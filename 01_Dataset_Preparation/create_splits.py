"""
create_splits.py  --  Stage 2: Leakage-free (lesion-aware) dataset split

Splits HAM10000 into train / val / test so that ALL images of a lesion
(same lesion_id) end up in exactly ONE split. Tries to keep class
proportions similar in each split and to hit the requested sizes
(val=1000, test=2000, train=everything else) as closely as grouping allows.

Reads : data/validation_summary.json, data/validation_image_report.csv  (from Stage 1)
Writes: data/split_manifest.csv, split_train.csv, split_val.csv,
        split_test.csv, split_summary.json, class_mapping.json
Only writes if ALL leakage checks pass. The original dataset is never touched.
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
PROJECT_DIR = Path(r"C:\Users\akash\Downloads\CPU-GPU-COA-Project")
DATA_DIR = PROJECT_DIR / "data"

VALIDATION_SUMMARY = DATA_DIR / "validation_summary.json"
IMAGE_REPORT_CSV = DATA_DIR / "validation_image_report.csv"

MANIFEST_CSV = DATA_DIR / "split_manifest.csv"
TRAIN_CSV = DATA_DIR / "split_train.csv"
VAL_CSV = DATA_DIR / "split_val.csv"
TEST_CSV = DATA_DIR / "split_test.csv"
SPLIT_SUMMARY_JSON = DATA_DIR / "split_summary.json"
CLASS_MAPPING_JSON = DATA_DIR / "class_mapping.json"

SEED = 42
TARGET_VAL = 1000
TARGET_TEST = 2000
# Train target = all remaining images (so no image is silently discarded)

# Fixed, alphabetical class order -> the same label number every run
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
SPLITS = ["train", "val", "test"]


def line(char="=", n=64):
    print(char * n)


def fail(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


def main():
    line()
    print("STAGE 2: LEAKAGE-FREE SPLIT (grouped by lesion_id)")
    line()

    # ---- 1. Make sure Stage 1 passed -------------------------------------
    if not VALIDATION_SUMMARY.is_file() or not IMAGE_REPORT_CSV.is_file():
        fail("Stage 1 output not found in the data folder. Run validate_images.py first.")
    with open(VALIDATION_SUMMARY, "r", encoding="utf-8") as f:
        val_summary = json.load(f)
    if not val_summary.get("validation_passed", False):
        fail("Stage 1 validation did not pass. Fix those problems first.")

    # ---- 2. Load usable images -------------------------------------------
    report = pd.read_csv(IMAGE_REPORT_CSV)
    df = report[report["status"] == "ok"][["image_id", "lesion_id", "dx", "image_path"]].copy()
    df = df.reset_index(drop=True)
    n_total = len(df)
    print(f"Usable images loaded : {n_total}")
    print(f"Unique lesions       : {df['lesion_id'].nunique()}")

    if df["image_id"].duplicated().any():
        fail("Duplicate image_id values found in the report.")
    unknown = sorted(set(df["dx"]) - set(CLASSES))
    if unknown:
        fail(f"Unknown class labels found: {unknown}")
    if n_total <= TARGET_VAL + TARGET_TEST:
        fail("Not enough images for the requested validation/test sizes.")

    target_sizes = {"val": TARGET_VAL, "test": TARGET_TEST, "train": n_total - TARGET_VAL - TARGET_TEST}
    fractions = {s: target_sizes[s] / n_total for s in SPLITS}
    print(f"Target sizes         : train={target_sizes['train']}, "
          f"val={target_sizes['val']}, test={target_sizes['test']}")

    # ---- 3. Build lesion groups ------------------------------------------
    # For each lesion: how many images of each class it has (normally one class)
    lesion_vectors = {}
    for lesion_id, group in df.groupby("lesion_id"):
        lesion_vectors[lesion_id] = Counter(group["dx"])

    class_totals = Counter(df["dx"])
    # Ideal number of images of each class in each split
    class_targets = {s: {c: class_totals[c] * fractions[s] for c in CLASSES} for s in SPLITS}

    lesions = []
    for lesion_id in sorted(lesion_vectors.keys()):          # sorted = deterministic
        vec = lesion_vectors[lesion_id]
        dominant = vec.most_common(1)[0][0]
        lesions.append({"lesion_id": lesion_id, "vec": vec, "n": sum(vec.values()), "dom": dominant})

    # Shuffle with a fixed seed, then process rare classes first, bigger lesions first.
    # (sort is stable, so the shuffled order breaks ties reproducibly)
    rng = random.Random(SEED)
    rng.shuffle(lesions)
    lesions.sort(key=lambda L: (class_totals[L["dom"]], -L["n"]))

    # ---- 4. Greedy assignment of whole lesions to splits -----------------
    current = {s: Counter() for s in SPLITS}
    assignment = {}
    for L in lesions:
        best_split, best_score = None, None
        for s in SPLITS:
            score = 0.0
            for c, count in L["vec"].items():
                target = max(class_targets[s][c], 1e-9)
                score += count * (class_targets[s][c] - current[s][c]) / target
            if best_score is None or score > best_score:
                best_split, best_score = s, score
        assignment[L["lesion_id"]] = best_split
        current[best_split].update(L["vec"])

    df["split"] = df["lesion_id"].map(assignment)
    df["label"] = df["dx"].map(CLASS_TO_IDX)

    # ---- 5. Leakage and integrity checks ---------------------------------
    checks = {}
    checks["every image has exactly one split"] = bool(df["split"].isin(SPLITS).all())
    checks["total images preserved (none discarded)"] = (len(df) == n_total)

    lesion_sets = {s: set(df.loc[df["split"] == s, "lesion_id"]) for s in SPLITS}
    image_sets = {s: set(df.loc[df["split"] == s, "image_id"]) for s in SPLITS}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        checks[f"no lesion_id overlap: {a} vs {b}"] = len(lesion_sets[a] & lesion_sets[b]) == 0
        checks[f"no image_id overlap: {a} vs {b}"] = len(image_sets[a] & image_sets[b]) == 0

    splits_per_lesion = df.groupby("lesion_id")["split"].nunique()
    checks["every lesion is in exactly one split"] = bool((splits_per_lesion == 1).all())

    for s in SPLITS:
        present = set(df.loc[df["split"] == s, "dx"])
        checks[f"all 7 classes present in {s}"] = present == set(CLASSES)

    missing_paths = [p for p in df["image_path"] if not Path(p).is_file()]
    checks["all image paths exist on disk"] = len(missing_paths) == 0

    # ---- 6. Print report --------------------------------------------------
    print()
    line()
    print("SPLIT RESULTS")
    line()
    print(f"{'split':<7}{'target':>8}{'actual':>8}{'diff':>7}{'lesions':>9}{'share':>8}")
    for s in SPLITS:
        actual = int((df["split"] == s).sum())
        print(f"{s:<7}{target_sizes[s]:>8}{actual:>8}{actual - target_sizes[s]:>+7}"
              f"{len(lesion_sets[s]):>9}{actual / n_total * 100:>7.1f}%")
    print(f"{'total':<7}{n_total:>8}{len(df):>8}")

    print("\nClass counts per split (images):")
    print(f"{'class':<8}{'train':>8}{'val':>8}{'test':>8}{'total':>8}")
    class_table = {}
    for c in CLASSES:
        row = {s: int(((df["split"] == s) & (df["dx"] == c)).sum()) for s in SPLITS}
        class_table[c] = row
        print(f"{c:<8}{row['train']:>8}{row['val']:>8}{row['test']:>8}{sum(row.values()):>8}")

    print("\nClass share inside each split (%):")
    print(f"{'class':<8}{'train':>8}{'val':>8}{'test':>8}")
    for c in CLASSES:
        shares = []
        for s in SPLITS:
            size = int((df["split"] == s).sum())
            shares.append(class_table[c][s] / size * 100)
        print(f"{c:<8}{shares[0]:>7.1f}%{shares[1]:>7.1f}%{shares[2]:>7.1f}%")

    print("\nLEAKAGE / INTEGRITY CHECKS:")
    all_passed = True
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_passed = all_passed and ok

    if not all_passed:
        print("\nRESULT: FAILED -- no files were saved. Share this output with me.")
        sys.exit(1)

    # ---- 7. Save files ----------------------------------------------------
    manifest = df[["image_id", "lesion_id", "dx", "label", "split", "image_path"]].sort_values(
        ["split", "image_id"]
    )
    manifest.to_csv(MANIFEST_CSV, index=False)
    for s, path in [("train", TRAIN_CSV), ("val", VAL_CSV), ("test", TEST_CSV)]:
        manifest[manifest["split"] == s].to_csv(path, index=False)

    with open(CLASS_MAPPING_JSON, "w", encoding="utf-8") as f:
        json.dump({"class_to_idx": CLASS_TO_IDX, "idx_to_class": {str(i): c for c, i in CLASS_TO_IDX.items()}},
                  f, indent=2)

    summary = {
        "seed": SEED,
        "total_images": n_total,
        "target_sizes": target_sizes,
        "actual_sizes": {s: int((df["split"] == s).sum()) for s in SPLITS},
        "unique_lesions": {s: len(lesion_sets[s]) for s in SPLITS},
        "class_counts": class_table,
        "checks": checks,
        "all_checks_passed": True,
        "note": "Whole lesions are kept together to prevent leakage, so sizes can differ slightly "
                "from the requested 7000/1000/2000. The extra images beyond 10000 are in train.",
    }
    with open(SPLIT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    line()
    print("RESULT: PASSED -- no data leakage detected")
    line()
    print(f"Saved: {MANIFEST_CSV}")
    print(f"Saved: {TRAIN_CSV}")
    print(f"Saved: {VAL_CSV}")
    print(f"Saved: {TEST_CSV}")
    print(f"Saved: {SPLIT_SUMMARY_JSON}")
    print(f"Saved: {CLASS_MAPPING_JSON}")


if __name__ == "__main__":
    main()
