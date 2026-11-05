"""
validate_images.py  --  Stage 1: Dataset validation for HAM10000

Reads the metadata CSV, indexes the image files, matches them, checks
every image can be decoded, and saves a validation report.
The original dataset is only READ, never modified or deleted.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

# ----------------------------------------------------------------------
# SETTINGS (change these paths if your folders are somewhere else)
# ----------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(r"C:\Users\akash\Downloads\ham10000")

METADATA_CSV = DATASET_DIR / "HAM10000_metadata.csv"
IMAGE_DIRS = [
    DATASET_DIR / "HAM10000_images_part_1",
    DATASET_DIR / "HAM10000_images_part_2",
]

DATA_DIR = PROJECT_DIR / "data"
SUMMARY_JSON = DATA_DIR / "validation_summary.json"
IMAGE_REPORT_CSV = DATA_DIR / "validation_image_report.csv"

EXPECTED_COLUMNS = ["lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"]
EXPECTED_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Class counts from your earlier inspection (used only as a cross-check)
EXPECTED_CLASS_COUNTS = {
    "akiec": 327, "bcc": 514, "bkl": 1099, "df": 115,
    "mel": 1113, "nv": 6705, "vasc": 142,
}
EXPECTED_TOTAL = 10015

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def build_image_index():
    """Scan both image folders. Returns (index, info).
    index: {image_id: Path}. info holds per-folder counts and problems."""
    index = {}
    info = {
        "folder_counts": {},
        "missing_folders": [],
        "duplicate_filenames": [],   # same image_id found more than once
        "non_image_files": [],       # files with unexpected extensions
    }
    for folder in IMAGE_DIRS:
        if not folder.is_dir():
            info["missing_folders"].append(str(folder))
            info["folder_counts"][folder.name] = 0
            continue
        count = 0
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                info["non_image_files"].append(str(path))
                continue
            count += 1
            image_id = path.stem
            if image_id in index:
                info["duplicate_filenames"].append(
                    {"image_id": image_id, "first": str(index[image_id]), "second": str(path)}
                )
            else:
                index[image_id] = path
        info["folder_counts"][folder.name] = count
    return index, info


def check_image(path):
    """Try to fully decode one image.
    Returns (status, width, height, mode, error_message)."""
    try:
        # verify() checks file structure but does not decode pixels
        with Image.open(path) as img:
            img.verify()
        # Re-open (verify invalidates the image) and decode all pixels
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            mode = img.mode
            img.convert("RGB")
        return "ok", width, height, mode, ""
    except Exception as e:  # any failure means the image is unusable
        return "corrupted", None, None, None, f"{type(e).__name__}: {e}"


def line(char="=", n=64):
    print(char * n)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    start = time.time()
    errors = []    # problems that must be fixed before Stage 2
    warnings = []  # things worth knowing, not blocking
    summary = {}

    line()
    print("STAGE 1: HAM10000 DATASET VALIDATION")
    line()

    # ---- 1. Read metadata ------------------------------------------------
    if not METADATA_CSV.is_file():
        print(f"ERROR: Metadata file not found:\n  {METADATA_CSV}")
        sys.exit(1)

    df = pd.read_csv(METADATA_CSV)
    print(f"Metadata file : {METADATA_CSV}")
    print(f"Records read  : {len(df)}")

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        print(f"Columns found: {list(df.columns)}")
        sys.exit(1)

    # Clean whitespace in key columns (in memory only; the CSV is untouched)
    ws_count = 0
    for col in ["image_id", "lesion_id", "dx"]:
        as_text = df[col].astype(str)
        stripped = as_text.str.strip()
        ws_count += int((as_text != stripped).sum())
        df[col] = stripped.where(df[col].notna(), None)
    if ws_count:
        warnings.append(f"{ws_count} key values had leading/trailing spaces (stripped in memory only).")

    summary["metadata_records"] = int(len(df))
    if len(df) != EXPECTED_TOTAL:
        warnings.append(f"Metadata has {len(df)} records; your earlier inspection reported {EXPECTED_TOTAL}.")

    # ---- 2. Metadata consistency ----------------------------------------
    missing_values = {c: int(df[c].isna().sum()) for c in EXPECTED_COLUMNS}
    summary["missing_values_per_column"] = missing_values
    for col in ["image_id", "lesion_id", "dx"]:
        if missing_values[col] > 0:
            errors.append(f"{missing_values[col]} records have a missing '{col}' (required column).")
    for col in ["age", "sex", "localization", "dx_type"]:
        if missing_values[col] > 0:
            warnings.append(f"{missing_values[col]} records have missing '{col}' (not used for training).")

    dup_ids = df[df["image_id"].duplicated(keep=False)]["image_id"].unique().tolist()
    summary["duplicate_image_ids"] = len(dup_ids)
    if dup_ids:
        errors.append(f"{len(dup_ids)} image_id values appear more than once, e.g. {dup_ids[:5]}")

    full_dup_rows = int(df.duplicated().sum())
    summary["fully_duplicated_rows"] = full_dup_rows
    if full_dup_rows:
        errors.append(f"{full_dup_rows} fully duplicated metadata rows.")

    summary["unique_image_ids"] = int(df["image_id"].nunique())
    summary["unique_lesion_ids"] = int(df["lesion_id"].nunique())

    # A lesion should have one diagnosis; more than one is inconsistent
    dx_per_lesion = df.dropna(subset=["lesion_id", "dx"]).groupby("lesion_id")["dx"].nunique()
    inconsistent = dx_per_lesion[dx_per_lesion > 1].index.tolist()
    summary["lesions_with_multiple_dx"] = len(inconsistent)
    if inconsistent:
        warnings.append(
            f"{len(inconsistent)} lesion_ids have more than one diagnosis label, e.g. {inconsistent[:5]}"
        )

    # ---- 3. Classes ------------------------------------------------------
    class_counts = df["dx"].value_counts().to_dict()
    summary["class_counts"] = {k: int(v) for k, v in class_counts.items()}
    found = set(class_counts.keys())
    missing_classes = [c for c in EXPECTED_CLASSES if c not in found]
    unexpected_classes = sorted(found - set(EXPECTED_CLASSES))
    summary["missing_classes"] = missing_classes
    summary["unexpected_classes"] = unexpected_classes
    if missing_classes:
        errors.append(f"Expected classes not found: {missing_classes}")
    if unexpected_classes:
        errors.append(f"Unexpected class labels found: {unexpected_classes}")
    for cls, expected in EXPECTED_CLASS_COUNTS.items():
        actual = class_counts.get(cls, 0)
        if actual != expected:
            warnings.append(f"Class '{cls}' has {actual} records; earlier inspection reported {expected}.")

    # ---- 4. Index image files and match to metadata ---------------------
    print("\nScanning image folders ...")
    index, idx_info = build_image_index()
    summary["image_folder_counts"] = idx_info["folder_counts"]
    summary["total_image_files_found"] = len(index)

    for folder in idx_info["missing_folders"]:
        errors.append(f"Image folder not found: {folder}")
    if idx_info["duplicate_filenames"]:
        errors.append(
            f"{len(idx_info['duplicate_filenames'])} image_ids exist in more than one folder "
            f"(first: {idx_info['duplicate_filenames'][0]})."
        )
    if idx_info["non_image_files"]:
        warnings.append(f"{len(idx_info['non_image_files'])} non-image files were ignored in the image folders.")

    meta_ids = set(df["image_id"].dropna())
    file_ids = set(index.keys())
    missing_files = sorted(meta_ids - file_ids)
    orphan_files = sorted(file_ids - meta_ids)
    summary["metadata_ids_without_file"] = len(missing_files)
    summary["files_without_metadata"] = len(orphan_files)
    summary["missing_file_examples"] = missing_files[:10]
    summary["orphan_file_examples"] = orphan_files[:10]
    if missing_files:
        errors.append(
            f"{len(missing_files)} metadata records have NO image file, e.g. {missing_files[:5]}. "
            "These cannot be used for training."
        )
    if orphan_files:
        warnings.append(
            f"{len(orphan_files)} image files have no metadata record, e.g. {orphan_files[:5]}. "
            "They will be ignored (no label)."
        )

    # ---- 5. Check each matched image can be decoded ---------------------
    to_check = df.dropna(subset=["image_id"]).drop_duplicates(subset="image_id")
    total = len(to_check)
    print(f"Checking integrity of {total} images (this can take a few minutes) ...")

    records = []
    size_counter = Counter()
    mode_counter = Counter()
    for i, row in enumerate(to_check.itertuples(index=False), start=1):
        image_id = row.image_id
        path = index.get(image_id)
        if path is None:
            records.append({
                "image_id": image_id, "lesion_id": row.lesion_id, "dx": row.dx,
                "image_path": "", "file_found": False, "status": "missing",
                "width": None, "height": None, "mode": None, "error": "No image file found",
            })
        else:
            status, w, h, mode, err = check_image(path)
            if status == "ok":
                size_counter[(w, h)] += 1
                mode_counter[mode] += 1
            records.append({
                "image_id": image_id, "lesion_id": row.lesion_id, "dx": row.dx,
                "image_path": str(path), "file_found": True, "status": status,
                "width": w, "height": h, "mode": mode, "error": err,
            })
        if i % 1000 == 0 or i == total:
            print(f"  checked {i}/{total}")

    report = pd.DataFrame(records)
    n_ok = int((report["status"] == "ok").sum())
    n_corrupt = int((report["status"] == "corrupted").sum())
    n_missing = int((report["status"] == "missing").sum())
    summary["images_ok"] = n_ok
    summary["images_corrupted"] = n_corrupt
    summary["images_missing"] = n_missing
    summary["image_sizes_top5"] = {f"{w}x{h}": c for (w, h), c in size_counter.most_common(5)}
    summary["image_modes"] = dict(mode_counter)

    corrupt_ids = report.loc[report["status"] == "corrupted", "image_id"].tolist()
    summary["corrupted_image_ids"] = corrupt_ids
    if n_corrupt:
        errors.append(f"{n_corrupt} images are corrupted/unreadable, e.g. {corrupt_ids[:5]}")
    if len(size_counter) > 1:
        warnings.append(f"Images come in {len(size_counter)} different sizes (we will resize in Stage 3).")
    non_rgb = {m: c for m, c in mode_counter.items() if m != "RGB"}
    if non_rgb:
        warnings.append(f"Some images are not RGB: {non_rgb} (they will be converted to RGB in Stage 3).")

    # Usable = metadata record with a readable image file
    usable = report[report["status"] == "ok"]
    summary["usable_records"] = int(len(usable))
    summary["usable_class_counts"] = {k: int(v) for k, v in usable["dx"].value_counts().to_dict().items()}
    summary["usable_unique_lesions"] = int(usable["lesion_id"].nunique())

    # ---- 6. Save results -------------------------------------------------
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary["errors"] = errors
    summary["warnings"] = warnings
    summary["validation_passed"] = len(errors) == 0
    summary["seconds_taken"] = round(time.time() - start, 1)

    report.to_csv(IMAGE_REPORT_CSV, index=False)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # ---- 7. Print readable summary --------------------------------------
    print()
    line()
    print("VALIDATION SUMMARY")
    line()
    print(f"Metadata records            : {summary['metadata_records']}")
    print(f"Unique image IDs            : {summary['unique_image_ids']}")
    print(f"Unique lesion IDs           : {summary['unique_lesion_ids']}")
    print(f"Duplicate image IDs         : {summary['duplicate_image_ids']}")
    print(f"Lesions with >1 diagnosis   : {summary['lesions_with_multiple_dx']}")
    for folder, count in idx_info["folder_counts"].items():
        print(f"Image files in {folder:<22}: {count}")
    print(f"Total image files indexed   : {summary['total_image_files_found']}")
    print(f"Metadata IDs without a file : {summary['metadata_ids_without_file']}")
    print(f"Files without metadata      : {summary['files_without_metadata']}")
    print(f"Readable images             : {n_ok}")
    print(f"Corrupted images            : {n_corrupt}")
    print(f"Common image sizes          : {summary['image_sizes_top5']}")
    print(f"Image color modes           : {summary['image_modes']}")
    print("\nClass counts (metadata):")
    for cls in EXPECTED_CLASSES:
        meta_n = class_counts.get(cls, 0)
        use_n = summary["usable_class_counts"].get(cls, 0)
        print(f"  {cls:<6} metadata={meta_n:<6} usable={use_n}")
    print(f"\nUsable records for later stages: {summary['usable_records']} "
          f"({summary['usable_unique_lesions']} unique lesions)")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print("\nERRORS (fix before Stage 2):")
        for e in errors:
            print(f"  - {e}")

    print()
    line()
    print("RESULT:", "PASSED" if not errors else "FAILED")
    line()
    print(f"Saved: {SUMMARY_JSON}")
    print(f"Saved: {IMAGE_REPORT_CSV}")
    print(f"Time taken: {summary['seconds_taken']} seconds")

    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
