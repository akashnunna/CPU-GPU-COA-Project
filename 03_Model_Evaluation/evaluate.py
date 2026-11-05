"""
evaluate.py  --  Final evaluation on the held-out TEST set (Stage 7)

Evaluates every saved best model (CPU-trained and GPU-trained, if they exist) on the SAME
test set, on the SAME device (CPU by default, so the device does not affect the numbers).
The test set is used only here, never for training or model selection.

Examples:
  python evaluate.py              evaluate best_cpu.pt and best_gpu.pt (whichever exist)
  python evaluate.py --smoke      evaluate the smoke-test models

Outputs: results/eval_<run>.json, results/per_class_<run>.csv, results/confusion_<run>.csv,
         graphs/confusion_matrix_<run>.png, results/model_agreement.json (if both models exist)
"""

import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support)

import config
from dataset import HAMDataset
from model import build_model


@torch.inference_mode()
def predict(model, dataset, device, batch_size=64):
    model.eval()
    preds, labels = [], []
    for i in range(0, len(dataset), batch_size):
        batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        x = torch.stack([b[0] for b in batch]).to(device)
        preds.append(model(x).argmax(dim=1).cpu())
        labels.append(torch.stack([b[1] for b in batch]))
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def plot_confusion(cm, title, path):
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(config.NUM_CLASSES)); ax.set_xticklabels(config.CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(config.NUM_CLASSES)); ax.set_yticklabels(config.CLASSES)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_title(title)
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Fraction of true class (row-normalized)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="evaluate the smoke-test models")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                        help="device used for evaluation (default cpu)")
    args = parser.parse_args()
    device = config.torch_device(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        print("GPU not available for evaluation; use --device cpu.")
        sys.exit(1)

    config.ensure_dirs()
    test_ds = HAMDataset("test", train=False)      # no augmentation
    print(f"Test images: {len(test_ds)}  | evaluation device: {device}")

    all_preds = {}
    found_any = False
    for tag in ["cpu", "gpu"]:
        name = config.run_name(tag, args.smoke)
        ckpt_path = config.MODELS_DIR / f"best_{name}.pt"
        if not ckpt_path.is_file():
            print(f"\n[{name}] checkpoint not found ({ckpt_path.name}) -> skipped")
            continue
        found_any = True
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model()
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        preds, labels = predict(model, test_ds, device)
        all_preds[tag] = preds

        labs = list(range(config.NUM_CLASSES))
        acc = accuracy_score(labels, preds)
        mp, mr, mf, _ = precision_recall_fscore_support(labels, preds, labels=labs, average="macro", zero_division=0)
        wp, wr, wf, _ = precision_recall_fscore_support(labels, preds, labels=labs, average="weighted", zero_division=0)
        p, r, f, s = precision_recall_fscore_support(labels, preds, labels=labs, zero_division=0)
        cm = confusion_matrix(labels, preds, labels=labs)

        per_class = pd.DataFrame({"class": config.CLASSES, "precision": p, "recall": r, "f1": f, "support": s})
        per_class.to_csv(config.RESULTS_DIR / f"per_class_{name}.csv", index=False)
        pd.DataFrame(cm, index=config.CLASSES, columns=config.CLASSES).to_csv(
            config.RESULTS_DIR / f"confusion_{name}.csv")
        plot_confusion(cm, f"Confusion matrix - model trained on {tag.upper()} (test set, n={len(labels)})",
                       config.GRAPHS_DIR / f"confusion_matrix_{name}.png")

        result = {
            "run": name, "model_trained_on": ckpt.get("device_trained_on"), "evaluated_on_device": device,
            "best_epoch": ckpt.get("best_epoch"), "test_samples": int(len(labels)),
            "accuracy": float(acc),
            "macro_precision": float(mp), "macro_recall": float(mr), "macro_f1": float(mf),
            "weighted_precision": float(wp), "weighted_recall": float(wr), "weighted_f1": float(wf),
        }
        with open(config.RESULTS_DIR / f"eval_{name}.json", "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

        print(f"\n[{name}] TEST RESULTS")
        print(f"  Accuracy            : {acc:.4f}")
        print(f"  Macro    P / R / F1 : {mp:.4f} / {mr:.4f} / {mf:.4f}")
        print(f"  Weighted P / R / F1 : {wp:.4f} / {wr:.4f} / {wf:.4f}")
        print(per_class.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"  Saved: results\\eval_{name}.json, per_class_{name}.csv, confusion_{name}.csv, "
              f"graphs\\confusion_matrix_{name}.png")

    if not found_any:
        print("\nNo trained models found. Run train.py first.")
        sys.exit(1)

    if "cpu" in all_preds and "gpu" in all_preds:
        agree = float(np.mean(all_preds["cpu"] == all_preds["gpu"]))
        with open(config.RESULTS_DIR / f"model_agreement{'_smoke' if args.smoke else ''}.json", "w") as fh:
            json.dump({"fraction_identical_predictions_cpu_model_vs_gpu_model": agree,
                       "note": "The two models were trained separately, so some disagreement is expected."}, fh, indent=2)
        print(f"\nCPU-trained vs GPU-trained model give the same prediction on {agree * 100:.1f}% of test images.")

    print("\nEVALUATION DONE")


if __name__ == "__main__":
    main()
