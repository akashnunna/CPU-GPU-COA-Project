"""
train.py  --  CPU / GPU training experiment (Stages 4, 5 and 6)

Examples (run from the project folder):
  python train.py --device cpu --smoke        quick pipeline test (2 epochs, small subset)
  python train.py --device cpu                full CPU experiment
  python train.py --device gpu                full GPU experiment (needs a CUDA GPU)

Both devices use the same model, split, epochs, batch size, optimizer, learning rate,
class weights, seed and preprocessing. The best model is chosen by VALIDATION macro-F1
(the test set is never used here).

Timing
  * Each epoch is timed with time.perf_counter(); on GPU torch.cuda.synchronize() is
    called before every reading, because GPU work is asynchronous.
  * A short warm-up (a few throw-away steps on a COPY of the model) runs before timing
    so one-time GPU start-up cost is not mixed into epoch 1. Its duration is reported separately.
  * 'total_training_time' = sum of epoch times (training + validation passes).
    'total_experiment_time' also includes model creation, data loading setup and warm-up.

Outputs (in models/ and results/)
  models/best_<run>.pt, results/history_<run>.csv, results/train_summary_<run>.json
  where <run> is cpu, gpu (or cpu_smoke / gpu_smoke for smoke tests)
"""

import argparse
import copy
import json
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import config
from dataset import make_loaders
from model import build_model, count_parameters


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over a loader. Trains if an optimizer is given, otherwise just evaluates.
    Metrics are accumulated on the device and read once at the end (avoids a GPU sync per batch)."""
    training = optimizer is not None
    model.train(training)
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    n = 0
    preds_all, labels_all = [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            if training:
                loss.backward()
                optimizer.step()
            bs = y.size(0)
            loss_sum += loss.detach() * bs
            pred = out.argmax(dim=1)
            correct += (pred == y).sum()
            n += bs
            if not training:
                preds_all.append(pred)
                labels_all.append(y)
    metrics = {"loss": loss_sum.item() / n, "acc": correct.item() / n}
    if not training:
        p = torch.cat(preds_all).cpu().numpy()
        t = torch.cat(labels_all).cpu().numpy()
        metrics["macro_f1"] = float(f1_score(t, p, average="macro", labels=list(range(config.NUM_CLASSES)),
                                             zero_division=0))
    return metrics


def warm_up(model, device, batch_size, class_weights):
    """A few throw-away training steps on a COPY of the model (real model/optimizer stay untouched)."""
    tmp = copy.deepcopy(model).to(device)
    opt = torch.optim.Adam(tmp.parameters(), lr=config.LEARNING_RATE)
    crit = nn.CrossEntropyLoss(weight=class_weights.to(device))
    tmp.train()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(3):
        x = torch.randn(batch_size, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
        y = torch.randint(0, config.NUM_CLASSES, (batch_size,), device=device)
        opt.zero_grad(set_to_none=True)
        crit(tmp(x), y).backward()
        opt.step()
    sync(device)
    elapsed = time.perf_counter() - t0
    del tmp, opt
    return elapsed


def sync(device):
    config.sync(device)


def main():
    parser = argparse.ArgumentParser(description="Train the CNN on CPU or GPU")
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--epochs", type=int, default=None, help=f"default {config.EPOCHS}")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--smoke", action="store_true", help="quick test: few epochs, small subset")
    args = parser.parse_args()

    device_tag = args.device
    device = config.torch_device(device_tag)
    name = config.run_name(device_tag, args.smoke)
    epochs = args.epochs if args.epochs is not None else (config.SMOKE_EPOCHS if args.smoke else config.EPOCHS)

    print("=" * 64)
    print(f"TRAINING RUN: {name}   (device={device}, epochs={epochs}, batch={args.batch_size}, lr={args.lr})")
    print("=" * 64)

    if device == "cuda" and not torch.cuda.is_available():
        print("\nGPU run requested but torch.cuda.is_available() is False.")
        print("No GPU experiment was performed (nothing is faked). To run it you need:")
        print("  1) an NVIDIA GPU with an up-to-date driver (check with: nvidia-smi)")
        print("  2) the CUDA build of PyTorch (see the install steps; the CPU-only build cannot use the GPU)")
        sys.exit(1)

    experiment_start = time.perf_counter()
    config.ensure_dirs()
    config.set_seed(config.SEED)

    env = config.hardware_info()
    print(f"Python {env['python_version']} | torch {env['torch_version']} | CPU: {env['cpu_name']}")
    if device == "cuda":
        print(f"GPU: {env['gpu_name']} | CUDA (torch build): {env['cuda_version_in_torch']}")

    # ---- data, model (always created on CPU first -> identical initial weights) ----
    train_loader, val_loader, _test_loader, class_weights, class_counts = make_loaders(
        batch_size=args.batch_size, pin_memory=(device == "cuda"), smoke=args.smoke)
    model = build_model()
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    print(f"Class weights: {[round(float(w), 3) for w in class_weights]}")

    # ---- warm-up (reported separately) ----
    warmup_time = warm_up(model, device, args.batch_size, class_weights)
    print(f"Warm-up time (not counted in training time): {warmup_time:.2f}s")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    config.set_seed(config.SEED)   # same random stream (shuffling/augmentation) start for every run

    # ---- training loop ----
    history = []
    best = {"macro_f1": -1.0, "val_loss": float("inf")}
    best_state, best_epoch = None, 0
    print(f"\n{'ep':>3} {'tr_loss':>8} {'va_loss':>8} {'tr_acc':>7} {'va_acc':>7} {'va_F1':>7} {'time(s)':>8}")
    for epoch in range(1, epochs + 1):
        sync(device)
        t0 = time.perf_counter()
        tr = run_epoch(model, train_loader, criterion, device, optimizer)
        sync(device)
        t1 = time.perf_counter()
        va = run_epoch(model, val_loader, criterion, device)
        sync(device)
        t2 = time.perf_counter()

        row = {
            "epoch": epoch, "train_loss": tr["loss"], "val_loss": va["loss"],
            "train_acc": tr["acc"], "val_acc": va["acc"], "val_macro_f1": va["macro_f1"],
            "train_time_s": t1 - t0, "val_time_s": t2 - t1, "epoch_time_s": t2 - t0,
        }
        history.append(row)

        improved = (va["macro_f1"] > best["macro_f1"]) or (
            va["macro_f1"] == best["macro_f1"] and va["loss"] < best["val_loss"])
        if improved:
            best = {"macro_f1": va["macro_f1"], "val_loss": va["loss"], "val_acc": va["acc"]}
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"{epoch:>3} {tr['loss']:>8.4f} {va['loss']:>8.4f} {tr['acc']:>7.4f} {va['acc']:>7.4f} "
              f"{va['macro_f1']:>7.4f} {t2 - t0:>8.1f}{'  *best' if improved else ''}")

    # ---- save everything ----
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(config.RESULTS_DIR / f"history_{name}.csv", index=False)

    torch.save({
        "model_state": best_state, "best_epoch": best_epoch, "device_trained_on": device_tag,
        "val_macro_f1": best["macro_f1"], "val_loss": best["val_loss"], "val_acc": best["val_acc"],
        "image_size": config.IMAGE_SIZE, "classes": config.CLASSES,
    }, config.MODELS_DIR / f"best_{name}.pt")

    total_training_time = float(hist_df["epoch_time_s"].sum())
    gpu_mem = {"peak_allocated_mb": "unavailable (CPU run)", "peak_reserved_mb": "unavailable (CPU run)"}
    if device == "cuda":
        gpu_mem = {"peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
                   "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024 ** 2, 1)}

    summary = {
        "run": name, "device": device_tag, "smoke_test": args.smoke,
        "settings": {
            "epochs": epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
            "weight_decay": config.WEIGHT_DECAY, "optimizer": "Adam", "loss": "CrossEntropyLoss (class-weighted)",
            "image_size": config.IMAGE_SIZE, "seed": config.SEED, "num_workers": config.NUM_WORKERS,
            "pin_memory": device == "cuda", "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset), "model_parameters": count_parameters(model),
        },
        "class_weights": [float(w) for w in class_weights],
        "best_epoch": best_epoch, "best_val_macro_f1": best["macro_f1"],
        "best_val_loss": best["val_loss"], "best_val_acc": best["val_acc"],
        "total_training_time_s": total_training_time,
        "total_train_only_time_s": float(hist_df["train_time_s"].sum()),
        "avg_epoch_time_s": float(hist_df["epoch_time_s"].mean()),
        "avg_epoch_time_excl_first_s": float(hist_df["epoch_time_s"].iloc[1:].mean()) if epochs > 1 else None,
        "warmup_time_s": warmup_time,
        "total_experiment_time_s": time.perf_counter() - experiment_start,
        "gpu_memory": gpu_mem,
        "environment": env,
        "notes": ("Models start from identical initial weights (created on CPU with the same seed). "
                  "Dropout uses a different random generator on CPU and GPU, and floating-point results "
                  "differ slightly between devices, so the two training curves are similar but not identical."),
    }
    with open(config.RESULTS_DIR / f"train_summary_{name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 64)
    print(f"DONE: {name}")
    print(f"  Best epoch (by val macro-F1): {best_epoch}  (val macro-F1 {best['macro_f1']:.4f})")
    print(f"  Total training time : {total_training_time:.1f}s   (avg {summary['avg_epoch_time_s']:.1f}s/epoch)")
    print(f"  Total experiment    : {summary['total_experiment_time_s']:.1f}s (includes setup + warm-up)")
    print(f"  GPU peak memory     : {gpu_mem['peak_allocated_mb']}")
    print(f"  Saved: models\\best_{name}.pt, results\\history_{name}.csv, results\\train_summary_{name}.json")
    print("=" * 64)


if __name__ == "__main__":
    main()
