"""
compare_results.py  --  CPU vs GPU comparison, tables and graphs (Stage 9)

Reads the result files produced by train.py, evaluate.py and benchmark_inference.py and:
  * computes training speedup and % reduction in training time (only if BOTH runs exist)
  * compares inference latency / throughput, accuracy, macro/weighted F1 and GPU memory
  * checks that both runs used the same settings (fair comparison)
  * saves tables (CSV/JSON), a plain-language text summary and PNG graphs

Examples:
  python compare_results.py            real experiment results
  python compare_results.py --smoke    smoke-test results (pipeline check only)

Every number comes from the saved result files. If something is missing it is shown
as 'unavailable'; nothing is estimated or invented.
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config

COLORS = {"cpu": "#1f77b4", "gpu": "#2ca02c", "gpu_transfer": "#98df8a"}


def load_json(path):
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_csv(path):
    return pd.read_csv(path) if path.is_file() else None


def safe_div(a, b):
    if a is None or b is None:
        return None
    try:
        return None if float(b) == 0 else float(a) / float(b)
    except (TypeError, ValueError):
        return None


def fmt(v, nd=3):
    if v is None:
        return "unavailable"
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}"
    return str(v)


def num_or_none(v):
    return float(v) if isinstance(v, (int, float)) else None


# ----------------------------- graphs ---------------------------------
def bar_cpu_gpu(cpu_val, gpu_val, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    top = max([v for v in (cpu_val, gpu_val) if v is not None] or [1])
    for x, (key, val) in enumerate([("cpu", cpu_val), ("gpu", gpu_val)]):
        if val is None:
            ax.text(x, top * 0.05, f"{key.upper()}\nunavailable", ha="center", va="bottom", color="gray")
        else:
            ax.bar(x, val, color=COLORS[key], width=0.55)
            ax.text(x, val, f"{val:.1f}", ha="center", va="bottom")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["CPU", "GPU"])
    ax.set_ylabel(ylabel); ax.set_title(title); ax.set_ylim(0, top * 1.15)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def line_plot(hist, column, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for key in ("cpu", "gpu"):
        h = hist.get(key)
        if h is not None:
            ax.plot(h["epoch"], h[column], marker="o", label=key.upper(), color=COLORS[key])
    if hist.get("gpu") is None:
        title += "  (GPU: unavailable)"
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def inference_bars(bench, value_col, ylabel, title, path):
    series = [("CPU", "cpu", "compute_only", COLORS["cpu"]),
              ("GPU (compute only)", "gpu", "compute_only", COLORS["gpu"]),
              ("GPU (incl. CPU<->GPU transfer)", "gpu", "with_transfer", COLORS["gpu_transfer"])]
    sizes = sorted(bench["batch_size"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    width = 0.26
    for k, (label, dev, mode, color) in enumerate(series):
        sub = bench[(bench["device"] == dev) & (bench["mode"] == mode)]
        if sub.empty:
            continue
        xs, ys = [], []
        for i, bs in enumerate(sizes):
            row = sub[sub["batch_size"] == bs]
            if not row.empty:
                xs.append(i + (k - 1) * width); ys.append(float(row[value_col].iloc[0]))
        ax.bar(xs, ys, width=width, label=label, color=color)
        for x, y in zip(xs, ys):
            ax.text(x, y, f"{y:.2f}" if y < 100 else f"{y:.0f}", ha="center", va="bottom", fontsize=8)
    if "gpu" not in set(bench["device"]):
        title += "  (GPU: unavailable)"
    ax.set_xticks(range(len(sizes))); ax.set_xticklabels([f"batch size {s}" for s in sizes])
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ----------------------------- main -----------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    sfx = "_smoke" if args.smoke else ""
    config.ensure_dirs()
    R, G = config.RESULTS_DIR, config.GRAPHS_DIR

    names = {k: config.run_name(k, args.smoke) for k in ("cpu", "gpu")}
    summ = {k: load_json(R / f"train_summary_{names[k]}.json") for k in names}
    hist = {k: load_csv(R / f"history_{names[k]}.csv") for k in names}
    ev = {k: load_json(R / f"eval_{names[k]}.json") for k in names}
    bench = load_csv(R / f"inference_benchmark{sfx}.csv")
    gpu_ok = summ["gpu"] is not None

    print("=" * 64)
    print("CPU vs GPU COMPARISON" + (" (SMOKE TEST - pipeline check only)" if args.smoke else ""))
    print("=" * 64)
    if summ["cpu"] is None:
        print("No CPU training results found. Run train.py --device cpu first.")
        return
    if not gpu_ok:
        print("GPU training results not found -> GPU comparison is UNAVAILABLE (CPU results only).")

    # ---- fairness check ----
    warnings = []
    if gpu_ok:
        sc, sg = summ["cpu"]["settings"], summ["gpu"]["settings"]
        for key in ("epochs", "batch_size", "learning_rate", "weight_decay", "optimizer", "loss", "image_size",
                    "seed", "train_samples", "val_samples", "model_parameters", "num_workers"):
            if sc.get(key) != sg.get(key):
                warnings.append(f"Setting '{key}' differs: CPU={sc.get(key)} vs GPU={sg.get(key)}")
    known_differences = ["pin_memory is enabled on the GPU run only (standard for CUDA transfers).",
                         "Dropout random numbers come from different generators on CPU and GPU, and "
                         "floating-point results differ slightly between devices."]

    # ---- training comparison ----
    def g(k, key):
        return summ[k].get(key) if summ[k] else None

    cpu_t, gpu_t = g("cpu", "total_training_time_s"), g("gpu", "total_training_time_s")
    speedup = safe_div(cpu_t, gpu_t) if gpu_ok else None
    reduction = (safe_div((cpu_t - gpu_t), cpu_t) * 100) if (gpu_ok and cpu_t and gpu_t is not None
                                                             and safe_div((cpu_t - gpu_t), cpu_t) is not None) else None
    cpu_e1, gpu_e1 = g("cpu", "avg_epoch_time_excl_first_s"), g("gpu", "avg_epoch_time_excl_first_s")
    speedup_excl = safe_div(cpu_e1, gpu_e1) if gpu_ok else None

    gpu_mem = g("gpu", "gpu_memory")
    gpu_peak = num_or_none(gpu_mem.get("peak_allocated_mb")) if gpu_mem else None

    rows = [
        ("Total training time (s)", g("cpu", "total_training_time_s"), g("gpu", "total_training_time_s")),
        ("Training passes only (s)", g("cpu", "total_train_only_time_s"), g("gpu", "total_train_only_time_s")),
        ("Average time per epoch (s)", g("cpu", "avg_epoch_time_s"), g("gpu", "avg_epoch_time_s")),
        ("Average epoch time excl. epoch 1 (s)", cpu_e1, gpu_e1),
        ("Warm-up time, not in training time (s)", g("cpu", "warmup_time_s"), g("gpu", "warmup_time_s")),
        ("Total experiment time incl. setup (s)", g("cpu", "total_experiment_time_s"), g("gpu", "total_experiment_time_s")),
        ("Epochs run", g("cpu", "settings") and g("cpu", "settings")["epochs"],
         g("gpu", "settings") and g("gpu", "settings")["epochs"]),
        ("Best epoch (val macro-F1)", g("cpu", "best_epoch"), g("gpu", "best_epoch")),
        ("Best validation macro-F1", g("cpu", "best_val_macro_f1"), g("gpu", "best_val_macro_f1")),
        ("GPU peak memory allocated (MB)", "not applicable", gpu_peak),
    ]
    for key, label in (("accuracy", "Test accuracy"), ("macro_f1", "Test macro-F1"),
                       ("weighted_f1", "Test weighted-F1"), ("macro_precision", "Test macro precision"),
                       ("macro_recall", "Test macro recall")):
        rows.append((label, ev["cpu"][key] if ev["cpu"] else None, ev["gpu"][key] if ev["gpu"] else None))

    inf_summary = []
    if bench is not None:
        for bs in sorted(bench["batch_size"].unique()):
            def pick(dev, mode, col):
                s = bench[(bench["device"] == dev) & (bench["mode"] == mode) & (bench["batch_size"] == bs)]
                return float(s[col].iloc[0]) if not s.empty else None
            c_ms, g_ms, gt_ms = (pick("cpu", "compute_only", "mean_ms_per_image"),
                                  pick("gpu", "compute_only", "mean_ms_per_image"),
                                  pick("gpu", "with_transfer", "mean_ms_per_image"))
            c_ips, g_ips, gt_ips = (pick("cpu", "compute_only", "mean_images_per_s"),
                                    pick("gpu", "compute_only", "mean_images_per_s"),
                                    pick("gpu", "with_transfer", "mean_images_per_s"))
            rows += [(f"Inference latency, batch {bs} (ms/image)", c_ms, g_ms),
                     (f"Inference latency, batch {bs}, GPU incl. transfer (ms/image)", c_ms, gt_ms),
                     (f"Inference throughput, batch {bs} (images/s)", c_ips, g_ips),
                     (f"Inference throughput, batch {bs}, GPU incl. transfer (images/s)", c_ips, gt_ips)]
            inf_summary.append({"batch_size": int(bs),
                                "latency_speedup_compute_only": safe_div(c_ms, g_ms),
                                "latency_speedup_with_transfer": safe_div(c_ms, gt_ms),
                                "throughput_ratio_gpu_over_cpu_compute_only": safe_div(g_ips, c_ips),
                                "throughput_ratio_gpu_over_cpu_with_transfer": safe_div(gt_ips, c_ips),
                                "transfer_overhead_ms_per_image": (gt_ms - g_ms) if (gt_ms is not None and g_ms is not None) else None})

    table = pd.DataFrame([(m, c if c is not None else "unavailable", gv if gv is not None else "unavailable")
                          for m, c, gv in rows], columns=["metric", "CPU", "GPU"])
    table.to_csv(R / f"comparison_table{sfx}.csv", index=False)

    perf = {
        "gpu_experiment_completed": gpu_ok,
        "training_speedup_cpu_time_div_gpu_time": speedup,
        "training_time_reduction_percent": reduction,
        "epoch_time_speedup_excluding_epoch1": speedup_excl,
        "inference": inf_summary if gpu_ok else "unavailable (no GPU results)",
        "settings_mismatches": warnings,
        "known_unavoidable_differences": known_differences,
        "single_training_run_per_device": True,
    }
    with open(R / f"comparison_summary{sfx}.json", "w", encoding="utf-8") as f:
        json.dump(perf, f, indent=2)

    # ---- graphs ----
    bar_cpu_gpu(cpu_t, gpu_t, "Seconds", "Total training time: CPU vs GPU", G / f"graph_total_training_time{sfx}.png")
    bar_cpu_gpu(g("cpu", "avg_epoch_time_s"), g("gpu", "avg_epoch_time_s"), "Seconds per epoch",
                "Average epoch time: CPU vs GPU", G / f"graph_avg_epoch_time{sfx}.png")
    line_plot(hist, "epoch_time_s", "Seconds", "Training time by epoch", G / f"graph_epoch_time_by_epoch{sfx}.png")
    line_plot(hist, "train_loss", "Loss", "Training loss vs epoch", G / f"graph_train_loss{sfx}.png")
    line_plot(hist, "val_loss", "Loss", "Validation loss vs epoch", G / f"graph_val_loss{sfx}.png")
    line_plot(hist, "train_acc", "Accuracy", "Training accuracy vs epoch", G / f"graph_train_accuracy{sfx}.png")
    line_plot(hist, "val_acc", "Accuracy", "Validation accuracy vs epoch", G / f"graph_val_accuracy{sfx}.png")
    if bench is not None:
        inference_bars(bench, "mean_ms_per_image", "ms per image (lower is faster)",
                       "Inference latency: CPU vs GPU", G / f"graph_inference_latency{sfx}.png")
        inference_bars(bench, "mean_images_per_s", "images per second (higher is faster)",
                       "Inference throughput: CPU vs GPU", G / f"graph_inference_throughput{sfx}.png")
    else:
        print("No inference benchmark file found -> inference graphs skipped.")

    # ---- plain-language summary ----
    env = summ["cpu"]["environment"]
    L = []
    L.append("CPU vs GPU COMPARISON SUMMARY" + (" (SMOKE TEST)" if args.smoke else ""))
    L.append("=" * 64)
    L.append(f"CPU: {env['cpu_name']} ({env['cpu_logical_cores']} logical cores, {env['torch_cpu_threads']} torch threads)")
    if gpu_ok:
        genv = summ["gpu"]["environment"]
        L.append(f"GPU: {genv['gpu_name']} | CUDA (torch build): {genv['cuda_version_in_torch']} | torch {genv['torch_version']}")
    else:
        L.append("GPU: unavailable - the GPU experiment was not completed, so no GPU speedup is reported.")
    L.append(f"Python {env['python_version']} | torch {env['torch_version']}")
    L.append("")
    L.append("1. TRAINING TIME")
    L.append(f"   CPU total training time : {fmt(cpu_t, 1)} s  (average {fmt(g('cpu','avg_epoch_time_s'), 1)} s/epoch)")
    if gpu_ok:
        L.append(f"   GPU total training time : {fmt(gpu_t, 1)} s  (average {fmt(g('gpu','avg_epoch_time_s'), 1)} s/epoch)")
        L.append(f"   Training speedup (CPU time / GPU time)  : {fmt(speedup, 2)}x")
        L.append(f"   Reduction in training time              : {fmt(reduction, 1)} %")
        if speedup is not None:
            if speedup > 1:
                L.append("   -> In this experiment the GPU finished training faster than the CPU.")
            elif speedup < 1:
                L.append("   -> In this experiment the GPU was SLOWER than the CPU (see notes below).")
            else:
                L.append("   -> Both devices took about the same time.")
        L.append(f"   GPU warm-up (reported separately): {fmt(g('gpu','warmup_time_s'), 2)} s; "
                 f"GPU peak memory allocated: {fmt(gpu_peak, 1)} MB")
    L.append("")
    L.append("2. PREDICTIVE QUALITY (same held-out test set)")
    for k in ("cpu", "gpu"):
        if ev[k]:
            e = ev[k]
            L.append(f"   {k.upper()}-trained model: accuracy {e['accuracy']:.4f}, macro-F1 {e['macro_f1']:.4f}, "
                     f"weighted-F1 {e['weighted_f1']:.4f}")
        elif k == "cpu" or gpu_ok:
            L.append(f"   {k.upper()}-trained model: evaluation not found (run evaluate.py)")
    if ev["cpu"] and ev["gpu"]:
        L.append(f"   Difference in macro-F1 (GPU - CPU): {ev['gpu']['macro_f1'] - ev['cpu']['macro_f1']:+.4f}; "
                 f"accuracy: {ev['gpu']['accuracy'] - ev['cpu']['accuracy']:+.4f}")
        L.append("   The two models were trained separately, so small differences can come from training "
                 "randomness; this experiment does not isolate the cause.")
    L.append("   Macro-F1 matters most here because the classes are imbalanced (nv is by far the largest).")
    L.append("")
    L.append("3. INFERENCE (same weights, same test images on both devices)")
    if bench is None:
        L.append("   No inference benchmark results found.")
    else:
        for item in inf_summary:
            bs = item["batch_size"]
            kind = "single-image" if bs == 1 else "batch"
            L.append(f"   Batch size {bs} ({kind} inference):")
            if gpu_ok and item["latency_speedup_compute_only"] is not None:
                L.append(f"     latency speedup, compute only        : {fmt(item['latency_speedup_compute_only'], 2)}x (CPU time / GPU time)")
                L.append(f"     latency speedup, incl. data transfer : {fmt(item['latency_speedup_with_transfer'], 2)}x")
                L.append(f"     transfer overhead on GPU             : {fmt(item['transfer_overhead_ms_per_image'], 3)} ms/image")
            else:
                L.append("     GPU numbers unavailable.")
    L.append("")
    L.append("4. FAIRNESS CHECK")
    if warnings:
        L.append("   WARNING - settings differ between runs; the speed comparison is NOT like-for-like:")
        L.extend(f"     - {w}" for w in warnings)
    elif gpu_ok:
        L.append("   Same model, split, epochs, batch size, optimizer, learning rate, class weights, seed and preprocessing.")
    L.append("   Known differences that cannot be removed:")
    L.extend(f"     - {d}" for d in known_differences)
    L.append("   Each device was trained once, so the speedup comes from a single run per device.")
    L.append("")
    L.append("5. BACKGROUND (general computer-architecture ideas; NOT measured by this experiment)")
    L.append("   - A GPU has thousands of simple cores that run many multiply-add operations in parallel,")
    L.append("     which suits the large matrix/convolution work in a CNN; a CPU has few, more complex cores.")
    L.append("   - A GPU is only fully used when each step has enough work (e.g. larger batches / bigger models).")
    L.append("     A small model with a small batch may leave much of the GPU idle.")
    L.append("   - Data loading and augmentation run on the CPU in both experiments, and each batch must be copied")
    L.append("     from CPU memory to GPU memory; this transfer cost is shown in the 'incl. transfer' inference results.")
    L.append("   - The first GPU use includes one-time start-up work; this run reports the warm-up separately.")
    L.append("")
    L.append("Educational experiment only - not a medical diagnostic tool.")
    text = "\n".join(L)
    with open(R / f"comparison_summary{sfx}.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print("\nSaved tables : results\\comparison_table%s.csv, comparison_summary%s.json / .txt" % (sfx, sfx))
    print("Saved graphs : graphs\\graph_*%s.png (confusion matrices come from evaluate.py)" % sfx)
    print("COMPARISON DONE")


if __name__ == "__main__":
    main()
