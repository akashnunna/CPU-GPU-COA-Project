"""
benchmark_inference.py  --  CPU vs GPU inference benchmark (Stage 8)

Examples:
  python benchmark_inference.py             full benchmark
  python benchmark_inference.py --smoke     tiny quick test

Procedure (identical for both devices)
  * The SAME weights are used on both devices (default: the CPU-trained checkpoint), so only
    the hardware differs. Change with --checkpoint gpu.
  * The SAME first 1000 test images (sorted order) are used. Images are preprocessed
    (resized + normalized) BEFORE timing, so image decoding is NOT part of the measurement.
  * model.eval() + torch.inference_mode() (no gradients). Model loading time is measured
    and reported separately; it is not part of latency.
  * Warm-up batches run first (not timed). GPU is synchronized before and after every
    timed batch, because GPU work is asynchronous.
  * Two batch sizes: 1 (single-image inference: latency of one request) and 32 (batch
    inference: throughput). Results are labelled by batch size; compare like with like.
  * Each setting is repeated several times; mean and standard deviation are reported.
  * Two GPU modes: 'compute_only' (inputs already on the GPU) and 'with_transfer'
    (CPU->GPU copy of the input and GPU->CPU copy of the result are inside the timing).
    On CPU there is no transfer, so only 'compute_only' exists.

Outputs: results/inference_benchmark.csv (summary) and results/inference_benchmark_raw.json
"""

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

import config
from dataset import HAMDataset
from model import build_model


def load_model(ckpt_path, device):
    """Load checkpoint -> model on device. Returns (model, load_time_seconds)."""
    t0 = time.perf_counter()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    config.sync(device)
    return model, time.perf_counter() - t0


def make_batches(images, batch_size):
    return [images[i:i + batch_size].contiguous() for i in range(0, len(images), batch_size)]


@torch.inference_mode()
def timed_pass(model, batches, device, with_transfer):
    """One full pass over all batches. Returns list of (seconds, n_images) per batch."""
    times = []
    for b in batches:
        config.sync(device)
        t0 = time.perf_counter()
        x = b.to(device) if with_transfer else b
        pred = model(x).argmax(dim=1)
        if with_transfer:
            pred = pred.cpu()
        config.sync(device)
        times.append((time.perf_counter() - t0, b.shape[0]))
    return times


@torch.inference_mode()
def warm_up(model, batches, device, with_transfer, n_batches):
    for i in range(n_batches):
        b = batches[i % len(batches)]
        x = b.to(device) if with_transfer else b
        model(x).argmax(dim=1)
    config.sync(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint", choices=["cpu", "gpu"], default="cpu",
                        help="which trained checkpoint's weights to benchmark on BOTH devices (default cpu)")
    args = parser.parse_args()

    n_images = config.SMOKE_BENCH_IMAGES if args.smoke else config.BENCH_IMAGES
    repeats = config.SMOKE_BENCH_REPEATS if args.smoke else config.BENCH_REPEATS
    warm_batches = 3 if args.smoke else config.BENCH_WARMUP_BATCHES
    suffix = "_smoke" if args.smoke else ""

    ckpt_path = config.MODELS_DIR / f"best_{config.run_name(args.checkpoint, args.smoke)}.pt"
    if not ckpt_path.is_file():
        print(f"Checkpoint not found: {ckpt_path}\nRun train.py first.")
        sys.exit(1)

    config.ensure_dirs()
    env = config.hardware_info()
    print("=" * 64)
    print("INFERENCE BENCHMARK")
    print("=" * 64)
    print(f"Weights: {ckpt_path.name} | images: {n_images} | repeats: {repeats} | batch sizes: {config.BENCH_BATCH_SIZES}")
    print(f"CPU: {env['cpu_name']} (torch threads: {env['torch_cpu_threads']})")

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
        print(f"GPU: {env['gpu_name']}")
    else:
        print("GPU: unavailable -> only the CPU benchmark will run (no GPU numbers are invented)")

    # Preprocessed test images (excluded from timing)
    ds = HAMDataset("test", train=False)
    n_images = min(n_images, len(ds))
    images = torch.stack([ds[i][0] for i in range(n_images)])
    print(f"Prepared {n_images} preprocessed test images: {tuple(images.shape)}")

    rows, raw = [], []
    for device in devices:
        model, load_time = load_model(ckpt_path, device)
        print(f"\n[{device}] model load time (reported separately): {load_time:.3f}s")
        modes = ["compute_only"] if device == "cpu" else ["compute_only", "with_transfer"]
        for bs in config.BENCH_BATCH_SIZES:
            cpu_batches = make_batches(images, bs)
            for mode in modes:
                with_transfer = (mode == "with_transfer")
                batches = cpu_batches if (device == "cpu" or with_transfer) else [b.to(device) for b in cpu_batches]
                warm_up(model, batches, device, with_transfer, warm_batches)
                per_repeat, all_batch_ms = [], []
                for r in range(repeats):
                    times = timed_pass(model, batches, device, with_transfer)
                    total = sum(t for t, _ in times)
                    per_repeat.append({"repeat": r + 1, "total_s": total,
                                       "ms_per_image": total / n_images * 1000, "images_per_s": n_images / total})
                    all_batch_ms.extend(t * 1000 for t, _ in times)
                ms = np.array([p["ms_per_image"] for p in per_repeat])
                ips = np.array([p["images_per_s"] for p in per_repeat])
                row = {
                    "device": "gpu" if device == "cuda" else "cpu", "mode": mode, "batch_size": bs,
                    "images": n_images, "repeats": repeats,
                    "mean_ms_per_image": ms.mean(), "std_ms_per_image": ms.std(),
                    "mean_images_per_s": ips.mean(), "std_images_per_s": ips.std(),
                    "batch_latency_p50_ms": float(np.percentile(all_batch_ms, 50)),
                    "batch_latency_p95_ms": float(np.percentile(all_batch_ms, 95)),
                    "model_load_time_s": load_time,
                }
                rows.append(row)
                raw.append({"device": row["device"], "mode": mode, "batch_size": bs, "per_repeat": per_repeat})
                print(f"  [{row['device']}] batch={bs:<3} {mode:<14} "
                      f"{row['mean_ms_per_image']:.3f} +/- {row['std_ms_per_image']:.3f} ms/image | "
                      f"{row['mean_images_per_s']:.1f} +/- {row['std_images_per_s']:.1f} images/s")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / f"inference_benchmark{suffix}.csv", index=False)
    with open(config.RESULTS_DIR / f"inference_benchmark_raw{suffix}.json", "w", encoding="utf-8") as f:
        json.dump({"weights": ckpt_path.name, "images": n_images, "repeats": repeats,
                   "warmup_batches": warm_batches, "environment": env, "runs": raw,
                   "gpu_benchmark_ran": "cuda" in devices}, f, indent=2)

    print("\nBatch size 1  = single-image inference (latency of one request).")
    print("Batch size 32 = batch inference (shows throughput).")
    print(f"Saved: results\\inference_benchmark{suffix}.csv and results\\inference_benchmark_raw{suffix}.json")
    print("INFERENCE BENCHMARK DONE")


if __name__ == "__main__":
    main()
