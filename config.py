"""
config.py  --  Shared settings and helper functions used by every script.

All paths and hyperparameters live HERE, so CPU and GPU runs always use
exactly the same values. Change a value once and every script sees it.
"""

import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# ----------------------------------------------------------------------
# PATHS (change PROJECT_DIR if your folder is somewhere else)
# ----------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
GRAPHS_DIR = PROJECT_DIR / "graphs"

TRAIN_CSV = DATA_DIR / "split_train.csv"
VAL_CSV = DATA_DIR / "split_val.csv"
TEST_CSV = DATA_DIR / "split_test.csv"
CLASS_MAPPING_JSON = DATA_DIR / "class_mapping.json"

# ----------------------------------------------------------------------
# DATA / MODEL / TRAINING SETTINGS (identical for CPU and GPU runs)
# ----------------------------------------------------------------------
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]   # label 0..6
NUM_CLASSES = len(CLASSES)

IMAGE_SIZE = 128            # images are resized to 128 x 128
NORM_MEAN = (0.485, 0.456, 0.406)   # standard ImageNet statistics
NORM_STD = (0.229, 0.224, 0.225)

SEED = 42
EPOCHS = 15                 # full experiment (use --epochs or --smoke to change)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0             # 0 = load in main process (safest on Windows, same for both devices)

# Smoke test (quick pipeline check)
SMOKE_EPOCHS = 2
SMOKE_TRAIN_SAMPLES = 512
SMOKE_VAL_SAMPLES = 256

# Inference benchmark
BENCH_BATCH_SIZES = [1, 32]
BENCH_IMAGES = 1000         # first 1000 test images (sorted order, same for both devices)
BENCH_WARMUP_BATCHES = 10
BENCH_REPEATS = 5
SMOKE_BENCH_IMAGES = 64
SMOKE_BENCH_REPEATS = 2


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def ensure_dirs():
    for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR, GRAPHS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_name(device_tag, smoke=False):
    """File-name stem for a run: 'cpu', 'gpu', 'cpu_smoke', 'gpu_smoke'."""
    return f"{device_tag}_smoke" if smoke else device_tag


def torch_device(device_tag):
    """Map the friendly tag ('cpu' / 'gpu') to a torch device string."""
    return "cuda" if device_tag == "gpu" else "cpu"


def sync(device):
    """Wait for all queued GPU work to finish (needed for correct timing)."""
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _cpu_name():
    """Try to get the marketing name of the CPU on Windows; fall back to platform info."""
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"],
                capture_output=True, text=True, timeout=15,
            )
            name = out.stdout.strip()
            if name:
                return name
        except Exception:
            pass
    return platform.processor() or "unavailable"


def hardware_info():
    """Collect the hardware / software configuration. Anything we cannot read is 'unavailable'."""
    info = {
        "python_version": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "torch_version": torch.__version__,
        "cpu_name": _cpu_name(),
        "cpu_logical_cores": os.cpu_count(),
        "torch_cpu_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version_in_torch": torch.version.cuda if torch.version.cuda else "unavailable",
        "cudnn_version": "unavailable",
        "gpu_name": "unavailable",
        "gpu_total_memory_mb": "unavailable",
        "gpu_compute_capability": "unavailable",
        "system_ram_gb": "unavailable",
    }
    try:
        import psutil
        info["system_ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_total_memory_mb"] = round(props.total_memory / 1024 ** 2)
            info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
            if torch.backends.cudnn.is_available():
                info["cudnn_version"] = torch.backends.cudnn.version()
        except Exception:
            pass
    return info
