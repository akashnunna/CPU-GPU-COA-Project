"""
dataset.py  --  Image preprocessing, PyTorch Dataset and DataLoaders (Stage 3a)

What it does
  * Reads image paths from the split CSVs (split_train.csv / split_val.csv / split_test.csv)
  * Opens each image, converts to RGB, resizes to IMAGE_SIZE x IMAGE_SIZE
  * Caches the resized images ONCE as uint8 tensors in data/cache_<split>_<size>.pt
    (so slow JPEG decoding is not repeated on every epoch, and CPU and GPU runs
     see exactly the same pixels; epoch times then reflect the model computation)
  * Augmentation (flips, small rotation, small brightness/contrast change) is applied
    ONLY to training images. Validation and test images are never augmented.
  * Normalizes with ImageNet mean/std
  * Computes class weights from the TRAINING set only

Run directly (python dataset.py) to build the caches and print a quick check.
"""

import sys
import time

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import v2

import config


def load_one_image(path, size):
    """Load one image -> uint8 tensor [3, size, size]. Raises a clear error if it fails."""
    try:
        with Image.open(path) as img:
            img = img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8).copy()
    except Exception as e:
        raise RuntimeError(f"Could not load image '{path}': {type(e).__name__}: {e}")
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def load_or_build_cache(split, csv_path, size=config.IMAGE_SIZE):
    """Return dict(images uint8 [N,3,H,W], labels long [N], image_ids list). Builds the cache if needed."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Split file not found: {csv_path}. Run create_splits.py first.")
    df = pd.read_csv(csv_path)
    ids = df["image_id"].tolist()
    cache_path = config.DATA_DIR / f"cache_{split}_{size}.pt"

    if cache_path.is_file():
        data = torch.load(cache_path, weights_only=False)
        if data["image_ids"] == ids:          # cache matches the current split file
            return data
        print(f"  [{split}] cache is out of date (split changed) -> rebuilding")

    print(f"  [{split}] building cache from {len(df)} images (one-time step) ...")
    start = time.time()
    images = torch.empty((len(df), 3, size, size), dtype=torch.uint8)
    for i, row in enumerate(df.itertuples(index=False)):
        images[i] = load_one_image(row.image_path, size)
        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{len(df)}")
    labels = torch.tensor(df["label"].values, dtype=torch.long)
    data = {"images": images, "labels": labels, "image_ids": ids}
    torch.save(data, cache_path)
    print(f"  [{split}] cache saved ({time.time() - start:.0f}s): {cache_path.name}")
    return data


class HAMDataset(Dataset):
    """Dataset over cached, resized images. train=True turns augmentation on."""

    def __init__(self, split, train=False):
        csv_map = {"train": config.TRAIN_CSV, "val": config.VAL_CSV, "test": config.TEST_CSV}
        data = load_or_build_cache(split, csv_map[split])
        self.images = data["images"]
        self.labels = data["labels"]
        self.image_ids = data["image_ids"]
        self.train = train
        self.augment = v2.Compose([
            v2.RandomHorizontalFlip(0.5),
            v2.RandomVerticalFlip(0.5),
            v2.RandomRotation(degrees=20),
            v2.ColorJitter(brightness=0.1, contrast=0.1),
        ]) if train else None
        self.mean = torch.tensor(config.NORM_MEAN).view(3, 1, 1)
        self.std = torch.tensor(config.NORM_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]                      # uint8 [3,H,W]
        if self.augment is not None:
            img = self.augment(img)
        img = img.float() / 255.0
        img = (img - self.mean) / self.std
        return img, self.labels[idx]


def class_weights_from_train(labels):
    """Inverse-frequency class weights: weight_c = N / (K * count_c). Uses TRAINING labels only."""
    counts = torch.bincount(labels, minlength=config.NUM_CLASSES).float()
    weights = counts.sum() / (config.NUM_CLASSES * counts.clamp(min=1))
    return weights, counts


def make_loaders(batch_size=config.BATCH_SIZE, pin_memory=False, smoke=False):
    """Build train / val / test DataLoaders. smoke=True uses small subsets for a quick test."""
    train_ds = HAMDataset("train", train=True)
    val_ds = HAMDataset("val", train=False)
    test_ds = HAMDataset("test", train=False)

    class_weights, class_counts = class_weights_from_train(train_ds.labels)   # from FULL train set

    if smoke:
        train_ds = Subset(train_ds, list(range(min(config.SMOKE_TRAIN_SAMPLES, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(config.SMOKE_VAL_SAMPLES, len(val_ds)))))

    gen = torch.Generator()
    gen.manual_seed(config.SEED)                    # same shuffling order every run
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen,
                              num_workers=config.NUM_WORKERS, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader, class_weights, class_counts


if __name__ == "__main__":
    config.ensure_dirs()
    print("Building / loading image caches ...")
    try:
        tr, va, te, w, counts = make_loaders()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    print("\nDataset check")
    print(f"  train images: {len(tr.dataset)}   val images: {len(va.dataset)}   test images: {len(te.dataset)}")
    x, y = next(iter(tr))
    print(f"  one training batch -> images {tuple(x.shape)}, labels {tuple(y.shape)}")
    print(f"  pixel range after normalization: min {x.min():.2f}, max {x.max():.2f}")
    print("\nTraining-set class counts and class weights:")
    for i, c in enumerate(config.CLASSES):
        print(f"  {c:<6} count={int(counts[i]):<6} weight={w[i]:.3f}")
    print("\nDATASET CHECK PASSED")
