"""
Copied verbatim from D:\\SIH\\model\\dataset.py — do not edit, this defines
what "same data" means for the challenge.
"""

import json
import os
import re

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

NUM_BANDS = 14


def _read_h5_array(path: str, preferred_key: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if preferred_key in f:
            return f[preferred_key][:]
        keys = list(f.keys())
        if not keys:
            raise ValueError(f"{path} has no datasets inside it.")
        return f[keys[0]][:]


def _natural_sort_key(path: str):
    nums = re.findall(r"\d+", os.path.basename(path))
    return int(nums[0]) if nums else path


class Landslide4SenseDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, stats_path=None,
                 augment=False, index_file=None):
        if index_file:
            with open(index_file) as f:
                self.img_paths = [l.strip() for l in f if l.strip()]
        else:
            import glob
            self.img_paths = sorted(
                glob.glob(os.path.join(img_dir, "image_*.h5")),
                key=_natural_sort_key)

        if not self.img_paths:
            raise FileNotFoundError(f"No image files found.")

        self.mask_dir = mask_dir
        self.augment = augment
        self.mean = self.std = None

        if stats_path and os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
            self.mean = np.array(stats["mean"], dtype=np.float32).reshape(NUM_BANDS, 1, 1)
            self.std = np.array(stats["std"], dtype=np.float32).reshape(NUM_BANDS, 1, 1)

    def __len__(self):
        return len(self.img_paths)

    def _mask_path_for(self, img_path):
        idx = _natural_sort_key(img_path)
        return os.path.join(self.mask_dir, f"mask_{idx}.h5")

    def _normalize(self, img_chw: np.ndarray) -> np.ndarray:
        if self.mean is not None:
            return (img_chw - self.mean) / (self.std + 1e-6)
        c = img_chw.shape[0]
        out = np.empty_like(img_chw)
        for i in range(c):
            band = img_chw[i]
            lo, hi = band.min(), band.max()
            out[i] = (band - lo) / (hi - lo + 1e-6)
        return out

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = _read_h5_array(img_path, "img").astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        img = self._normalize(img)

        if self.mask_dir:
            mask = _read_h5_array(self._mask_path_for(img_path), "mask").astype(np.float32)
            mask = mask[None, :, :]
        else:
            mask = np.zeros((1, img.shape[1], img.shape[2]), dtype=np.float32)

        if self.augment:
            img, mask = self._augment(img, mask)

        return torch.from_numpy(img.copy()), torch.from_numpy(mask.copy())

    @staticmethod
    def _augment(img, mask):
        if np.random.rand() < 0.5:
            img, mask = img[:, :, ::-1], mask[:, :, ::-1]
        if np.random.rand() < 0.5:
            img, mask = img[:, ::-1, :], mask[:, ::-1, :]
        k = np.random.randint(0, 4)
        if k:
            img = np.rot90(img, k, axes=(1, 2))
            mask = np.rot90(mask, k, axes=(1, 2))
        return img, mask


class CachedLandslide4SenseDataset(Dataset):
    """The same normalized patches as :class:`Landslide4SenseDataset`.

    Cache files are memory-mapped lazily in each DataLoader worker, so Windows
    worker spawning never copies the multi-gigabyte cache into every process.
    Augmentation remains live and byte-for-byte follows the original dataset.
    """

    def __init__(self, cache_dir, augment=False, in_memory=False):
        meta_path = os.path.join(cache_dir, "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("format") != "landslide4sense-normalized-v1":
            raise ValueError(f"Unsupported cache format in {cache_dir}")
        self.cache_dir = cache_dir
        self.augment = augment
        self.n_samples = int(meta["n_samples"])
        if in_memory:
            # Load once before epoch timing begins. This avoids a page fault
            # for every patch when the OS cannot keep the mmap hot.
            self._images = np.load(os.path.join(cache_dir, "images.npy"))
            self._masks = np.load(os.path.join(cache_dir, "masks.npy"))
        else:
            self._images = None
            self._masks = None

    def __len__(self):
        return self.n_samples

    def __getstate__(self):
        # DataLoader on Windows pickles datasets.  Keep the pickle tiny and
        # reopen the shared OS-backed mappings in the child instead.
        state = self.__dict__.copy()
        state["_images"] = state["_masks"] = None
        return state

    def _open(self):
        if self._images is None:
            self._images = np.load(os.path.join(self.cache_dir, "images.npy"), mmap_mode="r")
            self._masks = np.load(os.path.join(self.cache_dir, "masks.npy"), mmap_mode="r")

    def __getitem__(self, idx):
        self._open()
        img = self._images[idx]
        mask = self._masks[idx]
        if self.augment:
            img, mask = Landslide4SenseDataset._augment(img, mask)
        if not self.augment and self._images is not None:
            # DataLoader's stack makes one writable batch copy; avoid making
            # an additional per-sample copy for the RAM-resident cache.
            return torch.from_numpy(img), torch.from_numpy(mask)
        # Flips/rotations can have negative strides, so augmented samples need
        # a writable contiguous copy before entering PyTorch.
        return torch.from_numpy(np.array(img, dtype=np.float32, copy=True, order="C")), \
            torch.from_numpy(np.array(mask, dtype=np.float32, copy=True, order="C"))
