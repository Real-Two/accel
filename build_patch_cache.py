"""Build a one-time contiguous, normalized cache for the frozen dataset.

The cache changes neither the model nor the data: it stores exactly what
Landslide4SenseDataset produces immediately before its random augmentation.
It replaces thousands of HDF5 opens and per-band normalization passes in every
epoch with OS-cached memory-map reads.
"""

import argparse
import gc
import json
import os

import numpy as np

from dataset import Landslide4SenseDataset


def build(img_dir, mask_dir, out_dir):
    if os.path.exists(out_dir):
        if os.listdir(out_dir):
            raise FileExistsError(f"{out_dir} already exists and is not empty; reuse it or choose a new path")
    else:
        os.makedirs(out_dir)

    source = Landslide4SenseDataset(img_dir, mask_dir, augment=False)
    n = len(source)
    image_tmp = os.path.join(out_dir, "images.incomplete.npy")
    mask_tmp = os.path.join(out_dir, "masks.incomplete.npy")
    images = np.lib.format.open_memmap(image_tmp, mode="w+", dtype=np.float32,
                                        shape=(n, 14, 128, 128))
    masks = np.lib.format.open_memmap(mask_tmp, mode="w+", dtype=np.float32,
                                       shape=(n, 1, 128, 128))
    try:
        for idx in range(n):
            img, mask = source[idx]
            images[idx] = img.numpy()
            masks[idx] = mask.numpy()
            if (idx + 1) % 100 == 0 or idx + 1 == n:
                images.flush()
                masks.flush()
                print(f"cached {idx + 1}/{n}", flush=True)
        del images, masks
        gc.collect()
        os.replace(image_tmp, os.path.join(out_dir, "images.npy"))
        os.replace(mask_tmp, os.path.join(out_dir, "masks.npy"))
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump({"format": "landslide4sense-normalized-v1", "n_samples": n,
                       "source_img_dir": os.path.abspath(img_dir),
                       "source_mask_dir": os.path.abspath(mask_dir)}, f, indent=2)
        print(f"Cache ready: {out_dir}")
    except Exception:
        del images, masks
        gc.collect()
        for path in (image_tmp, mask_tmp):
            if os.path.exists(path):
                os.remove(path)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", required=True)
    p.add_argument("--mask_dir", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    build(args.img_dir, args.mask_dir, args.out_dir)
