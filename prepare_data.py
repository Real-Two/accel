"""
Gets the Landslide4Sense data onto this machine and carves the same
train/val split the original Kaggle notebook used (last 400 patches by
natural numeric order -> val, remaining 3399 -> train), so results are
directly comparable to the original run.

Step 1 — download (needs internet + `pip install kagglehub`):

    python prepare_data.py --download

    This calls kagglehub.dataset_download('tekbahadurkshetri/landslide4sense')
    and prints the local cache path it landed in.

Step 2 — split:

    python prepare_data.py --split --raw_dir <path printed above>

    Produces ./split/{TrainImg,TrainMask,ValImg,ValMask} next to this script.
    Uses hardlinks where possible (instant, no extra disk), falls back to a
    copy if the OS/filesystem doesn't support hardlinks across the source
    and destination (e.g. different drives).
"""

import argparse
import glob
import os
import re
import shutil

N_VAL = 400


def natural_key(path):
    nums = re.findall(r"\d+", os.path.basename(path))
    return int(nums[0]) if nums else path


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)  # hardlink — instant, no extra disk, same volume only
    except OSError:
        shutil.copy2(src, dst)  # cross-volume fallback


def download():
    import kagglehub
    path = kagglehub.dataset_download("tekbahadurkshetri/landslide4sense")
    print(f"Downloaded to: {path}")
    print("Pass this as --raw_dir to the --split step (point at the folder "
          "that directly contains TrainData/img and TrainData/mask).")


def split(raw_dir, out_base):
    src_img = os.path.join(raw_dir, "TrainData", "img")
    src_mask = os.path.join(raw_dir, "TrainData", "mask")
    if not os.path.isdir(src_img):
        raise FileNotFoundError(
            f"Expected {src_img} to exist. Pass --raw_dir pointing at the "
            f"folder that directly contains TrainData/img and TrainData/mask."
        )

    train_img_dir = os.path.join(out_base, "TrainImg")
    train_mask_dir = os.path.join(out_base, "TrainMask")
    val_img_dir = os.path.join(out_base, "ValImg")
    val_mask_dir = os.path.join(out_base, "ValMask")
    for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        os.makedirs(d, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(src_img, "image_*.h5")), key=natural_key)
    print(f"Found {len(img_paths)} total patches in {src_img}")
    if len(img_paths) <= N_VAL:
        raise RuntimeError(f"Not enough patches ({len(img_paths)}) to hold out {N_VAL} for val.")

    train_imgs = img_paths[:-N_VAL]
    val_imgs = img_paths[-N_VAL:]

    def place(img_path, dst_img_dir, dst_mask_dir):
        basename = os.path.basename(img_path)
        idx = natural_key(img_path)
        mask_path = os.path.join(src_mask, f"mask_{idx}.h5")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing mask for {basename}: expected {mask_path}")
        link_or_copy(img_path, os.path.join(dst_img_dir, basename))
        link_or_copy(mask_path, os.path.join(dst_mask_dir, f"mask_{idx}.h5"))

    for p in train_imgs:
        place(p, train_img_dir, train_mask_dir)
    for p in val_imgs:
        place(p, val_img_dir, val_mask_dir)

    print(f"Train: {len(train_imgs)} patches -> {train_img_dir}")
    print(f"Val:   {len(val_imgs)} patches -> {val_img_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--raw_dir", default=None)
    parser.add_argument("--out_base", default="split")
    args = parser.parse_args()

    if args.download:
        download()
    if args.split:
        if not args.raw_dir:
            raise SystemExit("--split requires --raw_dir")
        split(args.raw_dir, args.out_base)
    if not args.download and not args.split:
        print("Pass --download and/or --split. See file docstring for usage.")
