"""Measure the real CPU training workload before choosing acceleration flags.

This deliberately uses the frozen model, loss, data and batch size.  It
separates time spent waiting for the DataLoader from forward/backward/step
time, and sweeps PyTorch intra-op thread counts on already-loaded batches.
It is a diagnostic, not a training run: no checkpoints are written.
"""

import argparse
import os
import statistics
import time

import torch
from torch.utils.data import DataLoader

from dataset import Landslide4SenseDataset
from losses import BCEDiceLoss, compute_metrics
from model import UNet


def timed_train_step(model, optimizer, loss_fn, imgs, masks, metrics=False):
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    logits = model(imgs)
    loss = loss_fn(logits, masks)
    loss.backward()
    optimizer.step()
    if metrics:
        compute_metrics(logits, masks)
    return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_img", required=True)
    p.add_argument("--train_mask", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--threads", default="4,6,8,10,12,16")
    p.add_argument("--channels-last", action="store_true",
                   help="Measure oneDNN-friendly NHWC activations and weights.")
    p.add_argument("--metrics", action="store_true", help="Include the frozen metric calculation.")
    p.add_argument("--interop", type=int, default=None, help="Set PyTorch inter-op threads before work begins.")
    p.add_argument("--jit", action="store_true", help="Trace the fixed-shape model before timing.")
    args = p.parse_args()
    if args.interop is not None:
        torch.set_num_interop_threads(args.interop)

    ds = Landslide4SenseDataset(args.train_img, args.train_mask, augment=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=False,
                        persistent_workers=args.num_workers > 0)
    iterator = iter(loader)
    # Exclude worker startup and oneDNN lazy initialization from steady-state
    # measurements; both happen once per process, not once per epoch.
    warmup_imgs, warmup_masks = next(iterator)
    model = UNet(in_channels=14, out_channels=1, base_channels=32)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        warmup_imgs = warmup_imgs.contiguous(memory_format=torch.channels_last)
    if args.jit:
        model = torch.jit.trace(model, warmup_imgs, check_trace=False)
    loss_fn = BCEDiceLoss(pos_weight=15.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    candidates = [int(x) for x in args.threads.split(",")]
    batches = [next(iterator) for _ in range(args.batches)]
    print(f"CPU default threads: {torch.get_num_threads()} | logical CPUs: {os.cpu_count()}")
    for n in candidates:
        torch.set_num_threads(n)
        # Stabilize the thread pool and primitive caches at this count.
        warmup_imgs = warmup_imgs.contiguous(memory_format=torch.channels_last) if args.channels_last else warmup_imgs
        timed_train_step(model, optimizer, loss_fn, warmup_imgs, warmup_masks, args.metrics)
        timings = [timed_train_step(
                       model, optimizer, loss_fn,
                       imgs.contiguous(memory_format=torch.channels_last) if args.channels_last else imgs,
                       masks, args.metrics)
                   for imgs, masks in batches]
        print(f"threads={n:2d} median_step={statistics.median(timings):.3f}s "
              f"mean_step={statistics.mean(timings):.3f}s")


if __name__ == "__main__":
    main()
