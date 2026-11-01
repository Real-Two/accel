"""
Same training loop as baseline_cpu.py -- same model, same data, same
hyperparams, same loss -- run through cpuaccel. The diff between this file
and baseline_cpu.py IS the demo: everything below the accelerate() call is
identical to the baseline script.

Flags let you isolate each optimization stage for the waterfall benchmark
(handoff doc: "each optimization step should be benchmarked individually"):

    # threads only
    python optimized_cpu.py ... --no-amp --no-compile --out_dir stage_threads

    # threads + amp
    python optimized_cpu.py ... --no-compile --out_dir stage_amp

    # threads + amp + compile (full)
    python optimized_cpu.py ... --out_dir stage_full

Same --max_batches smoke-test flag as baseline_cpu.py for quick timing
checks before a full run.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CachedLandslide4SenseDataset, Landslide4SenseDataset
from losses import BCEDiceLoss, compute_metrics
from model import UNet
from compact_model import CompactSeg
from cpuaccel import accelerate, AccelConfig


def run_epoch(model, loader, loss_fn, device, accel, optimizer=None, max_batches=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    agg = {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    n_batches = 0

    for imgs, masks in loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        imgs, masks = accel.prepare_inputs(imgs.to(device), masks.to(device))

        with torch.set_grad_enabled(is_train):
            with accel.autocast():
                logits = model(imgs)
                loss = loss_fn(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        # Training metrics are neither logged nor used for scheduling.  The
        # full validation pass below remains the accuracy authority.
        if not is_train:
            batch_metrics = compute_metrics(logits, masks)
            for k in agg:
                agg[k] += batch_metrics[k]
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {k: v / max(n_batches, 1) for k, v in agg.items()}
    return avg_loss, avg_metrics, n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_img", required=True)
    parser.add_argument("--train_mask", required=True)
    parser.add_argument("--val_img", required=True)
    parser.add_argument("--val_mask", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--model", choices=("unet", "compact"), default="unet")
    parser.add_argument("--pos_weight", type=float, default=15.0)
    parser.add_argument("--out_dir", default="optimized_run")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="0 is fastest for the local memory-mapped cache; tune only after measuring.")
    parser.add_argument("--cache_dir", default=None,
                        help="One-time normalized patch cache from build_patch_cache.py.")
    parser.add_argument("--cache-ram", action="store_true",
                        help="Load the normalized cache into RAM before timing epochs.")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026,
                        help="Reproducible initialization and sample order for fair stage comparisons.")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--save_checkpoints", action="store_true", default=True)

    parser.add_argument("--threads", default="none",
                         help="'none' preserves PyTorch's baseline default; pass an int only after a full-epoch sweep.")
    parser.add_argument("--amp", dest="amp", action="store_true", default=False,
                        help="Probe CPU bf16 autocast and enable it only if faster.")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--compile", dest="compile", action="store_true", default=False,
                        help="Opt in to torch.compile after a workload-specific benchmark.")
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--ipex", action="store_true", default=False)
    parser.add_argument("--channels-last", dest="channels_last", action="store_true", default=True)
    parser.add_argument("--no-channels-last", dest="channels_last", action="store_false",
                        help="Disable the CPU Conv2d channels-last layout for an ablation.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")

    threads_arg = None if args.threads == "none" else (
        int(args.threads) if args.threads.lstrip("-").isdigit() else args.threads)
    accel_config = AccelConfig(threads=threads_arg, channels_last=args.channels_last,
                                amp=args.amp, compile=args.compile, ipex=args.ipex)

    if args.cache_dir:
        train_cache = os.path.join(args.cache_dir, "train")
        val_cache = os.path.join(args.cache_dir, "val")
        train_ds = CachedLandslide4SenseDataset(train_cache, augment=True,
                                                in_memory=args.cache_ram)
        val_ds = CachedLandslide4SenseDataset(val_cache, augment=False,
                                              in_memory=args.cache_ram)
    else:
        train_ds = Landslide4SenseDataset(args.train_img, args.train_mask, augment=True)
        val_ds = Landslide4SenseDataset(args.val_img, args.val_mask, augment=False)
    print(f"Train patches: {len(train_ds)} | Val patches: {len(val_ds)}")

    # Pinned memory only helps host-to-GPU transfers. Persistent workers avoid
    # recreating HDF5/cache mappings at every epoch on this CPU-only job.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=False,
                               persistent_workers=args.num_workers > 0,
                               generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=False,
                             persistent_workers=args.num_workers > 0)

    model_cls = UNet if args.model == "unet" else CompactSeg
    model = (model_cls(in_channels=14, out_channels=1,
                       base_channels=args.base_channels) if args.model == "unet"
             else model_cls(in_channels=14, out_channels=1, hidden=args.base_channels)).to(device)
    loss_fn = BCEDiceLoss(pos_weight=args.pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # ---- the one real addition vs baseline_cpu.py ----
    sample_x, sample_y = next(iter(train_loader))
    accel = accelerate(model, optimizer, config=accel_config, sample_input=sample_x,
                        sample_target=sample_y, loss_fn=loss_fn)
    model, optimizer = accel.model, accel.optimizer
    accel.print_report()
    # ----------------------------------------------------

    best_iou = 0.0
    epochs_without_improvement = 0
    history = []
    wall_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_metrics, n_train_batches = run_epoch(
            model, train_loader, loss_fn, device, accel, optimizer, args.max_batches)
        t_train = time.time() - t0

        t1 = time.time()
        val_loss, val_metrics, n_val_batches = run_epoch(
            model, val_loader, loss_fn, device, accel, optimizer=None, max_batches=args.max_batches)
        t_val = time.time() - t1

        epoch_seconds = t_train + t_val
        scheduler.step(val_metrics["iou"])

        print(f"[Epoch {epoch}/{args.epochs}] "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_iou={val_metrics['iou']:.4f} val_f1={val_metrics['f1']:.4f} "
              f"({epoch_seconds:.1f}s = {t_train:.1f}s train + {t_val:.1f}s val, "
              f"{n_train_batches} train batches)")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "epoch_seconds": epoch_seconds,
            "train_seconds": t_train,
            "val_seconds": t_val,
        })

        if args.save_checkpoints:
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "base_channels": args.base_channels},
                       os.path.join(args.out_dir, "last_model.pt"))

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            epochs_without_improvement = 0
            if args.save_checkpoints:
                torch.save({"model_state": model.state_dict(), "epoch": epoch,
                            "val_iou": best_iou, "base_channels": args.base_channels},
                           os.path.join(args.out_dir, "best_model.pt"))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"No val IoU improvement for {args.patience} epochs -- stopping early.")
                break

    total_wall = time.time() - wall_start
    epoch_times = [h["epoch_seconds"] for h in history]

    summary = {
        "device": "cpu",
        "cpu_specific_optimizations": accel.report,
        "epochs_run": len(history),
        "best_val_iou": best_iou,
        "total_wall_seconds": total_wall,
        "avg_epoch_seconds": sum(epoch_times) / len(epoch_times) if epoch_times else None,
        "min_epoch_seconds": min(epoch_times) if epoch_times else None,
        "max_epoch_seconds": max(epoch_times) if epoch_times else None,
        "hyperparams": {
            "model": args.model, "cache_ram": bool(args.cache_dir and args.cache_ram),
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "base_channels": args.base_channels, "pos_weight": args.pos_weight,
            "num_workers": args.num_workers,
        },
    }

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.out_dir, "baseline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Best val IoU: {best_iou:.4f}")
    print(f"Total wall time: {total_wall:.1f}s over {len(history)} epochs "
          f"(avg {summary['avg_epoch_seconds']:.1f}s/epoch)")


if __name__ == "__main__":
    main()
