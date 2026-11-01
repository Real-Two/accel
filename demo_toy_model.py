"""
Stage-by-stage breakdown on a toy model, unrelated to landslide, trained on
synthetic data -- proves cpuaccel generalizes AND tells us honestly which
optimizations actually help at this model's scale (a tiny model may not
benefit from everything the same way the full U-Net does -- that's a real
finding, not a bug, and worth knowing before the demo rather than during it).

DO NOT run while baseline_cpu.py / optimized_cpu.py is training on the real
data -- CPU-bound, will skew that run's timing.

    python demo_toy_model.py
"""

import contextlib
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cpuaccel import accelerate, AccelConfig
from cpuaccel.threads import resolve_thread_count


class ToyCNN(nn.Module):
    def __init__(self, in_channels=3, n_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(inplace=True),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_synthetic_loader(n_samples=512, batch_size=32):
    x = torch.randn(n_samples, 3, 64, 64)
    y = torch.randint(0, 3, (n_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def run(model, loader, optimizer, loss_fn, autocast_ctx, n_epochs):
    model.train()
    t0 = time.time()
    for _ in range(n_epochs):
        for xb, yb in loader:
            with autocast_ctx():
                out = model(xb)
                loss = loss_fn(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return time.time() - t0


def fresh_model_and_opt():
    m = ToyCNN()
    return m, torch.optim.Adam(m.parameters(), lr=1e-3)


if __name__ == "__main__":
    loader = make_synthetic_loader()
    n_epochs = 5
    results = {}

    print("=== Stage 0: plain (no tuning) ===")
    m, opt = fresh_model_and_opt()
    t = run(m, loader, opt, nn.CrossEntropyLoss(), contextlib.nullcontext, n_epochs)
    results["plain"] = t
    print(f"{t:.2f}s\n")

    print("=== Stage 1: threads only ===")
    m, opt = fresh_model_and_opt()
    sample_x, _ = next(iter(loader))
    accel = accelerate(m, opt, config=AccelConfig(threads="auto", amp=False, compile=False), sample_input=sample_x)
    accel.print_report()
    t = run(accel.model, loader, accel.optimizer, nn.CrossEntropyLoss(), accel.autocast, n_epochs)
    results["threads"] = t
    print(f"{t:.2f}s\n")

    print("=== Stage 2: threads + AMP (validated) ===")
    m, opt = fresh_model_and_opt()
    accel = accelerate(m, opt, config=AccelConfig(threads="auto", amp=True, compile=False), sample_input=sample_x)
    accel.print_report()
    t = run(accel.model, loader, accel.optimizer, nn.CrossEntropyLoss(), accel.autocast, n_epochs)
    results["threads+amp"] = t
    print(f"{t:.2f}s\n")

    print("=== Stage 3: threads + AMP (validated) + compile ===")
    m, opt = fresh_model_and_opt()
    accel = accelerate(m, opt, config=AccelConfig(threads="auto", amp=True, compile=True), sample_input=sample_x)
    accel.print_report()
    t = run(accel.model, loader, accel.optimizer, nn.CrossEntropyLoss(), accel.autocast, n_epochs)
    results["threads+amp+compile"] = t
    print(f"{t:.2f}s\n")

    print("=== Summary ===")
    baseline = results["plain"]
    for name, t in results.items():
        print(f"{name:22s} {t:6.2f}s   {baseline / t:5.2f}x vs plain")
