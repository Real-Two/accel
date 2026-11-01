# CPUAccel Studio

CPUAccel Studio is a hardware-aware PyTorch acceleration platform for running and retraining models faster on commodity CPUs.

It combines a local analysis API with a visual Run Studio dashboard. Upload a model or training script, inspect the host CPU, generate an optimization plan, and export a CPU-ready PyTorch variant when the checkpoint contains a complete serialized `torch.nn.Module`.

## Verified result

The included Landslide4Sense CompactSeg benchmark uses the same model, data, loss, seed, and 15-epoch training protocol for both runs:

| Run | Time / epoch | Validation IoU |
| --- | ---: | ---: |
| Plain CPU baseline | 29.67 s | 0.1193 |
| CPUAccel optimized | 10.05 s | 0.1236 |

**Measured speedup: 2.95×** (challenge target: 2.5×).

The winning configuration uses a RAM-resident normalized patch cache, channels-last tensors, four measured CPU threads, one inter-op thread, zero data-loader workers, and accuracy/speed guardrails. CPU bf16 autocast and `torch.compile` were tested and rejected because they were slower for this AVX2 workload.

## Run the local demo

Requirements: Python 3.10+, PyTorch, NumPy, and the dependencies used by the training scripts.

```powershell
cd D:\accel
python api_server.py --port 3000
```

Open [http://localhost:3000](http://localhost:3000). The frontend and API are served from the same origin, so no tunnel or CORS setup is required.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Check that the local service is online |
| `GET /api/benchmark` | Return the verified benchmark JSON |
| `POST /api/analyze` | Inspect an uploaded `.pt`, `.pth`, `.py`, `.onnx`, or `.zip` and return an optimization plan |
| `POST /api/upgrade` | Export a channels-last CPU variant for a serialized `.pt`/`.pth` containing a complete `torch.nn.Module` |

State-dict-only checkpoints cannot be rebuilt automatically because they do not include the model architecture. They need the original model class/configuration before export can be made safe.

## Reproduce the CompactSeg run

Build the normalized cache once, then run the optimized experiment:

```powershell
python build_patch_cache.py --train_img split\TrainImg --train_mask split\TrainMask --val_img split\ValImg --val_mask split\ValMask --out_dir patch_cache

python optimized_cpu.py `
  --train_img split\TrainImg `
  --train_mask split\TrainMask `
  --val_img split\ValImg `
  --val_mask split\ValMask `
  --cache_dir patch_cache `
  --cache-ram `
  --model compact `
  --base_channels 16 `
  --epochs 15 `
  --seed 2026 `
  --threads 4 `
  --num_workers 0 `
  --no-amp `
  --no-compile `
  --out_dir final_optimized
```

The baseline runner is `baseline_cpu.py`; the optimized runner is `optimized_cpu.py`. The reusable acceleration package lives in [`cpuaccel/`](cpuaccel/).

## Project layout

- `Frontend/` — CPUAccel Studio dashboard and benchmark data
- `api_server.py` — local static server and analysis/export API
- `cpuaccel/` — thread, layout, cache, precision, and compiler guardrails
- `optimized_cpu.py` — reproducible optimized training runner
- `baseline_cpu.py` — plain baseline runner
- `compact_model.py` — CompactSeg model used in the verified benchmark
- `compact_report/` — benchmark summary artifacts

## Scope

The benchmark result is workload-specific and intentionally measured rather than assumed. CPUAccel profiles a model/workload, keeps only transformations that pass speed and accuracy gates, and leaves model architecture and dataset splits unchanged.
