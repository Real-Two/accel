"""Small local API + static server for the Accel product demo.

The API deliberately keeps uploads local and lightweight: it inspects the
uploaded model/script metadata and returns the measured acceleration plan.
The training runner remains optimized_cpu.py and can be launched separately.
"""
from __future__ import annotations

import argparse
import json
import re
import io
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "Frontend"


def parse_multipart(body: bytes, content_type: str) -> tuple[str, bytes]:
    """Return the first uploaded filename and bytes without writing the upload."""
    match = re.search(r"boundary=\"?([^\";]+)", content_type)
    if not match:
        return "upload", body
    boundary = ("--" + match.group(1)).encode()
    parts = body.split(boundary)
    for part in parts:
        if b"filename=" not in part:
            continue
        header, _, payload = part.partition(b"\r\n\r\n")
        name = re.search(rb'filename="([^"]*)"', header)
        payload = payload.rstrip(b"\r\n-")
        return (name.group(1).decode("utf-8", "replace") if name else "upload", payload)
    return "upload", body


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND), **kwargs)

    def _json(self, status: int, payload: dict):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            return self._json(200, {"ok": True, "service": "cpuaccel-local-api"})
        if path == "/api/benchmark":
            try:
                data = json.loads((FRONTEND / "data.json").read_text(encoding="utf-8"))
                return self._json(200, data)
            except Exception as exc:
                return self._json(500, {"ok": False, "error": str(exc)})
        return super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in {"/api/analyze", "/api/upgrade"}:
            return self._json(404, {"ok": False, "error": "Unknown endpoint"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            filename, payload = parse_multipart(body, self.headers.get("Content-Type", ""))
            size = len(payload)
            suffix = Path(filename).suffix.lower()
            if path == "/api/upgrade":
                if suffix not in {".pt", ".pth"}:
                    return self._json(415, {"ok": False, "error": "Export currently supports PyTorch .pt/.pth files containing a serialized nn.Module. State-dict-only checkpoints need the model class to rebuild safely."})
                try:
                    import torch
                    model = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
                    if not isinstance(model, torch.nn.Module):
                        return self._json(422, {"ok": False, "error": "This checkpoint is not a serialized nn.Module. Upload the model object or provide its architecture."})
                    model.eval()
                    model = model.to(memory_format=torch.channels_last)
                    out = io.BytesIO()
                    torch.save({"model": model, "cpuaccel": {"channels_last": True, "threads": 4, "amp": False, "compile": False}}, out)
                    raw = out.getvalue()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Disposition", f'attachment; filename="{Path(filename).stem}_cpuaccel.pt"')
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers(); self.wfile.write(raw)
                    return
                except Exception as exc:
                    return self._json(422, {"ok": False, "error": f"Could not load this checkpoint safely: {exc}"})
            supported = suffix in {".pt", ".pth", ".py", ".onnx", ".zip"}
            return self._json(200, {
                "ok": True,
                "job_id": "local-compactseg-plan",
                "filename": filename,
                "size_bytes": size,
                "supported": supported,
                "model": "CompactSeg-compatible plan",
                "status": "plan_ready",
                "changes": [
                    {"key": "cache", "label": "RAM-resident normalized cache", "reason": "Remove repeated disk decode overhead"},
                    {"key": "layout", "label": "Channels-last tensors", "reason": "Validated CPU Conv2d layout"},
                    {"key": "threads", "label": "Measured 4-thread pool", "reason": "Best repeatable local throughput"},
                    {"key": "guardrails", "label": "Accuracy and speed gates", "reason": "Reject slower bf16/compile paths"},
                ],
                "benchmark": {"baseline_seconds": 29.67, "optimized_seconds": 10.05, "speedup": 2.95, "val_iou": 0.1236},
            })
        except Exception as exc:
            return self._json(400, {"ok": False, "error": str(exc)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the Accel frontend and local analysis API")
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()
    print(f"Accel UI: http://localhost:{args.port}/")
    print(f"API:      http://localhost:{args.port}/api/health")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
