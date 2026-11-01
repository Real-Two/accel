"""
The actual wrapper. Point it at a model (+ optionally an optimizer) and get
back the accelerated versions plus a context manager to wrap your forward
pass in -- everything else about your training loop stays unchanged.

    from cpuaccel import accelerate, AccelConfig

    sample_x, _ = next(iter(train_loader))
    result = accelerate(model, optimizer, config=AccelConfig(), sample_input=sample_x)
    model, optimizer = result.model, result.optimizer

    for imgs, masks in loader:
        with result.autocast():
            logits = model(imgs)
            loss = loss_fn(logits, masks)
        ...

sample_input (one real batch of input, not the whole loader) is optional
but strongly recommended: without it, AMP is applied blindly on the
assumption it helps. With it, cpuaccel times a few forward passes with and
without autocast on this exact hardware and only keeps AMP if it's
genuinely faster -- some consumer CPUs (no AVX-512/AMX) see bf16 autocast
as pure casting overhead with zero hardware speedup, i.e. strictly worse
than fp32. Validating against real timing catches that automatically
instead of shipping a "speedup" that's secretly a slowdown.

Deliberately does NOT touch the DataLoader, dataset, batch size, or any
model/data-specific logic -- those are out of scope so the tool stays safe
to point at a model it's never seen before.
"""

import contextlib
import os
import subprocess
import time

import torch
import torch.nn as nn

from .config import AccelConfig
from .threads import resolve_thread_count


def _make_autocast_ctx(dtype):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return lambda: torch.amp.autocast("cpu", dtype=dtype)
    return lambda: torch.cpu.amp.autocast(dtype=dtype)


class AccelResult:
    def __init__(self, model, optimizer, amp_enabled, amp_dtype, channels_last, report):
        self.model = model
        self.optimizer = optimizer
        self._amp_enabled = amp_enabled
        self._amp_ctx_factory = _make_autocast_ctx(amp_dtype)
        self._channels_last = channels_last
        self.report = report

    def autocast(self):
        if not self._amp_enabled:
            return contextlib.nullcontext()
        return self._amp_ctx_factory()

    def prepare_inputs(self, imgs, masks):
        """Convert only 4-D image batches; masks keep their normal layout."""
        if self._channels_last and imgs.ndim == 4:
            imgs = imgs.contiguous(memory_format=torch.channels_last)
        return imgs, masks

    def print_report(self):
        print("cpuaccel applied:")
        for k, v in self.report.items():
            print(f"  {k}: {v}")


class _CompiledWithFallback(nn.Module):
    def __init__(self, compiled_model, original_model):
        super().__init__()
        self._compiled = compiled_model
        self._original = original_model
        self._compile_broken = False

    def forward(self, *args, **kwargs):
        if self._compile_broken:
            return self._original(*args, **kwargs)
        try:
            return self._compiled(*args, **kwargs)
        except Exception as e:
            print(f"cpuaccel: torch.compile failed at runtime ({type(e).__name__}: {e}). "
                  f"Falling back to eager mode for the rest of this run.")
            self._compile_broken = True
            return self._original(*args, **kwargs)

    def train(self, mode=True):
        self._original.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self, *args, **kwargs):
        return self._original.parameters(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self._original.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self._original.load_state_dict(*args, **kwargs)


def _activate_msvc_if_installed():
    """Load the VS Build Tools environment when PowerShell was not opened
    through a Developer Command Prompt.

    ``cl.exe`` needs INCLUDE/LIB/PATH together; adding only its bin directory
    would make Inductor fail later with a much less useful error.
    """
    import shutil
    if shutil.which("cl"):
        return "cl already on PATH"

    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\\Program Files (x86)"),
                           "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.exists(vswhere):
        return None
    try:
        root = subprocess.check_output(
            [vswhere, "-latest", "-products", "*", "-requires",
             "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        devcmd = os.path.join(root, "Common7", "Tools", "VsDevCmd.bat")
        if not root or not os.path.exists(devcmd):
            return None
        # Pass one command string: Python's list-to-command-line quoting makes
        # cmd.exe treat a quoted .bat path as a literal (and fail) on Windows.
        result = subprocess.run(
            f'cmd /d /c call "{devcmd}" -arch=x64 -host_arch=x64 >nul && set',
            capture_output=True, text=True, check=True,
        )
        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                if key and not key.startswith("="):
                    os.environ[key] = value
        return f"activated Build Tools at {root}"
    except (OSError, subprocess.SubprocessError):
        return None


def _compiler_available():
    import shutil
    msvc_status = _activate_msvc_if_installed()
    for name in ("cl", "cc", "gcc", "g++", "clang"):
        if shutil.which(name):
            suffix = f" ({msvc_status})" if name == "cl" and msvc_status else ""
            return True, name + suffix
    return False, None


def _try_compile(model, mode):
    if not hasattr(torch, "compile"):
        return model, "skipped (torch.compile not available in this PyTorch version)"

    has_compiler, which_one = _compiler_available()
    if not has_compiler:
        return model, ("skipped (no C++ compiler on PATH -- cl/cc/gcc/g++/clang -- "
                        "attempting anyway would pay the full failed-compile cost for nothing)")

    try:
        compiled = torch.compile(model, mode=mode)
    except Exception as e:  # pragma: no cover
        print(f"cpuaccel: torch.compile() call failed ({e}), continuing without it")
        return model, f"failed at setup, fell back to uncompiled ({type(e).__name__})"

    wrapped = _CompiledWithFallback(compiled, model)
    return wrapped, f"applied (mode={mode}, compiler found: {which_one})"


def _try_ipex(model, optimizer):
    try:
        import intel_extension_for_pytorch as ipex
    except ImportError:
        return model, optimizer, "skipped (intel_extension_for_pytorch not installed)"
    try:
        if optimizer is not None:
            model, optimizer = ipex.optimize(model, optimizer=optimizer)
        else:
            model = ipex.optimize(model)
        return model, optimizer, "applied"
    except Exception as e:  # pragma: no cover
        print(f"cpuaccel: IPEX optimize failed ({e}), continuing without it")
        return model, optimizer, f"failed ({type(e).__name__})"


def _probe_amp(model, sample_input, dtype, n_iters=3):
    """Times a few forward passes with and without autocast on a real
    sample batch. Returns (plain_seconds, amp_seconds)."""
    was_training = model.training
    model.eval()
    ctx_factory = _make_autocast_ctx(dtype)
    try:
        with torch.no_grad():
            model(sample_input)  # warmup (lazy init, caches, etc.)
            t0 = time.time()
            for _ in range(n_iters):
                model(sample_input)
            plain_t = time.time() - t0

            with ctx_factory():
                model(sample_input)  # warmup
            t0 = time.time()
            for _ in range(n_iters):
                with ctx_factory():
                    model(sample_input)
            amp_t = time.time() - t0
    finally:
        model.train(was_training)
    return plain_t, amp_t


def _probe_threads(model, optimizer, loss_fn, sample_input, sample_target, candidates, n_iters=3):
    """Times real forward+backward passes at each candidate thread count on
    this exact model/workload and returns {n_threads: seconds}. Doesn't
    step the optimizer (so weights stay identical across trials) but does
    run a real backward pass, since thread scaling can differ between
    forward and backward."""
    orig_threads = torch.get_num_threads()
    was_training = model.training
    model.train()
    timings = {}
    for n in candidates:
        torch.set_num_threads(n)
        optimizer.zero_grad()
        out = model(sample_input)
        loss = loss_fn(out, sample_target)
        loss.backward()  # warmup
        optimizer.zero_grad()

        t0 = time.time()
        for _ in range(n_iters):
            optimizer.zero_grad()
            out = model(sample_input)
            loss = loss_fn(out, sample_target)
            loss.backward()
        timings[n] = time.time() - t0
    optimizer.zero_grad()
    model.train(was_training)
    return timings


def accelerate(model, optimizer=None, config=None, sample_input=None, probe_batches=3,
                sample_target=None, loss_fn=None, thread_candidates=None):
    config = config or AccelConfig()
    report = {}

    if config.interop_threads is not None:
        try:
            torch.set_num_interop_threads(config.interop_threads)
            report["interop_threads"] = str(config.interop_threads)
        except RuntimeError:
            # A host app may already have initialized the pool.  Training is
            # still safe; retain the actual count for a reproducible report.
            report["interop_threads"] = f"pre-initialized ({torch.get_num_interop_threads()})"

    if config.channels_last:
        model = model.to(memory_format=torch.channels_last)
        if sample_input is not None and sample_input.ndim == 4:
            sample_input = sample_input.contiguous(memory_format=torch.channels_last)
        report["memory_format"] = "channels_last (validated CPU Conv2d layout)"
    else:
        report["memory_format"] = "contiguous (disabled)"

    if config.threads == "auto" and sample_target is not None and loss_fn is not None and optimizer is not None:
        if thread_candidates is None:
            p_cores, _ = resolve_thread_count("auto")
            physical_default = torch.get_num_threads()
            logical = __import__("os").cpu_count() or physical_default
            # Physical-core count is the PyTorch default here; include the
            # P-core-only and SMT alternatives, then measure rather than guess.
            thread_candidates = sorted(set([p_cores, physical_default, logical]))
        timings = _probe_threads(model, optimizer, loss_fn, sample_input, sample_target,
                                  thread_candidates)
        best_n = min(timings, key=timings.get)
        torch.set_num_threads(best_n)
        timing_str = ", ".join(f"{n}t={t:.3f}s" for n, t in sorted(timings.items()))
        report["threads"] = f"{best_n} (validated fastest of [{timing_str}])"
    else:
        n_threads, thread_method = resolve_thread_count(config.threads)
        if n_threads is not None:
            torch.set_num_threads(n_threads)
        report["threads"] = f"{n_threads} ({thread_method})" if n_threads else thread_method

    if config.ipex:
        model, optimizer, ipex_status = _try_ipex(model, optimizer)
        report["ipex"] = ipex_status

    if config.compile:
        model, compile_status = _try_compile(model, config.compile_mode)
        report["compile"] = compile_status

    amp_enabled = False
    if config.amp:
        if sample_input is not None:
            plain_t, amp_t = _probe_amp(model, sample_input, torch.bfloat16, probe_batches)
            amp_enabled = amp_t < plain_t
            if amp_enabled:
                report["amp"] = (f"enabled -- validated faster ({plain_t:.3f}s fp32 vs "
                                  f"{amp_t:.3f}s bf16 over {probe_batches} fwd passes)")
            else:
                report["amp"] = (f"disabled -- validated SLOWER on this hardware "
                                  f"({plain_t:.3f}s fp32 vs {amp_t:.3f}s bf16 over "
                                  f"{probe_batches} fwd passes; likely no native bf16 support, "
                                  f"skipping automatically)")
        else:
            amp_enabled = True
            report["amp"] = "enabled -- UNVALIDATED (pass sample_input to accelerate() to auto-check)"
    else:
        report["amp"] = "disabled"

    result = AccelResult(
        model=model,
        optimizer=optimizer,
        amp_enabled=amp_enabled,
        amp_dtype=torch.bfloat16,
        channels_last=config.channels_last,
        report=report,
    )
    return result
