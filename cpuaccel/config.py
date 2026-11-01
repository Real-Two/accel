from dataclasses import dataclass
from typing import Union


@dataclass
class AccelConfig:
    # Keep PyTorch's observed baseline default unless a full-epoch experiment
    # has established a better explicit value.  Short micro-probes were too
    # noisy on this hybrid laptop to choose safely.
    threads: Union[str, int, None] = None
    channels_last: bool = True                  # oneDNN-friendly NHWC layout for Conv2d
    interop_threads: int | None = 1             # model is sequential; avoid scheduler overhead
    # This AVX2-only demo CPU emulates bf16 and is dramatically slower under
    # autocast. Keep it opt-in; --amp still probes before enabling it.
    amp: bool = False
    # The measured eager oneDNN path is faster for this small U-Net on the
    # demo CPU.  Keep compilation opt-in for future hardware/software tests.
    compile: bool = False
    compile_mode: str = "default"
    ipex: bool = False                         # opt-in, stretch goal, needs intel_extension_for_pytorch
