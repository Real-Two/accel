"""
P-core detection for hybrid Intel chips (12th/13th/14th gen and similar).

Getting an exact P-core count cross-platform without admin privileges is
genuinely annoying (the reliable API is Windows-only and needs ctypes struct
wrangling). Instead of over-engineering that for a hackathon timeline, this
uses a known-chip lookup table (covers the demo CPU exactly) with a
documented, honest fallback for anything not in the table. The pitch can
honestly say "auto-detects known Intel hybrid chips, safe fallback
otherwise" -- which is true and still demos well.

Pass threads=<int> to AccelConfig to skip detection entirely.
"""

import os
import platform

KNOWN_P_CORES = {
    "i7-13620H": 6,   # the demo CPU for this challenge
    "i5-13500H": 4,
    "i7-12700H": 6,
    "i9-13900H": 6,
    "i7-1360P": 4,
    "i5-1240P": 4,
    "i7-1260P": 4,
    "i9-13900K": 8,
    "i7-13700K": 8,
    "i5-13600K": 6,
    "i7-12700K": 8,
    "i5-12600K": 6,
    "i9-14900K": 8,
    "i7-14700K": 8,
}


def _cpu_brand_string():
    try:
        import cpuinfo  # py-cpuinfo
        return cpuinfo.get_cpu_info().get("brand_raw", "")
    except ImportError:
        pass
    return platform.processor() or ""


def detect_p_cores():
    """
    Returns (n_threads_to_use, method_str) where method_str explains how the
    number was derived, for the report that gets printed/logged.
    """
    brand = _cpu_brand_string()
    for model_name, p_cores in KNOWN_P_CORES.items():
        if model_name in brand:
            return p_cores, f"known-chip lookup ({model_name} -> {p_cores} P-cores)"

    try:
        import psutil
        phys = psutil.cpu_count(logical=False)
        if phys:
            return phys, f"unknown chip ('{brand or 'undetected'}'), falling back to physical core count ({phys})"
    except ImportError:
        pass

    logical = os.cpu_count() or 4
    fallback = max(1, logical // 2)
    return fallback, (f"unknown chip ('{brand or 'undetected'}'), psutil unavailable, "
                       f"falling back to logical_count/2 ({fallback})")


def resolve_thread_count(threads):
    """threads: 'auto' | int | None -> (n, method_str)"""
    if threads is None:
        return None, "unset (PyTorch default -- no tuning applied)"
    if isinstance(threads, int):
        return threads, "explicit override"
    if threads == "auto":
        return detect_p_cores()
    raise ValueError(f"threads must be 'auto', an int, or None -- got {threads!r}")
