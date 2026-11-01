"""
Standalone check -- does cpuaccel's chip lookup actually recognize this
machine? Doesn't train anything, just calls detect_p_cores() and prints
what it would do. Safe to run alongside the baseline training.

    python check_chip_detection.py
"""

import sys
sys.path.insert(0, ".")

from cpuaccel.threads import detect_p_cores, _cpu_brand_string

brand = _cpu_brand_string()
n, method = detect_p_cores()

print(f"Detected CPU brand string: {brand!r}")
print(f"cpuaccel would set torch threads to: {n}")
print(f"Method: {method}")

if "known-chip lookup" in method:
    print("\n-> Matched the known-chip table. Demo-safe.")
else:
    print("\n-> Did NOT match the known-chip table -- using fallback. "
          "Still works, just not P-core-exact. Worth checking the brand "
          "string above against KNOWN_P_CORES in cpuaccel/threads.py if "
          "you want it exact for the demo.")
