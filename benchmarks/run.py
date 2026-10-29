"""Time the benchmark cases in this process and write one JSON record.

Usage (one process; the interleaved protocol is ``compare.py``)::

    PYTHONPATH=<tree>/src python benchmarks/run.py --out FILE [--threads N] [--filter REGEX]

``PYTHONPATH`` must be absolute and name the tree to measure: the record says which ``pantr`` it
imported, and the caller checks it. Per case: one warm-up phase (Numba compiles and caches here,
and the import's background warmup is awaited first), then a loop count chosen so one repeat
lasts at least ``--min-time``, then ``--repeats`` timed repeats with ``time.perf_counter_ns``. The
record keeps every repeat's per-call time; the process statistic is their median.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "PANTR_NUM_THREADS",
    "PANTR_ISA",
)


def _git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _governor() -> str:
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        return path.read_text().strip()
    except OSError:
        return "unreadable"


WARMUP_CALLS = 3
"""Calls made before timing at the least (the first compiles a Numba kernel)."""

WARMUP_SECONDS = 0.05
"""Time spent calling before timing at the least."""


def _time_case(fn: Callable[[], object], repeats: int, min_time: float) -> dict[str, Any]:
    warm_until = time.perf_counter() + WARMUP_SECONDS
    calls = 0
    while calls < WARMUP_CALLS or time.perf_counter() < warm_until:
        fn()
        calls += 1
    t0 = time.perf_counter_ns()
    fn()
    one = max(time.perf_counter_ns() - t0, 1)
    loops = max(1, math.ceil(min_time * 1e9 / one))
    per_call: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(loops):
            fn()
        per_call.append((time.perf_counter_ns() - start) / loops)
    return {"loops": loops, "per_call_ns": per_call, "median_ns": statistics.median(per_call)}


def main() -> int:
    """Run the selected cases and write the record.

    Returns:
        int: 0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1, help="pantr.set_num_threads value")
    parser.add_argument("--filter", default=".", help="regex on case names")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--min-time", type=float, default=0.002, help="seconds per repeat")
    args = parser.parse_args()

    import pantr  # noqa: PLC0415

    try:
        from pantr._numba_compat import wait_for_jit_warmup  # noqa: PLC0415

        wait_for_jit_warmup()
    except ImportError:
        pass
    pantr.set_num_threads(args.threads)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cases import all_cases  # noqa: PLC0415

    isa = "none"
    try:
        from pantr._ext import ext  # noqa: PLC0415

        isa = f"{ext.__name__} ({ext.isa}, {Path(ext.__file__).name})"
    except ImportError:
        pass

    pattern = re.compile(args.filter)
    results: dict[str, Any] = {}
    for case in all_cases():
        if not pattern.search(case.name):
            continue
        fn = case.setup()
        results[case.name] = {
            "what": case.what,
            "stage": case.stage,
            "size": case.size,
            **_time_case(fn, args.repeats, args.min_time),
        }

    tree = Path(pantr.__file__).resolve().parents[2]
    record = {
        "pantr_file": str(Path(pantr.__file__).resolve()),
        "commit": _git_commit(tree),
        "extension": isa,
        "threads": args.threads,
        "python": sys.version,
        "machine": {
            "cpu": _cpu_model(),
            "platform": platform.platform(),
            "affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "governor": _governor(),
        },
        "env": {k: os.environ.get(k) for k in THREAD_VARS},
        "repeats": args.repeats,
        "cases": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
