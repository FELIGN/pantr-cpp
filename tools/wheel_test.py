"""Run the Python suite against the installed pantr, once per ISA rung this machine can run.

Usage::

    python tools/wheel_test.py <checkout> [-n WORKERS] [-- extra pytest arguments]

Run it with the interpreter whose environment has pantr installed (a wheel) and the test
dependencies. It runs the suite of ``<checkout>/tests`` with ``PANTR_TEST_INSTALLED=1`` and
``-o pythonpath=``, so the checkout's ``src`` is never imported (``tests/conftest.py`` refuses to
start otherwise), first under the rung the default ladder picks, then under ``PANTR_ISA=generic``
when that is a different module. A tuned rung the CPU cannot run is not tested here: the ladder
would never load it.

The script is run as a file, not imported from the checkout, so ``import pantr`` below resolves to
the installed package.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def rungs_to_test() -> list[str]:
    """Return the rungs to run the suite under: the default choice, then ``generic``.

    Returns:
        list[str]: One or two rung names.
    """
    from pantr import _isa  # noqa: PLC0415
    from pantr._ext import ext  # noqa: PLC0415

    rungs = [ext.isa]
    if ext.isa != _isa.GENERIC:
        rungs.append(_isa.GENERIC)
    return rungs


def main() -> int:
    """Run the suite per rung.

    Returns:
        int: 0 when every run passed, else the first nonzero pytest exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkout", type=Path, help="the source checkout holding tests/")
    parser.add_argument("-n", "--workers", default="8", help="pytest-xdist workers (default 8)")
    parser.add_argument("pytest_args", nargs="*", help="extra pytest arguments (after --)")
    args = parser.parse_args()
    root = args.checkout.resolve()
    status = 0
    for rung in rungs_to_test():
        env = dict(os.environ, PANTR_ISA=rung, PANTR_TEST_INSTALLED="1")
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "pythonpath=",
            "-p",
            "no:cacheprovider",
            "-n",
            args.workers,
            str(root / "tests"),
            *args.pytest_args,
        ]
        print(f"== PANTR_ISA={rung}: {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=root, env=env, check=False)
        if result.returncode != 0 and status == 0:
            status = result.returncode
    return status


if __name__ == "__main__":
    raise SystemExit(main())
