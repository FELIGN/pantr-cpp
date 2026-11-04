"""Keep the adversarial sweep alive as a test.

The sweep proper lives in ``tools/adversarial_sweep/`` and exists because pantr's kernels are
compiled code with no bounds checking of their own: an out-of-range write corrupts memory
silently and integer arithmetic overflows untrapped. The whole suite already runs clean, so
everything the sweep finds comes from inputs the suite does not contain. This test runs the
sweep's bounded ``smoke`` profile and fails when it finds something that is not already pinned
by a regression test.

It runs in a subprocess, so that a sanitizer's abort or a hard crash in a case fails this test
instead of taking the pytest process with it, and it is opt-in: following the precedent of
``tests/mpi/``, it is gated on an environment variable and marked ``slow``, so the default suite
pays nothing. CI runs it in a job of its own against an extension built with AddressSanitizer,
UndefinedBehaviorSanitizer and libstdc++'s assertions, which is what turns an out-of-range access
in a kernel into a failure; against an ordinary build, the sweep grades exceptions, non-finite
results and declared invariants only.

Run it with::

    PANTR_RUN_SWEEP=1 pytest tests/test_adversarial_sweep.py
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from tests._tree_env import subprocess_env

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_LAUNCHER = _ROOT / "tools" / "sweep.py"

_ENV_FLAG = "PANTR_RUN_SWEEP"
"""Set it to run the sweep. Absent, the test skips and the suite costs nothing."""

_MIN_CASES = 400
"""Floor on the smoke profile's case count.

Guards the failure mode a verdict-based assertion cannot see: a probe module that stops
yielding cases (an import error swallowed by a generator, a profile guard inverted) makes
the sweep pass by running almost nothing. The smoke profile yields a little over 500
cases; this floor leaves room to prune without becoming a maintenance chore.
"""

_KNOWN_FINDINGS = frozenset(
    {
        # `_degree_elevate_1d_core` returns a knot vector and a control-point array that
        # disagree once an interior knot reaches multiplicity degree + 1. Pinned by
        # `test_sweep_regressions.py::test_degree_elevation_outputs_are_mutually_consistent`.
        "elevate_degree_d0_m1_float64_random",
        "elevate_degree_d1_m2_float64_random",
    }
)
"""Findings the smoke profile is expected to reproduce.

The assertion is containment, not equality: a **new** finding fails the test, while fixing
a known one merely shrinks the set. Every entry must be *tracked*, not merely *seen* --
each is either pinned by an ``xfail(strict=True)`` test in ``test_sweep_regressions.py``,
which is what fails when the bug is fixed, or recorded above as a deliberate
contract-boundary probe. An entry with neither would make this list a way to silence a
finding, which is the one thing it must not become.

Two entries were removed when ``fix(bspline,grid): … (#289)`` landed: the degree-0
cardinal extraction reads out of bounds no longer occur, so the sweep stopped reporting
``cardinal_intervals_d0_m1_float64`` and ``extraction_build_d0_m1_float64_cardinal``.
Containment would have tolerated leaving them, but a stale entry misdescribes the state of
the code.

Six more went the same way when the knot-vector factories were made to refuse
``num_intervals=0``: the six ``create_uniform_{open,periodic}_knots_d{0,1,3}_float64_[0,1]_n0``
cases now decline the input, and the probe asserts that refusal with ``must_reject``.
"""


def _run_sweep(journal: pathlib.Path) -> subprocess.CompletedProcess[str]:
    """Run the smoke profile in a child process.

    Args:
        journal (pathlib.Path): File to receive the per-case JSONL records.

    Returns:
        subprocess.CompletedProcess[str]: The finished process, with output captured.
    """
    # Fixed argv, no shell, in-repo launcher.
    return subprocess.run(
        [sys.executable, str(_LAUNCHER), "--profile", "smoke", "--journal", str(journal)],
        capture_output=True,
        text=True,
        env=subprocess_env(),
        check=False,
    )


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get(_ENV_FLAG) != "1",
    reason=f"set {_ENV_FLAG}=1 to run the adversarial sweep",
)
def test_smoke_sweep_finds_nothing_new(tmp_path: pathlib.Path) -> None:
    journal = tmp_path / "sweep.jsonl"

    result = _run_sweep(journal)

    # 0 is clean, 1 is findings, and anything else is an incomplete run, a crash or a
    # signal (a sanitizer's abort among them). Only the first two may be interpreted.
    assert result.returncode in (0, 1), (
        f"the sweep did not complete (exit {result.returncode}):\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-4000:]}"
    )

    records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert len(records) >= _MIN_CASES, f"only {len(records)} cases ran, expected >= {_MIN_CASES}"

    found = {r["label"] for r in records if r["verdict"] == "BUG"}
    new = found - _KNOWN_FINDINGS
    details = "\n".join(
        f"  {r['group']}/{r['label']} [{r['kind']}]\n    {r['detail'].splitlines()[0]}"
        for r in records
        if r["verdict"] == "BUG" and r["label"] in new
    )
    assert not new, f"the sweep found {len(new)} finding(s) not in _KNOWN_FINDINGS:\n{details}"
