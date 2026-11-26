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

The tests after it are not gated: they check, in process, that the ``full`` profile's basis
probes grade the library's documented refusals as refusals, by running the cases of those probes
rather than the profile.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from importlib import import_module
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from tests._tree_env import subprocess_env

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# Imported by name, so that the type check of `tests` does not follow into `tools/`, which it
# does not cover.
_probes_basis: Any = import_module("tools.adversarial_sweep._probes_basis")
_sweep_core: Any = import_module("tools.adversarial_sweep._core")
Profile: Any = import_module("tools.adversarial_sweep._axes").Profile
Verdict: Any = _sweep_core.Verdict
run_case: Any = _sweep_core.run_case

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


def _sanitizer_expected() -> bool:
    """Say whether this run is meant to be sanitized.

    It is when the extension under test carries AddressSanitizer's instrumentation (it calls the
    runtime's ``__asan_report`` functions, which no ordinary build does) or the runtime is
    preloaded into the test process, as the CI job does. Neither is read from the sweep, whose
    report is what is checked against it.

    Returns:
        bool: ``True`` when the extension is instrumented or the ASan runtime is preloaded.
    """
    from pantr._ext import ext  # noqa: PLC0415

    instrumented = b"__asan_report" in pathlib.Path(ext.__file__).read_bytes()
    return instrumented or "libasan" in os.environ.get("LD_PRELOAD", "")


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

    # The canary. A clean verdict is worth something only if an out-of-range access in a kernel
    # would have been seen, and the sweep reports whether the sanitizer's runtime is in its
    # process: the report must be there, and where this run is meant to be sanitized it must say
    # the runtime is loaded. (The old canary's two deliberately out-of-range kernels are gone with
    # the Python kernels; CI separately checks that the extension is instrumented.)
    reports = [
        line
        for line in result.stderr.splitlines()
        if line.startswith("[sweep]") and "AddressSanitizer" in line
    ]
    assert len(reports) == 1, (
        f"the sweep did not report its sanitizer state:\n{result.stderr[:2000]}"
    )
    if _sanitizer_expected():
        assert "AddressSanitizer is loaded" in reports[0], (
            "this run is meant to be sanitized, but the sweep says otherwise (a clean result "
            f"would be blind to out-of-range access): {reports[0]}"
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


# The `full` profile's basis group, checked in process. These tests run the cases of the
# probes they concern, not the profile: CI runs only the smoke profile.


def test_full_basis_enumeration_calls_no_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    # A builder called while the cases are built, rather than inside a case, ends the group's
    # generator on its first refusal and drops every case after it. Every public builder
    # computes through `_build`, looked up when it is called, so patching it there reaches the
    # builders the probes hold in their tables too.
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a change-of-basis builder ran while the cases were being built")

    monkeypatch.setattr(import_module("pantr.change_basis"), "_build", refuse)

    cases = list(_probes_basis.cases(Profile.FULL))

    round_trips = [case for case in cases if case.label.startswith("roundtrip_")]
    assert round_trips
    # The patch is live: running a case does reach it.
    with pytest.raises(AssertionError, match="while the cases were being built"):
        round_trips[0].run()


@pytest.mark.parametrize(
    "generator",
    [
        _probes_basis._change_basis_square_cases,
        _probes_basis._change_basis_lagrange_pair_cases,
        _probes_basis._round_trip_cases,
    ],
    ids=["square", "lagrange_pair", "round_trip"],
)
def test_full_change_of_basis_cases_find_nothing(
    generator: Callable[[Any], Iterator[Any]],
) -> None:
    # The library decides what it refuses: a degree past a builder's supported domain is
    # refused with the documented ValueError, and a case that expects the opposite of what the
    # library does is graded BUG, in either direction.
    outcomes = [run_case(i, case) for i, case in enumerate(generator(Profile.FULL))]

    findings = [
        f"{o.case.label} [{o.verdict.name}]: {o.detail.splitlines()[0]}"
        for o in outcomes
        if o.verdict in (Verdict.BUG, Verdict.UNDOCUMENTED_REJECTION)
    ]
    assert not findings, "\n".join(findings)


def _accepts(func: Callable[..., Any], *args: Any) -> bool:
    """Say whether a builder returns rather than refusing with `ValueError`."""
    try:
        func(*args)
    except ValueError:
        return False
    return True


def test_probes_read_each_builder_domain_at_its_edge() -> None:
    # The swept degrees never sit on the edge of a domain, so the probes' reading of each domain
    # is checked here against the library itself, at both dtypes' last supported degree and the
    # one after it.
    domains = [
        (name, (lambda degree, dtype, func=func: func(degree, dtype)), limit)
        for name, func, limit in _probes_basis._SQUARE_BUILDERS
    ] + [
        (
            f"{name}_{variant.value}",
            (lambda degree, dtype, func=func, variant=variant: func(degree, variant, dtype)),
            limits[variant],
        )
        for name, func, limits in _probes_basis._LAGRANGE_PAIR_BUILDERS
        if limits is not None
        for variant in limits
    ]
    disagreements = []
    for name, build, limit in domains:
        if limit is None:
            continue
        edges = (limit.float32, limit.float32 + 1, limit.float64, limit.float64 + 1)
        for dtype in (np.dtype(np.float32), np.dtype(np.float64)):
            for degree in edges:
                probe = _probes_basis._supports(limit, degree, dtype)
                if probe != _accepts(build, degree, dtype):
                    disagreements.append(f"{name} degree {degree} {dtype.name}: probe says {probe}")
    assert len(domains) == 6 + 5  # six square builders, bernstein_to_lagrange per variant
    assert not disagreements, "\n".join(disagreements)


def test_full_nd_rejections_are_documented() -> None:
    # Graded against the `Raises:` of the tabulator the probe calls, not of the probe's adapter.
    suffixes = ("_nd_empty_degrees", "_nd_dim_mismatch", "_nd_negative_degree")
    cases = []
    for case in _probes_basis.cases(Profile.FULL):
        if case.label.endswith(suffixes):
            cases.append(case)
        if len(cases) == 9:  # three tabulators, three rejections each
            break

    assert len(cases) == 9
    verdicts = {case.label: run_case(0, case).verdict for case in cases}
    assert verdicts == dict.fromkeys(verdicts, Verdict.DOCUMENTED_REJECTION)
