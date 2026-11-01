"""Every Layer 2 entry point over a ``parallel=True`` kernel waits for the JIT warmup.

What remains of that rule is the hierarchical grid, whose kernels are still Numba's; the
tensor-product grid's point location and the inversion ``Bspline.locate`` run in C++ (whose
pool is safe against concurrent callers) and carry no barrier, so for them the last class
below starts threads on a fresh process instead.

``pantr/__init__.py`` compiles (and runs) Numba kernels on a background thread at import,
and Numba's default workqueue threading layer is not safe against a concurrent
``parallel=True`` call from another thread: the process *aborts* rather than raising, which
takes the whole session with it.  ``CLAUDE.md`` states the rule that follows -- a Layer 2
entry point over such kernels calls :func:`pantr._numba_compat.wait_for_jit_warmup` first --
and FELIGN/pantr#418 records the gap this module closes.

**These tests assert the call, not the absence of a crash, and that is deliberate.**  The
barrier is a once-per-process event, so by the time any in-process test runs the warmup is
long finished and nothing in-process can observe the race; a test that merely ran the entry
point and checked it did not abort would pass whether or not the call is there.  Each case
below patches the barrier *in the entry point's own module* together with a sentinel on the
way to the kernel, so deleting the call from any one entry point fails exactly that case.
The same reasoning, and the measurements behind it, are in
``tests/test_bspline_locate.py::TestNumbaWarmup``.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest

from pantr.bezier import Bezier
from pantr.grid import hierarchical_grid, uniform_grid
from tests._tree_env import TESTING_INSTALLED, TREE_SRC, subprocess_env

# ---------------------------------------------------------------------------
# Objects the drivers exercise
# ---------------------------------------------------------------------------


def _bezier_1d() -> Bezier:
    """Build a planar quadratic Bézier curve.

    Returns:
        ~pantr.bezier.Bezier: A ``dim == 1``, ``rank == 2`` Bézier.
    """
    return Bezier(np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]]))


def _bezier_2d() -> Bezier:
    """Build a bilinear Bézier surface in 3D.

    Returns:
        ~pantr.bezier.Bezier: A ``dim == 2``, ``rank == 3`` Bézier.
    """
    return Bezier(np.arange(12.0).reshape(2, 2, 3))


def _hierarchical_grid_2d() -> Any:
    """Build a refined two-dimensional hierarchical grid.

    Returns:
        Any: A :class:`~pantr.grid.HierarchicalGrid` over the unit square, 4 by 4 at the
        root with one block refined to level one.
    """
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4), 2)
    grid.refine(0, [1, 1], [3, 3])
    return grid


# ---------------------------------------------------------------------------
# The census: one case per Layer 2 entry point
# ---------------------------------------------------------------------------


class _BarrierCase(NamedTuple):
    """One Layer 2 entry point over ``parallel=True`` kernels, and how to drive it.

    Attributes:
        entry_point (str): ``module.function`` of the entry point, used as the test id.
        module (str): Module whose ``wait_for_jit_warmup`` name is patched. This is the
            entry point's own module, so the case fails if that one call is deleted even
            when a caller elsewhere still has one.
        sentinel (str): Attribute of ``module`` that the entry point calls on its way to a
            ``parallel=True`` kernel. The barrier must precede it.
        drive (Callable[[], Any]): Public-API call that reaches the entry point.
    """

    entry_point: str
    module: str
    sentinel: str
    drive: Callable[[], Any]


_CASES: tuple[_BarrierCase, ...] = (
    # -- pantr.bezier ------------------------------------------------------
    # Every Bezier kernel, root finding included, moved to C++ (whose pool is safe against
    # concurrent callers), so pantr.bezier carries no barrier and has no case here.
    # -- pantr.grid --------------------------------------------------------
    # TensorProductGrid.locate_many moved to C++ and carries no barrier: its case is the
    # threaded fresh-process test below.
    _BarrierCase(
        "_hierarchical_grid.HierarchicalGrid.locate_many",
        "pantr.grid._hierarchical_grid",
        "_hier_locate_points_core",
        lambda: _hierarchical_grid_2d().locate_many(np.full((3, 2), 0.3)),
    ),
    _BarrierCase(
        "_hierarchical_grid.HierarchicalGrid.collect_cell_bounds",
        "pantr.grid._hierarchical_grid",
        "_hier_collect_cell_bounds_core",
        lambda: _hierarchical_grid_2d().collect_cell_bounds(),
    ),
    # -- pantr.bspline -----------------------------------------------------
    # The basis tabulation (BsplineSpace1D.tabulate_basis and its derivatives), the evaluation
    # of a Bspline (evaluate, evaluate_derivatives) and the Bézier decomposition (to_beziers)
    # moved to C++ (whose pool is safe against concurrent callers), so they carry no barrier and
    # have no case here.
)
"""The Layer 2 entry points whose barrier this module pins, one case each."""


class TestNumbaWarmupBarrier:
    """Each Layer 2 entry point over parallel kernels reaches the barrier first."""

    @pytest.mark.parametrize("case", _CASES, ids=lambda case: case.entry_point)
    def test_the_barrier_runs_before_the_kernel_is_reached(
        self, case: _BarrierCase, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The entry point calls the barrier, and calls it before it reaches a kernel.

        Both names are patched in the entry point's *own* module namespace, so a case can
        only pass on the call that module makes: deleting ``wait_for_jit_warmup()`` from one
        entry point fails that entry point's case and no other.

        Args:
            case (_BarrierCase): The entry point under test.
            monkeypatch (pytest.MonkeyPatch): Patches the two module-level names.
        """
        calls: list[str] = []
        module = importlib.import_module(case.module)
        real_sentinel = getattr(module, case.sentinel)

        def record_barrier() -> None:
            calls.append("barrier")

        def record_sentinel(*args: Any, **kwargs: Any) -> Any:
            calls.append("kernel")
            return real_sentinel(*args, **kwargs)

        monkeypatch.setattr(module, "wait_for_jit_warmup", record_barrier)
        monkeypatch.setattr(module, case.sentinel, record_sentinel)

        case.drive()

        assert "barrier" in calls, f"{case.entry_point} must wait for the JIT warmup"
        assert "kernel" in calls, (
            f"{case.sentinel} was never reached, so this case proves nothing about "
            f"{case.entry_point}; the driver no longer takes the intended branch"
        )
        assert calls.index("barrier") < calls.index("kernel"), (
            f"the barrier must precede the first kernel call; got {calls[:3]}"
        )

    def test_every_case_names_a_distinct_entry_point(self) -> None:
        """No two cases cover the same entry point, so the count is the coverage.

        A copy-paste that left two cases pointing at one function would look like coverage
        of two entry points while leaving one unpinned.
        """
        names = [case.entry_point for case in _CASES]
        assert len(set(names)) == len(names), f"duplicate entry points in the census: {names}"


# ---------------------------------------------------------------------------
# Entry points over C++ that no barrier protects
# ---------------------------------------------------------------------------

_THREADED_FIRST_CALLS = textwrap.dedent(
    """
    import sys
    import threading

    import numpy as np

    import pantr
    from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
    from pantr.grid import uniform_grid

    n_threads = int(sys.argv[1])
    print(pantr.__file__)

    # The identity map on the unit square, and a grid over it. Nothing has been called yet.
    knots = np.array([0.0, 0.0, 1.0, 1.0])
    space = BsplineSpace([BsplineSpace1D(knots, 1), BsplineSpace1D(knots, 1)])
    control_points = np.array([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
    spline = Bspline(space, control_points)
    grid = uniform_grid([[0.0, 1.0], [0.0, 1.0]], [40, 40])

    rng = np.random.default_rng(0)
    targets = rng.uniform(0.05, 0.95, size=(200, 2))
    points = rng.uniform(-0.1, 1.1, size=(5000, 2))
    inside = np.all((points >= 0.0) & (points <= 1.0), axis=1)

    start = threading.Barrier(n_threads)
    failures = []

    def work(k):
        try:
            start.wait()
            if k % 2 == 0:
                cell_ids, ref_coords = spline.locate(targets)
                assert np.all(cell_ids >= 0), "a target of the identity map was not found"
                np.testing.assert_allclose(ref_coords, targets, rtol=0.0, atol=1e-9)
            else:
                ids = grid.locate_many(points)
                assert np.array_equal(ids >= 0, inside), "membership differs from the box test"
        except BaseException as exc:  # reported to the parent, which fails the test
            failures.append(f"thread {k}: {exc!r}")

    threads = [threading.Thread(target=work, args=(k,)) for k in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)
    """
)
"""A fresh process that makes its first ``Bspline.locate`` and ``TensorProductGrid.locate_many``
calls from several threads at once, started together by a barrier."""


class TestConcurrentFirstCalls:
    """The C++ entry points need no barrier: threads may make the first calls together."""

    @pytest.mark.parametrize("attempt", range(3))
    def test_threads_calling_locate_first_thing_succeed(self, attempt: int) -> None:
        """Eight threads call ``Bspline.locate`` and ``locate_many`` as a process's first act.

        This is the race the retired Numba barrier existed for (a parallel kernel called from
        a second thread while the first was still compiling), asked of the C++ entry points
        that replaced those kernels. It has to be a fresh process: the race is a once-per-process
        event, so a warm test process cannot observe it. The child checks its own results
        (the identity map is inverted, and the grid's membership agrees with the box test), so
        a clean exit means the calls succeeded, not only that they did not abort.

        Args:
            attempt (int): Repetition index; each attempt is its own process.
        """
        del attempt  # only distinguishes the processes
        result = subprocess.run(
            [sys.executable, "-c", _THREADED_FIRST_CALLS, "8"],
            capture_output=True,
            text=True,
            check=False,
            env=subprocess_env(),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        origin = Path(result.stdout.splitlines()[0]).resolve()
        if not TESTING_INSTALLED:
            assert origin.is_relative_to(TREE_SRC), f"the child imported pantr from {origin}"
