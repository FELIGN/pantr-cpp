"""Every Layer 2 entry point over a ``parallel=True`` kernel waits for the JIT warmup.

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
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import pytest

from pantr.bezier import Bezier
from pantr.grid import hierarchical_grid, uniform_grid

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
    _BarrierCase(
        "_tensor_product_grid.TensorProductGrid.locate_many",
        "pantr.grid._tensor_product_grid",
        "_locate_points_core",
        lambda: uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4).locate_many(np.full((3, 2), 0.3)),
    ),
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
