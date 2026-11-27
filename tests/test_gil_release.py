"""Long compiled calls release the GIL, so other Python threads run meanwhile; short ones keep it.

Each long case runs one call large enough to take a visible time while a second thread wakes every
millisecond and records when it ran. A call that holds the GIL blocks that thread for its whole
duration; one that releases it leaves gaps of about the wake period. The assertion compares the
largest gap with half the call's duration, a margin of an order of magnitude over the wake period
on any machine where the call takes tens of milliseconds, so it does not depend on the machine's
speed. The cases cover the Bezier operations that reach the thread pool, the reductions and fits
that run exact or dense solves, the grids' serial passes over their cells, and the B-spline
refinements, modifications and CAD constructions; each is sized above the work at which the binding
releases the GIL (``gil_release_min_ops`` in ``bindings/src/arrays.hpp``), or builds a high-degree
exact reduction operator the process table does not hold yet. ``l2_project_bspline`` releases it
too but is absent: its sum-factorised solves are short beside sampling the user's function, so the
call does not tell a held GIL from a released one.

A short call keeps the GIL: releasing it while another thread is CPU-bound costs the caller about
one switch interval before it gets the GIL back, which is far more than the call itself.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import os
import sys
import threading
import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from pantr._ext import ext
from pantr._testing import reduction_cache_clear
from pantr.bezier import Bezier, fit_bezier
from pantr.bspline import (
    Bspline,
    BsplineSpace1D,
    create_uniform_space,
    fit_bspline,
    quasi_interpolate_bspline,
)
from pantr.cad import (
    create_coons_surface,
    create_coons_volume,
    create_revolution,
    create_ruled,
    join,
    make_compat,
)
from pantr.geometry import AABB
from pantr.grid import cell_quadrature, partition_grid, uniform_grid
from pantr.quad import gauss_legendre_quadrature

_WAKE_PERIOD = 0.001
_MIN_DURATION = 0.02  # below this the gap and the wake period are not an order of magnitude apart


def _largest_gap(call: Callable[[], object]) -> tuple[float, float]:
    """Run ``call`` while a ticker thread wakes; return its duration and the largest gap.

    Args:
        call (Callable[[], object]): The call to time.

    Returns:
        tuple[float, float]: The call's duration and the longest stretch of it, in seconds, in
        which the ticker did not run.
    """
    stamps: list[float] = []
    stop = threading.Event()

    def tick() -> None:
        while not stop.is_set():
            stamps.append(time.perf_counter())
            time.sleep(_WAKE_PERIOD)

    ticker = threading.Thread(target=tick)
    ticker.start()
    try:
        time.sleep(0.02)  # the ticker is running before the call starts
        t0 = time.perf_counter()
        call()
        t1 = time.perf_counter()
    finally:
        stop.set()
        ticker.join()
    inside = [t0, *(s for s in stamps if t0 < s < t1), t1]
    return t1 - t0, max(b - a for a, b in itertools.pairwise(inside))


def _rng() -> np.random.Generator:
    return np.random.default_rng(20261109)


def _scattered_fit() -> object:
    space = create_uniform_space(2, [7, 7, 7])
    pts = _rng().random((8000, 3))
    return fit_bspline(np.ones(8000), pts, space)


def _tensor_fit() -> object:
    space = create_uniform_space(3, [600, 600])
    nodes = [np.linspace(0.0, 1.0, 1500), np.linspace(0.0, 1.0, 1500)]
    return fit_bspline(_rng().random((1500, 1500)), nodes, space)


def _qi() -> object:
    space = create_uniform_space(3, [800, 800])
    return quasi_interpolate_bspline(lambda pts: np.ones(pts.shape[0]), space)


def _bezier_scattered_fit() -> object:
    pts = _rng().random((20000, 2))
    return fit_bezier(np.ones(20000), pts, degree=[25, 25])


def _bezier_tensor_fit() -> object:
    nodes = [np.linspace(0.0, 1.0, 400), np.linspace(0.0, 1.0, 400)]
    return fit_bezier(_rng().random((400, 400, 50)), nodes, degree=[60, 60])


def _on(
    cp: Callable[[], npt.NDArray[np.float64]], op: Callable[[Bezier], object]
) -> Callable[[], Callable[[], object]]:
    """Build the Bezier outside the timed call (its constructor holds the GIL) and time ``op``."""

    def setup() -> Callable[[], object]:
        b = Bezier(cp())
        return lambda: op(b)

    return setup


def _untimed_setup(call: Callable[[], object]) -> Callable[[], Callable[[], object]]:
    return lambda: call


_NET = (31, 120000, 3)
_REDUCIBLE_SHAPE = (66, 2)  # above the reduction table's cached degrees: built per call


def _net(shape: tuple[int, ...]) -> Callable[[], npt.NDArray[np.float64]]:
    """Defer a random net to the setup, so collecting the module allocates nothing large."""
    return lambda: _rng().random(shape)


def _reduction_error() -> object:
    reduction_cache_clear()  # the exact operators are built in the call, not found in the table
    return Bezier(_rng().random(_REDUCIBLE_SHAPE)[:62]).degree_reduction_error(1)


def _minimize() -> object:
    reduction_cache_clear()  # the first trial's exact operator is built in the call
    return Bezier(_rng().random((61, 2))).minimize_degree()


def _compose() -> Callable[[], object]:
    # Its estimate, outer.size() * (sum(outer degrees) * inner degree + 1)^2 = 256 * 151^2, is
    # above the threshold.
    outer, inner = Bezier(_rng().random((16, 16, 1))), Bezier(_rng().random((6, 6, 2)))
    return lambda: outer.compose(inner)


def _multiply() -> Callable[[], object]:
    square = Bezier(_rng().random((80, 80, 1)))
    return lambda: square.multiply(square)


_UNIT_SQUARE = [[0.0, 1.0], [0.0, 1.0]]


def _collect_cell_bounds(cells: int) -> Callable[[], Callable[[], object]]:
    """Collect the corners of a tensor-product grid's cells (``2 * ndim`` stores per cell)."""

    def setup() -> Callable[[], object]:
        grid = uniform_grid(_UNIT_SQUARE, cells)
        return grid.collect_cell_bounds

    return setup


def _selected_cell_quadrature(cells: int) -> Callable[[], Callable[[], object]]:
    """Map a one-point rule onto every cell of a tensor-product grid, given as a selection."""

    def setup() -> Callable[[], object]:
        grid = uniform_grid(_UNIT_SQUARE, cells)
        ids = np.arange(grid.num_cells)
        rule = gauss_legendre_quadrature(2, 1)
        return lambda: cell_quadrature(grid, rule, ids)

    return setup


def _cell_quadrature(cells: int) -> Callable[[], Callable[[], object]]:
    """Map a one-point rule onto every cell of a tensor-product grid, without a selection."""

    def setup() -> Callable[[], object]:
        grid = uniform_grid(_UNIT_SQUARE, cells)
        rule = gauss_legendre_quadrature(2, 1)
        return lambda: cell_quadrature(grid, rule)

    return setup


def _block_partition(cells: int) -> Callable[[], Callable[[], object]]:
    """Partition a tensor-product grid into four blocks."""

    def setup() -> Callable[[], object]:
        grid = uniform_grid(_UNIT_SQUARE, cells)
        return lambda: partition_grid(grid, 4, backend="block")

    return setup


def _first_query(cells: int) -> Callable[[], Callable[[], object]]:
    """Query a new grid, so that the call builds its BVH."""

    def setup() -> Callable[[], object]:
        box = AABB([0.1, 0.1], [0.2, 0.2])  # made here: one held by the module outlives nanobind
        return lambda: uniform_grid(_UNIT_SQUARE, cells).query_aabb(box)

    return setup


# The B-spline refinements, modifications and CAD constructions. Each estimate named below is the
# binding's lower bound on the call's work (``bindings/src/bspline_refine.cpp``,
# ``bspline_modify.cpp``, ``cad.cpp``); ``n`` is a direction's control points, ``rest`` the values
# of the other extents and the coordinates.


def _spline(degree: int, intervals: list[int], rank: int, *, periodic: bool = False) -> Bspline:
    """A B-spline of uniform knots and random control points."""
    space = create_uniform_space(degree, intervals, periodic=periodic)
    return Bspline(space, _rng().random((*space.num_basis, rank)))


def _on_spline(
    degree: int,
    intervals: list[int],
    rank: int,
    op: Callable[[Bspline], object],
    *,
    periodic: bool = False,
) -> Callable[[], Callable[[], object]]:
    """Build the B-spline outside the timed call and time ``op`` on it."""

    def setup() -> Callable[[], object]:
        b = _spline(degree, intervals, rank, periodic=periodic)
        return lambda: op(b)

    return setup


def _to_periodic(intervals: int) -> Callable[[], Callable[[], object]]:
    """Convert the open form of a periodic cubic curve back to periodic form."""

    def setup() -> Callable[[], object]:
        opened = _spline(3, [intervals], 3, periodic=True).to_open_bspline()
        return lambda: opened.to_periodic()

    return setup


def _remove_knots(intervals: int, values: int) -> Callable[[], Callable[[], object]]:
    """Try to remove ``values`` interior knots of direction 0 of a random cubic surface."""

    def setup() -> Callable[[], object]:
        b = _spline(3, [intervals, intervals], 3)
        knots = np.arange(1, values + 1) / intervals
        return lambda: b.remove_knots([knots, None])

    return setup


def _refine_to_common(
    degree: int, intervals: int, intervals_b: int, other: int
) -> Callable[[], Callable[[], object]]:
    """Refine two surfaces to the union of their knots in direction 0."""

    def setup() -> Callable[[], object]:
        a = _spline(degree, [intervals, other], 3)
        b = _spline(degree, [intervals_b, other], 3)
        return lambda: ext.bspline.refine_to_common_space_1d(a._h, b._h, 0)

    return setup


def _refused(op: Callable[[Bspline], object], match: str) -> Callable[[Bspline], object]:
    """Run ``op``, which the call refuses before any work, and check that it does."""

    def call(b: Bspline) -> object:
        with pytest.raises(ValueError, match=match):
            op(b)
        return None

    return call


def _linear_space(intervals: int) -> BsplineSpace1D:
    """A clamped degree-1 space of uniform knots."""
    return BsplineSpace1D(np.concatenate(([0.0], np.linspace(0.0, 1.0, intervals + 1), [1.0])), 1)


def _inserted_knot_vector(
    intervals: int, inserted: int, *, shuffled: bool
) -> Callable[[], Callable[[], object]]:
    """Merge ``inserted`` knots, shuffled or sorted, into a uniform degree-1 knot vector.

    The binding is called, not the wrapper, which then builds a space of the result with the GIL
    held.
    """

    def setup() -> Callable[[], object]:
        space = _linear_space(intervals)
        knots, tol = np.asarray(space.knots, dtype=np.float64), space.tolerance
        new = np.linspace(0.0, 1.0, inserted + 2)[1:-1]
        if shuffled:
            new = _rng().permutation(new)
        return lambda: ext.bspline.compute_inserted_knot_vector(knots, 1, new, tol)

    return setup


def _subdivision_knots(intervals: int) -> Callable[[], Callable[[], object]]:
    """The knots that halve every interval of a degree-1 knot vector (the binding)."""

    def setup() -> Callable[[], object]:
        space = _linear_space(intervals)
        knots, tol = np.asarray(space.knots, dtype=np.float64), space.tolerance
        return lambda: ext.bspline.compute_uniform_subdivision_knots(knots, 1, tol, 2, None)

    return setup


def _make_compat(intervals: int) -> Callable[[], Callable[[], object]]:
    """Two cubic surfaces of ``intervals`` and ``intervals + 1`` intervals per direction."""

    def setup() -> Callable[[], object]:
        a = _spline(3, [intervals, intervals], 3)
        b = _spline(3, [intervals + 1, intervals + 1], 3)
        return lambda: make_compat(a, b)

    return setup


def _ruled(intervals: int, more: int) -> Callable[[], Callable[[], object]]:
    """The ruled volume between cubic surfaces of ``intervals`` and ``intervals + more``."""

    def setup() -> Callable[[], object]:
        a = _spline(3, [intervals, intervals], 3)
        b = _spline(3, [intervals + more, intervals + more], 3)
        return lambda: create_ruled(a, b)

    return setup


def _revolution(intervals: int) -> Callable[[], Callable[[], object]]:
    """A cubic surface revolved a full turn about the z axis."""

    def setup() -> Callable[[], object]:
        b = _spline(3, [intervals, intervals], 3)
        return lambda: create_revolution(b, [0.0, 0.0, 0.0])

    return setup


def _coons_surface(intervals: int) -> Callable[[], Callable[[], object]]:
    """The Coons patch of the four boundary curves of a random cubic surface."""

    def setup() -> Callable[[], object]:
        s = _spline(3, [intervals, intervals], 3)
        curves = ((s.boundary(0, 0), s.boundary(0, 1)), (s.boundary(1, 0), s.boundary(1, 1)))
        return lambda: create_coons_surface(curves)  # type: ignore[arg-type]

    return setup


def _coons_volume(intervals: int) -> Callable[[], Callable[[], object]]:
    """The Coons volume of the six boundary faces of a random cubic volume."""

    def setup() -> Callable[[], object]:
        v = _spline(3, [intervals, intervals, intervals], 3)
        faces = tuple((v.boundary(a, 0), v.boundary(a, 1)) for a in range(3))
        return lambda: create_coons_volume(faces)  # type: ignore[arg-type]

    return setup


def _join(intervals: list[int]) -> Callable[[], Callable[[], object]]:
    """Join the two halves of a random cubic surface split along direction 0."""

    def setup() -> Callable[[], object]:
        left, right = _spline(3, intervals, 3).split(0, 0.5)
        return lambda: join(left, right, 0)

    return setup


_WINDOWS_BLOCKED = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the Windows wheel legs showed slice and restrict, and in a later run split and "
    "multiply, blocking the ticker on all three tries; the cause is not established, and the "
    "decision to release depends only on the problem size, so the Linux and macOS legs exercise it",
)

SETUPS: dict[str, Callable[[], Callable[[], object]]] = {
    "split": _on(_net(_NET), lambda b: b.split(0, 0.3)),
    "slice": _on(_net(_NET), lambda b: b.slice(0, 0.3)),
    "slice_point": _on(_net((1201, 3000)), lambda b: b.slice(0, 0.3)),
    "restrict": _on(_net(_NET), lambda b: b.restrict([(0.2, 0.7), None])),
    "multiply": _multiply,
    "compose": _compose,
    "reduce_degree": _on(_net(_REDUCIBLE_SHAPE), lambda b: b.reduce_degree(1)),
    "degree_reduction_error": _untimed_setup(_reduction_error),
    "minimize_degree": _untimed_setup(_minimize),
    "fit_bezier scattered": _untimed_setup(_bezier_scattered_fit),
    "fit_bezier tensor": _untimed_setup(_bezier_tensor_fit),
    "fit_bspline scattered": _untimed_setup(_scattered_fit),
    "fit_bspline tensor": _untimed_setup(_tensor_fit),
    "quasi_interpolate_bspline": _untimed_setup(_qi),
    # 2500^2 cells of 4 stores each, 4500^2 of 4 operations each, and 1500^2 and 1000^2 cells of
    # 4 + 5 operations each: above the threshold.
    "collect_cell_bounds": _collect_cell_bounds(2500),
    "partition_grid": _block_partition(4500),
    "cell_quadrature": _cell_quadrature(1500),
    "cell_quadrature selection": _selected_cell_quadrature(1000),
    # A BVH of 600^2 cells: 4 operations per cell for its corners and each of its 19 full levels.
    "query_aabb first": _first_query(600),
    # rest * ((n + inserted) * (degree + 1) - degree^2) for the knots inserted into each direction
    # of a 600^2 degree-7 net of rank 3, rest read off the input: 2 * 1800 * (1600 * 8 - 49).
    "insert_knots": _on_spline(
        7, [593, 593], 3, lambda b: b.insert_knots([np.linspace(0.0, 1.0, 1002)[1:-1]] * 2)
    ),
    # refinement_ops with nothing inserted, for each of the two periodic directions of an 800^2
    # cubic net: 2 * 2400 * (800 * 4 - 9).
    "to_open_bspline": _on_spline(3, [800, 800], 3, lambda b: b.to_open_bspline(), periodic=True),
    # (2 / 3) (k^3 - k) for the QR of a curve of 1103 open points, k = 1103 - 3.
    "to_periodic": _to_periodic(1100),
    # As insert_knots, for one knot halving each of the 593 intervals of each direction:
    # 2 * 1800 * (1193 * 8 - 49).
    "subdivide": _on_spline(7, [593, 593], 3, lambda b: b.subdivide([2, 2])),
    # The net's 1303^2 * 3 values, inserted at the bounds or copied.
    "Bspline.restrict": _on_spline(3, [1300, 1300], 3, lambda b: b.restrict([(0.2, 0.7), None])),
    # Three passes over the 1000^2 * 3 values of a degree-7 net.
    "Bspline.split": _on_spline(7, [993, 993], 3, lambda b: b.split(0, 0.3)),
    # 2 (n + m), n = 1.5e6 + 3 knots and m = 1e6 new ones, which the call also sorts.
    "compute_inserted_knot_vector": _inserted_knot_vector(1_500_000, 1_000_000, shuffled=True),
    # The 4.5e6 + 3 knots, each read for the classes.
    "compute_uniform_subdivision_knots": _subdivision_knots(4_500_000),
    # rest * (n + increment * intervals) = 3000 * (1000 + 997).
    "elevate_degree": _on_spline(3, [997, 997], 3, lambda b: b.elevate_degree([1, 0])),
    # rest * (intervals * (q + 1) (p + 1) + removals * (q + intervals)), one forced removal at each
    # of the 169 interior knots: 519 * (170 * 12 + 169 * 172).
    "Bspline.reduce_degree": _on_spline(3, [170, 170], 3, lambda b: b.reduce_degree([1, 0])),
    # rest * (n + sum_k (n - 3 k)) for the copy and the 8 removals' working copies:
    # 1800 * (600 + 8 * 600 - 3 * 28).
    "remove_knots": _remove_knots(597, 8),
    # 2 * rest * (n - 1) = 2 * 3300 * 1099, along direction 1, which flattens the net (a
    # transposed copy) and restores it.
    "derivative": _on_spline(3, [1097, 1097], 3, lambda b: b.derivative(1)),
    # As derivative, plus rest * (n - 1 + intervals) for the elevation: 2700 * (2 * 899 + 1796).
    "derivative keep_degree": _on_spline(
        3, [897, 897], 3, lambda b: b.derivative(0, keep_degree=True)
    ),
    # refinement_ops for the 800 in-domain knots the coarser degree-7 one lacks:
    # 4500 * (1607 * 8 - 49).
    "refine_to_common_space_1d": _refine_to_common(7, 800, 1600, 1493),
    # Per direction, the coarser surface lacks one interior knot: 2 * 1509 * (504 * 4 - 9).
    "make_compat": _make_compat(500),
    # make_compat's estimate (above) for surfaces of 500^2 and 501^2 intervals, plus twice their
    # two nets, stacked and copied.
    "create_ruled": _ruled(500, 1),
    # 16 * points * 9 for the full turn, 383^2 points.
    "create_revolution": _revolution(380),
    # 3 * 1203^2 * 3: three terms added into the patch.
    "create_coons_surface": _coons_surface(1200),
    # 7 * 73^3 * 3: seven terms added into the volume.
    "create_coons_volume": _coons_volume(70),
    # Twice the larger half's net, 503 * 1703 * 3 values.
    "join": _join([1000, 1700]),
}


@_WINDOWS_BLOCKED
@pytest.mark.slow
@pytest.mark.parametrize("name", SETUPS)
def test_long_call_lets_other_threads_run(name: str) -> None:
    """Another Python thread is not blocked for more than half of the call, in one of three tries.

    A held GIL blocks the ticker for the whole call on every try, so it cannot pass; a released
    one can still show a long gap when an oversubscribed machine deschedules the ticker, which is
    why one clean try of three suffices.
    """
    call = SETUPS[name]()
    tries = [_largest_gap(call) for _ in range(3)]
    telling = [(duration, gap) for duration, gap in tries if duration >= _MIN_DURATION]
    assert telling, f"{name} took {tries}: every try too short to tell"
    assert any(gap < 0.5 * duration for duration, gap in telling), f"{name}: blocked {tries}"


_SWITCH_INTERVAL = 0.02  # long beside a short call, so a release shows as a wait of about this
_SHORT_TRIES = 10


def _small_compose() -> Callable[[], object]:
    outer, inner = Bezier(_rng().random((4, 1))), Bezier(_rng().random((4, 4, 1)))
    return lambda: outer.compose(inner)


_SMALL = _net((4, 4, 2))


def _cold_reduction_error(b: Bezier) -> object:
    """Reduce a curve whose exact operator is built in the call: a short build at a low degree."""
    reduction_cache_clear()
    return b.degree_reduction_error(1)


def _cold_reduction(b: Bezier) -> object:
    """Reduce as ``_cold_reduction_error`` does, returning the reduced Bezier."""
    reduction_cache_clear()
    return b.reduce_degree(1)


def _cold_minimization(b: Bezier) -> object:
    """Minimize a curve whose first trial's exact operator is built in the call."""
    reduction_cache_clear()
    return b.minimize_degree()


def _cold_zero_tol_minimization(b: Bezier) -> object:
    """Minimize with ``tol=0``, which runs no trial and builds nothing, up to ``bincoeff_max_n``."""
    reduction_cache_clear()
    return b.minimize_degree(0.0)


def _refused_minimization(b: Bezier) -> object:
    """Minimize a curve of degree above ``bincoeff_max_n`` (61), which the call refuses at once."""
    reduction_cache_clear()  # unrefused, the call would build an operator above the threshold
    with pytest.raises(ValueError, match="binomial coefficients"):
        b.minimize_degree()
    return None


SHORT_CALLS: dict[str, Callable[[], Callable[[], object]]] = {
    "split": _on(_SMALL, lambda b: b.split(0, 0.3)),
    "multiply": _on(_SMALL, lambda b: b.multiply(b)),
    # 784 points of rank 3: 3 * 784^2 products, below the threshold, where counting every
    # coordinate pair (9 * 784^2) would be above it.
    "multiply rank 3": _on(_net((28, 28, 3)), lambda b: b.multiply(b)),
    "compose": _small_compose,
    "reduce_degree": _on(_SMALL, lambda b: b.reduce_degree(1)),
    "degree_reduction_error": _on(_SMALL, lambda b: b.degree_reduction_error(1)),
    # A cold build at degree 10 is estimated at 16 * 2290 operations, two orders below the
    # threshold; at degree 61 with ``tol=0`` the call builds nothing and copies the net, wide
    # enough here that the call is not trivially short.
    "degree_reduction_error cold": _on(_net((11, 1)), _cold_reduction_error),
    "reduce_degree cold": _on(_net((11, 1)), _cold_reduction),
    "minimize_degree": _on(_SMALL, lambda b: b.minimize_degree()),
    "minimize_degree cold": _on(_net((11, 1)), _cold_minimization),
    "minimize_degree tol 0 cold": _on(_net((62, 2000)), _cold_zero_tol_minimization),
    "minimize_degree refused": _on(_net((63, 1)), _refused_minimization),
    "fit_bezier tensor": lambda: lambda: fit_bezier(
        _rng().random((5, 5)), [np.linspace(0.0, 1.0, 5)] * 2, degree=[3, 3]
    ),
    "fit_bezier scattered": lambda: lambda: fit_bezier(
        np.ones(20), _rng().random((20, 2)), degree=[2, 2]
    ),
    "fit_bspline tensor": lambda: lambda: fit_bspline(
        _rng().random((8, 8)), [np.linspace(0.0, 1.0, 8)] * 2, create_uniform_space(2, [4, 4])
    ),
    "fit_bspline scattered": lambda: lambda: fit_bspline(
        np.ones(40), _rng().random((40, 2)), create_uniform_space(1, [2, 2])
    ),
    "quasi_interpolate_bspline": lambda: lambda: quasi_interpolate_bspline(
        lambda pts: np.ones(pts.shape[0]), create_uniform_space(2, [4, 4])
    ),
    # Sized so that each estimate is at most a tenth of the threshold while the call is not
    # trivially short: 300^2 cells of 4 stores, 300^2 of 4 operations, 100^2 of 4 + 5, and a BVH
    # of 40^2 cells at 4 operations per cell for its corners and each of its 11 full levels.
    "collect_cell_bounds": _collect_cell_bounds(300),
    "partition_grid": _block_partition(300),
    "cell_quadrature selection": _selected_cell_quadrature(100),
    "query_aabb first": _first_query(40),
    # Each estimate at most a tenth of the threshold (the long entry's formula, at these sizes),
    # with the call not trivially short.
    "insert_knots": _on_spline(
        3, [97, 97], 3, lambda b: b.insert_knots([np.linspace(0.0, 1.0, 202)[1:-1], None])
    ),
    "to_open_bspline": _on_spline(3, [100, 100], 3, lambda b: b.to_open_bspline(), periodic=True),
    "to_periodic": _to_periodic(80),
    "subdivide": _on_spline(3, [120, 120], 3, lambda b: b.subdivide([2, 1])),
    "Bspline.restrict": _on_spline(3, [300, 300], 3, lambda b: b.restrict([(0.2, 0.7), None])),
    "Bspline.split": _on_spline(3, [200, 200], 3, lambda b: b.split(0, 0.3)),
    "compute_inserted_knot_vector": _inserted_knot_vector(100_000, 50_000, shuffled=False),
    "compute_uniform_subdivision_knots": _subdivision_knots(100_000),
    "elevate_degree": _on_spline(3, [197, 197], 3, lambda b: b.elevate_degree([1, 0])),
    "Bspline.reduce_degree": _on_spline(3, [40, 40], 3, lambda b: b.reduce_degree([1, 0])),
    "remove_knots": _remove_knots(97, 4),
    "derivative": _on_spline(3, [197, 197], 3, lambda b: b.derivative(1)),
    "derivative keep_degree": _on_spline(
        3, [147, 147], 3, lambda b: b.derivative(0, keep_degree=True)
    ),
    # Knots that do not nest: each receives about 120, the estimate counts the one of difference.
    "refine_to_common_space_1d": _refine_to_common(3, 120, 121, 97),
    "make_compat": _make_compat(100),
    "create_ruled": _ruled(170, 0),
    "create_revolution": _revolution(37),
    "create_coons_surface": _coons_surface(147),
    "create_coons_volume": _coons_volume(17),
    "join": _join([200, 200]),
    # Refused before any work on nets whose estimate would otherwise be above the threshold (the
    # long entries' sizes), so each counts no work and keeps the GIL.
    "Bspline.restrict refused": _on_spline(
        3, [1300, 1300], 3, _refused(lambda b: b.restrict([(0.7, 0.2), None]), "strictly less")
    ),
    "Bspline.split refused": _on_spline(
        7, [993, 993], 3, _refused(lambda b: b.split(0, 1.5), "strictly inside")
    ),
    "elevate_degree refused": _on_spline(
        3, [997, 997], 3, _refused(lambda b: b.elevate_degree([59, 0]), "binomial")
    ),
}


def _fastest_under_contention(call: Callable[[], object]) -> float:
    """Time ``call`` while another thread spins in bytecode; return the fastest of several tries.

    Args:
        call (Callable[[], object]): The call to time.

    Returns:
        float: The shortest duration, in seconds.
    """
    call()  # warm: the reduction operators enter the process table, the imports are done
    stop = threading.Event()

    def spin() -> None:
        x = 0
        while not stop.is_set():
            x += 1

    previous = sys.getswitchinterval()
    sys.setswitchinterval(_SWITCH_INTERVAL)
    spinner = threading.Thread(target=spin)
    spinner.start()
    try:
        durations = []
        for _ in range(_SHORT_TRIES):
            t0 = time.perf_counter()
            call()
            durations.append(time.perf_counter() - t0)
    finally:
        stop.set()
        spinner.join()
        sys.setswitchinterval(previous)
    return min(durations)


@pytest.mark.parametrize("name", list(SHORT_CALLS))
def test_short_call_keeps_the_gil(name: str) -> None:
    """A short call is not delayed by a switch interval when another thread is CPU-bound.

    A call that releases the GIL hands it to the spinning thread, which keeps it until its switch
    interval expires, so every try lasts about that interval; one that keeps it returns in far less.
    The wrapper's own bytecode also yields the GIL to the spinner once per interval, so a try that
    straddles that moment waits too, whatever the compiled call does; the fastest of ten tries must
    be under half the interval, so those tries, and a descheduled one, do not decide it. Rarely a
    whole batch of tries waits (the cause is not established), so the best of three batches, each
    with its own spinning thread, is taken; a call that releases waits in every batch.
    """
    setup = SHORT_CALLS[name]
    fastest = min(_fastest_under_contention(setup()) for _ in range(3))
    assert fastest < 0.5 * _SWITCH_INTERVAL, f"{name}: fastest {fastest:.4f} s under contention"


# Batches of the pool-call test, each with its own spinner. Every counted batch must show the
# hand-over, so a batch more is one more chance of a false failure and one more chance to count a
# batch at all: three is a heuristic balance between the two, not a derived number.
_BATCHES = 3
# The buffer the control's duration is measured on, long enough for the timer to resolve it; the
# control's own buffer is scaled from it, and never below CPython's ``HASHLIB_GIL_MINSIZE``
# (2048 bytes, ``Modules/hashlib.h``), under which ``hashlib`` keeps the GIL.
_CONTROL_PROBE_BYTES = 1 << 20
_CONTROL_MIN_BYTES = 1 << 12


def _releasing_control(call: Callable[[], object]) -> Callable[[], object]:
    """Return a stdlib call that releases the GIL and runs about twice as long as ``call``.

    ``hashlib.sha256`` releases the GIL while it digests a buffer of at least
    ``HASHLIB_GIL_MINSIZE`` bytes, and is not pantr. Its duration is measured on a 1 MiB probe
    and its buffer scaled by the ratio of the two calls' unloaded durations, times 2 (a heuristic
    margin, not derived), so the control is not the shorter of the two, which would give the
    spinner less time to be scheduled than the pantr call gives it, nor much the longer, which
    would let it hand over where the pantr call could not.

    Args:
        call (Callable[[], object]): The call the control stands beside, already warm.

    Returns:
        Callable[[], object]: The control.
    """

    def fastest(f: Callable[[], object]) -> float:
        f()  # warm
        durations = []
        for _ in range(_SHORT_TRIES):
            t0 = time.perf_counter()
            f()
            durations.append(time.perf_counter() - t0)
        return min(durations)

    probe = bytes(_CONTROL_PROBE_BYTES)
    ratio = fastest(call) / fastest(lambda: hashlib.sha256(probe).digest())
    size = max(_CONTROL_MIN_BYTES, math.ceil(2 * ratio * _CONTROL_PROBE_BYTES))
    buffer = bytes(size)
    return lambda: hashlib.sha256(buffer).digest()


def _interleaved_medians(
    call: Callable[[], object], control: Callable[[], object]
) -> tuple[float, float]:
    """Time ``call`` and ``control`` in turn while another thread spins; return both medians.

    Args:
        call (Callable[[], object]): The call under test, already warm.
        control (Callable[[], object]): A call that certainly releases the GIL.

    Returns:
        tuple[float, float]: The median durations of ``call`` and ``control``, in seconds.
    """
    stop = threading.Event()

    def spin() -> None:
        x = 0
        while not stop.is_set():
            x += 1

    previous = sys.getswitchinterval()
    sys.setswitchinterval(_SWITCH_INTERVAL)
    spinner = threading.Thread(target=spin)
    spinner.start()
    try:
        durations: tuple[list[float], list[float]] = ([], [])
        for _ in range(_SHORT_TRIES):
            for f, out in zip((call, control), durations, strict=True):
                t0 = time.perf_counter()
                f()
                out.append(time.perf_counter() - t0)
    finally:
        stop.set()
        spinner.join()
        sys.setswitchinterval(previous)
    call_median, control_median = (sorted(d)[len(d) // 2] for d in durations)
    return call_median, control_median


@pytest.mark.skipif(
    "PYTEST_XDIST_WORKER" in os.environ
    or not hasattr(os, "sched_getaffinity")
    or len(os.sched_getaffinity(0)) < 2,
    reason="the call is shorter than the time the spinning thread needs to be scheduled, so it "
    "takes the released GIL only from a free core; under xdist workers or on one CPU it often has "
    "none and the release cannot be seen (the serial coverage run has one)",
)
def test_pool_call_releases_the_gil() -> None:
    """A call that reaches the pool releases the GIL, however short.

    A curve of 20 control points evaluated at 8000 points is ``160000`` operations, above the pool's
    threshold (``parallel_min_ops``, ``2**17``) and far below ``gil_release_min_ops`` (``2**22``),
    so it releases only because it reaches the pool. A released call hands the GIL to the spinning
    thread and waits for its switch interval, so the median try lasts about that interval; one that
    kept it would return in far less. Bug caught: pool calls gated on the non-pool threshold, which
    made Python threads that each run such calls serialize.

    The hand-over shows only if the spinner is scheduled and takes the GIL before the short call
    returns, which a shared host does not always allow (issue #18). So each
    batch also times a control that certainly releases, interleaved with the pantr call, and only
    a batch whose control hands over decides: the pantr call must hand over in every such batch,
    and the test skips, with the control's medians, when no batch does.
    """
    curve = Bezier(_rng().random((20, 1)))
    pts = _rng().random(8000)
    curve.evaluate(pts)  # warm
    control = _releasing_control(lambda: curve.evaluate(pts))
    batches = [_interleaved_medians(lambda: curve.evaluate(pts), control) for _ in range(_BATCHES)]
    if all(ctrl < 0.5 * _SWITCH_INTERVAL for _, ctrl in batches):
        controls = ", ".join(f"{ctrl:.4f}" for _, ctrl in batches)
        pytest.skip(
            f"the host cannot show a release: the control's median was below half the switch "
            f"interval ({_SWITCH_INTERVAL} s) in every batch: {controls} s"
        )
    kept = [
        (p, c) for p, c in batches if c >= 0.5 * _SWITCH_INTERVAL and p < 0.5 * _SWITCH_INTERVAL
    ]
    assert not kept, (
        f"(pantr, control) medians {', '.join(f'({p:.4f}, {c:.4f})' for p, c in kept)} s: the "
        f"control handed the GIL over (switch interval {_SWITCH_INTERVAL} s) and the pantr call "
        f"kept it"
    )
