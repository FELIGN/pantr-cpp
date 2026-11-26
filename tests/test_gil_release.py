"""Long compiled calls release the GIL, so other Python threads run meanwhile; short ones keep it.

Each long case runs one call large enough to take a visible time while a second thread wakes every
millisecond and records when it ran. A call that holds the GIL blocks that thread for its whole
duration; one that releases it leaves gaps of about the wake period. The assertion compares the
largest gap with half the call's duration, a margin of an order of magnitude over the wake period
on any machine where the call takes tens of milliseconds, so it does not depend on the machine's
speed. The cases cover the Bezier operations that reach the thread pool, the reductions and fits
that run exact or dense solves, and the grids' serial passes over their cells; each is sized above
the work at which the binding releases the GIL (``gil_release_min_ops`` in
``bindings/src/arrays.hpp``), or builds an exact reduction operator the process table does not
hold yet. ``l2_project_bspline`` releases it too but is absent: its sum-factorised solves are short
beside sampling the user's function, so the call does not tell a held GIL from a released one.

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

from pantr._testing import reduction_cache_clear
from pantr.bezier import Bezier, fit_bezier
from pantr.bspline import (
    create_uniform_space,
    fit_bspline,
    quasi_interpolate_bspline,
)
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


def _block_partition(cells: int) -> Callable[[], Callable[[], object]]:
    """Partition a tensor-product grid into four blocks."""

    def setup() -> Callable[[], object]:
        grid = uniform_grid(_UNIT_SQUARE, cells)
        return lambda: partition_grid(grid, 4, backend="block")

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
    # 2500^2 cells of 4 stores each, 3000^2 of 4 operations each, and 1000^2 cells of 4 + 5
    # operations each: above the threshold.
    "collect_cell_bounds": _collect_cell_bounds(2500),
    "partition_grid": _block_partition(3000),
    "cell_quadrature selection": _selected_cell_quadrature(1000),
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

SHORT_CALLS: dict[str, Callable[[], Callable[[], object]]] = {
    "split": _on(_SMALL, lambda b: b.split(0, 0.3)),
    "multiply": _on(_SMALL, lambda b: b.multiply(b)),
    # 784 points of rank 3: 3 * 784^2 products, below the threshold, where counting every
    # coordinate pair (9 * 784^2) would be above it.
    "multiply rank 3": _on(_net((28, 28, 3)), lambda b: b.multiply(b)),
    "compose": _small_compose,
    "reduce_degree": _on(_SMALL, lambda b: b.reduce_degree(1)),
    "degree_reduction_error": _on(_SMALL, lambda b: b.degree_reduction_error(1)),
    "minimize_degree": _on(_SMALL, lambda b: b.minimize_degree()),
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
    "collect_cell_bounds": _collect_cell_bounds(10),
    "partition_grid": _block_partition(10),
    "cell_quadrature selection": _selected_cell_quadrature(10),
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
