"""The long compiled calls release the GIL, so other Python threads run meanwhile.

Each case runs one call large enough to take a visible time while a second thread wakes every
millisecond and records when it ran. A call that holds the GIL blocks that thread for its whole
duration; one that releases it leaves gaps of about the wake period. The assertion compares the
largest gap with half the call's duration, a margin of an order of magnitude over the wake period
on any machine where the call takes tens of milliseconds, so it does not depend on the machine's
speed. The cases cover the Bezier operations that reach the thread pool and the reductions and fits
that run exact or dense solves; each is sized to the work at which the binding releases the GIL.
``l2_project_bspline`` releases it too but is absent: its sum-factorised solves are short beside
sampling the user's function, so the call does not tell a held GIL from a released one.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable

import numpy as np
import pytest

from pantr._testing import reduction_cache_clear
from pantr.bezier import Bezier, fit_bezier
from pantr.bspline import (
    create_uniform_space,
    fit_bspline,
    quasi_interpolate_bspline,
)

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
    pts = _rng().random((6000, 3))
    return fit_bspline(np.ones(6000), pts, space)


def _tensor_fit() -> object:
    space = create_uniform_space(3, [600, 600])
    nodes = [np.linspace(0.0, 1.0, 1500), np.linspace(0.0, 1.0, 1500)]
    return fit_bspline(_rng().random((1500, 1500)), nodes, space)


def _qi() -> object:
    space = create_uniform_space(3, [400, 400])
    return quasi_interpolate_bspline(lambda pts: np.ones(pts.shape[0]), space)


def _bezier_scattered_fit() -> object:
    pts = _rng().random((20000, 2))
    return fit_bezier(np.ones(20000), pts, degree=[25, 25])


def _bezier_tensor_fit() -> object:
    nodes = [np.linspace(0.0, 1.0, 400), np.linspace(0.0, 1.0, 400)]
    return fit_bezier(_rng().random((400, 400, 50)), nodes, degree=[60, 60])


_NET = _rng().random((31, 40000, 3))
_CURVE = _rng().random((601, 3000))
_SQUARE = _rng().random((80, 80, 1))
_REDUCIBLE = _rng().random((66, 2))  # above the reduction table's cached degrees: built per call


def _reduction_error() -> object:
    reduction_cache_clear()  # the exact operators are built in the call, not found in the table
    return Bezier(_REDUCIBLE[:62]).degree_reduction_error(1)


CASES: dict[str, Callable[[], object]] = {
    "split": lambda: Bezier(_NET).split(0, 0.3),
    "slice": lambda: Bezier(_NET).slice(0, 0.3),
    "slice_point": lambda: Bezier(_CURVE).slice(0, 0.3),
    "restrict": lambda: Bezier(_NET).restrict([(0.2, 0.7), None]),
    "multiply": lambda: Bezier(_SQUARE).multiply(Bezier(_SQUARE)),
    "compose": lambda: Bezier(_rng().random((10, 10, 1))).compose(Bezier(_rng().random((4, 4, 2)))),
    "reduce_degree": lambda: Bezier(_REDUCIBLE).reduce_degree(1),
    "degree_reduction_error": _reduction_error,
    "fit_bezier scattered": _bezier_scattered_fit,
    "fit_bezier tensor": _bezier_tensor_fit,
    "fit_bspline scattered": _scattered_fit,
    "fit_bspline tensor": _tensor_fit,
    "quasi_interpolate_bspline": _qi,
}


@pytest.mark.slow
@pytest.mark.parametrize("name", list(CASES))
def test_long_call_lets_other_threads_run(name: str) -> None:
    """Another Python thread is never blocked for more than half of the call."""
    duration, gap = _largest_gap(CASES[name])
    assert duration >= _MIN_DURATION, f"{name} took {duration:.4f} s: too short to tell"
    assert gap < 0.5 * duration, f"{name}: blocked {gap:.4f} s of {duration:.4f} s"
