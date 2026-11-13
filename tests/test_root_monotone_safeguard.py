"""The monotone root solver bisects when a Newton step stalls, so ``float32`` converges.

The Newton/bisection hybrid keeps a bracket around the root and stops when it is narrower than
``tol``. A ``float32`` evaluation can leave every Newton iterate on one side of the root, so the
far end of the bracket never moves and the iteration cap was reached on ordinary
simple-root inputs. A step that fails to halve the bracket is now followed by a bisection.

Oracle: a second implementation, the ``float64`` solve of the same coefficients widened exactly to
``float64``. Both solves stop on a bracket narrower than their ``tol`` around a sign change of the
evaluated polynomial, so the two roots differ by at most the ``float32`` ``tol`` plus the
``float64`` one, and the ``float32`` evaluation's rounding moves its sign change by a few units of
``float32`` roundoff over the slope, which these well-separated inputs keep below ``tol``.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy import typing as npt

from pantr.bezier import Bezier, find_monotone_root
from pantr.tolerance import get_strict


def _random_monotone_float32(rng: np.random.Generator, count: int) -> list[npt.NDArray[np.float32]]:
    """Draw increasing (or decreasing) Bernstein coefficients with a sign change, in ``float32``.

    Monotone coefficients make a monotone polynomial, so each has exactly one simple root.

    Args:
        rng (np.random.Generator): The random source.
        count (int): How many to draw.

    Returns:
        list[npt.NDArray[np.float32]]: Coefficient vectors of degree 2 to 12.
    """
    out: list[npt.NDArray[np.float32]] = []
    while len(out) < count:
        n = int(rng.integers(2, 13))
        c = np.sort(rng.standard_normal(n + 1))
        c -= rng.uniform(c[0], c[-1])
        if rng.random() < 0.5:
            c = -c
        c32 = c.astype(np.float32)
        if np.sign(c32[0]) * np.sign(c32[-1]) < 0:
            out.append(c32)
    return out


@pytest.mark.parametrize("seed", [0, 1])
def test_float32_monotone_roots_converge_to_the_float64_root(seed: int) -> None:
    """Every random ``float32`` simple-root case converges within ``tol`` of ``float64``."""
    tol32 = get_strict(np.float32)
    tol64 = get_strict(np.float64)
    for c32 in _random_monotone_float32(np.random.default_rng(seed), 1500):
        reference = find_monotone_root(Bezier(c32.astype(np.float64)))
        root = find_monotone_root(Bezier(c32))
        assert abs(root - reference) <= tol32 + tol64, c32.tolist()


def test_float32_converges_at_a_tol_finer_than_its_own() -> None:
    """A ``float32`` solve at ``tol=1e-13`` converges: the bracket is held in ``float64``.

    The cap sized for ``float32``'s own strict tolerance (or the previous 64) refused some of these.
    Oracle as above, at the ``float32`` strict tolerance: the rounding of the ``float32``
    evaluation, not ``tol``, decides how far the sign change is from the ``float64`` one.
    """
    tol32 = get_strict(np.float32)
    tol64 = get_strict(np.float64)
    for c32 in _random_monotone_float32(np.random.default_rng(3), 1000):
        root = find_monotone_root(Bezier(c32), tol=1e-13)
        reference = find_monotone_root(Bezier(c32.astype(np.float64)))
        assert abs(root - reference) <= tol32 + tol64, c32.tolist()


def test_float32_batch_converges() -> None:
    """The batch form takes the same steps: one degree, no row raises."""
    rng = np.random.default_rng(2)
    rows = [c for c in _random_monotone_float32(rng, 4000) if c.size == 6][:300]
    roots = find_monotone_root([Bezier(c) for c in rows])
    reference = [find_monotone_root(Bezier(c.astype(np.float64))) for c in rows]
    np.testing.assert_allclose(
        roots, reference, rtol=0.0, atol=get_strict(np.float32) + get_strict(np.float64)
    )


@pytest.mark.parametrize(
    ("coeffs", "root"),
    [([1.7e308, -1.7e308], 0.5), ([-1.7e308, -8.5e307, 1.7e308], (np.sqrt(5.0) - 1.0) / 2.0)],
    ids=["line", "quadratic"],
)
def test_overflowing_coefficient_spread_still_converges(coeffs: list[float], root: float) -> None:
    """A derivative that overflows gives no Newton step; the solve bisects to the root.

    The end spread overflows ``float64`` (``1.7e308 - (-1.7e308)``), so the derivative is infinite
    and a Newton step from it does not move; taking that step made the solve reach its cap.
    """
    got = find_monotone_root(Bezier(np.array(coeffs)))
    assert abs(got - root) <= 2.0 * get_strict(np.float64)


def test_float64_triple_roots_converge_within_the_cap() -> None:
    """``(t - k/d)^3`` converges at the strict ``float64`` tol for every ``k/d`` with ``d <= 16``.

    The bisection safeguard halves the bracket at least every two iterations, so a cap of
    ``2 (ceil(log2(1 / tol)) + 1)`` iterations reaches ``tol``; the shared cap of 64 raised on some
    of these (``(t - 5/8)^3`` among them). Oracle: analytic. The Bernstein coefficients of
    ``(t - r)^3`` are ``(-r)^(3 - i) (1 - r)^i``. Each of de Casteljau's 3 levels forms convex
    combinations, two roundings each, so to first order the evaluation errs by at most
    ``2 * 3 * u * max|b|``; the evaluated sign change is then within the cube root of that of
    ``r``. The bar doubles the error (an admitted margin for the second-order terms) and adds
    ``tol``.
    """
    tol = float(get_strict(np.float64))
    u = float(np.finfo(np.float64).eps) / 2
    for d in range(2, 17):
        for k in range(1, d):
            r = k / d
            coeffs = np.array([(-r) ** (3 - i) * (1 - r) ** i for i in range(4)])
            x = float(find_monotone_root(Bezier(coeffs)))
            band = (2 * 6 * u * np.abs(coeffs).max()) ** (1 / 3) + tol
            assert abs(x - r) <= band, (k, d, x)
