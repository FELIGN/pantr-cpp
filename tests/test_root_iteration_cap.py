"""The Newton/bisection hybrid's iteration cap is reported, never passed off as a root.

The hybrid keeps a bracket around the root and stops when it is narrower than ``tol``. When it
cannot (a ``tol`` below the spacing of doubles at the root, or, for ``find_roots``' hybrid, a
root where the derivative also vanishes, whose Newton steps converge linearly) its iteration cap
is reached with a bracket wider than ``tol``, whose midpoint can be far from the root. The
previous release returned that midpoint as the root. A ``float32`` evaluation whose rounding
kept every Newton step on one side of a simple root reached the cap too; the monotone solver's
bisection safeguard now makes it converge (``find_roots``' Yuksel walk keeps the previous hybrid,
so it still warns).

Oracle: encoded truth, the polynomials written as products of linear factors with rational roots
and converted to Bernstein form in exact arithmetic.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb

import numpy as np
import pytest
from numpy import typing as npt

from pantr.bezier import Bezier, find_monotone_root, find_roots
from pantr.tolerance import get_strict


def _bernstein(roots: list[Fraction]) -> npt.NDArray[np.float64]:
    """Get the Bernstein coefficients of ``prod (t - r)``.

    Args:
        roots (list[Fraction]): The roots.

    Returns:
        npt.NDArray[np.float64]: ``len(roots) + 1`` coefficients.
    """
    p = [Fraction(1)]
    for r in roots:
        q = [Fraction(0)] * (len(p) + 1)
        for i, a in enumerate(p):
            q[i + 1] += a
            q[i] -= r * a
        p = q
    n = len(p) - 1
    return np.array(
        [
            float(sum(Fraction(comb(i, j), comb(n, j)) * p[j] for j in range(i + 1)))
            for i in range(n + 1)
        ]
    )


_TRIPLE = _bernstein([Fraction(1, 3)] * 3)
"""``(t - 1/3)^3``: the previous release's monotone solver returned about ``2/3``; it now converges
at the strict tolerance, and a ``tol`` of ``2^-60`` cannot bracket it."""

_SIMPLE = _bernstein([Fraction(3, 20)] + [Fraction(-1)] * 3)
"""``(t - 3/20)(t + 1)^3``: a simple root, which a ``tol`` of ``1e-20`` cannot bracket."""


@pytest.mark.parametrize(
    ("coeffs", "tol"),
    [(_TRIPLE, 2.0**-60), (_SIMPLE, 1e-20)],
    ids=["triple-tol-below-spacing", "tol-below-spacing"],
)
def test_monotone_root_raises_at_the_cap(
    coeffs: npt.NDArray[np.floating], tol: float | None
) -> None:
    """``find_monotone_root`` raises rather than return the midpoint of a wide bracket."""
    with pytest.raises(RuntimeError, match="did not narrow the bracket to tol"):
        find_monotone_root(Bezier(coeffs), tol=tol)
    with pytest.raises(RuntimeError, match="row 1: find_monotone_root"):
        # Row 0 is a line written at the same degree with integer coefficients symmetric about
        # zero, so false position starts at its root 1/2, an exact zero of its evaluation: it
        # converges whatever the tol. (`linspace(-1, 1, 4)` is not symmetric to the bit.)
        line = (2.0 * np.arange(coeffs.size) - (coeffs.size - 1)).astype(coeffs.dtype)
        find_monotone_root([Bezier(line), Bezier(coeffs)], tol=tol)


@pytest.mark.parametrize("root", [Fraction(5, 8), Fraction(3, 20)], ids=str)
def test_float32_simple_root_converges(root: Fraction) -> None:
    """``(t - r)(t + 1)^3`` in ``float32`` reached the cap; it now converges within ``tol``.

    The bound is ``tol`` plus the shift of the sign change by the coefficients' rounding to
    ``float32`` and the evaluation's: a few ``float32`` units of roundoff over the slope, which is
    of order one at these roots, so ``2 tol`` bounds it.
    """
    coeffs = _bernstein([root] + [Fraction(-1)] * 3).astype(np.float32)
    tol = get_strict(np.float32)
    assert abs(find_monotone_root(Bezier(coeffs)) - float(root)) <= 2.0 * tol
    np.testing.assert_allclose(
        find_monotone_root([Bezier(coeffs)] * 2), [float(root)] * 2, rtol=0.0, atol=2.0 * tol
    )


def test_find_roots_warns_when_yuksel_reaches_the_cap() -> None:
    """``find_roots`` reports a capped solve of Yuksel's walk as a gap, hence a warning.

    ``(t - 3/20)(t + 1)^3`` in ``float32``: the solve stalls and the value it returns (half the
    root) is not within ``tol``; the result says so instead of passing as complete.
    """
    c = _bernstein([Fraction(3, 20)] + [Fraction(-1)] * 3).astype(np.float32)
    with pytest.warns(RuntimeWarning, match="subdivision budget"):
        find_roots(Bezier(c))
    with pytest.warns(RuntimeWarning, match="1 of 1 polynomials"):
        find_roots([Bezier(c)])


def test_a_converging_solve_is_silent() -> None:
    """Control: a simple root in ``float64`` converges, with no error and no warning."""
    c = _bernstein([Fraction(5, 8)] + [Fraction(-1)] * 3)
    assert abs(find_monotone_root(Bezier(c)) - 0.625) <= 1e-12
    np.testing.assert_allclose(find_roots(Bezier(c)), [0.625], rtol=0.0, atol=1e-12)
