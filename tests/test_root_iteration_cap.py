"""The Newton/bisection hybrid's iteration cap is reported, never passed off as a root.

The hybrid keeps a bracket around the root and stops when it is narrower than ``tol``. When a
Newton sequence approaches the root from one side (a root where the derivative also vanishes, or
a ``float32`` evaluation whose rounding stalls the step) the far end never moves, the cap of 64
iterations is reached with a wide bracket, and its midpoint can be far from the root. The
previous release returned that midpoint as the root.

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
"""``(t - 1/3)^3``: the previous release's monotone solver returned about ``2/3``."""

_STALLS_IN_FLOAT32 = _bernstein([Fraction(5, 8)] + [Fraction(-1)] * 3).astype(np.float32)
"""``(t - 5/8)(t + 1)^3`` in ``float32``: the previous release returned about ``0.606``."""


@pytest.mark.parametrize("coeffs", [_TRIPLE, _STALLS_IN_FLOAT32], ids=["triple", "float32"])
def test_monotone_root_raises_at_the_cap(coeffs: npt.NDArray[np.floating]) -> None:
    """``find_monotone_root`` raises rather than return the midpoint of a wide bracket."""
    with pytest.raises(RuntimeError, match="did not narrow the bracket to tol"):
        find_monotone_root(Bezier(coeffs))
    with pytest.raises(RuntimeError, match="row 1: find_monotone_root"):
        # Row 0 is a line written at the same degree (root 1/2), which converges.
        line = np.linspace(-1.0, 1.0, coeffs.size).astype(coeffs.dtype)
        find_monotone_root([Bezier(line), Bezier(coeffs)])


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
