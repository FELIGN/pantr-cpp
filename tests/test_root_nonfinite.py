"""Root finding refuses non-finite data and a non-finite ``tol`` instead of answering.

``find_monotone_root`` of NaN coefficients returned a finite value (about 1), and ``find_roots``
an empty set, which reads as "no root" (the policy: no function returns a finite value or reports
success on non-finite data). ``minimize_degree(tol=inf)`` reduced anything to degree 0, and
``bspline.find_roots(tol=nan)`` returned no roots.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from pantr.bezier import Bezier, find_monotone_root, find_roots
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.bspline import find_roots as find_bspline_roots

_SOLVERS: dict[str, Callable[[list[Bezier]], object]] = {
    "find_roots": lambda b: find_roots(b[0]),
    "find_roots batch": lambda b: find_roots(b),
    "find_monotone_root": lambda b: find_monotone_root(b[0]),
    "find_monotone_root batch": lambda b: find_monotone_root(b),
}


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=str)
@pytest.mark.parametrize("solver", list(_SOLVERS), ids=str)
def test_non_finite_coefficients_raise(solver: str, bad: float, dtype: type) -> None:
    """One NaN or infinite coefficient in any row raises ``ValueError``."""
    good = np.array([-1.0, 0.5, 1.0], dtype=dtype)
    poisoned = good.copy()
    poisoned[1] = bad
    rows = [Bezier(poisoned)] if "batch" not in solver else [Bezier(good), Bezier(poisoned)]
    with pytest.raises(ValueError, match="coefficients must be finite"):
        _SOLVERS[solver](rows)


@pytest.mark.parametrize("solver", list(_SOLVERS), ids=str)
def test_finite_coefficients_are_solved(solver: str) -> None:
    """Control: the same rows with finite coefficients are solved."""
    good = Bezier(np.array([-1.0, 0.5, 1.0]))
    _SOLVERS[solver]([good, good])


@pytest.mark.parametrize("tol", [np.inf, np.nan], ids=str)
def test_minimize_degree_refuses_a_non_finite_tol(tol: float) -> None:
    """``tol=inf`` reduced a cubic to degree 0; NaN and infinity now raise."""
    cubic = Bezier(np.array([[0.0], [1.0], [-1.0], [2.0]]))
    with pytest.raises(ValueError, match="tol must be finite"):
        cubic.minimize_degree(tol=tol)
    assert cubic.minimize_degree(tol=1e-12).degree == (3,)


@pytest.mark.parametrize("tol", [np.inf, np.nan], ids=str)
def test_bspline_find_roots_refuses_a_non_finite_tol(tol: float) -> None:
    """``tol=nan`` returned no roots for a spline with a root at 0.5."""
    space = BsplineSpace([BsplineSpace1D(np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), 2)])
    spline = Bspline(space, np.array([[-1.0], [0.0], [1.0]]))
    with pytest.raises(ValueError, match="tol must be positive and finite"):
        find_bspline_roots(spline, tol=tol)
    np.testing.assert_allclose(find_bspline_roots(spline), [0.5], rtol=0.0, atol=1e-12)
