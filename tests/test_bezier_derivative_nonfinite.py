"""A derivative of an order above the degree on non-finite control points is NaN, not zero.

The derivative of order ``k > p`` of a degree-``p`` polynomial is identically zero, and the
evaluator writes that zero without forming the derivative. A NaN or infinite control point
makes the polynomial undefined, so its derivative must read NaN rather than an exact zero (the
policy: non-finite in, non-finite out).
"""

from __future__ import annotations

import numpy as np
import pytest

from pantr.bezier import Bezier


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_curve_with_a_bad_control_point(bad: float) -> None:
    """Above the degree: NaN in the bad column, zero in a finite one."""
    b = Bezier(np.array([[1.0, 1.0], [bad, 2.0], [4.0, 3.0]]))
    got = b.evaluate_derivatives(np.array([0.25, 0.5]), 3)
    assert np.isnan(got[:, 0]).all()
    assert (got[:, 1] == 0.0).all()


def test_surface_with_nan_control_points() -> None:
    """The same in two directions, the order above the degree in one direction only."""
    b = Bezier(np.full((2, 2, 1), np.nan))
    got = b.evaluate_derivatives(np.array([[0.3, 0.4]]), [2, 0])
    assert np.isnan(got).all()


def test_finite_control_points_still_give_zero() -> None:
    """Control: finite control points and points give an exact zero above the degree."""
    b = Bezier(np.array([[1.0], [-2.0], [4.0]]))
    got = b.evaluate_derivatives(np.array([0.0, 0.3, 1.0]), 3)
    assert (got == 0.0).all()
