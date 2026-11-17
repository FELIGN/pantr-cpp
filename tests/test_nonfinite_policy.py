"""Non-finite input is never accepted on a comparison that NaN fails.

``minimize_degree`` accepted a reduction when ``rel_error >= tol`` was false, which it is for a
NaN error; the knot factories refused ``start >= end``, which a NaN endpoint passes. Each now
refuses the non-finite case (the policy: non-finite in, non-finite out, and no validation or
acceptance reported on non-finite data).
"""

from __future__ import annotations

import numpy as np
import pytest

from pantr.bezier import Bezier
from pantr.bspline import create_uniform_open_knots, create_uniform_periodic_knots


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_minimize_degree_does_not_reduce_non_finite_control_points(bad: float) -> None:
    """A line written at degree 3 with one bad control point keeps its degree and its NaN."""
    cp = np.array([[0.0], [1.0 / 3.0], [2.0 / 3.0], [1.0]])
    cp[1, 0] = bad
    out = Bezier(cp).minimize_degree()
    assert out.degree == (3,)
    np.testing.assert_array_equal(np.asarray(out.control_points), cp)


def test_minimize_degree_still_reduces_finite_control_points() -> None:
    """Control: the same line with finite control points reduces to degree 1."""
    cp = np.array([[0.0], [1.0 / 3.0], [2.0 / 3.0], [1.0]])
    assert Bezier(cp).minimize_degree().degree == (1,)


@pytest.mark.parametrize("factory", [create_uniform_open_knots, create_uniform_periodic_knots])
@pytest.mark.parametrize(
    "domain", [(np.nan, 1.0), (0.0, np.nan), (-np.inf, 1.0), (0.0, np.inf)], ids=str
)
def test_knot_factories_refuse_a_non_finite_endpoint(
    factory: object, domain: tuple[float, float]
) -> None:
    """A NaN or infinite domain end raises instead of returning NaN knots."""
    assert callable(factory)
    with pytest.raises(ValueError, match="domain must be finite"):
        factory(2, 2, domain=domain)
