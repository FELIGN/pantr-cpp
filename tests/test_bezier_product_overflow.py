"""The Bernstein product (``Bezier.multiply``) at high degree and on special values.

Oracle: analytic. The product of two Béziers whose control points are all ``1`` is the constant
``1``, whose control points at any degree are all ``1`` (the Bernstein basis is a partition of
unity), so every product control point must be ``1`` up to the formula's rounding.
"""

from __future__ import annotations

import numpy as np
import pytest

from pantr.bezier import Bezier


@pytest.mark.parametrize("degree", [66, 70, 100])
@pytest.mark.parametrize("dim", [1, 2])
def test_float32_product_of_ones_is_one_at_high_degree(degree: int, dim: int) -> None:
    """Bug caught: the coefficient ``C(p, i) C(q, j) / C(p + q, i + j)`` formed in ``float32``.

    Its intermediate ``C(p, i) C(q, j)`` overflows ``float32`` near degree 66 although the
    coefficient is at most 1, so the product of two constants had infinite or NaN control
    points, silently. The tolerance is the sum's rounding: ``p + q + 1`` terms of at most one
    unit each, a few units of ``float32`` epsilon per term.
    """
    shape = (degree + 1,) * dim + (1,)
    one = Bezier(np.ones(shape, dtype=np.float32))
    cp = np.asarray(one.multiply(one).control_points)
    assert np.isfinite(cp).all()
    bound = 4.0 * (2 * degree + 1) ** dim * float(np.finfo(np.float32).eps)
    np.testing.assert_allclose(cp, 1.0, rtol=0.0, atol=bound)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_overflow_from_finite_control_points_raises(dtype: type[np.floating]) -> None:
    """A product beyond the dtype's range raises ``OverflowError`` instead of returning ``inf``."""
    big = float(np.sqrt(np.finfo(dtype).max)) * 4.0
    a = Bezier(np.full((3, 1), big, dtype=dtype))
    with pytest.raises(OverflowError, match="overflows the range of its dtype"):
        a.multiply(a)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_control_points_propagate(bad: float) -> None:
    """A NaN or infinite control point gives non-finite product control points, without raising.

    Where it sits matters: ``d_0 = f_0 g_0`` reads only the first points, so a bad last point
    leaves ``d_0`` finite and makes every ``d_k`` with ``k >= p`` non-finite.
    """
    f = Bezier(np.array([[1.0], [2.0], [bad]]))
    g = Bezier(np.array([[3.0], [4.0]]))
    cp = np.asarray(f.multiply(g).control_points).ravel()
    assert np.isfinite(cp[0])
    assert not np.isfinite(cp[2:]).any()
