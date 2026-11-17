"""A THB refinement whose two-scale matrix overflows its dtype raises instead of giving NaN.

The two-scale matrix between levels is computed in the knots' storage type. On a float32 knot
vector whose span exceeds the float32 range, a ratio of knot differences is ``inf / inf``, and the
prolongation built on it held NaN entries that its residual gate let through (the policy: a
non-finite value born from finite input raises).
"""

from __future__ import annotations

import numpy as np
import pytest

from pantr.bspline import BsplineSpace, BsplineSpace1D, create_thb_space


def _symmetric_knots(s: float) -> np.ndarray:
    """Degree-2 open knots on ``[-s, s]`` with one interior knot at 0, in float32."""
    v = np.float32(s)
    return np.array([-v, -v, -v, 0.0, v, v, v], dtype=np.float32)


def test_refining_past_the_float32_range_raises() -> None:
    """Knots on ``[-2e38, 2e38]`` span more than float32 holds: refining them raises."""
    coarse = create_thb_space(BsplineSpace([BsplineSpace1D(_symmetric_knots(2e38), 2)]))
    with pytest.raises(OverflowError, match="two-scale"):
        coarse.refine_region(0, [0], [2])


def test_refining_inside_the_float32_range_gives_a_finite_prolongation() -> None:
    """Control: the same knots on ``[-1, 1]`` refine and prolong to finite entries."""
    coarse = create_thb_space(BsplineSpace([BsplineSpace1D(_symmetric_knots(1.0), 2)]))
    fine = coarse.refine_region(0, [0], [2])
    prolongation = np.asarray(coarse.prolongation_to(fine))
    assert prolongation.shape == (6, 4)
    assert np.isfinite(prolongation).all()
