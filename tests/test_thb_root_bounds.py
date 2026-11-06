"""A THB space checks each axis of its grid's root bounds against that axis's own knot tolerance.

The check used the root space's tolerance, the largest over the axes, so on a space whose axes
differ in scale (``[0, 1e6] x [0, 1e-3]``) a bound of the small axis off by ``1e-9``, a relative
``1e-6`` of its span, was accepted.

Oracle: the per-axis knot tolerance, ``8 eps`` times the axis's knot scale.
"""

from __future__ import annotations

import numpy as np
import pytest

from pantr.bspline import BsplineSpace, BsplineSpace1D, THBSplineSpace
from pantr.grid import HierarchicalGrid, TensorProductGrid


def _open_knots(length: float, n: int = 4, degree: int = 2) -> np.ndarray:
    """Open knots on ``[0, length]`` with ``n`` uniform intervals."""
    return np.concatenate([[0.0] * degree, np.linspace(0.0, length, n + 1), [length] * degree])


def _grid(len0: float, len1: float, n: int = 4) -> HierarchicalGrid:
    """A factor-2 hierarchy over the ``n x n`` grid of ``[0, len0] x [0, len1]``."""
    return HierarchicalGrid(
        TensorProductGrid([np.linspace(0.0, len0, n + 1), np.linspace(0.0, len1, n + 1)]), 2
    )


_SPACE = BsplineSpace([BsplineSpace1D(_open_knots(1e6), 2), BsplineSpace1D(_open_knots(1e-3), 2)])


def test_small_axis_bound_off_by_a_millionth_of_its_span_raises() -> None:
    """The small axis's upper bound off by ``1e-9`` is refused, as it is on a same-scale space."""
    with pytest.raises(ValueError, match="grid root bounds must match root_space domain"):
        THBSplineSpace(_SPACE, _grid(1e6, 1e-3 + 1e-9))


def test_exact_bounds_are_accepted() -> None:
    """Control: the same space with the exact bounds is built."""
    THBSplineSpace(_SPACE, _grid(1e6, 1e-3))
