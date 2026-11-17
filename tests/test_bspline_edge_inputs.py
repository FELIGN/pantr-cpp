"""Edge inputs of the B-spline fits and spaces that the previous release handled one way.

- One derivative row per end on a direction with a single node: NumPy broadcast the row onto
  the empty slice of rows to replace, a no-op, so the fit is the plain interpolation.
- A space with no direction: the fits raised ``ValueError``, not an ``IndexError`` from inside.
- A stored state with an infinite knot: the constructor refuses one, so a pickle must too.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import (
    BsplineSpace,
    BsplineSpace1D,
    create_uniform_space,
    interpolate_bspline,
    l2_project_bspline,
    quasi_interpolate_bspline,
)
from pantr.quad import PointsLattice


def _square(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Get ``x ** 2`` of the first coordinate.

    Args:
        p (npt.NDArray[np.float64]): Points, one per row.

    Returns:
        npt.NDArray[np.float64]: One value per point.
    """
    return p[:, 0] ** 2


def _square_on_lattice(lattice: PointsLattice) -> npt.NDArray[np.float64]:
    """Get ``x ** 2`` on a one-direction lattice.

    Args:
        lattice (PointsLattice): The lattice.

    Returns:
        npt.NDArray[np.float64]: One value per lattice point.
    """
    return np.asarray(lattice.pts_per_dir[0], dtype=np.float64) ** 2


@pytest.mark.parametrize("derivs", [(1, 0), (0, 1), (1, 1)])
def test_single_node_with_one_derivative_row_is_the_plain_fit(derivs: tuple[int, int]) -> None:
    """Bug caught: the C++ broadcast check raised where NumPy broadcast one row onto none."""
    space = create_uniform_space(3, 4)
    nodes = [np.array([0.5])]
    plain = interpolate_bspline(_square_on_lattice, space, nodes=nodes)
    with_derivs = interpolate_bspline(
        _square_on_lattice, space, nodes=nodes, boundary_derivatives=[derivs]
    )
    np.testing.assert_array_equal(
        np.asarray(with_derivs.control_points), np.asarray(plain.control_points)
    )


@pytest.mark.parametrize(
    ("fit", "message"),
    [
        (l2_project_bspline, "Points lattice must have at least 1 dimension"),
        (quasi_interpolate_bspline, "need at least one array to stack"),
    ],
)
def test_a_space_with_no_direction_raises_value_error(fit: object, message: str) -> None:
    """Bug caught: ``BsplineSpace.dtype`` indexing an empty tuple, an ``IndexError``."""
    assert callable(fit)
    with pytest.raises(ValueError, match=message):
        fit(_square, BsplineSpace([]))


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_a_stored_state_with_an_infinite_knot_is_refused(bad: float) -> None:
    """The constructor refuses an infinite knot; a stored state skipped that check."""
    knots = np.array([0.0, 0.0, 1.0, 1.0])
    knots[-1 if bad > 0 else 0] = bad
    tol = BsplineSpace1D([0.0, 0.0, 1.0, 1.0], 1).tolerance
    with pytest.raises(ValueError, match="stored knot vector must be finite"):
        BsplineSpace1D._from_state(knots, 1, False, tol)
