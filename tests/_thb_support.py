"""Test-only helpers shared by the THB suites: dense basis rows and random cell points.

This module is **not** part of the public ``pantr`` API and is underscore-prefixed so pytest does
not collect it. It holds what more than one THB test file needs and nothing that evaluates the
hierarchical basis by another route than the public one:

- :func:`tensor_product_rows`: the dense values of a tensor-product space at points, built from
  ``BsplineSpace.tabulate_basis``, the second route every comparison with an unrefined or fully
  refined hierarchical space rests on;
- :func:`hierarchical_rows`: the dense values of a hierarchical space at points, located cell by
  cell through ``HierarchicalGrid.locate`` and ``THBSplineSpace.tabulate_basis``;
- :func:`random_cell_points`: seeded random points of one cell, the first two on the cell's
  corners so that the boundary is always sampled;
- :data:`VALUE_ATOL`: the bound under which two routes to one basis value agree.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pantr.bspline import BsplineSpace, THBSplineSpace
from pantr.tolerance import get_default

VALUE_ATOL = float(get_default(np.float64))
"""Absolute bound on a basis value, or an operator entry, that two routes compute from one input.

A basis value lies in ``[0, 1]``, so the relative tier of ``get_default`` (64 eps: a short
algorithm of about 16 operations plus build slack) is also the absolute one. The routes compared
multiply the same at most three 1D factors, or add a contraction over the refinement coefficients
of one function (at most ``(p + 2) ** d`` of them for a halving); an operator entry of a round
trip is the same code on the same inputs. All of them are inside that tier.
"""


def tensor_product_rows(
    space: BsplineSpace, pts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return the dense ``(n_pts, n_dofs)`` values of a tensor-product space at ``pts``.

    Built from ``BsplineSpace.tabulate_basis`` (the local ``(p + 1) ** d`` values and the first
    function per direction), scattered to the C-order dof of each function.

    Args:
        space (BsplineSpace): The tensor-product space.
        pts (npt.NDArray[np.float64]): Points ``(n_pts, dim)`` in the domain.

    Returns:
        npt.NDArray[np.float64]: One row per point; zero outside the point's local window.
    """
    values, first = space.tabulate_basis(pts)
    rows = np.zeros((pts.shape[0], space.num_total_basis))
    window = np.indices(tuple(d + 1 for d in space.degrees)).reshape(space.dim, -1).T
    for i in range(pts.shape[0]):
        multi = first[i] + window
        flat = np.ravel_multi_index(tuple(multi.T), space.num_basis)
        rows[i, flat] = values[i].reshape(-1)
    return rows


def hierarchical_rows(thb: THBSplineSpace, pts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the dense ``(n_pts, n_dofs)`` values of a hierarchical space at ``pts``.

    Each point is located in its leaf cell and tabulated there; a point on a cell boundary takes
    the cell ``locate`` assigns it to.

    Args:
        thb (THBSplineSpace): The hierarchical space.
        pts (npt.NDArray[np.float64]): Points ``(n_pts, dim)`` in the domain.

    Returns:
        npt.NDArray[np.float64]: One row per point; zero for the functions not listed on its cell.

    Raises:
        ValueError: If a point lies outside the domain.
    """
    rows = np.zeros((pts.shape[0], thb.num_total_basis))
    for i in range(pts.shape[0]):
        cid = thb.grid.locate(pts[i])
        if cid is None:
            raise ValueError(f"point {pts[i]} lies outside the domain")
        values, dofs = thb.tabulate_basis(cid, pts[i : i + 1])
        rows[i, dofs] = values[0]
    return rows


def random_cell_points(
    thb: THBSplineSpace, cid: int, n_pts: int, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Return ``n_pts`` seeded random points of cell ``cid``, the first two on its corners.

    Args:
        thb (THBSplineSpace): The space whose grid holds the cell.
        cid (int): The active cell.
        n_pts (int): Number of points, at least 2.
        rng (np.random.Generator): The seeded generator drawing the other points.

    Returns:
        npt.NDArray[np.float64]: Points ``(n_pts, dim)`` in the closed cell; the first is the
        lower corner and the second the upper one.
    """
    lo, hi = thb.grid.cell_bounds(cid)
    unit = rng.uniform(size=(n_pts, thb.dim))
    unit[0] = 0.0
    unit[1] = 1.0
    pts: npt.NDArray[np.float64] = lo + (hi - lo) * unit
    return pts
