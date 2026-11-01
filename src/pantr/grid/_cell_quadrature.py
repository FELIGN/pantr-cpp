"""Per-cell quadrature: map a reference rule on the unit cube onto grid cells.

A :class:`pantr.quad.QuadratureRule` lives on the reference unit cube
``[0, 1]^ndim``. :func:`cell_quadrature` pushes it forward onto a grid's cells
with the per-cell affine map ``T(u) = lo + (hi - lo) * u`` (see
:meth:`pantr.grid.Grid.reference_map`): points are mapped by ``T`` and weights
are scaled by the cell volume ``prod(hi - lo)`` (the Jacobian determinant of
``T``). The reference weights sum to ``1``, so the mapped weights of a cell sum
to that cell's volume and ``sum_i w_i f(x_i)`` approximates the integral of
``f`` over the cell.

This is the uncut/background-cell quadrature bridge: a downstream consumer takes
this rule for interior cells and substitutes its own cut-cell rule on cells
flagged as cut.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._ext import ext

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..quad import QuadratureRule
    from ._grid import Grid

_cpp = ext.grid


def cell_quadrature(
    grid: Grid,
    rule: QuadratureRule,
    cells: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Map a reference :class:`~pantr.quad.QuadratureRule` onto grid cells.

    For each selected cell, the rule's unit-cube points are affinely mapped into
    the cell box and its weights are scaled by the cell volume.

    Args:
        grid (Grid): The grid whose cells are integrated. ``grid.ndim`` must
            equal ``rule.ndim``.
        rule (QuadratureRule): Reference rule on ``[0, 1]^ndim``.
        cells (npt.ArrayLike | None): Cell ids to map the rule onto. ``None``
            (the default) selects every cell in id order. Otherwise a 1D
            integer array-like of ids, each in ``[0, num_cells)`` (a scalar is
            treated as a single id); order and duplicates are preserved.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: A pair
        ``(points, weights)`` where ``points`` has shape
        ``(num_selected, num_points, ndim)`` and ``weights`` has shape
        ``(num_selected, num_points)``, ordered to match ``cells``.

    Raises:
        ValueError: If ``rule.ndim != grid.ndim``, ``cells`` is not a 1D integer
            array-like, or any id is outside ``[0, num_cells)``.

    Note:
        Only the requested cells' boxes are computed. Without a selection the
        boxes of every cell come from one pass over the grid.
    """
    ids = None if cells is None else _resolve_cell_ids(cells)
    points, weights = _cpp.cell_quadrature(
        grid._h,
        np.ascontiguousarray(rule.points, dtype=np.float64),
        np.ascontiguousarray(rule.weights, dtype=np.float64),
        ids,
    )
    return points, weights


def _resolve_cell_ids(cells: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Normalize a ``cells`` selector to a 1D ``int64`` array.

    The range of the ids is checked by the compiled mapping.

    Args:
        cells (npt.ArrayLike): Cell-id selector (a scalar id or a 1D integer
            array-like).

    Returns:
        npt.NDArray[np.int64]: A 1D array of cell ids.

    Raises:
        ValueError: If ``cells`` does not have an integer dtype or is not 1D.
    """
    arr = np.atleast_1d(np.asarray(cells))
    if arr.ndim != 1:
        raise ValueError(f"cells must be 1D; got shape {arr.shape}.")
    if arr.size == 0:
        return np.empty(0, dtype=np.int64)
    if arr.dtype.kind not in ("i", "u"):
        raise ValueError(f"cells must be an integer array of cell ids; got dtype {arr.dtype!r}.")
    return np.ascontiguousarray(arr, dtype=np.int64)


__all__ = ["cell_quadrature"]
