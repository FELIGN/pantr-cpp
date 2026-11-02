"""Cell-coupling (dual) graph of a B-spline or THB-spline space.

Builds the graph whose vertices are the cells of a space's grid and whose edges
join cells that share at least one basis function (DOF), weighted by the number
of shared functions. This is the input a graph partitioner (METIS / Scotch)
needs to minimize cross-rank DOF coupling; it is produced serially with no MPI
and no external dependency, and consumed later by the optional graph-partition
backends.

The graph is emitted in the standard CSR adjacency format used by METIS and
Scotch (``xadj`` / ``adjncy`` / ``edge_weights`` / ``vertex_weights``, corresponding
to METIS ``xadj`` / ``adjncy`` / ``adjwgt`` / ``vwgt``), so a backend can hand it
to those libraries with no reshaping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from .._ext import ext
from ._bspline_space_nd import BsplineSpace
from ._thb_spline_space import THBSplineSpace

if TYPE_CHECKING:
    import numpy.typing as npt

_cpp: Any = ext.bspline


class CouplingGraph(NamedTuple):
    """Cell-coupling graph of a space, in METIS / Scotch CSR adjacency format.

    A :class:`typing.NamedTuple` returned by :func:`coupling_graph`. The graph is
    undirected (symmetric adjacency) and has no self-loops:

    - ``num_vertices`` -- number of cells (graph vertices).
    - ``xadj`` -- CSR row pointers, shape ``(num_vertices + 1,)``; the neighbors of
      cell ``c`` are ``adjncy[xadj[c]:xadj[c + 1]]``. (METIS ``xadj``.) Read-only.
    - ``adjncy`` -- concatenated neighbor cell ids, shape ``(xadj[-1],)``. (METIS
      ``adjncy``.) Read-only.
    - ``edge_weights`` -- per-adjacency-entry weight = number of basis functions the
      two cells share, aligned with ``adjncy``. (METIS ``adjwgt``.) Read-only.
    - ``vertex_weights`` -- per-cell weight (assembly cost), shape
      ``(num_vertices,)``; uniform ``1.0`` unless ``cell_weights`` was given. (METIS
      ``vwgt``.) Read-only.
    """

    num_vertices: int
    xadj: npt.NDArray[np.int64]
    adjncy: npt.NDArray[np.int64]
    edge_weights: npt.NDArray[np.int64]
    vertex_weights: npt.NDArray[np.float64]


def coupling_graph(
    space: BsplineSpace | THBSplineSpace,
    *,
    cell_weights: npt.ArrayLike | None = None,
) -> CouplingGraph:
    """Build the cell-coupling graph of a B-spline or THB-spline space.

    Two cells are joined by an edge when they share at least one basis function;
    the edge weight is the number of shared functions, and each vertex (cell)
    carries an optional assembly-cost weight. The result is the dual graph a
    graph partitioner uses to minimize cross-rank DOF coupling.

    Args:
        space (BsplineSpace | THBSplineSpace): The space whose cells to couple. A
            :class:`BsplineSpace` must be non-periodic.
        cell_weights (npt.ArrayLike | None): Optional per-cell assembly cost, shape
            ``(num_cells,)``, finite and non-negative. ``None`` means uniform.

    Returns:
        CouplingGraph: The coupling graph in METIS / Scotch CSR adjacency format.

    Raises:
        TypeError: If ``space`` is neither a :class:`BsplineSpace` nor a
            :class:`THBSplineSpace`.
        ValueError: If ``space`` is a periodic :class:`BsplineSpace`, or if
            ``cell_weights`` has the wrong shape or invalid values.

    Example:
        >>> from pantr.bspline import coupling_graph, create_uniform_space
        >>> space = create_uniform_space(2, 4)
        >>> graph = coupling_graph(space)
        >>> graph.num_vertices
        4
        >>> int(graph.xadj[-1]) == graph.adjncy.size
        True
    """
    if isinstance(space, THBSplineSpace):
        weights = _coerce_cell_weights(cell_weights, space.grid.num_cells)
        space._check_not_stale()
        arrays = _cpp.coupling_graph_thb(space._h, weights)
    elif isinstance(space, BsplineSpace):
        if space._h is None:
            raise ValueError("a B-spline space needs at least one direction")
        weights = _coerce_cell_weights(cell_weights, space.num_total_intervals)
        arrays = _cpp.coupling_graph(space._h, weights)
    else:
        raise TypeError(
            f"coupling_graph expects a BsplineSpace or THBSplineSpace; got {type(space).__name__}."
        )
    num_vertices, xadj, adjncy, edge_weights, vertex_weights = arrays
    return CouplingGraph(int(num_vertices), xadj, adjncy, edge_weights, vertex_weights)


def _coerce_cell_weights(
    cell_weights: npt.ArrayLike | None, n_cells: int
) -> npt.NDArray[np.float64] | None:
    """Coerce ``cell_weights`` to a 1-D ``float64`` array.

    The length and the values (finite, non-negative) are checked by the compiled builder.

    Args:
        cell_weights (npt.ArrayLike | None): Candidate per-cell weights, or ``None``.
        n_cells (int): Expected length (for the message).

    Returns:
        npt.NDArray[np.float64] | None: The coerced weights, or ``None`` if the input was
        ``None``.

    Raises:
        ValueError: If the weights are not 1-D.
    """
    if cell_weights is None:
        return None
    weights = np.asarray(cell_weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError(f"cell_weights must have shape ({n_cells},); got {weights.shape}.")
    return np.ascontiguousarray(weights)


__all__ = ["CouplingGraph", "coupling_graph"]
