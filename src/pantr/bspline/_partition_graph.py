"""Graph partitioning of a space's cell-coupling graph into rank subdomains.

:func:`partition_graph` turns a :class:`~pantr.bspline.CouplingGraph` (built by
:func:`~pantr.bspline.coupling_graph`) into a :class:`~pantr.grid.Partition`, minimizing
the cross-rank DOF coupling that geometric backends (``block`` / ``rcb`` in
:func:`pantr.grid.partition_grid`) cannot see. Two backends, both serial (the partition
is computed redundantly per rank -- no MPI):

- ``"spectral"`` (default) -- recursive Fiedler bisection, compiled; no extra
  dependency; weight- and activity-aware.
- ``"metis"`` -- METIS via the optional ``pymetis`` package (``pip install
  'pantr[metis]'``); higher-quality min-cut. Raises a clear error if ``pymetis`` is
  absent, so the default install never requires it.

Exports:
    - :func:`partition_graph`
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .._ext import ext
from ..grid import Partition
from ._coupling_graph import CouplingGraph

if TYPE_CHECKING:
    import numpy.typing as npt

_cpp: Any = ext.bspline

_BACKENDS = ("spectral", "metis")
"""Graph-partition backends recognized by :func:`partition_graph`."""


def partition_graph(
    coupling: CouplingGraph,
    n_parts: int,
    *,
    backend: str = "spectral",
    cell_active: npt.ArrayLike | None = None,
) -> Partition:
    """Partition a cell-coupling graph into ``n_parts`` rank subdomains.

    Balances :attr:`CouplingGraph.vertex_weights` across parts while minimizing the
    weight of cut edges (shared DOFs). Cells excluded by ``cell_active`` get owner
    ``-1`` and are dropped from the graph.

    Args:
        coupling (CouplingGraph): The cell-coupling graph (see
            :func:`~pantr.bspline.coupling_graph`); its ``vertex_weights`` drive the
            load balance and ``edge_weights`` the cut cost.
        n_parts (int): Number of parts (ranks); must be ``>= 1``.
        backend (str): ``"spectral"`` (default; recursive Fiedler bisection, no extra
            dependency, never leaves a rank empty) or ``"metis"`` (METIS via the optional
            ``pymetis`` package; higher-quality min-cut, but may leave a rank empty).
        cell_active (npt.ArrayLike | None): Optional boolean mask, shape
            ``(coupling.num_vertices,)``; inactive cells get owner ``-1`` and are
            excluded. ``None`` means all active.

    Returns:
        Partition: A per-cell owner assignment with ``n_parts`` parts; ``-1`` for
        inactive cells, otherwise a rank in ``range(n_parts)``.

    Raises:
        TypeError: If ``coupling`` is not a :class:`CouplingGraph`.
        ValueError: If ``backend`` is unknown; if ``n_parts < 1``; if ``cell_active`` has
            the wrong shape or marks no cell active; if ``n_parts`` exceeds the number
            of active cells; or if the graph's arrays are inconsistent, its edge weights
            not integers or its vertex weights negative or not finite.
        ImportError: If ``backend="metis"`` but ``pymetis`` is not installed.
    """
    if not isinstance(coupling, CouplingGraph):
        raise TypeError(f"coupling must be a CouplingGraph; got {type(coupling).__name__}.")
    if backend not in _BACKENDS:
        valid = ", ".join(repr(b) for b in _BACKENDS)
        raise ValueError(f"unknown backend {backend!r}; valid backends: {valid}.")
    if n_parts < 1:
        raise ValueError(f"n_parts must be >= 1; got {n_parts}.")

    n_parts = int(n_parts)
    n = int(coupling.num_vertices)
    active = _coerce_active(cell_active, n)
    arrays = _graph_arrays(coupling)
    if backend == "spectral":
        return Partition._from_handle(_cpp.partition_graph_spectral(n, *arrays, n_parts, active))
    # "metis": the compiled side checks the request and extracts the active subgraph (rows
    # sorted), pymetis partitions it, and the owners go back through the compiled Partition.
    active_idx, xadj, adjncy, edge_weights, vertex_weights = _cpp.graph_active_subgraph(
        n, *arrays, n_parts, active
    )
    owner = np.full(n, -1, dtype=np.int32)
    owner[active_idx] = _metis_partition(xadj, adjncy, edge_weights, vertex_weights, n_parts)
    return Partition(owner, n_parts)


def _graph_arrays(
    coupling: CouplingGraph,
) -> tuple[
    npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]
]:
    """Coerce a coupling graph's arrays to C-contiguous ``int64`` / ``float64``.

    Their lengths and values are checked by the compiled partitioner.

    Args:
        coupling (CouplingGraph): The graph.

    Returns:
        tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64],
        npt.NDArray[np.float64]]: ``xadj``, ``adjncy``, ``edge_weights``, ``vertex_weights``.

    Raises:
        ValueError: If ``edge_weights`` holds a value that is not an integer in the ``int64``
            range.
    """
    edge_weights = np.asarray(coupling.edge_weights)
    if not _int64_representable(edge_weights):
        # A conversion to int64 would truncate, wrap or (for inf, NaN and floats past 2**63) be
        # undefined, silently.
        raise ValueError("edge_weights must be integers within the int64 range.")
    return (
        np.ascontiguousarray(coupling.xadj, dtype=np.int64),
        np.ascontiguousarray(coupling.adjncy, dtype=np.int64),
        np.ascontiguousarray(edge_weights, dtype=np.int64),
        np.ascontiguousarray(coupling.vertex_weights, dtype=np.float64),
    )


def _int64_representable(values: npt.NDArray[Any]) -> bool:
    """Tell whether every value converts to ``int64`` exactly.

    Args:
        values (npt.NDArray[Any]): An array of any real dtype.

    Returns:
        bool: ``True`` for integers in ``[-2**63, 2**63)``: any signed integer dtype of at most
        64 bits, an unsigned one whose values stay below ``2**63``, or floats that are finite,
        integral and in that range (``2.0**63`` is exact, so the comparisons are too).
    """
    if np.issubdtype(values.dtype, np.signedinteger):
        return True
    if np.issubdtype(values.dtype, np.unsignedinteger):
        return values.size == 0 or int(values.max()) <= int(np.iinfo(np.int64).max)
    with np.errstate(invalid="ignore"):
        return bool(
            np.all(np.isfinite(values))
            and np.all(values == np.rint(values))
            and np.all((values >= -(2.0**63)) & (values < 2.0**63))
        )


def _metis_partition(
    xadj: npt.NDArray[np.int64],
    adjncy: npt.NDArray[np.int64],
    edge_weights: npt.NDArray[np.int64],
    vertex_weights: npt.NDArray[np.float64],
    n_parts: int,
) -> npt.NDArray[np.int32]:
    """Partition by METIS (k-way min-cut) via the optional ``pymetis`` package.

    Vertex weights are rounded to integers (clamped to ``>= 1``, as METIS requires
    positive integer weights) and edge weights to integers (must be ``> 0``); METIS then
    minimizes the cut while balancing the integer vertex weights. Unlike the spectral
    backend, METIS does not guarantee every part is non-empty.

    METIS implements the multilevel k-way partitioning scheme of
    :cite:t:`karypis1998metis`.

    Args:
        xadj (npt.NDArray[np.int64]): CSR row pointers of the (active) subgraph (coerced to
            ``pymetis.zero_copy_dtype()`` internally).
        adjncy (npt.NDArray[np.int64]): CSR neighbour indices, ascending per row.
        edge_weights (npt.NDArray[np.int64]): Per-edge weights aligned with ``adjncy``;
            must be ``>= 1``.
        vertex_weights (npt.NDArray[np.float64]): Per-vertex weights.
        n_parts (int): Number of parts (``>= 1``).

    Returns:
        npt.NDArray[np.int32]: Per-vertex owner in ``range(n_parts)``.

    Raises:
        ImportError: If ``pymetis`` is not installed or fails to load, and
            ``n_parts > 1``.
        ValueError: If any edge weight rounds to ``<= 0`` (METIS requires positive
            integer edge weights).
        RuntimeError: If the METIS library reports an internal error.
    """
    n_vertices = int(xadj.shape[0]) - 1
    if n_parts == 1:
        return np.zeros(n_vertices, dtype=np.int32)

    pymetis: Any = _require_pymetis()
    idx_dtype = pymetis.zero_copy_dtype()
    adjacency = pymetis.CSRAdjacency(
        np.asarray(xadj, dtype=idx_dtype),
        np.asarray(adjncy, dtype=idx_dtype),
    )
    eweights = np.asarray(edge_weights).astype(idx_dtype)
    zero_count = int(np.sum(eweights <= 0))
    if zero_count > 0:
        raise ValueError(
            f"METIS requires positive integer edge weights; after rounding, "
            f"{zero_count} edge(s) have weight <= 0. Scale up the graph edge weights "
            f"or use backend='spectral'."
        )
    vweights = np.maximum(1, np.rint(vertex_weights)).astype(idx_dtype)
    try:
        _, membership = pymetis.part_graph(n_parts, adjacency, vweights=vweights, eweights=eweights)
    except RuntimeError as exc:
        err = RuntimeError("METIS partitioning failed.")
        err.add_note(
            f"Graph: {n_vertices} vertices, {int(adjncy.shape[0])} edges, n_parts={n_parts}"
        )
        err.add_note("Try backend='spectral' as a fallback.")
        raise err from exc
    return cast("npt.NDArray[np.int32]", np.asarray(membership, dtype=np.int32))


def _require_pymetis() -> ModuleType:
    """Import and return the ``pymetis`` module, or raise a clear error if absent.

    Returns:
        ModuleType: The imported ``pymetis`` module.

    Raises:
        ImportError: If ``pymetis`` is not installed or fails to load (e.g. missing
            native library), with guidance on how to obtain it.
    """
    try:
        return importlib.import_module("pymetis")
    except ImportError as exc:
        raise ImportError(
            "backend='metis' requires 'pymetis', which is not installed or failed to load. "
            "Install it with \"pip install 'pantr[metis]'\" (or 'pip install pymetis'); or use "
            f"backend='spectral', which needs no extra dependency. (Original error: {exc})"
        ) from exc


def _coerce_active(cell_active: npt.ArrayLike | None, n: int) -> npt.NDArray[np.bool_] | None:
    """Coerce ``cell_active`` to a 1-D boolean array.

    The length and the presence of an active cell are checked by the compiled partitioner.

    Args:
        cell_active (npt.ArrayLike | None): Candidate activity mask, or ``None``.
        n (int): Expected length (number of graph vertices, for the message).

    Returns:
        npt.NDArray[np.bool_] | None: The coerced mask, or ``None`` if input was ``None``.

    Raises:
        ValueError: If the mask is not 1-D.
    """
    if cell_active is None:
        return None
    active = np.asarray(cell_active)
    if active.ndim != 1:
        raise ValueError(f"cell_active must have shape ({n},); got {active.shape}.")
    return np.ascontiguousarray(active.astype(bool))


__all__ = ["partition_graph"]
