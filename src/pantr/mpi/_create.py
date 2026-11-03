"""Convenience factory for building an MPI-distributed space directly from a space.

Provides :func:`create_distributed_space`, the one-call counterpart to the explicit
``partition -> DistributedSpace`` flow: it derives the grid, partitions it, and wraps
the result, so a caller need not assemble the grid and :class:`~pantr.grid.Partition`
by hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._ext import ext
from ..bspline import BsplineSpace, THBSplineSpace, coupling_graph, partition_graph
from ..grid import Partition, partition_grid, tensor_product_grid
from ._comm import as_communicator
from ._distributed_space import DistributedSpace
from ._thread_policy import _ensure_default_thread_policy

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..grid import Grid

_cpp: Any = ext.mpi
"""The compiled distributed algorithms (private)."""


def create_distributed_space(  # noqa: PLR0913 -- public factory mirrors the partitioner args
    global_space: BsplineSpace | THBSplineSpace,
    comm: Any,  # noqa: ANN401 -- an mpi4py.MPI.Comm; mpi4py is an optional, untyped dep
    *,
    method: str = "grid",
    backend: str | None = None,
    cell_weights: npt.ArrayLike | None = None,
    cell_active: npt.ArrayLike | None = None,
) -> DistributedSpace:
    """Build an MPI-distributed space directly from a global space.

    Convenience wrapper over the explicit flow (derive grid -> partition ->
    :class:`DistributedSpace`).  It derives the space's cell grid
    (:func:`~pantr.grid.tensor_product_grid` for a :class:`~pantr.bspline.BsplineSpace`,
    or :attr:`THBSplineSpace.grid <pantr.bspline.THBSplineSpace.grid>` for a hierarchical
    space), partitions it across ``comm.size`` ranks, and returns the per-rank handle.
    The partition is computed on rank 0 only and sent to the other ranks (one
    ``comm.allgather``, in which the other ranks send nothing), so every rank holds the
    same one even where a floating-point partitioner would round differently on another
    rank's machine. With one rank nothing is sent. The arguments are checked on every rank
    before the collective; if computing the partition raises on rank 0, rank 0 raises that
    exception and every other rank raises :class:`ValueError`.

    The explicit three-step flow remains available for full control; this factory just
    removes the boilerplate for the common case.

    Args:
        global_space (BsplineSpace | THBSplineSpace): The global space to distribute,
            identical on every rank.  A ``BsplineSpace`` must be non-periodic.
        comm (Any): An MPI communicator (e.g. ``mpi4py.MPI.COMM_WORLD``), or a duck-typed
            one: its ``size`` is read, and with more than one rank its ``rank`` and its
            ``allgather`` are used.
        method (str): Partitioning strategy.  ``"grid"`` (default) splits the cell grid
            geometrically via :func:`~pantr.grid.partition_grid`; ``"graph"`` partitions
            the cell-coupling graph (:func:`~pantr.bspline.coupling_graph` +
            :func:`~pantr.bspline.partition_graph`) to minimize cross-rank DOF coupling.
        backend (str | None): Partitioner backend.  ``None`` (default) selects each
            method's default -- ``"auto"`` for ``"grid"`` (``"block"`` or ``"rcb"``),
            ``"spectral"`` for ``"graph"`` (or ``"metis"``, needing the ``metis`` extra).
        cell_weights (npt.ArrayLike | None): Per-cell assembly-cost weights.  For
            ``"grid"`` they balance the geometric split; for ``"graph"`` they become the
            coupling graph's vertex weights.  Defaults to ``None`` (uniform).
        cell_active (npt.ArrayLike | None): Boolean per-cell activity mask; inactive
            cells get owner ``-1`` and drop out of the partition.  Defaults to ``None``
            (all active).

    Returns:
        DistributedSpace: The per-rank distributed-space handle.

    Raises:
        TypeError: If ``method="grid"`` and ``global_space`` is neither a
            ``BsplineSpace`` nor a ``THBSplineSpace``.
        ValueError: If ``method`` is not ``"grid"`` or ``"graph"``; if ``backend`` is
            invalid for the chosen method; or if the derived partition is incompatible
            with ``comm`` (e.g. ``comm.size`` mismatch), as raised downstream; on a rank
            other than 0, if rank 0 could not compute the partition.

    Example:
        Needs a live MPI communicator and several ranks, so it is not run as a doctest.

        >>> from mpi4py import MPI  # doctest: +SKIP
        >>> from pantr.bspline import create_uniform_space  # doctest: +SKIP
        >>> from pantr.mpi import create_distributed_space  # doctest: +SKIP
        >>> space = create_uniform_space([2, 2], [8, 8])  # doctest: +SKIP
        >>> ds = create_distributed_space(space, MPI.COMM_WORLD)  # doctest: +SKIP
    """
    _ensure_default_thread_policy()
    if method not in ("grid", "graph"):
        raise ValueError(f"method must be 'grid' or 'graph'; got {method!r}.")
    if method == "grid" and not isinstance(global_space, BsplineSpace | THBSplineSpace):
        raise TypeError(
            f"global_space must be a BsplineSpace or THBSplineSpace; "
            f"got {type(global_space).__name__!r}."
        )
    n_parts = int(comm.size)

    def compute() -> Any:  # noqa: ANN401 -- the compiled Partition
        if method == "grid":
            grid: Grid = (
                global_space.grid
                if isinstance(global_space, THBSplineSpace)
                else tensor_product_grid(global_space)
            )
            part = partition_grid(
                grid,
                n_parts,
                backend="auto" if backend is None else backend,
                cell_weights=cell_weights,
                cell_active=cell_active,
            )
        else:
            graph = coupling_graph(global_space, cell_weights=cell_weights)
            part = partition_graph(
                graph,
                n_parts,
                backend="spectral" if backend is None else backend,
                cell_active=cell_active,
            )
        return part._h

    partition = Partition._from_handle(_cpp.broadcast_partition(as_communicator(comm), compute))
    return DistributedSpace(global_space, partition, comm)
