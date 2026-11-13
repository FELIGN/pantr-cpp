"""Native, dependency-free partitioning of a structured grid into rank subdomains.

Produces a :class:`Partition` (per-cell owner assignment) without any external
dependency or MPI -- the zero-dependency default for distributing a grid. Two
backends are provided, with an ``"auto"`` dispatch:

- ``"block"`` -- a Cartesian split of a :class:`TensorProductGrid` into contiguous,
  aspect-ratio-aware box subdomains. Ideal for tensor-product grids whose part count
  factors reasonably across the axes; raises rather than producing empty ranks when
  it does not. Ignores cell weights and activity.
- ``"rcb"`` -- recursive coordinate bisection on cell centroids. Geometric and
  grid-agnostic (works on any :class:`Grid`, hierarchical or immersed), weight-aware
  (balances total cell cost, not cell count) and activity-aware (inactive cells get
  owner ``-1`` and are excluded). Handles arbitrary part counts, including prime ones.
- ``"auto"`` -- ``"block"`` when the grid is tensor-product, no weights/activity are
  given, and the part count factors onto the axes; otherwise ``"rcb"``.

External graph partitioners (ParMETIS / PT-Scotch) arrive in later work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np

from .._ext import ext
from ._partition import Partition

if TYPE_CHECKING:
    import numpy.typing as npt

    from ._grid import Grid

# The module alias is untyped: one call serves both grid types, which the stubs overload.
_cpp: Any = ext.grid

_BACKENDS: Final[dict[str, Any]] = {
    "auto": _cpp.PartitionBackend.automatic,
    "block": _cpp.PartitionBackend.block,
    "rcb": _cpp.PartitionBackend.rcb,
}
"""Partitioning backends recognized by :func:`partition_grid`, by name."""


def partition_grid(
    grid: Grid,
    n_parts: int,
    *,
    backend: str = "auto",
    cell_weights: npt.ArrayLike | None = None,
    cell_active: npt.ArrayLike | None = None,
) -> Partition:
    """Partition a grid's cells into ``n_parts`` rank subdomains.

    Returns a :class:`Partition` (a plain per-cell owner array) for the
    serial, communication-free distribution of a grid -- no MPI is involved.
    See the module docstring for the ``"block"``, ``"rcb"``, and ``"auto"``
    backends.

    The ``cell_weights`` and ``cell_active`` hooks let a consumer (e.g. an
    immersed/unfitted code) drive the partition without pantr storing any
    classification: per-cell assembly cost via ``cell_weights`` (the ``"rcb"``
    backend balances total weight), and an active subset via ``cell_active``
    (inactive cells get owner ``-1``). The ``"block"`` backend supports neither.

    Args:
        grid (Grid): The grid to partition. The ``"block"`` backend requires a
            :class:`TensorProductGrid`; ``"rcb"`` accepts any grid.
        n_parts (int): Number of parts (ranks); must be ``>= 1``.
        backend (str): ``"auto"`` (default), ``"block"``, or ``"rcb"``.
        cell_weights (npt.ArrayLike | None): Optional per-cell cost, shape
            ``(num_cells,)``, finite and non-negative. ``None`` means uniform.
            Used by ``"rcb"``; rejected by ``"block"``.
        cell_active (npt.ArrayLike | None): Optional boolean mask, shape
            ``(num_cells,)``; inactive cells get owner ``-1`` and are excluded
            from partitioning. ``None`` means all active. Used by ``"rcb"``;
            rejected by ``"block"``.

    Returns:
        Partition: A per-cell owner assignment with ``n_parts`` parts. Cells
        excluded by ``cell_active`` have owner ``-1``; every active cell is
        assigned a rank in ``range(n_parts)`` and no rank is left empty.

    Raises:
        ValueError: If ``n_parts < 1``; if ``backend`` is unknown; if ``"block"``
            is used on a non-:class:`TensorProductGrid` or with weights/activity;
            if ``n_parts`` cannot be factored onto the axes (``"block"``); if
            ``cell_weights`` / ``cell_active`` have the wrong shape or invalid
            values; or if ``n_parts`` exceeds the number of active cells
            (``"rcb"``).

    Example:
        >>> from pantr.grid import partition_grid, uniform_grid
        >>> grid = uniform_grid([[0.0, 1.0], [0.0, 1.0]], [4, 4])
        >>> part = partition_grid(grid, 4)
        >>> part.n_parts
        4
        >>> int(part.cell_owner.min()), int(part.cell_owner.max())
        (0, 3)
    """
    # The order of the checks is part of the contract: the part count, then the backend name.
    if n_parts < 1:
        raise ValueError(f"n_parts must be >= 1; got {n_parts}.")
    if backend not in _BACKENDS:
        valid = ", ".join(repr(b) for b in _BACKENDS)
        raise ValueError(f"unknown backend {backend!r}; valid backends: {valid}.")

    n_parts = int(n_parts)
    n_cells = grid.num_cells
    weights = _validate_weights(cell_weights, n_cells)
    active = _validate_active(cell_active, n_cells)
    return Partition._from_handle(
        _cpp.partition_grid(grid._h, n_parts, _BACKENDS[backend], weights, active)
    )


def _validate_weights(
    cell_weights: npt.ArrayLike | None, n_cells: int
) -> npt.NDArray[np.float64] | None:
    """Coerce ``cell_weights`` to a 1-D ``float64`` array of the expected rank.

    The values (length, finiteness, sign) are checked by the compiled partitioner.

    Args:
        cell_weights (npt.ArrayLike | None): Candidate per-cell weights, or ``None``.
        n_cells (int): Expected length (for the message).

    Returns:
        npt.NDArray[np.float64] | None: The coerced weights, or ``None`` if the
        input was ``None``.

    Raises:
        ValueError: If the weights are not 1-D.
    """
    if cell_weights is None:
        return None
    weights = np.asarray(cell_weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError(f"cell_weights must have shape ({n_cells},); got {weights.shape}.")
    return np.ascontiguousarray(weights)


def _validate_active(
    cell_active: npt.ArrayLike | None, n_cells: int
) -> npt.NDArray[np.bool_] | None:
    """Coerce ``cell_active`` to a 1-D boolean array of the expected rank.

    The length and the presence of an active cell are checked by the compiled
    partitioner.

    Args:
        cell_active (npt.ArrayLike | None): Candidate activity mask, or ``None``.
        n_cells (int): Expected length (for the message).

    Returns:
        npt.NDArray[np.bool_] | None: The coerced mask, or ``None`` if the input
        was ``None``.

    Raises:
        ValueError: If the mask is not 1-D.
    """
    if cell_active is None:
        return None
    active = np.asarray(cell_active)
    if active.ndim != 1:
        raise ValueError(f"cell_active must have shape ({n_cells},); got {active.shape}.")
    return np.ascontiguousarray(active.astype(bool))


__all__ = ["partition_grid"]
