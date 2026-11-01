"""Cell-ownership partition for distributing a structured grid.

A :class:`Partition` records, for every cell of a grid (or the knot-span grid of a
B-spline space), which rank owns it -- the serial, communication-free descriptor
consumed by the distributed-space machinery. It is produced either by consuming an
external partition (for example a dolfinx mesh) or by a native graph partitioner,
and is intentionally space-agnostic: it stores only an integer owner per cell.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._ext import ext

if TYPE_CHECKING:
    import numpy.typing as npt

_cpp = ext.grid


class Partition:
    """A per-cell owner assignment over a grid's cells.

    Records, for every cell, the rank that owns it -- or ``-1`` for an inactive cell
    excluded from the partition (e.g. an exterior / trimmed cell). The owner array is
    coerced to a read-only ``int32`` array on construction and the object is otherwise
    immutable. Owners and counts are exposed through the :attr:`cell_owner`,
    :attr:`n_parts`, :attr:`n_cells`, and :attr:`active_mask` properties.
    """

    __slots__ = ("_cell_owner", "_h", "_n_parts")

    #: The compiled partition (private).
    _h: Any
    _cell_owner: npt.NDArray[np.int32]

    def __init__(self, cell_owner: npt.ArrayLike, n_parts: int) -> None:
        """Build a partition from a per-cell owner array.

        Args:
            cell_owner (npt.ArrayLike): Per-cell owner ranks (``-1`` for inactive
                cells); coerced to a read-only 1D ``int32`` array.
            n_parts (int): Number of parts (ranks); must be ``>= 1``.

        Raises:
            ValueError: If ``n_parts < 1``, ``cell_owner`` is not 1D integer, or any
                owner is outside ``[-1, n_parts)``.
        """
        # The order of the checks is part of the contract: the part count is refused first.
        if n_parts < 1:
            raise ValueError(f"n_parts must be >= 1; got {n_parts}.")
        owner = np.asarray(cell_owner)
        if owner.ndim != 1 or not np.issubdtype(owner.dtype, np.integer):
            raise ValueError("cell_owner must be a 1D integer array.")
        self._init_from_handle(
            _cpp.Partition(np.ascontiguousarray(owner, dtype=np.int32), int(n_parts))
        )

    def _init_from_handle(self, h: Any) -> None:  # noqa: ANN401
        """Adopt a compiled partition and read its owner view and part count once.

        Args:
            h (Any): The compiled partition.
        """
        self._h = h
        self._cell_owner = h.cell_owner
        self._n_parts = int(h.n_parts)

    @classmethod
    def _from_handle(cls, h: Any) -> Partition:  # noqa: ANN401
        """Wrap a compiled partition without copying it.

        Args:
            h (Any): The compiled partition.

        Returns:
            Partition: The wrapper.
        """
        part = object.__new__(cls)
        part._init_from_handle(h)
        return part

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle and copy through the constructor.

        Returns:
            tuple[Any, ...]: The class and its arguments.
        """
        return (type(self), (self._cell_owner, self._n_parts))

    @property
    def cell_owner(self) -> npt.NDArray[np.int32]:
        """Get the read-only per-cell owner array.

        Returns:
            npt.NDArray[np.int32]: ``(n_cells,)`` owners; ``-1`` for inactive cells.
        """
        return self._cell_owner

    @property
    def n_parts(self) -> int:
        """Get the number of parts (ranks).

        Returns:
            int: The part count (``>= 1``).
        """
        return self._n_parts

    @property
    def n_cells(self) -> int:
        """Get the total number of cells (active and inactive).

        Returns:
            int: Length of :attr:`cell_owner`.
        """
        return int(self._cell_owner.shape[0])

    @property
    def active_mask(self) -> npt.NDArray[np.bool_]:
        r"""Get a boolean mask of the active cells (owned by some rank).

        Returns:
            npt.NDArray[np.bool\_]: Fresh ``(n_cells,)`` mask; ``True`` where the cell
            owner is not ``-1``.
        """
        mask: npt.NDArray[np.bool_] = self._h.active_mask()
        return mask

    def owned_cells(self, rank: int) -> npt.NDArray[np.int64]:
        """Return the flat ids of the cells owned by ``rank``, ascending.

        Args:
            rank (int): Owner rank in ``[0, n_parts)``.

        Returns:
            npt.NDArray[np.int64]: Sorted cell ids with ``cell_owner == rank``.

        Raises:
            ValueError: If ``rank`` is outside ``[0, n_parts)``.
        """
        owned: npt.NDArray[np.int64] = self._h.owned_cells(int(rank))
        return owned


__all__ = ["Partition"]
