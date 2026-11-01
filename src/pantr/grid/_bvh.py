"""Bounding-volume hierarchy over grid cells.

A simple but efficient BVH that indexes a fixed collection of axis-aligned
bounding boxes (one per grid cell). The tree is built once by iterative
median-of-longest-axis splits and queried by iterative descent, both in C++; this
module wraps the compiled tree (``_h``).

Layout
------

The BVH is held as five parallel arrays, matching the representation the traversal
consumes:

- ``node_lo`` / ``node_hi``: per-node AABB corners, shape ``(n_nodes, ndim)``.
- ``node_left`` / ``node_right``: child indices; ``-1`` on leaves.
- ``node_cell``: cell identifier on leaves; ``-1`` on internal nodes.

The root is always node ``0`` and covers every cell. For ``N`` cells the tree
has exactly ``2 * N - 1`` nodes (``N`` leaves, ``N - 1`` internal nodes).
Internal-node AABBs are tight: the union of the children's AABBs. Construction
stops at one cell per leaf, so leaves and cells are in one-to-one correspondence.
The construction order (preorder) is deterministic, which keeps query results
reproducible.

Queries
-------

:meth:`BVH.query_aabb` returns the ids of cells whose AABB is not separated from
the query box. A query is one descent that collects the matching cell ids in
preorder, returned as a fresh, compact ``int64`` array.

**The overlap test is a separating-axis test and nothing else**, and it is not
:meth:`pantr.geometry.AABB.overlaps`. It compares inclusively, so a query box
sharing a face with a cell is reported; but it has no emptiness branch, so a
*reversed* query interval is reported against any cell whose own interval contains
it, where the box reports nothing. One cell ``[0, 10]`` in one dimension, queried
with ``lo = 5, hi = 3``, returns cell ``0`` here. This docstring used to claim
agreement with the box, and the claim is withdrawn rather than the behavior
changed, because reconciling the two predicates has to move both at once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import numpy as np

from .._ext import ext
from ._grid_utils import _as_float64

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..geometry import AABB

_cpp = ext.grid


class BVH:
    """Bounding-volume hierarchy indexing a fixed collection of AABBs.

    Instances are immutable: the tree lives in the compiled library and the arrays
    this class exposes are read-only views of it. Queries allocate fresh ``int64``
    output arrays per call.

    Build by passing per-cell AABBs to :meth:`from_cell_bounds`; direct
    construction from the raw array representation is supported via the default
    constructor but is mostly intended for tests and round-trip serialization.
    """

    __slots__ = (
        "_h",
        "_node_cell",
        "_node_hi",
        "_node_left",
        "_node_lo",
        "_node_right",
        "n_cells",
        "n_nodes",
        "ndim",
    )

    #: The compiled tree (private).
    _h: Any
    _node_lo: npt.NDArray[np.float64]
    _node_hi: npt.NDArray[np.float64]
    _node_left: npt.NDArray[np.int64]
    _node_right: npt.NDArray[np.int64]
    _node_cell: npt.NDArray[np.int64]
    #: Spatial dimension of the indexed AABBs (``>= 1``).
    ndim: int
    #: Number of cells indexed (equal to the number of leaves).
    n_cells: int
    #: Total number of nodes (``2 * n_cells - 1`` for ``n_cells > 0``, else ``0``).
    n_nodes: int

    def __init__(  # noqa: PLR0913 -- BVH is a five-array flat struct
        self,
        node_lo: npt.NDArray[np.float64],
        node_hi: npt.NDArray[np.float64],
        node_left: npt.NDArray[np.int64],
        node_right: npt.NDArray[np.int64],
        node_cell: npt.NDArray[np.int64],
        *,
        n_cells: int,
    ) -> None:
        """Validate the raw BVH arrays and copy them into the compiled tree.

        Callers should prefer :meth:`from_cell_bounds`; this constructor is
        useful for tests that need to poke specific tree shapes. The arrays are
        copied: the caller's arrays are neither kept nor modified.

        Args:
            node_lo (npt.NDArray[np.float64]): Per-node AABB lo corners, shape
                ``(n_nodes, ndim)``.
            node_hi (npt.NDArray[np.float64]): Per-node AABB hi corners, shape
                ``(n_nodes, ndim)``.
            node_left (npt.NDArray[np.int64]): Left-child indices; ``-1`` on
                leaves. Shape ``(n_nodes,)``.
            node_right (npt.NDArray[np.int64]): Right-child indices; ``-1`` on
                leaves.
            node_cell (npt.NDArray[np.int64]): Leaf cell identifiers; ``-1`` on
                internal nodes.
            n_cells (int): Number of indexed cells (leaves).

        Raises:
            TypeError: If any array has the wrong dtype.
            ValueError: If shapes are inconsistent, ``ndim`` is ``< 1``,
                ``n_nodes != 2 * n_cells - 1`` (``0`` when ``n_cells == 0``), if
                ``node_cell`` and the child pointers disagree about which nodes are
                leaves, if an internal node's children or a leaf's cell id are out of
                range, if the arrays do not encode a tree, or if the tree's
                root-to-leaf depth exceeds the traversal's stack depth (128).
        """
        if node_lo.dtype != np.float64 or node_hi.dtype != np.float64:
            raise TypeError(
                f"node_lo / node_hi must be float64; got {node_lo.dtype!r} / {node_hi.dtype!r}."
            )
        indices = (
            (node_left, "node_left"),
            (node_right, "node_right"),
            (node_cell, "node_cell"),
        )
        for arr, name in indices:
            if arr.dtype != np.int64:
                raise TypeError(f"{name} must be int64; got {arr.dtype!r}.")
            if arr.ndim != 1 and node_lo.ndim == 2:  # noqa: PLR2004
                # The compiled constructor takes the index arrays flat; the message spells
                # the NumPy shape.
                raise ValueError(f"{name} must have shape ({node_lo.shape[0]},); got {arr.shape}.")
        self._init_from_handle(
            _cpp.BVH(
                np.require(node_lo, requirements="C"),
                np.require(node_hi, requirements="C"),
                np.require(node_left, requirements="C").ravel(),
                np.require(node_right, requirements="C").ravel(),
                np.require(node_cell, requirements="C").ravel(),
                int(n_cells),
            )
        )

    def _init_from_handle(self, h: Any) -> None:  # noqa: ANN401
        """Adopt a compiled tree and read its scalars and array views once.

        Args:
            h (Any): The compiled tree.
        """
        self._h = h
        self.ndim = int(h.ndim)
        self.n_cells = int(h.n_cells)
        self.n_nodes = int(h.n_nodes)
        self._node_lo = h.node_lo
        self._node_hi = h.node_hi
        self._node_left = h.node_left
        self._node_right = h.node_right
        self._node_cell = h.node_cell

    @classmethod
    def _from_handle(cls, h: Any) -> Self:  # noqa: ANN401
        """Wrap a compiled tree without copying it.

        Args:
            h (Any): The compiled tree.

        Returns:
            Self: The wrapper.
        """
        bvh = object.__new__(cls)
        bvh._init_from_handle(h)
        return bvh

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle and copy through the constructor, from the raw arrays.

        Returns:
            tuple[Any, ...]: The rebuild function and its arguments.
        """
        return (
            _rebuild_bvh,
            (
                type(self),
                self._node_lo,
                self._node_hi,
                self._node_left,
                self._node_right,
                self._node_cell,
                self.n_cells,
            ),
        )

    @classmethod
    def from_cell_bounds(
        cls,
        cell_lo: npt.ArrayLike,
        cell_hi: npt.ArrayLike,
    ) -> Self:
        """Build a BVH over ``n_cells`` axis-aligned cell AABBs.

        Uses a top-down recursive median-of-longest-axis split. Cells are sorted
        by centroid on the longest axis; the median splits the list into two
        halves of equal size (``+/- 1``). Each leaf indexes exactly one cell.

        Args:
            cell_lo (npt.ArrayLike): Per-cell lo corners; shape
                ``(n_cells, ndim)`` with ``ndim >= 1``. Validated, not mutated.
            cell_hi (npt.ArrayLike): Per-cell hi corners; same shape and
                conventions as ``cell_lo``. Each entry must satisfy
                ``cell_hi >= cell_lo``.

        Returns:
            Self: The constructed hierarchy.

        Raises:
            TypeError: If inputs cannot be cast to ``float64``.
            ValueError: If shapes are inconsistent, ``ndim`` is ``< 1``, any
                value is not finite, or any cell has ``hi < lo``.
        """
        lo = _as_float64(cell_lo, name="cell_lo")
        hi = _as_float64(cell_hi, name="cell_hi")
        return cls._from_handle(_cpp.BVH.from_cell_bounds(lo, hi))

    def query_aabb(self, aabb: AABB) -> npt.NDArray[np.int64]:
        """Return the ids of every leaf cell whose AABB is not separated from ``aabb``.

        This is a separating-axis test, not :meth:`pantr.geometry.AABB.overlaps`; see
        the module docstring for how they differ on an empty query box.

        Args:
            aabb (AABB): Query box; must match :attr:`ndim`.

        Returns:
            npt.NDArray[np.int64]: Overlapping cell ids. Order matches the
            internal preorder traversal; callers that need a particular order
            should sort the result.

        Raises:
            ValueError: If ``aabb.ndim != self.ndim``.
        """
        out: npt.NDArray[np.int64] = self._h.query_aabb(aabb._h)
        return out

    @property
    def node_lo(self) -> npt.NDArray[np.float64]:
        """Get the read-only view of per-node AABB lo corners.

        Returns:
            npt.NDArray[np.float64]: Shape ``(n_nodes, ndim)``.
        """
        return self._node_lo

    @property
    def node_hi(self) -> npt.NDArray[np.float64]:
        """Get the read-only view of per-node AABB hi corners.

        Returns:
            npt.NDArray[np.float64]: Shape ``(n_nodes, ndim)``.
        """
        return self._node_hi

    @property
    def node_left(self) -> npt.NDArray[np.int64]:
        """Get the read-only view of per-node left-child indices.

        Returns:
            npt.NDArray[np.int64]: Shape ``(n_nodes,)``; ``-1`` on leaves.
        """
        return self._node_left

    @property
    def node_right(self) -> npt.NDArray[np.int64]:
        """Get the read-only view of per-node right-child indices.

        Returns:
            npt.NDArray[np.int64]: Shape ``(n_nodes,)``; ``-1`` on leaves.
        """
        return self._node_right

    @property
    def node_cell(self) -> npt.NDArray[np.int64]:
        """Get the read-only view of per-leaf cell identifiers.

        Returns:
            npt.NDArray[np.int64]: Shape ``(n_nodes,)``; ``-1`` on internal
            nodes, ``0 <= id < n_cells`` on leaves.
        """
        return self._node_cell


def _rebuild_bvh(  # noqa: PLR0913 -- the five arrays and the cell count
    cls: type[BVH],
    node_lo: npt.NDArray[np.float64],
    node_hi: npt.NDArray[np.float64],
    node_left: npt.NDArray[np.int64],
    node_right: npt.NDArray[np.int64],
    node_cell: npt.NDArray[np.int64],
    n_cells: int,
) -> BVH:
    """Unpickle a :class:`BVH` from its raw arrays.

    Args:
        cls (type[BVH]): The class to build.
        node_lo (npt.NDArray[np.float64]): Per-node lo corners.
        node_hi (npt.NDArray[np.float64]): Per-node hi corners.
        node_left (npt.NDArray[np.int64]): Left-child indices.
        node_right (npt.NDArray[np.int64]): Right-child indices.
        node_cell (npt.NDArray[np.int64]): Leaf cell identifiers.
        n_cells (int): Number of indexed cells.

    Returns:
        BVH: The rebuilt hierarchy.
    """
    return cls(node_lo, node_hi, node_left, node_right, node_cell, n_cells=n_cells)


__all__ = ["BVH"]
