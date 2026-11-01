"""Hierarchical structured grid with uniform per-direction subdivision.

A :class:`HierarchicalGrid` wraps a root :class:`TensorProductGrid` and adds a
hierarchy of refinement levels.  Each ``refine`` call promotes a rectangular
region of currently-active cells to the next level, splitting every cell in that
region into ``m_0 * ... * m_{d-1}`` equal children (the per-direction *factor*).

Design highlights
-----------------

*No per-cell storage.*  Only **rectangular blocks** are stored — at most one
small list of ``(lo, hi)`` index pairs per level.  Memory is therefore
``O(total_blocks * ndim)``, independent of the total cell count.

*Union semantics.*  Calling :meth:`refine` with a region that overlaps an
already-refined area is silently correct: only the currently-active portion of
the region is refined.  Since the children of newly-active cells are always
disjoint from existing level blocks, no deduplication is needed.  Refining only
part of a box is what makes :meth:`refine` non-injective on the grid state, so
:meth:`coarsen` inverts it conditionally rather than always; both docstrings
state the hypothesis.

*No balance constraint.*  :meth:`refine` imposes no 2:1 grading: cells of any
two levels may share a facet.  Facet adjacency (:meth:`neighbor_across_facet`,
:meth:`hanging_neighbors`) walks as many levels up or down as the interface
requires.

Main exports:

- :class:`HierarchicalGrid`: hierarchical grid built on a
  :class:`TensorProductGrid`.
- :func:`hierarchical_grid`: standalone factory.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np

from .._ext import ext
from ._grid import Grid, GridRestriction
from ._grid_utils import _as_float64
from ._tensor_product_grid import TensorProductGrid

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

_cpp = ext.grid

# A Block is a pair (lo, hi) of per-direction inclusive-start / exclusive-end
# integer index tuples, all at the coordinate system of a specific level.
_Block = tuple[tuple[int, ...], tuple[int, ...]]


# ---------------------------------------------------------------------------
# HierarchicalGrid
# ---------------------------------------------------------------------------


class HierarchicalGrid(Grid):
    """Hierarchical grid with a fixed per-direction uniform subdivision factor.

    Built on a root :class:`TensorProductGrid`.  Active cells are stored as
    non-overlapping rectangular *blocks* at each level; no per-cell data is
    kept.  A new level is created by :meth:`refine`; the grid starts with all
    root cells active at level 0.

    Flat cell ids are assigned level-by-level (level 0 first), block-by-block
    within a level (sorted by ``lo`` tuple), and C-order within each block.

    The blocks live in an immutable compiled value (``_h``); :meth:`refine`,
    :meth:`coarsen` and their ``*_cells`` forms replace that value inside this
    object, and the object's cached counts and tag registries follow.

    Attributes:
        _root (TensorProductGrid): The level-0 root grid.
        _factor (tuple[int, ...]): Per-direction subdivision factor (``>= 1``).
        _ndim (int): Spatial dimension.
        _num_cells (int): Total active cell count of the current compiled value.
        _max_level (int): Index of the deepest non-empty level of the current value.
        _version (int): Monotonic mutation counter, incremented on every
            structural change (see :attr:`version`).
    """

    __slots__ = ("_factor", "_max_level", "_ndim", "_num_cells", "_root", "_version")

    _root: TensorProductGrid
    _factor: tuple[int, ...]

    def __init__(
        self,
        root: TensorProductGrid,
        factor: int | Sequence[int],
    ) -> None:
        """Create a hierarchical grid from a root and a subdivision factor.

        Args:
            root (TensorProductGrid): The level-0 grid.
            factor (int | Sequence[int]): Per-direction subdivision factor.
                A scalar is broadcast to every axis.  Each entry must be
                ``>= 1``; a factor of ``1`` on an axis prevents subdivision in
                that direction.

        Raises:
            TypeError: If ``root`` is not a :class:`TensorProductGrid`.
            ValueError: If ``factor`` has the wrong length or any entry is
                ``< 1``.
        """
        super().__init__()
        if not isinstance(root, TensorProductGrid):
            raise TypeError(f"root must be a TensorProductGrid; got {type(root).__name__!r}.")
        if isinstance(factor, int):
            fac: tuple[int, ...] = (int(factor),) * root.ndim
        else:
            fac = tuple(int(f) for f in factor)
        self._init_from_handle(_cpp.HierarchicalGrid(root._h, list(fac)), root, fac)

    def _init_from_handle(self, h: Any, root: TensorProductGrid, factor: tuple[int, ...]) -> None:  # noqa: ANN401
        """Adopt a compiled grid and read its metadata once.

        Args:
            h (Any): The compiled grid.
            root (TensorProductGrid): The wrapper of the level-0 grid of ``h``.
            factor (tuple[int, ...]): The subdivision factor of ``h``.
        """
        self._h = h
        self._root = root
        self._factor = factor
        self._ndim = root.ndim
        self._read_counts()

    def _read_counts(self) -> None:
        """Read the counts that change with a structural change from the compiled grid."""
        self._num_cells = int(self._h.num_cells)
        self._max_level = int(self._h.max_level)
        self._version = int(self._h.version)

    def _swap(self, h: Any) -> None:  # noqa: ANN401
        """Adopt the compiled grid a structural method returned, if it differs.

        The compiled grid returns the same value, with the same version, when the
        call changed nothing; the counts, the cached BVH wrapper and the tag
        registries then stay. After a change the flat ids are reassigned, so the
        cached BVH wrapper and both tag registries are dropped.

        Args:
            h (Any): The compiled grid the structural method returned.
        """
        if int(h.version) == self._version:
            return
        self._h = h
        self._read_counts()
        self._bvh_w = None
        self._cell_tags = None
        self._facet_tags = None

    @classmethod
    def _from_handle(cls, h: Any) -> Self:  # noqa: ANN401
        """Wrap a compiled grid without copying it.

        Args:
            h (Any): The compiled grid.

        Returns:
            Self: The wrapper.
        """
        grid = object.__new__(cls)
        Grid.__init__(grid)
        grid._init_from_handle(
            h, TensorProductGrid._from_handle(h.root), tuple(int(f) for f in h.factor)
        )
        return grid

    def _clone(self) -> Self:
        """Return a new wrapper of the same compiled grid, sharing the root wrapper.

        Returns:
            Self: A grid sharing this one's immutable compiled value.
        """
        grid = object.__new__(type(self))
        Grid.__init__(grid)
        grid._init_from_handle(self._h, self._root, self._factor)
        return grid

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """Deep-copy the grid, including its root wrapper.

        The compiled value is immutable and is shared, as in :meth:`__copy__`; the root is
        copied through ``memo`` because its tag registries are mutable, so a copy never
        shows a tag set on the original's root.

        Args:
            memo (dict[int, Any]): The copy memo.

        Returns:
            Self: The copy.
        """
        new = cast("Self", self.__copy__())
        new._root = copy.deepcopy(self._root, memo)
        return new

    def _state(self) -> tuple[Any, ...]:
        """Return the constructor state: the root, the factor, the blocks and the version.

        Returns:
            tuple[Any, ...]: What :meth:`_from_state` rebuilds the grid from.
        """
        return (self._root, self._factor, self._blocks, self._version)

    @classmethod
    def _from_state(cls, state: tuple[Any, ...]) -> Self:
        """Rebuild a grid from :meth:`_state`, at the version it was taken.

        Args:
            state (tuple[Any, ...]): The result of :meth:`_state`.

        Returns:
            Self: The rebuilt grid.
        """
        grid = object.__new__(cls)
        Grid.__init__(grid)
        grid._init_from_blocks(*state)
        return grid

    def _init_from_blocks(
        self,
        root: TensorProductGrid,
        factor: Sequence[int],
        blocks: Sequence[Sequence[_Block]],
        version: int,
    ) -> None:
        """Build the compiled grid from its blocks, at the given version, and adopt it.

        Args:
            root (TensorProductGrid): The level-0 grid.
            factor (Sequence[int]): The subdivision factor.
            blocks (Sequence[Sequence[_Block]]): The active-leaf blocks of every level.
            version (int): The mutation counter to restore.
        """
        h = _cpp.HierarchicalGrid.from_blocks(
            root._h,
            [int(f) for f in factor],
            [[(list(lo), list(hi)) for lo, hi in level] for level in blocks],
            int(version),
        )
        self._init_from_handle(h, root, tuple(int(f) for f in factor))

    def _restore_previous_release(self, slots: dict[str, Any]) -> None:
        """Rebuild the compiled grid from the blocks a previous release pickled.

        Args:
            slots (dict[str, Any]): The previous release's slots; ``_root``, ``_factor``,
                ``_blocks`` and ``_version`` determine the grid (the packed arrays it also
                held are derived from the blocks).
        """
        self._init_from_blocks(
            slots["_root"], slots["_factor"], slots["_blocks"], slots["_version"]
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Get the spatial dimension.

        Returns:
            int: Number of axes (``>= 1``).
        """
        return self._ndim

    @property
    def num_cells(self) -> int:
        """Get the total number of active cells across all levels.

        Returns:
            int: Total active cell count.
        """
        return self._num_cells

    @property
    def root(self) -> TensorProductGrid:
        """Get the level-0 root grid.

        Returns:
            TensorProductGrid: The root grid used to construct this hierarchy.
        """
        return self._root

    @property
    def factor(self) -> tuple[int, ...]:
        """Get the per-direction subdivision factor.

        Returns:
            tuple[int, ...]: Length-``ndim`` tuple; ``factor[k] >= 1``.
        """
        return self._factor

    @property
    def max_level(self) -> int:
        """Get the index of the deepest non-empty level.

        Returns:
            int: ``0`` before any refinement; increases with each :meth:`refine`
            call that adds a new level.
        """
        return self._max_level

    @property
    def version(self) -> int:
        """Get the monotonic mutation counter of this grid.

        Incremented on every structural change (:meth:`refine`, :meth:`coarsen`).
        Snapshot consumers (e.g. :class:`~pantr.bspline.THBSplineSpace`) compare
        it to detect *any* post-construction mutation -- ``max_level`` and
        ``num_cells`` alone cannot distinguish compensating refine/coarsen pairs.

        Returns:
            int: The current mutation count (``>= 1``).
        """
        return self._version

    @property
    def _blocks(self) -> list[list[_Block]]:
        """Get the active-leaf blocks of every level.

        Returns:
            list[list[_Block]]: ``_blocks[l]`` is the sorted list of non-overlapping
            ``(lo, hi)`` blocks of active cells at level ``l``, as tuples.
        """
        return [[(tuple(lo), tuple(hi)) for lo, hi in level] for level in self._h.blocks()]

    # ------------------------------------------------------------------
    # Grid contract
    # ------------------------------------------------------------------

    def cell_bounds(
        self,
        cid: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the axis-aligned ``(lo, hi)`` corners of cell ``cid``.

        Args:
            cid (int): Flat cell identifier.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Fresh,
            writeable length-``ndim`` ``float64`` arrays.

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        lo, hi = self._h.cell_bounds(int(cid))
        return lo, hi

    def locate(self, pt: npt.ArrayLike) -> int | None:
        """Return the active leaf cell containing ``pt``, or ``None``.

        Performs a top-down traversal from the root level.

        Args:
            pt (npt.ArrayLike): Length-``ndim`` point in parametric
                coordinates.

        Returns:
            int | None: Active cell flat id, or ``None`` when ``pt`` is
            outside the grid domain.

        Raises:
            ValueError: If ``pt`` does not have length ``ndim``.
        """
        found: int | None = self._h.locate(_as_float64(pt, name="pt").ravel())
        return found

    def locate_many(self, points: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Locate a batch of points by a top-down descent, in parallel over the points.

        Args:
            points (npt.ArrayLike): ``(npts, ndim)`` array-like of points, or a
                single length-``ndim`` point.

        Returns:
            npt.NDArray[np.int64]: Shape ``(npts,)`` cell ids; ``-1`` for points
            outside the grid (including points with NaN or infinite coordinates).

        Raises:
            ValueError: If the trailing axis of ``points`` is not ``ndim``.
        """
        out: npt.NDArray[np.int64] = self._h.locate_many(self._normalize_points(points))
        return out

    def neighbor_across_facet(self, cid: int, lfid: int) -> int | None:
        """Return the cell across local facet ``lfid`` of ``cid``, or ``None``.

        Handles hanging-node interfaces across **any** level difference (no 2:1
        balance is enforced by :meth:`refine`): when the neighbour is coarser, the
        active leaf covering the position -- however many levels up -- is returned.
        When the neighbour side is finer, the first active leaf descendant touching
        the face (lowest C-order along the face) is returned.  Use
        :meth:`hanging_neighbors` to retrieve *all* fine neighbours.

        Args:
            cid (int): Cell identifier.
            lfid (int): Local facet identifier in ``[0, 2 * ndim)``.

        Returns:
            int | None: Neighbouring cell id, or ``None`` on a boundary facet.

        Raises:
            IndexError: If ``cid`` or ``lfid`` is out of range.
        """
        neighbor: int | None = self._h.neighbor_across_facet(int(cid), int(lfid))
        return neighbor

    def hanging_neighbors(self, cid: int, lfid: int) -> tuple[int, ...]:
        """Return all active neighbours across facet ``lfid`` of ``cid``.

        Equivalent to :meth:`neighbor_across_facet` for conforming and coarser
        interfaces (a single neighbour, however many levels up).  For a hanging
        (fine-side) interface, returns *all* active leaves touching the face,
        descending as many levels as the interface requires, ordered depth-first
        along the face (C-order over the non-``axis`` directions at each level).

        Args:
            cid (int): Cell identifier.
            lfid (int): Local facet identifier in ``[0, 2 * ndim)``.

        Returns:
            tuple[int, ...]: Neighbouring cell ids; empty on a boundary facet.

        Raises:
            IndexError: If ``cid`` or ``lfid`` is out of range.
        """
        return tuple(self._h.hanging_neighbors(int(cid), int(lfid)))

    def boundary_facets(self) -> npt.NDArray[np.int64]:
        """Return every facet on the grid's outer boundary as ``(cid, lfid)`` rows.

        Same semantics as :meth:`pantr.grid.Grid.boundary_facets` -- the outer
        boundary only, so level interfaces are excluded -- computed from the packed
        active blocks instead of one neighbour query per facet.

        A level-``l`` cell's facet ``(axis, side)`` is on the outer boundary iff its
        level-``l`` index along ``axis`` is ``0`` (side 0) or
        ``level_cells_per_axis(l)[axis] - 1`` (side 1), which is exactly the
        out-of-range test that makes :meth:`neighbor_across_facet` return ``None``.
        Since a block is a rectangle of contiguous indices, that condition is decided
        **once per block face** rather than per cell, and the whole face is emitted
        vectorized. Cost is ``O(total boundary facets + num_blocks * 2 * ndim)``.

        Returns:
            npt.NDArray[np.int64]: Read-only array of shape ``(n, 2)`` whose rows are
            ``(cid, lfid)`` with ``lfid = 2 * axis + side``, sorted lexicographically
            by ``(cid, lfid)``.
        """
        out: npt.NDArray[np.int64] = self._h.boundary_facets()
        out.flags.writeable = False
        return out

    # ------------------------------------------------------------------
    # Restriction / windowing
    # ------------------------------------------------------------------

    def restrict(self, cell_ids: npt.ArrayLike) -> GridRestriction:
        """Return the root-cell-aligned bounding-box sub-grid spanning ``cell_ids``.

        The window is the multi-index bounding box, **in root-cell coordinates**,
        of the root cells containing the requested leaves (a leaf at
        ``(level, midx)`` lives in root cell ``midx[k] // factor[k] ** level``).
        The sub-grid's root is the matching slice of this grid's root breakpoints
        (never re-clamped) and it keeps the same ``factor``; its active leaves are
        the per-level intersections of this grid's blocks with the window.

        Because the window is root-cell-aligned, restricting a single deep leaf
        returns the whole leaf-tiling of its root cell, with only the requested
        leaf flagged in :attr:`GridRestriction.in_subset`.

        Args:
            cell_ids (npt.ArrayLike): Flat cell identifiers to span; duplicates
                are ignored. Each must satisfy ``0 <= cid < num_cells``.

        Returns:
            GridRestriction: The windowed :class:`HierarchicalGrid`, its
            ``local_to_global_cell`` map of shape ``(sub.num_cells,)``, and the
            boolean ``in_subset`` mask flagging requested versus bounding-box-fill cells.

        Raises:
            ValueError: If ``cell_ids`` is empty.
            IndexError: If any cell id is out of range ``[0, num_cells)``.
            TypeError: If ``cell_ids`` is not integer-valued.
            RuntimeError: If an internal invariant is violated (should be unreachable).
        """
        return super().restrict(cell_ids)

    # ------------------------------------------------------------------
    # Hierarchy accessors
    # ------------------------------------------------------------------

    def cell_level(self, cid: int) -> int:
        """Return the refinement level of cell ``cid``.

        Args:
            cid (int): Cell identifier.

        Returns:
            int: Refinement level (``0`` for unrefined root cells).

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        return int(self._h.cell_level(int(cid)))

    def cell_multi_index(self, cid: int) -> tuple[int, ...]:
        """Return the per-axis index of cell ``cid`` in its level's coordinates.

        Args:
            cid (int): Cell identifier.

        Returns:
            tuple[int, ...]: Length-``ndim`` per-axis index tuple at the cell's
            refinement level.

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        return tuple(self._h.cell_multi_index(int(cid)))

    def cell_id(self, level: int, midx: Sequence[int]) -> int | None:
        """Return the flat id of the active leaf ``(level, midx)``, or ``None``.

        The inverse of :meth:`cell_level` and :meth:`cell_multi_index` taken together.
        Because :meth:`refine` and :meth:`coarsen` reassign every flat id, a caller that
        must track cells across a mutation keeps them as ``(level, midx)`` pairs -- which
        are stable -- and resolves them back to ids here, once per mutation.

        Args:
            level (int): Hierarchy level.
            midx (Sequence[int]): Per-axis index in level-``level`` coordinates.

        Returns:
            int | None: The flat cell id when ``(level, midx)`` is currently an active
            (leaf) cell, else ``None`` -- out of range, not yet created, or refined away.
        """
        found: int | None = self._h.cell_id(int(level), [int(i) for i in midx])
        return found

    def is_active_leaf(self, level: int, midx: Sequence[int]) -> bool:
        """Return whether ``(level, midx)`` is an active (leaf) cell.

        Args:
            level (int): Hierarchy level.
            midx (Sequence[int]): Per-axis index in level-``level`` coordinates.

        Returns:
            bool: ``True`` iff a cell with this level and multi-index is currently
            active (a leaf); ``False`` if it is out of range, not yet created, or has
            been refined away.
        """
        return bool(self._h.is_active_leaf(int(level), [int(i) for i in midx]))

    # ------------------------------------------------------------------
    # Active-set accessors
    # ------------------------------------------------------------------

    def level_cells_per_axis(self, level: int) -> tuple[int, ...]:
        """Return the per-axis cell count of the level-``level`` grid.

        This is a formula — ``level`` need not be an existing hierarchy level.
        Values above ``max_level`` return the count for the hypothetical finer grid
        that would result from additional uniform subdivision, as long as the count
        of every axis fits in a signed 64-bit integer.  This differs from
        :meth:`active_blocks`, :meth:`active_leaf_mask`, and :meth:`subdomain_mask`,
        which all require ``level <= max_level``.

        Args:
            level (int): Hierarchy level.  Must be ``>= 0``; values above
                ``max_level`` are accepted and return the geometrically valid count,
                unless it exceeds ``2**63 - 1`` along some axis.

        Returns:
            tuple[int, ...]: ``root.cells_per_axis[k] * factor[k] ** level`` for
            every axis ``k``.

        Raises:
            ValueError: If ``level < 0``.
            OverflowError: If ``root.cells_per_axis[k] * factor[k] ** level`` exceeds
                ``2**63 - 1`` for some axis ``k``.
        """
        return tuple(int(n) for n in self._h.level_cells_per_axis(int(level)))

    def active_blocks(self, level: int) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        """Return the active-leaf blocks at ``level``.

        Args:
            level (int): Hierarchy level.  Must be in ``[0, max_level]``.

        Returns:
            tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]: The sorted,
            non-overlapping ``(lo, hi)`` integer rectangles of active (leaf) cells
            at ``level``, in level-``level`` coordinates.

        Raises:
            ValueError: If ``level`` is outside ``[0, max_level]``.
        """
        return tuple((tuple(lo), tuple(hi)) for lo, hi in self._h.active_blocks(int(level)))

    def active_leaf_mask(self, level: int) -> npt.NDArray[np.bool_]:
        r"""Return a boolean mask of the active-leaf cells at ``level``.

        Args:
            level (int): Hierarchy level.  Must be in ``[0, max_level]``.

        Returns:
            npt.NDArray[np.bool\_]: Fresh array of shape
            ``level_cells_per_axis(level)``; ``True`` where the level-``level``
            cell ``(level, midx)`` is an active (leaf) cell.

        Raises:
            ValueError: If ``level`` is outside ``[0, max_level]``.
        """
        mask: npt.NDArray[np.bool_] = self._h.active_leaf_mask(int(level))
        return mask

    def subdomain_mask(self, level: int) -> npt.NDArray[np.bool_]:
        r"""Return a boolean mask of the level-``level`` refined subdomain.

        A level-``level`` cell lies in the subdomain :math:`\Omega_{level}` (the
        region refined to at least ``level``) iff it is **not** covered by an
        active leaf of a coarser level.  The mask is computed at the level-``level``
        resolution by projecting every coarser active-leaf block up to ``level`` and
        clearing those cells.

        Args:
            level (int): Hierarchy level.  Must be in ``[0, max_level]``.

        Returns:
            npt.NDArray[np.bool\_]: Fresh array of shape
            ``level_cells_per_axis(level)``; ``True`` where the level-``level``
            cell lies in :math:`\Omega_{level}`.

        Raises:
            ValueError: If ``level`` is outside ``[0, max_level]``.

        Note:
            The mask is sized to the level-``level`` cell grid and computed on
            demand; it is never stored.
        """
        mask: npt.NDArray[np.bool_] = self._h.subdomain_mask(int(level))
        return mask

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def refine(
        self,
        level: int,
        lo: Sequence[int],
        hi: Sequence[int],
    ) -> None:
        """Refine the rectangular region ``[lo, hi)`` at ``level`` to ``level+1``.

        Union semantics: only the currently-active portion of ``[lo, hi)`` is
        refined, leaving cells that are already deeper untouched.  If the
        intersection with active blocks at ``level`` is empty, the call is a
        silent no-op.  Overlapping calls therefore safely extend the refined
        region.

        That convenience costs invertibility.  Promoting only part of a box sends
        several distinct grids to the same result, so :meth:`refine` is **not
        injective** on the grid state and no single :meth:`coarsen` can undo it in
        general: ``coarsen(level, lo, hi)`` reverses this call only when every cell
        of ``[lo, hi)`` was an active leaf at ``level`` beforehand.  When it was
        not, the paired :meth:`coarsen` either refuses the box, if part of it now
        sits deeper than ``level+1``, or demotes all of it, including the cells this
        call left alone, and so removes children an earlier :meth:`refine` created.
        Where that must not happen, name the cells rather than a box on
        the side that destroys them: :meth:`coarsen_cells` reactivates a parent only
        when all of its children are named active leaves, so nothing the caller did
        not name is removed.  (:meth:`refine_cells` also marks by id, but it refines
        the bounding box of the ids given at each level, so it can promote a cell the
        caller did not mark; that costs nothing, since refining removes no cell.)

        After the call all flat cell ids are **reassigned** (the BVH, cell tags,
        and facet tags are also invalidated).

        Args:
            level (int): Refinement level at which the region lives.  Must
                satisfy ``0 <= level <= max_level``.
            lo (Sequence[int]): Per-direction start index (inclusive), in
                level-``level`` coordinates.
            hi (Sequence[int]): Per-direction end index (exclusive), in
                level-``level`` coordinates.

        Raises:
            ValueError: If ``level`` is out of range, ``lo``/``hi`` have the
                wrong length, any ``lo[k] >= hi[k]``, or ``[lo, hi)`` falls
                entirely outside the level-``level`` domain.
            OverflowError: If ``level`` is ``max_level`` and the new level
                ``level + 1`` would have more than ``2**63 - 1`` cells along some
                axis.  The grid is unchanged.

        References:
            Refinement and coarsening algorithms for adaptive hierarchical-spline
            meshes :cite:p:`garau2018algorithms`.
        """
        self._swap(self._h.refined(int(level), [int(x) for x in lo], [int(x) for x in hi]))

    def refine_cells(self, cell_ids: Sequence[int]) -> None:
        """Refine a set of active cells using per-level bounding-box aggregation.

        Groups ``cell_ids`` by their level, computes the bounding box of all
        cells at each level (the smallest axis-aligned rectangle containing
        them), and calls :meth:`refine` once per level.

        Args:
            cell_ids (Sequence[int]): Flat cell ids to refine.  Cells from
                multiple levels are handled; repeated ids are silently ignored.

        Raises:
            IndexError: If any id in ``cell_ids`` is out of range.
            OverflowError: As :meth:`refine`: a refined level would have more than
                ``2**63 - 1`` cells along some axis.  The grid is unchanged.
        """
        if not cell_ids:
            return
        self._swap(self._h.refined_cells([int(cid) for cid in cell_ids]))

    def coarsen(
        self,
        level: int,
        lo: Sequence[int],
        hi: Sequence[int],
    ) -> None:
        """Demote the rectangular region ``[lo, hi)`` from ``level+1`` back to ``level``.

        Reactivates the level-``level`` cells in ``[lo, hi)`` and removes their
        level-``(level+1)`` children.  The region must be **fully refined to exactly
        level ``level+1``**: every child cell in ``[lo*factor, hi*factor)`` must be an
        active leaf at ``level+1`` (none further refined, none still a leaf at
        ``level``); otherwise the call raises and the message names the cells that
        break it.

        The **whole** box is demoted, while :meth:`refine` promotes only the box's
        currently-active portion.  :meth:`refine` is therefore not injective on the
        grid state, and this call inverts it **only when every cell of ``[lo, hi)``
        was an active leaf at ``level`` before that refine**.  When it was not, this
        call either refuses the box, as above, or removes children an earlier
        :meth:`refine` created: after ``refine(l, A)`` and an overlapping
        ``refine(l, B)``, ``coarsen(l, B)`` demotes all of ``B``, including the part
        of ``A`` that lies inside it.  It is demoting exactly the box it was given,
        but it is not undoing the second refine.  To coarsen without risking
        refinement the caller did not mean to lose, mark cells by id:
        :meth:`coarsen_cells` reactivates a parent only when all of its children are
        named active leaves.  This box form stays the cheaper call for a region known
        to be uniformly refined to ``level+1``.

        The opposite order carries no hypothesis: coarsening leaves the whole box
        active at ``level``, so ``refine(level, lo, hi)`` always undoes
        ``coarsen(level, lo, hi)``.

        After the call all flat cell ids are **reassigned** (the BVH, cell tags, and
        facet tags are invalidated).

        Args:
            level (int): Level whose cells are reactivated.  Must satisfy
                ``0 <= level < max_level``.
            lo (Sequence[int]): Per-direction start index (inclusive), in level-``level``
                coordinates.
            hi (Sequence[int]): Per-direction end index (exclusive), in level-``level``
                coordinates.

        Raises:
            ValueError: If ``level`` is out of range, ``lo``/``hi`` have the wrong
                length, any ``lo[k] >= hi[k]``, ``[lo, hi)`` is out of bounds, or the
                region is not fully refined to exactly level ``level+1``.

        References:
            Coarsening algorithms for adaptive hierarchical-spline meshes
            :cite:p:`garau2018algorithms`.
        """
        self._swap(self._h.coarsened(int(level), [int(x) for x in lo], [int(x) for x in hi]))

    def coarsen_cells(self, cell_ids: Sequence[int]) -> None:
        """Demote every parent whose children are all named in ``cell_ids``.

        The route that destroys only what the caller named.  ``cell_ids`` are grouped
        by parent cell, and a parent is reactivated -- its children removed -- only when
        **every one** of its ``prod(factor)`` children is both an active leaf and present
        in ``cell_ids``.  Three things are therefore skipped silently rather than
        refused: a parent only some of whose children are named, a parent one of whose
        children has been refined further, and an id at level 0, which has no parent.  A
        call that demotes nothing is not an error, and no cell outside ``cell_ids`` is
        ever removed.

        That is what separates this from :meth:`coarsen`, which demotes the whole box it
        is given and so can remove children an earlier :meth:`refine` created.  Prefer
        this call wherever losing unmarked refinement would be a defect; prefer
        :meth:`coarsen` for a region known to be uniformly refined, where naming one box
        is cheaper than enumerating its cells (this call demotes one parent at a time, so
        it repacks the block lists once per parent).

        Ids spanning several levels are handled in one call, deepest level first.  The
        order is observable and so is fixed: the reactivated parents are appended to the
        block lists in that order, the greedy merge in normalization turns that into a
        rectangle partition, and flat ids are handed out block by block.  Coarsening does
        **not** cascade, whichever order is used -- a cell reborn by this call was not an
        active leaf when the caller chose its ids, so it cannot be among them, and its own
        parent is therefore never complete.

        Note:
            :meth:`refine_cells` is *not* the mirror of this method: it refines the
            bounding box of the ids given at each level, so it can promote cells the
            caller never named.  That costs nothing there, because refining removes no
            cell, whereas coarsening destroys one.

        Args:
            cell_ids (Sequence[int]): Flat ids of active leaf cells to coarsen away.
                Repeated ids and ids at level 0 are ignored.  An empty sequence is a
                no-op.

        Raises:
            IndexError: If any id in ``cell_ids`` is out of range.
        """
        self._swap(self._h.coarsened_cells([int(cid) for cid in cell_ids]))

    # ------------------------------------------------------------------
    # Bulk accessors
    # ------------------------------------------------------------------

    def collect_cell_bounds(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Materialize ``(cell_lo, cell_hi)`` in flat-id order for the BVH.

        Computed in C++, in parallel over the active blocks; results are
        identical to calling :meth:`cell_bounds` per cell.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            ``(cell_lo, cell_hi)`` of shape ``(num_cells, ndim)``.
        """
        cell_lo, cell_hi = self._h.collect_cell_bounds()
        return cell_lo, cell_hi

    def export_cells(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Return deduplicated leaf-cell vertices and their connectivity.

        The mesh a dolfinx-style consumer needs: one vertex per distinct corner of the
        active leaves, shared exactly between the cells that meet there, including
        corners shared across levels (a hanging vertex appears once).

        Deduplication is **exact and tolerance-free**: corners are identified on the
        finest-level integer lattice, where a level-``l`` cell at ``midx`` contributes
        ``(midx[k] + bit[k]) * factor[k] ** (max_level - l)``. Equal corners are equal
        integers, and each distinct integer node is then mapped to a coordinate
        through one formula, so coincident corners come out bitwise identical.

        Corner order within a row is the tensor-product (basix/dolfinx) convention:
        corner ``c`` takes the ``hi`` bound on axis ``k`` iff bit ``k`` of ``c`` is set,
        with axis 0 the least-significant bit. In 2D that is
        ``[(lo0, lo1), (hi0, lo1), (lo0, hi1), (hi0, hi1)]``. This is a *corner*
        convention and is deliberately unlike pantr's C-order flat cell and dof ids,
        where the **last** axis varies fastest.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: ``(points, conn)``,
            both read-only. ``points`` has shape ``(num_vertices, ndim)``, sorted
            lexicographically by integer lattice coordinate; ``conn`` has shape
            ``(num_cells, 2 ** ndim)`` and indexes ``points``, one row per active leaf
            in flat cell-id order.

        Note:
            ``points[conn[cid]]`` reproduces the corners of :meth:`cell_bounds` to
            within ``2 gamma_2 |x| + 2 gamma_4 |x - b|``, where ``b`` is the lower
            breakpoint of the root cell holding the corner and
            ``gamma_m = m u / (1 - m u)`` with ``u = eps / 2``. Both sides build
            ``b + s * w / factor ** level`` from different expressions, and forming
            ``w``, the division, the integer-scaled multiply and the addition are four
            correctly rounded operations each, with a fifth on the :meth:`cell_bounds`
            side of a ``hi`` corner, which it writes as ``lo + size``.

            **The second term is what the bound is about.** ``|x - b|`` is bounded by
            the root cell's width and not by ``|x|``, so only where every coordinate
            dominates its own offset -- a domain with ``b >= 0`` is the usual reason --
            does this collapse to a multiple of ``eps |x|``. A corner small against the
            width of the root cell holding it is a counterexample rather than a corner
            case, and ``tests/test_grid_hierarchical.py`` pins one.

            It is bitwise whenever every intermediate is exactly representable -- a
            dyadic ``factor`` on dyadic root breakpoints, which covers the usual
            unit-domain power-of-two case. Bitwise agreement in general is not merely
            unimplemented, it is unattainable: :meth:`cell_bounds` scales a root cell's
            width by ``factor ** level``, so for a corner shared between two levels it
            evaluates a *different* expression on each side and can return two floats
            one ulp apart, and no single deduplicated vertex can equal both.

            **Stated hypothesis: coordinates are normal, not subnormal.** The bound is
            a product of relative factors with the magnitudes, so below roughly
            ``2e-308`` both terms underflow to exactly zero while the difference they
            bound is still a nonzero subnormal: the expression stops implementing the
            inequality, though the inequality itself still holds in the reals.
            Nothing excludes that domain, so this is a hypothesis and not a guarantee.

        Warning:
            Materializes ``O(num_cells * 2 ** ndim * ndim)`` integers before
            deduplication, so a large 3D hierarchy costs several times ``conn`` in peak
            memory.
        """
        points, conn = self._h.export_cells()
        points.flags.writeable = False
        conn.flags.writeable = False
        return points, conn

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a compact developer-friendly representation.

        Returns:
            str: Shows dimension, root cell counts, factor, active cells,
            and max level.
        """
        return (
            f"HierarchicalGrid(ndim={self._ndim}, "
            f"root_cells={self._root.cells_per_axis}, "
            f"factor={self._factor}, "
            f"num_cells={self._num_cells}, "
            f"max_level={self._max_level})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def hierarchical_grid(
    root: TensorProductGrid,
    factor: int | Sequence[int],
) -> HierarchicalGrid:
    """Build a :class:`HierarchicalGrid` from a root grid and a subdivision factor.

    Args:
        root (TensorProductGrid): The level-0 grid.
        factor (int | Sequence[int]): Per-direction subdivision factor.  A
            scalar is broadcast to all axes.  Each entry must be ``>= 1``.

    Returns:
        HierarchicalGrid: A new hierarchical grid starting with all root cells
        active at level 0.

    Raises:
        TypeError: If ``root`` is not a :class:`TensorProductGrid`.
        ValueError: If ``factor`` has the wrong length or any entry is ``< 1``.
    """
    return HierarchicalGrid(root, factor)


__all__ = ["HierarchicalGrid", "hierarchical_grid"]
