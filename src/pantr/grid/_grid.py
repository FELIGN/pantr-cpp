"""Abstract base class for structured cell grids.

A :class:`Grid` is a partition of a parametric domain into cells with *implicit*
(computed, not stored) connectivity. It is the shared grid abstraction for the
PaNTr stack: background grids for immersed / unfitted discretizations, knot-span
grids of B-spline spaces, and (later) hierarchical refinement grids all satisfy
this contract.

Design
------

The contract is deliberately small. A concrete grid must define only:

- :attr:`Grid.ndim`, :attr:`Grid.num_cells` -- size metadata;
- :meth:`Grid.cell_bounds` -- the axis-aligned ``(lo, hi)`` corners of a cell;
- :meth:`Grid.locate` -- the cell containing a point (or ``None``);
- :meth:`Grid.neighbor_across_facet` -- the cell across a given local facet.

Everything else has a concrete default built from those primitives, assuming the
common case of axis-aligned box cells with ``2 * ndim`` facets: cell AABBs,
reference maps, facet bounds, neighbour lists, boundary-facet detection, batch
point location, and an :class:`AABB` overlap query backed by a lazily-built
:class:`pantr.grid.BVH`. A concrete grid overrides any default for which it has a
cheaper specialization (for example, :class:`pantr.grid.TensorProductGrid`
replaces :meth:`locate`, :meth:`neighbor_across_facet`, and the cell-bounds
collection with per-axis arithmetic).

Cells live in parametric coordinates; mapping to physical space is the
responsibility of the geometry (for example, a B-spline) layered on top of the
grid -- the grid itself stores no geometry map.

Concrete grids
--------------

pantr's grids, :class:`pantr.grid.TensorProductGrid` and
:class:`pantr.grid.HierarchicalGrid`, are implemented in C++: each holds its
immutable compiled value in ``_h``, and the defaults above forward to it, so a
default method sees the compiled grid's own primitives. :class:`Grid` therefore
cannot be subclassed (nor can the two concrete grids): a subclass overriding a
primitive would be ignored by the compiled defaults, so subclassing raises
:class:`TypeError`. ``isinstance(g, Grid)`` still identifies a grid.

Tagging
-------

Each grid exposes two lazily-created sparse tag registries,
:attr:`Grid.cell_tags` and :attr:`Grid.facet_tags`, for attaching integer labels
to a subset of cells / facets (in / out / cut classification, boundary-condition
markers, ...). They stay empty until first use, so an untagged grid carries no
per-cell tag footprint. Deciding *what* to tag is the consumer's job.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, NamedTuple, Self

import numpy as np

from ._bvh import BVH
from ._grid_utils import _as_float64
from ._tags import CellTags, FacetTags

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy.typing as npt

    from ..geometry import AABB
    from ..transform import AffineTransform

_CONCRETE_GRIDS: frozenset[tuple[str, str]] = frozenset(
    {
        ("pantr.grid._tensor_product_grid", "TensorProductGrid"),
        ("pantr.grid._hierarchical_grid", "HierarchicalGrid"),
    }
)
"""``(module, qualified name)`` of the grids pantr defines, the only classes allowed to derive."""


class Grid(abc.ABC):
    """Abstract structured cell grid with implicit connectivity.

    See the module docstring for the contract and the set of methods a concrete
    grid must implement versus those provided as overridable defaults. The size
    metadata is exposed through the :attr:`ndim` and :attr:`num_cells`
    properties.

    The class cannot be subclassed except by pantr's own grids (see the module
    docstring).

    Attributes:
        _h (Any): The compiled grid every default forwards to (private).
    """

    __slots__ = ("_bvh_w", "_cell_tags", "_facet_tags", "_h")

    #: The compiled grid (private).
    _h: Any

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        """Refuse every subclass but pantr's two grids.

        Args:
            **kwargs (Any): Passed to the next ``__init_subclass__``.

        Raises:
            TypeError: If ``cls`` is not :class:`pantr.grid.TensorProductGrid` or
                :class:`pantr.grid.HierarchicalGrid`.
        """
        super().__init_subclass__(**kwargs)
        if (cls.__module__, cls.__qualname__) not in _CONCRETE_GRIDS:
            raise TypeError(
                f"{cls.__qualname__} cannot subclass Grid or a pantr grid: pantr's grids are "
                "TensorProductGrid and HierarchicalGrid, implemented in C++."
            )

    def __init__(self) -> None:
        """Initialize the lazy spatial-index and tag-registry slots.

        A concrete grid calls ``super().__init__()`` and then sets ``_h``.
        """
        self._bvh_w: BVH | None = None
        self._cell_tags: CellTags | None = None
        self._facet_tags: FacetTags | None = None

    # ------------------------------------------------------------------
    # Abstract contract
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def ndim(self) -> int:
        """Get the spatial dimension of the grid.

        Returns:
            int: Number of axes (``>= 1``).
        """

    @property
    @abc.abstractmethod
    def num_cells(self) -> int:
        """Get the number of cells in this (local) grid.

        Returns:
            int: Non-negative cell count.
        """

    @abc.abstractmethod
    def cell_bounds(
        self,
        cid: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the axis-aligned ``(lo, hi)`` corners of cell ``cid``.

        Args:
            cid (int): Cell identifier; must satisfy ``0 <= cid < num_cells``.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Fresh,
            writeable length-``ndim`` ``float64`` arrays.

        Raises:
            IndexError: If ``cid`` is out of range.
        """

    @abc.abstractmethod
    def locate(self, pt: npt.ArrayLike) -> int | None:
        """Return the cell containing ``pt``, or ``None`` if ``pt`` is outside.

        Args:
            pt (npt.ArrayLike): Length-``ndim`` point in parametric coordinates.

        Returns:
            int | None: Containing cell id, or ``None`` when ``pt`` lies outside
            every cell.

        Raises:
            ValueError: If ``pt`` does not have length ``ndim``.
        """

    @abc.abstractmethod
    def neighbor_across_facet(self, cid: int, lfid: int) -> int | None:
        """Return the cell across local facet ``lfid`` of ``cid``, or ``None``.

        The result is ``None`` iff the facet lies on the grid's outer boundary.

        Args:
            cid (int): Cell identifier.
            lfid (int): Local facet identifier in ``[0, num_local_facets(cid))``.

        Returns:
            int | None: Neighbouring cell id, or ``None`` on a boundary facet.

        Raises:
            IndexError: If ``cid`` or ``lfid`` is out of range.
        """

    if TYPE_CHECKING:
        # Declared for the type checker only: each concrete grid defines it (the only classes
        # that may derive from Grid), and an abstract method would change what ``Grid()``
        # reports as missing.
        @classmethod
        def _from_handle(cls, h: Any) -> Self:  # noqa: ANN401
            """Wrap a compiled grid of this kind without copying it (private)."""

    # ------------------------------------------------------------------
    # Cell accessors (defaults built on the contract)
    # ------------------------------------------------------------------

    def iter_cells(self) -> Iterator[int]:
        """Yield every cell identifier exactly once, in id order.

        Returns:
            Iterator[int]: ``iter(range(num_cells))``.
        """
        return iter(range(self.num_cells))

    def cell_aabb(self, cid: int) -> AABB:
        """Return cell ``cid``'s axis-aligned bounding box.

        Args:
            cid (int): Cell identifier.

        Returns:
            AABB: Equivalent to ``AABB(*self.cell_bounds(cid))``.

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        from ..geometry import AABB  # noqa: PLC0415

        return AABB._from_handle(self._h.cell_aabb(int(cid)))

    def cell_level(self, cid: int) -> int:
        """Return the refinement level of cell ``cid``.

        Non-hierarchical grids always return ``0``; hierarchical backends use
        level ``>= 1`` for refined cells.

        Args:
            cid (int): Cell identifier.

        Returns:
            int: Refinement level (``0`` for a flat grid).

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        return int(self._h.cell_level(int(cid)))

    def child_cells(self, cid: int) -> tuple[int, ...]:
        """Return the immediate refinement children of cell ``cid``.

        For a flat (non-hierarchical) grid this is always empty.

        Args:
            cid (int): Cell identifier.

        Returns:
            tuple[int, ...]: Child cell ids; empty for a flat grid.

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        return tuple(self._h.child_cells(int(cid)))

    def reference_map(self, cid: int) -> AffineTransform:
        """Return the affine map ``[0, 1]^ndim -> cell`` for cell ``cid``.

        For an axis-aligned cell this is ``T(u) = diag(hi - lo) @ u + lo``.

        Args:
            cid (int): Cell identifier.

        Returns:
            AffineTransform: Push-forward from the unit cube to the cell.

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        from ..transform import AffineTransform  # noqa: PLC0415

        return AffineTransform._from_handle(self._h.reference_map(int(cid)))

    def neighbors(self, cid: int) -> list[int]:
        """Return the facet-neighbour cell ids of ``cid``.

        Built by collecting :meth:`neighbor_across_facet` over every local facet
        and dropping boundary facets (``None``).

        Args:
            cid (int): Cell identifier.

        Returns:
            list[int]: Neighbouring cell ids; length between ``ndim`` (corner
            cell) and ``2 * ndim`` (interior cell) for a box grid.

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        return list(self._h.neighbors(int(cid)))

    # ------------------------------------------------------------------
    # Restriction / windowing
    # ------------------------------------------------------------------

    def restrict(self, cell_ids: npt.ArrayLike) -> GridRestriction:
        """Return the structured sub-grid spanning a subset of cells.

        The sub-grid is the smallest grid of the same kind that contains every
        requested cell -- for a tensor-product grid, the multi-index bounding box
        of ``cell_ids``; for a hierarchical grid, the bounding box in root-cell
        coordinates. Its breakpoints are *pure slices* of this grid's (never
        re-clamped or re-based), so a B-spline or field built on the sub-grid
        agrees pointwise with the global one over the shared cells.

        The bounding box may contain cells that were not requested (when
        ``cell_ids`` is non-convex); :attr:`GridRestriction.in_subset` flags, per
        local cell, whether it was requested (``True``) or is bounding-box fill
        (``False``).

        Args:
            cell_ids (npt.ArrayLike): Flat cell identifiers to span; duplicates
                are ignored. Each must satisfy ``0 <= cid < num_cells``.

        Returns:
            GridRestriction: The windowed sub-grid, the ``local_to_global_cell``
            map, and the ``in_subset`` mask.

        Raises:
            ValueError: If ``cell_ids`` is empty.
            IndexError: If any cell id is out of range.
            TypeError: If ``cell_ids`` is not integer-valued.
        """
        handle, local_to_global, in_subset = self._restrict_handle(cell_ids)
        return GridRestriction(type(self)._from_handle(handle), local_to_global, in_subset)

    # ------------------------------------------------------------------
    # Facet accessors (axis-aligned box defaults)
    # ------------------------------------------------------------------

    def num_local_facets(self, cid: int) -> int:
        """Return the number of local facets of cell ``cid``.

        Defaults to ``2 * ndim`` (an axis-aligned box).

        Args:
            cid (int): Cell identifier.

        Returns:
            int: ``2 * ndim``.

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        return int(self._h.num_local_facets(int(cid)))

    def local_facet_axis_side(self, cid: int, lfid: int) -> tuple[int, int]:
        """Return ``(axis, side)`` for local facet ``lfid`` of cell ``cid``.

        Uses the conventional ``lfid = 2 * axis + side`` encoding, with
        ``axis in [0, ndim)`` and ``side in {0, 1}`` (``0`` = low face,
        ``1`` = high face).

        Args:
            cid (int): Cell identifier.
            lfid (int): Local facet identifier in ``[0, 2 * ndim)``.

        Returns:
            tuple[int, int]: ``(axis, side)``.

        Raises:
            IndexError: If ``cid`` or ``lfid`` is out of range.
        """
        axis, side = self._h.local_facet_axis_side(int(cid), int(lfid))
        return int(axis), int(side)

    def local_facet_bounds(
        self,
        cid: int,
        lfid: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the degenerate ``(lo, hi)`` AABB of facet ``lfid`` of cell ``cid``.

        On the facet's normal axis both corners coincide (the cell's lo corner
        for ``side == 0``, hi corner for ``side == 1``); the other axes span the
        cell's extent.

        Args:
            cid (int): Cell identifier.
            lfid (int): Local facet identifier in ``[0, 2 * ndim)``.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Fresh,
            writeable length-``ndim`` ``float64`` arrays.

        Raises:
            IndexError: If ``cid`` or ``lfid`` is out of range.
        """
        lo, hi = self._h.local_facet_bounds(int(cid), int(lfid))
        return lo, hi

    def is_mesh_boundary_facet(self, cid: int, lfid: int) -> bool:
        """Return whether facet ``lfid`` of cell ``cid`` is on the grid's outer boundary.

        Defaults to ``neighbor_across_facet(cid, lfid) is None``.

        Args:
            cid (int): Cell identifier.
            lfid (int): Local facet identifier in ``[0, num_local_facets(cid))``.

        Returns:
            bool: ``True`` iff no neighbouring cell shares the facet.

        Raises:
            IndexError: If ``cid`` or ``lfid`` is out of range.
        """
        return bool(self._h.is_mesh_boundary_facet(int(cid), int(lfid)))

    def boundary_facets(self) -> npt.NDArray[np.int64]:
        """Return every facet on the grid's outer boundary as ``(cid, lfid)`` rows.

        Only the outer (domain) boundary: a facet shared with any neighbouring cell
        is excluded, whether that neighbour is conforming, coarser, or finer. For a
        hierarchical grid this means level interfaces are **not** boundary facets,
        even though the leaf-cell mesh makes them topologically exterior (hanging
        vertices) and so a consumer searching that mesh over-reports them.

        The default enumerates :meth:`is_mesh_boundary_facet` over every cell and
        local facet, costing ``O(num_cells * 2 * ndim)`` neighbour queries.
        A grid with exploitable structure overrides it.

        Returns:
            npt.NDArray[np.int64]: Read-only array of shape ``(n, 2)`` whose rows are
            ``(cid, lfid)`` with ``lfid = 2 * axis + side`` (see
            :meth:`local_facet_axis_side`), sorted lexicographically by
            ``(cid, lfid)``.
        """
        out: npt.NDArray[np.int64] = self._h.boundary_facets()
        out.flags.writeable = False
        return out

    def hanging_neighbors(self, cid: int, lfid: int) -> tuple[int, ...]:
        """Return all active cells sharing facet ``lfid`` of ``cid``.

        For conforming grids (flat :class:`TensorProductGrid`) this is
        equivalent to :meth:`neighbor_across_facet` wrapped in a tuple.  For
        hierarchical grids, a single coarse face may abut multiple fine cells
        (hanging nodes); this method returns all of them.

        Subclasses with hanging-node support should override this method.

        Args:
            cid (int): Cell identifier.
            lfid (int): Local facet identifier in ``[0, num_local_facets(cid))``.

        Returns:
            tuple[int, ...]: All neighbouring cell ids across the facet;
            empty when the facet lies on the grid's outer boundary.

        Raises:
            IndexError: If ``cid`` or ``lfid`` is out of range.
        """
        return tuple(self._h.hanging_neighbors(int(cid), int(lfid)))

    # ------------------------------------------------------------------
    # Point location and spatial queries
    # ------------------------------------------------------------------

    def locate_many(self, points: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Locate a batch of points, returning one cell id per point.

        Points outside the grid map to ``-1`` (including points with a NaN or
        infinite coordinate). The compiled grids run the search in parallel over
        the points.

        Args:
            points (npt.ArrayLike): ``(npts, ndim)`` array-like of points, or a
                single length-``ndim`` point.

        Returns:
            npt.NDArray[np.int64]: Shape ``(npts,)`` cell ids; ``-1`` for points
            outside the grid.

        Raises:
            ValueError: If the trailing axis of ``points`` is not ``ndim``.
        """
        out: npt.NDArray[np.int64] = self._h.locate_many(self._normalize_points(points))
        return out

    def query_aabb(self, aabb: AABB) -> npt.NDArray[np.int64]:
        """Return the ids of every cell whose AABB overlaps ``aabb``.

        Backed by a :class:`pantr.grid.BVH` over the grid's cell AABBs, built
        lazily on first call and cached for the grid's lifetime. The overlap test
        is inclusive on every axis, so cells touching ``aabb`` on any face, edge,
        or corner are included.

        Args:
            aabb (AABB): Query box; must match :attr:`ndim`.

        Returns:
            npt.NDArray[np.int64]: Overlapping cell ids (unordered).

        Raises:
            ValueError: If ``aabb.ndim != self.ndim``.
        """
        out: npt.NDArray[np.int64] = self._h.query_aabb(aabb._h)
        return out

    def cell_bvh(self) -> BVH:
        """Return the cached :class:`pantr.grid.BVH` over the grid's cell AABBs.

        Built lazily on first call from ``collect_cell_bounds`` and cached.
        Building the BVH materializes ``O(num_cells)`` node arrays, so it is
        deferred until an :meth:`query_aabb` (or direct) call needs it -- an
        untagged, un-queried grid never pays this cost.

        Returns:
            BVH: The grid's spatial index over its cell AABBs.

        Note:
            Thread-safe: the compiled grid builds the index once, whichever
            thread asks first. The returned wrapper is cached until the next
            structural change of the grid.
        """
        if self._bvh_w is None:
            self._bvh_w = BVH._from_handle(self._h.cell_bvh())
        return self._bvh_w

    def collect_cell_bounds(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Materialize per-cell ``(lo, hi)`` as ``(num_cells, ndim)`` arrays.

        Cells appear in id order.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            ``(cell_lo, cell_hi)`` of shape ``(num_cells, ndim)``.
        """
        cell_lo, cell_hi = self._h.collect_cell_bounds()
        return cell_lo, cell_hi

    # ------------------------------------------------------------------
    # Tagging
    # ------------------------------------------------------------------

    @property
    def cell_tags(self) -> CellTags:
        """Get the grid's sparse cell-tag registry (created lazily).

        Returns:
            CellTags: A registry of named ``(cell_ids, values)`` associations,
            empty until first use.
        """
        if self._cell_tags is None:
            self._cell_tags = CellTags(self.num_cells)
        return self._cell_tags

    @property
    def facet_tags(self) -> FacetTags:
        """Get the grid's sparse facet-tag registry (created lazily).

        Returns:
            FacetTags: A registry of named ``((cell_id, local_facet_id), value)``
            associations, empty until first use. Sized for ``2 * ndim`` facets
            per cell (an axis-aligned box grid).
        """
        if self._facet_tags is None:
            self._facet_tags = FacetTags(self.num_cells, 2 * self.ndim)
        return self._facet_tags

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _bvh(self) -> BVH | None:
        """Get the BVH if it has been built, else ``None`` (never builds it).

        Returns:
            BVH | None: The wrapper of the compiled grid's spatial index, or
            ``None`` before the first :meth:`query_aabb` / :meth:`cell_bvh`.
        """
        return self.cell_bvh() if self._h.bvh_built() else None

    def _state(self) -> tuple[Any, ...]:
        """Return the picklable constructor state of this grid (without its tags).

        Returns:
            tuple[Any, ...]: What :meth:`_from_state` rebuilds the grid from.

        Raises:
            NotImplementedError: Always; each concrete grid defines it.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define _state().")

    @classmethod
    def _from_state(cls, state: tuple[Any, ...]) -> Grid:
        """Rebuild a grid from :meth:`_state` (tags not restored).

        Args:
            state (tuple[Any, ...]): The result of :meth:`_state`.

        Returns:
            Grid: The rebuilt grid.

        Raises:
            NotImplementedError: Always; each concrete grid defines it.
        """
        raise NotImplementedError(f"{cls.__name__} does not define _from_state().")

    def _clone(self) -> Grid:
        """Return a new wrapper of the same grid value (tags not carried over).

        Returns:
            Grid: A grid equal to this one, sharing its immutable compiled value
            where the concrete grid can.
        """
        return type(self)._from_state(self._state())

    def __copy__(self) -> Grid:
        """Copy the grid: a new wrapper of the same value, with copies of its tag registries.

        The grid is immutable except through its own structural methods (which
        swap the compiled value inside one wrapper), so a copy shares the value
        and is independent of the original afterwards. The tag registries are
        copied as new wrappers over the same (immutable) tag values, so setting
        a tag on one grid does not affect the other.

        Returns:
            Grid: The copy.
        """
        new = self._clone()
        new._cell_tags = None if self._cell_tags is None else self._cell_tags._copy()
        new._facet_tags = None if self._facet_tags is None else self._facet_tags._copy()
        return new

    def __deepcopy__(self, memo: dict[int, Any]) -> Grid:
        """Deep-copy the grid; identical to :meth:`__copy__`, as the value is immutable.

        Args:
            memo (dict[int, Any]): The copy memo (unused).

        Returns:
            Grid: The copy.
        """
        return self.__copy__()

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle through the constructor state, carrying the tag registries.

        Returns:
            tuple[Any, ...]: The rebuild function and its arguments.
        """
        return (_rebuild_grid, (type(self), self._state(), self._cell_tags, self._facet_tags))

    def _restrict_handle(self, cell_ids: npt.ArrayLike) -> tuple[Any, Any, Any]:
        """Restrict the compiled grid to the window spanning ``cell_ids``.

        Args:
            cell_ids (npt.ArrayLike): Flat cell identifiers.

        Returns:
            tuple[Any, Any, Any]: The compiled sub-grid, the read-only
            ``local_to_global_cell`` map and the read-only boolean ``in_subset`` mask.

        Raises:
            TypeError: If ``cell_ids`` is not integer-valued.
            ValueError: If ``cell_ids`` is empty.
            IndexError: If a cell id is out of range.
        """
        ids = np.asarray(cell_ids).ravel()
        # An empty request reaches the compiled grid first, which refuses it.
        if ids.size > 0 and not np.issubdtype(ids.dtype, np.integer):
            raise TypeError(f"restrict: cell_ids must be integer-valued; got dtype {ids.dtype}.")
        handle, local_to_global, in_subset = self._h.restrict(
            np.ascontiguousarray(ids, dtype=np.int64)
        )
        local_to_global.flags.writeable = False
        in_subset.flags.writeable = False
        return handle, local_to_global, in_subset

    def _normalize_points(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Coerce ``points`` to a C-contiguous ``(npts, ndim)`` ``float64`` array.

        A single length-``ndim`` point is promoted to shape ``(1, ndim)``.

        Args:
            points (npt.ArrayLike): Points array-like.

        Returns:
            npt.NDArray[np.float64]: Shape ``(npts, ndim)``.

        Raises:
            ValueError: If the trailing axis is not ``ndim`` or the rank is not
                1 or 2.
        """
        arr = _as_float64(points, name="points")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != self.ndim:  # noqa: PLR2004
            raise ValueError(
                f"points must have shape (npts, {self.ndim}) or ({self.ndim},); "
                f"got shape {arr.shape}."
            )
        return np.ascontiguousarray(arr)


def _rebuild_grid(
    cls: type[Grid],
    state: tuple[Any, ...],
    cell_tags: CellTags | None,
    facet_tags: FacetTags | None,
) -> Grid:
    """Unpickle a grid: rebuild it from its constructor state and restore its tags.

    Args:
        cls (type[Grid]): The concrete grid class.
        state (tuple[Any, ...]): The grid's :meth:`~Grid._state`.
        cell_tags (CellTags | None): The grid's cell-tag registry, if it had created one.
        facet_tags (FacetTags | None): The grid's facet-tag registry, if it had created one.

    Returns:
        Grid: The rebuilt grid.
    """
    grid = cls._from_state(state)
    grid._cell_tags = cell_tags
    grid._facet_tags = facet_tags
    return grid


class GridRestriction(NamedTuple):
    """Result of :meth:`Grid.restrict`: a windowed sub-grid plus index maps.

    A :class:`typing.NamedTuple` returned by :meth:`Grid.restrict`:

    - ``grid`` -- the windowed sub-grid: the smallest structured grid containing
      every requested cell. For structured implementations (e.g.
      :class:`TensorProductGrid`) this is the same concrete type as the grid
      that produced it.
    - ``local_to_global_cell`` -- shape ``(grid.num_cells,)`` map from each
      sub-grid cell id to its id in the original grid, in the sub-grid's own
      cell-id order. Read-only.
    - ``in_subset`` -- shape ``(grid.num_cells,)`` boolean mask: ``True`` for
      cells that were in the requested ``cell_ids``, ``False`` for bounding-box
      fill cells (present only when the request was non-convex). Read-only.
    """

    grid: Grid
    local_to_global_cell: npt.NDArray[np.int64]
    in_subset: npt.NDArray[np.bool_]


__all__ = ["Grid", "GridRestriction"]
