"""Truncated hierarchical B-spline spaces (THB-splines).

This module defines :class:`THBSplineSpace`, a hierarchical spline space built on
a :class:`pantr.grid.HierarchicalGrid`.  It follows the G+Smo *self-evaluating*
model: per level it stores the Kraft selection of active tensor-product B-splines,
and a coefficient vector only for truncated functions.  Untruncated functions are
plain tensor-product B-splines.  Both the truncated (THB, default) and non-truncated
(HB) bases are supported via the ``truncate`` flag.

The space is built, evaluated, refined and prolonged in the C++ library
(``pantr::thb::THBSplineSpace``, ``include/pantr/thb/``). The class holds the compiled value
(``_h``), the Python objects it was built from, and the snapshot of the grid object that tells
whether that grid has changed since.

Main exports:

- :class:`THBSplineSpace`: hierarchical B-spline space on a
  :class:`~pantr.grid.HierarchicalGrid`.
"""

from __future__ import annotations

import copy
import operator
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from scipy import sparse

from .._ext import ext
from ..grid import HierarchicalGrid, hierarchical_grid, tensor_product_grid
from ._bspline_space_1d import BsplineSpace1D
from ._bspline_space_nd import BsplineSpace

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

_cpp: Any = ext.bspline
"""The compiled hierarchical spaces (private; untyped: one class per dtype)."""

_F32 = np.dtype(np.float32)

_MAX_COMPILED_INT = 2**62
"""Bound that keeps a Python int inside the compiled call's 64-bit integers.

An integer argument beyond it is far outside every range the compiled call checks, so clamping it
keeps that call's own range error where an overflow would have been another exception.
"""


def _clamp_int(value: int) -> int:
    """Clamp a Python int to the range the compiled call takes.

    Args:
        value (int): The integer.

    Returns:
        int: ``value``, or the nearest bound of ``[-2**62, 2**62]``.
    """
    return max(-_MAX_COMPILED_INT, min(_MAX_COMPILED_INT, value))


def _check_out_array(
    out: npt.NDArray[np.float64] | npt.NDArray[np.int64],
    shape: tuple[int, ...],
    dtype: npt.DTypeLike,
    name: str,
) -> None:
    """Validate an output array's shape, dtype, and writeability.

    Args:
        out (npt.NDArray[np.float64] | npt.NDArray[np.int64]): The output array.
        shape (tuple[int, ...]): The required shape.
        dtype (npt.DTypeLike): The required dtype.
        name (str): The parameter name, used in error messages.

    Raises:
        ValueError: If ``out`` has the wrong shape or dtype, or is not writeable.
    """
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {out.shape}.")
    if out.dtype != dtype:
        raise ValueError(f"{name} must have dtype {np.dtype(dtype).name}; got {out.dtype}.")
    if not out.flags.writeable:
        raise ValueError(f"{name} must be writeable.")


def _derivative_orders(orders: int | Sequence[int], dim: int) -> tuple[int, ...]:
    """Normalize derivative orders to one integer per direction.

    A scalar is broadcast to every direction; a sequence is converted entry by entry. The length
    and the sign are the compiled call's to check.

    Args:
        orders (int | Sequence[int]): The orders as the caller gave them.
        dim (int): Number of parametric directions.

    Returns:
        tuple[int, ...]: The orders, clamped into the compiled call's integers.

    Raises:
        TypeError: If an entry has no integer value, or ``orders`` is not iterable.
    """
    if isinstance(orders, int):
        return (_clamp_int(orders),) * dim
    return tuple(_clamp_int(int(o)) for o in orders)


def _new_space(
    h: Any,  # noqa: ANN401
    root_space: BsplineSpace,
    grid: HierarchicalGrid,
    truncate: bool,
    regularity: tuple[int | None, ...],
) -> THBSplineSpace:
    """Wrap a compiled space with the grid object that is its own grid.

    Args:
        h (Any): The compiled space.
        root_space (BsplineSpace): The level-0 space.
        grid (HierarchicalGrid): The wrapper of ``h``'s grid.
        truncate (bool): Whether the basis is truncated.
        regularity (tuple[int | None, ...]): The per-direction regularity.

    Returns:
        THBSplineSpace: The wrapper.
    """
    space = object.__new__(THBSplineSpace)
    space._adopt(
        h,
        root_space,
        grid,
        truncate,
        regularity,
        (grid.max_level, grid.num_cells, grid.version),
    )
    return space


class THBSplineSpace:
    r"""Hierarchical B-spline space on a :class:`~pantr.grid.HierarchicalGrid`.

    Built from a root :class:`~pantr.bspline.BsplineSpace` (level 0) and a
    :class:`~pantr.grid.HierarchicalGrid` carrying the active-cell hierarchy.  The
    per-level tensor-product spaces are obtained by uniformly subdividing the root
    space according to the grid's per-direction ``factor``.  The active hierarchical
    basis is the Kraft selection :cite:p:`kraft1997hierarchical,vuong2011hierarchical`:
    a level-``l`` tensor-product B-spline is active iff its support lies in the
    level-``l`` subdomain :math:`\Omega_l` but not entirely in the finer subdomain
    :math:`\Omega_{l+1}`.

    With ``truncate=True`` (the default) the *truncated* hierarchical basis (THB) is
    built: each active function that straddles a finer-level refinement boundary has its
    components on active finer functions removed (Giannelli-Jüttler-Speleers truncation
    :cite:p:`giannelli2012thb`), restoring the partition of unity.  Only truncated
    functions store a coefficient vector (in the finest tensor-product basis their support
    reaches); untruncated functions remain plain tensor-product B-splines.  With
    ``truncate=False`` the non-truncated hierarchical basis (HB) is built.

    This space is a snapshot of the grid at construction time.  Calling
    :meth:`~pantr.grid.HierarchicalGrid.refine` on the underlying grid after
    construction invalidates this space; subsequent calls to :meth:`active_basis` or
    :meth:`tabulate_basis` will raise :class:`RuntimeError`.  Create a new
    :class:`THBSplineSpace` from the updated grid instead.

    Note:
        :meth:`active_basis` lists functions whose *untruncated* support covers a
        cell; under truncation a few of those may evaluate to exactly zero on the
        cell.  :meth:`tabulate_basis` always returns the correct (possibly zero)
        values.

    References:
        Adaptive isogeometric algorithms for hierarchical splines
        :cite:p:`garau2018algorithms`.  Per-element multi-level Bézier extraction
        (used for element assembly and visualization) is provided by
        :class:`~pantr.bspline.MultiLevelExtraction`, following
        :cite:t:`dangella2018multilevel`.

    Attributes:
        _h (Any): The compiled space (``THBSplineSpaceF64`` or ``THBSplineSpaceF32``). It owns
            the grid it was built on, a value that does not follow later changes of ``_grid``.
        _root_space (BsplineSpace): The level-0 tensor-product space, as given.
        _grid (HierarchicalGrid): The active-cell hierarchy (snapshot reference), as given.
        _truncate (bool): Whether the truncated (THB) basis is used; ``False`` for
            the plain hierarchical (HB) basis.
        _regularity (tuple[int | None, ...]): Per-direction continuity used when
            subdividing to build finer levels.
        _grid_snapshot (tuple[int, int, int]): ``(max_level, num_cells, version)``
            captured at construction; used to detect post-construction grid
            mutations (the grid's :attr:`~pantr.grid.HierarchicalGrid.version`
            counter catches mutations the other two cannot distinguish).
        _num_levels (int): The number of levels, read from ``_h`` once.
        _num_total_basis (int): The number of active functions, read from ``_h`` once.
        _level_cache (dict[int, BsplineSpace]): The level spaces built so far, by level.
    """

    __slots__ = (
        "_grid",
        "_grid_snapshot",
        "_h",
        "_level_cache",
        "_num_levels",
        "_num_total_basis",
        "_regularity",
        "_root_space",
        "_truncate",
    )

    def __init__(
        self,
        root_space: BsplineSpace,
        grid: HierarchicalGrid,
        *,
        truncate: bool = True,
        regularity: int | Sequence[int | None] | None = None,
    ) -> None:
        """Create a hierarchical B-spline space.

        Args:
            root_space (BsplineSpace): The level-0 tensor-product B-spline space.
            grid (HierarchicalGrid): Hierarchical grid whose root knot-span grid
                matches ``root_space``.
            truncate (bool): If ``True`` (default), build the truncated (THB) basis;
                if ``False``, build the non-truncated hierarchical (HB) basis.
            regularity (int | Sequence[int | None] | None): Per-direction continuity
                at the knots inserted when subdividing to finer levels.  A scalar is
                broadcast to every axis; ``None`` (default) uses maximal smoothness.
                Each non-``None`` entry must satisfy ``-1 <= regularity[k] < degree[k]``.

        Raises:
            TypeError: If ``root_space`` is not a :class:`~pantr.bspline.BsplineSpace`
                or ``grid`` is not a :class:`~pantr.grid.HierarchicalGrid`.
            ValueError: If ``grid`` and ``root_space`` disagree on dimension or on
                the root knot-span grid, if ``regularity`` has the wrong length, or
                if any per-direction regularity value is out of range.
        """
        if not isinstance(root_space, BsplineSpace):
            raise TypeError(
                f"root_space must be a BsplineSpace; got {type(root_space).__name__!r}."
            )
        if not isinstance(grid, HierarchicalGrid):
            raise TypeError(f"grid must be a HierarchicalGrid; got {type(grid).__name__!r}.")
        if root_space._h is None:
            # A space with no direction has no compiled value, and a grid has at least one.
            raise ValueError(f"grid.ndim ({grid.ndim}) must equal root_space.dim (0).")
        if regularity is None or isinstance(regularity, int):
            reg: tuple[int | None, ...] = (regularity,) * root_space.dim
        else:
            reg = tuple(regularity)

        cls = _cpp.THBSplineSpaceF32 if root_space.dtype == _F32 else _cpp.THBSplineSpaceF64
        h = cls(
            root_space._h,
            grid._h,
            bool(truncate),
            [None if r is None else _clamp_int(int(r)) for r in reg],
        )
        snapshot = (grid.max_level, grid.num_cells, grid.version)
        self._adopt(h, root_space, grid, truncate, reg, snapshot)

    def _adopt(  # noqa: PLR0913
        self,
        h: Any,  # noqa: ANN401
        root_space: BsplineSpace,
        grid: HierarchicalGrid,
        truncate: bool,
        regularity: tuple[int | None, ...],
        snapshot: tuple[int, int, int],
    ) -> None:
        """Hold a compiled space and the Python objects it relates to, and cache its counts.

        Args:
            h (Any): The compiled space.
            root_space (BsplineSpace): The level-0 space.
            grid (HierarchicalGrid): The grid object the space reports as its own.
            truncate (bool): Whether the basis is truncated.
            regularity (tuple[int | None, ...]): The per-direction regularity.
            snapshot (tuple[int, int, int]): ``(max_level, num_cells, version)`` of ``grid`` that
                the compiled space corresponds to.
        """
        self._h = h
        self._root_space = root_space
        self._grid = grid
        self._truncate = truncate
        self._regularity = regularity
        self._grid_snapshot = snapshot
        self._num_levels = int(h.num_levels)
        self._num_total_basis = int(h.num_total_basis)
        self._level_cache: dict[int, BsplineSpace] = {}

    def _derive(self, h: Any) -> THBSplineSpace:  # noqa: ANN401
        """Wrap a compiled space derived from this one (refined or coarsened).

        The new space has the root, truncation and regularity of this one and a new grid object
        for the grid the compiled space owns.

        Args:
            h (Any): The compiled space.

        Returns:
            THBSplineSpace: The wrapper.
        """
        grid = HierarchicalGrid._from_handle(h.grid)
        return _new_space(h, self._root_space, grid, self._truncate, self._regularity)

    @classmethod
    def _from_state(  # noqa: PLR0913
        cls,
        root_space: BsplineSpace,
        own_grid: HierarchicalGrid,
        grid: HierarchicalGrid,
        truncate: bool,
        regularity: tuple[int | None, ...],
        snapshot: tuple[int, int, int],
    ) -> THBSplineSpace:
        """Rebuild a pickled space, as stale as it was.

        The compiled space is rebuilt on ``own_grid``, the grid it owned when it was pickled; the
        space reports ``grid``, the grid object it was built from, and the snapshot it had, so a
        space whose grid had changed unpickles as stale.

        Args:
            root_space (BsplineSpace): The level-0 space.
            own_grid (HierarchicalGrid): The grid the compiled space was built on.
            grid (HierarchicalGrid): The grid object the space reported.
            truncate (bool): Whether the basis is truncated.
            regularity (tuple[int | None, ...]): The per-direction regularity.
            snapshot (tuple[int, int, int]): The ``(max_level, num_cells, version)`` snapshot.

        Returns:
            THBSplineSpace: The rebuilt space (a subclass's extra attributes are restored by
            ``__setstate__``).
        """
        h_cls = _cpp.THBSplineSpaceF32 if root_space.dtype == _F32 else _cpp.THBSplineSpaceF64
        h = h_cls(
            root_space._h,
            own_grid._h,
            bool(truncate),
            [None if r is None else _clamp_int(int(r)) for r in regularity],
        )
        space = cls.__new__(cls)
        space._adopt(h, root_space, grid, truncate, regularity, snapshot)
        return space

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle through a state constructor, with a subclass's own attributes as state.

        The state is the root space, the grid the compiled space owns, the grid object the space
        reports, the truncation, the regularity and the snapshot of the grid, so that a stale
        space unpickles stale.

        Returns:
            tuple[Any, ...]: The rebuilding function, its arguments and the extra attributes.
        """
        return (
            type(self)._from_state,
            (
                self._root_space,
                HierarchicalGrid._from_handle(self._h.grid),
                self._grid,
                self._truncate,
                self._regularity,
                self._grid_snapshot,
            ),
            dict(self.__dict__) if hasattr(self, "__dict__") else None,
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a subclass's attributes.

        Args:
            state (dict[str, Any]): The pickled extra attributes.
        """
        self.__dict__.update(state)

    def __copy__(self) -> THBSplineSpace:
        """Copy: the new object shares the compiled value, the grid and the root space.

        Returns:
            THBSplineSpace: The copy, of the same class, with the same attributes.
        """
        new = object.__new__(type(self))
        new._adopt(
            self._h,
            self._root_space,
            self._grid,
            self._truncate,
            self._regularity,
            self._grid_snapshot,
        )
        new._level_cache = self._level_cache
        if hasattr(self, "__dict__"):
            new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo: dict[int, Any]) -> THBSplineSpace:
        """Deep copy: shares the compiled value, deep-copies the grid and the root space.

        The grid is copied through ``memo``, so a copied space and a copied function of it keep
        sharing one grid, and a copy is stale exactly when the original was.

        Args:
            memo (dict[int, Any]): The deep-copy memo.

        Returns:
            THBSplineSpace: The copy.
        """
        new = object.__new__(type(self))
        memo[id(self)] = new
        new._adopt(
            self._h,
            copy.deepcopy(self._root_space, memo),
            copy.deepcopy(self._grid, memo),
            self._truncate,
            self._regularity,
            self._grid_snapshot,
        )
        if hasattr(self, "__dict__"):
            for key, value in self.__dict__.items():
                new.__dict__[key] = copy.deepcopy(value, memo)
        return new

    # ------------------------------------------------------------------
    # Staleness
    # ------------------------------------------------------------------

    def _check_not_stale(self) -> None:
        """Raise if the grid has been modified since this space was constructed.

        Raises:
            RuntimeError: If the grid's ``max_level``, ``num_cells``, or mutation
                ``version`` differs from the snapshot taken at construction.
        """
        current = (self._grid.max_level, self._grid.num_cells, self._grid.version)
        if current != self._grid_snapshot:
            raise RuntimeError(
                "THBSplineSpace is stale: the underlying HierarchicalGrid has been modified "
                "after construction. Create a new THBSplineSpace from the updated grid."
            )

    def _cell_id(self, cid: int) -> int:
        """Convert and check a cell id as the per-cell accessors always have.

        The previous implementation memoized the active functions of a cell under ``cid`` itself,
        compared it with the cell count and converted it with ``int``: an array was refused as
        unhashable, a float was truncated, and an object that only converts to an integer (a
        string, an object with ``__index__``) was refused by the comparison. The check is made
        here, with the space's cell count, so that the message names ``cid`` as it was given.

        Args:
            cid (int): The cell id as the caller gave it.

        Returns:
            int: The cell id as a Python int.

        Raises:
            TypeError: If ``cid`` is unhashable (an array), has no integer value, or does not
                compare with integers.
            IndexError: If ``cid`` is out of range.
        """
        hash(cid)
        value = int(cid)
        num_cells = self._grid_snapshot[1]
        if not 0 <= cid < num_cells:
            raise IndexError(f"cell id {cid!r} is out of range [0, {num_cells}).")
        return value

    @property
    def _func_offset(self) -> npt.NDArray[np.int64]:
        """Get the global dof of each level's first active function.

        Returns:
            npt.NDArray[np.int64]: Read-only, ``num_levels + 1`` cumulative active-function
            counts: level ``l`` owns the dofs ``[_func_offset[l], _func_offset[l + 1])``.
        """
        offset: npt.NDArray[np.int64] = self._h.func_offset
        return offset

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def grid(self) -> HierarchicalGrid:
        """Get the underlying hierarchical grid.

        Returns:
            HierarchicalGrid: The active-cell hierarchy this space is built on.
        """
        return self._grid

    @property
    def root_space(self) -> BsplineSpace:
        """Get the level-0 tensor-product space.

        Returns:
            BsplineSpace: The root B-spline space.
        """
        return self._root_space

    @property
    def dim(self) -> int:
        """Get the parametric dimension.

        Returns:
            int: Number of parametric directions.
        """
        return self._root_space.dim

    @property
    def degrees(self) -> tuple[int, ...]:
        """Get the per-direction polynomial degrees.

        Returns:
            tuple[int, ...]: Degree per direction (the same at every level).
        """
        return self._root_space.degrees

    @property
    def num_levels(self) -> int:
        """Get the number of hierarchy levels at construction time.

        Returns:
            int: Number of levels; stable even if the grid is later refined.
        """
        return self._num_levels

    @property
    def truncate(self) -> bool:
        """Get whether the hierarchical basis is truncated.

        Returns:
            bool: ``True`` for the truncated (THB) basis, ``False`` for the plain
            hierarchical (HB) basis.
        """
        return self._truncate

    @property
    def num_total_basis(self) -> int:
        """Get the total number of active hierarchical basis functions.

        Mirrors :attr:`~pantr.bspline.BsplineSpace.num_total_basis` (the hierarchical
        basis is not tensor-product, so there is no per-direction ``num_basis``).

        Returns:
            int: Total active-function count across all levels.
        """
        return self._num_total_basis

    @property
    def num_basis_per_level(self) -> tuple[int, ...]:
        """Get the number of active basis functions at each level.

        Returns:
            tuple[int, ...]: Active-function count per level.
        """
        return tuple(int(n) for n in self._h.num_basis_per_level)

    @property
    def domain(self) -> npt.NDArray[np.float32 | np.float64]:
        """Get the parametric domain bounds.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Shape ``(dim, 2)`` ``[lo, hi]`` per
            direction (from the root space).
        """
        return self._root_space.domain

    @property
    def dtype(self) -> npt.DTypeLike:
        """Get the floating-point dtype of the space.

        Returns:
            npt.DTypeLike: Always ``numpy.float64`` (THB evaluation is float64).
        """
        return np.float64

    @property
    def tolerance(self) -> float:
        """Get the numerical tolerance.

        Returns:
            float: The root space's tolerance.
        """
        return self._root_space.tolerance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _check_level(self, level: int) -> int:
        """Check a hierarchy level against the level count.

        Args:
            level (int): The level as the caller gave it.

        Returns:
            int: The level as a Python int.

        Raises:
            ValueError: If ``level`` is out of range.
            TypeError: If ``level`` is in range but not an integer.
        """
        if not (0 <= level < self._num_levels):
            raise ValueError(f"level must be in [0, {self._num_levels - 1}]; got {level!r}.")
        return operator.index(level)

    def level_space(self, level: int) -> BsplineSpace:
        """Return the tensor-product space at ``level``.

        Returns a construction-time snapshot; not affected by subsequent grid
        mutations (no stale check is performed).

        Args:
            level (int): Hierarchy level in ``[0, num_levels)``.

        Returns:
            BsplineSpace: The root space subdivided to ``level``.

        Raises:
            ValueError: If ``level`` is out of range.
        """
        lvl = self._check_level(level)
        cached = self._level_cache.get(lvl)
        if cached is None:
            if lvl == 0:
                cached = self._root_space
            else:
                cached = BsplineSpace(
                    [BsplineSpace1D._from_handle(h) for h in self._h.level_space_1d(lvl)]
                )
            self._level_cache[lvl] = cached
        return cached

    def active_function_indices(self, level: int) -> npt.NDArray[np.int64]:
        """Return the flat indices of the active functions at ``level``.

        Returns a construction-time snapshot; not affected by subsequent grid
        mutations (no stale check is performed).

        Args:
            level (int): Hierarchy level in ``[0, num_levels)``.

        Returns:
            npt.NDArray[np.int64]: Sorted flat (C-order) level-``level`` function
            indices selected by the Kraft rule.  A fresh copy is returned.

        Raises:
            ValueError: If ``level`` is out of range.
        """
        indices: npt.NDArray[np.int64] = self._h.active_function_indices(self._check_level(level))
        return indices

    def active_basis(self, cid: int) -> npt.NDArray[np.int64]:
        """Return the global dofs of the active functions whose support intersects cell ``cid``.

        Args:
            cid (int): Active cell flat id in ``[0, grid.num_cells)``.

        Returns:
            npt.NDArray[np.int64]: Sorted global hierarchical-dof indices of the
            functions whose support intersects cell ``cid``.

        Raises:
            IndexError: If ``cid`` is out of range ``[0, grid.num_cells)``.
            RuntimeError: If the grid has been modified since construction.
        """
        self._check_not_stale()
        dofs: npt.NDArray[np.int64] = self._h.active_basis(self._cell_id(cid))
        return dofs

    def max_active_per_cell(self) -> int:
        """Return the largest number of active functions on any single cell.

        The width a fixed-size dofmap needs: ``max(active_basis(cid).size)`` over every
        active cell. On an unrefined space this is ``prod(degree + 1)``; near a level
        interface a cell also sees the coarser functions overlapping it, so the count
        grows there.

        Computed once and cached, since the space is a construction-time snapshot.

        Returns:
            int: Maximum active-function count over all cells (``>= 1``).

        Raises:
            RuntimeError: If the grid has been modified since construction.

        Note:
            Counts exactly what :meth:`active_basis` returns, which selects on
            tensor-product support. Truncation can only annihilate a function on a cell
            it supports, never add one, so with ``truncate=True`` this is an upper bound
            on the number of *non-zero* functions -- which is what a fixed-width dofmap
            wants. The value is therefore the same for the THB and HB bases.

            Visits every cell, so the first call populates the per-cell table of the
            whole grid -- the same table :meth:`active_basis` reads, but built in full.
        """
        self._check_not_stale()
        return int(self._h.max_active_per_cell())

    def restrict(self, cell_ids: npt.ArrayLike) -> THBSplineSpaceRestriction:
        """Return the windowed sub-space over a subset of active cells.

        Windows this space to the root-cell-aligned bounding box of ``cell_ids``: the
        hierarchical grid is restricted (:meth:`pantr.grid.HierarchicalGrid.restrict`),
        the root space is windowed (:meth:`pantr.bspline.BsplineSpace.restrict`), and a
        new :class:`THBSplineSpace` is rebuilt on the sub-grid (re-running the Kraft
        active-function selection and truncation).

        Unlike the tensor-product :meth:`pantr.bspline.BsplineSpace.restrict`, the
        windowed THB basis equals the global one only over the **interior** cells --
        those whose entire (cross-level) function-support-closure lies inside the
        window -- because Kraft selection and truncation depend on the subdomain near
        the window boundary. Callers make the cells they care about interior by padding
        ``cell_ids`` with a support-closure halo.

        Args:
            cell_ids (npt.ArrayLike): Active cell flat ids to span; duplicates ignored.

        Returns:
            THBSplineSpaceRestriction: The windowed :class:`THBSplineSpace` and a
            read-only ``local_to_global_dof`` map; entry ``d`` is the global
            hierarchical dof of local dof ``d`` when the local function matches a
            globally-active function of the same level and multi-index, else ``-1``.
            Values are exact over interior cells; functions near the window boundary
            may map to ``-1``.

        Raises:
            ValueError: If ``cell_ids`` is empty.
            TypeError: If ``cell_ids`` is not integer-valued.
            IndexError: If any cell id is out of range ``[0, grid.num_cells)``.
        """
        self._check_not_stale()
        ids = np.asarray(cell_ids).ravel()
        # An empty request reaches the compiled space first, which refuses it.
        if ids.size > 0 and not np.issubdtype(ids.dtype, np.integer):
            raise TypeError(f"restrict: cell_ids must be integer-valued; got dtype {ids.dtype}.")
        h, local_to_global_dof, local_to_global_cell = self._h.restrict(
            np.ascontiguousarray(ids, dtype=np.int64)
        )
        grid = HierarchicalGrid._from_handle(h.grid)
        root = BsplineSpace([BsplineSpace1D._from_handle(s) for s in h.level_space_1d(0)])
        sub_space = _new_space(h, root, grid, self._truncate, self._regularity)
        local_to_global_dof.flags.writeable = False
        local_to_global_cell.flags.writeable = False
        return THBSplineSpaceRestriction(sub_space, local_to_global_dof, local_to_global_cell)

    def _tabulate(
        self,
        cid: int,
        pts: npt.ArrayLike,
        orders: tuple[int, ...],
        out_basis: npt.NDArray[np.float64] | None,
        out_dofs: npt.NDArray[np.int64] | None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Evaluate the active functions' ``orders`` mixed partial on cell ``cid``.

        Shared implementation for :meth:`tabulate_basis` (``orders`` all zero) and
        :meth:`tabulate_basis_derivatives`.

        Args:
            cid (int): Active cell flat id in ``[0, grid.num_cells)``.
            pts (npt.ArrayLike): Parametric points of shape ``(..., dim)`` lying in
                cell ``cid``.  Per axis, ``8 * eps * max(|lo|, |hi|, hi - lo)`` of slack
                is allowed at the cell boundary; points further outside raise
                :class:`ValueError`.
            orders (tuple[int, ...]): Per-direction derivative orders.
            out_basis (npt.NDArray[np.float64] | None): Optional output array of shape
                ``(..., K)`` with ``K = active_basis(cid).size``.  Allocated when
                ``None``.
            out_dofs (npt.NDArray[np.int64] | None): Optional output array of shape
                ``(K,)`` for the global dofs.  Allocated when ``None``.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: ``(values, dofs)``
            of shapes ``(..., K)`` and ``(K,)``.

        Raises:
            IndexError: If ``cid`` is out of range ``[0, grid.num_cells)``.
            ValueError: If ``pts`` does not have trailing dimension ``dim``, if any
                point lies outside cell ``cid``, or if ``out_basis``/``out_dofs`` has
                the wrong shape, dtype, or is not writeable.
            RuntimeError: If the grid has been modified since construction.
        """
        self._check_not_stale()
        cell = self._cell_id(cid)

        dim = self.dim
        pts_arr = np.asarray(pts, dtype=np.float64)
        if pts_arr.ndim == 0 or pts_arr.shape[-1] != dim:
            raise ValueError(f"pts must have trailing dimension {dim}; got shape {pts_arr.shape}.")
        lead = pts_arr.shape[:-1]
        flat_pts = np.ascontiguousarray(pts_arr.reshape(-1, dim))

        # The compiled space checks the cell membership of the points. The outputs are checked
        # after it, so an error in the points is raised first, and a valid output is filled from
        # the result (a non-contiguous one as well).
        values, dofs = self._h.tabulate_basis(cell, flat_pts, list(orders))
        out_shape = (*lead, values.shape[1])
        result = values.reshape(out_shape)
        if out_basis is not None:
            _check_out_array(out_basis, out_shape, np.float64, "out_basis")
            out_basis[...] = result
            result = out_basis
        if out_dofs is not None:
            _check_out_array(out_dofs, (dofs.shape[0],), np.int64, "out_dofs")
            out_dofs[...] = dofs
            dofs = out_dofs
        return result, dofs

    def tabulate_basis(
        self,
        cid: int,
        pts: npt.ArrayLike,
        out_basis: npt.NDArray[np.float64] | None = None,
        out_dofs: npt.NDArray[np.int64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Evaluate the active hierarchical functions on cell ``cid`` at ``pts``.

        Untruncated functions are a single tensor-product B-spline (the product of
        their 1D B-spline values).  Truncated functions are evaluated from their
        stored coefficients in the finest tensor-product basis their support reaches.
        The returned columns are ordered as ``dofs`` (the sorted global dofs, equal to
        :meth:`active_basis`); a listed truncated function may evaluate to exactly zero
        on the cell.  Mirrors the ``(basis, first_basis)`` two-return of
        :meth:`~pantr.bspline.BsplineSpace.tabulate_basis`.

        Args:
            cid (int): Active cell flat id in ``[0, grid.num_cells)``.
            pts (npt.ArrayLike): Parametric points of shape ``(..., dim)`` lying in
                cell ``cid``.  Per axis, ``8 * eps * max(|lo|, |hi|, hi - lo)`` of
                slack is allowed at the boundary; points further outside raise
                :class:`ValueError`.
            out_basis (npt.NDArray[np.float64] | None): Optional output array of shape
                ``(..., K)`` with ``K = active_basis(cid).size``.  Allocated when
                ``None``.
            out_dofs (npt.NDArray[np.int64] | None): Optional output array of shape
                ``(K,)`` for the dofs.  Allocated when ``None``.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: ``(values, dofs)`` of
            shapes ``(..., K)`` and ``(K,)``.

        Raises:
            IndexError: If ``cid`` is out of range ``[0, grid.num_cells)``.
            ValueError: If ``pts`` does not have trailing dimension ``dim``, if any
                point lies outside the bounds of cell ``cid``, or if ``out_basis`` /
                ``out_dofs`` has the wrong shape, dtype, or is not writeable.
            RuntimeError: If the grid has been modified since construction.
        """
        return self._tabulate(cid, pts, (0,) * self.dim, out_basis, out_dofs)

    def tabulate_basis_derivatives(
        self,
        cid: int,
        pts: npt.ArrayLike,
        orders: int | Sequence[int],
        out_basis: npt.NDArray[np.float64] | None = None,
        out_dofs: npt.NDArray[np.int64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        r"""Evaluate a mixed partial derivative of the active functions on cell ``cid``.

        Computes the single mixed partial :math:`\partial^{orders}` of each active
        hierarchical function, where ``orders[k]`` is the derivative order in
        parametric direction ``k`` (derivatives are with respect to the parametric
        coordinates).  Untruncated functions differentiate as a tensor product of 1D
        B-spline derivatives; truncated functions apply their stored coefficients to
        the B-spline derivatives at their representation level.  The returned columns
        are ordered as ``dofs`` (the sorted global dofs, equal to :meth:`active_basis`).

        Args:
            cid (int): Active cell flat id in ``[0, grid.num_cells)``.
            pts (npt.ArrayLike): Parametric points of shape ``(..., dim)`` lying in
                cell ``cid``.  Per axis, ``8 * eps * max(|lo|, |hi|, hi - lo)`` of
                slack is allowed at the boundary; points further outside raise
                :class:`ValueError`.
            orders (int | Sequence[int]): Per-direction derivative orders.  A scalar
                is broadcast to every direction.  Each entry must be ``>= 0``; orders
                exceeding the degree yield zero.
            out_basis (npt.NDArray[np.float64] | None): Optional output array of shape
                ``(..., K)`` with ``K = active_basis(cid).size``.  Allocated when
                ``None``.
            out_dofs (npt.NDArray[np.int64] | None): Optional output array of shape
                ``(K,)`` for the dofs.  Allocated when ``None``.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: ``(values, dofs)`` of
            shapes ``(..., K)`` and ``(K,)``.

        Raises:
            IndexError: If ``cid`` is out of range ``[0, grid.num_cells)``.
            ValueError: If ``orders`` has the wrong length or a negative entry, if
                ``pts`` does not have trailing dimension ``dim``, if any point lies
                outside cell ``cid``, or if ``out_basis`` / ``out_dofs`` has the wrong
                shape, dtype, or is not writeable.
            RuntimeError: If the grid has been modified since construction.
        """
        return self._tabulate(cid, pts, _derivative_orders(orders, self.dim), out_basis, out_dofs)

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    @staticmethod
    def _admissible_class(admissible_class: int | None) -> int | None:
        """Check and convert the ``admissible_class`` argument shared by refine and coarsen.

        The class is compared as given, so the message names it as it was given and a value that
        does not order against integers is refused with a ``TypeError``.

        Args:
            admissible_class (int | None): The class as the caller gave it.

        Returns:
            int | None: The class as a Python int, or ``None``.

        Raises:
            ValueError: If ``admissible_class`` is below ``2``.
        """
        if admissible_class is None:
            return None
        if admissible_class < 2:  # noqa: PLR2004
            raise ValueError(
                f"admissible_class must be an integer >= 2 or None; got {admissible_class!r}."
            )
        return _clamp_int(int(admissible_class))

    def refine(
        self,
        cell_ids: npt.ArrayLike,
        *,
        admissible_class: int | None = 2,
    ) -> THBSplineSpace:
        """Return a new space with the marked cells refined.

        This method does not mutate ``self`` or its grid: a fresh grid is refined
        and a new :class:`THBSplineSpace` is built; ``self`` and its grid are unchanged.

        With ``admissible_class=m`` (the default ``m=2``) the refinement is graded so
        the resulting mesh is admissible of class ``m`` (the truncated functions
        acting on any cell span at most ``m`` successive levels), following the
        recursive refinement-neighborhood algorithm of Carraturo et al. (2019).  This
        assumes the current mesh is already admissible of class ``m`` (true for the
        root and for any mesh built via graded :meth:`refine`).  With
        ``admissible_class=None`` exactly the marked cells are refined (no grading).

        Args:
            cell_ids (npt.ArrayLike): Flat ids of active cells to refine.
            admissible_class (int | None): Admissibility class ``m >= 2`` to maintain,
                or ``None`` for ungraded refinement.  Defaults to ``2``.

        Returns:
            THBSplineSpace: A new space on the refined grid (same ``root_space``,
            ``truncate``, and ``regularity``).

        Raises:
            IndexError: If any id is outside ``[0, grid.num_cells)``.
            ValueError: If ``admissible_class`` is an integer ``< 2``.
            RuntimeError: If the grid has been modified since construction.
        """
        self._check_not_stale()
        adm = self._admissible_class(admissible_class)
        ids = np.ascontiguousarray(np.asarray(cell_ids, dtype=np.int64).ravel())
        return self._derive(self._h.refine(ids, adm))

    def refine_region(
        self,
        level: int,
        lo: Sequence[int],
        hi: Sequence[int],
        *,
        admissible_class: int | None = 2,
    ) -> THBSplineSpace:
        """Return a new space with the active cells in a rectangular region refined.

        The region is the integer cell-index box ``[lo, hi)`` at ``level`` (in
        level-``level`` coordinates), matching the convention of
        :meth:`pantr.grid.HierarchicalGrid.refine`.  Only the currently-active leaf
        cells inside the box are refined; the rest of the box (already refined, or
        not present at ``level``) is ignored.  If the box contains no active leaf
        cells, the call is a no-op and returns a space equivalent to ``self``.  This
        is the region-based counterpart of :meth:`refine`, which marks individual
        cells by flat id.

        Like :meth:`refine`, this does not mutate ``self`` or its grid: a fresh grid
        is refined and a new :class:`THBSplineSpace` is returned.  Calls chain, so
        successive regions refine progressively (graded by default).

        Args:
            level (int): Level at which the box lives.  Must satisfy
                ``0 <= level <= grid.max_level``.
            lo (Sequence[int]): Per-direction start index (inclusive), in
                level-``level`` coordinates.
            hi (Sequence[int]): Per-direction end index (exclusive), in
                level-``level`` coordinates.
            admissible_class (int | None): Admissibility class ``m >= 2`` to maintain
                (graded refinement), or ``None`` for ungraded refinement.  Defaults
                to ``2``.  See :meth:`refine`.

        Returns:
            THBSplineSpace: A new space on the refined grid (same ``root_space``,
            ``truncate``, and ``regularity``).

        Raises:
            ValueError: If ``admissible_class`` is an integer ``< 2``, ``level`` is
                out of range, ``lo``/``hi`` have the wrong length, any
                ``lo[k] >= hi[k]``, or any part of ``[lo, hi)`` lies outside the
                level domain.
            RuntimeError: If the grid has been modified since construction.
        """
        self._check_not_stale()
        adm = self._admissible_class(admissible_class)
        h = self._h.refine_region(
            _clamp_int(operator.index(level)),
            [_clamp_int(int(x)) for x in lo],
            [_clamp_int(int(x)) for x in hi],
            adm,
        )
        return self._derive(h)

    def coarsen(
        self,
        cell_ids: npt.ArrayLike,
        *,
        admissible_class: int | None = 2,
    ) -> THBSplineSpace:
        """Return a new space with the marked cells coarsened away.

        A parent cell is reactivated (its children removed) only when **all** of its
        children are marked active leaves, mirroring the coarsening algorithm of
        Carraturo et al. (2019, Alg. 5).  That rule is
        :meth:`~pantr.grid.HierarchicalGrid.coarsen_cells`, which this method drives one
        parent at a time so the admissibility guard below can veto a parent without
        affecting the rest.  With ``admissible_class=None`` this is the exact inverse of
        :meth:`refine`: ``space.refine(cells).coarsen(children_of(cells))`` recovers
        ``space``.  With ``admissible_class=m`` the guard may suppress some coarsenings,
        so the recovery holds only when the guard permits them all.

        With ``admissible_class=m`` (the default ``m=2``) a parent is reactivated only
        if its coarsening neighborhood (Def. 3.5) is empty, so the resulting mesh stays
        admissible of class ``m``.  With ``admissible_class=None`` that guard is skipped.

        The space is immutable: a fresh grid is coarsened and a new
        :class:`THBSplineSpace` is built; ``self`` and its grid are unchanged.
        An empty ``cell_ids`` is valid and returns an unchanged copy of the space.

        Args:
            cell_ids (npt.ArrayLike): Flat ids of active leaf cells to coarsen away.
                An empty array is valid and produces an unchanged copy.
            admissible_class (int | None): Admissibility class ``m >= 2`` to maintain,
                or ``None`` to skip the admissibility guard.  Defaults to ``2``.

        Returns:
            THBSplineSpace: A new space on the coarsened grid (same ``root_space``,
            ``truncate``, and ``regularity``).

        Raises:
            IndexError: If any id is outside ``[0, grid.num_cells)``.
            ValueError: If ``admissible_class`` is an integer ``< 2``.
            RuntimeError: If the grid has been modified since construction.
        """
        self._check_not_stale()
        adm = self._admissible_class(admissible_class)
        ids = np.ascontiguousarray(np.asarray(cell_ids, dtype=np.int64).ravel())
        return self._derive(self._h.coarsen(ids, adm))

    # ------------------------------------------------------------------
    # Prolongation and restriction
    # ------------------------------------------------------------------

    def _prolongation_csr(
        self, fine: THBSplineSpace
    ) -> tuple[int, int, npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Get the prolongation to a refinement as the compiled space's CSR arrays.

        Args:
            fine (THBSplineSpace): Candidate refinement.

        Returns:
            tuple: ``(rows, cols, indptr, indices, data)`` of the matrix.

        Raises:
            TypeError: If ``fine`` is not a :class:`THBSplineSpace`.
            ValueError: If ``fine`` is not a refinement of this space; the message lists every
                structural mismatch, or gives the residual of a column that is not reproduced.
        """
        if not isinstance(fine, THBSplineSpace):
            raise TypeError(f"fine must be a THBSplineSpace; got {type(fine).__name__!r}.")
        rows, cols, indptr, indices, data = self._h.prolongation_csr(fine._h)
        return int(rows), int(cols), indptr, indices, data

    def prolongation_to(self, fine: THBSplineSpace) -> npt.NDArray[np.float64]:
        """Return the prolongation matrix from this space to a refinement ``fine``.

        The hierarchical spaces are nested (``V_h ⊆ V_h'``), so every function of this
        (coarse) space lies in ``fine``.  The returned matrix ``P`` maps a
        coefficient vector in this space's basis to the coefficients of the **same
        function** in ``fine``'s basis: if ``u`` are coarse coefficients, ``P @ u`` are
        the fine coefficients.

        It is built column-by-column following the local two-scale construction used
        in practice (Garau & Vazquez 2018; D'Angella et al. 2018): each coarse
        function is matched against only the fine functions over its support,
        expressed in the deepest level present there (not the global finest level),
        and reproduced by a small local least-squares solve.  Functions far from the
        refinement yield trivial (identity) columns, so cost and sparsity follow the
        refined region.

        For a large refinement prefer :meth:`prolongation_to_sparse`, which computes the
        same entries with the same arithmetic but stores only the non-zeros: this dense
        variant allocates ``n_fine * n_coarse`` float64 regardless of how few columns the
        refinement actually touches.

        Args:
            fine (THBSplineSpace): A refinement of this space (same ``root_space``,
                ``factor``, ``regularity``, and ``truncate``; more levels / refined
                cells).

        Returns:
            npt.NDArray[np.float64]: Matrix ``P`` of shape
            ``(fine.num_total_basis, self.num_total_basis)``.

        Raises:
            TypeError: If ``fine`` is not a :class:`THBSplineSpace`.
            ValueError: If ``fine`` is not a refinement of this space (mismatched
                root/factor/regularity/truncation, fewer levels, or a prolongation
                residual above ``4096 * eps * (1 + max_coarse_value)``, where
                ``max_coarse_value`` is the largest coefficient of any coarse column).
        """
        rows, cols, indptr, indices, data = self._prolongation_csr(fine)
        # The stored entries are scattered as solved: an explicit or negative zero stays.
        prolongation: npt.NDArray[np.float64] = np.zeros((rows, cols), dtype=np.float64)
        row_of_entry = np.repeat(np.arange(rows, dtype=np.int64), np.diff(indptr))
        prolongation[row_of_entry, indices] = data
        return prolongation

    def prolongation_to_sparse(self, fine: THBSplineSpace) -> sparse.csr_array:
        """Return the prolongation to a refinement ``fine`` in sparse CSR form.

        The same matrix :meth:`prolongation_to` returns, entry for entry: both are the
        compiled space's stored entries, so no entry is rounded differently and
        ``toarray()`` **compares equal** to the dense result. Only the storage differs.

        **Equal as values, not bit for bit**, and the gap is exactly the sign of a zero.
        The *stored* entries are bitwise what the solves produced, this array's own
        ``data`` included; but :meth:`scipy.sparse.csr_array.toarray` densifies by adding
        into a ``+0.0``-filled buffer, and ``+0.0 + -0.0`` is ``+0.0``, so a column that
        solved to a negative zero comes back positive. ``-0.0 == 0.0``, so nothing that
        compares values can see it, and nothing downstream of a matrix-vector product can
        either. ``tests/test_thb_spline_space.py`` pins both halves.

        That storage is the whole point at scale. The dense variant is
        ``O(n_fine * n_coarse)`` -- for ``1e5`` fine by ``9e4`` coarse dofs, some 72 GB --
        while the matrix has only a handful of entries per column: a coarse function is
        matched against the fine functions overlapping its support, and away from the
        refined region that is one entry of value ``1.0``. Sparse storage is ``O(nnz)``.

        Entries are stored exactly as solved, including any that happen to be zero, since
        they are the two-scale coefficients rather than a thresholded approximation. Call
        ``eliminate_zeros()`` on the result if a caller wants them dropped.

        Args:
            fine (THBSplineSpace): A refinement of this space, on the same terms as
                :meth:`prolongation_to`.

        Returns:
            scipy.sparse.csr_array: Matrix ``P`` of shape
            ``(fine.num_total_basis, self.num_total_basis)``, ``float64``.

        Raises:
            TypeError: If ``fine`` is not a :class:`THBSplineSpace`.
            ValueError: If ``fine`` is not a refinement of this space -- the same
                mismatches and residual check as :meth:`prolongation_to`.
        """
        rows, cols, indptr, indices, data = self._prolongation_csr(fine)
        return sparse.csr_array((data, indices, indptr), shape=(rows, cols))

    def restriction_to(self, coarse: THBSplineSpace) -> npt.NDArray[np.float64]:
        """Return the restriction matrix from this space to a coarsening ``coarse``.

        ``self`` must be a refinement of ``coarse``.  The restriction is the algebraic
        pseudo-inverse of the prolongation, ``R = pinv(P)`` with
        ``P = coarse.prolongation_to(self)``.  It is assembly-free (no mass matrix) and
        satisfies ``R @ P == I``, so restricting a prolonged coarse field recovers it
        exactly; for a general fine field ``R @ u_fine`` is the least-squares
        (coefficient-space) projection onto the coarse space.

        Args:
            coarse (THBSplineSpace): A coarsening of this space (``self`` is a
                refinement of ``coarse``).

        Returns:
            npt.NDArray[np.float64]: Matrix ``R`` of shape
            ``(coarse.num_total_basis, self.num_total_basis)``.

        Raises:
            TypeError: If ``coarse`` is not a :class:`THBSplineSpace`.
            ValueError: If ``self`` is not a refinement of ``coarse``.
        """
        if not isinstance(coarse, THBSplineSpace):
            raise TypeError(f"coarse must be a THBSplineSpace; got {type(coarse)!r}.")
        prolongation = coarse.prolongation_to(self)
        restriction: npt.NDArray[np.float64] = np.asarray(
            np.linalg.pinv(prolongation), dtype=np.float64
        )
        return restriction

    def __repr__(self) -> str:
        """Return a compact string representation.

        Returns:
            str: Shows dimension, degrees, level count, active-function count, and
            truncation flag.
        """
        return (
            f"THBSplineSpace(dim={self.dim}, degrees={self.degrees}, "
            f"num_levels={self.num_levels}, num_total_basis={self._num_total_basis}, "
            f"truncate={self._truncate})"
        )


class THBSplineSpaceRestriction(NamedTuple):
    """Result of :meth:`THBSplineSpace.restrict`: a windowed THB space with its maps.

    - ``space`` -- the windowed :class:`THBSplineSpace` rebuilt on the restricted grid;
      its basis equals the global basis pointwise over interior cells (those whose
      function-support-closure lies inside the window).
    - ``local_to_global_dof`` -- read-only ``(space.num_total_basis,)`` map; entry ``d``
      is the global hierarchical dof of local dof ``d`` when the local function matches a
      globally-active function of the same level and multi-index, else ``-1`` (local
      function active in the sub-space but absent from the global active set -- arises
      near the window boundary where Kraft selection may differ). Values are exact over
      interior cells.
    - ``local_to_global_cell`` -- read-only ``(space.grid.num_cells,)`` map; entry ``c``
      is the global flat cell id of local cell ``c``. Same ordering as
      :attr:`space.grid <THBSplineSpace.grid>`.
    """

    space: THBSplineSpace
    local_to_global_dof: npt.NDArray[np.int64]
    local_to_global_cell: npt.NDArray[np.int64]


def create_thb_space(
    root: BsplineSpace,
    factor: int | Sequence[int] = 2,
    *,
    truncate: bool = True,
    regularity: int | Sequence[int | None] | None = None,
) -> THBSplineSpace:
    """Create a trivial (unrefined) THB-spline space from a B-spline space.

    Convenience factory that wraps ``root`` in a single-level
    :class:`~pantr.grid.HierarchicalGrid` (its knot-span grid, ready to subdivide by
    ``factor``) and builds the corresponding :class:`THBSplineSpace`.  The result has
    one level and its active basis coincides with that of ``root``; refine it with
    :meth:`THBSplineSpace.refine` or :meth:`THBSplineSpace.refine_region`.

    It is the ergonomic entry point for lifting an existing :class:`BsplineSpace` into
    a single-level hierarchy, leaving the two-argument :class:`THBSplineSpace`
    constructor for callers that build the hierarchical grid explicitly.

    Args:
        root (BsplineSpace): The level-0 tensor-product B-spline space.
        factor (int | Sequence[int]): Per-direction subdivision factor used when the
            space is later refined.  A scalar is broadcast to every axis.  Each entry
            must be ``>= 1``.  Defaults to ``2`` (dyadic refinement).
        truncate (bool): If ``True`` (default), build the truncated (THB) basis; if
            ``False``, build the non-truncated hierarchical (HB) basis.
        regularity (int | Sequence[int | None] | None): Per-direction continuity at
            the knots inserted when subdividing to finer levels.  See
            :class:`THBSplineSpace`.  Defaults to ``None`` (maximal smoothness).

    Returns:
        THBSplineSpace: A single-level THB space over ``root``.

    Raises:
        ValueError: If any ``factor`` entry is ``< 1``, ``factor`` has the wrong
            length, or ``regularity`` is out of range for ``root``'s degrees.

    Example:
        >>> from pantr.bspline import create_uniform_space, create_thb_space
        >>> thb = create_thb_space(create_uniform_space([2, 2], [8, 8]))
        >>> thb.num_levels
        1
        >>> thb = thb.refine_region(0, [0, 0], [4, 4])  # refine the lower-left quarter
        >>> thb.num_levels
        2
    """
    grid = hierarchical_grid(tensor_product_grid(root), factor)
    return THBSplineSpace(root, grid, truncate=truncate, regularity=regularity)
