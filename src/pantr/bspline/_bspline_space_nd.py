"""Multi-dimensional B-spline spaces using tensor products.

This module defines :class:`BsplineSpace`, which aggregates multiple
:class:`~pantr.bspline.BsplineSpace1D` objects to represent
multi-dimensional parameter domains. It handles tensor-product basis evaluation
by combining the 1D components, in the C++ library (``pantr::bspline::BsplineSpace``).
"""

from __future__ import annotations

import copy as _copy
import functools
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy import typing as npt

from .._ext import ext
from ..basis._basis_utils import _allocate_or_validate_out
from ..quad import PointsLattice
from ._bspline_cell_supports import _cell_supports_impl
from ._bspline_space_1d import _FIRST_DTYPE, _extra_state, _first_out, _kernel_buffer

if TYPE_CHECKING:
    from ._bspline_space_1d import BsplineSpace1D

_cpp: Any = ext.bspline
"""The compiled B-spline spaces (private; untyped: one class per dtype)."""

_VALUE_ATTRS_ND: frozenset[str] = frozenset(
    {
        "_spaces",
        "_h",
        "degrees",
        "tolerance",
        "num_basis",
        "num_total_basis",
        "num_intervals",
        "num_total_intervals",
        "domain",
    }
)
"""The attributes of a ``BsplineSpace`` that hold its value or are cached from it."""


class BsplineSpace:
    """A class representing a multi-dimensional B-spline space.

    This space is defined by a set of B-spline spaces, one for each dimension.

    This class provides methods to analyze B-spline properties, validate input
    parameters, compute various geometric characteristics of the spline,
    and access various properties of the B-spline.

    Attributes:
        _spaces (tuple[BsplineSpace1D, ...]): The B-spline spaces, one for each dimension,
            as given (so ``space.spaces[d] is`` the object passed in).
        _h (Any): The compiled tensor product (``BsplineSpaceF64`` or ``BsplineSpaceF32``)
            of their compiled values; None for an empty tuple of spaces.
    """

    _spaces: tuple[BsplineSpace1D, ...]
    _h: Any

    def __init__(
        self,
        spaces: Iterable[BsplineSpace1D],
    ) -> None:
        """Initialize a B-spline space object.

        Args:
            spaces (Iterable[BsplineSpace1D]): List of B-spline spaces, one for each dimension.

        Raises:
            ValueError: If the B-spline spaces have different data types.
        """
        self._spaces = tuple(spaces)

        if not all(space.dtype == self.dtype for space in self._spaces):
            raise ValueError("All B-spline spaces must have the same data type.")

        # An empty tuple is accepted, as it always was, and has no compiled value (the C++ space
        # needs a direction): its ``dtype`` raises ``IndexError`` and nothing can be tabulated.
        self._h = None
        if self._spaces:
            cls = _cpp.BsplineSpaceF32 if self.dtype == np.float32 else _cpp.BsplineSpaceF64
            self._h = cls([space._h for space in self._spaces])

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle (and copy) through the constructor, with a subclass's own attributes as state.

        Returns:
            tuple[Any, ...]: The class, its constructor argument and the other attributes.
        """
        return (type(self), (self._spaces,), _extra_state(self, _VALUE_ATTRS_ND))

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a subclass's attributes, or rebuild from a previous release's pickle.

        A previous release pickled the attribute ``_spaces`` (with whatever cached properties it
        held); such a state rebuilds the compiled value from it and drops the cached entries.

        Args:
            state (dict[str, Any]): The pickled state.
        """
        state = dict(state)
        if "_spaces" in state and "_h" not in self.__dict__:
            BsplineSpace.__init__(self, state["_spaces"])
            for key in _VALUE_ATTRS_ND:
                state.pop(key, None)
        self.__dict__.update(state)

    def __copy__(self) -> BsplineSpace:
        """Copy: the new object shares the 1D spaces and the compiled value.

        Returns:
            BsplineSpace: The copy, of the same class, with the same attributes.
        """
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo: dict[int, Any]) -> BsplineSpace:
        """Deep copy: deep-copies the 1D spaces (which share their values) and other attributes.

        Args:
            memo (dict[int, Any]): The deep-copy memo.

        Returns:
            BsplineSpace: The copy.
        """
        new = object.__new__(type(self))
        memo[id(self)] = new
        spaces = _copy.deepcopy(self._spaces, memo)
        BsplineSpace.__init__(new, spaces)
        for key, value in self.__dict__.items():
            if key not in _VALUE_ATTRS_ND:
                new.__dict__[key] = _copy.deepcopy(value, memo)
        return new

    @property
    def dim(self) -> int:
        """Get the dimension of the B-spline space.

        Returns:
            int: The dimension of the B-spline space.
        """
        return len(self._spaces)

    @property
    def spaces(self) -> tuple[BsplineSpace1D, ...]:
        """Get the B-spline spaces.

        Returns:
            tuple[BsplineSpace1D, ...]: The B-spline spaces.
        """
        return self._spaces

    @functools.cached_property
    def degrees(self) -> tuple[int, ...]:
        """Get the polynomial degree of the B-spline.

        Returns:
            tuple[int, ...]: The degree for each dimension.
        """
        return tuple(space.degree for space in self._spaces)

    @functools.cached_property
    def tolerance(self) -> float:
        """Get the absolute tolerance used for parametric comparisons on this space.

        The largest of the univariate spaces' tolerances, each of which already
        carries its own direction's knot magnitude (see
        :attr:`~pantr.bspline.BsplineSpace1D.tolerance`). Taking the largest is the
        conservative choice when the directions are scaled differently.

        Returns:
            float: The absolute parametric tolerance.
        """
        return max(space.tolerance for space in self._spaces)

    @property
    def dtype(self) -> npt.DTypeLike:
        """Get the data type of the B-spline space.

        Returns:
            npt.DTypeLike: The numpy data type of the B-spline space.
        """
        return self._spaces[0].dtype

    @functools.cached_property
    def num_basis(self) -> tuple[int, ...]:
        """Get the number of basis functions for each dimension.

        Returns:
            tuple[int, ...]: The number of basis functions for each dimension.
        """
        return tuple(space.num_basis for space in self._spaces)

    @functools.cached_property
    def num_total_basis(self) -> int:
        """Get the total number of basis functions.

        Returns:
            int: The total number of basis functions.
        """
        return int(np.prod(self.num_basis))

    @functools.cached_property
    def num_intervals(self) -> tuple[int, ...]:
        """Get the number of intervals for each dimension.

        Returns:
            tuple[int, ...]: The number of intervals for each dimension.
        """
        return tuple(space.num_intervals for space in self._spaces)

    @functools.cached_property
    def num_total_intervals(self) -> int:
        """Get the total number of intervals.

        Returns:
            int: The total number of intervals.
        """
        return int(np.prod(self.num_intervals))

    @functools.cached_property
    def domain(self) -> npt.NDArray[np.float32 | np.float64]:
        """Get the domain of the B-spline space.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The domain of the B-spline space.
            The shape is (dim, 2), where the last dimension contains the start
            and end values of the domain.
        """
        domain = np.empty((self.dim, 2), dtype=self.dtype)
        for i, space in enumerate(self._spaces):
            domain[i, :] = space.domain
        return domain

    def has_Bezier_like_knots(self) -> bool:
        """Check if the knot vector represents a Bézier-like configuration.

        A Bézier-like configuration has open ends and only one non-zero span
        for each dimension.

        Returns:
            bool: True if knots have open ends and only one span.

        Example:
            >>> from pantr.bspline import BsplineSpace1D
            >>> bspline_1D = BsplineSpace1D([1, 1, 1, 3, 3, 3], 2)
            >>> bspline_2D = BsplineSpace([bspline_1D, bspline_1D])
            >>> bspline_2D.has_Bezier_like_knots()
            True
        """
        return all(space.has_Bezier_like_knots() for space in self._spaces)

    def tabulate_basis(
        self,
        pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
        out_basis: npt.NDArray[np.float32 | np.float64] | None = None,
        out_first_basis: npt.NDArray[np.int_] | None = None,
    ) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
        """Tabulate the B-spline basis functions at the given points.

        Args:
            pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): The points
               at which to tabulate the basis functions.
               It can be a 2D array with shape (num_pts, dim) or a PointsLattice object.
            out_basis (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the
                basis values will be stored. If None, a new array is allocated. Must have the
                correct shape and dtype if provided. This follows NumPy's style for output arrays.
                Defaults to None.
            out_first_basis (npt.NDArray[numpy.intp] | None): Optional output array where the
                first basis indices will be stored. If None, a new array is allocated. Must have
                the correct shape and dtype numpy.intp if provided. This follows NumPy's style for
                output arrays. Defaults to None.

        Returns:
            tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[numpy.intp]]: The basis
            function values and the first basis function indices.

            In the case pts is a 2D array, the shape of the basis function values array
            is (num_pts, order[0], order[1], ..., order[d-1]), where d is the dimension
            of the B-spline space and num_pts is the number of points.
            In the case pts is a PointsLattice object, the shape of the
            basis function values array is
            (num_pts_0, num_pts_1, ..., num_pts_d, order[0], order[1], ..., order[d-1]),
            where num_pts_i is the number of points in the i-th dimension.

            The shape of the first basis function indices array is (num_pts, dim),
            if pts is a 2D array, or (num_pts_0, num_pts_1, ..., num_pts_d, dim),
            if pts is a PointsLattice object.

            If `out_basis` or `out_first_basis` was provided, the corresponding element of the tuple
            is the same array.

        Raises:
            ValueError: If pts is not a 2D array or a PointsLattice object.
            ValueError: If the pts dimension does not match the dimension of the B-spline space.
            ValueError: If one or more points are outside the domain of the B-spline space, or if
                `out_basis` or `out_first_basis` is provided and has incorrect shape or dtype.
        """
        orders = tuple(degree + 1 for degree in self.degrees)
        num_local = self._h.num_local_functions
        if isinstance(pts, PointsLattice):
            lattice_h = pts._h
            dtype = np.dtype(pts.dtype)
            # Refuses another dimension (C++, with the message the suite matches).
            outside = self._h.first_direction_outside(lattice_h)
            if outside >= 0:
                raise ValueError(
                    f"One or more values in pts.pts_per_dir[{outside}] are outside the knot "
                    f"vector domain {self._spaces[outside].domain}"
                )
            pts_shape = tuple(len(p) for p in pts.pts_per_dir)
            num_pts = int(np.prod(pts_shape))

            def call(values: npt.NDArray[Any], first: npt.NDArray[Any]) -> None:
                self._h.tabulate_lattice(lattice_h, values, first)
        else:
            arr = pts if isinstance(pts, np.ndarray) else np.asarray(pts)
            if arr.ndim != 2:  # noqa: PLR2004
                raise ValueError("pts must be a 2D array")
            if arr.dtype not in (np.float32, np.float64):
                arr = arr.astype(np.float64)
            # The binding is overloaded on the dtype; ``coords`` is either, so it is untyped.
            coords: Any = np.ascontiguousarray(arr)
            # Refuses another column count (C++, with the message the suite matches).
            outside = self._h.first_direction_outside(coords)
            if outside >= 0:
                raise ValueError(
                    f"One or more values in pts[:, {outside}] are outside the knot vector"
                    f" domain {self.domain}"
                )
            dtype = np.dtype(self.dtype)
            pts_shape = (arr.shape[0],)
            num_pts = arr.shape[0]

            def call(values: npt.NDArray[Any], first: npt.NDArray[Any]) -> None:
                self._h.tabulate_points(coords, values, first)

        values_shape = (*pts_shape, *orders)
        first_shape = (*pts_shape, self.dim)
        if out_basis is None and out_first_basis is None:
            values = np.empty((num_pts, num_local), dtype=dtype)
            first = np.empty((num_pts, self.dim), dtype=_FIRST_DTYPE)
            call(values, first)
            return (
                values.reshape(values_shape),
                first.astype(np.int_, copy=False).reshape(first_shape),
            )
        out_basis = _allocate_or_validate_out(out_basis, values_shape, dtype)
        out_first_basis = _first_out(out_first_basis, first_shape)
        with (
            _kernel_buffer(out_basis, (num_pts, num_local)) as values,
            _kernel_buffer(out_first_basis, (num_pts, self.dim), _FIRST_DTYPE) as first,
        ):
            call(values, first)
        return out_basis, out_first_basis

    def cell_supports(self, cell_ids: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Return the control points supported on each of the given cells.

        Every cell of a tensor-product space is supported by exactly
        ``prod(degree + 1)`` control points, contiguous per direction, so the result
        is a dense ``(n, W)`` table rather than a ragged structure. Column ``k``
        always means the same local offset: the one whose C-order index within
        ``degrees + 1`` is ``k``.

        Args:
            cell_ids (npt.ArrayLike): Flat knot-span cell ids in C-order over
                :attr:`num_intervals`, the same convention as
                :func:`pantr.grid.tensor_product_grid` and
                :class:`SpanwiseElementExtraction`. Each must satisfy
                ``0 <= cid < num_total_intervals``.

        Returns:
            npt.NDArray[np.int64]: Array of shape ``(n, prod(degree + 1))`` of flat
            C-order control-point ids. Empty input gives shape ``(0, W)``.

        Raises:
            ValueError: If any direction is periodic, ``cell_ids`` is not integral or
                one-dimensional, or an id is out of range.

        Example:
            >>> from pantr.bspline import BsplineSpace, BsplineSpace1D
            >>> first = BsplineSpace1D([0, 0, 0, 1, 2, 3, 3, 3], 2)
            >>> second = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)
            >>> space = BsplineSpace([first, second])
            >>> space.num_basis
            (5, 3)
            >>> space.cell_supports([0]).tolist()
            [[0, 1, 2, 3, 4, 5, 6, 7, 8]]
        """
        return _cell_supports_impl(self, cell_ids)

    def boundary_dofs(
        self,
        direction: int,
        side: int,
        layers: int = 1,
    ) -> npt.NDArray[np.int64]:
        """Return the control points of a boundary slab, as flat C-order ids.

        The slab holds the control points whose index along ``direction`` lies in the
        first (``side == 0``) or last (``side == 1``) ``layers`` positions, and every
        index on the remaining axes. For an open knot vector, ``layers == 1`` selects
        exactly the basis functions with non-zero trace on that face, which is what a
        strong Dirichlet condition needs; a thicker slab serves a clamped or
        :math:`C^1` condition.

        The result holds **control-point** ids over :attr:`num_basis`, never cell ids
        over :attr:`num_intervals`. Interior knot multiplicities do not affect it: a
        boundary slab is a notion in index space.

        Args:
            direction (int): Axis of the face, in ``[0, dim)``.
            side (int): ``0`` for the low face, ``1`` for the high face. Together with
                ``direction`` this is the ``lfid = 2 * direction + side`` encoding of
                :meth:`pantr.grid.Grid.local_facet_axis_side`.
            layers (int): Slab thickness in control-point indices, in
                ``[1, num_basis[direction]]``. Defaults to 1.

        Returns:
            npt.NDArray[np.int64]: Read-only, strictly ascending flat C-order
            control-point ids, of shape
            ``(layers * num_total_basis // num_basis[direction],)``.

        Raises:
            ValueError: If ``direction``, ``side`` or ``layers`` is out of range, or if
                any direction is periodic -- a periodic direction has no boundary.

        Example:
            >>> from pantr.bspline import BsplineSpace, BsplineSpace1D
            >>> first = BsplineSpace1D([0, 0, 0, 1, 2, 3, 3, 3], 2)
            >>> second = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
            >>> space = BsplineSpace([first, second])
            >>> space.num_basis
            (5, 4)
            >>> space.boundary_dofs(0, 0).tolist()
            [0, 1, 2, 3]
            >>> space.boundary_dofs(1, 0).tolist()
            [0, 4, 8, 12, 16]
            >>> space.boundary_dofs(1, 1, layers=2).tolist()
            [2, 3, 6, 7, 10, 11, 14, 15, 18, 19]
        """
        if any(space.periodic for space in self._spaces):
            raise ValueError("boundary_dofs: periodic B-spline spaces are not supported.")
        if not 0 <= direction < self.dim:
            raise ValueError(
                f"boundary_dofs: direction must lie in [0, {self.dim}); got {direction}."
            )
        if side not in (0, 1):
            raise ValueError(f"boundary_dofs: side must be 0 (low) or 1 (high); got {side}.")
        num_basis_dir = self.num_basis[direction]
        if not 1 <= layers <= num_basis_dir:
            raise ValueError(
                f"boundary_dofs: layers must lie in [1, {num_basis_dir}]; got {layers}."
            )

        first = 0 if side == 0 else num_basis_dir - layers
        axes = [np.arange(n, dtype=np.int64) for n in self.num_basis]
        axes[direction] = np.arange(first, first + layers, dtype=np.int64)

        # Each axis range is ascending and the C-order flat id is monotone in the
        # lexicographic order of the multi-index, so the slab comes out sorted and
        # unique without an explicit sort.
        mesh = np.meshgrid(*axes, indexing="ij")
        dofs = np.ravel_multi_index(tuple(m.ravel() for m in mesh), self.num_basis).astype(np.int64)
        dofs.flags.writeable = False
        return dofs

    def restrict(self, cell_ids: npt.ArrayLike) -> BsplineSpaceRestriction:
        """Return the bounding-box windowed sub-space spanning ``cell_ids``.

        The window is the per-axis multi-index bounding box of the requested
        knot-span cells (flat ids in C-order over :attr:`num_intervals`, the same
        convention as :func:`pantr.grid.tensor_product_grid` and
        :class:`SpanwiseElementExtraction`). Each axis is windowed by slicing its
        knot vector (never re-clamped), so the windowed basis equals this space's
        basis pointwise over the windowed cells.

        Args:
            cell_ids (npt.ArrayLike): Flat knot-span cell ids to span; duplicates
                are ignored. Each must satisfy ``0 <= cid < num_total_intervals``.

        Returns:
            BsplineSpaceRestriction: The windowed :class:`BsplineSpace` and the
            read-only ``local_to_global_dof`` map of shape
            ``(windowed_space.num_total_basis,)``.

        Raises:
            ValueError: If ``cell_ids`` is empty or any axis is periodic.
            IndexError: If any cell id is out of range ``[0, num_total_intervals)``.
            TypeError: If ``cell_ids`` is not integer-valued.
        """
        if any(space.periodic for space in self._spaces):
            raise ValueError("restrict: periodic B-spline spaces are not supported.")
        ids = np.asarray(cell_ids).ravel()
        if ids.size == 0:
            raise ValueError("restrict: cell_ids must be non-empty.")
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError(f"restrict: cell_ids must be integer-valued; got dtype {ids.dtype}.")
        ids = ids.astype(np.int64, copy=False)
        n_int = self.num_total_intervals
        lo_id, hi_id = int(ids.min()), int(ids.max())
        if lo_id < 0 or hi_id >= n_int:
            raise IndexError(
                f"restrict: cell id out of range [0, {n_int}); got [{lo_id}, {hi_id}]."
            )

        multi = np.unravel_index(ids, self.num_intervals)
        windowed_1d: list[BsplineSpace1D] = []
        dof_axes: list[npt.NDArray[np.int64]] = []
        for d, space in enumerate(self._spaces):
            w_space, dof_d = space.restrict(int(multi[d].min()), int(multi[d].max()) + 1)
            windowed_1d.append(w_space)
            dof_axes.append(dof_d)

        mesh = np.meshgrid(*dof_axes, indexing="ij")
        local_to_global_dof = np.ravel_multi_index(
            tuple(m.ravel() for m in mesh), self.num_basis
        ).astype(np.int64)
        local_to_global_dof.flags.writeable = False
        return BsplineSpaceRestriction(BsplineSpace(windowed_1d), local_to_global_dof)


class BsplineSpaceRestriction(NamedTuple):
    """Result of :meth:`BsplineSpace.restrict`: a windowed space and its DOF map.

    - ``space`` -- the windowed :class:`BsplineSpace`: per axis a pure knot-vector
      slice of the parent (never re-clamped), so its basis equals the parent's
      pointwise over the windowed cells.
    - ``local_to_global_dof`` -- read-only, shape ``(space.num_total_basis,)``,
      mapping each windowed DOF (flat, C-order over the windowed per-axis basis
      counts) to its flat index in the parent space.

    Unlike :class:`pantr.grid.GridRestriction` there is no ``in_subset`` mask: every
    windowed DOF is a genuine parent DOF (a windowed space spans a box of cells, so
    there are no fill DOFs).
    """

    space: BsplineSpace
    local_to_global_dof: npt.NDArray[np.int64]
