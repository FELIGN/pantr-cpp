"""THB spline function: a :class:`THBSplineSpace` paired with control points.

This module provides :class:`THBSpline`, the hierarchical analogue of
:class:`~pantr.bspline.Bspline`.  It represents a function
``f(u) = Σ_i c_i φ_i(u)`` where ``φ_i`` are the active (truncated) hierarchical
basis functions of a :class:`~pantr.bspline.THBSplineSpace` and ``c_i`` the control
points (one per active dof).  Evaluation locates each point's active leaf cell and
combines the cell's active-basis values with the matching control points, in the C++
library (``pantr::thb::THBSpline``, ``include/pantr/thb/spline.hpp``).

Main exports:

- :class:`THBSpline`: an evaluable hierarchical B-spline function.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy import typing as npt

from .._ext import ext
from ._thb_spline_space import THBSplineSpace, _derivative_orders

if TYPE_CHECKING:
    from collections.abc import Sequence

_cpp: Any = ext.bspline
"""The compiled THB splines (private; untyped: one class per dtype)."""

_F32 = np.dtype(np.float32)


class THBSpline:
    """An evaluable THB spline function ``f = Σ_i c_i φ_i``.

    The hierarchical analogue of :class:`~pantr.bspline.Bspline`: it pairs a
    :class:`~pantr.bspline.THBSplineSpace` with one control point per active
    hierarchical dof.  Because the hierarchical basis is not of tensor-product
    structure, the control points stay a flat ``(num_total_basis,)`` (scalar) or
    ``(num_total_basis, rank)`` (vector-valued) array, rather than the
    ``(*num_basis, rank)`` grid of :class:`~pantr.bspline.Bspline`.

    Attributes:
        _h (Any): The compiled spline (``THBSplineF64`` or ``THBSplineF32``).
        _space (THBSplineSpace): The hierarchical space.
        _control_points (npt.NDArray[np.float64]): Control points reshaped to
            ``(num_total_basis, rank)`` (read-only view of ``_h``'s).
        _scalar (bool): Whether the field is scalar (``control_points`` was 1-D).
    """

    __slots__ = ("_control_points", "_h", "_scalar", "_space")

    _control_points: npt.NDArray[np.float64]

    def __init__(self, space: THBSplineSpace, control_points: npt.ArrayLike) -> None:
        """Create a THB spline function.

        Args:
            space (THBSplineSpace): The hierarchical space.
            control_points (npt.ArrayLike): Control points of shape
                ``(num_total_basis,)`` (scalar) or ``(num_total_basis, rank)``
                (vector-valued), ordered by global dof.

        Raises:
            TypeError: If ``space`` is not a :class:`THBSplineSpace`.
            ValueError: If ``control_points`` is not 1-D/2-D or its leading dimension
                does not equal ``space.num_total_basis``.
        """
        if not isinstance(space, THBSplineSpace):
            raise TypeError(f"space must be a THBSplineSpace; got {type(space).__name__!r}.")
        arr = np.asarray(control_points, dtype=np.float64)
        n = space.num_total_basis
        if arr.ndim == 1:
            scalar = True
            if arr.shape[0] != n:
                raise ValueError(f"control_points must have length {n}; got {arr.shape[0]}.")
            arr = arr.reshape(n, 1)
        elif arr.ndim == 2:  # noqa: PLR2004
            scalar = False
            if arr.shape[0] != n:
                raise ValueError(
                    f"control_points leading dimension must be {n}; got shape {arr.shape!r}."
                )
        else:
            raise ValueError(f"control_points must be 1-D or 2-D; got {arr.ndim}-D.")
        cls = _cpp.THBSplineF32 if space._root_space.dtype == _F32 else _cpp.THBSplineF64
        self._adopt(cls(space._h, np.ascontiguousarray(arr)), space, scalar)

    def _adopt(self, h: Any, space: THBSplineSpace, scalar: bool) -> None:  # noqa: ANN401
        """Hold a compiled spline and cache its read-only control points.

        Args:
            h (Any): The compiled spline.
            space (THBSplineSpace): The space object it is a function of.
            scalar (bool): Whether the field is scalar.
        """
        self._h = h
        self._space = space
        self._control_points = h.control_points
        self._scalar = scalar

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle through the space and the control points, with a subclass's attributes as state.

        Returns:
            tuple[Any, ...]: The rebuilding function, its arguments and the extra attributes.
        """
        return (
            _rebuild_thb_spline,
            (type(self), self._space, self.control_points),
            dict(self.__dict__) if hasattr(self, "__dict__") else None,
        )

    def __setstate__(
        self, state: dict[str, Any] | tuple[dict[str, Any] | None, dict[str, Any]]
    ) -> None:
        """Restore a subclass's attributes, or rebuild from a previous release's pickle.

        A previous release pickled the slots of a plain object, a pair ``(attributes, slots)``
        with the attributes of a subclass (or ``None``) and the slot values ``_space``,
        ``_control_points`` (always two-dimensional) and ``_scalar``. Such a state rebuilds the
        compiled spline from the space and the control points.

        Args:
            state (dict[str, Any] | tuple[dict[str, Any] | None, dict[str, Any]]): The pickled
                state: a subclass's extra attributes, or the previous release's pair.
        """
        if isinstance(state, tuple):
            extra, slots = state
            control_points = slots["_control_points"]
            if slots["_scalar"]:
                control_points = control_points[:, 0]
            THBSpline.__init__(self, slots["_space"], control_points)
            state = extra or {}
        if state:
            self.__dict__.update(state)

    def __copy__(self) -> THBSpline:
        """Copy: the new object shares the compiled value and the space.

        Returns:
            THBSpline: The copy, of the same class, with the same attributes.
        """
        new = object.__new__(type(self))
        new._adopt(self._h, self._space, self._scalar)
        if hasattr(self, "__dict__"):
            new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo: dict[int, Any]) -> THBSpline:
        """Deep copy: shares the compiled value (immutable) and deep-copies the space.

        Args:
            memo (dict[int, Any]): The deep-copy memo.

        Returns:
            THBSpline: The copy.
        """
        new = object.__new__(type(self))
        memo[id(self)] = new
        new._adopt(self._h, copy.deepcopy(self._space, memo), self._scalar)
        if hasattr(self, "__dict__"):
            for key, value in self.__dict__.items():
                new.__dict__[key] = copy.deepcopy(value, memo)
        return new

    @property
    def space(self) -> THBSplineSpace:
        """Get the underlying hierarchical space.

        Returns:
            THBSplineSpace: The space supplied at construction time.
        """
        return self._space

    @property
    def control_points(self) -> npt.NDArray[np.float64]:
        """Get the control points (read-only view).

        Returns:
            npt.NDArray[np.float64]: Shape ``(num_total_basis,)`` for a scalar field,
            ``(num_total_basis, rank)`` otherwise.  The array is read-only; copy it
            before modifying.
        """
        if self._scalar:
            return self._control_points[:, 0]
        return self._control_points

    @property
    def dim(self) -> int:
        """Get the parametric dimension.

        Returns:
            int: Number of parametric directions.
        """
        return self._space.dim

    @property
    def degree(self) -> tuple[int, ...]:
        """Get the per-direction polynomial degrees.

        Returns:
            tuple[int, ...]: Degree per parametric direction (mirrors
            :attr:`~pantr.bspline.Bspline.degree`).
        """
        return self._space.degrees

    @property
    def rank(self) -> int:
        """Get the output rank (number of value components).

        Returns:
            int: ``1`` for a scalar field; the number of components otherwise.
        """
        return int(self._control_points.shape[1])

    @property
    def dtype(self) -> np.dtype[np.float64]:
        """Get the floating-point dtype of the control points.

        Returns:
            np.dtype[np.float64]: Dtype of the stored control points, always
            ``numpy.float64``.
        """
        return self._control_points.dtype

    def evaluate(
        self,
        pts: npt.ArrayLike,
        out: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Evaluate the THB spline at ``pts``.

        Each point is located in its active leaf cell; the cell's active-basis values
        (:meth:`THBSplineSpace.tabulate_basis`) are combined with the matching control
        points.  Mirrors :meth:`~pantr.bspline.Bspline.evaluate`.

        Args:
            pts (npt.ArrayLike): Parametric points of shape ``(..., dim)``.
            out (npt.NDArray[np.float64] | None): Optional output array of the result
                shape (``(...)`` for a scalar field, ``(..., rank)`` otherwise).
                Allocated when ``None``.

        Returns:
            npt.NDArray[np.float64]: Values of shape ``(...)`` for a scalar field
            or ``(..., rank)`` for a vector-valued field.

        Raises:
            ValueError: If ``pts`` does not have trailing dimension ``dim``, any point
                lies outside the grid domain, or ``out`` has the wrong shape/dtype or
                is not writeable.
            RuntimeError: If the grid has been modified since construction, or a cell
                has no active basis functions (inconsistent space).
        """
        return self._evaluate_orders(pts, (0,) * self.dim, out)

    def evaluate_derivatives(
        self,
        pts: npt.ArrayLike,
        orders: int | Sequence[int],
        out: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        r"""Evaluate a mixed partial derivative of the THB spline at ``pts``.

        Computes the single mixed partial :math:`\partial^{orders} f`, where
        ``orders[k]`` is the derivative order in parametric direction ``k`` (with
        respect to the parametric coordinates).  Mirrors
        :meth:`~pantr.bspline.Bspline.evaluate_derivatives`.

        Args:
            pts (npt.ArrayLike): Parametric points of shape ``(..., dim)``.
            orders (int | Sequence[int]): Per-direction derivative orders.  A scalar is
                broadcast to every direction.  Each entry must be ``>= 0``.
            out (npt.NDArray[np.float64] | None): Optional output array of the result
                shape.  Allocated when ``None``.

        Returns:
            npt.NDArray[np.float64]: Derivative values of shape ``(...)`` for a scalar
            field or ``(..., rank)`` for a vector-valued field.

        Raises:
            ValueError: If ``orders`` has the wrong length or a negative entry, if
                ``pts`` does not have trailing dimension ``dim``, any point lies
                outside the grid domain, or ``out`` has the wrong shape/dtype or is
                not writeable.
            RuntimeError: If the grid has been modified since construction, or a cell
                has no active basis functions (inconsistent space).
        """
        return self._evaluate_orders(pts, orders, out)

    def _evaluate_orders(
        self,
        pts: npt.ArrayLike,
        orders: int | Sequence[int],
        out: npt.NDArray[np.float64] | None,
    ) -> npt.NDArray[np.float64]:
        """Shared evaluation of the ``orders`` mixed partial (values when all zero).

        Args:
            pts (npt.ArrayLike): Parametric points of shape ``(..., dim)``.
            orders (int | Sequence[int]): Per-direction derivative orders (validated by
                :meth:`THBSplineSpace.tabulate_basis_derivatives`).
            out (npt.NDArray[np.float64] | None): Optional output array of the result
                shape.

        Returns:
            npt.NDArray[np.float64]: Values of shape ``(...)`` / ``(..., rank)``.

        Raises:
            ValueError: If ``pts``/``orders``/``out`` are invalid (see the public
                methods).
            RuntimeError: If the grid is stale or a cell has no active functions.
        """
        self._space._check_not_stale()
        arr = np.asarray(pts, dtype=np.float64)
        if arr.ndim == 0 or arr.shape[-1] != self.dim:
            raise ValueError(
                f"pts must have trailing dimension {self.dim}; got shape {arr.shape!r}."
            )
        batch_shape = arr.shape[:-1]
        flat = np.ascontiguousarray(arr.reshape(-1, self.dim))
        rank = self.rank

        # The orders are looked at only once there is a point to evaluate, as the cells' own
        # tabulation did.
        orders_t = _derivative_orders(orders, self.dim) if flat.shape[0] > 0 else ()
        raw: npt.NDArray[np.float64] = self._h.evaluate(flat, list(orders_t))

        result = raw[:, 0].reshape(batch_shape) if self._scalar else raw.reshape(*batch_shape, rank)
        if out is None:
            return result
        if out.shape != result.shape:
            raise ValueError(f"out must have shape {result.shape}; got {out.shape}.")
        if out.dtype != np.float64:
            raise ValueError(f"out must have dtype float64; got {out.dtype}.")
        if not out.flags.writeable:
            raise ValueError("out must be writeable.")
        out[...] = result
        return out

    def refine(self, cell_ids: npt.ArrayLike, *, admissible_class: int | None = 2) -> THBSpline:
        """Return the same function on a space with the marked cells refined.

        Refines the underlying :class:`THBSplineSpace` (see
        :meth:`THBSplineSpace.refine`) and prolongs the control points onto it, so the
        returned spline represents the **same function** -- hierarchical refinement is
        exact (the spaces are nested).  This does not mutate ``self``.

        Args:
            cell_ids (npt.ArrayLike): Flat ids of active cells to refine.
            admissible_class (int | None): Admissibility class ``m >= 2`` to maintain
                (graded refinement), or ``None`` for ungraded refinement.  Defaults
                to ``2``.  See :meth:`THBSplineSpace.refine`.

        Returns:
            THBSpline: A new spline on the refined space, equal to ``self`` as a
            function.

        Raises:
            IndexError: If any id is outside ``[0, grid.num_cells)``.
            ValueError: If ``admissible_class`` is an integer ``< 2``.
            RuntimeError: If the grid has been modified since construction.
        """
        fine = self._space.refine(cell_ids, admissible_class=admissible_class)
        return self._prolong_to(fine)

    def refine_region(
        self,
        level: int,
        lo: Sequence[int],
        hi: Sequence[int],
        *,
        admissible_class: int | None = 2,
    ) -> THBSpline:
        """Return the same function on a space with a rectangular region refined.

        Refines the underlying :class:`THBSplineSpace` (see
        :meth:`THBSplineSpace.refine_region`) and prolongs the control points onto it,
        so the returned spline represents the **same function**.  This does not mutate
        ``self``.

        Args:
            level (int): Level at which the box lives.  Must satisfy
                ``0 <= level <= grid.max_level``.
            lo (Sequence[int]): Per-direction start index (inclusive), in
                level-``level`` coordinates.
            hi (Sequence[int]): Per-direction end index (exclusive), in
                level-``level`` coordinates.
            admissible_class (int | None): Admissibility class ``m >= 2`` to maintain,
                or ``None`` for ungraded refinement.  Defaults to ``2``.  See
                :meth:`THBSplineSpace.refine_region`.

        Returns:
            THBSpline: A new spline on the refined space, equal to ``self`` as a
            function.

        Raises:
            ValueError: If ``admissible_class`` is an integer ``< 2``, ``level`` is
                out of range, ``lo``/``hi`` have the wrong length, any
                ``lo[k] >= hi[k]``, or ``[lo, hi)`` lies outside the level domain.
            RuntimeError: If the grid has been modified since construction.
        """
        fine = self._space.refine_region(level, lo, hi, admissible_class=admissible_class)
        return self._prolong_to(fine)

    def _prolong_to(self, fine: THBSplineSpace) -> THBSpline:
        """Prolong the control points onto a refinement ``fine`` (value-preserving).

        Args:
            fine (THBSplineSpace): A refinement of this spline's space.

        Returns:
            THBSpline: A new spline on ``fine`` equal to ``self`` as a function.
        """
        new = object.__new__(THBSpline)
        new._adopt(self._h.prolong_to(fine._h), fine, self._scalar)
        return new

    def __repr__(self) -> str:
        """Return a concise representation.

        Returns:
            str: Summary with dimension, rank, and active-function count.
        """
        return (
            f"THBSpline(dim={self.dim}, rank={self.rank}, num_total_basis="
            f"{self._space.num_total_basis})"
        )


def _rebuild_thb_spline(
    cls: type[THBSpline], space: THBSplineSpace, control_points: npt.NDArray[np.float64]
) -> THBSpline:
    """Rebuild a pickled THB spline of class ``cls``, bypassing ``cls.__init__``.

    A subclass's constructor may take other arguments; the previous release's pickles did not
    call it either.

    Args:
        cls (type[THBSpline]): The class of the pickled spline.
        space (THBSplineSpace): Its space.
        control_points (npt.NDArray[np.float64]): Its control points, scalar or vector-valued as
            they were.

    Returns:
        THBSpline: The spline (its extra attributes are restored by ``__setstate__``).
    """
    spline = cls.__new__(cls)
    THBSpline.__init__(spline, space, control_points)
    return spline
