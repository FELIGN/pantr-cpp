"""Bezier geometric object: the Bezier class."""

from __future__ import annotations

import copy as _copy
import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from numpy import typing as npt

from .._ext import ext
from ..basis._basis_utils import _validate_out_array
from ..quad import PointsLattice
from ._bezier_degree import (
    _degree_reduce_bezier,
    _degree_reduction_l2_error,
    _minimize_degree_bezier,
)

if TYPE_CHECKING:
    from ..bspline import Bspline
    from ..transform import AffineTransform

_cpp: Any = ext.bezier
"""The compiled Bezier values (private; untyped: one class per dtype)."""

_F32: np.dtype[np.float32] = np.dtype(np.float32)
_F64: np.dtype[np.float64] = np.dtype(np.float64)


class Bezier:
    """A parametric Bézier curve, surface, or volume defined by control points.

    Stores only control points and an ``is_rational`` flag. The polynomial
    degree in each parametric direction is inferred from the control point
    array shape: ``degree[d] = control_points.shape[d] - 1``.

    The value is immutable: the control points are copied in at construction
    and read-only, and the operations that take ``in_place=True`` replace the
    compiled value this object holds rather than writing into it. (The one
    exception, until :class:`~pantr.bspline.Bspline` is ported: a Bézier made
    by a ``copy=False`` conversion from a B-spline shares the B-spline's array,
    so an in-place change of the B-spline shows through it.)

    Attributes:
        _h (Any): The compiled value (``BezierF64`` or ``BezierF32``).
        _shape (tuple[int, ...]): Control point shape ``(*degrees_plus_1, cp_size)``,
            where the last axis includes the homogeneous weight for rational.
        _is_rational (bool): Whether the Bézier is rational (last control
            point coordinate is a homogeneous weight).
        _dtype (np.dtype): ``float32`` or ``float64``.

    The read-only control point view is read from ``_h`` on first use and kept
    in the instance (``control_points`` is a cached property that refuses
    assignment).
    """

    _h: Any
    _shape: tuple[int, ...]
    _is_rational: bool
    _dtype: np.dtype[np.float32] | np.dtype[np.float64]

    def __init__(
        self,
        control_points: npt.ArrayLike,
        is_rational: bool = False,
    ) -> None:
        """Initialize a Bézier from control points.

        Args:
            control_points (npt.ArrayLike): Control points. Shape must be at
                least 2D: ``(*degrees_plus_1, rank)``. A 1D input of shape
                ``(n,)`` is reshaped to ``(n, 1)`` (scalar field). They are
                copied. ``float32`` input stays ``float32``; any other real
                dtype (integers and booleans included) is cast to ``float64``.
            is_rational (bool): Whether the Bézier is rational. Defaults to
                False.

        Raises:
            ValueError: If the control points have fewer than 1 entry in any
                parametric direction.
            ValueError: If the Bézier has rank smaller than 1.
            TypeError: If the control points are complex.
        """
        cp = _as_control_points(control_points)
        rational = bool(is_rational)
        f32 = cp.dtype is _F32
        # Stored in the instance dict directly: an attribute write would go through `__setattr__`.
        d = self.__dict__
        d["_h"] = (_cpp.BezierF32 if f32 else _cpp.BezierF64)(cp, rational)
        d["_shape"] = cp.shape
        d["_is_rational"] = rational
        d["_dtype"] = _F32 if f32 else _F64

    def _set_handle(
        self,
        h: Any,  # noqa: ANN401
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype[Any] | None = None,
    ) -> None:
        """Hold a compiled value and cache what the properties return (DR-25).

        Args:
            h (Any): The compiled value.
            shape (tuple[int, ...] | None): Its control point shape, when the caller knows it.
            dtype (np.dtype | None): Its dtype, when the caller knows it.
        """
        f32 = isinstance(h, _cpp.BezierF32) if dtype is None else dtype == _F32
        d = self.__dict__
        d["_h"] = h
        d["_shape"] = tuple(h.shape) if shape is None else shape
        d["_is_rational"] = bool(h.rational)
        d["_dtype"] = _F32 if f32 else _F64
        # Drop the control-point view of the previous value; the next read takes the new one.
        d.pop("control_points", None)

    @classmethod
    def _from_handle(cls, h: Any) -> Bezier:  # noqa: ANN401
        """Wrap a compiled value without copying it.

        Args:
            h (Any): The compiled value.

        Returns:
            Bezier: The wrapper.
        """
        b = object.__new__(cls)
        b._set_handle(h)
        return b

    @classmethod
    def _borrow(cls, control_points: npt.ArrayLike, is_rational: bool) -> Bezier:
        """Build a Bézier over an array without copying it (a ``copy=False`` conversion).

        Interim, while :class:`~pantr.bspline.Bspline` is still a Python object over a NumPy
        array: a ``copy=False`` conversion from it shares that array, as the previous release
        did, so a later in-place change of the B-spline shows in the Bézier. An array that needs
        a conversion (dtype, layout) is converted first, which shares nothing.

        Args:
            control_points (npt.ArrayLike): The control points.
            is_rational (bool): Whether the Bézier is rational.

        Returns:
            Bezier: The Bézier.
        """
        cp = _as_control_points(control_points)
        b = object.__new__(cls)
        b._set_handle(_class_for(cp.dtype).borrow(cp, bool(is_rational)), cp.shape, cp.dtype)
        return b

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle through the constructor, with a subclass's own attributes as state.

        Returns:
            tuple[Any, ...]: The class, the constructor arguments and the extra attributes.
        """
        return (type(self), (self.control_points, self._is_rational), _extra_state(self))

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore extra attributes, or rebuild from a previous release's pickle.

        A previous release pickled the plain attributes ``_control_points`` and ``_is_rational``;
        such a state rebuilds the compiled value from them.

        Args:
            state (dict[str, Any]): The pickled state.
        """
        state = dict(state)
        if "_control_points" in state:
            cp = state.pop("_control_points")
            Bezier.__init__(self, cp, bool(state.pop("_is_rational", False)))
        self.__dict__.update(state)

    def __copy__(self) -> Bezier:
        """Copy: the new object shares the immutable compiled value.

        Returns:
            Bezier: The copy, of the same class, with the same attributes.
        """
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo: dict[int, Any]) -> Bezier:
        """Deep copy: shares the immutable compiled value and deep-copies any other attribute.

        Args:
            memo (dict[int, Any]): The deep-copy memo.

        Returns:
            Bezier: The copy.
        """
        new = object.__new__(type(self))
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            new.__dict__[key] = value if key in _VALUE_ATTRS else _copy.deepcopy(value, memo)
        return new

    @property
    def dim(self) -> int:
        """Get the parametric dimension of the Bézier.

        Returns:
            int: Number of parametric dimensions.
        """
        return len(self._shape) - 1

    @property
    def degree(self) -> tuple[int, ...]:
        """Get the polynomial degrees per parametric direction.

        Returns:
            tuple[int, ...]: Polynomial degree for each parametric dimension,
            computed as ``shape[d] - 1``.
        """
        return tuple(s - 1 for s in self._shape[:-1])

    @functools.cached_property
    def control_points(self) -> npt.NDArray[np.float32 | np.float64]:
        """Get the control points of the Bézier.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Control point array with
            shape ``(*degrees_plus_1, rank)``, read-only: a view of the
            immutable value, which keeps the values it showed when it was read
            even after an in-place operation on this object.
        """
        # Read once and kept in the instance dict, so later reads cost no call; `__setattr__` and
        # `__delattr__` refuse the name, as they did for the plain property it replaces.
        cp: npt.NDArray[np.float32 | np.float64] = self._h.control_points
        return cp

    def __setattr__(self, name: str, value: object) -> None:
        """Refuse assigning ``control_points``, a read-only property.

        Args:
            name (str): The attribute.
            value (object): The value.

        Raises:
            AttributeError: For ``control_points``.
        """
        if name == "control_points":
            raise AttributeError(
                f"property 'control_points' of {type(self).__name__!r} object has no setter"
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        """Refuse deleting ``control_points``, a read-only property.

        Args:
            name (str): The attribute.

        Raises:
            AttributeError: For ``control_points``.
        """
        if name == "control_points":
            raise AttributeError(
                f"property 'control_points' of {type(self).__name__!r} object has no deleter"
            )
        object.__delattr__(self, name)

    @property
    def is_rational(self) -> bool:
        """Check whether the Bézier is rational.

        Returns:
            bool: True if the Bézier is rational (i.e., the last control point
            coordinate is a homogeneous weight), False otherwise.
        """
        return self._is_rational

    @property
    def rank(self) -> int:
        """Get the output rank of the Bézier.

        The rank is the number of value dimensions produced by the mapping.
        For a scalar field it is 1; for a 3D curve it is 3. For rational
        Béziers the weight coordinate is excluded.

        Returns:
            int: Output rank of the Bézier.
        """
        rk = self._shape[-1]
        return rk - 1 if self._is_rational else rk

    @property
    def dtype(self) -> npt.DTypeLike:
        """Get the floating-point dtype of the Bézier.

        Returns:
            npt.DTypeLike: The numpy dtype (float32 or float64) of the control
            point array.
        """
        return self._dtype

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Evaluate the Bézier at the given parametric points.

        Args:
            pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): The
                parametric points at which to evaluate. For 1D Bézier, must be
                a 1D array of shape ``(n_pts,)``. For multi-dimensional Bézier,
                must be a 2D array of shape ``(n_pts, dim)`` or a
                :class:`~pantr.quad.PointsLattice`.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                array. Defaults to None.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Bézier values at the given
            points.

        Raises:
            ValueError: If the points dtype does not match the Bézier dtype,
                or if ``out`` has incorrect shape or dtype.
        """
        if isinstance(pts, PointsLattice):
            self._check_lattice(pts)
            raw = self._h.evaluate_lattice(pts._h)
        else:
            raw = self._h.evaluate(self._points_2d(pts))
        return _finish(raw, out, self._dtype)

    def evaluate_derivatives(
        self,
        pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
        orders: int | Sequence[int],
        out: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Evaluate a specific partial derivative of the Bézier.

        Computes the single partial derivative specified by ``orders``,
        where ``orders[d]`` is the derivative order in parametric direction
        ``d``. For rational Bézier the generalised quotient rule is applied.

        Args:
            pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): The
                parametric points at which to evaluate.
            orders (int | Sequence[int]): Derivative order(s). A single
                ``int`` is broadcast to all ``self.dim`` directions. A sequence
                must contain one non-negative integer per parametric direction.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional
                pre-allocated output array. Defaults to None.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Mixed partial derivative
            values.

        Raises:
            ValueError: If ``len(orders) != self.dim``, if any order is
                negative, or if the points dtype does not match the Bézier dtype.

        Example:
            >>> import numpy as np
            >>> f = Bezier(np.array([[0.0], [2.0]]))  # f(t) = 2t
            >>> np.allclose(f.evaluate_derivatives(np.array([0.3, 0.7]), 1), [2.0, 2.0])
            True
            >>> cp = np.zeros((2, 2, 1))
            >>> cp[1, 1, 0] = 1.0  # f(u, v) = u * v
            >>> surf = Bezier(cp)
            >>> np.allclose(surf.evaluate_derivatives(np.array([[0.3, 0.4]]), [1, 1]), [1.0])
            True
        """
        orders_seq: Sequence[int] = [orders] * self.dim if isinstance(orders, int) else orders
        orders_tuple = tuple(orders_seq)
        # Checked before the points, as before; the compiled call checks them again.
        if len(orders_tuple) != self.dim:
            raise ValueError(f"len(orders) ({len(orders_tuple)}) must match dim ({self.dim}).")
        if any(o < 0 for o in orders_tuple):
            raise ValueError("All derivative orders must be non-negative.")
        if isinstance(pts, PointsLattice):
            self._check_lattice(pts)
            raw = self._h.evaluate_derivatives_lattice(pts._h, list(orders_tuple))
        else:
            raw = self._h.evaluate_derivatives(self._points_2d(pts), list(orders_tuple))
        return _finish(raw, out, self._dtype)

    def _check_lattice(self, pts: PointsLattice) -> None:
        """Check a lattice's dimension and dtype against this Bézier, as before.

        Args:
            pts (PointsLattice): The lattice.

        Raises:
            ValueError: On a dimension or dtype mismatch.
        """
        dtype = self._dtype
        if self.dim == 1:
            if pts.dim != 1:
                raise ValueError("PointsLattice must be 1D for a 1D Bézier.")
            if pts.dtype != dtype:
                raise ValueError(f"Points dtype ({pts.dtype}) must match Bézier dtype ({dtype}).")
            return
        if pts.dim != self.dim:
            raise ValueError(f"PointsLattice dim ({pts.dim}) must match Bézier dim ({self.dim}).")
        if pts.dtype != dtype:
            raise ValueError(
                f"PointsLattice dtype ({pts.dtype}) must match Bézier dtype ({dtype})."
            )

    def _points_2d(self, pts: Any) -> npt.NDArray[np.float32 | np.float64]:  # noqa: ANN401
        """Check points against this Bézier and view them as a C-contiguous ``(n, dim)`` array.

        Args:
            pts (Any): The points: ``(n,)`` for a 1D Bézier, ``(n, dim)`` otherwise.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The points, ``(n, dim)``.

        Raises:
            ValueError: On a shape or dtype mismatch (the messages of the previous release).
        """
        arr = np.asarray(pts)
        dtype = self._dtype
        if self.dim == 1:
            if arr.ndim != 1:
                raise ValueError("Points must be a 1D array for a 1D Bézier.")
            if arr.dtype != dtype:
                raise ValueError(f"Points dtype ({arr.dtype}) must match Bézier dtype ({dtype}).")
            return np.ascontiguousarray(arr).reshape(-1, 1)
        if arr.ndim != 2 or arr.shape[1] != self.dim:  # noqa: PLR2004
            raise ValueError(f"pts must be a 2D array with {self.dim} columns.")
        if arr.dtype != dtype:
            raise ValueError(f"Points dtype ({arr.dtype}) must match Bézier dtype ({dtype}).")
        return np.ascontiguousarray(arr)

    # ------------------------------------------------------------------
    # Derivative (returns new Bezier)
    # ------------------------------------------------------------------

    def derivative(self, direction: int = 0, *, keep_degree: bool = False) -> Bezier:
        """Return a Bézier representing the first derivative in the given direction.

        Computes the hodograph: a new Bézier whose value at every parametric
        point equals the partial derivative of this Bézier with respect to
        parametric direction ``direction``.

        For non-rational Bézier of degree ``p`` in direction ``d``, the
        result has degree ``p - 1`` (or ``p`` when ``keep_degree=True``).

        For rational Bézier, the quotient rule is applied, producing a
        rational Bézier of degree ``2p`` in direction ``d`` (or the original
        degree when ``keep_degree=True``).

        Args:
            direction (int): Parametric direction for differentiation.
                Must be in ``[0, dim)``. Defaults to 0.
            keep_degree (bool): If ``True``, the result preserves the same
                degree as the original Bézier by fusing derivative and degree
                elevation. This is useful, for instance, when computing
                derivatives of rational polynomials (in the numerator).
                Defaults to ``False``.

        Returns:
            Bezier: A new Bézier representing the derivative.

        Raises:
            ValueError: If ``direction`` is out of range ``[0, dim)``.
            ValueError: If the degree in the given direction is 0.

        Example:
            >>> import numpy as np
            >>> f = Bezier(np.array([[0.0], [0.0], [1.0]]))  # Bernstein form of t**2
            >>> fp = f.derivative()
            >>> fp.degree
            (1,)
            >>> np.allclose(fp.evaluate(np.array([0.3])), [0.6])
            True
            >>> f.derivative(keep_degree=True).degree
            (2,)
        """
        return Bezier._from_handle(self._h.derivative(direction, bool(keep_degree)))

    # ------------------------------------------------------------------
    # Degree elevation
    # ------------------------------------------------------------------

    def elevate_degree(self, degree_increments: int | Sequence[int]) -> Bezier:
        """Elevate the polynomial degree of the Bézier.

        Creates a new Bézier that represents the same mapping as the original
        but with higher polynomial degree.

        Args:
            degree_increments (int | Sequence[int]): Number of degrees to
                increase. If an integer, the same increment is applied to all
                parametric directions. If a sequence, must have length equal
                to ``self.dim``.

        Returns:
            Bezier: A new Bézier with elevated degrees.

        Raises:
            ValueError: If any degree increment is negative.
            ValueError: If all degree increments are zero.
            ValueError: If the number of increments does not match the dimension.
        """
        if isinstance(degree_increments, int):
            increments = (degree_increments,) * self.dim
        else:
            increments = tuple(degree_increments)
        return Bezier._from_handle(self._h.elevate_degree(list(increments)))

    # ------------------------------------------------------------------
    # Degree reduction
    # ------------------------------------------------------------------

    def reduce_degree(self, degree_decrements: int | Sequence[int]) -> Bezier:
        r"""Reduce the polynomial degree of the Bézier, interpolating the endpoints.

        Creates a new Bézier whose degree is lower by the requested amount in
        each parametric direction.  Among the polynomials of the lower degree
        that reproduce this one at the ends of the parametric domain, the result
        is the one closest in :math:`L^2([0, 1]^{\dim})`; the endpoint values are
        reproduced exactly, bit for bit.

        Unlike :meth:`elevate_degree`, this operation is **not exact** in
        general: the result is an approximation of the original mapping.  Use
        :meth:`degree_reduction_error` to find out how close it is before
        committing to it.

        The endpoint conditions cost accuracy in the interior — dropping them
        would lower the :math:`L^2` error by a factor between 1.1 (degree 16) and
        4.5 (reduction to a straight line) — and buy an approximation that joins
        its neighbours exactly, which is what makes the B-spline case work.  A
        reduction to degree 0 cannot honour two conditions with one coefficient
        and returns the plain :math:`L^2` projection, the mean of the control
        points.

        Args:
            degree_decrements (int | Sequence[int]): Number of degrees to
                reduce. If an integer, the same decrement is applied to all
                parametric directions. If a sequence, must have length equal
                to ``self.dim``.

        Returns:
            Bezier: A new Bézier with reduced degrees.

        Raises:
            ValueError: If any degree decrement is negative.
            ValueError: If all degree decrements are zero.
            ValueError: If the number of decrements does not match the dimension.
            ValueError: If any decrement exceeds the current degree in that
                direction.
            ValueError: If the object is rational and the reduction drives a
                control weight to zero or below.
        """
        return _degree_reduce_bezier(self, self._checked_decrements(degree_decrements))

    def degree_reduction_error(self, degree_decrements: int | Sequence[int]) -> float:
        r"""Compute the :math:`L^2` error that :meth:`reduce_degree` would introduce.

        The value is exact rather than an estimate: it is
        :math:`\lVert f - g \rVert_{L^2}` for the reduction *g* this Bézier would
        produce, obtained from the Bernstein mass matrix rather than by sampling.
        Rank components are combined in the Euclidean sense, and for a rational
        Bézier the norm is taken over the homogeneous coefficients, not over the
        projected mapping.

        The convex-hull bound on the coefficient residual is deliberately not
        offered instead: measured against the true supremum of the error it runs
        from 1.3 to 1600 times too large, the ratio growing with degree, so it is
        of little use as a stopping criterion.

        Args:
            degree_decrements (int | Sequence[int]): Number of degrees to
                reduce, as in :meth:`reduce_degree`.

        Returns:
            float: The :math:`L^2` norm of the error, in the units of the
            control points.

        Raises:
            ValueError: If any degree decrement is negative.
            ValueError: If all degree decrements are zero.
            ValueError: If the number of decrements does not match the dimension.
            ValueError: If any decrement exceeds the current degree in that
                direction.

        Example:
            >>> import numpy as np
            >>> from pantr.bezier import Bezier
            >>> bezier = Bezier(np.array([[0.0], [1.0], [0.0], [1.0]]))
            >>> round(bezier.degree_reduction_error(1), 6)
            0.138013
        """
        decrements = self._checked_decrements(degree_decrements)
        return _degree_reduction_l2_error(self, decrements)

    def _checked_decrements(self, degree_decrements: int | Sequence[int]) -> tuple[int, ...]:
        """Normalise and validate a degree-decrement argument.

        Args:
            degree_decrements (int | Sequence[int]): Decrement for every
                direction, or one per direction.

        Returns:
            tuple[int, ...]: One decrement per parametric direction.

        Raises:
            ValueError: If any degree decrement is negative.
            ValueError: If all degree decrements are zero.
            ValueError: If the number of decrements does not match the dimension.
            ValueError: If any decrement exceeds the current degree in that
                direction.
        """
        if isinstance(degree_decrements, int):
            decrements = (degree_decrements,) * self.dim
        else:
            decrements = tuple(degree_decrements)

        if len(decrements) != self.dim:
            raise ValueError(
                f"Number of degree decrements ({len(decrements)}) "
                f"must match dimension ({self.dim})."
            )

        if any(dec < 0 for dec in decrements):
            raise ValueError("Degree decrements must be non-negative.")

        if all(dec == 0 for dec in decrements):
            raise ValueError("At least one degree decrement must be positive.")

        for d, dec in enumerate(decrements):
            if dec > self.degree[d]:
                raise ValueError(
                    f"Degree decrement ({dec}) in direction {d} exceeds "
                    f"current degree ({self.degree[d]})."
                )

        return decrements

    def minimize_degree(self, tol: float | None = None) -> Bezier:
        """Find the lowest degree that preserves accuracy within tolerance.

        Iterates over each parametric direction and repeatedly tries to
        reduce the degree by 1.  A reduction is accepted when the
        round-trip (reduce then elevate) relative L2 error stays below
        ``tol``.  For vector-valued Bézier, all rank components are
        checked simultaneously.

        For a **rational** Bézier that error is measured on the projected
        mapping, not on the homogeneous control coefficients, so ``tol`` is a
        budget on the geometry the caller sees at any coordinate scale.  Two
        consequences: it is then a quadrature estimate rather than an exact
        value, accurate to better than 3% of the true relative deviation over
        degrees 3 to 20 and weight ratios up to 100; and a reduction is refused
        outright when a weight is not strictly positive on the domain, since the
        mapping then has a pole and no projected deviation is defined.

        Args:
            tol (float | None): Relative tolerance for accepting a degree
                reduction.  If *None*, uses a default based on machine
                epsilon (``1e3 * eps``).

        Returns:
            Bezier: A new Bézier with the lowest degree that preserves
            accuracy.  If no reduction is possible, returns a copy.

        Example:
            >>> import numpy as np
            >>> b = Bezier(np.array([3.0, 3.0, 3.0, 1.0]).reshape(4, 1))
            >>> b.degree
            (3,)
            >>> b_min = b.minimize_degree()
        """
        return _minimize_degree_bezier(self, tol)

    # ------------------------------------------------------------------
    # Multiply
    # ------------------------------------------------------------------

    def multiply(self, other: Bezier) -> Bezier:
        """Return the pointwise product of this Bézier and another.

        Given Bézier ``self`` and ``other`` over the same parametric domain
        ``[0, 1]^dim``, returns a new Bézier ``h`` such that
        ``h(t) = self(t) * other(t)``. The result has degree ``p_d + q_d``
        per direction.

        That degree represents the product **exactly**: no approximation is made
        in choosing it. The control points are not exact, and this docstring used
        to say they were. Each is a binomial-weighted sum evaluated in floating
        point, so each carries that sum's roundings.

        Args:
            other (Bezier): The second Bézier operand. Must have the same
                dimension, dtype, and rank as ``self``.

        Returns:
            Bezier: A new Bézier representing ``self * other``.

        Raises:
            ValueError: If the operands have different dimensions, dtypes,
                or ranks.

        Example:
            >>> import numpy as np
            >>> f = Bezier(np.array([[0.0], [1.0]]))  # f(t) = t
            >>> g = Bezier(np.array([[0.0], [1.0]]))  # g(t) = t
            >>> h = f.multiply(g)
            >>> h.degree
            (2,)
            >>> np.allclose(h.evaluate(np.array([0.5])), [0.25])
            True
            >>> np.allclose((f * g).evaluate(np.array([0.5])), [0.25])  # same via __mul__
            True
        """
        # Checked here, in this order, for the messages that spell the dtypes; the compiled call
        # checks dimension and rank again.
        if self.dim != other.dim:
            raise ValueError(
                f"Operands must have the same dimension. Got {self.dim} and {other.dim}."
            )
        if self._dtype != other._dtype:
            raise ValueError(
                f"Operands must have the same dtype. Got {self._dtype} and {other._dtype}."
            )
        return Bezier._from_handle(self._h.multiply(other._h))

    __mul__ = multiply

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self, inner: Bezier) -> Bezier:
        """Compose this Bézier with another: ``result(t) = self(inner(t))``.

        Composes two non-rational Bézier objects. The result is a new Bézier with
        parametric dimension equal to ``inner.dim``, rank equal to ``self.rank``,
        and degree ``sum(self.degree) * inner.degree[s]`` in each direction ``s``.

        That degree represents the composition **exactly**; the control points do
        not, and this docstring used to call the composition itself exact. They
        are built from repeated Bernstein products in floating point -- one per
        power of the inner map and one per tensor term -- and carry every one of
        those roundings.

        Args:
            inner (Bezier): The inner Bézier (reparametrization). Must be
                non-rational and satisfy ``inner.rank == self.dim``.

        Returns:
            Bezier: A new Bézier representing ``self(inner(t))``.

        Raises:
            TypeError: If either Bézier is rational.
            ValueError: If ``inner.rank != self.dim``.
            ValueError: If the operands have different dtypes.
            ValueError: If a one-dimensional ``inner`` would drive the composed
                degree past the exactness envelope of the binomial-coefficient
                kernel. The envelope is *multiplicative* in the two degrees, so it
                binds at far lower inputs than degree elevation's additive one. See
                ``include/pantr/bezier/compose.hpp`` for which
                configurations are exempt and why.

        Example:
            >>> f = Bezier(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]))
            >>> g = Bezier(np.array([[0.2], [0.8]]))
            >>> h = f.compose(g)
        """
        if self._dtype != inner._dtype:
            # The checks that come before the dtype one, for their messages and order; with equal
            # dtypes the compiled call makes them all.
            if self._is_rational:
                raise TypeError(
                    "Composition is not supported for rational Béziers (outer is rational)."
                )
            if inner._is_rational:
                raise TypeError(
                    "Composition is not supported for rational Béziers (inner is rational)."
                )
            if inner.rank != self.dim:
                raise ValueError(
                    f"Inner Bézier rank ({inner.rank}) must equal outer Bézier "
                    f"parametric dimension ({self.dim})."
                )
            raise ValueError(
                f"Operands must have the same dtype. Got {self._dtype} and {inner._dtype}."
            )
        return Bezier._from_handle(self._h.compose(inner._h))

    # ------------------------------------------------------------------
    # Reverse and permute
    # ------------------------------------------------------------------

    @overload
    def reverse(self, direction: int = ..., *, in_place: Literal[False] = ...) -> Bezier: ...

    @overload
    def reverse(self, direction: int = ..., *, in_place: Literal[True]) -> None: ...

    def reverse(self, direction: int = 0, *, in_place: bool = False) -> Bezier | None:
        """Reverse the orientation of one parametric direction.

        Flips the control points along the given parametric axis so that the
        mapping is reparametrised in the opposite sense along that direction.

        Args:
            direction (int): Parametric direction to reverse. Must be in
                ``[0, dim)``. Defaults to 0.
            in_place (bool): If ``True``, modify this Bézier in place and
                return ``None``. If ``False`` (default), return a new Bézier.

        Returns:
            Bezier | None: The reversed Bézier, or ``None`` when
            ``in_place=True``.

        Raises:
            ValueError: If ``direction`` is out of range ``[0, dim)``.

        Example:
            >>> import numpy as np
            >>> f = Bezier(np.array([[0.0], [1.0]]))  # f(t) = t
            >>> rev = f.reverse()
            >>> np.allclose(rev.evaluate(np.array([0.3])), [0.7])  # rev(t) = 1 - t
            True
            >>> f.reverse(in_place=True)  # mutates in place, returns None
            >>> np.allclose(f.evaluate(np.array([0.3])), [0.7])
            True
        """
        h = self._h.reversed(direction)
        if in_place:
            self._set_handle(h, self._shape, self._dtype)
            return None
        return Bezier._from_handle(h)

    @overload
    def permute_directions(
        self, permutation: Sequence[int], *, in_place: Literal[False] = ...
    ) -> Bezier: ...

    @overload
    def permute_directions(
        self, permutation: Sequence[int], *, in_place: Literal[True]
    ) -> None: ...

    def permute_directions(
        self, permutation: Sequence[int], *, in_place: bool = False
    ) -> Bezier | None:
        """Reorder the parametric directions according to a permutation.

        Given a permutation ``[i_0, i_1, …]``, the new direction ``k`` is
        the old direction ``permutation[k]``. For example, ``[1, 2, 0]`` on
        a 3D volume maps old direction 1 → new 0, old 2 → new 1, old 0 → new 2.

        Args:
            permutation (Sequence[int]): A permutation of ``range(dim)``.
            in_place (bool): If ``True``, modify this Bézier in place and
                return ``None``. If ``False`` (default), return a new Bézier.

        Returns:
            Bezier | None: The permuted Bézier, or ``None`` when
            ``in_place=True``.

        Raises:
            ValueError: If ``permutation`` is not a valid permutation of
                ``range(dim)``.

        Example:
            >>> import numpy as np
            >>> surf = Bezier(np.zeros((2, 3, 1)))
            >>> surf.degree
            (1, 2)
            >>> surf.permute_directions([1, 0]).degree  # swap u <-> v
            (2, 1)
        """
        perm = list(permutation)
        # Checked here for the message, which spells the Python list; the compiled call checks it
        # again.
        if sorted(perm) != list(range(self.dim)):
            raise ValueError(f"permutation must be a permutation of range({self.dim}), got {perm}.")
        h = self._h.permuted(perm)
        if in_place:
            self._set_handle(h)
            return None
        return Bezier._from_handle(h)

    # ------------------------------------------------------------------
    # Affine transformation
    # ------------------------------------------------------------------

    @overload
    def transform(self, affine: AffineTransform, *, in_place: Literal[False] = ...) -> Bezier: ...

    @overload
    def transform(self, affine: AffineTransform, *, in_place: Literal[True]) -> None: ...

    def transform(
        self,
        affine: AffineTransform,
        *,
        in_place: bool = False,
    ) -> Bezier | None:
        """Apply an affine transformation to the control points.

        For non-rational Bézier, every control point ``P`` is mapped to
        ``A @ P + b``.  For rational Bézier the weighted homogeneous
        coordinates are updated so that the projected geometry undergoes the
        same affine map while the weights are preserved.

        Args:
            affine (~pantr.transform.AffineTransform): The affine
                transformation to apply.
            in_place (bool): If ``True``, the control points are modified in
                place and ``None`` is returned.  If ``False`` (default), a
                new :class:`Bezier` is returned.

        Returns:
            Bezier | None: The transformed Bézier, or ``None`` when
            ``in_place=True``.

        Raises:
            ValueError: If the transform dimension does not match the
                geometric rank of the Bézier.

        Example:
            >>> import numpy as np
            >>> from pantr.transform import AffineTransform
            >>> curve = Bezier(np.array([[0.0, 0.0], [1.0, 1.0]]))  # rank 2
            >>> T = AffineTransform.translation([1.0, 2.0])
            >>> shifted = curve.transform(T)
            >>> np.allclose(shifted.control_points, [[1.0, 2.0], [2.0, 3.0]])
            True
        """
        h = self._h.transformed(
            np.ascontiguousarray(affine.matrix, dtype=np.float64),
            np.ascontiguousarray(affine.offset, dtype=np.float64),
        )
        if in_place:
            self._set_handle(h, self._shape, self._dtype)
            return None
        return Bezier._from_handle(h)

    # ------------------------------------------------------------------
    # Restrict
    # ------------------------------------------------------------------

    def restrict(
        self,
        bounds: tuple[float, float] | Sequence[tuple[float, float] | None],
    ) -> Bezier:
        """Return a Bézier restricted to a sub-region of ``[0, 1]^dim``.

        Extracts the portion of the Bézier defined on the given sub-interval
        and reparametrizes the result back to ``[0, 1]^dim``.  The returned
        Bézier has the same degree but different control points that encode
        the restricted mapping.

        Uses two de Casteljau passes per direction for direct Bernstein
        coefficient computation without B-spline conversion. The order
        of the passes is chosen for numerical stability.

        Args:
            bounds (tuple[float, float] | Sequence[tuple[float, float] | None]):
                For a 1D Bézier, a ``(lower, upper)`` tuple within ``[0, 1]``.
                For multi-dimensional Bézier, a sequence of length ``dim``
                where each element is a ``(lower, upper)`` tuple for that
                direction, or ``None`` to keep the full ``[0, 1]`` range.
                At least one direction must have non-``None`` bounds that
                restrict the domain.

        Returns:
            Bezier: New Bézier on ``[0, 1]^dim`` representing the restriction.

        Raises:
            ValueError: If the sequence length does not match ``dim``.
            ValueError: If all directions are ``None`` or match the full domain.
            ValueError: If any bound lies outside ``[0, 1]``.
            ValueError: If ``lower >= upper`` in any direction.
        """
        if self.dim == 1:
            bounds_per_dim: list[tuple[float, float] | None] = [
                bounds  # type: ignore[list-item]
            ]
        else:
            seq = list(bounds)  # type: ignore[arg-type,unused-ignore]
            if len(seq) != self.dim:
                raise ValueError(
                    f"bounds sequence length ({len(seq)}) must match dim ({self.dim})."
                )
            bounds_per_dim = seq  # type: ignore[assignment]

        # Validated here for the messages, which spell the bounds as given; the compiled call
        # validates them again.
        for i, b in enumerate(bounds_per_dim):
            if b is None:
                continue
            lower, upper = b
            if lower >= upper:
                raise ValueError(
                    f"Lower bound ({lower}) must be strictly less than upper bound ({upper}) "
                    f"in direction {i}."
                )
            if lower < 0.0 or upper > 1.0:
                raise ValueError(
                    f"Bounds ({lower}, {upper}) must lie within [0, 1] in direction {i}."
                )

        return Bezier._from_handle(
            self._h.restrict(
                [None if b is None else (float(b[0]), float(b[1])) for b in bounds_per_dim]
            )
        )

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def split(self, direction: int, value: float) -> tuple[Bezier, Bezier]:
        """Split the Bézier into two at a parameter value in one direction.

        Uses the de Casteljau algorithm to subdivide the Bézier into a
        left half (representing the original on ``[0, value]``) and a right
        half (representing the original on ``[value, 1]``), both
        reparametrized to ``[0, 1]``.

        Args:
            direction (int): Parametric direction along which to split.
                Must be in ``[0, dim)``.
            value (float): Parameter value at which to split.  Must lie
                strictly inside ``(0, 1)``.

        Returns:
            tuple[Bezier, Bezier]: A pair ``(left, right)`` of Béziers on
            ``[0, 1]^dim``.

        Raises:
            ValueError: If ``direction`` is out of range ``[0, dim)``.
            ValueError: If ``value`` is not strictly inside ``(0, 1)``.

        Example:
            >>> import numpy as np
            >>> curve = Bezier(np.array([[0.0], [1.0]]))  # f(t) = t
            >>> left, right = curve.split(0, 0.5)
            >>> np.allclose(left.evaluate(np.array([1.0])), [0.5])
            True
            >>> np.allclose(right.evaluate(np.array([0.0])), [0.5])
            True
        """
        if direction < 0 or direction >= self.dim:
            raise ValueError(f"direction must be in [0, {self.dim}), got {direction}.")
        # Checked here for the message, which spells the value as given.
        if value <= 0.0 or value >= 1.0:
            raise ValueError(f"value must be strictly inside (0, 1), got {value}.")
        left, right = self._h.split(direction, float(value))
        return Bezier._from_handle(left), Bezier._from_handle(right)

    # ------------------------------------------------------------------
    # Slice and boundary
    # ------------------------------------------------------------------

    def slice(self, axis: int, value: float) -> Bezier | npt.NDArray[np.float32 | np.float64]:
        """Slice the Bézier by fixing one parametric direction at a given value.

        Reduces the parametric dimension by one using the de Casteljau
        algorithm on the control points.  A surface becomes a curve, a curve
        becomes a point (returned as a NumPy array).

        At the boundary values ``0`` and ``1`` the result is obtained in
        O(1) by direct control point lookup.

        Args:
            axis (int): Parametric direction to fix (0-indexed).
                Must be in ``[0, dim)``.
            value (float): Parameter value at which to slice.  Must lie
                within ``[0, 1]``.

        Returns:
            Bezier | npt.NDArray[np.float32 | np.float64]:
            A Bézier with ``dim - 1`` dimensions when ``dim >= 2``,
            or a NumPy array of shape ``(rank,)`` when ``dim == 1``.
            Rational Béziers preserve the rational structure when ``dim >= 2``;
            for ``dim == 1`` the result is projected to physical coordinates.

        Raises:
            ValueError: If ``axis`` is out of range ``[0, dim)``.
            ValueError: If ``value`` is outside ``[0, 1]``.

        Example:
            >>> import numpy as np
            >>> cp = np.zeros((2, 2, 1))
            >>> cp[1, 1, 0] = 1.0  # f(u, v) = u * v
            >>> surf = Bezier(cp)
            >>> curve = surf.slice(1, 0.5)  # fix v=0.5: curve(u) = 0.5u
            >>> curve.dim
            1
            >>> pt = surf.slice(1, 0.5).slice(0, 0.2)  # surface -> curve -> point
            >>> np.allclose(pt, [0.1])
            True
        """
        if axis < 0 or axis >= self.dim:
            raise ValueError(f"axis must be in [0, {self.dim}), got {axis}.")
        # Checked here for the message, which spells the value as given.
        if value < 0.0 or value > 1.0:
            raise ValueError(f"value must be in [0, 1], got {value}.")
        if self.dim == 1:
            point: npt.NDArray[np.float32 | np.float64] = self._h.slice_point(axis, float(value))
            return point
        return Bezier._from_handle(self._h.slice(axis, float(value)))

    def boundary(self, axis: int, side: int) -> Bezier | npt.NDArray[np.float32 | np.float64]:
        """Extract the boundary of the Bézier along one parametric direction.

        Returns the restriction of the Bézier to one end of the ``[0, 1]``
        domain in the given direction.

        Args:
            axis (int): Parametric direction (0-indexed).
                Must be in ``[0, dim)``.
            side (int): Which end of the domain: ``0`` for the start,
                ``1`` for the end.

        Returns:
            Bezier | npt.NDArray[np.float32 | np.float64]:
            A Bézier with ``dim - 1`` dimensions when ``dim >= 2``,
            or a NumPy array of shape ``(rank,)`` when ``dim == 1``.

        Raises:
            ValueError: If ``axis`` is out of range ``[0, dim)``.
            ValueError: If ``side`` is not 0 or 1.

        Example:
            >>> import numpy as np
            >>> cp = np.zeros((2, 2, 1))
            >>> cp[1, 1, 0] = 1.0  # f(u, v) = u * v
            >>> surf = Bezier(cp)
            >>> right_edge = surf.boundary(0, 1)  # u = 1: f(1, v) = v
            >>> np.allclose(right_edge.evaluate(np.array([0.7])), [0.7])
            True
        """
        if side not in (0, 1):
            raise ValueError(f"side must be 0 or 1, got {side}.")
        if axis < 0 or axis >= self.dim:
            raise ValueError(f"axis must be in [0, {self.dim}), got {axis}.")

        value = 0.0 if side == 0 else 1.0
        return self.slice(axis, value)

    # ------------------------------------------------------------------
    # Collapse along axis
    # ------------------------------------------------------------------

    def collapse_along_axis(
        self,
        axis: int,
        values: npt.ArrayLike,
    ) -> Bezier:
        """Collapse to a univariate Bézier along one parametric direction.

        Fixes all parametric directions except ``axis`` at the given parameter
        values, producing a 1D Bézier whose control points are the Bernstein
        coefficients along ``axis``.  This is a tensor contraction: for each
        collapsed direction, the Bernstein basis is evaluated at the given
        value and contracted with the control point array.

        Args:
            axis (int): Parametric direction to keep (0-indexed).
                Must be in ``[0, dim)``.
            values (npt.ArrayLike): Parameter values for all directions
                except ``axis``.  Must have length ``dim - 1`` with all
                values in ``[0, 1]``.  ``values[i]`` corresponds to
                direction ``i`` for ``i < axis``, and direction ``i + 1``
                for ``i >= axis``.

        Returns:
            Bezier: A 1D Bézier with degree ``self.degree[axis]`` and
            the same rank and rationality as the input.

        Raises:
            ValueError: If ``dim < 2`` (nothing to collapse).
            ValueError: If ``axis`` is out of range ``[0, dim)``.
            ValueError: If ``values`` does not have length ``dim - 1``.
            ValueError: If any value is outside ``[0, 1]``.

        Example:
            >>> import numpy as np
            >>> cp = np.zeros((2, 2, 2, 1))
            >>> cp[1, 1, 1, 0] = 1.0  # f(u, v, w) = u * v * w
            >>> vol = Bezier(cp)
            >>> curve = vol.collapse_along_axis(1, [0.3, 0.7])  # fix u=0.3, w=0.7
            >>> curve.degree
            (1,)
            >>> np.allclose(curve.evaluate(np.array([0.5])), [0.3 * 0.5 * 0.7])
            True
        """
        if self.dim < 2:  # noqa: PLR2004
            raise ValueError("collapse_along_axis requires dim >= 2.")
        if axis < 0 or axis >= self.dim:
            raise ValueError(f"axis must be in [0, {self.dim}), got {axis}.")

        values_arr = np.asarray(values, dtype=self._dtype)
        # Checked here for the message, which spells the array's shape.
        if values_arr.ndim != 1 or values_arr.shape[0] != self.dim - 1:
            raise ValueError(
                f"values must have length dim - 1 = {self.dim - 1}, got shape {values_arr.shape}."
            )
        return Bezier._from_handle(
            self._h.collapse_along_axis(axis, np.ascontiguousarray(values_arr))
        )

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_bspline(self, *, copy: bool = True) -> Bspline:
        """Convert to an equivalent B-spline with Bézier knot vectors.

        Creates a :class:`~pantr.bspline.Bspline` with open knot vectors
        ``[0]*(p+1) + [1]*(p+1)`` in each parametric direction.

        Args:
            copy (bool): If ``True`` (default), the control points are
                deep-copied into the new B-spline. If ``False``, the
                B-spline holds this Bézier's read-only control point view
                (shared memory, so the B-spline cannot be changed in place).

        Returns:
            ~pantr.bspline.Bspline: Equivalent B-spline representation.
        """
        from ..bspline import Bspline as BsplineCls  # noqa: PLC0415
        from ..bspline import BsplineSpace, BsplineSpace1D  # noqa: PLC0415

        dtype = self.dtype
        spaces: list[BsplineSpace1D] = []
        for p in self.degree:
            knots = np.zeros(2 * (p + 1), dtype=dtype)
            knots[p + 1 :] = 1.0
            spaces.append(BsplineSpace1D(knots, p))

        cp = self.control_points.copy() if copy else self.control_points
        return BsplineCls(BsplineSpace(spaces), cp, self._is_rational)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(
        self,
        *,
        color: str | None = None,
        show_control_polygon: bool = False,
        **plotter_kwargs: Any,  # noqa: ANN401
    ) -> object:
        """Quick interactive visualization of this Bézier (requires pyvista).

        For finer control, use ``pantr.viz.Scene`` directly.

        Args:
            color: Surface color.
            show_control_polygon: Render control polygon (points and wireframe).
            **plotter_kwargs: Additional keyword arguments for ``pv.Plotter()``.

        Returns:
            object: The pyvista ``Plotter`` after showing.

        Raises:
            ImportError: If pyvista is not installed.
        """
        from ..viz import plot as _plot  # noqa: PLC0415

        return _plot(
            self,
            color=color,
            show_control_polygon=show_control_polygon,
            **plotter_kwargs,
        )


def create_from_bspline(bspline: Bspline, *, copy: bool = True) -> Bezier:
    """Create a Bézier from a B-spline with Bézier-like knot vectors.

    Validates that the B-spline has Bézier-like knots (open knots with
    ``num_basis == degree + 1`` in each direction) and extracts the
    control points.

    Args:
        bspline (~pantr.bspline.Bspline): A B-spline with Bézier-like
            knot structure.
        copy (bool): If ``True`` (default), the control points are
            deep-copied into the new Bézier. If ``False``, the Bézier
            shares the same underlying control point array (read-only
            through the Bézier) when it is C-contiguous float32 or float64;
            another layout or dtype is converted, which shares nothing.

    Returns:
        Bezier: The equivalent Bézier.

    Raises:
        ValueError: If the B-spline does not have Bézier-like knots.
    """
    if not bspline.space.has_Bezier_like_knots():
        raise ValueError("B-spline does not have Bézier-like knots. Cannot convert to Bézier.")
    if copy:
        return Bezier(bspline.control_points, is_rational=bspline.is_rational)
    return Bezier._borrow(bspline.control_points, bspline.is_rational)


_VALUE_ATTRS = frozenset({"_h", "_shape", "_is_rational", "_dtype", "control_points"})
"""The attributes that hold the immutable value and what is cached from it (shared by copies)."""


def _extra_state(b: Bezier) -> dict[str, Any]:
    """Get a Bézier's attributes other than its value, for pickling a subclass.

    Args:
        b (Bezier): The Bézier.

    Returns:
        dict[str, Any]: The other attributes (empty for a plain Bézier).
    """
    return {k: v for k, v in b.__dict__.items() if k not in _VALUE_ATTRS}


def _as_control_points(control_points: npt.ArrayLike) -> npt.NDArray[np.float32 | np.float64]:
    """Normalise control points for the compiled constructor.

    Args:
        control_points (npt.ArrayLike): The control points.

    Returns:
        npt.NDArray[np.float32 | np.float64]: A C-contiguous native ``float32`` or ``float64``
        array of rank at least 2.

    Raises:
        ValueError: If the input is a 0-d array.
        TypeError: If the input is complex.
    """
    cp = np.asarray(control_points)
    dtype = cp.dtype
    if (dtype is _F64 or dtype is _F32) and cp.ndim >= 2 and cp.flags.c_contiguous:  # noqa: PLR2004
        return cp  # the common case, already normalised
    if cp.ndim < 1:
        raise ValueError("Control points must be at least 1D.")
    if cp.ndim == 1:
        cp = cp[:, np.newaxis]
    kind = cp.dtype.kind
    if kind == "c":
        raise TypeError(f"Bézier control points must be real; got dtype {cp.dtype}.")
    target = _F32 if kind == "f" and cp.dtype.itemsize == 4 else _F64  # noqa: PLR2004
    return np.ascontiguousarray(cp, dtype=target)


def _class_for(dtype: np.dtype[Any]) -> Any:  # noqa: ANN401
    """Get the compiled class of a dtype.

    Args:
        dtype (np.dtype): ``float32`` or ``float64``.

    Returns:
        Any: ``BezierF32`` or ``BezierF64``.
    """
    return _cpp.BezierF32 if dtype == _F32 else _cpp.BezierF64


def _finish(
    raw: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64] | None,
    dtype: np.dtype[Any],
) -> npt.NDArray[np.float32 | np.float64]:
    """Drop a unit rank axis and honour ``out``, as the previous release did.

    Args:
        raw (npt.NDArray[np.float32 | np.float64]): The compiled result, rank axis last.
        out (npt.NDArray[np.float32 | np.float64] | None): The caller's output array, if any.
        dtype (np.dtype): The Bézier's dtype.

    Returns:
        npt.NDArray[np.float32 | np.float64]: ``out`` filled, or the result.

    Raises:
        ValueError: If ``out`` has the wrong shape or dtype, or is not writeable.
    """
    result = raw[..., 0] if raw.shape[-1] == 1 else raw
    if out is None:
        return result
    _validate_out_array(out, result.shape, dtype)
    out[...] = result
    return out
