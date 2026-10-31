"""B-spline geometric objects: the Bspline class and evaluation helpers.

This module provides :class:`Bspline`, which pairs a
:class:`~pantr.bspline.BsplineSpace` with control points to represent a
parametric B-spline curve, surface, or volume. The value lives in the C++ library
(``pantr::bspline::Bspline``); evaluation at arbitrary points is done there too.
"""

from __future__ import annotations

import copy as _copy
import functools
import operator
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from numpy import typing as npt

from .._ext import ext
from .._transform_control_points import _apply_affine_to_control_points
from ._bspline_derivative import _derivative_bspline
from ._bspline_eval import _evaluate_Bspline, _evaluate_Bspline_deriv
from ._bspline_locate import _locate_impl
from ._bspline_slice import _slice_bspline
from ._bspline_to_beziers import _to_beziers_impl

_cpp: Any = ext.bspline
"""The compiled B-splines (private; untyped: one class per dtype)."""

_F32: np.dtype[np.float32] = np.dtype(np.float32)
_F64: np.dtype[np.float64] = np.dtype(np.float64)

if TYPE_CHECKING:
    from ..bezier import Bezier
    from ..quad import PointsLattice
    from ..transform import AffineTransform
    from ._bspline_locate import _LocateContext
    from ._bspline_space_nd import BsplineSpace


class Bspline:
    """A parametric B-spline curve/surface defined by a space and control points.

    Combines a :class:`~pantr.bspline.BsplineSpace` (knot vectors, degrees)
    with a set of control points to represent a B-spline mapping.

    The value is immutable: the control points are copied in at construction and read-only, and
    the operations that take ``in_place=True`` replace the compiled value this object holds rather
    than writing into it. An array obtained from :attr:`control_points` before such an operation
    keeps the values it showed.

    Attributes:
        _h (Any): The compiled value (``BsplineF64`` or ``BsplineF32``): the space and the
            control points.
        _space (pantr.bspline.BsplineSpace): The multi-dimensional B-spline space, the object the
            B-spline was built from (so ``spline.space is`` that object).
        _shape (tuple[int, ...]): Control point shape ``(*num_basis, cp_size)``, where the last
            axis includes the homogeneous weight for rational.
        _is_rational (bool): Whether the B-spline is rational (NURBS).
        _dtype (np.dtype): ``float32`` or ``float64``.
        _beziers_cache (``npt.NDArray[np.object_] | None``): Cached Bézier
            decomposition, or ``None`` if not yet computed.
        _locate_cache (``_LocateContext | None``): Cached point-inversion state
            (see :meth:`locate`), or ``None`` if not yet computed. Invalidated
            alongside ``_beziers_cache`` by the in-place mutators.

    The read-only control point view is read from ``_h`` on first use and kept in the instance
    (``control_points`` is a cached property that refuses assignment).
    """

    _h: Any
    _space: BsplineSpace
    _shape: tuple[int, ...]
    _is_rational: bool
    _dtype: np.dtype[np.float32] | np.dtype[np.float64]
    _beziers_cache: npt.NDArray[np.object_] | None
    _locate_cache: _LocateContext | None

    def __init__(
        self, space: BsplineSpace, control_points: npt.ArrayLike, is_rational: bool = False
    ) -> None:
        """Initialize a B-spline.

        Args:
            space (~pantr.bspline.BsplineSpace): The B-spline space.
            control_points (npt.ArrayLike): The control points: ``num_total_basis * cp_size``
                values, in any shape that flattens to ``(*num_basis, cp_size)`` in C order. They
                are copied, and must have the dtype of the space.
            is_rational (bool): Whether the B-spline is rational.

        Raises:
            ValueError: If the number of control points is not a multiple
                of the number of basis functions.
            ValueError: If the control points dtype does not match the
                B-spline space dtype.
            ValueError: If the B-spline has rank smaller than 1.
        """
        arr = np.asarray(control_points)
        dtype = np.dtype(space.dtype)  # IndexError for a space with no direction, as before
        cls = _class_for(dtype)
        # The count is checked before the dtype, as before; the compiled constructor checks it
        # again.
        cls.cp_size_for(space._h, int(arr.size))
        if arr.dtype != dtype:
            raise ValueError(
                f"The control points must have the same dtype as the B-spline space."
                f"Got {arr.dtype} control points and {dtype} B-spline space."
            )
        self._set_handle(cls(space._h, np.ascontiguousarray(arr), bool(is_rational)), space)

    def _set_handle(
        self,
        h: Any,  # noqa: ANN401
        space: BsplineSpace,
    ) -> None:
        """Hold a compiled value and cache what the properties return (DR-25).

        Args:
            h (Any): The compiled value.
            space (BsplineSpace): The space object it was built from.
        """
        d = self.__dict__
        d["_h"] = h
        d["_space"] = space
        d["_shape"] = tuple(h.shape)
        d["_is_rational"] = bool(h.is_rational)
        d["_dtype"] = _F32 if isinstance(h, _cpp.BsplineF32) else _F64
        d["_beziers_cache"] = None
        d["_locate_cache"] = None
        # Drop the control-point view of the previous value; the next read takes the new one.
        d.pop("control_points", None)

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle through the space and control points, with a subclass's attributes as state.

        Returns:
            tuple[Any, ...]: The rebuilding function, its arguments and the extra attributes.
        """
        return (
            _rebuild_bspline,
            (type(self), self._space, self.control_points, self._is_rational),
            _extra_state(self),
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore extra attributes, or rebuild from a previous release's pickle.

        A previous release pickled the plain attributes ``_space``, ``_control_points`` and
        ``_is_rational`` (and the caches); such a state rebuilds the compiled value from them and
        drops the caches.

        Args:
            state (dict[str, Any]): The pickled state.
        """
        state = dict(state)
        if "_control_points" in state:
            cp = state.pop("_control_points")
            space = state.pop("_space")
            Bspline.__init__(self, space, cp, bool(state.pop("_is_rational")))
            state.pop("_beziers_cache", None)
            state.pop("_locate_cache", None)
        self.__dict__.update(state)

    def __copy__(self) -> Bspline:
        """Copy: the new object shares the immutable compiled value and the space.

        Returns:
            Bspline: The copy, of the same class, with the same attributes.
        """
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo: dict[int, Any]) -> Bspline:
        """Deep copy: shares the immutable compiled value and deep-copies the rest.

        The caches are not copied; the copy rebuilds them on demand.

        Args:
            memo (dict[int, Any]): The deep-copy memo.

        Returns:
            Bspline: The copy.
        """
        new = object.__new__(type(self))
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            if key in _VALUE_ATTRS:
                new.__dict__[key] = value
            elif key in _CACHE_ATTRS:
                new.__dict__[key] = None
            else:
                new.__dict__[key] = _copy.deepcopy(value, memo)
        return new

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
    def dim(self) -> int:
        """Get the parametric dimension of the B-spline.

        Returns:
            int: Number of parametric dimensions (equals the dimension of the
            underlying B-spline space).
        """
        return len(self._shape) - 1

    @property
    def degree(self) -> tuple[int, ...]:
        """Get the B-spline degrees per parametric direction.

        Returns:
            tuple[int, ...]: Polynomial degree for each parametric dimension.
        """
        return self._space.degrees

    @property
    def space(self) -> BsplineSpace:
        """Get the underlying B-spline space.

        Returns:
            ~pantr.bspline.BsplineSpace: The multi-dimensional
            B-spline space defining the knot vectors and polynomial degrees.
        """
        return self._space

    @functools.cached_property
    def control_points(self) -> npt.NDArray[np.float32 | np.float64]:
        """Get the control points of the B-spline.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Control point array with
            shape ``(*num_basis, rank)``, read-only: a view of the immutable value, which keeps
            the values it showed when it was read even after an in-place operation on this
            object.
        """
        # Read once and kept in the instance dict, so later reads cost no call; `__setattr__` and
        # `__delattr__` refuse the name, as they did for the plain property it replaces.
        cp: npt.NDArray[np.float32 | np.float64] = self._h.control_points
        return cp

    @property
    def is_rational(self) -> bool:
        """Check whether the B-spline is rational (NURBS).

        Returns:
            bool: True if the B-spline is rational (i.e., the last control point
            coordinate is a homogeneous weight), False otherwise.
        """
        return self._is_rational

    @property
    def rank(self) -> int:
        """Get the output rank of the B-spline.

        The rank is the number of value dimensions produced by the mapping.
        For a scalar field it is 1; for a 3D curve it is 3.  For rational
        B-splines the weight coordinate is excluded.

        Returns:
            int: Output rank of the B-spline.
        """
        rk = self._shape[-1]
        return rk - 1 if self._is_rational else rk

    @property
    def dtype(self) -> npt.DTypeLike:
        """Get the floating-point dtype of the B-spline.

        Returns:
            npt.DTypeLike: The numpy dtype (float32 or float64) of the control
            point array.
        """
        return self._dtype

    def _adopt(self, other: Bspline) -> None:
        """Replace the compiled value (and space) this object holds by another's.

        The in-place operations build the new B-spline and swap it in, so an array read from
        :attr:`control_points` earlier keeps the old values.

        Args:
            other (Bspline): The B-spline whose value is taken.
        """
        self._set_handle(other._h, other._space)

    def _from_result(self, h: Any) -> Bspline:  # noqa: ANN401
        """Wrap a compiled value an operation returned as a new B-spline.

        The directions the operation left alone keep their Python space objects.

        Args:
            h (Any): The compiled B-spline (of this B-spline's dtype).

        Returns:
            Bspline: A plain ``Bspline`` over the value, whose space is built from the
            returned directions.
        """
        from ._bspline_space_1d import BsplineSpace1D  # noqa: PLC0415
        from ._bspline_space_nd import BsplineSpace  # noqa: PLC0415

        old = self._space.spaces
        spaces = [
            o if n.same_rep(o._h) else BsplineSpace1D._from_handle(n)
            for n, o in zip(h.directions, old, strict=True)
        ]
        result = object.__new__(Bspline)
        result._set_handle(h, BsplineSpace(spaces))
        return result

    def evaluate(
        self,
        pts: npt.ArrayLike | PointsLattice,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Evaluate the B-spline at the given points.

        Args:
            pts (npt.ArrayLike | PointsLattice): The
                parametric points at which to evaluate the B-spline, shaped
                ``(n_pts, dim)`` or, for a 1D B-spline, either ``(n_pts,)`` or
                ``(n_pts, 1)``. Array-likes are converted; the dtype must match
                the B-spline's.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                array where the result will be stored. If None, a new array is
                allocated. This follows NumPy's style for output arrays.
                Defaults to None.

        Returns:
            npt.NDArray[np.float32 | np.float64]: B-spline values at the given
            points.

        Raises:
            ValueError: If the points have an unusable shape, if their dtype does
                not match the B-spline dtype, or if `out` has incorrect shape or
                dtype.

        Note:
            Special values are not guarded. A rational B-spline divides by the
            interpolated weight: a zero weight gives ``inf`` or ``nan`` and a ``nan``
            weight gives ``nan``, in the affected values only, with no error or
            warning. A ``nan`` control point reaches every value whose supporting
            functions include it, a zero basis value there included. A ``nan`` point
            is outside the domain and raises.
        """
        return _evaluate_Bspline(self, pts, out)

    def evaluate_derivatives(
        self,
        pts: npt.ArrayLike | PointsLattice,
        orders: int | Sequence[int],
        out: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Evaluate a specific partial derivative of the B-spline.

        Computes the single partial derivative specified by ``orders``,
        where ``orders[d]`` is the derivative order in parametric direction ``d``.
        For rational B-splines the generalised quotient rule is applied so that
        the returned values are derivatives of the projected mapping.

        Args:
            pts (npt.ArrayLike | PointsLattice): The
                parametric points at which to evaluate. For 1D B-splines, an
                array of shape ``(n_pts,)`` or ``(n_pts, 1)``, or a 1D
                :class:`~pantr.quad.PointsLattice`. For multi-dimensional
                B-splines, a 2D array of shape ``(n_pts, dim)`` or a
                :class:`~pantr.quad.PointsLattice` with matching dimension.
                Array-likes are converted; the dtype must match the B-spline's.
            orders (int | Sequence[int]): Derivative order(s). A single
                ``int`` is broadcast to all ``self.dim`` directions. A sequence
                must contain one non-negative integer per parametric direction
                (``len(orders) == self.dim``). Pass ``0`` (or ``[0, ..., 0]``)
                to obtain the function value (equivalent to :meth:`evaluate`).
            out (npt.NDArray[np.float32 | np.float64] | None): Optional
                pre-allocated output array with the same shape and dtype as the
                returned array (see below). Filled in-place and returned.
                Defaults to None.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Mixed partial derivative
            values. Shape is ``(*pts_base_shape,)`` for scalar output or
            ``(*pts_base_shape, rank)`` for vector-valued output, where
            ``pts_base_shape`` is ``(n_pts,)`` for a points array or
            ``(*pts_grid_shape,)`` for a :class:`~pantr.quad.PointsLattice`.
            For rational B-splines the weight column is divided out and not
            included in the output.

        Raises:
            ValueError: If ``len(orders) != self.dim``, if any order is
                negative, if the points dtype does not match the B-spline dtype,
                or if ``out`` has incorrect shape or dtype. The orders are checked
                before the points.
            TypeError: If an order is not an integer (anything without
                ``__index__``).
            ZeroDivisionError: If the basis derivative recurrence divides by zero,
                which a knot repeated more than ``degree + 1`` times within a
                point's support produces; ``out`` is then unwritten.

        Note:
            Special values are not guarded. The quotient rule of a rational B-spline
            divides by the interpolated weight: a zero weight gives ``inf`` or
            ``nan`` and a ``nan`` weight gives ``nan``, in the affected values only,
            with no error or warning. A ``nan`` control point reaches every value
            whose supporting functions include it. A ``nan`` point is outside the
            domain and raises.

        Example:
            >>> import numpy as np
            >>> from pantr.bspline import create_uniform_space
            >>> spline = Bspline(create_uniform_space(2, 1), np.array([[0.0], [0.0], [1.0]]))
            >>> pts = np.array([[0.25], [0.75]])
            >>> np.allclose(spline.evaluate_derivatives(pts, 2), [2.0, 2.0])  # f(t) = t**2
            True
            >>> np.allclose(spline.evaluate_derivatives(pts, [2]), [2.0, 2.0])  # sequence form
            True
            >>> u_cp, v_cp = np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.0, 1.0])
            >>> srf = Bspline(create_uniform_space(2, [1, 1]), np.outer(u_cp, v_cp)[..., None])
            >>> uv = np.array([[0.3, 0.4]])
            >>> np.allclose(srf.evaluate_derivatives(uv, [1, 2]), [2.0])  # f = u*v**2, d3f/du dv2
            True
        """
        orders_seq: Sequence[int] = [orders] * self.dim if isinstance(orders, int) else orders
        return _evaluate_Bspline_deriv(self, pts, orders_seq, out)

    def locate(
        self,
        points: npt.ArrayLike,
        *,
        tol: float | None = None,
        max_iter: int = 30,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Invert the mapping: find the parametric coordinates of physical points.

        The inverse of :meth:`evaluate`. Candidate knot-span cells come from a cached
        hierarchy over the per-cell control-point boxes (the convex-hull property), and
        each candidate is refined by Newton's method on ``F(xi) - x = 0``. The
        hierarchy is built on the first call and reused afterwards.

        Only square mappings are inverted (``rank == dim``), which covers planar and
        volumetric geometry maps. For a rational B-spline every weight must be
        strictly positive: the candidate search relies on the projected control points
        bounding the cell's image, which is false for a non-positive weight.

        Args:
            points (npt.ArrayLike): Physical points, shape ``(n, rank)``. A 1-D input is
                one point of ``rank`` coordinates, except when ``rank == 1``, where it is
                ``n`` scalar coordinates.
            tol (float | None): Convergence threshold on ``||F(xi) - x||_2``, as an
                absolute distance in physical units, governing where the iteration stops
                as well as what counts as found. ``None`` (default) separates the two:
                :func:`pantr.tolerance.get_default` for this B-spline's dtype times the
                larger of the control-point bounding-box diagonal and the largest
                coordinate magnitude is where the iteration stops, and the same tier
                times that length maximised against the physical length one ulp of a
                parametric coordinate spans is what counts as found. The second is what
                keeps the verdict reachable for a knot vector far from the parametric
                origin, whose coordinates a float64 resolves only to ``eps * |xi|``; the
                first is what keeps the answer as accurate as the geometry allows anyway.
                They are equal for a knot vector spanning ``[0, L]``. Defaults to None.
            max_iter (int): Maximum number of Newton steps per candidate cell.
                Defaults to 30.

        Returns:
            tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]: ``(cell_ids,
            ref_coords)``, both read-only. ``cell_ids`` has shape ``(n,)`` and holds
            flat knot-span cell ids in C-order over ``space.num_intervals`` -- the
            convention of :func:`pantr.grid.tensor_product_grid` and
            :class:`SpanwiseElementExtraction`. ``ref_coords`` has shape ``(n, dim)``
            and holds *parametric* (not cell-local) coordinates satisfying
            ``evaluate(ref_coords[i]) == points[i]`` within ``tol``, or -- when ``tol`` is
            ``None`` -- within the acceptance threshold described under ``tol`` above. That
            is the bar a coordinate is certified against and not usually the accuracy it
            carries, which is the tighter of the two thresholds wherever the map can deliver
            it. A point not on the mapping's image gets ``cell_ids[i] = -1`` and
            ``ref_coords[i] = nan``.

        Raises:
            NotImplementedError: If ``rank > dim`` (an embedded curve or surface, which
                needs a Gauss-Newton closest-point solve).
            ValueError: If ``rank < dim``, any direction is periodic, a rational
                B-spline has a non-positive weight, ``points`` has the wrong shape or a
                non-finite entry, ``max_iter < 1``, or ``tol`` is not positive and
                finite. Also when ``tol`` is ``None`` and the knot vector sits so far
                from the parametric origin that one ulp of a coordinate moves the image
                by more than the geometry's own size, since no default threshold can
                mean anything there; pass an explicit ``tol`` to ask anyway.

        Note:
            What is guaranteed is that ``evaluate(ref_coords[i])`` reproduces
            ``points[i]``, not that ``ref_coords[i]`` is a particular preimage. A
            mapping that folds (its Jacobian determinant changes sign) sends several
            parametric points to the same physical point, and any of them is a correct
            answer; only for an injective mapping does :meth:`locate` invert
            :meth:`evaluate` back to the coordinates it was called with.

            A point on an interior knot line legitimately belongs to either adjacent
            cell; ``ref_coords`` is unique but the reported ``cell_ids`` entry is
            whichever cell :meth:`pantr.grid.TensorProductGrid.locate_many` assigns.

            The inversion always runs in ``float64``: a ``float32`` B-spline is promoted
            to an exact ``float64`` copy, since a ``float32`` residual cannot be resolved
            below the ``float32`` tolerance preset. Evaluating the returned coordinates
            on the original ``float32`` B-spline therefore reproduces the query point to
            ``float32`` accuracy only.

            The default *acceptance* threshold grows with how far the knot vector sits
            from the parametric origin, because that is how far one ulp of a coordinate
            moves the image. A "found" verdict on such a geometry therefore asserts less
            than the same verdict on one parametrized from the origin, though the
            coordinate itself is still refined as far as the map allows. It stays a small
            fraction of the geometry for any ordinary knot vector; at a reach of ``1e13`` -- a knot
            span ten trillion times further from the origin than it is wide -- it reaches
            ``0.14`` of the bounding-box diagonal, and a point ten percent outside the
            image is reported found. Past ``1 / (64 * eps)`` the default is refused
            outright rather than served. Pass an explicit ``tol`` wherever the distance
            matters more than the default's derivation.

        Example:
            >>> import numpy as np
            >>> from pantr.bspline import create_uniform_space
            >>> u, v = np.meshgrid([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], indexing="ij")
            >>> surface = Bspline(create_uniform_space(1, [2, 2]), np.stack([u, v], axis=-1))
            >>> pts = np.array([[0.25, 0.75], [0.6, 0.3]])
            >>> cell_ids, ref_coords = surface.locate(surface.evaluate(pts))
            >>> cell_ids.tolist()
            [1, 2]
            >>> np.allclose(ref_coords, pts)
            True
        """
        return _locate_impl(self, points, tol, max_iter)

    def derivative(self, direction: int = 0, *, keep_degree: bool = False) -> Bspline:
        """Return a B-spline representing the first derivative in the given direction.

        Computes the hodograph: a new B-spline whose value at every parametric
        point equals the partial derivative of this B-spline with respect to
        parametric direction ``direction``.

        For non-rational B-splines of degree ``p`` in direction ``d``, the
        result has degree ``p - 1`` in direction ``d`` and the same degree in
        all other directions (or ``p`` when ``keep_degree=True``).

        For rational B-splines (NURBS), the quotient rule is applied, producing
        a rational B-spline of degree ``2p`` in direction ``d``. That already
        exceeds ``p``, so ``keep_degree`` has no effect on a rational B-spline:
        both settings return degree ``2p``.

        Args:
            direction (int): Parametric direction for differentiation.
                Must be in ``[0, dim)``. Defaults to 0.
            keep_degree (bool): If ``True``, a non-rational result preserves the
                degree of the original B-spline by applying degree elevation after
                differentiation. This is useful, for instance, when computing
                derivatives of rational polynomials (in the numerator). It is
                ignored for a rational B-spline, whose derivative is already of
                higher degree than the original. Defaults to ``False``.

        Returns:
            Bspline: A new B-spline representing the derivative.

        Raises:
            ValueError: If ``direction`` is out of range ``[0, dim)``.
            ValueError: If the degree in the given direction is 0.
            ValueError: If the degree the result has to be built at exceeds the
                exactness envelope of the binomial-coefficient kernel
                (``bincoeff_max_n``, 61). Reachable with ``keep_degree=True``, and
                for a rational B-spline with either setting, since that path
                always re-elevates.
            ValueError: If the call re-elevates and the differentiated direction's
                knot vector is not clamped, meaning a run of ``degree + 1`` equal
                knots at each end. Re-elevating is ``keep_degree=True`` and any
                rational field; it runs Piegl and Tiller A5.9, which assumes those
                runs and without them returns a different function. Two things stay
                available on an unclamped direction: a non-rational
                ``keep_degree=False``, which asks nothing of the ends, and a
                direction of degree 1, whose hodograph is degree 0 and so has no
                end runs to lack.
            ValueError: If the call re-elevates and the differentiated direction has
                an interior knot of multiplicity ``degree + 1`` (a ``C^-1`` knot): the
                hodograph then repeats that knot more than its own degree allows, and
                A5.9 cannot walk it.
            ValueError: If the B-spline is rational and the differentiated direction
                is periodic with a C^0 seam (seam multiplicity equal to the degree).
                The derivative is discontinuous at the seam, so it has no periodic
                form.

        Example:
            >>> import numpy as np
            >>> from pantr.bspline import create_uniform_space
            >>> f = Bspline(create_uniform_space(2, 1), np.array([[0.0], [0.0], [1.0]]))
            >>> fp = f.derivative()  # first derivative of a 1D curve; f(t) = t**2
            >>> fp.degree
            (1,)
            >>> np.allclose(fp.evaluate(np.array([[0.3]])), [0.6])  # f'(t) = 2t
            True
            >>> f.derivative().derivative().degree  # second derivative (composable)
            (0,)
            >>> f.derivative(keep_degree=True).degree  # derivative preserving degree
            (2,)
            >>> u_cp, v_cp = np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.0, 1.0])
            >>> srf = Bspline(create_uniform_space(2, [1, 1]), np.outer(u_cp, v_cp)[..., None])
            >>> srf.derivative(direction=1).degree  # partial derivative along direction 1
            (2, 1)
        """
        if direction < 0 or direction >= self.dim:
            raise ValueError(f"direction must be in [0, {self.dim}), got {direction}.")
        if self.space.spaces[direction].degree < 1:
            raise ValueError("Derivative of a degree-0 B-spline is not defined.")
        return _derivative_bspline(self, direction, keep_degree=keep_degree)

    def elevate_degree(self, degree_increments: int | Sequence[int]) -> Bspline:
        """Elevate the polynomial degree of the B-spline.

        Creates a new B-spline that represents the same mapping as the original
        one but with higher-order polynomial basis functions. This is achieved
        by increasing the degree in each parametric direction and adjusting the
        control points and knot vectors accordingly.

        Args:
            degree_increments (int | Sequence[int]): Number of degrees to
                increase. If an integer, the same increment is applied to all
                parametric directions. If a sequence, must have length equal
                to the B-spline dimension.

        Returns:
            Bspline: A new B-spline with elevated degrees.

        Raises:
            ValueError: If any degree increment is negative.
            ValueError: If all degree increments are zero.
            ValueError: If the number of increments does not match the dimension.
            ValueError: If an elevated degree would exceed the exactness envelope
                of the binomial-coefficient kernel (``bincoeff_max_n``, 61).
            ValueError: If a direction to be elevated is not periodic and its knot
                vector is not clamped, meaning a run of ``degree + 1`` equal knots
                at each end. Elevation is Piegl and Tiller A5.9, which assumes those
                runs: without the closing one its segment walk reads past the
                control points, and without the opening one it returns a different
                function. A periodic direction is converted to its open form first
                and is unaffected.
            ValueError: If a knot of a direction to be elevated repeats more than
                ``degree + 1`` times (a space accepts one; A5.9 cannot walk it).

        References:
            Degree elevation of spline curves :cite:p:`piegl1997nurbs`.
        """
        if isinstance(degree_increments, int):
            increments = (degree_increments,) * self.dim
        else:
            increments = tuple(degree_increments)

        if len(increments) != self.dim:
            raise ValueError(
                f"Number of degree increments ({len(increments)}) "
                f"must match dimension ({self.dim})."
            )

        if any(inc < 0 for inc in increments):
            raise ValueError("Degree increments must be non-negative.")

        if all(inc == 0 for inc in increments):
            raise ValueError("At least one degree increment must be positive.")

        return self._from_result(self._h.elevate_degree([_index(inc) for inc in increments]))

    def reduce_degree(self, degree_decrements: int | Sequence[int]) -> Bspline:
        r"""Reduce the polynomial degree of the B-spline.

        Decomposes to Bézier segments, reduces each segment, and coarsens the
        knot vector to restore the original continuity structure.

        Unlike :meth:`elevate_degree`, this operation is **not exact** in
        general: the result is an approximation of the original mapping.

        Each Bézier segment is reduced on its own, to the closest lower-degree
        polynomial in :math:`L^2` among those reproducing the segment's endpoint
        values.  Adjacent segments therefore agree at every breakpoint bit for
        bit and the stitch is exactly C0, with neither side moved off its own
        optimum.  Knots are then removed to restore the original continuity
        structure, and that step, not the reduction, is what sets the final
        error on an open direction: it is a forced removal with no deviation
        bound, so the exact per-segment figure
        :meth:`~pantr.bezier.Bezier.degree_reduction_error` reports does not
        carry over to the assembled spline.

        A **periodic** direction has a second such step. The reduction runs on
        the open form and the periodic seam is then closed by a least-squares
        projection, also with no deviation bound, because the reduced spline no
        longer lies in the periodic subspace exactly -- for a smooth periodic
        spline that projection is percent-level, not round-off. So on a periodic
        direction neither of the two unbounded steps dominates by construction,
        and the error has to be measured rather than attributed.

        Args:
            degree_decrements (int | Sequence[int]): Number of degrees to
                reduce. If an integer, the same decrement is applied to all
                parametric directions. If a sequence, must have length equal
                to the B-spline dimension.

        Returns:
            Bspline: A new B-spline with reduced degrees.

        Raises:
            ValueError: If any degree decrement is negative.
            ValueError: If all degree decrements are zero.
            ValueError: If the number of decrements does not match the dimension.
            ValueError: If any decrement exceeds the current degree in that
                direction.
            ValueError: If the object is rational and the reduction drives a
                control weight to zero or below.
            ValueError: If a direction to be reduced is not periodic and its knot
                vector is not clamped, or a knot of it repeats more than
                ``degree + 1`` times.
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

        return self._from_result(self._h.reduce_degree([_index(dec) for dec in decrements]))

    def insert_knots(
        self,
        new_knots: npt.ArrayLike | Sequence[npt.ArrayLike | None],
    ) -> Bspline:
        """Return a geometrically equivalent B-spline with additional knots inserted.

        Args:
            new_knots (npt.ArrayLike | Sequence[npt.ArrayLike | None]):
                For a 1D B-spline, a flat non-empty 1D array-like of knot values
                to insert.  For multi-dimensional B-splines, a sequence of length
                ``dim`` where each element is a 1D array-like of knots to insert
                in that direction, or ``None`` to skip that direction.  At least
                one direction must have a non-empty array of knots to insert.
                Repeated values in an array insert the same knot multiple times.

        Returns:
            Bspline: New B-spline with the same geometry and refined knot vectors.

        Raises:
            ValueError: If the sequence length does not match ``dim`` (multi-dim case).
            ValueError: If all directions have empty or ``None`` knot arrays.
            ValueError: If any knot lies outside its direction's domain.
            ValueError: If any insertion would exceed maximum multiplicity.

        References:
            Knot insertion and refinement :cite:p:`piegl1997nurbs`.
        """
        dtype = self.dtype

        if self.dim == 1:
            arr = np.asarray(new_knots, dtype=dtype)
            new_knots_per_dim: list[npt.NDArray[np.float32 | np.float64] | None] = [arr]
        else:
            seq = list(new_knots)  # type: ignore[arg-type]
            if len(seq) != self.dim:
                raise ValueError(
                    f"new_knots sequence length ({len(seq)}) must match dim ({self.dim})."
                )
            new_knots_per_dim = [None if nk is None else np.asarray(nk, dtype=dtype) for nk in seq]

        # Require at least one non-empty direction.
        if all(nk is None or nk.size == 0 for nk in new_knots_per_dim):
            raise ValueError(
                "At least one direction must have a non-empty array of knots to insert."
            )

        return self._insert_knots_per_direction(new_knots_per_dim)

    def _insert_knots_per_direction(
        self, new_knots_per_dim: list[npt.NDArray[np.float32 | np.float64] | None]
    ) -> Bspline:
        """Insert knots given per direction (``None`` or an empty array skips a direction).

        Args:
            new_knots_per_dim (list[npt.NDArray | None]): One array of the B-spline's dtype, or
                ``None``, per direction.

        Returns:
            Bspline: The refined B-spline.

        Raises:
            ValueError: If an array to insert is not 1D.
        """
        per_dim: list[npt.NDArray[np.float32 | np.float64] | None] = []
        for nk in new_knots_per_dim:
            if nk is None or nk.size == 0:
                per_dim.append(None)
                continue
            if nk.ndim != 1:
                raise ValueError(f"new_knots must be a 1D array-like, got shape {nk.shape}")
            per_dim.append(np.ascontiguousarray(nk))
        return self._from_result(self._h.insert_knots(per_dim))

    def remove_knots(
        self,
        knot_values: float | npt.ArrayLike | Sequence[npt.ArrayLike | None],
        *,
        num: int | None = None,
        tol: float | None = None,
    ) -> Bspline:
        """Return a B-spline with specified interior knots removed.

        Each listed knot value is removed up to *num* times (or as many as
        possible when ``num=None``), provided the geometric deviation stays
        within *tol*.

        With no ``tol`` the operation is **lossless**: only knots the spline does not
        actually need are removed, to within the round-off of the reconstruction that
        decides it. Removing a knot at the cost of a geometric error is a trade only
        the caller can price, so it has to be asked for by passing a ``tol``.

        Args:
            knot_values (float | npt.ArrayLike | Sequence[npt.ArrayLike | None]):
                For a 1D B-spline, a single float or a 1D array-like of
                distinct interior knot values to remove. For multi-dimensional
                B-splines, a sequence of length ``dim`` where each element is a
                1D array-like of knot values to remove in that direction, or
                ``None`` to skip that direction. At least one direction must
                have a non-empty array of knot values.
            num (int | None): Maximum number of removals per distinct knot
                value. ``None`` (default) removes as many as possible (up to
                the current multiplicity, capped at the degree).
            tol (float | None): Maximum allowed geometric deviation for each
                removal step, as a distance in the B-spline's own physical units
                (projected coordinates, for a rational B-spline). ``None`` (default)
                derives the round-off floor ``8 * (degree + 1) * eps * scale *
                sqrt(n)``, with ``scale = max(control-net bounding-box diagonal, largest
                ||P_i||)`` and ``n`` the number of control points the removal test spans
                at once (one for a curve, the product of the other directions' basis
                counts for a surface or a volume), so a knot is removed only when doing
                so is exact to the arithmetic.

        Returns:
            Bspline: New B-spline with the same geometry (within tolerance)
            and reduced knot vectors.

        Raises:
            ValueError: If the B-spline is periodic in any direction.
            ValueError: If the sequence length does not match ``dim``
                (multi-dim case).
            ValueError: If all directions have empty or ``None`` knot arrays.
            ValueError: If any knot value is not found, is a boundary knot, or is
                not an interior knot of the domain (a ghost knot of an unclamped
                knot vector).
            ValueError: If a knot to remove is repeated more than ``degree + 1``
                times in the knot vector (a space accepts such a vector).
            TypeError: If ``num`` is not an integer.
        """
        # Periodic splines are not supported.
        for i, sp in enumerate(self._space.spaces):
            if sp.periodic:
                raise ValueError(
                    f"Knot removal is not supported for periodic B-splines "
                    f"(direction {i} is periodic)."
                )

        dtype = self.dtype

        if self.dim == 1:
            arr = np.atleast_1d(np.asarray(knot_values, dtype=dtype)).ravel()
            kv_per_dim: list[npt.NDArray[np.float32 | np.float64] | None] = [arr]
        else:
            seq = list(knot_values)  # type: ignore[arg-type]
            if len(seq) != self.dim:
                raise ValueError(
                    f"knot_values sequence length ({len(seq)}) must match dim ({self.dim})."
                )
            kv_per_dim = [
                None if kv is None else np.atleast_1d(np.asarray(kv, dtype=dtype)).ravel()
                for kv in seq
            ]

        # Require at least one non-empty direction.
        if all(kv is None or kv.size == 0 for kv in kv_per_dim):
            raise ValueError(
                "At least one direction must have a non-empty array of knot values to remove."
            )

        return self._from_result(
            self._h.remove_knots(
                [
                    None if kv is None or kv.size == 0 else np.ascontiguousarray(kv)
                    for kv in kv_per_dim
                ],
                None if num is None else _index(num),
                None if tol is None else _real(tol),
            )
        )

    def to_open_bspline(self) -> Bspline:
        """Return an open (clamped) non-periodic B-spline equivalent to this one.

        Converts each parametric direction to an open representation by inserting knots
        at the domain boundaries until each has multiplicity ``degree + 1``, then
        trimming any ghost knots outside the domain.  Works for 1D and multi-dimensional
        B-splines, and handles periodic, unclamped non-periodic, and mixed cases.

        For periodic splines the ``n_full = len(knots) - degree - 1`` control points are
        reconstructed by modulo-wrapping the ``n_periodic`` stored control points
        (``ctrl_full[i] = ctrl[i % n_periodic]``), and the Oslo algorithm is applied to
        this full set. The resulting open B-spline represents the mathematical periodic
        function defined by the periodic knot vector and the wrapped control points.

        Returns:
            Bspline: Open, non-periodic B-spline with clamped knot vectors.

        Raises:
            ValueError: If the B-spline is already open in every direction.
        """
        return self._from_result(self._h.to_open())

    def restrict(
        self,
        bounds: tuple[float, float] | Sequence[tuple[float, float] | None],
    ) -> Bspline:
        """Return a B-spline restricted to a sub-region of the parametric domain.

        Extracts the portion of the B-spline defined on the given sub-domain by
        inserting knots at the new boundaries until they reach multiplicity
        ``degree + 1``, then extracting the relevant knot sub-vector and control
        points.  Skips insertion when a bound already coincides with a clamped
        domain endpoint.  Periodic directions are automatically converted to
        open form before restricting.

        Args:
            bounds (tuple[float, float] | Sequence[tuple[float, float] | None]):
                For a 1D B-spline, a ``(lower, upper)`` tuple defining the new
                domain.  For multi-dimensional B-splines, a sequence of length
                ``dim`` where each element is a ``(lower, upper)`` tuple for
                that direction, or ``None`` to keep the full domain in that
                direction.  At least one direction must have non-``None`` bounds
                that actually restrict the domain.

        Returns:
            Bspline: New B-spline defined on the restricted domain.

        Raises:
            ValueError: If the sequence length does not match ``dim`` (multi-dim).
            ValueError: If all directions are ``None`` or match the full domain.
            ValueError: If any bound lies outside its direction's domain.
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

        return self._from_result(
            self._h.restrict(
                [None if b is None else (_real(b[0]), _real(b[1])) for b in bounds_per_dim]
            )
        )

    def split(self, direction: int, value: float) -> tuple[Bspline, Bspline]:
        """Split the B-spline into two at a parameter value in one direction.

        Inserts knots at ``value`` until the multiplicity reaches
        ``degree + 1``, then extracts the left and right sub-splines.
        Periodic directions are automatically converted to open form before
        splitting.

        Args:
            direction (int): Parametric direction along which to split.
                Must be in ``[0, dim)``.
            value (float): Parameter value at which to split.  Must lie
                strictly inside the domain of the specified direction.

        Returns:
            tuple[Bspline, Bspline]: A pair ``(left, right)`` of B-splines.
            The left sub-spline has domain ``[domain_start, value]`` and
            the right has domain ``[value, domain_end]`` in the split
            direction. Other directions are unchanged.

        Raises:
            ValueError: If ``direction`` is out of range ``[0, dim)``.
            ValueError: If ``value`` is not strictly inside the domain.

        Example:
            >>> import numpy as np
            >>> from pantr.bspline import create_uniform_space
            >>> spline = Bspline(create_uniform_space(2, 3), np.arange(5.0).reshape(5, 1))
            >>> left, right = spline.split(0, 0.5)
            >>> np.allclose(left.space.spaces[0].domain, [0.0, 0.5])
            True
            >>> np.allclose(right.space.spaces[0].domain, [0.5, 1.0])
            True
        """
        left, right = self._h.split(_index(direction), _real(value))
        return self._from_result(left), self._from_result(right)

    def to_periodic(self, continuity: int | tuple[int | None, ...] | None = None) -> Bspline:
        """Return a periodic B-spline equivalent to this one.

        Converts each non-periodic parametric direction to a periodic
        representation with the requested continuity at the seam (the domain
        boundary where the periodic function wraps around).  Directions that
        are already periodic are left unchanged.

        The conversion performs an exact change of basis: the Oslo algorithm
        defines the linear map from periodic control points (with modulo
        wrapping) to open control points, and inverting this map via QR
        factorization recovers the periodic representation.

        Warning:
            This method assumes the B-spline represents an inherently periodic
            function — i.e., the function values and derivatives match at the
            domain boundaries up to the requested continuity order.  If the
            function is not periodic, the result will **not** represent the
            same geometry.  Gross violations raise ``ValueError`` (via a
            residual check), but small deviations may pass silently.  Use
            :meth:`to_open_bspline` on the result to verify round-trip
            fidelity if in doubt.

        Args:
            continuity (int | tuple[int | None, ...] | None): Target continuity
                at the seam per direction.  ``None`` (default) requests maximum
                regularity ``C^{p-1}`` in every non-periodic direction.  An
                integer applies to all non-periodic directions; a tuple specifies
                per-direction values where ``None`` entries skip that direction
                (leave it unchanged).  Integer values must satisfy
                ``0 <= continuity <= degree - 1``.

        Returns:
            Bspline: B-spline with the requested directions made periodic.

        Raises:
            ValueError: If no direction would be converted (all already periodic
                or all skipped via ``None`` in the tuple).
            ValueError: If the function is not periodic (endpoint mismatch
                or residual exceeds tolerance).
            ValueError: If ``continuity`` is out of range.
        """
        cont: list[int | None]
        if continuity is None:
            cont = [_CONVERT_DEFAULT] * self.dim
        elif isinstance(continuity, int):
            cont = [_index(continuity)] * self.dim
        else:
            cont = [None if c is None else _index(c) for c in continuity]
        return self._from_result(self._h.to_periodic(cont))

    def to_bezier(self, *, copy: bool = True) -> Bezier:
        """Convert to an equivalent Bézier.

        Extracts a :class:`~pantr.bezier.Bezier` from this B-spline when
        it represents a single polynomial patch (one element per direction).
        Periodic and non-open B-splines are automatically converted to open
        form before the check.

        Args:
            copy (bool): If ``True`` (default), the control points are
                deep-copied into the new Bézier.  If ``False``, the Bézier
                shares the same underlying control point array when possible
                (direct extraction of a C-contiguous float32 or float64 array;
                another layout or dtype is converted, which shares nothing) or
                owns the freshly allocated array produced by the open-form
                conversion.

        Returns:
            ~pantr.bezier.Bezier: Equivalent Bézier representation.

        Raises:
            ValueError: If the B-spline has more than one element per
                direction and cannot be represented as a single Bézier.
        """
        from ..bezier import Bezier as BezierCls  # noqa: PLC0415

        if self._space.has_Bezier_like_knots():
            if copy:
                return BezierCls(self.control_points, is_rational=self._is_rational)
            return BezierCls._borrow(self.control_points, self._is_rational)

        # Already open but not Bezier-like → multi-element, cannot convert.
        spaces = self._space.spaces
        if all(s.has_open_knots() and not s.periodic for s in spaces):
            raise ValueError(
                "B-spline has more than one element per direction "
                "and cannot be represented as a single Bézier."
            )

        # Periodic / non-open: convert to open form first.
        open_bspline = self.to_open_bspline()
        if not open_bspline.space.has_Bezier_like_knots():
            raise ValueError(
                "B-spline has more than one element per direction "
                "and cannot be represented as a single Bézier."
            )
        cp = open_bspline.control_points.copy() if copy else open_bspline.control_points
        return BezierCls(cp, is_rational=self._is_rational)

    def to_beziers(self) -> npt.NDArray[np.object_]:
        """Decompose into Bézier patches.

        Decomposes the B-spline into its constituent Bézier patches by applying
        Bézier extraction operators direction by direction.  Periodic directions
        are automatically converted to open form first.

        The result is cached: the first call computes and stores the
        decomposition; subsequent calls return the cached array.  In-place
        mutations (``reverse``, ``permute_directions``, ``transform`` with
        ``in_place=True``) invalidate the cache.

        Returns:
            ``npt.NDArray[np.object_]``: Array of :class:`~pantr.bezier.Bezier`
            objects with shape ``(*num_intervals)`` following the tensor-product
            interval structure.  For a 1D B-spline with 3 intervals the shape
            is ``(3,)``.  For a 2D surface with 3x2 intervals the shape is
            ``(3, 2)``.  The Bézier control points are read-only.

        Example:
            >>> import numpy as np
            >>> from pantr.bspline import create_uniform_space
            >>> spline = Bspline(create_uniform_space(2, 3), np.arange(5.0).reshape(5, 1))
            >>> beziers = spline.to_beziers()
            >>> beziers.shape
            (3,)
            >>> beziers[0].degree
            (2,)

        References:
            Bézier extraction of NURBS :cite:p:`borden2011bezier` (and its
            T-spline generalization :cite:p:`scott2011tsplines`).
        """
        if self._beziers_cache is None:
            self._beziers_cache = _to_beziers_impl(self)
        return self._beziers_cache

    def multiply(self, other: Bspline) -> Bspline:
        """Return the exact pointwise product of this B-spline and another.

        Given B-splines ``self`` and ``other`` over the same parametric
        domain, returns a new B-spline ``h`` such that ``h(t) = self(t) *
        other(t)`` for all ``t`` in the domain.  The result lives in the product
        space of degree ``p_d + q_d`` per direction where ``p_d`` and ``q_d``
        are the degrees of the two operands in direction *d*.

        Both non-rational and rational (NURBS) operands are supported.  A
        non-rational operand is promoted to rational (unit weights) when the
        other is rational.

        Args:
            other (Bspline): The second B-spline operand. Must have the
                same dimension, dtype, rank, and parametric domain as ``self``.

        Returns:
            Bspline: A new B-spline representing ``self * other``.

        Raises:
            ValueError: If the operands have different dimensions.
            ValueError: If the operands have different dtypes.
            ValueError: If the operands have different ranks.
            ValueError: If the operands have different parametric domains.

        Note:
            The boundary structure of the operands is preserved per direction:
            both periodic → periodic, both non-open → non-open, either open → open.

        Example:
            >>> import numpy as np
            >>> from pantr.bspline import create_uniform_space
            >>> f = Bspline(create_uniform_space(1, 1), np.array([[0.0], [1.0]]))  # f(t) = t
            >>> g = Bspline(create_uniform_space(1, 1), np.array([[0.0], [1.0]]))  # g(t) = t
            >>> h = f.multiply(g)
            >>> h.degree
            (2,)
            >>> np.allclose(h.evaluate(np.array([[0.5]])), [0.25])  # (f g)(t) = t**2
            True
            >>> np.allclose((f * g).evaluate(np.array([[0.5]])), [0.25])  # same via __mul__
            True
        """
        if self.dim == 1:
            from ._bspline_product import _multiply_bspline_1d  # noqa: PLC0415

            return _multiply_bspline_1d(self, other)
        from ._bspline_product_nd import _multiply_bspline_nd  # noqa: PLC0415

        return _multiply_bspline_nd(self, other)

    __mul__ = multiply

    # ------------------------------------------------------------------
    # Reverse and permute
    # ------------------------------------------------------------------

    @overload
    def reverse(self, direction: int = ..., *, in_place: Literal[False] = ...) -> Bspline: ...

    @overload
    def reverse(self, direction: int = ..., *, in_place: Literal[True]) -> None: ...

    def reverse(self, direction: int = 0, *, in_place: bool = False) -> Bspline | None:
        """Reverse the orientation of one parametric direction.

        Flips the control points along the given parametric axis and reflects
        the corresponding knot vector so that the mapping is reparametrised in
        the opposite sense along that direction.

        Args:
            direction (int): Parametric direction to reverse. Must be in
                ``[0, dim)``. Defaults to 0.
            in_place (bool): If ``True``, modify this B-spline in place and
                return ``None``. If ``False`` (default), return a new B-spline.

        Returns:
            Bspline | None: The reversed B-spline, or ``None`` when
            ``in_place=True``.

        Raises:
            ValueError: If ``direction`` is out of range ``[0, dim)``.

        Example:
            >>> import numpy as np
            >>> from pantr.bspline import create_uniform_space
            >>> srf = Bspline(create_uniform_space(1, [1, 1]), np.arange(8.0).reshape(2, 2, 2))
            >>> rev = srf.reverse(direction=0)
            >>> rev.control_points[:, 0, 0].tolist()
            [4.0, 0.0]
            >>> srf.reverse(direction=1, in_place=True)  # mutates in place, returns None
            >>> srf.control_points[0, :, 0].tolist()
            [2.0, 0.0]
        """
        if direction < 0 or direction >= self.dim:
            raise ValueError(f"direction must be in [0, {self.dim}), got {direction}.")

        from .._control_points_utils import _reverse_control_points  # noqa: PLC0415
        from ._bspline_space_1d import BsplineSpace1D  # noqa: PLC0415
        from ._bspline_space_nd import BsplineSpace  # noqa: PLC0415

        # Reflect the knot vector: knots_new = a + b - knots[::-1].
        old_space = self._space.spaces[direction]
        knots = old_space.knots
        a, b = old_space.domain
        new_knots = (a + b) - knots[::-1]
        new_space_1d = BsplineSpace1D(new_knots, old_space.degree, periodic=old_space.periodic)

        new_spaces = list(self._space.spaces)
        new_spaces[direction] = new_space_1d
        new_space = BsplineSpace(new_spaces)

        new_cp = _reverse_control_points(self.control_points, direction, in_place=False)
        if old_space.periodic:
            # The stored periodic points expand as full[j] = stored[j % n_stored];
            # reversing the full sequence therefore needs a plain flip *plus* a
            # cyclic shift by the ghost count n_full - n_stored.
            n_stored = new_cp.shape[direction]
            n_full = knots.shape[0] - old_space.degree - 1
            shift = (n_full - n_stored) % n_stored
            if shift:
                new_cp = np.roll(new_cp, shift, axis=direction)

        result = Bspline(new_space, new_cp, is_rational=self._is_rational)
        if in_place:
            self._adopt(result)
            return None
        return result

    @overload
    def permute_directions(
        self, permutation: Sequence[int], *, in_place: Literal[False] = ...
    ) -> Bspline: ...

    @overload
    def permute_directions(
        self, permutation: Sequence[int], *, in_place: Literal[True]
    ) -> None: ...

    def permute_directions(
        self, permutation: Sequence[int], *, in_place: bool = False
    ) -> Bspline | None:
        """Reorder the parametric directions according to a permutation.

        Given a permutation ``[i_0, i_1, …]``, the new direction ``k`` is
        the old direction ``permutation[k]``. For example, ``[1, 2, 0]`` on
        a 3D volume maps old direction 1 → new 0, old 2 → new 1, old 0 → new 2.

        Args:
            permutation (Sequence[int]): A permutation of ``range(dim)``.
            in_place (bool): If ``True``, modify this B-spline in place and
                return ``None``. If ``False`` (default), return a new B-spline.

        Returns:
            Bspline | None: The permuted B-spline, or ``None`` when
            ``in_place=True``.

        Raises:
            ValueError: If ``permutation`` is not a valid permutation of
                ``range(dim)``.

        Example:
            >>> import numpy as np
            >>> from pantr.bspline import create_uniform_space
            >>> space = create_uniform_space([2, 1], [1, 1])
            >>> surface = Bspline(space, np.arange(6.0).reshape(3, 2, 1))
            >>> surface.degree
            (2, 1)
            >>> swapped = surface.permute_directions([1, 0])  # swap u <-> v
            >>> swapped.degree
            (1, 2)
            >>> swapped.control_points[0, 2].tolist()  # was control_points[2, 0]
            [4.0]
        """
        from .._control_points_utils import _permute_control_points  # noqa: PLC0415
        from ._bspline_space_nd import BsplineSpace  # noqa: PLC0415

        perm = list(permutation)
        if sorted(perm) != list(range(self.dim)):
            raise ValueError(f"permutation must be a permutation of range({self.dim}), got {perm}.")

        new_cp = _permute_control_points(self.control_points, perm, self.dim)

        # Reorder 1D spaces.
        old_spaces = self._space.spaces
        new_spaces = tuple(old_spaces[i] for i in perm)
        new_space = BsplineSpace(new_spaces)

        result = Bspline(new_space, new_cp, is_rational=self._is_rational)
        if in_place:
            self._adopt(result)
            return None
        return result

    # ------------------------------------------------------------------
    # Affine transformation
    # ------------------------------------------------------------------

    @overload
    def transform(self, affine: AffineTransform, *, in_place: Literal[False] = ...) -> Bspline: ...

    @overload
    def transform(self, affine: AffineTransform, *, in_place: Literal[True]) -> None: ...

    def transform(
        self,
        affine: AffineTransform,
        *,
        in_place: bool = False,
    ) -> Bspline | None:
        """Apply an affine transformation to the control points.

        For non-rational B-splines, every control point ``P`` is mapped to
        ``A @ P + b``.  For rational B-splines (NURBS) the weighted
        homogeneous coordinates are updated so that the projected geometry
        undergoes the same affine map while the weights are preserved.

        Args:
            affine (~pantr.transform.AffineTransform): The affine
                transformation to apply.
            in_place (bool): If ``True``, the control points are modified in
                place and ``None`` is returned.  If ``False`` (default), a
                new :class:`Bspline` is returned.

        Returns:
            Bspline | None: The transformed B-spline, or ``None`` when
            ``in_place=True``.

        Raises:
            ValueError: If the transform dimension does not match the
                geometric rank of the B-spline.

        Example:
            >>> import numpy as np
            >>> from pantr.bspline import create_uniform_space
            >>> from pantr.transform import AffineTransform
            >>> spline = Bspline(create_uniform_space(1, 1), np.array([[0.0, 0.0], [1.0, 1.0]]))
            >>> T = AffineTransform.translation([1.0, 2.0])
            >>> shifted = spline.transform(T)
            >>> np.allclose(shifted.control_points, [[1.0, 2.0], [2.0, 3.0]])
            True
        """
        # The transform always builds a new array (the control points are read-only).
        new_cp = _apply_affine_to_control_points(
            self.control_points,
            self._is_rational,
            affine.matrix,
            affine.offset,
            in_place=False,
        )
        result = Bspline(self._space, new_cp, is_rational=self._is_rational)
        if in_place:
            self._adopt(result)
            return None
        return result

    def subdivide(
        self,
        n_subdivisions: int | Sequence[int | None],
        regularity: int | None = None,
    ) -> Bspline:
        """Return a geometrically equivalent B-spline with uniformly refined knot vectors.

        For every non-zero knot span in each active parametric direction,
        inserts ``n_subdivisions - 1`` uniformly spaced knot values.  Each
        value is repeated ``degree - regularity`` times so that the B-spline
        has ``C^regularity`` continuity at every inserted knot.

        Args:
            n_subdivisions (int | Sequence[int | None]): Number of equal
                sub-spans per existing interval.  A single ``int`` is applied
                to all directions; must be >= 2.  A sequence of length ``dim``
                provides per-direction counts; use ``None`` to skip a direction.
                At least one direction must have a count >= 2.
            regularity (int | None): Continuity order at every inserted knot.
                Applied uniformly across all active directions.  Must be in
                ``[-1, degree - 1]`` for each active direction.  ``None``
                (default) uses ``degree - 1`` per direction.

        Returns:
            Bspline: New B-spline with refined knot vectors and same geometry.

        Raises:
            ValueError: If the sequence length does not match ``dim``.
            ValueError: If any subdivision count is < 1.
            ValueError: If no direction has a count >= 2.
            ValueError: If ``regularity`` is outside the valid range for any
                active direction.
        """
        counts: list[int | None]
        if isinstance(n_subdivisions, int):
            counts = [n_subdivisions] * self.dim
        else:
            counts = list(n_subdivisions)
        # None skips a direction, as a count of 1 does.
        return self._from_result(
            self._h.subdivide(
                [1 if c is None else _index(c) for c in counts],
                None if regularity is None else _index(regularity),
            )
        )

    def slice(self, axis: int, value: float) -> Bspline | npt.NDArray[np.float32 | np.float64]:
        """Slice the B-spline by fixing one parametric direction at a given value.

        Reduces the parametric dimension by one using de Boor corner cutting
        on the control points.  A volume becomes a surface, a surface becomes
        a curve, and a curve becomes a point (returned as a NumPy array).

        When the parameter value coincides with a knot of multiplicity ``s``,
        only ``p - s`` de Boor iterations are needed.  At a C0 knot
        (``s >= p``) the result is obtained in O(1) by direct control point
        lookup.

        Args:
            axis (int): Parametric direction to fix (0-indexed).
                Must be in ``[0, dim)``.
            value (float): Parameter value at which to slice.  Must lie
                within the domain of the specified direction.

        Returns:
            Bspline | npt.NDArray[np.float32 | np.float64]:
            A B-spline with ``dim - 1`` dimensions when ``dim >= 2``,
            or a NumPy array of shape ``(rank,)`` when ``dim == 1``.
            Rational B-splines preserve the NURBS structure when ``dim >= 2``;
            for ``dim == 1`` the result is projected to physical coordinates.

        Raises:
            ValueError: If ``axis`` is out of range ``[0, dim)``.
            ValueError: If ``value`` is outside the domain of the specified
                direction.

        Example:
            >>> import numpy as np
            >>> from pantr.bspline import create_uniform_space
            >>> space = create_uniform_space(1, [2, 2, 2])
            >>> volume = Bspline(space, np.arange(81.0).reshape(3, 3, 3, 3))
            >>> volume.slice(2, 0.3).dim  # slice a volume at w=0.3 to get a surface
            2
            >>> volume.slice(2, 0.3).slice(1, 0.5).dim  # and again at v=0.5 to get a curve
            1
            >>> pt = volume.slice(2, 0.3).slice(1, 0.5).slice(0, 0.2)  # composable down to a point
            >>> np.allclose(pt, volume.evaluate(np.array([[0.2, 0.5, 0.3]])))
            True
        """
        if axis < 0 or axis >= self.dim:
            raise ValueError(f"axis must be in [0, {self.dim}), got {axis}.")

        space_d = self.space.spaces[axis]
        domain = space_d.domain
        tol = float(space_d.tolerance)
        if value < float(domain[0]) - tol or value > float(domain[1]) + tol:
            raise ValueError(
                f"value {value} is outside the domain [{domain[0]}, {domain[1]}] "
                f"of direction {axis}."
            )

        return _slice_bspline(self, axis, value)

    def boundary(self, axis: int, side: int) -> Bspline | npt.NDArray[np.float32 | np.float64]:
        """Extract the boundary of the B-spline along one parametric direction.

        Returns the restriction of the B-spline to one end of the domain
        in the given direction.

        Args:
            axis (int): Parametric direction (0-indexed).
                Must be in ``[0, dim)``.
            side (int): Which end of the domain: ``0`` for the start,
                ``1`` for the end.

        Returns:
            Bspline | npt.NDArray[np.float32 | np.float64]:
            A B-spline with ``dim - 1`` dimensions when ``dim >= 2``,
            or a NumPy array of shape ``(rank,)`` when ``dim == 1``.

        Raises:
            ValueError: If ``axis`` is out of range ``[0, dim)``.
            ValueError: If ``side`` is not 0 or 1.

        Example:
            >>> import numpy as np
            >>> from pantr.bspline import create_uniform_space
            >>> space = create_uniform_space([2, 1], [1, 1])
            >>> surface = Bspline(space, np.arange(6.0).reshape(3, 2, 1))
            >>> left = surface.boundary(0, 0)  # left boundary along direction 0
            >>> left.degree, left.control_points.ravel().tolist()
            ((1,), [0.0, 1.0])
            >>> right = surface.boundary(1, 1)  # right boundary along direction 1
            >>> right.degree, right.control_points.ravel().tolist()
            ((2,), [1.0, 3.0, 5.0])
        """
        if side not in (0, 1):
            raise ValueError(f"side must be 0 or 1, got {side}.")
        if axis < 0 or axis >= self.dim:
            raise ValueError(f"axis must be in [0, {self.dim}), got {axis}.")

        space_d = self.space.spaces[axis]
        domain = space_d.domain
        value = float(domain[0]) if side == 0 else float(domain[1])
        return self.slice(axis, value)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(
        self,
        *,
        color: str | None = None,
        show_control_polygon: bool = False,
        show_knot_lines: bool = False,
        **plotter_kwargs: Any,  # noqa: ANN401
    ) -> object:
        """Quick interactive visualization of this B-spline (requires pyvista).

        For finer control, use ``pantr.viz.Scene`` directly.

        Args:
            color: Surface color.
            show_control_polygon: Render control polygon (points and wireframe).
            show_knot_lines: Render knot lines.
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
            show_knot_lines=show_knot_lines,
            **plotter_kwargs,
        )


def create_from_bezier(bezier: Bezier, *, copy: bool = True) -> Bspline:
    """Create a B-spline from a Bézier.

    Builds a :class:`Bspline` with Bézier-like knot vectors
    (``[0]*(p+1) + [1]*(p+1)`` per direction) whose control points
    are taken from the given Bézier.

    Args:
        bezier (~pantr.bezier.Bezier): The source Bézier.
        copy (bool): If ``True`` (default), the control points are
            deep-copied before they reach the B-spline.  If ``False``, they
            are passed on as they are.  Either way the B-spline constructor
            copies them into its own immutable buffer, so the result shares
            no memory with the Bézier.

    Returns:
        Bspline: Equivalent B-spline with Bézier-like knots.
    """
    return bezier.to_bspline(copy=copy)


_I64_MAX = 2**63 - 1
"""The largest integer the compiled library takes as an index or a count."""


def _index(x: Any) -> int:  # noqa: ANN401
    """Get an integer argument as the compiled library takes it.

    A Python integer beyond 64 bits is saturated: it is out of every range the library checks, so
    it is refused there, and only the number in the message is the saturated one.

    Args:
        x (Any): An ``int``, ``bool`` or NumPy integer (anything with ``__index__``).

    Returns:
        int: ``x``, limited to the 64-bit range.

    Raises:
        TypeError: If ``x`` has no ``__index__`` (a float, for instance).
    """
    return max(-_I64_MAX - 1, min(operator.index(x), _I64_MAX))


def _real(x: Any) -> float:  # noqa: ANN401
    """Get a real-number argument as a Python float.

    Args:
        x (Any): A Python or NumPy real number (a 0-d array too).

    Returns:
        float: ``x`` as a float.

    Raises:
        TypeError: If ``x`` is a string or bytes (``float()`` would parse it).
    """
    if isinstance(x, str | bytes):
        raise TypeError(f"expected a real number, got {type(x).__name__}")
    return float(x)


_CONVERT_DEFAULT = -1
"""The continuity that ``to_periodic`` reads as "the maximum regularity, ``degree - 1``"."""

_VALUE_ATTRS = frozenset({"_h", "_shape", "_is_rational", "_dtype", "control_points"})
"""The attributes that hold the immutable value and what is cached from it (shared by copies)."""

_CACHE_ATTRS = frozenset({"_beziers_cache", "_locate_cache"})
"""The caches derived from the value (not copied by a deep copy, dropped from a pickle)."""


def _extra_state(b: Bspline) -> dict[str, Any]:
    """Get a B-spline's attributes other than its value, for pickling a subclass.

    Args:
        b (Bspline): The B-spline.

    Returns:
        dict[str, Any]: The other attributes (empty for a plain B-spline).
    """
    skip = _VALUE_ATTRS | _CACHE_ATTRS | {"_space"}
    return {k: v for k, v in b.__dict__.items() if k not in skip}


def _class_for(dtype: np.dtype[Any]) -> Any:  # noqa: ANN401
    """Get the compiled class of a dtype.

    Args:
        dtype (np.dtype): ``float32`` or ``float64``.

    Returns:
        Any: ``BsplineF32`` or ``BsplineF64``.
    """
    return _cpp.BsplineF32 if dtype == _F32 else _cpp.BsplineF64


def _rebuild_bspline(
    cls: type[Bspline],
    space: BsplineSpace,
    control_points: npt.NDArray[np.float32 | np.float64],
    is_rational: bool,
) -> Bspline:
    """Rebuild a pickled B-spline of class ``cls``, bypassing ``cls.__init__``.

    A subclass's constructor may take other arguments; the previous release's pickles did not
    call it either.

    Args:
        cls (type[Bspline]): The class of the pickled B-spline.
        space (BsplineSpace): Its space.
        control_points (npt.NDArray[np.float32 | np.float64]): Its control points.
        is_rational (bool): Whether it is rational.

    Returns:
        Bspline: The B-spline (its extra attributes are restored by ``__setstate__``).
    """
    b = cls.__new__(cls)
    Bspline.__init__(b, space, control_points, is_rational)
    return b
