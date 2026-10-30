"""1D B-spline space: knot vector, degree, and basis evaluation.

This module exposes :class:`BsplineSpace1D`, which combines a knot vector and
polynomial degree to define a 1D B-spline space. It provides methods for
basis evaluation, Bézier/Lagrange/cardinal extraction operators, and several
geometric properties (domain, intervals, cardinality).

The space is a thin wrapper over the C++ value ``pantr::bspline::BsplineSpace1D``
(``pantr._pantr.bspline``): the wrapper normalizes arguments and handles ``out=``
arrays, and the C++ library validates, derives the knot classes and evaluates the
basis.
"""

from __future__ import annotations

import contextlib
import copy as _copy
import functools
from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy import typing as npt

from .._ext import ext
from ..basis import LagrangeVariant
from ..basis._basis_utils import (
    _allocate_or_validate_out,
    _compute_final_output_shape_1D,
    _compute_final_output_shape_1D_deriv,
    _normalize_points_1D,
    _reshaped_out,
    _validate_out_array,
)
from ._bspline_extraction import (
    _tabulate_Bspline_Bezier_1D_extraction_impl,
    _tabulate_Bspline_cardinal_1D_extraction_impl,
    _tabulate_Bspline_Lagrange_1D_extraction_impl,
)
from ._bspline_knot_insertion import (
    _compute_inserted_knot_vector_1d,
    _compute_uniform_subdivision_knots,
)

_cpp: Any = ext.bspline
"""The compiled B-spline spaces (private; untyped: one class per dtype)."""

_EMPTY_PTS: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
"""An empty point array, for a call made only to have the C++ library check an argument."""


def _as_knot_array(knots: npt.ArrayLike) -> npt.NDArray[np.float32 | np.float64]:
    """Convert the ``knots`` argument as the previous release accepted it.

    A list or a NumPy array (of any subclass) is accepted, anything else is a
    ``TypeError``; integer knots become ``float64``; the result must be 1D and of
    dtype ``float32`` or ``float64`` (native byte order).

    Args:
        knots (npt.ArrayLike): The knot vector as given.

    Returns:
        npt.NDArray[np.float32 | np.float64]: A C-contiguous 1D array (the caller's array
        when it already is one; the C++ space copies it).

    Raises:
        TypeError: If ``knots`` is not a list or an array, or is not 1D.
        ValueError: If the dtype is not ``float32`` or ``float64``.
    """
    if isinstance(knots, list):
        arr = np.array(knots)
    elif isinstance(knots, np.ndarray):
        arr = np.asarray(knots)
    else:
        raise TypeError("knots must be a 1D numpy array or Python list")
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float64)
    if arr.ndim != 1:
        raise TypeError("knots must be a 1D numpy array or Python list")
    if arr.dtype not in (np.float32, np.float64):
        raise ValueError("knots type must be float (32 or 64 bits)")
    return np.ascontiguousarray(arr)


_FIRST_DTYPE: np.dtype[np.int64] = np.dtype(np.int64)
"""The dtype the C++ library writes first-basis indices in (``np.int_`` is that on NumPy 2)."""


_VALUE_ATTRS_1D: frozenset[str] = frozenset(
    {
        "_h",
        "_knots",
        "_degree",
        "_periodic",
        "_tol",
        "_num_basis",
        "_num_intervals",
        "_unique_all",
        "_unique_in_domain",
        "_first_basis",
        "domain",
    }
)
"""The attributes of a ``BsplineSpace1D`` that hold its immutable value or are cached from it."""

_PREVIOUS_RELEASE_ATTRS_1D: frozenset[str] = frozenset(
    {
        "_knots",
        "_degree",
        "_periodic",
        "_tol",
        "num_basis",
        "num_intervals",
        "domain",
        "_first_basis_per_interval_cached",
        "_left_end_open",
        "_right_end_open",
        "_bezier_like_knots",
    }
)
"""The value and cached attributes a previous release's ``BsplineSpace1D`` pickled."""


def _extra_state(obj: object, value_attrs: frozenset[str]) -> dict[str, Any]:
    """Get an object's attributes other than its value, for pickling a subclass.

    Args:
        obj (object): The space.
        value_attrs (frozenset[str]): The attributes that hold the value or are cached from it.

    Returns:
        dict[str, Any]: The other attributes (empty for an instance of the class itself).
    """
    return {k: v for k, v in obj.__dict__.items() if k not in value_attrs}


def _first_out(out: npt.NDArray[np.int_] | None, shape: tuple[int, ...]) -> npt.NDArray[np.int_]:
    """Allocate a first-basis index array, or validate the caller's as the previous release did.

    Args:
        out (npt.NDArray[np.int_] | None): The caller's array, or None.
        shape (tuple[int, ...]): The shape it must have.

    Returns:
        npt.NDArray[np.int_]: ``out``, or a fresh ``np.int_`` array.

    Raises:
        ValueError: If ``out`` has another shape or dtype than ``np.int_``, or is not writeable.
    """
    if out is None:
        return np.empty(shape, dtype=np.int_)
    _validate_out_array(out, shape, np.int_)
    return out


@contextlib.contextmanager
def _kernel_buffer(
    out: npt.NDArray[Any], shape: tuple[int, ...], dtype: npt.DTypeLike | None = None
) -> Iterator[npt.NDArray[Any]]:
    """Give the C++ call a C-contiguous array of ``shape`` that ends up in ``out``.

    ``out`` is used directly when it reshapes to a C-contiguous view of the dtype the
    call writes; otherwise a temporary is handed out and copied into ``out`` once the
    block completes (not when it raises).

    Args:
        out (npt.NDArray[Any]): The validated output array.
        shape (tuple[int, ...]): The shape the C++ call writes.
        dtype (npt.DTypeLike | None): The dtype the C++ call writes; ``out``'s when None.

    Yields:
        npt.NDArray[Any]: The array to write into.
    """
    buffer, copy_back = _reshaped_out(out, shape)
    kernel_dtype = buffer.dtype if dtype is None else np.dtype(dtype)
    direct = buffer.flags.c_contiguous and buffer.dtype == kernel_dtype
    target = buffer if direct else np.empty(shape, dtype=kernel_dtype)
    yield target
    if target is not buffer:
        buffer[...] = target
    if copy_back:
        out[...] = buffer.reshape(out.shape)


class BsplineSpace1D:
    """A class representing a 1D B-spline with configurable degree and knot vector.

    This class provides methods to analyze B-spline properties, validate input
    parameters, compute various geometric characteristics of the spline,
    and access various properties of the B-spline.

    An instance always has :attr:`num_intervals` at least 1: a knot vector whose
    in-domain knots all fall in one class is refused at construction, since such a
    space has no cell and nothing can be evaluated, tabulated or located on it.

    The value is immutable: the knots are copied into a compiled value at
    construction, everything derived from them is computed then, and the arrays
    this object returns are read-only views of that value.

    Attributes:
        _h (Any): The compiled value (``BsplineSpace1DF64`` or ``BsplineSpace1DF32``).
        _tol (float): Absolute tolerance for parametric comparisons on this space,
            in the units of its knots. Derived once at construction as
            ``8 * eps(dtype) * max(span, |knots[0]|, |knots[-1]|)``, so it tracks
            the knot vector's own magnitude instead of being a per-dtype constant.
        _knots (npt.NDArray[np.float32 | np.float64]): Knot vector defining the
            B-spline, a read-only view of ``_h``.
        _degree (int): Polynomial degree of the B-spline.
        _periodic (bool): Whether the B-spline is periodic.
        _num_basis (int): Number of basis functions.
        _num_intervals (int): Number of intervals.
    """

    _h: Any
    _tol: float
    _knots: npt.NDArray[np.float32 | np.float64]
    _degree: int
    _periodic: bool
    _num_basis: int
    _num_intervals: int

    def __init__(
        self,
        knots: npt.ArrayLike,
        degree: int,
        periodic: bool = False,
        snap_knots: bool | None = True,
    ) -> None:
        """Initialize a B-spline 1D object.

        Args:
            knots (npt.ArrayLike): Knot vector defining the B-spline. Must be non-decreasing
                and have at least 2*degree+2 elements.
            degree (int): Polynomial degree of the B-spline. Must be non-negative.
            periodic (bool): Whether the B-spline is periodic. Defaults to False.
            snap_knots (bool | None): Whether to snap nearby knots to avoid numerical issues.
                Defaults to True.

        Raises:
            ValueError: If degree is negative, knots are insufficient, or knots are
                not non-decreasing; if the resulting space would span no interval,
                i.e. every step between ``knots[degree]`` and ``knots[-degree-1]`` is
                within :attr:`tolerance`, so the domain is one knot and the space has
                no cell; or if
                ``snap_knots`` is set and snapping is what collapsed a knot vector
                that had more than one distinct knot onto that single point, which
                means the requested spacing is finer than the dtype resolves at that
                coordinate magnitude. ``snap_knots=False`` bypasses the merging and
                the snapping diagnosis, but not the interval requirement. Also, at
                positive degree, if the last knot repeats more than ``degree + 1``
                times, which leaves a space whose basis does not sum to one at the
                right endpoint.
            TypeError: If knots cannot be converted to a numpy array.
        """
        if degree < 0:
            # Refused by the C++ call before the knots are examined, as it always was first.
            _cpp.BsplineSpace1DF64(_EMPTY_PTS, degree, False, False)
        arr = _as_knot_array(knots)
        cls = _cpp.BsplineSpace1DF32 if arr.dtype.type is np.float32 else _cpp.BsplineSpace1DF64
        self._set_handle(cls(arr, degree, bool(periodic), bool(snap_knots)))

    def _set_handle(self, handle: Any) -> None:  # noqa: ANN401
        """Hold a compiled space and cache what its properties return.

        The value is immutable, so what is read here never goes stale.

        Args:
            handle (Any): The compiled space.
        """
        self._h = handle
        self._knots = handle.knots
        self._degree = int(handle.degree)
        self._periodic = bool(handle.periodic)
        self._tol = float(handle.tolerance)
        self._num_basis = int(handle.num_basis)
        self._num_intervals = int(handle.num_intervals)

    @classmethod
    def _from_handle(cls, handle: Any) -> BsplineSpace1D:  # noqa: ANN401
        """Wrap a compiled space built by the library (no copy, no revalidation).

        Args:
            handle (Any): The compiled space.

        Returns:
            BsplineSpace1D: The wrapper.
        """
        space = cls.__new__(cls)
        space._set_handle(handle)
        return space

    @classmethod
    def _from_state(
        cls,
        knots: npt.NDArray[np.float32 | np.float64],
        degree: int,
        periodic: bool,
        tol: float,
    ) -> BsplineSpace1D:
        """Rebuild a space from its stored knots, degree, periodicity and tolerance.

        The knots are taken as stored (no snapping) with the tolerance the space was
        built with, so the rebuilt space equals the original.

        Args:
            knots (npt.NDArray[np.float32 | np.float64]): The stored knot vector.
            degree (int): The degree.
            periodic (bool): The periodicity.
            tol (float): The stored tolerance.

        Returns:
            BsplineSpace1D: The space.
        """
        arr = _as_knot_array(knots)
        h_cls = _cpp.BsplineSpace1DF32 if arr.dtype.type is np.float32 else _cpp.BsplineSpace1DF64
        return cls._from_handle(h_cls.from_state(arr, degree, bool(periodic), float(tol)))

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle (and copy) through :meth:`_from_state`, with a subclass's own attributes as state.

        Returns:
            tuple[Any, ...]: The rebuilding callable, its arguments and the other attributes.
        """
        return (
            type(self)._from_state,
            (self._knots, self._degree, self._periodic, self._tol),
            _extra_state(self, _VALUE_ATTRS_1D),
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a subclass's attributes, or rebuild from a previous release's pickle.

        A previous release pickled the attributes ``_knots``, ``_degree``, ``_periodic`` and
        ``_tol`` (with whatever cached properties it held); such a state rebuilds the compiled
        value from them and drops the cached entries.

        Args:
            state (dict[str, Any]): The pickled state.
        """
        state = dict(state)
        if "_knots" in state and "_h" not in self.__dict__:
            rebuilt = type(self)._from_state(
                np.asarray(state["_knots"]), state["_degree"], state["_periodic"], state["_tol"]
            )
            self._set_handle(rebuilt._h)
            for key in _PREVIOUS_RELEASE_ATTRS_1D:
                state.pop(key, None)
        self.__dict__.update(state)

    def __copy__(self) -> BsplineSpace1D:
        """Copy: the new object shares the immutable compiled value.

        Returns:
            BsplineSpace1D: The copy, of the same class, with the same attributes.
        """
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo: dict[int, Any]) -> BsplineSpace1D:
        """Deep copy: shares the immutable compiled value and deep-copies any other attribute.

        Args:
            memo (dict[int, Any]): The deep-copy memo.

        Returns:
            BsplineSpace1D: The copy.
        """
        new = object.__new__(type(self))
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            new.__dict__[key] = value if key in _VALUE_ATTRS_1D else _copy.deepcopy(value, memo)
        return new

    @property
    def degree(self) -> int:
        """Get the polynomial degree of the B-spline.

        Returns:
            int: The degree.
        """
        return self._degree

    @property
    def knots(self) -> npt.NDArray[np.float32 | np.float64]:
        """Get the knot vector.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The knot vector.
        """
        return self._knots

    @property
    def periodic(self) -> bool:
        """Check if the B-spline is periodic.

        Returns:
            bool: True if periodic, False otherwise.
        """
        return self._periodic

    @property
    def tolerance(self) -> float:
        """Get the absolute tolerance used for parametric comparisons on this space.

        It is in the units of the knot vector, being the dimensionless strict
        preset of :mod:`pantr.tolerance` scaled by the knot vector's own magnitude
        (``max(span, |knots[0]|, |knots[-1]|)``) and doubled for one extra rounding
        upstream. Two knots closer than this are the same knot, a point within this
        of an end is inside the domain, and two knot intervals differing by less
        than this are the same length.

        Being an absolute parametric length, it is *not* the factor to scale a
        physical coordinate or a spline coefficient by; take the dimensionless
        preset from :mod:`pantr.tolerance` for those and scale it by the magnitude
        that applies.

        Returns:
            float: The absolute parametric tolerance.
        """
        return self._tol

    @property
    def dtype(self) -> npt.DTypeLike:
        """Get the data type of the knot vector (and used in computations).

        Returns:
            npt.DTypeLike: The numpy data type of the knots.
        """
        return self._knots.dtype

    @property
    def num_basis(self) -> int:
        """Get the number of basis functions.

        This depends on the knot vector length and the degree, but
        also on whether the B-spline is periodic.

        Returns:
            int: Number of basis functions.
        """
        return self._num_basis

    def get_unique_knots_and_multiplicity(
        self,
        in_domain: bool = False,
    ) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
        """Get unique knots and their multiplicities.

        Args:
            in_domain (bool): If True, only consider knots in the domain.
                Defaults to False.

        Returns:
            tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[numpy.intp]]: Tuple of
            (unique_knots, multiplicities) where unique_knots contains the distinct knot values
            and multiplicities contains the corresponding multiplicity counts.
        """
        return self._unique_in_domain if in_domain else self._unique_all

    @functools.cached_property
    def _unique_all(self) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
        """Read-only views of every distinct knot and its multiplicity (cached).

        Returns:
            tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: The knots and
            multiplicities.
        """
        result: tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]] = (
            self._h.unique_knots_and_multiplicity(False)
        )
        return result

    @functools.cached_property
    def _unique_in_domain(
        self,
    ) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
        """Read-only views of the domain's distinct knots and their multiplicities (cached).

        Returns:
            tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: The knots and
            multiplicities.
        """
        result: tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]] = (
            self._h.unique_knots_and_multiplicity(True)
        )
        return result

    @property
    def num_intervals(self) -> int:
        """Get the number of intervals in the domain.

        Returns:
            int: Number of intervals.

        Example:
            >>> space = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
            >>> space.num_intervals
            2
        """
        return self._num_intervals

    def first_basis_per_interval(self) -> npt.NDArray[np.int64]:
        """Get the index of the first B-spline function supported on each interval.

        Entry ``j`` is the smallest global function index ``i`` whose function is
        non-zero on the open interval between in-domain unique knots ``j`` and
        ``j + 1``. The functions non-zero there are exactly ``i`` through
        ``i + degree``, so this one index describes the whole support of the interval.

        The result is cached and read-only: repeated calls return the same array.

        Returns:
            npt.NDArray[np.int64]: Read-only ``int64`` array of shape
            ``(num_intervals,)``, non-decreasing, whose successive differences are
            the interior knot multiplicities.

        Raises:
            ValueError: If the space is periodic.

        Example:
            >>> space = BsplineSpace1D([0, 0, 0, 1, 2, 2, 3, 3, 3], 2)
            >>> space.first_basis_per_interval().tolist()
            [0, 1, 3]
        """
        return self._first_basis

    @functools.cached_property
    def _first_basis(self) -> npt.NDArray[np.int64]:
        """Read-only view of the first supported function of each interval (cached).

        Returns:
            npt.NDArray[np.int64]: Shape ``(num_intervals,)``.

        Raises:
            ValueError: If the space is periodic (from the C++ call, with the message the
                suite matches); nothing is cached then.
        """
        result: npt.NDArray[np.int64] = self._h.first_basis_per_interval()
        return result

    @functools.cached_property
    def domain(self) -> tuple[np.float32 | np.float64, np.float32 | np.float64]:
        """Get the knot vector domain.

        Returns:
            tuple[np.float32 | np.float64, np.float64]: Tuple of
            (start_value, end_value) defining the domain.

        Example:
            >>> bspline = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
            >>> start, end = bspline.domain
            >>> float(start), float(end)
            (0.0, 2.0)
        """
        return (self._knots[self._degree], self._knots[self._knots.size - self._degree - 1])

    def has_left_end_open(self) -> bool:
        """Check if the left end of the B-spline is open.

        A left end is open if the first degree+1 knots are equal.

        Returns:
            bool: True if the left end is open, False otherwise.
        """
        return bool(self._h.left_open)

    def has_right_end_open(self) -> bool:
        """Check if the right end of the B-spline is open.

        A right end is open if the last degree+1 knots are equal.

        Returns:
            bool: True if the right end is open, False otherwise.
        """
        return bool(self._h.right_open)

    def has_open_knots(self) -> bool:
        """Check if the B-spline has open ends.

        Returns:
            bool: True if both ends are open, False otherwise.
        """
        return bool(self._h.left_open and self._h.right_open)

    def has_Bezier_like_knots(self) -> bool:
        """Check if the knot vector represents a Bézier-like configuration.

        A Bézier-like configuration has open ends and only one non-zero span.

        Returns:
            bool: True if knots have open ends and only one span.

        Example:
            >>> bspline = BsplineSpace1D([1, 1, 1, 3, 3, 3], 2)
            >>> bspline.has_Bezier_like_knots()
            True
        """
        return bool(self._h.bezier_like)

    def get_cardinal_intervals(
        self, out: npt.NDArray[np.bool_] | None = None
    ) -> npt.NDArray[np.bool_]:
        """Get boolean array indicating whether the intervals (non-zero spans) are cardinal or not.

        An interval is cardinal if has the same length as the degree-1
        previous and the degree-1 next intervals.

        Concretely, an interval ``[u_k, u_(k+1)]`` is cardinal when both conditions
        hold: neither bounding knot is repeated, and the ``2 * degree - 1`` knot spans
        of the window ``u_(k-degree+1), ..., u_(k+degree)`` -- the ``degree - 1`` spans
        before the interval, the interval itself, and the ``degree - 1`` spans after it
        -- all have the interval's own length, to within the space tolerance.

        Those ``2 * degree`` knots are exactly the ones an evaluation on that span
        reads. The standard basis-function recurrence works up from level ``1`` to
        ``degree``, and at level ``j`` it references only ``u_(k+1-j)`` and ``u_(k+j)``;
        across all levels those indices sweep ``k-degree+1`` through ``k+degree`` and no
        further. So the window holds every knot the ``degree + 1`` functions restricted
        to that interval depend on, and equal spacing across it is what makes those
        restrictions coincide with the cardinal (uniform-knot) ones.

        In the case of open knot vectors, this definition automatically
        discards the first degree-1 and the last degree-1 intervals: the repeated
        end knots put zero-length spans inside their window.

        At ``degree == 0`` the equal-length condition ranges over an empty set of
        neighbouring intervals, so it holds vacuously and the knot spacing does not
        affect the answer. What remains is the multiplicity gate that applies at every
        degree: an interval is reported cardinal unless one of the two knots bounding
        it is repeated. Either way this is consistent with the geometry, since a
        degree-0 space carries one basis function per interval and its cardinal
        extraction operator is the 1x1 identity on every interval regardless of the
        flag.

        Args:
            out (npt.NDArray[bool] | None): Optional output array where the result will be
                stored. If None, a new array is allocated. Must have the correct shape and dtype
                if provided. This follows NumPy's style for output arrays. Defaults to None.

        Returns:
            npt.NDArray[bool]: Boolean array where True indicates cardinal intervals.
                It has length equal to the number of intervals. If `out` was provided,
                returns the same array.

        Raises:
            ValueError: If `out` is provided and has incorrect shape or dtype.

        Example:
            Six unit intervals, degree 2: the window is one span either side, so
            only the two end intervals see a zero-length span from the repeated
            end knots.

            >>> bspline = BsplineSpace1D([0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6], 2)
            >>> bspline.get_cardinal_intervals().tolist()
            [False, True, True, True, True, False]

            Repeating the interior knot 5 disqualifies the two intervals it bounds:

            >>> bspline = BsplineSpace1D([0, 0, 0, 1, 2, 3, 4, 5, 5, 6, 6, 6], 2)
            >>> bspline.get_cardinal_intervals().tolist()
            [False, True, True, True, False, False]

            Degree 3 over the domain ``[3, 5]``: the window reaches two spans past
            each interval, so the irregular span ``[7, 10]`` falls outside both and
            neither interval is affected.

            >>> bspline = BsplineSpace1D([0, 1, 2, 3, 4, 5, 6, 7, 10], 3)
            >>> bspline.get_cardinal_intervals().tolist()
            [True, True]

        References:
            The recurrence whose knot footprint sets this window is Algorithm A2.2
            of :cite:p:`piegl1997nurbs`.
        """
        shape = (self._num_intervals,)
        if out is None:
            out = np.empty(shape, dtype=np.bool_)
        else:
            _validate_out_array(out, shape, np.bool_)
        # The C++ call writes C-contiguous memory; a strided ``out`` gets a temporary.
        target = out if out.flags.c_contiguous else np.empty(shape, dtype=np.bool_)
        self._h.cardinal_intervals(target)
        if target is not out:
            out[...] = target
        return out

    def _check_in_domain(self, pts: Any) -> None:  # noqa: ANN401
        """Refuse points outside the domain, the domain spelled as the previous release did.

        Args:
            pts (Any): Normalized, C-contiguous ``float32`` or ``float64`` points.

        Raises:
            ValueError: If a point (NaN included) is outside the domain up to the tolerance.
        """
        if not self._h.all_in_domain(pts):
            raise ValueError(
                f"One or more values in pts are outside the knot vector domain {self.domain}"
            )

    def tabulate_basis(
        self,
        pts: npt.ArrayLike,
        out_basis: npt.NDArray[np.float32 | np.float64] | None = None,
        out_first_basis: npt.NDArray[np.int_] | None = None,
        *,
        validate: bool = True,
    ) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
        """Evaluate the B-spline basis functions at the given points.

        Args:
            pts (npt.ArrayLike): Evaluation points.
            out_basis (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the
                basis values will be stored. If None, a new array is allocated. Must have the
                correct shape and dtype if provided. This follows NumPy's style for output arrays.
                Defaults to None.
            out_first_basis (npt.NDArray[numpy.intp] | None): Optional output array where the
                first basis indices will be stored. If None, a new array is allocated. Must have
                the correct shape and dtype numpy.intp if provided. This follows NumPy's style for
                output arrays. Defaults to None.
            validate (bool): If True (default), check that every point lies inside
                the spline domain. Pass False only when the caller guarantees
                in-domain points; out-of-domain points are then undefined behavior.
                Defaults to True.

        Returns:
            tuple[
                npt.NDArray[np.float32] | npt.NDArray[np.float64],
                npt.NDArray[numpy.intp]
            ]: Tuple containing:
                - basis_values: (npt.NDArray[np.float32] | npt.NDArray[np.float64])
                  Array of shape matching `pts` with the last dimension length (degree+1),
                  containing the basis function values evaluated at each point.
                  If `out_basis` was provided, returns the same array.
                - first_basis_indices: (npt.NDArray[numpy.intp])
                  1D integer array indicating the index of the first nonzero basis function
                  for each evaluation point. The length is the same as the number
                  of evaluation points. If `out_first_basis` was provided, returns the same array.

        Raises:
            ValueError: If ``validate`` is True and any evaluation point is outside the
                B-spline domain, or if `out_basis` or `out_first_basis` is provided and
                has incorrect shape or dtype.

        Example:
            >>> import numpy as np
            >>> bspline = BsplineSpace1D([0, 0, 0, 0.25, 0.7, 0.7, 1, 1, 1], 2)
            >>> values, first = bspline.tabulate_basis([0.0, 0.5, 0.75, 1.0])
            >>> np.allclose(
            ...     values,
            ...     [
            ...         [1.0, 0.0, 0.0],
            ...         [0.12698413, 0.5643739, 0.30864198],
            ...         [0.69444444, 0.27777778, 0.02777778],
            ...         [0.0, 0.0, 1.0],
            ...     ],
            ... )
            True
            >>> first.tolist()
            [0, 1, 3, 3]

        References:
            Basis values are computed with the stable Cox-de Boor recurrence
            :cite:p:`deboor2001splines`.
        """
        input_shape = np.shape(pts)
        # The binding is overloaded on the dtype; ``p`` is either, so it is passed untyped.
        p: Any = np.ascontiguousarray(_normalize_points_1D(pts))
        if validate:
            self._check_in_domain(p)
        num_pts = p.shape[0]
        order = self._degree + 1
        if out_basis is None and out_first_basis is None:
            # Allocated here rather than by the binding: an uninitialized NumPy array is the
            # cheaper hand-off for the one-point calls whose cost is all overhead.
            values = np.empty((num_pts, order), dtype=p.dtype)
            first = np.empty(num_pts, dtype=_FIRST_DTYPE)
            self._h.tabulate_basis(p, values, first, False)
            return (
                values.reshape(_compute_final_output_shape_1D(input_shape, order)),
                first.astype(np.int_, copy=False).reshape(input_shape),
            )
        out_basis = _allocate_or_validate_out(
            out_basis, _compute_final_output_shape_1D(input_shape, order), p.dtype
        )
        out_first_basis = _first_out(out_first_basis, input_shape)
        with (
            _kernel_buffer(out_basis, (num_pts, order)) as values,
            _kernel_buffer(out_first_basis, (num_pts,), _FIRST_DTYPE) as first,
        ):
            self._h.tabulate_basis(p, values, first, False)
        return out_basis, out_first_basis

    def tabulate_basis_derivatives(
        self,
        pts: npt.ArrayLike,
        n_deriv: int,
        out_deriv: npt.NDArray[np.float32 | np.float64] | None = None,
        out_first_basis: npt.NDArray[np.int_] | None = None,
        *,
        validate: bool = True,
    ) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
        """Evaluate B-spline basis function derivatives at the given points.

        Implements Algorithm A2.3 (DerBasisFuncs) from Piegl & Tiller.
        The 0th slice ``deriv_values[..., 0, :]`` is identical to the result
        of :meth:`tabulate_basis`.  For ``n_deriv > degree`` all rows beyond
        ``degree`` are identically zero.

        Args:
            pts (npt.ArrayLike): Evaluation points.
            n_deriv (int): Maximum derivative order to compute (>= 0).
            out_deriv (npt.NDArray[np.float32 | np.float64] | None): Optional output
                array for derivative values. If None, a new array is allocated. Must
                have shape ``(*pts_shape, n_deriv+1, degree+1)`` and dtype matching
                ``pts`` if provided. Defaults to None.
            out_first_basis (npt.NDArray[numpy.intp] | None): Optional output array
                for first basis indices. If None, a new array is allocated. Must have
                shape ``pts_shape`` and dtype ``numpy.intp`` if provided.
                Defaults to None.
            validate (bool): If True (default), check that every point lies inside
                the spline domain. Pass False only when the caller guarantees
                in-domain points; out-of-domain points are then undefined behavior.
                Defaults to True.

        Returns:
            tuple[
                npt.NDArray[np.float32] | npt.NDArray[np.float64],
                npt.NDArray[numpy.intp]
            ]: Tuple containing:
                - deriv_values: Array of shape ``(*pts_shape, n_deriv+1, degree+1)``.
                  ``deriv_values[..., k, i]`` is the k-th derivative of the i-th
                  local B-spline basis function at each point.
                - first_basis_indices: Integer array of shape ``pts_shape`` giving the
                  global index of the first nonzero basis function for each point.

        Raises:
            ValueError: If ``n_deriv < 0``, if ``validate`` is True and any evaluation
                point is outside the domain, or ``out_deriv`` / ``out_first_basis``
                has incorrect shape or dtype.
            ZeroDivisionError: If ``n_deriv >= 1`` and the recurrence divides by zero: a
                first knot repeated more than ``degree + 1`` times leaves an empty clamped span,
                and a point within the tolerance band left of the domain lands on it.

        Example:
            >>> import numpy as np
            >>> bspline = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)
            >>> d, first = bspline.tabulate_basis_derivatives([0.5], n_deriv=1)
            >>> d.shape
            (1, 2, 3)
            >>> # first derivatives at x=0.5: B0'=-1, B1'=0, B2'=1
            >>> np.allclose(d[0, 1, :], [-1.0, 0.0, 1.0])
            True
        """
        if n_deriv < 0:
            # Refused by the C++ call, before the points are read or any output is sized.
            self._h.tabulate_basis_derivatives(_EMPTY_PTS, n_deriv, None, None, bool(validate))
        input_shape = np.shape(pts)
        p: Any = np.ascontiguousarray(_normalize_points_1D(pts))
        if validate:
            self._check_in_domain(p)
        num_pts = p.shape[0]
        order = self._degree + 1
        kernel_shape = (num_pts, n_deriv + 1, order)
        if out_deriv is None and out_first_basis is None:
            values = np.empty(kernel_shape, dtype=p.dtype)
            first = np.empty(num_pts, dtype=_FIRST_DTYPE)
            self._h.tabulate_basis_derivatives(p, n_deriv, values, first, False)
            return (
                values.reshape(_compute_final_output_shape_1D_deriv(input_shape, n_deriv, order)),
                first.astype(np.int_, copy=False).reshape(input_shape),
            )
        out_deriv = _allocate_or_validate_out(
            out_deriv, _compute_final_output_shape_1D_deriv(input_shape, n_deriv, order), p.dtype
        )
        out_first_basis = _first_out(out_first_basis, input_shape)
        with (
            _kernel_buffer(out_deriv, kernel_shape) as values,
            _kernel_buffer(out_first_basis, (num_pts,), _FIRST_DTYPE) as first,
        ):
            self._h.tabulate_basis_derivatives(p, n_deriv, values, first, False)
        return out_deriv, out_first_basis

    def tabulate_Bezier_extraction_operators(
        self, out: npt.NDArray[np.float32 | np.float64] | None = None
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Create Bézier extraction operators of the B-spline.

        Args:
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the
                result will be stored. If None, a new array is allocated. Must have the correct
                shape and dtype if provided. This follows NumPy's style for output arrays.
                Defaults to None.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Array of extraction matrices with shape
                (n_intervals, degree+1, degree+1) where each matrix transforms
                Bernstein basis functions to B-spline basis functions for that interval.

                Each matrix C[i, :, :] transforms Bernstein basis functions
                to B-spline basis functions for the i-th interval as

                    C[i, :, :] @ [Bernstein values] = [B-spline values in interval].

                If `out` was provided, returns the same array.

        Raises:
            ValueError: If `out` is provided and has incorrect shape or dtype.
        """
        return _tabulate_Bspline_Bezier_1D_extraction_impl(
            self.knots, self.degree, self.tolerance, out=out
        )

    def tabulate_Lagrange_extraction_operators(
        self,
        lagrange_variant: LagrangeVariant = LagrangeVariant.EQUISPACES,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Create Lagrange extraction operators of the B-spline.

        Args:
            lagrange_variant (LagrangeVariant): Lagrange point distribution to use
                (e.g., equispaced, Gauss-Lobatto-Legendre, etc).
                Defaults to `LagrangeVariant.EQUISPACES`.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the
                result will be stored. If None, a new array is allocated. Must have the correct
                shape and dtype if provided. This follows NumPy's style for output arrays.
                Defaults to None.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Array of extraction matrices with shape
                (n_intervals, degree+1, degree+1) where each matrix transforms
                Lagrange basis functions to B-spline basis functions for that interval.

                Each matrix C[i, :, :] transforms Lagrange basis functions
                to B-spline basis functions for the i-th interval as

                    C[i, :, :] @ [Lagrange values] = [B-spline values in interval].

                If `out` was provided, returns the same array.

        Raises:
            ValueError: If `out` is provided and has incorrect shape or dtype.
        """
        return _tabulate_Bspline_Lagrange_1D_extraction_impl(
            self.knots, self.degree, self.tolerance, lagrange_variant=lagrange_variant, out=out
        )

    def tabulate_cardinal_extraction_operators(
        self, out: npt.NDArray[np.float32 | np.float64] | None = None
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Create cardinal B-spline extraction operators of the B-spline.

        Args:
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the
                result will be stored. If None, a new array is allocated. Must have the correct
                shape and dtype if provided. This follows NumPy's style for output arrays.
                Defaults to None.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Array of extraction matrices with shape
                (n_intervals, degree+1, degree+1) where each matrix transforms
                cardinal spline basis functions to B-spline basis functions for that interval.

                Each matrix C[i, :, :] transforms cardinal spline basis functions
                to B-spline basis functions for the i-th interval as

                    C[i, :, :] @ [cardinal values] = [B-spline values in interval].

                If `out` was provided, returns the same array.

        Raises:
            ValueError: If `out` is provided and has incorrect shape or dtype.
        """
        return _tabulate_Bspline_cardinal_1D_extraction_impl(
            self.knots, self.degree, self.tolerance, out=out
        )

    def insert_knots(self, new_knots: npt.ArrayLike) -> BsplineSpace1D:
        """Return a new BsplineSpace1D with additional knots inserted.

        Args:
            new_knots (npt.ArrayLike): 1D array-like of knot values to insert.
                Must be non-empty.  Values must lie in
                ``[knots[degree], knots[-degree-1]]``.  Inserting a value
                already present increases its multiplicity; multiplicity cannot
                exceed ``degree + 1``.  Repeated values in ``new_knots`` insert
                that knot multiple times in one call.

        Returns:
            BsplineSpace1D: New space with the inserted knots.

        Raises:
            ValueError: If ``new_knots`` is empty.
            ValueError: If ``new_knots`` is not 1D.
            ValueError: If any new knot lies outside the domain.
            ValueError: If inserting a knot would exceed maximum multiplicity
                ``degree + 1``.
        """
        arr = np.asarray(new_knots, dtype=self.dtype)
        if arr.size == 0:
            raise ValueError("new_knots must not be empty.")
        merged = _compute_inserted_knot_vector_1d(self._knots, self._degree, arr, self._tol)
        return BsplineSpace1D(merged, self._degree)

    def subdivide(self, n_subdivisions: int, regularity: int | None = None) -> BsplineSpace1D:
        """Return a new BsplineSpace1D with each knot span split into equal sub-spans.

        For every non-zero knot interval ``[t_i, t_{i+1})``, inserts
        ``n_subdivisions - 1`` uniformly spaced knot values.  Each value is
        repeated ``degree - regularity`` times so that the resulting B-spline
        has ``C^regularity`` continuity at every inserted knot.

        Args:
            n_subdivisions (int): Number of equal sub-spans per existing knot
                interval.  Must be >= 2.
            regularity (int | None): Continuity order at every inserted knot.
                Must be in ``[-1, degree - 1]``.  ``None`` (default) uses
                ``degree - 1``, giving maximum continuity (multiplicity 1).

        Returns:
            BsplineSpace1D: New space with uniformly refined knot vector.

        Raises:
            ValueError: If ``n_subdivisions < 2``.
            ValueError: If ``regularity`` is outside ``[-1, degree - 1]``.
        """
        if n_subdivisions < 2:  # noqa: PLR2004
            raise ValueError(f"n_subdivisions must be >= 2, got {n_subdivisions}")
        eff_regularity = self._degree - 1 if regularity is None else regularity
        if eff_regularity < -1 or eff_regularity > self._degree - 1:
            raise ValueError(
                f"regularity must be in [-1, degree - 1] = [-1, {self._degree - 1}], "
                f"got {eff_regularity}"
            )
        new_knots = _compute_uniform_subdivision_knots(
            self._knots, self._degree, self._tol, n_subdivisions, eff_regularity
        )
        return self.insert_knots(new_knots)

    def restrict(
        self, interval_lo: int, interval_hi: int
    ) -> tuple[BsplineSpace1D, npt.NDArray[np.int64]]:
        """Return the windowed sub-space over the intervals ``[interval_lo, interval_hi)``.

        The windowed knot vector is a pure slice of this space's knots (never
        re-clamped), chosen so the windowed basis functions are exactly the global
        basis functions supported on the windowed intervals; they therefore equal
        the global basis pointwise over those intervals.

        Args:
            interval_lo (int): First interval (knot span) of the window, inclusive.
            interval_hi (int): One past the last interval of the window, exclusive.
                Must satisfy ``0 <= interval_lo < interval_hi <= num_intervals``.

        Returns:
            tuple[BsplineSpace1D, npt.NDArray[np.int64]]: The windowed space and a
            read-only ``local_to_global_dof`` array of shape
            ``(windowed_space.num_basis,)`` mapping each windowed basis index to its
            index in this space.

        Raises:
            ValueError: If the space is periodic, or the interval range is invalid.
        """
        handle, local_to_global_dof = self._h.restrict(int(interval_lo), int(interval_hi))
        local_to_global_dof.flags.writeable = False
        windowed = BsplineSpace1D._from_handle(handle)
        return windowed, local_to_global_dof
