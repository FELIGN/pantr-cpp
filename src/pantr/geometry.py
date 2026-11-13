"""Axis-aligned bounding boxes and geometry primitives.

This module exposes :class:`AABB`, a lightweight, immutable value type for an
axis-aligned bounding box in any spatial dimension ``ndim >= 1``. It is the
shared box / domain primitive for PaNTr (for example, the parametric domain of
a spline space or the bounds of a grid cell) and for libraries built on PaNTr.

An :class:`AABB` stores two read-only ``float64`` corner arrays :attr:`AABB.lo` and
:attr:`AABB.hi` of shape ``(ndim,)``. Entries may be finite or ``+/- numpy.inf`` (for
unbounded axes); ``numpy.nan`` is rejected. A point ``x`` is inside the box when
``lo[i] <= x[i] <= hi[i]`` on every axis; a box with ``lo[i] > hi[i]`` on some
axis is *empty* and reported by :meth:`AABB.is_empty`.

Main exports:

- :class:`AABB` -- immutable axis-aligned bounding box.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy import typing as npt

from ._ext import ext

_cpp: Any = ext.geometry
"""The compiled AABB (private; untyped)."""


class _AffineMap(Protocol):
    """Structural protocol for an affine map ``T(x) = matrix @ x + offset``.

    Any object exposing :attr:`dim`, :attr:`matrix`, and :attr:`offset` satisfies
    :meth:`AABB.transform`. This decouples the box primitive from a specific
    affine-transform class (for example :class:`pantr.transform.AffineTransform`
    or a downstream library's own affine type).
    """

    @property
    def dim(self) -> int:
        """Get the spatial dimension of the map.

        Returns:
            int: The dimension ``n``.
        """
        ...

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        """Get the linear part ``A`` of the map.

        Returns:
            npt.NDArray[np.float64]: The ``(n, n)`` matrix.
        """
        ...

    @property
    def offset(self) -> npt.NDArray[np.float64]:
        """Get the translation part ``b`` of the map.

        Returns:
            npt.NDArray[np.float64]: The ``(n,)`` vector.
        """
        ...


def _as_float64(arr: npt.ArrayLike, *, name: str) -> npt.NDArray[np.float64]:
    """Coerce ``arr`` to a ``float64`` array, preserving rank.

    Integer and unsigned-integer inputs are cast to ``float64``; boolean and
    non-numeric inputs are rejected. Results of rank ``>= 1`` are made
    C-contiguous.

    Args:
        arr (npt.ArrayLike): Input array or array-like.
        name (str): Argument name, used in error messages.

    Returns:
        npt.NDArray[np.float64]: A ``float64`` view or copy of ``arr``.

    Raises:
        TypeError: If ``arr`` cannot be converted to an ndarray, or its dtype
            is neither integer nor floating-point (e.g. boolean, complex,
            object).
    """
    try:
        a = np.asarray(arr)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} could not be converted to an ndarray: {exc}") from exc
    if a.dtype.kind not in ("f", "i", "u"):
        raise TypeError(f"{name} must have a numeric (int or float) dtype; got {a.dtype!r}.")
    a = a.astype(np.float64, copy=False)
    if a.ndim >= 1 and not a.flags.c_contiguous:
        a = np.ascontiguousarray(a)
    return a


class AABB:
    """Axis-aligned bounding box in any spatial dimension ``ndim >= 1``.

    An :class:`AABB` stores two 1-D arrays :attr:`lo` and :attr:`hi` of equal
    shape ``(ndim,)``. Entries may be finite or ``+/- numpy.inf`` (for unbounded
    axes); ``numpy.nan`` is rejected. Instances are frozen: the stored arrays
    are C-contiguous, ``float64``, and flagged read-only, and ``__setattr__``
    rejects attempts to replace the bound attributes.

    The sign convention is the natural one: a point ``x`` is "inside" the box
    when ``lo[i] <= x[i] <= hi[i]`` on every axis. A box with ``lo[i] > hi[i]``
    on some axis is *empty* (no point satisfies the inequality);
    :meth:`is_empty` reports this.
    """

    __slots__ = ("_h", "hi", "lo")

    #: The compiled value (private).
    _h: Any
    #: Lower corner, shape ``(ndim,)``, read-only.
    lo: npt.NDArray[np.float64]
    #: Upper corner, shape ``(ndim,)``, read-only.
    hi: npt.NDArray[np.float64]

    def __init__(self, lo: npt.ArrayLike, hi: npt.ArrayLike) -> None:
        """Build and validate an AABB from array-like corners.

        The arguments are coerced to C-contiguous ``float64`` arrays and
        checked for NaN. The resulting buffers are cached as read-only
        attributes on ``self``.

        Args:
            lo (npt.ArrayLike): Lower corner; array-like of finite-or-infinite floats.
                Any rank is accepted and ravelled to 1-D of length ``ndim``.
            hi (npt.ArrayLike): Upper corner, same shape and semantics as ``lo``.

        Raises:
            ValueError: If the shapes mismatch, contain a NaN, or the implied
                ``ndim`` is less than 1.
            TypeError: If ``lo`` or ``hi`` has a non-numeric dtype.
        """
        lo_arr = _as_float64(lo, name="lo").ravel()
        hi_arr = _as_float64(hi, name="hi").ravel()
        _init_from_handle(self, _cpp.AABB(lo_arr, hi_arr))

    @classmethod
    def _from_handle(cls, h: Any) -> AABB:  # noqa: ANN401
        """Wrap a compiled AABB without copying it.

        Args:
            h (Any): The compiled value.

        Returns:
            AABB: The wrapper.
        """
        box = object.__new__(cls)
        _init_from_handle(box, h)
        return box

    def __reduce__(self) -> tuple[type[AABB], tuple[npt.NDArray[np.float64], ...]]:
        """Pickle and copy through the constructor.

        Returns:
            tuple[type[AABB], tuple[npt.NDArray[np.float64], ...]]: The class and its corners.
        """
        return (type(self), (self.lo, self.hi))

    def __setattr__(self, name: str, value: object) -> None:
        """Reject post-construction attribute writes.

        Raises:
            AttributeError: Always -- :class:`AABB` is immutable.
        """
        raise AttributeError(f"AABB is immutable; cannot set attribute {name!r}.")

    def __delattr__(self, name: str) -> None:
        """Reject attribute deletion.

        Raises:
            AttributeError: Always -- :class:`AABB` is immutable.
        """
        raise AttributeError(f"AABB is immutable; cannot delete attribute {name!r}.")

    def __repr__(self) -> str:
        """Return a compact representation useful for debugging.

        Returns:
            str: ``"AABB(lo=..., hi=...)"`` with corner values as Python lists.
        """
        return f"AABB(lo={self.lo.tolist()!r}, hi={self.hi.tolist()!r})"

    def __eq__(self, other: object) -> bool:
        """Compare value-based equality: two AABBs are equal iff their bounds match.

        This is *value* equality, not geometric equality: two empty AABBs whose
        corner arrays differ (e.g. ``AABB.empty(2)`` vs a hand-constructed empty)
        compare unequal even though they contain the same set of points.

        Args:
            other (object): Expected to be an :class:`AABB`.

        Returns:
            bool: ``True`` when both corner arrays are element-wise equal
            (``+inf == +inf`` and ``-inf == -inf`` as usual, and ``-0.0 == 0.0``,
            which :meth:`__hash__` normalizes away to stay compatible).

        Note:
            Returns :data:`NotImplemented` for non-:class:`AABB` ``other`` so
            that Python's reflected equality protocol works correctly.
        """
        if not isinstance(other, AABB):
            return NotImplemented
        return bool(self._h.equals(other._h))

    def __hash__(self) -> int:
        """Hash the corner bytes, with signed zero normalized to ``+0.0``.

        The bounds are hashed as bytes, so the raw bit patterns of ``-0.0`` and
        ``+0.0`` would hash differently even though :meth:`__eq__` considers them
        equal. Adding ``0.0`` maps ``-0.0`` to ``+0.0`` and leaves every other
        value bitwise untouched (both infinities included), which brings the two
        methods back into agreement: ``-0.0`` and ``+0.0`` are the only distinct
        ``float64`` patterns that compare equal, and NaN -- the other case where
        bitwise and IEEE comparison differ -- is rejected by :meth:`__init__`.
        Only the hash is normalized; the stored bounds keep the sign given.

        Returns:
            int: Hash of the normalized corner bytes; equal AABBs hash equal.
        """
        return hash(((self.lo + 0.0).tobytes(), (self.hi + 0.0).tobytes()))

    @property
    def ndim(self) -> int:
        """Get the spatial dimensionality of the box.

        Returns:
            int: The number of axes (``>= 1``).
        """
        return int(self.lo.shape[0])

    def is_empty(self) -> bool:
        """Check whether the box contains no points.

        A box is empty iff ``lo[i] > hi[i]`` on at least one axis. Useful when
        constructing boxes manually, accepting boxes from external sources, or
        after :meth:`pad` with a negative radius.

        Returns:
            bool: ``True`` when the box is empty.
        """
        return bool(self._h.is_empty())

    def contains_point(self, x: npt.ArrayLike) -> bool:
        """Check whether point ``x`` lies inside or on the boundary of the box.

        A point is inside iff ``lo[i] <= x[i] <= hi[i]`` on every axis.
        An empty box contains no points.

        Args:
            x (npt.ArrayLike): Point to test; array-like of floats, ravelled to
                1-D of length ``ndim``.

        Returns:
            bool: ``True`` when ``x`` is inside or on the boundary.

        Raises:
            ValueError: If the ravelled ``x`` does not have length ``ndim``,
                or if ``x`` contains NaN.
            TypeError: If ``x`` has a non-numeric dtype.
        """
        return bool(self._h.contains_point(_as_float64(x, name="x").ravel()))

    def overlaps(self, other: AABB) -> bool:
        """Check whether ``self`` and ``other`` share at least one point.

        Empty boxes (on either side) overlap nothing.

        Args:
            other (AABB): The box to test against; must share :attr:`ndim`.

        Returns:
            bool: ``True`` when the two boxes intersect, ``False`` otherwise.

        Raises:
            ValueError: If ``other.ndim != self.ndim``.
        """
        return bool(self._h.overlaps(other._h))

    def union(self, other: AABB) -> AABB:
        """Return the smallest axis-aligned box that contains ``self`` and ``other``.

        Empty boxes act as neutral elements: ``union(empty, x) == x``.

        Args:
            other (AABB): Box to union with; must share :attr:`ndim`.

        Returns:
            AABB: The union bounding box.

        Raises:
            ValueError: If ``other.ndim != self.ndim``.
        """
        h = self._h.union_with(other._h)
        if h.same_rep(other._h):
            return other
        if h.same_rep(self._h):
            return self
        return AABB._from_handle(h)

    def intersect(self, other: AABB) -> AABB | None:
        """Return the axis-aligned intersection, or ``None`` if disjoint.

        Args:
            other (AABB): Box to intersect with; must share :attr:`ndim`.

        Returns:
            AABB | None: The intersection, or ``None`` when the two boxes do not
            overlap (including the case where either operand is empty).

        Raises:
            ValueError: If ``other.ndim != self.ndim``.
        """
        h = self._h.intersect(other._h)
        return None if h is None else AABB._from_handle(h)

    def pad(self, r: float | npt.ArrayLike) -> AABB:
        """Inflate the box by ``r`` on every axis (symmetric on both sides).

        A scalar ``r`` expands every axis by the same amount; a length-``ndim``
        array expands each axis by its own entry. Padding an unbounded axis
        leaves it unbounded. Padding by a negative ``r`` shrinks the box and can
        produce an empty AABB, which is allowed and reported by
        :meth:`is_empty`.

        Args:
            r (float | npt.ArrayLike): Per-side padding amount. Scalar or
                length-``ndim`` array-like of finite reals.

        Returns:
            AABB: The padded box.

        Raises:
            ValueError: If ``r`` has the wrong shape or non-finite entries.
        """
        arr = _as_float64(r, name="r")
        if arr.ndim == 0:
            return AABB._from_handle(self._h.pad_scalar(float(arr)))
        return AABB._from_handle(self._h.pad(arr.ravel()))

    def transform(self, affine: _AffineMap) -> AABB:
        """Return the axis-aligned wrap of the image of ``self`` under ``affine``.

        For an affine map ``T(x) = A x + b``, the tight axis-aligned box
        containing ``T(self)`` is computed from the per-entry sign of ``A``:
        each output axis ``i`` receives contributions ``A[i, j] * lo[j]`` or
        ``A[i, j] * hi[j]`` -- whichever gives the minimum / maximum -- summed
        over ``j``. Zero entries of ``A`` contribute nothing even when
        ``lo[j]`` / ``hi[j]`` is infinite; this preserves the correct wrap for
        unbounded axes that the transform projects out.

        Args:
            affine (_AffineMap): The affine map to apply -- any object exposing
                ``dim``, ``matrix`` (``A``), and ``offset`` (``b``). Its
                dimension must match :attr:`ndim`.

        Returns:
            AABB: The axis-aligned wrap of the transformed box.

        Raises:
            ValueError: If ``affine.dim != self.ndim``, if ``affine.matrix`` is
                not square ``(ndim, ndim)``, or if the wrap produces NaN (e.g.
                a matrix containing NaN, or ``inf - inf`` arithmetic from
                opposing infinite bounds in the same row).
        """
        if self.is_empty():
            return AABB.empty(self.ndim)
        matrix = np.ascontiguousarray(affine.matrix, dtype=np.float64)
        offset = np.ascontiguousarray(affine.offset, dtype=np.float64).ravel()
        return AABB._from_handle(self._h.transform(affine.dim, matrix, offset))

    @staticmethod
    def unbounded(ndim: int) -> AABB:
        """Build the everywhere-true AABB ``(-inf, +inf)^ndim``.

        Args:
            ndim (int): Spatial dimension (``>= 1``).

        Returns:
            AABB: The unbounded AABB.

        Raises:
            ValueError: If ``ndim < 1``.
        """
        return AABB._from_handle(_cpp.AABB.unbounded(ndim))

    @staticmethod
    def empty(ndim: int) -> AABB:
        """Build an empty (zero-volume) AABB with ``lo > hi``.

        An empty AABB contains no points (detected by :meth:`is_empty`) and acts
        as the neutral element of :meth:`union` (``union(empty, x) == x``).
        Symmetric counterpart to :meth:`unbounded`.

        Args:
            ndim (int): Spatial dimension (``>= 1``).

        Returns:
            AABB: An empty AABB (``lo = +inf``, ``hi = -inf``).

        Raises:
            ValueError: If ``ndim < 1``.
        """
        return AABB._from_handle(_cpp.AABB.empty(ndim))

    @staticmethod
    def from_bounds(bounds: npt.ArrayLike) -> AABB:
        """Build an AABB from a ``(ndim, 2)`` ``[[lo, hi], ...]`` array.

        The dual of :meth:`as_bounds`; useful when interoperating with
        ``numpy.histogramdd``-style bounds arguments.

        Args:
            bounds (npt.ArrayLike): ``(ndim, 2)`` array-like of ``[lo, hi]`` rows.

        Returns:
            AABB: The constructed AABB.

        Raises:
            ValueError: If ``bounds`` does not have shape ``(ndim, 2)`` with
                ``ndim >= 1``.
            TypeError: If ``bounds`` has a non-numeric dtype.
        """
        return AABB._from_handle(_cpp.AABB.from_bounds(_as_float64(bounds, name="bounds")))

    def as_bounds(self) -> npt.NDArray[np.float64]:
        """Return the AABB as a ``(ndim, 2)`` ``[[lo, hi], ...]`` array.

        Returns:
            npt.NDArray[np.float64]: A freshly-allocated, writeable, C-contiguous
            ``(ndim, 2)`` array.
        """
        out: npt.NDArray[np.float64] = self._h.as_bounds()
        return out


def _init_from_handle(box: AABB, h: Any) -> None:  # noqa: ANN401
    """Set an AABB wrapper's handle and its cached read-only corners.

    The corners are read once (DR-25: the value is immutable), as views of the compiled value.

    Args:
        box (AABB): The wrapper being initialised.
        h (Any): The compiled value.
    """
    object.__setattr__(box, "_h", h)
    object.__setattr__(box, "lo", h.lo)
    object.__setattr__(box, "hi", h.hi)


__all__ = ["AABB"]
