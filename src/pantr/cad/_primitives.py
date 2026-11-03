"""Constructive geometry primitives.

Defines create_line, create_circle, create_bilinear, and create_trilinear.

Provides functions to create basic B-spline objects from geometric
descriptions (points, corners, radii, angles).  All primitives produce
rank-3 (3D) output so they can be freely composed with higher-level
operations such as ``create_extrusion`` and ``create_revolution``.
"""

from __future__ import annotations

import numpy as np
from numpy import typing as npt

from .._ext import ext
from ..bspline import Bspline, BsplineSpace1D


def _linear_space_1d(dtype: npt.DTypeLike = np.float64) -> BsplineSpace1D:
    """Create a degree-1 B-spline space on [0, 1].

    Args:
        dtype: Floating-point dtype for the knot vector.

    Returns:
        BsplineSpace1D: A clamped degree-1 space with knots ``[0, 0, 1, 1]``.
    """
    knots = np.array([0.0, 0.0, 1.0, 1.0], dtype=dtype)
    return BsplineSpace1D(knots, degree=1)


def _points(point: npt.ArrayLike) -> list[float]:
    """Flatten a point to the list of coordinates the compiled functions take.

    Args:
        point (npt.ArrayLike): Array-like of coordinates.

    Returns:
        list[float]: The coordinates; the compiled function zero-pads and checks their number.
    """
    return np.asarray(point, dtype=np.float64).ravel().tolist()


def _sweep_pair(angle: float | tuple[float, float] | None) -> tuple[float, float] | None:
    """Read a sweep specification as the ``(start, end)`` pair the compiled functions take.

    Args:
        angle (float | tuple[float, float] | None): ``None`` (the full turn), an end angle (the
            start is 0), or a ``(start, end)`` pair, in radians.

    Returns:
        tuple[float, float] | None: ``(start, end)``, or ``None`` for the full turn.
    """
    if angle is None:
        return None
    if isinstance(angle, tuple | list):
        start, end = angle
        return (float(start), float(end))
    return (0.0, float(angle))


def create_line(
    p0: npt.ArrayLike = (0.0, 0.0, 0.0),
    p1: npt.ArrayLike = (1.0, 0.0, 0.0),
) -> Bspline:
    """Construct a straight-line B-spline curve between two points.

    Creates a degree-1 (linear), non-rational B-spline curve with two
    control points.  Input points shorter than 3 elements are zero-padded
    to 3D.

    Args:
        p0: Start point.  Defaults to the origin.
        p1: End point.  Defaults to ``(1, 0, 0)``.

    Returns:
        Bspline: A 1D, degree-1, rank-3, non-rational B-spline curve.

    Example:
        >>> crv = create_line([0, 0], [1, 1])
        >>> crv.degree
        (1,)
        >>> crv.rank
        3
    """
    h = ext.cad.create_line(_points(p0), _points(p1))
    return Bspline._from_handle(h)


def create_circle(
    radius: float = 1.0,
    center: npt.ArrayLike | None = None,
    angle: float | tuple[float, float] | None = None,
) -> Bspline:
    """Construct a NURBS circular arc or full circle.

    Creates a degree-2 rational B-spline curve in the *xy*-plane.  The
    arc is split into spans of at most 90 degrees each.  Interior knots
    have multiplicity 2 (equal to the degree), giving C0 continuity at
    arc junctions.  This is the standard exact representation of conics
    using rational quadratic B-splines.

    The number of spans depends on the sweep angle:

    - ``|sweep| <= 90``: 1 span, 3 control points
    - ``90 < |sweep| <= 180``: 2 spans, 5 control points
    - ``180 < |sweep| <= 270``: 3 spans, 7 control points
    - ``270 < |sweep| <= 360``: 4 spans, 9 control points

    Args:
        radius: Circle radius.  Defaults to 1.
        center: Center point (up to 3D, zero-padded).  If ``None``,
            the circle is centered at the origin.
        angle: Sweep specification.

            - ``None`` -- full circle (360 degrees).
            - ``float`` -- arc from angle 0 to the given value (radians).
            - ``(start, end)`` -- arc from *start* to *end* (radians).

    Returns:
        Bspline: A 1D, degree-2, rank-3, rational B-spline curve.

    Example:
        >>> crv = create_circle()
        >>> crv.degree
        (2,)
        >>> crv.is_rational
        True
    """
    h = ext.cad.create_circle(
        radius, None if center is None else _points(center), _sweep_pair(angle)
    )
    return Bspline._from_handle(h)


def create_bilinear(corners: npt.ArrayLike | None = None) -> Bspline:
    """Construct a bilinear B-spline surface from four corner points.

    Creates a degree (1, 1), non-rational B-spline surface with 2 x 2
    control points.

    The corner ordering follows the tensor-product convention::

        corners[0, 1]       corners[1, 1]
        o------------------o
        |  v               |
        |  ^               |
        |  |               |
        |  +-------> u     |
        o------------------o
        corners[0, 0]       corners[1, 0]

    Args:
        corners: Array of shape ``(2, 2, rank)`` with ``rank <= 3``.
            Coordinates shorter than 3D are zero-padded.
            If ``None``, defaults to the unit square
            ``[-0.5, 0.5]^2 x {0}`` in the *xy*-plane.

    Returns:
        Bspline: A 2D, degree-(1, 1), rank-3, non-rational B-spline surface.

    Raises:
        ValueError: If *corners* does not have shape ``(2, 2, rank)``
            with ``1 <= rank <= 3``.

    Example:
        >>> srf = create_bilinear()
        >>> srf.degree
        (1, 1)
        >>> srf.dim
        2
    """
    if corners is None:
        return Bspline._from_handle(ext.cad.create_bilinear())
    arr = np.asarray(corners, dtype=np.float64)
    return Bspline._from_handle(ext.cad.create_bilinear(arr.ravel().tolist(), list(arr.shape)))


def create_trilinear(corners: npt.ArrayLike | None = None) -> Bspline:
    """Construct a trilinear B-spline volume from eight corner points.

    Creates a degree (1, 1, 1), non-rational B-spline volume with
    2 x 2 x 2 control points.

    The corner ordering follows the tensor-product convention::

           corners[0,1,1]       corners[1,1,1]
           o--------------------o
          /|                   /|
         / |                  / |                 w
        o--------------------o  |                 ^  v
        |  | corners[0,0,1]  |  | corners[1,0,1]  | /
        |  |                 |  |                 |/
        |  o-----------------|--o                 +------> u
        | / corners[0,1,0]   | / corners[1,1,0]
        |/                   |/
        o--------------------o
        corners[0,0,0]       corners[1,0,0]

    Args:
        corners: Array of shape ``(2, 2, 2, rank)`` with ``rank <= 3``.
            Coordinates shorter than 3D are zero-padded.
            If ``None``, defaults to the unit cube
            ``[-0.5, 0.5]^3`` centered at the origin.

    Returns:
        Bspline: A 3D, degree-(1, 1, 1), rank-3, non-rational B-spline volume.

    Raises:
        ValueError: If *corners* does not have shape ``(2, 2, 2, rank)``
            with ``1 <= rank <= 3``.

    Example:
        >>> vol = create_trilinear()
        >>> vol.degree
        (1, 1, 1)
        >>> vol.dim
        3
    """
    if corners is None:
        return Bspline._from_handle(ext.cad.create_trilinear())
    arr = np.asarray(corners, dtype=np.float64)
    return Bspline._from_handle(ext.cad.create_trilinear(arr.ravel().tolist(), list(arr.shape)))
