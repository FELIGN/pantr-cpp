"""Bezier geometric objects and approximation functions.

This module provides :class:`Bezier`, which stores control points to represent
a parametric Bezier curve, surface, or volume. Degree is derived from the
control point array shape. Evaluation, degree reduction and fitting run in pantr's
C++ library (``include/pantr/bezier/``).

- :func:`interpolate_bezier`, :func:`fit_bezier`: approximation functions.
- :func:`create_from_bspline`: create a Bézier from a B-spline.
"""

from ._bezier import Bezier, create_from_bspline
from ._bezier_interpolate import fit_bezier, interpolate_bezier

__all__ = [
    "Bezier",
    "create_from_bspline",
    "fit_bezier",
    "interpolate_bezier",
]
