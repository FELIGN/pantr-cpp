"""Bezier geometric objects, approximation functions, and Bernstein polynomial root finding.

This module provides :class:`Bezier`, which stores control points to represent
a parametric Bezier curve, surface, or volume. Degree is derived from the
control point array shape. Evaluation, degree reduction, fitting and root finding
run in pantr's C++ library (``include/pantr/bezier/``, ``include/pantr/roots/``).

- :func:`interpolate_bezier`, :func:`fit_bezier`: approximation functions.
- :func:`create_from_bspline`: create a Bézier from a B-spline.

Root-finding exports:

- :func:`find_roots` -- find all roots (single or batch, auto-dispatch).
- :func:`find_monotone_root` -- fast solver for monotone polynomials (single or batch).
- :func:`set_root_provider`, :func:`get_root_provider` -- select the root provider
  (pantr's own, or BeRF-FR's when ``pantr-berf`` is installed).
"""

from ._bezier import Bezier, create_from_bspline
from ._bezier_interpolate import fit_bezier, interpolate_bezier
from ._root_finding import (
    find_monotone_root,
    find_roots,
    get_root_provider,
    set_root_provider,
)

__all__ = [
    "Bezier",
    "create_from_bspline",
    "find_monotone_root",
    "find_roots",
    "fit_bezier",
    "get_root_provider",
    "interpolate_bezier",
    "set_root_provider",
]
