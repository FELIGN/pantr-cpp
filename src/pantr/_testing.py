"""The one private surface the test suite may import (not a public API).

Internals worth testing directly, which no public function exposes, are bound in the compiled
module's ``_testing`` submodule and re-exported here; a test imports them from ``pantr._testing``
and from no other private path (design/port_design.md §3.8). Nothing in the library imports this
module.

- The Bernstein recurrences at one point, the degree up to which the unmirrored one runs, and
  the Bernstein derivative tabulation (no public function returns derivatives).
- The generic Gram projection behind the change-of-basis builders.
- The change-of-basis process tables: their degree bound, size and reset.
"""

from __future__ import annotations

from typing import Final

from ._ext import ext

_t = ext._testing

BERNSTEIN_MAX_UNMIRRORED_DEGREE_FLOAT64: Final[int] = _t.bernstein_max_unmirrored_degree_double
"""Largest float64 degree tabulated by the plain forward Bernstein recurrence."""

BERNSTEIN_MAX_UNMIRRORED_DEGREE_FLOAT32: Final[int] = _t.bernstein_max_unmirrored_degree_float
"""Largest float32 degree tabulated by the plain forward Bernstein recurrence."""

bernstein_row_mirrored = _t.bernstein_row_mirrored
bernstein_row_mirrored_f32 = _t.bernstein_row_mirrored_f32
bernstein_row_unmirrored = _t.bernstein_row_unmirrored
bernstein_row_unmirrored_f32 = _t.bernstein_row_unmirrored_f32
bernstein_derivatives_point = _t.bernstein_derivatives_point
bernstein_derivatives_point_f32 = _t.bernstein_derivatives_point_f32
tabulate_bernstein_derivatives_1d = _t.tabulate_bernstein_derivatives_1d
tabulate_bernstein_derivatives_1d_f32 = _t.tabulate_bernstein_derivatives_1d_f32
gram_projection = _t.gram_projection
gram_projection_f32 = _t.gram_projection_f32

CHANGE_BASIS_CACHE_MAX_DEGREE: Final[int] = _t.change_basis_cache_max_degree
"""Largest degree whose change-of-basis matrix the process tables keep."""

change_basis_cache_size = _t.change_basis_cache_size
change_basis_cache_clear = _t.change_basis_cache_clear

BERNSTEIN: Final[int] = 0
"""Wire value of the Bernstein family, for :data:`gram_projection`."""
CARDINAL_BSPLINE: Final[int] = 1
"""Wire value of the cardinal B-spline family, for :data:`gram_projection`."""
LEGENDRE: Final[int] = 2
"""Wire value of the shifted Legendre family, for :data:`gram_projection`."""

__all__ = [
    "BERNSTEIN",
    "BERNSTEIN_MAX_UNMIRRORED_DEGREE_FLOAT32",
    "BERNSTEIN_MAX_UNMIRRORED_DEGREE_FLOAT64",
    "CARDINAL_BSPLINE",
    "CHANGE_BASIS_CACHE_MAX_DEGREE",
    "LEGENDRE",
    "bernstein_derivatives_point",
    "bernstein_derivatives_point_f32",
    "bernstein_row_mirrored",
    "bernstein_row_mirrored_f32",
    "bernstein_row_unmirrored",
    "bernstein_row_unmirrored_f32",
    "change_basis_cache_clear",
    "change_basis_cache_size",
    "gram_projection",
    "gram_projection_f32",
    "tabulate_bernstein_derivatives_1d",
    "tabulate_bernstein_derivatives_1d_f32",
]
