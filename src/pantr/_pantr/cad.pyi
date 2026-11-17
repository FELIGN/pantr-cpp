"""Constructive geometry (private)."""

from collections.abc import Sequence
from typing import overload

import pantr._pantr.bspline

def create_line(p0: Sequence[float], p1: Sequence[float]) -> pantr._pantr.bspline.BsplineF64:
    """A degree-1 curve between two points (up to three coordinates)."""

def create_circle(
    radius: float, center: Sequence[float] | None, sweep: Sequence[float] | None
) -> pantr._pantr.bspline.BsplineF64:
    """A rational quadratic circle or arc; `sweep` is (start, end) in radians, none for the full circle."""

@overload
def create_bilinear() -> pantr._pantr.bspline.BsplineF64:
    """The bilinear unit square [-0.5, 0.5]^2 in the xy-plane."""

@overload
def create_bilinear(
    values: Sequence[float], shape: Sequence[int]
) -> pantr._pantr.bspline.BsplineF64:
    """A bilinear surface from corner coordinates, row-major over `shape` = (2, 2, rank)."""

@overload
def create_trilinear() -> pantr._pantr.bspline.BsplineF64:
    """The trilinear unit cube [-0.5, 0.5]^3."""

@overload
def create_trilinear(
    values: Sequence[float], shape: Sequence[int]
) -> pantr._pantr.bspline.BsplineF64:
    """A trilinear volume from corner coordinates, row-major over `shape` = (2, 2, 2, rank)."""

def make_compat(
    bsplines: Sequence[pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64],
    axes: Sequence[int] | None,
) -> list[pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64]:
    """B-splines made compatible (degree and knots) along the axes; none means every axis."""

def create_ruled(
    bspline1: pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64,
    bspline2: pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64,
) -> pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64:
    """The ruled surface or volume between two B-splines."""

def create_revolution(
    bspline: pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64,
    point: Sequence[float],
    axis: Sequence[float],
    sweep: Sequence[float] | None,
) -> pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64:
    """A curve or surface revolved about the axis through `point` with direction `axis`."""

def create_coons_surface(
    c_v0: pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64,
    c_v1: pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64,
    c_u0: pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64,
    c_u1: pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64,
) -> pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64:
    """The Coons patch of four boundary curves."""

def create_coons_volume(
    face_u0: pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64,
    face_u1: pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64,
    face_v0: pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64,
    face_v1: pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64,
    face_w0: pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64,
    face_w1: pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64,
) -> pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64:
    """The Coons volume of six boundary faces."""

def join(
    bspline1: pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64,
    bspline2: pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64,
    axis: int,
) -> pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64:
    """Two B-splines joined along a parametric axis with C0 continuity."""
