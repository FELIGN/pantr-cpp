"""Patch interfaces (private)."""

from collections.abc import Sequence

import numpy
from numpy.typing import NDArray

import pantr._pantr.bspline

def validate_interface(
    patch_a: int,
    face_a: int,
    patch_b: int,
    face_b: int,
    axis_map: Sequence[int],
    flips: Sequence[bool],
) -> None:
    """Refuse an interface record that cannot hold (the checks of the C++ `Interface`)."""

def match_face_cps(
    num_basis_a: Sequence[int],
    face_a: int,
    num_basis_b: Sequence[int],
    face_b: int,
    axis_map: Sequence[int],
    flips: Sequence[bool],
) -> tuple[NDArray[numpy.int64], NDArray[numpy.int64]]:
    """The flat control-point ids of two faces, paired under an axis correspondence."""

def detect_interfaces(
    patches: Sequence[pantr._pantr.bspline.BsplineF32 | pantr._pantr.bspline.BsplineF64],
    tol: float | None,
) -> list[tuple[int, int, int, int, list[int], list[bool]]]:
    """The conforming interfaces among patches, as (patch_a, face_a, patch_b, face_b, axis_map, flips) tuples sorted by their first four entries."""

def multi_patch_interfaces(
    patches: Sequence[pantr._pantr.bspline.BsplineF32] | Sequence[pantr._pantr.bspline.BsplineF64],
    interfaces: Sequence[tuple[int, int, int, int, Sequence[int], Sequence[bool]]] | None,
    tol: float | None,
) -> list[tuple[int, int, int, int, list[int], list[bool]]]:
    """Validate a collection of patches of one dtype (and the supplied interfaces, each verified against the geometry), returning its interfaces: the supplied ones, or the detected ones."""
