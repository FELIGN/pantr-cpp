"""Basis tabulation (private)."""

from collections.abc import Sequence
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray

import pantr._pantr.quad

@overload
def tabulate_1d(
    family: int,
    degree: int,
    variant: int,
    pts: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")]
    | None,
) -> object: ...
@overload
def tabulate_1d(
    family: int,
    degree: int,
    variant: int,
    pts: Annotated[
        NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")]
    | None,
) -> object:
    """Tabulate the 1D basis `family` (0 Bernstein, 1 cardinal B-spline, 2 Legendre, 3 Lagrange on node family `variant`) of `degree` at `pts`, into `out` when given (then returning None)."""

@overload
def tabulate_lattice(
    family: int,
    degrees: Sequence[int],
    variant: int,
    lattice: pantr._pantr.quad.PointsLatticeF64,
    order: int,
    out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")]
    | None,
) -> object: ...
@overload
def tabulate_lattice(
    family: int,
    degrees: Sequence[int],
    variant: int,
    lattice: pantr._pantr.quad.PointsLatticeF32,
    order: int,
    out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")]
    | None,
) -> object:
    """Tabulate the tensor-product basis of `family` with `degrees` on a lattice, points and functions in `order` (0 C, 1 F), into `out` when given (then returning None)."""

@overload
def tabulate_points(
    family: int,
    degrees: Sequence[int],
    variant: int,
    pts: Annotated[
        NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu", writable=False)
    ],
    order: int,
    out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")]
    | None,
) -> object: ...
@overload
def tabulate_points(
    family: int,
    degrees: Sequence[int],
    variant: int,
    pts: Annotated[
        NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu", writable=False)
    ],
    order: int,
    out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")]
    | None,
) -> object:
    """Tabulate the tensor-product basis of `family` with `degrees` at scattered points `(n, dim)`, functions in `order` (0 C, 1 F), into `out` when given (then returning None)."""
