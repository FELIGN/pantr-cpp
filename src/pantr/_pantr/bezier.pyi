"""Bezier values and operations (private)."""

from collections.abc import Sequence
from typing import Annotated

import numpy
from numpy.typing import NDArray

import pantr._pantr.quad

class BezierF64:
    """An immutable Bezier map (control points in a shared, read-only buffer)."""

    def __init__(
        self,
        control_points: Annotated[
            NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)
        ],
        rational: bool,
    ) -> None:
        """Validate and copy the control points."""

    @staticmethod
    def borrow(
        control_points: Annotated[
            NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)
        ],
        rational: bool,
    ) -> BezierF64:
        """A Bezier over the caller's array, without copying it (interim)."""

    @property
    def shape(self) -> list[int]:
        """The control-point extents."""

    @property
    def rational(self) -> bool:
        """Whether the last coordinate is a weight."""

    @property
    def control_points(self) -> Annotated[NDArray[numpy.float64], dict(writable=False)]:
        """Read-only view of the control points."""

    def same_rep(self, other: BezierF64) -> bool:
        """Whether both share one representation."""

    def evaluate(
        self,
        pts: Annotated[
            NDArray[numpy.float64],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
    ) -> NDArray[numpy.float64]:
        """Values at (n, dim) points: (n, rank)."""

    def evaluate_lattice(
        self, lattice: pantr._pantr.quad.PointsLatticeF64
    ) -> NDArray[numpy.float64]:
        """Values on a lattice: (n_0, ..., n_{dim-1}, rank)."""

    def evaluate_derivatives(
        self,
        pts: Annotated[
            NDArray[numpy.float64],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
        orders: Sequence[int],
    ) -> NDArray[numpy.float64]:
        """A mixed partial derivative at (n, dim) points."""

    def evaluate_derivatives_lattice(
        self, lattice: pantr._pantr.quad.PointsLatticeF64, orders: Sequence[int]
    ) -> NDArray[numpy.float64]:
        """A mixed partial derivative on a lattice."""

    def derivative(self, direction: int, keep_degree: bool) -> BezierF64:
        """The hodograph in one direction."""

    def elevate_degree(self, increments: Sequence[int]) -> BezierF64:
        """Degree elevation per direction."""

    def multiply(self, other: BezierF64) -> BezierF64:
        """The pointwise product."""

    def compose(self, inner: BezierF64) -> BezierF64:
        """self(inner(t))."""

    def split(self, direction: int, value: float) -> tuple[BezierF64, BezierF64]:
        """Split at a parameter value: (left, right)."""

    def slice(self, axis: int, value: float) -> BezierF64:
        """Fix one direction (dim >= 2)."""

    def slice_point(self, axis: int, value: float) -> NDArray[numpy.float64]:
        """The point of a curve at a parameter (projected)."""

    def restrict(self, bounds: Sequence[tuple[float, float] | None]) -> BezierF64:
        """The restriction to a sub-box, reparametrised to [0, 1]^dim."""

    def collapse_along_axis(
        self,
        axis: int,
        values: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
    ) -> BezierF64:
        """Fix every direction but one."""

    def reversed(self, direction: int) -> BezierF64:
        """Reverse one direction."""

    def permuted(self, permutation: Sequence[int]) -> BezierF64:
        """Reorder the directions."""

    def transformed(
        self,
        matrix: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
        offset: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
    ) -> BezierF64:
        """Apply x -> A x + b to the geometry."""

class BezierF32:
    """An immutable Bezier map (control points in a shared, read-only buffer)."""

    def __init__(
        self,
        control_points: Annotated[
            NDArray[numpy.float32], dict(order="C", device="cpu", writable=False)
        ],
        rational: bool,
    ) -> None:
        """Validate and copy the control points."""

    @staticmethod
    def borrow(
        control_points: Annotated[
            NDArray[numpy.float32], dict(order="C", device="cpu", writable=False)
        ],
        rational: bool,
    ) -> BezierF32:
        """A Bezier over the caller's array, without copying it (interim)."""

    @property
    def shape(self) -> list[int]:
        """The control-point extents."""

    @property
    def rational(self) -> bool:
        """Whether the last coordinate is a weight."""

    @property
    def control_points(self) -> Annotated[NDArray[numpy.float32], dict(writable=False)]:
        """Read-only view of the control points."""

    def same_rep(self, other: BezierF32) -> bool:
        """Whether both share one representation."""

    def evaluate(
        self,
        pts: Annotated[
            NDArray[numpy.float32],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
    ) -> NDArray[numpy.float32]:
        """Values at (n, dim) points: (n, rank)."""

    def evaluate_lattice(
        self, lattice: pantr._pantr.quad.PointsLatticeF32
    ) -> NDArray[numpy.float32]:
        """Values on a lattice: (n_0, ..., n_{dim-1}, rank)."""

    def evaluate_derivatives(
        self,
        pts: Annotated[
            NDArray[numpy.float32],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
        orders: Sequence[int],
    ) -> NDArray[numpy.float32]:
        """A mixed partial derivative at (n, dim) points."""

    def evaluate_derivatives_lattice(
        self, lattice: pantr._pantr.quad.PointsLatticeF32, orders: Sequence[int]
    ) -> NDArray[numpy.float32]:
        """A mixed partial derivative on a lattice."""

    def derivative(self, direction: int, keep_degree: bool) -> BezierF32:
        """The hodograph in one direction."""

    def elevate_degree(self, increments: Sequence[int]) -> BezierF32:
        """Degree elevation per direction."""

    def multiply(self, other: BezierF32) -> BezierF32:
        """The pointwise product."""

    def compose(self, inner: BezierF32) -> BezierF32:
        """self(inner(t))."""

    def split(self, direction: int, value: float) -> tuple[BezierF32, BezierF32]:
        """Split at a parameter value: (left, right)."""

    def slice(self, axis: int, value: float) -> BezierF32:
        """Fix one direction (dim >= 2)."""

    def slice_point(self, axis: int, value: float) -> NDArray[numpy.float32]:
        """The point of a curve at a parameter (projected)."""

    def restrict(self, bounds: Sequence[tuple[float, float] | None]) -> BezierF32:
        """The restriction to a sub-box, reparametrised to [0, 1]^dim."""

    def collapse_along_axis(
        self,
        axis: int,
        values: Annotated[
            NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
    ) -> BezierF32:
        """Fix every direction but one."""

    def reversed(self, direction: int) -> BezierF32:
        """Reverse one direction."""

    def permuted(self, permutation: Sequence[int]) -> BezierF32:
        """Reorder the directions."""

    def transformed(
        self,
        matrix: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
        offset: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
    ) -> BezierF32:
        """Apply x -> A x + b to the geometry."""
