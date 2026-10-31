"""B-spline spaces and knot vectors (private)."""

from collections.abc import Sequence
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray

import pantr._pantr.bezier
import pantr._pantr.quad

class BsplineSpace1DF64:
    """An immutable 1D B-spline space (knots and degree)."""

    def __init__(
        self,
        knots: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        degree: int,
        periodic: bool,
        snap_knots: bool,
    ) -> None:
        """Validate and copy the knots into a new space."""

    @staticmethod
    def from_state(
        knots: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        degree: int,
        periodic: bool,
        tol: float,
    ) -> BsplineSpace1DF64:
        """Rebuild a space from its stored knots and tolerance (no snapping)."""

    def same_rep(self, other: BsplineSpace1DF64) -> bool:
        """Whether both share one representation."""

    @property
    def degree(self) -> int:
        """The degree."""

    @property
    def periodic(self) -> bool:
        """Whether the space is periodic."""

    @property
    def tolerance(self) -> float:
        """The absolute parametric tolerance."""

    @property
    def num_basis(self) -> int:
        """The number of basis functions."""

    @property
    def num_intervals(self) -> int:
        """The number of in-domain intervals."""

    @property
    def left_open(self) -> bool:
        """Whether the left end is open."""

    @property
    def right_open(self) -> bool:
        """Whether the right end is open."""

    @property
    def bezier_like(self) -> bool:
        """Whether the space is one piece."""

    @property
    def knots(self) -> Annotated[NDArray[numpy.float64], dict(writable=False)]:
        """Read-only view of the knot vector."""

    def unique_knots_and_multiplicity(
        self, in_domain: bool
    ) -> tuple[
        Annotated[NDArray[numpy.float64], dict(writable=False)],
        Annotated[NDArray[numpy.int64], dict(writable=False)],
    ]:
        """Read-only views of the distinct knots and their multiplicities."""

    def first_basis_per_interval(self) -> Annotated[NDArray[numpy.int64], dict(writable=False)]:
        """Read-only view of the first supported function of each interval."""

    def cardinal_intervals(
        self, out: Annotated[NDArray[numpy.bool_], dict(shape=(None,), order="C", device="cpu")]
    ) -> None:
        """Write the cardinal-interval flags into `out`."""

    def restrict(self, lo: int, hi: int) -> tuple[BsplineSpace1DF64, NDArray[numpy.int64]]:
        """The windowed space over intervals [lo, hi) and its map."""

    def greville(self) -> NDArray[numpy.float64]:
        """The Greville abscissae."""

    @overload
    def all_in_domain(
        self,
        pts: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
    ) -> bool: ...
    @overload
    def all_in_domain(
        self,
        pts: Annotated[
            NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
    ) -> bool:
        """Whether every point lies in the domain."""

    @overload
    def tabulate_basis(
        self,
        pts: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")],
        first: Annotated[NDArray[numpy.int64], dict(shape=(None,), order="C", device="cpu")],
        validate: bool,
    ) -> None: ...
    @overload
    def tabulate_basis(
        self,
        pts: Annotated[
            NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")],
        first: Annotated[NDArray[numpy.int64], dict(shape=(None,), order="C", device="cpu")],
        validate: bool,
    ) -> None:
        """Write the basis values and first indices at `pts` into `out` and `first`."""

    @overload
    def tabulate_basis_derivatives(
        self,
        pts: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        n_deriv: int,
        out: Annotated[
            NDArray[numpy.float64], dict(shape=(None, None, None), order="C", device="cpu")
        ]
        | None,
        first: Annotated[NDArray[numpy.int64], dict(shape=(None,), order="C", device="cpu")] | None,
        validate: bool,
    ) -> None: ...
    @overload
    def tabulate_basis_derivatives(
        self,
        pts: Annotated[
            NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        n_deriv: int,
        out: Annotated[
            NDArray[numpy.float32], dict(shape=(None, None, None), order="C", device="cpu")
        ]
        | None,
        first: Annotated[NDArray[numpy.int64], dict(shape=(None,), order="C", device="cpu")] | None,
        validate: bool,
    ) -> None:
        """Write the basis derivatives of orders 0..n_deriv at `pts` into `out` and `first`."""

    def bezier_extraction(
        self,
        out: Annotated[
            NDArray[numpy.float64], dict(shape=(None, None, None), order="C", device="cpu")
        ],
    ) -> None:
        """Write the Bezier extraction operators (intervals, degree + 1, degree + 1) into `out`."""

    def lagrange_extraction(
        self,
        variant: int,
        out: Annotated[
            NDArray[numpy.float64], dict(shape=(None, None, None), order="C", device="cpu")
        ],
    ) -> None:
        """Write the Lagrange extraction operators of a node variant into `out`."""

    def cardinal_extraction(
        self,
        out: Annotated[
            NDArray[numpy.float64], dict(shape=(None, None, None), order="C", device="cpu")
        ],
    ) -> None:
        """Write the cardinal B-spline extraction operators into `out`."""

    def bezier_identity_mask(
        self, out: Annotated[NDArray[numpy.bool_], dict(shape=(None,), order="C", device="cpu")]
    ) -> None:
        """Write into `out` which intervals are already Bezier patches (identity operator)."""

class BsplineSpace1DF32:
    """An immutable 1D B-spline space (knots and degree)."""

    def __init__(
        self,
        knots: Annotated[
            NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        degree: int,
        periodic: bool,
        snap_knots: bool,
    ) -> None:
        """Validate and copy the knots into a new space."""

    @staticmethod
    def from_state(
        knots: Annotated[
            NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        degree: int,
        periodic: bool,
        tol: float,
    ) -> BsplineSpace1DF32:
        """Rebuild a space from its stored knots and tolerance (no snapping)."""

    def same_rep(self, other: BsplineSpace1DF32) -> bool:
        """Whether both share one representation."""

    @property
    def degree(self) -> int:
        """The degree."""

    @property
    def periodic(self) -> bool:
        """Whether the space is periodic."""

    @property
    def tolerance(self) -> float:
        """The absolute parametric tolerance."""

    @property
    def num_basis(self) -> int:
        """The number of basis functions."""

    @property
    def num_intervals(self) -> int:
        """The number of in-domain intervals."""

    @property
    def left_open(self) -> bool:
        """Whether the left end is open."""

    @property
    def right_open(self) -> bool:
        """Whether the right end is open."""

    @property
    def bezier_like(self) -> bool:
        """Whether the space is one piece."""

    @property
    def knots(self) -> Annotated[NDArray[numpy.float32], dict(writable=False)]:
        """Read-only view of the knot vector."""

    def unique_knots_and_multiplicity(
        self, in_domain: bool
    ) -> tuple[
        Annotated[NDArray[numpy.float32], dict(writable=False)],
        Annotated[NDArray[numpy.int64], dict(writable=False)],
    ]:
        """Read-only views of the distinct knots and their multiplicities."""

    def first_basis_per_interval(self) -> Annotated[NDArray[numpy.int64], dict(writable=False)]:
        """Read-only view of the first supported function of each interval."""

    def cardinal_intervals(
        self, out: Annotated[NDArray[numpy.bool_], dict(shape=(None,), order="C", device="cpu")]
    ) -> None:
        """Write the cardinal-interval flags into `out`."""

    def restrict(self, lo: int, hi: int) -> tuple[BsplineSpace1DF32, NDArray[numpy.int64]]:
        """The windowed space over intervals [lo, hi) and its map."""

    def greville(self) -> NDArray[numpy.float32]:
        """The Greville abscissae."""

    @overload
    def all_in_domain(
        self,
        pts: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
    ) -> bool: ...
    @overload
    def all_in_domain(
        self,
        pts: Annotated[
            NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
    ) -> bool:
        """Whether every point lies in the domain."""

    @overload
    def tabulate_basis(
        self,
        pts: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")],
        first: Annotated[NDArray[numpy.int64], dict(shape=(None,), order="C", device="cpu")],
        validate: bool,
    ) -> None: ...
    @overload
    def tabulate_basis(
        self,
        pts: Annotated[
            NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")],
        first: Annotated[NDArray[numpy.int64], dict(shape=(None,), order="C", device="cpu")],
        validate: bool,
    ) -> None:
        """Write the basis values and first indices at `pts` into `out` and `first`."""

    @overload
    def tabulate_basis_derivatives(
        self,
        pts: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        n_deriv: int,
        out: Annotated[
            NDArray[numpy.float64], dict(shape=(None, None, None), order="C", device="cpu")
        ]
        | None,
        first: Annotated[NDArray[numpy.int64], dict(shape=(None,), order="C", device="cpu")] | None,
        validate: bool,
    ) -> None: ...
    @overload
    def tabulate_basis_derivatives(
        self,
        pts: Annotated[
            NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        n_deriv: int,
        out: Annotated[
            NDArray[numpy.float32], dict(shape=(None, None, None), order="C", device="cpu")
        ]
        | None,
        first: Annotated[NDArray[numpy.int64], dict(shape=(None,), order="C", device="cpu")] | None,
        validate: bool,
    ) -> None:
        """Write the basis derivatives of orders 0..n_deriv at `pts` into `out` and `first`."""

    def bezier_extraction(
        self,
        out: Annotated[
            NDArray[numpy.float32], dict(shape=(None, None, None), order="C", device="cpu")
        ],
    ) -> None:
        """Write the Bezier extraction operators (intervals, degree + 1, degree + 1) into `out`."""

    def lagrange_extraction(
        self,
        variant: int,
        out: Annotated[
            NDArray[numpy.float32], dict(shape=(None, None, None), order="C", device="cpu")
        ],
    ) -> None:
        """Write the Lagrange extraction operators of a node variant into `out`."""

    def cardinal_extraction(
        self,
        out: Annotated[
            NDArray[numpy.float32], dict(shape=(None, None, None), order="C", device="cpu")
        ],
    ) -> None:
        """Write the cardinal B-spline extraction operators into `out`."""

    def bezier_identity_mask(
        self, out: Annotated[NDArray[numpy.bool_], dict(shape=(None,), order="C", device="cpu")]
    ) -> None:
        """Write into `out` which intervals are already Bezier patches (identity operator)."""

class BsplineSpaceF64:
    """An immutable tensor-product B-spline space."""

    def __init__(self, spaces: Sequence[BsplineSpace1DF64]) -> None:
        """The tensor product of `spaces`."""

    @property
    def dim(self) -> int:
        """The number of directions."""

    @property
    def num_local_functions(self) -> int:
        """The non-zero functions per point."""

    @overload
    def first_direction_outside(
        self,
        pts: Annotated[
            NDArray[numpy.float64],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
    ) -> int: ...
    @overload
    def first_direction_outside(
        self,
        pts: Annotated[
            NDArray[numpy.float32],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
    ) -> int:
        """The first direction with a point outside the domain, or -1."""

    @overload
    def first_direction_outside(self, lattice: pantr._pantr.quad.PointsLatticeF64) -> int: ...
    @overload
    def first_direction_outside(self, lattice: pantr._pantr.quad.PointsLatticeF32) -> int:
        """The first direction with a coordinate outside the domain, or -1."""

    @overload
    def tabulate_points(
        self,
        pts: Annotated[
            NDArray[numpy.float64],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
        out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")],
        first: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order="C", device="cpu")],
    ) -> None: ...
    @overload
    def tabulate_points(
        self,
        pts: Annotated[
            NDArray[numpy.float32],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
        out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")],
        first: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order="C", device="cpu")],
    ) -> None:
        """Write the basis at points `(n, dim)` into `out` `(n, nfun)` and `first` `(n, dim)`."""

    @overload
    def tabulate_lattice(
        self,
        lattice: pantr._pantr.quad.PointsLatticeF64,
        out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")],
        first: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order="C", device="cpu")],
    ) -> None: ...
    @overload
    def tabulate_lattice(
        self,
        lattice: pantr._pantr.quad.PointsLatticeF32,
        out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")],
        first: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order="C", device="cpu")],
    ) -> None:
        """Write the basis on a lattice (C order) into `out` and `first`."""

class BsplineSpaceF32:
    """An immutable tensor-product B-spline space."""

    def __init__(self, spaces: Sequence[BsplineSpace1DF32]) -> None:
        """The tensor product of `spaces`."""

    @property
    def dim(self) -> int:
        """The number of directions."""

    @property
    def num_local_functions(self) -> int:
        """The non-zero functions per point."""

    @overload
    def first_direction_outside(
        self,
        pts: Annotated[
            NDArray[numpy.float64],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
    ) -> int: ...
    @overload
    def first_direction_outside(
        self,
        pts: Annotated[
            NDArray[numpy.float32],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
    ) -> int:
        """The first direction with a point outside the domain, or -1."""

    @overload
    def first_direction_outside(self, lattice: pantr._pantr.quad.PointsLatticeF64) -> int: ...
    @overload
    def first_direction_outside(self, lattice: pantr._pantr.quad.PointsLatticeF32) -> int:
        """The first direction with a coordinate outside the domain, or -1."""

    @overload
    def tabulate_points(
        self,
        pts: Annotated[
            NDArray[numpy.float64],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
        out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")],
        first: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order="C", device="cpu")],
    ) -> None: ...
    @overload
    def tabulate_points(
        self,
        pts: Annotated[
            NDArray[numpy.float32],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
        out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")],
        first: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order="C", device="cpu")],
    ) -> None:
        """Write the basis at points `(n, dim)` into `out` `(n, nfun)` and `first` `(n, dim)`."""

    @overload
    def tabulate_lattice(
        self,
        lattice: pantr._pantr.quad.PointsLatticeF64,
        out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")],
        first: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order="C", device="cpu")],
    ) -> None: ...
    @overload
    def tabulate_lattice(
        self,
        lattice: pantr._pantr.quad.PointsLatticeF32,
        out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")],
        first: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order="C", device="cpu")],
    ) -> None:
        """Write the basis on a lattice (C order) into `out` and `first`."""

def uniform_open_knots(
    num_intervals: int, degree: int, continuity: int, start: float, end: float
) -> NDArray[numpy.float64]:
    """A uniform open knot vector."""

def uniform_periodic_knots(
    num_intervals: int, degree: int, continuity: int, start: float, end: float
) -> NDArray[numpy.float64]:
    """A uniform periodic knot vector."""

def cardinal_knots(num_intervals: int, degree: int) -> NDArray[numpy.float64]:
    """The cardinal B-spline knot vector."""

def uniform_open_knots_f32(
    num_intervals: int, degree: int, continuity: int, start: float, end: float
) -> NDArray[numpy.float32]:
    """A uniform open knot vector."""

def uniform_periodic_knots_f32(
    num_intervals: int, degree: int, continuity: int, start: float, end: float
) -> NDArray[numpy.float32]:
    """A uniform periodic knot vector."""

def cardinal_knots_f32(num_intervals: int, degree: int) -> NDArray[numpy.float32]:
    """The cardinal B-spline knot vector."""

class BsplineF64:
    """An immutable B-spline map (a space and control points in a shared, read-only buffer)."""

    def __init__(
        self,
        space: BsplineSpaceF64,
        control_points: Annotated[
            NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)
        ],
        rational: bool,
    ) -> None:
        """Validate and copy the control points (any shape: only their number is read)."""

    @staticmethod
    def cp_size_for(space: BsplineSpaceF64, count: int) -> int:
        """The coordinates per control point that `count` values make in `space`, or the ValueError of a count that is not a multiple of the number of basis functions."""

    @property
    def shape(self) -> list[int]:
        """The control-point extents (*num_basis, cp_size)."""

    @property
    def cp_size(self) -> int:
        """The coordinates per control point."""

    @property
    def rank(self) -> int:
        """The output rank, the weight excluded."""

    @property
    def is_rational(self) -> bool:
        """Whether the last coordinate is a weight."""

    @property
    def space(self) -> BsplineSpaceF64:
        """The space."""

    @property
    def directions(self) -> list[BsplineSpace1DF64]:
        """The 1D space of each direction (each sharing its representation)."""

    @property
    def control_points(self) -> Annotated[NDArray[numpy.float64], dict(writable=False)]:
        """Read-only view of the control points."""

    def same_rep(self, other: BsplineF64) -> bool:
        """Whether both share one representation."""

    def evaluate(
        self,
        pts: Annotated[
            NDArray[numpy.float64],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
        out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")],
    ) -> None:
        """Write the values at (n, dim) points into `out` (n, cp_size); the weight column of a rational spline stays."""

    def evaluate_derivatives(
        self,
        pts: Annotated[
            NDArray[numpy.float64],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
        orders: Sequence[int],
        out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")],
    ) -> None:
        """Write a mixed partial derivative at (n, dim) points into `out` (n, rank)."""

    @overload
    def evaluate_lattice(
        self,
        lattice: pantr._pantr.quad.PointsLatticeF64,
        out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")],
    ) -> None: ...
    @overload
    def evaluate_lattice(
        self,
        lattice: pantr._pantr.quad.PointsLatticeF32,
        out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")],
    ) -> None:
        """Write the values on a lattice (C order) into `out` (num_points, cp_size); the weight column of a rational spline stays."""

    @overload
    def evaluate_derivatives_lattice(
        self,
        lattice: pantr._pantr.quad.PointsLatticeF64,
        orders: Sequence[int],
        out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")],
    ) -> None: ...
    @overload
    def evaluate_derivatives_lattice(
        self,
        lattice: pantr._pantr.quad.PointsLatticeF32,
        orders: Sequence[int],
        out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")],
    ) -> None:
        """Write a mixed partial derivative on a lattice into `out` (num_points, rank)."""

    def slice(self, axis: int, value: float) -> BsplineF64:
        """Fix one direction of a surface or a volume at a parameter value."""

    def slice_point(self, axis: int, value: float) -> NDArray[numpy.float64]:
        """The point of a curve at a parameter value (projected if rational)."""

    def boundary(self, axis: int, side: int) -> BsplineF64:
        """The boundary of a surface or a volume at the start (0) or the end (1) of a direction."""

    def boundary_point(self, axis: int, side: int) -> NDArray[numpy.float64]:
        """The end point of a curve (projected if rational)."""

    def reversed(self, direction: int) -> BsplineF64:
        """Reverse one direction."""

    def permuted(self, permutation: Sequence[int]) -> BsplineF64:
        """Reorder the directions."""

    def transformed(
        self,
        matrix: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
        offset: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
    ) -> BsplineF64:
        """Apply x -> A x + b to the geometry."""

    def insert_knots(
        self,
        new_knots: Sequence[
            Annotated[
                NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
            ]
            | None
        ],
    ) -> BsplineF64:
        """Insert knots per direction (none or empty skips a direction); a periodic direction is refined through its open form."""

    def to_open(self) -> BsplineF64:
        """Convert every direction to open (clamped) non-periodic form."""

    def to_periodic(self, continuity: Sequence[int | None]) -> BsplineF64:
        """Convert the directions with a continuity (-1 for the maximum; none skips) to periodic form."""

    def subdivide(self, counts: Sequence[int], regularity: int | None) -> BsplineF64:
        """Subdivide each direction whose count is at least 2 uniformly."""

    def restrict(self, bounds: Sequence[tuple[float, float] | None]) -> BsplineF64:
        """Restrict each direction with bounds to that interval."""

    def split(self, direction: int, value: float) -> tuple[BsplineF64, BsplineF64]:
        """Split at a parameter value in one direction."""

    def elevate_degree(self, increments: Sequence[int]) -> BsplineF64:
        """Elevate the degree in each direction by its increment (a periodic direction is elevated through its open form)."""

    def reduce_degree(self, decrements: Sequence[int]) -> BsplineF64:
        """Reduce the degree in each direction by its decrement (an approximation)."""

    def remove_knots(
        self,
        knot_values: Sequence[
            Annotated[
                NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
            ]
            | None
        ],
        num: int | None,
        tol: float | None,
    ) -> BsplineF64:
        """Remove the knots per direction (none or empty skips a direction), up to `num` times each, while the deviation stays within `tol` (the round-off floor when none)."""

    def derivative(self, direction: int) -> BsplineF64:
        """The partial derivative of a non-rational B-spline."""

    def derivative_keep_degree(self, direction: int) -> BsplineF64:
        """The partial derivative of a non-rational B-spline, elevated back to its degree."""

    def to_bezier(self, copy: bool) -> pantr._pantr.bezier.BezierF64:
        """The Bezier of a single-element B-spline (periodic and unclamped ones through open form); with `copy` false a Bezier-like B-spline shares its buffer."""

    def to_beziers(self) -> tuple[list[int], list[pantr._pantr.bezier.BezierF64]]:
        """The Bezier patches of every element, in C order, and the elements per direction."""

class BsplineF32:
    """An immutable B-spline map (a space and control points in a shared, read-only buffer)."""

    def __init__(
        self,
        space: BsplineSpaceF32,
        control_points: Annotated[
            NDArray[numpy.float32], dict(order="C", device="cpu", writable=False)
        ],
        rational: bool,
    ) -> None:
        """Validate and copy the control points (any shape: only their number is read)."""

    @staticmethod
    def cp_size_for(space: BsplineSpaceF32, count: int) -> int:
        """The coordinates per control point that `count` values make in `space`, or the ValueError of a count that is not a multiple of the number of basis functions."""

    @property
    def shape(self) -> list[int]:
        """The control-point extents (*num_basis, cp_size)."""

    @property
    def cp_size(self) -> int:
        """The coordinates per control point."""

    @property
    def rank(self) -> int:
        """The output rank, the weight excluded."""

    @property
    def is_rational(self) -> bool:
        """Whether the last coordinate is a weight."""

    @property
    def space(self) -> BsplineSpaceF32:
        """The space."""

    @property
    def directions(self) -> list[BsplineSpace1DF32]:
        """The 1D space of each direction (each sharing its representation)."""

    @property
    def control_points(self) -> Annotated[NDArray[numpy.float32], dict(writable=False)]:
        """Read-only view of the control points."""

    def same_rep(self, other: BsplineF32) -> bool:
        """Whether both share one representation."""

    def evaluate(
        self,
        pts: Annotated[
            NDArray[numpy.float32],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
        out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")],
    ) -> None:
        """Write the values at (n, dim) points into `out` (n, cp_size); the weight column of a rational spline stays."""

    def evaluate_derivatives(
        self,
        pts: Annotated[
            NDArray[numpy.float32],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
        orders: Sequence[int],
        out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")],
    ) -> None:
        """Write a mixed partial derivative at (n, dim) points into `out` (n, rank)."""

    @overload
    def evaluate_lattice(
        self,
        lattice: pantr._pantr.quad.PointsLatticeF64,
        out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")],
    ) -> None: ...
    @overload
    def evaluate_lattice(
        self,
        lattice: pantr._pantr.quad.PointsLatticeF32,
        out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")],
    ) -> None:
        """Write the values on a lattice (C order) into `out` (num_points, cp_size); the weight column of a rational spline stays."""

    @overload
    def evaluate_derivatives_lattice(
        self,
        lattice: pantr._pantr.quad.PointsLatticeF64,
        orders: Sequence[int],
        out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")],
    ) -> None: ...
    @overload
    def evaluate_derivatives_lattice(
        self,
        lattice: pantr._pantr.quad.PointsLatticeF32,
        orders: Sequence[int],
        out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")],
    ) -> None:
        """Write a mixed partial derivative on a lattice into `out` (num_points, rank)."""

    def slice(self, axis: int, value: float) -> BsplineF32:
        """Fix one direction of a surface or a volume at a parameter value."""

    def slice_point(self, axis: int, value: float) -> NDArray[numpy.float32]:
        """The point of a curve at a parameter value (projected if rational)."""

    def boundary(self, axis: int, side: int) -> BsplineF32:
        """The boundary of a surface or a volume at the start (0) or the end (1) of a direction."""

    def boundary_point(self, axis: int, side: int) -> NDArray[numpy.float32]:
        """The end point of a curve (projected if rational)."""

    def reversed(self, direction: int) -> BsplineF32:
        """Reverse one direction."""

    def permuted(self, permutation: Sequence[int]) -> BsplineF32:
        """Reorder the directions."""

    def transformed(
        self,
        matrix: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
        offset: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
    ) -> BsplineF32:
        """Apply x -> A x + b to the geometry."""

    def insert_knots(
        self,
        new_knots: Sequence[
            Annotated[
                NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
            ]
            | None
        ],
    ) -> BsplineF32:
        """Insert knots per direction (none or empty skips a direction); a periodic direction is refined through its open form."""

    def to_open(self) -> BsplineF32:
        """Convert every direction to open (clamped) non-periodic form."""

    def to_periodic(self, continuity: Sequence[int | None]) -> BsplineF32:
        """Convert the directions with a continuity (-1 for the maximum; none skips) to periodic form."""

    def subdivide(self, counts: Sequence[int], regularity: int | None) -> BsplineF32:
        """Subdivide each direction whose count is at least 2 uniformly."""

    def restrict(self, bounds: Sequence[tuple[float, float] | None]) -> BsplineF32:
        """Restrict each direction with bounds to that interval."""

    def split(self, direction: int, value: float) -> tuple[BsplineF32, BsplineF32]:
        """Split at a parameter value in one direction."""

    def elevate_degree(self, increments: Sequence[int]) -> BsplineF32:
        """Elevate the degree in each direction by its increment (a periodic direction is elevated through its open form)."""

    def reduce_degree(self, decrements: Sequence[int]) -> BsplineF32:
        """Reduce the degree in each direction by its decrement (an approximation)."""

    def remove_knots(
        self,
        knot_values: Sequence[
            Annotated[
                NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
            ]
            | None
        ],
        num: int | None,
        tol: float | None,
    ) -> BsplineF32:
        """Remove the knots per direction (none or empty skips a direction), up to `num` times each, while the deviation stays within `tol` (the round-off floor when none)."""

    def derivative(self, direction: int) -> BsplineF32:
        """The partial derivative of a non-rational B-spline."""

    def derivative_keep_degree(self, direction: int) -> BsplineF32:
        """The partial derivative of a non-rational B-spline, elevated back to its degree."""

    def to_bezier(self, copy: bool) -> pantr._pantr.bezier.BezierF32:
        """The Bezier of a single-element B-spline (periodic and unclamped ones through open form); with `copy` false a Bezier-like B-spline shares its buffer."""

    def to_beziers(self) -> tuple[list[int], list[pantr._pantr.bezier.BezierF32]]:
        """The Bezier patches of every element, in C order, and the elements per direction."""

def compute_inserted_knot_vector(
    knots: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    degree: int,
    new_knots: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    tol: float,
) -> NDArray[numpy.float64]:
    """Validate new knots and merge them into a knot vector."""

def compute_uniform_subdivision_knots(
    knots: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    degree: int,
    tol: float,
    n_subdivisions: int,
    regularity: int | None,
) -> NDArray[numpy.float64]:
    """The knots that subdivide every span uniformly."""

def compute_inserted_knot_vector_f32(
    knots: Annotated[
        NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    degree: int,
    new_knots: Annotated[
        NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    tol: float,
) -> NDArray[numpy.float32]:
    """Validate new knots and merge them into a knot vector."""

def compute_uniform_subdivision_knots_f32(
    knots: Annotated[
        NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    degree: int,
    tol: float,
    n_subdivisions: int,
    regularity: int | None,
) -> NDArray[numpy.float32]:
    """The knots that subdivide every span uniformly."""

def refine_to_common_space_1d(
    a: BsplineF64, b: BsplineF64, direction: int
) -> tuple[BsplineF64, BsplineF64]:
    """Insert into each of two B-splines the knots the other has in one direction and it lacks."""

def refine_to_common_space_1d_f32(
    a: BsplineF32, b: BsplineF32, direction: int
) -> tuple[BsplineF32, BsplineF32]:
    """Insert into each of two B-splines the knots the other has in one direction and it lacks."""

class SpanwiseExtractionF64:
    """An immutable set of element operators of a tensor-product space onto a target basis."""

    def __init__(self, space: BsplineSpaceF64, target: int, variant: int) -> None:
        """Build the operators (target 0 Bezier, 1 Lagrange, 2 cardinal); refuses periodic directions."""

    @property
    def dim(self) -> int:
        """The number of directions."""

    @property
    def num_intervals(self) -> list[int]:
        """The intervals per direction."""

    @property
    def input_shape_per_dir(self) -> list[int]:
        """The columns of each direction's operator."""

    @property
    def output_shape_per_dir(self) -> list[int]:
        """The rows of each direction's operator."""

    @property
    def num_identity_elements(self) -> int:
        """The elements whose operator is the identity in every direction."""

    def compact_ops(self, k: int) -> Annotated[NDArray[numpy.float64], dict(writable=False)]:
        """Read-only view of direction `k`'s non-identity operators."""

    def idx_map(self, k: int) -> Annotated[NDArray[numpy.int64], dict(writable=False)]:
        """Read-only view of direction `k`'s row of each interval."""

    def identity_mask(self, k: int) -> Annotated[NDArray[numpy.bool_], dict(writable=False)]:
        """Read-only view of which intervals of direction `k` are identity."""

    def required_scratch(self, op: int) -> int:
        """The scratch entries an application of one cell needs."""

    def kron_operator(
        self,
        cell: Sequence[int],
        out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")],
    ) -> None:
        """Write the Kronecker operator of the element `cell` into `out`."""

    def apply_many(
        self,
        v: Annotated[
            NDArray[numpy.float64],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
        cells: Annotated[
            NDArray[numpy.int64], dict(shape=(None, None), order="C", device="cpu", writable=False)
        ],
        out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")],
        scratch: Annotated[
            NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")
        ]
        | None,
    ) -> None:
        """Write M_c v[c] for every cell (rows of `cells`) into `out`."""

    def apply_transpose_many(
        self,
        v: Annotated[
            NDArray[numpy.float64],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
        cells: Annotated[
            NDArray[numpy.int64], dict(shape=(None, None), order="C", device="cpu", writable=False)
        ],
        out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")],
        scratch: Annotated[
            NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")
        ]
        | None,
    ) -> None:
        """Write M_c^T v[c] for every cell into `out`."""

    def apply_mt_k_m_many(
        self,
        k: Annotated[
            NDArray[numpy.float64],
            dict(shape=(None, None, None), order="C", device="cpu", writable=False),
        ],
        cells: Annotated[
            NDArray[numpy.int64], dict(shape=(None, None), order="C", device="cpu", writable=False)
        ],
        out: Annotated[
            NDArray[numpy.float64], dict(shape=(None, None, None), order="C", device="cpu")
        ],
        scratch: Annotated[
            NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")
        ]
        | None,
    ) -> None:
        """Write M_c^T K[c] M_c for every cell into `out`."""

    def apply_m_k_mt_many(
        self,
        k: Annotated[
            NDArray[numpy.float64],
            dict(shape=(None, None, None), order="C", device="cpu", writable=False),
        ],
        cells: Annotated[
            NDArray[numpy.int64], dict(shape=(None, None), order="C", device="cpu", writable=False)
        ],
        out: Annotated[
            NDArray[numpy.float64], dict(shape=(None, None, None), order="C", device="cpu")
        ],
        scratch: Annotated[
            NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")
        ]
        | None,
    ) -> None:
        """Write M_c K[c] M_c^T for every cell into `out`."""

    def apply(
        self,
        v: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        cell: Sequence[int],
        out: Annotated[NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu")],
        scratch: Annotated[NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu")]
        | None,
    ) -> None:
        """Write M v for the element `cell` (an interval per direction) into `out`."""

    def apply_transpose(
        self,
        v: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        cell: Sequence[int],
        out: Annotated[NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu")],
        scratch: Annotated[NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu")]
        | None,
    ) -> None:
        """Write M^T v for the element `cell` into `out`."""

    def apply_mt_k_m(
        self,
        k: Annotated[
            NDArray[numpy.float64],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
        cell: Sequence[int],
        out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")],
        scratch: Annotated[NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu")]
        | None,
    ) -> None:
        """Write M^T K M for the element `cell` into `out`."""

    def apply_m_k_mt(
        self,
        k: Annotated[
            NDArray[numpy.float64],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
        cell: Sequence[int],
        out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu")],
        scratch: Annotated[NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu")]
        | None,
    ) -> None:
        """Write M K M^T for the element `cell` into `out`."""

class SpanwiseExtractionF32:
    """An immutable set of element operators of a tensor-product space onto a target basis."""

    def __init__(self, space: BsplineSpaceF32, target: int, variant: int) -> None:
        """Build the operators (target 0 Bezier, 1 Lagrange, 2 cardinal); refuses periodic directions."""

    @property
    def dim(self) -> int:
        """The number of directions."""

    @property
    def num_intervals(self) -> list[int]:
        """The intervals per direction."""

    @property
    def input_shape_per_dir(self) -> list[int]:
        """The columns of each direction's operator."""

    @property
    def output_shape_per_dir(self) -> list[int]:
        """The rows of each direction's operator."""

    @property
    def num_identity_elements(self) -> int:
        """The elements whose operator is the identity in every direction."""

    def compact_ops(self, k: int) -> Annotated[NDArray[numpy.float32], dict(writable=False)]:
        """Read-only view of direction `k`'s non-identity operators."""

    def idx_map(self, k: int) -> Annotated[NDArray[numpy.int64], dict(writable=False)]:
        """Read-only view of direction `k`'s row of each interval."""

    def identity_mask(self, k: int) -> Annotated[NDArray[numpy.bool_], dict(writable=False)]:
        """Read-only view of which intervals of direction `k` are identity."""

    def required_scratch(self, op: int) -> int:
        """The scratch entries an application of one cell needs."""

    def kron_operator(
        self,
        cell: Sequence[int],
        out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")],
    ) -> None:
        """Write the Kronecker operator of the element `cell` into `out`."""

    def apply_many(
        self,
        v: Annotated[
            NDArray[numpy.float32],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
        cells: Annotated[
            NDArray[numpy.int64], dict(shape=(None, None), order="C", device="cpu", writable=False)
        ],
        out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")],
        scratch: Annotated[
            NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")
        ]
        | None,
    ) -> None:
        """Write M_c v[c] for every cell (rows of `cells`) into `out`."""

    def apply_transpose_many(
        self,
        v: Annotated[
            NDArray[numpy.float32],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
        cells: Annotated[
            NDArray[numpy.int64], dict(shape=(None, None), order="C", device="cpu", writable=False)
        ],
        out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")],
        scratch: Annotated[
            NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")
        ]
        | None,
    ) -> None:
        """Write M_c^T v[c] for every cell into `out`."""

    def apply_mt_k_m_many(
        self,
        k: Annotated[
            NDArray[numpy.float32],
            dict(shape=(None, None, None), order="C", device="cpu", writable=False),
        ],
        cells: Annotated[
            NDArray[numpy.int64], dict(shape=(None, None), order="C", device="cpu", writable=False)
        ],
        out: Annotated[
            NDArray[numpy.float32], dict(shape=(None, None, None), order="C", device="cpu")
        ],
        scratch: Annotated[
            NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")
        ]
        | None,
    ) -> None:
        """Write M_c^T K[c] M_c for every cell into `out`."""

    def apply_m_k_mt_many(
        self,
        k: Annotated[
            NDArray[numpy.float32],
            dict(shape=(None, None, None), order="C", device="cpu", writable=False),
        ],
        cells: Annotated[
            NDArray[numpy.int64], dict(shape=(None, None), order="C", device="cpu", writable=False)
        ],
        out: Annotated[
            NDArray[numpy.float32], dict(shape=(None, None, None), order="C", device="cpu")
        ],
        scratch: Annotated[
            NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")
        ]
        | None,
    ) -> None:
        """Write M_c K[c] M_c^T for every cell into `out`."""

    def apply(
        self,
        v: Annotated[
            NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        cell: Sequence[int],
        out: Annotated[NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu")],
        scratch: Annotated[NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu")]
        | None,
    ) -> None:
        """Write M v for the element `cell` (an interval per direction) into `out`."""

    def apply_transpose(
        self,
        v: Annotated[
            NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        cell: Sequence[int],
        out: Annotated[NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu")],
        scratch: Annotated[NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu")]
        | None,
    ) -> None:
        """Write M^T v for the element `cell` into `out`."""

    def apply_mt_k_m(
        self,
        k: Annotated[
            NDArray[numpy.float32],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
        cell: Sequence[int],
        out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")],
        scratch: Annotated[NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu")]
        | None,
    ) -> None:
        """Write M^T K M for the element `cell` into `out`."""

    def apply_m_k_mt(
        self,
        k: Annotated[
            NDArray[numpy.float32],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
        cell: Sequence[int],
        out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu")],
        scratch: Annotated[NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu")]
        | None,
    ) -> None:
        """Write M K M^T for the element `cell` into `out`."""
