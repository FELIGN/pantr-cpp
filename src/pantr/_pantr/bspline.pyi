"""B-spline spaces and knot vectors (private)."""

from collections.abc import Sequence
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray

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
