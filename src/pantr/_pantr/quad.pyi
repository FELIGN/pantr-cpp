"""Quadrature rules and lattices (private)."""

from collections.abc import Sequence
from typing import Annotated

import numpy
from numpy.typing import NDArray

def trapezoidal_1d(n_pts: int) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    """Trapezoidal rule on [0, 1]: (nodes, weights)."""

def gauss_legendre_1d(n_pts: int) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    """Gauss-Legendre rule on [0, 1]: (nodes, weights)."""

def gauss_lobatto_legendre_1d(n_pts: int) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    """Gauss-Lobatto-Legendre rule on [0, 1]: (nodes, weights)."""

def chebyshev_gauss_1st_kind_1d(
    n_pts: int,
) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    """Chebyshev-Gauss rule of the first kind on [0, 1]: (nodes, weights)."""

def chebyshev_gauss_2nd_kind_1d(
    n_pts: int,
) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    """Chebyshev-Gauss rule of the second kind on [0, 1]: (nodes, weights)."""

def tanh_sinh_1d(n_pts: int) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    """Tanh-sinh rule on [0, 1], truncated where the endpoint gap stops being resolvable."""

def modified_chebyshev_nodes_1d(n_pts: int) -> NDArray[numpy.float64]:
    """Chebyshev-Lobatto nodes on [0, 1]."""

def lagrange_points(variant: int, n_pts: int) -> NDArray[numpy.float64]:
    """Nodes of a Lagrange basis on [0, 1]."""

def trapezoidal_1d_f32(n_pts: int) -> tuple[NDArray[numpy.float32], NDArray[numpy.float32]]:
    """Trapezoidal rule on [0, 1]: (nodes, weights)."""

def gauss_legendre_1d_f32(n_pts: int) -> tuple[NDArray[numpy.float32], NDArray[numpy.float32]]:
    """Gauss-Legendre rule on [0, 1]: (nodes, weights)."""

def gauss_lobatto_legendre_1d_f32(
    n_pts: int,
) -> tuple[NDArray[numpy.float32], NDArray[numpy.float32]]:
    """Gauss-Lobatto-Legendre rule on [0, 1]: (nodes, weights)."""

def chebyshev_gauss_1st_kind_1d_f32(
    n_pts: int,
) -> tuple[NDArray[numpy.float32], NDArray[numpy.float32]]:
    """Chebyshev-Gauss rule of the first kind on [0, 1]: (nodes, weights)."""

def chebyshev_gauss_2nd_kind_1d_f32(
    n_pts: int,
) -> tuple[NDArray[numpy.float32], NDArray[numpy.float32]]:
    """Chebyshev-Gauss rule of the second kind on [0, 1]: (nodes, weights)."""

def tanh_sinh_1d_f32(n_pts: int) -> tuple[NDArray[numpy.float32], NDArray[numpy.float32]]:
    """Tanh-sinh rule on [0, 1], truncated where the endpoint gap stops being resolvable."""

def modified_chebyshev_nodes_1d_f32(n_pts: int) -> NDArray[numpy.float32]:
    """Chebyshev-Lobatto nodes on [0, 1]."""

def lagrange_points_f32(variant: int, n_pts: int) -> NDArray[numpy.float32]:
    """Nodes of a Lagrange basis on [0, 1]."""

class PointsLatticeF64:
    """An immutable tensor-product lattice of points."""

    def __init__(
        self,
        pts_per_dir: Sequence[
            Annotated[
                NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
            ]
        ],
    ) -> None:
        """Copy the coordinate arrays into a new lattice."""

    @property
    def dim(self) -> int:
        """The number of directions."""

    @property
    def num_points(self) -> int:
        """The number of points."""

    def pts(self, d: int) -> Annotated[NDArray[numpy.float64], dict(writable=False)]:
        """Read-only view of direction `d`'s coordinates."""

    def all_points(self, order: int) -> NDArray[numpy.float64]:
        """All points, (num_points, dim), in `order` (0 C, 1 F)."""

def lagrange_points_lattice(variant: int, n_pts_per_dir: Sequence[int]) -> PointsLatticeF64:
    """Lattice of the Lagrange nodes per direction."""

class PointsLatticeF32:
    """An immutable tensor-product lattice of points."""

    def __init__(
        self,
        pts_per_dir: Sequence[
            Annotated[
                NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
            ]
        ],
    ) -> None:
        """Copy the coordinate arrays into a new lattice."""

    @property
    def dim(self) -> int:
        """The number of directions."""

    @property
    def num_points(self) -> int:
        """The number of points."""

    def pts(self, d: int) -> Annotated[NDArray[numpy.float32], dict(writable=False)]:
        """Read-only view of direction `d`'s coordinates."""

    def all_points(self, order: int) -> NDArray[numpy.float32]:
        """All points, (num_points, dim), in `order` (0 C, 1 F)."""

def lagrange_points_lattice_f32(variant: int, n_pts_per_dir: Sequence[int]) -> PointsLatticeF32:
    """Lattice of the Lagrange nodes per direction."""

class QuadratureRule:
    """An immutable quadrature rule on the unit cube."""

    def __init__(
        self,
        points: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
        weights: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
    ) -> None:
        """Validate and copy the points and weights."""

    @property
    def ndim(self) -> int:
        """The number of axes."""

    @property
    def num_points(self) -> int:
        """The number of points."""

    @property
    def points(self) -> Annotated[NDArray[numpy.float64], dict(writable=False)]:
        """Read-only (num_points, ndim) points."""

    @property
    def weights(self) -> Annotated[NDArray[numpy.float64], dict(writable=False)]:
        """Read-only (num_points,) weights."""

def tensor_product_quadrature(
    rules: Sequence[
        tuple[
            Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
            Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
        ]
    ],
) -> QuadratureRule:
    """Tensor product of (nodes, weights) pairs on [0, 1]."""

def gauss_legendre_quadrature(ndim: int, npts: Sequence[int]) -> QuadratureRule:
    """Tensor-product Gauss-Legendre rule on the unit cube."""
