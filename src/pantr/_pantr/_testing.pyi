"""Probes for tests and benchmarks (private)."""

from typing import Annotated

import numpy
from numpy.typing import NDArray

def count_threads_used(chunks: int, hold_us: int) -> int:
    """Count the distinct threads that run a parallel loop of `chunks` chunks holding `hold_us` microseconds each."""

def routes_shared_parallelism() -> bool:
    """Report whether this module routes its parallel state through another module."""

bernstein_max_unmirrored_degree_double: int = ...

bernstein_max_unmirrored_degree_float: int = ...

def bernstein_row_mirrored(n: int, u: float) -> NDArray[numpy.float64]:
    """Bernstein values at one point by the mirrored ratio recurrence."""

def bernstein_row_unmirrored(n: int, u: float) -> NDArray[numpy.float64]:
    """Bernstein values at one point by the plain forward recurrence (low degrees only)."""

def bernstein_derivatives_point(n: int, s: float, n_deriv: int) -> NDArray[numpy.float64]:
    """Bernstein derivatives of orders 0..n_deriv at one point, (n_deriv + 1, n + 1)."""

def tabulate_bernstein_derivatives_1d(
    n: int,
    pts: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    n_deriv: int,
) -> NDArray[numpy.float64]:
    """Bernstein derivatives of orders 0..n_deriv at each point, (len(pts), n_deriv + 1, n + 1)."""

def bernstein_row_mirrored_f32(n: int, u: float) -> NDArray[numpy.float32]:
    """Bernstein values at one point by the mirrored ratio recurrence."""

def bernstein_row_unmirrored_f32(n: int, u: float) -> NDArray[numpy.float32]:
    """Bernstein values at one point by the plain forward recurrence (low degrees only)."""

def bernstein_derivatives_point_f32(n: int, s: float, n_deriv: int) -> NDArray[numpy.float32]:
    """Bernstein derivatives of orders 0..n_deriv at one point, (n_deriv + 1, n + 1)."""

def tabulate_bernstein_derivatives_1d_f32(
    n: int,
    pts: Annotated[
        NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    n_deriv: int,
) -> NDArray[numpy.float32]:
    """Bernstein derivatives of orders 0..n_deriv at each point, (len(pts), n_deriv + 1, n + 1)."""

def gram_projection(
    new_family: int, old_family: int, degree: int, n_quad_pts: int
) -> NDArray[numpy.float64]:
    """The matrix M with old = M new, by an n_quad_pts-point Gauss-Legendre Gram projection."""

def gram_projection_f32(
    new_family: int, old_family: int, degree: int, n_quad_pts: int
) -> NDArray[numpy.float32]:
    """The float instantiation of gram_projection."""

change_basis_cache_max_degree: int = ...

def change_basis_cache_size() -> int:
    """Number of matrices the change-of-basis process tables hold, both dtypes."""

def change_basis_cache_clear() -> None:
    """Empty the change-of-basis process tables (matrices handed out stay valid)."""
