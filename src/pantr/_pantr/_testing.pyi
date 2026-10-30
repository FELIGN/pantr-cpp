"""Probes for tests and benchmarks (private)."""

from collections.abc import Sequence
from typing import Annotated, overload

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

def have_same_sign(a: float, b: float) -> bool: ...
def have_opposite_signs(a: float, b: float) -> bool: ...
def spans_zero(a: float, b: float) -> bool: ...
def count_sign_changes(
    coeff: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
) -> int: ...
def clip_hull_to_zero(
    coeff: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
) -> tuple[float, float, bool]: ...

ROOTS_MAX_NEWTON_ITER: int = ...

ROOTS_CLIP_MAX_DEPTH: int = ...

@overload
def de_casteljau_eval_scalar(
    coeff: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    t: float,
) -> float: ...
@overload
def de_casteljau_eval_scalar(
    coeff: Annotated[
        NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    t: float,
) -> float: ...
@overload
def de_casteljau_eval_and_deriv_scalar(
    coeff: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    t: float,
) -> tuple[float, float]: ...
@overload
def de_casteljau_eval_and_deriv_scalar(
    coeff: Annotated[
        NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    t: float,
) -> tuple[float, float]: ...
@overload
def subdivide_scalar(
    coeff: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    t_min: float,
    t_max: float,
) -> NDArray[numpy.float64]: ...
@overload
def subdivide_scalar(
    coeff: Annotated[
        NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    t_min: float,
    t_max: float,
) -> NDArray[numpy.float64]: ...
@overload
def newton_polish_scalar(
    coeff: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    mid: float,
    lo: float,
    hi: float,
    param_tol: float,
) -> tuple[float, float, float]: ...
@overload
def newton_polish_scalar(
    coeff: Annotated[
        NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    mid: float,
    lo: float,
    hi: float,
    param_tol: float,
) -> tuple[float, float, float]: ...
@overload
def solve_monotone_root_kernel(
    coeff: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    param_tol: float,
    max_iter: int = ...,
) -> float: ...
@overload
def solve_monotone_root_kernel(
    coeff: Annotated[
        NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    param_tol: float,
    max_iter: int = ...,
) -> float: ...
@overload
def yuksel_roots(
    coeff: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    tol: float,
) -> tuple[NDArray[numpy.float64], int]: ...
@overload
def yuksel_roots(
    coeff: Annotated[
        NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    tol: float,
) -> tuple[NDArray[numpy.float64], int]: ...
@overload
def clip_roots_core(
    root_coeff: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    param_tol: float,
    geom_tol: float,
    max_depth: int = ...,
) -> tuple[NDArray[numpy.float64], int]: ...
@overload
def clip_roots_core(
    root_coeff: Annotated[
        NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    param_tol: float,
    geom_tol: float,
    max_depth: int = ...,
) -> tuple[NDArray[numpy.float64], int]: ...
@overload
def dedup_roots(
    raw_roots: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    n_roots: int,
    coeff: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    param_tol: float,
    geom_tol: float,
) -> NDArray[numpy.float64]: ...
@overload
def dedup_roots(
    raw_roots: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    n_roots: int,
    coeff: Annotated[
        NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    param_tol: float,
    geom_tol: float,
) -> NDArray[numpy.float64]: ...
def l2_reduction_operator(
    degree: int, decrement: int
) -> Annotated[NDArray[numpy.float64], dict(writable=False)]: ...
def elevation_matrix(degree: int, increment: int) -> NDArray[numpy.float64]: ...

reduction_cache_max_degree: int = ...

def reduction_cache_size() -> int: ...
def reduction_cache_clear() -> None: ...
def squared_l2_norm(
    coeffs: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
) -> float: ...
def projected_quadrature_size(degree: int) -> int: ...
def tensor_gauss_weights(num_nodes: Sequence[int]) -> NDArray[numpy.float64]: ...
def sample_projected(
    ctrl: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
    num_nodes: Sequence[int],
) -> NDArray[numpy.float64] | None: ...
def projected_relative_deviation(
    values: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)] | None,
    trial_values: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)]
    | None,
    quad_weights: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
) -> float: ...
@overload
def knot_scale(
    knots: Annotated[NDArray[numpy.float64], dict(shape=(None,), device="cpu", writable=False)],
) -> float: ...
@overload
def knot_scale(
    knots: Annotated[NDArray[numpy.float32], dict(shape=(None,), device="cpu", writable=False)],
) -> float:
    """max(span, |knots[0]|, |knots[-1]|) of a non-empty knot vector."""

@overload
def knot_tolerance(
    knots: Annotated[NDArray[numpy.float64], dict(shape=(None,), device="cpu", writable=False)],
) -> float: ...
@overload
def knot_tolerance(
    knots: Annotated[NDArray[numpy.float32], dict(shape=(None,), device="cpu", writable=False)],
) -> float:
    """The absolute parametric tolerance of a non-empty knot vector."""

@overload
def check_snapping_kept_an_interval(
    raw: Annotated[NDArray[numpy.float64], dict(shape=(None,), device="cpu", writable=False)],
    snapped: Annotated[NDArray[numpy.float64], dict(shape=(None,), device="cpu", writable=False)],
    tol: float,
) -> None: ...
@overload
def check_snapping_kept_an_interval(
    raw: Annotated[NDArray[numpy.float32], dict(shape=(None,), device="cpu", writable=False)],
    snapped: Annotated[NDArray[numpy.float32], dict(shape=(None,), device="cpu", writable=False)],
    tol: float,
) -> None:
    """Refuse a knot vector that snapping collapsed onto one point."""
