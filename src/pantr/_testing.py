"""The one private surface the test suite may import (not a public API).

Internals worth testing directly, which no public function exposes, are bound in the compiled
module's ``_testing`` submodule and re-exported here; a test imports them from ``pantr._testing``
and from no other private path (design/port_design.md §3.8). Nothing in the library imports this
module.

- The Bernstein recurrences at one point, the degree up to which the unmirrored one runs, and
  the Bernstein derivative tabulation (no public function returns derivatives).
- The generic Gram projection behind the change-of-basis builders.
- The change-of-basis process tables: their degree bound, size and reset, and the private
  wrappers over them the extraction code uses (``_cached_*``).
- The change-of-basis degree domains (``_*_MAX_DEGREE``, ``_DegreeLimit``) and the Lagrange nodes
  (``_get_lagrange_points``), which the domain derivation rebuilds in exact arithmetic.
- The Builtin root provider's internals (de Casteljau, sign predicates, hull clip, Newton polish,
  Yuksel's finder, Bezier clipping and the candidate merge), with the call conventions of the
  Numba kernels they replaced (a root array and a count).
- The degree-reduction operators and their process table, the exact elevation matrix, and the
  pieces of the rational round-trip measure (Gram norm, Gauss grid, projected sampling).
- The root provider's raw state (``root_provider_state``, ``set_root_provider_state``), so a test
  can return it to "resolve on first use".
- :func:`bernstein_interpolate`: values on the modified-Chebyshev grid to Bernstein coefficients.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import numpy.typing as npt

from ._ext import ext
from .basis._basis_1D import _Family
from .basis._basis_lagrange import _get_lagrange_points
from .bezier._find_roots import _get_state as root_provider_state
from .bezier._find_roots import _set_state as set_root_provider_state
from .change_basis import (
    _BERNSTEIN_TO_CARDINAL_MAX_DEGREE,
    _BERNSTEIN_TO_LAGRANGE_MAX_DEGREE,
    _CARDINAL_TO_BERNSTEIN_MAX_DEGREE,
    _CARDINAL_TO_LEGENDRE_MAX_DEGREE,
    _cached_cardinal_to_bernstein_matrix,
    _cached_cardinal_to_legendre_matrix,
    _cached_lagrange_to_bernstein_matrix,
    _cached_legendre_to_cardinal_matrix,
    _DegreeLimit,
)

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

# -- root finding (the Builtin provider) --
have_same_sign = _t.have_same_sign
have_opposite_signs = _t.have_opposite_signs
spans_zero = _t.spans_zero
count_sign_changes = _t.count_sign_changes
clip_hull_to_zero = _t.clip_hull_to_zero
de_casteljau_eval_scalar = _t.de_casteljau_eval_scalar
de_casteljau_eval_and_deriv_scalar = _t.de_casteljau_eval_and_deriv_scalar
subdivide_scalar = _t.subdivide_scalar
newton_polish_scalar = _t.newton_polish_scalar
solve_monotone_root_kernel = _t.solve_monotone_root_kernel
yuksel_roots = _t.yuksel_roots
clip_roots_core = _t.clip_roots_core
dedup_roots = _t.dedup_roots
find_roots_detailed = ext.roots.find_roots_detailed
find_roots_detailed_f32 = ext.roots.find_roots_detailed_f32
root_provider_capabilities = ext.roots.capabilities

ROOTS_MAX_NEWTON_ITER: Final[int] = _t.ROOTS_MAX_NEWTON_ITER
"""Iteration cap of the Newton/bisection hybrid."""

ROOTS_CLIP_MAX_DEPTH: Final[int] = _t.ROOTS_CLIP_MAX_DEPTH
"""Depth at which Bezier clipping gives an interval up (and reports a gap)."""

# -- degree reduction --
interpolating_reduction_operator = ext._kernels.interpolating_reduction_operator
l2_reduction_operator = _t.l2_reduction_operator
elevation_matrix = _t.elevation_matrix
reduction_cache_size = _t.reduction_cache_size
reduction_cache_clear = _t.reduction_cache_clear
projected_quadrature_size = _t.projected_quadrature_size
tensor_gauss_weights = _t.tensor_gauss_weights
projected_relative_deviation = _t.projected_relative_deviation


def squared_l2_norm(coeffs: npt.NDArray[np.floating[Any]]) -> float:
    """Get the squared ``L2([0, 1]^d)`` norm of a tensor Bernstein polynomial (in float64).

    Args:
        coeffs (npt.NDArray[np.floating[Any]]): Bernstein coefficients, any shape.

    Returns:
        float: ``c^T (G_0 x ... x G_{d-1}) c``.
    """
    return float(_t.squared_l2_norm(np.ascontiguousarray(coeffs, dtype=np.float64)))


def sample_projected(
    ctrl: npt.NDArray[np.floating[Any]], num_nodes: tuple[int, ...]
) -> npt.NDArray[np.float64] | None:
    """Sample a homogeneous net on the Gauss grid and divide out the weights (in float64).

    Args:
        ctrl (npt.NDArray[np.floating[Any]]): ``(*orders, rank + 1)``, the last column weights.
        num_nodes (tuple[int, ...]): Gauss-Legendre nodes per direction.

    Returns:
        npt.NDArray[np.float64] | None: ``(*num_nodes, rank)``, or *None* when a sampled weight
        is not strictly positive.
    """
    result: npt.NDArray[np.float64] | None = _t.sample_projected(
        np.ascontiguousarray(ctrl, dtype=np.float64), list(num_nodes)
    )
    return result


REDUCTION_CACHE_MAX_DEGREE: Final[int] = _t.reduction_cache_max_degree
"""Largest degree whose reduction operators the process table keeps."""


def bernstein_interpolate(
    f: npt.NDArray[np.floating[Any]], tol: float | None = None
) -> npt.NDArray[np.floating[Any]]:
    """Interpolate values on the modified-Chebyshev tensor grid to Bernstein coefficients.

    The values ``f`` of shape ``(n_0, ..., n_{N-1})`` sit at the tensor product of the
    ``n_k`` modified Chebyshev-Lobatto nodes per direction (one node, ``0.5``, for a size-1
    axis, which passes through); the result has the same shape and dtype. A thin route
    through :func:`pantr.bezier.fit_bezier`, kept for the tests and the adversarial sweep.

    Args:
        f (npt.NDArray[np.floating[Any]]): The values.
        tol (float | None): SVD truncation tolerance; ``100 * eps`` when *None*.

    Returns:
        npt.NDArray[np.floating[Any]]: The coefficients.
    """
    from .bezier import fit_bezier  # noqa: PLC0415
    from .quad import get_modified_chebyshev_nodes_1d  # noqa: PLC0415

    arr = np.asarray(f)
    nodes = [
        get_modified_chebyshev_nodes_1d(max(n, 2), arr.dtype)[:n]
        if n >= 2  # noqa: PLR2004
        else np.array([0.5], dtype=arr.dtype)
        for n in arr.shape
    ]
    cp = fit_bezier(arr, nodes, tol=tol).control_points
    return np.asarray(cp[..., 0], dtype=arr.dtype)


BERNSTEIN: Final = _Family.BERNSTEIN
"""The Bernstein family, for :data:`gram_projection`."""
CARDINAL_BSPLINE: Final = _Family.CARDINAL_BSPLINE
"""The cardinal B-spline family, for :data:`gram_projection`."""
LEGENDRE: Final = _Family.LEGENDRE
"""The shifted Legendre family, for :data:`gram_projection`."""

__all__ = [
    "BERNSTEIN",
    "BERNSTEIN_MAX_UNMIRRORED_DEGREE_FLOAT32",
    "BERNSTEIN_MAX_UNMIRRORED_DEGREE_FLOAT64",
    "CARDINAL_BSPLINE",
    "CHANGE_BASIS_CACHE_MAX_DEGREE",
    "LEGENDRE",
    "REDUCTION_CACHE_MAX_DEGREE",
    "ROOTS_CLIP_MAX_DEPTH",
    "ROOTS_MAX_NEWTON_ITER",
    "_BERNSTEIN_TO_CARDINAL_MAX_DEGREE",
    "_BERNSTEIN_TO_LAGRANGE_MAX_DEGREE",
    "_CARDINAL_TO_BERNSTEIN_MAX_DEGREE",
    "_CARDINAL_TO_LEGENDRE_MAX_DEGREE",
    "_DegreeLimit",
    "_cached_cardinal_to_bernstein_matrix",
    "_cached_cardinal_to_legendre_matrix",
    "_cached_lagrange_to_bernstein_matrix",
    "_cached_legendre_to_cardinal_matrix",
    "_get_lagrange_points",
    "bernstein_derivatives_point",
    "bernstein_derivatives_point_f32",
    "bernstein_interpolate",
    "bernstein_row_mirrored",
    "bernstein_row_mirrored_f32",
    "bernstein_row_unmirrored",
    "bernstein_row_unmirrored_f32",
    "change_basis_cache_clear",
    "change_basis_cache_size",
    "clip_hull_to_zero",
    "clip_roots_core",
    "count_sign_changes",
    "de_casteljau_eval_and_deriv_scalar",
    "de_casteljau_eval_scalar",
    "dedup_roots",
    "elevation_matrix",
    "find_roots_detailed",
    "find_roots_detailed_f32",
    "gram_projection",
    "gram_projection_f32",
    "have_opposite_signs",
    "have_same_sign",
    "interpolating_reduction_operator",
    "l2_reduction_operator",
    "newton_polish_scalar",
    "projected_quadrature_size",
    "projected_relative_deviation",
    "reduction_cache_clear",
    "reduction_cache_size",
    "root_provider_capabilities",
    "root_provider_state",
    "sample_projected",
    "set_root_provider_state",
    "solve_monotone_root_kernel",
    "spans_zero",
    "squared_l2_norm",
    "subdivide_scalar",
    "tabulate_bernstein_derivatives_1d",
    "tabulate_bernstein_derivatives_1d_f32",
    "tensor_gauss_weights",
    "yuksel_roots",
]
