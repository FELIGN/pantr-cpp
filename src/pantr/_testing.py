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
- The B-spline knot vector's scale and absolute tolerance (``knot_scale``, ``knot_tolerance``)
  and the snapping diagnosis (``check_snapping_kept_an_interval``), which a space applies at
  construction and no public function exposes on their own.
- The blossom (``evaluate_blossom_1d``, ``blossom_span``) of a 1D B-spline, which has no public
  API, and the Oslo refinement matrix in its banded (``oslo_rows``) and dense (``oslo_matrix``)
  forms, whose bands no public function returns.
- The exact-integer binomial kernel (``bincoeff``, ``check_bincoeff_envelope``,
  ``BINCOEFF_MAX_N``) and the 1D degree kernels (``degree_elevate_1d``, ``degree_reduce_1d``)
  behind ``Bspline.elevate_degree`` and ``reduce_degree``.
- The knot-removal helpers behind ``Bspline.remove_knots``: the control-point scale, the round-off
  floor, the stacked-point count, the homogeneous tolerance and the knot-class lookup.
- The Bezier, Lagrange and cardinal extraction operators and the cardinal intervals of a raw knot
  vector with an explicit tolerance (``bezier_extraction_1d`` and its siblings), which a space
  only reaches with its own tolerance.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import numpy.typing as npt

from ._ext import ext
from .basis import LagrangeVariant
from .basis._basis_1D import _Family
from .basis._basis_lagrange import _get_lagrange_points
from .basis._basis_tabulate import _variant_code
from .basis._basis_utils import _allocate_or_validate_out, _validate_out_array
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
evaluate_blossom_1d = _t.evaluate_blossom_1d
evaluate_blossom_1d_f32 = _t.evaluate_blossom_1d_f32
blossom_span = _t.blossom_span
blossom_span_f32 = _t.blossom_span_f32
oslo_rows = _t.oslo_rows
oslo_rows_f32 = _t.oslo_rows_f32
oslo_matrix = ext._kernels.oslo_matrix
oslo_matrix_f32 = ext._kernels.oslo_matrix_f32
knot_scale = _t.knot_scale
knot_tolerance = _t.knot_tolerance
check_snapping_kept_an_interval = _t.check_snapping_kept_an_interval
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
        get_modified_chebyshev_nodes_1d(n, arr.dtype)
        if n >= 2  # noqa: PLR2004
        else np.array([0.5], dtype=arr.dtype)
        for n in arr.shape
    ]
    cp = fit_bezier(arr, nodes, tol=tol).control_points
    return np.asarray(cp[..., 0], dtype=arr.dtype)


BINCOEFF_MAX_N: Final[int] = _t.bincoeff_max_n
"""Largest upper index for which :data:`bincoeff` honours its contract."""

bincoeff = _t.bincoeff
check_bincoeff_envelope = _t.check_bincoeff_envelope
num_stacked_control_points = _t.num_stacked_control_points


def _by_dtype(name: str, dtype: npt.DTypeLike) -> Any:  # noqa: ANN401
    """Get a compiled helper of the dtype's instantiation.

    Args:
        name (str): The helper's name without the dtype suffix.
        dtype (npt.DTypeLike): ``float32`` or ``float64``.

    Returns:
        Any: ``<name>_f32`` for ``float32``, else ``<name>``.
    """
    return getattr(_t, name + ("_f32" if np.dtype(dtype) == np.float32 else ""))


def control_point_scale(ctrl: npt.NDArray[np.floating[Any]]) -> float:
    """Get the magnitude a knot-removal distance is relative to.

    Args:
        ctrl (npt.NDArray[np.floating[Any]]): Control points of any shape, the last axis the
            coordinates.

    Returns:
        float: ``max(bounding-box diagonal, largest ||P_i||)``, in float64.
    """
    flat = np.ascontiguousarray(ctrl).reshape(-1, ctrl.shape[-1])
    return float(_by_dtype("control_point_scale", flat.dtype)(flat))


def roundoff_deviation_floor(
    ctrl: npt.NDArray[np.floating[Any]], degree: int, num_stacked: int
) -> float:
    """Get the deviation budget that admits exactly the removals round-off allows.

    Args:
        ctrl (npt.NDArray[np.floating[Any]]): Control points as the removal sees them, the last
            axis the coordinates.
        degree (int): The degree of the direction being reduced.
        num_stacked (int): The control points the removal's one distance spans.

    Returns:
        float: ``8 (degree + 1) eps sqrt(num_stacked)`` times :func:`control_point_scale`.
    """
    flat = np.ascontiguousarray(ctrl).reshape(-1, ctrl.shape[-1])
    return float(_by_dtype("roundoff_deviation_floor", flat.dtype)(flat, degree, num_stacked))


def homogeneous_deviation_tolerance(
    ctrl: npt.NDArray[np.floating[Any]], tol_euclidean: float
) -> float:
    """Pull a projected-space deviation budget back to homogeneous control points.

    Args:
        ctrl (npt.NDArray[np.floating[Any]]): Homogeneous control points, the last column the
            weights.
        tol_euclidean (float): The budget in projected space.

    Returns:
        float: ``d w_min / (1 + |P|_max)``.
    """
    flat = np.ascontiguousarray(ctrl).reshape(-1, ctrl.shape[-1])
    return float(_by_dtype("homogeneous_deviation_tolerance", flat.dtype)(flat, tol_euclidean))


def find_knot_index_and_multiplicity(
    knots: npt.NDArray[np.floating[Any]], degree: int, knot_value: float, tol: float
) -> tuple[int, int]:
    """Find the last index and the multiplicity of the knot class a value falls in.

    Args:
        knots (npt.NDArray[np.floating[Any]]): The knot vector.
        degree (int): The degree.
        knot_value (float): The value to locate.
        tol (float): The absolute parametric tolerance.

    Returns:
        tuple[int, int]: ``(r, s)``.
    """
    k = np.ascontiguousarray(knots)
    r, s = _by_dtype("find_knot_index_and_multiplicity", k.dtype)(k, degree, knot_value, tol)
    return int(r), int(s)


def degree_elevate_1d(
    degree: int,
    ctrl: npt.NDArray[np.floating[Any]],
    knots: npt.NDArray[np.floating[Any]],
    increment: int,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """Degree elevate a 1D B-spline given by its arrays (the knot vector must be clamped).

    Args:
        degree (int): The degree.
        ctrl (npt.NDArray[np.floating[Any]]): Control points ``(n, rank)``.
        knots (npt.NDArray[np.floating[Any]]): The clamped knot vector.
        increment (int): The degrees to add.

    Returns:
        tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]: The elevated control
        points and the elevated knot vector.
    """
    c = np.ascontiguousarray(ctrl)
    out: tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]] = _by_dtype(
        "degree_elevate_1d", c.dtype
    )(degree, c, np.ascontiguousarray(knots), increment)
    return (out[1], out[0])


def degree_reduce_1d(
    degree: int,
    ctrl: npt.NDArray[np.floating[Any]],
    knots: npt.NDArray[np.floating[Any]],
    decrement: int,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """Degree reduce a 1D B-spline given by its arrays to Bezier form (clamped knot vector).

    Args:
        degree (int): The degree.
        ctrl (npt.NDArray[np.floating[Any]]): Control points ``(n, rank)``.
        knots (npt.NDArray[np.floating[Any]]): The clamped knot vector.
        decrement (int): The degrees to remove.

    Returns:
        tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]: The reduced control
        points and the knot vector, in Bezier form.
    """
    c = np.ascontiguousarray(ctrl)
    out: tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]] = _by_dtype(
        "degree_reduce_1d", c.dtype
    )(degree, c, np.ascontiguousarray(knots), decrement)
    return (out[1], out[0])


def _extraction_intervals(
    knots: npt.NDArray[np.float32 | np.float64], degree: int, tol: float
) -> tuple[npt.NDArray[np.float32 | np.float64], int]:
    """Validate the arguments of a raw extraction call and count its intervals.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): The knot vector.
        degree (int): The degree.
        tol (float): The absolute parametric tolerance.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], int]: The knots as the compiled call takes
        them, and the number of in-domain intervals.

    Raises:
        TypeError: If ``knots`` is not 1D.
        ValueError: If ``tol`` is negative or the knots and the degree define no space.
    """
    k = np.ascontiguousarray(knots)
    if k.ndim != 1:
        raise TypeError("knots must be a 1D array")
    return k, int(_by_dtype("num_intervals_1d", k.dtype)(k, degree, tol))


def bezier_extraction_1d(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Get the Bezier extraction operators of a raw knot vector with an explicit tolerance.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): The knot vector.
        degree (int): The degree.
        tol (float): The absolute parametric tolerance, non-negative.
        out (npt.NDArray[np.float32 | np.float64] | None): A C-contiguous array to fill, or None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: ``(intervals, degree + 1, degree + 1)``; ``out``
        when given.

    Raises:
        TypeError: If ``knots`` is not 1D.
        ValueError: If ``tol`` is negative, the knots and the degree define no space, or ``out``
            has another shape or dtype or is not writeable.
    """
    k, n = _extraction_intervals(knots, degree, tol)
    ops = _allocate_or_validate_out(out, (n, degree + 1, degree + 1), k.dtype)
    _by_dtype("bezier_extraction_1d_into", k.dtype)(k, degree, tol, ops)
    return ops


def lagrange_extraction_1d(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    lagrange_variant: LagrangeVariant = LagrangeVariant.EQUISPACES,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Get the Lagrange extraction operators of a raw knot vector with an explicit tolerance.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): The knot vector.
        degree (int): The degree.
        tol (float): The absolute parametric tolerance, non-negative.
        lagrange_variant (LagrangeVariant): The node distribution.
        out (npt.NDArray[np.float32 | np.float64] | None): A C-contiguous array to fill, or None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: ``(intervals, degree + 1, degree + 1)``; ``out``
        when given.

    Raises:
        TypeError: If ``knots`` is not 1D.
        ValueError: As :func:`bezier_extraction_1d`, or if the degree is beyond the Lagrange
            matrix's.
    """
    k, n = _extraction_intervals(knots, degree, tol)
    ops = _allocate_or_validate_out(out, (n, degree + 1, degree + 1), k.dtype)
    _by_dtype("lagrange_extraction_1d_into", k.dtype)(
        k, degree, tol, _variant_code(lagrange_variant), ops
    )
    return ops


def cardinal_extraction_1d(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Get the cardinal extraction operators of a raw knot vector with an explicit tolerance.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): The knot vector.
        degree (int): The degree.
        tol (float): The absolute parametric tolerance, non-negative.
        out (npt.NDArray[np.float32 | np.float64] | None): A C-contiguous array to fill, or None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: ``(intervals, degree + 1, degree + 1)``; ``out``
        when given.

    Raises:
        TypeError: If ``knots`` is not 1D.
        ValueError: As :func:`lagrange_extraction_1d`.
    """
    k, n = _extraction_intervals(knots, degree, tol)
    ops = _allocate_or_validate_out(out, (n, degree + 1, degree + 1), k.dtype)
    _by_dtype("cardinal_extraction_1d_into", k.dtype)(k, degree, tol, ops)
    return ops


def cardinal_intervals_1d(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    out: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.bool_]:
    """Flag the cardinal intervals of a raw knot vector with an explicit tolerance.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): The knot vector.
        degree (int): The degree.
        tol (float): The absolute parametric tolerance, non-negative.
        out (npt.NDArray[np.bool_] | None): A C-contiguous boolean array to fill, or None.

    Returns:
        npt.NDArray[np.bool_]: One flag per in-domain interval; ``out`` when given.

    Raises:
        TypeError: If ``knots`` is not 1D.
        ValueError: If ``tol`` is negative, the knots and the degree define no space, or ``out``
            has another shape or dtype or is not writeable.
    """
    k, n = _extraction_intervals(knots, degree, tol)
    flags = np.empty(n, dtype=np.bool_) if out is None else out
    if out is not None:
        _validate_out_array(out, (n,), np.bool_)
    _by_dtype("cardinal_intervals_1d_into", k.dtype)(k, degree, tol, flags)
    return flags


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
    "BINCOEFF_MAX_N",
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
    "bezier_extraction_1d",
    "bincoeff",
    "blossom_span",
    "blossom_span_f32",
    "cardinal_extraction_1d",
    "cardinal_intervals_1d",
    "change_basis_cache_clear",
    "change_basis_cache_size",
    "check_bincoeff_envelope",
    "check_snapping_kept_an_interval",
    "clip_hull_to_zero",
    "clip_roots_core",
    "control_point_scale",
    "count_sign_changes",
    "de_casteljau_eval_and_deriv_scalar",
    "de_casteljau_eval_scalar",
    "dedup_roots",
    "degree_elevate_1d",
    "degree_reduce_1d",
    "elevation_matrix",
    "evaluate_blossom_1d",
    "evaluate_blossom_1d_f32",
    "find_knot_index_and_multiplicity",
    "find_roots_detailed",
    "find_roots_detailed_f32",
    "gram_projection",
    "gram_projection_f32",
    "have_opposite_signs",
    "have_same_sign",
    "homogeneous_deviation_tolerance",
    "interpolating_reduction_operator",
    "knot_scale",
    "knot_tolerance",
    "l2_reduction_operator",
    "lagrange_extraction_1d",
    "newton_polish_scalar",
    "num_stacked_control_points",
    "oslo_matrix",
    "oslo_matrix_f32",
    "oslo_rows",
    "oslo_rows_f32",
    "projected_quadrature_size",
    "projected_relative_deviation",
    "reduction_cache_clear",
    "reduction_cache_size",
    "root_provider_capabilities",
    "root_provider_state",
    "roundoff_deviation_floor",
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
