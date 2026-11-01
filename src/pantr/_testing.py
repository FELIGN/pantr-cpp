"""The one private surface the test suite may import (not a public API).

Internals worth testing directly, which no public function exposes, are bound in the compiled
module's ``_testing`` submodule and re-exported here; a test imports them from ``pantr._testing``
and from no other private path (design/port_design.md §3.8). Nothing in the library imports this
module.

- The internals of ``Bspline.locate`` (``locate_*``): the inversion context's lengths, the default
  thresholds, the parametric reach and scale, the per-cell boxes, and one Newton solve or second
  start at a time, which no public call isolates.
- :func:`empty_views`: the binding's read-only view of empty, null-data storage.
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
- The Kronecker application on arbitrary per-direction factors (``kron_apply``), which the element
  operators of ``SpanwiseElementExtraction`` are a special case of, its per-cell scratch size
  (``required_scratch_size``) and the structural identity masks of the Bezier and Lagrange
  extraction operators of a 1D space (``bezier_identity_mask``, ``lagrange_identity_mask``).
- The Mørken-Reimers root finder of a scalar B-spline: its kernel on an open knot vector
  (``bspline_morken_reimers_roots``, a root array and the counts), the merge of repeated reports
  (``bspline_merge_roots``), the insertion budget per zero (``bspline_max_insertions``), and
  :func:`bspline_find_roots_with_budget`, the public route with a budget small enough to exhaust.
- The knot classes of a raw knot vector at an explicit tolerance
  (:func:`unique_knots_and_multiplicity`), which a space only reaches with its own tolerance.
- The pure block helpers of the hierarchical grid (``_block_size``, ``_in_block``,
  ``_rect_intersect``, ``_peel``, ``_try_merge``, ``_normalize_blocks``, ``_mark_region_cells``,
  ``_name_marked_cells`` and the two message limits), and the Cartesian factorization behind the
  ``"block"`` partition backend (``_factor_blocks``, ``_divisors``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final, NamedTuple

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
from .bspline._bspline import Bspline
from .bspline._bspline_space_1d import BsplineSpace1D
from .bspline._bspline_space_nd import BsplineSpace
from .bspline.spanwise_element_extraction import (
    _OP_CODES,
    OpKind,
    _operation_shapes,
    _validate_op_kind,
)
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

empty_views = _t.empty_views
"""Two empty read-only views over null data (float64, int64), as the binding helper builds them."""

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
bspline_max_insertions = _t.bspline_max_insertions
bspline_morken_reimers_roots = _t.bspline_morken_reimers_roots
bspline_merge_roots = _t.bspline_merge_roots
check_snapping_kept_an_interval = _t.check_snapping_kept_an_interval
reduction_cache_size = _t.reduction_cache_size
reduction_cache_clear = _t.reduction_cache_clear
projected_quadrature_size = _t.projected_quadrature_size
tensor_gauss_weights = _t.tensor_gauss_weights
projected_relative_deviation = _t.projected_relative_deviation


def bspline_find_roots_with_budget(
    bspline: Any,  # noqa: ANN401
    max_insertions: int,
    tol: float | None = None,
) -> npt.NDArray[np.float64]:
    """Find the roots of a scalar B-spline as :func:`pantr.bspline.find_roots` does, on a budget.

    The insertion budget per zero is otherwise fixed by the degree; a small one is how a test
    reaches the two exhaustion warnings.

    Args:
        bspline (Any): A :class:`~pantr.bspline.Bspline`.
        max_insertions (int): Insertion budget per zero.
        tol (float | None): Relative parametric tolerance, or ``None`` for the dtype default.

    Returns:
        npt.NDArray[np.float64]: Sorted, read-only array of roots.
    """
    from pantr.bspline._bspline_roots import _find_roots_impl  # noqa: PLC0415

    return _find_roots_impl(bspline, tol=tol, max_insertions=max_insertions)


def unique_knots_and_multiplicity(
    knots: npt.NDArray[np.floating[Any]],
    degree: int,
    tol: float,
    in_domain: bool,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.int64]]:
    """Get the knot classes of a raw knot vector at an explicit absolute tolerance.

    Args:
        knots (npt.NDArray[np.floating[Any]]): Non-decreasing ``float32`` or ``float64`` knots.
        degree (int): The degree, which fixes the domain.
        tol (float): Absolute parametric tolerance grouping nearby knots.
        in_domain (bool): Report only the classes from the first domain knot's to the last's.

    Returns:
        tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.int64]]: The class values (of the
        knots' dtype) and their multiplicities.
    """
    k = np.ascontiguousarray(knots)
    fn: Any = _by_dtype_kernel("unique_knots_and_multiplicity_1d", k.dtype)
    values, mults = fn(k, int(degree), float(tol), bool(in_domain))
    return values, mults


def _by_dtype_kernel(name: str, dtype: npt.DTypeLike) -> Any:  # noqa: ANN401
    """Get an interim ``_kernels`` function by dtype (``_f32`` for ``float32``).

    Args:
        name (str): The ``float64`` function's name.
        dtype (npt.DTypeLike): ``float32`` or ``float64``.

    Returns:
        Any: The function.
    """
    return getattr(ext._kernels, name + ("_f32" if np.dtype(dtype) == np.float32 else ""))


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


def bezier_identity_mask(space_1d: BsplineSpace1D) -> npt.NDArray[np.bool_]:
    """Flag the intervals of a 1D space whose Bezier extraction operator is the identity.

    Args:
        space_1d (BsplineSpace1D): The space.

    Returns:
        npt.NDArray[np.bool_]: One flag per interval: both bounding knots have multiplicity
        ``degree + 1`` or more.
    """
    out = np.empty(space_1d.num_intervals, dtype=np.bool_)
    space_1d._h.bezier_identity_mask(out)
    return out


def lagrange_identity_mask(
    space_1d: BsplineSpace1D, variant: LagrangeVariant | str
) -> npt.NDArray[np.bool_]:
    """Flag the intervals of a 1D space whose Lagrange extraction operator is the identity.

    A degree-0 space flags every interval. Otherwise the intervals are those of
    :func:`bezier_identity_mask` when the Lagrange-to-Bernstein matrix of ``variant`` is the
    identity, and none when it is not.

    Args:
        space_1d (BsplineSpace1D): The space.
        variant (LagrangeVariant | str): The Lagrange node distribution.

    Returns:
        npt.NDArray[np.bool_]: One flag per interval.

    Raises:
        ValueError: If ``variant`` is not a Lagrange variant, or the degree is beyond the degrees
            the Lagrange-to-Bernstein matrix can be built for.
    """
    out = np.empty(space_1d.num_intervals, dtype=np.bool_)
    _by_dtype("lagrange_identity_mask", space_1d.knots.dtype)(
        space_1d._h, _variant_code(variant), out
    )
    return out


def required_scratch_size(
    input_shape_per_dir: tuple[int, ...], output_shape_per_dir: tuple[int, ...], op_kind: OpKind
) -> int:
    """Get the scratch entries an element operation needs for one cell.

    Args:
        input_shape_per_dir (tuple[int, ...]): The columns of each direction's operator.
        output_shape_per_dir (tuple[int, ...]): The rows of each direction's operator.
        op_kind (OpKind): ``"apply"``, ``"apply_T"``, ``"MT_K_M"`` or ``"M_K_MT"``.

    Returns:
        int: Twice the larger intermediate result of the contraction, 0 when one stage does it
        all; enough for any identity pattern.

    Raises:
        ValueError: If ``op_kind`` is unknown, or the two tuples differ in length or are empty.
    """
    kind = _validate_op_kind(op_kind)
    return int(
        _t.required_scratch_size(
            _OP_CODES[kind], list(output_shape_per_dir), list(input_shape_per_dir)
        )
    )


def kron_apply(  # noqa: PLR0913 -- the operation, its factors, flags, operand and two buffers.
    factors: Sequence[npt.NDArray[np.float32 | np.float64]],
    identity: Sequence[bool],
    operand: npt.NDArray[np.float32 | np.float64],
    op_kind: OpKind,
    *,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
    scratch: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Apply ``kron(F_0, ..., F_{d-1})`` to an operand, as one cell of an extraction does.

    The element operations of :class:`~pantr.bspline.SpanwiseElementExtraction` on factors that
    need not come from a space: ``"apply"`` is ``M v``, ``"apply_T"`` is ``M^T v``, ``"MT_K_M"``
    is ``M^T K M`` and ``"M_K_MT"`` is ``M K M^T``. A factor flagged identity must be square and
    is not read.

    Args:
        factors (Sequence[npt.NDArray[np.float32 | np.float64]]): The 2D factors, one per
            direction, of the operand's dtype.
        identity (Sequence[bool]): Per direction, whether the factor is the identity.
        operand (npt.NDArray[np.float32 | np.float64]): The vector or the square matrix.
        op_kind (OpKind): The operation.
        out (npt.NDArray[np.float32 | np.float64] | None): The result array to fill (C-contiguous;
            it may be ``operand`` when every direction is identity), or ``None`` to allocate.
        scratch (npt.NDArray[np.float32 | np.float64] | None): At least
            :func:`required_scratch_size` entries, or ``None`` to allocate.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The result.

    Raises:
        ValueError: If the factors or the flags are inconsistent, an identity factor is not
            square, the operand or ``out`` has another size than the operation needs, ``out``
            overlaps the operand (unless every direction is identity), or the scratch is too small
            or overlaps either.
    """
    kind = _validate_op_kind(op_kind)
    dtype = np.dtype(operand.dtype)
    rows = tuple(int(f.shape[0]) for f in factors)
    cols = tuple(int(f.shape[1]) for f in factors)
    _, out_shape = _operation_shapes(cols, rows, kind)
    result = _allocate_or_validate_out(out, out_shape, dtype)
    _by_dtype("kron_apply", dtype)(
        _OP_CODES[kind],
        [np.ascontiguousarray(f) for f in factors],
        list(identity),
        np.ascontiguousarray(operand),
        result,
        scratch,
    )
    return result


# -- grids --
_MAX_NAMED_CELLS: Final[int] = _t.max_named_cells
"""Cells a rejected-coarsening message spells out before it counts the rest."""

_MAX_DIAGNOSED_CELLS: Final[int] = _t.max_diagnosed_cells
"""Region size above which a coarsening diagnosis is not computed."""

_Block = tuple[tuple[int, ...], tuple[int, ...]]


def _block(raw: tuple[Sequence[int], Sequence[int]]) -> _Block:
    """Turn a compiled block (two lists) into a block of tuples.

    Args:
        raw (tuple[Sequence[int], Sequence[int]]): ``(lo, hi)`` as the compiled helper returns it.

    Returns:
        _Block: ``(lo, hi)`` as tuples of ints.
    """
    return tuple(raw[0]), tuple(raw[1])


def _block_size(lo: Sequence[int], hi: Sequence[int]) -> int:
    """Get the number of cells in the block ``[lo, hi)``.

    Args:
        lo (Sequence[int]): Lower bound per axis.
        hi (Sequence[int]): Upper bound per axis.

    Returns:
        int: The product of the extents.
    """
    return int(_t.block_size(list(lo), list(hi)))


def _in_block(midx: Sequence[int], lo: Sequence[int], hi: Sequence[int]) -> bool:
    """Get whether ``midx`` lies in the block ``[lo, hi)``.

    Args:
        midx (Sequence[int]): A multi-index.
        lo (Sequence[int]): Lower bound per axis (inclusive).
        hi (Sequence[int]): Upper bound per axis (exclusive).

    Returns:
        bool: Whether every coordinate is in range.
    """
    return bool(_t.in_block(list(midx), list(lo), list(hi)))


def _rect_intersect(
    lo1: Sequence[int], hi1: Sequence[int], lo2: Sequence[int], hi2: Sequence[int]
) -> _Block | None:
    """Get the intersection of two blocks.

    Args:
        lo1 (Sequence[int]): First block lower bound.
        hi1 (Sequence[int]): First block upper bound.
        lo2 (Sequence[int]): Second block lower bound.
        hi2 (Sequence[int]): Second block upper bound.

    Returns:
        _Block | None: ``(lo, hi)`` of the overlap, or ``None`` when it is empty.
    """
    found = _t.rect_intersect(list(lo1), list(hi1), list(lo2), list(hi2))
    return None if found is None else _block(found)


def _peel(
    outer_lo: Sequence[int],
    outer_hi: Sequence[int],
    inner_lo: Sequence[int],
    inner_hi: Sequence[int],
) -> list[_Block]:
    """Subtract the inner block from the outer one.

    Args:
        outer_lo (Sequence[int]): Outer block lower bound.
        outer_hi (Sequence[int]): Outer block upper bound.
        inner_lo (Sequence[int]): Inner block lower bound (inside the outer block).
        inner_hi (Sequence[int]): Inner block upper bound.

    Returns:
        list[_Block]: At most ``2 * ndim`` slabs covering the outer block minus the inner one.
    """
    slabs = _t.peel(list(outer_lo), list(outer_hi), list(inner_lo), list(inner_hi))
    return [_block(slab) for slab in slabs]


def _try_merge(
    lo1: Sequence[int], hi1: Sequence[int], lo2: Sequence[int], hi2: Sequence[int]
) -> _Block | None:
    """Merge two blocks that are adjacent along one axis and aligned on the others.

    Args:
        lo1 (Sequence[int]): First block lower bound.
        hi1 (Sequence[int]): First block upper bound.
        lo2 (Sequence[int]): Second block lower bound.
        hi2 (Sequence[int]): Second block upper bound.

    Returns:
        _Block | None: The merged block, or ``None`` when the two cannot merge.
    """
    merged = _t.try_merge(list(lo1), list(hi1), list(lo2), list(hi2))
    return None if merged is None else _block(merged)


def _normalize_blocks(blocks: Sequence[_Block]) -> list[_Block]:
    """Merge adjacent blocks greedily, then sort them.

    Args:
        blocks (Sequence[_Block]): ``(lo, hi)`` blocks.

    Returns:
        list[_Block]: The normalized blocks.
    """
    normalized = _t.normalize_blocks([(list(lo), list(hi)) for lo, hi in blocks])
    return [_block(block) for block in normalized]


def _mark_region_cells(
    mask: npt.NDArray[np.bool_], lo: Sequence[int], hi: Sequence[int], block: _Block
) -> None:
    """Set, in place, the cells of a region mask that a block covers.

    Args:
        mask (npt.NDArray[np.bool_]): C-contiguous region-shaped mask, indexed relative to ``lo``.
        lo (Sequence[int]): Region lower bound.
        hi (Sequence[int]): Region upper bound.
        block (_Block): The block to mark, in the region's coordinates.

    Raises:
        TypeError: If ``mask`` is not a ``bool`` array.
        ValueError: If ``mask`` is not shaped ``hi - lo`` (the compiled helper would write
            past it), or is not C-contiguous or not writeable (it would mark a converted copy
            and leave the caller's mask unchanged).
    """
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise TypeError(f"mask must be a bool numpy array; got {type(mask).__name__!r}.")
    if not (mask.flags.c_contiguous and mask.flags.writeable):
        raise ValueError("mask must be C-contiguous and writeable: it is marked in place.")
    shape = tuple(int(b) - int(a) for a, b in zip(lo, hi, strict=True))
    if mask.shape != shape:
        raise ValueError(f"mask must have the region's shape {shape}; got {mask.shape}.")
    _t.mark_region_cells(mask, list(lo), list(hi), (list(block[0]), list(block[1])))


def _name_marked_cells(mask: npt.NDArray[np.bool_], origin: Sequence[int]) -> str:
    """Spell the marked cells of a region mask as absolute index tuples.

    Args:
        mask (npt.NDArray[np.bool_]): C-contiguous region-shaped mask.
        origin (Sequence[int]): Index of the region's first cell.

    Returns:
        str: The first few marked cells, then ``and N more`` when there are others.
    """
    return str(_t.name_marked_cells(np.ascontiguousarray(mask), list(origin)))


def _factor_blocks(cells_per_axis: Sequence[int], n_parts: int) -> tuple[int, ...]:
    """Factor ``n_parts`` into a per-axis block count proportional to the cell counts.

    Args:
        cells_per_axis (Sequence[int]): Cell count along each axis.
        n_parts (int): Number of parts.

    Returns:
        tuple[int, ...]: Block count per axis; the product is ``n_parts``.
    """
    return tuple(int(b) for b in _t.factor_blocks(list(cells_per_axis), int(n_parts)))


def _divisors(n: int) -> list[int]:
    """Get all positive divisors of ``n``, ascending.

    Args:
        n (int): A positive integer.

    Returns:
        list[int]: The sorted divisors of ``n``.
    """
    return [int(d) for d in _t.divisors(int(n))]


BERNSTEIN: Final = _Family.BERNSTEIN
"""The Bernstein family, for :data:`gram_projection`."""
CARDINAL_BSPLINE: Final = _Family.CARDINAL_BSPLINE
"""The cardinal B-spline family, for :data:`gram_projection`."""
LEGENDRE: Final = _Family.LEGENDRE
"""The shifted Legendre family, for :data:`gram_projection`."""


class LocateThresholds(NamedTuple):
    """The stopping and acceptance thresholds of one ``locate`` solve, ``stop <= accept``.

    Attributes:
        stop (float): The residual at or below which the iteration stops.
        accept (float): The residual at or below which a result counts as found.
    """

    stop: float
    accept: float


class LocateContext(NamedTuple):
    """What a ``locate`` test reads of a spline's inversion context.

    Attributes:
        spline (Bspline): The spline the solver inverts: the caller's when it is float64, else an
            exact float64 copy (the solver's own precision contract).
        handle (Any): The compiled context, which the other ``locate_*`` probes take.
        scale (float): The length the default acceptance threshold is a multiple of.
        physical (float): The image-only length the default stopping threshold is a multiple of.
        diagonal (float): The control points' bounding-box diagonal.
        parametric (float): The parametric length (one ulp of a coordinate, in physical units).
    """

    spline: Bspline
    handle: Any
    scale: float
    physical: float
    diagonal: float
    parametric: float


def _promote_to_float64(spline: Bspline) -> Bspline:
    """Return an exact float64 copy of ``spline`` (``spline`` itself when already float64).

    Args:
        spline (Bspline): The spline.

    Returns:
        Bspline: The same mapping over float64 knots and control points.
    """
    if np.dtype(spline.dtype) == np.float64:
        return spline
    spaces = [
        BsplineSpace1D(np.asarray(sub.knots, dtype=np.float64), sub.degree, periodic=sub.periodic)
        for sub in spline.space.spaces
    ]
    return Bspline(
        BsplineSpace(spaces),
        np.asarray(spline.control_points, dtype=np.float64),
        is_rational=spline.is_rational,
    )


def locate_context(spline: Bspline) -> LocateContext:
    """Build ``spline``'s inversion context, as :meth:`Bspline.locate` does on its first call.

    Args:
        spline (Bspline): A spline ``locate`` accepts.

    Returns:
        LocateContext: The context's lengths and handle.
    """
    h = ext.bspline.locate_context(spline._h)
    return LocateContext(
        spline=_promote_to_float64(spline),
        handle=h,
        scale=float(h.scale),
        physical=float(h.physical),
        diagonal=float(h.diagonal),
        parametric=float(h.parametric),
    )


def locate_default_thresholds(spline: Bspline) -> LocateThresholds:
    """Return the default thresholds of ``spline``, or the refusal ``locate`` raises.

    Args:
        spline (Bspline): A spline ``locate`` accepts.

    Returns:
        LocateThresholds: ``get_default(dtype)`` times the physical and the geometric scale.
    """
    stop, accept = ext.bspline.locate_context(spline._h).default_thresholds()
    return LocateThresholds(float(stop), float(accept))


def locate_parametric_reach(domain: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return ``max(|lo|, |hi|) / (hi - lo)`` per direction of a ``(dim, 2)`` box (0 if empty).

    Args:
        domain (npt.ArrayLike): The parametric box, one ``(lo, hi)`` row per direction.

    Returns:
        npt.NDArray[np.float64]: Shape ``(dim,)``.
    """
    box = np.asarray(domain, dtype=np.float64)
    return np.asarray(
        _t.locate_parametric_reach(
            np.ascontiguousarray(box[:, 0]), np.ascontiguousarray(box[:, 1])
        ),
        dtype=np.float64,
    )


def locate_parametric_scale(spline: Bspline) -> float:
    """Return the parametric length of ``spline``'s default acceptance threshold.

    Args:
        spline (Bspline): A spline ``locate`` accepts.

    Returns:
        float: ``max_k reach_k * net_span_k``, times ``eps(float64) / eps(dtype)``.
    """
    return float(ext.bspline.locate_context(spline._h).parametric)


locate_geometric_scale = _t.locate_geometric_scale
"""``max(physical, parametric)``: the length the default acceptance threshold is a multiple of."""


def locate_cell_physical_bounds(
    spline: Bspline,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the physical box of every cell's supporting (projected) control points.

    Args:
        spline (Bspline): A spline ``locate`` accepts.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ``(lo, hi)``, each
        ``(num_cells, rank)``, cells flat in C order over ``space.num_intervals``.
    """
    lo, hi = _t.locate_cell_physical_bounds(ext.bspline.locate_context(spline._h))
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


def locate_nearest_corner_starts(
    spline: Bspline, cell_ids: npt.ArrayLike, targets: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Return the corner of each cell whose image is nearest its target (the second start).

    Args:
        spline (Bspline): A spline ``locate`` accepts.
        cell_ids (npt.ArrayLike): One flat cell id per target, ``(n,)``.
        targets (npt.ArrayLike): Physical points, ``(n, rank)``.

    Returns:
        npt.NDArray[np.float64]: ``(n, dim)`` corners.
    """
    ctx = ext.bspline.locate_context(spline._h)
    return np.asarray(
        _t.locate_nearest_corner_starts(
            ctx,
            np.ascontiguousarray(cell_ids, dtype=np.int64),
            np.ascontiguousarray(targets, dtype=np.float64),
        ),
        dtype=np.float64,
    )


def locate_newton_refine(
    spline: Bspline,
    targets: npt.ArrayLike,
    starts: npt.ArrayLike,
    thresholds: LocateThresholds,
    max_iter: int,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    """Run one damped Newton solve per target from its start, as ``locate`` does per candidate.

    Args:
        spline (Bspline): A spline ``locate`` accepts.
        targets (npt.ArrayLike): Physical points, ``(n, rank)``.
        starts (npt.ArrayLike): Starting parametric points, ``(n, dim)``; clipped to the domain.
        thresholds (LocateThresholds): The stopping and acceptance thresholds.
        max_iter (int): Maximum accepted Newton steps per solve.

    Returns:
        tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]: ``(converged, xi)``: whether each
        final residual is within ``thresholds.accept``, and each last accepted iterate.
    """
    ctx = ext.bspline.locate_context(spline._h)
    converged, xi = _t.locate_newton_refine(
        ctx,
        np.ascontiguousarray(targets, dtype=np.float64),
        np.ascontiguousarray(starts, dtype=np.float64),
        float(thresholds.stop),
        float(thresholds.accept),
        int(max_iter),
    )
    return np.asarray(converged, dtype=np.bool_), np.asarray(xi, dtype=np.float64)


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
    "_MAX_DIAGNOSED_CELLS",
    "_MAX_NAMED_CELLS",
    "LocateContext",
    "LocateThresholds",
    "_DegreeLimit",
    "_block_size",
    "_cached_cardinal_to_bernstein_matrix",
    "_cached_cardinal_to_legendre_matrix",
    "_cached_lagrange_to_bernstein_matrix",
    "_cached_legendre_to_cardinal_matrix",
    "_divisors",
    "_factor_blocks",
    "_get_lagrange_points",
    "_in_block",
    "_mark_region_cells",
    "_name_marked_cells",
    "_normalize_blocks",
    "_peel",
    "_rect_intersect",
    "_try_merge",
    "bernstein_derivatives_point",
    "bernstein_derivatives_point_f32",
    "bernstein_interpolate",
    "bernstein_row_mirrored",
    "bernstein_row_mirrored_f32",
    "bernstein_row_unmirrored",
    "bernstein_row_unmirrored_f32",
    "bezier_extraction_1d",
    "bezier_identity_mask",
    "bincoeff",
    "blossom_span",
    "blossom_span_f32",
    "bspline_find_roots_with_budget",
    "bspline_max_insertions",
    "bspline_merge_roots",
    "bspline_morken_reimers_roots",
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
    "empty_views",
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
    "kron_apply",
    "l2_reduction_operator",
    "lagrange_extraction_1d",
    "lagrange_identity_mask",
    "locate_cell_physical_bounds",
    "locate_context",
    "locate_default_thresholds",
    "locate_geometric_scale",
    "locate_nearest_corner_starts",
    "locate_newton_refine",
    "locate_parametric_reach",
    "locate_parametric_scale",
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
    "required_scratch_size",
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
    "unique_knots_and_multiplicity",
    "yuksel_roots",
]
