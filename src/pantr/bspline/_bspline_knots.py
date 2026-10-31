"""B-spline knot vector kernels still used by B-spline code not yet ported to C++.

The knot classes, domain test and basis count here are the Numba kernels the unported
modules (the products and the rational derivative) still call; the spaces, the knot
tolerance, the snapping and interval checks, the cardinal intervals, the extraction
operators and the knot factories are C++ (``include/pantr/bspline/knots.hpp``,
``extraction.hpp`` and ``knot_factories.hpp``).
Each kernel is deleted with its last caller (the coexistence rule of the C++ port).

Every ``tol`` argument in this module is an **absolute** parametric length, in the
units of the knot vector itself: the tolerance ``pantr::bspline::knot_tolerance``
derives from the knot vector's own magnitude, which
:class:`~pantr.bspline.BsplineSpace1D` exposes as
:attr:`~pantr.bspline.BsplineSpace1D.tolerance`.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt

from .._numba_compat import nb_jit


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _get_multiplicity_of_first_knot_in_domain_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
) -> int:
    """Get the multiplicity of the first knot in the domain (i.e., the `degree`-th knot).

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.

    Returns:
        int: Multiplicity of the first knot in the domain.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    first_knot = knots[degree]
    return int(np.sum(np.abs(knots[: degree + 1] - first_knot) <= tol))


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _get_unique_knots_and_multiplicity_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    in_domain: bool = False,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Get unique knots and their multiplicities.

    Knots are grouped by *gap*: the vector is non-decreasing, so one pass suffices,
    and a class ends where the step to the next knot exceeds ``tol``. Every member
    of a class is represented by the class's **first** knot, chosen rather than
    averaged, so the values returned are knots the vector actually contains and
    the grouping is idempotent -- regrouping the represented vector reproduces it.
    An average would be a fresh rounding, and could place two adjacent classes'
    representatives a single ulp apart.

    Grouping by gap rather than by a rounding grid is what makes the answer depend
    on the knots and not on where they sit relative to an origin: two values one
    ulp apart can straddle a grid line and never merge, for any grid spacing.

    The cost of the gap rule is chaining -- ``n`` knots each within ``tol`` of the
    next collapse into one class however far apart the ends are. That is accepted
    rather than capped: a class-width cap would break the idempotence above, since
    a class split by the cap leaves two representatives that may be within ``tol``
    of each other and would merge on the next pass. Chaining needs every step of
    the chain to sit at the format's noise floor, which is a mesh that has already
    stopped being resolvable.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): **Absolute** parametric tolerance: consecutive knots closer
            than this are one knot. See :attr:`~pantr.bspline.BsplineSpace1D.tolerance`
            for how it is derived from the knot vector's own magnitude.
        in_domain (bool): If True, only consider knots in the domain.
            Defaults to False.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: Tuple of
            (unique_knots, multiplicities) where unique_knots contains the distinct knot values
            and multiplicities contains the corresponding multiplicity counts. Both arrays have
            the same length. With ``in_domain=False`` the multiplicities sum to
            ``knots.size``, so ``np.repeat(unique_knots, multiplicities)`` rebuilds
            the whole vector with each class collapsed onto its representative.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = knots.size
    # Index of the first knot of each class, and how many knots the class holds.
    first_ids = np.empty(n, dtype=np.int_)
    mult = np.zeros(n, dtype=np.int_)

    # The class containing the first and the last knot of the domain. Their whole
    # class is reported, clamped copies outside the domain included, which is what
    # the multiplicity of a boundary knot means.
    lo_index, hi_index = degree, n - degree - 1
    lo_class, hi_class = 0, 0

    j = 0
    first_ids[0] = 0
    mult[0] = 1
    for i in range(1, n):
        if knots[i] - knots[i - 1] > tol:
            j += 1
            first_ids[j] = i
        mult[j] += 1
        if i == lo_index:
            lo_class = j
        if i == hi_index:
            hi_class = j

    lo, hi = (lo_class, hi_class) if in_domain else (0, j)

    unique_knots = knots[first_ids[lo : hi + 1]]
    mults = mult[lo : hi + 1]
    return unique_knots, mults


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _get_Bspline_num_basis_1D_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    periodic: bool,
    tol: float,
) -> int:
    """Compute the number of basis functions.

    In the non-periodic case, the number of basis functions is given by
    the number of knots minus the degree minus 1.

    In the periodic case, the number of basis functions is computed as
    before, minus the regularity at the domain's beginning/end minus 1.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        periodic (bool): Whether the B-spline is periodic.
        tol (np.float32 | np.float64): Tolerance for numerical comparisons.

    Returns:
        int: Number of basis functions.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    num_basis = int(len(knots) - degree - 1)

    if periodic:
        # Determining the number of extra basis required in the periodic case.
        # This depends on the regularity of the knot vector at domain's
        # begining.
        regularity = degree - _get_multiplicity_of_first_knot_in_domain_impl(knots, degree, tol)
        num_basis -= regularity + 1

    return num_basis


def _ensure_scalar_arrays(
    values: dict[str, float | int | np.floating[Any] | None],
) -> dict[str, npt.NDArray[np.generic]]:
    """Convert scalar inputs to zero-dimensional arrays.

    Args:
        values (dict[str, Optional[float | int | np.floating]]): Mapping of parameter
            names to scalar values.

    Returns:
        dict[str, npt.NDArray[np.generic]]: Mapping of provided names to 0-D arrays.

    Raises:
        ValueError: If any value is not scalar.
    """
    arrays: dict[str, npt.NDArray[np.generic]] = {}
    for name, value in values.items():
        if value is None:
            continue
        array_value = np.array(value)
        if array_value.ndim != 0:
            raise ValueError(f"{name} must be a scalar value")
        arrays[name] = array_value
    return arrays


def _resolve_dtype_from_arrays(
    arrays: dict[str, npt.NDArray[np.generic]],
    requested_dtype: npt.DTypeLike | None,
) -> np.dtype[np.floating[Any]]:
    """Resolve the floating dtype to use for knot endpoints.

    Args:
        arrays (dict[str, npt.NDArray[np.generic]]): Scalar arrays for each value.
        requested_dtype (Optional[npt.DTypeLike]): Explicit dtype request.

    Returns:
        np.dtype[np.floating[Any]]: Resolved floating-point dtype.

    Raises:
        ValueError: If the dtype is invalid or inconsistent across values.
    """
    if requested_dtype is not None:
        dtype_obj = np.dtype(requested_dtype)
        if dtype_obj.kind != "f":
            raise ValueError("dtype must be a floating-point type")
        for name, array_value in arrays.items():
            if array_value.dtype != dtype_obj:
                raise ValueError(f"{name} must be of type dtype {dtype_obj}")
        return cast(np.dtype[np.floating[Any]], dtype_obj)

    inferred_dtype: np.dtype[np.floating[Any]] | None = None
    for array_value in arrays.values():
        candidate = (
            cast(np.dtype[np.floating[Any]], np.dtype(array_value.dtype))
            if array_value.dtype.kind == "f"
            else cast(np.dtype[np.floating[Any]], np.dtype(np.float64))
        )
        if inferred_dtype is None:
            inferred_dtype = candidate
        elif candidate != inferred_dtype:
            raise ValueError("start and end must have the same dtype")

    return (
        inferred_dtype
        if inferred_dtype is not None
        else cast(np.dtype[np.floating[Any]], np.dtype(np.float64))
    )


def _coerce_scalar(
    array_value: npt.NDArray[np.generic] | None,
    dtype_obj: np.dtype[np.floating[Any]],
    default: float,
) -> np.floating[Any]:
    """Convert a scalar array to the target dtype or use the default value.

    Args:
        array_value (Optional[npt.NDArray[np.generic]]): Scalar array to convert.
        dtype_obj (np.dtype[np.floating[Any]]): Target floating dtype.
        default (float): Default value when the array is None.

    Returns:
        np.floating[Any]: Value converted to the requested dtype.
    """
    if array_value is None:
        return dtype_obj.type(default)
    return dtype_obj.type(array_value.astype(dtype_obj, copy=False).item())


def _get_knots_ends_and_dtype(
    start: float | int | np.floating[Any] | None = None,
    end: float | int | np.floating[Any] | None = None,
    dtype: npt.DTypeLike | None = None,
) -> tuple[np.floating[Any], np.floating[Any], np.dtype[np.floating[Any]]]:
    """Get the start, end, and dtype for a knot vector.

    Args:
        start (Optional[float | int | np.floating]): Start value of the domain.
            Defaults to 0.0 if not provided.
        end (Optional[float | int | np.floating]): End value of the domain.
            Defaults to 1.0 if not provided.
        dtype (Optional[npt.DTypeLike]): Data type for the knot vector.
            If ``None``, inferred from start/end or defaults to ``np.float64``.

    Returns:
        tuple[np.floating, np.floating, np.dtype]: Tuple of (start, end, dtype).

    Raises:
        ValueError: If inputs are non-scalar, have incompatible dtypes, or if end <= start.
    """
    arrays = _ensure_scalar_arrays({"start": start, "end": end})
    dtype_obj = _resolve_dtype_from_arrays(arrays, dtype)
    start_value = _coerce_scalar(arrays.get("start"), dtype_obj, 0.0)
    end_value = _coerce_scalar(arrays.get("end"), dtype_obj, 1.0)

    if end_value <= start_value:
        raise ValueError("end must be greater than start")

    return start_value, end_value, dtype_obj


def _warmup_numba_functions() -> None:
    """Precompile numba functions with float64 signatures for faster first call.

    This function triggers compilation of the numba-decorated functions
    with float64 arrays, ensuring they are cached and ready for use.
    """
    # Small dummy arrays for warmup
    knots_dummy = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    tol_dummy = 1e-10
    degree_dummy = 2

    # Warmup each numba function with float64
    _get_multiplicity_of_first_knot_in_domain_impl(knots_dummy, degree_dummy, tol_dummy)
    _get_unique_knots_and_multiplicity_impl(knots_dummy, degree_dummy, tol_dummy, False)
    _get_Bspline_num_basis_1D_impl(knots_dummy, degree_dummy, False, tol_dummy)


# Precompile numba functions on module import
# (Moved to central thread in __init__.py)


__all__ = [
    "_get_Bspline_num_basis_1D_impl",
    "_get_knots_ends_and_dtype",
    "_get_multiplicity_of_first_knot_in_domain_impl",
    "_get_unique_knots_and_multiplicity_impl",
]
