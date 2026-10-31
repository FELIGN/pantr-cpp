"""Python helpers for knot-vector factories: resolving the ends and the dtype.

The knot classes, the domain test, the basis count, the spaces, the knot tolerance, the
snapping and interval checks, the cardinal intervals, the extraction operators and the knot
factories are C++ (``include/pantr/bspline/knots.hpp``, ``extraction.hpp`` and
``knot_factories.hpp``); the Numba kernels this module held were deleted with their last
callers (the coexistence rule of the C++ port).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt


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


__all__ = [
    "_get_knots_ends_and_dtype",
]
