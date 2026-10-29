"""Shared array and dtype helpers for Layer 2 modules.

This module centralizes recurring array-reshape idioms (moveaxis +
flatten/unflatten) and dtype validators that would otherwise be duplicated
across packages. All helpers are private (not part of the public API).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


def _validate_float_dtype(dtype: npt.DTypeLike) -> None:
    """Validate that ``dtype`` is ``float32`` or ``float64``.

    The input is normalized through ``np.dtype()`` before comparison, so
    string aliases such as ``"float32"`` are accepted alongside NumPy type
    objects and ``np.dtype`` instances. ``None`` resolves to ``np.float64``
    per NumPy semantics and is therefore accepted.

    Args:
        dtype (npt.DTypeLike): The dtype to validate.

    Raises:
        ValueError: If the dtype does not resolve to ``np.float32`` or
            ``np.float64`` (e.g. ``np.int32``, ``np.float16``).
    """
    if np.dtype(dtype).type not in (np.float32, np.float64):
        raise ValueError("dtype must be float32 or float64")


_IS_FLOAT32_BY_DTYPE_LIKE: dict[object, bool] = {}
"""Whether each dtype-like argument already validated names ``float32`` (else ``float64``).

Keyed on the caller's own argument, so a repeated call skips ``np.dtype``; only valid arguments are
stored, which keeps the table as small as the set of spellings a program uses.
"""


def _is_float32(dtype: npt.DTypeLike) -> bool:
    """Validate ``dtype`` as :func:`_validate_float_dtype` does and say whether it is ``float32``.

    This is how a wrapper picks the ``float`` or the ``double`` instantiation of a C++ function.

    Args:
        dtype (npt.DTypeLike): The dtype to validate.

    Returns:
        bool: ``True`` for ``float32``, ``False`` for ``float64``.

    Raises:
        ValueError: If the dtype does not resolve to ``np.float32`` or ``np.float64``.
    """
    try:
        return _IS_FLOAT32_BY_DTYPE_LIKE[dtype]
    except (KeyError, TypeError):  # not seen yet, or unhashable (then invalid, refused below)
        pass
    _validate_float_dtype(dtype)
    is_f32 = bool(np.dtype(dtype).type is np.float32)
    _IS_FLOAT32_BY_DTYPE_LIKE[dtype] = is_f32
    return is_f32


def _flatten_along_axis(
    arr: npt.NDArray[np.float32 | np.float64],
    axis: int,
) -> tuple[npt.NDArray[np.float32 | np.float64], tuple[int, ...]]:
    """Move ``axis`` to position 0 and flatten trailing dimensions.

    Returns a C-contiguous 2-D view suitable for passing to Numba kernels
    that operate along a single axis, together with the trailing shape
    needed to invert the operation via :func:`_unflatten_along_axis`.

    Args:
        arr (npt.NDArray[np.float32 | np.float64]): Input N-D array.
        axis (int): Axis to be moved to the leading position.

    Returns:
        tuple: ``(pts_2d, trailing_shape)`` where ``pts_2d`` has shape
        ``(arr.shape[axis], prod(trailing_shape))`` and is C-contiguous,
        and ``trailing_shape`` is the tuple of dimensions that were
        flattened.
    """
    moved = np.moveaxis(arr, axis, 0)
    return np.ascontiguousarray(moved.reshape(moved.shape[0], -1)), moved.shape[1:]


def _unflatten_along_axis(
    pts_2d: npt.NDArray[np.float32 | np.float64],
    trailing_shape: tuple[int, ...],
    axis: int,
) -> npt.NDArray[np.float32 | np.float64]:
    """Invert :func:`_flatten_along_axis`.

    Reshape a 2-D kernel output to ``(pts_2d.shape[0], *trailing_shape)``
    and move the leading axis back to position ``axis``.

    Args:
        pts_2d (npt.NDArray[np.float32 | np.float64]): 2-D array produced
            by a kernel operating along the leading axis.
        trailing_shape (tuple[int, ...]): Trailing shape returned by
            :func:`_flatten_along_axis`. ``pts_2d.shape[0]`` may differ
            from the original ``arr.shape[axis]``; this supports kernels
            whose output size differs from the input point count (e.g. degree
            elevation or Bezier extraction changes the number of control points).
        axis (int): Target axis position (same value used when calling
            :func:`_flatten_along_axis`).

    Returns:
        npt.NDArray[np.float32 | np.float64]: N-D array with axis ordering
        restored.
    """
    return np.moveaxis(pts_2d.reshape(pts_2d.shape[0], *trailing_shape), 0, axis)
