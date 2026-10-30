"""Interim access to the compiled Oslo refinement matrix (DR-2).

Knot insertion, conversion to open and periodic form, restriction, splitting, subdivision, degree
elevation and reduction, knot removal and the non-rational derivative live in the compiled
library (``pantr::bspline``, ``include/pantr/bspline/``); the ``Bspline`` methods call them through
the compiled value. This module keeps the dense Oslo matrix, which Python callers not yet ported
(the products and the THB spaces) still reach, with the argument conventions of the Numba kernel it
replaced: arrays of the knot dtype, any strides, a new array returned. It is deleted with the last
of those callers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .._ext import ext

_kernels: Any = ext._kernels
"""The compiled kernels (interim surface, untyped: each has a ``_f32`` twin)."""


def _suffix(dtype: np.dtype[Any]) -> str:
    """Get the suffix of a kernel's name for a dtype.

    Args:
        dtype (np.dtype): The dtype of the knots.

    Returns:
        str: ``"_f32"`` for ``float32``, ``""`` otherwise.
    """
    return "_f32" if dtype == np.float32 else ""


def _compute_oslo_matrix_1d_core(
    degree: int,
    old_knots: npt.NDArray[np.float32 | np.float64],
    new_knots: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Get the dense Oslo refinement matrix of a knot vector and a refinement of it.

    Args:
        degree (int): Degree of the B-spline.
        old_knots (npt.NDArray[np.float32 | np.float64]): Original knot vector.
        new_knots (npt.NDArray[np.float32 | np.float64]): Refined knot vector, of the same dtype.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The matrix ``alpha`` of shape ``(m + 1, n + 1)``
        such that ``alpha @ P`` are the control points over the refined knots.

    Raises:
        ValueError: If a knot vector has fewer than ``degree + 2`` knots.
    """
    old = np.ascontiguousarray(old_knots)
    fn = getattr(_kernels, "oslo_matrix" + _suffix(old.dtype))
    result: npt.NDArray[np.float32 | np.float64] = fn(
        int(degree), old, np.ascontiguousarray(new_knots)
    )
    return result
