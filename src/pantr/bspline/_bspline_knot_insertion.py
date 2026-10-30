"""Interim access to the compiled one-direction refinement kernels (DR-2).

Knot insertion, conversion to open and periodic form, restriction, splitting and subdivision live
in the compiled library (``pantr::bspline``, ``include/pantr/bspline/knot_insertion.hpp``,
``periodic.hpp``, ``restrict.hpp``); the ``Bspline`` methods call them through the compiled value.
This module keeps the three one-direction entry points that Python callers not yet ported
(degree elevation and reduction, the derivative, the products and the THB spaces) still reach,
with the argument conventions of the Numba kernels and Layer 2 helpers they replaced: arrays of
the knot dtype, any strides, new arrays returned. It shrinks as those callers are ported and is
deleted with the last one.
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


def _to_open_bspline_1d_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    ctrl_2d: npt.NDArray[np.float32 | np.float64],
    periodic: bool,
    tol: float,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Convert a single parametric direction to an open (clamped) knot vector.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): Knot vector, ghost knots included if
            ``periodic``.
        degree (int): Polynomial degree.
        ctrl_2d (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(n_stored, cols)``.
        periodic (bool): Whether the direction is periodic.
        tol (float): Absolute parametric tolerance.

    Returns:
        tuple[npt.NDArray, npt.NDArray]: ``(open_knots, open_ctrl)``.

    Raises:
        ValueError: If the direction is already open and not periodic.
    """
    k = np.ascontiguousarray(knots)
    fn = getattr(_kernels, "to_open_1d" + _suffix(k.dtype))
    result: tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]] = fn(
        k, int(degree), np.ascontiguousarray(ctrl_2d), bool(periodic), float(tol)
    )
    return result


def _to_periodic_bspline_1d_impl(  # noqa: PLR0913
    open_knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    ctrl_2d: npt.NDArray[np.float32 | np.float64],
    m_bdy: int,
    tol: float,
    tol_deviation: float | None = None,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Convert an open (clamped) 1D B-spline to periodic form.

    Args:
        open_knots (npt.NDArray[np.float32 | np.float64]): Open knot vector.
        degree (int): Polynomial degree.
        ctrl_2d (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(n_open, cols)``.
        m_bdy (int): Target boundary multiplicity, in ``[1, degree]``.
        tol (float): Absolute parametric tolerance.
        tol_deviation (float | None): How far the open form may sit from the periodic subspace,
            in control-point units, or ``None`` for the round-off floor alone (``np.inf`` removes
            the closure test; the seam test keeps its floor). Defaults to ``None``.

    Returns:
        tuple[npt.NDArray, npt.NDArray]: ``(periodic_knots, periodic_ctrl)``.

    Raises:
        ValueError: If ``m_bdy`` is outside ``[1, degree]``, or the function is not periodic.
    """
    k = np.ascontiguousarray(open_knots)
    fn = getattr(_kernels, "to_periodic_1d" + _suffix(k.dtype))
    result: tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]] = fn(
        k,
        int(degree),
        np.ascontiguousarray(ctrl_2d),
        int(m_bdy),
        float(tol),
        None if tol_deviation is None else float(tol_deviation),
    )
    return result
