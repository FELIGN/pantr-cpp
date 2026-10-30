"""B-spline evaluation: the wrapper over the compiled evaluation.

This module normalises the points and the ``out`` array of :meth:`Bspline.evaluate` and
:meth:`Bspline.evaluate_derivatives` and hands them to the compiled value of the B-spline
(``pantr::bspline::evaluate`` and ``evaluate_derivatives`` in the C++ library). The Cox-de Boor
recurrence, the tensor-product contraction, the quotient rule of a rational B-spline and the
domain check run there; this module keeps what spells a Python value or a NumPy dtype: the point
and lattice checks, the ``out`` checks and the shape of what is returned.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy import typing as npt

from ..basis._basis_utils import _validate_out_array
from ..quad import PointsLattice

if TYPE_CHECKING:
    from . import Bspline


def _normalize_eval_points(
    pts: npt.ArrayLike,
    dim: int,
    dtype: npt.DTypeLike,
) -> npt.NDArray[np.float32 | np.float64]:
    """Coerce and validate a user-supplied array of evaluation points.

    Establishes the shape and dtype the compiled evaluation assumes: an array-like such as a plain
    list is converted, and a ``(n_pts, 1)`` column is a valid 1D input.

    For ``dim == 1`` both ``(n_pts,)`` and ``(n_pts, 1)`` are accepted and normalized
    to ``(n_pts,)``. A column of points is what the general ``(n_pts, dim)``
    convention degenerates to at ``dim == 1``, and it is the shape the library's own
    documented example uses, so it is normalized rather than rejected.

    Args:
        pts (npt.ArrayLike): Evaluation points, or anything ``np.asarray`` accepts.
        dim (int): Parametric dimension of the B-spline.
        dtype (npt.DTypeLike): Required dtype, i.e. the B-spline's own.

    Returns:
        npt.NDArray[np.float32 | np.float64]: C-contiguous array of shape
        ``(n_pts,)`` when ``dim == 1`` and ``(n_pts, dim)`` otherwise. The input is
        returned unchanged when it already satisfies both, so the common path copies
        nothing.

    Raises:
        ValueError: If ``pts`` cannot be converted to an array, if its shape is not
            one of the accepted forms, or if its dtype differs from ``dtype``.
    """
    try:
        arr = np.asarray(pts)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"pts could not be converted to an array: {exc}") from exc

    if dim == 1:
        if arr.ndim == 2 and arr.shape[1] == 1:  # noqa: PLR2004
            arr = np.ascontiguousarray(arr).reshape(-1)
        elif arr.ndim != 1:
            raise ValueError(
                f"pts must have shape (n_pts,) or (n_pts, 1) for a 1D B-spline; "
                f"got shape {arr.shape}"
            )
    elif arr.ndim != 2 or arr.shape[1] != dim:  # noqa: PLR2004
        raise ValueError(f"pts must be a 2D array with {dim} columns; got shape {arr.shape}")

    if arr.dtype != dtype:
        raise ValueError("Points dtype must match B-spline dtype")

    return np.ascontiguousarray(arr)


def _check_lattice(spline: Bspline, pts: PointsLattice) -> None:
    """Check a lattice's dimension (and, from two directions on, its dtype) against a B-spline.

    A 1D B-spline takes a lattice of any float dtype, as before; the compiled evaluation computes
    in the common type of the knots and the points.

    Args:
        spline (Bspline): The B-spline.
        pts (PointsLattice): The lattice.

    Raises:
        ValueError: On a dimension mismatch, or a dtype mismatch from two directions on.
    """
    dim = spline.dim
    if dim == 1:
        if pts.dim != 1:
            raise ValueError("Points lattice must be 1D")
        return
    if pts.dim != dim:
        raise ValueError(
            f"Points lattice dimension {pts.dim} does not match B-spline dimension {dim}"
        )
    if pts.dtype != spline.dtype:
        raise ValueError("Points dtype must match B-spline dtype")


def _points_and_base(
    spline: Bspline, pts: npt.ArrayLike | PointsLattice
) -> tuple[Any, tuple[int, ...], bool]:
    """Check the points and give the argument the compiled call takes.

    Args:
        spline (Bspline): The B-spline.
        pts (npt.ArrayLike | PointsLattice): The points or the lattice.

    Returns:
        tuple[Any, tuple[int, ...], bool]: The compiled lattice or the C-contiguous ``(n, dim)``
        points, the leading shape of the result (``(n,)`` or the points per direction), and
        whether the points are a lattice.

    Raises:
        ValueError: On a shape or dtype mismatch (the messages of the previous release).
    """
    if isinstance(pts, PointsLattice):
        _check_lattice(spline, pts)
        return pts._h, tuple(int(p.shape[0]) for p in pts.pts_per_dir), True
    arr = _normalize_eval_points(pts, spline.dim, spline.dtype)
    n = arr.shape[0]
    return (arr.reshape(n, 1) if spline.dim == 1 else arr), (n,), False


def _output(
    out: npt.NDArray[np.float32 | np.float64] | None,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    cols: int,
) -> tuple[
    npt.NDArray[np.float32 | np.float64],
    npt.NDArray[np.float32 | np.float64],
    npt.NDArray[np.float32 | np.float64] | None,
]:
    """Validate the caller's ``out`` or allocate, and give the C-contiguous buffer to write.

    A C-contiguous ``out`` is written in place. Any other layout is filled from a temporary once
    the compiled call returns (``reshape`` of a non-contiguous array may copy, and a write into a
    copy would be lost).

    Args:
        out (npt.NDArray[np.float32 | np.float64] | None): The caller's array.
        shape (tuple[int, ...]): The shape it must have.
        dtype (np.dtype): The dtype it must have.
        cols (int): The last extent of ``shape``.

    Returns:
        tuple: The array that holds the result (``out`` or a new one), the ``(rows, cols)`` buffer
        the compiled call writes, and the temporary to copy into ``out`` afterwards, if any.

    Raises:
        ValueError: If ``out`` has the wrong shape or dtype, or is not writeable.
    """
    rows = math.prod(shape) // cols
    if out is None:
        arr = np.empty(shape, dtype=dtype)
        return arr, arr.reshape(rows, cols), None
    _validate_out_array(out, shape, dtype)
    if out.flags.c_contiguous:
        return out, out.reshape(rows, cols), None
    tmp = np.empty(shape, dtype=dtype)
    return out, tmp.reshape(rows, cols), tmp


def _check_orders(dim: int, orders: Sequence[int]) -> None:
    """Check the number and sign of the derivative orders (the compiled call checks them too).

    The wrapper checks them first so that wrong orders are reported before a wrong point
    argument, as the previous release did; the compiled evaluation cannot, since it is reached
    only after the points are normalised.

    Args:
        dim (int): The parametric dimension of the B-spline.
        orders (Sequence[int]): The derivative orders, already integers.

    Raises:
        ValueError: If ``len(orders) != dim`` or an order is negative (the length first).
    """
    if len(orders) != dim:
        raise ValueError(f"len(orders) must equal spline.dim ({dim}), got {len(orders)}")
    for d, order in enumerate(orders):
        if order < 0:
            raise ValueError(f"orders[{d}] must be >= 0, got {order}")


def _evaluate_Bspline(
    spline: Bspline,
    pts: npt.ArrayLike | PointsLattice,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate the B-spline at the given points.

    Args:
        spline (Bspline): The B-spline.
        pts (npt.ArrayLike | PointsLattice): The points (a 1D B-spline takes ``(n_pts,)`` or
            ``(n_pts, 1)``) or a lattice.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array of shape
            ``(*pts_base_shape, cp_size)``, the weight column of a rational B-spline included
            (which keeps the weight, with the coordinates divided by it). Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The values, squeezed; a view of ``out`` when given.

    Raises:
        ValueError: On a bad point, lattice or ``out``, or a point outside the domain.
    """
    arg, base, lattice = _points_and_base(spline, pts)
    cp_size = spline._shape[-1]
    arr, buf, pending = _output(out, (*base, cp_size), spline._dtype, cp_size)
    if lattice:
        spline._h.evaluate_lattice(arg, buf)
    else:
        spline._h.evaluate(arg, buf)
    if pending is not None:
        arr[...] = pending
    if spline._is_rational:
        return arr[..., :-1].squeeze()
    return arr.squeeze()


def _evaluate_Bspline_deriv(
    spline: Bspline,
    pts: npt.ArrayLike | PointsLattice,
    orders: Sequence[int],
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate a partial derivative of the B-spline at the given points.

    Args:
        spline (Bspline): The B-spline.
        pts (npt.ArrayLike | PointsLattice): The points or the lattice.
        orders (Sequence[int]): One derivative order per direction (anything with ``__index__``).
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array of the shape
            of the result. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The derivative, of shape ``(*pts_base_shape,)`` for
        a rank-one B-spline or ``(*pts_base_shape, rank)`` otherwise (the weight of a rational
        B-spline is divided out); ``out`` when given.

    Raises:
        ValueError: On a bad order (the orders are checked before the points), point, lattice or
            ``out``, or a point outside the domain.
        TypeError: If an order is not an integer.
        ZeroDivisionError: If the basis derivative recurrence divides by zero, which a knot
            repeated more than ``degree + 1`` times within a point's support produces.
    """
    orders_list = [operator.index(o) for o in orders]
    _check_orders(spline.dim, orders_list)
    arg, base, lattice = _points_and_base(spline, pts)
    rank = spline.rank
    shape = (*base, rank) if rank > 1 else base
    arr, buf, pending = _output(out, shape, spline._dtype, rank)
    if lattice:
        spline._h.evaluate_derivatives_lattice(arg, orders_list, buf)
    else:
        spline._h.evaluate_derivatives(arg, orders_list, buf)
    if pending is not None:
        arr[...] = pending
    return arr
