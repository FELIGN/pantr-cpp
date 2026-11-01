"""Physical-to-parametric point inversion for :class:`~pantr.bspline.Bspline`.

Layer 2 wrapper behind :meth:`pantr.bspline.Bspline.locate`. The inversion runs in the C++
library (``pantr::bspline::locate`` in ``include/pantr/bspline/locate.hpp``, which also states
the derivation of its thresholds and of the line search's constants): candidate cells from a
hierarchy over the per-cell control-point boxes, then a damped Newton solve per candidate from
the cell midpoint and, failing that, from the nearest corner.

This module normalizes the arguments, raises the messages that interpolate a Python value (the
caller's ``max_iter``, ``tol`` and the shape of ``points`` as given), and keeps the per-spline
context cached on the B-spline, where the in-place mutators invalidate it.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

import numpy as np

from .._ext import ext

if TYPE_CHECKING:
    from numpy import typing as npt

    from ._bspline import Bspline

_cpp: Any = ext.bspline
"""The compiled B-splines (private; untyped)."""

_MAX_ITER_CAP: int = 2**63 - 1
"""The largest iteration budget the compiled loop counts to (an int64).

A larger budget is clamped to it. Every accepted step strictly reduces the residual (outside the
subnormal range, see ``newton_refine`` in ``include/pantr/bspline/locate.hpp``) and a solve that
stops reducing it is abandoned, so no solve on ordinary data comes near this many steps.
"""


def _locate_context(spline: Bspline) -> Any:  # noqa: ANN401 -- the compiled LocateContext
    """Return the cached inversion context of ``spline``, building it on first use.

    Building it validates the spline (square, non-periodic, positive weights, finite control
    points), so a spline that cannot be inverted is refused before any argument is looked at.

    Args:
        spline (Bspline): The spline to invert.

    Returns:
        Any: The compiled ``LocateContext``. Invalidated by the in-place mutators of
        :class:`~pantr.bspline.Bspline`, which reset the cache attribute to ``None``.

    Raises:
        NotImplementedError: If ``rank > dim``.
        ValueError: If ``rank < dim``, a direction is periodic, a weight is not positive, or a
            control point is not finite.
    """
    if spline._locate_cache is None:
        spline._locate_cache = _cpp.locate_context(spline._h)
    return spline._locate_cache


def _locate_impl(
    spline: Bspline,
    points: npt.ArrayLike,
    tol: float | None,
    max_iter: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Invert the mapping of ``spline`` at the given physical points.

    Args:
        spline (Bspline): The spline to invert.
        points (npt.ArrayLike): Physical points, ``(n, rank)``; a 1-D input is one point of
            ``rank`` coordinates, or ``n`` scalar coordinates when ``rank == 1``.
        tol (float | None): Explicit threshold on ``||F(xi) - x||_2`` (stopping and acceptance),
            or ``None`` for the default pair.
        max_iter (int): Maximum number of Newton steps per candidate cell and start.

    Returns:
        tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]: ``(cell_ids, ref_coords)``,
        both read-only; ``-1`` and ``nan`` where a point was not found.

    Raises:
        NotImplementedError: If ``rank > dim``.
        ValueError: If the spline cannot be inverted (see :func:`_locate_context`),
            ``max_iter < 1``, ``tol`` is not positive and finite, ``points`` has the wrong shape
            or a non-finite entry, or ``tol`` is ``None`` and the parametrization can carry no
            default threshold.
    """
    context = _locate_context(spline)
    if max_iter < 1:
        raise ValueError(f"locate: max_iter must be >= 1; got {max_iter}.")
    if tol is not None and not (float(tol) > 0.0 and np.isfinite(tol)):
        raise ValueError(f"locate: tol must be positive and finite; got {tol}.")
    rank = spline.rank
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 1) if rank == 1 else pts.reshape(1, -1)
    if pts.ndim != 2 or pts.shape[1] != rank:  # noqa: PLR2004 -- (n, rank) is 2-D
        raise ValueError(
            f"locate: points must have shape (n, {rank}) for a rank-{rank} B-spline; "
            f"got shape {np.asarray(points).shape}."
        )
    budget = min(operator.index(max_iter), _MAX_ITER_CAP)
    cell_ids, ref_coords = _cpp.locate(
        context, np.ascontiguousarray(pts), None if tol is None else float(tol), budget
    )
    cell_ids.flags.writeable = False
    ref_coords.flags.writeable = False
    return cell_ids, ref_coords


__all__ = ["_locate_impl"]
