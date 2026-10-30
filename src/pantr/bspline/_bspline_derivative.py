"""B-spline derivative (hodograph) computation.

This module provides :func:`_derivative_bspline`, which computes the exact
first derivative of a B-spline in a given parametric direction, returning a
new :class:`~pantr.bspline.Bspline` whose value at every point equals the
derivative of the original.

The non-rational derivative (``pantr::bspline::derivative_nonrational`` and
``derivative_keep_degree_nonrational``, ``include/pantr/bspline/derivative.hpp``) and the common
refinement of two B-splines (``refine_to_common_space_1d``) live in the compiled library. The
rational (NURBS) derivative stays here until ``Bspline.multiply``, which its quotient rule uses, is
ported; it calls the compiled pieces.

Works for non-rational and rational (NURBS) B-splines of any parametric
dimension, and correctly preserves per-direction boundary structure
(open / periodic / non-open).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._array_utils import _flatten_along_axis, _unflatten_along_axis
from .._ext import ext
from ._bspline_knots import _get_unique_knots_and_multiplicity_impl
from ._bspline_space_1d import BsplineSpace1D
from ._bspline_space_nd import BsplineSpace

if TYPE_CHECKING:
    import numpy.typing as npt

    from . import Bspline

_cpp: Any = ext.bspline
"""The compiled B-splines (private; untyped: one class and function set per dtype)."""

_kernels: Any = ext._kernels
"""The compiled one-direction kernels (interim surface, untyped): the periodic conversion."""


def _one_direction(name: str, dtype: npt.DTypeLike) -> Any:  # noqa: ANN401
    """Get a compiled one-direction kernel for a dtype.

    Args:
        name (str): The kernel, ``"to_open_1d"`` or ``"to_periodic_1d"``.
        dtype (npt.DTypeLike): ``float32`` or ``float64``.

    Returns:
        Any: The kernel of that dtype.
    """
    return getattr(_kernels, name + ("_f32" if np.dtype(dtype) == np.float32 else ""))


def _refine_to_common_space_1d(
    a: Bspline,
    b: Bspline,
    direction: int,
) -> tuple[Bspline, Bspline]:
    """Refine two B-splines to share a common knot vector in one direction.

    Computes the knots missing from each B-spline in direction ``direction``
    and inserts them so both share the same knot vector in that direction.

    Args:
        a (~pantr.bspline.Bspline): First B-spline.
        b (~pantr.bspline.Bspline): Second B-spline.
        direction (int): Parametric direction to refine.

    Returns:
        tuple[~pantr.bspline.Bspline, ~pantr.bspline.Bspline]: Both B-splines
        refined to the same knot vector in direction ``direction``; one that
        needed no insertion is returned as it is.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    fn = getattr(_cpp, "refine_to_common_space_1d" + ("_f32" if a.dtype == np.float32 else ""))
    ha, hb = fn(a._h, b._h, int(direction))
    return (
        a if ha.same_rep(a._h) else a._from_result(ha),
        b if hb.same_rep(b._h) else b._from_result(hb),
    )


def _derivative_keep_degree_nonrational(bspline: Bspline, direction: int) -> Bspline:
    """Compute derivative with degree preservation for a non-rational B-spline.

    Computes the derivative (degree ``p - 1``) and degree-elevates by 1 to
    restore the original degree ``p``, in the compiled library, without an intermediate
    :class:`~pantr.bspline.Bspline` object.

    Args:
        bspline (~pantr.bspline.Bspline): A non-rational B-spline.
        direction (int): Parametric direction for differentiation.

    Returns:
        ~pantr.bspline.Bspline: Derivative B-spline of the same degree as the
        input.  A periodic ``direction`` comes back in open representation
        (degree elevation does not support periodic); every other direction
        keeps the representation it had.

    Raises:
        ValueError: If the direction's degree exceeds the exactness envelope of the
            binomial-coefficient kernel (``bincoeff_max_n``, 61), or if the
            hodograph's own knot vector -- ``knots[1:-1]`` at degree ``p - 1`` -- is not
            clamped, since re-elevating it runs A5.9 and A5.9 assumes it is. A clamped
            direction always yields a clamped hodograph, and so does **any** direction of
            degree 1, whose hodograph is degree 0 and has no end runs to lack; every other
            unclamped direction is refused.
    """
    from ._bspline import _index  # noqa: PLC0415

    return bspline._from_result(bspline._h.derivative_keep_degree(_index(direction)))


def _tile_scalar_bspline(w: Bspline, target_rank: int) -> Bspline:
    """Tile a rank-1 B-spline to match a target rank.

    Repeats the single control point component along the last axis so that
    the resulting B-spline has ``target_rank`` components, each identical to
    the original scalar function.

    Args:
        w (~pantr.bspline.Bspline): A rank-1 (scalar) non-rational B-spline.
        target_rank (int): Desired number of output components.

    Returns:
        ~pantr.bspline.Bspline: B-spline with ``target_rank`` identical
        components.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    from . import Bspline as BsplineCls  # noqa: PLC0415

    ctrl = np.repeat(w.control_points, target_rank, axis=-1)
    return BsplineCls(w.space, ctrl, is_rational=False)


def _derivative_rational(bspline: Bspline, direction: int) -> Bspline:
    """Compute the partial derivative of a rational (NURBS) B-spline.

    Delegates to :func:`_derivative_rational_open`, converting ``direction`` to open
    form first when it is periodic and closing the seam again afterwards.

    **Why the round-trip.** The quotient rule raises the degree to ``2p`` through
    :meth:`~pantr.bspline.Bspline.multiply`, and the seam multiplicity that product
    needs does not fit a periodic knot vector: differentiating a periodic rational
    used to fail with ``Inserting these knots would exceed the maximum multiplicity``,
    raised four frames down with nothing naming the cause, for both values of
    ``keep_degree`` since the dispatcher routes both here. Going through the open form
    is what :meth:`~pantr.bspline.Bspline.elevate_degree` and
    :meth:`~pantr.bspline.Bspline.reduce_degree` already do for the same reason.

    **The seam multiplicity to close back to.** Continuity is the invariant to carry
    across, not multiplicity, because the degree changes. A periodic spline of degree
    ``p`` whose seam has multiplicity ``m`` is ``C^(p - m)`` there. The numerator
    ``A'w - Aw'`` carries one differentiation, so it is ``C^(p - m - 1)``; the
    denominator ``w^2`` is still ``C^(p - m)``; refined to their common space the pair
    carries the weaker of the two. At degree ``2p`` that continuity is multiplicity
    ``2p - (p - m - 1) = p + m + 1``, which is at most ``2p`` because a C^0 seam
    (``m == p``) is refused first: the derivative of such a spline is discontinuous at
    the seam, so it has no periodic form.

    Args:
        bspline (~pantr.bspline.Bspline): Rational B-spline to differentiate.
        direction (int): Parametric direction, in ``[0, dim)``.

    Returns:
        ~pantr.bspline.Bspline: Rational derivative B-spline, periodic in ``direction``
        exactly when the input was.

    Raises:
        ValueError: If ``direction`` is periodic and its seam has multiplicity equal to
            the degree (C^0 at the seam), where the derivative is discontinuous.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_derivative_bspline` instead.
    """
    from . import Bspline as BsplineCls  # noqa: PLC0415

    space_1d = bspline.space.spaces[direction]
    if not space_1d.periodic:
        return _derivative_rational_open(bspline, direction)

    p_deg = space_1d.degree
    tol = float(space_1d.tolerance)
    _, mults = _get_unique_knots_and_multiplicity_impl(space_1d.knots, p_deg, tol, in_domain=True)
    m_bdy = int(mults[0])
    if m_bdy >= p_deg:
        raise ValueError(
            "Cannot differentiate a periodic rational B-spline whose seam is C^0 "
            f"(seam multiplicity {m_bdy} equals the degree {p_deg}): its derivative is "
            "discontinuous at the seam and has no periodic representation."
        )

    # --- to open form in this direction only ---
    pts_2d, trailing_shape = _flatten_along_axis(bspline.control_points, direction)
    open_knots, open_pts_2d = _one_direction("to_open_1d", bspline.dtype)(
        np.ascontiguousarray(space_1d.knots), p_deg, pts_2d, True, tol
    )
    open_spaces = list(bspline.space.spaces)
    open_spaces[direction] = BsplineSpace1D(open_knots, p_deg)
    open_bspline = BsplineCls(
        BsplineSpace(open_spaces),
        _unflatten_along_axis(open_pts_2d, trailing_shape, direction),
        is_rational=True,
    )

    result = _derivative_rational_open(open_bspline, direction)

    # --- and back to periodic ---
    res_space_1d = result.space.spaces[direction]
    res_degree = res_space_1d.degree
    m_bdy_new = p_deg + m_bdy + 1
    res_tol = float(res_space_1d.tolerance)
    res_pts_2d, res_trailing = _flatten_along_axis(result.control_points, direction)
    per_knots, per_pts_2d = _one_direction("to_periodic_1d", bspline.dtype)(
        np.ascontiguousarray(res_space_1d.knots), res_degree, res_pts_2d, m_bdy_new, res_tol, None
    )
    per_spaces = list(result.space.spaces)
    per_spaces[direction] = BsplineSpace1D(per_knots, res_degree, periodic=True)
    return BsplineCls(
        BsplineSpace(per_spaces),
        _unflatten_along_axis(per_pts_2d, res_trailing, direction),
        is_rational=True,
    )


def _derivative_rational_open(bspline: Bspline, direction: int) -> Bspline:
    """Compute the partial derivative of a rational (NURBS) B-spline.

    Applies the quotient rule: for ``f = A/w``,
    ``f' = (A'w - Aw') / w^2``.

    Uses degree-preserving derivatives for ``A'`` and ``w'`` so that the
    products ``A'w`` and ``Aw'`` are degree ``2p`` (matching ``w^2``),
    eliminating the need for a separate numerator degree elevation.

    Since ``multiply()`` requires operands to have the same rank, the scalar
    weight B-splines (rank 1) are tiled to match the vector rank before
    multiplication.

    Args:
        bspline (~pantr.bspline.Bspline): A rational B-spline.
        direction (int): Parametric direction for differentiation.

    Returns:
        ~pantr.bspline.Bspline: Rational derivative B-spline.

    Note:
        ``direction`` must not be periodic: the quotient rule below raises the degree
        to ``2p`` through :meth:`~pantr.bspline.Bspline.multiply`, and a periodic knot
        vector cannot carry the seam multiplicity that product needs.
        :func:`_derivative_rational` is what enforces that, by converting the direction
        to open form first.
        Inputs are otherwise assumed to be correct (no validation performed).
        For general use, call :func:`_derivative_bspline` instead.
    """
    from . import Bspline as BsplineCls  # noqa: PLC0415

    ctrl = bspline.control_points
    vec_rank = ctrl.shape[-1] - 1  # number of non-weight components

    # Decompose into numerator A (weighted coords) and denominator w (weights).
    a_bspline = BsplineCls(bspline.space, ctrl[..., :-1], is_rational=False)
    w_bspline = BsplineCls(bspline.space, ctrl[..., -1:], is_rational=False)

    # Degree-preserving derivatives: A' and w' remain degree p.
    a_prime = _derivative_keep_degree_nonrational(a_bspline, direction)
    w_prime = _derivative_keep_degree_nonrational(w_bspline, direction)

    # Tile scalar B-splines to match vector rank for multiply().
    w_tiled = _tile_scalar_bspline(w_bspline, vec_rank)
    w_prime_tiled = _tile_scalar_bspline(w_prime, vec_rank)

    # Products are degree p + p = 2p (same as w^2).
    num1 = a_prime.multiply(w_tiled)
    num2 = a_bspline.multiply(w_prime_tiled)

    # Subtract CPs (same space guaranteed).
    numerator_ctrl = num1.control_points - num2.control_points
    numerator = BsplineCls(num1.space, numerator_ctrl, is_rational=False)

    # Compute denominator: w^2 (degree 2p).
    denom = w_bspline.multiply(w_bspline)

    # Refine both to a common knot vector in the target direction.
    numerator, denom = _refine_to_common_space_1d(numerator, denom, direction)

    # Assemble rational B-spline.
    result_ctrl = np.concatenate([numerator.control_points, denom.control_points], axis=-1)
    return BsplineCls(numerator.space, result_ctrl, is_rational=True)


def _derivative_bspline(bspline: Bspline, direction: int, *, keep_degree: bool = False) -> Bspline:
    """Compute the first partial derivative of a B-spline.

    This is the main entry point for derivative computation. It dispatches
    to the appropriate implementation based on the B-spline's properties
    (rational vs non-rational, dimension, periodicity).

    Args:
        bspline (~pantr.bspline.Bspline): The B-spline to differentiate.
        direction (int): Parametric direction for differentiation. Must be
            in ``[0, dim)``.
        keep_degree (bool): If ``True``, the result has the same degree as the
            input by applying degree elevation after differentiation. Defaults
            to ``False``.

    Returns:
        ~pantr.bspline.Bspline: A new B-spline representing the derivative.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :meth:`~pantr.bspline.Bspline.derivative` instead.
    """
    from ._bspline import _index  # noqa: PLC0415

    if bspline.is_rational:
        # The derivative of a rational B-spline already uses degree-preserving A'/w', producing
        # degree 2p which exceeds the original degree p: `keep_degree` changes nothing.
        return _derivative_rational(bspline, direction)
    if keep_degree:
        return _derivative_keep_degree_nonrational(bspline, direction)
    return bspline._from_result(bspline._h.derivative(_index(direction)))
