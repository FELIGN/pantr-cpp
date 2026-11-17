"""B-spline derivative (hodograph) computation.

This module provides :func:`_derivative_bspline`, which computes the exact
first derivative of a B-spline in a given parametric direction, returning a
new :class:`~pantr.bspline.Bspline` whose value at every point equals the
derivative of the original.

Every path is in the compiled library (``include/pantr/bspline/derivative.hpp`` for a
non-rational B-spline, ``derivative_rational.hpp`` for the quotient rule over
:meth:`~pantr.bspline.Bspline.multiply` that a rational one takes); this module dispatches.

Works for non-rational and rational (NURBS) B-splines of any parametric
dimension, and correctly preserves per-direction boundary structure
(open / periodic / non-open).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Bspline


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
            clamped or repeats a knot more than ``p`` times, since re-elevating it runs
            A5.9 and A5.9 assumes it is clamped and cannot walk such a run. A direction
            clamped at both ends yields a clamped hodograph, and so does **any** direction
            of degree 1, whose hodograph is degree 0 and has no end runs to lack; every
            other unclamped direction is refused. Clamped is not enough, though: an
            interior knot of multiplicity ``p + 1`` (a ``C^-1`` knot) becomes a run of
            ``p + 1`` in a hodograph of degree ``p - 1``, one more than a knot may repeat,
            and the direction is refused, degree 1 included.
    """
    from ._bspline import _index  # noqa: PLC0415

    return bspline._from_result(bspline._h.derivative_keep_degree(_index(direction)))


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
        # The quotient rule takes degree-preserving A' and w', producing degree 2p, which
        # exceeds the original degree p: `keep_degree` changes nothing.
        return bspline._from_result(bspline._h.derivative_rational(_index(direction)))
    if keep_degree:
        return _derivative_keep_degree_nonrational(bspline, direction)
    return bspline._from_result(bspline._h.derivative(_index(direction)))
