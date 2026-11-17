"""A point at or within tolerance of a domain end is evaluated on the nearest non-empty span.

Regression tests for two span searches that assigned such a point to the wrong span:

* The basis span search clamps into the in-domain spans, and the clamped span at an end is empty
  when the end knot is repeated past ``degree + 1`` (left) or reaches ``degree + 1`` before a
  trailing knot (right). The basis there came out all zeros and the spline evaluated to 0.
* The blossom's span search sent a parameter just below the first knot to the last span, so the
  blossom there was that of the end of the curve.

Oracles: the space's own values at the domain end (``0.0``), and the analytic end values of a
clamped vector (the basis is ``[0, 0, 1]`` at the right end of the last non-empty span, and the
spline interpolates its end control point there).
"""

from __future__ import annotations

import numpy as np
import pytest

from pantr._testing import blossom_span, evaluate_blossom_1d
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D

_EPS = float(np.finfo(np.float64).eps)
_UNIT_ROUNDOFF = _EPS / 2.0
_P = 2

_LEFT_HEAVY = np.array([0.0, 0, 0, 0, 1, 1, 1])
"""Degree 2, knot 0 repeated four times (p + 2): the clamped first span ``[0, 0)`` is empty."""

_RIGHT_HEAVY = np.array([0.0, 0, 0, 1, 2, 2, 2, 3, 3])
"""Degree 2, knot 2 repeated three times (p + 1) before a trailing knot: ``[2, 2)`` is empty."""

_BASIS_TOL = 8.0 * (_P + 1) * _EPS
"""Absolute ``8 (p + 1) eps``: the bound the C++ restriction test asserts for basis values."""


def _eval_tol(coefs: np.ndarray) -> float:
    """``8 (p + 1) eps * max|c|``, the bound #7 states for the evaluation.

    Not a worst-case derivation: summing the per-entry basis bound over the ``p + 1`` terms gives
    ``(p + 1)`` times this. The actual error is far below either at these points.
    """
    return _BASIS_TOL * float(np.abs(coefs).max())


@pytest.mark.parametrize("x", [-1e-17, -1e-16])
def test_a_point_within_tolerance_below_a_heavy_start_takes_the_first_non_empty_span(
    x: float,
) -> None:
    """Below the start, inside the band: the values and the evaluation of the start."""
    space = BsplineSpace1D(_LEFT_HEAVY, _P)
    assert abs(x) < float(space.tolerance)
    coefs = np.arange(1.0, 5.0)
    curve = Bspline(BsplineSpace([space]), coefs)
    at_start, first_start = space.tabulate_basis([0.0])
    vals, first = space.tabulate_basis([x])
    assert int(first[0]) == int(first_start[0]) == 1
    np.testing.assert_allclose(vals[0], at_start[0], rtol=0.0, atol=_BASIS_TOL)
    assert abs(float(np.asarray(curve.evaluate(np.array([x]))).reshape(-1)[0]) - 2.0) <= _eval_tol(
        coefs
    )


@pytest.mark.parametrize("x", [2.0, 2.0 + 1e-15])
def test_a_point_at_or_within_tolerance_above_a_heavy_end_takes_the_last_non_empty_span(
    x: float,
) -> None:
    """At the end, or above it inside the band: the left limit, ``[0, 0, 1]`` and ``c[3]``."""
    space = BsplineSpace1D(_RIGHT_HEAVY, _P)
    assert abs(x - 2.0) < float(space.tolerance)
    coefs = np.arange(1.0, space.num_basis + 1.0)
    curve = Bspline(BsplineSpace([space]), coefs)
    vals, first = space.tabulate_basis([x])
    assert int(first[0]) == 1
    np.testing.assert_allclose(vals[0], [0.0, 0.0, 1.0], rtol=0.0, atol=_BASIS_TOL)
    assert abs(float(np.asarray(curve.evaluate(np.array([x]))).reshape(-1)[0]) - 4.0) <= _eval_tol(
        coefs
    )


def test_the_derivatives_at_a_heavy_end_are_those_of_the_left_limit() -> None:
    """The first derivative at the right end equals the one just inside the last span.

    Oracle: on ``[1, 2)`` the basis is one polynomial piece, so its derivative at ``2`` is the
    limit of the derivative at ``2 - h``; with ``h = 1e-9`` the difference is ``h`` times the
    second derivative, at most ``4 h`` here (knot spacing 1, degree 2), plus rounding.
    """
    space = BsplineSpace1D(_RIGHT_HEAVY, _P)
    d_end, f_end = space.tabulate_basis_derivatives([2.0], 1)
    d_in, f_in = space.tabulate_basis_derivatives([2.0 - 1e-9], 1)
    assert int(f_end[0]) == int(f_in[0]) == 1
    np.testing.assert_allclose(d_end[0, 1], d_in[0, 1], rtol=0.0, atol=4e-9 + 1e-12)


def test_a_parameter_within_tolerance_below_a_clamped_start_is_in_the_first_span() -> None:
    """The blossom span of a value 4 eps below a clamped start is ``degree``, as at the start.

    The blossom there differs from the one at the start by at most the shift times the slope
    (``4 |u|``: slope ``p max|dP| / min span = 2 * 1 / 0.5``) plus ``32`` units of roundoff
    (8 roundings on values of magnitude at most 4), the bound the C++ pin asserts.
    """
    kn = np.array([0, 0, 0, 0.5, 1, 1, 1.0])
    cp = np.array([[1.0], [2.0], [3.0], [4.0]])
    u = -8.9e-16
    tol = 8.0 * _EPS
    assert int(blossom_span(kn, 0.0, 3)) == 2
    assert int(blossom_span(kn, u, 3)) == 2
    at_start = evaluate_blossom_1d(kn, 2, cp, np.array([0.0, 0.0]), tol)
    below = evaluate_blossom_1d(kn, 2, cp, np.array([u, u]), tol)
    assert float(at_start[0]) == 1.0
    assert abs(float(below[0]) - float(at_start[0])) <= 4.0 * abs(u) + 32.0 * _UNIT_ROUNDOFF


def test_a_parameter_below_a_heavy_start_takes_the_span_of_the_first_knot() -> None:
    """Below a start repeated ``p + 2`` times the blossom span is the one ``0.0`` itself gets."""
    n = _LEFT_HEAVY.shape[0] - _P - 2
    assert int(blossom_span(_LEFT_HEAVY, -1e-17, n)) == int(blossom_span(_LEFT_HEAVY, 0.0, n)) == 3


def test_a_parameter_at_a_heavy_end_takes_the_last_non_empty_span_in_the_blossom() -> None:
    """At the end beside an empty span the blossom starts from the last non-empty span.

    Oracle: de Boor on ``[1, 2)`` by hand with control points ``1 .. 6``: ``f[1.5, 2]`` is
    ``(P2 + P3) / 2 = 3.5`` and ``f[2, 2]`` the end value ``P3 = 4``; rounding of two levels of
    convex combinations of values at most 6, taken as ``8 * 6`` units of roundoff.
    """
    n = _RIGHT_HEAVY.shape[0] - _P - 2
    assert int(blossom_span(_RIGHT_HEAVY, 2.0, n)) == 3
    cp = np.arange(1.0, 7.0).reshape(-1, 1)
    tol = 8.0 * _EPS
    bound = 48.0 * _UNIT_ROUNDOFF
    mixed = evaluate_blossom_1d(_RIGHT_HEAVY, _P, cp, np.array([1.5, 2.0]), tol)
    end = evaluate_blossom_1d(_RIGHT_HEAVY, _P, cp, np.array([2.0, 2.0]), tol)
    assert abs(float(mixed[0]) - 3.5) <= bound
    assert abs(float(end[0]) - 4.0) <= bound


def test_a_point_past_a_heavy_end_without_validation_extrapolates_the_last_piece() -> None:
    """With ``validate=False`` a point past the end is evaluated on the last non-empty span.

    It used to land on the empty span ``[2, 2)`` and get zeros. Oracle: on ``[1, 2)`` the active
    functions are ``N1 = (2 - x)^2 / 2``, ``N3 = (x - 1)^2`` and ``N2 = 1 - N1 - N3``, so at
    ``x = 2.5`` they are ``[0.125, -1.375, 2.25]`` (a polynomial piece extrapolated, not a basis).
    """
    space = BsplineSpace1D(_RIGHT_HEAVY, _P)
    vals, first = space.tabulate_basis(np.array([2.5]), validate=False)
    assert int(first[0]) == 1
    np.testing.assert_allclose(vals[0], [0.125, -1.375, 2.25], rtol=0.0, atol=16 * _BASIS_TOL)


def test_a_zero_divisor_in_the_band_beside_a_sub_tolerance_span_raises() -> None:
    """The derivative status stays reachable with validation on, and maps to ``ZeroDivisionError``.

    On the unsnapped knots ``[0, 0, 0, 1e-32, 1, 1, 1]`` the first span is shorter than the
    rounding of the band point ``-1e-15``, so ``(1e-32 + 1e-15) + (-1e-15 - 0)`` rounds to zero.
    """
    space = BsplineSpace1D(np.array([0.0, 0, 0, 1e-32, 1, 1, 1]), _P, snap_knots=False)
    curve = Bspline(BsplineSpace([space]), np.arange(4.0))
    with pytest.raises(ZeroDivisionError):
        curve.evaluate_derivatives(np.array([-1e-15]), [1])
