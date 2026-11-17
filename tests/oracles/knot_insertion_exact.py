"""Exact-rational knot insertion and blossoms: oracles that share nothing with the library.

- **Knot insertion** is Boehm's one-knot insertion (Piegl and Tiller, algorithm A5.1 with one
  knot), applied once per inserted knot, in :class:`fractions.Fraction` arithmetic. It is the
  *definition* of knot insertion (a convex combination of two neighbouring control points); the
  library runs the discrete B-spline recurrence of the Oslo algorithm over all the knots at once,
  so the two are different computations of one linear map. Because the control points of a B-spline
  over given knots are unique, any order of insertion gives the same result.
- **The blossom** is computed from its definition as the polar form of the polynomial piece: the
  piece on a knot span is recovered exactly from ``degree + 1`` exact point evaluations of the
  Cox-de Boor sum (:mod:`tests.oracles.bspline_exact`), written in monomials, and its polar form is
  ``sum_k c_k e_k(u) / C(p, k)`` with ``e_k`` the elementary symmetric polynomials of the arguments.
  That is the symmetric multi-affine function whose diagonal is the piece, and it uses neither the
  generalized de Boor recurrence nor the control points directly.

Every knot, control point and argument is a float, hence a dyadic rational, so these values carry
no rounding at all.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from fractions import Fraction

import numpy as np
import numpy.typing as npt

from .bspline_exact import exact, homogeneous


def boehm_insert(
    knots: Sequence[Fraction],
    degree: int,
    ctrl: Sequence[Sequence[Fraction]],
    t: Fraction,
) -> tuple[list[Fraction], list[list[Fraction]]]:
    """Insert one knot (Piegl and Tiller A5.1, ``r = 1``), exactly.

    ``k`` is the last index in ``[degree, n]`` with ``knots[k] <= t`` and ``s`` the multiplicity
    of ``t`` among ``knots[0..k]``; the new point ``i`` in ``[k - degree + 1, k - s]`` is
    ``alpha P_i + (1 - alpha) P_{i-1}`` with ``alpha = (t - u_i) / (u_{i+degree} - u_i)``.

    Args:
        knots (Sequence[Fraction]): The knot vector.
        degree (int): The degree.
        ctrl (Sequence[Sequence[Fraction]]): The control points, one row per point.
        t (Fraction): The knot to insert, inside the domain, of multiplicity below ``degree + 1``.

    Returns:
        tuple[list[Fraction], list[list[Fraction]]]: The new knots and control points.
    """
    n = len(ctrl) - 1
    p = degree
    k = p
    for j in range(p, n + 1):
        if knots[j] <= t:
            k = j
    s = sum(1 for j in range(k + 1) if knots[j] == t)
    new_knots = [*knots[: k + 1], t, *knots[k + 1 :]]
    cols = len(ctrl[0])
    new_ctrl: list[list[Fraction]] = [[Fraction(0)] * cols for _ in range(n + 2)]
    for i in range(k - p + 1):
        new_ctrl[i] = list(ctrl[i])
    for i in range(k - s, n + 1):
        new_ctrl[i + 1] = list(ctrl[i])
    for i in range(k - p + 1, k - s + 1):
        alpha = (t - knots[i]) / (knots[i + p] - knots[i])
        new_ctrl[i] = [
            alpha * a + (1 - alpha) * b for a, b in zip(ctrl[i], ctrl[i - 1], strict=True)
        ]
    return new_knots, new_ctrl


def insert_knots(
    knots: npt.ArrayLike,
    degree: int,
    ctrl: npt.NDArray[np.floating],
    new_knots: npt.ArrayLike,
) -> tuple[list[Fraction], list[list[Fraction]]]:
    """Insert several knots, one at a time, exactly.

    Args:
        knots (npt.ArrayLike): The knot vector (floats).
        degree (int): The degree.
        ctrl (npt.NDArray[np.floating]): Control points ``(n, cols)``.
        new_knots (npt.ArrayLike): The knots to insert (floats), repeats allowed.

    Returns:
        tuple[list[Fraction], list[list[Fraction]]]: The refined knots and control points.
    """
    u = [exact(k) for k in np.asarray(knots).tolist()]
    rows = [[exact(v) for v in row] for row in np.atleast_2d(ctrl).tolist()]
    for t in np.asarray(new_knots).tolist():
        u, rows = boehm_insert(u, degree, rows, exact(t))
    return u, rows


def oslo_matrix(
    knots: npt.ArrayLike, degree: int, new_knots: npt.ArrayLike
) -> list[list[Fraction]]:
    """Get the refinement matrix: the insertion applied to the columns of the identity.

    Args:
        knots (npt.ArrayLike): The original knot vector.
        degree (int): The degree.
        new_knots (npt.ArrayLike): The knots to insert.

    Returns:
        list[list[Fraction]]: The matrix ``alpha`` of shape ``(m + 1, n + 1)``.
    """
    n1 = len(np.asarray(knots)) - degree - 1
    eye = np.eye(n1)
    return insert_knots(knots, degree, eye, new_knots)[1]


def _span_of(knots: Sequence[Fraction], degree: int, x: Fraction) -> int:
    """Get the span a point is evaluated on: the library's rule, as :class:`ExactBasis` has it."""
    last = len(knots) - degree - 2
    return next((j for j in range(degree, last) if x < knots[j + 1]), last)


def blossom(
    knots: npt.ArrayLike,
    degree: int,
    ctrl: npt.NDArray[np.floating],
    u_values: Sequence[float],
) -> list[Fraction]:
    """Get the blossom of a 1D B-spline from its definition as a polar form, exactly.

    The polynomial piece on the span of the largest argument is interpolated exactly from
    ``degree + 1`` points inside the span (Vandermonde system in fractions), and its polar form
    is evaluated at the arguments (which need not lie in the span).

    Args:
        knots (npt.ArrayLike): The knot vector.
        degree (int): The degree ``p``.
        ctrl (npt.NDArray[np.floating]): Control points ``(n, rank)``.
        u_values (Sequence[float]): The ``p`` arguments.

    Returns:
        list[Fraction]: The ``rank`` coordinates of ``f[u_1, ..., u_p]``.
    """
    p = degree
    u = [exact(k) for k in np.asarray(knots).tolist()]
    cp = np.atleast_2d(ctrl)
    rank = cp.shape[1]
    if p == 0:
        return [exact(v) for v in cp[-1].tolist()]
    args = [exact(v) for v in u_values]
    k = _span_of(u, p, max(args))
    lo, hi = u[k], u[k + 1]
    # p + 1 points strictly inside the span, at sixteenths of it: floats when the knots are
    # multiples of a small power of two (checked), so that they are evaluated exactly.
    xs = [lo + (hi - lo) * Fraction(i + 1, 16) for i in range(p + 1)]
    if any(Fraction(float(x)) != x for x in xs):
        raise ValueError("the knots need to be dyadic with few bits for the exact blossom")
    # Monomial coefficients c_0..c_p of each coordinate: solve the exact Vandermonde system.
    values = [homogeneous([knots], [p], cp, [float(x)], [0]) for x in xs]
    vander = [[x**j for j in range(p + 1)] for x in xs]
    coeffs = _solve(vander, values, rank)
    # Elementary symmetric polynomials of the arguments.
    sigma = [Fraction(0)] * (p + 1)
    for k_sub in range(p + 1):
        sigma[k_sub] = sum(
            (math.prod(c, start=Fraction(1)) for c in itertools.combinations(args, k_sub)),
            start=Fraction(0),
        )
    return [
        sum((coeffs[j][c] * sigma[j] / math.comb(p, j) for j in range(p + 1)), start=Fraction(0))
        for c in range(rank)
    ]


def _solve(a: list[list[Fraction]], b: list[list[Fraction]], cols: int) -> list[list[Fraction]]:
    """Solve ``A X = B`` exactly by Gauss-Jordan elimination (``A`` square, nonsingular).

    Args:
        a (list[list[Fraction]]): The matrix.
        b (list[list[Fraction]]): The right-hand sides, one row per equation.
        cols (int): The number of right-hand sides.

    Returns:
        list[list[Fraction]]: The solution, one row per unknown.
    """
    n = len(a)
    m = [[*a[i], *b[i]] for i in range(n)]
    for col in range(n):
        piv = next(r for r in range(col, n) if m[r][col] != 0)
        m[col], m[piv] = m[piv], m[col]
        scale = m[col][col]
        m[col] = [v / scale for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col], strict=True)]
    return [row[n : n + cols] for row in m]


__all__ = ["blossom", "boehm_insert", "insert_knots", "oslo_matrix"]
