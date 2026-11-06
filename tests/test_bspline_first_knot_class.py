"""Knot vectors whose first domain knot's class continues past ``knots[degree]``.

When ``knots[degree] == knots[degree + 1]`` on a non-clamped vector (or the first class is longer
than ``degree + 1``), the first in-domain interval starts at the last knot of that class, not at
``knots[degree]``. Bézier and Lagrange extraction, ``to_beziers``, the cardinal-interval flags and
``slice`` at the domain start all assumed it started at ``knots[degree]``, and were wrong there.

Oracles. ``Bspline.evaluate`` and ``tabulate_basis`` (Cox-de Boor, a different route from the
extraction by knot insertion) against the Bézier pieces; the Bernstein and equispaced Lagrange
polynomials written out with NumPy; for the cardinal flags, the definition (interval length and
the ``2 degree - 1`` neighbouring spans) checked by hand on a uniform vector.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pytest
from numpy import typing as npt

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D


def _bernstein(p: int, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Get the degree-``p`` Bernstein polynomials at ``u``, one row per polynomial.

    Args:
        p (int): The degree.
        u (npt.NDArray[np.float64]): Points in [0, 1].

    Returns:
        npt.NDArray[np.float64]: ``(p + 1, len(u))``.
    """
    return np.array([comb(p, i) * u**i * (1.0 - u) ** (p - i) for i in range(p + 1)])


def _lagrange_equispaced(p: int, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Get the Lagrange polynomials on ``p + 1`` equispaced nodes of [0, 1] at ``u``.

    Args:
        p (int): The degree.
        u (npt.NDArray[np.float64]): Points in [0, 1].

    Returns:
        npt.NDArray[np.float64]: ``(p + 1, len(u))``.
    """
    nodes = np.linspace(0.0, 1.0, p + 1)
    rows = []
    for i in range(p + 1):
        row = np.ones_like(u)
        for j in range(p + 1):
            if j != i:
                row = row * (u - nodes[j]) / (nodes[i] - nodes[j])
        rows.append(row)
    return np.array(rows)


def _random_spaces(n: int, seed: int) -> list[BsplineSpace1D]:
    """Get non-clamped spaces whose first domain knot is repeated past ``knots[degree]``.

    The domain is [0, 1]; ``degree`` distinct ghost knots precede the first ``0`` and follow the
    last ``1``, and ``0`` is repeated 2 to ``degree`` times, so ``knots[degree + 1] == 0``.

    Args:
        n (int): Number of spaces.
        seed (int): Random seed.

    Returns:
        list[BsplineSpace1D]: The spaces.
    """
    rng = np.random.default_rng(seed)
    out: list[BsplineSpace1D] = []
    while len(out) < n:
        p = int(rng.integers(2, 5))
        nint = int(rng.integers(1, 5))
        inner = np.sort(rng.uniform(0.05, 0.95, nint - 1))
        m0 = int(rng.integers(2, p + 1))
        mi = int(rng.integers(1, p + 1))
        left = -np.cumsum(rng.uniform(0.05, 0.3, p))[::-1]
        right = 1.0 + np.cumsum(rng.uniform(0.05, 0.3, p))
        knots = np.concatenate([left, [0.0] * m0, np.repeat(inner, mi), [1.0], right])
        sp = BsplineSpace1D(knots, p)
        assert sp.knots[p] == sp.knots[p + 1] == 0.0
        out.append(sp)
    return out


def test_to_beziers_matches_evaluate_on_every_interval() -> None:
    """Each Bézier piece of ``to_beziers`` is the spline on its interval.

    Bug caught: extraction started at ``knots[degree]``, so the first piece was built from the
    wrong knots (the reviewer's example gave a constant 4 where the spline is 2.7 to 3.6).
    """
    rng = np.random.default_rng(3)
    longer = BsplineSpace1D([0.0] * 4 + [0.3, 0.7] + [1.0] * 3, 2)  # first class of degree + 2
    for sp in [longer, *_random_spaces(60, seed=1)]:
        b = Bspline(BsplineSpace([sp]), rng.standard_normal((sp.num_basis, 1)))
        breaks = sp.get_unique_knots_and_multiplicity(True)[0]
        pieces = b.to_beziers()
        u = np.linspace(0.0, 1.0, 7)
        for e in range(sp.num_intervals):
            x = breaks[e] + (breaks[e + 1] - breaks[e]) * u
            np.testing.assert_allclose(
                pieces[e].evaluate(u).ravel(), b.evaluate(x).ravel(), rtol=0.0, atol=1e-12
            )


@pytest.mark.parametrize("family", ["bezier", "lagrange"])
def test_extraction_operators_map_the_local_basis_to_the_bspline_basis(family: str) -> None:
    """``C[e] @ local = B-spline values`` on every interval, the first one included."""
    for sp in _random_spaces(40, seed=2):
        p = sp.degree
        if family == "bezier":
            ops, local = sp.tabulate_Bezier_extraction_operators(), _bernstein
        else:
            ops, local = sp.tabulate_Lagrange_extraction_operators(), _lagrange_equispaced
        breaks = sp.get_unique_knots_and_multiplicity(True)[0]
        u = np.array([0.1, 0.35, 0.8])
        for e in range(sp.num_intervals):
            x = breaks[e] + (breaks[e + 1] - breaks[e]) * u
            vals, _first = sp.tabulate_basis(x)
            np.testing.assert_allclose(ops[e] @ local(p, u), vals.T, rtol=0.0, atol=1e-12)


def test_slice_at_the_domain_start_is_the_spline_value() -> None:
    """``slice`` at the first domain knot equals ``evaluate`` there.

    Bug caught: the span at the domain start was taken as ``degree``, which with
    ``knots = [0, 0, 0, 1, 2, 2]`` at degree 1 returned the first control point (10) where the
    spline is 20.
    """
    sp = BsplineSpace1D([0.0, 0.0, 0.0, 1.0, 2.0, 2.0], 1)
    b = Bspline(BsplineSpace([sp]), np.array([[10.0], [20.0], [30.0], [40.0]]))
    assert b.evaluate(np.array([0.0])).ravel().tolist() == [20.0]
    assert np.asarray(b.slice(0, 0.0)).ravel().tolist() == [20.0]
    rng = np.random.default_rng(5)
    for sp in _random_spaces(30, seed=4):
        b = Bspline(BsplineSpace([sp]), rng.standard_normal((sp.num_basis, 1)))
        np.testing.assert_allclose(
            np.asarray(b.slice(0, 0.0)).ravel(),
            b.evaluate(np.array([0.0])).ravel(),
            rtol=0.0,
            atol=1e-12,
        )


def test_cardinal_intervals_do_not_depend_on_extra_leading_copies() -> None:
    """A uniform vector's interior intervals are cardinal however often its first knot repeats.

    Bug caught: the walk started at ``knots[degree]``, so one extra leading copy shifted every
    window and interval [1, 2] read as not cardinal.
    """
    for lead in (3, 4):
        knots = [0.0] * lead + [1.0, 2.0, 3.0, 4.0, 5.0] + [6.0] * 3
        flags = BsplineSpace1D(knots, 2).get_cardinal_intervals().tolist()
        # [0, 1] and [5, 6] touch a repeated knot; the interior ones have uniform windows.
        assert flags == [False, True, True, True, True, False], lead
