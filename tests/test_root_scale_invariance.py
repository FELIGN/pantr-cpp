"""Scale invariance of :func:`pantr.bezier.find_roots` and the meaning of its ``tol``.

Oracles. Encoded truth: Bernstein coefficients of products of linear factors with rational
roots, converted in exact rational arithmetic (``fractions``, the textbook map
``b_i = sum_{j <= i} a_j C(i, j) / C(n, j)``) and rounded once. Metamorphic: multiplying the
coefficients by a power of two is exact away from under- and overflow and leaves the roots
alone, so the result must not move by one bit.

The bug these pin: the value-frame bounds of the root finder (its zero test, the merge radius,
two derivative tests) were absolute, so a degree-6 polynomial with six simple roots returned
twelve roots at ``1e-10`` times its coefficients, three at ``1e14`` and one at ``1e15``, and an
explicit ``tol`` acted as a floor on the coefficient values.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb

import numpy as np
import pytest
from numpy import typing as npt

from pantr.bezier import Bezier, find_monotone_root, find_roots

_DEG6_ROOTS = [Fraction(k, 10) for k in range(1, 7)]
"""Six simple roots, 1/10 apart: the degree-6 case, which takes Bezier clipping and the merge."""


_Coeffs = npt.NDArray[np.float32 | np.float64]
"""Coefficients of either storage type."""


def _bernstein(roots: list[Fraction], dtype: type[np.floating] = np.float64) -> _Coeffs:
    """Get the Bernstein coefficients of ``prod (t - r)``, scaled to ``max |c|`` in [0.5, 1).

    Args:
        roots (list[Fraction]): The roots.
        dtype (type[np.floating]): The storage type.

    Returns:
        _Coeffs: ``len(roots) + 1`` coefficients.
    """
    p = [Fraction(1)]
    for r in roots:
        q = [Fraction(0)] * (len(p) + 1)
        for i, a in enumerate(p):
            q[i + 1] += a
            q[i] -= r * a
        p = q
    n = len(p) - 1
    b = np.array(
        [
            float(sum(Fraction(comb(i, j), comb(n, j)) * p[j] for j in range(i + 1)))
            for i in range(n + 1)
        ]
    )
    _, e = np.frexp(np.max(np.abs(b)))
    out: _Coeffs = np.ldexp(b, -e).astype(dtype)
    return out


def _batch_row(c: _Coeffs, tol: float | None = None) -> npt.NDArray[np.float64]:
    """Get the roots of ``c`` as row 0 of a one-row batch.

    Args:
        c (_Coeffs): Coefficients.
        tol (float | None): The tolerance.

    Returns:
        npt.NDArray[np.float64]: The row's roots.
    """
    roots, counts = find_roots([Bezier(c)], tol=tol)
    return roots[0, : int(counts[0])]


@pytest.mark.parametrize("scale", [1e-15, 1e-10, 1e-5, 1.0, 1e5, 1e14, 1e15, 1e100])
def test_degree6_roots_at_any_value_scale(scale: float) -> None:
    """Six simple roots are found once each, single and batch, whatever the coefficients' scale.

    Bug caught: an absolute zero floor (twelve roots at ``1e-10``, none at ``1e-15``) or a merge
    radius in coefficient units (three roots at ``1e14``, one at ``1e15``).
    """
    c = _bernstein(_DEG6_ROOTS) * scale
    truth = np.array([float(r) for r in _DEG6_ROOTS])
    for got in (find_roots(Bezier(c)), _batch_row(c)):
        assert got.shape == (6,)
        np.testing.assert_allclose(got, truth, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("tol", [None, 1e-9])
@pytest.mark.parametrize("batch", [False, True])
def test_power_of_two_scaling_leaves_the_roots_bitwise(tol: float | None, batch: bool) -> None:
    """``c`` and ``2^k c`` give the same bits for every ``k`` in [-900, 900].

    Covers the clipping path (degree 6, and degree 6 with a double root) and the Yuksel path
    (degree 2), at the default and at an explicit ``tol``.
    """
    cases = [
        _DEG6_ROOTS,
        [Fraction(3, 10), Fraction(7, 10)],
        [Fraction(1, 10), Fraction(2, 5), Fraction(2, 5), Fraction(1, 2), Fraction(3, 5)],
    ]
    for roots in cases:
        c = _bernstein(roots)

        def run(x: _Coeffs) -> npt.NDArray[np.float64]:
            return _batch_row(x, tol) if batch else find_roots(Bezier(x), tol=tol)

        ref = run(c)
        assert ref.size > 0
        differing = [k for k in range(-900, 901) if not np.array_equal(run(np.ldexp(c, k)), ref)]
        assert differing == [], f"degree {len(roots)}: differs at 2^k for k in {differing[:10]}"


def test_power_of_two_scaling_float32() -> None:
    """``float32`` coefficients: the same bits for ``c`` and ``2^k c`` over ``float32``'s range."""
    c = _bernstein(_DEG6_ROOTS, np.float32)
    ref = find_roots(Bezier(c))
    ref_batch = _batch_row(c)
    assert ref.shape == (6,)  # accuracy in float32 is the C++ suite's encoded-truth test
    for k in range(-100, 101):
        scaled = np.ldexp(c, k).astype(np.float32)
        assert np.array_equal(find_roots(Bezier(scaled)), ref), k
        assert np.array_equal(_batch_row(scaled), ref_batch), k


def test_batch_rows_are_scaled_independently() -> None:
    """Each row of a batch is judged against its own largest coefficient.

    Bug caught: one value floor for the whole batch, which solves a row at ``2^-60`` against a
    floor sized for the row at ``2^60``.
    """
    c = _bernstein(_DEG6_ROOTS)
    ref = find_roots(Bezier(c))
    ks = [-900, -60, 0, 60, 900]
    roots, counts = find_roots([Bezier(np.ldexp(c, k)) for k in ks])
    assert counts.tolist() == [6] * len(ks)
    for i in range(len(ks)):
        assert np.array_equal(roots[i, :6], ref)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("tol", [None, 1e-3])
def test_zero_polynomial_has_no_roots(dtype: type[np.floating], tol: float | None) -> None:
    """The zero polynomial returns no roots, single and batch."""
    for n1 in (2, 5, 9):
        c = np.zeros(n1, dtype=dtype)
        assert find_roots(Bezier(c), tol=tol).size == 0
        roots, counts = find_roots([Bezier(c), Bezier(c)], tol=tol)
        assert counts.tolist() == [0, 0]
        assert np.isnan(roots).all()


def test_explicit_tol_is_not_a_floor_on_coefficient_values() -> None:
    """``tol`` bounds the parametric error only: small coefficients do not read as zero.

    Bug caught: the user's ``tol`` passed on as an absolute value floor, which returned no root
    for a polynomial whose coefficients are all below ``tol``.
    """
    c = _bernstein([Fraction(3, 10), Fraction(7, 10)]) * 1e-4
    for got in (find_roots(Bezier(c), tol=1e-3), _batch_row(c, 1e-3)):
        assert got.shape == (2,)
        np.testing.assert_allclose(got, [0.3, 0.7], rtol=0.0, atol=1e-3)


@pytest.mark.parametrize("tol", [np.inf, -np.inf, np.nan])
def test_non_finite_tol_raises(tol: float) -> None:
    """A non-finite ``tol`` raises ``ValueError`` (``inf`` used to return no roots, silently)."""
    b = Bezier(_bernstein(_DEG6_ROOTS))
    with pytest.raises(ValueError, match="tol must be positive and finite"):
        find_roots(b, tol=tol)
    with pytest.raises(ValueError, match="tol must be positive and finite"):
        find_roots([b, b], tol=tol)
    with pytest.raises(ValueError, match="tol must be positive and finite"):
        find_monotone_root(Bezier([1.0, -1.0]), tol=tol)
