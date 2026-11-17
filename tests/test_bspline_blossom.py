"""Tests for B-spline blossom (polar form) evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pantr import _testing
from pantr._testing import blossom_span as _blossom_span
from pantr._testing import evaluate_blossom_1d as _evaluate_blossom_1d
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from tests.oracles import bspline_exact, knot_insertion_exact

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eval_blossom(
    knots: list[float],
    degree: int,
    ctrl: list[float] | list[list[float]],
    u_values: list[float],
) -> npt.NDArray[Any]:
    """Thin wrapper: build numpy arrays and call _evaluate_blossom_1d."""
    kv = np.array(knots, dtype=np.float64)
    cp = np.array(ctrl, dtype=np.float64)
    if cp.ndim == 1:
        cp = cp[:, np.newaxis]
    u = np.array(u_values, dtype=np.float64)
    tol = float(np.finfo(np.float64).eps * max(1.0, float(np.abs(kv).max())) * 64)
    return _evaluate_blossom_1d(kv, degree, cp, u, tol)


def _eval_spline(
    knots: list[float],
    degree: int,
    ctrl: list[float],
    t: float,
) -> float:
    """Evaluate a 1D B-spline at a single point via de Boor."""
    kv = np.array(knots, dtype=np.float64)
    cp = np.array(ctrl, dtype=np.float64)[:, np.newaxis]
    space = BsplineSpace([BsplineSpace1D(kv, degree)])
    f = Bspline(space, cp)
    return float(f.evaluate(np.array([t])))


# ---------------------------------------------------------------------------
# Diagonal property: blossom(t, t, ..., t) == f(t)
# ---------------------------------------------------------------------------


class TestBlossomDiagonal:
    """Test that the diagonal of the blossom equals the B-spline evaluation."""

    def test_degree2_single_element(self) -> None:
        """Diagonal property for a degree-2 single-element B-spline."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        ctrl = [1.0, 2.0, 3.0]

        for t in np.linspace(0.0, 1.0, 9):
            blossom_val = _eval_blossom(knots, 2, ctrl, [t, t])
            direct_val = _eval_spline(knots, 2, ctrl, t)
            np.testing.assert_allclose(blossom_val[0], direct_val, atol=1e-12, err_msg=f"t={t}")

    def test_degree3_two_elements(self) -> None:
        """Diagonal property for a cubic B-spline with one interior knot."""
        knots = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]
        ctrl = [1.0, 0.5, 2.0, 1.5, 0.5]

        for t in np.linspace(0.0, 1.0, 11):
            blossom_val = _eval_blossom(knots, 3, ctrl, [t, t, t])
            direct_val = _eval_spline(knots, 3, ctrl, t)
            np.testing.assert_allclose(blossom_val[0], direct_val, atol=1e-12, err_msg=f"t={t}")

    def test_degree1_linear(self) -> None:
        """Diagonal property for a degree-1 B-spline (piecewise linear)."""
        knots = [0.0, 0.0, 0.5, 1.0, 1.0]
        ctrl = [1.0, 3.0, 2.0]

        for t in np.linspace(0.0, 1.0, 9):
            blossom_val = _eval_blossom(knots, 1, ctrl, [t])
            direct_val = _eval_spline(knots, 1, ctrl, t)
            np.testing.assert_allclose(blossom_val[0], direct_val, atol=1e-12, err_msg=f"t={t}")

    def test_vector_rank(self) -> None:
        """Diagonal property holds for each component of a vector-valued spline."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        ctrl = [[1.0, 2.0], [3.0, 0.5], [2.0, 1.5]]

        for t in np.linspace(0.0, 1.0, 7):
            blossom_val = _eval_blossom(knots, 2, ctrl, [t, t])
            # Evaluate each component separately
            val0 = _eval_spline(knots, 2, [row[0] for row in ctrl], t)
            val1 = _eval_spline(knots, 2, [row[1] for row in ctrl], t)
            np.testing.assert_allclose(blossom_val[0], val0, atol=1e-12)
            np.testing.assert_allclose(blossom_val[1], val1, atol=1e-12)


# ---------------------------------------------------------------------------
# Symmetry: blossom(u1, u2) == blossom(u2, u1)
# ---------------------------------------------------------------------------


class TestBlossomSymmetry:
    """Test that the blossom is symmetric in its arguments."""

    def test_degree2_symmetry(self) -> None:
        """Blossom is symmetric for degree-2 spline."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        ctrl = [1.0, 2.0, 0.5, 3.0]

        for u1, u2 in [(0.1, 0.4), (0.2, 0.8), (0.5, 0.5), (0.0, 1.0)]:
            b12 = _eval_blossom(knots, 2, ctrl, [u1, u2])
            b21 = _eval_blossom(knots, 2, ctrl, [u2, u1])
            np.testing.assert_allclose(b12, b21, atol=1e-12)

    def test_degree3_symmetry(self) -> None:
        """Blossom is symmetric for degree-3 spline with two elements."""
        knots = [0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0, 1.0]
        ctrl = [1.0, 0.5, 2.0, 1.5, 3.0, 0.5]

        u1, u2, u3 = 0.1, 0.5, 0.8
        orderings = [
            [u1, u2, u3],
            [u1, u3, u2],
            [u2, u1, u3],
            [u2, u3, u1],
            [u3, u1, u2],
            [u3, u2, u1],
        ]
        results = [_eval_blossom(knots, 3, ctrl, perm) for perm in orderings]
        for res in results[1:]:
            np.testing.assert_allclose(res, results[0], atol=1e-12)


# ---------------------------------------------------------------------------
# Control point recovery: blossom(t_{i+1}, ..., t_{i+p}) == P_i
# ---------------------------------------------------------------------------


class TestBlossomControlPointRecovery:
    """Test that the blossom at consecutive interior knots recovers control points."""

    def test_degree2_knot_values(self) -> None:
        """blossom(t_{i+1}, t_{i+2}) == P[i] for degree-2 B-spline."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        ctrl = [1.0, 2.0, 0.5, 3.0]
        n = len(ctrl)
        p = 2

        for i in range(n):
            u_vals = [float(knots[i + r]) for r in range(1, p + 1)]
            blossom_val = _eval_blossom(knots, p, ctrl, u_vals)
            np.testing.assert_allclose(blossom_val[0], ctrl[i], atol=1e-12, err_msg=f"i={i}")

    def test_degree3_knot_values(self) -> None:
        """blossom(t_{i+1}, t_{i+2}, t_{i+3}) == P[i] for degree-3 B-spline."""
        knots = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]
        ctrl = [1.0, 0.5, 2.0, 1.5, 0.5]
        n = len(ctrl)
        p = 3

        for i in range(n):
            u_vals = [float(knots[i + r]) for r in range(1, p + 1)]
            blossom_val = _eval_blossom(knots, p, ctrl, u_vals)
            np.testing.assert_allclose(blossom_val[0], ctrl[i], atol=1e-12, err_msg=f"i={i}")


# ---------------------------------------------------------------------------
# Input validation errors
# ---------------------------------------------------------------------------


class TestBlossomValidation:
    """Tests for input validation in _evaluate_blossom_1d."""

    def test_wrong_u_values_length(self) -> None:
        """Raises ValueError when len(u_values) != degree."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        ctrl = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        u = np.array([0.5], dtype=np.float64)  # degree=2 requires 2 values
        with pytest.raises(ValueError, match="length"):
            _evaluate_blossom_1d(
                np.array(knots, dtype=np.float64),
                2,
                ctrl,
                u,
                1e-12,
            )

    def test_u_values_out_of_domain(self) -> None:
        """Raises ValueError when a u value lies outside the domain."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        ctrl = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        u = np.array([0.5, 1.5], dtype=np.float64)  # 1.5 > domain [0, 1]
        with pytest.raises(ValueError, match="outside domain"):
            _evaluate_blossom_1d(
                np.array(knots, dtype=np.float64),
                2,
                ctrl,
                u,
                1e-12,
            )


# ---------------------------------------------------------------------------
# Knot-span search: the binary search must reproduce the linear scan
# ---------------------------------------------------------------------------


def _linear_scan_span(knots: npt.NDArray[Any], u_last: float, n: int, degree: int) -> int:
    """The ``O(n)`` scan ``_blossom_span`` replaced, kept here as the reference oracle.

    Verbatim from the kernel before the change, including its ``k = n`` fall-through when
    no half-open span contains ``u_last``. A consistency check (mirror M5): it is the defining
    scan the binary search must reproduce, not an independent computation of the blossom; the
    independent oracle is :class:`TestBlossomAgainstThePolarForm` below.
    """
    k = n
    for idx in range(n + degree):
        if knots[idx] <= u_last < knots[idx + 1]:
            k = idx
            break
    return k


def _random_open_knots(
    degree: int, n_interior: int, seed: int, *, repeat_first: bool = False
) -> npt.NDArray[np.float64]:
    """Return a clamped knot vector on ``[0, 1]`` with random interior knots."""
    rng = np.random.default_rng(seed)
    interior = np.sort(rng.uniform(0.0, 1.0, size=n_interior))
    if repeat_first and n_interior > 1:
        interior[1] = interior[0]
    return np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)]).astype(np.float64)


class TestBlossomSpan:
    """The binary span search agrees with the linear scan on every valid input."""

    @pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("n_interior", [0, 1, 2, 5])
    @pytest.mark.parametrize("repeat_first", [False, True])
    def test_matches_the_linear_scan(
        self, degree: int, n_interior: int, repeat_first: bool
    ) -> None:
        """Identical span indices, interior multiplicities and endpoints included.

        The probe set covers the interior, both domain endpoints, every interior knot
        exactly, and the band just outside the domain that Layer 2 still accepts within
        its tolerance.
        """
        knots = _random_open_knots(
            degree, n_interior, seed=100 * degree + n_interior, repeat_first=repeat_first
        )
        n = knots.shape[0] - degree - 2
        interior = knots[degree + 1 : knots.shape[0] - degree - 1]
        rng = np.random.default_rng(7)
        probes = [
            *rng.uniform(0.0, 1.0, size=30).tolist(),
            0.0,
            1.0,
            *interior.tolist(),
            -1e-12,
            1.0 + 1e-12,
        ]
        for u in probes:
            assert int(_blossom_span(knots, float(u), n)) == _linear_scan_span(
                knots, float(u), n, degree
            ), f"degree={degree} u={u!r} knots={knots.tolist()}"

    def test_below_the_first_knot_gives_the_rightmost_span(self) -> None:
        """Preserved behaviour: an under-range parameter falls through to span ``n``.

        Reachable through Layer 2, which accepts parameters up to its tolerance below the
        domain start. The scan returned ``n`` here because no span matched; the binary
        search would give ``-1`` without the guard.
        """
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        n = knots.shape[0] - 4
        assert int(_blossom_span(knots, -1e-9, n)) == n
        assert _linear_scan_span(knots, -1e-9, n, 2) == n

    def test_at_and_beyond_the_last_knot_clamps(self) -> None:
        """The right domain endpoint and anything past it both give span ``n``."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        n = knots.shape[0] - 4
        assert int(_blossom_span(knots, 1.0, n)) == n
        assert int(_blossom_span(knots, 2.0, n)) == n

    def test_non_clamped_knots_stay_in_range(self) -> None:
        """On a non-clamped vector a parameter past the domain still yields a valid index.

        Here the linear scan matched a span above ``n`` and the caller then read past the
        control points; the clamp removes that. Inputs like this are outside the Layer 2
        contract either way.
        """
        knots = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        n = knots.shape[0] - 4  # degree 2 => 3 control points
        assert int(_blossom_span(knots, 4.5, n)) == n
        assert _linear_scan_span(knots, 4.5, n, 2) > n

    @pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
    def test_diagonal_property_with_interior_multiplicities(self, degree: int) -> None:
        """``f[t, ..., t] == f(t)`` on knots with a repeated interior knot.

        The end-to-end guard on the span change: degrees 1 to 5, a non-uniform knot vector
        carrying an interior multiplicity, and ``t`` at both endpoints, exactly on every
        interior knot, and in between.
        """
        knots = _random_open_knots(degree, 4, seed=degree, repeat_first=True)
        space = BsplineSpace1D(knots, degree)
        rng = np.random.default_rng(degree + 50)
        ctrl = rng.uniform(-1.0, 1.0, size=(space.num_basis, 2))
        curve = Bspline(BsplineSpace([space]), np.ascontiguousarray(ctrl))
        interior = knots[degree + 1 : knots.shape[0] - degree - 1]
        tol = 64.0 * float(np.finfo(np.float64).eps)

        for t in [0.0, 1.0, *interior.tolist(), *rng.uniform(0.0, 1.0, size=10).tolist()]:
            blossom = _evaluate_blossom_1d(knots, degree, ctrl, np.full(degree, float(t)), tol)
            direct = np.asarray(curve.evaluate(np.array([float(t)]))).reshape(2)
            np.testing.assert_allclose(
                np.asarray(blossom).reshape(2), direct, atol=tol, rtol=0.0, err_msg=f"t={t}"
            )


# ---------------------------------------------------------------------------
# Independent oracle: the polar form of the polynomial piece, exact-rational
# ---------------------------------------------------------------------------


_BLOSSOM_KNOTS: list[tuple[int, list[float]]] = [
    (1, [0.0, 0.0, 0.5, 1.0, 1.0]),
    (2, [0.0, 0.0, 0.0, 0.25, 0.75, 1.0, 1.0, 1.0]),
    (3, [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]),
    (4, [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]),
    (5, [0.0] * 6 + [0.125, 0.5, 0.5, 0.5] + [1.0] * 6),
]
"""``(degree, knots)`` pairs: dyadic, clamped, interior multiplicities up to ``degree - 2``."""


class TestBlossomAgainstThePolarForm:
    """The blossom against its definition, and the properties that define it."""

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    @pytest.mark.parametrize(("degree", "knots"), _BLOSSOM_KNOTS)
    def test_equals_the_exact_polar_form_of_the_piece(
        self, degree: int, knots: list[float], dtype: type
    ) -> None:
        """The value is the polar form of the polynomial piece, evaluated exactly.

        Oracle: :func:`tests.oracles.knot_insertion_exact.blossom`, which recovers the piece on
        the span of the largest argument from exact Cox-de Boor values and evaluates its polar
        form ``sum_k c_k e_k(u) / C(p, k)`` (no de Boor recurrence). Tolerance: ``degree`` levels
        of a convex combination, each with the weight (two roundings), its complement and the two
        products and a sum, within ``6 u`` of the exact level; ``8 degree u max|P|`` keeps a margin
        (``u = eps / 2`` of the storage type; the float32 kernel rounds to float32 at each level).
        """
        rng = np.random.default_rng(degree)
        n = len(knots) - degree - 1
        ctrl: npt.NDArray[Any] = (rng.integers(-64, 65, (n, 2)) / 64.0).astype(dtype)
        kv: npt.NDArray[Any] = np.asarray(knots, dtype=dtype)
        u = float(np.finfo(dtype).eps) / 2
        tol = 8.0 * degree * u * float(np.max(np.abs(ctrl)))
        for _ in range(20):
            args = rng.integers(0, 65, degree) / 64.0
            blossom_fn = (
                _testing.evaluate_blossom_1d_f32 if dtype == np.float32 else _evaluate_blossom_1d
            )
            got = blossom_fn(kv, degree, ctrl, args.astype(dtype), 1e-6)
            want = knot_insertion_exact.blossom(kv, degree, ctrl.astype(np.float64), args.tolist())
            assert all(
                abs(bspline_exact.exact(g) - w) <= tol
                for g, w in zip(got.tolist(), want, strict=True)
            ), f"args={args.tolist()}"

    @pytest.mark.parametrize(("degree", "knots"), _BLOSSOM_KNOTS)
    def test_diagonal_is_the_exactly_evaluated_curve(self, degree: int, knots: list[float]) -> None:
        """``f[t, ..., t]`` equals the Cox-de Boor sum at ``t``, in fractions.

        Oracle: :mod:`tests.oracles.bspline_exact`. Tolerance: ``8 degree u max|P|`` as above.
        """
        rng = np.random.default_rng(10 + degree)
        n = len(knots) - degree - 1
        ctrl = rng.integers(-64, 65, (n, 2)) / 64.0
        kv = np.asarray(knots)
        tol = 8.0 * degree * float(np.finfo(np.float64).eps) / 2 * float(np.max(np.abs(ctrl)))
        for t in [0.0, 0.25, 0.5, 1.0, *(rng.integers(0, 65, 8) / 64.0).tolist()]:
            got = _evaluate_blossom_1d(kv, degree, ctrl, np.full(degree, t), 1e-12)
            want = bspline_exact.homogeneous([kv], [degree], ctrl, [t], [0])
            assert all(
                abs(bspline_exact.exact(g) - w) <= tol
                for g, w in zip(got.tolist(), want, strict=True)
            ), f"t={t}"

    @pytest.mark.parametrize(("degree", "knots"), _BLOSSOM_KNOTS)
    def test_symmetric_and_multi_affine(self, degree: int, knots: list[float]) -> None:
        """Every ordering of the arguments gives one value; the value is affine in each argument.

        Oracle: the two defining properties of a blossom, checked on the computed values:
        invariance under permutation, and ``f[(1 - s) a + s b, rest] = (1 - s) f[a, rest] + s
        f[b, rest]`` for an exact ``s = 1/2``, with ``a`` and ``b`` in one span so that the
        affine combination stays in the piece (the other arguments are fixed in one span).
        Tolerance: ``16 degree u max|P|`` (two blossoms and their combination).
        """
        rng = np.random.default_rng(20 + degree)
        n = len(knots) - degree - 1
        ctrl = rng.integers(-64, 65, (n, 2)) / 64.0
        kv = np.asarray(knots)
        tol = 16.0 * degree * float(np.finfo(np.float64).eps) / 2 * float(np.max(np.abs(ctrl)))
        args = (rng.integers(0, 65, degree) / 64.0).tolist()
        base = _evaluate_blossom_1d(kv, degree, ctrl, np.array(args), 1e-12)
        for perm in [args[::-1], sorted(args), [*args[1:], args[0]]]:
            np.testing.assert_allclose(
                _evaluate_blossom_1d(kv, degree, ctrl, np.array(perm), 1e-12),
                base,
                rtol=0.0,
                atol=tol,
            )
        # Affinity in the first argument: with the other arguments in [0.125, 0.1875] the largest
        # argument stays in one span (the knot 0.125 of the last case is the left end of that
        # span), so the three values are blossoms of the same piece.
        lo_arg = 0.0625
        hi_arg = 0.1875
        rest = [0.125 + 0.015625 * i for i in range(degree - 1)]
        f_a = _evaluate_blossom_1d(kv, degree, ctrl, np.array([lo_arg, *rest]), 1e-12)
        f_b = _evaluate_blossom_1d(kv, degree, ctrl, np.array([hi_arg, *rest]), 1e-12)
        f_m = _evaluate_blossom_1d(
            kv, degree, ctrl, np.array([0.5 * (lo_arg + hi_arg), *rest]), 1e-12
        )
        np.testing.assert_allclose(f_m, 0.5 * (f_a + f_b), rtol=0.0, atol=tol)
