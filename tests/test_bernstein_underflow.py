"""Regression tests for Bernstein ratio-recurrence underflow near u=1 (issue #258).

The O(p) ratio recurrence of the C++ Bernstein tabulation
(``include/pantr/basis/bernstein.hpp``, reached here through ``pantr._testing``)
and the fused kernel behind ``Bezier.evaluate`` (1D) now branch on ``u > 0.5`` and run from
whichever endpoint keeps the seed term (``(1-u)^p`` or ``u^p``) bounded below
by ``0.5^p``, using the symmetry ``B_i,p(u) = B_{p-i},p(1-u)``. Before the
fix, the forward-only recurrence seeded from ``(1-u)^p`` underflowed to exact
zero for ``u`` close enough to 1 at high degree, and every subsequent term (a
positive multiple of the previous one) stayed zero: ``sum_i B_i(u)`` collapsed
to 0 instead of 1.

This module covers:
    - Partition of unity across degrees and points spanning both recurrence
      branches and their shared midpoint (float32 and float64).
    - Agreement with an independent O(p^2) reference (corner-cutting
      "AllBernstein", Piegl & Tiller A1.3), which never underflows because
      every intermediate value is a convex combination weighted by ``u`` and
      ``1-u`` rather than a ratio that can blow up.
    - Symmetry between the forward branch (evaluated at ``u <= 0.5``) and the
      mirrored branch (evaluated at ``1-u > 0.5``).
    - Consistency between the 0th row of the derivative kernel
      (``bernstein_derivatives_point``, an unaffected O(p^2) ``ndu``-table
      recurrence) and the fixed value kernel.
    - An end-to-end guard for the fused Bezier-evaluation kernel
      (behind ``Bezier.evaluate``) against de Casteljau evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pantr import _testing
from pantr._numba_compat import wait_for_jit_warmup
from pantr.basis import tabulate_bernstein_1d
from pantr.bezier import Bezier

_MAX_SAFE_DEGREE_NO_MIRROR_FLOAT64 = _testing.BERNSTEIN_MAX_UNMIRRORED_DEGREE_FLOAT64
_MAX_SAFE_DEGREE_NO_MIRROR_FLOAT32 = _testing.BERNSTEIN_MAX_UNMIRRORED_DEGREE_FLOAT32

# This module calls parallel=True Numba kernels (the fused Bezier ones) directly (bypassing the
# public API) very early in the test session. Numba's default threading
# layer cannot safely compile the same kernel concurrently from two threads,
# so wait for pantr's background warmup thread to finish first — otherwise it
# can race with these direct calls and crash the interpreter (see
# `pantr/__init__.py`'s `_async_warmup` and `_numba_compat.wait_for_jit_warmup`).
wait_for_jit_warmup()

DEGREES = [2, 5, 10, 20, 30, 64]

_EPS64 = float(np.finfo(np.float64).eps)
_EPS32 = float(np.finfo(np.float32).eps)

# Points spanning both recurrence branches and their shared midpoint, plus the
# near-1 region where the pre-fix forward recurrence underflows.
_U64 = [0.0, 1e-17, 0.25, 0.5 - _EPS64, 0.5, 0.5 + _EPS64, 0.75, 1.0 - 1e-8, 1.0 - 1e-16, 1.0]
_U32 = [0.0, 1e-7, 0.25, 0.5 - _EPS32, 0.5, 0.5 + _EPS32, 0.75, 1.0 - 1e-4, 1.0 - _EPS32, 1.0]


_Float32Or64 = type[np.float32] | type[np.float64]
"""Concrete float scalar types accepted by this module's test helpers.

Narrower than `npt.DTypeLike`: calling the type directly (e.g. ``dtype(u)``)
gives mypy a single concrete overload instead of the ambiguous ``np.void``
fallback that ``np.dtype(some_DTypeLike).type(...)`` resolves to.
"""


def _eval_bernstein_row(n: int, u: float, dtype: _Float32Or64) -> npt.NDArray[np.floating[Any]]:
    """Evaluate the mirrored ratio recurrence at one point, in ``dtype``."""
    row = (
        _testing.bernstein_row_mirrored_f32
        if dtype is np.float32
        else _testing.bernstein_row_mirrored
    )
    result: npt.NDArray[np.floating[Any]] = row(n, float(dtype(u)))
    return result


def _eval_unmirrored_row(
    n: int, u: float, dtype: _Float32Or64 = np.float64
) -> npt.NDArray[np.floating[Any]]:
    """Evaluate the plain forward (unmirrored) recurrence at one point, in ``dtype``."""
    row = (
        _testing.bernstein_row_unmirrored_f32
        if dtype is np.float32
        else _testing.bernstein_row_unmirrored
    )
    result: npt.NDArray[np.floating[Any]] = row(n, float(dtype(u)))
    return result


def _all_bernstein_reference(
    n: int, u: float, dtype: _Float32Or64 = np.float64
) -> npt.NDArray[np.floating[Any]]:
    """Evaluate all degree-n Bernstein basis functions via corner-cutting (A1.3).

    Independent O(n^2) reference (Piegl & Tiller, "AllBernstein"): every
    intermediate value is a convex combination of already-computed entries
    weighted by ``u`` and ``1 - u``, both in ``[0, 1]``, so there is no ratio
    blow-up and hence no risk of the total-flush failure mode this module
    guards against.
    """
    one = dtype(1.0)
    uu = dtype(u)
    u1 = one - uu

    row = np.zeros(n + 1, dtype=dtype)
    row[0] = one
    for j in range(1, n + 1):
        saved = dtype(0.0)
        for k in range(j):
            temp = row[k]
            row[k] = saved + u1 * temp
            saved = uu * temp
        row[j] = saved
    return row


class TestPartitionOfUnity:
    """Partition of unity must hold across both recurrence branches."""

    @pytest.mark.parametrize("u", _U64, ids=lambda u: f"{u:.17g}")
    @pytest.mark.parametrize("degree", DEGREES)
    def test_float64(self, degree: int, u: float) -> None:
        """Sum of Bernstein basis values stays within 8*p*eps of 1 (float64).

        This is the fixed version of the regression covered by issue #258:
        the (degree=30, u=1-1e-16) cell used to fail before the mirrored
        recurrence was introduced.
        """
        row = _eval_bernstein_row(degree, u, np.float64)
        assert abs(float(row.sum()) - 1.0) <= 8 * degree * _EPS64

    @pytest.mark.parametrize("u", _U32, ids=lambda u: f"{u:.9g}")
    @pytest.mark.parametrize("degree", DEGREES)
    def test_float32(self, degree: int, u: float) -> None:
        """Sum of Bernstein basis values stays within 8*p*eps of 1 (float32)."""
        row = _eval_bernstein_row(degree, u, np.float32)
        assert abs(float(row.sum()) - 1.0) <= 8 * degree * _EPS32


class TestReferenceCrossCheck:
    """Compare against the independent O(p^2) AllBernstein reference."""

    @pytest.mark.parametrize("degree", DEGREES)
    def test_matches_all_bernstein_reference(self, degree: int) -> None:
        """Kernel values match the reference over a grid biased toward u=1.

        The grid mixes uniform random points with a geometric sequence
        approaching both endpoints, since uniform sampling in [0, 1] almost
        never lands within the underflow-prone region near u=1 that this fix
        targets.
        """
        rng = np.random.default_rng(1000 + degree)
        uniform_pts = rng.uniform(0.0, 1.0, size=64)
        near_boundary = np.concatenate(
            [1.0 - np.geomspace(1e-16, 1e-2, 15), np.geomspace(1e-16, 1e-2, 15)]
        )
        pts = np.concatenate([uniform_pts, near_boundary, [0.0, 0.5, 1.0]])

        for u in pts:
            row = _eval_bernstein_row(degree, float(u), np.float64)
            ref = _all_bernstein_reference(degree, float(u))
            np.testing.assert_allclose(row, ref, rtol=1e-14, atol=1e-14)


class TestForwardMirroredSymmetry:
    """Forward branch at u must match the mirrored branch at 1-u, reversed."""

    @pytest.mark.parametrize("degree", DEGREES)
    def test_symmetry(self, degree: int) -> None:
        """B_i(u) (forward branch) equals B_{p-i}(1-u) (mirrored branch).

        Values are compared within ``p*eps``, not bitwise: the two branches
        accumulate the same mathematical quantity through different sequences
        of floating-point multiplications, so rounding differs by up to a few
        ULPs per step.
        """
        rng = np.random.default_rng(2000 + degree)
        # u < 0.5 strictly, so `u` takes the forward branch and `1 - u` (> 0.5)
        # takes the mirrored branch.
        us = np.concatenate([rng.uniform(0.0, 0.5, size=20), [1e-17, 0.25]])
        for u_raw in us:
            u = float(u_raw)
            row_forward = _eval_bernstein_row(degree, u, np.float64)
            row_mirrored = _eval_bernstein_row(degree, 1.0 - u, np.float64)
            np.testing.assert_allclose(
                row_forward,
                row_mirrored[::-1],
                rtol=degree * _EPS64,
                atol=degree * _EPS64,
            )


class TestDerivativeConsistency:
    """The unaffected O(p^2) derivative kernel must agree with the fixed kernel."""

    @pytest.mark.parametrize("u", _U64, ids=lambda u: f"{u:.17g}")
    @pytest.mark.parametrize("degree", DEGREES)
    def test_row0_matches_value_kernel(self, degree: int, u: float) -> None:
        """Row 0 of the derivative kernel (ndu table) matches the ratio recurrence."""
        row = _eval_bernstein_row(degree, u, np.float64)

        out_pt = _testing.bernstein_derivatives_point(degree, u, 2)

        np.testing.assert_allclose(out_pt[0, :], row, rtol=1e-13, atol=1e-13)


class TestFusedBezierKernel:
    """End-to-end guard for the fused 1D evaluation kernel behind `Bezier.evaluate`."""

    def test_degree_30_near_u1_matches_de_casteljau(self) -> None:
        """Random-control-point degree-30 Bezier at u=1-1e-16 matches de Casteljau.

        `Bezier.slice` of a curve is a stable de Casteljau triangular reduction
        (no ratio recurrence, hence no underflow risk) used here as the
        reference for the fused kernel under test.
        """
        degree = 30
        rank = 4
        u = 1.0 - 1e-16
        rng = np.random.default_rng(42)
        ctrl = rng.normal(size=(degree + 1, rank))

        curve = Bezier(ctrl)
        out_fused = curve.evaluate(np.array([u]))
        out_ref = curve.slice(0, u)
        assert isinstance(out_ref, np.ndarray)  # a curve slices to its point

        np.testing.assert_allclose(out_fused[0], out_ref, rtol=1e-10, atol=1e-10)

    def test_batch_of_mixed_points_matches_de_casteljau(self) -> None:
        """A single batched call mixing both branches matches de Casteljau per point.

        The fused kernel runs its points in parallel chunks: a
        batch with more than one point, spanning the forward branch, the
        mirrored branch, and the shared boundary, guards against any
        cross-iteration state-sharing bug the branch could introduce (each
        point's ``b_curr`` must stay independent of every other point's).
        """
        degree = 30
        rank = 3
        pts = np.array([0.0, 0.1, 0.25, 0.5, 0.5 + _EPS64, 0.75, 1.0 - 1e-12, 1.0 - 1e-16, 1.0])
        rng = np.random.default_rng(7)
        ctrl = rng.normal(size=(degree + 1, rank))

        curve = Bezier(ctrl)
        out_fused = curve.evaluate(pts)

        for k, u in enumerate(pts):
            out_ref = curve.slice(0, float(u))
            assert isinstance(out_ref, np.ndarray)  # a curve slices to its point
            np.testing.assert_allclose(out_fused[k], out_ref, rtol=1e-10, atol=1e-10)


class TestBernsteinPointNoMirror:
    """Pin the fast path's "bit-identical to the pre-issue-#258 recurrence" contract."""

    @pytest.mark.parametrize("u", [u for u in _U64 if u <= 0.5] + [1.0], ids=lambda u: f"{u:.17g}")
    @pytest.mark.parametrize("degree", [0, 1, 2, 5, 10, _MAX_SAFE_DEGREE_NO_MIRROR_FLOAT64])
    def test_bitwise_identical_where_both_take_the_same_code_path(
        self, degree: int, u: float
    ) -> None:
        """The unmirrored recurrence is bit-identical to the mirrored recurrence.

        Only where they run the *textually identical* code: the mirrored recurrence
        takes its own forward branch (unbranched, matching
        the unmirrored recurrence line for line) for ``u <= 0.5``, and both
        functions special-case ``u == 1.0`` identically. This is the
        "byte-for-byte-unbranched fast path" contract from
        ``bernstein_row_unmirrored`` in ``include/pantr/basis/bernstein.hpp``.
        """
        out_no_mirror = _eval_unmirrored_row(degree, u)

        out_mirrored = _eval_bernstein_row(degree, u, np.float64)

        np.testing.assert_array_equal(out_no_mirror, out_mirrored)

    @pytest.mark.parametrize("u", [u for u in _U64 if 0.5 < u < 1.0], ids=lambda u: f"{u:.17g}")
    @pytest.mark.parametrize("degree", [2, 5, 10, _MAX_SAFE_DEGREE_NO_MIRROR_FLOAT64])
    def test_matches_mirrored_kernel_within_a_few_ulp_for_u_above_half(
        self, degree: int, u: float
    ) -> None:
        """For ``u > 0.5`` (excluding 1.0) the two paths take different code.

        the mirrored recurrence mirrors (forward recurrence on ``1-u`` then
        reversed) while the unmirrored recurrence always runs forward on
        ``u`` directly. Both are mathematically correct at these safe
        degrees, but accumulate the same value through different sequences
        of floating-point operations, so they are compared within a few ULPs
        rather than bit-for-bit (same rationale as
        `TestForwardMirroredSymmetry`).
        """
        out_no_mirror = _eval_unmirrored_row(degree, u)

        out_mirrored = _eval_bernstein_row(degree, u, np.float64)

        np.testing.assert_allclose(
            out_no_mirror, out_mirrored, rtol=degree * _EPS64, atol=degree * _EPS64
        )

    def test_partition_of_unity_at_max_safe_degree(self) -> None:
        """The fast path holds partition of unity at its own proven degree bound.

        Uses the exact worst-case point (`np.nextafter(1.0, 0.0)`, the
        representable float64 closest to 1) that the docstring's underflow
        analysis is based on.
        """
        degree = _MAX_SAFE_DEGREE_NO_MIRROR_FLOAT64
        u = np.nextafter(1.0, 0.0)
        out_row = _eval_unmirrored_row(degree, float(u))
        assert abs(float(out_row.sum()) - 1.0) <= 8 * degree * _EPS64

    def test_partition_of_unity_at_max_safe_degree_float32(self) -> None:
        """Float32 analogue of `test_partition_of_unity_at_max_safe_degree`."""
        degree = _MAX_SAFE_DEGREE_NO_MIRROR_FLOAT32
        u = np.nextafter(np.float32(1.0), np.float32(0.0))
        out_row = _eval_unmirrored_row(degree, float(u), np.float32)
        assert abs(float(out_row.sum()) - 1.0) <= 8 * degree * _EPS32


class TestDegreeGateDispatch:
    """The batch tabulation kernels must dispatch correctly at the safe-degree boundary."""

    @pytest.mark.parametrize(
        ("dtype", "max_safe_degree", "eps", "worst_case_u"),
        [
            pytest.param(
                np.float64,
                _MAX_SAFE_DEGREE_NO_MIRROR_FLOAT64,
                _EPS64,
                np.nextafter(np.float64(1.0), np.float64(0.0)),
                id="float64",
            ),
            pytest.param(
                np.float32,
                _MAX_SAFE_DEGREE_NO_MIRROR_FLOAT32,
                _EPS32,
                np.nextafter(np.float32(1.0), np.float32(0.0)),
                id="float32",
            ),
        ],
    )
    def test_partition_of_unity_across_the_boundary(
        self,
        dtype: _Float32Or64,
        max_safe_degree: int,
        eps: float,
        worst_case_u: np.floating[Any],
    ) -> None:
        """Both the branch-free fast path and the mirrored path stay correct.

        Degree ``max_safe_degree`` must take the fast (branch-free) dispatch
        path; degree ``max_safe_degree + 1`` is the first degree where the
        fast path would fail, so it must take the mirrored path instead —
        this is the regression guard that the dispatch boundary itself
        doesn't reintroduce issue #258 right where it matters. Runs through the
        public tabulation, whose one kernel holds the dispatch (the serial and
        parallel paths run the same chunks).
        """
        t: npt.NDArray[np.floating[Any]] = np.array([worst_case_u], dtype=dtype)
        for degree in (max_safe_degree, max_safe_degree + 1):
            out = tabulate_bernstein_1d(degree, t)
            total = float(out[0].sum())
            assert abs(total - 1.0) <= 8 * degree * eps, (
                f"degree={degree} (max_safe_degree={max_safe_degree}) sum={total}"
            )
