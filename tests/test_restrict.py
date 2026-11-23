"""Tests for Bspline.restrict() and Bezier.restrict()."""

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bezier import Bezier
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D, create_uniform_periodic_knots
from pantr.quad import PointsLattice

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_open_bspline_1d(
    degree: int = 2,
    n_intervals: int = 4,
    rank: int = 2,
    dtype: type = np.float64,
) -> Bspline:
    """Create a 1D open B-spline with random control points."""
    interior: npt.NDArray[np.float64] = np.linspace(0.0, 1.0, n_intervals + 1, dtype=dtype)[1:-1]
    knots: npt.NDArray[np.float64] = np.concatenate(
        [[0.0] * (degree + 1), interior, [1.0] * (degree + 1)]
    ).astype(dtype)
    space_1d = BsplineSpace1D(knots, degree)
    space = BsplineSpace([space_1d])
    rng = np.random.default_rng(42)
    ctrl: npt.NDArray[np.float64] = rng.random((space.num_total_basis, rank)).astype(dtype)
    return Bspline(space, ctrl)


def _make_open_bspline_2d(
    degrees: tuple[int, int] = (2, 1),
    n_intervals: tuple[int, int] = (4, 3),
    rank: int = 3,
    dtype: type = np.float64,
) -> Bspline:
    """Create a 2D open tensor-product B-spline with random control points."""
    spaces: list[BsplineSpace1D] = []
    for p, n_int in zip(degrees, n_intervals, strict=True):
        interior: npt.NDArray[np.float64] = np.linspace(0.0, 1.0, n_int + 1, dtype=dtype)[1:-1]
        knots: npt.NDArray[np.float64] = np.concatenate(
            [[0.0] * (p + 1), interior, [1.0] * (p + 1)]
        ).astype(dtype)
        spaces.append(BsplineSpace1D(knots, p))
    space = BsplineSpace(spaces)
    rng = np.random.default_rng(42)
    ctrl: npt.NDArray[np.float64] = rng.random((space.num_total_basis, rank)).astype(dtype)
    return Bspline(space, ctrl)


def _make_periodic_bspline(
    num_intervals: int = 4,
    degree: int = 2,
    rank: int = 2,
    dtype: type = np.float64,
) -> Bspline:
    """Create a 1D periodic B-spline with random control points."""
    knots = create_uniform_periodic_knots(num_intervals, degree, dtype=dtype)
    space_1d = BsplineSpace1D(knots, degree, periodic=True)
    space = BsplineSpace([space_1d])
    rng = np.random.default_rng(42)
    ctrl: npt.NDArray[np.float64] = rng.random((space.num_total_basis, rank)).astype(dtype)
    return Bspline(space, ctrl)


def _make_rational_bspline_1d(
    degree: int = 2,
    n_intervals: int = 3,
    dtype: type = np.float64,
) -> Bspline:
    """Create a 1D rational B-spline (NURBS) with random control points and weights."""
    interior: npt.NDArray[np.float64] = np.linspace(0.0, 1.0, n_intervals + 1, dtype=dtype)[1:-1]
    knots: npt.NDArray[np.float64] = np.concatenate(
        [[0.0] * (degree + 1), interior, [1.0] * (degree + 1)]
    ).astype(dtype)
    space_1d = BsplineSpace1D(knots, degree)
    space = BsplineSpace([space_1d])
    rng = np.random.default_rng(42)
    n = space.num_total_basis
    # rank 3: (x, y, w) in homogeneous coordinates
    ctrl: npt.NDArray[np.float64] = rng.random((n, 3)).astype(dtype)
    ctrl[:, 2] = rng.uniform(0.5, 2.0, size=n).astype(dtype)  # positive weights
    return Bspline(space, ctrl, is_rational=True)


# ---------------------------------------------------------------------------
# Bspline.restrict — 1D
# ---------------------------------------------------------------------------


class TestBsplineRestrict1D:
    """Tests for Bspline.restrict() on 1D B-splines."""

    def test_interior_subinterval(self) -> None:
        """Restrict to an interior sub-interval and verify evaluation agreement."""
        f = _make_open_bspline_1d()
        r = f.restrict((0.25, 0.75))

        pts = np.linspace(0.25, 0.75, 100)
        np.testing.assert_allclose(r.evaluate(pts), f.evaluate(pts), atol=1e-14)

    def test_left_at_domain(self) -> None:
        """Restrict with left bound at domain start (skip left insertion)."""
        f = _make_open_bspline_1d()
        r = f.restrict((0.0, 0.5))

        pts = np.linspace(0.0, 0.5, 100)
        np.testing.assert_allclose(r.evaluate(pts), f.evaluate(pts), atol=1e-14)

    def test_right_at_domain(self) -> None:
        """Restrict with right bound at domain end (skip right insertion)."""
        f = _make_open_bspline_1d()
        r = f.restrict((0.5, 1.0))

        pts = np.linspace(0.5, 1.0, 100)
        np.testing.assert_allclose(r.evaluate(pts), f.evaluate(pts), atol=1e-14)

    def test_single_span(self) -> None:
        """Restrict to a single knot span produces a Bézier-like result."""
        f = _make_open_bspline_1d(degree=2, n_intervals=4)
        r = f.restrict((0.25, 0.5))

        assert r.space.has_Bezier_like_knots()
        pts = np.linspace(0.25, 0.5, 50)
        np.testing.assert_allclose(r.evaluate(pts), f.evaluate(pts), atol=1e-14)

    def test_at_existing_knots(self) -> None:
        """Restrict at values that are already knots (less insertion needed)."""
        f = _make_open_bspline_1d(degree=2, n_intervals=4)
        # Knots are at 0, 0.25, 0.5, 0.75, 1.0
        r = f.restrict((0.25, 0.75))

        pts = np.linspace(0.25, 0.75, 100)
        np.testing.assert_allclose(r.evaluate(pts), f.evaluate(pts), atol=1e-14)

    def test_higher_degree(self) -> None:
        """Restrict works with higher degree B-splines."""
        f = _make_open_bspline_1d(degree=4, n_intervals=6)
        r = f.restrict((0.2, 0.8))

        pts = np.linspace(0.2, 0.8, 100)
        np.testing.assert_allclose(r.evaluate(pts), f.evaluate(pts), atol=1e-13)

    def test_periodic_auto_convert(self) -> None:
        """Restrict on a periodic B-spline auto-converts to open form."""
        f = _make_periodic_bspline(num_intervals=4, degree=2)
        f_open = f.to_open_bspline()
        r = f.restrict((0.25, 0.75))

        assert not r.space.spaces[0].periodic
        assert r.space.spaces[0].has_open_knots()
        pts = np.linspace(0.25, 0.75, 100)
        np.testing.assert_allclose(r.evaluate(pts), f_open.evaluate(pts), atol=1e-13)

    def test_rational(self) -> None:
        """Restrict preserves rationality and evaluates correctly."""
        f = _make_rational_bspline_1d()
        r = f.restrict((0.2, 0.8))

        assert r.is_rational
        pts = np.linspace(0.2, 0.8, 100)
        np.testing.assert_allclose(r.evaluate(pts), f.evaluate(pts), atol=1e-13)

    def test_domain_is_correct(self) -> None:
        """Restricted B-spline has the correct domain."""
        f = _make_open_bspline_1d()
        r = f.restrict((0.3, 0.7))

        domain = r.space.domain
        np.testing.assert_allclose(domain[0, 0], 0.3, atol=1e-15)
        np.testing.assert_allclose(domain[0, 1], 0.7, atol=1e-15)

    def test_float32(self) -> None:
        """Restrict works with float32 B-splines."""
        f = _make_open_bspline_1d(dtype=np.float32)
        r = f.restrict((0.25, 0.75))

        assert r.dtype == np.float32
        pts = np.linspace(0.25, 0.75, 50, dtype=np.float32)
        np.testing.assert_allclose(r.evaluate(pts), f.evaluate(pts), atol=1e-5)


class TestFloat32BoundNearAKnot:
    """A ``float32`` bound within the space's tolerance of a knot is that knot.

    The quadratic on ``[0, 1]`` has one interior knot at 0.5 and tolerance ``2^-20`` (``8 eps``
    of ``float32``, span 1). The bound ``0.5 + 2^-20 + 2e-8`` rounds in ``float32`` to
    ``0.5 + 2^-20``, within the tolerance of the knot, while in ``double`` it is ``2e-8`` beyond
    it; the previous release counted it as the knot and cut as if it were not, and raised.
    """

    _NEAR = 0.5 + 2.0**-20 + 2e-8

    @staticmethod
    def _curve(dtype: type) -> Bspline:
        """Build the quadratic with one interior knot at 0.5.

        Args:
            dtype (type): The storage type.

        Returns:
            Bspline: The curve.
        """
        space = BsplineSpace([BsplineSpace1D(np.array([0, 0, 0, 0.5, 1, 1, 1], dtype=dtype), 2)])
        return Bspline(space, np.array([[1.0], [2.0], [3.0], [4.0]], dtype=dtype))

    @staticmethod
    def _assert_same(a: Bspline, b: Bspline) -> None:
        """Assert two splines have equal knots and control points.

        Args:
            a (Bspline): One spline.
            b (Bspline): The other.
        """
        np.testing.assert_array_equal(a.space.spaces[0].knots, b.space.spaces[0].knots)
        np.testing.assert_array_equal(a.control_points, b.control_points)

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_reproduction_does_not_raise(self, dtype: type) -> None:
        """The issue's reproduction: both calls succeed in both types, and at a clear bound."""
        curve = self._curve(dtype)
        for bound in (self._NEAR, 0.501):
            curve.restrict((bound, 0.9))
            curve.split(0, bound)

    def test_float32_result_is_the_one_at_the_knot(self) -> None:
        """Oracle: the same call at the knot itself, same inputs to the same build."""
        curve = self._curve(np.float32)
        self._assert_same(curve.restrict((self._NEAR, 0.9)), curve.restrict((0.5, 0.9)))
        for near_part, knot_part in zip(
            curve.split(0, self._NEAR), curve.split(0, 0.5), strict=True
        ):
            self._assert_same(near_part, knot_part)

    @pytest.mark.parametrize("j", range(-16, 17))
    def test_bounds_across_the_tolerance_give_clamped_vectors(self, j: int) -> None:
        """Bounds ``0.5 + j 2^-22`` span the tolerance ``2^-20`` on both sides of the knot."""
        curve = self._curve(np.float32)
        bound = 0.5 + j * 2.0**-22
        results = [curve.restrict((bound, 0.9)), *curve.split(0, bound)]
        for r in results:
            knots = r.space.spaces[0].knots
            assert np.all(np.diff(knots) >= 0)
            assert np.count_nonzero(knots == knots[0]) == 3
            assert np.count_nonzero(knots == knots[-1]) == 3


_EPS64 = float(np.finfo(np.float64).eps)
_UNSNAPPED_NEIGHBOUR = "is within the tolerance"


def _make_squares(knots: npt.ArrayLike, degree: int, dtype: type, snap: bool) -> Bspline:
    """Build a 1D B-spline with control points ``i^2 + 1``.

    Args:
        knots (npt.ArrayLike): The knot vector.
        degree (int): The degree.
        dtype (type): The storage type.
        snap (bool): Whether the space snaps its knots.

    Returns:
        Bspline: The curve.
    """
    sp = BsplineSpace1D(np.asarray(knots, dtype=dtype), degree, False, snap)
    n = sp.num_basis
    cp: npt.NDArray[np.floating] = (np.arange(n, dtype=dtype) ** 2 + 1).reshape(n, 1)
    return Bspline(BsplineSpace([sp]), cp)


def _assert_clamped_on(b: Bspline, lo: float, hi: float) -> None:
    """Assert exactly ``degree + 1`` bitwise-equal knots at each end and the domain ``[lo, hi]``.

    Args:
        b (Bspline): The 1D B-spline.
        lo (float): The expected domain start.
        hi (float): The expected domain end.
    """
    sp = b.space.spaces[0]
    knots = sp.knots
    p = sp.degree
    assert np.count_nonzero(knots == knots[0]) == p + 1, knots
    assert np.count_nonzero(knots == knots[-1]) == p + 1, knots
    assert np.all(knots[: p + 1] == knots[0])
    assert np.all(knots[-p - 1 :] == knots[-1])
    assert float(knots[0]) == lo
    assert float(knots[-1]) == hi


class TestBoundKnotsByOneRule:
    """The knots at a restriction or split bound are decided once, for the count and the cut alike.

    The four cases are the issue's reproduction verbatim. The count of the bound's multiplicity
    and the cut of the refined knot vector used to classify the knots beside the bound by two
    rules, and where they disagreed the result was not clamped at the bound (over-collection) or
    had a wrong domain (under-collection), silently. A split, or a restriction, whose piece would
    hold a distinct knot of an unsnapped space within the tolerance of the bound refuses, naming
    the bound.
    """

    @pytest.mark.parametrize("k", [26, 27])
    def test_over_collection_left_key_rounds_onto_a_knot(self, k: int) -> None:
        """Case (1): ``fl(at - tol) == 0.3`` at 26 ulp, which the count does not take as 0.3."""
        b = _make_squares([0, 0, 0, 0.3, 0.8, 0.8, 0.8], 2, np.float64, True)
        at = 0.3 + k * float(np.spacing(0.3))
        _assert_clamped_on(b.restrict((at, 0.8)), at, 0.8)
        left, right = b.split(0, at)
        _assert_clamped_on(left, 0.0, at)
        _assert_clamped_on(right, at, 0.8)

    @pytest.mark.parametrize("factor", [1, 2])
    def test_over_collection_float32_bound_near_a_knot_in_double(self, factor: int) -> None:
        """Case (2): within ``tol`` of 0.5 in ``double``, not after rounding to ``float32``."""
        s = np.float32(1.2375)
        b = _make_squares(np.array([0, 0, 0, 0.5, s, s, s], dtype=np.float32), 2, np.float32, True)
        tol = b.space.spaces[0].tolerance
        bound = 0.5 + factor * tol
        at = float(np.float32(bound))
        _assert_clamped_on(b.restrict((bound, float(s))), at, float(s))
        left, right = b.split(0, bound)
        _assert_clamped_on(left, 0.0, at)
        _assert_clamped_on(right, at, float(s))

    @pytest.mark.parametrize("m", [32, 34])
    def test_under_collection_unsnapped_knot_tol_above_the_bound(self, m: int) -> None:
        """Case (3): a knot exactly ``tol`` above the bound 2 was counted and then cut away."""
        knots = [0, 0, 0, 1.0, 2.0, 2.0 + m * _EPS64, 4.0, 4.0, 4.0]
        b = _make_squares(knots, 2, np.float64, False)
        _assert_clamped_on(b.restrict((0.0, 2.0)), 0.0, 2.0)

    def test_under_collection_unsnapped_split_refuses_naming_the_bound(self) -> None:
        """Case (3) split: the right half would hold ``2 + 32 eps`` beside its clamped start."""
        knots = [0, 0, 0, 1.0, 2.0, 2.0 + 32 * _EPS64, 4.0, 4.0, 4.0]
        b = _make_squares(knots, 2, np.float64, False)
        with pytest.raises(ValueError, match=r"The bound 2\.0 is within the tolerance"):
            b.split(0, 2.0)
        with pytest.raises(ValueError, match=r"The bound 2\.0 is within the tolerance"):
            b.restrict((2.0, 4.0))

    def test_under_collection_float32_count_rounds_a_knot_onto_the_bound(self) -> None:
        """Case (4): the count rounds ``2^-19 + 2^-43`` to ``2^-19 <= tol`` in ``float32``."""
        k3 = -(2.0**-20 + 2.0**-43)
        b = _make_squares(
            np.array([-1, -1, -1, k3, 2.0**-20, 0.5, 0.75, 1, 1, 1], dtype=np.float32),
            2,
            np.float32,
            False,
        )
        _assert_clamped_on(b.restrict((2.0**-20, 1.0)), 2.0**-20, 1.0)
        with pytest.raises(ValueError, match=r"The bound 9\.5367431640625e-07 is within"):
            b.split(0, 2.0**-20)
        # Without the knots 0.5 and 0.75 the previous release refused with the knot-count error.
        short = _make_squares(
            np.array([-1, -1, -1, k3, 2.0**-20, 1, 1, 1], dtype=np.float32), 2, np.float32, False
        )
        _assert_clamped_on(short.restrict((2.0**-20, 1.0)), 2.0**-20, 1.0)

    def test_nan_bound_is_refused(self) -> None:
        """No knot equals a NaN, so it is refused rather than cut as if every knot were at it."""
        b = _make_squares([0, 0, 0, 0.5, 1, 1, 1], 2, np.float64, True)
        with pytest.raises(ValueError, match="outside the domain"):
            b.restrict((0.2, np.nan))
        with pytest.raises(ValueError, match="strictly inside the domain"):
            b.split(0, np.nan)


class TestBoundKnotsProperty:
    """Bounds a few ulps either side of the tolerance from a knot give clamped pieces, or refuse.

    Random open knot vectors on a span ``L`` drawn so that ``tol / ulp`` is generally not an
    integer, which is where the old left key ``at - tol`` rounded onto a knot. Unsnapped vectors
    carry one interior knot doubled by a neighbour within the tolerance, which is where the old
    count included a knot the cut dropped. Every restriction and split at the bound
    ``t + k ulp(t)``, for ``k`` around ``tol / ulp(t)``, must give pieces clamped on a domain whose
    cut end is the bound rounded to the storage type or a knot within the tolerance of it, or,
    only for an unsnapped vector, refuse naming the bound.
    """

    _SEEDS = range(8)

    @staticmethod
    def _within(knots: npt.NDArray[np.floating], value: float, tol: float) -> npt.NDArray[np.bool_]:
        """Return which knots are within ``tol`` of ``value``, the difference in the storage type.

        Args:
            knots (npt.NDArray[np.floating]): The knots.
            value (float): The value, rounded to the knots' type.
            tol (float): The tolerance.

        Returns:
            npt.NDArray[np.bool_]: One flag per knot.
        """
        diff = np.abs(knots - knots.dtype.type(value))
        return np.asarray(diff.astype(np.float64) <= tol, dtype=np.bool_)

    def _vector(
        self, rng: np.random.Generator, degree: int, dtype: type, snap: bool
    ) -> tuple[Bspline, float]:
        """Draw a curve and the interior knot the bounds are placed around.

        Args:
            rng (np.random.Generator): The generator.
            degree (int): The degree.
            dtype (type): The storage type.
            snap (bool): Whether the space snaps its knots.

        Returns:
            tuple[Bspline, float]: The curve and the knot.
        """
        span = float(rng.uniform(0.5, 4.0))
        tol_est = 8.0 * float(np.finfo(dtype).eps) * span
        while True:
            interior = np.sort(rng.uniform(0.1 * span, 0.9 * span, size=int(rng.integers(2, 5))))
            interior = interior.astype(dtype).astype(np.float64)
            if np.all(np.diff(interior) > 16.0 * tol_est):
                break
        target = float(interior[int(rng.integers(0, interior.size))])
        if not snap:
            neighbour = float(dtype(target + rng.uniform(0.3, 1.0) * tol_est * rng.choice([-1, 1])))
            interior = np.sort(np.append(interior, neighbour))
        knots = np.concatenate([[0.0] * (degree + 1), interior, [span] * (degree + 1)])
        return _make_squares(knots.astype(dtype), degree, dtype, snap), target

    def _check_piece(
        self,
        piece: Bspline,
        cut: float,
        at_start: bool,
        parent: npt.NDArray[np.floating],
        tol: float,
    ) -> None:
        """Assert a piece is clamped and its cut end is the bound rounded, or a knot near it.

        Args:
            piece (Bspline): The piece.
            cut (float): The requested bound.
            at_start (bool): Whether the cut is the piece's start (else its end).
            parent (npt.NDArray[np.floating]): The parent's knots.
            tol (float): The parent's tolerance.
        """
        sp = piece.space.spaces[0]
        knots = sp.knots
        p = sp.degree
        assert np.count_nonzero(knots == knots[0]) == p + 1, knots
        assert np.count_nonzero(knots == knots[-1]) == p + 1, knots
        end = float(knots[0] if at_start else knots[-1])
        if end == float(knots.dtype.type(cut)):
            return
        near = parent[self._within(parent, cut, tol)]
        assert bool(np.any(near.astype(np.float64) == end)), (end, cut, near)

    @pytest.mark.parametrize("degree", [1, 2, 3])
    @pytest.mark.parametrize("snap", [True, False])
    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_bounds_around_the_tolerance(self, dtype: type, snap: bool, degree: int) -> None:
        """Every bound in the window gives clamped pieces, or refuses only when unsnapped."""
        for seed in self._SEEDS:
            rng = np.random.default_rng(1000 * seed + 10 * degree + int(snap))
            b, target = self._vector(rng, degree, dtype, snap)
            sp = b.space.spaces[0]
            parent = sp.knots
            tol = float(sp.tolerance)
            lo, hi = float(parent[0]), float(parent[-1])
            ulp = float(np.spacing(dtype(target)))
            m = int(tol // ulp)
            # Half steps give bounds a storage type does not hold, which round onto the grid.
            whole = sorted({0, 1, m - 1, m, m + 1, m + 2, 2 * m, 2 * m + 1})
            for k in [s + h for s in whole for h in (0.0, 0.5)]:
                for sign in (-1, 1):
                    cut = target + sign * k * ulp
                    for kind, starts in (
                        ("lo", (True,)),
                        ("hi", (False,)),
                        ("split", (False, True)),
                    ):
                        try:
                            if kind == "lo":
                                pieces = [b.restrict((cut, hi))]
                            elif kind == "hi":
                                pieces = [b.restrict((lo, cut))]
                            else:
                                pieces = list(b.split(0, cut))
                        except ValueError as err:
                            assert not snap, (seed, k, sign, err)
                            assert _UNSNAPPED_NEIGHBOUR in str(err), err
                            continue
                        for piece, at_start in zip(pieces, starts, strict=True):
                            self._check_piece(piece, cut, at_start, parent, tol)


# ---------------------------------------------------------------------------
# Bspline.restrict — error cases
# ---------------------------------------------------------------------------


class TestBsplineRestrictErrors:
    """Tests for Bspline.restrict() error handling."""

    def test_full_domain_raises(self) -> None:
        """Restrict to the full domain raises ValueError."""
        f = _make_open_bspline_1d()
        with pytest.raises(ValueError, match="full domain"):
            f.restrict((0.0, 1.0))

    def test_lower_ge_upper_raises(self) -> None:
        """Lower bound >= upper bound raises ValueError."""
        f = _make_open_bspline_1d()
        with pytest.raises(ValueError, match="strictly less"):
            f.restrict((0.5, 0.5))
        with pytest.raises(ValueError, match="strictly less"):
            f.restrict((0.7, 0.3))

    def test_out_of_domain_lower_raises(self) -> None:
        """Lower bound below domain raises ValueError."""
        f = _make_open_bspline_1d()
        with pytest.raises(ValueError, match="below the domain"):
            f.restrict((-0.1, 0.5))

    def test_out_of_domain_upper_raises(self) -> None:
        """Upper bound above domain raises ValueError."""
        f = _make_open_bspline_1d()
        with pytest.raises(ValueError, match="above the domain"):
            f.restrict((0.5, 1.1))

    def test_wrong_dim_raises(self) -> None:
        """Wrong sequence length for nD raises ValueError."""
        f = _make_open_bspline_2d()
        with pytest.raises(ValueError, match="must match dim"):
            f.restrict([(0.1, 0.9)])  # only 1 direction for a 2D spline

    def test_all_none_raises(self) -> None:
        """All directions None raises ValueError."""
        f = _make_open_bspline_2d()
        with pytest.raises(ValueError, match="non-None bounds"):
            f.restrict([None, None])


# ---------------------------------------------------------------------------
# Bspline.restrict — nD
# ---------------------------------------------------------------------------


class TestBsplineRestrictND:
    """Tests for Bspline.restrict() on multi-dimensional B-splines."""

    def test_restrict_one_direction(self) -> None:
        """Restrict 2D B-spline in one direction only."""
        f = _make_open_bspline_2d()
        r = f.restrict([(0.25, 0.75), None])

        # Direction 0 is restricted, direction 1 unchanged.
        domain = r.space.domain
        np.testing.assert_allclose(domain[0], [0.25, 0.75], atol=1e-15)
        np.testing.assert_allclose(domain[1], [0.0, 1.0], atol=1e-15)

    def test_restrict_both_directions(self) -> None:
        """Restrict 2D B-spline in both directions."""
        f = _make_open_bspline_2d()
        r = f.restrict([(0.25, 0.75), (0.33, 0.67)])

        domain = r.space.domain
        np.testing.assert_allclose(domain[0], [0.25, 0.75], atol=1e-15)
        np.testing.assert_allclose(domain[1], [0.33, 0.67], atol=1e-15)

    def test_restrict_2d_evaluation(self) -> None:
        """Restricted 2D B-spline evaluates the same as original on subdomain."""
        f = _make_open_bspline_2d()
        r = f.restrict([(0.25, 0.75), (0.33, 0.67)])

        pts_u = np.linspace(0.25, 0.75, 20)
        pts_v = np.linspace(0.33, 0.67, 15)
        lattice = PointsLattice([pts_u, pts_v])

        np.testing.assert_allclose(r.evaluate(lattice), f.evaluate(lattice), atol=1e-13)


class TestBsplineRestrictScaleCovariance:
    """The verdicts restrict reaches must not depend on where the domain sits.

    Restriction adds no tolerance of its own: every comparison it makes is a knot
    identity question answered by ``BsplineSpace1D.tolerance``, an absolute
    parametric length derived from the knot vector's own magnitude. These tests pin
    the property that follows: send ``x -> lambda * x`` and nothing changes but the
    coordinates.
    """

    @staticmethod
    def _shifted(lo: float, span: float, degree: int = 3, n_intervals: int = 5) -> Bspline:
        """Build an open 1D B-spline on ``[lo, lo + span]`` with a fixed control polygon."""
        interior = np.linspace(lo, lo + span, n_intervals + 1)[1:-1]
        knots = np.concatenate([[lo] * (degree + 1), interior, [lo + span] * (degree + 1)])
        space = BsplineSpace([BsplineSpace1D(knots, degree)])
        rng = np.random.default_rng(11)
        return Bspline(space, rng.random((space.num_total_basis, 2)))

    @pytest.mark.parametrize("lam", [1.0e-6, 1.0, 1.0e6, 1.0e9])
    def test_restriction_keeps_the_whole_boundary_group(self, lam: float) -> None:
        """The extraction must not truncate the clamped end at any coordinate magnitude.

        ``searchsorted(refined_knots, b_new + tol)`` is what cuts the sub-vector, and
        with the previous fixed ``1e-12`` the offset was absorbed by ``b_new`` itself
        once one ulp of the coordinate exceeded it (at ``lam = 1e6`` an ulp is about
        ``1.2e-10``), so the cut fell before the first copy of ``b_new`` and the last
        ``degree + 1`` knots were dropped. A tolerance of ``8 * eps * scale`` is at
        least four ulp of any coordinate present, so the offset always bites.
        """
        f = self._shifted(0.0, lam)
        degree = f.degree[0]
        r = f.restrict((0.2 * lam, 0.6 * lam))

        lo, hi = r.space.domain[0]
        assert lo == pytest.approx(0.2 * lam, rel=1.0e-14)
        assert hi == pytest.approx(0.6 * lam, rel=1.0e-14)
        # Clamped at both ends: the boundary group survived the cut intact.  Counted
        # through the space's own notion of knot identity, not bitwise: a boundary that
        # coincides with an existing knot keeps that knot's stored representative, which
        # may differ from the requested bound in the last bit.
        _, mult = r.space.spaces[0].get_unique_knots_and_multiplicity()
        assert mult[0] == degree + 1
        assert mult[-1] == degree + 1
        assert r.space.spaces[0].knots.size == mult.sum()

    @pytest.mark.parametrize("lam", [1.0e-6, 1.0, 1.0e6, 1.0e9])
    def test_the_restricted_map_still_agrees_with_the_original(self, lam: float) -> None:
        """Restriction is exact geometry, so the two evaluate alike at every scale."""
        f = self._shifted(0.0, lam)
        r = f.restrict((0.2 * lam, 0.6 * lam))
        pts = np.linspace(0.2 * lam, 0.6 * lam, 50)
        np.testing.assert_allclose(r.evaluate(pts), f.evaluate(pts), atol=0.0, rtol=1.0e-12)

    @pytest.mark.parametrize("lam", [1.0e-6, 1.0, 1.0e6, 1.0e9])
    def test_a_bound_one_ulp_outside_the_domain_is_still_the_domain(self, lam: float) -> None:
        """One ulp of slack is forgiven wherever the domain sits.

        The bound is snapped onto the endpoint rather than rejected, so the right
        insertion is skipped exactly as it is for the endpoint itself.
        """
        f = self._shifted(0.0, lam)
        just_outside = float(np.nextafter(lam, np.inf))
        r = f.restrict((0.4 * lam, just_outside))
        assert r.space.domain[0][1] == pytest.approx(lam, rel=1.0e-15)

    def test_an_offset_domain_reaches_the_same_verdicts_as_a_centered_one(self) -> None:
        """Translating the knot vector changes the arithmetic, not what restrict decides.

        Only the *verdicts* are covariant.  The control points are not expected to
        agree bitwise: at an offset of ``1e6`` the insertion weights are formed from
        coordinates of that magnitude, so they carry a relative error of about
        ``eps * 1e6 / span = 2.2e-10``, and the control points inherit it.  Measured
        agreement here is ``2.9e-10`` relative, which is that figure and not a defect;
        what must be identical is the knot structure the two runs produce.
        """
        base = self._shifted(0.0, 1.0)
        moved = self._shifted(1.0e6, 1.0)
        r_base = base.restrict((0.2, 0.6))
        r_moved = moved.restrict((1.0e6 + 0.2, 1.0e6 + 0.6))

        _, mult_base = r_base.space.spaces[0].get_unique_knots_and_multiplicity()
        _, mult_moved = r_moved.space.spaces[0].get_unique_knots_and_multiplicity()
        np.testing.assert_array_equal(mult_moved, mult_base)
        assert r_moved.control_points.shape == r_base.control_points.shape
        # The insertion-weight error above, with a factor of four for the chain.
        weight_error = 4.0 * np.finfo(np.float64).eps * 1.0e6
        np.testing.assert_allclose(
            r_moved.control_points, r_base.control_points, rtol=weight_error, atol=0.0
        )


# ---------------------------------------------------------------------------
# Bezier.restrict
# ---------------------------------------------------------------------------


class TestBezierRestrict:
    """Tests for Bezier.restrict()."""

    def test_1d_restrict(self) -> None:
        """Restrict a 1D Bézier to a sub-interval and verify reparametrization."""
        ctrl = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
        bez = Bezier(ctrl)
        r = bez.restrict((0.2, 0.8))

        assert r.degree == (2,)
        # r evaluates on [0,1], maps to bez on [0.2, 0.8]
        t_restr = np.linspace(0.0, 1.0, 100)
        t_mapped = 0.2 + 0.6 * t_restr
        np.testing.assert_allclose(r.evaluate(t_restr), bez.evaluate(t_mapped), atol=1e-14)

    def test_1d_restrict_higher_degree(self) -> None:
        """Restrict a cubic Bézier."""
        rng = np.random.default_rng(42)
        ctrl = rng.random((4, 3))
        bez = Bezier(ctrl)
        r = bez.restrict((0.1, 0.9))

        t_restr = np.linspace(0.0, 1.0, 100)
        t_mapped = 0.1 + 0.8 * t_restr
        np.testing.assert_allclose(r.evaluate(t_restr), bez.evaluate(t_mapped), atol=1e-13)

    def test_nd_restrict_one_direction(self) -> None:
        """Restrict a 2D Bézier in one direction."""
        rng = np.random.default_rng(42)
        ctrl = rng.random((3, 4, 2))
        bez = Bezier(ctrl)
        r = bez.restrict([(0.2, 0.8), None])

        assert r.degree == (2, 3)

    def test_rational(self) -> None:
        """Restrict a rational Bézier preserves rationality."""
        rng = np.random.default_rng(42)
        ctrl = rng.random((3, 3))
        ctrl[:, 2] = rng.uniform(0.5, 2.0, size=3)
        bez = Bezier(ctrl, is_rational=True)
        r = bez.restrict((0.2, 0.8))

        assert r.is_rational
        t_restr = np.linspace(0.0, 1.0, 100)
        t_mapped = 0.2 + 0.6 * t_restr  # [0,1] -> [0.2, 0.8]
        np.testing.assert_allclose(r.evaluate(t_restr), bez.evaluate(t_mapped), atol=1e-13)

    def test_full_domain_raises(self) -> None:
        """Restrict to full [0, 1] raises ValueError."""
        ctrl = np.array([[0.0], [1.0], [2.0]])
        bez = Bezier(ctrl)
        with pytest.raises(ValueError, match="full domain"):
            bez.restrict((0.0, 1.0))
