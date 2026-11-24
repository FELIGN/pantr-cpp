"""Tests for opening, restricting and splitting a periodic B-spline with a repeated seam knot.

A periodic knot vector whose seam knot is repeated and whose domain starts at the first copy holds
copies of the domain start ``a = knots[p]`` past index ``p``, inside the domain; a non-periodic
vector can hold copies of an end inside its domain past an external knot too. The conversion to
open form counts every copy of a domain end, wherever it lies, and a space built with
``snap_knots=False`` holding a knot distinct from the seam but within its tolerance is refused with
a message naming the seam and that knot, as ``restrict`` and ``split`` refuse at a cut end.

Oracle: :mod:`tests.oracles.bspline_exact`, the Cox-de Boor recursion on the unwrapped periodic
knot vector with the control point of function ``i`` read at ``i % n_stored``, in exact rational
arithmetic; it shares no code with the library's conversion or evaluation. The tolerance is the
``atol=1e-12`` the existing ``to_open_bspline`` tests assert (``tests/test_bspline_conversion.py``).
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from tests.oracles import bspline_exact

_ATOL = 1e-12
"""The tolerance of the existing ``to_open_bspline`` correctness tests."""


def _periodic_knots(cycle: list[float], degree: int) -> npt.NDArray[np.float64]:
    """Build the periodic knot vector of period 1 whose domain knots are ``cycle`` and ``1``.

    Knot ``i`` is ``cycle[(i - degree) % len(cycle)]`` shifted by its period, so ``knots[degree]``
    is the first copy of the seam ``0`` and the copies of ``1`` start at ``n - degree - 1``.

    Args:
        cycle (list[float]): The knots of one period, in ``[0, 1)``, starting at the seam ``0``.
        degree (int): The degree.

    Returns:
        npt.NDArray[np.float64]: ``len(cycle) + 2 * degree + 1`` knots, with ``degree`` ghost
        knots on each side of the domain ``[0, 1]``.
    """
    n = len(cycle) + 2 * degree + 1
    return np.array(
        [cycle[(i - degree) % len(cycle)] + ((i - degree) // len(cycle)) for i in range(n)],
        dtype=np.float64,
    )


def _periodic(cycle: list[float], degree: int, *, snap: bool) -> Bspline:
    """Build a periodic scalar B-spline over `_periodic_knots`, control points ``0, 1, ...``.

    Args:
        cycle (list[float]): The knots of one period, starting at the seam ``0``.
        degree (int): The degree.
        snap (bool): The space's ``snap_knots``.

    Returns:
        Bspline: The periodic B-spline.
    """
    space = BsplineSpace1D(_periodic_knots(cycle, degree), degree, True, snap)
    return Bspline(
        BsplineSpace([space]), np.arange(space.num_basis, dtype=np.float64).reshape(-1, 1)
    )


def _reference(f: Bspline, pts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Evaluate the periodic B-spline ``f`` at ``pts`` by its definition, exactly, rounded once.

    Args:
        f (Bspline): A 1D periodic B-spline.
        pts (npt.NDArray[np.float64]): Points inside its domain.

    Returns:
        npt.NDArray[np.float64]: The values, ``(len(pts), 1)``.
    """
    space = f.space.spaces[0]
    rows = [
        bspline_exact.derivative([space.knots], [space.degree], f.control_points, [float(x)])
        for x in pts
    ]
    return bspline_exact.to_float_array(rows)


def _interior(lo: float, hi: float) -> npt.NDArray[np.float64]:
    """Points strictly inside ``[lo, hi]``, as the existing ``to_open`` tests sample.

    Args:
        lo (float): The lower end.
        hi (float): The upper end.

    Returns:
        npt.NDArray[np.float64]: 39 points.
    """
    return np.linspace(lo, hi, 41, dtype=np.float64)[1:-1]


def _check_open_piece(f: Bspline, piece: Bspline, lo: float, hi: float) -> None:
    """Check that ``piece`` is clamped on ``[lo, hi]`` and agrees with ``f`` there.

    Args:
        f (Bspline): The periodic B-spline.
        piece (Bspline): The open result.
        lo (float): Its expected lower domain end.
        hi (float): Its expected upper domain end.
    """
    space = piece.space.spaces[0]
    assert not space.periodic
    assert space.has_open_knots()
    assert tuple(float(v) for v in space.domain) == (lo, hi)
    pts = _interior(lo, hi)
    got = np.asarray(piece.evaluate(pts), dtype=np.float64).reshape(-1, 1)
    np.testing.assert_allclose(got, _reference(f, pts), atol=_ATOL, rtol=0.0)


# Each case: the cycle, the degree, and the copies of the seam in the cycle. With ``p + 1`` copies
# (``p1-double``) nothing is inserted and only the ghost knots are trimmed.
_SEAM_CASES = [
    pytest.param([0.0, 0.0, 0.25, 0.5, 0.75], 2, 2, id="p2-double"),
    pytest.param([0.0, 0.0, 0.25, 0.5, 0.75], 3, 2, id="p3-double"),
    pytest.param([0.0, 0.0, 0.0, 0.25, 0.5, 0.75], 3, 3, id="p3-triple"),
    pytest.param([0.0, 0.0, 0.25, 0.5, 0.75], 1, 2, id="p1-double"),
]


class TestRepeatedSeamKnot:
    """A periodic space whose seam knot is repeated opens, restricts and splits."""

    @pytest.mark.parametrize(("cycle", "degree", "copies"), _SEAM_CASES)
    def test_premise(self, cycle: list[float], degree: int, copies: int) -> None:
        """Copies of the start lie past index ``p``, and the space has one function per knot."""
        space = _periodic(cycle, degree, snap=False).space.spaces[0]
        knots = np.asarray(space.knots)
        assert space.num_basis == len(cycle)
        assert np.all(knots[degree : degree + copies] == 0.0)
        assert knots[degree + copies] > 0.0
        assert knots[degree - 1] < 0.0

    @pytest.mark.parametrize(("cycle", "degree", "copies"), _SEAM_CASES)
    @pytest.mark.parametrize("snap", [False, True])
    def test_to_open(self, cycle: list[float], degree: int, copies: int, snap: bool) -> None:
        """``to_open_bspline`` clamps at the seam and keeps the periodic curve."""
        del copies
        f = _periodic(cycle, degree, snap=snap)
        _check_open_piece(f, f.to_open_bspline(), 0.0, 1.0)

    @pytest.mark.parametrize(("cycle", "degree", "copies"), _SEAM_CASES)
    def test_restrict(self, cycle: list[float], degree: int, copies: int) -> None:
        """``restrict`` far from the seam keeps the periodic curve on the window."""
        del copies
        f = _periodic(cycle, degree, snap=False)
        _check_open_piece(f, f.restrict((0.3, 0.6)), 0.3, 0.6)

    @pytest.mark.parametrize(("cycle", "degree", "copies"), _SEAM_CASES)
    def test_split(self, cycle: list[float], degree: int, copies: int) -> None:
        """``split`` at ``0.5`` gives two halves that keep the periodic curve."""
        del copies
        f = _periodic(cycle, degree, snap=False)
        left, right = f.split(0, 0.5)
        _check_open_piece(f, left, 0.0, 0.5)
        _check_open_piece(f, right, 0.5, 1.0)

    # ``p1-double`` is left out: converting back refuses a seam with ``degree + 1`` copies.
    @pytest.mark.parametrize(("cycle", "degree", "copies"), _SEAM_CASES[:3])
    def test_callers_that_open_the_direction(
        self, cycle: list[float], degree: int, copies: int
    ) -> None:
        """Knot insertion, degree elevation and the same-degree derivative keep the curve.

        Each opens the periodic direction through the same conversion; the first two convert back,
        so the result is periodic again.
        """
        del copies
        f = _periodic(cycle, degree, snap=False)
        pts = _interior(0.0, 1.0)
        for g in (f.insert_knots([0.375]), f.elevate_degree(1)):
            assert g.space.spaces[0].periodic
            got = np.asarray(g.evaluate(pts), dtype=np.float64).reshape(-1, 1)
            np.testing.assert_allclose(got, _reference(f, pts), atol=_ATOL, rtol=0.0)
        space = f.space.spaces[0]
        want = bspline_exact.to_float_array(
            [
                bspline_exact.derivative([space.knots], [degree], f.control_points, [float(x)], [1])
                for x in pts
            ]
        )
        got = np.asarray(f.derivative(keep_degree=True).evaluate(pts), dtype=np.float64)
        np.testing.assert_allclose(got.reshape(-1, 1), want, atol=_ATOL, rtol=0.0)


_NEAR_START = [0.0, 1e-15, 0.25, 0.5, 0.75]
"""A cycle whose second knot is within the tolerance of the domain start."""

_NEAR_END = [0.0, 0.25, 0.5, 0.75, 1.0 - 1e-15]
"""A cycle whose last knot is within the tolerance of the domain end."""


class TestNearSeamKnot:
    """A knot distinct from the seam but within the tolerance of it."""

    def test_premise(self) -> None:
        """The near knots are distinct from the seam and within the space's tolerance of it."""
        start = _periodic(_NEAR_START, 2, snap=False).space.spaces[0]
        knots = np.asarray(start.knots)
        assert knots[2] == 0.0
        assert knots[3] != 0.0
        assert abs(knots[3]) <= start.tolerance
        end = _periodic(_NEAR_END, 2, snap=False).space.spaces[0]
        knots = np.asarray(end.knots)
        n = knots.size
        assert knots[n - 3] == 1.0
        assert knots[n - 4] != 1.0
        assert abs(knots[n - 4] - 1.0) <= end.tolerance

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda f: f.to_open_bspline(), id="to_open"),
            pytest.param(lambda f: f.restrict((0.3, 0.6)), id="restrict"),
            pytest.param(lambda f: f.split(0, 0.5), id="split"),
        ],
    )
    @pytest.mark.parametrize(
        ("cycle", "seam", "near"),
        [
            pytest.param(_NEAR_START, r"0\.0", r"1e-15", id="start"),
            pytest.param(_NEAR_END, r"1\.0", r"0\.999999999999999", id="end"),
        ],
    )
    def test_unsnapped_refuses_naming_the_seam(
        self, call: Callable[[Bspline], object], cycle: list[float], seam: str, near: str
    ) -> None:
        """Unsnapped, every opening call refuses naming the seam and the near knot."""
        f = _periodic(cycle, 2, snap=False)
        with pytest.raises(ValueError, match=rf"^The seam {seam} .* distinct knot {near},") as err:
            call(f)
        assert "snap_knots=True" in str(err.value)
        assert "multiplicity" not in str(err.value)

    def test_snapped_is_the_double_seam_knot(self) -> None:
        """Snapped, the pair next to the start is ``0`` twice (``p2-double``), and it opens."""
        f = _periodic(_NEAR_START, 2, snap=True)
        knots = np.asarray(f.space.spaces[0].knots)
        np.testing.assert_array_equal(knots, _periodic_knots([0.0, 0.0, 0.25, 0.5, 0.75], 2))
        _check_open_piece(f, f.to_open_bspline(), 0.0, 1.0)
        _check_open_piece(f, f.restrict((0.3, 0.6)), 0.3, 0.6)
        left, right = f.split(0, 0.5)
        _check_open_piece(f, left, 0.0, 0.5)
        _check_open_piece(f, right, 0.5, 1.0)


def _open_of(knots: list[float], degree: int, *, snap: bool = True) -> Bspline:
    """Open a non-periodic scalar B-spline with control points ``1, 2, ...`` over ``knots``.

    Args:
        knots (list[float]): The knot vector.
        degree (int): The degree.
        snap (bool): The space's ``snap_knots``. Defaults to True.

    Returns:
        Bspline: ``to_open_bspline()`` of it.
    """
    space = BsplineSpace1D(np.array(knots, dtype=np.float64), degree, False, snap)
    assert not space.has_open_knots()
    f = Bspline(BsplineSpace([space]), np.arange(1.0, space.num_basis + 1.0))
    return f.to_open_bspline()


class TestUnclampedEndCopies:
    """A non-periodic unclamped space with copies of an end beyond its first or last ``p + 1``."""

    def test_start_copies_after_an_external_knot(self) -> None:
        """Nothing is inserted at the start; the external knot and its control point are trimmed."""
        g = _open_of([-1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0], 2)
        np.testing.assert_array_equal(g.space.spaces[0].knots, [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0])
        np.testing.assert_array_equal(np.ravel(g.control_points), [2.0, 3.0, 4.0, 5.0])

    def test_end_copies_before_an_external_knot(self) -> None:
        """Nothing is inserted at the end; the external knot and its control point are trimmed."""
        g = _open_of([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 3.0], 2)
        np.testing.assert_array_equal(g.space.spaces[0].knots, [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0])
        np.testing.assert_array_equal(np.ravel(g.control_points), [1.0, 2.0, 3.0, 4.0])

    def test_an_external_knot_near_the_start_is_trimmed(self) -> None:
        """A knot below the start within its tolerance is not a copy: the result starts at ``0``.

        Counting copies under the tolerance took ``-1e-15`` for one and started the result there.
        The values on the domain are unchanged: the knots and control points of the result are the
        exact ones of Boehm insertion of ``0`` once and ``1.5`` twice, checked against the original
        at interior points.
        """
        knots = [-1e-15, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        g = _open_of(knots, 2, snap=False)
        space = g.space.spaces[0]
        np.testing.assert_array_equal(space.knots, [0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 1.5, 1.5])
        assert float(space.domain[0]) == 0.0
        original = BsplineSpace1D(np.array(knots), 2, False, False)
        ctrl = np.arange(1.0, original.num_basis + 1.0).reshape(-1, 1)
        pts = _interior(0.0, 1.5)
        want = bspline_exact.to_float_array(
            [bspline_exact.derivative([original.knots], [2], ctrl, [float(x)]) for x in pts]
        )
        got = np.asarray(g.evaluate(pts), dtype=np.float64).reshape(-1, 1)
        np.testing.assert_allclose(got, want, atol=_ATOL, rtol=0.0)

    def test_more_than_degree_plus_one_copies_is_refused(self) -> None:
        """Four copies of the start at degree 2 are not a knot vector of that degree."""
        with pytest.raises(
            ValueError, match=r"^to_open: the domain end 0\.0 has 4 copies, more than degree"
        ):
            _open_of([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0], 2)

    def test_more_than_degree_plus_one_copies_of_the_end_is_refused(self) -> None:
        """Four copies of the domain end at degree 2, before an external knot, are refused."""
        with pytest.raises(
            ValueError, match=r"^to_open: the domain end 1\.0 has 4 copies, more than degree"
        ):
            _open_of([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0], 2)

    def test_unsnapped_knot_near_a_domain_end_is_refused(self) -> None:
        """A distinct knot within the tolerance of the start, inside the domain, is refused."""
        with pytest.raises(
            ValueError, match=r"^The domain end 0\.0 is within the tolerance .* knot 1e-15,"
        ) as err:
            _open_of([-1.0, 0.0, 0.0, 1e-15, 1.0, 2.0, 3.0, 4.0], 2, snap=False)
        assert "snap_knots=True" in str(err.value)
