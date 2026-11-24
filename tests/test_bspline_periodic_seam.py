"""Tests for opening, restricting and splitting a periodic B-spline with a repeated seam knot.

A periodic knot vector whose seam knot is repeated holds copies of the domain start ``a = knots[p]``
past index ``p``, or copies of the end ``b = knots[n - p - 1]`` before index ``n - p - 1``, inside
the domain, depending on which copy the vector starts its domain at. The conversion to open form
counts every copy of a domain end, wherever it lies, and a space built with ``snap_knots=False``
holding a knot distinct from the seam but within its tolerance is refused with a message naming the
seam and that knot, as ``restrict`` and ``split`` refuse at a cut end.

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


def _periodic_knots(
    cycle: list[float], degree: int, *, mirror: bool = False
) -> npt.NDArray[np.float64]:
    """Build the periodic knot vector of period 1 whose domain knots are ``cycle`` and ``1``.

    Knot ``i`` is ``cycle[(i - degree) % len(cycle)]`` shifted by its period, so ``knots[degree]``
    is the first copy of the seam ``0`` and the copies of ``1`` start at ``n - degree - 1``.

    Args:
        cycle (list[float]): The knots of one period, in ``[0, 1)``, starting at the seam ``0``.
        degree (int): The degree.
        mirror (bool): Return ``1 - knots`` reversed instead, whose ``knots[n - degree - 1]`` is
            the last copy of the seam ``1`` and whose copies of ``0`` end at ``degree``. Defaults
            to False.

    Returns:
        npt.NDArray[np.float64]: ``len(cycle) + 2 * degree + 1`` knots, with ``degree`` ghost
        knots on each side of the domain ``[0, 1]``.
    """
    n = len(cycle) + 2 * degree + 1
    knots = np.array(
        [cycle[(i - degree) % len(cycle)] + ((i - degree) // len(cycle)) for i in range(n)],
        dtype=np.float64,
    )
    return 1.0 - knots[::-1] if mirror else knots


def _periodic(cycle: list[float], degree: int, *, snap: bool, mirror: bool = False) -> Bspline:
    """Build a periodic scalar B-spline over `_periodic_knots`, control points ``0, 1, ...``.

    Args:
        cycle (list[float]): The knots of one period, starting at the seam ``0``.
        degree (int): The degree.
        snap (bool): The space's ``snap_knots``.
        mirror (bool): Passed to `_periodic_knots`. Defaults to False.

    Returns:
        Bspline: The periodic B-spline.
    """
    space = BsplineSpace1D(_periodic_knots(cycle, degree, mirror=mirror), degree, True, snap)
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


# Each case: the cycle, the degree, the copies of the seam in the cycle, and `_periodic_knots`'s
# ``mirror``. Unmirrored, the copies of ``a`` past index ``p`` are missed by a count over the first
# ``p + 1`` knots; mirrored, those of ``b`` before ``n - p - 1`` by a count over the last ``p + 1``.
_SEAM_CASES = [
    pytest.param([0.0, 0.0, 0.25, 0.5, 0.75], 2, 2, False, id="p2-double"),
    pytest.param([0.0, 0.0, 0.25, 0.5, 0.75], 3, 2, False, id="p3-double"),
    pytest.param([0.0, 0.0, 0.0, 0.25, 0.5, 0.75], 3, 3, False, id="p3-triple"),
    pytest.param([0.0, 0.0, 0.25, 0.5, 0.75], 2, 2, True, id="p2-double-mirror"),
    pytest.param([0.0, 0.0, 0.0, 0.25, 0.5, 0.75], 3, 3, True, id="p3-triple-mirror"),
]


class TestRepeatedSeamKnot:
    """A periodic space whose seam knot is repeated opens, restricts and splits."""

    @pytest.mark.parametrize(("cycle", "degree", "copies", "mirror"), _SEAM_CASES)
    def test_premise(self, cycle: list[float], degree: int, copies: int, mirror: bool) -> None:
        """Copies of a domain end lie inside the domain, outside its end's ``p + 1`` knots."""
        knots = np.asarray(
            _periodic(cycle, degree, snap=False, mirror=mirror).space.spaces[0].knots
        )
        n = knots.size
        if mirror:
            assert knots[n - degree - 1] == 1.0
            assert np.all(knots[n - degree - copies : n - degree] == 1.0)
            assert knots[n - degree - copies - 1] < 1.0
            assert knots[n - degree] > 1.0
        else:
            assert knots[degree] == 0.0
            assert np.all(knots[degree : degree + copies] == 0.0)
            assert knots[degree + copies] > 0.0
            assert knots[degree - 1] < 0.0

    @pytest.mark.parametrize(("cycle", "degree", "copies", "mirror"), _SEAM_CASES)
    @pytest.mark.parametrize("snap", [False, True])
    def test_to_open(
        self, cycle: list[float], degree: int, copies: int, mirror: bool, snap: bool
    ) -> None:
        """``to_open_bspline`` clamps at the seam and keeps the periodic curve."""
        del copies
        f = _periodic(cycle, degree, snap=snap, mirror=mirror)
        _check_open_piece(f, f.to_open_bspline(), 0.0, 1.0)

    @pytest.mark.parametrize(("cycle", "degree", "copies", "mirror"), _SEAM_CASES)
    def test_restrict(self, cycle: list[float], degree: int, copies: int, mirror: bool) -> None:
        """``restrict`` far from the seam keeps the periodic curve on the window."""
        del copies
        f = _periodic(cycle, degree, snap=False, mirror=mirror)
        _check_open_piece(f, f.restrict((0.3, 0.6)), 0.3, 0.6)

    @pytest.mark.parametrize(("cycle", "degree", "copies", "mirror"), _SEAM_CASES)
    def test_split(self, cycle: list[float], degree: int, copies: int, mirror: bool) -> None:
        """``split`` at ``0.5`` gives two halves that keep the periodic curve."""
        del copies
        f = _periodic(cycle, degree, snap=False, mirror=mirror)
        left, right = f.split(0, 0.5)
        _check_open_piece(f, left, 0.0, 0.5)
        _check_open_piece(f, right, 0.5, 1.0)


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

    @pytest.mark.parametrize("cycle", [_NEAR_START, _NEAR_END])
    def test_snapped_is_a_double_seam_knot(self, cycle: list[float]) -> None:
        """Snapped, the near pair is one knot twice at the seam, and the space opens.

        Next to the start the pair snaps onto ``0``: the vector of the ``p2-double`` case, with a
        second copy of ``a`` at index ``p + 1``. Next to the end it snaps onto ``1 - 1e-15``,
        which moves the seam there: the shape of ``p2-double-mirror``, a second copy of ``b`` at
        ``n - p - 2``.
        """
        f = _periodic(cycle, 2, snap=True)
        space = f.space.spaces[0]
        knots = np.asarray(space.knots)
        n = knots.size
        a, b = (float(v) for v in space.domain)
        assert np.count_nonzero(knots == a) == 2
        assert np.count_nonzero(knots == b) == 2
        if cycle is _NEAR_START:
            assert (a, b) == (0.0, 1.0)
            assert knots[3] == a
        else:
            assert (a, b) == (-(1.0 - (1.0 - 1e-15)), 1.0 - 1e-15)
            assert knots[n - 4] == b
        _check_open_piece(f, f.to_open_bspline(), a, b)
        _check_open_piece(f, f.restrict((0.3, 0.6)), 0.3, 0.6)
        left, right = f.split(0, 0.5)
        _check_open_piece(f, left, a, 0.5)
        _check_open_piece(f, right, 0.5, b)


class TestUnclampedEndCopies:
    """A non-periodic unclamped space whose start has ``p + 1`` copies after an external knot."""

    def test_to_open_trims_the_external_knot(self) -> None:
        """Nothing is inserted; the external knot and its control point are trimmed."""
        space = BsplineSpace1D(np.array([-1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0]), 2)
        assert not space.has_open_knots()
        f = Bspline(BsplineSpace([space]), np.arange(1.0, 6.0))
        g = f.to_open_bspline()
        np.testing.assert_array_equal(g.space.spaces[0].knots, [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0])
        np.testing.assert_array_equal(np.ravel(g.control_points), [2.0, 3.0, 4.0, 5.0])
