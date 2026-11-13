"""Knot-layer predicates must use an absolute tolerance, not ``np.isclose``.

Every knot-layer predicate grades a parametric length against
:attr:`~pantr.bspline.BsplineSpace1D.tolerance`, which is **absolute**: it already
carries the knot vector's own magnitude, so the comparison must be a plain
``|a - b| <= tol`` and nothing else. But ``np.isclose(a, b, atol=tol)`` keeps
numpy's default ``rtol=1e-5``, so the test it actually performs is
``|a - b| <= tol + 1e-5 * |b|``. Every predicate written that way therefore widened
by ``1e-5`` times the operand magnitude -- a second, undeclared relative term on top
of the derived one -- which is harmless on a unit domain and catastrophic on a large
one.

(The presets in :mod:`pantr.tolerance` are dimensionless relative factors; it is
``pantr._testing.knot_tolerance`` that turns one into the absolute length this file
is about. ``TestDerivedKnotTolerance`` at the end pins that derivation and the
properties the knot layer rests on.)

Two further properties of the leak are worth pinning, because they are what makes
it hard to spot:

* the ``rtol`` leg attaches to the **second** operand only, so the predicate is
  asymmetric in its arguments, and a comparison against an operand that happens to
  be ``0.0`` shows no leak at all;
* the widening is *relative*, so a test on a ``[0, 1]`` domain cannot see it.

Each test below therefore drives the same construction at two or three coordinate
scales and asserts the verdict is scale-invariant. The perturbations are hardcoded
at ``1e-6`` and ``1e-7`` relative, i.e. inside the leaked ``1e-5`` window and far
outside the strict float64 tolerance of ``1e-15``, so a leaky predicate answers
the perturbed case exactly as it answers the exact one.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from numpy import typing as npt

from pantr._testing import find_knot_index_and_multiplicity as _find_knot_index_and_multiplicity
from pantr._testing import knot_tolerance, unique_knots_and_multiplicity
from pantr.bspline import (
    Bspline,
    BsplineSpace,
    BsplineSpace1D,
    THBSplineSpace,
    create_uniform_space,
)
from pantr.grid import hierarchical_grid, uniform_grid

# Coordinate scales exercised throughout. 1.0 is the regime where the leak is
# invisible; 1e6 is where a 1e-5 relative window is 10.0 wide in absolute terms.
SCALES = (1.0, 100.0, 1e6)

# Relative perturbation sitting strictly inside the leaked ``rtol=1e-5`` window and
# far outside the strict float64 knot tolerance (1e-15).
INSIDE_LEAK = 1e-6


def _in_domain(
    knots: npt.NDArray[np.float64], tol: float, pts: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """Return, per point, whether a degree-2 space with tolerance ``tol`` accepts it.

    The space is built with the stored tolerance ``tol`` (``_from_state``) and the points are
    probed one at a time through ``tabulate_basis``, which refuses a point outside the domain.

    Args:
        knots (npt.NDArray[np.float64]): The knot vector, taken as stored.
        tol (float): The absolute tolerance of the space.
        pts (npt.NDArray[np.float64]): The points.

    Returns:
        npt.NDArray[np.bool_]: ``False`` where the point is refused as outside the domain.
    """
    space = BsplineSpace1D._from_state(knots, 2, False, tol)
    out = np.empty(pts.shape, dtype=np.bool_)
    for i, p in enumerate(pts):
        try:
            space.tabulate_basis(np.array([p]))
        except ValueError as exc:
            if "outside the knot vector domain" not in str(exc):
                raise
            out[i] = False
        else:
            out[i] = True
    return out


class TestMultiplicityCounting:
    """Knot-multiplicity counts must not absorb a near-but-distinct knot."""

    @pytest.mark.parametrize("base", [1.0, 1e6])
    def test_periodic_num_basis_ignores_near_first_knot(self, base: float) -> None:
        """A periodic space's ``num_basis`` must not depend on a ``1e-6``-relative gap.

        The multiplicity of the first knot in the domain counts how many of
        ``knots[:degree+1]`` equal ``knots[degree]``. The leak made two knots a
        relative ``1e-6`` below it count as equal, raising the multiplicity from 1
        to 3, driving the periodic regularity correction from ``degree - 1 == 1``
        to ``-1``, and reporting ``num_basis == 8`` instead of 6.
        """
        knots = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]) * base
        gap = INSIDE_LEAK * base
        knots[0] -= gap
        knots[1] -= gap

        space = BsplineSpace1D(knots, 2, periodic=True, snap_knots=False)

        # len(knots) - degree - 1 == 8 raw, minus (regularity + 1) == 2.
        assert space.num_basis == 6

    @pytest.mark.parametrize("base", SCALES)
    def test_has_open_knots_rejects_a_near_clamped_vector(self, base: float) -> None:
        """A knot vector that is not clamped must not read as clamped at any scale.

        At base 1e6 the leaked window is 10.0 wide, so gaps of a full unit read as
        "the first ``degree + 1`` knots are equal" -- and at a gap of ``0.5 * base``
        it would have accepted half the domain length.
        """
        knots = np.array([1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0]) * base
        knots[0] -= INSIDE_LEAK * base

        space = BsplineSpace1D(knots, 2, snap_knots=False)

        assert space.has_open_knots() is False

    @pytest.mark.parametrize("base", SCALES)
    def test_has_open_knots_accepts_an_exactly_clamped_vector(self, base: float) -> None:
        """The tightened predicate must still accept a genuinely clamped vector."""
        knots = np.array([1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0]) * base

        space = BsplineSpace1D(knots, 2, snap_knots=False)

        assert space.has_open_knots() is True

    @pytest.mark.parametrize("base", SCALES)
    def test_boundary_multiplicities_ignore_a_near_boundary_knot(self, base: float) -> None:
        """The boundary multiplicities must report the true multiplicity at any scale.

        Read as the first and last in-domain multiplicities of the space. The end-run count the
        product's boundary conversion uses (``detail::boundary_multiplicities`` in
        ``include/pantr/bspline/product.hpp``) is pinned on the same data in
        ``cpp/tests/test_bspline_product.cpp``.
        """
        knots = np.array([1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0]) * base
        knots[0] -= INSIDE_LEAK * base
        knots[-1] += INSIDE_LEAK * base
        space = BsplineSpace1D(knots, 2, snap_knots=False)

        _, mults = space.get_unique_knots_and_multiplicity(in_domain=True)

        assert (int(mults[0]), int(mults[-1])) == (2, 2)


class TestDomainGate:
    """The in-domain gate must be an absolute, argument-order-independent window."""

    @pytest.mark.parametrize("hi", SCALES)
    def test_overshoot_inside_the_leaked_window_is_rejected(self, hi: float) -> None:
        """A point overshooting the domain end must be rejected at any scale.

        The leak attached ``rtol`` to the second operand, which was ``knot_end`` at
        the right end, so overshoot of ``1e-5 * knot_end`` was accepted: 1.0e-5 on
        ``[0, 1]``, 1.0e-3 on ``[0, 100]`` and 10.0 on ``[0, 1e6]``.
        """
        knots = np.array([0.0, 0.0, 0.0, hi, hi, hi], dtype=np.float64)
        tol = 1e-10
        pts = np.array([hi + INSIDE_LEAK * hi, hi + 10.0])

        assert not np.any(_in_domain(knots, tol, pts))

    @pytest.mark.parametrize("hi", SCALES)
    def test_gate_is_symmetric_in_undershoot_and_overshoot(self, hi: float) -> None:
        """The window must be exactly ``tol`` wide at both ends.

        The old asymmetry was an accident of argument order: the left end compared
        ``knot_begin`` against ``pts`` and the right end ``pts`` against
        ``knot_end``, so undershoot at a domain starting at 0 was judged on a bare
        ``atol`` while overshoot at the far end got the full relative window.

        ``tol`` is scaled with ``hi`` so the probes stay representable. With a fixed
        ``tol = 1e-10``, ``hi + 0.5 * tol`` rounds back to exactly ``hi`` once
        ``hi = 1e6`` (half a ulp there is 5.8e-11), and the "just inside" probe would
        silently degenerate into an exact-boundary check at the very scale this
        module is about.
        """
        knots = np.array([0.0, 0.0, 0.0, hi, hi, hi], dtype=np.float64)
        tol = 1e-10 * hi
        assert hi + 0.5 * tol != hi, "probe must be representable at this scale"

        just_in = np.array([-0.5 * tol, hi + 0.5 * tol])
        just_out = np.array([-2.0 * tol, hi + 2.0 * tol])

        np.testing.assert_array_equal(_in_domain(knots, tol, just_in), [True, True])
        np.testing.assert_array_equal(_in_domain(knots, tol, just_out), [False, False])

    def test_band_stays_tol_wide_when_tol_falls_below_one_ulp(self) -> None:
        """The admitted band must be ``tol`` wide even where ``tol < ulp(knot_end)``.

        The obvious way to write an absolute gate is to shift the bound,
        ``pts <= knot_end + tol``. In floating point that is a *different* predicate:
        ``knot_end + tol`` rounds to the nearest representable value, so once ``tol``
        falls below ``ulp(knot_end)`` the effective tolerance silently becomes
        ``ulp(knot_end)``. On a domain ending at 1e6 that ulp is 1.164e-10, so a
        requested ``tol`` of 1e-10 would admit points 1.164e-10 past the end.

        Forming the difference first and only then comparing against ``tol`` avoids
        it; the subtraction is exact for the nearby points that decide the boundary.
        This is the same shape of defect as the leak the rest of this module fixes --
        a tolerance quietly becoming something other than what was asked for -- at one
        ulp instead of 1e-5 relative.
        """
        hi = 1e6
        tol = 1e-10
        knots = np.array([0.0, 0.0, 0.0, hi, hi, hi], dtype=np.float64)
        assert np.spacing(hi) > tol, "construction requires tol below one ulp of hi"

        # One ulp past the end: admitted by the shifted-bound form, outside the true
        # tol-wide band.
        just_past = np.array([np.nextafter(hi, np.inf)])
        assert float(just_past[0]) - hi > tol

        assert not bool(_in_domain(knots, tol, just_past)[0])

    @pytest.mark.parametrize("hi", SCALES)
    def test_evaluate_rejects_an_out_of_domain_point(self, hi: float) -> None:
        """The public evaluation path inherits the tightened gate."""
        knots = np.array([0.0, 0.0, 0.0, hi, hi, hi], dtype=np.float64)
        spline = Bspline(
            BsplineSpace([BsplineSpace1D(knots, 2)]),
            np.array([[0.0], [1.0], [0.0]]),
        )

        with pytest.raises(ValueError, match="outside the knot vector domain"):
            spline.evaluate(np.array([hi + INSIDE_LEAK * hi]))


class TestCardinalClassification:
    """A near-uniform knot vector is not a uniform one."""

    # ``knots[9]`` is moved. The kernel judges interval ``k`` (0-based, over the
    # 8 in-domain intervals of the vector below) on the ``2 * degree - 1 == 5``
    # knot-interval window ``knots[1 + k : 7 + k]``, which contains index 9 exactly
    # for ``k >= 3``. So the first three intervals stay cardinal and the rest do not
    # -- a localized pattern, which a blanket all-``True`` or all-``False`` answer
    # cannot imitate.
    EXPECTED: ClassVar[list[bool]] = [True, True, True, False, False, False, False, False]

    @staticmethod
    def _perturbed_uniform(rel: float) -> BsplineSpace1D:
        """Degree-3 uniform space on ``[3, 11]`` with ``knots[9]`` moved by ``rel``."""
        knots = np.arange(0.0, 15.0)
        knots[9] += rel
        return BsplineSpace1D(knots, 3, snap_knots=False)

    def test_uniform_knots_are_all_cardinal(self) -> None:
        """Baseline: the unperturbed uniform vector really is cardinal throughout."""
        space = self._perturbed_uniform(0.0)

        np.testing.assert_array_equal(space.get_cardinal_intervals(), [True] * 8)

    @pytest.mark.parametrize("rel", [1e-7, INSIDE_LEAK, 3e-6])
    def test_near_uniform_knots_are_not_cardinal(self, rel: float) -> None:
        """Moving one knot by ``rel`` must un-cardinal every window that sees it.

        The leak compared ``lengths`` against a reference of magnitude 1, so the
        leaked window was ``1e-5`` wide and every ``rel`` here was absorbed: all
        eight intervals came back cardinal.
        """
        space = self._perturbed_uniform(rel)

        np.testing.assert_array_equal(space.get_cardinal_intervals(), self.EXPECTED)

    @pytest.mark.parametrize("rel", [1e-7, INSIDE_LEAK, 3e-6])
    def test_near_uniform_operators_are_not_replaced_by_the_identity(self, rel: float) -> None:
        """A misclassified interval had its exact operator discarded for ``np.eye``.

        The cardinal extraction operators (``cardinal_extraction_1d``) compute the exact
        operator and then overwrite it with the identity on every interval flagged
        cardinal, so a misclassification did not merely mislabel: it substituted a
        wrong operator for a correct one that had already been computed two lines
        earlier.
        """
        space = self._perturbed_uniform(rel)
        ops = space.tabulate_cardinal_extraction_operators()
        identity = np.eye(4)

        is_identity = [bool(np.array_equal(ops[i], identity)) for i in range(ops.shape[0])]
        assert is_identity == self.EXPECTED

    def test_uniform_operators_are_the_identity(self) -> None:
        """Baseline: a genuinely cardinal interval does get the identity."""
        space = self._perturbed_uniform(0.0)
        ops = space.tabulate_cardinal_extraction_operators()

        for i in range(ops.shape[0]):
            np.testing.assert_array_equal(ops[i], np.eye(4))


class TestKnotLookup:
    """Locating a knot by value must not silently pick a neighbour."""

    @pytest.mark.parametrize("base", SCALES)
    def test_near_miss_is_not_matched(self, base: float) -> None:
        """A query a relative ``1e-6`` away from every knot must not be found.

        The leak matched a knot up to ``1e-5 * |knot_value|`` away, so
        ``remove_knots(v)`` removed a *different* knot than requested: up to
        1.0e-5 away at a knot of 1.0, and 10.0 away at a knot of 1e6.
        """
        knots = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0]) * base
        query = 1.0 * base + INSIDE_LEAK * base

        with pytest.raises(ValueError, match="not found in knot vector"):
            _find_knot_index_and_multiplicity(knots, 2, query, 1e-15)

    @pytest.mark.parametrize("base", SCALES)
    def test_exact_hit_is_matched(self, base: float) -> None:
        """The tightened lookup must still find the knot it is given exactly."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0]) * base

        r, s = _find_knot_index_and_multiplicity(knots, 2, 1.0 * base, 1e-15)

        assert (r, s) == (3, 1)

    @pytest.mark.parametrize("base", SCALES)
    def test_remove_knots_rejects_a_near_miss(self, base: float) -> None:
        """Public ``remove_knots`` surfaces the tightened lookup as an error."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0]) * base
        space = BsplineSpace1D(knots, 2, snap_knots=False)
        n = space.num_basis
        ctrl = np.stack([np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n) ** 2], axis=1)
        spline = Bspline(BsplineSpace([space]), ctrl)

        with pytest.raises(ValueError, match="not found in knot vector"):
            spline.remove_knots(1.0 * base + INSIDE_LEAK * base)


class TestSplitMultiplicity:
    """``Bspline.split`` must clamp the two pieces at the split value."""

    @staticmethod
    def _curve(offset: float, degree: int = 2, n_int: int = 6) -> Bspline:
        """Degree-2 curve on ``[offset, offset + n_int]`` with unit knot spacing."""
        knots = np.concatenate(
            [
                np.full(degree + 1, offset),
                np.arange(1.0, n_int) + offset,
                np.full(degree + 1, offset + n_int),
            ]
        )
        space = BsplineSpace1D(knots, degree)
        n = space.num_basis
        ctrl = np.stack([np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n) ** 2], axis=1)
        return Bspline(BsplineSpace([space]), ctrl)

    @pytest.mark.parametrize("offset", [0.0, 1e5, 1e6])
    def test_left_piece_interpolates_its_last_control_point(self, offset: float) -> None:
        """The left piece must end exactly at its own last control point.

        ``_split_bspline_1d_impl`` counts the split value's current multiplicity to
        decide how many knots to insert. The leak inflated that count -- measured 3
        instead of 1 at offset 1e5, 18 at 1e6 and 45 at 1e8 -- so the deficit
        collapsed to zero, no knots were inserted, and the split point kept
        multiplicity 1 instead of ``degree + 1``. The left piece was then not
        clamped there, and stopped interpolating its own last control point (a
        measured 1.22e-2 discrepancy at offset 1e5).
        """
        spline = self._curve(offset)
        value = float(spline.space.spaces[0].domain[0]) + 5.0

        left, _ = spline.split(0, value)

        np.testing.assert_allclose(
            np.asarray(left.evaluate(np.array([value]))),
            np.asarray(left.control_points)[-1],
            atol=1e-12,
            rtol=0.0,
        )

    @pytest.mark.parametrize("offset", [0.0, 1e5, 1e6])
    def test_split_point_reaches_full_multiplicity(self, offset: float) -> None:
        """The split value must end up clamped in both pieces."""
        spline = self._curve(offset)
        value = float(spline.space.spaces[0].domain[0]) + 5.0

        left, right = spline.split(0, value)

        assert left.space.spaces[0].has_open_knots() is True
        assert right.space.spaces[0].has_open_knots() is True


class TestRestrictBounds:
    """Restriction bounds must be snapped and validated absolutely.

    These exercise ``Bspline.restrict`` itself (they exercised the Layer-2 helpers
    ``_validate_restrict_bounds`` and ``_compute_boundary_knots_to_insert`` before those moved
    into the compiled library): the snapping shows in the domain of the restricted B-spline, and
    the knots that needed inserting show in its knot vector. The tolerance is the space's own,
    ``8 eps max(span, |knots[0]|, |knots[-1]|)``.
    """

    @staticmethod
    def _knots(base: float) -> npt.NDArray[np.float64]:
        """Clamped degree-2 knot vector on ``[base, 3 * base]``."""
        return np.array([1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]) * base

    @staticmethod
    def _curve(base: float) -> Bspline:
        """A degree-2 curve on the knots of :meth:`_knots`."""
        knots = TestRestrictBounds._knots(base)
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        return Bspline(space, np.arange(4.0).reshape(4, 1))

    @pytest.mark.parametrize("base", SCALES)
    def test_a_bound_inside_the_domain_is_not_snapped_to_the_end(self, base: float) -> None:
        """A bound a relative ``1e-6`` inside the domain must stay where it is.

        The leak read such a bound as coinciding with the domain endpoint, which
        both snapped it to the wrong value and -- when both bounds snapped -- made
        ``restrict`` raise "Bounds match the full domain and the direction is
        already open" for a perfectly valid request.
        """
        knots = self._knots(base)
        a, b = float(knots[2]), float(knots[-3])
        a_new = a + INSIDE_LEAK * base
        b_new = b - INSIDE_LEAK * base

        domain = self._curve(base).restrict((a_new, b_new)).space.spaces[0].domain

        assert (float(domain[0]), float(domain[1])) == (a_new, b_new)

    @pytest.mark.parametrize("base", SCALES)
    def test_a_bound_at_the_domain_end_is_still_snapped(self, base: float) -> None:
        """A bound within the tolerance of an end is that end, exactly.

        The tightened snap must still fire: a bound half a tolerance off each end (inside it,
        and far inside the leaked window) lands on the stored end value.
        """
        knots = self._knots(base)
        a, b = float(knots[2]), float(knots[-3])
        tol = knot_tolerance(knots)
        mid = 2.0 * base

        lower = self._curve(base).restrict((a + 0.5 * tol, mid)).space.spaces[0].domain
        upper = self._curve(base).restrict((mid, b - 0.5 * tol)).space.spaces[0].domain

        assert float(lower[0]) == a
        assert float(upper[1]) == b

    @pytest.mark.parametrize("base", SCALES)
    def test_interior_bounds_still_require_boundary_knots(self, base: float) -> None:
        """Bounds a relative ``1e-6`` inside the domain still need clamping knots.

        With the leak, both bounds were taken to be at the domain ends of an
        already-open vector, so no knots were queued and the caller raised instead.
        """
        knots = self._knots(base)
        a, b = float(knots[2]), float(knots[-3])
        a_new = a + INSIDE_LEAK * base
        b_new = b - INSIDE_LEAK * base

        restricted = self._curve(base).restrict((a_new, b_new))

        # Neither bound coincides with an existing knot, so each gets degree + 1 copies; the
        # interior knot stays.
        np.testing.assert_array_equal(
            restricted.space.spaces[0].knots, [a_new] * 3 + [2.0 * base] + [b_new] * 3
        )


class TestTHBRootBounds:
    """The THB grid/space domain-consistency check needs a real tolerance."""

    def test_mismatched_root_bounds_are_rejected(self) -> None:
        """A factor-of-two wrong domain must be rejected.

        ``np.allclose`` with no tolerance argument at all applies numpy's defaults
        (``rtol=1e-5``, ``atol=1e-8``). Through the ``atol`` leg, ``[0, 1e-9]``
        against ``[0, 2e-9]`` -- twice the domain -- compared equal.
        """
        root = create_uniform_space(2, 4, domain=(0.0, 1e-9))
        grid = hierarchical_grid(uniform_grid([[0.0, 2e-9]], 4), 2)

        with pytest.raises(ValueError, match="grid root bounds must match"):
            THBSplineSpace(root, grid)

    def test_relatively_mismatched_root_bounds_are_rejected(self) -> None:
        """A 3e-6 relative domain mismatch must be rejected, at any scale."""
        root = create_uniform_space(2, 4, domain=(0.0, 1.0))
        grid = hierarchical_grid(uniform_grid([[0.0, 1.0 + 3e-6]], 4), 2)

        with pytest.raises(ValueError, match="grid root bounds must match"):
            THBSplineSpace(root, grid)

    def test_matching_root_bounds_are_accepted(self) -> None:
        """The tightened check must still accept a consistent pair."""
        root = create_uniform_space(2, 4, domain=(0.0, 1e-9))
        grid = hierarchical_grid(uniform_grid([[0.0, 1e-9]], 4), 2)

        space = THBSplineSpace(root, grid)

        assert space.dim == 1


class TestDerivedKnotTolerance:
    """The absolute knot tolerance itself: its formula, and what rests on it.

    Everything above tests a *consequence* of ``BsplineSpace1D.tolerance``. These
    test the quantity, so that a one-line slip in the formula -- dropping the
    coordinate-magnitude term, taking a ``min`` where a ``max`` belongs -- surfaces
    here rather than several steps downstream as a puzzling merge verdict.
    """

    @staticmethod
    @pytest.mark.parametrize(
        ("dtype", "knots", "expected_scale", "why"),
        [
            (np.float64, [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0], 2.0, "span wins"),
            (np.float64, [-3.0, -3.0, -3.0, -2.0, -1.0, -1.0, -1.0], 3.0, "|knots[0]| wins"),
            (
                np.float64,
                [1e6, 1e6, 1e6, 1e6 + 0.5, 1e6 + 1, 1e6 + 1, 1e6 + 1],
                1e6 + 1.0,
                "|knots[-1]| wins over a span of 1",
            ),
            (np.float32, [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], 1.0, "the same in float32"),
        ],
    )
    def test_tolerance_is_eight_epsilons_of_the_knot_magnitude(
        dtype: type[np.floating], knots: list[float], expected_scale: float, why: str
    ) -> None:
        """``tolerance == 8 * eps(dtype) * max(span, |knots[0]|, |knots[-1]|)``.

        ``expected_scale`` is written out per case rather than computed by the
        helper under test, so the two cannot agree by sharing a bug.
        """
        space = BsplineSpace1D(np.asarray(knots, dtype=dtype), 2)
        eps = float(np.finfo(dtype).eps)
        assert space.tolerance == 8.0 * eps * expected_scale, why

    @staticmethod
    @pytest.mark.parametrize("lam", [1e-6, 1.0, 1e6])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_merge_verdict_is_scale_covariant(dtype: type[np.floating], lam: float) -> None:
        """The same gap-to-domain ratio gets the same verdict at every scale.

        This is the central claim of the derived tolerance, and it is the one the
        previous absolute constant failed: it merged every knot of this fixture at
        ``lam = 1e-6`` and none of them at ``lam = 1e6``.
        """
        for ratio, want in ((1e-15, [3, 2, 3]), (1e-3, [3, 1, 1, 3])):
            mid = dtype(0.5 * lam)
            knots = np.asarray(
                [0.0, 0.0, 0.0, mid, dtype(float(mid) * (1.0 + ratio)), lam, lam, lam],
                dtype=dtype,
            )
            space = BsplineSpace1D(knots, 2)
            _, mult = space.get_unique_knots_and_multiplicity()
            assert mult.tolist() == want, (
                f"gap ratio {ratio:g} at lam={lam:g} in {np.dtype(dtype).name} gave "
                f"multiplicities {mult.tolist()}, expected {want}"
            )

    @staticmethod
    def test_snapping_is_idempotent() -> None:
        """Snapping a snapped vector must not move it again, bit for bit.

        The exact ``denom == 0.0`` Cox-de Boor guard wants a knot vector that is a
        fixed point of the grouping, and electing each class's first knot rather
        than its mean is what delivers one. An average would be a fresh rounding
        and could place two neighbouring representatives back within tolerance.
        """
        base = 1000.0
        eps = float(np.finfo(np.float64).eps)
        near = 8.0 * eps * base * 0.25  # a quarter of the merge tolerance
        raw = np.array(
            [
                0.0,
                0.0,
                0.0,
                250.0,
                250.0 + near,
                250.0 + 2 * near,
                500.0,
                750.0 - near,
                750.0,
                base,
                base,
                base,
            ],
            dtype=np.float64,
        )
        once = np.asarray(BsplineSpace1D(raw, 2).knots)
        assert not np.array_equal(once, raw), "precondition: the fixture must actually merge"

        twice = np.asarray(BsplineSpace1D(once, 2).knots)
        assert np.array_equal(once, twice), (
            f"snapping moved an already-snapped vector: {once.tolist()} -> {twice.tolist()}"
        )

    @staticmethod
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_classes_partition_the_knot_vector(dtype: type[np.floating]) -> None:
        """``np.repeat(unique, mults)`` rebuilds the whole vector, as snapping needs.

        The grouping must be exactly the one the C++ space snaps with (each class onto its
        first knot), so if the multiplicities stopped summing to ``knots.size`` a vector
        rebuilt from them would change length. Checked directly on the knot-class kernel,
        with ``in_domain=False``.
        """
        raw = np.asarray([0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0], dtype=dtype)
        space = BsplineSpace1D(raw, 2)
        knots = np.asarray(space.knots)
        unique, mults = unique_knots_and_multiplicity(knots, space.degree, space.tolerance, False)
        assert int(mults.sum()) == knots.size
        assert np.array_equal(np.repeat(np.asarray(unique), mults), knots)
        assert np.all(np.isin(np.asarray(unique), knots)), "reported knots must be stored knots"

    @staticmethod
    def test_a_knot_vector_with_no_scale_compares_exactly() -> None:
        """An identically-zero knot vector leaves no magnitude to be relative to.

        ``knot_scale`` returns 0.0 and the tolerance with it, so grouping falls back
        to exact equality. That is the only defensible answer -- there is no scale to
        read off the input -- and the vector is all one knot either way. Pinned
        because a zero tolerance is the kind of degenerate value that invites a
        well-meaning floor.

        Exercised through the knot layer rather than through a space, because
        ``knot_scale`` is ``max(span, |knots[0]|, |knots[-1]|)`` and so vanishes only
        when the whole vector is zero -- which is exactly the zero-interval case
        ``BsplineSpace1D`` refuses since issue #320. The scale-free regime is
        therefore reachable in the knot layer and nowhere above it, and that is a
        property of the derivation worth stating rather than a gap in the test.
        """
        knots = np.zeros(8, dtype=np.float64)
        assert knot_tolerance(knots) == 0.0

        unique, mults = unique_knots_and_multiplicity(knots, 3, 0.0, False)
        assert unique.tolist() == [0.0]
        assert mults.tolist() == [8]

        with pytest.raises(ValueError, match="spans no interval"):
            BsplineSpace1D(knots, 3)
