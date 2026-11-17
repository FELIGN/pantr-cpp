"""Oracles for the C++ THB port, none of which is computed by the code under test.

The THB space, spline, quasi-interpolant, multilevel extraction and windowing functions run in
C++. This file checks them against routes that share nothing with that code:

* the truncated basis against the Giannelli-Jüttler-Speleers definition worked in exact rational
  arithmetic (Cox-de Boor and Boehm insertion, :mod:`tests.oracles`), for two levels;
* the partition of unity and non-negativity of the truncated basis, with the plain hierarchical
  basis as the control that the check can fail;
* reproduction by the quasi-interpolant: of a random THB spline (the coefficients are the answer)
  and of a polynomial of the space's degree (the polynomial is the answer), with the
  non-truncated basis as the control;
* the refinement relation: a coarse function equals its prolongation image in the refined space,
  a fully refined space is the finest tensor-product space, and every THB function lies in the span
  of the finest level's functions;
* the halo, the dof owner and the coupling graph against the overlap of each function's support,
  computed from the knots of its level space and the bounds of the cells, never from
  ``active_basis``.

Tolerances come from the project's tiers (``pantr.tolerance``) with the reason beside each.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

import numpy as np
import numpy.typing as npt
import pytest

from pantr._testing import prolongation_residual_tolerance, thb_dof_owner, thb_halo
from pantr.bspline import (
    THBSpline,
    THBSplineSpace,
    coupling_graph,
    create_thb_space,
    create_uniform_space,
    quasi_interpolate_thb_spline,
)
from pantr.grid import Partition, hierarchical_grid, tensor_product_grid, uniform_grid
from pantr.tolerance import get_conservative
from tests._thb_support import (
    VALUE_ATOL,
    hierarchical_rows,
    random_cell_points,
    tensor_product_rows,
)
from tests.oracles import bspline_exact, knot_insertion_exact

_ACCUMULATION_ATOL = float(get_conservative(np.float64))
"""Absolute bound on a quantity summed over many basis values or solved for by many samples.

The partition-of-unity sum adds at most ``max_active_per_cell`` values, each already a
contraction, and the quasi-interpolant contracts ``(p + 1) ** d`` samples with weights of order
one to a few tens: a long accumulation, which is what the ``get_conservative`` tier (4096 eps)
states. Scaled by ``max(1, |largest value|)`` where the quantity is not bounded by one.
"""

_VISIBLE = 1.0e-4
"""A gap between two functions that cannot be rounding.

The control checks (plain hierarchical basis against the partition of unity, non-truncated
quasi-interpolation against reproduction, exact truncation against the plain B-spline, a kink
against the finest spline space) fail by amounts of the order of a refinement coefficient, or of
the squared cell width times a slope jump: orders of magnitude above ``_ACCUMULATION_ATOL`` and
well above this value, which sits between the two.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures: refined spaces, built fresh on every call (a grid is mutable)
# ──────────────────────────────────────────────────────────────────────────────


def _space_1d(*, truncate: bool = True) -> THBSplineSpace:
    """Degree 2 on eight cells: two separated level-0 regions and one level-1 region."""
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 8), 2)
    grid.refine(0, [0], [3])
    grid.refine(0, [5], [7])
    grid.refine(1, [0], [2])
    return THBSplineSpace(create_uniform_space(2, 8), grid, truncate=truncate)


def _space_2d(*, truncate: bool = True) -> THBSplineSpace:
    """Degree (2, 2) on 6 x 6 cells with three levels."""
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]] * 2, 6), 2)
    grid.refine(0, [1, 1], [4, 4])
    grid.refine(1, [2, 2], [7, 7])
    return THBSplineSpace(create_uniform_space([2, 2], [6, 6]), grid, truncate=truncate)


def _space_2d_anisotropic(*, truncate: bool = True) -> THBSplineSpace:
    """Degrees (2, 3) on 5 x 4 cells with three levels."""
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]] * 2, [5, 4]), 2)
    grid.refine(0, [0, 1], [3, 3])
    grid.refine(1, [1, 2], [4, 5])
    return THBSplineSpace(create_uniform_space([2, 3], [5, 4]), grid, truncate=truncate)


def _space_3d(*, truncate: bool = True) -> THBSplineSpace:
    """Degrees (2, 2, 1) on 3 x 3 x 3 cells with two levels."""
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]] * 3, 3), 2)
    grid.refine(0, [0, 0, 0], [2, 2, 2])
    return THBSplineSpace(create_uniform_space([2, 2, 1], [3, 3, 3]), grid, truncate=truncate)


def _space_factor_3(*, truncate: bool = True) -> THBSplineSpace:
    """Degree 2 on four cells, each subdivided into three: a refinement factor that is not 2."""
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 3)
    grid.refine(0, [1], [3])
    return THBSplineSpace(create_uniform_space(2, 4), grid, truncate=truncate)


_SPACES = [
    pytest.param(_space_1d, id="1d_holes"),
    pytest.param(_space_2d, id="2d_three_levels"),
    pytest.param(_space_2d_anisotropic, id="2d_degrees_2_3"),
    pytest.param(_space_3d, id="3d"),
    pytest.param(_space_factor_3, id="1d_factor_3"),
]
"""The fixtures every property test runs over, each a builder taking ``truncate``."""

_Builder = Callable[..., THBSplineSpace]


def _sample_points(thb: THBSplineSpace, per_cell: int, seed: int) -> npt.NDArray[np.float64]:
    """Return ``per_cell`` seeded random points in every active cell, corners included."""
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [random_cell_points(thb, cid, per_cell, rng) for cid in range(thb.grid.num_cells)]
    )


# ──────────────────────────────────────────────────────────────────────────────
# The truncated basis against the exact definition (two levels)
# ──────────────────────────────────────────────────────────────────────────────


def _two_level_1d(*, truncate: bool = True) -> THBSplineSpace:
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 2)
    grid.refine(0, [0], [2])
    return THBSplineSpace(create_uniform_space(2, 4), grid, truncate=truncate)


def _two_level_2d_corner(*, truncate: bool = True) -> THBSplineSpace:
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]] * 2, 4), 2)
    grid.refine(0, [0, 0], [2, 2])
    return THBSplineSpace(create_uniform_space([2, 2], [4, 4]), grid, truncate=truncate)


def _two_level_2d_interior(*, truncate: bool = True) -> THBSplineSpace:
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]] * 2, [4, 8]), 2)
    grid.refine(0, [1, 2], [3, 6])
    return THBSplineSpace(create_uniform_space([2, 3], [4, 8]), grid, truncate=truncate)


def _inserted_knots(
    coarse: npt.NDArray[np.float64], fine: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return the knots that, inserted into ``coarse``, give ``fine`` (a multiset difference)."""
    values, counts = np.unique(fine, return_counts=True)
    have = np.array([np.count_nonzero(coarse == v) for v in values])
    return np.repeat(values, counts - have)


class _ExactTwoLevels:
    """The two-level truncated basis worked in exact arithmetic.

    For a space with levels 0 and 1, the truncated function of a level-0 function ``N`` is
    ``N - sum_{j active at level 1} c_j N^1_j``, where ``c`` are the exact refinement coefficients
    of ``N`` in the level-1 basis (Boehm insertion of the new knots, a tensor product of the 1D
    matrices), and the level-1 functions are not truncated. The values are the Cox-de Boor
    recursion in :class:`fractions.Fraction`.
    """

    def __init__(self, thb: THBSplineSpace) -> None:
        assert thb.num_levels == 2
        self._thb = thb
        self._space = [thb.level_space(0), thb.level_space(1)]
        self._knots = [
            [np.asarray(sp.knots, dtype=np.float64) for sp in lv.spaces] for lv in self._space
        ]
        # alpha[k][j][i]: the coefficient of the level-1 function j in the refinement of the
        # level-0 function i, in direction k.
        self._alpha = [
            knot_insertion_exact.oslo_matrix(
                self._knots[0][k],
                thb.degrees[k],
                _inserted_knots(self._knots[0][k], self._knots[1][k]),
            )
            for k in range(thb.dim)
        ]
        self._offsets = np.concatenate([[0], np.cumsum(thb.num_basis_per_level)])
        self._active = [
            np.array(
                [
                    np.unravel_index(int(f), self._space[lv].num_basis)
                    for f in thb.active_function_indices(lv)
                ],
                dtype=np.int64,
            ).reshape(-1, thb.dim)
            for lv in range(2)
        ]

    def values(self, pt: npt.NDArray[np.float64], dofs: npt.NDArray[np.int64]) -> list[Fraction]:
        """Return the exact values at ``pt`` of the functions ``dofs`` of the truncated basis."""
        dim = self._thb.dim
        basis = [
            [
                bspline_exact.ExactBasis(self._knots[lv][k], self._thb.degrees[k], pt[k])
                for k in range(dim)
            ]
            for lv in range(2)
        ]

        def plain(lv: int, multi: npt.NDArray[np.int64]) -> Fraction:
            out = Fraction(1)
            for k in range(dim):
                out *= basis[lv][k].value(int(multi[k]))
            return out

        fine_values = [plain(1, m) for m in self._active[1]]
        result = []
        for dof in dofs:
            level = int(np.searchsorted(self._offsets, dof, side="right")) - 1
            multi = self._active[level][int(dof - self._offsets[level])]
            value = plain(level, multi)
            if level == 0 and self._thb.truncate:
                for j, fine_multi in enumerate(self._active[1]):
                    coeff = Fraction(1)
                    for k in range(dim):
                        coeff *= self._alpha[k][int(fine_multi[k])][int(multi[k])]
                    value -= coeff * fine_values[j]
            result.append(value)
        return result


class TestTruncationAgainstExactRefinement:
    """The values of a two-level THB or HB basis equal the exact definition, point by point.

    Oracle: Cox-de Boor and Boehm knot insertion in rational arithmetic (``tests.oracles``),
    which share no code with the library. Bug caught: a wrong refinement coefficient, a component
    left on (or removed from) a finer active function, a wrong support at the interface, in the
    C++ truncation or its evaluation. Control: for the truncated space the exact truncated values
    differ from the plain B-spline values by a clear margin, so the comparison can fail.
    """

    @pytest.mark.parametrize("truncate", [True, False], ids=["thb", "hb"])
    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(_two_level_1d, id="1d"),
            pytest.param(_two_level_2d_corner, id="2d_corner"),
            pytest.param(_two_level_2d_interior, id="2d_interior_degrees_2_3"),
        ],
    )
    def test_tabulated_values_equal_the_exact_values(self, build: _Builder, truncate: bool) -> None:
        thb = build(truncate=truncate)
        exact = _ExactTwoLevels(thb)
        plain = _ExactTwoLevels(build(truncate=False))
        rng = np.random.default_rng(7)
        worst = 0.0
        gap_to_plain = 0.0
        for cid in range(thb.grid.num_cells):
            pts = random_cell_points(thb, cid, 3, rng)
            values, dofs = thb.tabulate_basis(cid, pts)
            for row, pt in zip(values, pts, strict=True):
                want = np.array([float(w) for w in exact.values(pt, dofs)])
                untruncated = np.array([float(w) for w in plain.values(pt, dofs)])
                worst = max(worst, float(np.abs(want - row).max()))
                gap_to_plain = max(gap_to_plain, float(np.abs(want - untruncated).max()))
        assert worst <= VALUE_ATOL, f"max |tabulated - exact| = {worst:.3e} > {VALUE_ATOL:.3e}"
        if truncate:
            assert gap_to_plain >= _VISIBLE, "the exact truncation equals the plain B-splines"


# ──────────────────────────────────────────────────────────────────────────────
# Partition of unity
# ──────────────────────────────────────────────────────────────────────────────


class TestPartitionOfUnity:
    """The truncated basis sums to one at random points of every cell; the plain one does not.

    Oracle: the invariant itself (Giannelli-Jüttler-Speleers 2012: the truncated basis is a
    non-negative partition of unity), whose error is bounded by the long-accumulation tier.
    Bug caught: a truncation coefficient wrong or skipped at some level, a function missing from
    a cell's active set (the sum drops below one), one listed twice. Control: the plain
    hierarchical basis of the same grid exceeds one by a clear margin where levels overlap.
    """

    @pytest.mark.parametrize("build", _SPACES)
    def test_truncated_basis_is_a_non_negative_partition_of_unity(self, build: _Builder) -> None:
        thb = build()
        rng = np.random.default_rng(21)
        worst = 0.0
        lowest = 0.0
        for cid in range(thb.grid.num_cells):
            values, _ = thb.tabulate_basis(cid, random_cell_points(thb, cid, 8, rng))
            worst = max(worst, float(np.abs(values.sum(axis=1) - 1.0).max()))
            lowest = min(lowest, float(values.min()))
        assert worst <= _ACCUMULATION_ATOL, f"max |sum - 1| = {worst:.3e}"
        assert lowest >= -_ACCUMULATION_ATOL, f"lowest value {lowest:.3e}"

    @pytest.mark.parametrize("build", _SPACES)
    def test_the_plain_hierarchical_basis_exceeds_one_where_levels_overlap(
        self, build: _Builder
    ) -> None:
        hb = build(truncate=False)
        rng = np.random.default_rng(21)
        excess = max(
            float(
                (
                    hb.tabulate_basis(cid, random_cell_points(hb, cid, 8, rng))[0].sum(axis=1) - 1.0
                ).max()
            )
            for cid in range(hb.grid.num_cells)
        )
        assert excess >= _VISIBLE, f"largest sum - 1 = {excess:.3e}"


# ──────────────────────────────────────────────────────────────────────────────
# Quasi-interpolation reproduces what the space contains
# ──────────────────────────────────────────────────────────────────────────────


def _polynomial(
    degrees: tuple[int, ...],
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Return ``prod_k (x_k + 1/2) ** degrees[k]``, a polynomial of the space's degree."""
    return lambda x: np.prod((x + 0.5) ** np.asarray(degrees), axis=1)


class TestQuasiInterpolationReproduction:
    """The THB quasi-interpolant reproduces what the truncated space contains.

    Oracles: for a random THB spline the answer is the coefficient vector that made it (an
    invertible construction, the quasi-interpolant is a projector); for a polynomial of the
    space's degree in every direction it is the polynomial, in closed form. Both are contained in
    the THB space, which holds the level-0 tensor-product polynomials. Bug caught: a functional
    sampled on the wrong cell or level (a truncated function's coefficient off), a weight row
    wrong, rank mixed up. Control: on the plain hierarchical basis, which is not a partition of
    unity, the same quasi-interpolant is not exact, so these checks can fail.
    """

    @pytest.mark.parametrize("rank", [0, 2], ids=["scalar", "vector"])
    @pytest.mark.parametrize("build", _SPACES)
    def test_reproduces_a_random_thb_spline(self, build: _Builder, rank: int) -> None:
        thb = build()
        shape = (thb.num_total_basis,) if rank == 0 else (thb.num_total_basis, rank)
        coeffs = np.random.default_rng(31).uniform(-1.0, 1.0, shape)
        target = THBSpline(thb, coeffs)

        recovered = quasi_interpolate_thb_spline(target.evaluate, thb)

        assert recovered.control_points.shape == coeffs.shape
        err = float(np.abs(recovered.control_points - coeffs).max())
        assert err <= _ACCUMULATION_ATOL, f"max coefficient error {err:.3e}"
        pts = np.random.default_rng(32).uniform(size=(50, thb.dim))
        np.testing.assert_allclose(
            recovered.evaluate(pts), target.evaluate(pts), rtol=0.0, atol=_ACCUMULATION_ATOL
        )

    @pytest.mark.parametrize("build", _SPACES)
    def test_reproduces_a_polynomial_of_the_spaces_degree(self, build: _Builder) -> None:
        thb = build()
        poly = _polynomial(thb.degrees)
        pts = np.random.default_rng(33).uniform(size=(200, thb.dim))

        recovered = quasi_interpolate_thb_spline(poly, thb)

        want = poly(pts)
        err = float(np.abs(recovered.evaluate(pts) - want).max())
        bound = _ACCUMULATION_ATOL * max(1.0, float(np.abs(want).max()))
        assert err <= bound, f"max |QI - polynomial| = {err:.3e} > {bound:.3e}"

    def test_reproduces_a_vector_valued_polynomial(self) -> None:
        thb = _space_2d_anisotropic()
        poly = _polynomial(thb.degrees)

        def field(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.stack([poly(x), 1.0 + 2.0 * x[:, 0] - x[:, 1]], axis=1)

        pts = np.random.default_rng(34).uniform(size=(100, 2))
        recovered = quasi_interpolate_thb_spline(field, thb)
        assert recovered.rank == 2
        want = field(pts)
        err = float(np.abs(recovered.evaluate(pts) - want).max())
        assert err <= _ACCUMULATION_ATOL * max(1.0, float(np.abs(want).max()))

    @pytest.mark.parametrize("build", _SPACES)
    def test_the_plain_hierarchical_quasi_interpolant_is_not_exact(self, build: _Builder) -> None:
        """Control: the constant 1 comes back as the over-unit sum of the plain basis."""
        hb = build(truncate=False)
        recovered = quasi_interpolate_thb_spline(lambda x: np.ones(x.shape[0]), hb)
        pts = np.random.default_rng(35).uniform(size=(400, hb.dim))
        assert np.abs(recovered.evaluate(pts) - 1.0).max() >= _VISIBLE


# ──────────────────────────────────────────────────────────────────────────────
# The refinement relation
# ──────────────────────────────────────────────────────────────────────────────


class TestRefinementRelation:
    """Coarse functions are combinations of finer ones, and the finest level contains them all.

    Oracles: the nestedness of the spaces, read through two independent evaluations (the coarse
    and the fine space's own ``tabulate_basis``) and, for a fully refined grid, through the
    finest tensor-product space's ``BsplineSpace.tabulate_basis``. Bug caught: a prolongation
    column wrong or missing (the identity ``A_fine P = A_coarse`` fails at a point), a refined
    space that is not the finest tensor-product space, an active function that is not in it.
    """

    @pytest.mark.parametrize("truncate", [True, False], ids=["thb", "hb"])
    @pytest.mark.parametrize("build", _SPACES)
    def test_a_coarse_function_equals_its_prolongation_image(
        self, build: _Builder, truncate: bool
    ) -> None:
        coarse = build(truncate=truncate)
        marked = np.random.default_rng(41).choice(
            coarse.grid.num_cells, size=max(1, coarse.grid.num_cells // 4), replace=False
        )
        fine = coarse.refine(marked, admissible_class=None)
        assert fine.grid.num_cells > coarse.grid.num_cells

        prolongation = coarse.prolongation_to(fine)

        pts = _sample_points(fine, 3, seed=42)
        coarse_rows = hierarchical_rows(coarse, pts)
        fine_rows = hierarchical_rows(fine, pts)
        gap = float(np.abs(fine_rows @ prolongation - coarse_rows).max())
        bound = prolongation_residual_tolerance(1.0)
        assert gap <= bound, f"max |A_fine P - A_coarse| = {gap:.3e} > {bound:.3e}"
        assert float(np.abs(coarse_rows).max()) > 0.5, "the check ran over empty rows"

    @pytest.mark.parametrize(
        ("degrees", "cells", "factor"),
        [
            pytest.param([2], [4], 2, id="1d"),
            pytest.param([2, 3], [3, 2], 2, id="2d_degrees_2_3"),
            pytest.param([2, 2], [2, 2], [2, 3], id="2d_factors_2_3"),
            pytest.param([1, 2, 2], [2, 2, 2], 2, id="3d"),
        ],
    )
    def test_a_fully_refined_space_is_the_finest_tensor_product_space(
        self, degrees: list[int], cells: list[int], factor: int | list[int]
    ) -> None:
        thb = create_thb_space(create_uniform_space(degrees, cells), factor=factor)
        factors = [factor] * len(cells) if isinstance(factor, int) else factor
        for level in range(2):
            top = [c * f**level for c, f in zip(cells, factors, strict=True)]
            thb = thb.refine_region(level, [0] * len(cells), top, admissible_class=None)
        finest = thb.level_space(2)
        assert thb.num_basis_per_level == (0, 0, finest.num_total_basis)

        pts = _sample_points(thb, 3, seed=51)
        gap = float(np.abs(hierarchical_rows(thb, pts) - tensor_product_rows(finest, pts)).max())
        assert gap <= VALUE_ATOL, f"max |THB - finest TP| = {gap:.3e} > {VALUE_ATOL:.3e}"

    @staticmethod
    def _finest_cell_points(thb: THBSplineSpace) -> npt.NDArray[np.float64]:
        """Three seeded random points in every cell of the finest level's tensor-product grid."""
        grid = tensor_product_grid(thb.level_space(thb.num_levels - 1))
        rng = np.random.default_rng(61)
        blocks = []
        for cid in range(grid.num_cells):
            lo, hi = grid.cell_bounds(cid)
            blocks.append(lo + (hi - lo) * rng.uniform(size=(3, thb.dim)))
        return np.concatenate(blocks)

    @pytest.mark.parametrize("build", [_space_1d, _space_2d, _space_2d_anisotropic])
    def test_every_thb_function_lies_in_the_span_of_the_finest_level(self, build: _Builder) -> None:
        thb = build()
        finest = thb.level_space(thb.num_levels - 1)
        pts = self._finest_cell_points(thb)
        matrix = tensor_product_rows(finest, pts)
        # Over-determined and of full column rank, so that a zero residual says something.
        assert matrix.shape[0] >= 2 * matrix.shape[1]
        assert np.linalg.matrix_rank(matrix) == matrix.shape[1]

        coeffs = np.random.default_rng(62).uniform(-1.0, 1.0, thb.num_total_basis)
        values = THBSpline(thb, coeffs).evaluate(pts)
        fit = np.linalg.lstsq(matrix, values, rcond=None)[0]
        residual = float(np.abs(matrix @ fit - values).max())
        bound = prolongation_residual_tolerance(float(np.abs(fit).max()))
        assert residual <= bound, f"residual {residual:.3e} > {bound:.3e}"

        # Control: a function with a kink inside a cell is not in a C^1 spline space.
        kink = np.abs(pts[:, 0] - 0.37)
        kink_fit = np.linalg.lstsq(matrix, kink, rcond=None)[0]
        assert float(np.abs(matrix @ kink_fit - kink).max()) >= _VISIBLE


# ──────────────────────────────────────────────────────────────────────────────
# Supports, from the knots: the halo, the owner and the coupling graph
# ──────────────────────────────────────────────────────────────────────────────


def _midpoint(thb: THBSplineSpace, cid: int) -> npt.NDArray[np.float64]:
    """Return the centre of cell ``cid``."""
    lo, hi = thb.grid.cell_bounds(cid)
    mid: npt.NDArray[np.float64] = 0.5 * (lo + hi)
    return mid


def _function_supports(thb: THBSplineSpace) -> list[npt.NDArray[np.int64]]:
    """Return, per global dof, the active cells inside the function's untruncated support.

    The support of the function of level ``l`` and multi-index ``m`` is the box
    ``prod_k [knots_k[m_k], knots_k[m_k + degree_k + 1]]`` of its level space. A leaf cell is in
    it when the cell's midpoint is strictly inside the box: a leaf cell is a union of nothing
    smaller than a cell of its own level, and a function is only active where its support is
    covered by cells of its level or finer, so a leaf cell either lies in the box or misses its
    interior. Nothing here reads ``active_basis``; the function of each dof is
    identified through ``active_function_indices`` (level by level, in the order of the dofs).

    Args:
        thb (THBSplineSpace): The space.

    Returns:
        list[npt.NDArray[np.int64]]: Sorted cell ids, one array per dof.
    """
    mids = np.array([_midpoint(thb, cid) for cid in range(thb.grid.num_cells)])
    supports: list[npt.NDArray[np.int64]] = []
    for level in range(thb.num_levels):
        space = thb.level_space(level)
        knots = [np.asarray(sp.knots, dtype=np.float64) for sp in space.spaces]
        for flat in thb.active_function_indices(level):
            multi = np.unravel_index(int(flat), space.num_basis)
            lo = np.array([knots[k][multi[k]] for k in range(thb.dim)])
            hi = np.array([knots[k][multi[k] + thb.degrees[k] + 1] for k in range(thb.dim)])
            supports.append(np.flatnonzero(np.all((mids > lo) & (mids < hi), axis=1)))
    return supports


def _brute_halo(
    supports: list[npt.NDArray[np.int64]], owned: npt.NDArray[np.int64], n_cells: int
) -> npt.NDArray[np.int64]:
    """Cells that share a function with an owned cell, owned cells excluded."""
    is_owned = np.zeros(n_cells, dtype=bool)
    is_owned[owned] = True
    touched = np.zeros(n_cells, dtype=bool)
    for cells in supports:
        if is_owned[cells].any():
            touched[cells] = True
    return np.flatnonzero(touched & ~is_owned).astype(np.int64)


def _brute_owner(
    supports: list[npt.NDArray[np.int64]], cell_owner: npt.NDArray[np.int32]
) -> npt.NDArray[np.int32]:
    """Owner of the first active cell of each support, ``-1`` where there is none."""
    owners = np.full(len(supports), -1, dtype=np.int32)
    for dof, cells in enumerate(supports):
        active = cells[cell_owner[cells] >= 0]
        if active.size:
            owners[dof] = cell_owner[active[0]]
    return owners


def _brute_coupling(supports: list[npt.NDArray[np.int64]], n_cells: int) -> npt.NDArray[np.int64]:
    """Number of functions shared by each pair of distinct cells."""
    shared = np.zeros((n_cells, n_cells), dtype=np.int64)
    for cells in supports:
        shared[np.ix_(cells, cells)] += 1
    np.fill_diagonal(shared, 0)
    return shared


class TestSupportOverlap:
    """The windowing functions equal the overlap of the functions' supports, read off the knots.

    Oracle: each function's support box from the knots of its level space, the leaf cells'
    midpoints from the grid, and set arithmetic in numpy; ``active_basis`` is not used, so a
    support the C++ code gets wrong cannot hide behind the same support in the reference. Bug
    caught: a halo too small at a level interface (a function's support not closed), a cell owned
    by the wrong rank, a coupling weight off by a shared function, a function listed on a cell it
    does not touch.
    """

    @pytest.mark.parametrize("build", _SPACES)
    def test_active_basis_is_the_set_of_functions_whose_support_holds_the_cell(
        self, build: _Builder
    ) -> None:
        thb = build()
        supports = _function_supports(thb)
        assert len(supports) == thb.num_total_basis
        for cid in range(thb.grid.num_cells):
            expected = [dof for dof, cells in enumerate(supports) if cid in cells]
            np.testing.assert_array_equal(thb.active_basis(cid), expected, err_msg=f"cell {cid}")

    @pytest.mark.parametrize("build", _SPACES)
    def test_halo_is_the_cells_sharing_a_function_with_an_owned_cell(self, build: _Builder) -> None:
        thb = build()
        supports = _function_supports(thb)
        n_cells = thb.grid.num_cells
        rng = np.random.default_rng(71)
        owned_sets = [
            np.array([], dtype=np.int64),
            np.arange(n_cells, dtype=np.int64),
            np.array([n_cells // 2], dtype=np.int64),
            np.flatnonzero(rng.uniform(size=n_cells) < 0.15).astype(np.int64),
            np.flatnonzero(rng.uniform(size=n_cells) < 0.6).astype(np.int64),
        ]
        sizes = []
        for owned in owned_sets:
            got = thb_halo(thb, owned)
            np.testing.assert_array_equal(got, _brute_halo(supports, owned, n_cells))
            sizes.append(got.size)
        assert max(sizes) > 0, "no owned set had a halo: the check ran over nothing"

    @pytest.mark.parametrize("build", _SPACES)
    def test_dof_owner_is_the_owner_of_the_first_active_cell_of_the_support(
        self, build: _Builder
    ) -> None:
        thb = build()
        supports = _function_supports(thb)
        n_cells = thb.grid.num_cells
        rng = np.random.default_rng(72)
        partitions = [
            Partition(np.zeros(n_cells, dtype=np.int32), n_parts=1),
            Partition(rng.integers(0, 3, size=n_cells).astype(np.int32), n_parts=3),
            Partition(rng.integers(-1, 3, size=n_cells).astype(np.int32), n_parts=3),
            # A slab of inactive cells: the functions supported inside it are dead.
            Partition(
                np.where(
                    [_midpoint(thb, c)[0] < 0.4 for c in range(n_cells)],
                    -1,
                    rng.integers(0, 3, size=n_cells),
                ).astype(np.int32),
                n_parts=3,
            ),
        ]
        dead = 0
        for partition in partitions:
            want = _brute_owner(supports, partition.cell_owner)
            np.testing.assert_array_equal(thb_dof_owner(thb, partition), want)
            dead += int((want < 0).sum())
        assert dead > 0, "no partition left a dof dead: the -1 path ran over nothing"

    @pytest.mark.parametrize("build", _SPACES)
    def test_coupling_graph_weights_count_the_functions_two_cells_share(
        self, build: _Builder
    ) -> None:
        thb = build()
        n_cells = thb.grid.num_cells
        graph = coupling_graph(thb)
        dense = np.zeros((n_cells, n_cells), dtype=np.int64)
        for cell in range(n_cells):
            row = slice(graph.xadj[cell], graph.xadj[cell + 1])
            dense[cell, graph.adjncy[row]] = graph.edge_weights[row]
        expected = _brute_coupling(_function_supports(thb), n_cells)
        np.testing.assert_array_equal(dense, expected)
        assert graph.adjncy.size > 0
