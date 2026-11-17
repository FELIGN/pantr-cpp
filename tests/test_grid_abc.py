"""Tests for the ``pantr.grid.Grid`` base class and the defaults it gives the two grids.

``Grid`` is the base of pantr's two grids, :class:`TensorProductGrid` and
:class:`HierarchicalGrid`, and nothing else can derive from it: the defaults are
implemented once, in C++, over each concrete grid's own primitives, so a subclass
overriding a primitive would be ignored by them. The defaults are therefore tested on the
two concrete grids, each against an oracle that uses nothing but ``cell_bounds`` and
brute force over every cell: two cells share a facet when their boxes touch on the facet's
plane and overlap with positive extent on the other axes; a point is in a cell when it is in
the closed box; a query box meets a cell when the closed boxes meet.

All coordinates are dyadic rationals, so the exact comparisons the oracles make are exact.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

import pantr.grid
from pantr.geometry import AABB
from pantr.grid import (
    BVH,
    Grid,
    GridRestriction,
    HierarchicalGrid,
    TensorProductGrid,
    hierarchical_grid,
    uniform_grid,
)

# ---------------------------------------------------------------------------
# The grids under test
# ---------------------------------------------------------------------------


def _tpg_2d() -> TensorProductGrid:
    """A non-uniform 2-D grid, 3 x 3 cells."""
    return TensorProductGrid([[0.0, 1.0, 2.5, 4.0], [0.0, 1.0, 2.0, 3.0]])


def _tpg_3d() -> TensorProductGrid:
    """A 3-D grid, 2 x 3 x 2 cells."""
    return TensorProductGrid([[0.0, 1.0, 3.0], [0.0, 0.5, 1.0, 2.0], [0.0, 2.0, 3.0]])


def _hg_2d() -> HierarchicalGrid:
    """A 2-D hierarchical grid over 4 x 3 root cells with two levels of refinement.

    The refined block meets coarse cells along a hanging interface on every side that
    is not the domain boundary.
    """
    g = hierarchical_grid(uniform_grid([[0.0, 4.0], [0.0, 3.0]], [4, 3]), 2)
    g.refine(0, [1, 1], [3, 2])
    g.refine(1, [2, 2], [4, 3])
    return g


def _hg_3d() -> HierarchicalGrid:
    """A 3-D hierarchical grid over 2 x 2 x 2 root cells with one refined cell."""
    g = hierarchical_grid(uniform_grid([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]], 2), 2)
    g.refine(0, [0, 0, 0], [1, 1, 1])
    return g


_BUILDERS: dict[str, Callable[[], Grid]] = {
    "tpg_2d": _tpg_2d,
    "tpg_3d": _tpg_3d,
    "hg_2d": _hg_2d,
    "hg_3d": _hg_3d,
}


@pytest.fixture(params=list(_BUILDERS), ids=list(_BUILDERS))
def grid(request: pytest.FixtureRequest) -> Grid:
    """Each grid under test, freshly built.

    Args:
        request (pytest.FixtureRequest): Carries the builder's name.

    Returns:
        Grid: The grid.
    """
    return _BUILDERS[request.param]()


# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------

_Box = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]


def _boxes(g: Grid) -> list[_Box]:
    """The ``(lo, hi)`` box of every cell, through the primitive ``cell_bounds`` only."""
    return [g.cell_bounds(cid) for cid in range(g.num_cells)]


def _across(boxes: list[_Box], cid: int, lfid: int) -> set[int]:
    """The cells sharing facet ``lfid`` of cell ``cid``, by brute force over every cell."""
    axis, side = divmod(lfid, 2)
    lo, hi = boxes[cid]
    plane = hi[axis] if side == 1 else lo[axis]
    found: set[int] = set()
    for other, (lo2, hi2) in enumerate(boxes):
        if other == cid:
            continue
        touches = lo2[axis] == plane if side == 1 else hi2[axis] == plane
        if not touches:
            continue
        overlaps = all(max(lo[k], lo2[k]) < min(hi[k], hi2[k]) for k in range(len(lo)) if k != axis)
        if overlaps:
            found.add(other)
    return found


def _containing(boxes: list[_Box], point: npt.NDArray[np.float64]) -> set[int]:
    """The cells whose closed box contains ``point``, by brute force."""
    return {
        cid
        for cid, (lo, hi) in enumerate(boxes)
        if bool(np.all(lo <= point)) and bool(np.all(point <= hi))
    }


def _meeting(boxes: list[_Box], qlo: npt.ArrayLike, qhi: npt.ArrayLike) -> set[int]:
    """The cells whose closed box meets the closed box ``[qlo, qhi]``, by brute force."""
    qlo_a = np.asarray(qlo, dtype=np.float64)
    qhi_a = np.asarray(qhi, dtype=np.float64)
    return {
        cid
        for cid, (lo, hi) in enumerate(boxes)
        if bool(np.all(lo <= qhi_a)) and bool(np.all(qlo_a <= hi))
    }


def _probe_points(g: Grid, seed: int) -> npt.NDArray[np.float64]:
    """Points inside, on the cell boundaries, outside, and non-finite."""
    boxes = _boxes(g)
    all_lo = np.min([lo for lo, _ in boxes], axis=0)
    all_hi = np.max([hi for _, hi in boxes], axis=0)
    rng = np.random.default_rng(seed)
    inside = rng.uniform(all_lo, all_hi, size=(60, g.ndim))
    corners = np.array([c for lo, hi in boxes for c in (lo, hi, 0.5 * (lo + hi))])
    outside = np.array([all_lo - 1.0, all_hi + 0.25, all_lo - 1e-3])
    nonfinite = np.full((3, g.ndim), np.nan)
    nonfinite[1, 0] = np.inf
    nonfinite[2, 0] = -np.inf
    nonfinite[2, 1:] = all_lo[1:]
    return np.vstack([inside, corners, outside, nonfinite])


# ---------------------------------------------------------------------------
# The class and its subclassing rule
# ---------------------------------------------------------------------------


def test_grid_is_abstract() -> None:
    """Grid cannot be instantiated directly."""
    with pytest.raises(TypeError, match="abstract"):
        Grid()  # type: ignore[abstract]


def test_isinstance_holds_for_both_grids() -> None:
    """Both concrete grids are Grids, and nothing else in the package is."""
    assert isinstance(_tpg_2d(), Grid)
    assert isinstance(_hg_2d(), Grid)
    assert not isinstance(BVH.from_cell_bounds([[0.0]], [[1.0]]), Grid)
    assert issubclass(TensorProductGrid, Grid)
    assert issubclass(HierarchicalGrid, Grid)


@pytest.mark.parametrize("base", [Grid, TensorProductGrid, HierarchicalGrid])
def test_subclassing_a_grid_raises(base: type[Grid]) -> None:
    """Grid, and each concrete grid, refuse a subclass at class creation."""
    with pytest.raises(TypeError, match="cannot subclass Grid"):
        type("_Mine", (base,), {})


def test_subclassing_names_the_two_grids() -> None:
    """The refusal says which classes exist, so a user knows what to use instead."""
    with pytest.raises(TypeError, match="TensorProductGrid and HierarchicalGrid"):

        class _Mine(Grid):
            pass


def test_a_same_named_class_elsewhere_is_still_refused() -> None:
    """The exemption is for pantr's own classes, identified by module and name together."""
    with pytest.raises(TypeError, match="cannot subclass"):
        type("TensorProductGrid", (Grid,), {"__module__": "somewhere.else"})


def test_the_grids_live_where_the_exemption_looks() -> None:
    """The module and name the exemption checks are where the two classes really live."""
    assert TensorProductGrid.__module__ == "pantr.grid._tensor_product_grid"
    assert HierarchicalGrid.__module__ == "pantr.grid._hierarchical_grid"
    assert pantr.grid.TensorProductGrid is TensorProductGrid


# ---------------------------------------------------------------------------
# Cell accessors
# ---------------------------------------------------------------------------


def test_iter_cells_is_the_id_range(grid: Grid) -> None:
    """Every cell id appears once, in id order."""
    assert list(grid.iter_cells()) == list(range(grid.num_cells))


def test_collect_cell_bounds_stacks_cell_bounds(grid: Grid) -> None:
    """The batched bounds are the per-cell bounds, row by row."""
    lo, hi = grid.collect_cell_bounds()
    assert lo.shape == hi.shape == (grid.num_cells, grid.ndim)
    for cid, (clo, chi) in enumerate(_boxes(grid)):
        np.testing.assert_array_equal(lo[cid], clo)
        np.testing.assert_array_equal(hi[cid], chi)


def test_cell_aabb_is_the_cell_box(grid: Grid) -> None:
    """``cell_aabb`` is an AABB with the corners of ``cell_bounds``."""
    for cid, (lo, hi) in enumerate(_boxes(grid)):
        box = grid.cell_aabb(cid)
        assert isinstance(box, AABB)
        np.testing.assert_array_equal(box.lo, lo)
        np.testing.assert_array_equal(box.hi, hi)


def test_reference_map_sends_the_unit_cube_onto_the_cell(grid: Grid) -> None:
    """The map is ``lo + (hi - lo) * u``: the unit corners and the centre land on the cell's."""
    unit = np.array(
        [[(corner >> k) & 1 for k in range(grid.ndim)] for corner in range(1 << grid.ndim)],
        dtype=np.float64,
    )
    unit = np.vstack([unit, np.full((1, grid.ndim), 0.5)])
    for cid, (lo, hi) in enumerate(_boxes(grid)):
        image = grid.reference_map(cid)(unit)
        np.testing.assert_array_equal(image, lo + unit * (hi - lo))


def test_level_and_children(grid: Grid) -> None:
    """A flat grid is at level 0; a hierarchical cell's box is its root cell's, divided down."""
    for cid, (lo, hi) in enumerate(_boxes(grid)):
        level = grid.cell_level(cid)
        assert isinstance(level, int)
        if isinstance(grid, HierarchicalGrid):
            root = grid.root
            root_cell = root.locate(0.5 * (lo + hi))
            assert root_cell is not None
            rlo, rhi = root.cell_bounds(root_cell)
            np.testing.assert_array_equal(
                hi - lo, (rhi - rlo) / np.array(grid.factor, dtype=np.float64) ** level
            )
        else:
            assert level == 0
        assert grid.child_cells(cid) == ()


# ---------------------------------------------------------------------------
# Facets and neighbours
# ---------------------------------------------------------------------------


def test_facet_count_axis_side_and_bounds(grid: Grid) -> None:
    """``lfid = 2 * axis + side``; a facet box is degenerate on its axis."""
    for cid, (lo, hi) in enumerate(_boxes(grid)):
        assert grid.num_local_facets(cid) == 2 * grid.ndim
        for lfid in range(2 * grid.ndim):
            axis, side = grid.local_facet_axis_side(cid, lfid)
            assert (axis, side) == (lfid // 2, lfid % 2)
            flo, fhi = grid.local_facet_bounds(cid, lfid)
            expect_lo, expect_hi = lo.copy(), hi.copy()
            plane = hi[axis] if side == 1 else lo[axis]
            expect_lo[axis] = expect_hi[axis] = plane
            np.testing.assert_array_equal(flo, expect_lo)
            np.testing.assert_array_equal(fhi, expect_hi)
            assert flo.flags.writeable
            assert fhi.flags.writeable


def test_neighbors_match_the_brute_force_adjacency(grid: Grid) -> None:
    """Across every facet the neighbour is one of the cells that touch it, or None."""
    boxes = _boxes(grid)
    for cid in range(grid.num_cells):
        expected_list = []
        for lfid in range(2 * grid.ndim):
            adjacent = _across(boxes, cid, lfid)
            nbr = grid.neighbor_across_facet(cid, lfid)
            if adjacent:
                assert nbr in adjacent
                expected_list.append(nbr)
            else:
                assert nbr is None
            assert grid.is_mesh_boundary_facet(cid, lfid) == (not adjacent)
            hanging = grid.hanging_neighbors(cid, lfid)
            assert isinstance(hanging, tuple)
            assert set(hanging) == adjacent
            assert len(hanging) == len(adjacent)
        assert grid.neighbors(cid) == expected_list


def test_boundary_facets_are_the_facets_without_a_neighbour(grid: Grid) -> None:
    """The rows are exactly the ``(cid, lfid)`` with no adjacent cell, sorted and read-only."""
    boxes = _boxes(grid)
    expected = [
        (cid, lfid)
        for cid in range(grid.num_cells)
        for lfid in range(2 * grid.ndim)
        if not _across(boxes, cid, lfid)
    ]
    rows = grid.boundary_facets()
    assert rows.dtype == np.int64
    assert rows.shape == (len(expected), 2)
    assert [tuple(int(v) for v in row) for row in rows] == expected
    assert not rows.flags.writeable


def test_hanging_interfaces_exist_in_the_hierarchical_fixtures() -> None:
    """The refined fixtures really have a facet with several neighbours (the premise)."""
    for build in (_hg_2d, _hg_3d):
        g = build()
        boxes = _boxes(g)
        assert any(
            len(_across(boxes, cid, lfid)) > 1
            for cid in range(g.num_cells)
            for lfid in range(2 * g.ndim)
        )


@pytest.mark.parametrize(
    ("call", "args"),
    [
        ("cell_bounds", (10_000,)),
        ("cell_aabb", (-1,)),
        ("reference_map", (10_000,)),
        ("cell_level", (10_000,)),
        ("child_cells", (-1,)),
        ("num_local_facets", (-1,)),
        ("neighbors", (10_000,)),
        ("local_facet_axis_side", (0, 99)),
        ("local_facet_bounds", (0, -1)),
        ("is_mesh_boundary_facet", (0, 99)),
        ("neighbor_across_facet", (0, 99)),
        ("hanging_neighbors", (10_000, 0)),
    ],
)
def test_out_of_range_ids_raise_index_error(grid: Grid, call: str, args: tuple[int, ...]) -> None:
    """A cell or facet id out of range is an IndexError from every accessor."""
    with pytest.raises(IndexError, match="out of range"):
        getattr(grid, call)(*args)


# ---------------------------------------------------------------------------
# Point location and spatial queries
# ---------------------------------------------------------------------------


def test_locate_many_matches_the_brute_force_membership(grid: Grid) -> None:
    """A point maps to a cell whose closed box contains it, and to -1 when none does."""
    boxes = _boxes(grid)
    pts = _probe_points(grid, seed=3)
    ids = grid.locate_many(pts)
    assert ids.dtype == np.int64
    assert ids.shape == (pts.shape[0],)
    for i, point in enumerate(pts):
        containing = _containing(boxes, point) if np.all(np.isfinite(point)) else set()
        if containing:
            assert int(ids[i]) in containing, f"point {point}"
        else:
            assert int(ids[i]) == -1, f"point {point}"
        located = grid.locate(point)
        assert (located is None) == (not containing)
        if located is not None:
            assert located in containing


def test_locate_many_accepts_one_point_and_rejects_a_wrong_width(grid: Grid) -> None:
    """A single point is a batch of one; a wrong trailing axis is a ValueError."""
    lo, hi = grid.cell_bounds(0)
    assert grid.locate_many(0.5 * (lo + hi)).shape == (1,)
    with pytest.raises(ValueError, match="shape"):
        grid.locate_many(np.zeros((4, grid.ndim + 1)))


def test_query_aabb_matches_the_brute_force_overlap(grid: Grid) -> None:
    """The query returns exactly the cells whose closed box meets the query box."""
    boxes = _boxes(grid)
    all_lo = np.min([lo for lo, _ in boxes], axis=0)
    all_hi = np.max([hi for _, hi in boxes], axis=0)
    rng = np.random.default_rng(5)
    queries = [(all_lo, all_hi), (all_lo - 2.0, all_lo - 1.0), (all_hi, all_hi + 1.0)]
    for cid in (0, grid.num_cells // 2, grid.num_cells - 1):
        lo, hi = boxes[cid]
        queries.append((lo, hi))  # the cell's own box: it and every touching cell
        queries.append((hi, hi))  # a point on a corner
    for _ in range(8):
        corner = rng.uniform(all_lo, all_hi)
        queries.append((corner, corner + rng.uniform(0.0, 1.5, size=grid.ndim)))
    for qlo, qhi in queries:
        found = grid.query_aabb(AABB(qlo, qhi))
        assert found.dtype == np.int64
        assert sorted(found.tolist()) == sorted(_meeting(boxes, qlo, qhi))


def test_query_aabb_rejects_a_box_of_the_wrong_dimension(grid: Grid) -> None:
    """The box must have the grid's dimension."""
    with pytest.raises(ValueError, match=r"aabb\.ndim"):
        grid.query_aabb(AABB(np.zeros(grid.ndim + 1), np.ones(grid.ndim + 1)))


def _built_index(g: Grid) -> BVH | None:
    """Read ``g._bvh`` through a function, so the type checker does not narrow it across calls."""
    return g._bvh


def test_the_bvh_is_built_lazily_and_cached(grid: Grid) -> None:
    """No query, no index; the first query builds it once, and ``cell_bvh`` returns that wrapper."""
    assert _built_index(grid) is None
    first = grid.cell_bvh()
    assert isinstance(first, BVH)
    assert first.n_cells == grid.num_cells
    assert grid.cell_bvh() is first
    assert _built_index(grid) is first
    lo, hi = grid.collect_cell_bounds()
    rebuilt = BVH.from_cell_bounds(lo, hi)
    np.testing.assert_array_equal(first.node_cell, rebuilt.node_cell)


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


def test_tag_registries_are_lazy_and_sized_by_the_grid(grid: Grid) -> None:
    """Tags exist on first use, are the same object after, and match the grid's size."""
    assert grid._cell_tags is None
    assert grid._facet_tags is None
    cell_tags = grid.cell_tags
    facet_tags = grid.facet_tags
    assert grid.cell_tags is cell_tags
    assert grid.facet_tags is facet_tags
    assert cell_tags.num_cells == grid.num_cells
    assert facet_tags.num_cells == grid.num_cells
    assert facet_tags.facets_per_cell == 2 * grid.ndim
    cell_tags.set("picked", [0, grid.num_cells - 1], 4)
    expected = np.zeros(grid.num_cells, dtype=np.int64)
    expected[[0, grid.num_cells - 1]] = 4
    np.testing.assert_array_equal(cell_tags.to_dense("picked"), expected)


# ---------------------------------------------------------------------------
# Restriction
# ---------------------------------------------------------------------------


def test_restrict_spans_the_requested_cells(grid: Grid) -> None:
    """The window's cells are the global cells ``local_to_global_cell`` names, box for box."""
    request = [grid.num_cells // 3, grid.num_cells // 2]
    result = grid.restrict(request)
    assert isinstance(result, GridRestriction)
    assert type(result.grid) is type(grid)
    sub, l2g, in_subset = result
    assert l2g.shape == in_subset.shape == (sub.num_cells,)
    assert not l2g.flags.writeable
    assert not in_subset.flags.writeable
    assert in_subset.dtype == np.bool_
    assert sorted(l2g[in_subset].tolist()) == sorted(request)
    for local in range(sub.num_cells):
        lo, hi = sub.cell_bounds(local)
        glo, ghi = grid.cell_bounds(int(l2g[local]))
        np.testing.assert_array_equal(lo, glo)
        np.testing.assert_array_equal(hi, ghi)
