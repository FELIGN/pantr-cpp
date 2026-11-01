"""Value semantics of the grid wrappers over the compiled grids.

``TensorProductGrid``, ``HierarchicalGrid``, ``CellTags`` and ``FacetTags`` hold an immutable
compiled value in ``_h``. What their wrappers promise on top of it is pinned here:

* point location gives the same ids at any thread count, and the right ones;
* the properties of an immutable grid are cached objects, not fresh ones per read;
* a grid, with its tag registries, pickles and copies by value, and the copy changes independently;
* a hierarchical grid swaps its compiled value when, and only when, a refinement or coarsening
  changes it: ``version`` counts the changes, and the tag registries and the cached BVH go with
  the value they were made for;
* a tag registry replaces the value it holds on ``set`` and ``remove``, so arrays read earlier
  keep their values, a failed ``set`` changes nothing, and a replaced tag keeps its place in the
  order.

Oracles are NumPy and brute force, never the wrappers' own helpers.
"""

from __future__ import annotations

import copy
import pickle
from typing import Any

import numba as nb
import numpy as np
import numpy.typing as npt
import pytest

import pantr
from pantr._ext import ext
from pantr.grid import (
    BVH,
    CellTags,
    FacetTags,
    HierarchicalGrid,
    Partition,
    TensorProductGrid,
    hierarchical_grid,
    uniform_grid,
)

# ---------------------------------------------------------------------------
# TensorProductGrid
# ---------------------------------------------------------------------------


def _numpy_locate(g: TensorProductGrid, points: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    """Locate points with numpy only: per-axis ``searchsorted``, a range test, C-order strides.

    An oracle for ``locate_many`` that shares no code with the grid: ``side="left"`` puts a
    point on an interior breakpoint into the lower cell, the first and last breakpoints belong
    to the end cells, and a point outside the box (or with a non-finite coordinate) is ``-1``.
    """
    strides = np.cumprod([1, *g.cells_per_axis[:0:-1]])[::-1]
    out = np.zeros(points.shape[0], dtype=np.int64)
    inside = np.ones(points.shape[0], dtype=bool)
    for d, bp in enumerate(g.breakpoints):
        x = points[:, d]
        inside &= (bp[0] <= x) & (x <= bp[-1])
        idx = np.searchsorted(bp, x, side="left") - 1
        out += np.clip(idx, 0, g.cells_per_axis[d] - 1) * strides[d]
    out[~inside] = -1
    return out


def _many_threads() -> int:
    """Pick a thread count above one for the parallel run.

    Returns:
        int: Eight, capped at the most threads `pantr.num_threads` accepts (a CI runner may
        allow four) and at the C++ pool's capacity.
    """
    numba_cap: int = nb.config.NUMBA_NUM_THREADS
    return min(8, numba_cap, ext.max_threads())


def test_locate_many_is_independent_of_the_thread_count() -> None:
    """One thread and several threads give the same ids, bit for bit, and the right ones.

    The batch is far above the size at which the search runs in parallel (about 10^4 points
    for two axes), so the several-thread run really splits it; the expected ids come from a
    numpy ``searchsorted`` oracle, so equality of the two runs is not the only thing checked.
    """
    if _many_threads() < 2:
        pytest.skip("a single thread is available, so there is no parallel run to compare")
    rng = np.random.default_rng(11)
    g = TensorProductGrid(
        [np.sort(rng.uniform(0.0, 10.0, 129)) + np.arange(129) * 1e-3, np.linspace(-2.0, 3.0, 97)]
    )
    pts = rng.uniform(-3.0, 13.0, size=(300_000, 2))
    pts[::997, 0] = np.nan
    pts[1::991, 1] = np.inf
    pts[2::983] = g.breakpoints[0][3], g.breakpoints[1][5]  # exactly on interior breakpoints
    with pantr.num_threads(1):
        serial = g.locate_many(pts)
    with pantr.num_threads(_many_threads()):
        parallel = g.locate_many(pts)
    np.testing.assert_array_equal(parallel, serial)
    expected = _numpy_locate(g, pts)
    nonfinite = ~np.isfinite(pts).all(axis=1)
    expected[nonfinite] = -1
    np.testing.assert_array_equal(serial, expected)
    assert (serial == -1).any()
    assert (serial >= 0).any()


def test_wrapper_properties_return_the_cached_objects() -> None:
    """Properties of an immutable grid return one object each, not a fresh one per read."""
    g = uniform_grid([[0.0, 1.0], [0.0, 2.0]], [3, 4])
    assert g.breakpoints is g.breakpoints
    assert g.bounds is g.bounds
    assert g.cells_per_axis is g.cells_per_axis


@pytest.mark.parametrize("how", ["pickle", "pickle2", "deepcopy", "copy"])
def test_a_grid_with_tags_survives_pickle_and_copy(how: str) -> None:
    """Pickling or copying a grid keeps its breakpoints and both tag registries.

    The copy is independent: a tag set on it afterwards does not appear on the original, and
    the registries keep their identity on the copy (``copy.cell_tags is copy.cell_tags``).
    """
    g = TensorProductGrid([[0.0, 1.0, 2.5, 4.0], [0.0, 2.0, 3.0]])
    g.cell_tags.set("zone", [4, 1], [7, 8])
    g.cell_tags.set("other", [0], 5)
    g.facet_tags.set("wall", [[0, 0], [5, 3]], 2)
    if how == "pickle":
        clone = pickle.loads(pickle.dumps(g))
    elif how == "pickle2":
        clone = pickle.loads(pickle.dumps(g, protocol=2))
    elif how == "deepcopy":
        clone = copy.deepcopy(g)
    else:
        clone = copy.copy(g)

    assert isinstance(clone, TensorProductGrid)
    assert clone is not g
    assert clone.cells_per_axis == g.cells_per_axis
    for a, b in zip(clone.breakpoints, g.breakpoints, strict=True):
        np.testing.assert_array_equal(a, b)
        assert not a.flags.writeable
    assert clone.cell_tags is clone.cell_tags
    assert clone.cell_tags.names == ("zone", "other")
    for name in ("zone", "other"):
        for got, want in zip(clone.cell_tags[name], g.cell_tags[name], strict=True):
            np.testing.assert_array_equal(got, want)
    assert clone.facet_tags.names == ("wall",)
    for got, want in zip(clone.facet_tags["wall"], g.facet_tags["wall"], strict=True):
        np.testing.assert_array_equal(got, want)

    clone.cell_tags.set("late", [2], 1)
    clone.cell_tags.remove("zone")
    assert g.cell_tags.names == ("zone", "other")
    assert clone.cell_tags.names == ("other", "late")


def test_an_untagged_grid_stays_untagged_through_pickle() -> None:
    """A grid that never made its registries does not make them on the way through pickle."""
    clone = pickle.loads(pickle.dumps(uniform_grid([[0.0, 1.0]], 4)))
    assert clone._cell_tags is None
    assert clone._facet_tags is None
    assert clone._bvh is None


# ---------------------------------------------------------------------------
# Handle swap: a registry is a wrapper over an immutable value
# ---------------------------------------------------------------------------


def test_arrays_read_before_a_set_keep_their_values() -> None:
    """``set`` replaces the registry's value; arrays fetched earlier are not altered."""
    tags = CellTags(num_cells=10)
    tags.set("a", [3, 1], [30, 10])
    ids_before, values_before = tags["a"]
    tags.set("a", [5, 6, 7], 99)
    np.testing.assert_array_equal(ids_before, [1, 3])
    np.testing.assert_array_equal(values_before, [10, 30])
    ids_after, values_after = tags["a"]
    np.testing.assert_array_equal(ids_after, [5, 6, 7])
    np.testing.assert_array_equal(values_after, [99, 99, 99])


def test_arrays_read_before_a_remove_stay_valid() -> None:
    """A removed tag's arrays, fetched earlier, still read the old values."""
    tags = FacetTags(num_cells=4, facets_per_cell=4)
    tags.set("wall", [[2, 1], [0, 3]], [6, 5])
    keys_before, values_before = tags["wall"]
    tags.remove("wall")
    assert "wall" not in tags
    np.testing.assert_array_equal(keys_before, [[0, 3], [2, 1]])
    np.testing.assert_array_equal(values_before, [5, 6])


def test_registry_arrays_are_read_only_views() -> None:
    """The arrays a registry returns cannot be written through."""
    cell = CellTags(num_cells=5)
    cell.set("a", [1, 2], 3)
    for arr in cell["a"]:
        assert not arr.flags.writeable
    facet = FacetTags(num_cells=5, facets_per_cell=2)
    facet.set("a", [[1, 0]], 3)
    for arr in facet["a"]:
        assert not arr.flags.writeable


def test_replacing_a_tag_keeps_its_place_in_the_order() -> None:
    """Setting an existing name replaces it in place; removing and re-adding moves it last."""
    tags = CellTags(num_cells=6)
    tags.set("a", [0], 1)
    tags.set("b", [1], 2)
    tags.set("a", [2], 3)
    assert tags.names == ("a", "b")
    assert list(tags) == ["a", "b"]
    tags.remove("a")
    tags.set("a", [3], 4)
    assert tags.names == ("b", "a")


def test_a_failed_set_leaves_the_registry_unchanged() -> None:
    """A rejected ``set`` does not replace the registry's value."""
    tags = CellTags(num_cells=4)
    tags.set("a", [0, 1], 7)
    with pytest.raises(ValueError, match="unique"):
        tags.set("a", [2, 2], 1)
    with pytest.raises(ValueError, match="cell ids must be in"):
        tags.set("b", [9], 1)
    assert tags.names == ("a",)
    np.testing.assert_array_equal(tags["a"][0], [0, 1])


def test_missing_names_raise_key_error() -> None:
    """Reading or removing an unknown tag, or one that is not a string, is a KeyError."""
    tags = CellTags(num_cells=3)
    with pytest.raises(KeyError):
        tags["nope"]
    with pytest.raises(KeyError):
        tags.remove("nope")
    assert 3 not in tags


def test_copied_grid_tags_change_independently() -> None:
    """A grid copy gets copies of the registries: a tag set on one is not seen on the other."""
    g = uniform_grid([[0.0, 4.0]], 4)
    g.cell_tags.set("a", [0], 1)
    clone = copy.copy(g)
    assert clone.cell_tags is not g.cell_tags
    clone.cell_tags.set("a", [1, 2], 5)
    clone.cell_tags.set("b", [3], 6)
    assert g.cell_tags.names == ("a",)
    np.testing.assert_array_equal(g.cell_tags["a"][0], [0])
    np.testing.assert_array_equal(clone.cell_tags["a"][0], [1, 2])


@pytest.mark.parametrize("protocol", [2, pickle.HIGHEST_PROTOCOL])
def test_registries_round_trip_through_pickle(protocol: int) -> None:
    """Both registries pickle with their names in insertion order and their sorted arrays."""
    cell = CellTags(num_cells=8)
    cell.set("z", [5, 2], [1, 2])
    cell.set("a", [7], 9)
    facet = FacetTags(num_cells=8, facets_per_cell=4)
    facet.set("z", [[3, 1], [0, 2]], [1, 2])
    facet.set("a", [[7, 0]], 9)

    cell2 = pickle.loads(pickle.dumps(cell, protocol=protocol))
    facet2 = pickle.loads(pickle.dumps(facet, protocol=protocol))

    assert cell2.num_cells == 8
    assert cell2.names == ("z", "a")
    for name in ("z", "a"):
        for got, want in zip(cell2[name], cell[name], strict=True):
            np.testing.assert_array_equal(got, want)
    assert facet2.num_cells == 8
    assert facet2.facets_per_cell == 4
    assert facet2.names == ("z", "a")
    for name in ("z", "a"):
        for got, want in zip(facet2[name], facet[name], strict=True):
            np.testing.assert_array_equal(got, want)


def test_views_of_empty_arrays_are_read_only_too() -> None:
    """Tags with no entry, a tree with no node and a partition with no cell give read-only arrays.

    An empty array has no data to point at, which is where a view could silently fall back to a
    writable array of its own.
    """
    cell = CellTags(num_cells=3)
    cell.set("none", np.array([], dtype=np.int64), 1)
    facet = FacetTags(num_cells=3, facets_per_cell=2)
    facet.set("none", np.zeros((0, 2), dtype=np.int64), 1)
    tree = BVH.from_cell_bounds(np.zeros((0, 2)), np.zeros((0, 2)))
    part = Partition(np.array([], dtype=np.int32), 1)
    arrays: list[npt.NDArray[Any]] = [
        *cell["none"],
        *facet["none"],
        tree.node_lo,
        tree.node_hi,
        tree.node_left,
        tree.node_right,
        tree.node_cell,
        part.cell_owner,
    ]
    assert facet["none"][0].shape == (0, 2)
    assert tree.node_lo.shape == (0, 2)
    assert all(a.size == 0 for a in arrays)
    assert not any(a.flags.writeable for a in arrays)


# ---------------------------------------------------------------------------
# HierarchicalGrid
# ---------------------------------------------------------------------------


def _names(tags: CellTags) -> tuple[str, ...]:
    """Read a registry's names through a call, so the type checker does not narrow them."""
    return tags.names


def _built_bvh(g: HierarchicalGrid) -> BVH | None:
    """Read the cached BVH wrapper (``None`` before one is built) through a call."""
    return g._bvh


def _refined_grid() -> HierarchicalGrid:
    """Build a 4 by 4 grid with a level-1 block and, inside it, a level-2 block."""
    g = hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 2.0]], [4, 4]), 2)
    g.refine(0, [0, 0], [2, 2])
    g.refine(1, [0, 0], [2, 2])
    return g


def test_version_counts_the_structural_changes() -> None:
    """``version`` rises once per change that alters the grid and stays put on a no-op.

    Expected values are the documented sequence: a new grid is at 1; a ``refine`` that
    changes it adds one; one that finds every cell of the region already deeper adds none; an
    empty ``refine_cells`` adds none; ``refine_cells`` over leaves of two levels refines once per
    level, so adds two; ``coarsen_cells`` adds one per parent it demotes.
    """
    g = hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4), 2)
    assert g.version == 1
    g.refine(0, [0, 0], [2, 2])
    assert g.version == 2
    n = g.num_cells
    g.refine(0, [0, 0], [2, 2])  # nothing active left at level 0 in the region
    assert (g.version, g.num_cells) == (2, n)
    g.refine_cells([])
    assert g.version == 2

    ids = [g.cell_id(0, (3, 3)), g.cell_id(1, (0, 0))]  # leaves of levels 0 and 1
    assert None not in ids
    g.refine_cells(ids)  # type: ignore[arg-type]
    assert g.version == 4
    assert g.max_level == 2

    # Demote the level-2 children of the level-1 cell (0, 0): one parent, one change.
    children = [g.cell_id(2, (i, j)) for i in (0, 1) for j in (0, 1)]
    g.coarsen_cells(children)  # type: ignore[arg-type]
    assert g.version == 5
    # Naming only three of the four children demotes nothing and changes nothing.
    g.refine(1, [0, 0], [1, 1])
    assert g.version == 6
    three = [g.cell_id(2, (i, j)) for i, j in ((0, 0), (0, 1), (1, 0))]
    g.coarsen_cells(three)  # type: ignore[arg-type]
    assert g.version == 6
    # Two parents named in full: two demotions, two bumps.
    g.refine(1, [1, 0], [2, 2])
    full = [g.cell_id(2, (2 + i, 2 * j + k)) for j in (0, 1) for i in (0, 1) for k in (0, 1)]
    before = g.version
    g.coarsen_cells(full)  # type: ignore[arg-type]
    assert g.version == before + 2


def test_a_change_resets_the_tags_and_the_bvh_and_a_no_op_keeps_them() -> None:
    """Tags and the cached BVH belong to one compiled value: a change drops them, a no-op not."""
    g = hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4), 2)
    g.refine(0, [0, 0], [2, 2])
    cell_tags = g.cell_tags
    facet_tags = g.facet_tags
    cell_tags.set("zone", [0, 1], 3)
    tree = g.cell_bvh()
    assert g._bvh is tree

    g.refine(0, [0, 0], [2, 2])  # nothing left to refine there
    g.refine_cells([])
    g.coarsen_cells([g.cell_id(0, (3, 3))])  # type: ignore[list-item]  # level 0: no parent
    assert g.cell_tags is cell_tags
    assert g.facet_tags is facet_tags
    assert g.cell_tags.names == ("zone",)
    assert g._bvh is tree

    n_before = g.num_cells
    g.refine(0, [2, 2], [3, 3])
    assert g.num_cells == n_before + 3
    assert _built_bvh(g) is None
    assert id(g.cell_tags) != id(cell_tags)  # `is not` would be narrowed by the asserts above
    assert _names(g.cell_tags) == ()
    assert g.cell_tags.num_cells == g.num_cells
    assert id(g.facet_tags) != id(facet_tags)
    assert g.facet_tags.num_cells == g.num_cells
    # The new BVH is over the new cells.
    assert g.cell_bvh().n_cells == g.num_cells
    # The old tags still read what they held.
    assert cell_tags.names == ("zone",)


def test_a_rejected_change_leaves_the_grid_as_it_was() -> None:
    """A refinement or coarsening that raises changes neither the grid nor its tags."""
    g = _refined_grid()
    g.cell_tags.set("zone", [0], 1)
    version, n = g.version, g.num_cells
    with pytest.raises(ValueError, match="cannot coarsen"):
        g.coarsen(0, [0, 0], [2, 2])  # part of it is two levels deep
    with pytest.raises(ValueError, match="out of bounds"):
        g.refine(0, [0, 0], [9, 9])
    with pytest.raises(IndexError, match="out of range"):
        g.refine_cells([0, g.num_cells])
    assert (g.version, g.num_cells) == (version, n)
    assert g.cell_tags.names == ("zone",)


def test_root_is_the_object_that_was_passed_in() -> None:
    """``g.root is root`` holds, through refinement and for a copy; a restriction has its own."""
    root = uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4)
    g = hierarchical_grid(root, 2)
    assert g.root is root
    g.refine(0, [0, 0], [2, 2])
    assert g.root is root
    assert copy.copy(g).root is root
    # A deep copy has a root of its own: the root's tag registries are mutable.
    deep_root = copy.deepcopy(g).root
    assert deep_root is not root
    assert deep_root.cells_per_axis == root.cells_per_axis
    sub = g.restrict([0]).grid
    assert isinstance(sub, HierarchicalGrid)
    assert isinstance(sub.root, TensorProductGrid)
    assert sub.root is sub.root
    assert sub.root is not root
    assert sub.factor == g.factor


def test_a_deep_copy_does_not_share_the_root_tag_registry() -> None:
    """Tags set on a deep copy's root do not show on the original's, and the reverse.

    A plain copy shares the root wrapper (the previous release's shallow copy did).
    """
    root = uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4)
    g = hierarchical_grid(root, 2)
    g.root.cell_tags.set("zone", [1], 5)
    deep = copy.deepcopy(g)
    assert deep.root.cell_tags.names == ("zone",)
    deep.root.cell_tags.set("only_deep", [2], 6)
    deep.root.cell_tags.remove("zone")
    assert g.root.cell_tags.names == ("zone",)
    g.root.cell_tags.set("only_orig", [3], 7)
    assert deep.root.cell_tags.names == ("only_deep",)
    assert copy.copy(g).root is g.root


@pytest.mark.parametrize("how", ["pickle", "pickle2", "deepcopy", "copy"])
def test_a_refined_hierarchical_grid_survives_pickle_and_copy(how: str) -> None:
    """Pickling or copying keeps the blocks, the version and both tag registries.

    The copy is independent in both directions: refining it does not change the original, and
    refining the original does not change it; each swaps only its own compiled value.
    """
    g = _refined_grid()
    g.refine(0, [3, 3], [4, 4])
    g.cell_tags.set("zone", [5, 1], [7, 8])
    g.facet_tags.set("wall", [[0, 0], [5, 3]], 2)
    if how == "pickle":
        clone = pickle.loads(pickle.dumps(g))
    elif how == "pickle2":
        clone = pickle.loads(pickle.dumps(g, protocol=2))
    elif how == "deepcopy":
        clone = copy.deepcopy(g)
    else:
        clone = copy.copy(g)

    assert isinstance(clone, HierarchicalGrid)
    assert clone is not g
    assert clone.factor == g.factor
    assert clone.version == g.version
    assert clone.num_cells == g.num_cells
    assert clone.max_level == g.max_level
    assert clone._blocks == g._blocks
    np.testing.assert_array_equal(clone.root.breakpoints[1], g.root.breakpoints[1])
    for level in range(g.max_level + 1):
        np.testing.assert_array_equal(clone.active_leaf_mask(level), g.active_leaf_mask(level))
    assert clone.cell_tags is clone.cell_tags
    assert clone.cell_tags.names == ("zone",)
    for got, want in zip(clone.cell_tags["zone"], g.cell_tags["zone"], strict=True):
        np.testing.assert_array_equal(got, want)
    assert clone.facet_tags.names == ("wall",)
    for got, want in zip(clone.facet_tags["wall"], g.facet_tags["wall"], strict=True):
        np.testing.assert_array_equal(got, want)

    n, version = g.num_cells, g.version
    clone.refine(0, [2, 2], [3, 3])
    assert _names(clone.cell_tags) == ()  # the clone's change reset its registry, not g's
    clone.cell_tags.set("late", [0], 1)
    assert (g.num_cells, g.version) == (n, version)
    assert g.cell_tags.names == ("zone",)

    clone2 = copy.copy(g)
    g.refine(0, [2, 3], [3, 4])
    assert clone2.num_cells == n
    assert clone2.version == version
    assert clone2.cell_tags.names == ("zone",)


def test_an_untagged_hierarchical_grid_stays_untagged_through_pickle() -> None:
    """A grid that never made its registries does not make them on the way through pickle."""
    clone = pickle.loads(pickle.dumps(_refined_grid()))
    assert clone._cell_tags is None
    assert clone._facet_tags is None
    assert clone._bvh is None


def _two_level_grid_for_threads() -> HierarchicalGrid:
    """Build a 2-D grid of about 10^5 cells over irregular breakpoints, three levels deep."""
    rng = np.random.default_rng(3)
    root = TensorProductGrid(
        [np.sort(rng.uniform(0.0, 1.0, 120)) + np.arange(120) * 1e-3, np.linspace(-1.0, 2.0, 101)]
    )
    g = hierarchical_grid(root, (2, 3))
    g.refine(0, [10, 10], [70, 60])
    g.refine(1, [30, 40], [90, 150])
    g.refine(2, [100, 200], [200, 500])
    return g


def test_hierarchical_locate_many_is_independent_of_the_thread_count() -> None:
    """One thread and several threads give the same ids, bit for bit, and plausible ones.

    The batch is far above the size at which the search runs in parallel. The ids are integers,
    so equality is the criterion. The independent check is brute force: the located cell must
    contain its point (inclusively, as a point on a shared face may go to either cell), and a
    point no cell contains must give ``-1``.
    """
    if _many_threads() < 2:
        pytest.skip("a single thread is available, so there is no parallel run to compare")
    g = _two_level_grid_for_threads()
    rng = np.random.default_rng(12)
    lo, hi = g.root.bounds[:, 0], g.root.bounds[:, 1]
    pts = rng.uniform(lo - 0.2, hi + 0.2, size=(400_000, 2))
    pts[::997, 0] = np.nan
    pts[1::991, 1] = np.inf
    pts[2::983] = g.root.breakpoints[0][11], g.root.breakpoints[1][12]  # on root breakpoints
    with pantr.num_threads(1):
        serial = g.locate_many(pts)
    with pantr.num_threads(_many_threads()):
        parallel = g.locate_many(pts)
    np.testing.assert_array_equal(parallel, serial)
    assert (serial == -1).any()
    assert (serial >= 0).any()

    cell_lo, cell_hi = g.collect_cell_bounds()
    for i in range(0, pts.shape[0], 97):
        p = pts[i]
        cid = int(serial[i])
        if cid == -1:
            assert not np.isfinite(p).all() or not (
                np.all((cell_lo <= p) & (p <= cell_hi), axis=1).any()
            )
        else:
            assert np.all((cell_lo[cid] <= p) & (p <= cell_hi[cid]))


def test_hierarchical_collect_cell_bounds_is_independent_of_the_thread_count() -> None:
    """One thread and several threads give the same corners, exactly.

    Every cell's corners come from its own block and multi-index by a fixed expression; no
    value is combined across cells, so splitting the cells among threads cannot change a bit and
    the criterion is equality, not a tolerance. The corners also equal ``cell_bounds`` cell by
    cell, which is computed by a different route (one cell at a time).
    """
    if _many_threads() < 2:
        pytest.skip("a single thread is available, so there is no parallel run to compare")
    g = _two_level_grid_for_threads()
    with pantr.num_threads(1):
        lo1, hi1 = g.collect_cell_bounds()
    with pantr.num_threads(_many_threads()):
        lo8, hi8 = g.collect_cell_bounds()
    assert lo1.shape == (g.num_cells, 2)
    np.testing.assert_array_equal(lo8, lo1)
    np.testing.assert_array_equal(hi8, hi1)
    for cid in range(0, g.num_cells, 211):
        lo, hi = g.cell_bounds(cid)
        np.testing.assert_array_equal(lo1[cid], lo)
        np.testing.assert_array_equal(hi1[cid], hi)
