"""Value semantics of the grid wrappers over the compiled grids.

``TensorProductGrid``, ``CellTags`` and ``FacetTags`` hold an immutable compiled value in
``_h``. What their wrappers promise on top of it is pinned here:

* point location gives the same ids at any thread count, and the right ones;
* the properties of an immutable grid are cached objects, not fresh ones per read;
* a grid, with its tag registries, pickles and copies by value, and the copy changes independently;
* a tag registry replaces the value it holds on ``set`` and ``remove``, so arrays read earlier
  keep their values, a failed ``set`` changes nothing, and a replaced tag keeps its place in the
  order.

Oracles are NumPy and brute force, never the wrappers' own helpers.
"""

from __future__ import annotations

import copy
import pickle
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import pantr
from pantr._ext import ext
from pantr.grid import BVH, CellTags, FacetTags, Partition, TensorProductGrid, uniform_grid

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


def test_locate_many_is_independent_of_the_thread_count() -> None:
    """One thread and eight threads give the same ids, bit for bit, and the right ones.

    The batch is far above the size at which the search runs in parallel (about 10^4 points
    for two axes), so the eight-thread run really splits it; the expected ids come from a
    numpy ``searchsorted`` oracle, so equality of the two runs is not the only thing checked.
    """
    if ext.max_threads() < 2:
        pytest.skip("the C++ pool has a single thread, so there is no parallel run to compare")
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
    with pantr.num_threads(8):
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
