"""Copy and pickle of ``THBSplineSpace``, ``THBSpline`` and ``MultiLevelExtraction``.

These classes hold a compiled value that Python can neither copy nor pickle by itself, so each
wrapper rebuilds it from its constructor arguments (design record DR-5; the Python implementation
they replace was made of plain objects and pickled by default, which the port must not remove).
What that promises, checked here by ``copy.copy``, ``copy.deepcopy`` and ``pickle`` at every
protocol:

* a round trip is equal by value to the original, the original being the oracle: the same levels,
  active functions, cells and basis values for a space, the same control points and values for a
  spline, the same operators for an extraction;
* a space made stale by a change of its grid stays stale, in the copy as in the original, and a
  copy whose own grid is refined is stale and the original is not (the space watches the grid
  object it reports);
* the relations between objects of one graph survive a deep copy and a pickle of a tuple: a space
  reports the grid it was copied with, a spline the space, an extraction the space and its grid;
* a subclass keeps its class and its attributes, whatever its constructor takes, and a deep copy
  or a pickle does not share a mutable attribute with the original.
"""

from __future__ import annotations

import copy
import pickle
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import (
    BsplineSpace,
    MultiLevelExtraction,
    THBSpline,
    THBSplineSpace,
    create_uniform_space,
)
from pantr.grid import HierarchicalGrid, hierarchical_grid, uniform_grid
from tests._thb_support import VALUE_ATOL


def _pickled(protocol: int) -> Callable[[Any], Any]:
    return lambda obj: pickle.loads(pickle.dumps(obj, protocol=protocol))


_COPIERS = [
    pytest.param(copy.copy, id="copy"),
    pytest.param(copy.deepcopy, id="deepcopy"),
    *[pytest.param(_pickled(p), id=f"pickle{p}") for p in range(pickle.HIGHEST_PROTOCOL + 1)],
]
"""Every way a wrapper is copied."""

_DETACHING_COPIERS = _COPIERS[1:]
"""The copiers that give the copy a grid, a space and attributes of its own."""


def _grid() -> HierarchicalGrid:
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]] * 2, 6), 2)
    grid.refine(0, [1, 1], [5, 5])
    grid.refine(1, [3, 3], [9, 9])
    return grid


def _root() -> BsplineSpace:
    return create_uniform_space([2, 3], [6, 6])


def _space(*, truncate: bool = True, regularity: int | None = None) -> THBSplineSpace:
    return THBSplineSpace(_root(), _grid(), truncate=truncate, regularity=regularity)


_SPACES = [
    pytest.param({}, id="thb"),
    pytest.param({"truncate": False}, id="hb"),
    pytest.param({"regularity": 0}, id="c0"),
]
"""Keyword arguments of the spaces a round trip must reproduce (truncation, regularity)."""


def _assert_same_space(copied: THBSplineSpace, original: THBSplineSpace) -> None:
    """Assert two spaces are equal by value, through public methods only."""
    assert copied.num_levels == original.num_levels
    assert copied.num_basis_per_level == original.num_basis_per_level
    assert copied.num_total_basis == original.num_total_basis
    assert copied.truncate == original.truncate
    assert copied.degrees == original.degrees
    assert copied.grid.num_cells == original.grid.num_cells
    assert copied.max_active_per_cell() == original.max_active_per_cell()
    for level in range(original.num_levels):
        np.testing.assert_array_equal(
            copied.active_function_indices(level), original.active_function_indices(level)
        )
        for k, space_1d in enumerate(original.level_space(level).spaces):
            np.testing.assert_array_equal(copied.level_space(level).spaces[k].knots, space_1d.knots)
    rng = np.random.default_rng(3)
    for cid in range(original.grid.num_cells):
        assert copied.grid.cell_level(cid) == original.grid.cell_level(cid)
        np.testing.assert_array_equal(copied.active_basis(cid), original.active_basis(cid))
        lo, hi = original.grid.cell_bounds(cid)
        pts = lo + (hi - lo) * rng.uniform(size=(3, 2))
        want, _ = original.tabulate_basis(cid, pts)
        got, _ = copied.tabulate_basis(cid, pts)
        np.testing.assert_allclose(got, want, rtol=0.0, atol=VALUE_ATOL)


class _TaggedSpace(THBSplineSpace):
    """A space subclass whose constructor takes another argument than the base class's."""

    def __init__(self, tag: str, root: BsplineSpace, grid: HierarchicalGrid) -> None:
        super().__init__(root, grid)
        self.tag = [tag]


class _TaggedSpline(THBSpline):
    """A spline subclass whose constructor takes another argument than the base class's."""

    def __init__(self, tag: str, space: THBSplineSpace, control_points: npt.ArrayLike) -> None:
        super().__init__(space, control_points)
        self.tag = [tag]


class _TaggedExtraction(MultiLevelExtraction):
    """An extraction subclass whose constructor takes another argument than the base class's."""

    def __init__(self, tag: str, space: THBSplineSpace) -> None:
        super().__init__(space)
        self.tag = [tag]


# ──────────────────────────────────────────────────────────────────────────────
# THBSplineSpace
# ──────────────────────────────────────────────────────────────────────────────


class TestSpace:
    """``THBSplineSpace`` copies and pickles by value, stale or not."""

    @pytest.mark.parametrize("copier", _COPIERS)
    @pytest.mark.parametrize("kwargs", _SPACES)
    def test_a_round_trip_is_equal_by_value(
        self, copier: Callable[[Any], Any], kwargs: dict[str, Any]
    ) -> None:
        original = _space(**kwargs)
        assert original.num_levels == 3
        copied = copier(original)
        assert copied is not original
        assert type(copied) is THBSplineSpace
        _assert_same_space(copied, original)

    @pytest.mark.parametrize("copier", _COPIERS)
    def test_a_stale_space_stays_stale(self, copier: Callable[[Any], Any]) -> None:
        grid = _grid()
        space = THBSplineSpace(_root(), grid)
        grid.refine(2, [8, 8], [10, 10])
        with pytest.raises(RuntimeError, match="stale"):
            space.active_basis(0)
        copied = copier(space)
        with pytest.raises(RuntimeError, match="stale"):
            copied.active_basis(0)
        with pytest.raises(RuntimeError, match="stale"):
            copied.tabulate_basis(0, np.array([[0.01, 0.01]]))
        with pytest.raises(RuntimeError, match="stale"):
            copied.max_active_per_cell()

    @pytest.mark.parametrize("copier", _DETACHING_COPIERS)
    def test_refining_the_copys_grid_stales_the_copy_and_not_the_original(
        self, copier: Callable[[Any], Any]
    ) -> None:
        original = _space()
        copied = copier(original)
        assert copied.grid is not original.grid
        copied.grid.refine(2, [8, 8], [10, 10])
        with pytest.raises(RuntimeError, match="stale"):
            copied.active_basis(0)
        assert original.active_basis(0).size > 0
        assert original.grid.num_cells == _space().grid.num_cells

    def test_a_shallow_copy_shares_the_grid_and_so_the_staleness(self) -> None:
        original = _space()
        copied = copy.copy(original)
        assert copied.grid is original.grid
        original.grid.refine(2, [8, 8], [10, 10])
        with pytest.raises(RuntimeError, match="stale"):
            copied.active_basis(0)

    @pytest.mark.parametrize("copier", _DETACHING_COPIERS)
    def test_a_space_reports_the_grid_it_was_copied_with(
        self, copier: Callable[[Any], Any]
    ) -> None:
        space = _space()
        copied_space, copied_grid, copied_root = copier((space, space.grid, space.root_space))
        assert copied_space.grid is copied_grid
        assert copied_space.root_space is copied_root

    @pytest.mark.parametrize("copier", _COPIERS)
    def test_a_subclass_keeps_its_class_and_attributes(self, copier: Callable[[Any], Any]) -> None:
        original = _TaggedSpace("a", _root(), _grid())
        copied = copier(original)
        assert type(copied) is _TaggedSpace
        assert copied.tag == ["a"]
        _assert_same_space(copied, original)

    @pytest.mark.parametrize("copier", _DETACHING_COPIERS)
    def test_a_deep_copy_does_not_share_a_mutable_attribute(
        self, copier: Callable[[Any], Any]
    ) -> None:
        original = _TaggedSpace("a", _root(), _grid())
        copied = copier(original)
        copied.tag.append("changed")
        assert original.tag == ["a"]


# ──────────────────────────────────────────────────────────────────────────────
# THBSpline
# ──────────────────────────────────────────────────────────────────────────────


def _spline(rank: int) -> THBSpline:
    space = _space()
    shape = (space.num_total_basis,) if rank == 0 else (space.num_total_basis, rank)
    return THBSpline(space, np.random.default_rng(5).uniform(-1.0, 1.0, shape))


class TestSpline:
    """``THBSpline`` copies and pickles by value, with its space."""

    @pytest.mark.parametrize("copier", _COPIERS)
    @pytest.mark.parametrize("rank", [0, 2], ids=["scalar", "vector"])
    def test_a_round_trip_is_equal_by_value(self, copier: Callable[[Any], Any], rank: int) -> None:
        original = _spline(rank)
        copied = copier(original)
        assert copied is not original
        assert type(copied) is THBSpline
        np.testing.assert_array_equal(copied.control_points, original.control_points)
        assert not copied.control_points.flags.writeable
        assert copied.rank == original.rank
        assert copied.degree == original.degree
        _assert_same_space(copied.space, original.space)
        pts = np.random.default_rng(6).uniform(size=(40, 2))
        got = copied.evaluate(pts)
        assert got.shape == original.evaluate(pts).shape
        np.testing.assert_allclose(got, original.evaluate(pts), rtol=0.0, atol=VALUE_ATOL)

    @pytest.mark.parametrize("copier", _COPIERS)
    def test_a_spline_of_a_stale_space_stays_stale(self, copier: Callable[[Any], Any]) -> None:
        grid = _grid()
        space = THBSplineSpace(_root(), grid)
        spline = THBSpline(space, np.ones(space.num_total_basis))
        grid.refine(2, [8, 8], [10, 10])
        with pytest.raises(RuntimeError, match="stale"):
            copier(spline).evaluate(np.array([[0.3, 0.3]]))

    @pytest.mark.parametrize("copier", _DETACHING_COPIERS)
    def test_a_spline_reports_the_space_it_was_copied_with(
        self, copier: Callable[[Any], Any]
    ) -> None:
        spline = _spline(0)
        copied_spline, copied_space = copier((spline, spline.space))
        assert copied_spline.space is copied_space
        assert copied_space is not spline.space

    def test_a_shallow_copy_shares_the_space(self) -> None:
        spline = _spline(0)
        assert copy.copy(spline).space is spline.space

    @pytest.mark.parametrize("copier", _COPIERS)
    def test_a_subclass_keeps_its_class_and_attributes(self, copier: Callable[[Any], Any]) -> None:
        space = _space()
        original = _TaggedSpline("b", space, np.arange(space.num_total_basis, dtype=float))
        copied = copier(original)
        assert type(copied) is _TaggedSpline
        assert copied.tag == ["b"]
        np.testing.assert_array_equal(copied.control_points, original.control_points)

    @pytest.mark.parametrize("copier", _DETACHING_COPIERS)
    def test_a_deep_copy_does_not_share_a_mutable_attribute(
        self, copier: Callable[[Any], Any]
    ) -> None:
        space = _space()
        original = _TaggedSpline("b", space, np.ones(space.num_total_basis))
        copied = copier(original)
        copied.tag.append("changed")
        assert original.tag == ["b"]


# ──────────────────────────────────────────────────────────────────────────────
# MultiLevelExtraction
# ──────────────────────────────────────────────────────────────────────────────


class TestExtraction:
    """``MultiLevelExtraction`` copies and pickles by value, with its space and target."""

    @pytest.mark.parametrize("copier", _COPIERS)
    @pytest.mark.parametrize("target", ["bezier", "lagrange", "cardinal"])
    def test_a_round_trip_is_equal_by_value(
        self, copier: Callable[[Any], Any], target: str
    ) -> None:
        original = MultiLevelExtraction(_space(), target)  # type: ignore[arg-type]
        copied = copier(original)
        assert copied is not original
        assert type(copied) is MultiLevelExtraction
        assert copied.target == target
        assert copied.num_elements == original.num_elements
        _assert_same_space(copied.space, original.space)
        for cid in range(0, original.num_elements, 3):
            np.testing.assert_array_equal(copied.active_basis(cid), original.active_basis(cid))
            for method in ("operator", "multilevel_operator"):
                np.testing.assert_allclose(
                    getattr(copied, method)(cid),
                    getattr(original, method)(cid),
                    rtol=0.0,
                    atol=VALUE_ATOL,
                )

    @pytest.mark.parametrize("copier", _COPIERS)
    def test_an_extraction_of_a_stale_space_stays_stale(self, copier: Callable[[Any], Any]) -> None:
        grid = _grid()
        extraction = MultiLevelExtraction(THBSplineSpace(_root(), grid))
        grid.refine(2, [8, 8], [10, 10])
        copied = copier(extraction)
        with pytest.raises(RuntimeError, match="stale"):
            copied.operator(0)
        with pytest.raises(RuntimeError, match="stale"):
            copied.multilevel_operator(0)

    @pytest.mark.parametrize("copier", _DETACHING_COPIERS)
    def test_an_extraction_reports_the_space_and_grid_it_was_copied_with(
        self, copier: Callable[[Any], Any]
    ) -> None:
        extraction = MultiLevelExtraction(_space())
        copied, copied_space, copied_grid = copier(
            (extraction, extraction.space, extraction.space.grid)
        )
        assert copied.space is copied_space
        assert copied_space.grid is copied_grid
        assert copied_space is not extraction.space

    @pytest.mark.parametrize("copier", _COPIERS)
    def test_a_subclass_keeps_its_class_and_attributes(self, copier: Callable[[Any], Any]) -> None:
        original = _TaggedExtraction("c", _space())
        copied = copier(original)
        assert type(copied) is _TaggedExtraction
        assert copied.tag == ["c"]
        np.testing.assert_allclose(
            copied.operator(0), original.operator(0), rtol=0.0, atol=VALUE_ATOL
        )

    @pytest.mark.parametrize("copier", _DETACHING_COPIERS)
    def test_a_deep_copy_does_not_share_a_mutable_attribute(
        self, copier: Callable[[Any], Any]
    ) -> None:
        original = _TaggedExtraction("c", _space())
        copied = copier(original)
        copied.tag.append("changed")
        assert original.tag == ["c"]
