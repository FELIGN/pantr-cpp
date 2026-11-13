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
  or a pickle does not share a mutable attribute with the original;
* a pickle written by the previous release, whose objects were plain ``__slots__`` objects, still
  loads and equals the object this release builds from the same pieces. Those pickles are not
  kept as bytes: the state they held is built by hand from stand-in classes with the previous
  release's slots (the way the ``Bspline`` tests of the previous release's pickles do), and what
  the previous release derived from the pieces is filled with values that fail the comparison if
  the loader reads them.
"""

from __future__ import annotations

import copy
import pickle
import sys
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
from pantr.bspline import _thb_spline as thb_spline_module
from pantr.bspline import _thb_spline_space as thb_space_module
from pantr.bspline import multilevel_extraction as extraction_module
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


# ──────────────────────────────────────────────────────────────────────────────
# Pickles written by the previous release
# ──────────────────────────────────────────────────────────────────────────────

_PREVIOUS_PROTOCOLS = range(2, pickle.HIGHEST_PROTOCOL + 1)
"""Protocols that pickle a ``__slots__`` object; the previous release refused 0 and 1."""

_StandIn = Callable[..., Any]


def _previous_space_slots(
    grid: HierarchicalGrid, *, truncate: bool = True, regularity: int | None = None
) -> dict[str, Any]:
    """Return the slots the previous release's space held, with nonsense for what it derived.

    The pieces the loader reads (root space, grid, truncation, regularity, snapshot) are the real
    ones. What the previous release derived from them (active functions, truncated coefficients,
    supports, caches) is filled with values that differ from any real one, so a loader that read
    them could not equal a freshly built space.
    """
    return {
        "_root_space": _root(),
        "_grid": grid,
        "_truncate": truncate,
        "_regularity": (regularity, regularity),
        "_grid_snapshot": (grid.max_level, grid.num_cells, grid.version),
        "_active_funcs": (np.array([-1]),),
        "_func_offset": np.array([0, -1]),
        "_num_active": -1,
        "_trunc": {0: thb_space_module._TruncCoeffs(0, (0, 0), np.full((1, 1), np.nan))},
        "_support": (),
        "_level_spaces": (),
        "_contrib_cache": {"unread": None},
        "_max_active_per_cell": -1,
    }


def _dump_previous(protocol: int, build: Callable[[_StandIn], Any]) -> bytes:
    """Pickle what ``build`` makes from stand-ins for the previous release's classes.

    ``build`` receives ``make(module, name, slots, subclass=None)``, which returns an instance of a
    class named ``name`` in ``module`` with exactly ``slots`` as its ``__slots__``, holding those
    values. The stand-in takes the place of the module's class while the pickle is written, so the
    pickle names the class as the previous release's did and loading it finds this release's class.

    With ``subclass`` the instance is of a subclass of that name in this test module, which has no
    ``__slots__`` and the attribute ``tag``, laid out as the previous release's subclass was.

    Args:
        protocol (int): The pickle protocol.
        build (Callable[[_StandIn], Any]): Builds the object given ``make``.

    Returns:
        bytes: The pickle.
    """
    with pytest.MonkeyPatch.context() as patch:

        def make(
            module: Any,
            name: str,
            slots: dict[str, Any],
            subclass: str | None = None,
        ) -> Any:
            cls = type(name, (), {"__slots__": tuple(slots), "__module__": module.__name__})
            patch.setattr(module, name, cls)
            if subclass is not None:
                cls = type(subclass, (cls,), {"__module__": __name__})
                patch.setattr(sys.modules[__name__], subclass, cls)
            obj = cls()
            for key, value in slots.items():
                setattr(obj, key, value)
            if subclass is not None:
                obj.tag = ["c"]
            return obj

        return pickle.dumps(build(make), protocol=protocol)


def _previous_space(make: _StandIn, grid: HierarchicalGrid, **kwargs: Any) -> Any:
    return make(thb_space_module, "THBSplineSpace", _previous_space_slots(grid, **kwargs))


@pytest.mark.parametrize("protocol", _PREVIOUS_PROTOCOLS)
class TestPreviousRelease:
    """Pickles of the previous release's plain objects load into the compiled wrappers."""

    @pytest.mark.parametrize("kwargs", _SPACES)
    def test_a_space_loads_and_equals_a_new_one(
        self, protocol: int, kwargs: dict[str, Any]
    ) -> None:
        reg = kwargs.get("regularity")
        data = _dump_previous(
            protocol,
            lambda make: _previous_space(
                make, _grid(), truncate=kwargs.get("truncate", True), regularity=reg
            ),
        )

        loaded = pickle.loads(data)

        assert type(loaded) is THBSplineSpace
        _assert_same_space(loaded, _space(**kwargs))

    def test_a_space_loaded_from_a_grid_refined_since_is_stale(self, protocol: int) -> None:
        grid = _grid()
        slots = _previous_space_slots(grid)
        grid.refine(2, [8, 8], [10, 10])

        loaded = pickle.loads(
            _dump_previous(protocol, lambda make: make(thb_space_module, "THBSplineSpace", slots))
        )

        with pytest.raises(RuntimeError, match="stale"):
            loaded.active_basis(0)

    def test_a_space_watches_the_grid_it_was_loaded_with(self, protocol: int) -> None:
        loaded = pickle.loads(_dump_previous(protocol, lambda make: _previous_space(make, _grid())))
        loaded.active_basis(0)

        loaded.grid.refine(2, [8, 8], [10, 10])

        with pytest.raises(RuntimeError, match="stale"):
            loaded.active_basis(0)

    def test_a_subclass_of_a_space_keeps_its_class_and_attributes(self, protocol: int) -> None:
        data = _dump_previous(
            protocol,
            lambda make: make(
                thb_space_module, "THBSplineSpace", _previous_space_slots(_grid()), "_TaggedSpace"
            ),
        )

        loaded = pickle.loads(data)

        assert type(loaded) is _TaggedSpace
        assert loaded.tag == ["c"]
        _assert_same_space(loaded, _space())

    @pytest.mark.parametrize("rank", [0, 2], ids=["scalar", "vector"])
    def test_a_spline_loads_and_equals_a_new_one(self, protocol: int, rank: int) -> None:
        want = _spline(rank)
        # The previous release held the control points as a read-only (num_total_basis, rank) array.
        stored = want.control_points.reshape(want.space.num_total_basis, -1).copy()
        stored.flags.writeable = False

        def build(make: _StandIn) -> Any:
            slots = {
                "_space": _previous_space(make, _grid()),
                "_control_points": stored,
                "_scalar": rank == 0,
            }
            return make(thb_spline_module, "THBSpline", slots)

        loaded = pickle.loads(_dump_previous(protocol, build))

        assert type(loaded) is THBSpline
        assert loaded.rank == want.rank
        np.testing.assert_array_equal(loaded.control_points, want.control_points)
        assert loaded.control_points.shape == want.control_points.shape
        assert not loaded.control_points.flags.writeable
        _assert_same_space(loaded.space, want.space)
        pts = np.random.default_rng(6).uniform(size=(40, 2))
        np.testing.assert_allclose(
            loaded.evaluate(pts), want.evaluate(pts), rtol=0.0, atol=VALUE_ATOL
        )

    def test_a_subclass_of_a_spline_keeps_its_class_and_attributes(self, protocol: int) -> None:
        n = _space().num_total_basis

        def build(make: _StandIn) -> Any:
            slots = {
                "_space": _previous_space(make, _grid()),
                "_control_points": np.arange(float(n)).reshape(n, 1),
                "_scalar": True,
            }
            return make(thb_spline_module, "THBSpline", slots, "_TaggedSpline")

        loaded = pickle.loads(_dump_previous(protocol, build))

        assert type(loaded) is _TaggedSpline
        assert loaded.tag == ["c"]
        np.testing.assert_array_equal(loaded.control_points, np.arange(float(n)))

    @pytest.mark.parametrize("target", ["bezier", "lagrange", "cardinal"])
    def test_an_extraction_loads_and_equals_a_new_one(self, protocol: int, target: str) -> None:
        def build(make: _StandIn) -> Any:
            slots = {
                "_space": _previous_space(make, _grid()),
                "_target": target,
                "_oslo": (),
                "_ext": {},
                "_coeffs_cache": {(0, (0, 0), 0): ((), np.full(1, np.nan))},
            }
            return make(extraction_module, "MultiLevelExtraction", slots)

        loaded = pickle.loads(_dump_previous(protocol, build))
        want = MultiLevelExtraction(_space(), target)  # type: ignore[arg-type]

        assert type(loaded) is MultiLevelExtraction
        assert loaded.target == target
        assert loaded.num_elements == want.num_elements
        _assert_same_space(loaded.space, want.space)
        for cid in range(0, want.num_elements, 3):
            for method in ("operator", "multilevel_operator"):
                np.testing.assert_allclose(
                    getattr(loaded, method)(cid),
                    getattr(want, method)(cid),
                    rtol=0.0,
                    atol=VALUE_ATOL,
                )

    def test_an_extraction_of_a_stale_space_loads_stale(self, protocol: int) -> None:
        grid = _grid()
        space_slots = _previous_space_slots(grid)
        grid.refine(2, [8, 8], [10, 10])

        def build(make: _StandIn) -> Any:
            slots = {
                "_space": make(thb_space_module, "THBSplineSpace", space_slots),
                "_target": "bezier",
                "_oslo": (),
                "_ext": {},
                "_coeffs_cache": {},
            }
            return make(extraction_module, "MultiLevelExtraction", slots)

        loaded = pickle.loads(_dump_previous(protocol, build))

        with pytest.raises(RuntimeError, match="stale"):
            loaded.operator(0)

    def test_a_subclass_of_an_extraction_keeps_its_class_and_attributes(
        self, protocol: int
    ) -> None:
        def build(make: _StandIn) -> Any:
            slots = {
                "_space": _previous_space(make, _grid()),
                "_target": "bezier",
                "_oslo": (),
                "_ext": {},
                "_coeffs_cache": {},
            }
            return make(extraction_module, "MultiLevelExtraction", slots, "_TaggedExtraction")

        loaded = pickle.loads(_dump_previous(protocol, build))

        assert type(loaded) is _TaggedExtraction
        assert loaded.tag == ["c"]
        assert loaded.target == "bezier"
