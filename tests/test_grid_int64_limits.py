"""The ``int64`` limits of the grids: what is refused, with what error, and that nothing changes.

The compiled grids count cells in ``int64``. A level whose per-axis cell count, or a grid whose
cell count, would pass ``2**63 - 1`` is refused with ``OverflowError`` (the previous release
computed with Python integers and refused nothing). A cell id that is not an ``int64`` is a
``TypeError`` at the call, as for any argument of the compiled library.
"""

from __future__ import annotations

from typing import Any

import pytest

from pantr.grid import HierarchicalGrid, TensorProductGrid, hierarchical_grid, uniform_grid

# One root cell and factor 2: level L has 2**L cells on the axis, so level 62 is the last whose
# count (2**62) fits in int64 and level 63 (2**63) does not.
_LAST_LEVEL = 62


def _deepest_grid() -> HierarchicalGrid:
    """Return a 1-D grid refined down the first cell to level ``_LAST_LEVEL``."""
    g = hierarchical_grid(uniform_grid([[0.0, 1.0]], 1), 2)
    for level in range(_LAST_LEVEL):
        g.refine(level, [0], [1])
    assert g.max_level == _LAST_LEVEL
    return g


def _snapshot(g: HierarchicalGrid) -> tuple[Any, ...]:
    """Return what a refused refinement must leave alone: version, counts and blocks."""
    return (g.version, g.num_cells, g.max_level, g._blocks)


def test_refine_past_the_int64_level_raises_and_leaves_the_grid_unchanged() -> None:
    """Refining the deepest level into a level with >= 2**63 cells per axis is refused."""
    g = _deepest_grid()
    before = _snapshot(g)
    with pytest.raises(OverflowError, match=r"level 62.*level 63.*axis 0"):
        g.refine(_LAST_LEVEL, [0], [1])
    assert _snapshot(g) == before


def test_refine_cells_past_the_int64_level_raises_and_leaves_the_grid_unchanged() -> None:
    """The same refusal through ``refine_cells``."""
    g = _deepest_grid()
    cid = g.cell_id(_LAST_LEVEL, [0])
    assert cid is not None
    before = _snapshot(g)
    with pytest.raises(OverflowError, match=r"level 62.*level 63.*axis 0"):
        g.refine_cells([cid])
    assert _snapshot(g) == before


def test_level_cells_per_axis_is_exact_up_to_int64_and_raises_past_it() -> None:
    """``level_cells_per_axis`` is a formula only while the count fits in int64."""
    g = hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], [1, 3]), [2, 1])
    assert g.level_cells_per_axis(_LAST_LEVEL) == (2**_LAST_LEVEL, 3)
    with pytest.raises(OverflowError, match=r"level 63.*axis 0"):
        g.level_cells_per_axis(_LAST_LEVEL + 1)
    with pytest.raises(OverflowError, match=r"level 64.*axis 0"):
        g.level_cells_per_axis(_LAST_LEVEL + 2)


def test_a_tensor_product_grid_with_2_to_the_63_cells_or_more_raises() -> None:
    """Five axes of 8192 cells give 2**65 cells: refused at construction."""
    axes = [[float(i) for i in range(8193)] for _ in range(5)]
    with pytest.raises(OverflowError):
        TensorProductGrid(axes)
    with pytest.raises(OverflowError):
        uniform_grid([[0.0, 1.0]] * 5, 8192)


def test_a_tensor_product_grid_just_under_the_limit_is_accepted() -> None:
    """Four axes of 8192 cells (2**52 cells) are not refused."""
    g = uniform_grid([[0.0, 1.0]] * 4, 8192)
    assert g.num_cells == 8192**4


@pytest.mark.parametrize("big", [2**63, 2**64, -(2**63) - 1])
def test_a_cell_id_that_is_not_an_int64_is_a_type_error(big: int) -> None:
    """A cell id past 64 bits cannot cross into the compiled library: ``TypeError``."""
    tp = uniform_grid([[0.0, 1.0]], 4)
    hg = hierarchical_grid(tp, 2)
    for grid in (tp, hg):
        with pytest.raises(TypeError):
            grid.cell_bounds(big)
    with pytest.raises(TypeError):
        hg.cell_level(big)
    with pytest.raises(TypeError):
        hg.refine_cells([big])
