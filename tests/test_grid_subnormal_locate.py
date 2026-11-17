"""``locate_many`` on cells narrower than the double spacing agrees with the same grid scaled up.

Refining ``[0, 5e-324, 1e-323]`` by 2 makes children whose bounds cannot be represented; the
located cell ids follow the rule of the normal range, so they equal those of ``[0, 1, 2]``.

Oracle: a metamorphic relation, scaling by ``2**1074`` (exact for these breakpoints).
"""

from __future__ import annotations

import numpy as np

from pantr.grid import TensorProductGrid, hierarchical_grid


def _located(breakpoints: list[float]) -> list[int]:
    """Locate the three breakpoints on the grid refined once by 2."""
    grid = hierarchical_grid(TensorProductGrid([np.array(breakpoints)]), 2)
    grid.refine(0, [0], [2])
    return [int(i) for i in grid.locate_many(np.array([[b] for b in breakpoints]))]


def test_subnormal_cells_locate_as_the_scaled_grid_does() -> None:
    """``[0, 5e-324, 1e-323]`` gives the ids ``[0, 1, 2]`` gives (the previous release did not)."""
    assert _located([0.0, 5e-324, 1e-323]) == _located([0.0, 1.0, 2.0]) == [0, 1, 3]
