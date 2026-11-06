"""A part count above the cell count is refused before the block factorization searches it.

``partition_grid`` on a tensor-product grid enumerated the divisors of ``n_parts`` (``O(sqrt(n))``)
before finding that no factorization fits the grid, so ``n_parts = 2**62`` on a 4-cell grid held
the interpreter for tens of seconds before the refusal it was always going to give.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from pantr.grid import TensorProductGrid, partition_grid


@pytest.mark.parametrize("backend", ["block", "auto"])
def test_huge_part_count_is_refused_at_once(backend: str) -> None:
    """``n_parts = 2**62`` on 4 cells raises within a second (the divisor search took tens)."""
    grid = TensorProductGrid([np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3)])
    start = time.perf_counter()
    with pytest.raises(ValueError, match="n_parts"):
        partition_grid(grid, 2**62, backend=backend)
    assert time.perf_counter() - start < 1.0


def test_a_part_count_that_fits_is_partitioned() -> None:
    """Control: 4 parts on 4 cells give one cell each."""
    grid = TensorProductGrid([np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3)])
    owners = np.asarray(partition_grid(grid, 4, backend="block").cell_owner)
    assert sorted(owners.tolist()) == [0, 1, 2, 3]
