"""Threads may make a process's first parallel calls together.

pantr's parallel entry points run on its C++ thread pool, which concurrent callers share: a call
from one Python thread never has to wait for another to finish setting anything up. The previous
release needed a barrier here (its import compiled kernels on a background thread, and a
concurrent parallel call aborted the process), so what is pinned is that a fresh process whose
first calls come from several threads at once completes them correctly. It has to be a fresh
process: a first call is a once-per-process event, so a warm test process cannot observe it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests._tree_env import TESTING_INSTALLED, TREE_SRC, subprocess_env

_THREADED_FIRST_CALLS = textwrap.dedent(
    """
    import sys
    import threading

    import numpy as np

    import pantr
    from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
    from pantr.grid import hierarchical_grid, uniform_grid

    n_threads = int(sys.argv[1])
    print(pantr.__file__)

    # The identity map on the unit square, and a grid over it. Nothing has been called yet.
    knots = np.array([0.0, 0.0, 1.0, 1.0])
    space = BsplineSpace([BsplineSpace1D(knots, 1), BsplineSpace1D(knots, 1)])
    control_points = np.array([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
    spline = Bspline(space, control_points)
    grid = uniform_grid([[0.0, 1.0], [0.0, 1.0]], [40, 40])
    hgrid = hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], [8, 8]), 2)
    hgrid.refine(0, [2, 2], [6, 6])
    hgrid.refine(1, [6, 6], [10, 10])

    rng = np.random.default_rng(0)
    targets = rng.uniform(0.05, 0.95, size=(200, 2))
    points = rng.uniform(-0.1, 1.1, size=(5000, 2))
    inside = np.all((points >= 0.0) & (points <= 1.0), axis=1)

    start = threading.Barrier(n_threads)
    failures = []

    def work(k):
        try:
            start.wait()
            if k % 3 == 0:
                cell_ids, ref_coords = spline.locate(targets)
                assert np.all(cell_ids >= 0), "a target of the identity map was not found"
                np.testing.assert_allclose(ref_coords, targets, rtol=0.0, atol=1e-9)
            elif k % 3 == 1:
                ids = grid.locate_many(points)
                assert np.array_equal(ids >= 0, inside), "membership differs from the box test"
            else:
                ids = hgrid.locate_many(points)
                assert np.array_equal(ids >= 0, inside), "membership differs from the box test"
                cell_lo, cell_hi = hgrid.collect_cell_bounds()
                assert cell_lo.shape == (hgrid.num_cells, 2)
                assert np.all(cell_lo < cell_hi), "a collected cell is empty"
                area = np.prod(cell_hi - cell_lo, axis=1).sum()
                assert abs(area - 1.0) < 1e-12, f"the cells do not tile the unit square: {area}"
        except BaseException as exc:  # reported to the parent, which fails the test
            failures.append(f"thread {k}: {exc!r}")

    threads = [threading.Thread(target=work, args=(k,)) for k in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)
    """
)
"""A fresh process that makes its first ``Bspline.locate``, ``TensorProductGrid.locate_many`` and
``HierarchicalGrid.locate_many`` / ``collect_cell_bounds`` calls from several threads at once,
started together by a barrier."""


class TestConcurrentFirstCalls:
    """Threads may make the first calls together."""

    @pytest.mark.parametrize("attempt", range(3))
    def test_threads_calling_locate_first_thing_succeed(self, attempt: int) -> None:
        """Eight threads call ``Bspline.locate`` and the grids' batch methods as a first act.

        This is the race the previous release's barrier existed for, asked of the C++ entry
        points. The child checks its own results
        (the identity map is inverted, the grids' membership agrees with the box test, and the
        hierarchical grid's cells tile the unit square), so a clean exit means the calls
        succeeded, not only that they did not abort.

        Args:
            attempt (int): Repetition index; each attempt is its own process.
        """
        del attempt  # only distinguishes the processes
        result = subprocess.run(
            [sys.executable, "-c", _THREADED_FIRST_CALLS, "8"],
            capture_output=True,
            text=True,
            check=False,
            env=subprocess_env(),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        origin = Path(result.stdout.splitlines()[0]).resolve()
        if not TESTING_INSTALLED:
            assert origin.is_relative_to(TREE_SRC), f"the child imported pantr from {origin}"
