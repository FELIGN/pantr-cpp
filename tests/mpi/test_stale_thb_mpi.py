"""Real-MPI tests: a stale THB space is refused on every rank (run under ``mpiexec``).

Skipped unless ``PANTR_RUN_MPI`` is set (and ``mpi4py`` is importable). The dedicated CI
job runs them as ``PANTR_RUN_MPI=1 mpiexec -n {2,3} python -m pytest tests/mpi/``.

A rank that owns no cell builds no local space, which is where staleness used to be checked;
it constructed while the ranks with cells raised, and then waited forever in the next
collective. The tests here never enter such a collective after construction: they gather each
rank's outcome instead, a collective every rank reaches whatever happened.
"""

from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import THBSplineSpace, create_thb_space, create_uniform_space
from pantr.grid import Partition
from pantr.mpi import DistributedSpace, quasi_interpolate_thb_spline_distributed

MPI = pytest.importorskip("mpi4py.MPI")

pytestmark = pytest.mark.skipif(
    not os.environ.get("PANTR_RUN_MPI"),
    reason="MPI test: set PANTR_RUN_MPI=1 and run under mpiexec",
)


def _thb_space(*, stale: bool) -> THBSplineSpace:
    """Return the space of issue #40, its grid refined after construction when ``stale``.

    Args:
        stale (bool): Whether to refine the grid after the space is built.

    Returns:
        THBSplineSpace: The space, identical on every rank.
    """
    thb = create_thb_space(create_uniform_space([2, 2], [4, 4]))
    if stale:
        thb.grid.refine(0, [0, 0], [1, 1])
    return thb


def _outcomes(thb: THBSplineSpace, owner: npt.NDArray[np.int32]) -> list[str]:
    """Construct the distributed space on every rank and gather what each rank got.

    Args:
        thb (THBSplineSpace): The global space.
        owner (npt.NDArray[np.int32]): The owner of every cell of ``thb.grid``.

    Returns:
        list[str]: Per rank, ``"constructed"`` or the exception's type and message.
    """
    comm = MPI.COMM_WORLD
    try:
        DistributedSpace(thb, Partition(owner, comm.size), comm)
        outcome = "constructed"
    except Exception as e:  # any exception must still reach the gather below
        outcome = f"{type(e).__name__}: {e}"
    return list(comm.allgather(outcome))


def test_stale_thb_space_is_refused_on_every_rank_including_one_without_cells() -> None:
    """Rank 0 owns every cell; every rank, the empty ones included, raises the same error."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("needs a rank that owns no cell: run under mpiexec -n 2 or more")
    thb = _thb_space(stale=True)
    outcomes = _outcomes(thb, np.zeros(thb.grid.num_cells, dtype=np.int32))
    for rank, got in enumerate(outcomes):
        assert got.startswith("RuntimeError: THBSplineSpace is stale"), (rank, got)
    assert len(set(outcomes)) == 1, outcomes  # the same exception, message included


def test_stale_thb_space_split_across_ranks_is_refused_on_every_rank() -> None:
    """Every rank owns cells: they all raised before the fix too, and still do."""
    comm = MPI.COMM_WORLD
    thb = _thb_space(stale=True)
    outcomes = _outcomes(thb, (np.arange(thb.grid.num_cells) % comm.size).astype(np.int32))
    for rank, got in enumerate(outcomes):
        assert got.startswith("RuntimeError: THBSplineSpace is stale"), (rank, got)


def test_fresh_thb_space_with_a_rank_without_cells_quasi_interpolates() -> None:
    """The control: the same empty ranks on a fresh space construct and finish a collective."""
    comm = MPI.COMM_WORLD
    thb = _thb_space(stale=False)
    owner = np.zeros(thb.grid.num_cells, dtype=np.int32)
    ds = DistributedSpace(thb, Partition(owner, comm.size), comm)
    assert ds.owns_cells == (comm.rank == 0)
    dfn = quasi_interpolate_thb_spline_distributed(lambda p: p[:, 0] + p[:, 1], ds)
    assert dfn.global_function.control_points.shape[0] == thb.num_total_basis
