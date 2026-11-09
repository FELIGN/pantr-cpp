"""A refused ``tol`` is refused before the user's callable is sampled."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import create_uniform_space, interpolate_bspline, l2_project_bspline
from pantr.grid import partition_grid, tensor_product_grid
from pantr.mpi import DistributedSpace, l2_project_bspline_distributed
from pantr.quad import PointsLattice


class _Recorder:
    """A projected function that records whether it was called."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, lattice: PointsLattice) -> npt.NDArray[np.float64]:
        self.calls += 1
        return np.asarray(lattice.pts_per_dir[0] ** 2, dtype=np.float64)


@pytest.mark.parametrize("tol", [float("nan"), float("inf"), float("-inf")])
def test_l2_project_bspline_refuses_tol_before_sampling(tol: float) -> None:
    """Bug caught: the callable, costly or with side effects, ran before ``tol`` was checked."""
    func = _Recorder()
    space = create_uniform_space(3, 4)
    with pytest.raises(ValueError, match="tol must be finite"):
        l2_project_bspline(func, space, tol=tol)
    assert func.calls == 0


def test_l2_project_bspline_samples_for_a_finite_tol() -> None:
    """Control: the recorder does see a call when ``tol`` is accepted."""
    func = _Recorder()
    space = create_uniform_space(3, 4)
    l2_project_bspline(func, space, tol=1e-12)
    assert func.calls >= 1


@pytest.mark.parametrize("tol", [float("nan"), float("inf"), float("-inf")])
def test_interpolate_bspline_refuses_tol_before_sampling(tol: float) -> None:
    """Bug caught: the callable ran on the node lattice before ``tol`` was checked."""
    func = _Recorder()
    space = create_uniform_space(3, 4)
    with pytest.raises(ValueError, match="tol must be finite"):
        interpolate_bspline(func, space, tol=tol)
    assert func.calls == 0


def test_interpolate_bspline_samples_for_a_finite_tol() -> None:
    """Control: the recorder does see a call when ``tol`` is accepted."""
    func = _Recorder()
    interpolate_bspline(func, create_uniform_space(3, 4), tol=1e-12)
    assert func.calls >= 1


class _SingleRankComm:
    """A one-rank stand-in for an mpi4py communicator (rank, size and a SUM allreduce)."""

    rank = 0
    size = 1

    def allreduce(self, data: Any) -> Any:
        """Return the one rank's contribution, which is the sum.

        Args:
            data (Any): The rank's contribution.

        Returns:
            Any: ``data``.
        """
        return data


@pytest.mark.parametrize("tol", [float("nan"), float("inf"), float("-inf")])
def test_l2_project_bspline_distributed_refuses_tol_before_sampling(tol: float) -> None:
    """Bug caught: the callable ran on the lattice and its traces before ``tol`` was checked."""
    func = _Recorder()
    space = create_uniform_space(3, 4)
    part = partition_grid(tensor_product_grid(space), 1)
    ds = DistributedSpace(space, part, _SingleRankComm())
    with pytest.raises(ValueError, match="tol must be finite"):
        l2_project_bspline_distributed(func, ds, tol=tol)
    assert func.calls == 0


def test_l2_project_bspline_distributed_samples_for_a_finite_tol() -> None:
    """Control: the recorder does see a call when ``tol`` is accepted."""
    func = _Recorder()
    space = create_uniform_space(3, 4)
    part = partition_grid(tensor_product_grid(space), 1)
    ds = DistributedSpace(space, part, _SingleRankComm())
    l2_project_bspline_distributed(func, ds, tol=1e-12)
    assert func.calls >= 1
