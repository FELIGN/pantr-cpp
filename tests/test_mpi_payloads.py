"""What :mod:`pantr.mpi` sends a duck-typed communicator (design/port_design.md §5.4).

The suite's other fakes return what they are given, so they would not notice a payload
changing dtype or shape. A recording communicator here keeps every payload, and each test
pins the format the previous release sent and the fakes elsewhere preset: the compiled
algorithms must keep calling a duck-typed communicator exactly so.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pantr.bspline import create_thb_space, create_uniform_space
from pantr.grid import Partition, partition_grid, tensor_product_grid
from pantr.mpi import (
    DistributedSpace,
    from_dolfinx,
    interpolate_bspline_distributed,
    l2_project_bspline_distributed,
    quasi_interpolate_bspline_distributed,
    quasi_interpolate_thb_spline_distributed,
)


class _Recording:
    """One rank of ``size``: records every payload; gathers return it at ``rank``'s slot.

    The other slots hold ``filler(payload)``, so a multi-rank run can be simulated far enough
    to record what this rank sends.
    """

    def __init__(self, rank: int = 0, size: int = 1, filler: Any = None) -> None:
        self.rank = rank
        self.size = size
        self.gathered: list[Any] = []
        self.reduced: list[np.ndarray] = []
        self._filler = filler

    def allgather(self, obj: Any) -> list[Any]:
        self.gathered.append(obj)
        fill = obj if self._filler is None else self._filler(obj)
        return [obj if r == self.rank else fill for r in range(self.size)]

    def allreduce(self, arr: np.ndarray) -> np.ndarray:
        self.reduced.append(np.array(arr, copy=True))
        return arr


def _flat(p: np.ndarray) -> np.ndarray:
    return np.sin(p[:, 0]) + p[:, -1]


def _flat2(p: np.ndarray) -> np.ndarray:
    return np.stack([_flat(p), 2.0 * _flat(p)], axis=1)


def _one_rank(space: Any, comm: _Recording) -> DistributedSpace:
    grid = space.grid if hasattr(space, "grid") else tensor_product_grid(space)
    return DistributedSpace(space, Partition(np.zeros(grid.num_cells, dtype=np.int32), 1), comm)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(("func", "cols"), [(_flat, None), (_flat2, 2)])
def test_collocation_sends_one_float64_block(dtype: type, func: Any, cols: int | None) -> None:
    """One gather of a C-contiguous float64 ``(m,)`` or ``(m, r)`` block, whatever the space."""
    comm = _Recording()
    space = create_uniform_space([2, 2], [3, 3], dtype=dtype)
    interpolate_bspline_distributed(func, _one_rank(space, comm))
    assert len(comm.gathered) == 1
    sent = comm.gathered[0]
    assert isinstance(sent, np.ndarray) and sent.dtype == np.float64 and sent.flags.c_contiguous
    n = space.num_total_basis
    assert sent.shape == ((n,) if cols is None else (n, cols))


def test_collocation_empty_rank_sends_empty_block() -> None:
    """A rank with no grid points sends a float64 ``(0,)`` block (a scalar function)."""
    space = create_uniform_space([1], [1])  # 2 nodes, 3 ranks: rank 2 gets none
    comm = _Recording(rank=2, size=3, filler=lambda o: np.ones(1))
    part = Partition(np.array([0], dtype=np.int32), 3)
    interpolate_bspline_distributed(_flat, DistributedSpace(space, part, comm))
    sent = comm.gathered[0]
    assert sent.dtype == np.float64 and sent.shape == (0,)


@pytest.mark.parametrize("cols", [None, 2])
def test_qi_sends_one_tuple(cols: int | None) -> None:
    """One gather of ``(int64 (k,), float64 (k, r))``, a scalar field as ``r = 1``."""
    comm = _Recording()
    space = create_uniform_space([2, 2], [3, 3])
    quasi_interpolate_bspline_distributed(_flat if cols is None else _flat2, _one_rank(space, comm))
    assert len(comm.gathered) == 1
    dofs, coeffs = comm.gathered[0]
    k = space.num_total_basis
    assert dofs.dtype == np.int64 and dofs.shape == (k,)
    assert coeffs.dtype == np.float64 and coeffs.shape == (k, 1 if cols is None else cols)


def test_qi_rank_without_cells_sends_empty_tuple() -> None:
    """A rank owning no cell sends ``(int64 (0,), float64 (0, 0))``."""
    space = create_uniform_space([2], [2])
    owner = np.zeros(2, dtype=np.int32)
    k = space.num_total_basis
    comm = _Recording(
        rank=1,
        size=2,
        filler=lambda o: (np.arange(k, dtype=np.int64), np.ones((k, 1))),
    )
    quasi_interpolate_bspline_distributed(_flat, DistributedSpace(space, Partition(owner, 2), comm))
    dofs, coeffs = comm.gathered[0]
    assert dofs.dtype == np.int64 and dofs.shape == (0,)
    assert coeffs.dtype == np.float64 and coeffs.shape == (0, 0)


@pytest.mark.parametrize("cols", [None, 2])
def test_thb_qi_keeps_a_scalar_field_one_dimensional(cols: int | None) -> None:
    """The THB quasi-interpolant sends ``(int64 (k,), float64 (k,))`` for a scalar field."""
    comm = _Recording()
    thb = create_thb_space(create_uniform_space([2, 2], [4, 4])).refine_region(0, [0, 0], [2, 2])
    quasi_interpolate_thb_spline_distributed(
        _flat if cols is None else _flat2, _one_rank(thb, comm)
    )
    dofs, coeffs = comm.gathered[0]
    k = thb.num_total_basis
    assert dofs.dtype == np.int64 and dofs.shape == (k,)
    assert coeffs.dtype == np.float64 and coeffs.shape == ((k,) if cols is None else (k, cols))


@pytest.mark.parametrize(
    ("space_dtype", "func_dtype", "sent_dtype"),
    [
        (np.float64, np.float64, np.float64),
        (np.float32, np.float32, np.float32),
        (np.float32, np.float64, np.float64),
        (np.float64, np.float32, np.float64),  # the previous release sent float32 here
    ],
)
def test_l2_reduces_one_tensor(space_dtype: type, func_dtype: type, sent_dtype: type) -> None:
    """One ``allreduce`` of the ``(*num_basis, n_components)`` load, nothing gathered."""
    comm = _Recording()
    space = create_uniform_space([2, 2], [3, 3], dtype=space_dtype)

    def func(lat: Any) -> np.ndarray:
        return _flat2(lat.get_all_points()).astype(func_dtype)

    l2_project_bspline_distributed(func, _one_rank(space, comm))
    assert comm.gathered == []
    assert len(comm.reduced) == 1
    assert comm.reduced[0].dtype == sent_dtype
    assert comm.reduced[0].shape == (*space.num_basis, 2)


def test_from_dolfinx_sends_int64_ids() -> None:
    """One gather of the rank's int64 ``(k,)`` pantr cell ids; ``rank`` is never read."""

    class SizeOnly:
        size = 1

        def __init__(self) -> None:
            self.gathered: list[Any] = []

        def allgather(self, obj: Any) -> list[Any]:
            self.gathered.append(obj)
            return [obj]

    comm = SizeOnly()
    mesh = SimpleNamespace(
        comm=comm,
        topology=SimpleNamespace(
            dim=2,
            original_cell_index=np.array([3, 1, 2, 0], dtype=np.int32),
            index_map=lambda _dim: SimpleNamespace(size_local=4),
        ),
    )
    from_dolfinx(mesh, 4)
    sent = comm.gathered[0]
    assert sent.dtype == np.int64 and sent.shape == (4,)
    np.testing.assert_array_equal(sent, [3, 1, 2, 0])


def test_partition_broadcast_sends_int64() -> None:
    """With more than one rank, rank 0 sends its partition as one int64 ``(n + 2,)`` array."""
    from pantr.mpi import create_distributed_space  # noqa: PLC0415

    comm = _Recording(rank=0, size=2, filler=lambda o: np.asarray(o)[:0])
    space = create_uniform_space([2, 2], [4, 4])
    ds = create_distributed_space(space, comm)
    sent = comm.gathered[0]
    assert sent.dtype == np.int64 and sent.shape == (16 + 2,)
    np.testing.assert_array_equal(sent[2:], ds.partition.cell_owner)
    assert (
        ds.partition.cell_owner.tolist()
        == partition_grid(tensor_product_grid(space), 2).cell_owner.tolist()
    )
