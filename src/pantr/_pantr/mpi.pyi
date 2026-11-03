"""Distributed algorithms over a communicator (private)."""

from collections.abc import Callable, Sequence
from typing import Annotated

import numpy
from numpy.typing import NDArray

import pantr._pantr.bspline
import pantr._pantr.grid

COMM_CAPSULE_NAME: str = ...

COMM_ABI_VERSION: int = ...

def broadcast_partition(
    comm: object, compute: Callable[[], pantr._pantr.grid.Partition]
) -> pantr._pantr.grid.Partition:
    """Rank 0's partition on every rank; `compute()` runs on rank 0 only (every rank at size 1)."""

def block_bounds(n_total: int, n_parts: int, rank: int) -> tuple[int, int]:
    """This rank's half-open block of a contiguous range split evenly."""

def collocate_f64_f64(
    comm: object, system: pantr._pantr.bspline.TensorCollocationF64, local: object, tol: float
) -> NDArray[numpy.float64]:
    """Gather every rank's value block and solve the collocation system (nodes P, space T)."""

def collocate_f64_f32(
    comm: object, system: pantr._pantr.bspline.TensorCollocationF64, local: object, tol: float
) -> NDArray[numpy.float64]:
    """Gather every rank's value block and solve the collocation system (nodes P, space T)."""

def collocate_f32_f64(
    comm: object, system: pantr._pantr.bspline.TensorCollocationF32, local: object, tol: float
) -> NDArray[numpy.float64]:
    """Gather every rank's value block and solve the collocation system (nodes P, space T)."""

def collocate_f32_f32(
    comm: object, system: pantr._pantr.bspline.TensorCollocationF32, local: object, tol: float
) -> NDArray[numpy.float32]:
    """Gather every rank's value block and solve the collocation system (nodes P, space T)."""

def l2_project_f64_f64(
    comm: object,
    projection: pantr._pantr.bspline.L2ProjectionF64,
    cell_owner: Annotated[
        NDArray[numpy.int32], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    values: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
    n_components: int,
    traces: Sequence[
        Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)]
    ],
    tol: float,
) -> NDArray[numpy.float64]:
    """Sum the owned-cell loads over the ranks and solve (space T, values Cm)."""

def l2_project_f32_f64(
    comm: object,
    projection: pantr._pantr.bspline.L2ProjectionF32,
    cell_owner: Annotated[
        NDArray[numpy.int32], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    values: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
    n_components: int,
    traces: Sequence[
        Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)]
    ],
    tol: float,
) -> NDArray[numpy.float64]:
    """Sum the owned-cell loads over the ranks and solve (space T, values Cm)."""

def l2_project_f32_f32(
    comm: object,
    projection: pantr._pantr.bspline.L2ProjectionF32,
    cell_owner: Annotated[
        NDArray[numpy.int32], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    values: Annotated[NDArray[numpy.float32], dict(order="C", device="cpu", writable=False)],
    n_components: int,
    traces: Sequence[
        Annotated[NDArray[numpy.float32], dict(order="C", device="cpu", writable=False)]
    ],
    tol: float,
) -> NDArray[numpy.float32]:
    """Sum the owned-cell loads over the ranks and solve (space T, values Cm)."""

def gather_owned(
    comm: object,
    n_global: int,
    local_to_global_dof: Annotated[
        NDArray[numpy.int64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    owned_dof_mask: Annotated[
        NDArray[numpy.bool_], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    coeffs: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
) -> tuple[NDArray[numpy.float64], bool]:
    """Gather every rank's owned (DOF, coefficient) pairs into the global field."""

def partition_from_owner(
    comm: object, local_ids: object, n_cells: int
) -> pantr._pantr.grid.Partition:
    """The partition whose owner of each cell is the rank that sent it."""
