"""Distributed L2 projection onto tensor-product B-spline spaces.

Provides :func:`l2_project_bspline_distributed`, the MPI-parallel counterpart of
:func:`~pantr.bspline.l2_project_bspline`.  The L2 load is a sum over cells, so it splits
over the cell partition: every rank evaluates ``func`` on the whole global quadrature
lattice, keeps the values at the quadrature points of the cells it owns (the others are set
to zero) and contracts them into a per-component load tensor; a single ``allreduce`` sums
the loads across ranks, and the (replicated) boundary rows and Kronecker solve recover the
global coefficients.  The per-direction mass matrices are partition-independent and built
identically on every rank, so only the load is communicated.  The result is a
:class:`~pantr.mpi.DistributedFunction` whose global coefficients equal the serial L2
projection's up to the order of summation.  Evaluating ``func`` only on the owned cells is a
possible later speed-up; the computation itself runs in the compiled extension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._ext import ext
from .._interpolation_utils import resolve_svd_tolerance
from ..bspline import Bspline, BsplineSpace
from ..bspline._bspline_interpolate import _evaluate_func_on_lattice, _l2_projection, _tag
from ..quad import PointsLattice
from ._comm import as_communicator
from ._distributed_function import DistributedFunction
from ._distributed_space import DistributedSpace
from ._thread_policy import _ensure_default_thread_policy

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy.typing as npt

_cpp: Any = ext.mpi
"""The compiled distributed algorithms (private)."""


def _trace_values(
    func: Callable[..., npt.ArrayLike],
    projection: Any,  # noqa: ANN401 -- the compiled L2 projection
    trace: int,
    n_components: int,
    dtype: np.dtype[Any],
) -> npt.NDArray[np.floating[Any]]:
    """Sample ``func`` on one boundary trace's lattice, every component at once.

    Args:
        func (Callable): Function being projected.
        projection (Any): The compiled projection.
        trace (int): The trace's index.
        n_components (int): The number of output components.
        dtype (np.dtype): The computation dtype.

    Returns:
        npt.NDArray: The values, shape ``(*trace_grid, n_components)``, C-contiguous in
        ``dtype``.

    Raises:
        ValueError: The function returned an array that does not reshape to the lattice.
    """
    boundary_nodes = list(projection.trace_nodes(trace))
    grid_shape = tuple(a.shape[0] for a in boundary_nodes)
    raw = np.asarray(func(PointsLattice(boundary_nodes)))
    if not np.issubdtype(raw.dtype, np.floating):
        raw = raw.astype(np.float64)
    return np.ascontiguousarray(raw.reshape(*grid_shape, n_components), dtype=dtype)


def l2_project_bspline_distributed(  # noqa: PLR0913
    func: Callable[..., npt.ArrayLike],
    distributed_space: DistributedSpace,
    *,
    n_quad: int | Sequence[int] | None = None,
    quadrature: Literal["gauss-legendre", "gauss-lobatto"] = "gauss-legendre",
    boundary_interpolation: bool | Sequence[tuple[bool, bool]] = False,
    tol: float | None = None,
) -> DistributedFunction:
    """L2-project a callable onto a distributed tensor-product B-spline space.

    The MPI-parallel counterpart of :func:`~pantr.bspline.l2_project_bspline`.  L2
    assembly is per-element and maps directly onto the cell partition: each rank
    evaluates ``func`` on the whole global quadrature lattice, keeps the values at the
    quadrature points of its *owned* cells (the others count as zero) and contracts them
    into a per-component load tensor; a single ``allreduce`` sums the global load across
    ranks; and the replicated Kronecker solve recovers the global coefficients.  The
    returned :class:`~pantr.mpi.DistributedFunction` agrees with the serial L2 projection,
    up to the order in which the load is summed.

    The per-direction mass matrices are partition-independent and built identically on
    every rank (cheap ``n_dofs_i x n_dofs_i`` systems), so only the load is communicated.
    ``boundary_interpolation`` rows are handled after the reduce: each boundary trace is
    partition-independent, so every rank recomputes the same boundary row (the boundary
    trace spans the whole face, so it is not attributable to a single owning cell -- every
    rank recomputes it identically from the global lattice).  Construction requires one
    MPI collective (``comm.allreduce``) after the local assembly.

    Args:
        func (Callable[..., npt.ArrayLike]): Function to project.  Called as
            ``func(lattice)`` where ``lattice`` is a :class:`~pantr.quad.PointsLattice`
            of quadrature points (the serial convention); must return an array of shape
            ``(n_total,)`` for scalar or ``(n_total, rank)`` for vector-valued functions.
        distributed_space (DistributedSpace): The distributed space to project onto.  Its
            ``global_space`` must be a :class:`~pantr.bspline.BsplineSpace`.
        n_quad (int | Sequence[int] | None): Quadrature points per element per direction.
            Defaults to ``degree + 1``.
        quadrature (Literal["gauss-legendre", "gauss-lobatto"]): Quadrature rule type.
            Defaults to ``"gauss-legendre"``.
        boundary_interpolation (bool | Sequence[tuple[bool, bool]]): Replace boundary rows
            with interpolation conditions.  ``False`` (default) is a pure L2 projection,
            ``True`` interpolates at all non-periodic boundaries, and a sequence of
            ``(left, right)`` pairs sets per-direction flags.
        tol (float | None): SVD truncation tolerance for the per-direction solves.  If
            ``None``, defaults to ``100 * machine_epsilon``.

    Returns:
        DistributedFunction: A distributed function whose
        :attr:`~pantr.mpi.DistributedFunction.local` L2-projects ``func`` over this
        rank's owned cells, and whose
        :attr:`~pantr.mpi.DistributedFunction.global_function` holds the full assembled
        global coefficient field (identical on every rank after the ``allreduce``).

    Raises:
        TypeError: If ``distributed_space.global_space`` is not a
            :class:`~pantr.bspline.BsplineSpace`.
        ValueError: If ``n_quad`` or ``boundary_interpolation`` is inconsistent with the
            global space, if ``func`` returns an output with an invalid shape, or if the
            reduced load does not match the expected stacked shape ``(*num_basis,
            n_components)`` (a symptom of ``func`` returning inconsistent shapes across
            ranks), or if ``tol`` is NaN or infinite.

    Note:
        ``func`` MUST be rank-independent: for a given quadrature lattice it must return
        the same shape and dtype on every rank.  The reduction (``comm.allreduce``) is a
        collective that every rank must reach with a matching contribution; if ``func``
        raised on a subset of ranks (e.g. a shape error seen by some ranks only) those
        ranks would abort before the collective and deadlock the rest, and mixed dtypes
        would corrupt the reduction.

    Note:
        The output dtype is inferred from the return value of ``func`` (as in the serial
        :func:`~pantr.bspline.l2_project_bspline`).  Unlike the serial path -- where
        :class:`~pantr.bspline.Bspline` raises on a control-point/space dtype mismatch --
        the distributed path is more lenient: it coerces the assembled global control
        points to ``global_space.dtype`` rather than raising.

    Example:
        Needs a live MPI communicator and several ranks, so it is not run as a doctest.

        >>> from mpi4py import MPI  # doctest: +SKIP
        >>> import numpy as np  # doctest: +SKIP
        >>> from pantr.bspline import create_uniform_space  # doctest: +SKIP
        >>> from pantr.mpi import create_distributed_space  # doctest: +SKIP
        >>> from pantr.mpi import l2_project_bspline_distributed  # doctest: +SKIP
        >>> space = create_uniform_space([2, 2], [8, 8])  # doctest: +SKIP
        >>> ds = create_distributed_space(space, MPI.COMM_WORLD)  # doctest: +SKIP
        >>> dfn = l2_project_bspline_distributed(  # doctest: +SKIP
        ...     lambda lat: np.sin(lat.pts_per_dir[0]), ds
        ... )
        >>> local = dfn.local  # rank-local Bspline on the windowed space  # doctest: +SKIP
    """
    _ensure_default_thread_policy()

    global_space = distributed_space.global_space
    if not isinstance(global_space, BsplineSpace):
        raise TypeError(
            f"distributed_space.global_space must be a BsplineSpace; "
            f"got {type(global_space).__name__!r}."
        )

    projection, _, _ = _l2_projection(global_space, n_quad, quadrature, boundary_interpolation)
    # Refuse a bad tol before the callable is sampled: it may be costly or have side effects.
    resolved_tol = float(resolve_svd_tolerance(global_space.dtype, tol))

    # Evaluate func on the whole global quadrature lattice and on every boundary trace; the
    # compiled code masks the lattice values to this rank's owned cells.
    quad_nodes_per_dir = list(projection.nodes)
    quad_grid_shape = tuple(a.shape[0] for a in quad_nodes_per_dir)
    components, out_dtype = _evaluate_func_on_lattice(
        func, PointsLattice(quad_nodes_per_dir), quad_grid_shape
    )
    n_components = len(components)
    compute = np.dtype(np.result_type(global_space.dtype, out_dtype))
    values = np.ascontiguousarray(np.stack(components, axis=-1), dtype=compute)
    traces = [
        _trace_values(func, projection, t, n_components, compute)
        for t in range(len(projection.traces))
    ]

    project = getattr(_cpp, f"l2_project_{_tag(np.dtype(global_space.dtype))}_{_tag(compute)}")
    coeffs = project(
        as_communicator(distributed_space.comm),
        projection,
        np.ascontiguousarray(distributed_space.partition.cell_owner, dtype=np.int32),
        values,
        n_components,
        traces,
        resolved_tol,
    )
    global_cp = np.asarray(coeffs).astype(global_space.dtype, copy=False)
    global_bspline = Bspline(global_space, global_cp)
    return DistributedFunction(global_bspline, distributed_space)


__all__ = ["l2_project_bspline_distributed"]
