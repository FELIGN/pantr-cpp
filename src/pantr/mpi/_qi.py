"""Distributed quasi-interpolation onto tensor-product B-spline spaces.

Provides :func:`quasi_interpolate_bspline_distributed`, the MPI-parallel counterpart of
:func:`~pantr.bspline.quasi_interpolate_bspline`.  Each rank evaluates the function only
on the points required by its *owned* DOFs (no redundant evaluation of halo-DOF points),
then a single ``allgather`` collective assembles the full global coefficient field.
The result is a :class:`~pantr.mpi.DistributedFunction` whose
:attr:`~pantr.mpi.DistributedFunction.local` reproduces the serial quasi-interpolant
exactly over the rank's owned cells.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_args

import numpy as np

from .._ext import ext
from ..bspline import Bspline, BsplineSpace, quasi_interpolate_bspline
from ..bspline._bspline_quasi_interpolation import QIKind
from ._comm import as_communicator
from ._distributed_function import DistributedFunction
from ._distributed_space import DistributedSpace
from ._thread_policy import _ensure_default_thread_policy

_cpp: Any = ext.mpi
"""The compiled distributed algorithms (private)."""

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt


def quasi_interpolate_bspline_distributed(
    func: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
    distributed_space: DistributedSpace,
    *,
    kind: QIKind = "llm",
) -> DistributedFunction:
    """Quasi-interpolate a callable onto a distributed tensor-product B-spline space.

    The MPI-parallel counterpart of :func:`~pantr.bspline.quasi_interpolate_bspline`.
    Each rank evaluates ``func`` only on the interior points required by its *owned*
    basis functions (Lee-Lyche-Mørken functionals); a single ``allgather`` at the end
    assembles the global coefficient field.  The returned
    :class:`~pantr.mpi.DistributedFunction` agrees with the serial quasi-interpolant
    pointwise over every owned cell.

    Construction requires one MPI collective (``comm.allgather``) after the local
    computation.  Per-rank ``func`` evaluation is purely local: no rank ever evaluates
    ``func`` outside its windowed parametric sub-domain.

    Args:
        func (Callable): Function to quasi-interpolate.  Called on a flat
            ``(M, dim)`` point array; must return ``(M,)`` (scalar) or ``(M, rank)``
            (vector-valued).
        distributed_space (DistributedSpace): The distributed space to interpolate onto.
            Its ``global_space`` must be a :class:`~pantr.bspline.BsplineSpace`.
        kind (QIKind): Quasi-interpolant kind.  Only ``"llm"`` (Lee-Lyche-Mørken) is
            currently supported.  Defaults to ``"llm"``.

    Returns:
        DistributedFunction: A distributed function whose
        :attr:`~pantr.mpi.DistributedFunction.local` quasi-interpolates ``func`` over
        this rank's owned cells, and whose
        :attr:`~pantr.mpi.DistributedFunction.global_function` holds the full assembled
        global coefficient field (identical on every rank after the ``allgather``).

    Raises:
        TypeError: If ``distributed_space.global_space`` is not a
            :class:`~pantr.bspline.BsplineSpace`.
        ValueError: If ``kind`` is not recognized, or if ``func`` returns an output
            with an invalid shape (0-D, more than 2-D, or wrong leading dimension).

    Note:
        All internal computation uses ``float64``.  The global control points are cast
        to ``global_space.dtype`` before assembly, consistent with the serial
        :func:`~pantr.bspline.quasi_interpolate_bspline`.

    Example:
        Needs a live MPI communicator and several ranks, so it is not run as a doctest.

        >>> from mpi4py import MPI  # doctest: +SKIP
        >>> import numpy as np  # doctest: +SKIP
        >>> from pantr.bspline import create_uniform_space  # doctest: +SKIP
        >>> from pantr.mpi import create_distributed_space  # doctest: +SKIP
        >>> from pantr.mpi import quasi_interpolate_bspline_distributed  # doctest: +SKIP
        >>> space = create_uniform_space([2, 2], [8, 8])  # doctest: +SKIP
        >>> ds = create_distributed_space(space, MPI.COMM_WORLD)  # doctest: +SKIP
        >>> dfn = quasi_interpolate_bspline_distributed(  # doctest: +SKIP
        ...     lambda p: np.sin(p[:, 0]) * np.cos(p[:, 1]), ds
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
    if kind not in get_args(QIKind):
        valid = ", ".join(repr(v) for v in get_args(QIKind))
        raise ValueError(f"Unknown kind {kind!r}; expected one of {valid}.")

    local = distributed_space.local
    if local is not None:
        # local.space is BsplineSpace because global_space is BsplineSpace (checked above).
        assert isinstance(local.space, BsplineSpace)
        # Run serial LLM QI on the windowed local space.
        local_bspline = quasi_interpolate_bspline(func, local.space, kind=kind)
        # Flatten control points to (n_local_dofs, rank_dim); shape is (*num_basis, rank_dim).
        cp = np.asarray(local_bspline.control_points, dtype=np.float64)
        cp_flat = np.ascontiguousarray(cp.reshape(local.space.num_total_basis, -1))
        local_to_global = np.ascontiguousarray(local.local_to_global_dof, dtype=np.int64)
        owned_mask = np.ascontiguousarray(local.owned_dof_mask, dtype=np.bool_)
    else:
        local_to_global = np.empty(0, dtype=np.int64)
        owned_mask = np.empty(0, dtype=np.bool_)
        cp_flat = np.empty((0, 0), dtype=np.float64)

    # One collective: every rank sends its owned (global DOF, coefficient) pairs.
    global_cp_flat, _ = _cpp.gather_owned(
        as_communicator(distributed_space.comm),
        global_space.num_total_basis,
        local_to_global,
        owned_mask,
        cp_flat,
    )
    num_basis = tuple(global_space.num_basis)
    global_cp = np.asarray(global_cp_flat).astype(global_space.dtype, copy=False)
    global_bspline = Bspline(global_space, global_cp.reshape(*num_basis, -1))
    return DistributedFunction(global_bspline, distributed_space)


__all__ = ["quasi_interpolate_bspline_distributed"]
