"""Distributed quasi-interpolation onto THB-spline spaces.

Provides :func:`quasi_interpolate_thb_spline_distributed`, the MPI-parallel counterpart
of :func:`~pantr.bspline.quasi_interpolate_thb_spline`.  Each rank runs the serial
Speleers-Manni hierarchical quasi-interpolant on its *windowed* local space -- which
covers every owned active DOF's full support, including the level-``l`` active leaf cell
on which the per-level functional samples ``func`` -- then keeps only its *owned* DOFs'
coefficients.  A
single ``allgather`` collective assembles the full global coefficient field.  The result
is a :class:`~pantr.mpi.DistributedFunction` whose
:attr:`~pantr.mpi.DistributedFunction.local` reproduces the serial quasi-interpolant
exactly over the rank's owned cells.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_args

import numpy as np

from .._ext import ext
from ..bspline import THBSpline, THBSplineSpace, quasi_interpolate_thb_spline
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


def quasi_interpolate_thb_spline_distributed(
    func: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
    distributed_space: DistributedSpace,
    *,
    kind: QIKind = "llm",
) -> DistributedFunction:
    """Quasi-interpolate a callable onto a distributed THB-spline space.

    The MPI-parallel counterpart of
    :func:`~pantr.bspline.quasi_interpolate_thb_spline`.  Each rank runs the serial
    Speleers-Manni hierarchical quasi-interpolant on its windowed local space and keeps
    only its *owned* DOFs' coefficients; a single ``allgather`` at the end assembles the
    global coefficient field.  The returned
    :class:`~pantr.mpi.DistributedFunction` agrees with the serial quasi-interpolant
    pointwise over every owned cell.

    The windowed local space covers the support closure of every owned DOF, so the
    per-level functional's level-``l`` active leaf cell -- the one the serial routine
    samples ``func`` on -- always lies inside the rank's windowed parametric domain.  No
    rank ever evaluates ``func`` outside that domain.  Construction requires one MPI
    collective (``comm.allgather``) after the local computation.

    Args:
        func (Callable): Function to quasi-interpolate.  Called on a flat
            ``(M, dim)`` point array; must return ``(M,)`` (scalar) or ``(M, rank)``
            (vector-valued).
        distributed_space (DistributedSpace): The distributed space to interpolate onto.
            Its ``global_space`` must be a :class:`~pantr.bspline.THBSplineSpace`.
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
            :class:`~pantr.bspline.THBSplineSpace`.
        ValueError: If ``kind`` is not recognized, or if ``func`` returns an output
            with an invalid shape (0-D, more than 2-D, or wrong leading dimension).

    Note:
        Scalar vs. vector kind is preserved (a scalar ``func`` yields a scalar
        :class:`~pantr.bspline.THBSpline`).  Like the serial
        :func:`~pantr.bspline.quasi_interpolate_thb_spline`, the result is always
        ``float64``: :class:`~pantr.bspline.THBSpline` coerces its coefficients to
        ``float64`` on construction, regardless of ``global_space.dtype``.

    Example:
        Needs a live MPI communicator and several ranks, so it is not run as a doctest.

        >>> from mpi4py import MPI  # doctest: +SKIP
        >>> import numpy as np  # doctest: +SKIP
        >>> from pantr.bspline import create_uniform_space, create_thb_space  # doctest: +SKIP
        >>> from pantr.mpi import create_distributed_space  # doctest: +SKIP
        >>> from pantr.mpi import quasi_interpolate_thb_spline_distributed  # doctest: +SKIP
        >>> thb = create_thb_space(create_uniform_space([2, 2], [8, 8]))  # doctest: +SKIP
        >>> thb = thb.refine_region(0, [0, 0], [4, 4])  # doctest: +SKIP
        >>> ds = create_distributed_space(thb, MPI.COMM_WORLD)  # doctest: +SKIP
        >>> dfn = quasi_interpolate_thb_spline_distributed(  # doctest: +SKIP
        ...     lambda p: np.sin(p[:, 0]) * np.cos(p[:, 1]), ds
        ... )
        >>> local = dfn.local  # rank-local THBSpline on the windowed space  # doctest: +SKIP
    """
    _ensure_default_thread_policy()

    global_space = distributed_space.global_space
    if not isinstance(global_space, THBSplineSpace):
        raise TypeError(
            f"distributed_space.global_space must be a THBSplineSpace; "
            f"got {type(global_space).__name__!r}."
        )
    if kind not in get_args(QIKind):
        valid = ", ".join(repr(v) for v in get_args(QIKind))
        raise ValueError(f"Unknown kind {kind!r}; expected one of {valid}.")

    local = distributed_space.local
    if local is not None:
        # local.space is THBSplineSpace because global_space is (checked above).
        assert isinstance(local.space, THBSplineSpace)
        # Run serial Speleers-Manni QI on the windowed local space.  THBSpline stores
        # coefficients per active dof already: (num_total_basis,) scalar or
        # (num_total_basis, rank) vector.  That rank is kept, so scalar funcs stay scalar,
        # matching the serial quasi-interpolant.
        local_thb = quasi_interpolate_thb_spline(func, local.space, kind=kind)
        cp = np.ascontiguousarray(local_thb.control_points, dtype=np.float64)
        local_to_global = np.ascontiguousarray(local.local_to_global_dof, dtype=np.int64)
        owned_mask = np.ascontiguousarray(local.owned_dof_mask, dtype=np.bool_)
    else:
        local_to_global = np.empty(0, dtype=np.int64)
        owned_mask = np.empty(0, dtype=np.bool_)
        cp = np.empty(0, dtype=np.float64)

    # One collective: every rank sends its owned (global DOF, coefficient) pairs.  THBSpline
    # always stores float64, so the global control points are float64 whatever
    # global_space.dtype is (consistent with the serial quasi-interpolant).
    global_cp, _ = _cpp.gather_owned(
        as_communicator(distributed_space.comm),
        global_space.num_total_basis,
        local_to_global,
        owned_mask,
        cp,
    )
    global_thb = THBSpline(global_space, np.asarray(global_cp))
    return DistributedFunction(global_thb, distributed_space)


__all__ = ["quasi_interpolate_thb_spline_distributed"]
