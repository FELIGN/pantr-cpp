"""Shared Layer 2 helpers for the ``bezier`` package.

Utilities that allocate output arrays and invoke the C++ Bernstein kernel.
These helpers are private and not part of the public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._ext import ext

_kernels: Any = ext._kernels
"""The ported Bernstein kernels (interim surface, untyped: overloaded on the dtype)."""

if TYPE_CHECKING:
    import numpy.typing as npt


def _tabulate_bernstein_1d_fast(
    degree: int,
    pts: npt.NDArray[np.float32 | np.float64],
    dtype: npt.DTypeLike,
) -> npt.NDArray[np.float32 | np.float64]:
    """Allocate an output array and evaluate the Bernstein basis of given degree.

    Thin Layer 2 wrapper around the C++ Bernstein kernel (``pantr._pantr._kernels``).
    Inputs are assumed already validated: ``degree >= 0``, ``pts`` is 1-D, and
    ``pts.dtype`` matches ``dtype`` exactly (a mismatch is refused by the binding with a
    ``TypeError``).

    Args:
        degree (int): Polynomial degree.
        pts (npt.NDArray[np.float32 | np.float64]): Evaluation points, shape
            ``(n_pts,)``.
        dtype (npt.DTypeLike): Target floating-point dtype for the output.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Basis values of shape
        ``(n_pts, degree + 1)``.
    """
    basis = np.empty((pts.shape[0], degree + 1), dtype=dtype)
    _kernels.tabulate_bernstein_1d(degree, pts, basis)
    return basis
