"""Shared control-point manipulation helpers (homogeneous promotion).

These functions operate purely on control-point arrays and are used across the
:mod:`~pantr.bspline` and :mod:`~pantr.cad` packages.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _append_unit_weight_column(
    control_points: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Append a trailing column of unit weights (homogeneous promotion).

    Converts Euclidean control points of shape ``(*num_basis, rank)`` into
    rational (homogeneous) control points of shape ``(*num_basis, rank + 1)``
    whose weights are all ``1``.

    Args:
        control_points (npt.NDArray[np.float32 | np.float64]): Control points of
            shape ``(*num_basis, rank)``.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Control points with a trailing
        unit-weight column, shape ``(*num_basis, rank + 1)``.
    """
    weights = np.ones((*control_points.shape[:-1], 1), dtype=control_points.dtype)
    return np.concatenate([control_points, weights], axis=-1)
