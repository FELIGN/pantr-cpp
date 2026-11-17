"""Lagrange basis nodes.

The node families of the Lagrange basis on ``[0, 1]``, computed by the C++ library
(``pantr::basis::lagrange_points``). The tabulation itself is the C++ library's
barycentric formula (``include/pantr/basis/lagrange.hpp``), reached through
:func:`pantr.basis.tabulate_lagrange_1d`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from .._array_utils import _is_float32
from .._ext import ext

if TYPE_CHECKING:
    from . import LagrangeVariant


def _get_lagrange_points(
    variant: LagrangeVariant | str, n_pts: int, dtype: npt.DTypeLike = np.float64
) -> npt.NDArray[np.float32 | np.float64]:
    """Get nodes for a Lagrange basis on [0, 1] for the given variant and size.

    The nodes are those of the quadrature rule of the same name (trapezoidal points for
    equispaced nodes, Gauss-Legendre, Gauss-Lobatto-Legendre, Chebyshev-Gauss of the first
    kind) except for ``CHEBYSHEV_2ND``, whose nodes are the Chebyshev-Lobatto points
    (endpoints included, :func:`pantr.quad.get_modified_chebyshev_nodes_1d`). Note that for
    Gauss-Lobatto-Legendre and Chebyshev second kind the number of points must be at least 2.

    Args:
        variant (LagrangeVariant | str): The variant of the Lagrange basis.
        n_pts (int): The number of points.
        dtype (npt.DTypeLike): The dtype of the nodes. If must be float32 or float64.
            Defaults to float64.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The nodes, ascending.

    Raises:
        ValueError: If dtype is not float32 or float64, ``n_pts`` is below the variant's
            minimum, or ``variant`` is not a :class:`~pantr.basis.LagrangeVariant` value.
    """
    # Lazy import: the enum's module imports pantr.quad, which imports this package.
    from ._basis_tabulate import _variant_code  # noqa: PLC0415

    build = ext.quad.lagrange_points_f32 if _is_float32(dtype) else ext.quad.lagrange_points
    return cast("npt.NDArray[np.float32 | np.float64]", build(_variant_code(variant), n_pts))


__all__ = ["_get_lagrange_points"]
