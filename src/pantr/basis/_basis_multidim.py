"""Multidimensional basis function evaluation.

The Layer 2 wrapper of the tensor-product tabulation: it normalizes the points,
validates or allocates ``out``, and has the C++ library combine the per-direction
1D bases (``pantr._pantr.basis``), at scattered points or on a points lattice.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt

from .._ext import ext
from ..quad import PointsLattice
from ._basis_1D import _Family
from ._basis_utils import _allocate_or_validate_out


class _Order(IntEnum):
    """Index order of a flattened tensor product, with the wire values of ``pantr::Order``."""

    C = 0
    """The last index varies fastest."""
    F = 1
    """The first index varies fastest."""


def _tabulate_tensor(  # noqa: PLR0913
    family: _Family,
    degrees: tuple[int, ...],
    variant: int,
    pts: npt.ArrayLike | PointsLattice,
    order: Literal["C", "F"] = "C",
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Tabulate a tensor-product basis at scattered points or on a points lattice.

    Row ``p`` is a point, column ``f`` a tensor-product function, each value the
    product of the per-direction 1D values. Functions are flattened in ``order``
    ('C': the last direction's index varies fastest, 'F': the first's); on a
    lattice the points are flattened in the same order.

    Args:
        family (_Family): The basis family.
        degrees (tuple[int, ...]): Per-direction degrees, already validated.
        variant (int): The Lagrange node family's wire value; read for the
            Lagrange family only.
        pts (npt.ArrayLike | PointsLattice): A 2D ``(n_points, dim)`` array of
            scattered points, or a points lattice.
        order (Literal["C", "F"]): Ordering of the functions (and of the lattice's
            points). Defaults to 'C'.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. This follows NumPy's
            style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of shape (n_points, n_basis_functions).
        If `out` was provided, returns the same array.

    Raises:
        ValueError: If the points array is not 2D or has no coordinate, the number of
            degrees differs from the dimension of the points, or `out` is provided and
            has incorrect shape or dtype.
    """
    order_code = _Order.F if order == "F" else _Order.C
    # The bindings are overloaded on the dtype (float and double instantiations); the points
    # are either, so they and the output reach the binding untyped. The enumerations cross as
    # plain ints (.value): an int subclass would make nanobind skip its exact-match pass and take
    # the first overload that converts, the double one.
    tabulate: Any
    handle: Any
    if isinstance(pts, PointsLattice):
        handle = pts._h
        dim = pts.dim
        n_points = int(handle.num_points)
        dtype = pts.dtype
        tabulate = ext.basis.tabulate_lattice
    else:
        arr = np.asarray(pts)
        if arr.ndim != 2:  # noqa: PLR2004
            raise ValueError("Input points must be a 2D array.")
        if arr.dtype not in (np.float32, np.float64):  # a non-native byte order converts too
            arr = arr.astype(np.float64)
        handle = np.ascontiguousarray(arr)
        dim = arr.shape[1]
        n_points = arr.shape[0]
        dtype = arr.dtype
        tabulate = ext.basis.tabulate_points

    if out is None or dim < 1 or len(degrees) != dim:
        # An inconsistent dimension is refused by the C++ call, before any output is looked at.
        return cast(
            "npt.NDArray[np.float32 | np.float64]",
            tabulate(family.value, list(degrees), variant, handle, order_code.value, None),
        )

    n_functions = int(np.prod([d + 1 for d in degrees]))
    out = _allocate_or_validate_out(out, (n_points, n_functions), dtype)
    # The C++ call writes C-contiguous memory; any other layout gets a temporary, copied in after.
    target = out if out.flags.c_contiguous else np.empty(out.shape, out.dtype)
    tabulate(family.value, list(degrees), variant, handle, order_code.value, target)
    if target is not out:
        out[...] = target
    return out


__all__ = ["_tabulate_tensor"]
