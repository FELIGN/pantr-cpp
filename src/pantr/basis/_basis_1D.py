"""1D basis function evaluation implementations.

This module provides the Layer 2 wrappers for evaluating the 1D basis functions
(Bernstein, cardinal B-spline, Lagrange, Legendre): point normalization and
``out=`` handling here, the computation (and its validation) in the C++ library
through ``pantr._pantr.basis``.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from .._ext import ext
from ._basis_utils import (
    _allocate_or_validate_out,
    _compute_final_output_shape_1D,
    _normalize_points_1D,
    _reshaped_out,
)

if TYPE_CHECKING:
    from . import LagrangeVariant


class _Family(IntEnum):
    """Basis family, with the wire values of ``pantr::basis::Family`` in the C++ library."""

    BERNSTEIN = 0
    """Bernstein polynomials."""
    CARDINAL_BSPLINE = 1
    """Central cardinal B-splines."""
    LEGENDRE = 2
    """Orthonormal shifted Legendre polynomials."""
    LAGRANGE = 3
    """Lagrange polynomials on a node family."""


def _tabulate_basis_1D_impl_helper(
    family: _Family,
    n: int,
    t: npt.ArrayLike,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
    variant: int = 0,
) -> npt.NDArray[np.float32 | np.float64]:
    """Common implementation for tabulating 1D basis functions.

    Normalizes the points, validates or allocates ``out`` and has the C++ library
    tabulate into a C-contiguous buffer.

    Args:
        family (_Family): The basis family.
        n (int): Degree of the basis polynomials. Must be non-negative.
        t (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy array.
            Types different from float32 or float64 are automatically converted to float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. Defaults to None.
        variant (int): The Lagrange node family's wire value; read for the Lagrange family only.
            Defaults to 0.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions, with the same shape
        as the input points and the last dimension equal to (degree + 1).
        If `out` was provided, returns the same array.

    Raises:
        ValueError: If degree is negative, or if `out` is provided and has incorrect
            shape or dtype.
    """
    # Get input shape before normalization (handle scalars and lists)
    if isinstance(t, np.ndarray):
        input_shape = t.shape
    elif isinstance(t, list | tuple):
        input_shape = np.array(t).shape
    else:  # scalar
        input_shape = ()

    # The binding is overloaded on the dtype; ``pts`` is either, so it is passed untyped.
    pts: Any = np.ascontiguousarray(_normalize_points_1D(t))
    if n < 0:
        # Refused by the C++ call (with the message the suite matches) before any output is
        # sized or looked at; it raises.
        ext.basis.tabulate_1d(family.value, n, variant, pts, None)
    if out is None:
        # Allocated here rather than by the binding: an uninitialized NumPy array is the cheaper
        # hand-off for the one-point calls whose cost is all overhead.
        result: npt.NDArray[np.float32 | np.float64] = np.empty(
            (pts.shape[0], n + 1), dtype=pts.dtype
        )
        ext.basis.tabulate_1d(family.value, n, variant, pts, cast(Any, result))
        return result.reshape(_compute_final_output_shape_1D(input_shape, n + 1))

    expected_final_shape = _compute_final_output_shape_1D(input_shape, n + 1)
    out = _allocate_or_validate_out(out, expected_final_shape, pts.dtype)
    buffer, copy_back = _reshaped_out(out, (pts.shape[0], n + 1))
    # The C++ call writes C-contiguous memory; a strided view gets a temporary, copied in after.
    target = buffer if buffer.flags.c_contiguous else np.empty(buffer.shape, buffer.dtype)
    ext.basis.tabulate_1d(family.value, n, variant, pts, cast(Any, target))
    if target is not buffer:
        buffer[...] = target
    if copy_back:
        out[...] = buffer.reshape(out.shape)
    return out


def _tabulate_Bernstein_basis_1D_impl(
    n: int,
    t: npt.ArrayLike,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate the Bernstein basis polynomials of the given degree at the given points.

    Args:
        n (int): Degree of the Bernstein polynomials. Must be non-negative.
        t (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy array.
            Types different from float32 or float64 are automatically converted to float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions, with the same shape
        as the input points and the last dimension equal to (degree + 1).
        If `out` was provided, returns the same array.

    Raises:
        ValueError: If degree is negative, or if `out` is provided and has incorrect
            shape or dtype.

    Example:
        >>> vals = _tabulate_Bernstein_basis_1D_impl(2, [0.0, 0.5, 0.75, 1.0])
        >>> np.allclose(
        ...     vals,
        ...     [[1.0, 0.0, 0.0], [0.25, 0.5, 0.25], [0.0625, 0.375, 0.5625], [0.0, 0.0, 1.0]],
        ... )
        True
    """
    return _tabulate_basis_1D_impl_helper(_Family.BERNSTEIN, n, t, out)


def _tabulate_cardinal_Bspline_basis_1D_impl(
    n: int,
    t: npt.ArrayLike,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Evaluate the cardinal B-spline basis polynomials of given degree at given points.

    The cardinal B-spline basis is the set of B-spline basis functions defined
    on an interval of maximum continuity that has degree-1 contiguous
    knot spans on each side with the same length as the interval itself.
    These basis functions appear in the central knot spans
    in the case of maximum regularity uniform knot vectors.

    Explicit expression:
    \[
    B_{n,i}(t) = (1/n!) * sum_{j=0}^{n-i} binom(n+1, j) * (-1)^j * (t + n - i - j)^n
    \]
    where \( B_{n,i}(t) \) is the B-spline basis function of degree \( n \) and index \( i \)
     at point \( t \), and \( binom(a, b) \) is the binomial coefficient.

    Its actual implementation is based on the Cox-de Boor recursion formula for the
    central cardinal B-spline basis.

    Args:
        n (int): Degree of the B-spline basis. Must be non-negative.
        t (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy array.
            Types different from float32 or float64 are automatically converted to float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions, with the same shape
        as the input points and the last dimension equal to (degree + 1).
        If `out` was provided, returns the same array.

    Raises:
        ValueError: If provided degree is negative, or if `out` is provided and has incorrect
            shape or dtype.

    Example:
        >>> vals = _tabulate_cardinal_Bspline_basis_1D_impl(2, [0.0, 0.5, 1.0])
        >>> np.allclose(vals, [[0.5, 0.5, 0.0], [0.125, 0.75, 0.125], [0.0, 0.5, 0.5]])
        True
    """
    return _tabulate_basis_1D_impl_helper(_Family.CARDINAL_BSPLINE, n, t, out)


def _tabulate_Lagrange_basis_1D_impl(
    n: int,
    variant: LagrangeVariant,
    t: npt.ArrayLike,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Evaluate Lagrange basis polynomials at points using the specified variant.

    The polynomials are defined in the interval [0, 1] and are given by the formula:
    \[
    L_{n,i}(t) = \prod_{j=0}^{n} \frac{t - x_j}{x_i - x_j}
    \]
    where \( x_i \) are the points at which the basis is evaluated.

    The variant determines the points at which the basis is evaluated. Computed by the
    barycentric formula in double precision, a point within ``16 * eps`` of a node getting that
    node's Kronecker row exactly.

    Args:
        n (int): Degree of the B-spline basis. Must be non-negative.
        variant (LagrangeVariant): Variant of the Lagrange basis.
        t (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy array.
            Types different from float32 or float64 are automatically converted to float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions, with the same shape
        as the input points and the last dimension equal to (degree + 1).
        If `out` was provided, returns the same array.

    Raises:
        ValueError: If provided degree is negative, if `out` is provided and has incorrect
            shape or dtype, or if `variant` is not a :class:`LagrangeVariant` value.
    """
    # Lazy import: the enum's module imports this one.
    from ._basis_tabulate import _variant_code  # noqa: PLC0415

    return _tabulate_basis_1D_impl_helper(_Family.LAGRANGE, n, t, out, _variant_code(variant))


def _tabulate_Legendre_basis_1D_impl(
    n: int,
    t: npt.ArrayLike,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate the normalized Shifted Legendre basis polynomials of the given degree.

    Args:
        n (int): Degree of the Legendre polynomials. Must be non-negative.
        t (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy array.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions.
        If `out` was provided, returns the same array.

    Raises:
        ValueError: If degree is negative, or if `out` is provided and has incorrect
            shape or dtype.
    """
    return _tabulate_basis_1D_impl_helper(_Family.LEGENDRE, n, t, out)


__all__ = [
    "_tabulate_Bernstein_basis_1D_impl",
    "_tabulate_Lagrange_basis_1D_impl",
    "_tabulate_Legendre_basis_1D_impl",
    "_tabulate_cardinal_Bspline_basis_1D_impl",
]
