"""Bernstein product coefficients for the B-spline product modules (interim).

:meth:`pantr.bezier.Bezier.multiply` runs in C++ (``include/pantr/bezier/product.hpp``). The
B-spline product modules, not yet ported, still multiply Bézier segments given as arrays; the two
helpers here reach the same C++ formula through ``_pantr._kernels``, so that one implementation
serves both (DR-2's coexistence surface, removed with its last caller).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .._ext import ext

_kernels: Any = ext._kernels
"""The ported Bernstein kernels (interim surface, untyped: overloaded on the dtype)."""


def _bernstein_product_coefficients(
    b_f: npt.NDArray[np.float32 | np.float64],
    b_g: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Compute Bézier control points of the product of two Bézier segments.

    Applies the Bernstein product formula element-wise over the rank axis:

    .. math::

        d_k = \frac{1}{\binom{p+q}{k}} \sum_{i=\max(0,k-q)}^{\min(p,k)}
              \binom{p}{i} \binom{q}{k-i}\, b_f[i] \cdot b_g[k-i]

    Args:
        b_f (npt.NDArray[np.float32 | np.float64]): Control points of the first
            Bézier segment, shape ``(p+1, rank)``.
        b_g (npt.NDArray[np.float32 | np.float64]): Control points of the second
            Bézier segment, shape ``(q+1, rank)``, cast to ``b_f``'s dtype.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Product Bézier control points of
        shape ``(p+q+1, rank)``, in ``b_f``'s dtype.

    Raises:
        ValueError: If the shapes are not ``(p+1, rank)`` and ``(q+1, rank)`` with one rank.
    """
    return _product(b_f, b_g, "bernstein_product")


def _bernstein_product_coefficients_nd(
    b_f: npt.NDArray[np.float32 | np.float64],
    b_g: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Compute tensor-product Bézier control points of the product of two Bézier patches.

    The multivariate Bernstein product formula, per direction as in
    :func:`_bernstein_product_coefficients`, over every direction.

    Args:
        b_f (npt.NDArray[np.float32 | np.float64]): Control points of the first patch,
            shape ``(*p+1, rank)``.
        b_g (npt.NDArray[np.float32 | np.float64]): Control points of the second patch,
            shape ``(*q+1, rank)`` with the same number of directions, cast to ``b_f``'s dtype.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Product control points of shape
        ``(*(p+q+1), rank)``, in ``b_f``'s dtype.

    Raises:
        ValueError: If the operands differ in their number of directions or in rank.
    """
    return _product(b_f, b_g, "bernstein_product_nd")


def _product(
    b_f: npt.NDArray[np.float32 | np.float64],
    b_g: npt.NDArray[np.float32 | np.float64],
    kernel: str,
) -> npt.NDArray[np.float32 | np.float64]:
    """Call a compiled product on C-contiguous operands of ``b_f``'s dtype.

    Args:
        b_f (npt.NDArray[np.float32 | np.float64]): The first operand.
        b_g (npt.NDArray[np.float32 | np.float64]): The second operand.
        kernel (str): ``"bernstein_product"`` or ``"bernstein_product_nd"`` (the float32
            function carries an ``_f32`` suffix).

    Returns:
        npt.NDArray[np.float32 | np.float64]: The product's control points.
    """
    f32 = b_f.dtype == np.float32
    dtype = np.float32 if f32 else np.float64
    result: npt.NDArray[np.float32 | np.float64] = getattr(
        _kernels, kernel + ("_f32" if f32 else "")
    )(np.ascontiguousarray(b_f, dtype=dtype), np.ascontiguousarray(b_g, dtype=dtype))
    return result
