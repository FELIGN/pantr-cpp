"""Ported kernels for Python callers not yet ported (private, interim)."""

from typing import Annotated

import numpy
from numpy.typing import NDArray

def bezier_elevate_degree_1d(
    degree: int,
    ctrl: Annotated[
        NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu", writable=False)
    ],
    increment: int,
) -> NDArray[numpy.float64]:
    """Elevate the degree of a (degree + 1, n_cols) Bernstein array."""

def bernstein_product(
    a: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
    b: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
) -> NDArray[numpy.float64]:
    """Bernstein coefficients of the pointwise product of two (*degrees_plus_1, rank) arrays (the 1D formula for one direction, the tensor formula otherwise)."""

def bernstein_product_nd(
    a: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
    b: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
) -> NDArray[numpy.float64]:
    """As bernstein_product, always by the tensor formula."""

def bezier_elevate_degree_1d_f32(
    degree: int,
    ctrl: Annotated[
        NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu", writable=False)
    ],
    increment: int,
) -> NDArray[numpy.float32]:
    """Elevate the degree of a (degree + 1, n_cols) Bernstein array."""

def bernstein_product_f32(
    a: Annotated[NDArray[numpy.float32], dict(order="C", device="cpu", writable=False)],
    b: Annotated[NDArray[numpy.float32], dict(order="C", device="cpu", writable=False)],
) -> NDArray[numpy.float32]:
    """Bernstein coefficients of the pointwise product of two (*degrees_plus_1, rank) arrays (the 1D formula for one direction, the tensor formula otherwise)."""

def bernstein_product_nd_f32(
    a: Annotated[NDArray[numpy.float32], dict(order="C", device="cpu", writable=False)],
    b: Annotated[NDArray[numpy.float32], dict(order="C", device="cpu", writable=False)],
) -> NDArray[numpy.float32]:
    """As bernstein_product, always by the tensor formula."""

def interpolating_reduction_operator(
    degree: int, decrement: int
) -> Annotated[NDArray[numpy.float64], dict(writable=False)]:
    """The endpoint-interpolating reduction operator (read-only, from the process table)."""
