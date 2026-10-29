"""Ported kernels for Python callers not yet ported (private, interim)."""

from typing import Annotated, overload

import numpy
from numpy.typing import NDArray

@overload
def tabulate_bernstein_1d(
    degree: int,
    pts: Annotated[NDArray[numpy.float64], dict(shape=(None,), device="cpu", writable=False)],
    out: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device="cpu")],
) -> None: ...
@overload
def tabulate_bernstein_1d(
    degree: int,
    pts: Annotated[NDArray[numpy.float32], dict(shape=(None,), device="cpu", writable=False)],
    out: Annotated[NDArray[numpy.float32], dict(shape=(None, None), device="cpu")],
) -> None:
    """Write the Bernstein basis of `degree` at `pts` into `out` (any strides, same dtype)."""

@overload
def tabulate_bernstein_derivatives_1d(
    degree: int,
    pts: Annotated[NDArray[numpy.float64], dict(shape=(None,), device="cpu", writable=False)],
    n_deriv: int,
    out: Annotated[NDArray[numpy.float64], dict(shape=(None, None, None), device="cpu")],
) -> None: ...
@overload
def tabulate_bernstein_derivatives_1d(
    degree: int,
    pts: Annotated[NDArray[numpy.float32], dict(shape=(None,), device="cpu", writable=False)],
    n_deriv: int,
    out: Annotated[NDArray[numpy.float32], dict(shape=(None, None, None), device="cpu")],
) -> None:
    """Write the Bernstein derivatives of orders 0..n_deriv at `pts` into `out` (n, n_deriv + 1, degree + 1), any strides, same dtype."""
