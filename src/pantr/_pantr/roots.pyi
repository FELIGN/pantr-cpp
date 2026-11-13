"""The Builtin root provider (private)."""

from typing import Annotated

import numpy
from numpy.typing import NDArray

CLAIM_NONE: int = ...

CLAIM_COMPUTED: int = ...

CLAIM_CERTIFIED: int = ...

COMPLETE: int = ...

GAP_LOCALIZED: int = ...

GAP_UNLOCALIZED: int = ...

def capabilities() -> tuple[int, bool, bool, bool]:
    """The Builtin provider's capabilities: (max_claim, batch, monotone, multiplicity)."""

def find_roots(
    coeff: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    param_tol: float,
    value_tol: float,
) -> tuple[NDArray[numpy.float64], int]:
    """Roots of one Bernstein polynomial: (roots, completeness)."""

def find_roots_detailed(
    coeff: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    param_tol: float,
    value_tol: float,
) -> tuple[
    NDArray[numpy.float64], NDArray[numpy.int32], NDArray[numpy.int32], int, NDArray[numpy.float64]
]:
    """Roots with claims: (roots, claims, multiplicities, completeness, gaps (k, 2))."""

def find_roots_batch(
    coeffs: Annotated[
        NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu", writable=False)
    ],
    param_tol: float,
    value_tol: float,
) -> tuple[NDArray[numpy.float64], NDArray[numpy.int64], NDArray[numpy.int32]]:
    """Roots of each row: (roots (n, max(degree, 1)) NaN-padded, counts, completeness)."""

def find_monotone_root(
    coeff: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    param_tol: float,
) -> float:
    """The root of a monotone polynomial, or NaN."""

def find_monotone_root_batch(
    coeffs: Annotated[
        NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu", writable=False)
    ],
    param_tol: float,
) -> NDArray[numpy.float64]:
    """The monotone root of each row, NaN for none."""

def find_roots_f32(
    coeff: Annotated[
        NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    param_tol: float,
    value_tol: float,
) -> tuple[NDArray[numpy.float64], int]:
    """Roots of one Bernstein polynomial: (roots, completeness)."""

def find_roots_detailed_f32(
    coeff: Annotated[
        NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    param_tol: float,
    value_tol: float,
) -> tuple[
    NDArray[numpy.float64], NDArray[numpy.int32], NDArray[numpy.int32], int, NDArray[numpy.float64]
]:
    """Roots with claims: (roots, claims, multiplicities, completeness, gaps (k, 2))."""

def find_roots_batch_f32(
    coeffs: Annotated[
        NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu", writable=False)
    ],
    param_tol: float,
    value_tol: float,
) -> tuple[NDArray[numpy.float64], NDArray[numpy.int64], NDArray[numpy.int32]]:
    """Roots of each row: (roots (n, max(degree, 1)) NaN-padded, counts, completeness)."""

def find_monotone_root_f32(
    coeff: Annotated[
        NDArray[numpy.float32], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    param_tol: float,
) -> float:
    """The root of a monotone polynomial, or NaN."""

def find_monotone_root_batch_f32(
    coeffs: Annotated[
        NDArray[numpy.float32], dict(shape=(None, None), order="C", device="cpu", writable=False)
    ],
    param_tol: float,
) -> NDArray[numpy.float64]:
    """The monotone root of each row, NaN for none."""
