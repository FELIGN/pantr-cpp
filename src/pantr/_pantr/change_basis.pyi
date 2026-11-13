"""Change-of-basis matrices (private)."""

from typing import Annotated

import numpy
from numpy.typing import NDArray

def build(builder: int, degree: int, variant: int) -> NDArray[numpy.float64]:
    """Build a change-of-basis matrix (a fresh array)."""

def cached(
    kind: int, degree: int, variant: int
) -> Annotated[NDArray[numpy.float64], dict(writable=False)]:
    """The process table's matrix for a key, as a read-only view of the shared matrix."""

def build_f32(builder: int, degree: int, variant: int) -> NDArray[numpy.float32]:
    """Build a change-of-basis matrix (a fresh array)."""

def cached_f32(
    kind: int, degree: int, variant: int
) -> Annotated[NDArray[numpy.float32], dict(writable=False)]:
    """The process table's matrix for a key, as a read-only view of the shared matrix."""

def cardinal_to_bernstein_max_degree() -> tuple[int, int]:
    """The (float32, float64) degree limit of cardinal_to_bernstein."""

def bernstein_to_cardinal_max_degree() -> tuple[int, int]:
    """The (float32, float64) degree limit of bernstein_to_cardinal."""

def cardinal_to_legendre_max_degree() -> tuple[int, int]:
    """The (float32, float64) degree limit of cardinal_to_legendre and the dual coefficients."""

def bernstein_to_lagrange_max_degree(variant: int) -> tuple[int, int]:
    """The (float32, float64) degree limit of bernstein_to_lagrange."""
