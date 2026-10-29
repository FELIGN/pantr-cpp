"""Probes for tests and benchmarks (private)."""

from typing import Annotated

import numpy
from numpy.typing import NDArray

def count_threads_used(chunks: int, hold_us: int) -> int:
    """Count the distinct threads that run a parallel loop of `chunks` chunks holding `hold_us` microseconds each."""

def routes_shared_parallelism() -> bool:
    """Report whether this module routes its parallel state through another module."""

def probe_scalar(x: float) -> float:
    """Return 2 * x: the binding's cost for one float in, one float out."""

def probe_bernstein(
    degree: int,
    pts: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
    """Bernstein basis by de Casteljau: the binding's cost for one small array in and out."""

class ProbeObject:
    """An object with one read-only integer property."""

    def __init__(self, degree: int) -> None: ...
    @property
    def degree(self) -> int:
        """The value given at construction."""
