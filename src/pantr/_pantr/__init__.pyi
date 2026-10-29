"""pantr's compiled core (private: use the pantr package)."""

from pantr._pantr import _testing as _testing

interface_version: int = ...

isa: str = ...

simd_isa: str = ...

def max_threads() -> int:
    """Return the C++ pool capacity: the affinity CPU count, lowered by PANTR_NUM_THREADS."""

def get_num_threads() -> int:
    """Return the calling thread's parallelism setting for the C++ pool."""

def set_num_threads(n: int) -> None:
    """Set the calling thread's parallelism for the C++ pool (1 <= n <= max_threads())."""

_shared_parallelism: object
