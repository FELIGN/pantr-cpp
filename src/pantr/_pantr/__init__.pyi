"""pantr's compiled core (private: use the pantr package)."""

from pantr._pantr import _kernels as _kernels
from pantr._pantr import _testing as _testing
from pantr._pantr import basis as basis
from pantr._pantr import bezier as bezier
from pantr._pantr import change_basis as change_basis
from pantr._pantr import geometry as geometry
from pantr._pantr import quad as quad
from pantr._pantr import roots as roots
from pantr._pantr import tolerance as tolerance
from pantr._pantr import transform as transform

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
