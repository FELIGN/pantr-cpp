"""Choose the MPI thread level of a ``tests/mpi`` run before any test module initializes MPI.

``PANTR_MPI_THREAD_LEVEL`` (``single``, ``funneled``, ``serialized`` or ``multiple``) is handed to
``mpi4py.rc.thread_level``, which only takes effect before ``mpi4py.MPI`` is first imported: MPI is
initialized at that import, and the level it provides is fixed for the life of the process. A run
without the variable keeps mpi4py's default (``multiple``).
"""

from __future__ import annotations

import os

_LEVEL = os.environ.get("PANTR_MPI_THREAD_LEVEL", "")

if _LEVEL:
    import mpi4py

    mpi4py.rc.thread_level = _LEVEL
