"""Process-level thread policy for MPI runs.

Without coordination, every MPI rank evaluating PaNTr's parallel kernels runs one
thread per CPU it may use, so a node running ``R`` ranks executes ``R x n_cores``
compute threads -- heavy oversubscription. To prevent this, every MPI-engaging entry
point of :mod:`pantr.mpi` (:class:`~pantr.mpi.DistributedSpace`,
:func:`~pantr.mpi.from_dolfinx`, and any future ones) applies a process-level default
on first use: the calling thread's parallelism on the compiled extension's thread pool
(:mod:`pantr._parallel`) is limited to **one thread per rank** (flat MPI). A malformed
``PANTR_NUM_THREADS`` makes applying it raise :class:`ValueError`.

The default is sticky (applied at most once per process) and never overrides explicit
user configuration; see :func:`_ensure_default_thread_policy`. The pool's capacity is
never modified, so the count can always be raised back at runtime (e.g. for rank-0
postprocessing) via ``pantr.set_num_threads`` or :func:`configure_threads`.

This module never imports ``mpi4py`` -- it only touches the thread setting of the
compiled extension -- so it works with duck-typed communicators (serial tests) and in
MPI-free installs alike. It does not limit BLAS threads: pantr's own dense linear
algebra is compiled Eigen, which starts no thread of its own.

Note:
    The thread setting is **thread-local**: the policy and :func:`configure_threads`
    only affect kernels launched from the calling thread -- in SPMD practice, the rank's
    main thread. Kernels launched from other user-created threads keep the full default
    width; call ``pantr.set_num_threads`` inside each such thread if needed.
"""

from __future__ import annotations

import threading
from typing import Final

from .. import _parallel

_DEFAULT_THREADS_PER_RANK: Final[int] = 1
"""Threads granted to each MPI rank by the default policy (flat MPI)."""

_policy_applied: Final[threading.Event] = threading.Event()
"""Sticky guard applied at most once per process.

Uses :class:`threading.Event` so that ``is_set()`` and ``set()`` are atomic
without a separate lock.
"""


def configure_threads(threads_per_rank: int = 1) -> None:
    """Explicitly set the number of threads this process (MPI rank) uses.

    The explicit alternative to the default policy: call it on every rank -- at any
    point, including before any :mod:`pantr.mpi` object is built -- to choose how many
    threads each rank may use in a hybrid MPI + threads run. Marks the thread count as
    explicitly configured, so the default one-thread policy never overrides it. Also
    the right call when running serial PaNTr under your own ``mpiexec`` without any
    :mod:`pantr.mpi` machinery.

    Args:
        threads_per_rank (int): Threads this process may use. Must be >= 1 and at most
            the thread pool's capacity (the CPUs this process may run on, lowered by
            ``PANTR_NUM_THREADS``). Defaults to 1.

    Raises:
        ValueError: If *threads_per_rank* is less than 1 or exceeds the thread pool's
            capacity, or if ``PANTR_NUM_THREADS`` is not a positive integer.

    Note:
        The thread setting is thread-local: this call governs kernels launched from
        the calling thread only. Call it on each thread individually if running PaNTr
        kernels from multiple threads; in SPMD practice, calling it once on the rank's
        main thread is sufficient.
    """
    _parallel.set_num_threads(threads_per_rank)
    _policy_applied.set()


def _ensure_default_thread_policy() -> None:
    """Apply the default MPI thread policy (1 thread per rank), at most once.

    Called by every MPI-engaging entry point of :mod:`pantr.mpi` before any other
    work. No-op when the policy already ran in this process or when the thread count
    was explicitly configured (a non-empty ``PANTR_NUM_THREADS`` environment variable,
    ``pantr.set_num_threads`` / ``pantr.num_threads``, or :func:`configure_threads`).
    Sticky: once applied, later user changes are never overridden by subsequent
    entry-point calls.

    Note:
        Every new MPI entry point added to :mod:`pantr.mpi` must call this hook
        first; there is no structural chokepoint (communicators are duck-typed and
        ``mpi4py`` is never imported here).
    """
    if _policy_applied.is_set() or _parallel._threads_explicitly_configured():
        return
    _parallel._set_num_threads_raw(_DEFAULT_THREADS_PER_RANK)
    _policy_applied.set()


def _reset_policy_for_testing() -> None:
    """Reset all sticky thread-policy state (testing only).

    Clears the policy-applied flag and the explicit-configuration flag in
    :mod:`pantr._parallel`. The thread count itself is not restored; callers must snapshot
    and restore it around tests.
    """
    _policy_applied.clear()
    _parallel._user_configured.clear()


__all__ = ["configure_threads"]
