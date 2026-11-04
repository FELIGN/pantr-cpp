"""Thread-count control for PaNTr's parallel kernels.

Provides a thin wrapper around the thread-pool configuration so that users can
limit (or disable) the parallelism used by PaNTr without touching environment
variables directly.

PaNTr's parallel kernels run on the thread pool of its compiled extension. The pool's
capacity, and so the largest count :func:`set_num_threads` accepts, is the number of CPUs
this process may run on (its affinity mask), lowered by the ``PANTR_NUM_THREADS``
environment variable when that is set; both are read once per process. The setting is
per thread: :func:`set_num_threads` changes the parallelism of the calling thread only,
and every other thread keeps its own, the capacity by default.

Typical usage::

    import pantr

    # Query / set for the calling thread
    pantr.set_num_threads(4)
    print(pantr.get_num_threads())

    # Scoped override (restores previous value on exit)
    with pantr.num_threads(1):
        result = space.tabulate_basis(pts)  # runs serially
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Generator
from typing import Final

from pantr._ext import ext

_user_configured: Final[threading.Event] = threading.Event()
"""Set by the public setters; queried by default policies (e.g. the per-rank MPI
default in :mod:`pantr.mpi`) so they never override an explicit user decision."""


def get_num_threads() -> int:
    """Return the number of threads the calling thread's parallel kernels may use.

    Returns:
        int: The calling thread's setting, at most the pool's capacity.
    """
    return int(ext.get_num_threads())


def _set_num_threads_raw(n: int) -> None:
    """Set the calling thread's thread count without marking it as explicit configuration.

    Internal hook for default policies (e.g. the per-rank default applied by
    :mod:`pantr.mpi`): performs the same validation as :func:`set_num_threads`
    but leaves the explicit-configuration flag untouched, so a policy-applied
    value remains distinguishable from a user decision.

    Args:
        n (int): Desired thread count. Must be >= 1 and at most the pool's capacity.

    Raises:
        ValueError: If *n* is less than 1 or exceeds the pool's capacity, or if the
            ``PANTR_NUM_THREADS`` environment variable is not a positive integer.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    capacity = int(ext.max_threads())
    if n > capacity:
        raise ValueError(
            f"n must be <= the thread pool's capacity ({capacity}: the CPUs this process may "
            f"use, lowered by PANTR_NUM_THREADS), got {n}"
        )
    ext.set_num_threads(n)


def set_num_threads(n: int) -> None:
    """Set the number of threads the calling thread's parallel kernels may use.

    Calling this marks the thread count as explicitly configured: default
    policies (e.g. the per-rank MPI default in :mod:`pantr.mpi`) will never
    override it afterwards.

    Args:
        n (int): Desired thread count. Must be >= 1 and at most the pool's capacity:
            the CPUs this process may run on, lowered by ``PANTR_NUM_THREADS``.

    Raises:
        ValueError: If *n* is less than 1 or exceeds the pool's capacity, or if the
            ``PANTR_NUM_THREADS`` environment variable is not a positive integer.
    """
    _set_num_threads_raw(n)
    _user_configured.set()


def _threads_explicitly_configured() -> bool:
    """Report whether the user explicitly configured the thread count.

    ``True`` when :func:`set_num_threads` was called directly, when the
    ``PANTR_NUM_THREADS`` environment variable is set to a non-empty value (an empty one
    leaves the pool's capacity unchanged, as if unset), or when the :func:`num_threads`
    context manager was used. The flag is set on both entry *and* exit of
    :func:`num_threads` (exit restores the previous count via :func:`set_num_threads`),
    so any use of that context manager permanently marks the process as explicitly
    configured for the rest of its lifetime.

    Returns:
        bool: Whether explicit thread configuration is in effect.
    """
    return _user_configured.is_set() or os.environ.get("PANTR_NUM_THREADS", "") != ""


@contextlib.contextmanager
def num_threads(n: int) -> Generator[None, None, None]:
    """Context manager that temporarily sets the calling thread's thread count.

    On entry the calling thread's setting is changed to *n*; on exit the previous
    value is restored.

    Args:
        n (int): Desired thread count for the block.

    Yields:
        None

    Note:
        Using this context manager permanently marks the thread count as explicitly
        configured (via :func:`set_num_threads` on both entry and exit).  Default
        policies -- e.g. the per-rank MPI default in :mod:`pantr.mpi` -- will not
        override the thread count for the rest of the process lifetime, even after
        the ``with`` block exits.

    Example:
        >>> import pantr
        >>> with pantr.num_threads(1):
        ...     pantr.get_num_threads()  # all pantr operations run serially here
        1
    """
    prev = get_num_threads()
    set_num_threads(n)
    try:
        yield
    finally:
        set_num_threads(prev)


__all__ = [
    "get_num_threads",
    "num_threads",
    "set_num_threads",
]
