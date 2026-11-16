"""Real-MPI tests of distributed calls from two Python threads of one rank (run under ``mpiexec``).

Skipped unless ``PANTR_RUN_MPI`` is set (and ``mpi4py`` is importable); see
``tests/mpi/test_distributed_mpi.py`` for the launch convention. The contract under test
(``docs/guide/distributed.md``, "Distributed calls from several Python threads"):

- below ``MPI_THREAD_MULTIPLE`` the collectives of two threads never overlap inside MPI, which
  ``MPI_THREAD_SERIALIZED`` forbids; they keep the GIL for the duration of the MPI call;
- at ``MPI_THREAD_MULTIPLE`` they release it, so another thread of a rank runs while that rank
  waits for the others.

The level MPI provides is fixed when ``mpi4py.MPI`` is first imported, so each test skips unless the
process has the level it needs. Run the serialized test as
``PANTR_RUN_MPI=1 PANTR_MPI_THREAD_LEVEL=serialized mpiexec -n 2 python -m pytest
tests/mpi/test_threads_mpi.py`` (``tests/mpi/conftest.py`` hands the variable to mpi4py); the
default run provides ``MPI_THREAD_MULTIPLE``.
"""

from __future__ import annotations

import faulthandler
import os
import threading
import time
from collections.abc import Iterator

import pytest

from pantr.bspline import create_uniform_space
from pantr.mpi import create_distributed_space

MPI = pytest.importorskip("mpi4py.MPI")

pytestmark = pytest.mark.skipif(
    not os.environ.get("PANTR_RUN_MPI"),
    reason="MPI test: set PANTR_RUN_MPI=1 and run under mpiexec",
)

_ITERATIONS = 200
"""Calls per thread; the failure the test pins (a crash or a hang) needs overlapping calls."""

_TIMEOUT_SECONDS = 60.0
"""Wall-clock bound of a test; a hang is turned into a failed run when it expires."""


@pytest.fixture
def watchdog(capfd: pytest.CaptureFixture[str]) -> Iterator[None]:
    """End the process with a traceback of every thread if the test outlasts the timeout.

    A hung MPI call cannot be interrupted from Python, and the timer of ``faulthandler`` runs on
    its own thread, so it does not need the GIL that the hung call may hold. Output capture is off
    for the test, since a capture file is lost when the process ends this way.
    """
    with capfd.disabled():
        faulthandler.dump_traceback_later(_TIMEOUT_SECONDS, exit=True)
        try:
            yield
        finally:
            faulthandler.cancel_dump_traceback_later()


def test_two_threads_under_serialized_do_not_overlap_in_mpi(watchdog: None) -> None:
    """Two threads call a pantr collective at once, on duplicated communicators.

    Under ``MPI_THREAD_SERIALIZED`` two calls inside MPI at once crash a rank or hang the job (seen
    on Intel MPI 2021.18); the pantr collectives must therefore not overlap. A hang expires the
    ``watchdog`` timeout, which ends the process.
    """
    if MPI.Query_thread() != MPI.THREAD_SERIALIZED:
        pytest.skip("needs MPI_THREAD_SERIALIZED: set PANTR_MPI_THREAD_LEVEL=serialized")
    comm = MPI.COMM_WORLD
    space = create_uniform_space([2, 2], [6, 6])
    create_distributed_space(space, comm)  # warm-up on the main thread
    comms = [comm.Dup(), comm.Dup()]
    done = [0, 0]
    errors: list[BaseException] = []

    def work(i: int) -> None:
        try:
            for _ in range(_ITERATIONS):
                create_distributed_space(space, comms[i])
                done[i] += 1
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert done == [_ITERATIONS, _ITERATIONS]
    for c in comms:
        c.Free()


def test_a_second_thread_runs_while_a_collective_waits_at_multiple(watchdog: None) -> None:
    """At ``MPI_THREAD_MULTIPLE`` a rank blocked in a collective does not hold the GIL.

    Rank 0 enters a collective on a thread that the other ranks reach only after a pause, and its
    main thread sleeps for far less than that pause. The main thread wakes while the collective
    still waits only if the call released the GIL.
    """
    if MPI.Query_thread() != MPI.THREAD_MULTIPLE:
        pytest.skip("needs MPI_THREAD_MULTIPLE (the mpi4py default)")
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("needs at least two ranks, so that a rank waits for another")
    space = create_uniform_space([2, 2], [6, 6])
    create_distributed_space(space, comm)  # warm-up
    late, wake = 2.0, 0.2
    comm.Barrier()
    if comm.rank == 0:
        finished = threading.Event()
        errors: list[BaseException] = []

        def collective() -> None:
            try:
                create_distributed_space(space, comm)
            except BaseException as e:
                errors.append(e)
            finally:
                finished.set()

        thread = threading.Thread(target=collective, daemon=True)
        thread.start()
        time.sleep(wake)
        waiting = not finished.is_set()
        thread.join()
        assert errors == []
        assert waiting, "the main thread woke only after the collective returned"
    else:
        time.sleep(late)
        create_distributed_space(space, comm)
