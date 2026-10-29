"""One pool and one per-thread setting per process across pantr's extension modules (DR-20).

``_pantr_twin`` is a second extension built from the same headers, as ``_pantr_mpi`` and
``_pantr_berf`` will be, that attaches to ``_pantr``'s table at import. ``_pantr_twin_unrouted`` is
the same source built without the attach: the control, showing on every run what the routing
prevents (each module its own pool, a setting made through pantr invisible to it). Both are
test-only and installed by ``make ext``; a wheel does not ship them.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types

import pytest

import pantr
from pantr._ext import ext, source_root
from tests._tree_env import subprocess_env

pytestmark = pytest.mark.skipif(
    source_root(ext.__file__) is None,
    reason="the twin modules are test-only builds, installed by `make ext` and not in a wheel",
)

# Chunks and hold time for the thread-count probe: enough chunks that every admitted thread finds
# one, held long enough that a worker wakes before the caller has claimed them all.
_CHUNKS = 64
_HOLD_US = 2000


def _twin(name: str) -> types.ModuleType:
    return importlib.import_module(f"pantr.{name}")


@pytest.fixture
def capacity() -> int:
    """The pool capacity, skipping where it leaves no room for a parallel setting below it."""
    cap = ext.max_threads()
    if cap < 3:
        pytest.skip(f"needs a pool capacity of at least 3, this process has {cap}")
    return cap


def test_routed_twin_attaches_and_the_control_does_not() -> None:
    """The routed build reports the attach; the control keeps its own state."""
    assert _twin("_pantr_twin").routes_shared_parallelism()
    assert not _twin("_pantr_twin_unrouted").routes_shared_parallelism()
    assert not ext._testing.routes_shared_parallelism()


def test_twin_sees_the_owners_capacity() -> None:
    """Capacity is the owner's, and both read the same affinity, so the control agrees too."""
    assert _twin("_pantr_twin").max_threads() == ext.max_threads()


def test_setting_made_through_pantr_reaches_the_twin(capacity: int) -> None:
    """``pantr.set_num_threads`` is seen by the routed twin, never by the control."""
    twin = _twin("_pantr_twin")
    control = _twin("_pantr_twin_unrouted")
    n = 2
    pantr.set_num_threads(n)
    assert ext.get_num_threads() == n
    assert twin.get_num_threads() == n
    assert control.get_num_threads() == capacity  # its own, unset, setting


def test_setting_made_through_the_twin_reaches_pantr(capacity: int) -> None:
    """The setting is one per thread across modules, whichever module writes it."""
    _twin("_pantr_twin").set_num_threads(2)
    assert ext.get_num_threads() == 2
    assert ext._testing.count_threads_used(_CHUNKS, _HOLD_US) <= 2


def test_twin_loops_run_on_the_owners_pool(capacity: int) -> None:
    """A parallel loop in the twin uses the owner's workers, within the owner's setting."""
    twin = _twin("_pantr_twin")
    pantr.set_num_threads(3)
    assert twin.count_threads_used(_CHUNKS, _HOLD_US) == 3


# Run in a fresh process: which pools exist depends on everything the process did before.
_THREAD_COUNT_SCRIPT = """
import sys
from pathlib import Path
import pantr
from pantr._ext import ext
from pantr._numba_compat import wait_for_jit_warmup

def os_threads():
    return len(list(Path("/proc/self/task").iterdir()))

wait_for_jit_warmup()  # the import's warmup thread must not exit mid-count
twin = __import__(f"pantr.{sys.argv[1]}", fromlist=["_"])
cap = ext.max_threads()
ext.set_num_threads(cap)
ext._testing.count_threads_used(64, 2000)  # the owner's pool exists from here on
before = os_threads()
twin.set_num_threads(cap)
twin.count_threads_used(64, 2000)
print(cap, os_threads() - before)
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="counts threads in /proc")
@pytest.mark.parametrize(
    ("twin", "routed"), [("_pantr_twin", True), ("_pantr_twin_unrouted", False)]
)
def test_twin_starts_no_threads_of_its_own(capacity: int, twin: str, routed: bool) -> None:
    """After the owner's pool exists, the routed twin adds no thread; the control adds a pool."""
    result = subprocess.run(
        [sys.executable, "-c", _THREAD_COUNT_SCRIPT, twin],
        capture_output=True,
        text=True,
        check=False,
        env=subprocess_env(),
    )
    assert result.returncode == 0, result.stderr
    cap, added = (int(x) for x in result.stdout.split())
    assert added == (0 if routed else cap - 1)


# The owner's own failure, reported through the table: a malformed PANTR_NUM_THREADS. The real
# owner's status path (a value_error becomes status 1 and is raised again in the attached module)
# runs only in a fresh process, since the capacity is read once.
_OWNER_ERROR_SCRIPT = """
import pantr._pantr_twin as twin

for call in (twin.max_threads, lambda: twin.count_threads_used(8, 10), twin.get_num_threads):
    try:
        call()
    except ValueError as e:
        assert "PANTR_NUM_THREADS must be a positive integer" in str(e), e
    else:
        raise AssertionError("no ValueError")
try:
    twin.set_num_threads(0)
except ValueError as e:
    assert "n must be >= 1" in str(e), e
else:
    raise AssertionError("no ValueError")
print("ok")
"""


def test_owner_errors_reach_the_twin_as_python_errors() -> None:
    """The owner's ValueError crosses the table as a status and is raised by the twin as one."""
    env = subprocess_env()
    env["PANTR_NUM_THREADS"] = "abc"
    result = subprocess.run(
        [sys.executable, "-c", _OWNER_ERROR_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


_FORK_SCRIPT = """
import os
from pantr._ext import ext
import pantr._pantr_twin as twin

ext.set_num_threads(3)
assert twin.count_threads_used(64, 2000) == 3  # the owner's pool exists in the parent
pid = os.fork()
if pid == 0:
    # The child inherits the pool's memory but none of its threads; through the table the owner
    # builds a new pool in the child.
    ok = twin.count_threads_used(64, 2000) == 3
    os._exit(0 if ok else 1)
_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0, status
assert twin.count_threads_used(64, 2000) == 3
print("ok")
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_twin_loops_survive_a_fork(capacity: int) -> None:
    """A forked child's loop through the table runs on a pool the owner builds in the child."""
    result = subprocess.run(
        [sys.executable, "-c", _FORK_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
        env=subprocess_env(),
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
