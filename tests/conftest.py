"""Pytest configuration: choose which ``pantr`` the suite tests, and refuse a stale build.

By default the suite tests this tree: `src` is prepended to ``sys.path`` so the package resolves
without installation, with the extension modules ``make ext`` installed into ``src/pantr``. With
``PANTR_TEST_INSTALLED=1`` (and ``-o pythonpath=``, which empties ``pytest.ini``'s), it tests the
installed ``pantr`` instead, a wheel: `src` is not added, and the session refuses to start if
``pantr`` still resolves to this tree.

The repository root is added so test modules can import shared, non-collected helpers from the
regular ``tests`` package (e.g. ``from tests._thb_assembly import ...``); under pytest this
already resolves via ``rootdir``, so the addition is a belt-and-suspenders guard for direct
(non-pytest) invocation.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests._tree_env import TESTING_INSTALLED, TREE_SRC


def _ensure_on_sys_path() -> None:
    """Prepend the repository `src` directory (unless testing the installed package) and root."""
    repo_root: Path = Path(__file__).resolve().parents[1]
    paths = [str(repo_root)] if TESTING_INSTALLED else [str(TREE_SRC), str(repo_root)]
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)


_ensure_on_sys_path()


def pytest_sessionstart(session: pytest.Session) -> None:
    """Refuse to run against the wrong ``pantr`` or a stale extension build.

    Args:
        session (pytest.Session): The test session (unused).

    Raises:
        pytest.UsageError: In installed mode, ``pantr`` resolves to this tree; in tree mode, a
            C++ source under ``include/``, ``bindings/`` or ``providers/*/include`` is newer than
            the compiled module (rebuild with ``make ext``).
    """
    import pantr  # noqa: PLC0415
    from pantr._ext import ext, source_root, stale_sources  # noqa: PLC0415

    if TESTING_INSTALLED:
        if Path(pantr.__file__).resolve().is_relative_to(TREE_SRC):
            raise pytest.UsageError(
                f"PANTR_TEST_INSTALLED=1, but pantr resolves to this tree ({pantr.__file__}); "
                "run pytest with `-o pythonpath=` from an environment where pantr is installed"
            )
        return
    root = source_root(ext.__file__)
    if root is None:
        return
    stale = stale_sources(ext.__file__, root)
    if stale:
        listed = ", ".join(str(p.relative_to(root)) for p in stale[:5])
        more = f" and {len(stale) - 5} more" if len(stale) > 5 else ""
        raise pytest.UsageError(
            f"the compiled module {ext.__file__} is older than {listed}{more}; "
            "rebuild it with `make ext`"
        )


@pytest.fixture(autouse=True)
def _isolate_thread_state() -> Iterator[None]:
    """Keep the thread counts (Numba's and the C++ pool's) and the thread policy test-local.

    MPI-engaging entry points (:mod:`pantr.mpi`) apply a sticky process-level default
    of one Numba thread per rank; without isolation, a single such test would
    serialize every later parallel-kernel test in the session.

    Yields:
        None
    """
    import numba as nb  # noqa: PLC0415

    from pantr._ext import ext  # noqa: PLC0415

    prev = int(nb.get_num_threads())
    prev_cpp = ext.get_num_threads()
    yield
    from pantr.mpi import _thread_policy  # noqa: PLC0415

    try:
        _thread_policy._reset_policy_for_testing()
    finally:
        nb.set_num_threads(prev)
        ext.set_num_threads(prev_cpp)
