"""Tests for the parallel thread-count control module."""

from __future__ import annotations

import pytest

import pantr
from pantr._parallel import get_num_threads, num_threads, set_num_threads


class TestGetSetNumThreads:
    """Tests for get_num_threads / set_num_threads."""

    def test_get_returns_positive(self) -> None:
        """get_num_threads returns a positive integer."""
        n = get_num_threads()
        assert isinstance(n, int)
        assert n >= 1

    def test_set_and_get_roundtrip(self) -> None:
        """set_num_threads followed by get_num_threads returns the set value."""
        prev = get_num_threads()
        try:
            set_num_threads(1)
            assert get_num_threads() == 1
        finally:
            set_num_threads(prev)

    def test_set_zero_raises(self) -> None:
        """set_num_threads(0) raises ValueError."""
        with pytest.raises(ValueError, match="n must be >= 1"):
            set_num_threads(0)

    def test_set_negative_raises(self) -> None:
        """set_num_threads(-1) raises ValueError."""
        with pytest.raises(ValueError, match="n must be >= 1"):
            set_num_threads(-1)

    def test_set_exceeds_max_raises(self) -> None:
        """set_num_threads beyond the thread pool's capacity raises ValueError."""
        from pantr._ext import ext  # noqa: PLC0415

        with pytest.raises(ValueError, match="thread pool's capacity"):
            set_num_threads(ext.max_threads() + 1)

    def test_set_at_max_is_accepted(self) -> None:
        """The capacity itself is accepted: the bound is inclusive."""
        from pantr._ext import ext  # noqa: PLC0415

        set_num_threads(ext.max_threads())
        assert get_num_threads() == ext.max_threads()


class TestNumThreadsContextManager:
    """Tests for the num_threads context manager."""

    def test_restores_previous_value(self) -> None:
        """Thread count is restored after the context manager exits."""
        prev = get_num_threads()
        with num_threads(1):
            assert get_num_threads() == 1
        assert get_num_threads() == prev

    def test_restores_on_exception(self) -> None:
        """Thread count is restored even when an exception is raised."""
        prev = get_num_threads()
        with pytest.raises(RuntimeError), num_threads(1):
            raise RuntimeError("test error")
        assert get_num_threads() == prev  # type: ignore[unreachable, unused-ignore]

    def test_nested_context_managers(self) -> None:
        """Nested context managers restore correctly."""
        prev = get_num_threads()
        with num_threads(1):
            assert get_num_threads() == 1
            with num_threads(1):
                assert get_num_threads() == 1
            assert get_num_threads() == 1
        assert get_num_threads() == prev

    def test_limit_blas_is_gone(self) -> None:
        """``limit_blas`` was removed with threadpoolctl: the keyword is refused."""
        with pytest.raises(TypeError, match="limit_blas"), num_threads(1, limit_blas=True):  # type: ignore[call-arg]
            pass


class TestPublicReExports:
    """Tests for the public API re-exports."""

    def test_set_num_threads_reexported(self) -> None:
        """pantr.set_num_threads is the same function as _parallel.set_num_threads."""
        assert pantr.set_num_threads is set_num_threads

    def test_get_num_threads_reexported(self) -> None:
        """pantr.get_num_threads is the same function as _parallel.get_num_threads."""
        assert pantr.get_num_threads is get_num_threads

    def test_num_threads_reexported(self) -> None:
        """pantr.num_threads is the same function as _parallel.num_threads."""
        assert pantr.num_threads is num_threads


class TestDrivesTheCppPool:
    """``set_num_threads`` is the C++ pool's per-thread setting."""

    def test_setting_reaches_the_cpp_pool(self) -> None:
        """The calling thread's C++ setting follows, and bounds the threads a C++ loop uses."""
        from pantr._ext import ext  # noqa: PLC0415

        if ext.max_threads() < 3:
            pytest.skip("needs a C++ pool capacity of at least 3")
        set_num_threads(2)
        assert ext.get_num_threads() == 2
        assert ext._testing.count_threads_used(64, 2000) == 2
        set_num_threads(1)
        assert ext._testing.count_threads_used(64, 200) == 1

    def test_maximum_is_pantr_num_threads(self) -> None:
        """``PANTR_NUM_THREADS`` sets the maximum; ``NUMBA_NUM_THREADS`` is not read.

        A fresh process with ``PANTR_NUM_THREADS=2`` and ``NUMBA_NUM_THREADS=4`` (each read
        once at start): the default and the maximum are 2, and 3 is refused.
        """
        import subprocess  # noqa: PLC0415
        import sys  # noqa: PLC0415

        from pantr._ext import ext  # noqa: PLC0415
        from tests._tree_env import subprocess_env  # noqa: PLC0415

        if ext.max_threads() < 2:  # the child inherits this process's affinity
            pytest.skip("needs a C++ pool capacity of at least 2")
        env = subprocess_env()
        env["PANTR_NUM_THREADS"] = "2"
        env["NUMBA_NUM_THREADS"] = "4"
        script = (
            "import pantr; from pantr._ext import ext\n"
            "print(pantr.get_num_threads(), ext.max_threads())\n"
            "try:\n"
            "    pantr.set_num_threads(3)\n"
            "except ValueError as e:\n"
            "    print('refused', 'PANTR_NUM_THREADS' in str(e))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False, env=env
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.split() == ["2", "2", "refused", "True"]

    def test_setting_is_per_thread(self) -> None:
        """A setting made in one thread leaves another thread's at the capacity."""
        import threading  # noqa: PLC0415

        from pantr._ext import ext  # noqa: PLC0415

        set_num_threads(1)
        seen: list[int] = []
        worker = threading.Thread(target=lambda: seen.append(get_num_threads()))
        worker.start()
        worker.join()
        assert seen == [ext.max_threads()]
        assert get_num_threads() == 1

    def test_context_manager_restores_the_cpp_setting(self) -> None:
        """``num_threads`` restores the C++ setting on exit."""
        from pantr._ext import ext  # noqa: PLC0415

        if ext.max_threads() < 2:
            pytest.skip("needs a C++ pool capacity of at least 2")
        set_num_threads(1)
        with num_threads(2):
            assert ext.get_num_threads() == 2
        assert ext.get_num_threads() == 1
