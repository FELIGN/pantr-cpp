"""The test session and its child processes import this checkout's ``pantr``.

Guards against the failure where a worktree's tests, or a tool run beside them, silently exercise
another checkout's ``pantr`` through an editable install.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests._tree_env import (
    TESTING_INSTALLED,
    TREE_SRC,
    check_extension_from_tree,
    check_pantr_from_tree,
    pantr_origin,
    subprocess_env,
)

pytestmark = pytest.mark.skipif(
    TESTING_INSTALLED, reason="tests this tree's own resolution; the run tests an installed pantr"
)


def test_session_imports_this_tree() -> None:
    """The pytest process itself resolves ``pantr`` under this tree's ``src``."""
    check_pantr_from_tree()
    import pantr  # noqa: PLC0415

    assert Path(pantr.__file__).resolve().is_relative_to(TREE_SRC)
    assert pantr_origin() == Path(pantr.__file__).resolve()


def test_session_loads_this_trees_extension() -> None:
    """The compiled extension the session uses is the one ``make ext`` built into this tree."""
    assert check_extension_from_tree().is_relative_to(TREE_SRC)


def test_subprocess_env_points_a_child_at_this_tree() -> None:
    """A child given ``subprocess_env()`` imports ``pantr`` from this tree."""
    result = subprocess.run(
        [sys.executable, "-c", "import pantr; print(pantr.__file__)"],
        capture_output=True,
        text=True,
        check=False,
        env=subprocess_env(),
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve().is_relative_to(TREE_SRC)


def test_subprocess_env_keeps_the_inherited_path_after_the_tree() -> None:
    """The tree's ``src`` goes first; an inherited ``PYTHONPATH`` is kept behind it."""
    assert subprocess_env({"PYTHONPATH": "/elsewhere"})["PYTHONPATH"] == (
        f"{TREE_SRC}{os.pathsep}/elsewhere"
    )
    assert subprocess_env({})["PYTHONPATH"] == str(TREE_SRC)


def test_check_rejects_another_tree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The check fails when ``pantr`` resolves outside the tree (its reason to exist)."""
    other = tmp_path / "pantr" / "__init__.py"
    monkeypatch.setattr("tests._tree_env.pantr_origin", lambda: other)
    with pytest.raises(RuntimeError, match="outside this tree"):
        check_pantr_from_tree()
