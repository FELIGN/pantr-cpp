"""Run child processes against this checkout's ``pantr``, and prove that they do.

``pytest.ini``'s ``pythonpath = src`` reaches the pytest process only. A child process, and any
tool run outside pytest (``lint-imports``), resolves ``pantr`` through ``sys.path`` instead, where
an editable install of another checkout can stand first: the child then exercises the wrong tree
and passes or fails for reasons that have nothing to do with this one. This module gives such
processes an absolute ``PYTHONPATH`` pointing at this tree's ``src`` and checks where ``pantr``
actually resolves.

Run as ``python -m tests._tree_env`` (with the ``PYTHONPATH`` the caller intends to use) to check
the current interpreter's resolution and exit non-zero when it is not this tree.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final

TREE_SRC: Final[Path] = Path(__file__).resolve().parents[1] / "src"
"""Absolute path of this checkout's ``src`` directory."""

TESTING_INSTALLED: Final[bool] = os.environ.get("PANTR_TEST_INSTALLED", "") == "1"
"""Whether the suite tests the installed ``pantr`` (a wheel) rather than this tree's."""


def subprocess_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an environment whose ``PYTHONPATH`` starts with this tree's ``src``, absolute.

    Args:
        base (Mapping[str, str] | None): Environment to extend. Defaults to ``os.environ``.

    Returns:
        dict[str, str]: A copy of ``base`` with ``TREE_SRC`` prepended to ``PYTHONPATH``; when
        testing the installed package (``PANTR_TEST_INSTALLED=1``), an unchanged copy, so a child
        imports the same installed ``pantr``.
    """
    env = dict(os.environ if base is None else base)
    if TESTING_INSTALLED:
        return env
    inherited = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{TREE_SRC}{os.pathsep}{inherited}" if inherited else str(TREE_SRC)
    return env


def pantr_origin() -> Path:
    """Return the file ``import pantr`` would load in this interpreter, without importing it.

    Returns:
        pathlib.Path: The resolved ``pantr/__init__.py`` that the import system finds.

    Raises:
        ModuleNotFoundError: ``pantr`` is not importable at all.
    """
    spec = importlib.util.find_spec("pantr")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("pantr is not importable from this interpreter")
    return Path(spec.origin).resolve()


def check_pantr_from_tree() -> None:
    """Raise unless ``pantr`` resolves to this tree's ``src``.

    Raises:
        RuntimeError: ``pantr`` resolves to another location (typically an editable install of a
            different checkout).
    """
    origin = pantr_origin()
    if not origin.is_relative_to(TREE_SRC):
        raise RuntimeError(f"pantr resolves to {origin}, outside this tree's {TREE_SRC}")


def check_extension_from_tree() -> Path:
    """Import ``pantr`` and raise unless its compiled extension is this tree's build.

    Returns:
        pathlib.Path: The loaded ``_pantr`` module file.

    Raises:
        RuntimeError: The extension loaded from outside this tree's ``src``.
        ImportError: No usable extension (not built: run ``make ext``).
    """
    from pantr._ext import ext  # noqa: PLC0415

    module_file = Path(ext.__file__).resolve()
    if not module_file.is_relative_to(TREE_SRC):
        raise RuntimeError(f"pantr's extension loads from {module_file}, outside {TREE_SRC}")
    return module_file


if __name__ == "__main__":
    try:
        check_pantr_from_tree()
        extension = check_extension_from_tree()
    except (RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"pantr resolves to {pantr_origin()}, its extension to {extension}")
