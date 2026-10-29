"""Load pantr's compiled extension, and refuse one that does not match this source tree.

``ext`` is the ``_pantr`` rung this process uses (see :mod:`pantr._isa`). Two guards keep a stale
module from passing as a regression somewhere else:

- **Interface version.** ``ext.interface_version`` must equal :data:`EXPECTED_INTERFACE`. Both
  are bumped together with any change to what the module exposes, so a module built from older
  bindings fails at import.
- **Staleness against the sources.** In a development tree (the module sits in the tree's
  ``src/pantr``, where ``make ext`` installs it), :func:`stale_sources` lists the C++ sources
  modified after the module was built. The test session refuses to start while that list is not
  empty (``tests/conftest.py``). An installed wheel has no sources beside it and is not checked.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pantr import _isa

__all__ = [
    "EXPECTED_INTERFACE",
    "PROVIDER_INCLUDE_GLOB",
    "SOURCE_DIRS",
    "SOURCE_FILES",
    "check_interface",
    "ext",
    "shared_parallelism_capsule",
    "source_root",
    "stale_sources",
]

EXPECTED_INTERFACE: Final[int] = 5
"""The ``interface_version`` this package expects of the compiled module."""

SOURCE_DIRS: Final[tuple[str, ...]] = ("include", "bindings", "cmake")
"""Directories, relative to a source tree's root, whose files the module is built from (cmake/
carries the flags and the dependency pins)."""

SOURCE_FILES: Final[tuple[str, ...]] = ("CMakeLists.txt",)
"""Files, relative to a source tree's root, the module is built from. ``pyproject.toml`` is left
out: it also holds what has nothing to do with the build (dependencies, tool settings), and the one
build input in it, the nanobind pin, is checked by the build itself."""

PROVIDER_INCLUDE_GLOB: Final[str] = "providers/*/include"
"""Provider header directories, relative to a source tree's root, also built into modules."""


def check_interface(module: types.ModuleType, expected: int = EXPECTED_INTERFACE) -> None:
    """Raise unless ``module`` was built from bindings with the expected interface version.

    Args:
        module (types.ModuleType): A loaded ``_pantr`` rung.
        expected (int): The version this package expects. Defaults to
            :data:`EXPECTED_INTERFACE`.

    Raises:
        ImportError: The module has no ``interface_version`` or a different one.
    """
    found = getattr(module, "interface_version", None)
    if found != expected:
        raise ImportError(
            f"{module.__name__} ({getattr(module, '__file__', '?')}) has interface version "
            f"{found}, but this pantr expects {expected}: the compiled module is stale or from "
            "another version; rebuild it with `make ext`, or reinstall pantr"
        )


def _load() -> types.ModuleType:
    """Import the ISA rung of ``_pantr`` and check its interface version.

    Returns:
        types.ModuleType: The checked module.

    Raises:
        ImportError: No usable rung, or an interface mismatch.
    """
    module = _isa.load("pantr", "_pantr")
    check_interface(module)
    return module


if TYPE_CHECKING:
    from pantr import _pantr as ext
else:
    ext = _load()
"""The ``_pantr`` rung this process uses; typed by the committed stub ``_pantr.pyi``."""


def source_root(module_file: str | Path) -> Path | None:
    """Return the source tree a development build of the module lives in.

    Args:
        module_file (str | pathlib.Path): Path of a loaded extension module.

    Returns:
        pathlib.Path | None: The tree's root when the module sits in ``<root>/src/pantr`` beside an
        ``include/pantr`` directory (where ``make ext`` installs it), else ``None`` (an installed
        wheel).
    """
    package_dir = Path(module_file).resolve().parent
    if package_dir.name != "pantr" or package_dir.parent.name != "src":
        return None
    root = package_dir.parent.parent
    return root if (root / "include" / "pantr").is_dir() else None


def stale_sources(module_file: str | Path, root: Path) -> list[Path]:
    """List the sources under ``root`` modified after the module was built.

    Args:
        module_file (str | pathlib.Path): Path of the built module.
        root (pathlib.Path): The source tree's root.

    Returns:
        list[pathlib.Path]: Files under :data:`SOURCE_DIRS` and :data:`PROVIDER_INCLUDE_GLOB`,
        and :data:`SOURCE_FILES`, whose modification time is later than the module's, sorted.
    """
    built = Path(module_file).stat().st_mtime_ns
    dirs = [root / d for d in SOURCE_DIRS] + sorted(root.glob(PROVIDER_INCLUDE_GLOB))
    files = [path for d in dirs if d.is_dir() for path in d.rglob("*") if path.is_file()]
    files += [root / f for f in SOURCE_FILES if (root / f).is_file()]
    return sorted(path for path in files if path.stat().st_mtime_ns > built)


def shared_parallelism_capsule() -> object:
    """Return the capsule through which other pantr extension modules share ``ext``'s pool.

    Returns:
        object: A capsule named ``pantr.shared_parallelism.v1`` holding the owner's table
        (``pantr/parallel/shared.hpp``).
    """
    return ext._shared_parallelism
