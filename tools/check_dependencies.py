"""Fail when a retired dependency returns, scipy leaves its boundary, or a mirrored file drifts.

Usage::

    python tools/check_dependencies.py

Four checks, over this source tree only (no installed package is consulted, so it runs with no
dependency installed):

- **Retired dependencies.** ``numba`` and ``threadpoolctl`` are named by no requirement in any
  ``pyproject.toml`` of the tree (the runtime dependencies, every optional extra and the build
  requirements) and imported by no Python file of the packages under ``src/``. pantr's kernels
  and thread pool are compiled C++; these two went with the Python kernels they served.
- **scipy at the boundary.** scipy is imported only by the modules in :data:`SCIPY_MODULES`,
  each of which builds a scipy object a public function returns. Everything else computes in
  the compiled extension and NumPy.
- **Mirrored build pins.** Each distribution in :data:`MIRRORED_PINS` that pantr's
  ``pyproject.toml`` pins exactly (``==``) is required with the same string by every sibling
  ``pyproject.toml``: every pantr extension module builds against one nanobind ABI and one build
  backend, and the sibling repeats the pin because its sdist builds without pantr's tree.
- **Mirrored licence files.** Each file of :data:`MIRRORED_FILES` whose original exists has a
  byte-identical copy in the sibling distribution, which cannot reach outside its own directory
  for ``license-files``.

An import is any ``import`` or absolute ``from ... import`` statement, or a call of
``importlib.import_module`` or ``__import__`` with a string literal, wherever it sits in a
``.py`` or ``.pyi`` file (function bodies and ``TYPE_CHECKING`` blocks included), found by
parsing, never by matching text. A module name computed at run time is out of its reach, and
so is any file outside ``src/``. Exits 1 and lists every violation when one is found.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Final

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
"""The source tree checked."""

RETIRED: Final[frozenset[str]] = frozenset({"numba", "threadpoolctl"})
"""Distributions and top-level modules no longer used anywhere in the packages."""

SCIPY_MODULES: Final[frozenset[str]] = frozenset({"src/pantr/bspline/_thb_spline_space.py"})
"""Files allowed to import scipy, relative to :data:`ROOT` (``prolongation_to_sparse`` returns
a ``scipy.sparse.csr_array``)."""

PYPROJECTS: Final[tuple[str, ...]] = ("pyproject.toml", "bindings/*/pyproject.toml")
"""The distributions' manifests, relative to :data:`ROOT`: pantr's and its siblings' (pantr-mpi)."""

MIRRORED_PINS: Final[tuple[str, ...]] = ("scikit-build-core", "nanobind")
"""Build requirements every sibling distribution repeats from pantr's ``pyproject.toml``."""

MIRRORED_FILES: Final[dict[str, str]] = {
    "bindings/mpi/LICENSE": "LICENSE",
    "bindings/mpi/licenses/nanobind-LICENSE": "licenses/nanobind-LICENSE",
    "bindings/mpi/licenses/robin_map-LICENSE": "licenses/robin_map-LICENSE",
}
"""Copy, relative to :data:`ROOT`, mapped to its original: the licences pantr-mpi's wheel ships
(its own, and those of the nanobind code compiled into its module)."""

_NAME: Final[re.Pattern[str]] = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
"""The distribution name at the start of a requirement string."""


def _requirements(pyproject: Path) -> Iterator[str]:
    """Yield every requirement string a ``pyproject.toml`` declares.

    Args:
        pyproject (Path): The file.

    Yields:
        str: The runtime dependencies, each optional extra's, each dependency group's and the
        build-system requirements.
    """
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    yield from project.get("dependencies", [])
    for group in project.get("optional-dependencies", {}).values():
        yield from group
    for group in data.get("dependency-groups", {}).values():
        yield from (req for req in group if isinstance(req, str))
    yield from data.get("build-system", {}).get("requires", [])


def _normalized(name: str) -> str:
    """Normalize a distribution name (PEP 503: case and runs of ``-_.`` do not matter).

    Args:
        name (str): The name.

    Returns:
        str: The normalized name.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def _imported_modules(path: Path) -> Iterator[tuple[int, str]]:
    """Yield the top-level module of every absolute import in a Python file or stub.

    Args:
        path (Path): The file.

    Yields:
        tuple[int, str]: The line number and the top-level module name.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            yield node.lineno, node.module.split(".")[0]
        elif (
            isinstance(node, ast.Call)
            and _called_name(node.func) in {"import_module", "__import__"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and not node.args[0].value.startswith(".")
        ):
            yield node.lineno, node.args[0].value.split(".")[0]


def _called_name(func: ast.expr) -> str | None:
    """Get the name a call is made through: ``f`` for ``f(...)`` and ``m.f(...)``.

    Args:
        func (ast.expr): The call's function expression.

    Returns:
        str | None: The name, or None for any other callee.
    """
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _build_pins(pyproject: Path) -> dict[str, str]:
    """Map each build requirement of a ``pyproject.toml`` from its normalized name.

    Args:
        pyproject (Path): The file.

    Returns:
        dict[str, str]: The requirement strings of ``[build-system] requires``, by name.
    """
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    pins: dict[str, str] = {}
    for req in data.get("build-system", {}).get("requires", []):
        match = _NAME.match(req)
        if match:
            pins[_normalized(match.group(1))] = req
    return pins


def _mirror_violations(root: Path) -> list[str]:
    """List the mirrored build pins and licence files that differ from their originals.

    Args:
        root (Path): The tree's root.

    Returns:
        list[str]: One line per difference, ``path: message``.
    """
    found: list[str] = []
    main = root / "pyproject.toml"
    pinned: dict[str, str] = {}
    if main.is_file():
        main_pins = _build_pins(main)
        pinned = {
            name: main_pins[name]
            for name in map(_normalized, MIRRORED_PINS)
            if "==" in main_pins.get(name, "")
        }
    for sibling in sorted(root.glob(PYPROJECTS[1])):
        pins = _build_pins(sibling)
        for name, req in pinned.items():
            if pins.get(name) != req:
                found.append(
                    f"{sibling.relative_to(root).as_posix()}: build requirement "
                    f"{pins.get(name)!r} differs from pyproject.toml's {req!r}"
                )
    for copy, original in MIRRORED_FILES.items():
        if (root / original).is_file() and (
            not (root / copy).is_file()
            or (root / copy).read_bytes() != (root / original).read_bytes()
        ):
            found.append(f"{copy}: not a byte-identical copy of {original}")
    return found


def violations(root: Path = ROOT) -> list[str]:
    """List every violation of the four checks in a source tree.

    Args:
        root (Path): The tree's root. Defaults to :data:`ROOT`.

    Returns:
        list[str]: One line per violation, ``path:line: message``; empty when the tree passes.
    """
    found: list[str] = []
    retired = {_normalized(n) for n in RETIRED}
    for pyproject in sorted(p for pattern in PYPROJECTS for p in root.glob(pattern)):
        for req in _requirements(pyproject):
            match = _NAME.match(req)
            if match and _normalized(match.group(1)) in retired:
                found.append(f"{pyproject.relative_to(root).as_posix()}: requires {req!r}")
    sources = [*(root / "src").glob("**/*.py"), *(root / "src").glob("**/*.pyi")]
    for path in sorted(sources):
        rel = path.relative_to(root).as_posix()
        for line, module in _imported_modules(path):
            if module in RETIRED:
                found.append(f"{rel}:{line}: imports {module}")
            elif module == "scipy" and rel not in SCIPY_MODULES:
                found.append(f"{rel}:{line}: imports scipy outside the boundary modules")
    found.extend(_mirror_violations(root))
    return found


def main() -> int:
    """Run the checks on this tree and report.

    Returns:
        int: 0 when the tree passes, 1 otherwise.
    """
    found = violations()
    for line in found:
        print(line, file=sys.stderr)
    if found:
        print(f"{len(found)} dependency violation(s)", file=sys.stderr)
        return 1
    print("dependencies: no retired dependency, scipy only at the boundary, mirrors in step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
