"""Tests for ``tools/check_dependencies.py``, the gate that keeps numba and threadpoolctl out.

Each test builds a small source tree and asserts what the check reports on it: a gate that
cannot fail is no gate, so every rule is shown firing on a planted violation, and the clean
tree is shown passing.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType

import pytest

_TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "check_dependencies.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_dependencies", _TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check = _load()

_CLEAN_PYPROJECT = """
[build-system]
requires = ["scikit-build-core"]

[project]
name = "pantr"
dependencies = ["numpy>=1.26", "scipy>=1.15"]

[project.optional-dependencies]
mpi = ["mpi4py>=4"]
"""


def _tree(tmp_path: pathlib.Path, files: dict[str, str]) -> pathlib.Path:
    """Write a source tree: a clean pyproject and the boundary module, plus ``files``."""
    layout = {
        "pyproject.toml": _CLEAN_PYPROJECT,
        "src/pantr/__init__.py": "from . import geometry\n",
        "src/pantr/bspline/_thb_spline_space.py": "from scipy import sparse\n",
        **files,
    }
    for rel, text in layout.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def test_clean_tree_passes(tmp_path: pathlib.Path) -> None:
    assert check.violations(_tree(tmp_path, {})) == []


def test_the_repository_passes() -> None:
    assert check.violations() == []


@pytest.mark.parametrize(
    ("source", "line", "module"),
    [
        ("import numba\n", 1, "numba"),
        ("import numba.core.errors as e\n", 1, "numba"),
        ("from threadpoolctl import threadpool_limits\n", 1, "threadpoolctl"),
        ("def f():\n    import numba\n", 2, "numba"),
        ("import importlib\nimportlib.import_module('numba')\n", 2, "numba"),
        ("__import__('threadpoolctl')\n", 1, "threadpoolctl"),
    ],
)
def test_a_retired_import_is_reported(
    tmp_path: pathlib.Path, source: str, line: int, module: str
) -> None:
    found = check.violations(_tree(tmp_path, {"src/pantr/geometry.py": source}))
    assert found == [f"src/pantr/geometry.py:{line}: imports {module}"]


def test_a_relative_import_of_a_local_name_is_not_reported(tmp_path: pathlib.Path) -> None:
    tree = _tree(tmp_path, {"src/pantr/geometry.py": "from . import numba\n"})
    assert check.violations(tree) == []


def test_a_stub_is_scanned(tmp_path: pathlib.Path) -> None:
    found = check.violations(_tree(tmp_path, {"src/pantr/_pantr/x.pyi": "import numba\n"}))
    assert found == ["src/pantr/_pantr/x.pyi:1: imports numba"]


def test_scipy_outside_the_boundary_is_reported(tmp_path: pathlib.Path) -> None:
    tree = _tree(
        tmp_path,
        {"src/pantr/tolerance.py": "try:\n    import scipy\nexcept ImportError:\n    pass\n"},
    )
    assert check.violations(tree) == [
        "src/pantr/tolerance.py:2: imports scipy outside the boundary modules"
    ]


@pytest.mark.parametrize(
    "table",
    [
        'dependencies = ["numpy", "Numba>=0.60"]',
        'dependencies = ["numpy"]\n'
        "[project.optional-dependencies]\n"
        "x = [\"threadpoolctl; python_version>'3'\"]",
    ],
)
def test_a_retired_requirement_is_reported(tmp_path: pathlib.Path, table: str) -> None:
    pyproject = f'[project]\nname = "pantr"\n{table}\n'
    found = check.violations(_tree(tmp_path, {"pyproject.toml": pyproject}))
    assert len(found) == 1
    assert found[0].startswith("pyproject.toml: requires ")


def test_a_sibling_distribution_and_a_dependency_group_are_checked(tmp_path: pathlib.Path) -> None:
    files = {
        "bindings/mpi/pyproject.toml": '[project]\nname = "pantr-mpi"\ndependencies = ["numba"]\n',
        "pyproject.toml": _CLEAN_PYPROJECT + '\n[dependency-groups]\ndev = ["threadpoolctl"]\n',
    }
    found = check.violations(_tree(tmp_path, files))
    assert sorted(found) == [
        "bindings/mpi/pyproject.toml: requires 'numba'",
        "pyproject.toml: requires 'threadpoolctl'",
    ]
