"""Smoke tests for package metadata.

Validates public attributes exposed via the package API.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Final

import pytest

import pantr


def test_package_all_exports() -> None:
    """Ensure all expected symbols are exported."""
    expected_all: Final[set[str]] = {
        "__version__",
        "__license__",
        "__author__",
        "basis",
        "bezier",
        "bspline",
        "cad",
        "change_basis",
        "cmake_dir",
        "geometry",
        "get_num_threads",
        "grid",
        "num_threads",
        "quad",
        "set_num_threads",
        "tolerance",
        "transform",
    }
    assert set(pantr.__all__) == expected_all


def test_package_metadata_values() -> None:
    """Validate the package metadata constants."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", pantr.__version__)
    assert pantr.__license__ == "MIT"
    assert pantr.__author__ == "Pablo Antolin <pablo.antolin@epfl.ch>"


def test_metadata_import_stability() -> None:
    """Verify metadata survives module reloads."""
    version = pantr.__version__
    module = importlib.reload(pantr)
    assert module.__version__ == version


def test_citation_file_states_the_package_version() -> None:
    """``CITATION.cff`` names the version in ``pantr.__version__``, so a bump cannot miss it."""
    path = Path(__file__).resolve().parents[1] / "CITATION.cff"
    if not path.exists():
        pytest.skip("not run from a source tree")
    versions = re.findall(r"^version: (\S+)\s*$", path.read_text(), flags=re.MULTILINE)
    assert versions == [pantr.__version__]


def test_submodule_all_exports() -> None:
    """Ensure each submodule exports the expected symbols."""
    from pantr import basis, bezier, bspline, change_basis, quad, tolerance  # noqa: PLC0415

    assert "LagrangeVariant" in basis.__all__
    assert "tabulate_bernstein_1d" in basis.__all__

    assert "Bezier" in bezier.__all__

    assert "Bspline" in bspline.__all__
    assert "BsplineSpace1D" in bspline.__all__
    assert "BsplineSpace" in bspline.__all__
    assert "create_uniform_open_knots" in bspline.__all__

    assert "compute_bernstein_to_lagrange_1d" in change_basis.__all__

    assert "PointsLattice" in quad.__all__
    assert "get_gauss_legendre_1d" in quad.__all__

    assert "ToleranceInfo" in tolerance.__all__
    assert "get_default" in tolerance.__all__
