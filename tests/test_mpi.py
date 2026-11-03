"""Tests for the optional :mod:`pantr.mpi` package skeleton and import boundary."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import pytest

import pantr
import pantr.mpi
from tests._tree_env import subprocess_env


def test_import_and_public_surface() -> None:
    """`import pantr.mpi` works regardless of mpi4py, exposing the skeleton API."""
    assert callable(pantr.mpi.mpi_available)
    assert callable(pantr.mpi.require_mpi)
    assert callable(pantr.mpi.from_dolfinx)
    assert callable(pantr.mpi.configure_threads)
    assert callable(pantr.mpi.create_distributed_space)
    assert callable(pantr.mpi.create_distributed_function)
    assert isinstance(pantr.mpi.DistributedSpace, type)
    assert isinstance(pantr.mpi.DistributedFunction, type)
    assert isinstance(pantr.mpi.HAS_MPI, bool)
    assert callable(pantr.mpi.l2_project_bspline_distributed)
    assert callable(pantr.mpi.quasi_interpolate_bspline_distributed)
    assert callable(pantr.mpi.quasi_interpolate_thb_spline_distributed)
    assert callable(pantr.mpi.interpolate_bspline_distributed)
    assert callable(pantr.mpi.fit_bspline_distributed)
    assert set(pantr.mpi.__all__) == {
        "DistributedFunction",
        "DistributedSpace",
        "HAS_MPI",
        "configure_threads",
        "create_distributed_function",
        "create_distributed_space",
        "fit_bspline_distributed",
        "from_dolfinx",
        "interpolate_bspline_distributed",
        "l2_project_bspline_distributed",
        "mpi_available",
        "quasi_interpolate_bspline_distributed",
        "quasi_interpolate_thb_spline_distributed",
        "require_mpi",
    }


def test_mpi_available_matches_find_spec() -> None:
    """`mpi_available()` agrees with a direct find_spec probe and with `HAS_MPI`."""
    expected = importlib.util.find_spec("mpi4py") is not None
    assert pantr.mpi.mpi_available() is expected
    assert pantr.mpi.HAS_MPI is expected


def test_require_mpi_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """`require_mpi()` raises an informative ImportError when mpi4py is absent."""
    monkeypatch.setattr(pantr.mpi, "mpi_available", lambda: False)
    with pytest.raises(ImportError, match=r"pantr\[mpi\]"):
        pantr.mpi.require_mpi()


def test_require_mpi_imports_mpi4py_mpi(monkeypatch: pytest.MonkeyPatch) -> None:
    """When available, `require_mpi()` imports and returns exactly `mpi4py.MPI`."""
    sentinel = object()
    imported: dict[str, str] = {}

    def fake_import(name: str) -> object:
        imported["name"] = name
        return sentinel

    monkeypatch.setattr(pantr.mpi, "mpi_available", lambda: True)
    monkeypatch.setattr(importlib, "import_module", fake_import)
    assert pantr.mpi.require_mpi() is sentinel
    assert "name" in imported, "fake_import was never called — monkeypatch target may be wrong"
    assert imported["name"] == "mpi4py.MPI"


def test_require_mpi_raises_on_broken_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """`require_mpi()` wraps non-ImportError load failures as ImportError."""

    def broken_import(name: str) -> object:
        raise OSError("libmpi.so: cannot open shared object file")

    monkeypatch.setattr(pantr.mpi, "mpi_available", lambda: True)
    monkeypatch.setattr(importlib, "import_module", broken_import)
    with pytest.raises(ImportError, match="mpi4py.MPI"):
        pantr.mpi.require_mpi()


def test_require_mpi_returns_module_when_available() -> None:
    """Integration: with a real mpi4py installed, the returned module is usable."""
    if not pantr.mpi.mpi_available():
        pytest.skip("mpi4py not installed")
    mpi = pantr.mpi.require_mpi()
    assert hasattr(mpi, "COMM_WORLD")


def test_pantr_toplevel_does_not_import_mpi() -> None:
    """``import pantr`` must not cause ``pantr.mpi`` to appear in ``sys.modules``.

    The subprocess is pointed at this checkout's source tree (``tests._tree_env``).
    ``pantr`` reaches this process through ``pytest.ini``'s ``pythonpath = src``, which a
    subprocess does not inherit; without the hand-off it would resolve whatever the
    editable install points at, and fail outright when that target no longer exists.
    """
    env = subprocess_env()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pantr, sys; assert 'pantr.mpi' not in sys.modules, list(sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def test_core_does_not_import_pantr_mpi() -> None:
    """No serial-core module directly imports pantr.mpi (mirrors the import-linter contract).

    Note: only direct (one-hop) imports are checked here. Transitive coverage is
    provided by the import-linter contract in pyproject.toml (``make import-lint``).
    """
    grimp = pytest.importorskip("grimp")
    graph = grimp.build_graph("pantr")
    mpi_modules = {m for m in graph.modules if m == "pantr.mpi" or m.startswith("pantr.mpi.")}
    assert mpi_modules, "pantr.mpi not found in the import graph"

    offenders: dict[str, list[str]] = {}
    for module in graph.modules:
        if module in mpi_modules:
            continue
        bad = sorted(graph.find_modules_directly_imported_by(module) & mpi_modules)
        if bad:
            offenders[module] = bad
    assert not offenders, f"serial-core modules importing pantr.mpi: {offenders}"


# --------------------------------------------------------------------------- #
# The pantr-mpi package: its MPI family check and its version pin
# --------------------------------------------------------------------------- #


class _FakeMpiComm:
    """Stands in for ``mpi4py.MPI.Comm``: the type an mpi4py communicator is recognised by."""

    rank = 0
    size = 1


def _fake_mpi4py(monkeypatch: pytest.MonkeyPatch, library_version: str) -> None:
    """Install a fake ``mpi4py.MPI`` whose library reports ``library_version``."""
    fake = types.ModuleType("mpi4py.MPI")
    fake.Comm = _FakeMpiComm  # type: ignore[attr-defined]
    fake.Get_library_version = lambda: library_version  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", fake)


def _fake_build_info(monkeypatch: pytest.MonkeyPatch, family: str, built: str) -> None:
    """Install a fake ``pantr_mpi._build_info`` recording ``family``."""
    info = types.ModuleType("pantr_mpi._build_info")
    info.FAMILY = family  # type: ignore[attr-defined]
    info.BUILT_AGAINST = built  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pantr_mpi._build_info", info)


def _distribute(comm: object) -> None:
    from pantr.bspline import create_uniform_space  # noqa: PLC0415

    pantr.mpi.create_distributed_space(create_uniform_space([2], [4]), comm)


@pytest.mark.parametrize(
    ("family", "built", "loaded", "expected"),
    [
        ("openmpi", "Open MPI v5.0.3", "MPICH Version: 4.2.0\nMPICH Release date", "Open MPI"),
        ("mpich", "MPICH 4.2.0", "Open MPI v5.0.3, package: Open MPI", "MPICH ABI"),
        ("mpich", "Intel(R) MPI Library 2021.18.0", "Some MPI 5 ABI library", "MPICH ABI"),
    ],
)
def test_mpi_family_mismatch_names_both(
    monkeypatch: pytest.MonkeyPatch, family: str, built: str, loaded: str, expected: str
) -> None:
    """An mpi4py communicator of another MPI family than pantr-mpi's raises ImportError."""
    _fake_mpi4py(monkeypatch, loaded)
    _fake_build_info(monkeypatch, family, built)
    with pytest.raises(ImportError, match=expected) as info:
        _distribute(_FakeMpiComm())
    assert built in str(info.value)
    assert loaded.splitlines()[0].split(",")[0] in str(info.value)


def test_mpi_family_match_then_loader_error_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    """A matching family that fails to load reports the MPI it was built for."""
    _fake_mpi4py(monkeypatch, "Intel(R) MPI Library 2021.18.0 for Linux* OS")
    _fake_build_info(monkeypatch, "mpich", "MPICH 4.2.0")
    monkeypatch.setitem(sys.modules, "pantr_mpi._pantr_mpi", None)  # import raises ImportError
    with pytest.raises(ImportError, match="failed to load.*Intel"):
        _distribute(_FakeMpiComm())


def test_custom_family_skips_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """``custom`` (a source build against any MPI) goes straight to loading the extension."""
    _fake_mpi4py(monkeypatch, "Some MPI 5 ABI library")
    _fake_build_info(monkeypatch, "custom", "unknown MPI")
    monkeypatch.setitem(sys.modules, "pantr_mpi._pantr_mpi", None)
    with pytest.raises(ImportError, match="failed to load"):
        _distribute(_FakeMpiComm())


def test_mpi4py_comm_without_pantr_mpi(monkeypatch: pytest.MonkeyPatch) -> None:
    """An mpi4py communicator without pantr-mpi installed asks for ``pantr[mpi]``."""
    _fake_mpi4py(monkeypatch, "MPICH Version: 4.2.0")
    monkeypatch.setitem(sys.modules, "pantr_mpi._build_info", None)
    with pytest.raises(ImportError, match=r"pantr-mpi.*pantr\[mpi\]"):
        _distribute(_FakeMpiComm())


def test_mpi_extra_pins_pantr_mpi_at_this_version() -> None:
    """``pantr[mpi]`` pulls pantr-mpi at pantr's own version, which bindings/mpi builds."""
    root = Path(__file__).resolve().parents[1]
    if not (root / "pyproject.toml").exists():
        pytest.skip("not run from a source tree")
    extras = tomllib.loads((root / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    assert f"pantr-mpi=={pantr.__version__}" in extras["mpi"]
    mpi_project = tomllib.loads((root / "bindings/mpi/pyproject.toml").read_text())
    version_input = mpi_project["tool"]["scikit-build"]["metadata"]["version"]["input"]
    assert (root / "bindings/mpi" / version_input).resolve() == (
        root / "src/pantr/__init__.py"
    ).resolve()
