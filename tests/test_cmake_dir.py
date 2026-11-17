"""``pantr.cmake_dir()`` and the headers and CMake package the pantr wheel ships."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import pantr
from tests._tree_env import TESTING_INSTALLED

_MPI_INCLUDE = re.compile(r"#[ \t]*include[ \t]*[<\"]mpi\.h[>\"]")
_HEADERS_WITH_MPI_H = {"pantr/mpi/mpi_table.hpp", "pantr/mpi/mpi_communicator.hpp"}

installed_only = pytest.mark.skipif(
    not TESTING_INSTALLED, reason="the CMake package ships in the wheel (PANTR_TEST_INSTALLED=1)"
)


@pytest.mark.skipif(TESTING_INSTALLED, reason="a source checkout's pantr has no CMake package")
def test_cmake_dir_of_a_checkout_raises() -> None:
    """``make ext`` installs no CMake package into ``src/pantr``, and ``cmake_dir`` says so."""
    with pytest.raises(FileNotFoundError, match="no CMake package"):
        pantr.cmake_dir()


@installed_only
def test_cmake_dir_holds_the_package() -> None:
    """The directory holds the config, its version file and the capsule targets, in the package."""
    path = Path(pantr.cmake_dir())
    assert path.parent.parent.parent == Path(pantr.__file__).resolve().parent
    for name in ("pantrConfig.cmake", "pantrConfigVersion.cmake", "pantrMpiCapsuleTargets.cmake"):
        assert (path / name).is_file(), name


@installed_only
def test_wheel_headers_are_pantrs_own() -> None:
    """``pantr/include`` has pantr's headers, no bundled third-party ones, and mpi.h in two only."""
    include = Path(pantr.__file__).resolve().parent / "include"
    assert (include / "pantr/mpi/comm_capsule.h").is_file()
    assert (include / "pantr/pantr.hpp").is_file()
    assert not (include / "pantr/third_party").exists()
    headers = sorted(
        p.relative_to(include).as_posix()
        for p in (include / "pantr").rglob("*")
        if p.suffix in {".h", ".hpp"}
    )
    assert headers
    with_mpi = {h for h in headers if _MPI_INCLUDE.search((include / h).read_text("utf-8"))}
    assert with_mpi == _HEADERS_WITH_MPI_H
