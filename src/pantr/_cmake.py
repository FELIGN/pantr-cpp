"""Where an installed pantr keeps its CMake package (:func:`pantr.cmake_dir`)."""

from __future__ import annotations

from pathlib import Path


def cmake_dir() -> str:
    """Return the directory holding the installed pantr's CMake package.

    Pass it to CMake as ``pantr_DIR``. The pantr wheel ships pantr's own headers (under
    ``pantr/include``) and this package, but not the third-party headers it bundles in a
    CMake install, so a build against it may ask only for
    ``find_package(pantr COMPONENTS mpi-capsule)``, which is what ``pantr-mpi`` builds from.

    Returns:
        str: The directory holding ``pantrConfig.cmake``.

    Raises:
        FileNotFoundError: This pantr has no CMake package, as in a source checkout's
            ``src/pantr`` (``make ext`` installs the extension modules only).

    Example:
        >>> import pantr
        >>> pantr.cmake_dir()  # doctest: +SKIP
        '.../site-packages/pantr/share/cmake/pantr'
    """
    path = Path(__file__).resolve().parent / "share" / "cmake" / "pantr"
    if not (path / "pantrConfig.cmake").is_file():
        raise FileNotFoundError(
            f"pantr has no CMake package at {path}: it ships in the pantr wheel, and a source "
            "checkout has none; build against the checkout's include/ directory, or install "
            "pantr with CMake (cmake --install)"
        )
    return str(path)
