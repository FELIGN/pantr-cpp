"""The communicator a caller passes, as pantr's compiled distributed algorithms take it.

The algorithms of :mod:`pantr.mpi` run in the compiled extension, over one of two kinds of
communicator (design/mpi.md):

- an ``mpi4py.MPI.Comm`` goes through the ``pantr-mpi`` package, whose extension turns it
  into a table of MPI collectives in a capsule. ``pantr-mpi`` is built for one MPI family,
  so before its extension is first imported the family it was built for is compared with
  the MPI library ``mpi4py`` loaded, and a mismatch, or a library that fails to load, raises
  :class:`ImportError` naming both;
- any other object is a duck-typed communicator (its ``rank`` and ``size`` attributes,
  ``allgather`` and ``allreduce``), called back from the compiled code with the payloads
  :mod:`pantr.mpi` has always sent.

``mpi4py`` is never imported here: an object is an mpi4py communicator only if
``mpi4py.MPI`` is already loaded, which creating one requires.
"""

from __future__ import annotations

import importlib
import sys
import threading
from types import ModuleType
from typing import Any, Final

_FAMILY_NAMES: Final[dict[str, str]] = {
    "openmpi": "Open MPI",
    "mpich": "MPICH ABI (MPICH, Intel MPI, MVAPICH)",
}
"""The MPI families pantr-mpi is built for, by the name the import check prints."""

_lock: Final[threading.Lock] = threading.Lock()
_loaded: dict[str, ModuleType] = {}


def mpi_library_family(library_version: str) -> str:
    """Return the MPI family a library version string names.

    The family is read from the first line of the string, up to its first comma, as
    dolfinx's wheel check reads it: ``"openmpi"`` when it names Open MPI, ``"mpich"`` when it
    names an implementation of the MPICH ABI (MPICH, Intel MPI, MVAPICH), else ``"unknown"``.

    Args:
        library_version (str): What ``MPI_Get_library_version`` returned.

    Returns:
        str: ``"openmpi"``, ``"mpich"`` or ``"unknown"``.
    """
    lines = library_version.splitlines()
    first = " ".join(lines[0].split(",")[0].split()) if lines else ""
    if "Open MPI" in first:
        return "openmpi"
    if any(name in first for name in ("MPICH", "Intel", "MVAPICH")):
        return "mpich"
    return "unknown"


def check_mpi_family(family: str, built_against: str, library_version: str) -> None:
    """Raise unless the loaded MPI library belongs to the family pantr-mpi was built for.

    Args:
        family (str): The family recorded at build time (``"openmpi"``, ``"mpich"``, or
            ``"custom"``, which disables the check).
        built_against (str): The library version line recorded at build time.
        library_version (str): What the loaded MPI library reports.

    Raises:
        ImportError: The families differ, naming the expected and the loaded library.
    """
    if family == "custom":
        return
    loaded = mpi_library_family(library_version)
    if loaded == family:
        return
    lines = library_version.splitlines()
    loaded_text = " ".join(lines[0].split()) if lines else "no MPI library"
    expected = _FAMILY_NAMES.get(family, family)
    raise ImportError(
        f"pantr-mpi was built for {expected} ({built_against}), but mpi4py loaded "
        f"{loaded_text}. Use an MPI of that family, or build pantr-mpi from its sdist against "
        "this one, with this MPI's compiler wrappers on PATH: "
        "pip install --no-binary pantr-mpi --force-reinstall --no-deps pantr-mpi"
    )


def _load_pantr_mpi() -> ModuleType:
    """Import pantr-mpi's extension, after checking its MPI family (once per process).

    Returns:
        ModuleType: ``pantr_mpi._pantr_mpi``.

    Raises:
        ImportError: pantr-mpi is not installed, its MPI family differs from the loaded MPI
            library's, or its extension fails to load (for instance a missing ``libmpi``).
    """
    with _lock:
        module = _loaded.get("module")
        if module is not None:
            return module
        try:
            info = importlib.import_module("pantr_mpi._build_info")
        except ImportError as exc:
            raise ImportError(
                "pantr.mpi needs the 'pantr-mpi' package to use an mpi4py communicator; "
                "install it with 'pip install \"pantr[mpi]\"'."
            ) from exc
        mpi = importlib.import_module("mpi4py.MPI")
        check_mpi_family(str(info.FAMILY), str(info.BUILT_AGAINST), mpi.Get_library_version())
        try:
            module = importlib.import_module("pantr_mpi._pantr_mpi")
        except ImportError as exc:
            expected = _FAMILY_NAMES.get(str(info.FAMILY), str(info.FAMILY))
            raise ImportError(
                f"pantr-mpi's extension, built for {expected} ({info.BUILT_AGAINST}), failed "
                f"to load against the MPI library mpi4py loaded "
                f"({(mpi.Get_library_version().splitlines() or ['no MPI library'])[0]}): {exc}"
            ) from exc
        _loaded["module"] = module
        return module


def as_communicator(comm: Any) -> Any:  # noqa: ANN401 -- an mpi4py.MPI.Comm or a duck-typed one
    """Return the object the compiled distributed algorithms take for ``comm``.

    Args:
        comm (Any): An ``mpi4py.MPI.Comm``, or any object with ``rank`` and ``size``
            attributes and ``allgather`` and ``allreduce`` methods.

    Returns:
        Any: A capsule over the MPI communicator for an mpi4py communicator; ``comm`` itself
        otherwise.

    Raises:
        ImportError: See :func:`_load_pantr_mpi`.
        RuntimeError: MPI was initialized below ``MPI_THREAD_FUNNELED``, or is finalized.
    """
    mpi = sys.modules.get("mpi4py.MPI")
    comm_type = getattr(mpi, "Comm", None)
    if isinstance(comm_type, type) and isinstance(comm, comm_type):
        return _load_pantr_mpi().comm_capsule(comm)
    return comm


__all__: list[str] = []
