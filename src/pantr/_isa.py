"""Choose which ISA rung of a compiled extension to import.

A multi-ISA build ships one copy of an extension per instruction-set rung: on x86-64 the base
module (the architecture baseline) and ``<base>_v3`` / ``<base>_v4`` (the x86-64-v3 and -v4
microarchitecture levels); on arm64 the base module (scalar) and ``<base>_neon``. A single-rung
build ships the base module alone. This module picks one, once per process.

The rule:

- ``PANTR_ISA`` set means exactly that rung (``generic`` names the base module). A rung that is not
  installed, or that the CPU cannot run, raises :class:`ImportError`: an explicit choice never
  falls through to another module, so a benchmark cannot silently measure the wrong one.
- Unset means the best installed rung the CPU supports, then the base module. On x86-64 the
  ``pantr._pantr_cpu`` probe decides (CPUID and the OS's vector state); NEON is the AArch64
  baseline and needs no probe.

The loaded module carries its rung as ``isa``, so the choice is observable. ``pantr-berf`` reuses
this ladder for its own modules.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import types
from typing import Final

ENV_VAR: Final[str] = "PANTR_ISA"
"""Environment variable naming the one rung to import."""

GENERIC: Final[str] = "generic"
"""Name of the base rung (the module without a suffix)."""

X86_RUNGS: Final[tuple[str, ...]] = ("v4", "v3")
"""Tuned x86-64 rungs, best first."""

ARM_RUNGS: Final[tuple[str, ...]] = ("neon",)
"""Tuned arm64 rungs, best first."""

_PROBE: Final[str] = "pantr._pantr_cpu"
"""The baseline CPU probe module."""


def machine_arch() -> str:
    """Return this machine's architecture family.

    Returns:
        str: ``"x86_64"``, ``"arm64"`` or ``"other"``.
    """
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return "other"


def cpu_supports(rung: str) -> bool:
    """Report whether this CPU can run the rung ``rung``.

    Args:
        rung (str): A rung name (``generic``, ``v3``, ``v4``, ``neon``).

    Returns:
        bool: ``True`` for ``generic``; for ``neon``, whether the machine is arm64; for ``v3`` and
        ``v4``, what the ``pantr._pantr_cpu`` probe reports on an x86-64 machine.

    Raises:
        ValueError: ``rung`` names no rung.
        ImportError: The probe is needed but cannot be imported.
    """
    if rung == GENERIC:
        return True
    if rung in ARM_RUNGS:
        return machine_arch() == "arm64"
    if rung in X86_RUNGS:
        if machine_arch() != "x86_64":
            return False
        probe = importlib.import_module(_PROBE)
        return bool(probe.supports(rung))
    raise ValueError(
        f"unknown ISA rung {rung!r}; expected one of {(GENERIC, *X86_RUNGS, *ARM_RUNGS)}"
    )


def module_name(package: str, base: str, rung: str) -> str:
    """Return the fully qualified module name of a rung.

    Args:
        package (str): Package holding the modules (``"pantr"``).
        base (str): Base module name (``"_pantr"``).
        rung (str): Rung name.

    Returns:
        str: ``f"{package}.{base}"`` for ``generic``, else ``f"{package}.{base}_{rung}"``.
    """
    return f"{package}.{base}" if rung == GENERIC else f"{package}.{base}_{rung}"


def _installed(name: str) -> bool:
    """Report whether a module can be found, without importing it."""
    return importlib.util.find_spec(name) is not None


def candidates() -> list[str]:
    """Return the rungs the default ladder tries on this machine, best first.

    Returns:
        list[str]: Tuned rungs for this architecture, then ``generic``.
    """
    arch = machine_arch()
    tuned = X86_RUNGS if arch == "x86_64" else ARM_RUNGS if arch == "arm64" else ()
    return [*tuned, GENERIC]


def load(package: str, base: str) -> types.ModuleType:
    """Import the rung of ``package.base`` this process should use.

    Args:
        package (str): Package holding the modules.
        base (str): Base module name.

    Returns:
        types.ModuleType: The imported module.

    Raises:
        ImportError: ``PANTR_ISA`` names a rung that is not installed or that this CPU cannot run,
            or no rung is installed at all.
        ValueError: ``PANTR_ISA`` names no rung.
    """
    override = os.environ.get(ENV_VAR)
    if override is not None and override.strip():
        rung = override.strip().lower()
        name = module_name(package, base, rung)
        if not cpu_supports(rung):
            raise ImportError(f"{ENV_VAR}={rung!r}, but this CPU cannot run the {name} module")
        if not _installed(name):
            raise ImportError(f"{ENV_VAR}={rung!r}, but {name} is not installed")
        return importlib.import_module(name)
    for rung in candidates():
        name = module_name(package, base, rung)
        if not _installed(name):
            continue
        if rung != GENERIC and not cpu_supports(rung):
            continue
        return importlib.import_module(name)
    raise ImportError(
        f"no {package}.{base} extension module is installed; build it with `make ext` "
        "(development) or install a pantr wheel"
    )
