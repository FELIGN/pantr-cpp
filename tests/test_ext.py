"""The compiled extension's loading: the ISA ladder, the CPU probe and the two stale-module guards.

Oracles: the ladder's rule (design section 1.8) checked against a fake set of installed modules;
the CPU probe against the kernel's own feature list (``/proc/cpuinfo``), an independent reading
of the same CPUID and XCR0 state; the guards against modules and files built to fail them.
"""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from pantr import _ext, _isa
from pantr._ext import EXPECTED_INTERFACE, check_interface, ext, source_root, stale_sources
from tests._tree_env import TESTING_INSTALLED, subprocess_env

# --- The ladder ----------------------------------------------------------------------------------


@pytest.fixture
def fake_modules(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Replace module discovery and import with a fake set, and the CPU with a v3 x86-64 one.

    Returns:
        dict[str, object]: ``installed`` (set of module names, edit it) and ``imported`` (list of
        names imported, in order).
    """
    state: dict[str, object] = {"installed": set(), "imported": []}

    def installed(name: str) -> bool:
        return name in state["installed"]  # type: ignore[operator]

    def import_module(name: str) -> types.ModuleType:
        state["imported"].append(name)  # type: ignore[attr-defined]
        return types.ModuleType(name)

    def cpu_supports(rung: str) -> bool:
        if rung not in (_isa.GENERIC, *_isa.X86_RUNGS, *_isa.ARM_RUNGS):
            raise ValueError(f"unknown ISA rung {rung!r}")
        return rung in (_isa.GENERIC, "v3")

    monkeypatch.setattr(_isa, "_installed", installed)
    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(_isa, "cpu_supports", cpu_supports)
    monkeypatch.setattr(_isa, "machine_arch", lambda: "x86_64")
    monkeypatch.delenv(_isa.ENV_VAR, raising=False)
    return state


def _install(state: dict[str, object], *rungs: str) -> None:
    state["installed"] = {_isa.module_name("pkg", "_m", r) for r in rungs}


def test_default_picks_the_best_supported_rung(fake_modules: dict[str, object]) -> None:
    """v4 is installed but the CPU stops at v3: v3 loads."""
    _install(fake_modules, "generic", "v3", "v4")
    assert _isa.load("pkg", "_m").__name__ == "pkg._m_v3"
    assert fake_modules["imported"] == ["pkg._m_v3"]


def test_default_skips_rungs_not_installed(fake_modules: dict[str, object]) -> None:
    """A single-rung build loads its base module."""
    _install(fake_modules, "generic")
    assert _isa.load("pkg", "_m").__name__ == "pkg._m"


def test_default_raises_when_nothing_is_installed(fake_modules: dict[str, object]) -> None:
    """No module at all is an ImportError naming the build step."""
    _install(fake_modules)
    with pytest.raises(ImportError, match="make ext"):
        _isa.load("pkg", "_m")


@pytest.mark.parametrize("rung", ["generic", "v3"])
def test_override_loads_exactly_that_rung(
    fake_modules: dict[str, object], monkeypatch: pytest.MonkeyPatch, rung: str
) -> None:
    """PANTR_ISA names the module, even when a better rung would be chosen by default."""
    _install(fake_modules, "generic", "v3")
    monkeypatch.setenv(_isa.ENV_VAR, rung)
    assert _isa.load("pkg", "_m").__name__ == _isa.module_name("pkg", "_m", rung)


def test_override_of_a_missing_rung_raises(
    fake_modules: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rung the CPU runs but the build lacks raises; nothing else is imported."""
    _install(fake_modules, "generic")
    monkeypatch.setenv(_isa.ENV_VAR, "v3")
    with pytest.raises(ImportError, match="not installed"):
        _isa.load("pkg", "_m")
    assert fake_modules["imported"] == []


def test_override_of_an_unsupported_rung_raises(
    fake_modules: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rung the CPU cannot run raises instead of faulting at the first instruction."""
    _install(fake_modules, "generic", "v3", "v4")
    monkeypatch.setenv(_isa.ENV_VAR, "v4")
    with pytest.raises(ImportError, match="cannot run"):
        _isa.load("pkg", "_m")
    assert fake_modules["imported"] == []


def test_override_of_an_unknown_rung_raises(
    fake_modules: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A misspelt rung is an error, not a fallback."""
    _install(fake_modules, "generic")
    monkeypatch.setenv(_isa.ENV_VAR, "avx3")
    with pytest.raises(ValueError, match="unknown ISA rung"):
        _isa.load("pkg", "_m")


def test_loaded_module_reports_its_rung() -> None:
    """The module the session uses names its rung, so the choice is observable."""
    assert ext.isa in (_isa.GENERIC, *_isa.X86_RUNGS, *_isa.ARM_RUNGS)
    assert ext.__name__ == _isa.module_name("pantr", "_pantr", ext.isa)


def _installed_rungs() -> list[str]:
    return [
        r
        for r in _isa.candidates()
        if _isa._installed(_isa.module_name("pantr", "_pantr", r)) and _isa.cpu_supports(r)
    ]


@pytest.mark.parametrize("rung", _installed_rungs())
def test_each_runnable_rung_loads_under_the_override(rung: str) -> None:
    """Every installed rung this CPU runs loads under ``PANTR_ISA`` and reports itself.

    One process per rung: two rungs in one process would register their classes twice.
    """
    env = subprocess_env()
    env[_isa.ENV_VAR] = rung
    result = subprocess.run(
        [sys.executable, "-c", "from pantr._ext import ext; print(ext.__name__, ext.isa)"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [_isa.module_name("pantr", "_pantr", rung), rung]


# --- The CPU probe ------------------------------------------------------------------------------

# x86-64 microarchitecture levels (x86-64 psABI) in the kernel's flag names. The kernel lists a
# vector feature only when the OS saves its register state, so this is the same question the probe
# answers through CPUID and XGETBV, read independently.
_V2_FLAGS = {"cx16", "lahf_lm", "popcnt", "pni", "sse4_1", "sse4_2", "ssse3"}
_V3_FLAGS = _V2_FLAGS | {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "abm", "movbe"}
_V4_FLAGS = _V3_FLAGS | {"avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"}


def _cpuinfo_flags() -> set[str]:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("flags"):
            return set(line.split(":", 1)[1].split())
    return set()


@pytest.mark.skipif(
    not (sys.platform.startswith("linux") and _isa.machine_arch() == "x86_64"),
    reason="the oracle is Linux's /proc/cpuinfo on x86-64",
)
@pytest.mark.parametrize(
    ("level", "flags"), [("v2", _V2_FLAGS), ("v3", _V3_FLAGS), ("v4", _V4_FLAGS)]
)
def test_probe_agrees_with_the_kernel(level: str, flags: set[str]) -> None:
    """The probe reports a level exactly when the kernel lists every feature of it."""
    from pantr import _pantr_cpu  # noqa: PLC0415

    assert _pantr_cpu.arch == "x86_64"
    assert _pantr_cpu.supports(level) == flags.issubset(_cpuinfo_flags())


def test_probe_rejects_an_unknown_level() -> None:
    """A misspelt level raises rather than reading as unsupported."""
    from pantr import _pantr_cpu  # noqa: PLC0415

    with pytest.raises(ValueError, match="unknown x86-64 level"):
        _pantr_cpu.supports("v5")


def test_probe_never_offers_x86_levels_elsewhere() -> None:
    """Off x86-64, no x86 level is supported (the probe fails closed)."""
    from pantr import _pantr_cpu  # noqa: PLC0415

    if platform.machine().lower() in ("x86_64", "amd64"):
        pytest.skip("this machine is x86-64")
    assert not any(_pantr_cpu.supports(level) for level in ("v2", "v3", "v4"))


# --- The interface-version guard -----------------------------------------------------------------


def test_session_module_has_the_expected_interface() -> None:
    """The loaded module passed the guard."""
    assert ext.interface_version == EXPECTED_INTERFACE
    check_interface(ext)


def test_interface_guard_rejects_another_version() -> None:
    """The real module, checked against a different expected version, is refused."""
    with pytest.raises(ImportError, match="interface version .* expects .*make ext"):
        check_interface(ext, EXPECTED_INTERFACE + 1)


def test_interface_guard_rejects_a_module_without_one() -> None:
    """A module that is not a pantr build at all is refused."""
    with pytest.raises(ImportError, match="interface version None"):
        check_interface(types.ModuleType("pantr._pantr"))


# --- The staleness guard -------------------------------------------------------------------------


def _fake_tree(root: Path) -> Path:
    """Build a source tree with a module in src/pantr and one header; return the module."""
    (root / "include" / "pantr").mkdir(parents=True)
    (root / "bindings" / "src").mkdir(parents=True)
    (root / "providers" / "berf" / "include").mkdir(parents=True)
    (root / "src" / "pantr").mkdir(parents=True)
    module = root / "src" / "pantr" / "_pantr.abi3.so"
    module.write_bytes(b"")
    return module


def _set_mtime(path: Path, when: float) -> None:
    os.utime(path, (when, when))


def test_source_root_finds_a_development_build(tmp_path: Path) -> None:
    """A module in <root>/src/pantr beside include/pantr belongs to the tree at <root>."""
    module = _fake_tree(tmp_path)
    assert source_root(module) == tmp_path.resolve()


def test_source_root_ignores_an_installed_module(tmp_path: Path) -> None:
    """A module in site-packages/pantr has no tree to be stale against."""
    module = tmp_path / "site-packages" / "pantr" / "_pantr.abi3.so"
    module.parent.mkdir(parents=True)
    module.write_bytes(b"")
    assert source_root(module) is None


@pytest.mark.parametrize(
    "relative",
    ["include/pantr/x.hpp", "bindings/src/module.cpp", "providers/berf/include/p.hpp"],
)
def test_a_source_newer_than_the_module_is_stale(tmp_path: Path, relative: str) -> None:
    """Each source directory the module is built from is watched."""
    module = _fake_tree(tmp_path)
    now = time.time()
    _set_mtime(module, now - 100)
    source = tmp_path / relative
    source.write_text("// edited\n")
    _set_mtime(source, now)
    assert stale_sources(module, tmp_path) == [source]


def test_sources_older_than_the_module_are_not_stale(tmp_path: Path) -> None:
    """A build after the last edit passes; files outside the source directories never count."""
    module = _fake_tree(tmp_path)
    now = time.time()
    header = tmp_path / "include" / "pantr" / "x.hpp"
    header.write_text("// old\n")
    _set_mtime(header, now - 100)
    _set_mtime(module, now - 50)
    unrelated = tmp_path / "src" / "pantr" / "later.py"
    unrelated.write_text("")
    _set_mtime(unrelated, now)
    assert stale_sources(module, tmp_path) == []


@pytest.mark.skipif(TESTING_INSTALLED, reason="the session check guards a development build")
def test_session_refuses_a_stale_module(tmp_path: Path) -> None:
    """A test session whose module is older than a source refuses to start (conftest)."""
    module = Path(ext.__file__)
    root = source_root(module)
    assert root is not None
    # Age the module rather than touch a source: the tree's sources stay as they are.
    original = module.stat().st_mtime
    try:
        _set_mtime(module, 0.0)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                str(root / "tests" / "test_metadata.py"),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=root,
            env=subprocess_env(),
        )
    finally:
        _set_mtime(module, original)
    assert result.returncode != 0
    assert "rebuild it with `make ext`" in result.stdout + result.stderr


def test_capsule_is_the_shared_parallelism_table() -> None:
    """The capsule other modules attach to is the one ``_pantr`` exports."""
    assert _ext.shared_parallelism_capsule() is ext._shared_parallelism
    assert "pantr.shared_parallelism.v1" in repr(_ext.shared_parallelism_capsule())
