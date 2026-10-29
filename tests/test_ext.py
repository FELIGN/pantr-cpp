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
from tests._tree_env import subprocess_env

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


# The SIMD level each rung must be compiled for: a rung built without its ISA flag still loads and
# still reports its rung name, so the level its headers saw is what shows the flag reached it.
_SIMD_OF_RUNG = {"generic": "scalar", "v3": "avx2", "v4": "avx512", "neon": "neon"}


@pytest.mark.parametrize("rung", _installed_rungs())
def test_each_rung_is_compiled_for_its_isa(rung: str) -> None:
    """Every runnable rung's headers saw the ISA the rung is named for."""
    env = subprocess_env()
    env[_isa.ENV_VAR] = rung
    result = subprocess.run(
        [sys.executable, "-c", "from pantr._ext import ext; print(ext.simd_isa)"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == _SIMD_OF_RUNG[rung]


def test_cpu_supports_rejects_an_unknown_rung() -> None:
    """The real rung check, not a fake, refuses a misspelt rung."""
    with pytest.raises(ValueError, match="unknown ISA rung"):
        _isa.cpu_supports("avx3")


@pytest.mark.parametrize(
    ("arch", "expected"), [("x86_64", ["v4", "v3", "generic"]), ("arm64", ["neon", "generic"]),
                           ("other", ["generic"])],
)  # fmt: skip
def test_candidates_follow_the_architecture(
    monkeypatch: pytest.MonkeyPatch, arch: str, expected: list[str]
) -> None:
    """The default ladder offers each architecture's rungs, best first, then the base."""
    monkeypatch.setattr(_isa, "machine_arch", lambda: arch)
    assert _isa.candidates() == expected


@pytest.mark.parametrize(("arch", "runs"), [("arm64", True), ("x86_64", False)])
def test_neon_runs_on_arm64_only(monkeypatch: pytest.MonkeyPatch, arch: str, runs: bool) -> None:
    """NEON is the AArch64 baseline: supported there without a probe, nowhere else."""
    monkeypatch.setattr(_isa, "machine_arch", lambda: arch)
    assert _isa.cpu_supports("neon") is runs


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


# Register bits per level, from the x86-64 psABI's microarchitecture-level table, written out
# here independently of the probe: (register, bit) pairs.
_V2_BITS = [("l1", 0), ("l1", 9), ("l1", 13), ("l1", 19), ("l1", 20), ("l1", 23), ("e1", 0)]
_V3_BITS = [("l1", 28), ("l7", 5), ("l7", 3), ("l7", 8), ("l1", 29), ("l1", 12), ("e1", 5),
            ("l1", 22)]  # fmt: skip
_V4_BITS = [("l7", 16), ("l7", 30), ("l7", 28), ("l7", 17), ("l7", 31)]
_OSXSAVE = ("l1", 27)
_XCR0_FULL = 0xE7  # x87, XMM, YMM, opmask, ZMM_Hi256, Hi16_ZMM


def _registers(bits: list[tuple[str, int]]) -> dict[str, int]:
    regs = {"l1": 0, "l7": 0, "e1": 0}
    for reg, n in bits:
        regs[reg] |= 1 << n
    return regs


def _levels(bits: list[tuple[str, int]], xcr0: int) -> tuple[bool, bool, bool]:
    from pantr import _pantr_cpu  # noqa: PLC0415

    r = _registers(bits)
    return _pantr_cpu._levels_from_registers(r["l1"], r["l7"], r["e1"], xcr0)


_ALL_BITS = [*_V2_BITS, *_V3_BITS, *_V4_BITS, _OSXSAVE]


def test_levels_with_every_feature_and_state() -> None:
    """Every bit and the full vector state: all three levels."""
    assert _levels(_ALL_BITS, _XCR0_FULL) == (True, True, True)


@pytest.mark.parametrize(
    ("xcr0", "expected"),
    [(0x7, (True, True, False)), (0x3, (True, False, False)), (0x1, (True, False, False))],
)
def test_levels_fail_closed_on_os_vector_state(xcr0: int, expected: tuple[bool, ...]) -> None:
    """A CPU with every feature, under an OS that does not save ZMM or YMM state, gets no v4/v3."""
    assert _levels(_ALL_BITS, xcr0) == expected


def test_levels_without_osxsave_ignore_xcr0() -> None:
    """Without OSXSAVE, XCR0 is not consulted: no vector level, whatever the value passed."""
    bits = [b for b in _ALL_BITS if b != _OSXSAVE]
    assert _levels(bits, _XCR0_FULL) == (True, False, False)


@pytest.mark.parametrize("missing", _ALL_BITS)
def test_levels_need_every_bit(missing: tuple[str, int]) -> None:
    """Removing any one feature bit removes its level and every level above it."""
    got = _levels([b for b in _ALL_BITS if b != missing], _XCR0_FULL)
    if missing in _V2_BITS:
        assert got == (False, False, False)
    elif missing in _V3_BITS or missing == _OSXSAVE:
        assert got == (True, False, False)
    else:
        assert got == (True, True, False)


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
    (root / "cmake").mkdir()
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
    [
        "include/pantr/x.hpp",
        "bindings/src/module.cpp",
        "providers/berf/include/p.hpp",
        "cmake/PantrFlags.cmake",
        "CMakeLists.txt",
    ],
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


def test_session_refuses_a_stale_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The session hook (conftest) refuses a module older than a source, and names the rebuild.

    Run in-process against a fake tree, so the real module's timestamp is never touched.
    """
    from tests import conftest  # noqa: PLC0415

    module = _fake_tree(tmp_path)
    now = time.time()
    _set_mtime(module, now - 100)
    header = tmp_path / "include" / "pantr" / "x.hpp"
    header.write_text("// edited\n")
    _set_mtime(header, now)
    fake = types.ModuleType("pantr._pantr")
    fake.__file__ = str(module)
    monkeypatch.setattr(_ext, "ext", fake)
    monkeypatch.setattr(conftest, "TESTING_INSTALLED", False)
    with pytest.raises(pytest.UsageError, match="rebuild it with `make ext`"):
        conftest.pytest_sessionstart(None)  # type: ignore[arg-type]
    _set_mtime(module, now + 100)
    conftest.pytest_sessionstart(None)  # type: ignore[arg-type]


def test_capsule_is_the_shared_parallelism_table() -> None:
    """The capsule other modules attach to is the one ``_pantr`` exports."""
    assert _ext.shared_parallelism_capsule() is ext._shared_parallelism
    assert "pantr.shared_parallelism.v1" in repr(_ext.shared_parallelism_capsule())
