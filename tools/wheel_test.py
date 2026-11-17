"""Check the installed pantr wheel's contents, then run the suite under each rung it tests.

Usage::

    python tools/wheel_test.py <checkout> [-n WORKERS] [-- extra pytest arguments]
    python tools/wheel_test.py <checkout> --mpi-wheel

Run it with the interpreter whose environment has pantr installed (a wheel) and the test
dependencies. It runs the suite of ``<checkout>/tests`` with ``PANTR_TEST_INSTALLED=1`` and
``-o pythonpath=``, so the checkout's ``src`` is never imported (``tests/conftest.py`` refuses to
start otherwise), first under the rung the default ladder picks, then under ``PANTR_ISA=generic``
when that is a different module. A tuned rung the CPU cannot run is not tested here: the ladder
would never load it.

Before the suite it checks what the wheel installed: exactly the extension modules of this
machine's rung ladder (``_pantr``, ``_pantr_cpu`` and one module per tuned rung of the
architecture, so a build that lost the multi-ISA ladder fails here), and in the distribution's
``.dist-info/licenses`` the ``LICENSE``, ``THIRD_PARTY_NOTICES.md`` and every file of the
checkout's ``licenses/``.

With ``--mpi-wheel`` it checks the installed pantr-mpi wheel instead and runs no suite: the package
holds only ``__init__.py``, ``_build_info.py`` and the one ``_pantr_mpi`` extension module, with no
vendored library (the MPI library is the user's, and a bundled copy would ship without its licence),
and ``.dist-info/licenses`` holds the ``LICENSE`` and every file of ``bindings/mpi/licenses/``.

The script is run as a file, not imported from the checkout, so ``import pantr`` below resolves to
the installed package.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path


def rungs_to_test() -> list[str]:
    """Return the rungs to run the suite under: the default choice, then ``generic``.

    Returns:
        list[str]: One or two rung names.
    """
    from pantr import _isa  # noqa: PLC0415
    from pantr._ext import ext  # noqa: PLC0415

    rungs = [ext.isa]
    if ext.isa != _isa.GENERIC:
        rungs.append(_isa.GENERIC)
    return rungs


def expected_modules() -> set[str]:
    """Return the extension modules the wheel must install for this machine's architecture.

    Returns:
        set[str]: Module names relative to the package, such as ``_pantr_v3``.
    """
    from pantr import _isa  # noqa: PLC0415

    arch = _isa.machine_arch()
    tuned = _isa.X86_RUNGS if arch == "x86_64" else _isa.ARM_RUNGS if arch == "arm64" else ()
    return {"_pantr", "_pantr_cpu", *(f"_pantr_{rung}" for rung in tuned)}


def installed_modules() -> set[str]:
    """Return the extension modules installed in the pantr package directory.

    Returns:
        set[str]: Module names, the part of each ``_pantr*`` file name before its first dot.
    """
    import pantr  # noqa: PLC0415

    package = Path(pantr.__file__).parent
    return {
        path.name.split(".")[0]
        for path in package.iterdir()
        if path.name.startswith("_pantr") and path.suffix in (".so", ".pyd")
    }


def installed_licences(files: list[importlib.metadata.PackagePath]) -> set[str]:
    """Return the licence files a distribution installed, relative to its ``licenses`` directory.

    Args:
        files (list[importlib.metadata.PackagePath]): The distribution's recorded files.

    Returns:
        set[str]: POSIX paths such as ``LICENSE`` or ``licenses/Eigen-MPL-2.0.txt``.
    """
    # A licence file's record path is ``<name>.dist-info/licenses/<path>``.
    return {
        Path(*f.parts[2:]).as_posix()
        for f in files
        if f.parts[:1] and f.parts[0].endswith(".dist-info") and f.parts[1:2] == ("licenses",)
    }


def content_problems(checkout: Path) -> list[str]:
    """List what the installed wheel lacks or carries beyond what it should.

    Args:
        checkout (Path): The source checkout, whose ``licenses/`` the wheel must ship.

    Returns:
        list[str]: One line per problem; empty when the contents are as expected.
    """
    problems: list[str] = []
    expected, installed = expected_modules(), installed_modules()
    if installed != expected:
        problems.append(
            f"extension modules: expected {sorted(expected)}, installed {sorted(installed)}"
        )
    files = importlib.metadata.distribution("pantr").files or []
    licences = installed_licences(files)
    wanted = {"LICENSE", "THIRD_PARTY_NOTICES.md"}
    wanted |= {f"licenses/{p.name}" for p in (checkout / "licenses").iterdir()}
    if missing := sorted(wanted - licences):
        problems.append(f"licence files missing from .dist-info/licenses: {missing}")
    return problems


def mpi_content_problems(checkout: Path) -> list[str]:
    """List what the installed pantr-mpi wheel lacks or carries beyond what it should.

    Args:
        checkout (Path): The source checkout, whose ``bindings/mpi/licenses/`` the wheel must ship.

    Returns:
        list[str]: One line per problem; empty when the contents are as expected.
    """
    problems: list[str] = []
    files = importlib.metadata.distribution("pantr-mpi").files or []
    # The package's entries: its files and any subdirectory, ``__pycache__`` aside.
    package = sorted(
        {f.parts[1] for f in files if f.parts[0] == "pantr_mpi" and f.parts[1] != "__pycache__"}
    )
    modules = [name for name in package if name.startswith("_pantr_mpi")]
    others = [name for name in package if name not in modules]
    if len(modules) != 1 or others != ["__init__.py", "_build_info.py"]:
        problems.append(
            "pantr_mpi: expected __init__.py, _build_info.py and one _pantr_mpi extension, "
            f"installed {package}"
        )
    # auditwheel vendors into ``pantr_mpi.libs``; nothing but the package and its .dist-info belongs
    # beside it.
    stray = sorted(
        {
            f.parts[0]
            for f in files
            if f.parts[0] != "pantr_mpi" and not f.parts[0].endswith(".dist-info")
        }
    )
    if stray:
        problems.append(f"top-level entries beyond the package and its .dist-info: {stray}")
    wanted = {"LICENSE"}
    wanted |= {f"licenses/{p.name}" for p in (checkout / "bindings" / "mpi" / "licenses").iterdir()}
    if missing := sorted(wanted - installed_licences(files)):
        problems.append(f"licence files missing from .dist-info/licenses: {missing}")
    return problems


def main() -> int:
    """Run the suite per rung.

    Returns:
        int: 0 when every run passed, else the first nonzero pytest exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkout", type=Path, help="the source checkout holding tests/")
    parser.add_argument("-n", "--workers", default="8", help="pytest-xdist workers (default 8)")
    parser.add_argument(
        "--mpi-wheel",
        action="store_true",
        help="check the installed pantr-mpi wheel's contents and run no suite",
    )
    parser.add_argument("pytest_args", nargs="*", help="extra pytest arguments (after --)")
    args = parser.parse_args()
    root = args.checkout.resolve()
    if args.mpi_wheel:
        mpi_problems = mpi_content_problems(root)
        for problem in mpi_problems:
            print(f"pantr-mpi wheel contents: {problem}", file=sys.stderr)
        if not mpi_problems:
            print("== pantr-mpi wheel contents: one extension, no vendored library, licences")
        return 1 if mpi_problems else 0
    problems = content_problems(root)
    for problem in problems:
        print(f"wheel contents: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"== wheel contents: modules {sorted(installed_modules())}, licence files present")
    status = 0
    for rung in rungs_to_test():
        env = dict(os.environ, PANTR_ISA=rung, PANTR_TEST_INSTALLED="1")
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "pythonpath=",
            "-p",
            "no:cacheprovider",
            "-n",
            args.workers,
            str(root / "tests"),
            *args.pytest_args,
        ]
        print(f"== PANTR_ISA={rung}: {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=root, env=env, check=False)
        if result.returncode != 0 and status == 0:
            status = result.returncode
    return status


if __name__ == "__main__":
    raise SystemExit(main())
