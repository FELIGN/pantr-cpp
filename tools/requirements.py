"""Print pantr's requirements from pyproject.toml, one per line, for ``pip install -r``.

Usage::

    python tools/requirements.py [EXTRA ...] > requirements.txt

Prints the runtime dependencies, then those of each extra named, expanding ``pantr[...]``
self-references. CI installs the tools a job needs this way without building pantr itself, which
since the C++ port compiles the extension (``make ext`` builds it where a job needs it).
``build`` names the build-system requirements (scikit-build-core and the pinned nanobind).
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
SELF = re.compile(r"^pantr\[([^\]]+)\]$")


def requirements(extras: list[str]) -> list[str]:
    """Return the requirement strings for the runtime dependencies plus ``extras``.

    Args:
        extras (list[str]): Extra names, or ``build`` for the build-system requirements.

    Returns:
        list[str]: Requirements in first-seen order, without duplicates.

    Raises:
        KeyError: An extra that pyproject.toml does not define.
    """
    data = tomllib.loads(PYPROJECT.read_text())
    optional: dict[str, list[str]] = data["project"]["optional-dependencies"]
    out: list[str] = []
    seen_extras: set[str] = set()

    def add(req: str) -> None:
        match = SELF.match(req.replace(" ", ""))
        if match:
            for name in match.group(1).split(","):
                add_extra(name)
        elif req not in out:
            out.append(req)

    def add_extra(name: str) -> None:
        if name in seen_extras:
            return
        seen_extras.add(name)
        for req in optional[name]:
            add(req)

    for req in data["project"]["dependencies"]:
        add(req)
    for name in extras:
        if name == "build":
            for req in data["build-system"]["requires"]:
                add(req)
        else:
            add_extra(name)
    return out


if __name__ == "__main__":
    print("\n".join(requirements(sys.argv[1:])))
