#!/usr/bin/env python
"""List the test assertions a commit range changes, as a ledger to classify.

The C++ port keeps the Python suite as its oracle: every assertion a stage changes must fall in
one of four categories (a derived tolerance, an independent oracle replacing a mirror, a replaced
Numba-specific test, an API change listed in the changelog). This tool produces the list of
changed assertions from the diff itself, so the ledger cannot omit one by oversight; the category
column is left for the author to fill.

An "assertion line" is a removed or added line, under the given paths, that contains an assertion
construct: ``assert``, ``pytest.raises``/``pytest.warns``/``pytest.approx``, a ``numpy.testing``
or ``assert_*`` helper call, or a ``match=``, ``atol=`` or ``rtol=`` argument. A changed line
shows up twice, once removed and once added. Lines are attributed to the enclosing ``test_*``
function in the old (removed) or new (added) version of the file.

Usage::

    python tools/ac1_ledger.py main..HEAD
    python tools/ac1_ledger.py 11ab436..feat/cpp-port --paths tests tests/mpi
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

ASSERTION_RE: Final[re.Pattern[str]] = re.compile(
    r"\bassert\b|\bpytest\.(?:raises|warns|approx)\b|\bassert_\w+\(|\b(?:match|atol|rtol)\s*="
)
"""Matches a line that carries an assertion construct."""

_HUNK_RE: Final[re.Pattern[str]] = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_DEF_RE: Final[re.Pattern[str]] = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(")
_CLASS_RE: Final[re.Pattern[str]] = re.compile(r"^(\s*)class\s+(\w+)")

Kind = Literal["removed", "added"]
"""Whether a line left the old version or entered the new one."""


@dataclass(frozen=True)
class DiffLine:
    """One removed or added line of a unified diff.

    Attributes:
        path (str): Repository-relative path (the new path for an added line, the old path for a
            removed one).
        kind (Kind): ``"removed"`` or ``"added"``.
        lineno (int): Line number in the old (removed) or new (added) version.
        text (str): The line, without its diff marker.
    """

    path: str
    kind: Kind
    lineno: int
    text: str


@dataclass(frozen=True)
class LedgerRow:
    """One changed assertion line, attributed to its test.

    Attributes:
        line (DiffLine): The diff line.
        test (str | None): ``Class::test_name`` or ``test_name`` of the enclosing test, or ``None``
            when the line is outside any test function (a helper or a module-level constant).
    """

    line: DiffLine
    test: str | None


def parse_unified_diff(diff: str) -> list[DiffLine]:
    """Extract every removed and added line from ``git diff -U0`` output.

    Args:
        diff (str): The diff text.

    Returns:
        list[DiffLine]: The lines in diff order.
    """
    out: list[DiffLine] = []
    old_path = new_path = ""
    old_no = new_no = 0
    for raw in diff.splitlines():
        if raw.startswith("--- "):
            old_path = raw[6:] if raw.startswith("--- a/") else ""
            continue
        if raw.startswith("+++ "):
            new_path = raw[6:] if raw.startswith("+++ b/") else ""
            continue
        hunk = _HUNK_RE.match(raw)
        if hunk:
            old_no, new_no = int(hunk.group(1)), int(hunk.group(2))
            continue
        if raw.startswith("-") and old_path:
            out.append(DiffLine(old_path, "removed", old_no, raw[1:]))
            old_no += 1
        elif raw.startswith("+") and new_path:
            out.append(DiffLine(new_path, "added", new_no, raw[1:]))
            new_no += 1
    return out


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def enclosing_test(source: Sequence[str], lineno: int) -> str | None:
    """Name the ``test_*`` function that contains line ``lineno`` (1-based) of ``source``.

    The enclosing function is the nearest ``def`` above the line that is indented less than it;
    the line belongs to a test only when that function is a ``test_*`` one.

    Args:
        source (Sequence[str]): The file's lines.
        lineno (int): The 1-based line number.

    Returns:
        str | None: ``Class::test_name`` for a method of an enclosing class, ``test_name`` for a
        module-level test, ``None`` when no test function encloses the line (a helper, a class
        body, module level).
    """
    if not 1 <= lineno <= len(source):
        return None
    target = source[lineno - 1]
    target_indent = _indent(target) if target.strip() else sys.maxsize
    for i in range(lineno - 2, -1, -1):
        line = source[i]
        if not line.strip() or _indent(line) >= target_indent:
            continue
        func = _DEF_RE.match(line)
        if func is None:
            if _CLASS_RE.match(line):
                return None  # a class body, outside any function
            continue
        name = func.group(1)
        if not name.startswith("test_"):
            return None
        def_indent = _indent(line)
        for j in range(i - 1, -1, -1):
            cls = _CLASS_RE.match(source[j])
            if cls and len(cls.group(1)) < def_indent:
                return f"{cls.group(2)}::{name}"
            if source[j].strip() and _indent(source[j]) < def_indent and not cls:
                break
        return name
    return None


def _git(*args: str, cwd: str | None) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, encoding="utf-8", cwd=cwd
    ).stdout


def ledger(base: str, head: str, paths: Sequence[str], repo: str | None = None) -> list[LedgerRow]:
    """Collect the changed assertion lines between two revisions.

    Args:
        base (str): Old revision.
        head (str): New revision.
        paths (Sequence[str]): Pathspecs to diff (directories or files).
        repo (str | None): Repository directory. Defaults to the current directory.

    Returns:
        list[LedgerRow]: One row per removed or added assertion line.
    """
    diff = _git("diff", "-U0", "--no-color", "--no-renames", base, head, "--", *paths, cwd=repo)
    cache: dict[tuple[str, str], list[str]] = {}

    def source(rev: str, path: str) -> list[str]:
        key = (rev, path)
        if key not in cache:
            cache[key] = _git("show", f"{rev}:{path}", cwd=repo).splitlines()
        return cache[key]

    rows: list[LedgerRow] = []
    for line in parse_unified_diff(diff):
        if not line.path.endswith(".py") or not ASSERTION_RE.search(line.text):
            continue
        rev = base if line.kind == "removed" else head
        rows.append(LedgerRow(line, enclosing_test(source(rev, line.path), line.lineno)))
    return rows


def render(rows: Sequence[LedgerRow], base: str, head: str) -> str:
    """Render the ledger as a Markdown table with an empty category column.

    Args:
        rows (Sequence[LedgerRow]): The ledger rows.
        base (str): Old revision, for the heading.
        head (str): New revision, for the heading.

    Returns:
        str: The Markdown text.
    """
    lines = [
        f"# Changed test assertions, {base}..{head}",
        "",
        f"{len(rows)} assertion lines changed. Category: 1 derived tolerance, 2 independent oracle "
        "replacing a mirror, 3 replaced Numba-specific test, 4 changelog-listed API change.",
        "",
        "| Location | Test | Change | Line | Category |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        text = row.line.text.strip().replace("|", "\\|")
        where = f"{row.line.path}:{row.line.lineno}"
        lines.append(
            f"| `{where}` | {row.test or '(outside a test)'} | {row.line.kind} | `{text}` | |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Print the ledger for a commit range.

    Args:
        argv (Sequence[str] | None): Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        int: The process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("range", help="commit range BASE..HEAD")
    parser.add_argument("--paths", nargs="+", default=["tests"], help="pathspecs (default: tests)")
    args = parser.parse_args(argv)
    base, sep, head = str(args.range).partition("..")
    if not sep or not base or not head:
        parser.error(f"range must be BASE..HEAD, got {args.range!r}")
    sys.stdout.write(render(ledger(base, head, args.paths), base, head))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
