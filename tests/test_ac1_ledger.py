"""The changed-assertion ledger tool (``tools/ac1_ledger.py``) finds what a diff changed.

Oracle: hand-written diffs and sources whose assertion lines, line numbers and enclosing tests
are known by construction.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.ac1_ledger import ASSERTION_RE, enclosing_test, ledger, parse_unified_diff, render

_DIFF = """\
diff --git a/tests/test_x.py b/tests/test_x.py
index 1111111..2222222 100644
--- a/tests/test_x.py
+++ b/tests/test_x.py
@@ -10,2 +10,2 @@ def test_a():
-    assert np.allclose(a, b, atol=1e-12)
-    y = 2
+    assert np.allclose(a, b, atol=1e-10)
+    y = 3
@@ -20,0 +21,1 @@ def test_b():
+    with pytest.raises(ValueError, match="bad"):
diff --git a/tests/test_new.py b/tests/test_new.py
new file mode 100644
--- /dev/null
+++ b/tests/test_new.py
@@ -0,0 +1,1 @@
+assert True
"""


def test_parse_unified_diff_numbers_old_and_new_lines() -> None:
    lines = parse_unified_diff(_DIFF)
    got = [(d.path, d.kind, d.lineno, d.text.strip()) for d in lines]
    assert got == [
        ("tests/test_x.py", "removed", 10, "assert np.allclose(a, b, atol=1e-12)"),
        ("tests/test_x.py", "removed", 11, "y = 2"),
        ("tests/test_x.py", "added", 10, "assert np.allclose(a, b, atol=1e-10)"),
        ("tests/test_x.py", "added", 11, "y = 3"),
        ("tests/test_x.py", "added", 21, 'with pytest.raises(ValueError, match="bad"):'),
        ("tests/test_new.py", "added", 1, "assert True"),
    ]


@pytest.mark.parametrize(
    ("text", "is_assertion"),
    [
        ("    assert x == 1", True),
        ("    with pytest.raises(ValueError):", True),
        ("    with pytest.warns(RuntimeWarning, match='cap'):", True),
        ("    np.testing.assert_allclose(a, b)", True),
        ("    assert_array_equal(a, b)", True),
        ("    check(x, atol = 1e-9)", True),
        ("    x = compute(tolerance=1e-9)", False),
        ("    reassert = 3", False),
        ("    # nothing here", False),
    ],
)
def test_assertion_pattern(text: str, is_assertion: bool) -> None:
    assert bool(ASSERTION_RE.search(text)) is is_assertion


def test_enclosing_test_names_functions_and_methods() -> None:
    source = [
        "import numpy as np",  # 1
        "",  # 2
        "def test_top():",  # 3
        "    assert 1",  # 4
        "",  # 5
        "class TestThing:",  # 6
        "    def helper(self):",  # 7
        "        pass",  # 8
        "    def test_method(self):",  # 9
        "        assert 2",  # 10
        "",  # 11
        "_CONST = 1",  # 12
    ]
    assert enclosing_test(source, 1) is None
    assert enclosing_test(source, 4) == "test_top"
    assert enclosing_test(source, 8) is None  # in a helper method, not in the test above it
    assert enclosing_test(source, 10) == "TestThing::test_method"
    assert enclosing_test(source, 12) is None


def test_ledger_on_a_real_repository(tmp_path: Path) -> None:
    """End to end through git: one tolerance changed, one assertion added, one non-assertion."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_a.py").write_text("def test_a():\n    x = 1\n    assert x == 1\n")
    git("add", ".")
    git("commit", "-q", "-m", "base")
    base = git("rev-parse", "HEAD")
    (tests / "test_a.py").write_text(
        "def test_a():\n    x = 2\n    assert x == 2\n    assert x > 0\n"
    )
    git("commit", "-q", "-am", "head")
    head = git("rev-parse", "HEAD")

    rows = ledger(base, head, ["tests"], repo=str(tmp_path))
    got = [(r.line.kind, r.line.lineno, r.line.text.strip(), r.test) for r in rows]
    assert got == [
        ("removed", 3, "assert x == 1", "test_a"),
        ("added", 3, "assert x == 2", "test_a"),
        ("added", 4, "assert x > 0", "test_a"),
    ]
    table = render(rows, "base", "head")
    assert "3 assertion lines changed" in table
    assert "| `tests/test_a.py:4` | test_a | added | `assert x > 0` | |" in table
