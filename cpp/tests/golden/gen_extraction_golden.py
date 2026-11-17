# Recorded-data script, kept as it was run: outside the project style (ruff, ruff format).
# ruff: noqa
# fmt: off
"""Turn the JSON file of probe_extraction_case.py into cpp/tests/thb_extraction_golden.hpp.

usage: gen_extraction_golden.py <dir with case_mle_t.json> <out header>
"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(f"{src}/case_mle_t.json"))
commit = d["main_commit"]


def short_hex(h):
    mant, exp = h.split("p")
    if "." in mant:
        mant = mant.rstrip("0").rstrip(".")
    return mant + "p" + exp


def wrapped(decl, items, width=100):
    lines = []
    cur = decl + " = {"
    first = True
    for it in items:
        piece = it + ","
        if len(cur) + 1 + len(piece) > width and not first:
            lines.append(cur)
            cur = "    " + piece
        else:
            cur = (cur + " " + piece) if not cur.endswith("{") else (cur + piece)
        first = False
    lines.append(cur.rstrip(",") + "};")
    return "\n".join(lines)


out = []
w = out.append
w(f"""// Recorded from the Python MultiLevelExtraction (src/pantr/bspline/multilevel_extraction.py, Numba
// build, at main commit {commit}) by cpp/tests/golden/probe_extraction_case.py and merged into this
// header by cpp/tests/golden/gen_extraction_golden.py; do not edit by hand (the scripts' own
// headers say how to regenerate).
// One case, `mle_t`: a 2D truncated space of degree (2, 2) on the root knots
// {{0,0,0,.25,.5,.75,1,1,1}} in both directions (4 x 4 root cells), factor (2, 2), maximal
// smoothness, with `refine_region(0, [0,0], [2,2])` then `refine_region(1, [0,0], [2,2])` (no
// class), and the target "bezier". Five cells are recorded, `(level, index along axis 0, index
// along axis 1)` as in `cells` below.
// For each: the cell id, the active dofs (the rows), the multi-level operator M and the multi-level
// Bezier extraction C = M E, row-major `(rows, 9)`.
// Doubles are C++ hexfloat literals, so each recorded value is the Python double bit for bit; the
// tests compare computed values to them within a derived bound, not for equality.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

// clang-format off
namespace pantr_thb_extraction_golden {{

/// One recorded cell.
struct Cell {{
  std::int64_t level;
  std::int64_t i;
  std::int64_t j;
  std::int64_t cid;
  std::int64_t rows;
  std::int64_t cols;
  std::span<const std::int64_t> dofs;
  std::span<const double> m;
  std::span<const double> c;
}};

inline constexpr std::int64_t num_cells = {d['num_cells']};
inline constexpr std::int64_t num_total = {d['num_total']};
""")
names = []
for n, cell in enumerate(d["cells"]):
    w(wrapped(f"inline constexpr std::int64_t cell{n}_dofs[]", [str(v) for v in cell["dofs"]]))
    w(wrapped(f"inline constexpr double cell{n}_m[]", [short_hex(v) for v in cell["M"]]))
    w(wrapped(f"inline constexpr double cell{n}_c[]", [short_hex(v) for v in cell["C"]]))
    names.append(f"Cell{{{cell['level']}, {cell['index'][0]}, {cell['index'][1]}, {cell['cid']}, "
                 f"{cell['rows']}, {cell['cols']}, cell{n}_dofs, cell{n}_m, cell{n}_c}}")
w("inline constexpr Cell cells[] = {\n    " + ",\n    ".join(names) + "};")
w("\n}  // namespace pantr_thb_extraction_golden\n// clang-format on")
open(dst, "w").write("\n".join(out) + "\n")
