# Recorded-data script, kept as it was run: outside the project style (ruff, ruff format).
# ruff: noqa
# fmt: off
"""Turn the per-case JSON files of probe_refine_case.py into cpp/tests/thb_refine_golden.hpp.

usage: gen_refine_golden.py <dir with case_*.json> <out header>
"""
import json
import sys

REF = ["ref_2", "ref_3", "ref_n", "reg_3"]
COA = ["coa_2", "coa_3", "coa_n"]
PRO = ["pro_a", "pro_b"]
RES = ["res_box", "res_deep"]
src, dst = sys.argv[1], sys.argv[2]
data = {c: json.load(open(f"{src}/case_{c}.json")) for c in REF + COA + PRO + RES}
commit = data["ref_2"]["main_commit"]
assert all(d["main_commit"] == commit for d in data.values())


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


def ints(name, vals):
    # A zero-size array is ill-formed: an empty list is a one-element array a span of size 0 views.
    return wrapped(f"inline constexpr std::int64_t {name}[]", [str(v) for v in vals] or ["0"])


def summary(name, s):
    flat = [v for cell in s["cells"] for v in cell]
    out = [
        ints(f"{name}_per_level", s["per_level"]),
        ints(f"{name}_cells", flat),
        f"inline constexpr Summary {name} = {{{s['num_levels']}, {s['num_total']}, "
        f"{len(s['cells'])}, {name}_per_level, {name}_cells}};",
    ]
    return "\n".join(out)


out = []
w = out.append
w(f"""// Recorded from the Python THBSplineSpace (src/pantr/bspline/_thb_spline_space.py, Numba build, at
// main commit {commit}) by cpp/tests/golden/probe_refine_case.py, one fresh process per
// case, and merged into this header by cpp/tests/golden/gen_refine_golden.py; do not edit by hand
// (the scripts' own headers say how to regenerate).
// All cases are 2D, degree (2, 2), on the root knots {{0,0,0,.25,.5,.75,1,1,1}} in both directions
// (4 x 4 root cells), factor (2, 2), maximal smoothness, truncated, except the one 1D case.
// A cell is `(level, index along axis 0, index along axis 1)`, listed in the grid's cell-id order.
//
// Cases:
//   ref_2, ref_3, ref_n   four successive `refine` calls of the cell (level l, index (0, 0)), for
//                         l = 0, 1, 2, 3, with admissible class 2, 3 and none
//   reg_3                 `refine_region` with class 3: (0, [1,1], [3,3]), (1, [2,2], [4,4]),
//                         (2, [4,4], [6,6])
//   coa_2, coa_3, coa_n   `coarsen` of every active level-2 cell of the ref_2 space, class 2, 3 and
//                         none (the guard vetoes every family for 2 and 3)
//   pro_a                 the prolongation of the THB space with `refine_region(0, [0,0], [2,2])`
//                         to the one that adds `refine_region(1, [0,0], [2,2])` (no class)
//   pro_b                 the 1D case: degree 3 on {{0,0,0,0,.25,.5,.75,1,1,1,1}}, regularity 0, with
//                         `refine_region(0, [0], [2])`, then `refine_region(1, [0], [2])`
//   res_box               `restrict` of the cells of the pro_a fine space whose centre lies in
//                         [.1, .6]^2
//   res_deep              `restrict` of the one deepest cell of that space
// Doubles are C++ hexfloat literals, so each recorded value is the Python double bit for bit; the
// tests compare computed values to them within a derived bound, not for equality.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

// clang-format off
namespace pantr_thb_refine_golden {{

/// A space's grid and dof counts: levels, total dofs, cells, per-level dofs, and the cells as
/// `(level, index...)` rows of `dim + 1` integers.
struct Summary {{
  std::int64_t num_levels;
  std::int64_t num_total;
  std::int64_t num_cells;
  std::span<const std::int64_t> per_level;
  std::span<const std::int64_t> cells;
}};

/// A prolongation in CSR form.
struct Csr {{
  std::int64_t rows;
  std::int64_t cols;
  std::span<const std::int64_t> indptr;
  std::span<const std::int64_t> indices;
  std::span<const double> data;
}};

/// A restriction: the cell ids asked for, the windowed space's summary, the maps, and the
/// windowed space's active functions per level.
struct Restriction {{
  std::span<const std::int64_t> ids;
  Summary space;
  std::span<const std::int64_t> dof_map;
  std::span<const std::int64_t> cell_map;
  std::span<const std::span<const std::int64_t>> active;
}};

/// A coarsening: the ids asked for and the result.
struct Coarsening {{
  std::span<const std::int64_t> ids;
  Summary result;
}};
""")
for c in REF:
    w(f"\n// ---- case {c} ----")
    w(summary(c, data[c]))
for c in COA:
    w(f"\n// ---- case {c} ----")
    d = data[c]
    w(ints(f"{c}_ids", d["ids"]))
    w(summary(f"{c}_result", d))
    w(f"inline constexpr Coarsening {c} = {{{c}_ids, {c}_result}};")
for c in PRO:
    w(f"\n// ---- case {c} ----")
    d = data[c]
    w(ints(f"{c}_indptr", d["indptr"]))
    w(ints(f"{c}_indices", d["indices"]))
    w(wrapped(f"inline constexpr double {c}_data[]", [short_hex(v) for v in d["data"]]))
    w(f"inline constexpr Csr {c} = {{{d['shape'][0]}, {d['shape'][1]}, {c}_indptr, {c}_indices, "
      f"{c}_data}};")
for c in RES:
    w(f"\n// ---- case {c} ----")
    d = data[c]
    w(ints(f"{c}_ids", d["ids"]))
    w(summary(f"{c}_space", d["space"]))
    w(ints(f"{c}_dof_map", d["dof_map"]))
    w(ints(f"{c}_cell_map", d["cell_map"]))
    names = []
    for l, act in enumerate(d["active"]):
        w(ints(f"{c}_active_{l}", act))
        names.append(f"std::span<const std::int64_t>({c}_active_{l}, {len(act)})")
    w(f"inline constexpr std::span<const std::int64_t> {c}_active[] = {{{', '.join(names)}}};")
    w(f"inline constexpr Restriction {c} = {{{c}_ids, {c}_space, {c}_dof_map, {c}_cell_map, "
      f"{c}_active}};")
w("\n}  // namespace pantr_thb_refine_golden\n// clang-format on")
open(dst, "w").write("\n".join(out) + "\n")
