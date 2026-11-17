# Recorded-data script, kept as it was run: outside the project style (ruff, ruff format).
# ruff: noqa
# fmt: off
"""Turn the per-case JSON files of probe_thb_case.py into cpp/tests/thb_golden.hpp.

usage: gen_thb_golden.py <dir with case_*.json> <out header>
"""
import json
import sys

CASES = ["a2t", "a2h", "a3t", "a3h", "b2t", "b3t", "b3h", "f32", "fac21", "d3t", "unc"]
src, dst = sys.argv[1], sys.argv[2]
data = {c: json.load(open(f"{src}/case_{c}.json")) for c in CASES}
commit = data["a2t"]["main_commit"]
assert all(d["main_commit"] == commit for d in data.values())


def short_hex(h):
    """float.hex() without the trailing zeros of the mantissa: 0x1.8000000000000p-1 -> 0x1.8p-1."""
    mant, exp = h.split("p")
    if "." in mant:
        mant = mant.rstrip("0").rstrip(".")
    return mant + "p" + exp


def arr(vals, kind):
    if kind == "i":
        return ", ".join(str(v) for v in vals)
    return ", ".join(short_hex(v) for v in vals)


def wrapped(decl, items, width=100):
    """`decl = {items};` with the items filled into lines of at most `width` columns."""
    lines = []
    cur = decl + " = {"
    indent = "    "
    first = True
    for it in items:
        piece = it + ","
        if len(cur) + 1 + len(piece) > width and not first:
            lines.append(cur)
            cur = indent + piece
        else:
            cur = (cur + " " + piece) if not cur.endswith("{") else (cur + piece)
        first = False
    lines.append(cur.rstrip(",") + "};")
    return "\n".join(lines)


out = []
w = out.append
w(f"""// Recorded from the Python THBSplineSpace (src/pantr/bspline/_thb_spline_space.py, Numba build, at
// main commit {commit}) by cpp/tests/golden/probe_thb_case.py, one fresh process per
// case, and merged into this header by cpp/tests/golden/gen_thb_golden.py; do not edit by hand
// (the scripts' own headers say how to regenerate).
// Each case is a space built from a root B-spline space, a hierarchical grid refined as the case's
// comment says, and a regularity; the data are the per-level active functions, the truncated
// coefficient boxes, the level spaces' knots, the tolerance, and the values or mixed partial
// derivatives of the active functions at three points of four cells. Doubles are C++ hexfloat
// literals, so each recorded value is the Python double bit for bit; the tests compare computed
// values to it within a derived bound, not for equality.
//
// Cases (root knots K2 = {{0,0,0,.25,.5,.75,1,1,1}}, K3 = {{0,0,0,0,.25,.5,.75,1,1,1,1}}, K2N =
// {{0,0,0,.3,.55,.8,1,1,1}}; refine(level, lo, hi) as HierarchicalGrid::refined):
//   a2t, a2h  2D degree (2,2) on K2 x K2, factor (2,2), refine(0,[0,0],[2,2]); THB and HB
//   a3t, a3h  as a2 plus refine(1,[0,0],[2,2]); THB and HB
//   b2t       1D degree 3 on K3, factor 2, regularity 0, refine(0,[0],[2]); THB
//   b3t, b3h  as b2t plus refine(1,[0],[2]); THB and HB
//   f32       as a2t on K2N x K2N with float32 knots (the 1D values are computed in float64)
//   fac21     as a2t with factor (2,1): the second axis is never subdivided
//   unc       1D degree 2 on the unclamped knots -0.5, -0.25, ..., 1.5 (domain [0, 1]), factor 2,
//             refine(0,[0],[2]); THB
//   d3t       3D degree (2,2,2) on {{0,0,0,.5,1,1,1}}^3, factor 2, refine(0,[0,0,0],[1,1,1]); THB
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

// clang-format off
namespace pantr_thb_golden {{

/// A truncated function: dof, representation level, box lower corner and shape, coefficients.
struct TruncEntry {{
  std::int64_t dof;
  std::int64_t rep;
  std::int64_t lo[3];
  std::int64_t shape[3];
  const double* coeffs;
}};

/// One tabulation: the cell, the derivative orders, three points, the K dofs and the (3, K)
/// values, row-major.
struct EvalEntry {{
  std::int64_t cid;
  std::int64_t order[3];
  double pts[3][3];
  std::int64_t k;
  const std::int64_t* dofs;
  const double* values;
}};

struct Case {{
  int dim;
  bool truncate;
  std::int64_t num_total;
  std::int64_t num_cells;
  std::int64_t max_active;
  double tolerance;
  std::span<const std::int64_t> per_level;
  std::span<const std::span<const std::int64_t>> active;
  /// `level_knots[level * dim + axis]`.
  std::span<const std::span<const double>> level_knots;
  std::span<const TruncEntry> trunc;
  std::span<const EvalEntry> evals;
}};
""")

for c in CASES:
    d = data[c]
    dim = d["dim"]
    w(f"\n// ---- case {c}: {d['num_total']} functions, {len(d['trunc'])} truncated ----")
    w(wrapped(f"inline constexpr std::int64_t {c}_per_level[]", [str(v) for v in d["per_level"]]))
    for l, a in enumerate(d["active"]):
        w(wrapped(f"inline constexpr std::int64_t {c}_active_{l}[]", [str(v) for v in a]))
    w(f"inline constexpr std::span<const std::int64_t> {c}_active[] = {{"
      + ", ".join(f"{c}_active_{l}" for l in range(d["num_levels"])) + "};")
    for l in range(d["num_levels"]):
        for k in range(dim):
            w(wrapped(f"inline constexpr double {c}_knots_{l}_{k}[]", [short_hex(v) for v in d["level_knots"][l][k]]))
    w(f"inline constexpr std::span<const double> {c}_knots[] = {{"
      + ", ".join(f"{c}_knots_{l}_{k}" for l in range(d["num_levels"]) for k in range(dim)) + "};")
    for i, t in enumerate(d["trunc"]):
        w(wrapped(f"inline constexpr double {c}_coeffs_{i}[]", [short_hex(v) for v in t["coeffs"]]))
    if d["trunc"]:
        w(f"inline constexpr TruncEntry {c}_trunc[] = {{")
        for i, t in enumerate(d["trunc"]):
            lo = t["lo"] + [0] * (3 - dim)
            sh = t["shape"] + [0] * (3 - dim)
            w(f"    {{{t['dof']}, {t['rep']}, {{{arr(lo, 'i')}}}, {{{arr(sh, 'i')}}}, {c}_coeffs_{i}}},")
        w("};")
        trunc_ref = f"{c}_trunc"
    else:
        trunc_ref = "{}"
    for i, e in enumerate(d["evals"]):
        w(wrapped(f"inline constexpr std::int64_t {c}_dofs_{i}[]", [str(v) for v in e["dofs"]]))
        flat = [v for row in e["values"] for v in row]
        w(wrapped(f"inline constexpr double {c}_values_{i}[]", [short_hex(v) for v in flat]))
    w(f"inline constexpr EvalEntry {c}_evals[] = {{")
    for i, e in enumerate(d["evals"]):
        order = e["order"] + [0] * (3 - dim)
        pts = []
        for p in e["pts"]:
            pts.append("{" + arr(p + ["0x0p+0"] * (3 - dim), "f") + "}")
        w(f"    {{{e['cid']}, {{{arr(order, 'i')}}}, {{{', '.join(pts)}}}, {len(e['dofs'])}, "
          f"{c}_dofs_{i}, {c}_values_{i}}},")
    w("};")
    w(f"inline constexpr Case {c} = {{{dim}, {'true' if d['truncate'] else 'false'}, "
      f"{d['num_total']}, {d['num_cells']}, {d['max_active']}, {short_hex(d['tolerance'])}, {c}_per_level, "
      f"{c}_active, {c}_knots, {trunc_ref}, {c}_evals}};")

w("\n}  // namespace pantr_thb_golden\n// clang-format on\n")
open(dst, "w").write("\n".join(out))
