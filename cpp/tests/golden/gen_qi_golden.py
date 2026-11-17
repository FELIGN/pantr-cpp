# Recorded-data script, kept as it was run: outside the project style (ruff, ruff format).
# ruff: noqa
# fmt: off
"""Turn the JSON file of probe_qi_case.py into cpp/tests/thb_qi_golden.hpp.

usage: gen_qi_golden.py <dir with case_qi_t.json> <out header>
"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(f"{src}/case_qi_t.json"))
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


out = f"""// Recorded from the Python `quasi_interpolate_thb_spline` (src/pantr/bspline/
// _thb_quasi_interpolation.py, Numba build, at main commit {commit}) by
// cpp/tests/golden/probe_qi_case.py and merged into this header by
// cpp/tests/golden/gen_qi_golden.py; do not edit by hand (the scripts' own headers say how to
// regenerate).
// One case, `qi_t`: a 2D truncated space of degree (2, 2) on the root knots
// {{0,0,0,.25,.5,.75,1,1,1}} in both directions (4 x 4 root cells), factor (2, 2), maximal
// smoothness, with `refine_region(0, [0,0], [2,2])` then `refine_region(1, [0,0], [2,2])` (no
// class). The function is the vector-valued
//   f(x, y) = (x^3 - 2 x y^2 + 0.5 y + 1 / (1 + x y), x y + y^3),
// spelled with `+ - * /` only. `points` are the `(num_total * 9, 2)` points the quasi-interpolant
// asked `f` at (function by function, row-major), `coeffs` the `(num_total, 2)` control points of
// the result.
// Doubles are C++ hexfloat literals, so each recorded value is the Python double bit for bit; the
// tests compare computed values to them within a derived bound, not for equality.
#pragma once

#include <cstddef>
#include <cstdint>

// clang-format off
namespace pantr_thb_qi_golden {{

inline constexpr std::int64_t num_total = {d['num_total']};
{wrapped("inline constexpr double points[]", [short_hex(v) for v in d["points"]])}
{wrapped("inline constexpr double coeffs[]", [short_hex(v) for v in d["coeffs"]])}

}}  // namespace pantr_thb_qi_golden
// clang-format on
"""
open(dst, "w").write(out)
