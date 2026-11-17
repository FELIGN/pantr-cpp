# Recorded-data script, kept as it was run: outside the project style (ruff, ruff format).
# ruff: noqa
# fmt: off
"""Record the THB quasi-interpolation of one case from main's Numba pantr.

usage: probe_qi_case.py <case id> <out.json> <main checkout>   (one fresh process per case)

Regenerate cpp/tests/thb_qi_golden.hpp (the header names the main commit it was recorded at):
  1. Check out that main commit as <main checkout>; the `pantr-cpp` conda environment imports it
     (an editable install), so run with no PYTHONPATH and from a directory that is not a pantr
     checkout. The script refuses to run if `pantr` is imported from anywhere else.
  2. For each case id gen_qi_golden.py lists (qi_t), in its own process:
         conda run -n pantr-cpp python probe_qi_case.py <id> <dir>/case_<id>.json <main checkout>
  3. python3 gen_qi_golden.py <dir> cpp/tests/thb_qi_golden.hpp
"""
import json
import os
import subprocess
import sys

import numpy as np

import pantr

MAIN = os.path.abspath(sys.argv[3])
assert pantr.__file__.startswith(os.path.join(MAIN, "src") + os.sep), pantr.__file__

from pantr.bspline import BsplineSpace, BsplineSpace1D, create_thb_space, quasi_interpolate_thb_spline

K2 = [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0]


def hx(x):
    return float(x).hex()


def func(pts):
    # Only +, -, *, / so that the C++ test's evaluation of the same expressions agrees to rounding.
    x = pts[:, 0]
    y = pts[:, 1]
    f1 = x * x * x - 2.0 * x * y * y + 0.5 * y + 1.0 / (1.0 + x * y)
    f2 = x * y + y * y * y
    seen.append(pts.copy())
    return np.stack([f1, f2], axis=-1)


CASE = sys.argv[1]
out = {"case": CASE}
seen = []
if CASE == "qi_t":
    root = BsplineSpace([BsplineSpace1D(np.asarray(K2), 2), BsplineSpace1D(np.asarray(K2), 2)])
    thb = create_thb_space(root, truncate=True)
    thb = thb.refine_region(0, [0, 0], [2, 2], admissible_class=None)
    thb = thb.refine_region(1, [0, 0], [2, 2], admissible_class=None)
    spline = quasi_interpolate_thb_spline(func, thb)
    assert len(seen) == 1
    out["num_total"] = thb.num_total_basis
    out["points"] = [hx(v) for v in seen[0].ravel()]
    out["coeffs"] = [hx(v) for v in np.asarray(spline.control_points).ravel()]
else:
    raise SystemExit("unknown case " + CASE)

out["main_commit"] = subprocess.run(
    ["git", "-C", MAIN, "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True, check=True,
).stdout.strip()
with open(sys.argv[2], "w") as f:
    json.dump(out, f)
