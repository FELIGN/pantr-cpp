# Recorded-data script, kept as it was run: outside the project style (ruff, ruff format).
# ruff: noqa
# fmt: off
"""Record the multi-level extraction operators of one THB case from main's Numba pantr.

usage: probe_extraction_case.py <case id> <out.json> <main checkout>   (one fresh process per case)

Regenerate cpp/tests/thb_extraction_golden.hpp (the header names the main commit it was recorded at):
  1. Check out that main commit as <main checkout>; the `pantr-cpp` conda environment imports it
     (an editable install), so run with no PYTHONPATH and from a directory that is not a pantr
     checkout. The script refuses to run if `pantr` is imported from anywhere else.
  2. For each case id gen_extraction_golden.py lists (mle_t), in its own process:
         conda run -n pantr-cpp python probe_extraction_case.py <id> <dir>/case_<id>.json <main checkout>
  3. python3 gen_extraction_golden.py <dir> cpp/tests/thb_extraction_golden.hpp
"""
import json
import os
import subprocess
import sys

import numpy as np

import pantr

MAIN = os.path.abspath(sys.argv[3])
assert pantr.__file__.startswith(os.path.join(MAIN, "src") + os.sep), pantr.__file__

from pantr.bspline import BsplineSpace, BsplineSpace1D, MultiLevelExtraction, create_thb_space

K2 = [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
# (level, index along axis 0, index along axis 1) of the cells recorded: a level-0 cell next to the
# refined region, level-1 cells at both interfaces, level-2 cells at the corner and at the interface.
CELLS = [(0, 2, 0), (1, 3, 3), (1, 2, 1), (2, 3, 3), (2, 0, 0)]


def hx(x):
    return float(x).hex()


CASE = sys.argv[1]
out = {"case": CASE}
if CASE == "mle_t":
    root = BsplineSpace([BsplineSpace1D(np.asarray(K2), 2), BsplineSpace1D(np.asarray(K2), 2)])
    thb = create_thb_space(root, truncate=True)
    thb = thb.refine_region(0, [0, 0], [2, 2], admissible_class=None)
    thb = thb.refine_region(1, [0, 0], [2, 2], admissible_class=None)
    mle = MultiLevelExtraction(thb, "bezier")
    out["num_cells"] = thb.grid.num_cells
    out["num_total"] = thb.num_total_basis
    out["cells"] = []
    for lvl, i, j in CELLS:
        cid = thb.grid.cell_id(lvl, [i, j])
        assert cid is not None
        m = mle.multilevel_operator(cid)
        c = mle.operator(cid)
        out["cells"].append({
            "level": lvl, "index": [i, j], "cid": int(cid),
            "dofs": [int(v) for v in mle.active_basis(cid)],
            "rows": int(m.shape[0]), "cols": int(m.shape[1]),
            "M": [hx(v) for v in m.ravel()],
            "C": [hx(v) for v in c.ravel()],
        })
else:
    raise SystemExit("unknown case " + CASE)

out["main_commit"] = subprocess.run(
    ["git", "-C", MAIN, "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True, check=True,
).stdout.strip()
with open(sys.argv[2], "w") as f:
    json.dump(out, f)
