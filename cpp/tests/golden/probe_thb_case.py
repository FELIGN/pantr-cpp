# Recorded-data script, kept as it was run: outside the project style (ruff, ruff format).
# ruff: noqa
# fmt: off
"""Record one THB case from main's Numba pantr (one fresh process per case).

usage: probe_thb_case.py <case id> <out.json> <main checkout>

Regenerate cpp/tests/thb_golden.hpp (the header names the main commit it was recorded at):
  1. Check out that main commit as <main checkout>; the `pantr-cpp` conda environment imports it
     (an editable install), so run with no PYTHONPATH and from a directory that is not a pantr
     checkout. The script refuses to run if `pantr` is imported from anywhere else.
  2. For each case id of CASES below that gen_thb_golden.py lists (a2t a2h a3t a3h b2t b3t b3h f32
     fac21 d3t unc), in its own process:
         conda run -n pantr-cpp python probe_thb_case.py <id> <dir>/case_<id>.json <main checkout>
  3. python3 gen_thb_golden.py <dir> cpp/tests/thb_golden.hpp
"""
import json
import os
import subprocess
import sys

import numpy as np

import pantr

MAIN = os.path.abspath(sys.argv[3])
assert pantr.__file__.startswith(os.path.join(MAIN, "src") + os.sep), pantr.__file__

from pantr.bspline import BsplineSpace, BsplineSpace1D, THBSplineSpace
from pantr.grid import hierarchical_grid, tensor_product_grid

K2 = [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
K3 = [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]
K2_3D = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
KUN = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
K2N = [0.0, 0.0, 0.0, 0.3, 0.55, 0.8, 1.0, 1.0, 1.0]

# id: (knots per direction, degrees, dtype, factor, regularity, refinements, truncate)
# refinements: list of (level, lo, hi)
CASES = {
    "a2t": ([K2, K2], [2, 2], np.float64, [2, 2], None, [(0, [0, 0], [2, 2])], True),
    "a2h": ([K2, K2], [2, 2], np.float64, [2, 2], None, [(0, [0, 0], [2, 2])], False),
    "a3t": ([K2, K2], [2, 2], np.float64, [2, 2], None,
            [(0, [0, 0], [2, 2]), (1, [0, 0], [2, 2])], True),
    "a3h": ([K2, K2], [2, 2], np.float64, [2, 2], None,
            [(0, [0, 0], [2, 2]), (1, [0, 0], [2, 2])], False),
    "b2t": ([K3], [3], np.float64, [2], [0], [(0, [0], [2])], True),
    "b3t": ([K3], [3], np.float64, [2], [0], [(0, [0], [2]), (1, [0], [2])], True),
    "b3h": ([K3], [3], np.float64, [2], [0], [(0, [0], [2]), (1, [0], [2])], False),
    "f32": ([K2N, K2N], [2, 2], np.float32, [2, 2], None, [(0, [0, 0], [2, 2])], True),
    "f64n": ([K2N, K2N], [2, 2], np.float64, [2, 2], None, [(0, [0, 0], [2, 2])], True),
    "unc": ([KUN], [2], np.float64, [2], None, [(0, [0], [2])], True),
    "fac21": ([K2, K2], [2, 2], np.float64, [2, 1], None, [(0, [0, 0], [2, 2])], True),
    "d3t": ([K2_3D] * 3, [2, 2, 2], np.float64, [2, 2, 2], None, [(0, [0, 0, 0], [1, 1, 1])], True),
}

case = sys.argv[1]
knots, degrees, dtype, factor, reg, refs, truncate = CASES[case]
root = BsplineSpace(
    [BsplineSpace1D(np.asarray(k, dtype=dtype), d) for k, d in zip(knots, degrees)]
)
grid = hierarchical_grid(tensor_product_grid(root), factor)
for level, lo, hi in refs:
    grid.refine(level, lo, hi)
thb = THBSplineSpace(root, grid, truncate=truncate, regularity=reg)
dim = thb.dim


def hx(x):
    return float(x).hex()


out = {
    "case": case,
    "dim": dim,
    "truncate": truncate,
    "num_levels": thb.num_levels,
    "num_total": thb.num_total_basis,
    "num_cells": grid.num_cells,
    "per_level": list(thb.num_basis_per_level),
    "active": [thb.active_function_indices(l).tolist() for l in range(thb.num_levels)],
    "tolerance": hx(thb.tolerance),
    "level_knots": [
        [[hx(v) for v in thb.level_space(l).spaces[k].knots] for k in range(dim)]
        for l in range(thb.num_levels)
    ],
    "max_active": thb.max_active_per_cell(),
    "trunc": [],
    "evals": [],
    "active_basis": {},
}
for dof, e in sorted(thb._trunc.items()):
    out["trunc"].append(
        {
            "dof": int(dof),
            "rep": int(e.rep_level),
            "lo": [int(v) for v in e.box_lo],
            "shape": [int(v) for v in e.coeffs.shape],
            "coeffs": [hx(v) for v in e.coeffs.ravel()],
        }
    )

n = grid.num_cells
cids = sorted(set(int(v) for v in np.linspace(0, n - 1, 4).astype(int)))
u_rel = [np.zeros(dim), np.array([0.37, 0.81, 0.55][:dim]), np.ones(dim)]
order_list = [[0] * dim]
for k in range(dim):
    o = [0] * dim
    o[k] = 1
    order_list.append(o)
if dim >= 2:
    order_list.append([1, 1] + [0] * (dim - 2))
for cid in cids:
    lo, hi = grid.cell_bounds(cid)
    pts = np.stack([lo + u * (hi - lo) for u in u_rel])
    out["active_basis"][str(cid)] = [int(v) for v in thb.active_basis(cid)]
    for o in order_list:
        vals, dofs = thb.tabulate_basis_derivatives(cid, pts, o)
        assert vals.dtype == np.float64
        out["evals"].append(
            {
                "cid": cid,
                "order": o,
                "pts": [[hx(v) for v in p] for p in pts],
                "dofs": [int(v) for v in dofs],
                "values": [[hx(v) for v in row] for row in vals],
            }
        )
out["main_commit"] = subprocess.run(
    ["git", "-C", MAIN, "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True, check=True,
).stdout.strip()
with open(sys.argv[2], "w") as f:
    json.dump(out, f)
print(case, "ok", thb.num_total_basis, len(out["trunc"]), len(out["evals"]))
