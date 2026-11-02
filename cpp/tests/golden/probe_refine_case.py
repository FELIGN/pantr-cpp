# Recorded-data script, kept as it was run: outside the project style (ruff, ruff format).
# ruff: noqa
# fmt: off
"""Record one THB refine / coarsen / prolongation / restrict case from main's Numba pantr.

usage: probe_refine_case.py <case id> <out.json> <main checkout>   (one fresh process per case)

Regenerate cpp/tests/thb_refine_golden.hpp (the header names the main commit it was recorded at):
  1. Check out that main commit as <main checkout>; the `pantr-cpp` conda environment imports it
     (an editable install), so run with no PYTHONPATH and from a directory that is not a pantr
     checkout. The script refuses to run if `pantr` is imported from anywhere else.
  2. For each case id gen_refine_golden.py lists (ref_2 ref_3 ref_n reg_3 coa_2 coa_3 coa_n pro_a
     pro_b res_box res_deep), in its own process:
         conda run -n pantr-cpp python probe_refine_case.py <id> <dir>/case_<id>.json <main checkout>
  3. python3 gen_refine_golden.py <dir> cpp/tests/thb_refine_golden.hpp
"""
import json
import os
import subprocess
import sys

import numpy as np

import pantr

MAIN = os.path.abspath(sys.argv[3])
assert pantr.__file__.startswith(os.path.join(MAIN, "src") + os.sep), pantr.__file__

from pantr.bspline import BsplineSpace, BsplineSpace1D, create_thb_space

K2 = [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
K3 = [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]


def root2d():
    return BsplineSpace([BsplineSpace1D(np.asarray(K2), 2), BsplineSpace1D(np.asarray(K2), 2)])


def root1d():
    return BsplineSpace([BsplineSpace1D(np.asarray(K3), 3)])


def hx(x):
    return float(x).hex()


def cells_of(thb):
    g = thb.grid
    return [
        [int(g.cell_level(c))] + [int(v) for v in g.cell_multi_index(c)] for c in range(g.num_cells)
    ]


def summary(thb):
    return {
        "dim": thb.dim,
        "num_levels": thb.num_levels,
        "num_total": thb.num_total_basis,
        "per_level": [int(v) for v in thb.num_basis_per_level],
        "cells": cells_of(thb),
    }


def corner_refine(m, steps):
    thb = create_thb_space(root2d())
    for lvl in range(steps):
        cid = thb.grid.cell_id(lvl, [0, 0])
        assert cid is not None
        thb = thb.refine([cid], admissible_class=m)
    return thb


def level_ids(thb, level):
    return [c for c in range(thb.grid.num_cells) if thb.grid.cell_level(c) == level]


CASE = sys.argv[1]
out = {"case": CASE}

if CASE.startswith("ref_"):
    # ref_<m>: four successive corner refinements, class m ("n" = ungraded)
    m = None if CASE.endswith("n") else int(CASE[-1])
    thb = corner_refine(m, 4)
    out.update(summary(thb))
    out["m"] = m
elif CASE == "reg_3":
    # graded region refinements away from the corner, class 3
    thb = create_thb_space(root2d())
    thb = thb.refine_region(0, [1, 1], [3, 3], admissible_class=3)
    thb = thb.refine_region(1, [2, 2], [4, 4], admissible_class=3)
    thb = thb.refine_region(2, [4, 4], [6, 6], admissible_class=3)
    out.update(summary(thb))
elif CASE.startswith("coa_"):
    # coa_<m>: coarsen every active level-2 cell of the class-2 corner space
    m = None if CASE.endswith("n") else int(CASE[-1])
    thb = corner_refine(2, 4)
    ids = level_ids(thb, 2)
    out["ids"] = ids
    out["input"] = summary(thb)
    res = thb.coarsen(ids, admissible_class=m)
    out.update(summary(res))
    out["m"] = m
elif CASE.startswith("pro_"):
    which = CASE[4:]
    if which == "a":
        root = root2d()
        c = create_thb_space(root, truncate=True).refine_region(0, [0, 0], [2, 2], admissible_class=None)
        f = c.refine_region(1, [0, 0], [2, 2], admissible_class=None)
    elif which == "h":
        root = root2d()
        c = create_thb_space(root, truncate=False).refine_region(0, [0, 0], [2, 2], admissible_class=None)
        f = c.refine_region(1, [0, 0], [2, 2], admissible_class=None)
    elif which == "b":
        root = root1d()
        c = create_thb_space(root, factor=2, truncate=True, regularity=0).refine_region(
            0, [0], [2], admissible_class=None
        )
        f = c.refine_region(1, [0], [2], admissible_class=None)
    else:
        raise SystemExit("unknown prolongation case")
    P = c.prolongation_to_sparse(f)
    assert P.has_sorted_indices
    out["coarse"] = summary(c)
    out["fine"] = summary(f)
    out["shape"] = [int(v) for v in P.shape]
    out["indptr"] = [int(v) for v in P.indptr]
    out["indices"] = [int(v) for v in P.indices]
    out["data"] = [hx(v) for v in P.data]
elif CASE.startswith("res_"):
    which = CASE[4:]
    thb = create_thb_space(root2d())
    thb = thb.refine_region(0, [0, 0], [2, 2], admissible_class=None)
    thb = thb.refine_region(1, [0, 0], [2, 2], admissible_class=None)
    g = thb.grid
    if which == "box":
        # cells whose centre lies in [0.1, 0.6]^2
        ids = []
        for c in range(g.num_cells):
            lo, hi = g.cell_bounds(c)
            ctr = 0.5 * (lo + hi)
            if np.all(ctr >= 0.1) and np.all(ctr <= 0.6):
                ids.append(c)
    elif which == "deep":
        # one deepest cell: the whole tiling of its root cell comes with it
        ids = [max(range(g.num_cells), key=lambda c: (g.cell_level(c), -c))]
    else:
        raise SystemExit("unknown restriction case")
    out["parent"] = summary(thb)
    out["ids"] = ids
    r = thb.restrict(ids)
    out["space"] = summary(r.space)
    out["dof_map"] = [int(v) for v in r.local_to_global_dof]
    out["cell_map"] = [int(v) for v in r.local_to_global_cell]
    out["active"] = [r.space.active_function_indices(l).tolist() for l in range(r.space.num_levels)]
else:
    raise SystemExit("unknown case " + CASE)

out["main_commit"] = subprocess.run(
    ["git", "-C", MAIN, "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True, check=True,
).stdout.strip()
with open(sys.argv[2], "w") as f:
    json.dump(out, f)
