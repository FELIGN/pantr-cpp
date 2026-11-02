"""Test-only snapshot of the THB outputs whose determinism the port promises.

This module is **not** part of the public ``pantr`` API and is underscore-prefixed so pytest does
not collect it. It builds fixed refined spaces and records, as arrays, what the C++ THB code
returns for them: the discrete outputs (active sets, owners, the halo, the coupling graph, the
prolongation's sparsity pattern), which a thread count or an instruction-set rung must not change
at all, and the floating-point outputs (basis values, spline values, quasi-interpolation
coefficients, extraction operators, prolongation entries), which they may move within a derived
bound. ``test_thb_determinism`` compares two snapshots taken under different settings; run as
``python -m tests._thb_snapshot <path>`` the module writes one to ``path`` from a fresh process
(the generic instruction-set rung is chosen by the caller's ``PANTR_ISA``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from pantr._testing import thb_dof_owner, thb_halo
from pantr.bspline import (
    MultiLevelExtraction,
    THBSpline,
    THBSplineSpace,
    coupling_graph,
    create_uniform_space,
    quasi_interpolate_thb_spline,
)
from pantr.grid import Partition, hierarchical_grid, uniform_grid

_Ints = npt.NDArray[np.int64]
_Floats = npt.NDArray[np.float64]


class Snapshot(NamedTuple):
    """What the THB code returns for one space, as arrays.

    Attributes:
        level_sizes (_Ints): Active functions per level.
        active_functions (_Ints): ``active_function_indices`` of every level, concatenated.
        active_ptr (_Ints): Row pointers of the per-cell active sets.
        active_flat (_Ints): ``active_basis`` of every cell, concatenated.
        extraction_ptr (_Ints): Row pointers of the extraction's per-cell active sets.
        extraction_flat (_Ints): ``MultiLevelExtraction.active_basis`` of every cell, concatenated.
        dof_owner (_Ints): Owner of every dof under a seeded partition with inactive cells.
        halo (_Ints): Halo of a seeded owned set.
        graph_xadj (_Ints): Coupling graph row pointers.
        graph_adjncy (_Ints): Coupling graph neighbours.
        graph_weights (_Ints): Coupling graph edge weights.
        tabulated (_Floats): ``tabulate_basis`` at seeded points of seeded cells, concatenated.
        spline_values (_Floats): ``THBSpline.evaluate`` at seeded points of the domain.
        qi_coefficients (_Floats): Coefficients of the quasi-interpolant of a smooth function.
        operators (_Floats): Multilevel and Bézier extraction operators of seeded cells.
    """

    level_sizes: _Ints
    active_functions: _Ints
    active_ptr: _Ints
    active_flat: _Ints
    extraction_ptr: _Ints
    extraction_flat: _Ints
    dof_owner: _Ints
    halo: _Ints
    graph_xadj: _Ints
    graph_adjncy: _Ints
    graph_weights: _Ints
    tabulated: _Floats
    spline_values: _Floats
    qi_coefficients: _Floats
    operators: _Floats


class ProlongationSnapshot(NamedTuple):
    """The CSR prolongation between two spaces.

    Attributes:
        indptr (_Ints): Row pointers.
        indices (_Ints): Column indices, the sparsity pattern.
        data (_Floats): Stored entries.
    """

    indptr: _Ints
    indices: _Ints
    data: _Floats


def large_space() -> THBSplineSpace:
    """Build a degree-(2, 2) space of 48 x 48 root cells with three levels.

    Large enough that the per-cell table, the coupling graph, the extraction, the quasi-
    interpolant and a batched evaluation are above the work model's threshold for a parallel loop
    (``parallel_min_ops``, ``include/pantr/parallel/work_model.hpp``).

    Returns:
        THBSplineSpace: Two nested refined regions, away from the boundary.
    """
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]] * 2, 48), 2)
    grid.refine(0, [4, 4], [44, 44])
    grid.refine(1, [12, 12], [80, 80])
    return THBSplineSpace(create_uniform_space([2, 2], [48, 48]), grid)


def small_space() -> THBSplineSpace:
    """Build a degree-(2, 2) space of 6 x 6 root cells with three levels.

    Small enough to prolong in a fraction of a second, and already above the parallel threshold of
    the prolongation, whose items are whole column solves.

    Returns:
        THBSplineSpace: Two nested refined regions, away from the boundary.
    """
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]] * 2, 6), 2)
    grid.refine(0, [1, 1], [5, 5])
    grid.refine(1, [3, 3], [9, 9])
    return THBSplineSpace(create_uniform_space([2, 2], [6, 6]), grid)


def _csr(rows: list[_Ints]) -> tuple[_Ints, _Ints]:
    """Concatenate arrays into row pointers and a flat array."""
    ptr = np.concatenate([[0], np.cumsum([r.size for r in rows])]).astype(np.int64)
    return ptr, np.concatenate(rows).astype(np.int64)


def take_snapshot(thb: THBSplineSpace) -> Snapshot:
    """Record what the THB code returns for ``thb`` (seeded, at the current settings).

    Args:
        thb (THBSplineSpace): A two-dimensional space with scalar-field evaluation.

    Returns:
        Snapshot: The outputs.
    """
    n_cells = thb.grid.num_cells
    rng = np.random.default_rng(2026)
    ptr, flat = _csr([thb.active_basis(c) for c in range(n_cells)])
    extraction = MultiLevelExtraction(thb)
    e_ptr, e_flat = _csr([extraction.active_basis(c) for c in range(n_cells)])
    partition = Partition(rng.integers(-1, 4, size=n_cells).astype(np.int32), n_parts=4)
    owned = np.flatnonzero(rng.uniform(size=n_cells) < 0.3).astype(np.int64)
    graph = coupling_graph(thb)

    sampled = rng.choice(n_cells, size=min(n_cells, 64), replace=False)
    tabulated = []
    for cid in sampled:
        lo, hi = thb.grid.cell_bounds(int(cid))
        tabulated.append(thb.tabulate_basis(int(cid), lo + (hi - lo) * rng.uniform(size=(5, 2)))[0])
    operators = []
    for cid in sampled:
        operators.append(extraction.multilevel_operator(int(cid)).ravel())
        operators.append(extraction.operator(int(cid)).ravel())

    spline = THBSpline(thb, rng.uniform(-1.0, 1.0, thb.num_total_basis))
    smooth = quasi_interpolate_thb_spline(
        lambda x: np.sin(3.0 * x[:, 0]) * np.cos(2.0 * x[:, 1]) + x[:, 0] * x[:, 1], thb
    )
    return Snapshot(
        level_sizes=np.asarray(thb.num_basis_per_level, dtype=np.int64),
        active_functions=np.concatenate(
            [thb.active_function_indices(lv) for lv in range(thb.num_levels)]
        ),
        active_ptr=ptr,
        active_flat=flat,
        extraction_ptr=e_ptr,
        extraction_flat=e_flat,
        dof_owner=thb_dof_owner(thb, partition).astype(np.int64),
        halo=thb_halo(thb, owned),
        graph_xadj=graph.xadj,
        graph_adjncy=graph.adjncy,
        graph_weights=graph.edge_weights,
        tabulated=np.concatenate([t.ravel() for t in tabulated]),
        spline_values=spline.evaluate(rng.uniform(size=(100_000, 2))),
        qi_coefficients=np.asarray(smooth.control_points, dtype=np.float64),
        operators=np.concatenate(operators),
    )


def take_prolongation(coarse: THBSplineSpace, fine: THBSplineSpace) -> ProlongationSnapshot:
    """Record the sparse prolongation from ``coarse`` to ``fine``.

    Args:
        coarse (THBSplineSpace): The coarse space.
        fine (THBSplineSpace): A refinement of ``coarse``.

    Returns:
        ProlongationSnapshot: Its CSR arrays.
    """
    matrix = coarse.prolongation_to_sparse(fine)
    return ProlongationSnapshot(
        indptr=np.asarray(matrix.indptr, dtype=np.int64),
        indices=np.asarray(matrix.indices, dtype=np.int64),
        data=np.asarray(matrix.data, dtype=np.float64),
    )


def refinement_pair() -> tuple[THBSplineSpace, THBSplineSpace]:
    """Build the small space and an ungraded refinement of it, from every ninth cell.

    Returns:
        tuple[THBSplineSpace, THBSplineSpace]: ``(coarse, fine)``.
    """
    coarse = small_space()
    marked = np.arange(0, coarse.grid.num_cells, 9)
    return coarse, coarse.refine(marked, admissible_class=None)


def take_small() -> tuple[Snapshot, ProlongationSnapshot]:
    """Take the snapshots of the small fixtures, in this process at its current settings.

    Returns:
        tuple[Snapshot, ProlongationSnapshot]: The small space's outputs and the prolongation of
        the small refinement pair.
    """
    coarse, fine = refinement_pair()
    return take_snapshot(coarse), take_prolongation(coarse, fine)


def save(path: Path, snapshot: Snapshot, prolongation: ProlongationSnapshot) -> None:
    """Write the two snapshots to an ``.npz`` file.

    Args:
        path (Path): The file to write.
        snapshot (Snapshot): The space's outputs.
        prolongation (ProlongationSnapshot): The prolongation's CSR arrays.
    """
    arrays = {f"s_{k}": v for k, v in snapshot._asdict().items()}
    arrays |= {f"p_{k}": v for k, v in prolongation._asdict().items()}
    np.savez(path, **arrays)


def load(path: Path) -> tuple[Snapshot, ProlongationSnapshot]:
    """Read the two snapshots :func:`save` wrote.

    Args:
        path (Path): The ``.npz`` file.

    Returns:
        tuple[Snapshot, ProlongationSnapshot]: The space's outputs and the prolongation's arrays.
    """
    with np.load(path) as data:
        return (
            Snapshot(**{k: data[f"s_{k}"] for k in Snapshot._fields}),
            ProlongationSnapshot(**{k: data[f"p_{k}"] for k in ProlongationSnapshot._fields}),
        )


if __name__ == "__main__":
    import pantr
    from pantr._ext import ext
    from tests._tree_env import check_extension_from_tree, check_pantr_from_tree

    check_pantr_from_tree()
    check_extension_from_tree()
    print(ext.isa)
    print(pantr.__file__)
    save(Path(sys.argv[1]), *take_small())
