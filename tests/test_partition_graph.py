"""Tests for :func:`pantr.bspline.partition_graph` (spectral and METIS backends)."""

from __future__ import annotations

import importlib.util

import numba as nb
import numpy as np
import numpy.typing as npt
import pytest
from scipy import sparse

import pantr
from pantr._ext import ext
from pantr._testing import DENSE_FIEDLER_MAX, SpectralInfo, spectral_partition
from pantr.bspline import (
    CouplingGraph,
    THBSplineSpace,
    coupling_graph,
    create_uniform_space,
    partition_graph,
)
from pantr.grid import hierarchical_grid, uniform_grid

requires_pymetis = pytest.mark.skipif(
    importlib.util.find_spec("pymetis") is None, reason="pymetis is not installed"
)


def _make_graph(
    adjacency: npt.NDArray[np.int64], vertex_weights: npt.ArrayLike | None = None
) -> CouplingGraph:
    """Build a CouplingGraph from a dense symmetric integer adjacency matrix."""
    n = adjacency.shape[0]
    csr = sparse.csr_matrix(np.asarray(adjacency, dtype=np.int64))
    csr.sort_indices()
    vweights = (
        np.ones(n) if vertex_weights is None else np.asarray(vertex_weights, dtype=np.float64)
    )
    return CouplingGraph(
        n,
        csr.indptr.astype(np.int64),
        csr.indices.astype(np.int64),
        csr.data.astype(np.int64),
        vweights,
    )


def _cliques(clusters: list[list[int]], n: int, *, intra: int = 10) -> npt.NDArray[np.int64]:
    """Dense adjacency with a strong clique on each cluster."""
    adjacency = np.zeros((n, n), dtype=np.int64)
    for cluster in clusters:
        for i in cluster:
            for j in cluster:
                if i != j:
                    adjacency[i, j] = intra
    return adjacency


# --------------------------------------------------------------------------- #
# Hand-built graphs: the optimal cut is obvious (space-free correctness)
# --------------------------------------------------------------------------- #


def test_two_clusters_separated_by_weak_bridge() -> None:
    adjacency = _cliques([[0, 1, 2], [3, 4, 5]], 6)
    adjacency[2, 3] = adjacency[3, 2] = 1
    owner = partition_graph(_make_graph(adjacency), 2).cell_owner
    assert owner[0] == owner[1] == owner[2]
    assert owner[3] == owner[4] == owner[5]
    assert owner[0] != owner[3]


def test_disconnected_components_separated() -> None:
    # No bridge -> two connected components; exercises the component branch.
    adjacency = _cliques([[0, 1, 2], [3, 4, 5]], 6)
    owner = partition_graph(_make_graph(adjacency), 2).cell_owner
    assert owner[0] == owner[1] == owner[2]
    assert owner[3] == owner[4] == owner[5]
    assert owner[0] != owner[3]


def test_clique_chain_into_four_parts() -> None:
    clusters = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    adjacency = _cliques(clusters, 12)
    for i, j in [(2, 3), (5, 6), (8, 9)]:  # weak chain bridges
        adjacency[i, j] = adjacency[j, i] = 1
    owner = partition_graph(_make_graph(adjacency), 4).cell_owner
    for cluster in clusters:
        assert len(set(owner[cluster].tolist())) == 1, "each clique should be one part"
    assert len(set(owner.tolist())) == 4


def test_large_path_graph_single_cut() -> None:
    # n > dense threshold exercises the sparse eigsh path; a path bisects with one cut.
    n = 600
    diag = np.ones(n - 1, dtype=np.int64)
    csr = sparse.diags([diag, diag], [1, -1], format="csr", dtype=np.int64)
    graph = CouplingGraph(
        n,
        csr.indptr.astype(np.int64),
        csr.indices.astype(np.int64),
        csr.data.astype(np.int64),
        np.ones(n),
    )
    owner = partition_graph(graph, 2).cell_owner
    assert set(owner.tolist()) == {0, 1}
    assert int(np.sum(owner[:-1] != owner[1:])) == 1


def _grid_graph(dims: tuple[int, ...]) -> CouplingGraph:
    """The unit-weight grid graph on ``dims`` cells, vertices in C order."""
    adjacency = None
    for axis, n in enumerate(dims):
        ones = np.ones(n - 1, dtype=np.int64)
        path = sparse.diags([ones, ones], [1, -1], format="csr", dtype=np.int64)
        left = sparse.identity(int(np.prod(dims[:axis], dtype=np.int64)), dtype=np.int64)
        right = sparse.identity(int(np.prod(dims[axis + 1 :], dtype=np.int64)), dtype=np.int64)
        term = sparse.kron(sparse.kron(left, path), right)
        adjacency = term if adjacency is None else adjacency + term
    csr = sparse.csr_matrix(adjacency)
    csr.sort_indices()
    n_total = int(np.prod(dims))
    return CouplingGraph(
        n_total,
        csr.indptr.astype(np.int64),
        csr.indices.astype(np.int64),
        csr.data.astype(np.int64),
        np.ones(n_total),
    )


@pytest.mark.parametrize("dims", [(32, 27), (10, 8, 7)])
def test_large_grid_bisection_follows_the_fiedler_vector(dims: tuple[int, ...]) -> None:
    # Regression: above 512 vertices the bisection must order by the Fiedler vector, not by
    # another eigenvector. Oracle (closed form): the grid graph's Laplacian has the eigenpairs
    # prod_d cos(pi k_d (i_d + 1/2) / n_d), sum_d (2 - 2 cos(pi k_d / n_d)). For these shapes the
    # second-smallest eigenvalue is simple, attained by k = (1, 0, ...) on the first (longest)
    # axis, so the Fiedler vector is cos(pi (i_0 + 1/2) / n_0): positive for i_0 < n_0 / 2,
    # negative after (n_0 is even, so no entry is zero), and the two halves are these.
    lam = sorted(
        float(sum(2.0 - 2.0 * np.cos(np.pi * k / n) for k, n in zip(ks, dims, strict=True)))
        for ks in np.ndindex(*(min(n, 3) for n in dims))
    )
    lam_fiedler = 2.0 - 2.0 * np.cos(np.pi / dims[0])
    assert lam[1] == pytest.approx(lam_fiedler, rel=1e-12)
    assert lam[2] > lam[1] * (1.0 + 1e-3), "the oracle needs a simple second eigenvalue"

    graph = _grid_graph(dims)
    i0 = np.arange(graph.num_vertices) // int(np.prod(dims[1:]))
    fiedler = np.cos(np.pi * (i0 + 0.5) / dims[0])
    expected_left = fiedler > 0.0

    owner = partition_graph(graph, 2).cell_owner
    assert set(owner[expected_left].tolist()) == {int(owner[0])}
    assert set(owner[~expected_left].tolist()) == {1 - int(owner[0])}


def _path_graph(n: int) -> CouplingGraph:
    """The unit-weight path graph on ``n`` vertices."""
    diag = np.ones(n - 1, dtype=np.int64)
    csr = sparse.diags([diag, diag], [1, -1], format="csr", dtype=np.int64)
    return CouplingGraph(
        n,
        csr.indptr.astype(np.int64),
        csr.indices.astype(np.int64),
        csr.data.astype(np.int64),
        np.ones(n),
    )


def test_lanczos_failure_falls_back_to_dense() -> None:
    # One Lanczos restart cannot converge on a 600-vertex path; the dense fallback must still
    # partition, and the fallback is recorded rather than silent.
    part, info = spectral_partition(_path_graph(600), 2, lanczos_max_iter=1)
    assert info == SpectralInfo(
        dense_solves=0, lanczos_solves=0, dense_fallbacks=1, component_orders=0
    )
    owner = part.cell_owner
    assert set(owner.tolist()) == {0, 1}
    assert int(np.sum(owner[:-1] != owner[1:])) == 1
    # Without the limit the same graph takes the Lanczos path, to the same partition.
    default_part, default_info = spectral_partition(_path_graph(600), 2)
    assert default_info.lanczos_solves == 1
    assert default_info.dense_fallbacks == 0
    # The path's Fiedler vector is antisymmetric, so its two largest-magnitude entries tie up to
    # rounding and the two solvers may label the halves the other way round.
    other = default_part.cell_owner
    assert np.array_equal(other, owner) or np.array_equal(other, 1 - owner)


def test_clique_chain_into_three_parts() -> None:
    clusters = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    adjacency = _cliques(clusters, 9)
    for i, j in [(2, 3), (5, 6)]:
        adjacency[i, j] = adjacency[j, i] = 1
    owner = partition_graph(_make_graph(adjacency), 3).cell_owner
    for cluster in clusters:
        assert len(set(owner[cluster].tolist())) == 1, "each clique should be one part"
    assert len(set(owner.tolist())) == 3


def test_dense_threshold_boundary() -> None:
    # m=512 must use the dense solver (<=); m=513 the Lanczos solve (>).
    assert DENSE_FIEDLER_MAX == 512
    for n in (512, 513):
        graph = _path_graph(n)
        owner = partition_graph(graph, 2).cell_owner
        assert int(np.sum(owner[:-1] != owner[1:])) == 1
        _, info = spectral_partition(graph, 2)
        assert (info.dense_solves, info.lanczos_solves) == ((1, 0) if n <= 512 else (0, 1))


# --------------------------------------------------------------------------- #
# Real spaces
# --------------------------------------------------------------------------- #


def _assert_full_partition(owner: npt.NDArray[np.int32], n_parts: int, n_active: int) -> None:
    assert int(np.count_nonzero(owner >= 0)) == n_active
    assert set(owner[owner >= 0].tolist()) == set(range(n_parts))


def test_real_space_partition_is_valid() -> None:
    space = create_uniform_space([2, 2], [6, 6])
    part = partition_graph(coupling_graph(space), 4)
    assert part.n_parts == 4
    _assert_full_partition(part.cell_owner, 4, 36)
    assert np.bincount(part.cell_owner, minlength=4).min() >= 1


def test_thb_space_partition_is_valid() -> None:
    root = create_uniform_space(2, 4)
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 2)
    grid.refine(0, [0], [2])
    thb = THBSplineSpace(root, grid)
    part = partition_graph(coupling_graph(thb), 2)
    _assert_full_partition(part.cell_owner, 2, thb.grid.num_cells)


def test_weight_balanced_not_count_balanced() -> None:
    space = create_uniform_space(2, 8)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0])
    part = partition_graph(coupling_graph(space, cell_weights=weights), 2)
    owner = part.cell_owner
    part_weights = np.array([weights[owner == r].sum() for r in range(2)])
    np.testing.assert_allclose(np.sort(part_weights), [7.0, 7.0])


def test_is_deterministic() -> None:
    graph = coupling_graph(create_uniform_space([2, 2], [5, 5]))
    a = partition_graph(graph, 4).cell_owner
    b = partition_graph(graph, 4).cell_owner
    np.testing.assert_array_equal(a, b)


def test_is_independent_of_the_thread_count() -> None:
    # Nothing in the partitioner runs in parallel; one thread and several give the same owners.
    many = min(8, nb.config.NUMBA_NUM_THREADS, ext.max_threads())
    if many < 2:
        pytest.skip("a single thread is available, so there is no parallel run to compare")
    for graph in (coupling_graph(create_uniform_space([2, 2], [30, 25])), _grid_graph((32, 27))):
        with pantr.num_threads(1):
            serial = partition_graph(graph, 7).cell_owner
        with pantr.num_threads(many):
            parallel = partition_graph(graph, 7).cell_owner
        np.testing.assert_array_equal(parallel, serial)


def _edge_cut(graph: CouplingGraph, owner: npt.NDArray[np.int32]) -> int:
    """The weight of the edges whose ends have different owners, each edge counted once."""
    rows = np.repeat(np.arange(graph.num_vertices), np.diff(graph.xadj))
    return int(graph.edge_weights[owner[rows] != owner[graph.adjncy]].sum()) // 2


def test_two_way_balance_is_within_one_vertex_weight() -> None:
    # The first part in Fiedler order ends at the first vertex whose running weight reaches half
    # the total, so each part's weight is within the largest vertex weight of half.
    space = create_uniform_space([2, 1], [30, 20])
    weights = 1.0 + (np.arange(600) * 7 % 5)
    owner = partition_graph(coupling_graph(space, cell_weights=weights), 2).cell_owner
    total = float(weights.sum())
    for r in (0, 1):
        assert abs(float(weights[owner == r].sum()) - total / 2) < float(weights.max())


def test_grid_graph_cut_is_one_slab_of_edges() -> None:
    # The Fiedler split of the 32 x 27 grid graph separates slabs 15 and 16 of the first axis
    # (see the regression test above), so the cut is the 27 unit edges between them.
    graph = _grid_graph((32, 27))
    assert _edge_cut(graph, partition_graph(graph, 2).cell_owner) == 27


def test_n_parts_one() -> None:
    owner = partition_graph(coupling_graph(create_uniform_space(2, 4)), 1).cell_owner
    assert np.all(owner == 0)


def test_n_parts_equals_vertices_is_bijection() -> None:
    owner = partition_graph(coupling_graph(create_uniform_space(2, 5)), 5).cell_owner
    np.testing.assert_array_equal(np.sort(owner), np.arange(5))


def test_respects_cell_active() -> None:
    space = create_uniform_space(2, 8)
    active = np.array([True] * 4 + [False] * 4)
    part = partition_graph(coupling_graph(space), 2, cell_active=active)
    owner = part.cell_owner
    assert np.all(owner[4:] == -1)
    assert set(owner[:4].tolist()) == {0, 1}
    np.testing.assert_array_equal(part.active_mask, active)


def test_cell_active_int_coerced_to_bool() -> None:
    graph = coupling_graph(create_uniform_space(2, 4))
    part = partition_graph(graph, 2, cell_active=[1, 1, 0, 0])
    assert np.all(part.cell_owner[2:] == -1)


def test_cell_owner_is_read_only() -> None:
    part = partition_graph(coupling_graph(create_uniform_space(2, 4)), 2)
    assert not part.cell_owner.flags.writeable


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_parts", [0, -1])
def test_invalid_n_parts_raises(n_parts: int) -> None:
    graph = coupling_graph(create_uniform_space(2, 4))
    with pytest.raises(ValueError, match="n_parts must be >= 1"):
        partition_graph(graph, n_parts)


def test_n_parts_exceeds_active_raises() -> None:
    graph = coupling_graph(create_uniform_space(2, 4))
    with pytest.raises(ValueError, match="exceeds the number of active cells"):
        partition_graph(graph, 5)


def test_n_parts_exceeds_active_with_mask_raises() -> None:
    graph = coupling_graph(create_uniform_space(2, 8))
    active = np.array([True] * 2 + [False] * 6)
    with pytest.raises(ValueError, match="exceeds the number of active cells"):
        partition_graph(graph, 3, cell_active=active)


def test_non_coupling_graph_raises() -> None:
    with pytest.raises(TypeError, match="CouplingGraph"):
        partition_graph(np.zeros(3), 2)  # type: ignore[arg-type]


def test_cell_active_wrong_shape_raises() -> None:
    graph = coupling_graph(create_uniform_space(2, 4))
    with pytest.raises(ValueError, match="cell_active must have shape"):
        partition_graph(graph, 2, cell_active=[True, False])


def test_cell_active_all_false_raises() -> None:
    graph = coupling_graph(create_uniform_space(2, 4))
    with pytest.raises(ValueError, match="at least one cell active"):
        partition_graph(graph, 2, cell_active=[False, False, False, False])


def test_unknown_backend_raises() -> None:
    graph = coupling_graph(create_uniform_space(2, 4))
    with pytest.raises(ValueError, match="unknown backend"):
        partition_graph(graph, 2, backend="nope")


def test_metis_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate pymetis unavailable (or broken install) by making import_module raise.
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == "pymetis":
            raise ImportError("no module named 'pymetis'")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    graph = coupling_graph(create_uniform_space(2, 4))
    with pytest.raises(ImportError, match="requires 'pymetis'"):
        partition_graph(graph, 2, backend="metis")


# --------------------------------------------------------------------------- #
# METIS backend (optional pymetis)
# --------------------------------------------------------------------------- #


@requires_pymetis
def test_metis_separates_clusters() -> None:
    adjacency = _cliques([[0, 1, 2], [3, 4, 5]], 6)
    adjacency[2, 3] = adjacency[3, 2] = 1
    owner = partition_graph(_make_graph(adjacency), 2, backend="metis").cell_owner
    assert owner[0] == owner[1] == owner[2]
    assert owner[3] == owner[4] == owner[5]
    assert owner[0] != owner[3]


@requires_pymetis
def test_metis_real_space_is_valid() -> None:
    space = create_uniform_space([2, 2], [6, 6])
    part = partition_graph(coupling_graph(space), 4, backend="metis")
    owner = part.cell_owner
    assert part.n_parts == 4
    assert int(np.count_nonzero(owner >= 0)) == 36  # all active cells assigned
    # METIS does not guarantee every part is non-empty; check only valid range.
    assert owner.min() >= 0 and owner.max() < 4


@requires_pymetis
def test_metis_thb_space_is_valid() -> None:
    root = create_uniform_space(2, 4)
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 2)
    grid.refine(0, [0], [2])
    thb = THBSplineSpace(root, grid)
    owner = partition_graph(coupling_graph(thb), 2, backend="metis").cell_owner
    # METIS does not guarantee both parts non-empty; check only valid range.
    assert np.all(owner >= 0) and owner.max() < 2


@requires_pymetis
def test_metis_respects_cell_active() -> None:
    space = create_uniform_space(2, 8)
    active = np.array([True] * 4 + [False] * 4)
    part = partition_graph(coupling_graph(space), 2, backend="metis", cell_active=active)
    owner = part.cell_owner
    assert np.all(owner[4:] == -1)
    assert set(owner[:4].tolist()) <= {0, 1}
    np.testing.assert_array_equal(part.active_mask, active)


@requires_pymetis
def test_metis_n_parts_one() -> None:
    graph = coupling_graph(create_uniform_space(2, 4))
    owner = partition_graph(graph, 1, backend="metis").cell_owner
    assert np.all(owner == 0)


@requires_pymetis
def test_metis_is_deterministic() -> None:
    graph = coupling_graph(create_uniform_space([2, 2], [5, 5]))
    a = partition_graph(graph, 4, backend="metis").cell_owner
    b = partition_graph(graph, 4, backend="metis").cell_owner
    np.testing.assert_array_equal(a, b)


@requires_pymetis
def test_metis_zero_vertex_weight_clamped() -> None:
    # A vertex with weight 0 must be clamped to 1 (METIS requires positive int weights).
    adjacency = _cliques([[0, 1, 2], [3, 4, 5]], 6)
    adjacency[2, 3] = adjacency[3, 2] = 1
    vertex_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    graph = _make_graph(adjacency, vertex_weights=vertex_weights)
    owner = partition_graph(graph, 2, backend="metis").cell_owner
    assert set(owner.tolist()).issubset({0, 1})


@requires_pymetis
def test_metis_nonuniform_vertex_weights_valid() -> None:
    # Verify non-uniform vertex weights flow through integer conversion without error
    # and produce a valid partition (all cells assigned, all owners in range).
    adjacency = _cliques([[0, 1, 2], [3, 4, 5]], 6)
    adjacency[2, 3] = adjacency[3, 2] = 1
    vertex_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
    graph = _make_graph(adjacency, vertex_weights=vertex_weights)
    owner = partition_graph(graph, 2, backend="metis").cell_owner
    assert set(owner.tolist()).issubset({0, 1})
