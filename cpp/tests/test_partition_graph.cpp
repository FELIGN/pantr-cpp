// The coupling graph and the spectral graph partitioner.
//
// Oracles: the tensor-product graph against a brute-force count over every pair of cells and every
// basis function, with each function's support read off the knot vector (not the first function
// per interval the builder uses); the incidence builder against `B B^T` formed densely; the
// partitions of grid and path graphs against their closed-form Fiedler vectors (the Laplacian of a
// path on n vertices has eigenvectors cos(pi k (i + 1/2) / n), and a grid graph's are products);
// balance and edge cut against sums over the owners written in the test.
#include <doctest/doctest.h>

#include <pantr/bspline/coupling_graph.hpp>
#include <pantr/bspline/knot_factories.hpp>
#include <pantr/bspline/space.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/parallel/affinity.hpp>
#include <pantr/parallel/pool.hpp>
#include <pantr/partition/coupling_graph.hpp>
#include <pantr/partition/partition_graph.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace bs = pantr::bspline;
namespace pp = pantr::partition;

namespace {

using Dense = std::vector<std::vector<std::int64_t>>;

std::string message_of(const std::function<void()>& call) {
  try {
    call();
  } catch (const pantr::value_error& e) {
    return e.what();
  }
  return "<no value_error>";
}

/// Densify a graph's edge weights, checking along the way that each row is ascending.
Dense dense_of(const pp::CouplingGraph& g) {
  const auto n = static_cast<std::size_t>(g.num_vertices());
  Dense out(n, std::vector<std::int64_t>(n, 0));
  for (std::size_t c = 0; c < n; ++c) {
    for (auto k = static_cast<std::size_t>(g.xadj()[c]);
         k < static_cast<std::size_t>(g.xadj()[c + 1]); ++k) {
      if (k > static_cast<std::size_t>(g.xadj()[c])) {
        CHECK(g.adjncy()[k - 1] < g.adjncy()[k]);
      }
      out[c][static_cast<std::size_t>(g.adjncy()[k])] = g.edge_weights()[k];
    }
  }
  return out;
}

/// Per direction and interval, the functions supported there, read off the knots: function `i` is
/// non-zero on interval `[u_j, u_{j+1}]` iff `knots[i] <= u_j` and `u_{j+1} <= knots[i + p + 1]`.
template <class T>
std::vector<std::vector<std::int64_t>> supports_from_knots(const bs::BsplineSpace1D<T>& s) {
  const auto knots = s.knots();
  const auto p = static_cast<std::size_t>(s.degree());
  const auto [unique, mult] = s.unique_knots_and_multiplicity(true);
  std::vector<std::vector<std::int64_t>> out(unique.size() - 1);
  for (std::size_t j = 0; j + 1 < unique.size(); ++j) {
    for (std::size_t i = 0; i + p + 1 < knots.size(); ++i) {
      if (knots[i] <= unique[j] && unique[j + 1] <= knots[i + p + 1]) {
        out[j].push_back(static_cast<std::int64_t>(i));
      }
    }
  }
  return out;
}

/// The coupling of every pair of cells by brute force over every pair of global functions.
template <class T>
Dense brute_force(const bs::BsplineSpace<T>& space) {
  const auto& spaces = space.spaces();
  std::vector<std::vector<std::vector<std::int64_t>>> sup;
  std::vector<std::size_t> cells;
  std::vector<std::size_t> funcs;
  for (const auto& s : spaces) {
    sup.push_back(supports_from_knots(s));
    cells.push_back(sup.back().size());
    funcs.push_back(static_cast<std::size_t>(s.num_basis()));
  }
  std::size_t n_cells = 1;
  std::size_t n_funcs = 1;
  for (std::size_t d = 0; d < spaces.size(); ++d) {
    n_cells *= cells[d];
    n_funcs *= funcs[d];
  }
  auto unflat = [](std::size_t flat, const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> m(shape.size());
    for (std::size_t d = shape.size(); d-- > 0;) {
      m[d] = flat % shape[d];
      flat /= shape[d];
    }
    return m;
  };
  // supported[c] = the global functions non-zero on cell c.
  std::vector<std::set<std::size_t>> supported(n_cells);
  for (std::size_t f = 0; f < n_funcs; ++f) {
    const auto fm = unflat(f, funcs);
    for (std::size_t c = 0; c < n_cells; ++c) {
      const auto cm = unflat(c, cells);
      bool on = true;
      for (std::size_t d = 0; d < spaces.size() && on; ++d) {
        const auto& list = sup[d][cm[d]];
        on = std::find(list.begin(), list.end(), static_cast<std::int64_t>(fm[d])) != list.end();
      }
      if (on) {
        supported[c].insert(f);
      }
    }
  }
  Dense out(n_cells, std::vector<std::int64_t>(n_cells, 0));
  for (std::size_t a = 0; a < n_cells; ++a) {
    for (std::size_t b = 0; b < n_cells; ++b) {
      if (a != b) {
        for (const std::size_t f : supported[a]) {
          out[a][b] += static_cast<std::int64_t>(supported[b].count(f));
        }
      }
    }
  }
  return out;
}

template <class T>
bs::BsplineSpace1D<T> open_space(long long cells, int degree, long long continuity) {
  const auto knots = bs::uniform_open_knots<T>(cells, degree, continuity, T(0), T(1));
  return bs::BsplineSpace1D<T>(std::span<const T>(knots), degree);
}

/// A CSR graph from a dense symmetric adjacency.
pp::CouplingGraph graph_from_dense(const Dense& a, std::vector<double> weights = {}) {
  const std::size_t n = a.size();
  std::vector<std::int64_t> xadj(n + 1, 0);
  std::vector<std::int64_t> adjncy;
  std::vector<std::int64_t> ew;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (a[i][j] != 0) {
        adjncy.push_back(static_cast<std::int64_t>(j));
        ew.push_back(a[i][j]);
      }
    }
    xadj[i + 1] = static_cast<std::int64_t>(adjncy.size());
  }
  if (weights.empty()) {
    weights.assign(n, 1.0);
  }
  return {static_cast<std::int64_t>(n), xadj, adjncy, ew, weights};
}

/// The unit-weight grid graph on `dims` (C order), 2*ndim neighbors per interior vertex.
pp::CouplingGraph grid_graph(const std::vector<std::int64_t>& dims) {
  std::int64_t n = 1;
  for (const auto d : dims) {
    n *= d;
  }
  std::vector<std::int64_t> stride(dims.size(), 1);
  for (std::size_t d = dims.size(); d-- > 1;) {
    stride[d - 1] = stride[d] * dims[d];
  }
  std::vector<std::int64_t> xadj{0};
  std::vector<std::int64_t> adjncy;
  for (std::int64_t v = 0; v < n; ++v) {
    std::vector<std::int64_t> row;
    for (std::size_t d = 0; d < dims.size(); ++d) {
      const std::int64_t i = (v / stride[d]) % dims[d];
      if (i > 0) {
        row.push_back(v - stride[d]);
      }
      if (i + 1 < dims[d]) {
        row.push_back(v + stride[d]);
      }
    }
    std::sort(row.begin(), row.end());
    adjncy.insert(adjncy.end(), row.begin(), row.end());
    xadj.push_back(static_cast<std::int64_t>(adjncy.size()));
  }
  std::vector<std::int64_t> ew(adjncy.size(), 1);
  return {n, xadj, adjncy, ew, std::vector<double>(static_cast<std::size_t>(n), 1.0)};
}

/// Whether `owner` is the split `first axis index < half` (either labelling).
bool splits_first_axis(std::span<const std::int32_t> owner, std::int64_t per_slab,
                       std::int64_t half) {
  const std::int32_t lo = owner[0];
  for (std::size_t v = 0; v < owner.size(); ++v) {
    const bool left = static_cast<std::int64_t>(v) / per_slab < half;
    if ((owner[v] == lo) != left) {
      return false;
    }
  }
  return true;
}

/// The weight of the edges whose ends have different owners, each edge counted once.
std::int64_t edge_cut(const pp::CouplingGraph& g, std::span<const std::int32_t> owner) {
  std::int64_t cut = 0;
  for (std::size_t c = 0; c < owner.size(); ++c) {
    for (auto k = static_cast<std::size_t>(g.xadj()[c]);
         k < static_cast<std::size_t>(g.xadj()[c + 1]); ++k) {
      if (owner[static_cast<std::size_t>(g.adjncy()[k])] != owner[c]) {
        cut += g.edge_weights()[k];
      }
    }
  }
  return cut / 2;
}

}  // namespace

// --------------------------------------------------------------------------- coupling graph --

TEST_CASE_TEMPLATE("coupling_graph: tensor product matches the brute-force support overlap", T,
                   double, float) {
  struct Axis {
    long long cells;
    int degree;
    long long continuity;
  };
  const std::vector<std::vector<Axis>> cases = {
      {{1, 2, 1}},
      {{4, 0, -1}},
      {{5, 1, 0}},
      {{6, 2, 1}},
      {{7, 3, 2}},
      {{5, 2, 0}},
      {{6, 3, 1}},
      {{3, 1, 0}, {4, 2, 1}},
      {{4, 2, 0}, {3, 3, 2}},
      {{2, 1, 0}, {3, 2, 1}, {2, 1, 0}},
  };
  for (const auto& axes : cases) {
    std::vector<bs::BsplineSpace1D<T>> dirs;
    for (const auto& a : axes) {
      dirs.push_back(open_space<T>(a.cells, a.degree, a.continuity));
    }
    const bs::BsplineSpace<T> space(dirs);
    const pp::CouplingGraph g = bs::coupling_graph(space);
    CAPTURE(axes.size());
    CAPTURE(axes[0].cells);
    CAPTURE(axes[0].degree);
    CAPTURE(axes[0].continuity);
    const Dense expected = brute_force(space);
    CHECK(g.num_vertices() == static_cast<std::int64_t>(expected.size()));
    CHECK(dense_of(g) == expected);
    CHECK(g.xadj().size() == expected.size() + 1);
    for (const double w : g.vertex_weights()) {
      CHECK(w == 1.0);
    }
  }
}

TEST_CASE("coupling_graph: non-uniform knots with a repeated interior knot") {
  // Knots 0 0 0 0.1 0.5 0.5 0.7 1 1 1 (degree 2): a C^0 knot at 0.5 among C^1 ones.
  const std::vector<double> knots{0, 0, 0, 0.1, 0.5, 0.5, 0.7, 1, 1, 1};
  const bs::BsplineSpace<double> space({bs::BsplineSpace1D<double>(knots, 2)});
  const pp::CouplingGraph g = bs::coupling_graph(space);
  CHECK(dense_of(g) == brute_force(space));
}

TEST_CASE("coupling_graph: cell weights are copied and checked") {
  const bs::BsplineSpace<double> space({open_space<double>(4, 2, 1)});
  const std::vector<double> w{1.0, 2.0, 3.0, 4.0};
  const pp::CouplingGraph g = bs::coupling_graph(space, std::span<const double>(w));
  CHECK(std::vector<double>(g.vertex_weights().begin(), g.vertex_weights().end()) == w);
  const std::vector<double> zeros(4, 0.0);
  CHECK(bs::coupling_graph(space, std::span<const double>(zeros)).vertex_weights()[0] == 0.0);
  const std::vector<double> short_w{1.0, 2.0};
  CHECK(message_of([&] {
          static_cast<void>(bs::coupling_graph(space, std::span<const double>(short_w)));
        }) == "cell_weights must have shape (4,); got (2,).");
  for (const double bad :
       {-1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    const std::vector<double> bw{1.0, bad, 1.0, 1.0};
    CHECK(message_of([&] {
            static_cast<void>(bs::coupling_graph(space, std::span<const double>(bw)));
          }) == "cell_weights must be finite and non-negative.");
  }
}

TEST_CASE("coupling_graph: a periodic direction is refused") {
  const auto knots = bs::uniform_periodic_knots<double>(4, 2, 1, 0.0, 1.0);
  const bs::BsplineSpace<double> space(
      {open_space<double>(3, 2, 1), bs::BsplineSpace1D<double>(knots, 2, true)});
  CHECK(message_of([&] { static_cast<void>(bs::coupling_graph(space)); }) ==
        "periodic B-spline spaces are not supported.");
}

TEST_CASE("coupling_graph_from_incidence: matches B B^T without its diagonal") {
  // 4 cells, 6 DOFs, an irregular incidence (a THB-like overlap pattern).
  const std::vector<std::vector<std::int64_t>> rows = {{0, 1, 2}, {1, 2, 3, 4}, {4, 5}, {0, 5}};
  std::vector<std::int64_t> ptr{0};
  std::vector<std::int64_t> dofs;
  for (const auto& r : rows) {
    dofs.insert(dofs.end(), r.begin(), r.end());
    ptr.push_back(static_cast<std::int64_t>(dofs.size()));
  }
  const pp::CouplingGraph g = pp::coupling_graph_from_incidence(4, 6, ptr, dofs);
  Dense expected(4, std::vector<std::int64_t>(4, 0));
  for (std::size_t a = 0; a < 4; ++a) {
    for (std::size_t b = 0; b < 4; ++b) {
      if (a != b) {
        for (const auto d : rows[a]) {
          expected[a][b] += std::count(rows[b].begin(), rows[b].end(), d);
        }
      }
    }
  }
  CHECK(dense_of(g) == expected);
  // A DOF listed twice counts twice, as a summed incidence entry does.
  const std::vector<std::int64_t> dup_ptr{0, 2, 3};
  const std::vector<std::int64_t> dup_dofs{0, 0, 0};
  CHECK(dense_of(pp::coupling_graph_from_incidence(2, 1, dup_ptr, dup_dofs)) ==
        Dense{{0, 2}, {2, 0}});
  // A cell with no DOF has no edge.
  const std::vector<std::int64_t> empty_ptr{0, 0, 1};
  const std::vector<std::int64_t> one{0};
  CHECK(pp::coupling_graph_from_incidence(2, 1, empty_ptr, one).adjncy().empty());
}

TEST_CASE("coupling_graph_from_incidence: refusals") {
  const std::vector<std::int64_t> ptr{0, 1, 2};
  const std::vector<std::int64_t> dofs{0, 1};
  CHECK(message_of([&] {
          static_cast<void>(pp::coupling_graph_from_incidence(-1, 2, ptr, dofs));
        }).find("must be >= 0") != std::string::npos);
  CHECK(message_of([&] {
          static_cast<void>(pp::coupling_graph_from_incidence(3, 2, ptr, dofs));
        }).find("cell_ptr must have") != std::string::npos);
  const std::vector<std::int64_t> bad_ptr{0, 2, 1};
  CHECK(message_of([&] {
          static_cast<void>(pp::coupling_graph_from_incidence(2, 2, bad_ptr, dofs));
        }).find("cell_ptr must have") != std::string::npos);
  const std::vector<std::int64_t> bad_dofs{0, 2};
  CHECK(message_of([&] {
          static_cast<void>(pp::coupling_graph_from_incidence(2, 2, ptr, bad_dofs));
        }) == "dof ids must lie in [0, 2).");
}

TEST_CASE("CouplingGraph: the constructor checks the arrays") {
  using V = std::vector<std::int64_t>;
  const std::vector<double> w2{1.0, 1.0};
  CHECK(message_of([&] { pp::CouplingGraph(-1, V{0}, V{}, V{}, {}); }) ==
        "num_vertices must be >= 0; got -1.");
  CHECK(message_of([&] { pp::CouplingGraph(2, V{0, 1}, V{1}, V{1}, w2); }) ==
        "xadj must have shape (3,); got (2,).");
  CHECK(message_of([&] { pp::CouplingGraph(2, V{1, 1, 1}, V{1}, V{1}, w2); }) ==
        "xadj must start at 0; got 1.");
  CHECK(message_of([&] { pp::CouplingGraph(2, V{0, 2, 1}, V{1, 0}, V{1, 1}, w2); }) ==
        "xadj must be non-decreasing.");
  CHECK(message_of([&] { pp::CouplingGraph(2, V{0, 1, 1}, V{1, 0}, V{1, 1}, w2); }) ==
        "xadj[-1] must equal the length of adjncy (2); got 1.");
  CHECK(message_of([&] { pp::CouplingGraph(2, V{0, 1, 2}, V{1, 0}, V{1}, w2); }) ==
        "edge_weights must have shape (2,); got (1,).");
  CHECK(message_of([&] { pp::CouplingGraph(2, V{0, 1, 2}, V{1, 2}, V{1, 1}, w2); }) ==
        "adjncy entries must lie in [0, 2).");
  CHECK(message_of([&] { pp::CouplingGraph(2, V{0, 1, 2}, V{1, -1}, V{1, 1}, w2); }) ==
        "adjncy entries must lie in [0, 2).");
  CHECK(message_of([&] { pp::CouplingGraph(2, V{0, 1, 2}, V{1, 0}, V{1, 1}, {1.0}); }) ==
        "vertex_weights must have shape (2,); got (1,).");
  CHECK(message_of([&] { pp::CouplingGraph(2, V{0, 1, 2}, V{1, 0}, V{1, 1}, {1.0, -2.0}); }) ==
        "vertex_weights must be finite and non-negative.");
  const pp::CouplingGraph empty(0, V{0}, V{}, V{}, {});
  CHECK(empty.num_vertices() == 0);
}

// ---------------------------------------------------------------------- spectral partition --

TEST_CASE("Spectral: a grid graph above the dense limit splits where its Fiedler vector does") {
  // The regression for picking a non-Fiedler eigenvector above 512 vertices. On a 32 x 27 grid
  // graph lambda_2 = 2 - 2 cos(pi / 32) is simple (the next, 2 - 2 cos(pi / 27), is larger) and its
  // eigenvector cos(pi (i + 1/2) / 32), constant along the second axis, changes sign between rows
  // 15 and 16: the bisection's halves are rows 0-15 and 16-31, and the cut is the 27 edges between
  // them. Taking the second of two Lanczos pairs, as the Python version did, chose the
  // second-axis vector here. The first axis has an even length in every case, so no entry of the
  // Fiedler vector is zero and the halves are determined.
  for (const std::vector<std::int64_t>& dims :
       {std::vector<std::int64_t>{32, 27}, std::vector<std::int64_t>{10, 8, 7},
        std::vector<std::int64_t>{600}}) {
    CAPTURE(dims.size());
    CAPTURE(dims[0]);
    const pp::CouplingGraph g = grid_graph(dims);
    const pp::GraphPartition r = pp::partition_graph(g, 2);
    const std::int64_t per_slab = g.num_vertices() / dims[0];
    CHECK(splits_first_axis(r.partition.cell_owner(), per_slab, dims[0] / 2));
    CHECK(edge_cut(g, r.partition.cell_owner()) == per_slab);
    CHECK(r.info.lanczos_solves == 1);
    CHECK(r.info.dense_fallbacks == 0);
    CHECK(r.info.dense_solves == 0);
  }
}

TEST_CASE("Spectral: a Lanczos solve that does not converge falls back to the dense solver") {
  const pp::CouplingGraph g = grid_graph({600});
  const pp::Spectral one_restart{pp::dense_fiedler_max, 1};
  const pp::GraphPartition r = one_restart.partition(g, 2);
  CHECK(r.info.dense_fallbacks == 1);
  CHECK(r.info.lanczos_solves == 0);
  CHECK(splits_first_axis(r.partition.cell_owner(), 1, 300));
  CHECK(edge_cut(g, r.partition.cell_owner()) == 1);
}

TEST_CASE("Spectral: the dense limit is inclusive") {
  for (const std::int64_t n : {std::int64_t{512}, std::int64_t{513}}) {
    const pp::GraphPartition r = pp::partition_graph(grid_graph({n}), 2);
    CHECK(r.info.dense_solves == (n <= 512 ? 1 : 0));
    CHECK(r.info.lanczos_solves == (n <= 512 ? 0 : 1));
    CHECK(edge_cut(grid_graph({n}), r.partition.cell_owner()) == 1);
  }
}

TEST_CASE("Spectral: weak bridges between cliques are the cuts") {
  // Four cliques of three (internal weight 10) chained by weight-1 bridges: the optimal 4-way
  // cut is the three bridges, total weight 3.
  Dense a(12, std::vector<std::int64_t>(12, 0));
  for (std::size_t c = 0; c < 4; ++c) {
    for (std::size_t i = 3 * c; i < 3 * c + 3; ++i) {
      for (std::size_t j = 3 * c; j < 3 * c + 3; ++j) {
        a[i][j] = i == j ? 0 : 10;
      }
    }
  }
  for (const auto& [i, j] : {std::pair{2, 3}, std::pair{5, 6}, std::pair{8, 9}}) {
    a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = 1;
    a[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = 1;
  }
  const pp::CouplingGraph g = graph_from_dense(a);
  const pp::GraphPartition r = pp::partition_graph(g, 4);
  const auto owner = r.partition.cell_owner();
  CHECK(edge_cut(g, owner) == 3);
  std::set<std::int32_t> seen(owner.begin(), owner.end());
  CHECK(seen.size() == 4);
  CHECK(r.info.dense_solves == 3);
}

TEST_CASE("Spectral: a disconnected subgraph is ordered by component") {
  Dense a(6, std::vector<std::int64_t>(6, 0));
  for (std::size_t i = 0; i < 6; ++i) {
    for (std::size_t j = 0; j < 6; ++j) {
      if (i != j && (i < 3) == (j < 3)) {
        a[i][j] = 10;
      }
    }
  }
  const pp::GraphPartition r = pp::partition_graph(graph_from_dense(a), 2);
  CHECK(r.info.component_orders == 1);
  CHECK(splits_first_axis(r.partition.cell_owner(), 1, 3));
}

TEST_CASE("Spectral: weights are balanced, not counts") {
  // A 1D degree-2 space of 8 cells, weights 1 x 7 and 7: each half carries 7.
  const bs::BsplineSpace<double> space({open_space<double>(8, 2, 1)});
  const std::vector<double> w{1, 1, 1, 1, 1, 1, 1, 7};
  const pp::GraphPartition r =
      pp::partition_graph(bs::coupling_graph(space, std::span<const double>(w)), 2);
  double part0 = 0.0;
  double part1 = 0.0;
  for (std::size_t c = 0; c < w.size(); ++c) {
    (r.partition.cell_owner()[c] == 0 ? part0 : part1) += w[c];
  }
  CHECK(part0 == 7.0);
  CHECK(part1 == 7.0);
}

TEST_CASE("Spectral: two-way balance is within one vertex weight of the target") {
  // The split takes the first vertex whose running weight reaches half the total, so the first
  // part's weight is in [W/2, W/2 + w_max).
  const bs::BsplineSpace<double> space(
      {open_space<double>(20, 2, 1), open_space<double>(30, 1, 0)});
  std::vector<double> w(600);
  for (std::size_t c = 0; c < w.size(); ++c) {
    w[c] = 1.0 + static_cast<double>((c * 7) % 5);
  }
  const pp::CouplingGraph g = bs::coupling_graph(space, std::span<const double>(w));
  const pp::GraphPartition r = pp::partition_graph(g, 2);
  const auto owner = r.partition.cell_owner();
  const std::int32_t first_part = owner[0];
  double total = 0.0;
  double w_max = 0.0;
  for (const double x : w) {
    total += x;
    w_max = (std::max)(w_max, x);
  }
  double first = 0.0;
  for (std::size_t c = 0; c < w.size(); ++c) {
    if (owner[c] == first_part) {
      first += w[c];
    }
  }
  // The first half in the Fiedler order holds [W/2, W/2 + w_max), so the other holds
  // (W/2 - w_max, W/2]: either way within w_max of W/2.
  CHECK(std::abs(first - total / 2) < w_max);
  CHECK(r.info.lanczos_solves == 1);
}

TEST_CASE("Spectral: every part is non-empty, including n_parts == number of vertices") {
  const bs::BsplineSpace<double> space({open_space<double>(5, 2, 1)});
  const pp::GraphPartition r = pp::partition_graph(bs::coupling_graph(space), 5);
  std::vector<std::int32_t> owner(r.partition.cell_owner().begin(), r.partition.cell_owner().end());
  std::sort(owner.begin(), owner.end());
  CHECK(owner == std::vector<std::int32_t>{0, 1, 2, 3, 4});
}

TEST_CASE("Spectral: inactive vertices get owner -1 and are dropped") {
  const bs::BsplineSpace<double> space({open_space<double>(8, 2, 1)});
  const std::vector<std::uint8_t> active{1, 1, 1, 1, 0, 0, 0, 0};
  const pp::GraphPartition r =
      pp::partition_graph(bs::coupling_graph(space), 2, std::span<const std::uint8_t>(active));
  const auto owner = r.partition.cell_owner();
  for (std::size_t c = 4; c < 8; ++c) {
    CHECK(owner[c] == -1);
  }
  CHECK(std::set<std::int32_t>(owner.begin(), owner.begin() + 4) == std::set<std::int32_t>{0, 1});
}

TEST_CASE("Spectral: the partition does not depend on the thread count") {
  const std::size_t many = (std::min)(std::size_t{8}, pantr::parallel::max_threads());
  const bs::BsplineSpace<double> space(
      {open_space<double>(30, 2, 1), open_space<double>(25, 2, 1)});
  const pp::CouplingGraph g = bs::coupling_graph(space);
  const std::size_t before = pantr::parallel::get_num_threads();
  pantr::parallel::set_num_threads(1);
  const pp::GraphPartition one = pp::partition_graph(g, 7);
  pantr::parallel::set_num_threads(static_cast<std::int64_t>(many));
  const pp::GraphPartition more = pp::partition_graph(g, 7);
  pantr::parallel::set_num_threads(static_cast<std::int64_t>(before));
  CHECK(std::vector<std::int32_t>(one.partition.cell_owner().begin(),
                                  one.partition.cell_owner().end()) ==
        std::vector<std::int32_t>(more.partition.cell_owner().begin(),
                                  more.partition.cell_owner().end()));
}

TEST_CASE("Spectral: unusual graphs (zero weights, self loops, repeated entries)") {
  using V = std::vector<std::int64_t>;
  // 0-1 joined by a zero-weight entry, 2-3 by a self loop on 2 and a repeated entry: the zero
  // weight still joins 0 and 1 for connectivity, so each pair is a component.
  const pp::CouplingGraph g(4, V{0, 1, 2, 5, 6}, V{1, 0, 2, 3, 3, 2}, V{0, 0, 5, 1, 2, 3},
                            std::vector<double>(4, 1.0));
  const pp::GraphPartition r = pp::partition_graph(g, 2);
  const auto owner = r.partition.cell_owner();
  CHECK(owner[0] == owner[1]);
  CHECK(owner[2] == owner[3]);
  CHECK(owner[0] != owner[2]);
  CHECK(r.info.component_orders == 1);
  // A connected graph whose Laplacian is zero: any order is a Fiedler order; it must not fail.
  const pp::CouplingGraph flat(3, V{0, 1, 3, 4}, V{1, 0, 2, 1}, V{0, 0, 0, 0},
                               std::vector<double>(3, 1.0));
  const pp::GraphPartition rf = pp::partition_graph(flat, 3);
  CHECK(std::set<std::int32_t>(rf.partition.cell_owner().begin(),
                               rf.partition.cell_owner().end()) == std::set<std::int32_t>{0, 1, 2});
}

TEST_CASE("partition_graph: refusals, in order") {
  const bs::BsplineSpace<double> space({open_space<double>(4, 2, 1)});
  const pp::CouplingGraph g = bs::coupling_graph(space);
  CHECK(message_of([&] { static_cast<void>(pp::partition_graph(g, 0)); }) ==
        "n_parts must be >= 1; got 0.");
  CHECK(message_of([&] { static_cast<void>(pp::partition_graph(g, 5)); }) ==
        "n_parts=5 exceeds the number of active cells (4); cannot assign every rank a cell.");
  const std::vector<std::uint8_t> two{1, 0};
  CHECK(message_of([&] {
          static_cast<void>(pp::partition_graph(g, 2, std::span<const std::uint8_t>(two)));
        }) == "cell_active must have shape (4,); got (2,).");
  const std::vector<std::uint8_t> none(4, 0);
  CHECK(message_of([&] {
          static_cast<void>(pp::partition_graph(g, 2, std::span<const std::uint8_t>(none)));
        }) == "cell_active must mark at least one cell active.");
  const std::vector<std::uint8_t> one{1, 0, 0, 0};
  CHECK(message_of([&] {
          static_cast<void>(pp::partition_graph(g, 2, std::span<const std::uint8_t>(one)));
        }) == "n_parts=2 exceeds the number of active cells (1); cannot assign every rank a cell.");
  const pp::Spectral bad{pp::dense_fiedler_max, 0};
  CHECK(message_of([&] { static_cast<void>(bad.partition(g, 2)); }) ==
        "lanczos_max_iter must be >= 1; got 0.");
  const pp::GraphPartition single = pp::partition_graph(g, 1);
  for (const auto o : single.partition.cell_owner()) {
    CHECK(o == 0);
  }
}

TEST_CASE("select_active: the subgraph keeps the active edges, renumbered and ascending") {
  using V = std::vector<std::int64_t>;
  // A 4-cycle 0-1-2-3-0 with each row listed descending; drop vertex 1.
  const pp::CouplingGraph g(4, V{0, 2, 4, 6, 8}, V{3, 1, 2, 0, 3, 1, 2, 0},
                            V{4, 1, 2, 1, 3, 2, 3, 4}, {1.0, 2.0, 3.0, 4.0});
  const std::vector<std::uint8_t> active{1, 0, 1, 1};
  const auto sub = pp::detail::select_active(g, 2, std::span<const std::uint8_t>(active));
  CHECK(sub.active_idx == V{0, 2, 3});
  CHECK(sub.xadj == V{0, 1, 2, 4});
  CHECK(sub.adjncy == V{2, 2, 0, 1});
  CHECK(sub.edge_weights == V{4, 3, 4, 3});
  CHECK(sub.vertex_weights == std::vector<double>{1.0, 3.0, 4.0});
}

TEST_CASE("Spectral: a connected graph of zero weights above the dense limit falls back") {
  // The deflation shift is the largest absolute row sum of the Laplacian, zero here, so the
  // Lanczos operator is zero and Spectra gives up (it throws); the dense route answers instead.
  pp::CouplingGraph path = grid_graph({700});
  const pp::CouplingGraph flat(
      700, std::vector<std::int64_t>(path.xadj().begin(), path.xadj().end()),
      std::vector<std::int64_t>(path.adjncy().begin(), path.adjncy().end()),
      std::vector<std::int64_t>(path.adjncy().size(), 0), std::vector<double>(700, 1.0));
  const pp::GraphPartition r = pp::partition_graph(flat, 2);
  CHECK(r.info.dense_fallbacks == 1);
  CHECK(r.info.lanczos_solves == 0);
  std::set<std::int32_t> parts(r.partition.cell_owner().begin(), r.partition.cell_owner().end());
  CHECK(parts == std::set<std::int32_t>{0, 1});
}

TEST_CASE("Spectral: self loops are dropped and repeated entries summed, on both solver paths") {
  // The Laplacian of a graph with each edge weight split into two stored entries and a self loop
  // on every vertex equals the clean graph's (scipy's csgraph.laplacian ignores the diagonal and
  // sums duplicates), so the partitions agree, below and above the dense limit.
  for (const std::vector<std::int64_t>& dims :
       {std::vector<std::int64_t>{20, 13}, std::vector<std::int64_t>{30, 21}}) {
    const pp::CouplingGraph clean = grid_graph(dims);
    const auto n = static_cast<std::size_t>(clean.num_vertices());
    std::vector<std::int64_t> xadj{0};
    std::vector<std::int64_t> adjncy;
    std::vector<std::int64_t> ew;
    for (std::size_t v = 0; v < n; ++v) {
      adjncy.push_back(static_cast<std::int64_t>(v));
      ew.push_back(5);
      for (auto k = static_cast<std::size_t>(clean.xadj()[v]);
           k < static_cast<std::size_t>(clean.xadj()[v + 1]); ++k) {
        // Weight 3 split into 1 + 2 (the clean graph's weight is 1: scale it to 3 on both).
        adjncy.push_back(clean.adjncy()[k]);
        ew.push_back(1);
        adjncy.push_back(clean.adjncy()[k]);
        ew.push_back(2);
      }
      xadj.push_back(static_cast<std::int64_t>(adjncy.size()));
    }
    const pp::CouplingGraph messy(static_cast<std::int64_t>(n), xadj, adjncy, ew,
                                  std::vector<double>(n, 1.0));
    std::vector<std::int64_t> w3(clean.adjncy().size(), 3);
    const pp::CouplingGraph scaled(
        static_cast<std::int64_t>(n), {clean.xadj().begin(), clean.xadj().end()},
        {clean.adjncy().begin(), clean.adjncy().end()}, w3, std::vector<double>(n, 1.0));
    const pp::GraphPartition a = pp::partition_graph(messy, 2);
    const pp::GraphPartition b = pp::partition_graph(scaled, 2);
    CAPTURE(n);
    // Equal Laplacians, rounded differently in the products: the halves agree, and on these grids
    // (symmetric under reversing the vertex order, so the sign rule ties) maybe not their labels.
    std::vector<std::int32_t> oa(a.partition.cell_owner().begin(), a.partition.cell_owner().end());
    std::vector<std::int32_t> ob(b.partition.cell_owner().begin(), b.partition.cell_owner().end());
    if (oa[0] != ob[0]) {
      for (auto& o : ob) {
        o = 1 - o;
      }
    }
    CHECK(oa == ob);
    CHECK(a.info.lanczos_solves == (n > 512 ? 1 : 0));
  }
}

TEST_CASE("dense_fiedler: agrees with the full symmetric eigensolver's second eigenvector") {
  // Weighted random connected graphs (a path plus random chords, weights 1 to 9). Oracle: the
  // second column of Eigen::SelfAdjointEigenSolver on the same matrix. Both are backward stable,
  // each eigenvector within O(m eps ||L|| / gap) of the exact one (Davis-Kahan); the bound below
  // takes 10 m eps ||L|| / gap, the factor 10 an admitted heuristic margin on the constant.
  std::uint64_t state = 12345;
  auto next = [&state] {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return state >> 33;
  };
  for (const std::size_t m : {std::size_t{2}, std::size_t{7}, std::size_t{60}, std::size_t{300}}) {
    pp::detail::Laplacian lap;
    std::vector<std::vector<double>> w(m, std::vector<double>(m, 0.0));
    for (std::size_t i = 0; i + 1 < m; ++i) {
      w[i][i + 1] = w[i + 1][i] = static_cast<double>(1 + next() % 9);
    }
    for (std::size_t c = 0; c < m; ++c) {
      const std::size_t i = next() % m;
      const std::size_t j = next() % m;
      if (i != j) {
        w[i][j] = w[j][i] = static_cast<double>(1 + next() % 9);
      }
    }
    lap.ptr.assign(m + 1, 0);
    lap.diag.assign(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = 0; j < m; ++j) {
        if (i != j && w[i][j] != 0.0) {
          lap.col.push_back(j);
          lap.val.push_back(-w[i][j]);
          lap.diag[j] += w[i][j];
        }
      }
      lap.ptr[i + 1] = lap.col.size();
    }
    const Eigen::MatrixXd dense = lap.dense();
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> full(dense);
    const double gap = (std::min)(full.eigenvalues()(1) - full.eigenvalues()(0),
                                  m > 2 ? full.eigenvalues()(2) - full.eigenvalues()(1)
                                        : std::numeric_limits<double>::infinity());
    REQUIRE(gap > 0.0);
    const double norm = full.eigenvalues().cwiseAbs().maxCoeff();
    const Eigen::VectorXd fast = pp::detail::dense_fiedler(lap);
    // The distance to the nearer of +-v is 2 sin(theta / 2) >= sin(theta) for the angle theta
    // between them; it is measured directly, since sqrt(1 - cos^2) cannot resolve an angle below
    // sqrt(eps).
    const Eigen::VectorXd v = full.eigenvectors().col(1);
    const double distance = (std::min)((fast - v).norm(), (fast + v).norm());
    const double eps = std::numeric_limits<double>::epsilon();
    CAPTURE(m);
    // Normalized, then mapped by an orthogonal Q applied as m Householder reflections.
    CHECK(std::abs(fast.norm() - 1.0) <= 8.0 * static_cast<double>(m) * eps);
    CHECK(distance <= 10.0 * static_cast<double>(m) * eps * norm / gap);
  }
}

// Tests added after the test review: each kills a mutant the suite above let through.
namespace {

pp::detail::Laplacian laplacian_of(const pp::CouplingGraph& g) {
  const auto n = static_cast<std::size_t>(g.num_vertices());
  pp::detail::Laplacian lap;
  lap.ptr.assign(n + 1, 0);
  lap.diag.assign(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (auto k = static_cast<std::size_t>(g.xadj()[i]);
         k < static_cast<std::size_t>(g.xadj()[i + 1]); ++k) {
      const auto j = static_cast<std::size_t>(g.adjncy()[k]);
      const double w = static_cast<double>(g.edge_weights()[k]);
      lap.col.push_back(j);
      lap.val.push_back(-w);
      lap.diag[j] += w;
    }
    lap.ptr[i + 1] = lap.col.size();
  }
  return lap;
}

std::vector<std::int32_t> owners(const pp::GraphPartition& r) {
  return {r.partition.cell_owner().begin(), r.partition.cell_owner().end()};
}

}  // namespace

TEST_CASE(
    "Spectral probes: sign rule: largest-magnitude Fiedler entry positive (weighted 3-path)") {
  // w01 = 1, w12 = 2: lambda_2 = 3 - sqrt(3), v = (1, sqrt(3)-2, 1-sqrt(3)) up to scale. Largest
  // magnitude at vertex 0 -> positive -> order 2, 1, 0 -> first half {2, 1}.
  Dense a(3, std::vector<std::int64_t>(3, 0));
  a[0][1] = a[1][0] = 1;
  a[1][2] = a[2][1] = 2;
  const pp::CouplingGraph g = graph_from_dense(a);
  CHECK(owners(pp::partition_graph(g, 2)) == std::vector<std::int32_t>{1, 0, 0});
  const pp::Spectral lanczos{2, std::nullopt};  // m = 3 > dense_max: the Lanczos route
  const auto r = lanczos.partition(g, 2);
  CHECK(r.info.lanczos_solves == 1);
  CHECK(owners(r) == std::vector<std::int32_t>{1, 0, 0});
}

TEST_CASE(
    "Spectral probes: component order: smallest-vertex labels, order kept inside, split "
    "mid-component") {
  Dense a(7, std::vector<std::int64_t>(7, 0));
  for (const auto& [i, j] : {std::pair{0, 4}, std::pair{1, 2}, std::pair{2, 5}, std::pair{3, 6}}) {
    a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = 3;
    a[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = 3;
  }
  const auto r = pp::partition_graph(graph_from_dense(a), 2);
  CHECK(r.info.component_orders == 1);
  CHECK(owners(r) == std::vector<std::int32_t>{0, 0, 0, 1, 0, 1, 1});
}

TEST_CASE("Spectral probes: mask that disconnects the graph: components of the active subgraph") {
  const pp::CouplingGraph g = grid_graph({7});
  const std::vector<std::uint8_t> active{1, 1, 1, 0, 1, 1, 1};
  const auto r = pp::partition_graph(g, 2, std::span<const std::uint8_t>(active));
  CHECK(r.info.component_orders == 1);
  CHECK(owners(r) == std::vector<std::int32_t>{0, 0, 0, -1, 1, 1, 1});
}

TEST_CASE(
    "Spectral probes: interleaved mask equals the partition of the densely induced subgraph") {
  const pp::CouplingGraph g = grid_graph({5, 6});
  const std::vector<std::uint8_t> active{1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1,
                                         0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1};
  const Dense full = dense_of(g);
  std::vector<std::size_t> keep;
  for (std::size_t v = 0; v < active.size(); ++v) {
    if (active[v] != 0) {
      keep.push_back(v);
    }
  }
  Dense sub(keep.size(), std::vector<std::int64_t>(keep.size(), 0));
  for (std::size_t i = 0; i < keep.size(); ++i) {
    for (std::size_t j = 0; j < keep.size(); ++j) {
      sub[i][j] = full[keep[i]][keep[j]];
    }
  }
  for (const std::int64_t parts : {2, 3, 5}) {
    const auto masked =
        owners(pp::partition_graph(g, parts, std::span<const std::uint8_t>(active)));
    const auto direct = owners(pp::partition_graph(graph_from_dense(sub), parts));
    std::vector<std::int32_t> expect(active.size(), -1);
    for (std::size_t i = 0; i < keep.size(); ++i) {
      expect[keep[i]] = direct[i];
    }
    CAPTURE(parts);
    CHECK(masked == expect);
  }
}

TEST_CASE("Spectral probes: k-way sizes with unit weights are floor(n/k) or ceil(n/k)") {
  for (const std::vector<std::int64_t>& dims :
       {std::vector<std::int64_t>{30, 25}, std::vector<std::int64_t>{20, 13}}) {
    const pp::CouplingGraph g = grid_graph(dims);
    const std::int64_t n = g.num_vertices();
    for (const std::int64_t k : {3, 5, 6, 7, 9, 16}) {
      const auto o = owners(pp::partition_graph(g, k));
      std::vector<std::int64_t> size(static_cast<std::size_t>(k), 0);
      for (const auto v : o) {
        ++size[static_cast<std::size_t>(v)];
      }
      CAPTURE(n);
      CAPTURE(k);
      for (const auto s : size) {
        CHECK((s == n / k || s == (n + k - 1) / k));
      }
    }
  }
}

TEST_CASE("Spectral probes: from_incidence: cell weights pass through and are checked") {
  const std::vector<std::int64_t> ptr{0, 2, 3, 5};
  const std::vector<std::int64_t> dofs{0, 1, 1, 1, 2};
  const std::vector<double> w{2.0, 0.0, 3.5};
  const auto g = pp::coupling_graph_from_incidence(3, 3, ptr, dofs, std::span<const double>(w));
  CHECK(std::vector<double>(g.vertex_weights().begin(), g.vertex_weights().end()) == w);
  const std::vector<double> bad{1.0, 2.0};
  CHECK(message_of([&] {
          static_cast<void>(
              pp::coupling_graph_from_incidence(3, 3, ptr, dofs, std::span<const double>(bad)));
        }) == "cell_weights must have shape (3,); got (2,).");
  const std::vector<double> nan{1.0, std::numeric_limits<double>::quiet_NaN(), 1.0};
  CHECK(message_of([&] {
          static_cast<void>(
              pp::coupling_graph_from_incidence(3, 3, ptr, dofs, std::span<const double>(nan)));
        }) == "cell_weights must be finite and non-negative.");
}

TEST_CASE(
    "Spectral probes: from_incidence: each malformed-pointer / id condition alone is refused") {
  const std::vector<std::int64_t> dofs3{0, 1, 1};
  // Non-monotone, but starts at 0 and ends at dofs.size(): only the monotone check refuses it.
  const std::vector<std::int64_t> nonmono{0, 2, 1, 3};
  CHECK(message_of([&] {
          static_cast<void>(pp::coupling_graph_from_incidence(3, 2, nonmono, dofs3));
        }).find("cell_ptr must have") != std::string::npos);
  // Sorted, ends at dofs.size(), but starts at 1: only the front check refuses it.
  const std::vector<std::int64_t> front{1, 2, 3};
  CHECK(message_of([&] {
          static_cast<void>(pp::coupling_graph_from_incidence(2, 2, front, dofs3));
        }).find("cell_ptr must have") != std::string::npos);
  // Sorted, starts at 0, ends short of dofs.size(): only the back check refuses it.
  const std::vector<std::int64_t> back{0, 1, 2};
  CHECK(message_of([&] {
          static_cast<void>(pp::coupling_graph_from_incidence(2, 2, back, dofs3));
        }).find("cell_ptr must have") != std::string::npos);
  const std::vector<std::int64_t> ok{0, 1, 3};
  const std::vector<std::int64_t> neg{0, -1, 1};
  CHECK(message_of([&] { static_cast<void>(pp::coupling_graph_from_incidence(2, 2, ok, neg)); }) ==
        "dof ids must lie in [0, 2).");
  CHECK(message_of([&] {
          static_cast<void>(pp::coupling_graph_from_incidence(2, -1, ok, dofs3));
        }).find("must be >= 0") != std::string::npos);
}

TEST_CASE("Spectral probes: zero-cell graph") {
  const std::vector<std::int64_t> ptr{0};
  const auto g = pp::coupling_graph_from_incidence(0, 0, ptr, {});
  CHECK(g.num_vertices() == 0);
  CHECK(message_of([&] { static_cast<void>(pp::partition_graph(g, 1)); }) ==
        "n_parts=1 exceeds the number of active cells (0); cannot assign every rank a cell.");
}

TEST_CASE(
    "Spectral probes: dense_fiedler on repeated lambda_2: residual, orthogonality, unit norm") {
  struct Case {
    const char* name;
    pp::CouplingGraph g;
    double lambda2;
  };
  Dense star(7, std::vector<std::int64_t>(7, 0));
  for (std::size_t i = 1; i < 7; ++i) {
    star[0][i] = star[i][0] = 1;
  }
  Dense k8(8, std::vector<std::int64_t>(8, 0));
  for (std::size_t i = 0; i < 8; ++i) {
    for (std::size_t j = 0; j < 8; ++j) {
      k8[i][j] = i == j ? 0 : 1;
    }
  }
  const double pi = std::acos(-1.0);
  std::vector<Case> cases;
  cases.push_back({"star7 (lambda2=1, mult 5)", graph_from_dense(star), 1.0});
  cases.push_back({"K8 (lambda2=8, mult 7)", graph_from_dense(k8), 8.0});
  cases.push_back({"grid 8x8 (mult 2)", grid_graph({8, 8}), 2.0 - 2.0 * std::cos(pi / 8.0)});
  cases.push_back({"grid 20x20 (mult 2)", grid_graph({20, 20}), 2.0 - 2.0 * std::cos(pi / 20.0)});
  for (const auto& c : cases) {
    const pp::detail::Laplacian lap = laplacian_of(c.g);
    const Eigen::VectorXd v = pp::detail::dense_fiedler(lap);
    const Eigen::MatrixXd dense = lap.dense();
    const double m = static_cast<double>(v.size());
    const double eps = std::numeric_limits<double>::epsilon();
    const double norm = 2.0 * dense.diagonal().maxCoeff();  // Gershgorin bound on ||L||
    CAPTURE(c.name);
    CHECK(std::abs(v.norm() - 1.0) <= 8.0 * m * eps);
    CHECK(std::abs(v.sum()) <= 10.0 * m * eps * std::sqrt(m));
    CHECK((dense * v - c.lambda2 * v).norm() <= 10.0 * m * eps * norm);
  }
}

TEST_CASE("Spectral probes: shifted_tridiagonal_solve against a known solution") {
  struct Sys {
    const char* name;
    std::vector<double> diag, off;
    double sigma;
  };
  const std::vector<Sys> systems = {
      {"swap everywhere", {0.1, 0.2, 0.1, 0.3, 0.2, 0.1}, {1.0, 2.0, 1.5, 1.0, 2.5}, 0.05},
      {"no swap (dominant)", {5.0, 6.0, 7.0, 8.0, 9.0}, {1.0, 1.0, 1.0, 1.0}, 0.5},
      {"swap only on the last row", {5.0, 1.0, 0.1}, {1.0, 3.0}, 0.0},
      {"swap on the first row only", {0.1, 5.0, 5.0}, {3.0, 1.0}, 0.0},
      {"n = 2 with a swap", {0.1, 0.2}, {1.0}, 0.0},
      {"n = 2 no swap", {3.0, 0.2}, {1.0}, 0.0},
  };
  for (const auto& s : systems) {
    const auto n = static_cast<Eigen::Index>(s.diag.size());
    Eigen::VectorXd d = Eigen::Map<const Eigen::VectorXd>(s.diag.data(), n);
    Eigen::VectorXd o = Eigen::Map<const Eigen::VectorXd>(s.off.data(), n - 1);
    Eigen::MatrixXd t = Eigen::MatrixXd::Zero(n, n);
    for (Eigen::Index i = 0; i < n; ++i) {
      t(i, i) = d(i) - s.sigma;
    }
    for (Eigen::Index i = 0; i + 1 < n; ++i) {
      t(i, i + 1) = t(i + 1, i) = o(i);
    }
    Eigen::VectorXd y(n);
    for (Eigen::Index i = 0; i < n; ++i) {
      y(i) = (i % 2 == 0 ? 1.0 : -1.0) * (1.0 + 0.5 * static_cast<double>(i));
    }
    Eigen::VectorXd x = t * y;
    pp::detail::shifted_tridiagonal_solve(d, o, s.sigma, 1e-300, x);
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(t);
    const double kappa = svd.singularValues()(0) / svd.singularValues()(n - 1);
    const double eps = std::numeric_limits<double>::epsilon();
    CAPTURE(s.name);
    CHECK((x - y).norm() <= 10.0 * static_cast<double>(n) * eps * kappa * y.norm());
  }
}

TEST_CASE("Spectral probes: dense_fiedler on a zero Laplacian is finite and unit") {
  pp::detail::Laplacian lap;
  lap.ptr = {0, 1, 3, 4};
  lap.col = {1, 0, 2, 1};
  lap.val = {0.0, 0.0, 0.0, 0.0};
  lap.diag = {0.0, 0.0, 0.0};
  const Eigen::VectorXd v = pp::detail::dense_fiedler(lap);
  CHECK(v.allFinite());
  CHECK(std::abs(v.norm() - 1.0) <= 1e-14);
}

TEST_CASE(
    "Spectral probes: tensor graph: no stored zero weight, brute force on discontinuous/one-cell "
    "spaces") {
  std::vector<bs::BsplineSpace<double>> spaces;
  spaces.push_back(bs::BsplineSpace<double>({open_space<double>(3, 2, -1)}));
  spaces.push_back(
      bs::BsplineSpace<double>({open_space<double>(1, 2, 1), open_space<double>(3, 1, 0)}));
  spaces.push_back(
      bs::BsplineSpace<double>({open_space<double>(2, 2, -1), open_space<double>(3, 2, 1)}));
  spaces.push_back(bs::BsplineSpace<double>(
      {open_space<double>(6, 2, 1), open_space<double>(5, 3, 2), open_space<double>(1, 1, 0)}));
  const std::vector<double> k1{0, 0, 0, 0.3, 0.3, 0.3, 1, 1, 1};
  spaces.push_back(bs::BsplineSpace<double>({bs::BsplineSpace1D<double>(k1, 2)}));
  const std::vector<double> k2{0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1};
  spaces.push_back(bs::BsplineSpace<double>({bs::BsplineSpace1D<double>(k2, 3)}));
  std::size_t idx = 0;
  for (const auto& space : spaces) {
    const pp::CouplingGraph g = bs::coupling_graph(space);
    CAPTURE(idx++);
    CHECK(dense_of(g) == brute_force(space));
    for (const auto w : g.edge_weights()) {
      CHECK(w >= 1);
    }
  }
}

TEST_CASE("Spectral probes: incidence: an empty cell between two coupled cells") {
  const std::vector<std::int64_t> ptr{0, 1, 1, 2};
  const std::vector<std::int64_t> dofs{0, 0};
  const auto g = pp::coupling_graph_from_incidence(3, 1, ptr, dofs);
  CHECK(dense_of(g) == Dense{{0, 0, 1}, {0, 0, 0}, {1, 0, 0}});
  CHECK(std::vector<std::int64_t>(g.xadj().begin(), g.xadj().end()) ==
        std::vector<std::int64_t>{0, 1, 1, 2});
}

TEST_CASE("dense_fiedler: a chain of three cliques, whose tridiagonal form has a repeated block") {
  // Regression: three 3-cliques (weight 10) joined by weight-1 bridges 2-3 and 5-6. The Laplacian
  // has the eigenvalue 30 three times, so its tridiagonal form ends in a block whose off-diagonal
  // entries are at rounding level; the tridiagonal QR then failed to converge on the unscaled
  // matrix in a build without FMA, and the unconverged, unsorted values gave a shift that is not
  // lambda_2 (the vector found belonged to lambda_3). Oracle: Eigen's full symmetric solver on the
  // same matrix. The Rayleigh quotient of a unit vector within distance delta of the eigenvector
  // is within ||L|| delta^2 + O(eps ||L||) of lambda_2; the bound takes 10 m eps ||L|| (the factor
  // an admitted margin), and the distance bound is the one of the test above.
  const std::size_t m = 9;
  pp::detail::Laplacian lap;
  std::vector<std::vector<double>> w(m, std::vector<double>(m, 0.0));
  for (std::size_t c = 0; c < 3; ++c) {
    for (std::size_t i = 3 * c; i < 3 * c + 3; ++i) {
      for (std::size_t j = 3 * c; j < 3 * c + 3; ++j) {
        w[i][j] = i == j ? 0.0 : 10.0;
      }
    }
  }
  w[2][3] = w[3][2] = 1.0;
  w[5][6] = w[6][5] = 1.0;
  lap.ptr.assign(m + 1, 0);
  lap.diag.assign(m, 0.0);
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      if (i != j && w[i][j] != 0.0) {
        lap.col.push_back(j);
        lap.val.push_back(-w[i][j]);
        lap.diag[j] += w[i][j];
      }
    }
    lap.ptr[i + 1] = lap.col.size();
  }
  const Eigen::MatrixXd dense = lap.dense();
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> full(dense);
  REQUIRE(full.info() == Eigen::Success);
  const double lambda2 = full.eigenvalues()(1);
  const double gap = (std::min)(lambda2 - full.eigenvalues()(0), full.eigenvalues()(2) - lambda2);
  const double norm = full.eigenvalues().cwiseAbs().maxCoeff();
  const double eps = std::numeric_limits<double>::epsilon();
  const Eigen::VectorXd fast = pp::detail::dense_fiedler(lap);
  const double rayleigh = fast.dot(dense * fast);
  const Eigen::VectorXd v = full.eigenvectors().col(1);
  const double distance = (std::min)((fast - v).norm(), (fast + v).norm());
  CHECK(std::abs(rayleigh - lambda2) <= 10.0 * static_cast<double>(m) * eps * norm);
  CHECK(distance <= 10.0 * static_cast<double>(m) * eps * norm / gap);
  // And the partition the Python suite checks: each clique is one part.
  const pp::GraphPartition r =
      pp::partition_graph(graph_from_dense(Dense{{0, 10, 10, 0, 0, 0, 0, 0, 0},
                                                 {10, 0, 10, 0, 0, 0, 0, 0, 0},
                                                 {10, 10, 0, 1, 0, 0, 0, 0, 0},
                                                 {0, 0, 1, 0, 10, 10, 0, 0, 0},
                                                 {0, 0, 0, 10, 0, 10, 0, 0, 0},
                                                 {0, 0, 0, 10, 10, 0, 1, 0, 0},
                                                 {0, 0, 0, 0, 0, 1, 0, 10, 10},
                                                 {0, 0, 0, 0, 0, 0, 10, 0, 10},
                                                 {0, 0, 0, 0, 0, 0, 10, 10, 0}}),
                          3);
  const auto owner = r.partition.cell_owner();
  for (std::size_t c = 0; c < 3; ++c) {
    CHECK(owner[3 * c] == owner[3 * c + 1]);
    CHECK(owner[3 * c + 1] == owner[3 * c + 2]);
  }
  CHECK(std::set<std::int32_t>(owner.begin(), owner.end()) == std::set<std::int32_t>{0, 1, 2});
}
