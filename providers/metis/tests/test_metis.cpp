// The METIS provider: a GraphPartitioner over the user's METIS. Oracles: graphs whose minimum
// cut is evident (cliques joined by weak bridges), the validity of every owner, the refusals the
// Python backend raises, and the cut summed in the test.
#include <doctest/doctest.h>

#include <pantr/bspline/coupling_graph.hpp>
#include <pantr/bspline/knot_factories.hpp>
#include <pantr/bspline/space.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/partition/partition_graph.hpp>
#include <pantr/providers/metis/partitioner.hpp>

#include <cstddef>
#include <cstdint>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pp = pantr::partition;
namespace bs = pantr::bspline;

namespace {

/// Two cliques of three (internal weight 10) joined by one weight-1 bridge between 2 and 3.
pp::CouplingGraph two_cliques(std::vector<double> weights = std::vector<double>(6, 1.0)) {
  std::vector<std::int64_t> xadj{0};
  std::vector<std::int64_t> adjncy;
  std::vector<std::int64_t> ew;
  for (std::int64_t i = 0; i < 6; ++i) {
    for (std::int64_t j = 0; j < 6; ++j) {
      const bool same = (i < 3) == (j < 3);
      const bool bridge = (i == 2 && j == 3) || (i == 3 && j == 2);
      if (i != j && (same || bridge)) {
        adjncy.push_back(j);
        ew.push_back(bridge ? 1 : 10);
      }
    }
    xadj.push_back(static_cast<std::int64_t>(adjncy.size()));
  }
  return {6, xadj, adjncy, ew, std::move(weights)};
}

bs::BsplineSpace<double> space_2d(long long n) {
  const auto k = bs::uniform_open_knots<double>(n, 2, 1, 0.0, 1.0);
  const bs::BsplineSpace1D<double> d(std::span<const double>(k), 2);
  return bs::BsplineSpace<double>({d, d});
}

}  // namespace

TEST_CASE("metis: the weak bridge is the cut") {
  const pp::GraphPartition r =
      pp::partition_graph(two_cliques(), 2, std::nullopt, pantr::metis::Partitioner{});
  const auto o = r.partition.cell_owner();
  CHECK(o[0] == o[1]);
  CHECK(o[1] == o[2]);
  CHECK(o[3] == o[4]);
  CHECK(o[4] == o[5]);
  CHECK(o[0] != o[3]);
  CHECK(r.info.dense_solves + r.info.lanczos_solves + r.info.dense_fallbacks +
            r.info.component_orders ==
        0);
}

TEST_CASE("metis: zero vertex weights are clamped to one") {
  const pp::GraphPartition r =
      pantr::metis::Partitioner{}.partition(two_cliques({1, 1, 1, 1, 1, 0}), 2);
  for (const auto o : r.partition.cell_owner()) {
    CHECK((o == 0 || o == 1));
  }
}

TEST_CASE("metis: recursive (<= 8 parts) and k-way (> 8 parts) give valid owners") {
  const pp::CouplingGraph g = bs::coupling_graph(space_2d(12));
  for (const std::int64_t parts : {std::int64_t{4}, std::int64_t{12}}) {
    const pp::GraphPartition r = pantr::metis::Partitioner{}.partition(g, parts);
    for (const auto o : r.partition.cell_owner()) {
      CHECK(o >= 0);
      CHECK(o < parts);
    }
    CHECK(r.partition.n_parts() == parts);
  }
}

TEST_CASE("metis: inactive vertices get -1, one part is all zeros") {
  const pp::CouplingGraph g = bs::coupling_graph(space_2d(4));
  std::vector<std::uint8_t> active(16, 1);
  active[0] = active[5] = 0;
  const pp::GraphPartition r =
      pantr::metis::Partitioner{}.partition(g, 3, std::span<const std::uint8_t>(active));
  CHECK(r.partition.cell_owner()[0] == -1);
  CHECK(r.partition.cell_owner()[5] == -1);
  CHECK(r.partition.cell_owner()[1] >= 0);
  const pp::GraphPartition one = pantr::metis::Partitioner{}.partition(g, 1);
  for (const auto o : one.partition.cell_owner()) {
    CHECK(o == 0);
  }
}

TEST_CASE("metis: refusals") {
  using V = std::vector<std::int64_t>;
  const pp::CouplingGraph zero(2, V{0, 1, 2}, V{1, 0}, V{0, 0}, {1.0, 1.0});
  std::string message;
  try {
    static_cast<void>(pantr::metis::Partitioner{}.partition(zero, 2));
  } catch (const pantr::value_error& e) {
    message = e.what();
  }
  CHECK(message ==
        "METIS requires positive integer edge weights; after rounding, 2 edge(s) have weight "
        "<= 0. Scale up the graph edge weights or use backend='spectral'.");
  CHECK_THROWS_AS(static_cast<void>(pantr::metis::Partitioner{}.partition(zero, 3)),
                  pantr::value_error);
}

namespace {
std::vector<std::int32_t> mowners(const pp::GraphPartition& r) {
  return {r.partition.cell_owner().begin(), r.partition.cell_owner().end()};
}
}  // namespace

TEST_CASE("metis: edge weights steer the cut: weighted 4-cycle") {
  // 0-1 (1), 1-2 (10), 2-3 (1), 3-0 (10): balanced optimum {0,3} | {1,2}, cut weight 2.
  using V = std::vector<std::int64_t>;
  const pp::CouplingGraph g(4, V{0, 2, 4, 6, 8}, V{1, 3, 0, 2, 1, 3, 2, 0},
                            V{1, 10, 1, 10, 10, 1, 1, 10}, std::vector<double>(4, 1.0));
  const auto o = mowners(pantr::metis::Partitioner{}.partition(g, 2));
  CHECK(o[0] == o[3]);
  CHECK(o[1] == o[2]);
  CHECK(o[0] != o[1]);
}

TEST_CASE("metis: zero vertex weights are clamped to one (observable: balance)") {
  const auto o = mowners(pantr::metis::Partitioner{}.partition(two_cliques({0, 0, 0, 1, 1, 1}), 2));
  CHECK(o[0] == o[1]);
  CHECK(o[1] == o[2]);
  CHECK(o[3] == o[4]);
  CHECK(o[4] == o[5]);
  CHECK(o[0] != o[3]);
}

TEST_CASE("metis: a vertex weight past the index type is a capacity_error") {
  CHECK_THROWS_AS(static_cast<void>(pantr::metis::Partitioner{}.partition(
                      two_cliques({1, 1, 1, 1, 1, 3.0e9}), 2)),
                  pantr::capacity_error);
  CHECK_THROWS_AS(static_cast<void>(
                      pantr::metis::Partitioner{}.partition(two_cliques({1, 1, 1, 1, 1, 1e30}), 2)),
                  pantr::capacity_error);
}

TEST_CASE("metis: one part never reaches METIS: zero edge weights are not refused") {
  using V = std::vector<std::int64_t>;
  const pp::CouplingGraph zero(2, V{0, 1, 2}, V{1, 0}, V{0, 0}, {1.0, 1.0});
  const auto r = pantr::metis::Partitioner{}.partition(zero, 1);
  CHECK(mowners(r) == std::vector<std::int32_t>{0, 0});
}

TEST_CASE("metis: k-way branch (> 8 parts) finds ten cliques joined by weak bridges") {
  const std::int64_t cliques = 10;
  const std::int64_t size = 4;
  const std::int64_t n = cliques * size;
  std::vector<std::int64_t> xadj{0};
  std::vector<std::int64_t> adjncy;
  std::vector<std::int64_t> ew;
  for (std::int64_t i = 0; i < n; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      const bool same = i / size == j / size;
      const bool bridge =
          (j == i + 1 && i % size == size - 1) || (i == j + 1 && j % size == size - 1);
      if (i != j && (same || bridge)) {
        adjncy.push_back(j);
        ew.push_back(bridge ? 1 : 10);
      }
    }
    xadj.push_back(static_cast<std::int64_t>(adjncy.size()));
  }
  const pp::CouplingGraph g(n, xadj, adjncy, ew,
                            std::vector<double>(static_cast<std::size_t>(n), 1.0));
  const auto o = mowners(pantr::metis::Partitioner{}.partition(g, cliques));
  std::set<std::int32_t> distinct;
  for (std::int64_t c = 0; c < cliques; ++c) {
    for (std::int64_t i = 1; i < size; ++i) {
      CHECK(o[static_cast<std::size_t>(c * size + i)] == o[static_cast<std::size_t>(c * size)]);
    }
    distinct.insert(o[static_cast<std::size_t>(c * size)]);
  }
  CHECK(distinct.size() == static_cast<std::size_t>(cliques));
}

TEST_CASE("metis: interleaved mask equals the partition of the induced subgraph") {
  const pp::CouplingGraph g = bs::coupling_graph(space_2d(6));
  const std::size_t n = static_cast<std::size_t>(g.num_vertices());
  std::vector<std::uint8_t> active(n, 1);
  for (std::size_t v = 1; v < n; v += 4) {
    active[v] = 0;
  }
  std::vector<std::size_t> keep;
  std::vector<std::int64_t> local(n, -1);
  for (std::size_t v = 0; v < n; ++v) {
    if (active[v] != 0) {
      local[v] = static_cast<std::int64_t>(keep.size());
      keep.push_back(v);
    }
  }
  std::vector<std::int64_t> xadj{0};
  std::vector<std::int64_t> adjncy;
  std::vector<std::int64_t> ew;
  for (const std::size_t v : keep) {
    for (auto k = static_cast<std::size_t>(g.xadj()[v]);
         k < static_cast<std::size_t>(g.xadj()[v + 1]); ++k) {
      const std::int64_t c = local[static_cast<std::size_t>(g.adjncy()[k])];
      if (c >= 0) {
        adjncy.push_back(c);
        ew.push_back(g.edge_weights()[k]);
      }
    }
    xadj.push_back(static_cast<std::int64_t>(adjncy.size()));
  }
  const pp::CouplingGraph sub(static_cast<std::int64_t>(keep.size()), xadj, adjncy, ew,
                              std::vector<double>(keep.size(), 1.0));
  const auto masked =
      mowners(pantr::metis::Partitioner{}.partition(g, 3, std::span<const std::uint8_t>(active)));
  const auto direct = mowners(pantr::metis::Partitioner{}.partition(sub, 3));
  std::vector<std::int32_t> expect(n, -1);
  for (std::size_t i = 0; i < keep.size(); ++i) {
    expect[keep[i]] = direct[i];
  }
  CHECK(masked == expect);
}

TEST_CASE("metis: vertex weights are rounded, not truncated (path of 6)") {
  // rint {1.6,1.6,.6,.6,.6,.6} = {2,2,1,1,1,1}: the balanced cut is after vertex 1 (4 | 4);
  // truncation gives {1,1,1,1,1,1} (clamped), whose balanced cut is after vertex 2.
  using V = std::vector<std::int64_t>;
  const pp::CouplingGraph g(6, V{0, 1, 3, 5, 7, 9, 10}, V{1, 0, 2, 1, 3, 2, 4, 3, 5, 4},
                            V{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                            std::vector<double>{1.6, 1.6, 0.6, 0.6, 0.6, 0.6});
  const auto o = mowners(pantr::metis::Partitioner{}.partition(g, 2));
  CHECK(o[0] == o[1]);
  CHECK(o[1] != o[2]);
  CHECK(o[2] == o[5]);
}

TEST_CASE("metis: k-way branch uses the edge weights (ring of ten heavy pairs)") {
  const std::int64_t n = 20;
  std::vector<std::int64_t> xadj{0};
  std::vector<std::int64_t> adjncy;
  std::vector<std::int64_t> ew;
  for (std::int64_t i = 0; i < n; ++i) {
    std::vector<std::pair<std::int64_t, std::int64_t>> nb;
    const std::int64_t next = (i + 1) % n;
    const std::int64_t prev = (i + n - 1) % n;
    nb.emplace_back(next, i % 2 == 0 ? 10 : 1);  // edge (i, i+1) heavy when i even
    nb.emplace_back(prev, prev % 2 == 0 ? 10 : 1);
    std::sort(nb.begin(), nb.end());
    for (const auto& [j, w] : nb) {
      adjncy.push_back(j);
      ew.push_back(w);
    }
    xadj.push_back(static_cast<std::int64_t>(adjncy.size()));
  }
  const pp::CouplingGraph g(n, xadj, adjncy, ew,
                            std::vector<double>(static_cast<std::size_t>(n), 1.0));
  const auto o = mowners(pantr::metis::Partitioner{}.partition(g, 10));
  for (std::int64_t i = 0; i < n; i += 2) {
    CHECK(o[static_cast<std::size_t>(i)] == o[static_cast<std::size_t>(i + 1)]);
  }
}

TEST_CASE("metis: a total vertex weight past the index type is a capacity_error") {
  // Each weight fits 32 bits, their sum does not; METIS would accumulate it in idx_t.
  if (sizeof(idx_t) == 4) {
    CHECK_THROWS_AS(static_cast<void>(pantr::metis::Partitioner{}.partition(
                        two_cliques({1e9, 1e9, 1e9, 1e9, 1e9, 1e9}), 2)),
                    pantr::capacity_error);
  }
}
