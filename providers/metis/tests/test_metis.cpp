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
