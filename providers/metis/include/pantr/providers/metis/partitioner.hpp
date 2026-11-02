/// @file partitioner.hpp
/// @brief `pantr::metis::Partitioner`, a `GraphPartitioner` backed by the METIS library.
///
/// Built and exported only with `-DPANTR_WITH_METIS=ON` (target `pantr::metis`, which links the
/// user's METIS); the core headers never include it. It does what the Python
/// `partition_graph(backend="metis")` does through pymetis: vertex weights rounded to integers
/// and clamped to at least 1, edge weights required positive, `METIS_PartGraphRecursive` for at
/// most 8 parts and `METIS_PartGraphKway` above, METIS's default options. Unlike `Spectral`, METIS
/// does not promise that every part receives a vertex.
#pragma once

#include "pantr/core/errors.hpp"
#include "pantr/partition/coupling_graph.hpp"
#include "pantr/partition/partition_graph.hpp"

#include <metis.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace pantr::metis {

namespace detail {

/// Narrow a count or an id to METIS's index type, refusing one that does not fit.
///
/// Args:
///     v (std::int64_t): The value.
///
/// Returns:
///     idx_t: The value.
///
/// Raises:
///     capacity_error: `v` is outside `idx_t` (a 32-bit METIS build and a graph past 2^31).
[[nodiscard]] inline idx_t to_idx(std::int64_t v) {
  if (v < static_cast<std::int64_t>((std::numeric_limits<idx_t>::min)()) ||
      v > static_cast<std::int64_t>((std::numeric_limits<idx_t>::max)())) {
    throw capacity_error("the graph does not fit METIS's " + std::to_string(8 * sizeof(idx_t)) +
                         "-bit index type.");
  }
  return static_cast<idx_t>(v);
}

/// Refuse weights whose total does not fit METIS's index type, which METIS accumulates them in.
///
/// Args:
///     weights (const std::vector<idx_t>&): Positive weights.
///     what (const char*): `"vertex"` or `"edge"`, for the message.
///
/// Raises:
///     capacity_error: The sum exceeds the largest `idx_t`.
inline void check_sum_fits(const std::vector<idx_t>& weights, const char* what) {
  constexpr idx_t top = (std::numeric_limits<idx_t>::max)();
  idx_t sum = 0;
  for (const idx_t w : weights) {
    if (w > top - sum) {
      throw capacity_error(std::string("the total ") + what + " weight does not fit METIS's " +
                           std::to_string(8 * sizeof(idx_t)) + "-bit index type.");
    }
    sum += w;
  }
}

}  // namespace detail

/// A graph partitioner backed by METIS (multilevel recursive bisection or k-way partitioning).
struct Partitioner {
  /// Partition a graph (see `partition::partition_graph`).
  ///
  /// Args:
  ///     graph (const partition::CouplingGraph&): The graph.
  ///     n_parts (std::int64_t): Number of parts, at least 1.
  ///     active (std::optional<std::span<const std::uint8_t>>): One flag per vertex, or none.
  ///
  /// Returns:
  ///     partition::GraphPartition: The partition; its `info` counts nothing (METIS reports no
  ///     solver path).
  ///
  /// Raises:
  ///     value_error: As `partition_graph`, or an active edge weight is not positive.
  ///     capacity_error: The graph, a weight or a weight total does not fit METIS's index type.
  ///     std::runtime_error: METIS reports an error.
  [[nodiscard]] partition::GraphPartition partition(
      const partition::CouplingGraph& graph, std::int64_t n_parts,
      std::optional<std::span<const std::uint8_t>> active = std::nullopt) const {
    const partition::detail::ActiveSubgraph sub =
        partition::detail::select_active(graph, n_parts, active);
    const std::size_t n = sub.active_idx.size();
    std::vector<std::int32_t> owner(n, 0);
    if (n_parts > 1) {
      std::int64_t non_positive = 0;
      for (const std::int64_t w : sub.edge_weights) {
        non_positive += w <= 0 ? 1 : 0;
      }
      if (non_positive > 0) {
        throw value_error("METIS requires positive integer edge weights; after rounding, " +
                          std::to_string(non_positive) +
                          " edge(s) have weight <= 0. Scale up the graph edge weights or use "
                          "backend='spectral'.");
      }
      std::vector<idx_t> xadj(sub.xadj.size());
      for (std::size_t i = 0; i < xadj.size(); ++i) {
        xadj[i] = detail::to_idx(sub.xadj[i]);
      }
      std::vector<idx_t> adjncy(sub.adjncy.size());
      std::vector<idx_t> adjwgt(sub.adjncy.size());
      for (std::size_t k = 0; k < adjncy.size(); ++k) {
        adjncy[k] = detail::to_idx(sub.adjncy[k]);
        adjwgt[k] = detail::to_idx(sub.edge_weights[k]);
      }
      std::vector<idx_t> vwgt(n);
      for (std::size_t v = 0; v < n; ++v) {
        // Finite and non-negative (the graph checks it); rint rounds half to even, as np.rint.
        const double r = std::rint(sub.vertex_weights[v]);
        if (r >= static_cast<double>((std::numeric_limits<idx_t>::max)())) {
          throw capacity_error("a vertex weight does not fit METIS's " +
                               std::to_string(8 * sizeof(idx_t)) + "-bit index type.");
        }
        vwgt[v] = r < 1.0 ? idx_t{1} : static_cast<idx_t>(r);
      }
      // METIS sums the vertex weights (part loads) and the edge weights (the cut) in idx_t.
      detail::check_sum_fits(vwgt, "vertex");
      detail::check_sum_fits(adjwgt, "edge");
      idx_t nvtxs = detail::to_idx(static_cast<std::int64_t>(n));
      idx_t ncon = 1;
      idx_t nparts = detail::to_idx(n_parts);
      idx_t options[METIS_NOPTIONS];
      METIS_SetDefaultOptions(options);
      idx_t edgecut = 0;
      std::vector<idx_t> part(n, 0);
      const int status =
          n_parts <= 8 ? METIS_PartGraphRecursive(&nvtxs, &ncon, xadj.data(), adjncy.data(),
                                                  vwgt.data(), nullptr, adjwgt.data(), &nparts,
                                                  nullptr, nullptr, options, &edgecut, part.data())
                       : METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(), vwgt.data(),
                                             nullptr, adjwgt.data(), &nparts, nullptr, nullptr,
                                             options, &edgecut, part.data());
      if (status != METIS_OK) {
        throw std::runtime_error("METIS partitioning failed (status " + std::to_string(status) +
                                 ").");
      }
      for (std::size_t v = 0; v < n; ++v) {
        owner[v] = static_cast<std::int32_t>(part[v]);
      }
    }
    return {partition::detail::scatter_owners(graph.num_vertices(), sub.active_idx, owner, n_parts),
            {}};
  }
};

static_assert(partition::GraphPartitioner<Partitioner>);

}  // namespace pantr::metis
