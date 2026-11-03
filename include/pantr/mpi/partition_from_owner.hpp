/// @file partition_from_owner.hpp
/// @brief A cell partition assembled from the cells each rank owns elsewhere (in a finite-element
/// mesh distributed by another library). MPI-free.
///
/// Each rank sends the ids of the cells it owns; one `allgatherv` gives every rank all of them, and
/// every rank assembles the same owner array, `-1` for a cell no rank sent (one absent from the
/// mesh). The assembly is integer-only, so every rank's result is the same without a broadcast.
/// Design note: design/mpi.md.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/grid/partition.hpp"
#include "pantr/mpi/communicator.hpp"
#include "pantr/mpi/distributed_space.hpp"

#include <algorithm>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace pantr::mpi {

/// Assemble the partition whose owner of each cell is the rank that sent it.
///
/// Reads `comm.size()` and never `comm.rank()`.
///
/// Args:
///     comm (const C&): The communicator.
///     local_ids (std::span<const std::int64_t>): The cells this rank owns, in `[0, n_cells)`.
///     n_cells (std::int64_t): The number of cells, `>= 1`.
///
/// Returns:
///     grid::Partition: `comm.size()` parts; `-1` for a cell nobody sent.
///
/// Raises:
///     value_error: `n_cells < 1` (before the collective); another number of blocks than ranks;
///         a block that is not one-dimensional; an id outside `[0, n_cells)`; a cell sent by two
///         ranks.
template <Communicator C>
[[nodiscard]] grid::Partition partition_from_owner(const C& comm,
                                                   std::span<const std::int64_t> local_ids,
                                                   std::int64_t n_cells) {
  if (n_cells < 1) {
    throw value_error("n_cells must be >= 1; got " + std::to_string(n_cells) + ".");
  }
  const Gathered<std::int64_t> g = comm.allgatherv(block_1d(local_ids));
  const int n_parts = comm.size();
  detail::check_gather_count(g, n_parts);
  std::vector<std::int32_t> owner(static_cast<std::size_t>(n_cells), -1);
  for (std::size_t r = 0; r < g.count(); ++r) {
    if (g.shapes[r].size() != 1) {
      throw value_error("allgather result for rank " + std::to_string(r) +
                        " is not a 1-D array (got shape " +
                        py_repr(std::span<const std::size_t>(g.shapes[r])) + ").");
    }
    const std::span<const std::int64_t> ids = g.block(r);
    if (ids.empty()) {
      continue;
    }
    const auto [lo, hi] = std::minmax_element(ids.begin(), ids.end());
    if (*lo < 0 || *hi >= n_cells) {
      throw value_error("gathered cell ids from rank " + std::to_string(r) + " are outside [0, " +
                        std::to_string(n_cells) + "): got range [" + std::to_string(*lo) + ", " +
                        std::to_string(*hi) + "].");
    }
    for (const std::int64_t c : ids) {
      const auto ci = static_cast<std::size_t>(c);
      if (owner[ci] != -1) {
        // A repeat inside one rank's block is accepted, as the previous release accepted it.
        if (owner[ci] != static_cast<std::int32_t>(r)) {
          throw value_error("inconsistent partition: cell " + std::to_string(c) +
                            " is claimed by both rank " + std::to_string(owner[ci]) + " and rank " +
                            std::to_string(r) + ".");
        }
        continue;
      }
      owner[ci] = static_cast<std::int32_t>(r);
    }
  }
  return grid::Partition(std::move(owner), n_parts);
}

}  // namespace pantr::mpi
