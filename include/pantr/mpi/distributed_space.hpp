/// @file distributed_space.hpp
/// @brief A B-spline or THB space distributed over a communicator, and the partition broadcast that
/// makes every rank agree on its cell owners. MPI-free.
///
/// The storage model is redundant: every rank holds the global space and the partition, and builds
/// its own windowed `LocalSpace`, with no exchange of degrees of freedom (cross-rank coupling is
/// the caller's job). That model needs every rank to hold the same partition. A partition derived
/// from floating-point work (a spectral bisection, a weighted recursive bisection) can differ
/// between two machines that round differently (another ISA rung, another compiler), so
/// `create_distributed_space` computes it on rank 0 only and sends it to the others
/// (`broadcast_partition`). At one rank there is nobody to disagree with and nothing is sent.
/// Design note: design/mpi.md.
#pragma once

#include "pantr/bspline/local_space.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/grid/partition.hpp"
#include "pantr/mpi/communicator.hpp"
#include "pantr/thb/local_space.hpp"
#include "pantr/thb/space.hpp"

#include <concepts>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pantr::mpi {

namespace detail {

/// The first entry of rank 0's broadcast payload: the partition follows.
inline constexpr std::int64_t kPartitionOk = 1;
/// The first entry of rank 0's broadcast payload when it could not compute the partition.
inline constexpr std::int64_t kPartitionFailed = 0;

/// Raise unless a gather returned one block per rank.
///
/// Raises:
///     value_error: Another number of blocks than `size`.
template <class T>
void check_gather_count(const Gathered<T>& g, int size) {
  if (g.count() != static_cast<std::size_t>(size)) {
    throw value_error("allgather returned " + std::to_string(g.count()) +
                      " entries but comm.size is " + std::to_string(size) +
                      "; the communicator's allgather is inconsistent.");
  }
}

}  // namespace detail

/// Compute a partition on rank 0 and give every rank rank 0's.
///
/// At `comm.size() == 1` this is `compute()`. Otherwise rank 0 calls `compute()` and sends
/// `[1, n_parts, owner_0, ..., owner_{n-1}]` through `allgatherv` (the other ranks send nothing);
/// every other rank builds its partition from that payload without calling `compute`. If
/// `compute()` throws on rank 0, rank 0 still takes part in the collective, sending `[0]`, and then
/// rethrows its exception, so the other ranks do not wait forever: they raise `value_error`.
///
/// Args:
///     comm (const C&): The communicator.
///     compute (F&&): Returns the partition (`grid::Partition`); called on rank 0 only.
///
/// Returns:
///     grid::Partition: Rank 0's partition, on every rank.
///
/// Raises:
///     value_error: On a rank other than 0: rank 0 failed, or its payload is malformed; the
///         gather returned another number of blocks than ranks.
///     Whatever `compute` raises, on rank 0.
template <Communicator C, class F>
  requires std::same_as<std::invoke_result_t<F&>, grid::Partition>
[[nodiscard]] grid::Partition broadcast_partition(const C& comm, F&& compute) {
  const int size = comm.size();
  if (size == 1) {
    return compute();
  }
  const int rank = comm.rank();
  std::optional<grid::Partition> mine;
  std::exception_ptr failure;
  std::vector<std::int64_t> payload;
  if (rank == 0) {
    try {
      mine.emplace(compute());
      const std::span<const std::int32_t> owner = mine->cell_owner();
      payload.reserve(owner.size() + 2);
      payload.push_back(detail::kPartitionOk);
      payload.push_back(mine->n_parts());
      payload.insert(payload.end(), owner.begin(), owner.end());
    } catch (...) {
      failure = std::current_exception();
      mine.reset();
      payload.assign(1, detail::kPartitionFailed);
    }
  }
  const Gathered<std::int64_t> g = comm.allgatherv(block_1d<std::int64_t>(payload));
  if (failure) {
    std::rethrow_exception(failure);
  }
  if (mine) {
    return *std::move(mine);
  }
  detail::check_gather_count(g, size);
  const std::span<const std::int64_t> root = g.block(0);
  if (root.empty() || root[0] != detail::kPartitionOk) {
    throw value_error("rank 0 could not compute the partition (its exception is raised there).");
  }
  if (root.size() < 2) {
    throw value_error("rank 0's partition payload is malformed.");
  }
  std::vector<std::int32_t> owner;
  owner.reserve(root.size() - 2);
  for (std::size_t i = 2; i < root.size(); ++i) {
    if (root[i] < std::numeric_limits<std::int32_t>::min() ||
        root[i] > (std::numeric_limits<std::int32_t>::max)()) {
      throw value_error("rank 0's partition payload holds an owner outside int32.");
    }
    owner.push_back(static_cast<std::int32_t>(root[i]));
  }
  return grid::Partition(std::move(owner), root[1]);
}

/// The local-space type of a global space type.
template <class Space>
struct local_space_of;
template <StorageFloat T>
struct local_space_of<bspline::BsplineSpace<T>> {
  using type = bspline::LocalSpace<bspline::BsplineSpace<T>>;
};
template <StorageFloat T>
struct local_space_of<thb::THBSplineSpace<T>> {
  using type = bspline::LocalSpace<thb::THBSplineSpace<T>>;
};

/// The number of cells of a space's grid.
template <StorageFloat T>
[[nodiscard]] std::int64_t global_cell_count(const bspline::BsplineSpace<T>& space) {
  std::int64_t n = 1;
  for (const bspline::BsplineSpace1D<T>& s : space.spaces()) {
    n *= s.num_intervals();
  }
  return n;
}
/// The number of cells of a space's grid.
template <StorageFloat T>
[[nodiscard]] std::int64_t global_cell_count(const thb::THBSplineSpace<T>& space) {
  return space.grid().num_cells();
}

/// A B-spline (`bspline::BsplineSpace<T>`) or THB (`thb::THBSplineSpace<T>`) space distributed over
/// `n_parts` ranks, as seen from one of them. Immutable.
///
/// Attributes:
///     global_space (Space): The global space, identical on every rank.
///     partition (grid::Partition): The owner of every cell, identical on every rank.
///     rank (int): This rank.
///     owned_cells (std::vector<std::int64_t>): The cells this rank owns, ascending.
///     local (std::optional<LocalSpace>): This rank's windowed space; empty when it owns no cell.
template <class Space>
class DistributedSpace {
 public:
  using Local = typename local_space_of<Space>::type;

  /// Build this rank's view.
  ///
  /// Args:
  ///     global_space (Space): The global space (non-periodic).
  ///     partition (grid::Partition): The owner of every cell of the space's grid.
  ///     rank (int): This rank, in `[0, size)`.
  ///     size (int): The number of ranks; must equal `partition.n_parts()`.
  ///
  /// Raises:
  ///     value_error: A periodic B-spline space; `size != partition.n_parts()`; another cell count;
  ///         a rank outside `[0, size)`.
  DistributedSpace(Space global_space, grid::Partition partition, int rank, int size)
      : global_(std::move(global_space)), partition_(std::move(partition)), rank_(rank) {
    if constexpr (requires { global_.spaces(); }) {
      for (const auto& s : global_.spaces()) {
        if (s.periodic()) {
          throw value_error("periodic B-spline spaces are not supported.");
        }
      }
    }
    if (size != partition_.n_parts()) {
      throw value_error("comm.size (" + std::to_string(size) + ") must equal partition.n_parts (" +
                        std::to_string(partition_.n_parts()) + ").");
    }
    const std::int64_t n_cells = global_cell_count(global_);
    if (partition_.n_cells() != n_cells) {
      throw value_error("partition has " + std::to_string(partition_.n_cells()) +
                        " cells; expected " + std::to_string(n_cells) +
                        " (the global space's cell count).");
    }
    owned_ = partition_.owned_cells(rank_);
    if (!owned_.empty()) {
      if constexpr (requires { global_.spaces(); }) {
        local_.emplace(bspline::build_local(global_, partition_, rank_));
      } else {
        local_.emplace(thb::build_local(global_, partition_, rank_));
      }
    }
  }

  /// The global space.
  [[nodiscard]] const Space& global_space() const noexcept { return global_; }
  /// The partition.
  [[nodiscard]] const grid::Partition& partition() const noexcept { return partition_; }
  /// This rank.
  [[nodiscard]] int rank() const noexcept { return rank_; }
  /// The number of parts.
  [[nodiscard]] std::int64_t n_parts() const noexcept { return partition_.n_parts(); }
  /// The cells this rank owns, ascending.
  [[nodiscard]] const std::vector<std::int64_t>& owned_cells() const noexcept { return owned_; }
  /// This rank's local space, or empty when it owns no cell.
  [[nodiscard]] const std::optional<Local>& local() const noexcept { return local_; }
  /// Whether this rank owns a cell.
  [[nodiscard]] bool owns_cells() const noexcept { return local_.has_value(); }

 private:
  Space global_;
  grid::Partition partition_;
  int rank_;
  std::vector<std::int64_t> owned_;
  std::optional<Local> local_;
};

/// Distribute a space over a communicator with a partition every rank already holds.
template <Communicator C, class Space>
[[nodiscard]] DistributedSpace<Space> distribute(Space global_space, grid::Partition partition,
                                                 const C& comm) {
  const int size = comm.size();
  return DistributedSpace<Space>(std::move(global_space), std::move(partition), comm.rank(), size);
}

/// Distribute a space, computing its partition on rank 0 and broadcasting it
/// (`broadcast_partition`).
///
/// Args:
///     global_space (Space): The global space, identical on every rank.
///     comm (const C&): The communicator.
///     partitioner (P&&): Called on rank 0 as `partitioner(global_space, n_parts)`, returning a
///         `grid::Partition` (for instance over `grid::partition_grid` or
///         `partition::partition_graph`).
///
/// Returns:
///     DistributedSpace<Space>: This rank's view.
template <Communicator C, class Space, class P>
  requires std::same_as<std::invoke_result_t<P&, const Space&, std::int64_t>, grid::Partition>
[[nodiscard]] DistributedSpace<Space> create_distributed_space(Space global_space, const C& comm,
                                                               P&& partitioner) {
  const int size = comm.size();
  grid::Partition p = broadcast_partition(
      comm, [&]() { return partitioner(std::as_const(global_space), std::int64_t{size}); });
  return DistributedSpace<Space>(std::move(global_space), std::move(p), comm.rank(), size);
}

}  // namespace pantr::mpi
