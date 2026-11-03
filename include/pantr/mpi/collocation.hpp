/// @file collocation.hpp
/// @brief Distributed collocation interpolation and fitting onto tensor-product B-spline spaces.
/// MPI-free.
///
/// The collocation solve is a sequence of small one-dimensional solves; what is worth spreading
/// over ranks is producing the values on the node grid, whose size is the global DOF count. Each
/// rank supplies one contiguous block of the C-order flattened grid (`block_bounds`), one
/// `allgatherv` assembles the full value field, and every rank runs the same Kronecker solve
/// (`bspline::TensorCollocation`), so every rank holds the same global coefficients.
/// Design note: design/mpi.md.
#pragma once

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/fit.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/mpi/communicator.hpp"
#include "pantr/mpi/distributed_space.hpp"
#include "pantr/quad/points_lattice.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::mpi {

/// The half-open block `[start, stop)` of `range(n_total)` that `rank` evaluates: contiguous
/// blocks as even as possible, the first `n_total % n_parts` one element longer.
///
/// Args:
///     n_total (std::size_t): The length of the range.
///     n_parts (std::size_t): The number of blocks, `>= 1`.
///     rank (std::size_t): The block, in `[0, n_parts)`.
///
/// Returns:
///     std::pair<std::size_t, std::size_t>: `(start, stop)`; `start == stop` for an empty block.
[[nodiscard]] inline std::pair<std::size_t, std::size_t> block_bounds(std::size_t n_total,
                                                                      std::size_t n_parts,
                                                                      std::size_t rank) {
  if (n_parts == 0 || rank >= n_parts) {
    throw value_error("block_bounds: rank " + std::to_string(rank) + " is outside [0, " +
                      std::to_string(n_parts) + ").");
  }
  const std::size_t base = n_total / n_parts;
  const std::size_t extra = n_total % n_parts;
  const std::size_t start = rank * base + (std::min)(rank, extra);
  return {start, start + base + (rank < extra ? 1 : 0)};
}

/// A full value field: `values` row-major over `(n_total, rank)`.
///
/// Attributes:
///     values (std::vector<double>): The field.
///     rank (std::size_t): Components per point.
struct ValueField {
  std::vector<double> values;
  std::size_t rank;
};

/// Concatenate every rank's value block into the full flattened field.
///
/// The number of components is read from the first non-empty block (a scalar field when every
/// block is empty); empty blocks are skipped, whatever their extents.
///
/// Args:
///     blocks (const Gathered<double>&): The blocks in rank order, each `(m,)` or `(m, r)`.
///     n_total (std::size_t): The number of grid points.
///
/// Returns:
///     ValueField: `(n_total, rank)` values.
///
/// Raises:
///     value_error: The blocks do not tile the grid (their lengths do not sum to `n_total`), or a
///         block has another number of components than the first.
[[nodiscard]] inline ValueField assemble_blocks(const Gathered<double>& blocks,
                                                std::size_t n_total) {
  std::size_t rank_dim = 1;
  for (const std::vector<std::size_t>& s : blocks.shapes) {
    if (!s.empty() && s[0] != 0) {
      rank_dim = s.size() >= 2 ? s[1] : 1;
      break;
    }
  }
  std::size_t covered = 0;
  for (const std::vector<std::size_t>& s : blocks.shapes) {
    if (!s.empty() && s[0] != 0) {
      covered += s[0];
    }
  }
  if (covered != n_total) {
    throw value_error(
        "Per-rank value blocks do not tile the flattened grid: they cover " +
        std::to_string(covered) + " of " + std::to_string(n_total) +
        " points. On the interpolate path this means func returned the wrong number of values "
        "for a block; on the fit path (values_distributed=True) it means the supplied blocks do "
        "not match the contiguous C-order split of the grid (see fit_bspline_distributed's "
        "documented block layout).");
  }
  ValueField out{std::vector<double>(n_total * rank_dim), rank_dim};
  std::size_t row = 0;
  for (std::size_t r = 0; r < blocks.count(); ++r) {
    const std::vector<std::size_t>& s = blocks.shapes[r];
    if (s.empty() || s[0] == 0) {
      continue;
    }
    const std::span<const double> b = blocks.block(r);
    if (b.size() != s[0] * rank_dim) {
      throw value_error("the value block of rank " + std::to_string(r) + " has " +
                        std::to_string(b.size() / s[0]) + " components per point; expected " +
                        std::to_string(rank_dim) + ".");
    }
    std::copy(b.begin(), b.end(), out.values.begin() + static_cast<std::ptrdiff_t>(row * rank_dim));
    row += s[0];
  }
  return out;
}

/// Solve a tensor-product collocation system for the values every rank holds one block of: the
/// staged form of the distributed collocation.
///
/// Every rank builds the same `system` from the nodes before calling, so an inconsistent node set
/// raises on every rank before any rank enters the collective. The gathered values are rounded to
/// `T` (the space's type, as the previous release did) and solved in `Cm`.
///
/// Args:
///     comm (const C&): The communicator.
///     system (const bspline::TensorCollocation<P>&): The collocation system.
///     local (const Block<double>&): This rank's block, `(m,)` or `(m, r)`, rows the C-order grid
///         points `block_bounds` assigns it (or any contiguous split the ranks agree on).
///     tol (double): Truncation tolerance of the SVD path (see `bspline::solve_collocation`).
///
/// Returns:
///     NdArray<Cm>: The coefficients `(n_0, ..., n_{d-1}, rank)`, identical on every rank.
///
/// Raises:
///     value_error: See `assemble_blocks`, `detail::check_gather_count` and
///         `bspline::TensorCollocation::solve`.
template <StorageFloat T, StorageFloat Cm, Communicator C, StorageFloat P>
[[nodiscard]] NdArray<Cm> collocate_values_distributed(const C& comm,
                                                       const bspline::TensorCollocation<P>& system,
                                                       const Block<double>& local, double tol) {
  const std::size_t n_total = checked_product(system.grid_shape());
  const Gathered<double> blocks = comm.allgatherv(local);
  detail::check_gather_count(blocks, comm.size());
  const ValueField field = assemble_blocks(blocks, n_total);
  std::vector<Cm> values(field.values.size());
  std::transform(field.values.begin(), field.values.end(), values.begin(),
                 [](double v) { return static_cast<Cm>(static_cast<T>(v)); });
  return system.template solve<Cm>(std::span<const Cm>(values), field.rank, tol);
}

/// Fit the global B-spline from every rank's block of values on a tensor-product node grid.
///
/// Args:
///     comm (const C&): The communicator.
///     space (const bspline::BsplineSpace<T>&): The global target space.
///     nodes (const std::vector<std::span<const T>>&): One node array per direction.
///     local (const Block<double>&): This rank's block (see `collocate_values_distributed`).
///     derivatives (const std::vector<std::optional<bspline::BoundaryDerivatives>>&): See
///         `bspline::TensorCollocation`.
///     tol (std::optional<double>): Truncation tolerance; `100 * eps(T)` when absent.
///
/// Returns:
///     bspline::Bspline<T>: The global B-spline, identical on every rank.
///
/// Raises:
///     value_error: See `bspline::TensorCollocation` and `collocate_values_distributed`.
template <Communicator C, StorageFloat T>
[[nodiscard]] bspline::Bspline<T> collocate_distributed(
    const C& comm, const bspline::BsplineSpace<T>& space,
    const std::vector<std::span<const T>>& nodes, const Block<double>& local,
    const std::vector<std::optional<bspline::BoundaryDerivatives>>& derivatives = {},
    std::optional<double> tol = std::nullopt) {
  const bspline::TensorCollocation<T> system(space, nodes, derivatives);
  return bspline::Bspline<T>(
      space,
      collocate_values_distributed<T, T>(comm, system, local, bspline::default_fit_tol<T>(tol)),
      false);
}

/// Interpolate a function at a tensor-product grid of nodes (by default the Greville abscissae),
/// each rank evaluating it on its block of the grid: the convenience form taking a C++ callable.
///
/// Args:
///     ds (const DistributedSpace<bspline::BsplineSpace<T>>&): The distributed space.
///     comm (const C&): Its communicator.
///     f (F&&): Called once per rank as `f(points)` on this rank's `(m, dim)` block of the C-order
///         node grid (`mdview<const double, 2>`, possibly `m == 0`); returns an `NdArray<double>`
///         of `m` values or `(m, rank)`.
///     nodes (std::vector<std::vector<T>>): One node array per direction; Greville when empty.
///     derivatives (const std::vector<std::optional<bspline::BoundaryDerivatives>>&): See
///         `bspline::TensorCollocation`.
///     tol (std::optional<double>): Truncation tolerance.
///
/// Returns:
///     bspline::Bspline<T>: The global interpolant, identical on every rank.
///
/// Raises:
///     value_error: `f` returned another number of values than its block (raised on that rank,
///         before the collective); see `collocate_distributed`.
template <Communicator C, StorageFloat T, class F>
  requires std::invocable<F&, mdview<const double, 2>>
[[nodiscard]] bspline::Bspline<T> interpolate_distributed(
    const DistributedSpace<bspline::BsplineSpace<T>>& ds, const C& comm, F&& f,
    std::vector<std::vector<T>> nodes = {},
    const std::vector<std::optional<bspline::BoundaryDerivatives>>& derivatives = {},
    std::optional<double> tol = std::nullopt) {
  const bspline::BsplineSpace<T>& space = ds.global_space();
  if (nodes.empty()) {
    for (const bspline::BsplineSpace1D<T>& s : space.spaces()) {
      nodes.push_back(s.greville());
    }
  }
  const quad::PointsLattice<T> lattice(nodes);
  const NdArray<T> all = lattice.all_points(Order::C);
  const std::size_t dim = space.dim();
  const auto [start, stop] = block_bounds(all.shape()[0], static_cast<std::size_t>(comm.size()),
                                          static_cast<std::size_t>(comm.rank()));
  std::vector<double> pts((stop - start) * dim);
  std::transform(all.data() + start * dim, all.data() + stop * dim, pts.begin(),
                 [](T v) { return static_cast<double>(v); });
  const NdArray<double> values = f(mdview<const double, 2>(pts.data(), stop - start, dim));
  if ((values.rank() != 1 && values.rank() != 2) || values.shape()[0] != stop - start) {
    throw value_error(
        "interpolate_distributed: the function returned another number of values "
        "than the " +
        std::to_string(stop - start) + " points of this rank's block.");
  }
  const Block<double> local{values.flat(),
                            std::vector<std::size_t>(values.shape().begin(), values.shape().end())};
  const std::vector<std::span<const T>> spans(nodes.begin(), nodes.end());
  return collocate_distributed(comm, space, spans, local, derivatives, tol);
}

}  // namespace pantr::mpi
