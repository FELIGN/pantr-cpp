/// @file quasi_interpolate.hpp
/// @brief The quasi-interpolant onto a THB space (Speleers and Manni), staged as points then
/// coefficients.
///
/// Each active function `(l, beta)` gets the coefficient of a per-level tensor-product
/// Lee-Lyche-Moerken functional, with no coupling between levels:
/// `c[d] = lambda^l_beta(f)`, the truncation already living in the basis. The functional samples
/// `f` on `prod(degree + 1)` points of a level-`l` leaf cell inside the function's support; on
/// such a cell truncation zeroes every coarser active function's component on `B^l_beta`, so
/// `lambda^l_beta(s) = c_beta` for any THB spline `s` and the operator reproduces the truncated
/// space. For the plain hierarchical basis it is a local approximant and not a projector.
///
/// The leaf cell of a function is the candidate nearest to its Greville target, where the
/// candidates are the cells of the function's own level on whose functions it is listed, and the
/// target is the Greville abscissa, at the function's level, of the first candidate's multi-index
/// (the first candidate in increasing cell id). Of equally near candidates the one with the
/// smallest cell id is taken.
#pragma once

#include "pantr/bspline/quasi_interpolate.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/parallel/parallel_for.hpp"
#include "pantr/parallel/work_model.hpp"
#include "pantr/thb/space.hpp"
#include "pantr/thb/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::thb {

namespace detail {

/// Contract one function's block of values with its per-direction weights, the first direction
/// first, summing over each direction's samples in increasing order.
///
/// A `noexcept` function of its own so that GCC vectorises it, as `contract_cell`.
///
/// Args:
///     values (const double*): Row-major `(prod(orders), rank)` samples of the function.
///     weights (const double*): The per-direction weight rows, direction after direction,
///         `sum(orders)` entries.
///     orders (const std::size_t*): `dim` samples per direction.
///     dim (std::size_t): Number of directions.
///     rank (std::size_t): Components.
///     cur (double*): Scratch of `prod(orders) * rank` entries.
///     next (double*): Scratch of `prod(orders) * rank` entries.
///     out (double*): `rank` entries, written in full.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
inline void contract_block(const double* values, const double* weights, const std::size_t* orders,
                           std::size_t dim, std::size_t rank, double* cur, double* next,
                           double* out) noexcept {
  std::size_t rest = rank;
  for (std::size_t k = 0; k < dim; ++k) {
    rest *= orders[k];
  }
  std::copy(values, values + rest, cur);
  const double* w = weights;
  for (std::size_t k = 0; k < dim; ++k) {
    const std::size_t order = orders[k];
    rest /= order;
    for (std::size_t r = 0; r < rest; ++r) {
      double acc = 0.0;
      for (std::size_t j = 0; j < order; ++j) {
        acc += w[j] * cur[j * rest + r];
      }
      next[r] = acc;
    }
    std::swap(cur, next);
    w += order;
  }
  std::copy(cur, cur + rank, out);
}

}  // namespace detail

/// The staged THB quasi-interpolant: the points to sample, then the coefficients from the samples.
/// Immutable.
///
/// The points are listed function by function (in global dof order), each function's
/// `prod(degree + 1)` points as the tensor grid of its per-direction points, in C order (the last
/// direction fastest).
///
/// Attributes:
///     space_ (THBSplineSpace<T>): The target space.
///     orders_ (std::vector<std::size_t>): Samples per direction, `degree + 1`.
///     leaf_ (std::vector<std::int64_t>): Per function, the leaf cell it samples.
///     points_ (std::vector<double>): Per function, the per-direction points, direction after
///         direction (`sum(orders_)` entries each).
///     weights_ (std::vector<double>): The matching weight rows.
template <StorageFloat T>
class THBQuasiInterpolation {
 public:
  /// Choose each function's leaf cell and build its per-direction points and weights.
  ///
  /// Args:
  ///     space (THBSplineSpace<T>): The target space.
  ///
  /// Raises:
  ///     state_error: An active function has no leaf cell at its level (an invalid space).
  ///     value_error, index_error: A local collocation block is singular, or its inverse is not
  ///         finite (`bspline::detail::llm_weight_row`).
  explicit THBQuasiInterpolation(THBSplineSpace<T> space) : space_(std::move(space)) {
    const std::size_t d = space_.dim();
    for (const int p : space_.degrees()) {
      orders_.push_back(static_cast<std::size_t>(p) + 1);
    }
    for (const std::size_t o : orders_) {
      block_ *= o;
      width_ += o;
    }
    const auto n = static_cast<std::size_t>(space_.num_total_basis());
    const auto levels = static_cast<std::size_t>(space_.num_levels());
    std::vector<bspline::BsplineSpace<T>> level_spaces;
    for (std::size_t l = 0; l < levels; ++l) {
      level_spaces.push_back(space_.level_space(static_cast<std::int64_t>(l)));
    }
    // The Greville abscissae of every level and direction, widened to double.
    std::vector<std::vector<std::vector<double>>> greville(levels);
    for (std::size_t l = 0; l < levels; ++l) {
      for (const bspline::BsplineSpace1D<T>& s : level_spaces[l].spaces()) {
        const std::vector<T> g = s.greville();
        greville[l].emplace_back(g.begin(), g.end());
      }
    }

    // One pass over the cells in increasing id. The first candidate of a function fixes its
    // target, so the nearest candidate is tracked as the cells go by; `<` keeps the first of equal
    // distances.
    std::vector<std::int64_t> level_of(n, -1);
    std::vector<double> target(n * d, 0.0);
    std::vector<double> best(n, std::numeric_limits<double>::infinity());
    std::vector<std::int64_t> multi(n * d, 0);
    leaf_.assign(n, -1);
    const grid::HierarchicalGrid grid = space_.grid();
    std::vector<double> centre(d);
    for (std::int64_t cid = 0; cid < grid.num_cells(); ++cid) {
      const std::int64_t level = grid.cell_level(cid);
      const CellContributions cell = space_.contributions(cid);
      const auto [lo, hi] = grid.cell_bounds(cid);
      for (std::size_t k = 0; k < d; ++k) {
        centre[k] = 0.5 * (lo[k] + hi[k]);
      }
      for (std::size_t j = 0; j < cell.size(); ++j) {
        if (cell.levels[j] != level) {
          continue;
        }
        const auto dof = static_cast<std::size_t>(cell.dofs[j]);
        const std::span<const std::int64_t> idx = cell.multi_index(j);
        if (level_of[dof] < 0) {
          level_of[dof] = level;
          for (std::size_t k = 0; k < d; ++k) {
            target[dof * d + k] =
                greville[static_cast<std::size_t>(level)][k][static_cast<std::size_t>(idx[k])];
          }
        }
        // The Euclidean distance as the square root of the sum of squares in direction order,
        // each square its own statement: no contraction, so mirrored candidates tie exactly.
        double squares = 0.0;
        for (std::size_t k = 0; k < d; ++k) {
          const double diff = centre[k] - target[dof * d + k];
          const double square = diff * diff;
          squares += square;
        }
        const double dist = std::sqrt(squares);
        if (dist < best[dof]) {
          best[dof] = dist;
          leaf_[dof] = cid;
          std::copy(idx.begin(), idx.end(), multi.begin() + static_cast<std::ptrdiff_t>(dof * d));
        }
      }
    }

    points_.assign(n * width_, 0.0);
    weights_.assign(n * width_, 0.0);
    for (std::size_t dof = 0; dof < n; ++dof) {
      if (level_of[dof] < 0) {
        throw state_error("active dof " + std::to_string(dof) +
                          " has no leaf cell at its level; the THB space is inconsistent.");
      }
      const auto [lo, hi] = grid.cell_bounds(leaf_[dof]);
      const bspline::BsplineSpace<T>& level_space =
          level_spaces[static_cast<std::size_t>(level_of[dof])];
      const std::vector<bspline::BsplineSpace1D<T>>& dirs = level_space.spaces();
      std::size_t offset = dof * width_;
      for (std::size_t k = 0; k < d; ++k) {
        const std::span<double> pts(points_.data() + offset, orders_[k]);
        bspline::detail::llm_interval_points(lo[k], hi[k], pts);
        // `multi` is the function's index in its level's space, which the block's functions are
        // numbered in, not a global dof.
        bspline::detail::llm_weight_row<T>(dirs[k], static_cast<std::size_t>(multi[dof * d + k]),
                                           pts,
                                           std::span<double>(weights_.data() + offset, orders_[k]));
        offset += orders_[k];
      }
    }
  }

  /// Get the target space. Holds: a reference valid while the object lives.
  [[nodiscard]] const THBSplineSpace<T>& space() const noexcept { return space_; }

  /// Get the number of samples per function, `prod(degree + 1)`.
  [[nodiscard]] std::size_t block() const noexcept { return block_; }

  /// Get the number of functions, `space().num_total_basis()`.
  [[nodiscard]] std::size_t num_functions() const noexcept { return leaf_.size(); }

  /// Get the leaf cell function `dof` samples.
  ///
  /// Raises:
  ///     index_error: `dof` is out of range.
  [[nodiscard]] std::int64_t leaf_cell(std::int64_t dof) const { return leaf_[checked(dof)]; }

  /// Get the weight rows of function `dof`, direction after direction.
  ///
  /// Returns:
  ///     std::span<const double>: `sum(degree + 1)` entries; direction `k`'s row is the
  ///     `degree[k] + 1` entries after those of the directions before it.
  ///
  /// Raises:
  ///     index_error: `dof` is out of range.
  [[nodiscard]] std::span<const double> weights(std::int64_t dof) const {
    return std::span<const double>(weights_).subspan(checked(dof) * width_, width_);
  }

  /// Get every sample point.
  ///
  /// Returns:
  ///     NdArray<double>: Extents `(num_functions() * block(), dim)`.
  [[nodiscard]] NdArray<double> points() const {
    const std::size_t d = orders_.size();
    const std::size_t n = num_functions();
    NdArray<double> out({n * block_, d}, for_overwrite);
    double* dst = out.data();
    std::vector<std::size_t> idx(d);
    for (std::size_t dof = 0; dof < n; ++dof) {
      for (std::size_t kk = 0; kk < block_; ++kk) {
        std::size_t rest = kk;
        for (std::size_t k = d; k-- > 0;) {
          idx[k] = rest % orders_[k];
          rest /= orders_[k];
        }
        std::size_t offset = dof * width_;
        for (std::size_t k = 0; k < d; ++k) {
          *dst++ = points_[offset + idx[k]];
          offset += orders_[k];
        }
      }
    }
    return out;
  }

  /// Get the coefficients from the samples at `points()`.
  ///
  /// Each function's block of values is contracted with its weights one direction at a time, the
  /// first direction first, summing over a direction's samples in increasing order. The functions
  /// are contracted in parallel, each writing its own coefficient, so the result does not depend
  /// on the thread count. The work of one function is two operations per entry of each successive
  /// intermediate array, `block() * rank` entries shrinking by the direction's samples at each
  /// step.
  ///
  /// Args:
  ///     values (mdview<const double, 2>): Extents `(num_functions() * block(), rank)`.
  ///
  /// Returns:
  ///     NdArray<double>: Extents `(num_functions(), rank)`.
  ///
  /// Raises:
  ///     value_error: Another number of rows.
  [[nodiscard]] NdArray<double> coefficients(mdview<const double, 2> values) const {
    const std::size_t n = num_functions();
    const std::size_t rank = values.extent(1);
    if (values.extent(0) != n * block_) {
      throw value_error("THBQuasiInterpolation: expected " + std::to_string(n * block_) +
                        " rows of values, got " + std::to_string(values.extent(0)) + ".");
    }
    NdArray<double> out({n, rank}, for_overwrite);  // every row is written below
    double ops = 0.0;
    std::size_t size = block_ * rank;
    for (const std::size_t order : orders_) {
      ops += 2.0 * static_cast<double>(size);
      size /= order;
    }
    const parallel::WorkEstimate work{(std::max)(1.0, ops)};
    const double* src = values.data_handle();
    double* dst = out.data();
    parallel::parallel_for(n, work, [&](std::size_t begin, std::size_t end) {
      std::vector<double> cur(block_ * rank);
      std::vector<double> next(block_ * rank);
      for (std::size_t dof = begin; dof < end; ++dof) {
        detail::contract_block(src + dof * block_ * rank, weights_.data() + dof * width_,
                               orders_.data(), orders_.size(), rank, cur.data(), next.data(),
                               dst + dof * rank);
      }
    });
    return out;
  }

 private:
  [[nodiscard]] std::size_t checked(std::int64_t dof) const {
    if (!(0 <= dof && static_cast<std::size_t>(dof) < leaf_.size())) {
      throw index_error("dof " + std::to_string(dof) + " is out of range [0, " +
                        std::to_string(leaf_.size()) + ").");
    }
    return static_cast<std::size_t>(dof);
  }

  THBSplineSpace<T> space_;
  std::vector<std::size_t> orders_;
  std::size_t block_ = 1;
  std::size_t width_ = 0;
  std::vector<std::int64_t> leaf_;
  std::vector<double> points_;
  std::vector<double> weights_;
};

}  // namespace pantr::thb
