/// @file coupling_graph.hpp
/// @brief The cell-coupling graph of a tensor-product B-spline space.
#pragma once

#include "pantr/bspline/space.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/partition/coupling_graph.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace pantr::bspline {

namespace detail {

/// The cells one interval of a 1D space couples to, with the number of functions they share.
///
/// Attributes:
///     ptr (std::vector<std::int64_t>): Row pointers, one row per interval.
///     other (std::vector<std::int64_t>): Coupled intervals, ascending, the interval itself
///         included.
///     shared (std::vector<std::int64_t>): Functions shared with each, aligned with `other`.
struct AxisCoupling {
  std::vector<std::int64_t> ptr;
  std::vector<std::int64_t> other;
  std::vector<std::int64_t> shared;
};

/// Tabulate the interval-to-interval coupling of one non-periodic direction.
///
/// Interval `i` is supported by the `degree + 1` functions `fb[i], ..., fb[i] + degree` (`fb` the
/// first function per interval), so intervals `i` and `j` share
/// `max(0, degree + 1 - |fb[i] - fb[j]|)` functions. `fb` is strictly increasing, so the intervals
/// sharing any function with `i` form one contiguous run around it, found by two pointers.
///
/// Args:
///     first_basis (std::span<const std::int64_t>): First function per interval, increasing.
///     degree (int): The degree, at least 0.
///
/// Returns:
///     AxisCoupling: One row per interval.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `coupling_graph` instead.
[[nodiscard]] inline AxisCoupling axis_coupling(std::span<const std::int64_t> first_basis,
                                                int degree) {
  const std::size_t n = first_basis.size();
  const auto width = static_cast<std::int64_t>(degree) + 1;
  AxisCoupling out;
  out.ptr.assign(n + 1, 0);
  std::size_t lo = 0;
  for (std::size_t i = 0; i < n; ++i) {
    while (first_basis[i] - first_basis[lo] >= width) {
      ++lo;
    }
    for (std::size_t j = lo; j < n && first_basis[j] - first_basis[i] < width; ++j) {
      out.other.push_back(static_cast<std::int64_t>(j));
      out.shared.push_back(width - (first_basis[j] > first_basis[i]
                                        ? first_basis[j] - first_basis[i]
                                        : first_basis[i] - first_basis[j]));
    }
    out.ptr[i + 1] = static_cast<std::int64_t>(out.other.size());
  }
  return out;
}

}  // namespace detail

/// Build the cell-coupling graph of a non-periodic tensor-product B-spline space.
///
/// Cells are flat-indexed in C order over the per-direction interval counts. Two distinct cells
/// share the product over directions of the functions their intervals share in that direction,
/// and are joined when that product is positive; each row lists its neighbors in ascending order.
///
/// Args:
///     space (const BsplineSpace<T>&): The space; no direction may be periodic.
///     cell_weights (std::optional<std::span<const double>>): Per-cell weights, finite and
///         non-negative; uniform 1 when absent.
///
/// Returns:
///     partition::CouplingGraph: The graph.
///
/// Raises:
///     value_error: A direction is periodic, or the weights are invalid.
template <StorageFloat T>
[[nodiscard]] partition::CouplingGraph coupling_graph(
    const BsplineSpace<T>& space,
    std::optional<std::span<const double>> cell_weights = std::nullopt) {
  const auto& spaces = space.spaces();
  if (std::any_of(spaces.begin(), spaces.end(), [](const auto& s) { return s.periodic(); })) {
    throw value_error("periodic B-spline spaces are not supported.");
  }
  const std::size_t dim = spaces.size();
  std::vector<detail::AxisCoupling> axes;
  std::vector<std::size_t> cells_per_axis(dim);
  std::size_t n_cells = 1;
  for (std::size_t d = 0; d < dim; ++d) {
    axes.push_back(detail::axis_coupling(spaces[d].first_basis_per_interval(), spaces[d].degree()));
    cells_per_axis[d] = static_cast<std::size_t>(spaces[d].num_intervals());
    n_cells *= cells_per_axis[d];  // the space's constructor refuses an overflowing product
  }
  std::vector<double> weights = partition::detail::resolve_cell_weights(cell_weights, n_cells);

  // C-order strides of the cell grid.
  std::vector<std::int64_t> stride(dim, 1);
  for (std::size_t d = dim; d-- > 1;) {
    stride[d - 1] = stride[d] * static_cast<std::int64_t>(cells_per_axis[d]);
  }

  std::vector<std::int64_t> xadj(n_cells + 1, 0);
  std::vector<std::int64_t> adjncy;
  std::vector<std::int64_t> edge_weights;
  std::vector<std::size_t> multi(dim, 0);  // the cell's multi-index
  std::vector<std::size_t> pos(dim, 0);    // the odometer over the per-axis rows
  for (std::size_t c = 0; c < n_cells; ++c) {
    // Walk the product of the per-axis rows in lexicographic order, which is ascending flat id.
    for (std::size_t d = 0; d < dim; ++d) {
      pos[d] = static_cast<std::size_t>(axes[d].ptr[multi[d]]);
    }
    for (;;) {
      std::int64_t id = 0;
      std::int64_t w = 1;
      for (std::size_t d = 0; d < dim; ++d) {
        id += axes[d].other[pos[d]] * stride[d];
        w *= axes[d].shared[pos[d]];
      }
      if (id != static_cast<std::int64_t>(c)) {
        adjncy.push_back(id);
        edge_weights.push_back(w);
      }
      std::size_t d = dim;
      while (d-- > 0) {
        if (++pos[d] < static_cast<std::size_t>(axes[d].ptr[multi[d] + 1])) {
          break;
        }
        pos[d] = static_cast<std::size_t>(axes[d].ptr[multi[d]]);
      }
      if (d == static_cast<std::size_t>(-1)) {
        break;
      }
    }
    xadj[c + 1] = static_cast<std::int64_t>(adjncy.size());
    for (std::size_t d = dim; d-- > 0;) {
      if (++multi[d] < cells_per_axis[d]) {
        break;
      }
      multi[d] = 0;
    }
  }
  return partition::CouplingGraph(static_cast<std::int64_t>(n_cells), std::move(xadj),
                                  std::move(adjncy), std::move(edge_weights), std::move(weights));
}

}  // namespace pantr::bspline
