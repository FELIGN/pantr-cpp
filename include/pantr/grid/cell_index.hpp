/// @file cell_index.hpp
/// @brief Row-major (C-order) cell-index helpers for tensor-product grids.
///
/// A tensor-product grid with per-axis cell counts `cells_per_axis` numbers its cells in row-major
/// order: the *last* axis varies fastest, as `numpy.ravel_multi_index` / `numpy.unravel_index` do
/// with their default order. The helpers centralize that convention so no other header hard-codes a
/// strides formula.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/grid/detail/checked.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace pantr::grid {

/// Get the row-major (C-order) flat strides for `cells_per_axis`.
///
/// The flat id of the multi-index `(i_0, ..., i_{d-1})` is `sum_k i_k * strides[k]` with
/// `strides[d-1] == 1` and `strides[k] == prod(cells_per_axis[k+1:])`.
///
/// Args:
///     cells_per_axis (std::span<const std::int64_t>): Per-axis cell counts.
///
/// Returns:
///     std::vector<std::int64_t>: One stride per axis.
///
/// Raises:
///     std::overflow_error: A stride does not fit in `int64`.
[[nodiscard]] inline std::vector<std::int64_t> c_order_strides(
    std::span<const std::int64_t> cells_per_axis) {
  const std::size_t d = cells_per_axis.size();
  std::vector<std::int64_t> strides(d);
  std::int64_t stride = 1;
  for (std::size_t k = d; k-- > 0;) {
    strides[k] = stride;
    // The product after the first axis is never stored: it would be the total cell count.
    if (k > 0) {
      stride = detail::checked_mul(stride, cells_per_axis[k]);
    }
  }
  return strides;
}

/// Convert a flat cell id to its per-axis multi-index (C-order).
///
/// Args:
///     flat (std::int64_t): Flat cell id in `[0, prod(cells_per_axis))`.
///     cells_per_axis (std::span<const std::int64_t>): Per-axis cell counts.
///
/// Returns:
///     std::vector<std::int64_t>: One index per axis.
///
/// Raises:
///     index_error: `flat` is out of range.
///     std::overflow_error: The cell count does not fit in `int64`.
[[nodiscard]] inline std::vector<std::int64_t> flat_to_multi(
    std::int64_t flat, std::span<const std::int64_t> cells_per_axis) {
  const std::int64_t total = detail::checked_product(cells_per_axis);
  if (!(0 <= flat && flat < total)) {
    throw index_error("flat cell id " + std::to_string(flat) + " is out of range [0, " +
                      std::to_string(total) + ").");
  }
  std::vector<std::int64_t> multi(cells_per_axis.size());
  std::int64_t rest = flat;
  for (std::size_t k = cells_per_axis.size(); k-- > 0;) {
    multi[k] = rest % cells_per_axis[k];
    rest /= cells_per_axis[k];
  }
  return multi;
}

/// Convert a per-axis multi-index to its flat cell id (C-order).
///
/// Args:
///     multi (std::span<const std::int64_t>): One index per axis, each in
///         `[0, cells_per_axis[k])`.
///     cells_per_axis (std::span<const std::int64_t>): Per-axis cell counts.
///
/// Returns:
///     std::int64_t: The flat cell id.
///
/// Raises:
///     value_error: `multi` and `cells_per_axis` differ in length.
///     index_error: A per-axis index is out of range.
///     std::overflow_error: The cell count does not fit in `int64`.
[[nodiscard]] inline std::int64_t multi_to_flat(std::span<const std::int64_t> multi,
                                                std::span<const std::int64_t> cells_per_axis) {
  if (multi.size() != cells_per_axis.size()) {
    throw value_error("multi-index has length " + std::to_string(multi.size()) + "; expected " +
                      std::to_string(cells_per_axis.size()) + ".");
  }
  for (std::size_t k = 0; k < multi.size(); ++k) {
    if (!(0 <= multi[k] && multi[k] < cells_per_axis[k])) {
      throw index_error("axis " + std::to_string(k) + " index " + std::to_string(multi[k]) +
                        " out of range [0, " + std::to_string(cells_per_axis[k]) + ").");
    }
  }
  std::int64_t flat = 0;
  for (std::size_t k = 0; k < multi.size(); ++k) {
    flat = detail::checked_mul(flat, cells_per_axis[k]) + multi[k];
  }
  return flat;
}

}  // namespace pantr::grid
