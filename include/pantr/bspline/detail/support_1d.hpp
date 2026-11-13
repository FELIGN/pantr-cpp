/// @file support_1d.hpp
/// @brief The cell support of every function of a 1D B-spline space.
#pragma once

#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/scalar.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace pantr::bspline::detail {

/// Per-direction function support at one level.
///
/// Attributes:
///     first_basis (std::vector<std::int64_t>): First supported function of each interval,
///         `num_intervals` entries.
///     first_cell (std::vector<std::int64_t>): First interval (cell) of each function's support,
///         `num_basis` entries.
///     last_cell (std::vector<std::int64_t>): Last interval of each function's support (inclusive).
struct Support1D {
  std::vector<std::int64_t> first_basis;
  std::vector<std::int64_t> first_cell;
  std::vector<std::int64_t> last_cell;
};

/// Compute the cell support of every B-spline function of a 1D space.
///
/// `first_basis_per_interval` is inverted: function `i` is supported on the intervals whose range
/// `[first_basis, first_basis + degree]` holds it, which is the contiguous range
/// `[first_cell, last_cell]`.
///
/// Args:
///     space (const BsplineSpace1D<T>&): The 1D space.
///
/// Returns:
///     Support1D: The three arrays.
///
/// Raises:
///     value_error: The space is periodic.
///     state_error: A function has no supported interval (an invalid space).
template <StorageFloat T>
[[nodiscard]] Support1D func_support_1d(const BsplineSpace1D<T>& space) {
  const std::span<const std::int64_t> first = space.first_basis_per_interval();
  const auto degree = static_cast<std::int64_t>(space.degree());
  const auto n_basis = static_cast<std::size_t>(space.num_basis());
  Support1D out;
  out.first_basis.assign(first.begin(), first.end());
  out.first_cell.assign(n_basis, -1);
  out.last_cell.assign(n_basis, -1);
  for (std::size_t interval = 0; interval < first.size(); ++interval) {
    for (std::int64_t i = first[interval]; i <= first[interval] + degree; ++i) {
      const auto at = static_cast<std::size_t>(i);
      if (out.first_cell[at] < 0) {
        out.first_cell[at] = static_cast<std::int64_t>(interval);
      }
      out.last_cell[at] = static_cast<std::int64_t>(interval);
    }
  }
  std::string empty;
  for (std::size_t i = 0; i < n_basis; ++i) {
    if (out.first_cell[i] < 0) {
      empty += (empty.empty() ? "" : ", ") + std::to_string(i);
    }
  }
  if (!empty.empty()) {
    throw state_error("B-spline function(s) with empty support detected at indices [" + empty +
                      "]. This indicates an invalid B-spline space.");
  }
  return out;
}

}  // namespace pantr::bspline::detail
