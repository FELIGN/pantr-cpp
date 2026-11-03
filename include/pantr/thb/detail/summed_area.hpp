/// @file summed_area.hpp
/// @brief `FalseCountTable`: box queries over a boolean mask in `O(2^dim)` each.
///
/// The Kraft selection and the truncation walk ask, for very many axis-aligned boxes of cells,
/// whether a mask is true on all of a box or on some of it. A summed-area table over the cells
/// where the mask is false answers both from `2^dim` lookups (inclusion-exclusion over the box's
/// corners), at the cost of one pass over the mask per axis to build it.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace pantr::thb::detail {

/// A summed-area table of the cells where a boolean mask is false.
///
/// Attributes:
///     table_ (std::vector<std::int64_t>): Row-major, extents `counts[k] + 1` per axis; entry
///         `(i_0, ..., i_{d-1})` is the number of false cells in `[0, i_0) x ... x [0, i_{d-1})`.
///     stride_ (std::vector<std::size_t>): Row-major strides of `table_`.
class FalseCountTable {
 public:
  /// Build the table of `mask`.
  ///
  /// Args:
  ///     mask (std::span<const std::uint8_t>): Row-major mask of extents `counts`; zero means
  ///         false.
  ///     counts (std::span<const std::int64_t>): Cells per axis, each at least 1.
  ///
  /// Raises:
  ///     capacity_error: The table, `prod(counts[k] + 1)` entries, does not fit in `size_t`. Its
  ///         extents exceed the mask's by one per axis, so a mask that fits can still overflow
  ///         it: one cell on each of 64 axes needs `2^64` entries.
  ///
  /// Note:
  ///     Inputs are otherwise assumed to be correct (no validation performed): `mask.size()` is
  ///     the product of `counts`.
  FalseCountTable(std::span<const std::uint8_t> mask, std::span<const std::int64_t> counts)
      : stride_(counts.size()) {
    const std::size_t dim = counts.size();
    std::vector<std::size_t> extent(dim);
    std::size_t total = 1;
    for (std::size_t k = dim; k-- > 0;) {
      extent[k] = static_cast<std::size_t>(counts[k]) + 1;
      stride_[k] = total;
      if (total > (std::numeric_limits<std::size_t>::max)() / extent[k]) {
        throw capacity_error(
            "THBSplineSpace: a level's summed-area table needs more entries than fit in memory "
            "addresses (" +
            std::to_string(dim) + " directions).");
      }
      total *= extent[k];
    }
    table_.assign(total, 0);
    // Scatter the false cells to the shifted position (index + 1 per axis): the table's first
    // slice along every axis stays zero, which is what makes a box sum a difference of corners.
    std::vector<std::size_t> idx(dim, 0);
    for (std::size_t cell = 0; cell < mask.size(); ++cell) {
      if (mask[cell] == 0) {
        std::size_t at = 0;
        for (std::size_t k = 0; k < dim; ++k) {
          at += (idx[k] + 1) * stride_[k];
        }
        table_[at] = 1;
      }
      for (std::size_t k = dim; k-- > 0;) {
        if (++idx[k] < static_cast<std::size_t>(counts[k])) {
          break;
        }
        idx[k] = 0;
      }
    }
    // One cumulative sum per axis: `outer` blocks of `extent[k]` slices of `inner` entries.
    for (std::size_t k = 0; k < dim; ++k) {
      const std::size_t inner = stride_[k];
      const std::size_t outer = total / (extent[k] * inner);
      for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t m = 1; m < extent[k]; ++m) {
          std::int64_t* row = table_.data() + (o * extent[k] + m) * inner;
          const std::int64_t* prev = row - inner;
          for (std::size_t i = 0; i < inner; ++i) {
            row[i] += prev[i];
          }
        }
      }
    }
  }

  /// Count the false cells of the box `[lo, hi)`.
  ///
  /// Args:
  ///     lo (std::span<const std::int64_t>): Box lower corner, in `[0, counts]` per axis.
  ///     hi (std::span<const std::int64_t>): Box upper corner (exclusive), `lo <= hi <= counts`.
  ///
  /// Returns:
  ///     std::int64_t: The number of cells of the box where the mask is false.
  ///
  /// Note:
  ///     Inputs are assumed to be correct (no validation performed).
  [[nodiscard]] std::int64_t count_false(std::span<const std::int64_t> lo,
                                         std::span<const std::int64_t> hi) const noexcept {
    return corners(0, 0, lo, hi);
  }

  /// Report whether the mask is true on every cell of the box `[lo, hi)`.
  ///
  /// Note:
  ///     Inputs are assumed to be correct (no validation performed).
  [[nodiscard]] bool all_true(std::span<const std::int64_t> lo,
                              std::span<const std::int64_t> hi) const noexcept {
    return count_false(lo, hi) == 0;
  }

  /// Report whether the mask is true on at least one cell of the box `[lo, hi)`.
  ///
  /// Note:
  ///     Inputs are assumed to be correct (no validation performed).
  [[nodiscard]] bool any_true(std::span<const std::int64_t> lo,
                              std::span<const std::int64_t> hi) const noexcept {
    std::int64_t volume = 1;
    for (std::size_t k = 0; k < lo.size(); ++k) {
      volume *= hi[k] - lo[k];
    }
    return count_false(lo, hi) < volume;
  }

 private:
  // The box sum is the alternating sum over the corners, `+` for an upper bound and `-` for a
  // lower one in each axis; recursing over the axes keeps it free of a per-query allocation and
  // of any bound on the dimension.
  [[nodiscard]] std::int64_t corners(std::size_t axis, std::size_t offset,
                                     std::span<const std::int64_t> lo,
                                     std::span<const std::int64_t> hi) const noexcept {
    if (axis == stride_.size()) {
      return table_[offset];
    }
    return corners(axis + 1, offset + static_cast<std::size_t>(hi[axis]) * stride_[axis], lo, hi) -
           corners(axis + 1, offset + static_cast<std::size_t>(lo[axis]) * stride_[axis], lo, hi);
  }

  std::vector<std::int64_t> table_;
  std::vector<std::size_t> stride_;
};

}  // namespace pantr::thb::detail
