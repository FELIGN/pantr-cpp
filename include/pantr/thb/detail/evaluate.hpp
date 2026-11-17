/// @file evaluate.hpp
/// @brief Kernels that evaluate one hierarchical function at one point from per-direction 1D
/// basis tabulations.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed). For general use, call
///     `pantr::thb::THBSplineSpace::tabulate_basis`.
#pragma once

#include "pantr/config.hpp"
#include "pantr/thb/types.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace pantr::thb::detail {

/// The 1D basis tabulation of one level and direction at every query point: the `degree + 1`
/// non-zero functions' values, or their derivative of one order, and the first of those functions.
///
/// Attributes:
///     data (std::vector<double>): The kernel's output; point `i`'s values start at
///         `i * stride + offset`.
///     first (std::vector<std::int64_t>): Per point, the index of the first non-zero function.
///     stride (std::size_t): Doubles between consecutive points' blocks in `data`.
///     offset (std::size_t): Position of the wanted order within a point's block.
///     degree (std::int64_t): The direction's degree.
struct Basis1D {
  std::vector<double> data;
  std::vector<std::int64_t> first;
  std::size_t stride = 0;
  std::size_t offset = 0;
  std::int64_t degree = 0;

  /// Get the `degree + 1` values at point `pt`.
  [[nodiscard]] const double* values(std::size_t pt) const noexcept {
    return data.data() + pt * stride + offset;
  }
};

/// Evaluate an untruncated function (a tensor-product B-spline) at one point.
///
/// The product runs over the directions in order, from `1.0`, and stops at the first direction in
/// which the function is not supported at the point (its local index outside `[0, degree]`),
/// returning `0.0`.
///
/// Args:
///     bases (const Basis1D* const*): One tabulation per direction, at the function's level.
///     dim (std::size_t): Number of directions.
///     pt (std::size_t): Index of the point.
///     multi (std::span<const std::int64_t>): The function's per-direction index.
///
/// Returns:
///     double: The product, or its mixed partial when the tabulations hold derivatives.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
[[nodiscard]] inline double tensor_product_value(const Basis1D* const* bases, std::size_t dim,
                                                 std::size_t pt,
                                                 std::span<const std::int64_t> multi) noexcept {
  double product = 1.0;
  for (std::size_t k = 0; k < dim; ++k) {
    const std::int64_t local = multi[k] - bases[k]->first[pt];
    if (local < 0 || local > bases[k]->degree) {
      return 0.0;
    }
    product *= bases[k]->values(pt)[local];
  }
  return product;
}

/// Evaluate a truncated function at one point: `sum_multi coeffs[multi] * prod_k
/// factor_k[multi_k]`.
///
/// `factor_k[j]` is the 1D value of the level-`rep_level` function `box_lo[k] + j` at the point,
/// zero where that function is not supported there. The sum contracts the coefficient box one
/// direction at a time, last direction first, adding in increasing index within each; the order is
/// fixed, so the value does not depend on how the work is scheduled.
///
/// Args:
///     f (const TruncatedFunction&): The function.
///     bases (const Basis1D* const*): One tabulation per direction, at `f.rep_level`.
///     pt (std::size_t): Index of the point.
///     work (double*): Scratch of at least `sum(box_shape) + coeffs.size()` doubles.
///
/// Returns:
///     double: The function's value, or its mixed partial when the tabulations hold derivatives.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
[[nodiscard]] inline double truncated_value(const TruncatedFunction& f, const Basis1D* const* bases,
                                            std::size_t pt, double* work) noexcept {
  const std::size_t dim = f.box_shape.size();
  double* factors = work;
  std::size_t total_width = 0;
  for (std::size_t k = 0; k < dim; ++k) {
    const Basis1D& b = *bases[k];
    for (std::size_t j = 0; j < f.box_shape[k]; ++j) {
      const std::int64_t local = f.box_lo[k] + static_cast<std::int64_t>(j) - b.first[pt];
      factors[total_width + j] = (local >= 0 && local <= b.degree) ? b.values(pt)[local] : 0.0;
    }
    total_width += f.box_shape[k];
  }
  double* buffer = work + total_width;
  const double* source = f.coeffs.data();
  std::size_t size = f.coeffs.size();
  std::size_t width_end = total_width;
  for (std::size_t k = dim; k-- > 0;) {
    const std::size_t width = f.box_shape[k];
    const double* factor = factors + (width_end - width);
    const std::size_t outer = size / width;
    for (std::size_t o = 0; o < outer; ++o) {
      double sum = 0.0;
      for (std::size_t j = 0; j < width; ++j) {
        sum += source[o * width + j] * factor[j];
      }
      buffer[o] = sum;
    }
    source = buffer;
    size = outer;
    width_end -= width;
  }
  return source[0];
}

}  // namespace pantr::thb::detail
