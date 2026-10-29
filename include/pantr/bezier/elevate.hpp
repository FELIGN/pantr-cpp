/// @file elevate.hpp
/// @brief Degree elevation of a `Bezier`: the same map in a Bernstein basis of higher degree.
#pragma once

#include "pantr/bezier/bezier.hpp"
#include "pantr/bezier/detail/along_axis.hpp"
#include "pantr/bezier/detail/kernels_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/binomial.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bezier {

/// Degree-elevate the columns of a `(p + 1, n_cols)` array of control points.
///
/// Each column is the control polygon of a degree-`p` Bezier; the result holds the control
/// polygons of the same polynomials in the Bernstein basis of degree `p + increment`.
///
/// Args:
///     degree (std::size_t): The degree `p`.
///     ctrl (mdview<const T, 2>): Control points, extents `(p + 1, n_cols)`.
///     increment (std::size_t): The degrees to add, `>= 1`.
///
/// Returns:
///     NdArray<T>: Elevated control points, extents `(p + increment + 1, n_cols)`.
///
/// Raises:
///     value_error: `ctrl.extent(0) != degree + 1`, `increment == 0`, or `degree + increment`
///         beyond `bincoeff_max_n`.
template <StorageFloat T>
[[nodiscard]] NdArray<T> elevate_degree_1d(std::size_t degree, mdview<const T, 2> ctrl,
                                           std::size_t increment) {
  if (ctrl.extent(0) != degree + 1) {
    throw value_error("ctrl has " + std::to_string(ctrl.extent(0)) + " rows, but a degree-" +
                      std::to_string(degree) + " Bézier has " + std::to_string(degree + 1) + ".");
  }
  if (increment == 0) {
    throw value_error("The degree increment must be positive.");
  }
  const std::size_t elevated = degree + increment;
  check_bincoeff_envelope(static_cast<long long>(elevated),
                          "Degree elevation to degree " + std::to_string(elevated));
  NdArray<T> out({elevated + 1, ctrl.extent(1)}, for_overwrite);
  detail::elevate_degree_1d_kernel<T>(degree, increment, ctrl, out.template view<2>());
  return out;
}

/// Degree-elevate a Bezier in one or more parametric directions.
///
/// The elevated Bezier represents the same map (a rational one the same projected map). Each
/// direction with a positive increment is elevated in turn along its axis, the weight column of a
/// rational Bezier included.
///
/// Args:
///     b (const Bezier<T>&): The Bezier.
///     increments (std::span<const long long>): The degrees to add per direction, `dim` values
///         `>= 0`, at least one positive.
///
/// Returns:
///     Bezier<T>: The elevated Bezier.
///
/// Raises:
///     value_error: `increments` has the wrong length, a negative entry, or no positive entry
///         (checked in that order), or an elevated degree exceeds `bincoeff_max_n`.
template <StorageFloat T>
[[nodiscard]] Bezier<T> elevate_degree(const Bezier<T>& b, std::span<const long long> increments) {
  if (increments.size() != b.dim()) {
    throw value_error("Number of degree increments (" + std::to_string(increments.size()) +
                      ") must match dimension (" + std::to_string(b.dim()) + ").");
  }
  bool any_positive = false;
  for (const long long inc : increments) {
    if (inc < 0) {
      throw value_error("Degree increments must be non-negative.");
    }
    any_positive = any_positive || inc > 0;
  }
  if (!any_positive) {
    throw value_error("At least one degree increment must be positive.");
  }
  // The coefficient table is built from C(p + inc, i), so the elevated degree is what has to stay
  // in range. The sum saturates at the largest `long long`: past `bincoeff_max_n` the exact value
  // only reaches the message, and no array of such a degree can exist.
  for (std::size_t d = 0; d < b.dim(); ++d) {
    if (increments[d] > 0) {
      const auto deg = static_cast<long long>(b.degree(d));
      const long long elevated = increments[d] > (std::numeric_limits<long long>::max)() - deg
                                     ? (std::numeric_limits<long long>::max)()
                                     : deg + increments[d];
      check_bincoeff_envelope(elevated, "Degree elevation to degree " + std::to_string(elevated) +
                                            " in direction " + std::to_string(d));
    }
  }
  std::vector<std::size_t> shape(b.shape().begin(), b.shape().end());
  std::optional<NdArray<T>> current;
  for (std::size_t d = 0; d < b.dim(); ++d) {
    if (increments[d] == 0) {
      continue;
    }
    const std::size_t p = shape[d] - 1;
    const auto inc = static_cast<std::size_t>(increments[d]);
    const std::span<const T> src = current ? current->flat() : b.control_points();
    const NdArray<T> flat = detail::flatten_along_axis<T>(src, shape, d);
    NdArray<T> elevated({p + inc + 1, flat.shape()[1]}, for_overwrite);
    detail::elevate_degree_1d_kernel<T>(p, inc, flat.template view<2>(),
                                        elevated.template view<2>());
    current = detail::unflatten_along_axis<T>(elevated.template view<2>(), shape, d);
    shape[d] = p + inc + 1;
  }
  return Bezier<T>(std::move(*current), b.rational());
}

}  // namespace pantr::bezier
