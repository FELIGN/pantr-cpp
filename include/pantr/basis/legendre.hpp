/// @file legendre.hpp
/// @brief The orthonormal shifted Legendre basis on `[0, 1]`.
#pragma once

#include "pantr/basis/detail/tabulate_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>
#include <vector>

namespace pantr::basis {

namespace detail {

/// Operations one point of the Legendre tabulation costs: about four per value of the
/// three-term recurrence.
inline double legendre_ops_per_point(int degree) {
  return 4.0 * (static_cast<double>(degree) + 1.0);
}

/// Write the orthonormal shifted Legendre values of degree `degree` at every point into `out`.
///
/// `p_0 = 1`, `p_1 = sqrt(3) (2x - 1)`, and for `i >= 2`
/// `p_i = a_i (2x - 1) p_{i-1} - b_i p_{i-2}` with `a_i = sqrt(2i - 1) sqrt(2i + 1) / i` and
/// `b_i = ((i - 1) / i) sqrt(2i + 1) / sqrt(2i - 3)`, the coefficients computed once.
/// Intermediates in double, each value stored in `T`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `tabulate_legendre_1d` instead.
template <StorageFloat T>
void tabulate_legendre_1d_kernel(int degree, std::span<const T> pts, mdview<T, 2> out) {
  T* data = out.data_handle();
  const std::size_t stride = static_cast<std::size_t>(degree) + 1;
  if (degree == 0) {
    detail::for_each_point(pts.size(), 1.0, sizeof(T),
                           [&](std::size_t j) { data[j] = static_cast<T>(1); });
    return;
  }
  std::vector<double> a(stride, 0.0), b(stride, 0.0);
  for (int i = 2; i <= degree; ++i) {
    const auto id = static_cast<double>(i);
    const double s_m1 = std::sqrt(2.0 * id - 1.0);
    const double s_p1 = std::sqrt(2.0 * id + 1.0);
    const double s_m3 = std::sqrt(2.0 * id - 3.0);
    a[static_cast<std::size_t>(i)] = (s_m1 * s_p1) / id;
    b[static_cast<std::size_t>(i)] = ((id - 1.0) / id) * (s_p1 / s_m3);
  }
  const double sqrt3 = std::sqrt(3.0);
  detail::for_each_point(pts.size(), legendre_ops_per_point(degree), sizeof(T) * stride,
                         [&](std::size_t j) {
                           T* row = data + j * stride;
                           const double x = 2.0 * static_cast<double>(pts[j]) - 1.0;
                           row[0] = static_cast<T>(1);
                           row[1] = static_cast<T>(sqrt3 * x);
                           for (std::size_t i = 2; i < stride; ++i) {
                             row[i] = static_cast<T>(a[i] * x * static_cast<double>(row[i - 1]) -
                                                     b[i] * static_cast<double>(row[i - 2]));
                           }
                         });
}

}  // namespace detail

/// Evaluate the orthonormal shifted Legendre basis of degree `degree` at `pts` into `out`.
///
/// Orthonormal on `[0, 1]`: the integral of `p_n p_m` over `[0, 1]` is `delta_nm`.
///
/// Args:
///     degree (int): Degree, `>= 0`.
///     pts (std::span<const T>): The points.
///     out (mdview<T, 2>): Output, extents `(pts.size(), degree + 1)`.
///
/// Raises:
///     value_error: `degree < 0`, or `out` has other extents.
template <StorageFloat T>
void tabulate_legendre_1d(int degree, std::span<const T> pts, mdview<T, 2> out) {
  detail::check_degree(degree);
  detail::check_out<T, 2>(out, {pts.size(), static_cast<std::size_t>(degree) + 1});
  detail::tabulate_legendre_1d_kernel<T>(degree, pts, out);
}

/// Evaluate the orthonormal shifted Legendre basis of degree `degree` at `pts`.
///
/// Returns:
///     NdArray<T>: Extents `(pts.size(), degree + 1)`; see the `out` overload.
///
/// Raises:
///     value_error: `degree < 0`.
template <StorageFloat T>
[[nodiscard]] NdArray<T> tabulate_legendre_1d(int degree, std::span<const T> pts) {
  detail::check_degree(degree);
  NdArray<T> out({pts.size(), static_cast<std::size_t>(degree) + 1}, for_overwrite);
  detail::tabulate_legendre_1d_kernel<T>(degree, pts, out.template view<2>());
  return out;
}

}  // namespace pantr::basis
