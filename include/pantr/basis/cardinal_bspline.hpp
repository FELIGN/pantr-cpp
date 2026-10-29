/// @file cardinal_bspline.hpp
/// @brief The central cardinal B-spline basis on `[0, 1]`.
#pragma once

#include "pantr/basis/detail/tabulate_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"

#include <concepts>
#include <cstddef>
#include <span>

namespace pantr::basis {

namespace detail {

/// Write the cardinal B-spline values of degree `degree` at every point into `out`.
///
/// The `degree + 1` B-splines nonzero on the central unit span `[0, 1]` of the uniform knot
/// vector with unit spacing, by the Cox-de Boor recurrence (Piegl and Tiller's BasisFuns)
/// specialised to that span. Intermediates in double, each value stored in `T`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `tabulate_cardinal_bspline_1d` instead.
template <std::floating_point T>
void tabulate_cardinal_bspline_1d_kernel(int degree, std::span<const T> pts, mdview<T, 2> out) {
  T* data = out.data_handle();
  const std::size_t stride = static_cast<std::size_t>(degree) + 1;
  const double ops = 4.0 * static_cast<double>(stride * stride);
  detail::for_each_point(pts.size(), ops, [&](std::size_t j) {
    T* row = data + j * stride;
    row[0] = static_cast<T>(1);
    const double one_minus_u = 1.0 - static_cast<double>(pts[j]);
    for (int k = 1; k <= degree; ++k) {
      const double inv_k = 1.0 / k;
      double saved = 0.0;
      for (int r = 0; r < k; ++r) {
        const auto nr_old = static_cast<double>(row[r]);
        const double term = (r + one_minus_u) * inv_k;
        row[r] = static_cast<T>(saved + nr_old * term);
        saved = nr_old * (1.0 - term);
      }
      row[k] = static_cast<T>(saved);
    }
  });
}

}  // namespace detail

/// Evaluate the central cardinal B-spline basis of degree `degree` at `pts` into `out`.
///
/// These are the B-splines of maximal smoothness on a uniform unit knot vector that are nonzero on
/// its central span, taken as `[0, 1]`: the basis of the inner spans of a maximally regular uniform
/// spline. The polynomial pieces on `[0, 1]` are evaluated at every point, inside or not.
///
/// Args:
///     degree (int): Degree, `>= 0`.
///     pts (std::span<const T>): The points.
///     out (mdview<T, 2>): Output, extents `(pts.size(), degree + 1)`.
///
/// Raises:
///     value_error: `degree < 0`, or `out` has other extents.
template <std::floating_point T>
void tabulate_cardinal_bspline_1d(int degree, std::span<const T> pts, mdview<T, 2> out) {
  detail::check_degree(degree);
  detail::check_out<T, 2>(out, {pts.size(), static_cast<std::size_t>(degree) + 1});
  detail::tabulate_cardinal_bspline_1d_kernel<T>(degree, pts, out);
}

/// Evaluate the central cardinal B-spline basis of degree `degree` at `pts`.
///
/// Returns:
///     NdArray<T>: Extents `(pts.size(), degree + 1)`; see the `out` overload.
///
/// Raises:
///     value_error: `degree < 0`.
template <std::floating_point T>
[[nodiscard]] NdArray<T> tabulate_cardinal_bspline_1d(int degree, std::span<const T> pts) {
  detail::check_degree(degree);
  NdArray<T> out({pts.size(), static_cast<std::size_t>(degree) + 1});
  detail::tabulate_cardinal_bspline_1d_kernel<T>(degree, pts, out.template view<2>());
  return out;
}

}  // namespace pantr::basis
