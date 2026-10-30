/// @file blossom.hpp
/// @brief The blossom (polar form) of a 1D B-spline, by the generalized de Boor algorithm.
///
/// The blossom `f[u_0, ..., u_{p-1}]` is the unique symmetric multi-affine function of `p`
/// arguments whose diagonal is the curve, `f[t, ..., t] = f(t)`. It has no public Python API: it is
/// bound in the private test module for the tests that check the algorithm against its defining
/// properties.
#pragma once

#include "pantr/bspline/detail/refine_common.hpp"
#include "pantr/bspline/knots.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/scalar.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace pantr::bspline {

namespace detail {

/// Return the knot span the blossom's de Boor recurrence starts from.
///
/// The position of the last knot not exceeding `u_last`, found by binary search. For a value inside
/// a non-empty span this is exactly what a scan for the first `idx` with `knots[idx] <= u_last <
/// knots[idx + 1]` returns, repeated knots included, because that first match is the last
/// occurrence of the span's left knot. Both out-of-span cases return the rightmost span `n`, which
/// is what the linear scan this replaced also returned for `u_last` at or beyond the last knot and
/// for `u_last` below the first knot: the public entry point accepts parameters up to its
/// tolerance outside the domain, so the second case is reachable. The clamp also keeps the index in
/// range for a parameter past the domain of a non-clamped knot vector. A NaN is ordered after every
/// knot, as NumPy orders it, so it lands past the last span.
///
/// Args:
///     knots (std::span<const T>): Non-decreasing knot vector.
///     u_last (T): The largest of the blossom's parameter values.
///     n (std::ptrdiff_t): Index of the last control point.
///
/// Returns:
///     std::ptrdiff_t: The span index in `[0, n]`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] std::ptrdiff_t blossom_span(std::span<const T> knots, T u_last, std::ptrdiff_t n) {
  if (std::isnan(u_last)) {
    return n;
  }
  const auto above = std::upper_bound(knots.begin(), knots.end(), u_last);
  const std::ptrdiff_t span = (above - knots.begin()) - 1;
  if (span < 0 || span > n) {
    return n;
  }
  return span;
}

/// Evaluate the blossom of a 1D B-spline at `degree` parameter values already sorted ascending.
///
/// `d[j]` starts as the control point `P[k - p + j]` of the span `k` of the largest value; level
/// `l` uses `u_values[l - 1]`. The weight `alpha` and the convex combination are formed in
/// `double`, as the Numba kernel this replaces formed them (its `0.0` literal made the weight
/// `float64`), and the result rounded to the storage type when stored.
///
/// Args:
///     knots (std::span<const T>): Knot vector, `n + degree + 2` entries.
///     degree (int): Degree `p`.
///     control_points (mdview<const T, 2>): Control points, extents `(n + 1, rank)`.
///     u_values (std::span<const T>): `degree` values, ascending.
///
/// Returns:
///     std::vector<T>: The blossom value, `rank` entries.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): `u_values` has `degree`
///     entries in ascending order, and the knots and points agree in size.
template <StorageFloat T>
[[nodiscard]] std::vector<T> blossom_core(std::span<const T> knots, int degree,
                                          mdview<const T, 2> control_points,
                                          std::span<const T> u_values) {
  const std::size_t rank = control_points.extent(1);
  const auto n = static_cast<std::ptrdiff_t>(control_points.extent(0)) - 1;
  const auto p = static_cast<std::ptrdiff_t>(degree);

  if (degree == 0) {
    // By convention the last active knot span.
    return std::vector<T>(control_points.data_handle() + static_cast<std::size_t>(n) * rank,
                          control_points.data_handle() + static_cast<std::size_t>(n + 1) * rank);
  }

  // The span of the largest value, at least `degree`: on a non-clamped knot vector a value within
  // the tolerance below the domain start lands on a span left of the first one, whose window
  // would reach before the first control point (the Numba kernel read the last one there, by
  // negative indexing). Such a value belongs to the first span.
  const std::ptrdiff_t k =
      (std::max)(blossom_span<T>(knots, u_values[static_cast<std::size_t>(p - 1)], n), p);
  const auto width = static_cast<std::size_t>(p) + 1;

  // Local copy d[j] = P[k - p + j], j = 0..p.
  std::vector<T> d(width * rank);
  for (std::size_t j = 0; j < width; ++j) {
    const auto idx = static_cast<std::size_t>(k - p + static_cast<std::ptrdiff_t>(j));
    for (std::size_t r = 0; r < rank; ++r) {
      d[j * rank + r] = control_points(idx, r);
    }
  }

  for (std::ptrdiff_t level = 1; level <= p; ++level) {
    const T u = u_values[static_cast<std::size_t>(level - 1)];
    for (std::ptrdiff_t j = p; j >= level; --j) {
      const auto global_j = static_cast<std::size_t>(k - p + j);
      const T denom = knots[global_j + static_cast<std::size_t>(p + 1 - level)] - knots[global_j];
      double alpha = 0.0;
      if (denom > static_cast<T>(0)) {
        alpha = static_cast<double>((u - knots[global_j]) / denom);
      }
      for (std::size_t r = 0; r < rank; ++r) {
        const auto lo = static_cast<double>(d[static_cast<std::size_t>(j - 1) * rank + r]);
        const auto hi = static_cast<double>(d[static_cast<std::size_t>(j) * rank + r]);
        d[static_cast<std::size_t>(j) * rank + r] = static_cast<T>((1.0 - alpha) * lo + alpha * hi);
      }
    }
  }
  return std::vector<T>(d.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(p) * rank),
                        d.end());
}

}  // namespace detail

/// Evaluate the blossom of a 1D B-spline at `degree` parameter values.
///
/// Computes the symmetric multi-affine polar form `f[u_0, ..., u_{p-1}]` by the generalized de Boor
/// algorithm. The diagonal property gives `f[t, ..., t] == f(t)`, and the blossom is symmetric in
/// its arguments.
///
/// Args:
///     knots (std::span<const T>): Knot vector, `n + degree + 2` entries.
///     degree (int): Degree `p`.
///     control_points (mdview<const T, 2>): Control points, extents `(n + 1, rank)`.
///     u_values (std::span<const T>): Exactly `p` parameter values; need not be sorted, and may
///         coincide.
///     tol (double): Absolute parametric tolerance for the domain-membership test.
///
/// Returns:
///     std::vector<T>: The blossom value, `rank` entries.
///
/// Raises:
///     value_error: `u_values.size() != degree`; an entry lies outside
///         `[knots[degree], knots[size - degree - 1]]` by more than `tol`; or the control points do
///         not match the knots.
template <StorageFloat T>
[[nodiscard]] std::vector<T> blossom(std::span<const T> knots, int degree,
                                     mdview<const T, 2> control_points, std::span<const T> u_values,
                                     double tol) {
  if (static_cast<long long>(u_values.size()) != degree) {
    throw value_error("u_values must have length degree=" + std::to_string(degree) + ", got " +
                      std::to_string(u_values.size()) + ".");
  }
  detail::check_net_matches("blossom", control_points.extent(0), knots.size(), degree);
  if (knots.size() < 2 * static_cast<std::size_t>(degree) + 2) {
    throw value_error("knots must have at least 2*degree+2 elements");
  }

  if (degree == 0) {
    return detail::blossom_core<T>(knots, degree, control_points, u_values);
  }

  const auto d = static_cast<std::size_t>(degree);
  const auto a = static_cast<double>(knots[d]);
  const auto b = static_cast<double>(knots[knots.size() - d - 1]);
  for (const T u : u_values) {
    const auto v = static_cast<double>(u);
    if (v < a - tol || v > b + tol) {
      throw value_error("u_values entry " + py_repr(v) + " is outside domain [" + py_repr(a) +
                        ", " + py_repr(b) + "] (tol=" + py_repr(tol) + ").");
    }
  }

  // Ascending, a NaN last (NumPy's order): the kernel relies on ascending order to find the span.
  std::vector<T> sorted(u_values.begin(), u_values.end());
  std::sort(sorted.begin(), sorted.end(),
            [](T x, T y) { return std::isnan(y) ? !std::isnan(x) : x < y; });
  return detail::blossom_core<T>(knots, degree, control_points, sorted);
}

}  // namespace pantr::bspline
