/// @file derivative.hpp
/// @brief The derivative of a `Bezier` in one parametric direction, as a `Bezier` (the hodograph).
#pragma once

#include "pantr/bezier/bezier.hpp"
#include "pantr/bezier/detail/along_axis.hpp"
#include "pantr/bezier/product.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bezier {

namespace detail {

/// Derivative control points of the columns of a `(p + 1, n_cols)` array: `Q_i = p (P_{i+1} -
/// P_i)`, `(p, n_cols)`, in `T`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed), `p >= 1` included.
template <StorageFloat T>
[[nodiscard]] NdArray<T> derivative_1d(mdview<const T, 2> ctrl) {
  const std::size_t p = ctrl.extent(0) - 1;
  const std::size_t n_cols = ctrl.extent(1);
  NdArray<T> out({p, n_cols}, for_overwrite);
  const T pt = static_cast<T>(p);
  for (std::size_t i = 0; i < p; ++i) {
    for (std::size_t c = 0; c < n_cols; ++c) {
      out.data()[i * n_cols + c] = pt * (ctrl(i + 1, c) - ctrl(i, c));
    }
  }
  return out;
}

/// Derivative control points that keep the degree: the derivative `p (P_{i+1} - P_i)` fused with
/// one degree elevation, `R_j = (p - j) (P_{j+1} - P_j) + j (P_j - P_{j-1})` for `0 <= j <= p`
/// (the ends reduce to `R_0 = p (P_1 - P_0)` and `R_p = p (P_p - P_{p-1})`), `(p + 1, n_cols)`, in
/// `T`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed), `p >= 1` included.
template <StorageFloat T>
[[nodiscard]] NdArray<T> derivative_keep_degree_1d(mdview<const T, 2> ctrl) {
  const std::size_t p = ctrl.extent(0) - 1;
  const std::size_t n_cols = ctrl.extent(1);
  NdArray<T> out({p + 1, n_cols}, for_overwrite);
  T* d = out.data();
  const T pt = static_cast<T>(p);
  for (std::size_t c = 0; c < n_cols; ++c) {
    d[c] = pt * (ctrl(1, c) - ctrl(0, c));
    d[p * n_cols + c] = pt * (ctrl(p, c) - ctrl(p - 1, c));
  }
  for (std::size_t j = 1; j < p; ++j) {
    const T lo = static_cast<T>(p - j);
    const T hi = static_cast<T>(j);
    for (std::size_t c = 0; c < n_cols; ++c) {
      d[j * n_cols + c] = lo * (ctrl(j + 1, c) - ctrl(j, c)) + hi * (ctrl(j, c) - ctrl(j - 1, c));
    }
  }
  return out;
}

/// Apply a 1D control-point derivative along one axis of an N-D array of control points.
///
/// Args:
///     data (std::span<const T>): Control points over `shape`.
///     shape (std::span<const std::size_t>): The extents, the axis one among them.
///     axis (std::size_t): The axis to differentiate, with extent `>= 2`.
///     keep_degree (bool): Whether to keep the degree (`derivative_keep_degree_1d`) or lower it.
///
/// Returns:
///     NdArray<T>: The derivative control points, `shape` with `[axis]` lowered by one or kept.
template <StorageFloat T>
[[nodiscard]] NdArray<T> derivative_along_axis(std::span<const T> data,
                                               std::span<const std::size_t> shape, std::size_t axis,
                                               bool keep_degree) {
  const NdArray<T> flat = flatten_along_axis<T>(data, shape, axis);
  const NdArray<T> d = keep_degree ? derivative_keep_degree_1d<T>(flat.template view<2>())
                                   : derivative_1d<T>(flat.template view<2>());
  return unflatten_along_axis<T>(d.template view<2>(), shape, axis);
}

/// Copy columns `[first, last)` of control points `(..., cp)` into a Bezier of its own, with
/// the same parametric extents.
template <StorageFloat T>
[[nodiscard]] Bezier<T> column_range(const Bezier<T>& b, std::size_t first, std::size_t last) {
  const std::size_t cp = b.cp_size();
  const std::size_t rows = b.size() / cp;
  std::vector<std::size_t> shape(b.shape().begin(), b.shape().end());
  shape.back() = last - first;
  NdArray<T> out(std::move(shape), for_overwrite);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = first; c < last; ++c) {
      out.data()[r * (last - first) + (c - first)] = b.control_points()[r * cp + c];
    }
  }
  return Bezier<T>(std::move(out), false);
}

/// Repeat the single column of a rank-1 Bezier `rank` times.
template <StorageFloat T>
[[nodiscard]] Bezier<T> tile_scalar(const Bezier<T>& w, std::size_t rank) {
  const std::size_t rows = w.size();
  std::vector<std::size_t> shape(w.shape().begin(), w.shape().end());
  shape.back() = rank;
  NdArray<T> out(std::move(shape), for_overwrite);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < rank; ++c) {
      out.data()[r * rank + c] = w.control_points()[r];
    }
  }
  return Bezier<T>(std::move(out), false);
}

}  // namespace detail

/// Compute the first partial derivative of a Bezier in one parametric direction, as a Bezier.
///
/// For a non-rational Bezier of degree `p` in the direction the result has degree `p - 1`, or `p`
/// when `keep_degree` (the derivative fused with a degree elevation). For a rational Bezier
/// `f = A / w` the quotient rule `f' = (A' w - A w') / w^2` gives a rational Bezier whose
/// numerator and weight have degree `2 p` in the direction, and `keep_degree` is not consulted:
/// the degree-preserving derivatives of `A` and `w` are used in the products, so that `A' w`, `A
/// w'` and `w^2` share one degree.
///
/// Args:
///     b (const Bezier<T>&): The Bezier.
///     direction (long long): The parametric direction, in `[0, dim)`.
///     keep_degree (bool): Whether a non-rational result keeps the degree of `b`.
///
/// Returns:
///     Bezier<T>: The derivative.
///
/// Raises:
///     value_error: `direction` is out of range, or the degree in that direction is zero (checked
///         in that order).
///     std::overflow_error: A binomial coefficient of the product degree exceeds the `double`
///         range (rational case).
template <StorageFloat T>
[[nodiscard]] Bezier<T> derivative(const Bezier<T>& b, long long direction, bool keep_degree) {
  if (direction < 0 || direction >= static_cast<long long>(b.dim())) {
    throw value_error("direction must be in [0, " + std::to_string(b.dim()) + "), got " +
                      std::to_string(direction) + ".");
  }
  const auto axis = static_cast<std::size_t>(direction);
  if (b.degree(axis) < 1) {
    throw value_error("Derivative of a degree-0 Bézier is not defined.");
  }
  if (!b.rational()) {
    return Bezier<T>(
        detail::derivative_along_axis<T>(b.control_points(), b.shape(), axis, keep_degree), false);
  }
  const std::size_t rank = b.rank();
  const Bezier<T> a_part = detail::column_range(b, 0, rank);
  const Bezier<T> w_part = detail::column_range(b, rank, rank + 1);
  // Degree-preserving derivatives: A' and w' keep degree p, so the products below all have 2 p.
  const Bezier<T> a_prime(
      detail::derivative_along_axis<T>(a_part.control_points(), a_part.shape(), axis, true), false);
  const Bezier<T> w_prime(
      detail::derivative_along_axis<T>(w_part.control_points(), w_part.shape(), axis, true), false);
  const Bezier<T> num1 = multiply(a_prime, detail::tile_scalar(w_part, rank));
  const Bezier<T> num2 = multiply(a_part, detail::tile_scalar(w_prime, rank));
  const Bezier<T> denom = multiply(w_part, w_part);
  const std::size_t rows = denom.size();
  std::vector<std::size_t> shape(denom.shape().begin(), denom.shape().end());
  shape.back() = rank + 1;
  NdArray<T> out(std::move(shape), for_overwrite);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < rank; ++c) {
      out.data()[r * (rank + 1) + c] =
          num1.control_points()[r * rank + c] - num2.control_points()[r * rank + c];
    }
    out.data()[r * (rank + 1) + rank] = denom.control_points()[r];
  }
  return Bezier<T>(std::move(out), true);
}

}  // namespace pantr::bezier
