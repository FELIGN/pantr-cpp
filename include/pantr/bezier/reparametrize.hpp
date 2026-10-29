/// @file reparametrize.hpp
/// @brief Changes of parametrization and of space that move a `Bezier`'s control points without
///        changing its image: reversing a direction, permuting the directions, an affine map.
#pragma once

#include "pantr/bezier/bezier.hpp"
#include "pantr/bezier/detail/along_axis.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bezier {

/// Reverse the orientation of one parametric direction.
///
/// The control points are flipped along `direction`, so the result is `b` with `1 - t` in place of
/// `t` in that direction. A copy of the values: nothing is rounded.
///
/// Args:
///     b (const Bezier<T>&): The Bezier.
///     direction (long long): The direction to reverse, in `[0, dim)`.
///
/// Returns:
///     Bezier<T>: A Bezier `r` with `r(t) = b(t')`, `t'_direction = 1 - t_direction`, the other
///     parameters unchanged.
///
/// Raises:
///     value_error: `direction` is out of range.
template <StorageFloat T>
[[nodiscard]] Bezier<T> reversed(const Bezier<T>& b, long long direction) {
  if (direction < 0 || direction >= static_cast<long long>(b.dim())) {
    throw value_error("direction must be in [0, " + std::to_string(b.dim()) + "), got " +
                      std::to_string(direction) + ".");
  }
  const auto d = static_cast<std::size_t>(direction);
  const detail::AxisSplit s = detail::split_around(b.shape(), d);
  const std::size_t n = b.shape()[d];
  NdArray<T> out(std::vector<std::size_t>(b.shape().begin(), b.shape().end()), for_overwrite);
  const T* src = b.control_points().data();
  T* dst = out.data();
  for (std::size_t o = 0; o < s.outer; ++o) {
    for (std::size_t a = 0; a < n; ++a) {
      std::copy_n(src + (o * n + a) * s.inner, s.inner, dst + (o * n + (n - 1 - a)) * s.inner);
    }
  }
  return Bezier<T>(std::move(out), b.rational());
}

/// Reorder the parametric directions.
///
/// The new direction `k` is the old direction `permutation[k]`: for `[1, 2, 0]` on a volume the
/// old direction 1 becomes the new 0, the old 2 the new 1 and the old 0 the new 2. A copy of the
/// values: nothing is rounded; the coordinate (and weight) axis stays last.
///
/// Args:
///     b (const Bezier<T>&): The Bezier.
///     permutation (std::span<const long long>): A permutation of `0..dim - 1`.
///
/// Returns:
///     Bezier<T>: A Bezier `r` with `r(t_0, ..., t_{dim-1}) = b(s)` where `s_{permutation[k]} =
///     t_k`.
///
/// Raises:
///     value_error: `permutation` is not a permutation of `0..dim - 1`.
template <StorageFloat T>
[[nodiscard]] Bezier<T> permuted(const Bezier<T>& b, std::span<const long long> permutation) {
  const std::size_t dim = b.dim();
  bool valid = permutation.size() == dim;
  std::vector<bool> seen(dim, false);
  for (std::size_t k = 0; valid && k < dim; ++k) {
    const long long p = permutation[k];
    valid = p >= 0 && p < static_cast<long long>(dim) && !seen[static_cast<std::size_t>(p)];
    if (valid) {
      seen[static_cast<std::size_t>(p)] = true;
    }
  }
  if (!valid) {
    throw value_error("permutation must be a permutation of range(" + std::to_string(dim) + ")");
  }
  const std::span<const std::size_t> shape = b.shape();
  const std::size_t cp = b.cp_size();
  // Row-major strides of the old control points, in units of rows of `cp` values.
  std::vector<std::size_t> old_stride(dim, 1);
  for (std::size_t d = dim - 1; d-- > 0;) {
    old_stride[d] = old_stride[d + 1] * shape[d + 1];
  }
  std::vector<std::size_t> new_shape(dim + 1, cp);
  std::vector<std::size_t> stride(dim);  // old stride of the old direction that becomes `k`
  for (std::size_t k = 0; k < dim; ++k) {
    const auto old_dir = static_cast<std::size_t>(permutation[k]);
    new_shape[k] = shape[old_dir];
    stride[k] = old_stride[old_dir];
  }
  NdArray<T> out(new_shape, for_overwrite);
  const std::size_t rows = out.size() / cp;
  const T* src = b.control_points().data();
  T* dst = out.data();
  std::vector<std::size_t> idx(dim, 0);
  for (std::size_t row = 0; row < rows; ++row) {
    std::size_t offset = 0;
    for (std::size_t k = 0; k < dim; ++k) {
      offset += idx[k] * stride[k];
    }
    std::copy_n(src + offset * cp, cp, dst + row * cp);
    for (std::size_t k = dim; k-- > 0;) {
      if (++idx[k] < new_shape[k]) {
        break;
      }
      idx[k] = 0;
    }
  }
  return Bezier<T>(std::move(out), b.rational());
}

/// Apply the affine map `x -> A x + b` to the image of a Bezier.
///
/// `A` and the offset are first rounded to `T`, and every operation is in `T`, as NumPy computes
/// `cp @ A.T + b`: each coordinate is the sum over the columns of `A` of the products with the
/// control point, added in the order of the column index, plus the offset. For a rational Bezier
/// the stored control points are weighted coordinates `(w x, w)`, so the linear part acts on the
/// weighted coordinates and the offset is scaled by the weight, `A (w x) + w b`; the weights are
/// copied unchanged and the projected map undergoes `A x + b`.
///
/// Args:
///     b (const Bezier<T>&): The Bezier, of rank `n`.
///     matrix (std::span<const double>): The matrix `A`, row-major over `matrix_shape`.
///     matrix_shape (std::span<const std::size_t>): `(n, n)`.
///     offset (std::span<const double>): The offset, `n` values.
///
/// Returns:
///     Bezier<T>: The transformed Bezier, of the same shape and rationality as `b`.
///
/// Raises:
///     value_error: `matrix_shape` has not two extents, or is not `(n, n)` (the geometric rank
///         excludes the weight), or `matrix` or `offset` does not hold the number of values it
///         should (checked in that order).
template <StorageFloat T>
[[nodiscard]] Bezier<T> transformed(const Bezier<T>& b, std::span<const double> matrix,
                                    std::span<const std::size_t> matrix_shape,
                                    std::span<const double> offset) {
  const std::size_t n = b.rank();
  if (matrix_shape.size() != 2) {
    throw value_error("matrix must be a 2D array, got shape " + py_repr(matrix_shape) + ".");
  }
  if (matrix_shape[0] != n || matrix_shape[1] != n) {
    throw value_error("Transform dimension (" + std::to_string(matrix_shape[0]) +
                      ") does not match the geometric rank (" + std::to_string(n) +
                      ") of the control points.");
  }
  if (matrix.size() != n * n) {
    throw value_error("matrix holds " + std::to_string(matrix.size()) + " values, but the shape " +
                      py_repr(matrix_shape) + " needs " + std::to_string(n * n) + ".");
  }
  if (offset.size() != n) {
    throw value_error("offset must have length " + std::to_string(n) + ", got " +
                      std::to_string(offset.size()) + ".");
  }
  std::vector<T> a(n * n);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<T>(matrix[i]);
  }
  std::vector<T> t(n);
  for (std::size_t i = 0; i < n; ++i) {
    t[i] = static_cast<T>(offset[i]);
  }
  const std::size_t cp = b.cp_size();
  const bool rational = b.rational();
  NdArray<T> out(std::vector<std::size_t>(b.shape().begin(), b.shape().end()), for_overwrite);
  const std::size_t rows = out.size() / cp;
  const T* src = b.control_points().data();
  T* dst = out.data();
  for (std::size_t r = 0; r < rows; ++r) {
    const T* x = src + r * cp;
    T* y = dst + r * cp;
    for (std::size_t i = 0; i < n; ++i) {
      T acc = static_cast<T>(0);
      for (std::size_t k = 0; k < n; ++k) {
        acc = acc + x[k] * a[i * n + k];
      }
      y[i] = acc + (rational ? x[n] * t[i] : t[i]);
    }
    if (rational) {
      y[n] = x[n];
    }
  }
  return Bezier<T>(std::move(out), rational);
}

}  // namespace pantr::bezier
