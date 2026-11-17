/// @file net_ops.hpp
/// @brief What reversing a direction, permuting the directions and applying an affine map do to a
///        control net, shared by `Bezier` and `Bspline`: they act on the row-major control points
///        and know nothing of the type that holds them.
#pragma once

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
#include <string_view>
#include <utility>
#include <vector>

namespace pantr::bezier::detail {

/// Check a direction (a parametric axis) against the dimension.
///
/// Args:
///     direction (long long): The direction.
///     dim (std::size_t): The parametric dimension.
///     noun (std::string_view): What the caller calls it in its signature (`"direction"`,
///         `"axis"`), the first word of the message.
///
/// Returns:
///     std::size_t: `direction`.
///
/// Raises:
///     value_error: `direction` is not in `[0, dim)`; the message starts with `noun`.
[[nodiscard]] inline std::size_t checked_direction(long long direction, std::size_t dim,
                                                   std::string_view noun) {
  if (direction < 0 || direction >= static_cast<long long>(dim)) {
    throw value_error(std::string(noun) + " must be in [0, " + std::to_string(dim) + "), got " +
                      std::to_string(direction) + ".");
  }
  return static_cast<std::size_t>(direction);
}

/// Flip a row-major array along one axis, optionally rolled: `out[o, (i + shift) mod n, t] =
/// data[o, n - 1 - i, t]`. A copy of the values: nothing is rounded.
///
/// Args:
///     data (std::span<const T>): The elements, row-major over `shape`.
///     shape (std::span<const std::size_t>): The extents.
///     axis (std::size_t): The axis to flip, `< shape.size()`.
///     shift (std::size_t): The cyclic shift along the axis, `< shape[axis]`; 0 for a plain flip.
///
/// Returns:
///     NdArray<T>: The array of the same shape.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] NdArray<T> flip_along_axis(std::span<const T> data,
                                         std::span<const std::size_t> shape, std::size_t axis,
                                         std::size_t shift = 0) {
  const AxisSplit s = split_around(shape, axis);
  const std::size_t n = shape[axis];
  NdArray<T> out(std::vector<std::size_t>(shape.begin(), shape.end()), for_overwrite);
  const T* src = data.data();
  T* dst = out.data();
  for (std::size_t o = 0; o < s.outer; ++o) {
    for (std::size_t a = 0; a < n; ++a) {
      std::copy_n(src + (o * n + a) * s.inner, s.inner,
                  dst + (o * n + (n - 1 - a + shift) % n) * s.inner);
    }
  }
  return out;
}

/// Check that a list is a permutation of `0..dim - 1`.
///
/// Args:
///     permutation (std::span<const long long>): The list.
///     dim (std::size_t): The parametric dimension.
///
/// Raises:
///     value_error: It is not ("permutation must be a permutation of range(dim)").
inline void check_permutation(std::span<const long long> permutation, std::size_t dim) {
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
}

/// Reorder the parametric axes of a control net: the new axis `k` is the old axis
/// `permutation[k]`, and the last axis (the coordinates) stays last. A copy of the values.
///
/// Args:
///     data (std::span<const T>): The control points, row-major over `shape`.
///     shape (std::span<const std::size_t>): `(n_0, ..., n_{dim-1}, cp_size)`.
///     permutation (std::span<const long long>): A permutation of `0..dim - 1`.
///
/// Returns:
///     NdArray<T>: Extents `(n_{permutation[0]}, ..., n_{permutation[dim-1]}, cp_size)`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] NdArray<T> permute_axes(std::span<const T> data, std::span<const std::size_t> shape,
                                      std::span<const long long> permutation) {
  const std::size_t dim = shape.size() - 1;
  const std::size_t cp = shape[dim];
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
  const T* src = data.data();
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
  return out;
}

/// An affine map `x -> A x + b` rounded to the storage type.
///
/// Attributes:
///     matrix (std::vector<T>): `A`, row-major `n x n`.
///     offset (std::vector<T>): `b`, `n` values.
template <StorageFloat T>
struct RoundedAffine {
  std::vector<T> matrix;
  std::vector<T> offset;
};

/// Check an affine map against the geometric rank of a control net, and round it to `T`.
///
/// Args:
///     rank (std::size_t): The geometric rank `n` (the weight excluded).
///     matrix (std::span<const double>): The matrix `A`, row-major over `matrix_shape`.
///     matrix_shape (std::span<const std::size_t>): It must be `(n, n)`.
///     offset (std::span<const double>): The offset, `n` values.
///
/// Returns:
///     RoundedAffine<T>: `A` and `b` in `T`.
///
/// Raises:
///     value_error: `matrix_shape` has not two extents, or is not `(n, n)`, or `matrix` or
///         `offset` does not hold the number of values it should (checked in that order).
template <StorageFloat T>
[[nodiscard]] RoundedAffine<T> round_affine(std::size_t rank, std::span<const double> matrix,
                                            std::span<const std::size_t> matrix_shape,
                                            std::span<const double> offset) {
  const std::size_t n = rank;
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
  RoundedAffine<T> out{std::vector<T>(n * n), std::vector<T>(n)};
  for (std::size_t i = 0; i < out.matrix.size(); ++i) {
    out.matrix[i] = static_cast<T>(matrix[i]);
  }
  for (std::size_t i = 0; i < n; ++i) {
    out.offset[i] = static_cast<T>(offset[i]);
  }
  return out;
}

/// Apply a rounded affine map to every row of a control net.
///
/// Every operation is in `T`, as NumPy computed `cp @ A.T + b`: each coordinate is the sum over
/// the columns of `A` of the products with the control point, added in the order of the column
/// index, plus the offset. For a rational net the stored rows are weighted coordinates `(w x, w)`,
/// so the linear part acts on the weighted coordinates and the offset is scaled by the weight; the
/// weights are copied unchanged.
///
/// Args:
///     src (const T*): `rows * cp` values.
///     dst (T*): `rows * cp` values written; must not overlap `src`.
///     rows (std::size_t): The control points.
///     cp (std::size_t): The coordinates per point, the weight included.
///     rational (bool): Whether the last coordinate is a weight.
///     affine (const RoundedAffine<T>&): The map, of rank `cp - (rational ? 1 : 0)`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed). `noexcept`, so the contracted
///     sums stay vectorisable wherever it is called from.
template <StorageFloat T>
void apply_affine_rows(const T* src, T* dst, std::size_t rows, std::size_t cp, bool rational,
                       const RoundedAffine<T>& affine) noexcept {
  const std::size_t n = cp - (rational ? 1 : 0);
  const T* a = affine.matrix.data();
  const T* t = affine.offset.data();
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
}

}  // namespace pantr::bezier::detail
