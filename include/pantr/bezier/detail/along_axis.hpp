/// @file along_axis.hpp
/// @brief View an N-D row-major array along one axis as a matrix `(n_axis, rest)`, and back.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace pantr::bezier::detail {

/// Products of the extents before and after one axis of a row-major array.
///
/// Attributes:
///     outer (std::size_t): Product of the extents before the axis (1 for the first axis).
///     inner (std::size_t): Product of the extents after the axis (1 for the last axis).
struct AxisSplit {
  std::size_t outer;
  std::size_t inner;
};

/// Split a shape around one axis.
///
/// Args:
///     shape (std::span<const std::size_t>): The extents.
///     axis (std::size_t): The axis, `< shape.size()` (unchecked).
///
/// Returns:
///     AxisSplit: The products of the extents before and after `axis`.
[[nodiscard]] inline AxisSplit split_around(std::span<const std::size_t> shape, std::size_t axis) {
  AxisSplit s{1, 1};
  for (std::size_t d = 0; d < axis; ++d) {
    s.outer *= shape[d];
  }
  for (std::size_t d = axis + 1; d < shape.size(); ++d) {
    s.inner *= shape[d];
  }
  return s;
}

/// Move `axis` to the front of a row-major array and flatten the other axes, in their original
/// order, into columns (NumPy's `moveaxis(arr, axis, 0).reshape(n, -1)`, made contiguous).
///
/// Element `(a, o * inner + i)` of the result is `data[(o * n + a) * inner + i]`, `n =
/// shape[axis]`, for `o < outer` and `i < inner`.
///
/// Args:
///     data (std::span<const T>): The elements, row-major over `shape`.
///     shape (std::span<const std::size_t>): The extents.
///     axis (std::size_t): The axis to move, `< shape.size()` (unchecked).
///
/// Returns:
///     NdArray<T>: Extents `(shape[axis], prod(shape) / shape[axis])`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] NdArray<T> flatten_along_axis(std::span<const T> data,
                                            std::span<const std::size_t> shape, std::size_t axis) {
  const AxisSplit s = split_around(shape, axis);
  const std::size_t n = shape[axis];
  const std::size_t rest = s.outer * s.inner;
  NdArray<T> out({n, rest}, for_overwrite);
  T* dst = out.data();
  for (std::size_t a = 0; a < n; ++a) {
    for (std::size_t o = 0; o < s.outer; ++o) {
      const T* src = data.data() + (o * n + a) * s.inner;
      T* row = dst + a * rest + o * s.inner;
      for (std::size_t i = 0; i < s.inner; ++i) {
        row[i] = src[i];
      }
    }
  }
  return out;
}

/// Invert `flatten_along_axis`: reshape a matrix `(m, rest)` to the original shape with the extent
/// of `axis` replaced by `m` (the row count may differ from the original extent, as after a degree
/// elevation or a derivative).
///
/// Args:
///     flat (mdview<const T, 2>): The matrix, `rest = outer * inner` columns of the original
///         shape (unchecked).
///     original_shape (std::span<const std::size_t>): The extents before the flattening.
///     axis (std::size_t): The axis the matrix's rows go to, `< original_shape.size()`.
///
/// Returns:
///     NdArray<T>: Extents `original_shape` with `[axis]` replaced by `flat.extent(0)`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] NdArray<T> unflatten_along_axis(mdview<const T, 2> flat,
                                              std::span<const std::size_t> original_shape,
                                              std::size_t axis) {
  const AxisSplit s = split_around(original_shape, axis);
  const std::size_t m = flat.extent(0);
  const std::size_t rest = s.outer * s.inner;
  std::vector<std::size_t> shape(original_shape.begin(), original_shape.end());
  shape[axis] = m;
  NdArray<T> out(std::move(shape), for_overwrite);
  const T* src = flat.data_handle();
  T* dst = out.data();
  for (std::size_t a = 0; a < m; ++a) {
    for (std::size_t o = 0; o < s.outer; ++o) {
      const T* row = src + a * rest + o * s.inner;
      T* d = dst + (o * m + a) * s.inner;
      for (std::size_t i = 0; i < s.inner; ++i) {
        d[i] = row[i];
      }
    }
  }
  return out;
}

}  // namespace pantr::bezier::detail
