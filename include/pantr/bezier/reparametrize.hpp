/// @file reparametrize.hpp
/// @brief Changes of parametrization and of space that move a `Bezier`'s control points without
///        changing its image: reversing a direction, permuting the directions, an affine map.
#pragma once

#include "pantr/bezier/bezier.hpp"
#include "pantr/bezier/detail/net_ops.hpp"
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
  const std::size_t d = detail::checked_direction(direction, b.dim());
  return Bezier<T>(detail::flip_along_axis<T>(b.control_points(), b.shape(), d), b.rational());
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
  detail::check_permutation(permutation, b.dim());
  return Bezier<T>(detail::permute_axes<T>(b.control_points(), b.shape(), permutation),
                   b.rational());
}

/// Apply the affine map `x -> A x + b` to the image of a Bezier.
///
/// `A` and the offset are first rounded to `T`, and every operation is in `T`, as NumPy computed
/// `cp @ A.T + b`: each coordinate is the sum over the columns of `A` of the products with the
/// control point, added in the order of the column index, plus the offset (NumPy's BLAS product
/// may sum in another order, so the two agree to rounding, not bit for bit). For a rational Bezier
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
  const detail::RoundedAffine<T> affine =
      detail::round_affine<T>(b.rank(), matrix, matrix_shape, offset);
  NdArray<T> out(std::vector<std::size_t>(b.shape().begin(), b.shape().end()), for_overwrite);
  detail::apply_affine_rows<T>(b.control_points().data(), out.data(), out.size() / b.cp_size(),
                               b.cp_size(), b.rational(), affine);
  return Bezier<T>(std::move(out), b.rational());
}

}  // namespace pantr::bezier
