/// @file reparametrize.hpp
/// @brief Changes of parametrization and of space that move a `Bspline`'s knots and control points
///        without changing its image: reversing a direction, permuting the directions, an affine
///        map.
///
/// The control-point arithmetic is `bezier/detail/net_ops.hpp`'s, shared with `Bezier`.
#pragma once

#include "pantr/bezier/detail/net_ops.hpp"
#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace pantr::bspline {

/// Reverse the orientation of one parametric direction.
///
/// The knot vector is reflected about the middle of the domain, `a + b - knots` read backwards, so
/// the domain is unchanged, and the control points are flipped along the direction. A periodic
/// direction stores the first `n_stored` of its `n_full` wrapped control points; the full
/// sequence, reversed, then needs a cyclic shift of `n_full - n_stored` as well, which is applied.
/// The control points are copied, and each knot is one subtraction in `T` from the sum of the
/// domain ends.
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///     direction (long long): The direction to reverse, in `[0, dim)`.
///
/// Returns:
///     Bspline<T>: A B-spline `r` with `r(t) = b(t')`, `t'_direction = a + b - t_direction`, the
///     other parameters unchanged.
///
/// Raises:
///     value_error: `direction` is out of range; or the reflected knots define no space.
template <StorageFloat T>
[[nodiscard]] Bspline<T> reversed(const Bspline<T>& b, long long direction) {
  const std::size_t d = bezier::detail::checked_direction(direction, b.dim());
  const BsplineSpace<T> whole = b.space();
  const BsplineSpace1D<T>& old_space = whole.spaces()[d];
  const std::span<const T> knots = old_space.knots();
  const auto [lo, hi] = old_space.domain();
  const T sum = lo + hi;
  std::vector<T> reflected(knots.size());
  for (std::size_t i = 0; i < knots.size(); ++i) {
    reflected[i] = sum - knots[knots.size() - 1 - i];
  }
  std::vector<BsplineSpace1D<T>> spaces = whole.spaces();
  spaces[d] =
      BsplineSpace1D<T>(std::span<const T>(reflected), old_space.degree(), old_space.periodic());
  std::size_t shift = 0;
  if (old_space.periodic()) {
    const std::size_t n_stored = b.shape()[d];
    const std::size_t n_full = knots.size() - static_cast<std::size_t>(old_space.degree()) - 1;
    shift = (n_full - n_stored) % n_stored;
  }
  return Bspline<T>(BsplineSpace<T>(std::move(spaces)),
                    bezier::detail::flip_along_axis<T>(b.control_points(), b.shape(), d, shift),
                    b.rational());
}

/// Reorder the parametric directions.
///
/// The new direction `k` is the old direction `permutation[k]`: for `[1, 2, 0]` on a volume the
/// old direction 1 becomes the new 0, the old 2 the new 1 and the old 0 the new 2. A copy of the
/// values: nothing is rounded; the coordinate (and weight) axis stays last.
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///     permutation (std::span<const long long>): A permutation of `0..dim - 1`.
///
/// Returns:
///     Bspline<T>: A B-spline `r` with `r(t_0, ..., t_{dim-1}) = b(s)` where `s_{permutation[k]} =
///     t_k`.
///
/// Raises:
///     value_error: `permutation` is not a permutation of `0..dim - 1`.
template <StorageFloat T>
[[nodiscard]] Bspline<T> permuted(const Bspline<T>& b, std::span<const long long> permutation) {
  bezier::detail::check_permutation(permutation, b.dim());
  const BsplineSpace<T> whole = b.space();
  std::vector<BsplineSpace1D<T>> spaces;
  spaces.reserve(b.dim());
  for (const long long p : permutation) {
    spaces.push_back(whole.spaces()[static_cast<std::size_t>(p)]);
  }
  return Bspline<T>(BsplineSpace<T>(std::move(spaces)),
                    bezier::detail::permute_axes<T>(b.control_points(), b.shape(), permutation),
                    b.rational());
}

/// Apply the affine map `x -> A x + b` to the image of a B-spline.
///
/// `A` and the offset are first rounded to `T`, and every operation is in `T`, as NumPy computed
/// `cp @ A.T + b`: each coordinate is the sum over the columns of `A` of the products with the
/// control point, added in the order of the column index, plus the offset (NumPy's BLAS product
/// may sum in another order, so the two agree to rounding, not bit for bit). For a rational
/// B-spline the stored control points are weighted coordinates `(w x, w)`, so the linear part acts
/// on the weighted coordinates and the offset is scaled by the weight, `A (w x) + w b`; the
/// weights are copied unchanged and the projected map undergoes `A x + b`. The space is kept.
///
/// Args:
///     b (const Bspline<T>&): The B-spline, of rank `n`.
///     matrix (std::span<const double>): The matrix `A`, row-major over `matrix_shape`.
///     matrix_shape (std::span<const std::size_t>): `(n, n)`.
///     offset (std::span<const double>): The offset, `n` values.
///
/// Returns:
///     Bspline<T>: The transformed B-spline, over the same space and rational as `b`.
///
/// Raises:
///     value_error: `matrix_shape` has not two extents, or is not `(n, n)` (the geometric rank
///         excludes the weight), or `matrix` or `offset` does not hold the number of values it
///         should (checked in that order).
template <StorageFloat T>
[[nodiscard]] Bspline<T> transformed(const Bspline<T>& b, std::span<const double> matrix,
                                     std::span<const std::size_t> matrix_shape,
                                     std::span<const double> offset) {
  const bezier::detail::RoundedAffine<T> affine =
      bezier::detail::round_affine<T>(b.rank(), matrix, matrix_shape, offset);
  NdArray<T> out(std::vector<std::size_t>(b.shape().begin(), b.shape().end()), for_overwrite);
  bezier::detail::apply_affine_rows<T>(b.control_points().data(), out.data(),
                                       out.size() / b.cp_size(), b.cp_size(), b.rational(), affine);
  return Bspline<T>(b.space(), std::move(out), b.rational());
}

}  // namespace pantr::bspline
