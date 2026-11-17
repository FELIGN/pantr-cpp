/// @file subdivide.hpp
/// @brief Splitting, slicing, restricting and collapsing a `Bezier`: the maps that keep a piece of
///        its domain or fix some of its parameters.
#pragma once

#include "pantr/basis/bernstein.hpp"
#include "pantr/bezier/bezier.hpp"
#include "pantr/bezier/detail/along_axis.hpp"
#include "pantr/bezier/detail/kernels_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pantr::bezier {

namespace detail {

/// Refuse a direction or axis outside `[0, dim)`.
///
/// Args:
///     name (std::string_view): The parameter's name in the message (`direction` or `axis`).
///     dim (std::size_t): The parametric dimension.
///     value (long long): The direction or axis.
///
/// Raises:
///     value_error: `value < 0` or `value >= dim`.
inline void check_in_dim(std::string_view name, std::size_t dim, long long value) {
  if (value < 0 || value >= static_cast<long long>(dim)) {
    throw value_error(std::string(name) + " must be in [0, " + std::to_string(dim) + "), got " +
                      std::to_string(value) + ".");
  }
}

/// Refuse the arguments of a slice: an axis outside `[0, dim)`, then a value outside `[0, 1]`.
///
/// Args:
///     dim (std::size_t): The parametric dimension.
///     axis (long long): The axis.
///     value (double): The parameter value.
///
/// Raises:
///     value_error: The axis or the value is out of range, checked in that order. A NaN value
///         passes, as it does in the Python check.
inline void check_slice_args(std::size_t dim, long long axis, double value) {
  check_in_dim("axis", dim, axis);
  if (value < 0.0 || value > 1.0) {
    throw value_error("value must be in [0, 1], got " + py_repr(value) + ".");
  }
}

/// The extents with one axis removed.
[[nodiscard]] inline std::vector<std::size_t> shape_without(std::span<const std::size_t> shape,
                                                            std::size_t axis) {
  std::vector<std::size_t> out;
  out.reserve(shape.size() - 1);
  for (std::size_t d = 0; d < shape.size(); ++d) {
    if (d != axis) {
      out.push_back(shape[d]);
    }
  }
  return out;
}

/// The control points of the slice of a Bezier at `value` along `axis`, the extent of `axis`
/// removed (so `(cp_size,)` for a curve).
///
/// At `0` and `1` the first or last slab is copied without the de Casteljau pass or the flattening
/// to a matrix; the values are the ones `slice_1d_kernel` copies at those parameters.
///
/// Args:
///     b (const Bezier<T>&): The Bezier.
///     axis (std::size_t): The axis, `< b.dim()` (unchecked).
///     value (double): The parameter value, in `[0, 1]` (unchecked).
///
/// Returns:
///     NdArray<T>: The control points of the slice, rational coordinates included.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] NdArray<T> slice_values(const Bezier<T>& b, std::size_t axis, double value) {
  const std::span<const std::size_t> shape = b.shape();
  NdArray<T> out(shape_without(shape, axis), for_overwrite);
  if (value == 0.0 || value == 1.0) {
    const AxisSplit s = split_around(shape, axis);
    const std::size_t n = shape[axis];
    const std::size_t row = value == 0.0 ? 0 : n - 1;
    const T* src = b.control_points().data();
    T* dst = out.data();
    for (std::size_t o = 0; o < s.outer; ++o) {
      for (std::size_t i = 0; i < s.inner; ++i) {
        dst[o * s.inner + i] = src[(o * n + row) * s.inner + i];
      }
    }
    return out;
  }
  const NdArray<T> flat = flatten_along_axis<T>(b.control_points(), shape, axis);
  slice_1d_kernel<T>(flat.template view<2>(), value, out.flat());
  return out;
}

}  // namespace detail

/// Split a Bezier into two at a parameter value in one direction.
///
/// One forward de Casteljau pass per column gives the left half, the Bezier on `[0, value]`, and
/// the right half, the Bezier on `[value, 1]`, each reparametrized to `[0, 1]`. The control points
/// of a rational Bezier, the weight included, are split alike, so the halves are rational with
/// the same projected map.
///
/// Args:
///     b (const Bezier<T>&): The Bezier.
///     direction (long long): The parametric direction, in `[0, dim)`.
///     value (double): The parameter value, strictly inside `(0, 1)`.
///
/// Returns:
///     std::pair<Bezier<T>, Bezier<T>>: `(left, right)`, of the same shape as `b`.
///
/// Raises:
///     value_error: `direction` is out of range, or `value` is not strictly inside `(0, 1)`
///         (checked in that order; a NaN value passes, as in the Python check).
template <StorageFloat T>
[[nodiscard]] std::pair<Bezier<T>, Bezier<T>> split(const Bezier<T>& b, long long direction,
                                                    double value) {
  detail::check_in_dim("direction", b.dim(), direction);
  if (value <= 0.0 || value >= 1.0) {
    throw value_error("value must be strictly inside (0, 1), got " + py_repr(value) + ".");
  }
  const auto d = static_cast<std::size_t>(direction);
  const NdArray<T> flat = detail::flatten_along_axis<T>(b.control_points(), b.shape(), d);
  const std::vector<std::size_t> flat_shape(flat.shape().begin(), flat.shape().end());
  NdArray<T> left(flat_shape, for_overwrite);
  NdArray<T> right(flat_shape, for_overwrite);
  detail::split_1d_kernel<T>(flat.template view<2>(), value, left.template view<2>(),
                             right.template view<2>());
  NdArray<T> left_ctrl = detail::unflatten_along_axis<T>(left.template view<2>(), b.shape(), d);
  NdArray<T> right_ctrl = detail::unflatten_along_axis<T>(right.template view<2>(), b.shape(), d);
  return {Bezier<T>(std::move(left_ctrl), b.rational()),
          Bezier<T>(std::move(right_ctrl), b.rational())};
}

/// Fix one parametric direction of a surface or a volume (or higher) at a value.
///
/// The control points along `axis` go through de Casteljau; the values `0` and `1` copy the first
/// or last slab. A rational Bezier stays rational, the weight sliced with the coordinates. A curve
/// has no Bezier slice: its slice is a point, `slice_point`.
///
/// Args:
///     b (const Bezier<T>&): The Bezier, `dim() >= 2`.
///     axis (long long): The direction to fix, in `[0, dim)`.
///     value (double): The parameter value, in `[0, 1]`.
///
/// Returns:
///     Bezier<T>: A Bezier of dimension `dim() - 1`.
///
/// Raises:
///     value_error: `b` is a curve, `axis` is out of range, or `value` is outside `[0, 1]` (checked
///         in that order; a NaN value passes, as in the Python check).
template <StorageFloat T>
[[nodiscard]] Bezier<T> slice(const Bezier<T>& b, long long axis, double value) {
  if (b.dim() < 2) {
    throw value_error("A 1D Bézier slices to a point, not a Bézier: call slice_point.");
  }
  detail::check_slice_args(b.dim(), axis, value);
  return Bezier<T>(detail::slice_values<T>(b, static_cast<std::size_t>(axis), value), b.rational());
}

/// Evaluate a curve at one parameter value, by de Casteljau: the slice of a 1D Bezier.
///
/// The values `0` and `1` copy the first or last control point. A rational curve is projected
/// to physical coordinates, each coordinate divided by the weight in `T`.
///
/// Args:
///     b (const Bezier<T>&): The Bezier, `dim() == 1`.
///     axis (long long): The direction to fix, which must be `0`.
///     value (double): The parameter value, in `[0, 1]`.
///
/// Returns:
///     NdArray<T>: The point, extents `(rank,)`.
///
/// Raises:
///     value_error: `b` is not a curve, `axis` is not `0`, or `value` is outside `[0, 1]` (checked
///         in that order; a NaN value passes, as in the Python check).
template <StorageFloat T>
[[nodiscard]] NdArray<T> slice_point(const Bezier<T>& b, long long axis, double value) {
  if (b.dim() != 1) {
    throw value_error("A Bézier of dimension " + std::to_string(b.dim()) +
                      " slices to a Bézier, not a point: call slice.");
  }
  detail::check_slice_args(b.dim(), axis, value);
  NdArray<T> raw = detail::slice_values<T>(b, 0, value);
  if (!b.rational()) {
    return raw;
  }
  const std::size_t rank = b.rank();
  const T weight = raw.data()[rank];
  NdArray<T> point({rank}, for_overwrite);
  for (std::size_t c = 0; c < rank; ++c) {
    point.data()[c] = raw.data()[c] / weight;
  }
  return point;
}

/// Extract the face of a surface, a volume (or higher) at one end of a direction: the slice at `0`
/// or `1`.
///
/// Args:
///     b (const Bezier<T>&): The Bezier, `dim() >= 2`.
///     axis (long long): The direction, in `[0, dim)`.
///     side (long long): `0` for the start of the domain, `1` for the end.
///
/// Returns:
///     Bezier<T>: A Bezier of dimension `dim() - 1`.
///
/// Raises:
///     value_error: `b` is a curve, `side` is not `0` or `1`, or `axis` is out of range (checked
///         in that order).
template <StorageFloat T>
[[nodiscard]] Bezier<T> boundary(const Bezier<T>& b, long long axis, long long side) {
  if (b.dim() < 2) {
    throw value_error("The boundary of a 1D Bézier is a point, not a Bézier: call boundary_point.");
  }
  if (side != 0 && side != 1) {
    throw value_error("side must be 0 or 1, got " + std::to_string(side) + ".");
  }
  return slice<T>(b, axis, side == 0 ? 0.0 : 1.0);
}

/// Extract the end point of a curve: the slice at `0` or `1`, projected when rational.
///
/// Args:
///     b (const Bezier<T>&): The Bezier, `dim() == 1`.
///     axis (long long): The direction, which must be `0`.
///     side (long long): `0` for the start of the domain, `1` for the end.
///
/// Returns:
///     NdArray<T>: The point, extents `(rank,)`.
///
/// Raises:
///     value_error: `b` is not a curve, `side` is not `0` or `1`, or `axis` is not `0` (checked in
///         that order).
template <StorageFloat T>
[[nodiscard]] NdArray<T> boundary_point(const Bezier<T>& b, long long axis, long long side) {
  if (b.dim() != 1) {
    throw value_error("A Bézier of dimension " + std::to_string(b.dim()) +
                      " has a Bézier as boundary, not a point: call boundary.");
  }
  if (side != 0 && side != 1) {
    throw value_error("side must be 0 or 1, got " + std::to_string(side) + ".");
  }
  return slice_point<T>(b, axis, side == 0 ? 0.0 : 1.0);
}

/// Restrict a Bezier to a box inside `[0, 1]^dim`, reparametrized back to `[0, 1]^dim`.
///
/// The same degrees and different control points. Each restricted direction takes two de Casteljau
/// passes, ordered so that the divisor of the second pass is not small (`restrict_1d_kernel`); a
/// direction without bounds, or with the bounds `(0.0, 1.0)`, is left alone. The control points of
/// a rational Bezier, the weight included, are restricted alike.
///
/// Args:
///     b (const Bezier<T>&): The Bezier.
///     bounds (std::span<const std::optional<std::pair<double, double>>>): One entry per direction
///         (`dim` entries; a curve takes one): `(lower, upper)` with `0 <= lower < upper <= 1`, or
///         no value to keep the direction whole. At least one direction must be restricted.
///
/// Returns:
///     Bezier<T>: The restriction, of the same shape as `b`.
///
/// Raises:
///     value_error: `bounds.size() != dim`; then, in direction order, a direction with `lower >=
///         upper` or one with a bound outside `[0, 1]` (a NaN bound passes both, as in the Python
///         checks); then every direction unrestricted.
template <StorageFloat T>
[[nodiscard]] Bezier<T> restrict(const Bezier<T>& b,
                                 std::span<const std::optional<std::pair<double, double>>> bounds) {
  if (bounds.size() != b.dim()) {
    throw value_error("bounds sequence length (" + std::to_string(bounds.size()) +
                      ") must match dim (" + std::to_string(b.dim()) + ").");
  }
  for (std::size_t i = 0; i < bounds.size(); ++i) {
    if (!bounds[i]) {
      continue;
    }
    const double lower = bounds[i]->first;
    const double upper = bounds[i]->second;
    if (lower >= upper) {
      throw value_error("Lower bound (" + py_repr(lower) +
                        ") must be strictly less than upper bound (" + py_repr(upper) +
                        ") in direction " + std::to_string(i) + ".");
    }
    if (lower < 0.0 || upper > 1.0) {
      throw value_error("Bounds (" + py_repr(lower) + ", " + py_repr(upper) +
                        ") must lie within [0, 1] in direction " + std::to_string(i) + ".");
    }
  }
  const std::span<const std::size_t> shape = b.shape();
  std::optional<NdArray<T>> current;
  for (std::size_t i = 0; i < bounds.size(); ++i) {
    if (!bounds[i]) {
      continue;
    }
    const double lower = bounds[i]->first;
    const double upper = bounds[i]->second;
    if (lower == 0.0 && upper == 1.0) {
      continue;
    }
    const std::span<const T> src = current ? current->flat() : b.control_points();
    const NdArray<T> flat = detail::flatten_along_axis<T>(src, shape, i);
    NdArray<T> restricted(std::vector<std::size_t>(flat.shape().begin(), flat.shape().end()),
                          for_overwrite);
    detail::restrict_1d_kernel<T>(flat.template view<2>(), lower, upper,
                                  restricted.template view<2>());
    current = detail::unflatten_along_axis<T>(restricted.template view<2>(), shape, i);
  }
  if (!current) {
    throw value_error("Bounds match the full domain; at least one direction must be restricted.");
  }
  return Bezier<T>(std::move(*current), b.rational());
}

/// Collapse a surface, a volume (or higher) to a curve by fixing every direction but one.
///
/// The Bernstein basis of each collapsed direction is tabulated at its value (`basis::
/// tabulate_bernstein_1d`, in `T`) and contracted with the control points, the highest direction
/// first, each product rounded to `T` and the terms added in `T` in the order of the index. The
/// result has the degree of `axis` and the rank and rationality of `b`.
///
/// Args:
///     b (const Bezier<T>&): The Bezier, `dim() >= 2`.
///     axis (long long): The direction to keep, in `[0, dim)`.
///     values (std::span<const T>): The values of the other `dim - 1` directions in `[0, 1]`, in
///         direction order: `values[i]` is for direction `i` when `i < axis`, else `i + 1`.
///
/// Returns:
///     Bezier<T>: A curve.
///
/// Raises:
///     value_error: `b` is a curve, `axis` is out of range, `values.size() != dim - 1`, or a
///         value is outside `[0, 1]` (checked in that order; a NaN value passes, as in the Python
///         check).
template <StorageFloat T>
[[nodiscard]] Bezier<T> collapse_along_axis(const Bezier<T>& b, long long axis,
                                            std::span<const T> values) {
  if (b.dim() < 2) {
    throw value_error("collapse_along_axis requires dim >= 2.");
  }
  detail::check_in_dim("axis", b.dim(), axis);
  const auto keep = static_cast<std::size_t>(axis);
  if (values.size() != b.dim() - 1) {
    const std::array<std::size_t, 1> got = {values.size()};
    throw value_error("values must have length dim - 1 = " + std::to_string(b.dim() - 1) +
                      ", got shape " + py_repr(std::span<const std::size_t>(got)) + ".");
  }
  for (const T v : values) {
    if (v < static_cast<T>(0) || v > static_cast<T>(1)) {
      throw value_error("All values must be in [0, 1].");
    }
  }
  std::vector<std::size_t> shape(b.shape().begin(), b.shape().end());
  std::optional<NdArray<T>> current;
  // Highest direction first, so the axis index of a direction still to contract is its own.
  for (std::size_t d = b.dim(); d-- > 0;) {
    if (d == keep) {
      continue;
    }
    const T point = values[d < keep ? d : d - 1];
    const std::size_t n = shape[d];
    const NdArray<T> basis =
        basis::tabulate_bernstein_1d<T>(static_cast<int>(n - 1), std::span<const T>(&point, 1));
    const std::span<const T> src = current ? current->flat() : b.control_points();
    const detail::AxisSplit s = detail::split_around(shape, d);
    std::vector<std::size_t> next_shape = detail::shape_without(shape, d);
    NdArray<T> next(next_shape, for_overwrite);
    for (std::size_t o = 0; o < s.outer; ++o) {
      for (std::size_t j = 0; j < s.inner; ++j) {
        T acc = static_cast<T>(0);
        for (std::size_t i = 0; i < n; ++i) {
          acc = acc + basis.data()[i] * src[(o * n + i) * s.inner + j];
        }
        next.data()[o * s.inner + j] = acc;
      }
    }
    current = std::move(next);
    shape = std::move(next_shape);
  }
  return Bezier<T>(std::move(*current), b.rational());
}

}  // namespace pantr::bezier
