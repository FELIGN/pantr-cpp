/// @file validation.hpp
/// @brief What the `pantr::cad` functions share: padding a point to 3D, promoting a B-spline to
///        rational form, the column groups a comparison of control points is graded by, reading a
///        curve's end as a coefficient, and the tolerance tier of a comparison.
///
/// The helpers are general on purpose: the construction and joining functions that combine the
/// control points of several B-splines (`create_ruled` here, the Coons patches and `join` after
/// it) build on these and on nothing else, so each decision has one spelling.
#pragma once

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/slice.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/tolerance.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace pantr::cad::detail {

/// The physical dimension of every CAD primitive: their control points have three coordinates.
inline constexpr std::size_t physical_dim = 3;

/// Convert a point to a 3D vector, zero-padding a shorter one.
///
/// Args:
///     point (std::span<const double>): One to three coordinates (none is padded to the origin).
///
/// Returns:
///     std::array<double, 3>: The coordinates, then zeros.
///
/// Raises:
///     value_error: `point` has more than three coordinates.
[[nodiscard]] inline std::array<double, physical_dim> pad_to_3d(std::span<const double> point) {
  if (point.size() > physical_dim) {
    throw value_error("Point must have at most " + std::to_string(physical_dim) +
                      " coordinates, got " + std::to_string(point.size()) + ".");
  }
  std::array<double, physical_dim> out = {0.0, 0.0, 0.0};
  std::copy(point.begin(), point.end(), out.begin());
  return out;
}

/// Add unit weights to a non-rational B-spline, giving a rational one over the same space.
///
/// Args:
///     b (const bspline::Bspline<T>&): A B-spline curve, surface or volume.
///
/// Returns:
///     bspline::Bspline<T>: `b` itself when it is already rational (sharing its representation),
///     otherwise a rational B-spline whose control points are `b`'s with a trailing weight of 1.
template <StorageFloat T>
[[nodiscard]] bspline::Bspline<T> promote_to_rational(const bspline::Bspline<T>& b) {
  if (b.rational()) {
    return b;
  }
  const std::size_t width = b.cp_size();
  const std::size_t points = b.size() / width;
  const std::span<const T> src = b.control_points();
  std::vector<T> out;
  out.reserve(points * (width + 1));
  for (std::size_t p = 0; p < points; ++p) {
    out.insert(out.end(), src.begin() + static_cast<std::ptrdiff_t>(p * width),
               src.begin() + static_cast<std::ptrdiff_t>((p + 1) * width));
    out.push_back(static_cast<T>(1));
  }
  return bspline::Bspline<T>(b.space(), std::span<const T>(out), true);
}

/// One group of control-point columns that share a physical dimension.
///
/// A rational control point carries homogeneous coordinates, whose units are length times weight,
/// beside a dimensionless weight. Grading both against one magnitude makes the verdict depend on
/// where the model sits, so each group is graded against its own.
///
/// Attributes:
///     name (std::string_view): What the group holds, as an error message names it.
///     begin (std::size_t): The first column of the group.
///     end (std::size_t): One past the last column of the group.
struct ColumnGroup {
  std::string_view name;
  std::size_t begin;
  std::size_t end;
};

/// Split a control-point array's columns into groups that share one physical dimension.
///
/// A relative tolerance's scale has to span exactly one dimension, so the caller grades each group
/// against its own magnitude: sending `x -> lambda x` scales the coordinates and leaves the
/// weights alone, and one magnitude over both would make the verdict on a weight drift with the
/// size of the model.
///
/// Args:
///     rational (bool): Whether the trailing column is a weight.
///     cp_size (std::size_t): The number of columns, the weight included (at least 2 when
///         `rational`).
///
/// Returns:
///     std::vector<ColumnGroup>: One group `coordinate` over every column when non-rational;
///     otherwise `coordinate` over `[0, cp_size - 1)` and `weight` over `[cp_size - 1, cp_size)`.
///     The coordinates come first: data given in the wrong place fails there, and it is the more
///     informative half to report when both are out.
[[nodiscard]] inline std::vector<ColumnGroup> column_groups(bool rational, std::size_t cp_size) {
  if (!rational) {
    return {ColumnGroup{"coordinate", 0, cp_size}};
  }
  return {ColumnGroup{"coordinate", 0, cp_size - 1}, ColumnGroup{"weight", cp_size - 1, cp_size}};
}

/// Read one end of a curve as a coefficient, without projecting a rational one.
///
/// `bspline::boundary_point` **projects**: sliced down to a point, a rational curve is divided by
/// its weight and returns `rank` components where its coefficients have `rank + 1`. Every other
/// term of a Coons blend is combined homogeneously, so a corner read that way is both one
/// component short and in a different space from the terms it is subtracted from.
///
/// Viewing the same coefficients as an ordinary curve of one rank higher is what stops before the
/// division: the de Boor recurrence underneath is linear and does not consult the rationality
/// (`slice_point` branches on it only for the final `X / w`), so the arithmetic is identical and
/// the result is the homogeneous corner the blend needs. The view shares the curve's buffer.
///
/// Args:
///     curve (const bspline::Bspline<T>&): A curve, rational or not.
///     side (long long): Which end of the domain: 0 for the start, 1 for the end.
///
/// Returns:
///     NdArray<T>: The curve's coefficient at that end, extents `(rank,)` when `curve` is
///     non-rational and `(rank + 1,)` when it is rational, in `curve`'s own storage type.
///
/// Raises:
///     value_error: `side` is neither 0 nor 1, or `curve` is not a curve.
template <StorageFloat T>
[[nodiscard]] NdArray<T> homogeneous_endpoint(const bspline::Bspline<T>& curve, long long side) {
  const bspline::Bspline<T> homogeneous(curve.space(), curve.rep()->data, curve.size(),
                                        curve.cp_size(), false);
  return bspline::boundary_point<T>(homogeneous, 0, side);
}

/// Get the conservative relative tolerance tier of the coarsest storage type among those about to
/// be compared.
///
/// A tolerance tier read at `double` while the data is `float` refuses agreement the inputs cannot
/// express: if the true shared value is `F`, two stored readings satisfy `|r1 - r2| <= 0.5 (eps1 +
/// eps2) |F|`, a hard floor on any achievable agreement. Reading the tier at the coarsest
/// precision present clears that floor; taking the *promoted* type instead is backwards by the
/// ratio of the two epsilons, about `5.4e8`.
///
/// The rule grades by **container** rather than by provenance, so `float`-resolution data already
/// copied into a `double` container is graded at `double`. A homogeneous call passes one type and
/// gets its own tier; the mixed-precision path of a binding, which converts every input to one
/// type, passes the types it converted from.
///
/// Returns:
///     double: `tolerance::get_conservative` of the type with the largest machine epsilon among
///     `T` and `Rest`.
template <StorageFloat T, StorageFloat... Rest>
[[nodiscard]] constexpr double conservative_tier() noexcept {
  double tier = static_cast<double>(tolerance::get_conservative<T>());
  ((tier = (std::max)(tier, static_cast<double>(tolerance::get_conservative<Rest>()))), ...);
  return tier;
}

}  // namespace pantr::cad::detail
