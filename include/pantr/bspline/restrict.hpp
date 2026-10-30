/// @file restrict.hpp
/// @brief Restriction of a B-spline to a sub-region of its domain, and splitting at a parameter.
///
/// Both insert knots at the new boundaries until they reach multiplicity `degree + 1`, then cut the
/// knot vector and the control points; a periodic direction is converted to open form first.
///
/// Tolerance policy. Restriction introduces **no tolerance of its own**, and that is the policy
/// rather than an omission. Every comparison it makes is between two parametric coordinates of one
/// knot vector (a requested bound against a domain endpoint, against an existing knot, against the
/// refined vector), so every one of them is the question `knot_tolerance` already answers: are
/// these two values the same knot? The single `tol` threaded through this file is
/// `BsplineSpace1D::tolerance()`, the absolute parametric length `8 eps max(span, |knots[0]|,
/// |knots[-1]|)` the space derived once at construction. A second number here would be a second
/// notion of knot identity for the same vector.
///
/// Three properties of that inheritance are load-bearing.
///
/// **The scale does not move between the stages.** `tol` is derived from the input knot vector,
/// then applied to the *refined* one after boundary insertion. Knot insertion never changes the
/// first or last knot, so the scale is identical for both. The one stage that does change the
/// endpoints is the periodic-to-open conversion, which trims the ghost knots and leaves the vector
/// spanning `[a, b] = [knots[p], knots[-p - 1]]`. That can only *shrink* the scale: `[a, b]` lies
/// within `[knots[0], knots[-1]]`, so `b - a` is at most the original span, and the largest
/// absolute value on a real interval is attained at an endpoint, so `|a|` and `|b|` are at most
/// `max(|knots[0]|, |knots[-1]|)`. Each of the three terms of the scale only decreases, and the
/// inherited tolerance is then conservative rather than tight, the safe direction for a merge test.
///
/// **The extraction offsets widen, and that direction is deliberate.** The sub-vector is cut at
/// the first knot not below `a_new - tol` and one before the first knot not below `b_new + tol`:
/// both move the cut *away* from the
/// interval, so a boundary knot can never be dropped by a coordinate landing a rounding short of
/// its own value. At a domain based at `1e6` one ulp of the bound is about `1.2e-10`, so a fixed
/// absolute `1e-12` would be absorbed entirely by `b_new + tol == b_new` and the cut would fall
/// before the first copy of `b_new`, silently truncating the last `degree + 1` knots. A tolerance
/// of `8 eps scale` is at least four ulp of any coordinate in the vector, so the offset always
/// changes the value it is added to and the cut always lands past the whole boundary group.
///
/// **Widening cannot over-collect.** `BsplineSpace1D` snaps its knots by the same rule, so two
/// *distinct* stored knots differ by more than `tol` and a cut widened by `tol` cannot reach the
/// next one. The one configuration where that is not enough is a bound falling between two
/// distinct knots less than `2 tol` apart, which requires a mesh already at the format's
/// resolution floor for its magnitude: the regime `check_snapping_kept_an_interval` reports on.
#pragma once

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/detail/refine_common.hpp"
#include "pantr/bspline/knot_insertion.hpp"
#include "pantr/bspline/knots.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bspline {

/// A parametric interval `[lo, hi]`.
///
/// Attributes:
///     lo (double): The lower bound.
///     hi (double): The upper bound.
struct Bounds {
  double lo;
  double hi;
};

namespace detail {

/// Count the knots within `tol` of `value`, the difference formed in the storage type with `value`
/// rounded to it (as NumPy forms `knots - value` for a Python float).
template <StorageFloat T>
[[nodiscard]] std::size_t count_within(std::span<const T> knots, double value, double tol) {
  const auto v = static_cast<T>(value);
  return static_cast<std::size_t>(std::count_if(knots.begin(), knots.end(), [&](T k) {
    return static_cast<double>(std::abs(static_cast<T>(k - v))) <= tol;
  }));
}

/// The first index whose knot is not below `key`, the comparison made in `double` (as NumPy's
/// `searchsorted` makes it for a Python-float key).
template <StorageFloat T>
[[nodiscard]] std::size_t first_not_below(std::span<const T> knots, double key) {
  const auto it = std::lower_bound(knots.begin(), knots.end(), key,
                                   [](T k, double v) { return static_cast<double>(k) < v; });
  return static_cast<std::size_t>(it - knots.begin());
}

}  // namespace detail

/// Validate and snap restriction bounds for a 1D B-spline.
///
/// A bound within `tol` of a domain endpoint *is* that endpoint: it is accepted rather than
/// rejected as out of range, and snapped onto the stored value so the rest compares bitwise.
///
/// Args:
///     knots (std::span<const T>): The knot vector.
///     degree (int): Degree.
///     tol (double): Absolute parametric tolerance.
///     bounds (Bounds): The requested bounds.
///
/// Returns:
///     Bounds: The snapped bounds.
///
/// Raises:
///     value_error: `lo >= hi`, or a bound outside the domain by more than `tol`.
template <StorageFloat T>
[[nodiscard]] Bounds validate_restrict_bounds(std::span<const T> knots, int degree, double tol,
                                              Bounds bounds) {
  const auto d = static_cast<std::size_t>(degree);
  const auto a = static_cast<double>(knots[d]);
  const auto b = static_cast<double>(knots[knots.size() - d - 1]);
  double a_new = bounds.lo;
  double b_new = bounds.hi;

  if (a_new >= b_new) {
    throw value_error("Lower bound (" + py_repr(a_new) +
                      ") must be strictly less than upper bound (" + py_repr(b_new) + ").");
  }
  if (a_new < a && std::abs(a_new - a) > tol) {
    throw value_error("Lower bound (" + py_repr(a_new) + ") is below the domain start (" +
                      py_repr(a) + ").");
  }
  if (b_new > b && std::abs(b_new - b) > tol) {
    throw value_error("Upper bound (" + py_repr(b_new) + ") is above the domain end (" +
                      py_repr(b) + ").");
  }
  if (std::abs(a_new - a) <= tol) {
    a_new = a;
  }
  if (std::abs(b_new - b) <= tol) {
    b_new = b;
  }
  return {a_new, b_new};
}

/// Compute the knots to insert at the restriction boundaries.
///
/// For each boundary, enough copies to reach multiplicity `degree + 1`; none when the boundary
/// coincides with an already-open domain endpoint.
///
/// Args:
///     knots (std::span<const T>): The knot vector (non-periodic or open-compatible).
///     degree (int): Degree.
///     tol (double): Absolute parametric tolerance: it decides which existing knots already *are*
///         the boundary, and so how many copies the boundary is short of.
///     bounds (Bounds): The (snapped) bounds of the restricted domain.
///
/// Returns:
///     std::vector<T>: The knot values to insert (possibly empty).
///
/// Raises:
///     value_error: The bounds match the full domain and the direction is already open.
template <StorageFloat T>
[[nodiscard]] std::vector<T> compute_boundary_knots_to_insert(std::span<const T> knots, int degree,
                                                              double tol, Bounds bounds) {
  const auto p = static_cast<std::size_t>(degree);
  const auto a = static_cast<double>(knots[p]);
  const auto b = static_cast<double>(knots[knots.size() - p - 1]);

  const bool left_at_domain = std::abs(bounds.lo - a) <= tol;
  const bool right_at_domain = std::abs(bounds.hi - b) <= tol;
  const bool left_open =
      std::abs(static_cast<double>(knots[0]) - static_cast<double>(knots[p])) <= tol;
  const bool right_open = std::abs(static_cast<double>(knots[knots.size() - p - 1]) -
                                   static_cast<double>(knots.back())) <= tol;

  if (left_at_domain && right_at_domain && left_open && right_open) {
    throw value_error("Bounds match the full domain and the direction is already open.");
  }

  std::vector<T> out;
  if (!(left_at_domain && left_open)) {
    const std::size_t m_left = detail::count_within<T>(knots, bounds.lo, tol);
    if (p + 1 > m_left) {
      out.insert(out.end(), p + 1 - m_left, static_cast<T>(bounds.lo));
    }
  }
  if (!(right_at_domain && right_open)) {
    const std::size_t m_right = detail::count_within<T>(knots, bounds.hi, tol);
    if (p + 1 > m_right) {
      out.insert(out.end(), p + 1 - m_right, static_cast<T>(bounds.hi));
    }
  }
  return out;
}

/// Restrict a 1D B-spline to a sub-interval of its parametric domain.
///
/// Inserts knots at the boundaries until each has multiplicity `degree + 1`, then extracts the
/// knot sub-vector and control points of the restricted interval. A periodic direction is first
/// converted to open form.
///
/// Args:
///     knots (std::span<const T>): The knot vector.
///     degree (int): Degree.
///     ctrl (mdview<const T, 2>): Control points, extents `(n, cols)`.
///     periodic (bool): Whether the direction is periodic.
///     tol (double): Absolute parametric tolerance, serving every comparison here.
///     bounds (Bounds): The requested bounds.
///
/// Returns:
///     KnotsAndCtrl<T>: The clamped knot vector on `[lo, hi]` and its control points.
///
/// Raises:
///     value_error: As `validate_restrict_bounds`, `compute_boundary_knots_to_insert` and
///         `insert_knots_1d`; or the cut found no complete sub-vector.
template <StorageFloat T>
[[nodiscard]] KnotsAndCtrl<T> restrict_1d(std::span<const T> knots, int degree,
                                          mdview<const T, 2> ctrl, bool periodic, double tol,
                                          Bounds bounds) {
  const auto p = static_cast<std::size_t>(degree);
  const std::size_t cols = ctrl.extent(1);
  const Bounds snapped = validate_restrict_bounds<T>(knots, degree, tol, bounds);

  // A periodic direction is converted to open form first.
  KnotsAndCtrl<T> open{{}, detail::empty_net<T>()};
  std::span<const T> k = knots;
  mdview<const T, 2> c = ctrl;
  if (periodic) {
    open = to_open_1d<T>(knots, degree, ctrl, true, tol);
    k = open.knots;
    c = open.ctrl.template view<2>();
  } else {
    detail::check_net_matches("restrict", ctrl.extent(0), knots.size(), degree);
  }

  const std::vector<T> to_insert = compute_boundary_knots_to_insert<T>(k, degree, tol, snapped);
  KnotsAndCtrl<T> refined{{}, detail::empty_net<T>()};
  if (!to_insert.empty()) {
    refined = insert_knots_1d<T>(k, degree, c, to_insert, tol);
  } else {
    refined.knots.assign(k.begin(), k.end());
    refined.ctrl = NdArray<T>({c.extent(0), cols}, for_overwrite);
    std::copy(c.data_handle(), c.data_handle() + c.extent(0) * cols, refined.ctrl.data());
  }

  // Extract [lo, hi]. After insertion lo has multiplicity p + 1 starting at i_start, and hi has
  // multiplicity p + 1 ending at i_end. Both offsets widen the cut away from the interval (see the
  // file note).
  const std::size_t i_start = detail::first_not_below<T>(refined.knots, snapped.lo - tol);
  const std::size_t past_end = detail::first_not_below<T>(refined.knots, snapped.hi + tol);
  if (past_end == 0 || past_end - 1 < i_start + p || past_end > refined.knots.size() ||
      past_end - 1 - p > refined.ctrl.shape()[0]) {
    throw value_error("The restriction found no complete sub-vector of the refined knot vector.");
  }
  const std::size_t i_end = past_end - 1;
  KnotsAndCtrl<T> out{
      std::vector<T>(refined.knots.begin() + static_cast<std::ptrdiff_t>(i_start),
                     refined.knots.begin() + static_cast<std::ptrdiff_t>(i_end) + 1),
      NdArray<T>({i_end - p - i_start, cols}, for_overwrite)};
  std::copy(refined.ctrl.data() + i_start * cols, refined.ctrl.data() + (i_end - p) * cols,
            out.ctrl.data());
  return out;
}

/// Restrict a B-spline to a sub-region of its parametric domain.
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///     bounds (std::span<const std::optional<Bounds>>): For each direction its new bounds, `dim`
///         entries; no value keeps the full domain.
///
/// Returns:
///     Bspline<T>: The B-spline on the restricted domain; each restricted direction is a
///     non-periodic space whose knots are not snapped.
///
/// Raises:
///     value_error: `bounds.size() != dim` ("bounds sequence length ..."); no direction has bounds;
///         or as `restrict_1d` for a direction.
template <StorageFloat T>
[[nodiscard]] Bspline<T> restrict(const Bspline<T>& b,
                                  std::span<const std::optional<Bounds>> bounds) {
  if (bounds.size() != b.dim()) {
    throw value_error("bounds sequence length (" + std::to_string(bounds.size()) +
                      ") must match dim (" + std::to_string(b.dim()) + ").");
  }
  const bool any = std::any_of(bounds.begin(), bounds.end(),
                               [](const std::optional<Bounds>& x) { return x.has_value(); });
  if (!any) {
    throw value_error("At least one direction must have non-None bounds that restrict the domain.");
  }
  return detail::map_directions<T>(
      b, [&](std::size_t d) { return bounds[d].has_value(); },
      [&](std::size_t d, const BsplineSpace1D<T>& space,
          mdview<const T, 2> flat) -> std::pair<BsplineSpace1D<T>, NdArray<T>> {
        KnotsAndCtrl<T> r = restrict_1d<T>(space.knots(), space.degree(), flat, space.periodic(),
                                           space.tolerance(), *bounds[d]);
        return {BsplineSpace1D<T>(std::span<const T>(r.knots), space.degree(), false, false),
                std::move(r.ctrl)};
      });
}

/// Split a 1D B-spline at a parameter value.
///
/// Inserts knots at `value` until multiplicity `degree + 1`, then extracts the left
/// `[domain_start, value]` and right `[value, domain_end]` sub-splines. A periodic direction is
/// first converted to open form.
///
/// Args:
///     knots (std::span<const T>): The knot vector.
///     degree (int): Degree.
///     ctrl (mdview<const T, 2>): Control points, extents `(n, cols)`.
///     periodic (bool): Whether the direction is periodic.
///     tol (double): Absolute parametric tolerance.
///     value (double): The parameter value, strictly inside the domain (not checked here).
///
/// Returns:
///     std::pair<KnotsAndCtrl<T>, KnotsAndCtrl<T>>: The left and the right sub-spline's knots and
///     control points.
///
/// Raises:
///     value_error: As `insert_knots_1d` and `to_open_1d`; or the cut found no split index.
template <StorageFloat T>
[[nodiscard]] std::pair<KnotsAndCtrl<T>, KnotsAndCtrl<T>> split_1d(std::span<const T> knots,
                                                                   int degree,
                                                                   mdview<const T, 2> ctrl,
                                                                   bool periodic, double tol,
                                                                   double value) {
  const auto p = static_cast<std::size_t>(degree);
  const std::size_t cols = ctrl.extent(1);

  KnotsAndCtrl<T> open{{}, detail::empty_net<T>()};
  std::span<const T> k = knots;
  mdview<const T, 2> c = ctrl;
  if (periodic) {
    open = to_open_1d<T>(knots, degree, ctrl, true, tol);
    k = open.knots;
    c = open.ctrl.template view<2>();
  } else {
    detail::check_net_matches("split", ctrl.extent(0), knots.size(), degree);
  }

  // Insert up to multiplicity p + 1 at the split value.
  const std::size_t m_current = detail::count_within<T>(k, value, tol);
  KnotsAndCtrl<T> refined{{}, detail::empty_net<T>()};
  if (p + 1 > m_current) {
    const std::vector<T> to_insert(p + 1 - m_current, static_cast<T>(value));
    refined = insert_knots_1d<T>(k, degree, c, to_insert, tol);
  } else {
    refined.knots.assign(k.begin(), k.end());
    refined.ctrl = NdArray<T>({c.extent(0), cols}, for_overwrite);
    std::copy(c.data_handle(), c.data_handle() + c.extent(0) * cols, refined.ctrl.data());
  }

  // The split index: the value now has multiplicity p + 1.
  const std::size_t i_split = detail::first_not_below<T>(refined.knots, value - tol);
  if (i_split + p + 1 > refined.knots.size() || i_split > refined.ctrl.shape()[0]) {
    throw value_error("The split found no index in the refined knot vector.");
  }
  const std::size_t rows = refined.ctrl.shape()[0];

  std::pair<KnotsAndCtrl<T>, KnotsAndCtrl<T>> out{
      KnotsAndCtrl<T>{
          std::vector<T>(refined.knots.begin(),
                         refined.knots.begin() + static_cast<std::ptrdiff_t>(i_split + p + 1)),
          NdArray<T>({i_split, cols}, for_overwrite)},
      KnotsAndCtrl<T>{std::vector<T>(refined.knots.begin() + static_cast<std::ptrdiff_t>(i_split),
                                     refined.knots.end()),
                      NdArray<T>({rows - i_split, cols}, for_overwrite)}};
  std::copy(refined.ctrl.data(), refined.ctrl.data() + i_split * cols, out.first.ctrl.data());
  std::copy(refined.ctrl.data() + i_split * cols, refined.ctrl.data() + rows * cols,
            out.second.ctrl.data());
  return out;
}

/// Split a B-spline in two at a parameter value in one direction.
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///     direction (long long): The parametric direction, in `[0, dim)`.
///     value (double): The parameter value, strictly inside the domain of that direction (further
///         than the space's tolerance from both ends).
///
/// Returns:
///     std::pair<Bspline<T>, Bspline<T>>: `(left, right)`: the sub-splines on
///     `[domain_start, value]` and `[value, domain_end]` in the split direction; other directions
///     are unchanged.
///
/// Raises:
///     value_error: `direction` is outside `[0, dim)`; `value` is not strictly inside the domain;
///         or as `split_1d`.
template <StorageFloat T>
[[nodiscard]] std::pair<Bspline<T>, Bspline<T>> split(const Bspline<T>& b, long long direction,
                                                      double value) {
  if (direction < 0 || direction >= static_cast<long long>(b.dim())) {
    throw value_error("direction must be in [0, " + std::to_string(b.dim()) + "), got " +
                      std::to_string(direction) + ".");
  }
  const auto d = static_cast<std::size_t>(direction);
  const BsplineSpace<T> whole = b.space();
  const BsplineSpace1D<T>& space = whole.spaces()[d];
  const auto [dom_lo, dom_hi] = space.domain();
  const auto a = static_cast<double>(dom_lo);
  const auto e = static_cast<double>(dom_hi);
  const double tol = space.tolerance();
  if (value <= a + tol || value >= e - tol) {
    throw value_error("value must be strictly inside the domain (" + py_repr(a) + ", " +
                      py_repr(e) + "), got " + py_repr(value) + ".");
  }

  const std::vector<BsplineSpace1D<T>>& spaces = whole.spaces();
  const NdArray<T> flat = bezier::detail::flatten_along_axis<T>(b.control_points(), b.shape(), d);
  auto parts = split_1d<T>(space.knots(), space.degree(), flat.template view<2>(), space.periodic(),
                           tol, value);

  const std::vector<std::size_t> shape(b.shape().begin(), b.shape().end());
  const auto build = [&](KnotsAndCtrl<T>& part) {
    std::vector<BsplineSpace1D<T>> s = spaces;
    s[d] = BsplineSpace1D<T>(std::span<const T>(part.knots), space.degree(), false, false);
    NdArray<T> ctrl =
        bezier::detail::unflatten_along_axis<T>(part.ctrl.template view<2>(), shape, d);
    return Bspline<T>(BsplineSpace<T>(std::move(s)), std::move(ctrl), b.rational());
  };
  Bspline<T> left = build(parts.first);
  Bspline<T> right = build(parts.second);
  return {std::move(left), std::move(right)};
}

}  // namespace pantr::bspline
