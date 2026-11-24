/// @file restrict.hpp
/// @brief Restriction of a B-spline to a sub-region of its domain, and splitting at a parameter.
///
/// Both insert knots at the new boundaries until they reach multiplicity `degree + 1`, then cut the
/// knot vector and the control points; a periodic direction is converted to open form first.
///
/// Tolerance policy. Restriction introduces **no tolerance of its own**: the single `tol` threaded
/// through this file is `BsplineSpace1D::tolerance()`, the absolute parametric length `8 eps
/// max(span, |knots[0]|, |knots[-1]|)` the space derived once at construction. It answers one
/// question, the one `knot_tolerance` answers: which value is a bound (and are two values one
/// knot, for the refusals below)?
///
/// The one rule: which knots are at a bound. `tol` decides which value the bound *is*, once, up
/// front: a bound within `tol` of a domain end is that end (`validate_restrict_bounds`, in
/// `double`); otherwise a bound within `tol` of a knot is the nearest such knot (`snap_to_knot`,
/// which rounds the bound to the storage type and forms `knot - bound` in it, as NumPy did for a
/// Python float); otherwise it is the bound rounded to the storage type. After that a knot is at
/// the bound when it is **bitwise equal** to that value, and nothing else is: the count of the
/// copies already there, the insertion's multiplicity check (run with a zero tolerance), the cut
/// of the refined vector (`equal_range` on the value, in the storage type) and whether a domain end
/// is already open (`degree + 1` knots equal to it) all use that identity. The inserted copies
/// carry the value itself, so every end a cut makes has exactly `degree + 1` equal knots and the
/// value the bound was taken as.
///
/// Two refusals follow from reading the pieces under the same `tol`, which their spaces keep. Two
/// bounds within `tol` of each other are one knot, so the window holds no interval and is refused.
/// And a space built with `snap_knots = false` may hold a knot distinct from the bound but within
/// `tol` of it: identity leaves it a knot of its own, and where a cut would place it inside a piece
/// beside the clamped end, that end's class under `tol` would have more than `degree + 1` members.
/// A space refuses that at its last knot; the call refuses it at either cut end, naming the bound
/// and the knot (`check_piece_end`). An end the cut leaves as the space had it (an open domain end)
/// is not checked: it is the space's own.
///
/// Two properties are load-bearing.
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
/// **A bound in the domain stays in it once rounded**, which is what lets the insertion check its
/// values against the domain with a zero tolerance: the domain ends are values of the storage type
/// and rounding to it is monotone.
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

/// The knot that `value` is: the stored knot nearest `value` among those within `tol` of it (the
/// difference formed in the storage type with `value` rounded to it, as NumPy forms `knots - value`
/// for a Python float; the first such knot on a tie), or `value` itself when no knot is.
template <StorageFloat T>
[[nodiscard]] double snap_to_knot(std::span<const T> knots, double value, double tol) {
  const auto v = static_cast<T>(value);
  double out = value;
  double nearest = 0.0;
  bool found = false;
  for (const T k : knots) {
    const auto d = static_cast<double>(std::abs(static_cast<T>(k - v)));
    if (d <= tol && (!found || d < nearest)) {
      out = static_cast<double>(k);
      nearest = d;
      found = true;
    }
  }
  return out;
}

/// The value a bound is, by the file's one rule: the domain end `end` when `value` is within `tol`
/// of it (in `double`, as `validate_restrict_bounds` compares), else the knot `snap_to_knot` takes
/// it as, else `value` rounded to the storage type.
///
/// Args:
///     knots (std::span<const T>): The knot vector.
///     end (T): The domain end on the bound's side.
///     value (double): The bound.
///     tol (double): Absolute parametric tolerance.
///
/// Returns:
///     T: The value the bound is.
template <StorageFloat T>
[[nodiscard]] T bound_value(std::span<const T> knots, T end, double value, double tol) {
  if (std::abs(value - static_cast<double>(end)) <= tol) {
    return end;
  }
  return static_cast<T>(snap_to_knot<T>(knots, value, tol));
}

/// Refuse a window whose two ends are one knot under `tol`.
///
/// Args:
///     lo (T): The value the lower bound is.
///     hi (T): The value the upper bound is.
///     bounds (Bounds): The bounds as requested, for the message.
///     tol (double): Absolute parametric tolerance.
///
/// Raises:
///     value_error: `hi - lo`, formed in `T`, is at most `tol`: the window holds no interval of a
///         space with that tolerance.
template <StorageFloat T>
void check_window(T lo, T hi, Bounds bounds, double tol) {
  if (!(static_cast<double>(static_cast<T>(hi - lo)) > tol)) {
    throw value_error("The bounds (" + py_repr(bounds.lo) + ", " + py_repr(bounds.hi) +
                      ") are within the tolerance " + py_repr(tol) +
                      " of one value, so they enclose no interval.");
  }
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
///     value_error: `degree < 0` or fewer than `2 * degree + 2` knots; a NaN bound; `lo >= hi`; or
///         a bound outside the domain by more than `tol`.
template <StorageFloat T>
[[nodiscard]] Bounds validate_restrict_bounds(std::span<const T> knots, int degree, double tol,
                                              Bounds bounds) {
  detail::check_domain_indexable(knots.size(), degree);
  const auto d = static_cast<std::size_t>(degree);
  const auto a = static_cast<double>(knots[d]);
  const auto b = static_cast<double>(knots[knots.size() - d - 1]);
  double a_new = bounds.lo;
  double b_new = bounds.hi;

  // A NaN fails every comparison below, and no knot equals it, so it is refused here.
  for (const auto& [name, v] : {std::pair{"Lower", a_new}, std::pair{"Upper", b_new}}) {
    if (std::isnan(v)) {
      throw value_error(std::string(name) + " bound (" + py_repr(v) + ") is outside the domain [" +
                        py_repr(a) + ", " + py_repr(b) + "].");
    }
  }
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
/// Each bound is first taken as the value it is (the file's one rule): a domain end within `tol`
/// of it, else the nearest knot within `tol` (`detail::snap_to_knot`), else itself rounded to the
/// storage type. Then enough copies of that value to make `degree + 1` knots bitwise equal to it;
/// none when the boundary is a domain end already open (`degree + 1` equal knots).
///
/// Args:
///     knots (std::span<const T>): The knot vector (non-periodic or open-compatible).
///     degree (int): Degree.
///     tol (double): Absolute parametric tolerance: it decides which value each bound is, and with
///         it which existing knots are already copies of the boundary.
///     bounds (Bounds): The bounds of the restricted domain (snapped or not).
///
/// Returns:
///     std::vector<T>: The knot values to insert (possibly empty).
///
/// Raises:
///     value_error: `degree < 0` or fewer than `2 * degree + 2` knots; or the bounds match the full
///         domain and the direction is already open.
template <StorageFloat T>
[[nodiscard]] std::vector<T> compute_boundary_knots_to_insert(std::span<const T> knots, int degree,
                                                              double tol, Bounds bounds) {
  detail::check_domain_indexable(knots.size(), degree);
  const auto p = static_cast<std::size_t>(degree);
  const T a = knots[p];
  const T b = knots[knots.size() - p - 1];

  const T lo = detail::bound_value<T>(knots, a, bounds.lo, tol);
  const T hi = detail::bound_value<T>(knots, b, bounds.hi, tol);
  // An end is kept as the space has it when the bound is that end and it is already open.
  const bool left_kept = lo == a && knots[0] == a;
  const bool right_kept = hi == b && knots.back() == b;
  if (left_kept && right_kept) {
    throw value_error("Bounds match the full domain and the direction is already open.");
  }

  std::vector<T> out;
  if (!left_kept) {
    const auto [first, past] = detail::equal_knots<T>(knots, lo);
    out.insert(out.end(), p + 1 - (std::min)(p + 1, past - first), lo);
  }
  if (!right_kept) {
    const auto [first, past] = detail::equal_knots<T>(knots, hi);
    out.insert(out.end(), p + 1 - (std::min)(p + 1, past - first), hi);
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
///         `insert_knots_1d`, and for a periodic direction as `to_open_1d`; the two bounds are
///         values within `tol` of each other ("... enclose no interval"); or a cut end would hold a
///         distinct knot within `tol` of it ("The bound ... is within the tolerance").
template <StorageFloat T>
[[nodiscard]] KnotsAndCtrl<T> restrict_1d(std::span<const T> knots, int degree,
                                          mdview<const T, 2> ctrl, bool periodic, double tol,
                                          Bounds bounds) {
  const auto p = static_cast<std::size_t>(degree);
  const std::size_t cols = ctrl.extent(1);
  if (!periodic) {
    detail::check_net_matches("restrict", ctrl.extent(0), knots.size(), degree);
  }
  const Bounds snapped = validate_restrict_bounds<T>(knots, degree, tol, bounds);

  // A periodic direction is converted to open form first.
  KnotsAndCtrl<T> open{{}, detail::empty_net<T>()};
  std::span<const T> k = knots;
  mdview<const T, 2> c = ctrl;
  if (periodic) {
    open = to_open_1d<T>(knots, degree, ctrl, true, tol);
    k = open.knots;
    c = open.ctrl.template view<2>();
  }

  // Each bound becomes the value it is (the file's one rule); from here a knot is at a bound when
  // it equals that value, for the count, the insertion and the cut alike.
  const T a = k[p];
  const T b = k[k.size() - p - 1];
  const T lo = detail::bound_value<T>(k, a, snapped.lo, tol);
  const T hi = detail::bound_value<T>(k, b, snapped.hi, tol);
  detail::check_window<T>(lo, hi, bounds, tol);
  const std::vector<T> to_insert = compute_boundary_knots_to_insert<T>(k, degree, tol, snapped);
  KnotsAndCtrl<T> refined{{}, detail::empty_net<T>()};
  if (!to_insert.empty()) {
    // A zero tolerance: the multiplicity check counts copies of a value, as the rule does, not
    // classes under `tol` that an unsnapped neighbour of the bound would join.
    refined = insert_knots_1d<T>(k, degree, c, to_insert, 0.0);
  } else {
    refined.knots.assign(k.begin(), k.end());
    refined.ctrl = NdArray<T>({c.extent(0), cols}, for_overwrite);
    std::copy(c.data_handle(), c.data_handle() + c.extent(0) * cols, refined.ctrl.data());
  }

  // Extract [lo, hi]: from the first copy of lo to the last copy of hi, each p + 1 copies now and
  // lo < hi (`check_window`), so the piece has at least 2 p + 2 knots and p + 1 rows.
  const std::span<const T> rk = refined.knots;
  const std::size_t i_start = detail::equal_knots<T>(rk, lo).first;
  const std::size_t past_end = detail::equal_knots<T>(rk, hi).second;
  const std::size_t i_end = past_end - 1;
  const std::span<const T> piece = rk.subspan(i_start, past_end - i_start);
  // An end the space already had (an open domain end, as `compute_boundary_knots_to_insert` keeps
  // it) is its own; a cut end is checked.
  if (!(lo == a && k[0] == a)) {
    detail::check_piece_end<T>(piece, lo, "bound", bounds.lo, tol);
  }
  if (!(hi == b && k.back() == b)) {
    detail::check_piece_end<T>(piece, hi, "bound", bounds.hi, tol);
  }
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
/// A bound is compared with the domain ends in `double`, and one within the space's tolerance of
/// an end is that end; otherwise one within it of a knot, by the comparison `detail::snap_to_knot`
/// makes (in the storage type), is that knot.
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///     bounds (std::span<const std::optional<Bounds>>): For each direction its new bounds, `dim`
///         entries; no value keeps the full domain.
///
/// Returns:
///     Bspline<T>: The B-spline on the restricted domain; each restricted direction is a
///     non-periodic space whose knots are not snapped and whose tolerance is the parent
///     direction's.
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
        // The parent's tolerance, as `BsplineSpace1D::restrict` keeps it: one derived from the
        // piece's knots can be smaller and split a class of an unsnapped parent into two knots.
        return {BsplineSpace1D<T>::from_state(std::span<const T>(r.knots), space.degree(), false,
                                              space.tolerance()),
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
///     value_error: As `insert_knots_1d` and `to_open_1d`; `value` is NaN; or a half would hold
///         a distinct knot within `tol` of `value` ("The bound ... is within the tolerance").
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

  // The value becomes the knot it is, or itself rounded (the file's one rule); from here a knot is
  // at the split when it equals that value, for the count, the insertion and the cut alike.
  const auto at = static_cast<T>(detail::snap_to_knot<T>(k, value, tol));
  const auto [first, past] = detail::equal_knots<T>(k, at);
  KnotsAndCtrl<T> refined{{}, detail::empty_net<T>()};
  if (p + 1 > past - first) {
    const std::vector<T> to_insert(p + 1 - (past - first), at);
    // A zero tolerance, as in `restrict_1d`: copies of the value, not classes under `tol`.
    refined = insert_knots_1d<T>(k, degree, c, to_insert, 0.0);
  } else {
    refined.knots.assign(k.begin(), k.end());
    refined.ctrl = NdArray<T>({c.extent(0), cols}, for_overwrite);
    std::copy(c.data_handle(), c.data_handle() + c.extent(0) * cols, refined.ctrl.data());
  }

  // The split index: the first of the p + 1 copies of the value, so `i_split + p + 1 <= size`.
  const std::span<const T> rk = refined.knots;
  const std::size_t i_split = detail::equal_knots<T>(rk, at).first;
  detail::check_piece_end<T>(rk.first(i_split + p + 1), at, "bound", value, tol);
  detail::check_piece_end<T>(rk.subspan(i_split), at, "bound", value, tol);
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
///         than the space's tolerance from both ends). A value within the tolerance of a knot by
///         the comparison `detail::snap_to_knot` makes is that knot, which must be inside too.
///
/// Returns:
///     std::pair<Bspline<T>, Bspline<T>>: `(left, right)`: the sub-splines on
///     `[domain_start, value]` and `[value, domain_end]` in the split direction, each a
///     non-periodic space with unsnapped knots and the parent direction's tolerance; other
///     directions are unchanged.
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
  const auto refuse = [&](const std::string& why) {
    return value_error("value must be strictly inside the domain (" + py_repr(a) + ", " +
                       py_repr(e) + "), got " + py_repr(value) + why + ".");
  };
  // A NaN fails both comparisons, and no knot equals it, so it is refused by name.
  if (std::isnan(value) || value <= a + tol || value >= e - tol) {
    throw refuse("");
  }
  // The value is also judged as the split will use it: one that is an end knot by the snap's
  // comparison is that end, though it be further than `tol` from it in `double`.
  const double at = detail::snap_to_knot<T>(space.knots(), value, tol);
  if (at <= a + tol || at >= e - tol) {
    throw refuse(", which rounded to the storage type is within the tolerance " + py_repr(tol) +
                 " of the end knot " + py_repr(at));
  }

  const std::vector<BsplineSpace1D<T>>& spaces = whole.spaces();
  const NdArray<T> flat = bezier::detail::flatten_along_axis<T>(b.control_points(), b.shape(), d);
  auto parts = split_1d<T>(space.knots(), space.degree(), flat.template view<2>(), space.periodic(),
                           tol, value);

  const std::vector<std::size_t> shape(b.shape().begin(), b.shape().end());
  const auto build = [&](KnotsAndCtrl<T>& part) {
    std::vector<BsplineSpace1D<T>> s = spaces;
    // The parent's tolerance, as in `restrict`.
    s[d] = BsplineSpace1D<T>::from_state(std::span<const T>(part.knots), space.degree(), false,
                                         space.tolerance());
    NdArray<T> ctrl =
        bezier::detail::unflatten_along_axis<T>(part.ctrl.template view<2>(), shape, d);
    return Bspline<T>(BsplineSpace<T>(std::move(s)), std::move(ctrl), b.rational());
  };
  Bspline<T> left = build(parts.first);
  Bspline<T> right = build(parts.second);
  return {std::move(left), std::move(right)};
}

}  // namespace pantr::bspline
