/// @file compat.hpp
/// @brief `make_compat`: B-splines with one degree and one knot vector per axis.
///
/// The prerequisite of the operations that combine the control points of several B-splines
/// (`create_ruled`, the Coons patches, `join`): `make_compat` returns new B-splines that share
/// their degree and knot vector along the chosen axes and represent the same geometry.
#pragma once

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/convert.hpp"
#include "pantr/bspline/degree.hpp"
#include "pantr/bspline/knot_insertion.hpp"
#include "pantr/bspline/knots.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/scalar.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace pantr::cad {

/// A B-spline of either storage type, as the mixed-precision path of a binding holds it.
using AnyBspline = std::variant<bspline::Bspline<float>, bspline::Bspline<double>>;

namespace detail {

/// The interior breakpoints of a direction and their multiplicities.
///
/// Attributes:
///     points (std::vector<double>): The interior knot classes, increasing, as doubles.
///     multiplicities (std::vector<std::int64_t>): The multiplicity of each.
struct InteriorBreakpoints {
  std::vector<double> points;
  std::vector<std::int64_t> multiplicities;
};

/// Get the interior breakpoints and their multiplicities of a direction.
///
/// The knot classes of the domain at `tol` (rounded to the knot type), with the two boundary
/// classes stripped. Both are empty when the space has a single Bezier element.
///
/// Args:
///     space (const bspline::BsplineSpace1D<T>&): The direction.
///     tol (double): Absolute tolerance for grouping nearby knots.
///
/// Returns:
///     InteriorBreakpoints: The interior classes.
template <StorageFloat T>
[[nodiscard]] InteriorBreakpoints interior_breakpoints(const bspline::BsplineSpace1D<T>& space,
                                                       double tol) {
  const double tol_typed = static_cast<double>(static_cast<T>(tol));
  const bspline::KnotClasses<T> classes =
      bspline::unique_knots_and_multiplicity<T>(space.knots(), space.degree(), tol_typed, true);
  InteriorBreakpoints out;
  for (std::size_t i = 1; i + 1 < classes.knots.size(); ++i) {
    out.points.push_back(static_cast<double>(classes.knots[i]));
    out.multiplicities.push_back(classes.multiplicities[i]);
  }
  return out;
}

/// Look up the multiplicity of each merged breakpoint within a single space.
///
/// For each entry of `merged`, the entry of `points` that `numpy.searchsorted` finds (the first
/// not below it) is taken as its counterpart when it lies within `tol`; its multiplicity is the
/// answer, and a breakpoint absent from the space has multiplicity 0. Both sequences are sorted in
/// ascending order.
///
/// Args:
///     merged (const std::vector<double>&): The union of the interior breakpoints, ascending.
///     points (const std::vector<double>&): Interior breakpoints of one space, ascending.
///     multiplicities (const std::vector<std::int64_t>&): Their multiplicities.
///     tol (double): Tolerance for coincidence.
///
/// Returns:
///     std::vector<std::int64_t>: One multiplicity per entry of `merged`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
[[nodiscard]] inline std::vector<std::int64_t> lookup_multiplicities(
    const std::vector<double>& merged, const std::vector<double>& points,
    const std::vector<std::int64_t>& multiplicities, double tol) {
  std::vector<std::int64_t> result(merged.size(), 0);
  if (points.empty() || merged.empty()) {
    return result;
  }
  for (std::size_t i = 0; i < merged.size(); ++i) {
    const auto first_not_below = std::lower_bound(points.begin(), points.end(), merged[i]);
    const auto index = static_cast<std::size_t>(first_not_below - points.begin());
    if (index < points.size() && std::abs(points[index] - merged[i]) <= tol) {
      result[i] = multiplicities[index];
    }
  }
  return result;
}

/// Remap one axis of a B-spline to a new domain (a reparametrization only).
///
/// Each knot `k` becomes `a_new + (k - a_old) * scale` with `scale = (b_new - a_new) / (b_old -
/// a_old)`, computed in `double` and cast to `T`; the control points are shared unchanged, and the
/// periodic flag of the direction is kept.
///
/// Args:
///     b (const bspline::Bspline<T>&): The B-spline.
///     axis (std::size_t): The direction to remap, `< b.dim()`.
///     a_new (double): The new start of the domain.
///     b_new (double): The new end of the domain.
///
/// Returns:
///     bspline::Bspline<T>: The B-spline with the remapped knot vector in `axis`.
///
/// Raises:
///     value_error: A remapped knot is finite and beyond the range of `T`, or the remapped knots
///         define no space.
template <StorageFloat T>
[[nodiscard]] bspline::Bspline<T> remap_domain(const bspline::Bspline<T>& b, std::size_t axis,
                                               double a_new, double b_new) {
  const bspline::BsplineSpace<T> whole = b.space();
  const bspline::BsplineSpace1D<T>& old_space = whole.spaces()[axis];
  const double a_old = static_cast<double>(old_space.domain().first);
  const double b_old = static_cast<double>(old_space.domain().second);
  const double scale = (b_new - a_new) / (b_old - a_old);
  std::vector<T> knots;
  knots.reserve(old_space.knots().size());
  for (const T k : old_space.knots()) {
    // Two statements: a fused multiply-add would differ from NumPy's separate product and sum.
    const double scaled = (static_cast<double>(k) - a_old) * scale;
    const double value = a_new + scaled;
    knots.push_back(bspline::detail::convert_value<T>(value));
  }
  std::vector<bspline::BsplineSpace1D<T>> spaces = whole.spaces();
  spaces[axis] = bspline::BsplineSpace1D<T>(std::span<const T>(knots), old_space.degree(),
                                            old_space.periodic());
  return bspline::Bspline<T>(bspline::BsplineSpace<T>(std::move(spaces)), b.rep()->data, b.size(),
                             b.cp_size(), b.rational());
}

/// Remap every B-spline to the common domain `[min(starts), max(ends)]` along each axis.
///
/// Args:
///     results (std::vector<bspline::Bspline<T>>&): The B-splines, replaced by their remapped
///         versions where a domain differs.
///     axes (const std::vector<std::size_t>&): The axes to process.
template <StorageFloat T>
void same_domain(std::vector<bspline::Bspline<T>>& results, const std::vector<std::size_t>& axes) {
  for (const std::size_t a : axes) {
    std::vector<std::pair<double, double>> domains;
    domains.reserve(results.size());
    for (const auto& b : results) {
      const auto dom = b.space().spaces()[a].domain();
      domains.emplace_back(static_cast<double>(dom.first), static_cast<double>(dom.second));
    }
    double common_start = domains.front().first;
    double common_end = domains.front().second;
    for (const auto& d : domains) {
      common_start = (std::min)(common_start, d.first);
      common_end = (std::max)(common_end, d.second);
    }
    for (std::size_t i = 0; i < results.size(); ++i) {
      if (domains[i].first != common_start || domains[i].second != common_end) {
        results[i] = remap_domain<T>(results[i], a, common_start, common_end);
      }
    }
  }
}

/// Elevate every B-spline to the largest degree along each axis.
///
/// Args:
///     results (std::vector<bspline::Bspline<T>>&): The B-splines, replaced by their elevated
///         versions where a degree is below the largest.
///     axes (const std::vector<std::size_t>&): The axes to process.
///     dim (std::size_t): The parametric dimension.
template <StorageFloat T>
void same_degree(std::vector<bspline::Bspline<T>>& results, const std::vector<std::size_t>& axes,
                 std::size_t dim) {
  std::vector<long long> max_degrees(dim, 0);
  for (const std::size_t a : axes) {
    for (const auto& b : results) {
      max_degrees[a] = (std::max)(max_degrees[a], static_cast<long long>(b.degrees()[a]));
    }
  }
  for (std::size_t i = 0; i < results.size(); ++i) {
    std::vector<long long> increments(dim, 0);
    bool any = false;
    for (std::size_t d = 0; d < dim; ++d) {
      if (std::find(axes.begin(), axes.end(), d) != axes.end()) {
        increments[d] = max_degrees[d] - static_cast<long long>(results[i].degrees()[d]);
        any = any || increments[d] > 0;
      }
    }
    if (any) {
      results[i] = bspline::elevate_degree<T>(results[i], increments);
    }
  }
}

/// Merge the interior breakpoints of N spaces, returning per-space deficits.
///
/// Args:
///     points (const std::vector<std::vector<double>>&): The interior breakpoints per B-spline,
///         ascending.
///     multiplicities (const std::vector<std::vector<std::int64_t>>&): Their multiplicities.
///     tol (double): Tolerance for coincidence.
///
/// Returns:
///     std::pair<std::vector<double>, std::vector<std::vector<std::int64_t>>>: The sorted exact
///     union of every space's breakpoints, and for each space how many knots it lacks at each
///     union breakpoint to reach the largest multiplicity there (empty when no space has an
///     interior breakpoint).
[[nodiscard]] inline std::pair<std::vector<double>, std::vector<std::vector<std::int64_t>>>
merge_breakpoints(const std::vector<std::vector<double>>& points,
                  const std::vector<std::vector<std::int64_t>>& multiplicities, double tol) {
  std::vector<double> merged;
  for (const auto& p : points) {
    merged.insert(merged.end(), p.begin(), p.end());
  }
  if (merged.empty()) {
    return {{}, std::vector<std::vector<std::int64_t>>(points.size())};
  }
  std::sort(merged.begin(), merged.end());
  merged.erase(std::unique(merged.begin(), merged.end()), merged.end());

  // The target multiplicity is the largest over the spaces.
  std::vector<std::int64_t> target(merged.size(), 0);
  std::vector<std::vector<std::int64_t>> per_space;
  per_space.reserve(points.size());
  for (std::size_t s = 0; s < points.size(); ++s) {
    per_space.push_back(lookup_multiplicities(merged, points[s], multiplicities[s], tol));
    for (std::size_t i = 0; i < merged.size(); ++i) {
      target[i] = (std::max)(target[i], per_space.back()[i]);
    }
  }
  std::vector<std::vector<std::int64_t>> deficits = std::move(per_space);
  for (auto& d : deficits) {
    for (std::size_t i = 0; i < d.size(); ++i) {
      d[i] = target[i] - d[i];
    }
  }
  return {std::move(merged), std::move(deficits)};
}

/// Insert knots so every B-spline shares one knot vector per axis.
///
/// The tolerance along an axis is the **first** B-spline's space tolerance there (read after the
/// two earlier stages), applied to every B-spline's classes after rounding to its storage type.
///
/// Args:
///     results (std::vector<bspline::Bspline<T>>&): The B-splines, of one domain and degree per
///         axis.
///     axes (const std::vector<std::size_t>&): The axes to process.
///     dim (std::size_t): The parametric dimension.
template <StorageFloat T>
void merge_knots(std::vector<bspline::Bspline<T>>& results, const std::vector<std::size_t>& axes,
                 std::size_t dim) {
  for (const std::size_t a : axes) {
    const double tol = results.front().space().spaces()[a].tolerance();
    std::vector<std::vector<double>> points;
    std::vector<std::vector<std::int64_t>> multiplicities;
    for (const auto& b : results) {
      InteriorBreakpoints ip = interior_breakpoints<T>(b.space().spaces()[a], tol);
      points.push_back(std::move(ip.points));
      multiplicities.push_back(std::move(ip.multiplicities));
    }
    const auto& [merged, deficits] = merge_breakpoints(points, multiplicities, tol);
    if (merged.empty()) {
      continue;
    }
    for (std::size_t i = 0; i < results.size(); ++i) {
      std::vector<T> to_insert;
      for (std::size_t j = 0; j < merged.size(); ++j) {
        to_insert.insert(to_insert.end(), static_cast<std::size_t>(deficits[i][j]),
                         bspline::detail::convert_value<T>(merged[j]));
      }
      if (to_insert.empty()) {
        continue;
      }
      std::vector<std::span<const T>> per_dim(dim);
      per_dim[a] = std::span<const T>(to_insert);
      results[i] = bspline::insert_knots<T>(results[i], per_dim);
    }
  }
}

/// Spell a set of dimensions as Python spells a set of ints: `{1, 2}`.
[[nodiscard]] inline std::string set_repr(const std::set<std::size_t>& values) {
  std::string out = "{";
  bool first = true;
  for (const std::size_t v : values) {
    out += (first ? "" : ", ") + std::to_string(v);
    first = false;
  }
  return out + "}";
}

}  // namespace detail

/// Make B-splines compatible along the chosen parametric axes.
///
/// Returns new B-splines that share the same degree and knot vector along the axes and represent
/// the same geometry. Three stages, each over the selected axes in the order given:
///
///  1. Same domain: every knot vector is remapped affinely onto the envelope
///     `[min(starts), max(ends)]` (a reparametrization; the control points are unchanged).
///  2. Same degree: every B-spline is elevated to the largest degree.
///  3. Merge knots: knots are inserted so every B-spline has the same interior breakpoints at the
///     largest multiplicity any of them has. The breakpoints are the knot classes at the first
///     B-spline's space tolerance (rounded to each B-spline's storage type); the inserted values
///     are the exact sorted union of the breakpoints.
///
/// A periodic direction (on a selected axis) is first converted to open form, which opens every
/// direction of that B-spline. Fewer than two B-splines are returned unchanged, before any check.
///
/// Args:
///     bsplines (std::span<const bspline::Bspline<T>>): The B-splines, of one parametric
///         dimension.
///     axes (const std::optional<std::vector<long long>>&): The axes to operate on; no value
///         means every axis. An empty list returns the B-splines unchanged.
///
/// Returns:
///     std::vector<bspline::Bspline<T>>: One B-spline per input, in order, sharing the knot
///     structure along the axes (an input that needed no change shares its representation).
///
/// Raises:
///     value_error: The parametric dimensions differ ("All B-splines must have the same
///         parametric dimension, got {1, 2}."); an axis is out of range ("Axis a out of range for
///         dim=d."); or an elevation, insertion or conversion refuses (their messages).
template <StorageFloat T>
[[nodiscard]] std::vector<bspline::Bspline<T>> make_compat(
    std::span<const bspline::Bspline<T>> bsplines,
    const std::optional<std::vector<long long>>& axes = std::nullopt) {
  if (bsplines.size() < 2) {
    return std::vector<bspline::Bspline<T>>(bsplines.begin(), bsplines.end());
  }

  std::set<std::size_t> dims;
  for (const auto& b : bsplines) {
    dims.insert(b.dim());
  }
  if (dims.size() != 1) {
    throw value_error("All B-splines must have the same parametric dimension, got " +
                      detail::set_repr(dims) + ".");
  }
  const std::size_t dim = *dims.begin();

  std::vector<long long> selected;
  if (axes) {
    selected = *axes;
  } else {
    for (std::size_t a = 0; a < dim; ++a) {
      selected.push_back(static_cast<long long>(a));
    }
  }
  for (const long long a : selected) {
    if (a < 0 || a >= static_cast<long long>(dim)) {
      throw value_error("Axis " + std::to_string(a) +
                        " out of range for dim=" + std::to_string(dim) + ".");
    }
  }
  if (selected.empty()) {
    return std::vector<bspline::Bspline<T>>(bsplines.begin(), bsplines.end());
  }
  const std::vector<std::size_t> axis_list(selected.begin(), selected.end());

  // Convert periodic to open.
  std::vector<bspline::Bspline<T>> results;
  results.reserve(bsplines.size());
  for (const auto& b : bsplines) {
    const bspline::BsplineSpace<T> whole = b.space();
    const bool needs_open = std::any_of(axis_list.begin(), axis_list.end(), [&](std::size_t a) {
      return whole.spaces()[a].periodic();
    });
    results.push_back(needs_open ? bspline::to_open_bspline<T>(b) : b);
  }

  detail::same_domain<T>(results, axis_list);
  detail::same_degree<T>(results, axis_list, dim);
  detail::merge_knots<T>(results, axis_list, dim);
  return results;
}

namespace detail {

/// Make B-splines of either storage type compatible, the way a mixed-precision binding needs it.
///
/// Every input is converted to `double` (`bspline::astype`), made compatible there with
/// `cad::make_compat<double>` (so the merge tolerance is the converted first B-spline's, a
/// `double` tolerance, whatever type the first had), and each result is cast back to the storage
/// type of the input at its position.
///
/// Args:
///     bsplines (std::span<const AnyBspline>): The B-splines, of one parametric dimension.
///     axes (const std::optional<std::vector<long long>>&): As `make_compat`.
///
/// Returns:
///     std::vector<AnyBspline>: One B-spline per input, in order, of the input's storage type.
///
/// Raises:
///     value_error: As `make_compat`; a value is finite and beyond the range of `float` on the
///         way down or the cast knots define no space (`astype`'s refusals).
[[nodiscard]] inline std::vector<AnyBspline> make_compat(
    std::span<const AnyBspline> bsplines, const std::optional<std::vector<long long>>& axes) {
  if (bsplines.size() < 2) {
    return std::vector<AnyBspline>(bsplines.begin(), bsplines.end());
  }
  std::vector<bspline::Bspline<double>> widened;
  widened.reserve(bsplines.size());
  for (const AnyBspline& b : bsplines) {
    widened.push_back(std::visit([](const auto& v) { return bspline::astype<double>(v); }, b));
  }
  const std::vector<bspline::Bspline<double>> made =
      cad::make_compat<double>(std::span<const bspline::Bspline<double>>(widened), axes);
  std::vector<AnyBspline> out;
  out.reserve(made.size());
  for (std::size_t i = 0; i < made.size(); ++i) {
    if (std::holds_alternative<bspline::Bspline<float>>(bsplines[i])) {
      out.emplace_back(bspline::astype<float>(made[i]));
    } else {
      out.emplace_back(made[i]);
    }
  }
  return out;
}

}  // namespace detail

}  // namespace pantr::cad
