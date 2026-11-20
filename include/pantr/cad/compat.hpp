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
#include "pantr/bspline/detail/refine_common.hpp"
#include "pantr/bspline/knot_insertion.hpp"
#include "pantr/bspline/knots.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/scalar.hpp"

#include <algorithm>
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
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
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

/// Remap one axis of a B-spline to a new domain (a reparametrization only).
///
/// Each knot `k` becomes `a_new + (k - a_old) * scale` with `scale = (b_new - a_new) / (b_old -
/// a_old)`, computed in `double` and cast to `T`; the control points are shared unchanged, and the
/// periodic flag of the direction is kept.
///
/// The remap is computed in `double` and rounded once to `T`, while the Python implementation's
/// `float32` remap ran in `float32` arithmetic, so a `float` end knot can differ from that
/// implementation's by one `float` ulp.
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
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): `axis < b.dim()`.
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
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): `results` is not empty and
///     every axis is below the parametric dimension.
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
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): every axis is below `dim`, which
///     is the parametric dimension of every B-spline.
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

/// Merge the interior breakpoints of N spaces into tolerance classes, returning the knots each
/// space lacks.
///
/// The exact union of every space's breakpoints is grouped with the classifier the spaces group
/// their own knots with (`bspline::unique_knots_and_multiplicity` in `T` at `tol`), so breakpoints
/// of different spaces within `tol` of each other are one class. A space's multiplicity at a class
/// is the total multiplicity of its own breakpoints in the class, the target is the largest over
/// the spaces, and a space below it lacks the difference. A lacking knot takes the space's own
/// smallest breakpoint in the class where it has one, so that the space does not move its own knot
/// (a space merges each of its classes onto the class's first knot, which would shift a knot it
/// already has), and the class's smallest breakpoint where it has none.
///
/// Args:
///     points (const std::vector<std::vector<double>>&): The interior breakpoints per B-spline,
///         ascending, each a value of `T`.
///     multiplicities (const std::vector<std::vector<std::int64_t>>&): Their multiplicities.
///     tol (double): Absolute tolerance for grouping, already rounded to `T`.
///
/// Returns:
///     std::vector<std::vector<double>>: For each space, the knots to insert, ascending (empty when
///     it lacks none).
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] std::vector<std::vector<double>> merge_breakpoints(
    const std::vector<std::vector<double>>& points,
    const std::vector<std::vector<std::int64_t>>& multiplicities, double tol) {
  std::vector<T> merged;
  for (const auto& p : points) {
    for (const double v : p) {
      merged.push_back(static_cast<T>(v));
    }
  }
  std::vector<std::vector<double>> to_insert(points.size());
  if (merged.empty()) {
    return to_insert;
  }
  std::sort(merged.begin(), merged.end());
  merged.erase(std::unique(merged.begin(), merged.end()), merged.end());

  // The union holds distinct values, so a class's multiplicity is its number of union entries:
  // entry `i` belongs to class `class_of[i]`.
  const bspline::KnotClasses<T> classes =
      bspline::unique_knots_and_multiplicity<T>(std::span<const T>(merged), 0, tol, false);
  std::vector<std::size_t> class_of;
  class_of.reserve(merged.size());
  for (std::size_t c = 0; c < classes.knots.size(); ++c) {
    class_of.insert(class_of.end(), static_cast<std::size_t>(classes.multiplicities[c]), c);
  }

  // Every breakpoint of a space is an entry of the union, found exactly; a space's breakpoints
  // ascend, so the first one met in a class is its smallest there.
  const std::size_t num_classes = classes.knots.size();
  std::vector<std::int64_t> target(num_classes, 0);
  std::vector<std::vector<std::int64_t>> owned(points.size(),
                                               std::vector<std::int64_t>(num_classes, 0));
  std::vector<std::vector<double>> value(points.size());
  for (std::size_t s = 0; s < points.size(); ++s) {
    for (const T k : classes.knots) {
      value[s].push_back(static_cast<double>(k));
    }
    for (std::size_t j = 0; j < points[s].size(); ++j) {
      const auto entry = static_cast<std::size_t>(
          std::lower_bound(merged.begin(), merged.end(), static_cast<T>(points[s][j])) -
          merged.begin());
      const std::size_t c = class_of[entry];
      if (owned[s][c] == 0) {
        value[s][c] = points[s][j];
      }
      owned[s][c] += multiplicities[s][j];
    }
    for (std::size_t c = 0; c < num_classes; ++c) {
      target[c] = (std::max)(target[c], owned[s][c]);
    }
  }
  for (std::size_t s = 0; s < points.size(); ++s) {
    for (std::size_t c = 0; c < num_classes; ++c) {
      to_insert[s].insert(to_insert[s].end(), static_cast<std::size_t>(target[c] - owned[s][c]),
                          value[s][c]);
    }
  }
  return to_insert;
}

/// Insert knots so every B-spline shares one knot vector per axis.
///
/// The tolerance along an axis is the **first** B-spline's space tolerance there (read after the
/// two earlier stages), rounded to the storage type, applied to every B-spline's classes and to the
/// classes of their union (`merge_breakpoints`, which also says where a lacking knot goes).
///
/// Args:
///     results (std::vector<bspline::Bspline<T>>&): The B-splines, of one domain and degree per
///         axis.
///     axes (const std::vector<std::size_t>&): The axes to process.
///     dim (std::size_t): The parametric dimension.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): `results` is not empty, every
///     axis is below `dim`, and the B-splines share one domain and one degree along each axis.
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
    const double tol_typed = static_cast<double>(static_cast<T>(tol));
    const std::vector<std::vector<double>> lacking =
        merge_breakpoints<T>(points, multiplicities, tol_typed);
    for (std::size_t i = 0; i < results.size(); ++i) {
      std::vector<T> to_insert;
      for (const double k : lacking[i]) {
        to_insert.push_back(bspline::detail::convert_value<T>(k));
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
///  3. Merge knots: knots are inserted so every B-spline has the same interior knot classes at the
///     largest multiplicity any of them has. The classes are those of the union of every
///     B-spline's interior breakpoints, grouped as a space groups its own knots, at the first
///     B-spline's space tolerance rounded to the storage type; a multiplicity at a class counts all
///     of a B-spline's knots in it, and a knot inserted at a class takes the B-spline's own
///     smallest knot there, or the class's smallest breakpoint where it has none, so that no
///     B-spline's own knot moves (for a B-spline whose own tolerance is the first's; one with a
///     larger tolerance can merge an inserted knot into a class of its own). The outputs then
///     agree in their number of knots per class, but not necessarily in their values: those differ
///     by less than the tolerance, or by more where breakpoints each within the tolerance of the
///     next chain into one class.
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

/// Refuse a `float` B-spline whose knots, rounded down from `double`, repeat a value more than
/// `degree + 1` times in some direction.
///
/// Two distinct `double` knots of the merged vector can round to one `float`; the cast space would
/// then carry a multiplicity no operation of the library accepts afterwards. The previous Python
/// implementation inserted the merged knots in `float` and refused there, with this message.
///
/// Args:
///     b (const bspline::Bspline<float>&): The B-spline cast down.
///
/// Raises:
///     value_error: A knot value of some direction repeats more than `degree + 1` times (at the
///         direction's tolerance).
inline void check_narrowed_multiplicity(const bspline::Bspline<float>& b) {
  const bspline::BsplineSpace<float> whole = b.space();
  for (const auto& space : whole.spaces()) {
    const long long max_allowed = static_cast<long long>(space.degree()) + 1;
    const std::int64_t found =
        bspline::detail::max_multiplicity<float>(space.knots(), space.tolerance());
    if (found > max_allowed) {
      throw value_error("Inserting these knots would exceed the maximum multiplicity of " +
                        std::to_string(max_allowed) +
                        ". Maximum multiplicity found: " + std::to_string(found) + ".");
    }
  }
}

/// Make B-splines of either storage type compatible, the way a mixed-precision binding needs it.
///
/// Every input is converted to `double` (`bspline::astype`), made compatible there with
/// `cad::make_compat<double>` (so the merge tolerance is the converted first B-spline's, a
/// `double` tolerance, whatever type the first had), and each result is cast back to the storage
/// type of the input at its position; an input the compatibility left unchanged is returned as
/// itself.
///
/// Widening to `double` can treat a `float` knot and a nearby `double` knot differently from the
/// Python implementation, which kept them apart: for a `float` `0.3f` against a `double` `0.3` it
/// returned unequal numbers of knots, while this returns equal ones.
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
///         way down or the cast knots define no space (`astype`'s refusals); or two merged knots
///         round to one `float` beyond the maximum multiplicity (`check_narrowed_multiplicity`).
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
    if (made[i].same_rep(widened[i])) {
      // Left as it was: the input itself, so that a caller sees the value it gave.
      out.push_back(bsplines[i]);
    } else if (std::holds_alternative<bspline::Bspline<float>>(bsplines[i])) {
      bspline::Bspline<float> narrowed = bspline::astype<float>(made[i]);
      check_narrowed_multiplicity(narrowed);
      out.emplace_back(std::move(narrowed));
    } else {
      out.emplace_back(made[i]);
    }
  }
  return out;
}

}  // namespace detail

}  // namespace pantr::cad
