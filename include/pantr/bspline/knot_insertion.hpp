/// @file knot_insertion.hpp
/// @brief Knot insertion by the Oslo algorithm, conversion to open form, and uniform subdivision.
///
/// The validated entry points over the kernels of `detail/oslo.hpp`: a knot vector with more knots
/// and the control points that keep the geometry exactly. A periodic direction is converted to open
/// form, refined and converted back, so that periodicity is preserved.
#pragma once

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/detail/oslo.hpp"
#include "pantr/bspline/detail/refine_common.hpp"
#include "pantr/bspline/knots.hpp"
#include "pantr/bspline/periodic.hpp"
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
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bspline {

/// Compute the dense Oslo refinement matrix `alpha` with `Q = alpha P`.
///
/// Args:
///     degree (int): Degree of the B-spline, `>= 0`.
///     old_knots (std::span<const T>): Original knot vector, `n + degree + 2` entries.
///     new_knots (std::span<const T>): Refined knot vector, `m + degree + 2` entries. It should be
///         a superset of `old_knots`; that is not checked (callers hand knot vectors that agree
///         only up to a tolerance), and a vector that is not one gives the recurrence's value.
///
/// Returns:
///     NdArray<T>: Extents `(m + 1, n + 1)`.
///
/// Raises:
///     value_error: `degree < 0`, or a knot vector with fewer than `degree + 2` knots.
template <StorageFloat T>
[[nodiscard]] NdArray<T> oslo_matrix(int degree, std::span<const T> old_knots,
                                     std::span<const T> new_knots) {
  if (degree < 0) {
    throw value_error("degree must be non-negative");
  }
  const auto min_size = static_cast<std::size_t>(degree) + 2;
  if (old_knots.size() < min_size || new_knots.size() < min_size) {
    throw value_error("Both knot vectors need at least degree + 2 = " + std::to_string(min_size) +
                      " knots, got " + std::to_string(old_knots.size()) + " and " +
                      std::to_string(new_knots.size()) + ".");
  }
  return detail::oslo_matrix_dense<T>(degree, old_knots, new_knots);
}

/// Validate new knots and compute the merged (refined) knot vector.
///
/// Args:
///     knots (std::span<const T>): The original knot vector, non-decreasing.
///     degree (int): Degree.
///     new_knots (std::span<const T>): Knot values to insert, of the storage type already.
///     tol (double): Absolute parametric tolerance for the domain and multiplicity checks.
///
/// Returns:
///     std::vector<T>: The merged, sorted knot vector.
///
/// Raises:
///     value_error: `new_knots` is empty; a value lies outside the domain (up to `tol`); or a
///         knot's resulting multiplicity would exceed `degree + 1`.
template <StorageFloat T>
[[nodiscard]] std::vector<T> compute_inserted_knot_vector(std::span<const T> knots, int degree,
                                                          std::span<const T> new_knots,
                                                          double tol) {
  if (new_knots.empty()) {
    throw value_error("new_knots_to_insert must not be empty.");
  }

  std::string bad;
  for (const T v : new_knots) {
    if (!is_in_domain<T, T>(knots, degree, v, tol)) {
      bad += bad.empty() ? "" : " ";
      bad += detail::repr_of<T>(v);
    }
  }
  if (!bad.empty()) {
    const auto d = static_cast<std::size_t>(degree);
    throw value_error(
        "new_knots contains values outside the domain [" + py_repr(static_cast<double>(knots[d])) +
        ", " + py_repr(static_cast<double>(knots[knots.size() - d - 1])) + "]: [" + bad + "]");
  }

  // The sorted union: the new knots sorted (a no-op for the usual, sorted request, which is then
  // used in place) and merged with the old ones, which a space keeps non-decreasing.
  std::vector<T> sorted_copy;
  std::span<const T> sorted_new = new_knots;
  if (!std::is_sorted(new_knots.begin(), new_knots.end())) {
    sorted_copy.assign(new_knots.begin(), new_knots.end());
    std::sort(sorted_copy.begin(), sorted_copy.end());
    sorted_new = sorted_copy;
  }
  std::vector<T> merged(knots.size() + new_knots.size());
  std::merge(knots.begin(), knots.end(), sorted_new.begin(), sorted_new.end(), merged.begin());

  // No value may appear more than degree + 1 times.
  const long long max_allowed = static_cast<long long>(degree) + 1;
  const std::int64_t found = detail::max_multiplicity<T>(merged, tol);
  if (found > max_allowed) {
    throw value_error("Inserting these knots would exceed the maximum multiplicity of " +
                      std::to_string(max_allowed) +
                      ". Maximum multiplicity found: " + std::to_string(found) + ".");
  }
  return merged;
}

/// Insert knots into a 1D B-spline and compute the new control points.
///
/// Args:
///     knots (std::span<const T>): The original knot vector.
///     degree (int): Degree.
///     ctrl (mdview<const T, 2>): Control points, extents `(n + 1, cols)`.
///     new_knots (std::span<const T>): Knot values to insert.
///     tol (double): Absolute parametric tolerance.
///
/// Returns:
///     KnotsAndCtrl<T>: The refined knot vector and the new control points, `(m + 1, cols)`.
///
/// Raises:
///     value_error: The control points do not match the knots, or as
///         `compute_inserted_knot_vector`.
template <StorageFloat T>
[[nodiscard]] KnotsAndCtrl<T> insert_knots_1d(std::span<const T> knots, int degree,
                                              mdview<const T, 2> ctrl, std::span<const T> new_knots,
                                              double tol) {
  detail::check_net_matches("insert_knots", ctrl.extent(0), knots.size(), degree);
  std::vector<T> refined = compute_inserted_knot_vector<T>(knots, degree, new_knots, tol);
  NdArray<T> out = detail::insert_knots_core<T>(degree, knots, ctrl, refined);
  return KnotsAndCtrl<T>{std::move(refined), std::move(out)};
}

/// Convert a single parametric direction to an open (clamped) knot vector.
///
/// Inserts `degree + 1 - m_left` copies of the left boundary knot and `degree + 1 - m_right` of the
/// right one so that both ends have multiplicity `degree + 1`, then trims the ghost (or external)
/// knots, `degree + 1 - m_end` entries outside the domain on each side, each side with its own
/// boundary multiplicity. For a periodic direction the `n_full = size - degree - 1` control points
/// are first rebuilt by wrapping the stored ones (`full[i] = stored[i mod n_stored]`).
///
/// Args:
///     knots (std::span<const T>): The knot vector (ghost knots included if `periodic`).
///     degree (int): Degree.
///     ctrl (mdview<const T, 2>): Control points, extents `(n_stored, cols)`; `n_stored` is
///         `size - degree - 1` for a non-periodic direction.
///     periodic (bool): Whether the direction is periodic.
///     tol (double): Absolute parametric tolerance.
///
/// Returns:
///     KnotsAndCtrl<T>: The clamped knot vector and the corresponding control points.
///
/// Raises:
///     value_error: The control points do not match the knots; or the knot vector already has
///         multiplicity `degree + 1` at both ends and is not periodic.
template <StorageFloat T>
[[nodiscard]] KnotsAndCtrl<T> to_open_1d(std::span<const T> knots, int degree,
                                         mdview<const T, 2> ctrl, bool periodic, double tol) {
  const auto p = static_cast<std::size_t>(degree);
  const std::size_t n_stored = ctrl.extent(0);
  const std::size_t cols = ctrl.extent(1);
  if (periodic) {
    if (knots.size() < 2 * p + 2 || n_stored == 0) {
      throw value_error(
          "to_open: " + std::to_string(n_stored) + " control-point rows cannot be wrapped over " +
          std::to_string(knots.size()) + " knots at degree " + std::to_string(degree) + ".");
    }
  } else {
    detail::check_net_matches("to_open", n_stored, knots.size(), degree);
  }

  const T a = knots[p];
  const T b = knots[knots.size() - p - 1];
  std::size_t m_left = 0;
  std::size_t m_right = 0;
  for (std::size_t i = 0; i <= p; ++i) {
    if (static_cast<double>(std::abs(static_cast<T>(knots[i] - a))) <= tol) {
      ++m_left;
    }
    if (static_cast<double>(std::abs(static_cast<T>(knots[knots.size() - p - 1 + i] - b))) <= tol) {
      ++m_right;
    }
  }
  if (m_left == p + 1 && m_right == p + 1 && !periodic) {
    throw value_error(
        "B-spline is already open (both boundary knots have multiplicity degree + 1).");
  }

  // The full control points of the non-periodic form.
  NdArray<T> wrapped = detail::empty_net<T>();
  mdview<const T, 2> full = ctrl;
  if (periodic) {
    const std::size_t n_full = knots.size() - p - 1;
    wrapped = NdArray<T>({n_full, cols}, for_overwrite);
    for (std::size_t i = 0; i < n_full; ++i) {
      const T* src = ctrl.data_handle() + (i % n_stored) * cols;
      std::copy(src, src + cols, wrapped.data() + i * cols);
    }
    full = wrapped.template view<2>();
  }

  // Insert boundary knots to reach multiplicity degree + 1.
  std::vector<T> to_insert;
  to_insert.insert(to_insert.end(), p + 1 - m_left, a);
  to_insert.insert(to_insert.end(), p + 1 - m_right, b);

  std::vector<T> new_knots;
  NdArray<T> new_ctrl = detail::empty_net<T>();
  if (!to_insert.empty()) {
    KnotsAndCtrl<T> inserted = insert_knots_1d<T>(knots, degree, full, to_insert, tol);
    new_knots = std::move(inserted.knots);
    new_ctrl = std::move(inserted.ctrl);
  } else {
    new_knots.assign(knots.begin(), knots.end());
    new_ctrl = NdArray<T>({full.extent(0), cols}, for_overwrite);
    std::copy(full.data_handle(), full.data_handle() + full.extent(0) * cols, new_ctrl.data());
  }

  // Trim the external / ghost knots, each side counted with its own boundary multiplicity.
  const std::size_t n_trim_left = p + 1 - m_left;
  const std::size_t n_trim_right = p + 1 - m_right;
  const std::size_t end = new_knots.size() - n_trim_right;
  std::vector<T> open_knots(new_knots.begin() + static_cast<std::ptrdiff_t>(n_trim_left),
                            new_knots.begin() + static_cast<std::ptrdiff_t>(end));
  const std::size_t n_open = open_knots.size() - p - 1;
  NdArray<T> open_ctrl({n_open, cols}, for_overwrite);
  std::copy(new_ctrl.data() + n_trim_left * cols, new_ctrl.data() + (n_trim_left + n_open) * cols,
            open_ctrl.data());
  return KnotsAndCtrl<T>{std::move(open_knots), std::move(open_ctrl)};
}

/// Compute the new knots that subdivide every knot span uniformly.
///
/// For each non-zero span `[u_i, u_{i+1})` of the in-domain unique knots, `n_subdivisions - 1`
/// uniformly spaced values, each repeated `degree - regularity` times to give the requested
/// continuity. The values are `lo + i * step`, `step = (hi - lo) / n_subdivisions`, formed in
/// `double` as `numpy.linspace` forms them, then cast to the storage type.
///
/// Args:
///     knots (std::span<const T>): The knot vector.
///     degree (int): Degree.
///     tol (double): Absolute parametric tolerance for the knot classes.
///     n_subdivisions (long long): Equal sub-spans per interval, `>= 2`.
///     regularity (std::optional<long long>): Continuity order at the inserted knots, in
///         `[-1, degree - 1]`; no value means `degree - 1`.
///
/// Returns:
///     std::vector<T>: The knot values to insert, ascending.
///
/// Raises:
///     value_error: `n_subdivisions < 2`, or the regularity is outside `[-1, degree - 1]`.
template <StorageFloat T>
[[nodiscard]] std::vector<T> compute_uniform_subdivision_knots(
    std::span<const T> knots, int degree, double tol, long long n_subdivisions,
    std::optional<long long> regularity = std::nullopt) {
  if (n_subdivisions < 2) {
    throw value_error("n_subdivisions must be >= 2, got " + std::to_string(n_subdivisions));
  }
  const long long eff = regularity.has_value() ? *regularity : static_cast<long long>(degree) - 1;
  if (eff < -1 || eff > static_cast<long long>(degree) - 1) {
    throw value_error("regularity must be in [-1, degree - 1] = [-1, " +
                      std::to_string(degree - 1) + "], got " + std::to_string(eff));
  }
  const auto repeat = static_cast<std::size_t>(static_cast<long long>(degree) - eff);

  const KnotClasses<T> classes = unique_knots_and_multiplicity<T>(knots, degree, tol, true);
  const auto count = static_cast<std::size_t>(n_subdivisions);
  const auto steps = static_cast<double>(n_subdivisions);
  // The number of knots, refused when it does not fit an array (NumPy's own bound on a size).
  constexpr auto max_bytes = static_cast<std::size_t>((std::numeric_limits<std::ptrdiff_t>::max)());
  const std::size_t max_knots = max_bytes / sizeof(T);
  const std::size_t per_span = (count - 1) * repeat;
  if (count - 1 > max_knots / repeat ||
      classes.knots.size() - 1 > max_knots / (std::max)(per_span, std::size_t{1})) {
    throw capacity_error("n_subdivisions = " + std::to_string(n_subdivisions) +
                         " would insert more knots than an array can hold.");
  }
  std::vector<T> out;
  out.reserve((classes.knots.size() - 1) * per_span);
  for (std::size_t k = 0; k + 1 < classes.knots.size(); ++k) {
    const auto lo = static_cast<double>(classes.knots[k]);
    const auto hi = static_cast<double>(classes.knots[k + 1]);
    const double step = (hi - lo) / steps;
    for (std::size_t i = 1; i < count; ++i) {
      // Two statements: a fused multiply-add would differ from NumPy's separate product and sum.
      const double scaled = static_cast<double>(i) * step;
      const double value = scaled + lo;
      out.insert(out.end(), repeat, static_cast<T>(value));
    }
  }
  return out;
}

/// Insert knots in the directions of a B-spline.
///
/// The control points are updated so that the new B-spline represents the same geometry. A
/// periodic direction is converted to open form, refined and converted back.
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///     new_knots (std::span<const std::span<const T>>): For each direction the knot values to
///         insert; an empty span skips the direction. Repeated values insert the same knot
///         repeatedly.
///
/// Returns:
///     Bspline<T>: The B-spline with the same geometry and refined knot vectors.
///
/// Raises:
///     value_error: `new_knots.size() != dim` ("new_knots sequence length ..."); every direction is
///         skipped; or as `compute_inserted_knot_vector` for a direction.
template <StorageFloat T>
[[nodiscard]] Bspline<T> insert_knots(const Bspline<T>& b,
                                      std::span<const std::span<const T>> new_knots) {
  if (new_knots.size() != b.dim()) {
    throw value_error("new_knots sequence length (" + std::to_string(new_knots.size()) +
                      ") must match dim (" + std::to_string(b.dim()) + ").");
  }
  const bool any = std::any_of(new_knots.begin(), new_knots.end(),
                               [](std::span<const T> k) { return !k.empty(); });
  if (!any) {
    throw value_error("At least one direction must have a non-empty array of knots to insert.");
  }
  return detail::map_directions<T>(
      b, [&](std::size_t d) { return !new_knots[d].empty(); },
      [&](std::size_t d, const BsplineSpace1D<T>& space,
          mdview<const T, 2> flat) -> std::pair<BsplineSpace1D<T>, NdArray<T>> {
        const int p = space.degree();
        const double tol = space.tolerance();
        if (space.periodic()) {
          // Round-trip through open form to preserve periodicity.
          const long long m_bdy = space.unique_knots_and_multiplicity(true).second[0];
          KnotsAndCtrl<T> open = to_open_1d<T>(space.knots(), p, flat, true, tol);
          KnotsAndCtrl<T> refined =
              insert_knots_1d<T>(open.knots, p, open.ctrl.template view<2>(), new_knots[d], tol);
          KnotsAndCtrl<T> per =
              to_periodic_1d<T>(refined.knots, p, refined.ctrl.template view<2>(), m_bdy, tol);
          return {BsplineSpace1D<T>(std::span<const T>(per.knots), p, true), std::move(per.ctrl)};
        }
        KnotsAndCtrl<T> refined = insert_knots_1d<T>(space.knots(), p, flat, new_knots[d], tol);
        return {BsplineSpace1D<T>(std::span<const T>(refined.knots), p), std::move(refined.ctrl)};
      });
}

/// Convert every direction of a B-spline to open (clamped) non-periodic form.
///
/// Each direction that is periodic or has unclamped boundary knots gets knots inserted at the
/// domain boundaries up to multiplicity `degree + 1`, and its ghost knots stripped; directions
/// already open are left as they are.
///
/// Args:
///     b (const Bspline<T>&): The B-spline, periodic, unclamped or multi-dimensional.
///
/// Returns:
///     Bspline<T>: The B-spline with open knot vectors in all directions.
///
/// Raises:
///     value_error: Every direction is already open and non-periodic.
template <StorageFloat T>
[[nodiscard]] Bspline<T> to_open_bspline(const Bspline<T>& b) {
  const BsplineSpace<T> whole = b.space();
  const std::vector<BsplineSpace1D<T>>& spaces = whole.spaces();
  const auto is_open = [&](std::size_t d) {
    return spaces[d].has_open_knots() && !spaces[d].periodic();
  };
  bool any = false;
  for (std::size_t d = 0; d < spaces.size(); ++d) {
    any = any || !is_open(d);
  }
  if (!any) {
    throw value_error("B-spline is already open in every direction.");
  }
  return detail::map_directions<T>(
      b, [&](std::size_t d) { return !is_open(d); },
      [&](std::size_t /*d*/, const BsplineSpace1D<T>& space,
          mdview<const T, 2> flat) -> std::pair<BsplineSpace1D<T>, NdArray<T>> {
        KnotsAndCtrl<T> open =
            to_open_1d<T>(space.knots(), space.degree(), flat, space.periodic(), space.tolerance());
        return {BsplineSpace1D<T>(std::span<const T>(open.knots), space.degree(), false),
                std::move(open.ctrl)};
      });
}

/// Subdivide every knot span of the active directions uniformly.
///
/// Inserts `n_subdivisions - 1` equally spaced values in each non-zero span, each repeated
/// `degree - regularity` times, so the B-spline has `C^regularity` continuity at every inserted
/// knot. The geometry is unchanged.
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///     counts (std::span<const long long>): Equal sub-spans per interval for each direction, `dim`
///         entries; `1` leaves the direction as it is. At least one must be `>= 2`.
///     regularity (std::optional<long long>): Continuity at every inserted knot, applied to every
///         active direction, in `[-1, degree - 1]` for each; no value means `degree - 1`.
///
/// Returns:
///     Bspline<T>: The refined B-spline.
///
/// Raises:
///     value_error: `counts.size() != dim`; a count `< 1`; no count `>= 2`; or the regularity is
///         outside the range of an active direction.
template <StorageFloat T>
[[nodiscard]] Bspline<T> subdivide(const Bspline<T>& b, std::span<const long long> counts,
                                   std::optional<long long> regularity = std::nullopt) {
  const std::size_t dim = b.dim();
  if (counts.size() != dim) {
    throw value_error("n_subdivisions sequence length (" + std::to_string(counts.size()) +
                      ") must match dim (" + std::to_string(dim) + ").");
  }
  for (const long long c : counts) {
    if (c < 1) {
      throw value_error("n_subdivisions must be >= 1, got " + std::to_string(c));
    }
  }
  if (std::all_of(counts.begin(), counts.end(), [](long long c) { return c == 1; })) {
    throw value_error("At least one direction must have n_subdivisions >= 2.");
  }
  const BsplineSpace<T> whole = b.space();
  const std::vector<BsplineSpace1D<T>>& spaces = whole.spaces();
  std::vector<std::vector<T>> new_knots(dim);
  for (std::size_t d = 0; d < dim; ++d) {
    if (counts[d] == 1) {
      continue;
    }
    const long long deg = spaces[d].degree();
    const long long eff = regularity.has_value() ? *regularity : deg - 1;
    if (eff < -1 || eff > deg - 1) {
      throw value_error("regularity must be in [-1, degree - 1] = [-1, " + std::to_string(deg - 1) +
                        "] for direction " + std::to_string(d) + ", got " + std::to_string(eff));
    }
    new_knots[d] = compute_uniform_subdivision_knots<T>(spaces[d].knots(), spaces[d].degree(),
                                                        spaces[d].tolerance(), counts[d], eff);
  }
  std::vector<std::span<const T>> views;
  views.reserve(dim);
  for (const auto& k : new_knots) {
    views.emplace_back(k);
  }
  return insert_knots<T>(b, views);
}

}  // namespace pantr::bspline
