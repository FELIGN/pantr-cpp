/// @file derivative.hpp
/// @brief The first partial derivative (hodograph) of a non-rational B-spline, and the common
/// refinement of two B-splines in one direction.
///
/// The derivative of a non-rational B-spline of degree `p` in a direction is a B-spline of degree
/// `p - 1` there, over the knot vector without its first and last knots. A rational B-spline's
/// derivative runs through the quotient rule, which multiplies B-splines and is not part of this
/// header: `refine_to_common_space_1d` and `derivative_keep_degree_nonrational` are the pieces it
/// calls.
#pragma once

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/degree.hpp"
#include "pantr/bspline/detail/refine_common.hpp"
#include "pantr/bspline/knot_insertion.hpp"
#include "pantr/bspline/knots.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/binomial.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bspline {

/// Compute the derivative control points for a 1D non-rational B-spline.
///
/// The standard B-spline derivative formula, `Q[i] = p (P[i+1] - P[i]) / (T[i+p+1] - T[i+1])`, in
/// the storage type. A multiplicity-`(degree + 1)` interior knot makes the denominator zero (a
/// `C^-1` spline): the derivative control point there is zero, as the previous release's masked
/// division gave it.
///
/// Args:
///     knots (std::span<const T>): The knot vector of the original B-spline, `n + p + 1` entries.
///     degree (int): Degree of the original B-spline, `p >= 1`.
///     ctrl (mdview<const T, 2>): Control points, extents `(n, cols)`.
///
/// Returns:
///     NdArray<T>: Derivative control points, extents `(n - 1, cols)`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): `n >= 1`, `p >= 1` and
///     `knots.size() == n + p + 1`.
template <StorageFloat T>
[[nodiscard]] NdArray<T> derivative_ctrl_1d(std::span<const T> knots, int degree,
                                            mdview<const T, 2> ctrl) {
  const std::size_t n = ctrl.extent(0);
  const std::size_t cols = ctrl.extent(1);
  const auto p = static_cast<std::size_t>(degree);
  const auto scale = static_cast<T>(degree);
  NdArray<T> out({n - 1, cols}, for_overwrite);
  for (std::size_t i = 0; i + 1 < n; ++i) {
    const T denom = static_cast<T>(knots[p + 1 + i] - knots[1 + i]);
    for (std::size_t c = 0; c < cols; ++c) {
      const T diff = static_cast<T>(ctrl(i + 1, c) - ctrl(i, c));
      out.data()[i * cols + c] =
          denom != T{0} ? static_cast<T>(static_cast<T>(scale * diff) / denom) : T{0};
    }
  }
  return out;
}

namespace detail {

/// Check the direction and the degree of a derivative, and that the B-spline is non-rational.
///
/// Raises:
///     value_error: The direction is out of range; the degree there is 0; or the B-spline is
///         rational.
template <StorageFloat T>
void check_derivative_request(const Bspline<T>& b, long long direction, const char* what) {
  if (b.rational()) {
    throw value_error(std::string(what) + " is for non-rational B-splines.");
  }
  if (direction < 0 || direction >= static_cast<long long>(b.dim())) {
    throw value_error("direction must be in [0, " + std::to_string(b.dim()) + "), got " +
                      std::to_string(direction) + ".");
  }
  if (b.space().spaces()[static_cast<std::size_t>(direction)].degree() < 1) {
    throw value_error("Derivative of a degree-0 B-spline is not defined.");
  }
}

}  // namespace detail

/// Compute the partial derivative of a non-rational B-spline.
///
/// Handles open and periodic directions. For a periodic one, the control points are first expanded
/// to the full representation (`full[i] = stored[i mod n_stored]`), the derivative formula is
/// applied, and the result is trimmed back to the periodic count. The other directions are
/// unchanged.
///
/// Args:
///     b (const Bspline<T>&): A non-rational B-spline.
///     direction (long long): The parametric direction, in `[0, dim)`, of degree at least 1.
///
/// Returns:
///     Bspline<T>: The derivative: degree `p - 1` in `direction`, the other degrees unchanged.
///
/// Raises:
///     value_error: `direction` is out of range; the degree there is 0; or `b` is rational.
template <StorageFloat T>
[[nodiscard]] Bspline<T> derivative_nonrational(const Bspline<T>& b, long long direction) {
  detail::check_derivative_request(b, direction, "derivative_nonrational");
  const auto dir = static_cast<std::size_t>(direction);
  return detail::map_directions<T>(
      b, [&](std::size_t d) { return d == dir; },
      [&](std::size_t /*d*/, const BsplineSpace1D<T>& space,
          mdview<const T, 2> flat) -> std::pair<BsplineSpace1D<T>, NdArray<T>> {
        const std::span<const T> knots = space.knots();
        const int p = space.degree();
        const bool periodic = space.periodic();
        const std::span<const T> new_knots = knots.subspan(1, knots.size() - 2);
        const int new_degree = p - 1;

        NdArray<T> deriv = detail::empty_net<T>();
        if (periodic) {
          // Expand the stored control points to the full representation.
          const std::size_t n_stored = flat.extent(0);
          const std::size_t cols = flat.extent(1);
          const std::size_t n_full = knots.size() - static_cast<std::size_t>(p) - 1;
          NdArray<T> full({n_full, cols}, for_overwrite);
          for (std::size_t i = 0; i < n_full; ++i) {
            const T* src = flat.data_handle() + (i % n_stored) * cols;
            std::copy(src, src + cols, full.data() + i * cols);
          }
          deriv = derivative_ctrl_1d<T>(knots, p, full.template view<2>());
        } else {
          deriv = derivative_ctrl_1d<T>(knots, p, flat);
        }
        if (periodic) {
          // Trim to the periodic control point count of the derivative's knot vector.
          const auto n_new = static_cast<std::size_t>(
              num_basis<T>(new_knots, new_degree, true, space.tolerance()));
          const std::size_t cols = deriv.shape()[1];
          NdArray<T> trimmed({n_new, cols}, for_overwrite);
          std::copy(deriv.data(), deriv.data() + n_new * cols, trimmed.data());
          deriv = std::move(trimmed);
        }
        return {BsplineSpace1D<T>(new_knots, new_degree, periodic), std::move(deriv)};
      });
}

/// Compute the partial derivative of a non-rational B-spline, keeping its degree.
///
/// The derivative (degree `p - 1`) is degree-elevated by 1 to restore the degree `p`, working on
/// the arrays of the one direction (no intermediate B-spline). A periodic direction comes back in
/// open representation (degree elevation does not support periodic); every other direction keeps
/// the representation it had.
///
/// Degree elevation requires an open knot vector. Only `direction` is converted: opening every
/// direction would change the representation of the axes the operation never touches.
///
/// Args:
///     b (const Bspline<T>&): A non-rational B-spline.
///     direction (long long): The parametric direction, in `[0, dim)`, of degree at least 1.
///
/// Returns:
///     Bspline<T>: The derivative, of the degree of `b`.
///
/// Raises:
///     value_error: `direction` is out of range; the degree there is 0; `b` is rational; the
///         degree exceeds the exactness envelope of the binomial kernel (`bincoeff_max_n`); or the
///         hodograph's own knot vector, `knots[1:-1]` at degree `p - 1`, is not clamped, since
///         re-elevating it runs A5.9 and A5.9 assumes it is. A clamped direction always yields a
///         clamped hodograph, and so does **any** direction of degree 1, whose hodograph is
///         degree 0 and has no end runs to lack; every other unclamped direction is refused.
template <StorageFloat T>
[[nodiscard]] Bspline<T> derivative_keep_degree_nonrational(const Bspline<T>& b,
                                                            long long direction) {
  detail::check_derivative_request(b, direction, "derivative_keep_degree_nonrational");
  const auto dir = static_cast<std::size_t>(direction);
  // Re-elevating from `p - 1` back to `p` builds the Bézier table from `C(p, i)`. This is the only
  // degree-elevating path a caller can reach without asking for elevation: every derivative of a
  // *rational* B-spline routes here, `keep_degree` or not.
  const int p0 = b.space().spaces()[dir].degree();
  check_bincoeff_envelope(
      p0, "Degree-preserving derivative of a degree-" + std::to_string(p0) + " B-spline");
  return detail::map_directions<T>(
      b, [&](std::size_t d) { return d == dir; },
      [&](std::size_t d, const BsplineSpace1D<T>& space,
          mdview<const T, 2> flat) -> std::pair<BsplineSpace1D<T>, NdArray<T>> {
        const int p = space.degree();
        KnotsAndCtrl<T> open{std::vector<T>(space.knots().begin(), space.knots().end()),
                             detail::empty_net<T>()};
        mdview<const T, 2> pts = flat;
        if (space.periodic()) {
          open = to_open_1d<T>(space.knots(), p, flat, true, space.tolerance());
          pts = open.ctrl.template view<2>();
        }
        const std::span<const T> knots =
            space.periodic() ? std::span<const T>(open.knots) : space.knots();
        // Derivative (degree p to p - 1) and degree elevation (p - 1 to p), on arrays.
        const NdArray<T> deriv = derivative_ctrl_1d<T>(knots, p, pts);
        const std::span<const T> deriv_knots = knots.subspan(1, knots.size() - 2);
        // A5.9's precondition, checked on the vector that actually reaches it rather than on the
        // space: an original clamped at both ends implies `knots[1:-1]` is too, and the converse
        // does not hold, so testing the original would refuse a direction this path serves
        // correctly. `p == 1` is that case and it is not a loophole: the hodograph is degree 0,
        // whose domain is its whole knot vector, so A5.9 asks nothing of its ends and
        // re-elevating it is right whatever the original's ends were. The difference quotient
        // above assumes nothing about the ends either, which is why the plain derivative stays
        // available on an unclamped non-rational direction.
        check_clamped_knots<T>(deriv_knots, p - 1,
                               "Degree-preserving derivative in direction " + std::to_string(d) +
                                   " (its degree-" + std::to_string(p - 1) + " hodograph)");
        KnotsAndCtrl<T> elevated =
            detail::degree_elevate_kernel<T>(p - 1, deriv.template view<2>(), deriv_knots, 1);
        return {BsplineSpace1D<T>(std::span<const T>(elevated.knots), p), std::move(elevated.ctrl)};
      });
}

/// Compute the knots present in one knot vector but missing, or of lower multiplicity, in another.
///
/// For each unique knot of `unique_other`, how many more insertions `unique_self` needs to match
/// its multiplicity there.
///
/// Args:
///     unique_self (std::span<const T>): Unique knots of the first, increasing.
///     mults_self (std::span<const std::int64_t>): Their multiplicities.
///     unique_other (std::span<const T>): Unique knots of the other, increasing.
///     mults_other (std::span<const std::int64_t>): Their multiplicities.
///     tol (double): Absolute parametric tolerance for the coincidence tests.
///
/// Returns:
///     std::vector<T>: The knots to insert into the first, sorted, repeated as needed.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] std::vector<T> compute_missing_knots(std::span<const T> unique_self,
                                                   std::span<const std::int64_t> mults_self,
                                                   std::span<const T> unique_other,
                                                   std::span<const std::int64_t> mults_other,
                                                   double tol) {
  std::vector<T> out;
  std::size_t j = 0;
  for (std::size_t i = 0; i < unique_other.size(); ++i) {
    const auto xi = static_cast<double>(unique_other[i]);
    const std::int64_t m_other = mults_other[i];
    std::int64_t m_self = 0;
    // Find the matching knot in the first.
    while (j < unique_self.size() && static_cast<double>(unique_self[j]) < xi - tol) {
      ++j;
    }
    if (j < unique_self.size() && std::abs(static_cast<double>(unique_self[j]) - xi) <= tol) {
      m_self = mults_self[j];
    }
    if (m_other - m_self > 0) {
      out.insert(out.end(), static_cast<std::size_t>(m_other - m_self), unique_other[i]);
    }
  }
  return out;
}

/// Refine two B-splines to share a knot vector in one direction.
///
/// Computes the knots missing from each in `direction` (up to the larger of the two directions'
/// tolerances) and inserts them, so both have the same knots there.
///
/// Args:
///     a (const Bspline<T>&): The first B-spline.
///     b (const Bspline<T>&): The second B-spline, with the same dimension as `a`.
///     direction (std::size_t): The parametric direction, `< dim` (unchecked).
///
/// Returns:
///     std::pair<Bspline<T>, Bspline<T>>: Both refined (a value that needed no insertion is
///     returned as it is, sharing its representation).
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] std::pair<Bspline<T>, Bspline<T>> refine_to_common_space_1d(const Bspline<T>& a,
                                                                          const Bspline<T>& b,
                                                                          std::size_t direction) {
  const BsplineSpace1D<T> space_a = a.space().spaces()[direction];
  const BsplineSpace1D<T> space_b = b.space().spaces()[direction];
  const double tol = (std::max)(space_a.tolerance(), space_b.tolerance());

  const auto [unique_a, mults_a] = space_a.unique_knots_and_multiplicity(true);
  const auto [unique_b, mults_b] = space_b.unique_knots_and_multiplicity(true);

  // Knots to insert into `a` (present in `b` but missing or of lower multiplicity in `a`), and into
  // `b`, the other way.
  const std::vector<T> missing_a =
      compute_missing_knots<T>(unique_a, mults_a, unique_b, mults_b, tol);
  const std::vector<T> missing_b =
      compute_missing_knots<T>(unique_b, mults_b, unique_a, mults_a, tol);

  const auto refine = [&](const Bspline<T>& s, const std::vector<T>& missing) {
    if (missing.empty()) {
      return s;
    }
    std::vector<std::span<const T>> per_dim(s.dim());
    per_dim[direction] = missing;
    return insert_knots<T>(s, per_dim);
  };
  Bspline<T> ra = refine(a, missing_a);
  Bspline<T> rb = refine(b, missing_b);
  return {std::move(ra), std::move(rb)};
}

}  // namespace pantr::bspline
