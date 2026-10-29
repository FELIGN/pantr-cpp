/// @file yuksel.hpp
/// @brief Yuksel's monotone-decomposition root finder and the Newton/bisection hybrid it solves
///        each monotone piece with (Layer 3 of the Builtin provider).
///
/// Yuksel, *High-Performance Polynomial Root Finding for Graphics* (2022): the roots of the
/// derivative split `[0, 1]` into intervals on which the polynomial is monotone; an interval whose
/// end values change sign holds exactly one root. The derivative chain is built bottom-up to
/// degree 1 and walked back up. Ported from the Numba kernels operation by operation (see
/// `scalar.hpp` for the precision rules). None of these functions validates anything.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/roots/detail/scalar.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace pantr::roots::detail {

/// Iteration cap of the Newton/bisection hybrid. Each iteration keeps a bracket that holds the
/// root, so the cap ends a loop whose bracket has not narrowed to the tolerance (a tolerance below
/// the spacing of doubles never does) and the bracket's midpoint is returned, a point within the
/// current bracket of the root: what is lost at the cap is width, not the root. Pure bisection
/// reaches the strict float64 tolerance in 50 halvings. An admitted bound, kept from the previous
/// release.
inline constexpr int max_newton_iter = 64;

/// Scale-relative zero test factor of Yuksel's endpoint checks: an endpoint value within
/// `8 * eps * (max - min)` of zero counts as a root. Inherited, not derived (the previous release
/// says so too); it has no absolute floor, which keeps the finder invariant under scaling.
inline constexpr double boundary_eps_factor = 8.0;

/// `np.min` / `np.max` of a span: NaN propagates (any NaN element gives NaN).
template <class S>
[[nodiscard]] double nan_extreme(std::span<const S> v, bool want_max) noexcept {
  double r = static_cast<double>(v[0]);
  for (const S x : v) {
    const auto d = static_cast<double>(x);
    if (std::isnan(d)) {
      return d;
    }
    r = want_max ? std::max(r, d) : std::min(r, d);
  }
  return r;
}

/// Find the root of a polynomial known to be monotone on `[0, 1]` (the public monotone solver).
///
/// Starts from false position clamped to `[0.1, 0.9]`, then alternates a bracket update with a
/// Newton step, bisecting when Newton leaves the bracket. Stops when the bracket is no wider than
/// `param_tol`, or after `max_iter` iterations (see `max_newton_iter`).
///
/// Args:
///     coeff (std::span<const S>): Coefficients.
///     param_tol (double): Bracket width at which to stop.
///     max_iter (int): Iteration cap.
///
/// Returns:
///     double: The midpoint of the final bracket, or NaN when the end values do not change sign.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat S>
[[nodiscard]] double solve_monotone(std::span<const S> coeff, double param_tol,
                                    int max_iter = max_newton_iter) {
  double lo = 0.0;
  double hi = 1.0;
  // The end values stay in `S`, and so does the false-position quotient: Numba's `float()` of a
  // `float32` element is a `float32`.
  const S f_lo = coeff[0];
  const S f_hi = coeff.back();
  if (have_same_sign(f_lo, f_hi)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double x = 0.5 * (lo + hi);
  const S span_f = f_hi - f_lo;
  if (std::abs(span_f) > 0) {
    const S quotient = (-f_lo) / span_f;
    const double step = static_cast<double>(quotient) * (hi - lo);
    x = lo + step;
    const double margin = 0.1 * (hi - lo);
    x = std::max(lo + margin, std::min(x, hi - margin));
  }
  for (int it = 0; it < max_iter; ++it) {
    const auto [fx, dfx] = eval_and_deriv<S>(coeff, x);
    if (spans_zero(f_lo, fx)) {
      hi = x;
    } else {
      lo = x;
    }
    if ((hi - lo) <= param_tol) {
      break;
    }
    const double x_new = std::abs(dfx) > 0.0 ? x - fx / dfx : 0.5 * (lo + hi);
    x = (lo < x_new && x_new < hi) ? x_new : 0.5 * (lo + hi);
  }
  return 0.5 * (lo + hi);
}

/// Find the one root of a polynomial on a monotone sub-interval `[lo, hi]` (no reparametrisation),
/// by the same hybrid started at the midpoint.
///
/// Args:
///     coeff (std::span<const S>): Coefficients on `[0, 1]`.
///     lo (double): Left end.
///     hi (double): Right end.
///     f_lo (double): `f(lo)`.
///     tol (double): Bracket width at which to stop.
///     max_iter (int): Iteration cap (see `max_newton_iter`).
///
/// Returns:
///     double: The midpoint of the final bracket.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat S>
[[nodiscard]] double solve_on_interval(std::span<const S> coeff, double lo, double hi, double f_lo,
                                       double tol, int max_iter = max_newton_iter) {
  double x = 0.5 * (lo + hi);
  for (int it = 0; it < max_iter; ++it) {
    const auto [fx, dfx] = eval_and_deriv<S>(coeff, x);
    if (spans_zero(f_lo, fx)) {
      hi = x;
    } else {
      lo = x;
    }
    if ((hi - lo) <= tol) {
      return 0.5 * (lo + hi);
    }
    const double x_new = std::abs(dfx) > 0.0 ? x - fx / dfx : 0.5 * (lo + hi);
    x = (lo < x_new && x_new < hi) ? x_new : 0.5 * (lo + hi);
  }
  return 0.5 * (lo + hi);
}

/// Walk the monotone intervals cut by the sorted critical points `crit` and collect the roots: an
/// endpoint whose value is within `boundary_eps` of zero, and one root per interval whose end
/// values change sign. At most `n` roots (the degree) are kept.
///
/// Args:
///     coeff (std::span<const S>): Coefficients, degree `n = coeff.size() - 1 >= 1`.
///     crit (std::span<const double>): Sorted critical points.
///     tol (double): Bracket width, and the width below which an interval is skipped.
///
/// Returns:
///     std::vector<double>: The roots, ascending.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat S>
[[nodiscard]] std::vector<double> roots_at_level(std::span<const S> coeff,
                                                 std::span<const double> crit, double tol) {
  const std::size_t n = coeff.size() - 1;
  std::vector<double> roots;
  roots.reserve(n);
  // The extremes are `S` values (Numba's `float()` keeps `float32`), so is their difference.
  const auto hi_c = static_cast<S>(nan_extreme(coeff, true));
  const auto lo_c = static_cast<S>(nan_extreme(coeff, false));
  const S spread = hi_c - lo_c;
  const double scale = std::abs(static_cast<double>(spread));
  const double boundary_eps = scale * dbl_epsilon * boundary_eps_factor;
  const auto value_at = [&](double t) {
    return t < 1.0 ? eval<S>(coeff, t) : static_cast<double>(coeff[n]);
  };
  const auto push = [&](double r) {
    if (roots.size() < n) {
      roots.push_back(r);
    }
  };
  double prev_t = 0.0;
  double f_prev = static_cast<double>(coeff[0]);
  for (std::size_t k = 0; k <= crit.size(); ++k) {
    const double curr_t = k < crit.size() ? crit[k] : 1.0;
    if (curr_t - prev_t < tol) {
      f_prev = value_at(curr_t);
      prev_t = curr_t;
      continue;
    }
    const double f_curr = value_at(curr_t);
    if (std::abs(f_prev) <= boundary_eps) {
      if (roots.empty() || std::abs(roots.back() - prev_t) > tol) {
        push(prev_t);
        f_prev = f_curr;
        prev_t = curr_t;
        continue;
      }
    }
    if (have_opposite_signs(f_prev, f_curr)) {
      push(solve_on_interval<S>(coeff, prev_t, curr_t, f_prev, tol));
    }
    f_prev = f_curr;
    prev_t = curr_t;
  }
  if (std::abs(f_prev) <= boundary_eps && (roots.empty() || std::abs(roots.back() - 1.0) > tol)) {
    push(1.0);
  }
  return roots;
}

/// Find all roots of a Bernstein polynomial on `[0, 1]` by Yuksel's algorithm.
///
/// Args:
///     coeff (std::span<const S>): Coefficients, `n + 1` of them.
///     tol (double): Parametric tolerance.
///
/// Returns:
///     std::vector<double>: The roots, ascending (the order the walk produces them in).
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat S>
[[nodiscard]] std::vector<double> yuksel_roots(std::span<const S> coeff, double tol) {
  const std::size_t n = coeff.size() - 1;
  if (n == 0) {
    return {};
  }
  S d_min = coeff[0];
  S d_max = coeff[0];
  for (std::size_t i = 1; i <= n; ++i) {
    d_min = std::min(d_min, coeff[i]);
    d_max = std::max(d_max, coeff[i]);
  }
  if (d_min > 0 || d_max < 0) {
    return {};
  }
  if (n == 1) {
    const S c0 = coeff[0];
    const S c1 = coeff[1];
    if (c0 == c1) {
      return {};
    }
    const S root = c0 / (c0 - c1);  // in S, as the Numba kernel divided array elements
    if (0 <= root && root <= 1) {
      return {static_cast<double>(root)};
    }
    return {};
  }
  // derivs[lev] has degree n - 1 - lev: n - lev coefficients (row length n, padded).
  std::vector<double> derivs((n - 1) * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    const S diff = coeff[i + 1] - coeff[i];
    derivs[i] = static_cast<double>(diff);
  }
  for (std::size_t lev = 1; lev + 1 < n; ++lev) {
    const std::size_t sz_prev = n - lev;
    for (std::size_t i = 0; i < sz_prev; ++i) {
      derivs[lev * n + i] = derivs[(lev - 1) * n + i + 1] - derivs[(lev - 1) * n + i];
    }
  }
  const std::size_t deepest = n - 2;
  std::vector<double> crit;
  {
    const double c0 = derivs[deepest * n];
    const double c1 = derivs[deepest * n + 1];
    if (c0 != c1) {
      const double r = c0 / (c0 - c1);
      if (0.0 <= r && r <= 1.0) {
        crit.push_back(r);
      }
    }
  }
  for (std::size_t lev = deepest; lev-- > 0;) {
    const std::size_t deg_lev = n - 1 - lev;
    const std::span<const double> coeff_lev(derivs.data() + lev * n, deg_lev + 1);
    double lo_val = coeff_lev[0];
    double hi_val = coeff_lev[0];
    for (std::size_t i = 1; i <= deg_lev; ++i) {
      lo_val = std::min(lo_val, coeff_lev[i]);
      hi_val = std::max(hi_val, coeff_lev[i]);
    }
    if (lo_val > 0.0 || hi_val < 0.0) {
      crit.clear();
      continue;
    }
    if (crit.empty()) {
      // Monotone at this level: at most one root.
      const double f_lo = coeff_lev[0];
      const double f_hi = coeff_lev[deg_lev];
      const double boundary_eps = std::abs(hi_val - lo_val) * dbl_epsilon * boundary_eps_factor;
      if (std::abs(f_lo) <= boundary_eps) {
        crit.push_back(0.0);
      } else if (have_opposite_signs(f_lo, f_hi)) {
        crit.push_back(solve_on_interval<double>(coeff_lev, 0.0, 1.0, f_lo, tol));
      } else if (std::abs(f_hi) <= boundary_eps) {
        crit.push_back(1.0);
      }
      continue;
    }
    crit = roots_at_level<double>(coeff_lev, crit, tol);
  }
  return roots_at_level<S>(coeff, crit, tol);
}

}  // namespace pantr::roots::detail
