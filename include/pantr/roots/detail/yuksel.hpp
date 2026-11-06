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
#include "pantr/core/errors.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/roots/detail/scalar.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::roots::detail {

/// Iteration cap of the Newton/bisection hybrid. Each iteration keeps a bracket that holds the
/// root, so the cap ends a loop whose bracket has not narrowed to the tolerance: a tolerance below
/// the spacing of doubles never does. The bisection safeguard of `safeguarded_newton` halves the
/// bracket at least every two iterations, so 64 iterations reach a width of `2^-32` whatever the
/// Newton steps do; a finer tolerance (the strict float64 one is `2^-49`) also relies on Newton's
/// progress, which a multiple root of the polynomial (as in `(t - 1/3)^3`) slows to a linear rate
/// and its evaluation's rounding blurs. The bracket's midpoint can then be far from the root, so
/// the solvers say that the cap was reached (`Bracketed::converged`) and their callers report it.
/// An admitted bound, kept from the previous release.
inline constexpr int max_newton_iter = 64;

/// The result of the Newton/bisection hybrid.
///
/// Attributes:
///     x (double): The midpoint of the final bracket (NaN for `solve_monotone` without a sign
///         change).
///     lo (double): Left end of the final bracket.
///     hi (double): Right end of the final bracket.
///     converged (bool): Whether the bracket narrowed to the tolerance; false when the iteration
///         cap was reached first, and `x` is then only somewhere in `[lo, hi]`.
struct [[nodiscard]] Bracketed {
  double x;
  double lo;
  double hi;
  bool converged;
};

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
    r = want_max ? (std::max)(r, d) : (std::min)(r, d);
  }
  return r;
}

/// Narrow a bracket of a polynomial's sign change by safeguarded Newton steps, from `x`.
///
/// Each iteration evaluates at `x`, replaces the end of `[lo, hi]` on the same side of the sign
/// change, and stops once the bracket is no wider than `tol`, or at once, with the bracket `[x,
/// x]`, where the evaluation is an exact zero (the sign change is then at `x` itself). The next
/// point is the Newton step from whichever end has the smaller `|f|`, with two safeguards, the ones
/// of `rtsafe` (Press et al., Numerical Recipes, section 9.4) and Brent's zeroin (Brent,
/// Algorithms for Minimization without Derivatives, 1973, chapter 4), recalled and not re-read:
///
/// - **Bisection**: the midpoint is taken
///   when the Newton step leaves the bracket or the derivative there is zero, and when the
///   iteration just made, not itself a bisection, failed to halve the bracket. A Newton sequence
///   that stays on one side of the root, which a `float32` evaluation's rounding produces near a
///   simple root and a multiple root produces in any precision, otherwise never moves the far end.
///   With it the bracket at least halves every two iterations.
/// - **Minimum step**: a Newton point is kept at least `tol / 2` inside both ends. Once
///   that end is within `tol / 2` of the sign change, the point is then on its far side and
///   the bracket closes to `tol`, where a Newton step of its own length would leave the far end
///   in place.
///
/// Args:
///     coeff (std::span<const S>): Coefficients on `[0, 1]`.
///     lo (double): Left end of the bracket.
///     hi (double): Right end of the bracket.
///     x (double): First point, inside the bracket.
///     f_lo (double): The polynomial's value at `lo`, whose sign marks the left side.
///     tol (double): Bracket width at which to stop.
///     max_iter (int): Iteration cap (see `max_newton_iter`).
///
/// Returns:
///     Bracketed: The midpoint of the final bracket, and whether it narrowed to `tol` within
///     `max_iter`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat S>
[[nodiscard]] Bracketed safeguarded_newton(std::span<const S> coeff, double lo, double hi, double x,
                                           double f_lo, double tol, int max_iter) {
  // Value and derivative at each end of the bracket (0: `lo`, 1: `hi`) once it has been evaluated
  // there; the starting ends have none.
  std::array<double, 2> f_end{0.0, 0.0};
  std::array<double, 2> df_end{0.0, 0.0};
  std::array<bool, 2> evaluated{false, false};
  bool bisected = false;
  for (int it = 0; it < max_iter; ++it) {
    const double width = hi - lo;
    const auto [fx, dfx] = eval_and_deriv<S>(coeff, x);
    if (fx == 0.0) {
      // An exact zero of the evaluated polynomial: `x` is the sign change itself.
      return {x, x, x, true};
    }
    const std::size_t side = spans_zero(f_lo, fx) ? 1 : 0;
    (side == 1 ? hi : lo) = x;
    f_end[side] = fx;
    df_end[side] = dfx;
    evaluated[side] = true;
    if ((hi - lo) <= tol) {
      return {0.5 * (lo + hi), lo, hi, true};
    }
    const std::size_t other = 1 - side;
    const std::size_t best =
        evaluated[other] && std::abs(f_end[other]) < std::abs(f_end[side]) ? other : side;
    const double at = best == 1 ? hi : lo;
    const double newton = std::abs(df_end[best]) > 0.0 ? at - f_end[best] / df_end[best]
                                                       : std::numeric_limits<double>::quiet_NaN();
    const bool stalled = !bisected && (hi - lo) > 0.5 * width;
    if (stalled || !(lo <= newton && newton <= hi)) {
      x = 0.5 * (lo + hi);
      bisected = true;
    } else {
      x = (std::min)((std::max)(newton, lo + 0.5 * tol), hi - 0.5 * tol);
      bisected = false;
    }
  }
  return {0.5 * (lo + hi), lo, hi, false};
}

/// Find the root of a polynomial known to be monotone on `[0, 1]` (the public monotone solver).
///
/// Starts from false position clamped to `[0.1, 0.9]`, then narrows the bracket by
/// `safeguarded_newton`. Stops when the bracket is no wider than `param_tol`, or after `max_iter`
/// iterations (see `max_newton_iter`).
///
/// Args:
///     coeff (std::span<const S>): Coefficients.
///     param_tol (double): Bracket width at which to stop.
///     max_iter (int): Iteration cap.
///
/// Returns:
///     Bracketed: The midpoint of the final bracket (NaN, converged, when the end values do not
///     change sign), and whether the bracket narrowed to `param_tol` within `max_iter`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat S>
[[nodiscard]] Bracketed solve_monotone(std::span<const S> coeff, double param_tol,
                                       int max_iter = max_newton_iter) {
  double lo = 0.0;
  double hi = 1.0;
  // The end values stay in `S`, and so does the false-position quotient: Numba's `float()` of a
  // `float32` element is a `float32`.
  const S f_lo = coeff[0];
  const S f_hi = coeff.back();
  if (have_same_sign(static_cast<double>(f_lo), static_cast<double>(f_hi))) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, lo, hi, true};
  }
  double x = 0.5 * (lo + hi);
  const S span_f = f_hi - f_lo;
  if (std::abs(span_f) > 0) {
    const S quotient = (-f_lo) / span_f;
    const double step = static_cast<double>(quotient) * (hi - lo);
    x = lo + step;
    const double margin = 0.1 * (hi - lo);
    x = (std::max)(lo + margin, (std::min)(x, hi - margin));
  }
  return safeguarded_newton<S>(coeff, lo, hi, x, static_cast<double>(f_lo), param_tol, max_iter);
}

/// Find the one root of a polynomial on a monotone sub-interval `[lo, hi]` (no reparametrisation),
/// by the Newton/bisection hybrid of the previous release started at the midpoint: a Newton step,
/// bisecting only when it leaves the bracket.
///
/// It places the roots and the critical points of `find_roots`' Yuksel walk, and has no
/// `safeguarded_newton` safeguard: the walk's guard against reporting a root twice at a critical
/// point (`roots_at_level`) relies on where this hybrid ends near a multiple root, and the
/// safeguarded steps end elsewhere within the root's rounding band, which reports a triple root
/// twice where the previous release reported it once. A one-sided stall here reaches the cap,
/// which the walk reports as a gap.
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
///     Bracketed: The midpoint of the final bracket, and whether it narrowed to `tol` within
///     `max_iter`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat S>
[[nodiscard]] Bracketed solve_on_interval(std::span<const S> coeff, double lo, double hi,
                                          double f_lo, double tol, int max_iter = max_newton_iter) {
  double x = 0.5 * (lo + hi);
  for (int it = 0; it < max_iter; ++it) {
    const auto [fx, dfx] = eval_and_deriv<S>(coeff, x);
    if (spans_zero(f_lo, fx)) {
      hi = x;
    } else {
      lo = x;
    }
    if ((hi - lo) <= tol) {
      return {0.5 * (lo + hi), lo, hi, true};
    }
    const double x_new = std::abs(dfx) > 0.0 ? x - fx / dfx : 0.5 * (lo + hi);
    x = (lo < x_new && x_new < hi) ? x_new : 0.5 * (lo + hi);
  }
  return {0.5 * (lo + hi), lo, hi, false};
}

/// Record a hybrid solve's bracket as a gap when it reached the iteration cap, and get its point.
///
/// Args:
///     r (const Bracketed&): The solve's result.
///     gaps (std::vector<std::pair<double, double>>&): The gaps, appended to.
///
/// Returns:
///     double: `r.x`.
[[nodiscard]] inline double take_bracketed(const Bracketed& r,
                                           std::vector<std::pair<double, double>>& gaps) {
  if (!r.converged) {
    gaps.emplace_back(r.lo, r.hi);
  }
  return r.x;
}

/// What Yuksel's algorithm found.
///
/// Attributes:
///     roots (std::vector<double>): The roots, ascending.
///     gaps (std::vector<std::pair<double, double>>): The brackets of the solves that reached the
///         iteration cap (of a root, or of a critical point of a derivative, which places the
///         monotone intervals); a root near one may be misplaced within it, or missed.
struct YukselResult {
  std::vector<double> roots;
  std::vector<std::pair<double, double>> gaps;
};

/// Walk the monotone intervals cut by the sorted critical points `crit` and collect the roots: an
/// endpoint whose value is within `boundary_eps` of zero, and one root per interval whose end
/// values change sign. At most `n` roots (the degree) are kept.
///
/// Args:
///     coeff (std::span<const S>): Coefficients, degree `n = coeff.size() - 1 >= 1`.
///     crit (std::span<const double>): Sorted critical points.
///     tol (double): Bracket width, and the width below which an interval is skipped.
///     gaps (std::vector<std::pair<double, double>>&): Appended to: the bracket of every solve
///         that reached the iteration cap.
///
/// Returns:
///     std::vector<double>: The roots, ascending.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat S>
[[nodiscard]] std::vector<double> roots_at_level(std::span<const S> coeff,
                                                 std::span<const double> crit, double tol,
                                                 std::vector<std::pair<double, double>>& gaps) {
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
      push(take_bracketed(solve_on_interval<S>(coeff, prev_t, curr_t, f_prev, tol), gaps));
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
///     YukselResult: The roots, ascending (the order the walk produces them in), and the brackets
///     of the solves that reached the iteration cap.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat S>
[[nodiscard]] YukselResult yuksel_roots(std::span<const S> coeff, double tol) {
  const std::size_t n = coeff.size() - 1;
  if (n == 0) {
    return {};
  }
  S d_min = coeff[0];
  S d_max = coeff[0];
  for (std::size_t i = 1; i <= n; ++i) {
    d_min = (std::min)(d_min, coeff[i]);
    d_max = (std::max)(d_max, coeff[i]);
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
      return {{static_cast<double>(root)}, {}};
    }
    return {};
  }
  // derivs[lev] has degree n - 1 - lev: n - lev coefficients (row length n, padded). `(n - 1) * n`
  // wraps `size_t` for `n` above 2^32 on a 64-bit target (a coefficient array of 16 GiB of
  // `float`), and the writes below would then overrun the buffer: refused instead.
  if (n - 1 > (std::numeric_limits<std::size_t>::max)() / n) {
    throw capacity_error("degree " + std::to_string(n) +
                         " is too large for the derivative table of Yuksel's algorithm");
  }
  YukselResult out;
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
      lo_val = (std::min)(lo_val, coeff_lev[i]);
      hi_val = (std::max)(hi_val, coeff_lev[i]);
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
        crit.push_back(
            take_bracketed(solve_on_interval<double>(coeff_lev, 0.0, 1.0, f_lo, tol), out.gaps));
      } else if (std::abs(f_hi) <= boundary_eps) {
        crit.push_back(1.0);
      }
      continue;
    }
    crit = roots_at_level<double>(coeff_lev, crit, tol, out.gaps);
  }
  out.roots = roots_at_level<S>(coeff, crit, tol, out.gaps);
  return out;
}

}  // namespace pantr::roots::detail
