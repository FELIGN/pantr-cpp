/// @file scalar.hpp
/// @brief Layer 3 helpers of the Builtin root provider: one scalar Bernstein polynomial on `[0,
/// 1]`.
///
/// Precision follows the Numba kernels these replace, site by site (design §2.3). Arithmetic is in
/// `double`; where the Numba code worked on a copy of the input (`coeff.copy()`), the copy keeps
/// the input's storage type `S`, so a `float` polynomial is rounded to `float` after every de
/// Casteljau step, as before. Each product is formed in a statement of its own before it is added:
/// under `-ffp-contract=on` (the flag `pantr::pantr` sets; GCC >= 14 and Clang contract only within
/// one expression) that keeps it from being fused into an `fma`. The branches of these algorithms
/// are sign tests on such sums, a discrete decision a last-bit change can flip, and the Numba
/// kernels never contracted. A consumer building with `-ffp-contract=fast` loses that property.
///
/// None of these helpers validates anything.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/scalar.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace pantr::roots::detail {

/// Machine epsilon of `double`, the unit every scale-relative bound of the provider is written in
/// (the Numba code's `_DBL_EPSILON`; used for both storage types, as there).
inline constexpr double dbl_epsilon = 2.2204460492503131e-16;

/// `(1 - t) * a + t * b` as two rounded products and a rounded sum (no contraction).
///
/// Args:
///     s (double): `1 - t`, precomputed.
///     t (double): The parameter.
///     a (double): Left value.
///     b (double): Right value.
///
/// Returns:
///     double: The convex combination, rounded as the uncontracted expression is.
[[nodiscard]] inline double lerp_uncontracted(double s, double t, double a, double b) noexcept {
  const double left = s * a;
  const double right = t * b;
  return left + right;
}

/// Test whether both values are strictly positive or both strictly negative.
///
/// The product-free form of `a * b > 0`, which underflows to zero for tiny operands; a zero or a
/// NaN operand makes it false.
///
/// Args:
///     a (double): First value.
///     b (double): Second value.
///
/// Returns:
///     bool: Whether the signs are equal and nonzero.
[[nodiscard]] constexpr bool have_same_sign(double a, double b) noexcept {
  return (a > 0.0 && b > 0.0) || (a < 0.0 && b < 0.0);
}

/// Test whether the values have strictly opposite signs (the product-free `a * b < 0`).
///
/// Args:
///     a (double): First value.
///     b (double): Second value.
///
/// Returns:
///     bool: Whether one is strictly positive and the other strictly negative.
[[nodiscard]] constexpr bool have_opposite_signs(double a, double b) noexcept {
  return (a > 0.0 && b < 0.0) || (a < 0.0 && b > 0.0);
}

/// Test whether a bracket spans zero, an exact zero beside a finite partner counting as spanned
/// (the product-free `a * b <= 0`, which is NaN, hence false, for `0 * inf`).
///
/// Args:
///     a (double): Value at one end.
///     b (double): Value at the other end.
///
/// Returns:
///     bool: Whether the bracket spans zero.
[[nodiscard]] inline bool spans_zero(double a, double b) noexcept {
  if (have_opposite_signs(a, b)) {
    return true;
  }
  return (a == 0.0 && std::isfinite(b)) || (b == 0.0 && std::isfinite(a));
}

/// Evaluate a scalar Bernstein polynomial at `t` by the de Casteljau triangle.
///
/// Args:
///     coeff (std::span<const S>): Coefficients, `n + 1 >= 1` of them.
///     t (double): The parameter.
///
/// Returns:
///     double: The value (rounded to `S` at every step).
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat S>
[[nodiscard]] double eval(std::span<const S> coeff, double t) {
  std::vector<S> work(coeff.begin(), coeff.end());
  const std::size_t n = work.size() - 1;
  const double s = 1.0 - t;
  for (std::size_t k = 1; k <= n; ++k) {
    for (std::size_t i = 0; i + k <= n; ++i) {
      work[i] = static_cast<S>(lerp_uncontracted(s, t, work[i], work[i + 1]));
    }
  }
  return static_cast<double>(work[0]);
}

/// Value and first derivative of a scalar Bernstein polynomial at `t`.
///
/// Runs the triangle to its penultimate row `(d0, d1)`; the derivative is `n * (d1 - d0)`, the
/// difference taken in `S`, as the Numba kernel took it in the array's dtype.
///
/// Args:
///     coeff (std::span<const S>): Coefficients, `n + 1 >= 1` of them.
///     t (double): The parameter.
///
/// Returns:
///     std::pair<double, double>: `(f(t), f'(t))`; the derivative of a constant is 0.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat S>
[[nodiscard]] std::pair<double, double> eval_and_deriv(std::span<const S> coeff, double t) {
  const std::size_t n = coeff.size() - 1;
  if (n == 0) {
    return {static_cast<double>(coeff[0]), 0.0};
  }
  const double s = 1.0 - t;
  std::vector<S> row(coeff.begin(), coeff.end());
  for (std::size_t k = 1; k < n; ++k) {
    for (std::size_t i = 0; i + k <= n; ++i) {
      row[i] = static_cast<S>(lerp_uncontracted(s, t, row[i], row[i + 1]));
    }
  }
  const S d0 = row[0];
  const S d1 = row[1];
  const double f = lerp_uncontracted(s, t, d0, d1);
  const S diff = d1 - d0;
  return {f, static_cast<double>(n) * static_cast<double>(diff)};
}

/// Restrict a scalar Bernstein polynomial to `[lower, upper]`, reparametrised to `[0, 1]`.
///
/// Two de Casteljau passes in `double`, ordered to avoid dividing by a small number: a left pass
/// at `upper` then a right pass at `lower / upper` when `|upper| >= |lower - 1|`, else a right pass
/// at `lower` then a left pass at `(upper - lower) / (1 - lower)`.
///
/// Args:
///     coeff (std::span<const S>): Coefficients.
///     lower (double): Left bound, in `[0, 1)`.
///     upper (double): Right bound, in `(0, 1]`.
///
/// Returns:
///     std::vector<double>: The restricted coefficients.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat S>
[[nodiscard]] std::vector<double> restrict_to(std::span<const S> coeff, double lower,
                                              double upper) {
  const std::size_t p = coeff.size() - 1;
  std::vector<double> d(coeff.begin(), coeff.end());
  // d[j] = d[j] * tau + d[j - 1] * (1 - tau), descending (keeps the right part).
  const auto left_pass = [&](double tau) {
    const double om = 1.0 - tau;
    for (std::size_t step = 1; step <= p; ++step) {
      for (std::size_t j = p; j >= step; --j) {
        const double a = d[j] * tau;
        const double b = d[j - 1] * om;
        d[j] = a + b;
      }
    }
  };
  // d[j] = d[j] * (1 - tau) + d[j + 1] * tau, ascending (keeps the left part).
  const auto right_pass = [&](double tau) {
    const double om = 1.0 - tau;
    for (std::size_t step = 1; step <= p; ++step) {
      for (std::size_t j = 0; j + step <= p; ++j) {
        const double a = d[j] * om;
        const double b = d[j + 1] * tau;
        d[j] = a + b;
      }
    }
  };
  if (std::abs(upper) >= std::abs(lower - 1.0)) {
    left_pass(upper);
    right_pass(upper != 0.0 ? lower / upper : 0.0);
  } else {
    right_pass(lower);
    left_pass(lower != 1.0 ? (upper - lower) / (1.0 - lower) : 0.0);
  }
  return d;
}

/// Coefficients of the sub-interval `[t_min, t_max]`, clamped to `[0, 1]`; the whole interval is
/// returned as is (widened to `double`).
///
/// Args:
///     coeff (std::span<const S>): Coefficients on `[0, 1]`.
///     t_min (double): Sub-interval start.
///     t_max (double): Sub-interval end.
///
/// Returns:
///     std::vector<double>: The sub-interval's coefficients on `[0, 1]`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat S>
[[nodiscard]] std::vector<double> subdivide(std::span<const S> coeff, double t_min, double t_max) {
  if (t_min <= 0.0 && t_max >= 1.0) {
    return {coeff.begin(), coeff.end()};
  }
  t_min = std::min(std::max(t_min, 0.0), 1.0);
  t_max = std::min(std::max(t_max, 0.0), 1.0);
  return restrict_to<S>(coeff, t_min, t_max);
}

/// Count the sign changes of a coefficient sequence, zeros skipped (Descartes' bound for the
/// Bernstein basis: at most that many roots in `(0, 1)`, with the same parity).
///
/// Args:
///     coeff (std::span<const double>): The coefficients.
///
/// Returns:
///     int: The number of sign changes.
[[nodiscard]] inline int count_sign_changes(std::span<const double> coeff) noexcept {
  int changes = 0;
  int prev = 0;
  for (const double v : coeff) {
    int s = 0;
    if (v > 0.0) {
      s = 1;
    } else if (v < 0.0) {
      s = -1;
    } else {
      continue;
    }
    if (prev != 0 && prev != s) {
      ++changes;
    }
    prev = s;
  }
  return changes;
}

/// The parameter range a convex-hull clip keeps.
///
/// Attributes:
///     t_lo (double): Smallest crossing of `y = 0` by a hull edge (or zero vertex).
///     t_hi (double): Largest such crossing.
///     found (bool): Whether any crossing exists.
struct HullClip {
  double t_lo;
  double t_hi;
  bool found;
};

/// Clip `[0, 1]` to where the convex hull of the control polygon `(i / n, c_i)` meets `y = 0`.
///
/// Andrew's monotone chain builds the upper and lower hulls in `O(n)` (the abscissae are already
/// sorted); every hull edge whose ends have opposite signs, and every zero vertex, contributes its
/// crossing.
///
/// Args:
///     coeff (std::span<const double>): The coefficients, `n + 1` of them.
///
/// Returns:
///     HullClip: The clipped range; `found` false when the hull misses zero.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
[[nodiscard]] inline HullClip clip_hull_to_zero(std::span<const double> coeff) {
  const std::size_t n = coeff.size() - 1;
  if (n < 1) {
    return {0.0, 0.0, false};
  }
  const double inv_n = 1.0 / static_cast<double>(n);
  HullClip clip{1.0, 0.0, false};
  const auto take = [&](double t) {
    clip.t_lo = std::min(clip.t_lo, t);
    clip.t_hi = std::max(clip.t_hi, t);
    clip.found = true;
  };
  const auto cross = [&](std::size_t j0, std::size_t j1, std::size_t i) {
    const double a = static_cast<double>(j1 - j0) * (coeff[i] - coeff[j0]);
    const double b = (coeff[j1] - coeff[j0]) * static_cast<double>(i - j0);
    return a - b;
  };
  std::vector<std::size_t> hull(n + 1);
  for (const bool upper : {true, false}) {
    std::size_t size = 0;
    for (std::size_t i = 0; i <= n; ++i) {
      while (size >= 2) {
        const double c = cross(hull[size - 2], hull[size - 1], i);
        if (upper ? c >= 0.0 : c <= 0.0) {
          --size;
        } else {
          break;
        }
      }
      hull[size++] = i;
    }
    for (std::size_t k = 0; k + 1 < size; ++k) {
      const std::size_t ia = hull[k];
      const std::size_t ib = hull[k + 1];
      const double da = coeff[ia];
      const double db = coeff[ib];
      if (have_opposite_signs(da, db)) {
        const double ta = static_cast<double>(ia) * inv_n;
        const double tb = static_cast<double>(ib) * inv_n;
        const double step = (-da) / (db - da) * (tb - ta);
        take(ta + step);
      }
      if (da == 0.0) {
        take(static_cast<double>(ia) * inv_n);
      }
    }
    const std::size_t last = hull[size - 1];
    if (coeff[last] == 0.0) {
      take(static_cast<double>(last) * inv_n);
    }
  }
  return clip;
}

/// A polished root candidate.
///
/// Attributes:
///     t (double): The (possibly improved) parameter.
///     f (double): The value at `t`.
///     df (double): The derivative at the candidate before polishing.
struct Polished {
  double t;
  double f;
  double df;
};

/// Polish a root candidate with one Newton step, kept only if it stays within `param_tol` of
/// `[lo, hi]` (then clamped to `[0, 1]`) and does not increase the residual.
///
/// Args:
///     coeff (std::span<const S>): Coefficients on `[0, 1]`.
///     mid (double): The candidate.
///     lo (double): Left end of its interval.
///     hi (double): Right end of its interval.
///     param_tol (double): Width of the acceptance margin around `[lo, hi]`.
///
/// Returns:
///     Polished: The candidate, its value and the derivative at the original `mid`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat S>
[[nodiscard]] Polished newton_polish(std::span<const S> coeff, double mid, double lo, double hi,
                                     double param_tol) {
  const auto [f_mid, df_mid] = eval_and_deriv<S>(coeff, mid);
  if (std::abs(df_mid) > dbl_epsilon) {
    double newton = mid - f_mid / df_mid;
    if (lo - param_tol <= newton && newton <= hi + param_tol) {
      newton = std::max(0.0, std::min(1.0, newton));
      const double f_newton = eval<S>(coeff, newton);
      if (std::abs(f_newton) <= std::abs(f_mid)) {
        return {newton, f_newton, df_mid};
      }
    }
  }
  return {mid, f_mid, df_mid};
}

}  // namespace pantr::roots::detail
