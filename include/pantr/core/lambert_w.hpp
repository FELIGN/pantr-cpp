/// @file lambert_w.hpp
/// @brief The principal branch of the Lambert W function on `[0, inf)`, by Halley's iteration.
///
/// `W(x)` is the `w >= 0` solving `w * exp(w) = x`. pantr needs it on `x >= 0` only (the step of
/// the tanh-sinh rule), where the principal branch is the unique real solution and
/// `f(w) = w exp(w) - x` is increasing and convex, so Halley's iteration from a starting point
/// within a factor of the root converges cubically.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"

#include <cmath>
#include <limits>
#include <numbers>
#include <string>

namespace pantr {

/// Iterations `lambert_w0` may take. Cubic convergence from the starting point below needs at
/// most a handful (the starting point is within a factor of two of the root for every finite
/// `x >= 0`); 64 is a loud bound on a loop whose termination in floating point is not otherwise
/// guaranteed, an admitted heuristic rather than a derived count.
inline constexpr int lambert_w_max_iterations = 64;

/// Evaluate the principal branch `W0(x)` for `x >= 0`.
///
/// Halley's iteration on `f(w) = w e^w - x`. It stops once a step satisfies
/// `|dw| <= 4 eps (1 + |w|)`: both an absolute term (`W(0) = 0`, where a relative test alone
/// never terminates) and a relative one, each four roundings of the update, the step itself being
/// computed from three roundings of `f` and a quotient.
///
/// Args:
///     x (double): The argument; finite and `>= 0`.
///
/// Returns:
///     double: `w >= 0` with `w e^w = x`.
///
/// Raises:
///     value_error: `x` is negative, NaN or infinite.
///     state_error: The iteration did not converge within `lambert_w_max_iterations` steps.
[[nodiscard]] inline double lambert_w0(double x) {
  if (!(x >= 0.0) || !std::isfinite(x)) {
    throw value_error("lambert_w0 needs a finite argument >= 0, got " + py_repr(x));
  }
  if (x == 0.0) {
    return 0.0;
  }
  // Starting point: log1p(x) bounds W(x) from above on [0, inf) and is close for small x; past e,
  // the asymptotic log(x) - log(log(x)) is within a few percent.
  double w = x < std::numbers::e ? std::log1p(x) : std::log(x) - std::log(std::log(x));
  constexpr double tol = 4.0 * std::numeric_limits<double>::epsilon();
  for (int it = 0; it < lambert_w_max_iterations; ++it) {
    const double ew = std::exp(w);
    const double f = w * ew - x;
    const double wp1 = w + 1.0;
    const double dw = f / (ew * wp1 - (w + 2.0) * f / (2.0 * wp1));
    w -= dw;
    if (std::fabs(dw) <= tol * (1.0 + std::fabs(w))) {
      return w;
    }
  }
  throw state_error("lambert_w0(" + py_repr(x) + ") did not converge in " +
                    std::to_string(lambert_w_max_iterations) + " Halley steps");
}

}  // namespace pantr
