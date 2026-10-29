/// @file rules_1d.hpp
/// @brief One-dimensional quadrature rules and node sets on `[0, 1]`.
///
/// Every rule is computed in double precision and stored in `T` (`float` or `double`): a
/// Newton iteration run in `float` loses weight accuracy (design/lessons_first_port.md), and the
/// rounding to `T` then happens once, at the end. The one exception is
/// `modified_chebyshev_nodes_1d`, a closed form evaluated in `T`, as the Python implementation it
/// replaces did. Rules on `[-1, 1]` are mapped to `[0, 1]` as `(x + 1) / 2` and `w / 2`.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/lambert_w.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/tolerance.hpp"

#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

namespace pantr::quad {

/// Nodes and weights of a one-dimensional rule.
///
/// Attributes:
///     nodes (NdArray<T>): The nodes, shape `(n,)`, ascending.
///     weights (NdArray<T>): The weights, shape `(n,)`.
template <StorageFloat T>
struct Rule1D {
  NdArray<T> nodes;
  NdArray<T> weights;
};

namespace detail {

inline void check_n_pts(long long n_pts, long long min_pts) {
  if (n_pts < min_pts) {
    throw value_error("n_pts must be at least " + std::to_string(min_pts));
  }
}

/// Map a rule on `[-1, 1]` (in double) to `[0, 1]` and round it to `T`.
template <StorageFloat T>
Rule1D<T> to_unit_interval(const std::vector<double>& x, const std::vector<double>& w) {
  Rule1D<T> rule{NdArray<T>({x.size()}), NdArray<T>({w.size()})};
  for (std::size_t i = 0; i < x.size(); ++i) {
    rule.nodes.data()[i] = static_cast<T>((x[i] + 1.0) * 0.5);
    rule.weights.data()[i] = static_cast<T>(w[i] * 0.5);
  }
  return rule;
}

/// Legendre `P_n(x)` and `P_{n-1}(x)` by the three-term recurrence (`n >= 1`).
inline std::pair<double, double> legendre_pair(std::size_t n, double x) {
  double p_prev = 1.0;  // P_0
  double p = x;         // P_1
  for (std::size_t k = 2; k <= n; ++k) {
    const auto kd = static_cast<double>(k);
    const double p_next = ((2.0 * kd - 1.0) * x * p - (kd - 1.0) * p_prev) / kd;
    p_prev = p;
    p = p_next;
  }
  return {p, p_prev};
}

/// Iterations a Newton root polish may take. From the asymptotic starting points used here
/// Newton converges quadratically in a few steps; the cap is a loud bound on a floating-point
/// loop, an admitted heuristic rather than a derived count.
inline constexpr int newton_max_iterations = 100;

/// Newton's stopping test on `[-1, 1]`: a step below `4 eps (1 + |x|)`, both terms present (the
/// nodes have unit scale, and the middle node of an odd rule is 0, where a relative test alone
/// never terminates). Four roundings of the update is the admitted margin.
inline bool newton_converged(double dx, double x) {
  return std::fabs(dx) <= 4.0 * std::numeric_limits<double>::epsilon() * (1.0 + std::fabs(x));
}

[[noreturn]] inline void newton_failed(const char* rule, std::size_t n, int max_iterations) {
  throw state_error(std::string(rule) + ": Newton's iteration for n_pts = " + std::to_string(n) +
                    " did not converge in " + std::to_string(max_iterations) + " steps");
}

}  // namespace detail

/// Get the `n_pts`-point trapezoidal rule on `[0, 1]`.
///
/// Nodes `k / (n_pts - 1)`, weights `h` inside and `h / 2` at the ends, `h = 1 / (n_pts - 1)`;
/// one point gives node `0.5` and weight `1`.
///
/// Args:
///     n_pts (long long): Number of points, at least 1.
///
/// Returns:
///     Rule1D<T>: The rule.
///
/// Raises:
///     value_error: `n_pts < 1`.
template <StorageFloat T>
[[nodiscard]] Rule1D<T> trapezoidal_1d(long long n_pts) {
  detail::check_n_pts(n_pts, 1);
  const auto n = static_cast<std::size_t>(n_pts);
  Rule1D<T> rule{NdArray<T>({n}), NdArray<T>({n})};
  if (n == 1) {
    rule.nodes.data()[0] = static_cast<T>(0.5);
    rule.weights.data()[0] = static_cast<T>(1.0);
    return rule;
  }
  // As numpy.linspace: k * step in double, the last node set to the endpoint exactly.
  const double h = 1.0 / static_cast<double>(n - 1);
  for (std::size_t k = 0; k < n; ++k) {
    rule.nodes.data()[k] = static_cast<T>(static_cast<double>(k) * h);
    rule.weights.data()[k] = static_cast<T>(h);
  }
  rule.nodes.data()[n - 1] = static_cast<T>(1.0);
  rule.weights.data()[0] = rule.weights.data()[n - 1] = static_cast<T>(0.5 * h);
  return rule;
}

namespace detail {

/// `gauss_legendre_1d` with the Newton iteration cap as an argument, so a test can exhaust it.
template <StorageFloat T>
[[nodiscard]] Rule1D<T> gauss_legendre_1d_capped(long long n_pts, int max_iterations) {
  check_n_pts(n_pts, 1);
  const auto n = static_cast<std::size_t>(n_pts);
  std::vector<double> x(n), w(n);
  const double nd = static_cast<double>(n);
  for (std::size_t i = 0; i < (n + 1) / 2; ++i) {
    // Root i counted from +1 downward; its mirror is -root.
    const bool middle = (n % 2 == 1) && (i == n / 2);
    double r =
        middle ? 0.0 : std::cos(std::numbers::pi * (static_cast<double>(i) + 0.75) / (nd + 0.5));
    double dp = 0.0;
    bool converged = middle;
    for (int it = 0; it < max_iterations && !converged; ++it) {
      const auto [p, p_prev] = legendre_pair(n, r);
      dp = nd * (r * p - p_prev) / (r * r - 1.0);
      const double dx = p / dp;
      r -= dx;
      converged = newton_converged(dx, r);
    }
    if (!converged) {
      newton_failed("gauss_legendre_1d", n, max_iterations);
    }
    // Derivative at the converged root.
    const auto [p, p_prev] = legendre_pair(n, r);
    static_cast<void>(p);
    dp = nd * (r * p - p_prev) / (r * r - 1.0);
    const double weight = 2.0 / ((1.0 - r * r) * dp * dp);
    x[n - 1 - i] = r;
    x[i] = -r;
    w[n - 1 - i] = weight;
    w[i] = weight;
  }
  double sum = 0.0;
  for (const double wi : w) {
    sum += wi;
  }
  for (double& wi : w) {
    wi *= 2.0 / sum;
  }
  return to_unit_interval<T>(x, w);
}

}  // namespace detail

/// Get the `n_pts`-point Gauss-Legendre rule on `[0, 1]`.
///
/// The nodes are the roots of `P_n`, found by Newton's iteration on the three-term recurrence
/// from the asymptotic starting points `cos(pi (i + 3/4) / (n + 1/2))`, one per root; the weights
/// are `2 / ((1 - x^2) P_n'(x)^2)`. The rule is built symmetric (each pair of nodes exact
/// negatives on `[-1, 1]`, the middle node of an odd rule exactly 0) and its weights are rescaled
/// onto the measure 2 of `[-1, 1]`, as numpy.polynomial.legendre.leggauss does. Exact for
/// polynomials of degree up to `2 n_pts - 1`.
///
/// Args:
///     n_pts (long long): Number of points, at least 1.
///
/// Returns:
///     Rule1D<T>: The rule, nodes ascending.
///
/// Raises:
///     value_error: `n_pts < 1`.
///     state_error: Newton's iteration did not converge (not observed; a loud bound).
template <StorageFloat T>
[[nodiscard]] Rule1D<T> gauss_legendre_1d(long long n_pts) {
  return detail::gauss_legendre_1d_capped<T>(n_pts, detail::newton_max_iterations);
}

namespace detail {

/// `gauss_lobatto_legendre_1d` with the Newton iteration cap as an argument, so a test can exhaust
/// it.
template <StorageFloat T>
[[nodiscard]] Rule1D<T> gauss_lobatto_legendre_1d_capped(long long n_pts, int max_iterations) {
  check_n_pts(n_pts, 2);
  const auto n = static_cast<std::size_t>(n_pts);
  const std::size_t big_n = n - 1;
  const double nd = static_cast<double>(big_n);
  const double nn1 = nd * (nd + 1.0);
  std::vector<double> x(n), w(n);
  x[0] = -1.0;
  x[n - 1] = 1.0;
  for (std::size_t k = 1; k <= (n - 1) / 2; ++k) {
    const bool middle = (big_n % 2 == 0) && (2 * k == big_n);
    double r = middle ? 0.0 : std::cos(std::numbers::pi * static_cast<double>(k) / nd);
    bool converged = middle;
    for (int it = 0; it < max_iterations && !converged; ++it) {
      const auto [p, p_prev] = legendre_pair(big_n, r);
      const double one_minus_r2 = 1.0 - r * r;
      const double d1 = nd * (p_prev - r * p) / one_minus_r2;     // P_N'
      const double d2 = (2.0 * r * d1 - nn1 * p) / one_minus_r2;  // P_N''
      const double dx = d1 / d2;
      r -= dx;
      converged = newton_converged(dx, r);
    }
    if (!converged) {
      newton_failed("gauss_lobatto_legendre_1d", n, max_iterations);
    }
    x[n - 1 - k] = r;
    x[k] = -r;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const double p = legendre_pair(big_n, x[i]).first;
    w[i] = 2.0 / nn1 / (p * p);
  }
  // Symmetric by construction of the nodes; the weights of a pair are computed at exact
  // negatives, so P_N^2 agrees and they are equal.
  return to_unit_interval<T>(x, w);
}

}  // namespace detail

/// Get the `n_pts`-point Gauss-Lobatto-Legendre rule on `[0, 1]`.
///
/// With `N = n_pts - 1`, the nodes are `-1`, the roots of `P_N'`, and `1`; the interior roots
/// are found by Newton's iteration on `P_N'` (its derivative from the Legendre equation,
/// `P_N'' = (2 x P_N' - N (N + 1) P_N) / (1 - x^2)`) from the Chebyshev-Gauss-Lobatto starting
/// points `cos(pi k / N)`. Weights `2 / (N (N + 1) P_N(x)^2)`. Built symmetric. Exact for
/// polynomials of degree up to `2 n_pts - 3`.
///
/// Args:
///     n_pts (long long): Number of points, at least 2.
///
/// Returns:
///     Rule1D<T>: The rule, nodes ascending, starting at 0 and ending at 1.
///
/// Raises:
///     value_error: `n_pts < 2`.
///     state_error: Newton's iteration did not converge (not observed; a loud bound).
template <StorageFloat T>
[[nodiscard]] Rule1D<T> gauss_lobatto_legendre_1d(long long n_pts) {
  return detail::gauss_lobatto_legendre_1d_capped<T>(n_pts, detail::newton_max_iterations);
}

/// Get the `n_pts`-point Chebyshev-Gauss rule of the first kind on `[0, 1]`.
///
/// Nodes `sin(pi (2 k - n + 1) / (2 n))` (numpy.polynomial.chebyshev.chebpts1's formula, which is
/// ascending and exactly antisymmetric), weights `pi / n`, both mapped to `[0, 1]`.
///
/// Args:
///     n_pts (long long): Number of points, at least 1.
///
/// Returns:
///     Rule1D<T>: The rule, nodes ascending.
///
/// Raises:
///     value_error: `n_pts < 1`.
template <StorageFloat T>
[[nodiscard]] Rule1D<T> chebyshev_gauss_1st_kind_1d(long long n_pts) {
  detail::check_n_pts(n_pts, 1);
  const auto n = static_cast<std::size_t>(n_pts);
  std::vector<double> x(n), w(n, std::numbers::pi / static_cast<double>(n));
  const double scale = 0.5 * std::numbers::pi / static_cast<double>(n);
  for (std::size_t k = 0; k < n; ++k) {
    const double m = 2.0 * static_cast<double>(k) - static_cast<double>(n) + 1.0;
    x[k] = std::sin(scale * m);
  }
  return detail::to_unit_interval<T>(x, w);
}

/// Get the `n_pts`-point Chebyshev-Gauss rule of the second kind on `[0, 1]`.
///
/// Integrates against the second-kind weight mapped to `[0, 1]`, exactly for degree up to
/// `2 n_pts - 1`. Nodes `-cos(theta_k)`, weights `pi / (n + 1) sin^2(theta_k)`, with
/// `theta_k = k pi / (n + 1)`, `k = 1..n`: interior, ascending.
///
/// Args:
///     n_pts (long long): Number of points, at least 2.
///
/// Returns:
///     Rule1D<T>: The rule.
///
/// Raises:
///     value_error: `n_pts < 2`.
template <StorageFloat T>
[[nodiscard]] Rule1D<T> chebyshev_gauss_2nd_kind_1d(long long n_pts) {
  detail::check_n_pts(n_pts, 2);
  const auto n = static_cast<std::size_t>(n_pts);
  std::vector<double> x(n), w(n);
  const double np1 = static_cast<double>(n + 1);
  for (std::size_t k = 0; k < n; ++k) {
    const double angle = static_cast<double>(k + 1) * std::numbers::pi / np1;
    const double s = std::sin(angle);
    x[k] = -std::cos(angle);
    w[k] = std::numbers::pi / np1 * (s * s);
  }
  return detail::to_unit_interval<T>(x, w);
}

/// Get the `n_pts` Chebyshev-Lobatto (second-kind extrema) nodes on `[0, 1]`.
///
/// `0.5 - 0.5 cos(pi i / (n - 1))`, evaluated in `T`, from 0 to 1. Interpolation nodes, so no
/// weights.
///
/// Args:
///     n_pts (long long): Number of nodes, at least 2.
///
/// Returns:
///     NdArray<T>: The nodes, shape `(n_pts,)`.
///
/// Raises:
///     value_error: `n_pts < 2`.
template <StorageFloat T>
[[nodiscard]] NdArray<T> modified_chebyshev_nodes_1d(long long n_pts) {
  detail::check_n_pts(n_pts, 2);
  const auto n = static_cast<std::size_t>(n_pts);
  NdArray<T> nodes({n});
  const T pi = std::numbers::pi_v<T>;
  const T half = static_cast<T>(0.5);
  const auto denom = static_cast<T>(n - 1);
  for (std::size_t i = 0; i < n; ++i) {
    using std::cos;
    nodes.data()[i] = half - half * cos(pi * static_cast<T>(i) / denom);
  }
  return nodes;
}

/// Decay-rate factor selecting the uniform step `h` of the tanh-sinh rule in transform space.
/// Fixing the truncation so the smallest retained node sits a constant number of decay e-folds
/// from the endpoint turns the equation for `h` into `u e^u = 2 c (pi / 2) (n - 1)`, solved by
/// the Lambert W function: `h = 2 W(...) / n`. The value is the one the Python implementation
/// fixed, which keeps the rule interchangeable across releases.
inline constexpr double tanh_sinh_decay_factor = 0.6;

/// Get the tanh-sinh (double-exponential) rule on `[0, 1]` with at most `n_pts` points.
///
/// Nodes `x = tanh((pi / 2) sinh t)` on `[-1, 1]` at a uniform step `h` (see
/// `tanh_sinh_decay_factor`), weights `(pi / 2) cosh t / cosh^2((pi / 2) sinh t)`, symmetric
/// about 0. Generation stops at the first node whose endpoint gap `1 - |x|` is below `eps(T)`, so
/// no node lies on an endpoint, and fewer than `n_pts` nodes may be returned (fewer in `float` than
/// in `double`). The weights are rescaled onto the measure 2 of `[-1, 1]` before the mapping, up
/// to the rounding of their own sum.
///
/// Why the gap is `eps(T)`: the node near `+1` reaches `[0, 1]` as `((1 - gap) + 1) / 2` rounded
/// to `T`, and that whole expression must stay below 1. With `u = eps / 2` the spacing below 1,
/// `1 - gap` rounds to `1 - u` while `gap < 1.5 u` (to `1 - 2u` from there on); `+ 1` maps `1 - u`
/// to the midpoint `2 - u` of `[2 - 2u, 2]`, which ties to even and becomes 2, and maps `1 - 2u`
/// to `2 - 2u` exactly; the halving is exact. So the node collapses onto 1 exactly when
/// `gap < 0.75 eps`, and `eps` clears that by one representable step of the gap. (Reasoning on
/// `2 - gap` alone gives `eps / 2`, a factor 1.5 too low, because it ignores the rounding of
/// `1 - gap`.) The argument is about the binade boundary at 1, so it holds in `float` and `double`
/// alike. Truncating there costs nothing: with `gap = 1 - tanh(omega)`,
/// `cosh(omega)^-2 = gap (2 - gap)`, so a discarded weight is at most `pi cosh(t) gap`, which at
/// the threshold is below the rounding of the weight sum it would have joined. The bound does not
/// claim the returned nodes are distinct: two consecutive samples can clear the threshold and
/// still round onto one representable coordinate next to 1, the format running out of coordinates
/// rather than a node reaching the endpoint.
///
/// Args:
///     n_pts (long long): Requested number of points, at least 1.
///
/// Returns:
///     Rule1D<T>: The rule, nodes in `(0, 1)` in generation order: the centre node (odd
///     `n_pts`), then each symmetric pair, lower member first, from the centre outward.
///
/// Raises:
///     value_error: `n_pts < 1`.
///     state_error: The Lambert W iteration did not converge (not observed; a loud bound).
template <StorageFloat T>
[[nodiscard]] Rule1D<T> tanh_sinh_1d(long long n_pts) {
  detail::check_n_pts(n_pts, 1);
  const auto n = static_cast<std::size_t>(n_pts);
  std::vector<double> x;
  std::vector<double> w;
  if (n == 1) {
    x.push_back(0.0);
    w.push_back(2.0);
    return detail::to_unit_interval<T>(x, w);
  }
  constexpr double half_pi = 0.5 * std::numbers::pi;
  const double decay_arg = 2.0 * tanh_sinh_decay_factor * half_pi * static_cast<double>(n - 1);
  const double h = 2.0 * lambert_w0(decay_arg) / static_cast<double>(n);
  const double min_gap = static_cast<double>(tolerance::machine_epsilon<T>());
  x.reserve(n);
  w.reserve(n);
  const bool odd = n % 2 == 1;
  if (odd) {
    x.push_back(0.0);
    w.push_back(half_pi);
  }
  for (std::size_t i = 0; i < n / 2; ++i) {
    const double t = odd ? static_cast<double>(i + 1) * h : (static_cast<double>(i) + 0.5) * h;
    const double omega = half_pi * std::sinh(t);
    const double c = std::cosh(omega);
    const double weight = half_pi * std::cosh(t) / (c * c);
    // gap = 1 - tanh(omega) in the form 2 / (1 + e^{2 omega}), which stays a nonzero float up to
    // the resolution limit where 1 - tanh(omega) would saturate to 0 a step early.
    const double gap = 2.0 / (1.0 + std::exp(2.0 * omega));
    if (gap < min_gap) {
      break;  // the gap decreases monotonically in t: every later node would collapse too
    }
    x.push_back(-(1.0 - gap));
    w.push_back(weight);
    x.push_back(1.0 - gap);
    w.push_back(weight);
  }
  double sum = 0.0;
  for (const double wi : w) {
    sum += wi;
  }
  for (double& wi : w) {
    wi *= 2.0 / sum;
  }
  return detail::to_unit_interval<T>(x, w);
}

}  // namespace pantr::quad
