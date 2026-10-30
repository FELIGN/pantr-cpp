/// @file knot_factories.hpp
/// @brief The uniform open, periodic and cardinal knot vectors, and the Greville abscissae.
///
/// Both reproduce the NumPy computation the previous Python implementation made (`linspace`,
/// `mean`, `mod`) in the knots' own type, so the knot vectors and abscissae are the same values.
/// The reproduction assumes operations are rounded one at a time, which holds under
/// `-ffp-contract=on` or `off`; `-ffp-contract=fast` (or fast-math) may fuse them and move the last
/// bits.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/scalar.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace pantr::bspline {

namespace detail {

/// `n` points spaced as `numpy.linspace(start, stop, n)` spaces them in `T`: `i * step + start`
/// with `step = (stop - start) / (n - 1)`, each operation rounded in `T` (two statements, which
/// `-ffp-contract=on` does not fuse), `(i / (n - 1)) * delta + start` when the step rounds to zero,
/// and the last point set to `stop`. One point is `start`.
template <StorageFloat T>
[[nodiscard]] std::vector<T> linspace(T start, T stop, std::size_t n) {
  std::vector<T> y(n);
  if (n == 0) {
    return y;
  }
  if (n == 1) {
    y[0] = start;
    return y;
  }
  const auto div = static_cast<T>(n - 1);
  const T delta = stop - start;
  const T step = delta / div;
  for (std::size_t i = 0; i < n; ++i) {
    T scaled;
    if (step == static_cast<T>(0)) {
      const T frac = static_cast<T>(i) / div;
      scaled = frac * delta;
    } else {
      scaled = static_cast<T>(i) * step;
    }
    y[i] = scaled + start;
  }
  y[n - 1] = stop;
  return y;
}

/// Refuse knot-factory arguments that describe no knot vector.
///
/// Raises:
///     value_error: `start >= end`, `num_intervals < 1`, `degree < 0`, or the continuity outside
///         `[-1, degree - 1]`, checked in that order with the messages the suite matches.
template <StorageFloat T>
void validate_knot_input(long long num_intervals, long long degree, long long continuity, T start,
                         T end) {
  if (start >= end) {
    throw value_error("domain[0] must be less than domain[1]");
  }
  if (num_intervals < 1) {
    throw value_error("num_intervals must be at least 1");
  }
  if (degree < 0) {
    throw value_error("degree must be non-negative");
  }
  if (continuity < -1 || continuity >= degree) {
    throw value_error("Continuity must be between -1 and " + std::to_string(degree - 1) +
                      " for degree " + std::to_string(degree) + ".");
  }
}

/// The largest knot count a factory forms: a quarter of `long long`, far past memory, so that
/// the sums and products leading to a count cannot overflow.
inline constexpr long long max_knot_count = std::numeric_limits<long long>::max() / 4;

/// Count the knots `2 * per_end + repeated * mult` of a factory's vector, refusing a count past
/// `max_knot_count` before it is formed.
///
/// Args:
///     num_intervals (long long): The interval count, `>= 1`.
///     degree (long long): The degree, `>= 0`.
///     per_end (long long): Knots at each end, at most `degree + 1`.
///     repeated (long long): Knots repeated `mult` times, at most `num_intervals + 1`.
///     mult (long long): Their multiplicity, in `[1, degree + 1]`.
///
/// Returns:
///     std::size_t: The count.
///
/// Raises:
///     capacity_error: `num_intervals`, `degree` or the count exceeds `max_knot_count`.
inline std::size_t knot_count(long long num_intervals, long long degree, long long per_end,
                              long long repeated, long long mult) {
  // num_intervals and degree within the limit bound per_end, repeated and mult by it too, so the
  // quotient below is formed from values that cannot overflow.
  if (num_intervals > max_knot_count || degree > max_knot_count ||
      repeated > (max_knot_count - 2 * per_end) / mult) {
    throw capacity_error("knot vector too large: the number of knots exceeds the supported size");
  }
  return static_cast<std::size_t>(2 * per_end + repeated * mult);
}

}  // namespace detail

/// Create a uniform open knot vector: `start` and `end` repeated `degree + 1` times, the interior
/// knots of `numpy.linspace(start, end, num_intervals + 1)` each repeated `degree - continuity`
/// times.
///
/// Args:
///     num_intervals (long long): Number of intervals, `>= 1`.
///     degree (long long): Degree, `>= 0`.
///     continuity (long long): Continuity at the interior knots, in `[-1, degree - 1]`.
///     start (T): Domain start.
///     end (T): Domain end, `> start`.
///
/// Returns:
///     std::vector<T>: The knot vector.
///
/// Raises:
///     value_error: An argument out of range (see `detail::validate_knot_input`).
///     capacity_error: The knot count exceeds a quarter of `long long` (a smaller count the
///         allocator cannot serve raises `std::length_error` or `std::bad_alloc`).
template <StorageFloat T>
[[nodiscard]] std::vector<T> uniform_open_knots(long long num_intervals, long long degree,
                                                long long continuity, T start, T end) {
  detail::validate_knot_input<T>(num_intervals, degree, continuity, start, end);
  if (num_intervals > detail::max_knot_count || degree > detail::max_knot_count) {
    // Refused before `degree + 1` or `num_intervals + 1` below is formed.
    static_cast<void>(detail::knot_count(num_intervals, degree, 0, 0, 1));
  }
  const long long mult = degree - continuity;
  const std::size_t size =
      detail::knot_count(num_intervals, degree, degree + 1, num_intervals - 1, mult);
  const std::vector<T> unique =
      detail::linspace<T>(start, end, static_cast<std::size_t>(num_intervals) + 1);
  std::vector<T> knots;
  knots.reserve(size);
  knots.insert(knots.end(), static_cast<std::size_t>(degree) + 1, start);
  for (std::size_t i = 1; i + 1 < unique.size(); ++i) {
    knots.insert(knots.end(), static_cast<std::size_t>(mult), unique[i]);
  }
  knots.insert(knots.end(), static_cast<std::size_t>(degree) + 1, end);
  return knots;
}

/// Create a uniform periodic knot vector: the knots of `numpy.linspace(start, end,
/// num_intervals + 1)` each repeated `degree - continuity` times, extended past both ends by
/// `continuity + 1` knots at the interval length `(end - start) / num_intervals`.
///
/// Args:
///     num_intervals (long long): Number of intervals, `>= 1`.
///     degree (long long): Degree, `>= 0`.
///     continuity (long long): Continuity at the interior knots, in `[-1, degree - 1]`.
///     start (T): Domain start.
///     end (T): Domain end, `> start`.
///
/// Returns:
///     std::vector<T>: The knot vector.
///
/// Raises:
///     value_error: An argument out of range (see `detail::validate_knot_input`).
///     capacity_error: The knot count exceeds a quarter of `long long` (a smaller count the
///         allocator cannot serve raises `std::length_error` or `std::bad_alloc`).
template <StorageFloat T>
[[nodiscard]] std::vector<T> uniform_periodic_knots(long long num_intervals, long long degree,
                                                    long long continuity, T start, T end) {
  detail::validate_knot_input<T>(num_intervals, degree, continuity, start, end);
  if (num_intervals > detail::max_knot_count || degree > detail::max_knot_count) {
    // Refused before `degree + 1` or `num_intervals + 1` below is formed.
    static_cast<void>(detail::knot_count(num_intervals, degree, 0, 0, 1));
  }
  const long long mult = degree - continuity;
  const long long ext = degree - mult + 1;  // continuity + 1 knots past each end
  const std::size_t size = detail::knot_count(num_intervals, degree, ext, num_intervals + 1, mult);
  const std::vector<T> unique =
      detail::linspace<T>(start, end, static_cast<std::size_t>(num_intervals) + 1);
  const T span = end - start;
  const T length = span / static_cast<T>(num_intervals);
  const T reach = length * static_cast<T>(ext);
  const T before = start - reach;
  const T after = end + reach;
  const std::vector<T> left = detail::linspace<T>(before, start, static_cast<std::size_t>(ext) + 1);
  const std::vector<T> right = detail::linspace<T>(end, after, static_cast<std::size_t>(ext) + 1);
  std::vector<T> knots;
  knots.reserve(size);
  knots.insert(knots.end(), left.begin(), left.end() - 1);
  for (const T u : unique) {
    knots.insert(knots.end(), static_cast<std::size_t>(mult), u);
  }
  knots.insert(knots.end(), right.begin() + 1, right.end());
  return knots;
}

/// Create the knot vector of the cardinal B-splines: the uniform periodic vector of maximum
/// continuity over `[0, num_intervals]`.
///
/// Raises:
///     value_error: `num_intervals < 1` or `degree < 0` (checked here, so a zero count is not
///         reported as an empty domain).
///     capacity_error: The knot count exceeds a quarter of `long long` (a smaller count the
///         allocator cannot serve raises `std::length_error` or `std::bad_alloc`).
template <StorageFloat T>
[[nodiscard]] std::vector<T> cardinal_knots(long long num_intervals, long long degree) {
  if (num_intervals < 1) {
    throw value_error("num_intervals must be at least 1");
  }
  if (degree < 0) {
    throw value_error("degree must be non-negative");
  }
  return uniform_periodic_knots<T>(num_intervals, degree, degree - 1, static_cast<T>(0),
                                   static_cast<T>(num_intervals));
}

namespace detail {

/// Sum `n` values as NumPy's pairwise summation does (blocks of 8 partial sums up to 128 values,
/// halves past that), in `T`.
template <StorageFloat T>
[[nodiscard]] T pairwise_sum(const T* a, std::size_t n) {
  if (n < 8) {
    T res = static_cast<T>(-0.0);
    for (std::size_t i = 0; i < n; ++i) {
      res += a[i];
    }
    return res;
  }
  if (n <= 128) {
    T r[8];
    for (std::size_t j = 0; j < 8; ++j) {
      r[j] = a[j];
    }
    std::size_t i = 8;
    for (; i < n - (n % 8); i += 8) {
      for (std::size_t j = 0; j < 8; ++j) {
        r[j] += a[i + j];
      }
    }
    T res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) {
      res += a[i];
    }
    return res;
  }
  std::size_t n2 = n / 2;
  n2 -= n2 % 8;
  return pairwise_sum<T>(a, n2) + pairwise_sum<T>(a + n2, n - n2);
}

/// The mean of `n >= 1` values as `numpy.mean` computes it in `T`: their pairwise sum divided by
/// `n`.
template <StorageFloat T>
[[nodiscard]] T numpy_mean(const T* a, std::size_t n) {
  // NumPy's reduction starts from +0, which turns an all-negative-zero sum into +0 and changes no
  // other value (signed zero follows NumPy 2.x; earlier versions are not reproduced).
  const T sum = static_cast<T>(0) + pairwise_sum<T>(a, n);
  return sum / static_cast<T>(n);
}

/// `x mod m` as `numpy.mod` computes it for floats: `fmod`, moved by `m` when its sign differs from
/// `m`'s, and a zero carrying `m`'s sign.
template <StorageFloat T>
[[nodiscard]] T numpy_mod(T x, T m) {
  T mod = std::fmod(x, m);
  if (m == static_cast<T>(0)) {
    return mod;
  }
  if (mod != static_cast<T>(0)) {
    if ((m < static_cast<T>(0)) != (mod < static_cast<T>(0))) {
      mod += m;
    }
  } else {
    mod = std::copysign(static_cast<T>(0), m);
  }
  return mod;
}

}  // namespace detail

/// Compute the Greville abscissae of a 1D space: `g_i` is the mean of `knots[i+1 .. i+degree]`
/// (midpoints of the spans at degree 0), for `i < num_basis`; a periodic space's are wrapped into
/// `[a, b)` and sorted. The means are taken as `numpy.mean` takes them, in `T`.
///
/// Args:
///     knots (std::span<const T>): Knot vector of at least `num_basis + degree` knots.
///     degree (int): Degree, `>= 0`.
///     num_basis (std::size_t): Number of basis functions.
///     periodic (bool): Whether to wrap into the domain.
///     a (T): Domain start (read when periodic).
///     b (T): Domain end (read when periodic).
///
/// Returns:
///     std::vector<T>: One abscissa per basis function.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed); `BsplineSpace1D::greville`
///     is the checked entry point.
template <StorageFloat T>
[[nodiscard]] std::vector<T> greville_abscissae(std::span<const T> knots, int degree,
                                                std::size_t num_basis, bool periodic, T a, T b) {
  std::vector<T> g(num_basis);
  const auto p = static_cast<std::size_t>(degree);
  for (std::size_t i = 0; i < num_basis; ++i) {
    if (degree == 0) {
      const T s = knots[i] + knots[i + 1];
      g[i] = s / static_cast<T>(2);
    } else {
      g[i] = detail::numpy_mean<T>(knots.data() + i + 1, p);
    }
  }
  if (periodic) {
    const T period = b - a;
    for (T& x : g) {
      const T shifted = x - a;
      x = a + detail::numpy_mod<T>(shifted, period);
    }
    // NumPy's order: NaN after everything (an infinite knot can produce one).
    std::sort(g.begin(), g.end(),
              [](T x, T y) { return x < y || (!std::isnan(x) && std::isnan(y)); });
  }
  return g;
}

}  // namespace pantr::bspline
