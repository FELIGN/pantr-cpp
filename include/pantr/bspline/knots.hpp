/// @file knots.hpp
/// @brief Knot vectors: the parametric tolerance, knot classes, domain tests and the factories.
///
/// Every `tol` here is an **absolute** parametric length, in the units of the knot vector.
/// `knot_tolerance` turns the dimensionless strict preset of `pantr/tolerance.hpp` into one by
/// multiplying by the knot vector's own magnitude; `BsplineSpace1D` derives it once at construction
/// and hands it to everything below.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/tolerance.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pantr::bspline {

/// Extra factor on the strict tier for deciding that two knots are the same knot.
///
/// Two knots a caller means to be equal, obtained by different routes (a uniform
/// `a + i * (b - a) / n` against a barycentric `(a * (n - i) + b * i) / n`, a refinement, a
/// split), are each four non-amplifying operations from the exact value, so each is within
/// `4u * scale = 2 eps scale` of it (`u = eps / 2`) and the two differ by at most
/// `4 eps scale`. Twice that, `8 eps scale`, covers what the count does not see: an FMA
/// contraction, another vector width, one further rounding upstream. The strict preset is
/// `4 eps`, so the factor is 2. An admitted margin over a derived bound, not a tight one.
inline constexpr double knot_merge_safety = 2.0;

/// The name NumPy gives a storage type's dtype, as error messages spell it.
template <StorageFloat T>
[[nodiscard]] constexpr const char* dtype_name() noexcept {
  return std::same_as<T, float> ? "float32" : "float64";
}

/// Get the magnitude a parametric comparison on a knot vector is relative to.
///
/// The span alone is not enough: on a knot vector based at `1e6` a knot difference is formed from
/// two coordinates of that size, so it carries an absolute error of `eps * 1e6` however short the
/// span. The coordinate magnitudes are taken alongside it and the largest wins (no floor, which
/// would break scale covariance on a domain smaller than one unit).
///
/// Args:
///     knots (std::span<const T>): Non-decreasing knot vector, non-empty.
///
/// Returns:
///     double: `max(span, |knots[0]|, |knots[-1]|)` in double, taken left to right as Python's
///     `max` takes it; zero only for an identically zero vector.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] double knot_scale(std::span<const T> knots) {
  const auto lo = static_cast<double>(knots.front());
  const auto hi = static_cast<double>(knots.back());
  double result = hi - lo;
  for (const double v : {std::abs(lo), std::abs(hi)}) {
    if (v > result) {
      result = v;
    }
  }
  return result;
}

/// Get the absolute tolerance for comparing parametric coordinates of a knot vector:
/// `knot_merge_safety * get_strict<T>() * knot_scale(knots)`, i.e. `8 eps scale`.
///
/// Args:
///     knots (std::span<const T>): Non-decreasing knot vector, non-empty.
///
/// Returns:
///     double: Absolute parametric tolerance, in the units of `knots`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] double knot_tolerance(std::span<const T> knots) {
  return knot_merge_safety * static_cast<double>(tolerance::get_strict<T>()) * knot_scale<T>(knots);
}

/// Distinct knots (each class's first knot) and their multiplicities.
///
/// Attributes:
///     knots (std::vector<T>): The representatives, increasing.
///     multiplicities (std::vector<std::int64_t>): How many knots each class holds.
template <StorageFloat T>
struct KnotClasses {
  std::vector<T> knots;
  std::vector<std::int64_t> multiplicities;
};

/// Group a knot vector into classes of knots that are the same knot.
///
/// A class ends where the step to the next knot exceeds `tol`, so one pass over the
/// non-decreasing vector suffices; each class is represented by its **first** knot, chosen rather
/// than averaged, so the representatives are knots the vector contains and the grouping is
/// idempotent. Grouping by gap rather than by a rounding grid makes the answer independent of an
/// origin; its cost is chaining (knots each within `tol` of the next collapse into one class),
/// accepted because a class-width cap would break idempotence.
///
/// Args:
///     knots (std::span<const T>): Non-decreasing knot vector, non-empty.
///     degree (int): Degree, which fixes the domain `[knots[degree], knots[size - degree - 1]]`.
///     tol (double): Absolute parametric tolerance.
///     in_domain (bool): Report only the classes from the one holding the first domain knot to the
///         one holding the last (whole classes, clamped copies outside the domain included).
///
/// Returns:
///     KnotClasses<T>: With `in_domain == false` the multiplicities sum to `knots.size()`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] KnotClasses<T> unique_knots_and_multiplicity(std::span<const T> knots, int degree,
                                                           double tol, bool in_domain) {
  const std::size_t n = knots.size();
  const auto lo_index = static_cast<std::size_t>(degree);
  const std::size_t hi_index = n - lo_index - 1;
  std::vector<std::size_t> first_ids{0};
  std::vector<std::int64_t> mult{1};
  std::size_t lo_class = 0;
  std::size_t hi_class = 0;
  for (std::size_t i = 1; i < n; ++i) {
    // The step is formed in T and compared in double, as the Numba kernel did.
    if (static_cast<double>(static_cast<T>(knots[i] - knots[i - 1])) > tol) {
      first_ids.push_back(i);
      mult.push_back(0);
    }
    ++mult.back();
    if (i == lo_index) {
      lo_class = first_ids.size() - 1;
    }
    if (i == hi_index) {
      hi_class = first_ids.size() - 1;
    }
  }
  const std::size_t lo = in_domain ? lo_class : 0;
  const std::size_t hi = in_domain ? hi_class : first_ids.size() - 1;
  KnotClasses<T> out;
  out.knots.reserve(hi - lo + 1);
  for (std::size_t c = lo; c <= hi; ++c) {
    out.knots.push_back(knots[first_ids[c]]);
  }
  out.multiplicities.assign(mult.begin() + static_cast<std::ptrdiff_t>(lo),
                            mult.begin() + static_cast<std::ptrdiff_t>(hi) + 1);
  return out;
}

/// Count the knots among `knots[0..degree]` within `tol` of the first domain knot `knots[degree]`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] int multiplicity_of_first_knot_in_domain(std::span<const T> knots, int degree,
                                                       double tol) {
  const T first = knots[static_cast<std::size_t>(degree)];
  int count = 0;
  for (int i = 0; i <= degree; ++i) {
    if (static_cast<double>(std::abs(static_cast<T>(knots[static_cast<std::size_t>(i)] - first))) <=
        tol) {
      ++count;
    }
  }
  return count;
}

/// Number of basis functions: `size - degree - 1`, less `regularity + 1` for a periodic space,
/// where the regularity at the domain start is `degree` minus the first domain knot's multiplicity.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] long long num_basis(std::span<const T> knots, int degree, bool periodic, double tol) {
  long long n = static_cast<long long>(knots.size()) - degree - 1;
  if (periodic) {
    const int regularity = degree - multiplicity_of_first_knot_in_domain<T>(knots, degree, tol);
    n -= regularity + 1;
  }
  return n;
}

/// Whether `pt` lies in `[knots[degree], knots[size - degree - 1]]` up to `tol` at each end.
///
/// The difference is formed first and only then compared with `tol` (shifting the bound instead,
/// `pt <= end + tol`, rounds the bound and widens the band to an ulp of `end` once `tol` is
/// smaller); near a boundary the subtraction is exact (Sterbenz). A NaN point is outside.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat K, StorageFloat P>
[[nodiscard]] bool is_in_domain(std::span<const K> knots, int degree, P pt, double tol) {
  using W = std::common_type_t<K, P>;
  const auto begin = static_cast<W>(knots[static_cast<std::size_t>(degree)]);
  const auto end = static_cast<W>(knots[knots.size() - static_cast<std::size_t>(degree) - 1]);
  const auto p = static_cast<W>(pt);
  const bool above = (begin < p) || (static_cast<double>(std::abs(begin - p)) <= tol);
  const bool below = (p < end) || (static_cast<double>(std::abs(p - end)) <= tol);
  return above && below;
}

/// Flag the cardinal intervals: neither bounding knot repeated, and the `2 * degree - 1` knot
/// spans of the window `u[k-degree+1] .. u[k+degree]` all of the interval's own length within
/// `tol`. At degree 0 the length condition is vacuous and only the multiplicity gate applies.
///
/// Args:
///     knots (std::span<const T>): Knot vector.
///     degree (int): Degree.
///     tol (double): Absolute parametric tolerance.
///     out (std::span<bool>): One flag per in-domain interval.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed), `out`'s size included.
template <StorageFloat T>
void cardinal_intervals(std::span<const T> knots, int degree, double tol, std::span<bool> out) {
  const KnotClasses<T> cls = unique_knots_and_multiplicity<T>(knots, degree, tol, true);
  const std::vector<std::int64_t>& mult = cls.multiplicities;
  std::fill(out.begin(), out.end(), false);
  if (std::all_of(mult.begin(), mult.end(), [](std::int64_t m) { return m > 1; })) {
    return;
  }
  const std::size_t num_intervals = mult.size() - 1;
  auto knot_id = static_cast<std::size_t>(degree);
  for (std::size_t e = 0; e < num_intervals; ++e) {
    if (mult[e] == 1 && mult[e + 1] == 1) {
      if (degree == 0) {
        out[e] = true;
      } else {
        const std::size_t first = knot_id - static_cast<std::size_t>(degree) + 1;
        const T ref = static_cast<T>(knots[knot_id + 1] - knots[knot_id]);
        bool equal = true;
        for (std::size_t i = first; i < knot_id + static_cast<std::size_t>(degree); ++i) {
          const T len = static_cast<T>(knots[i + 1] - knots[i]);
          if (!(static_cast<double>(std::abs(static_cast<T>(len - ref))) <= tol)) {
            equal = false;
            break;
          }
        }
        out[e] = equal;
      }
    }
    knot_id += static_cast<std::size_t>(mult[e + 1]);
  }
}

/// Refuse a knot vector that snapping collapsed onto a single point.
///
/// Only that cause: the knots were distinct and none was distinguishable at its own magnitude. A
/// vector supplied flat passes and is refused next by `check_space_has_an_interval`, which owns the
/// consequence both share. Reachable without any defect: a mesh of `n` intervals over a span `s`
/// at offset `x` survives only while `s / n > 8 eps max(s, |x|)`.
///
/// Args:
///     raw (std::span<const T>): The knot vector as supplied, non-decreasing.
///     snapped (std::span<const T>): The same vector after snapping.
///     tol (double): The tolerance snapping used.
///
/// Raises:
///     value_error: `raw` held more than one distinct knot and `snapped` holds one.
template <StorageFloat T>
void check_snapping_kept_an_interval(std::span<const T> raw, std::span<const T> snapped,
                                     double tol) {
  if (raw.front() == raw.back() || snapped.front() != snapped.back()) {
    return;
  }
  const double scale = knot_scale<T>(raw);
  const auto scale_t = static_cast<T>(scale);
  const double ulp =
      static_cast<double>(std::nextafter(scale_t, std::numeric_limits<T>::infinity()) - scale_t);
  double closest = std::numeric_limits<double>::infinity();
  for (std::size_t i = 1; i < raw.size(); ++i) {
    const double gap = static_cast<double>(raw[i]) - static_cast<double>(raw[i - 1]);
    if (gap > 0.0 && gap < closest) {
      closest = gap;
    }
  }
  const std::string name = dtype_name<T>();
  const std::string remedy =
      std::same_as<T, float>
          ? "Use float64, move the domain nearer the origin, or coarsen the mesh."
          : "Move the domain nearer the origin, or coarsen the mesh.";
  throw value_error(
      "knot snapping collapsed every knot onto " + py_repr(static_cast<double>(snapped.front())) +
      ": in " + name + " at |coordinate| ~ " + py_format(scale, 3, false) +
      " two knots are the same knot unless they differ by more than " + py_format(tol, 3, false) +
      " (" + py_format(tol / ulp, 0, true) + " ulp there), and the closest pair in [" +
      py_repr(static_cast<double>(raw.front())) + ", " + py_repr(static_cast<double>(raw.back())) +
      "] is " + py_format(closest, 3, false) + " apart. This mesh is finer than " + name +
      " resolves at that magnitude. " + remedy);
}

/// Refuse a knot vector whose domain is a single point (no interval, so no cell).
///
/// The predicate is the interval count, decided by the same knot grouping the space uses, which
/// also catches an interior knot of multiplicity high enough to swallow the domain. The grouping
/// chains, so the message reports what holds of every case it fires on: each consecutive step
/// across the domain is at most `tol`.
///
/// Raises:
///     value_error: `num_intervals == 0`.
template <StorageFloat T>
void check_space_has_an_interval(std::span<const T> knots, int degree, long long num_intervals,
                                 double tol) {
  if (num_intervals > 0) {
    return;
  }
  const std::size_t last = knots.size() - static_cast<std::size_t>(degree) - 1;
  const std::string d = std::to_string(degree);
  const std::string t = py_format(tol, 3, false);
  throw value_error(
      "knot vector spans no interval: at degree " + d + " the domain runs from knots[" + d +
      "] = " + py_repr(static_cast<double>(knots[static_cast<std::size_t>(degree)])) +
      " to knots[" + std::to_string(last) + "] = " + py_repr(static_cast<double>(knots[last])) +
      ", and every step between them is at most this vector's tolerance of " + t +
      ", so its in-domain knots are all one knot, the space has no cell, and nothing can be "
      "evaluated, tabulated or located on it. The domain needs two consecutive knots more than " +
      t + " apart.");
}

namespace detail {

/// `n` points spaced as `numpy.linspace(start, stop, n)` spaces them in `T`: `i * step + start`
/// with `step = (stop - start) / (n - 1)`, each operation rounded in `T` (two statements, so no
/// contraction fuses them), `(i / (n - 1)) * delta + start` when the step rounds to zero, and the
/// last point set to `stop`. One point is `start`.
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

/// Count the knots `2 * per_end + repeated * mult` of a factory's vector, refusing a count past a
/// quarter of `long long` (far past memory) before any of it is formed.
///
/// Raises:
///     capacity_error: The count, or `degree` or `num_intervals` on the way to it, is that large.
inline std::size_t knot_count(long long num_intervals, long long degree, long long per_end,
                              long long repeated, long long mult) {
  constexpr long long limit = std::numeric_limits<long long>::max() / 4;
  // Checked first: every term below is then a small multiple of numbers within `limit`.
  if (num_intervals > limit || degree > limit ||
      (mult != 0 && repeated > (limit - 2 * per_end) / mult)) {
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
///     capacity_error: The vector would not fit in memory.
template <StorageFloat T>
[[nodiscard]] std::vector<T> uniform_open_knots(long long num_intervals, long long degree,
                                                long long continuity, T start, T end) {
  detail::validate_knot_input<T>(num_intervals, degree, continuity, start, end);
  if (num_intervals > std::numeric_limits<long long>::max() / 4 ||
      degree > std::numeric_limits<long long>::max() / 4) {
    static_cast<void>(detail::knot_count(num_intervals, degree, 0, 0, 0));  // raises
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
///     capacity_error: The vector would not fit in memory.
template <StorageFloat T>
[[nodiscard]] std::vector<T> uniform_periodic_knots(long long num_intervals, long long degree,
                                                    long long continuity, T start, T end) {
  detail::validate_knot_input<T>(num_intervals, degree, continuity, start, end);
  if (num_intervals > std::numeric_limits<long long>::max() / 4 ||
      degree > std::numeric_limits<long long>::max() / 4) {
    static_cast<void>(detail::knot_count(num_intervals, degree, 0, 0, 0));  // raises
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
///     capacity_error: The vector would not fit in memory.
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
  const T sum = pairwise_sum<T>(a, n);
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
