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
/// Every knot class of a knot vector, with the classes holding the first and the last domain knot.
///
/// Attributes:
///     classes (KnotClasses<T>): All the classes (multiplicities summing to the knot count).
///     lo_class (std::size_t): The class holding `knots[degree]`.
///     hi_class (std::size_t): The class holding `knots[size - degree - 1]`.
template <StorageFloat T>
struct KnotClassification {
  KnotClasses<T> classes;
  std::size_t lo_class = 0;
  std::size_t hi_class = 0;
};

/// Group a knot vector into classes (see `unique_knots_and_multiplicity`), all of them, in one
/// pass, recording which classes hold the domain's ends.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] KnotClassification<T> classify_knots(std::span<const T> knots, int degree,
                                                   double tol) {
  const std::size_t n = knots.size();
  const auto lo_index = static_cast<std::size_t>(degree);
  const std::size_t hi_index = n - lo_index - 1;
  KnotClassification<T> out;
  std::vector<T>& reps = out.classes.knots;
  std::vector<std::int64_t>& mult = out.classes.multiplicities;
  reps.reserve(n);
  mult.reserve(n);
  reps.push_back(knots[0]);
  mult.push_back(1);
  for (std::size_t i = 1; i < n; ++i) {
    // The step is formed in T and compared in double, as the Numba kernel did.
    if (static_cast<double>(static_cast<T>(knots[i] - knots[i - 1])) > tol) {
      reps.push_back(knots[i]);
      mult.push_back(0);
    }
    ++mult.back();
    if (i == lo_index) {
      out.lo_class = reps.size() - 1;
    }
    if (i == hi_index) {
      out.hi_class = reps.size() - 1;
    }
  }
  return out;
}

template <StorageFloat T>
[[nodiscard]] KnotClasses<T> unique_knots_and_multiplicity(std::span<const T> knots, int degree,
                                                           double tol, bool in_domain) {
  KnotClassification<T> all = classify_knots<T>(knots, degree, tol);
  if (!in_domain) {
    return std::move(all.classes);
  }
  const auto lo = static_cast<std::ptrdiff_t>(all.lo_class);
  const auto hi = static_cast<std::ptrdiff_t>(all.hi_class) + 1;
  KnotClasses<T> out;
  out.knots.assign(all.classes.knots.begin() + lo, all.classes.knots.begin() + hi);
  out.multiplicities.assign(all.classes.multiplicities.begin() + lo,
                            all.classes.multiplicities.begin() + hi);
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

}  // namespace pantr::bspline
