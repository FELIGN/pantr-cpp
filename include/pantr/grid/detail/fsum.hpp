/// @file fsum.hpp
/// @brief `fsum`: the exactly rounded sum of doubles, as Python's `math.fsum`.
#pragma once

#include "pantr/config.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace pantr::grid::detail {

/// Sum doubles with a single rounding of the exact sum (Shewchuk's partials algorithm, as CPython's
/// `math.fsum`, including its half-even correction across partials).
///
/// The sum is exactly rounded, so it does not depend on the order of the terms: two ways of
/// reaching the same terms agree bit for bit, which is what the block partition relies on to break
/// ties between candidate factorizations as the Python implementation does.
///
/// Note:
///     The inputs must be finite and the partial sums must not overflow; no check is made
///     (`math.fsum` raises there). The callers sum logarithms and their squared deviations.
///
/// Args:
///     xs (std::span<const double>): The terms.
///
/// Returns:
///     double: The exactly rounded sum; `0.0` for no terms.
[[nodiscard]] inline double fsum(std::span<const double> xs) {
  std::vector<double> partials;
  for (const double term : xs) {
    double x = term;
    std::size_t i = 0;
    for (std::size_t j = 0; j < partials.size(); ++j) {
      double y = partials[j];
      if (std::fabs(x) < std::fabs(y)) {
        const double t = x;
        x = y;
        y = t;
      }
      const double hi = x + y;
      const double lo = y - (hi - x);
      if (lo != 0.0) {
        partials[i] = lo;
        ++i;
      }
      x = hi;
    }
    partials.resize(i);
    // As CPython does, a zero adds nothing and is not kept, so a sum of zeros (of either sign) is
    // `+0.0`, as `math.fsum` returns it.
    if (x != 0.0) {
      partials.push_back(x);
    }
  }
  double hi = 0.0;
  std::size_t n = partials.size();
  if (n > 0) {
    --n;
    hi = partials[n];
    double lo = 0.0;
    while (n > 0) {
      const double x = hi;
      --n;
      const double y = partials[n];
      hi = x + y;
      const double yr = hi - x;
      lo = y - yr;
      if (lo != 0.0) {
        break;
      }
    }
    // Make half-even rounding work across multiple partials.
    if (n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0))) {
      const double y = lo * 2.0;
      const double x = hi + y;
      const double yr = x - hi;
      if (y == yr) {
        hi = x;
      }
    }
  }
  return hi;
}

}  // namespace pantr::grid::detail
