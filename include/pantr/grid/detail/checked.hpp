/// @file checked.hpp
/// @brief Overflow-checked `int64` arithmetic, portable (no compiler builtins).
#pragma once

#include "pantr/config.hpp"

#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace pantr::grid::detail {

/// Multiply two `int64` values, refusing a product that does not fit.
///
/// Python's integers are unbounded, so where the Python implementation formed a product of cell
/// counts or block coordinates and then stored it in an `int64` array, the C++ one checks.
///
/// Args:
///     a (std::int64_t): First factor.
///     b (std::int64_t): Second factor.
///
/// Returns:
///     std::int64_t: `a * b`.
///
/// Raises:
///     std::overflow_error: The product is outside the `int64` range.
[[nodiscard]] inline std::int64_t checked_mul(std::int64_t a, std::int64_t b) {
  constexpr std::int64_t lo = (std::numeric_limits<std::int64_t>::min)();
  constexpr std::int64_t hi = (std::numeric_limits<std::int64_t>::max)();
  if (a == 0 || b == 0) {
    return 0;
  }
  bool overflow = false;
  if (a > 0) {
    overflow = b > 0 ? a > hi / b : b < lo / a;
  } else {
    // a < 0. The division `lo / -1` would trap, so a == -1 is decided without it.
    if (b > 0) {
      overflow = a < lo / b;
    } else {
      overflow = (a == -1 && b == lo) || (a != -1 && b < hi / a);
    }
  }
  if (overflow) {
    throw std::overflow_error("integer overflow in a product of grid sizes");
  }
  return a * b;
}

/// Multiply the entries of a span, refusing a product that does not fit.
///
/// Args:
///     xs (std::span<const std::int64_t>): The factors.
///
/// Returns:
///     std::int64_t: Their product (1 for none).
///
/// Raises:
///     std::overflow_error: A partial product is outside the `int64` range.
[[nodiscard]] inline std::int64_t checked_product(std::span<const std::int64_t> xs) {
  std::int64_t p = 1;
  for (const std::int64_t x : xs) {
    p = checked_mul(p, x);
  }
  return p;
}

}  // namespace pantr::grid::detail
