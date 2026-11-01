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
/// Note:
///     Both operands must be non-negative; no check is made. Every caller multiplies cell counts,
///     block counts, indices and ranks, which are, and a negative operand has no reachable case
///     here to be licensed.
///
/// Args:
///     a (std::int64_t): First factor, at least 0.
///     b (std::int64_t): Second factor, at least 0.
///
/// Returns:
///     std::int64_t: `a * b`.
///
/// Raises:
///     std::overflow_error: The product is outside the `int64` range.
[[nodiscard]] inline std::int64_t checked_mul(std::int64_t a, std::int64_t b) {
  constexpr std::int64_t hi = (std::numeric_limits<std::int64_t>::max)();
  if (a == 0 || b == 0) {
    return 0;
  }
  // For b > 0, `a * b <= hi` iff `a <= floor(hi / b)`.
  if (a > hi / b) {
    throw std::overflow_error("integer overflow in a product of grid sizes");
  }
  return a * b;
}

/// Multiply the entries of a span, refusing a product that does not fit.
///
/// Note:
///     The factors must be non-negative; no check is made (see `checked_mul`).
///
/// Args:
///     xs (std::span<const std::int64_t>): The factors, each at least 0.
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
