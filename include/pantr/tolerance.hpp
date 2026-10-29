/// @file tolerance.hpp
/// @brief Dimensionless relative tolerance presets: a fixed number of machine epsilons.
///
/// Each preset is `K * eps(T)`, with `K` the same for every floating-point format, so a preset
/// means the same number of roundings in every precision. None of them is a comparison on its own:
/// a call site multiplies one by the magnitude of the quantity it compares and names that
/// magnitude. `K` is an acknowledged safety factor, stated rather than tuned, and a power of two
/// so it is exact in every format:
///
/// | preset             | `K`  | what it pays for                                                |
/// |--------------------|------|-----------------------------------------------------------------|
/// | `get_strict`       | 4    | a handful of roundings: one convex combination, then a subtract |
/// | `get_default`      | 64   | a short algorithm (about 16 operations) plus build slack        |
/// | `get_conservative` | 4096 | a long accumulation, or a condition number up to about 1e3      |
///
/// The Python module `pantr.tolerance` documents each factor at length and serves the same
/// presets for every NumPy floating dtype from these factors.
#pragma once

#include "pantr/config.hpp"

#include <concepts>
#include <limits>

namespace pantr::tolerance {

/// Machine epsilons a `strict` comparison allows for: a value reached by one convex combination
/// (a subtract, a multiply and an add, three roundings), then differenced against another (the
/// fourth).
inline constexpr double strict_eps_factor = 4.0;

/// Machine epsilons a `default` comparison allows for: a short algorithm, whose worst-case
/// accumulation is about `16 * eps`, times four for build slack (FMA contraction, vector width,
/// libm), rounded to the power of two it lands on.
inline constexpr double default_eps_factor = 64.0;

/// Machine epsilons a `conservative` comparison allows for: a long accumulation, or a condition
/// number up to about 1e3, with a factor of four of headroom and the same build slack.
inline constexpr double conservative_eps_factor = 4096.0;

/// Get the machine epsilon of a floating-point format.
///
/// Returns:
///     T: `std::numeric_limits<T>::epsilon()`, the gap between 1 and the next representable value.
template <std::floating_point T>
[[nodiscard]] constexpr T machine_epsilon() noexcept {
  return std::numeric_limits<T>::epsilon();
}

/// Get the strict relative tolerance, `strict_eps_factor * eps(T)`.
///
/// Returns:
///     T: The dimensionless tolerance; exact, the factor being a power of two.
template <std::floating_point T>
[[nodiscard]] constexpr T get_strict() noexcept {
  return static_cast<T>(strict_eps_factor) * machine_epsilon<T>();
}

/// Get the default relative tolerance, `default_eps_factor * eps(T)`.
///
/// Returns:
///     T: The dimensionless tolerance; exact, the factor being a power of two.
template <std::floating_point T>
[[nodiscard]] constexpr T get_default() noexcept {
  return static_cast<T>(default_eps_factor) * machine_epsilon<T>();
}

/// Get the conservative relative tolerance, `conservative_eps_factor * eps(T)`.
///
/// Returns:
///     T: The dimensionless tolerance; exact, the factor being a power of two.
template <std::floating_point T>
[[nodiscard]] constexpr T get_conservative() noexcept {
  return static_cast<T>(conservative_eps_factor) * machine_epsilon<T>();
}

}  // namespace pantr::tolerance
