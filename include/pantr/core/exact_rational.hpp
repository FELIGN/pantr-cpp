/// @file exact_rational.hpp
/// @brief Exact rational arithmetic, for operators that must be assembled without rounding.
///
/// A thin shim over Boost.Multiprecision's `cpp_rational` (arbitrary-precision numerator and
/// denominator, always reduced), used in standalone mode. Exact assembly is how pantr builds the
/// operators whose floating-point counterparts would lose the property under test (for example
/// degree-reduction operators); the rounding happens once, at the conversion to `double`.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"

#include <boost/multiprecision/cpp_int.hpp>

#include <cstdint>

namespace pantr::exact {

/// Arbitrary-precision rational number.
using Rational = boost::multiprecision::cpp_rational;

/// Arbitrary-precision integer.
using Integer = boost::multiprecision::cpp_int;

/// Build the reduced rational `num / den`.
///
/// Args:
///     num (std::int64_t): Numerator.
///     den (std::int64_t): Denominator; must be non-zero.
///
/// Returns:
///     Rational: `num / den` in lowest terms with a positive denominator.
///
/// Raises:
///     value_error: `den` is zero.
[[nodiscard]] inline Rational make_rational(std::int64_t num, std::int64_t den) {
  if (den == 0) {
    throw value_error("rational with a zero denominator");
  }
  return Rational(Integer(num), Integer(den));
}

/// Convert an exact rational to a double with Boost.Multiprecision's conversion.
///
/// No rounding guarantee beyond Boost's own is claimed here; callers that need a correctly rounded
/// result must establish it for their inputs.
///
/// Args:
///     q (const Rational&): The value.
///
/// Returns:
///     double: Boost.Multiprecision's conversion of `q`.
[[nodiscard]] inline double to_double(const Rational& q) { return q.convert_to<double>(); }

}  // namespace pantr::exact
