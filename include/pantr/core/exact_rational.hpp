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

#include <cmath>
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

/// Divide two positive integers, rounding the quotient once, to nearest with ties to even.
///
/// The result is the double nearest `num / den`, subnormal results included, as Python's
/// `int / int` gives it: the quotient is formed with 55 or 56 significant bits by an exact integer
/// division, the remainder kept as a sticky bit, and the rounding done on those.
///
/// Args:
///     num (const Integer&): Numerator, `> 0`.
///     den (const Integer&): Denominator, `> 0`.
///
/// Returns:
///     double: The correctly rounded quotient; `inf` past the largest double.
///
/// Raises:
///     value_error: `num` or `den` is not positive.
[[nodiscard]] inline double correctly_rounded_quotient(const Integer& num, const Integer& den) {
  if (num <= 0 || den <= 0) {
    throw value_error("correctly_rounded_quotient needs positive integers");
  }
  const auto bits_num = static_cast<long long>(boost::multiprecision::msb(num)) + 1;
  const auto bits_den = static_cast<long long>(boost::multiprecision::msb(den)) + 1;
  // q = floor(num 2^shift / den) then has bit length 55 or 56.
  const long long shift = bits_den - bits_num + 55;
  Integer q;
  Integer r;
  // In-place shifts throughout: GCC 14 at -O3 reports a false -Wstringop-overflow inside
  // Boost's shift of a temporary (`Integer(x) << n`).
  if (shift >= 0) {
    Integer scaled = num;
    scaled <<= static_cast<unsigned>(shift);
    boost::multiprecision::divide_qr(scaled, den, q, r);
  } else {
    Integer scaled = den;
    scaled <<= static_cast<unsigned>(-shift);
    boost::multiprecision::divide_qr(num, scaled, q, r);
  }
  const bool sticky = r != 0;
  const auto qbits = static_cast<long long>(boost::multiprecision::msb(q)) + 1;
  // The value lies in [2^e, 2^(e + 1)); a double keeps 53 bits there, fewer below 2^-1022 (the
  // subnormal grid, spacing 2^-1074).
  const long long e = qbits - 1 - shift;
  const long long keep = e < -1022 ? 53 - (-1022 - e) : 53;
  if (keep < 0) {
    return 0.0;  // below half the smallest subnormal
  }
  const long long excess = qbits - keep;  // >= 2: qbits >= 55 and keep <= 53
  Integer mant = q;
  mant >>= static_cast<unsigned>(excess);
  Integer low = mant;
  low <<= static_cast<unsigned>(excess);
  low = q - low;  // the discarded bits
  // Round to nearest, ties to even: up when the first discarded bit is set and either a later
  // discarded bit, the division's remainder or the kept last bit is.
  const auto half_bit = static_cast<unsigned>(excess - 1);
  const bool above_half = boost::multiprecision::bit_test(low, half_bit);
  boost::multiprecision::bit_unset(low, half_bit);
  if (above_half && (low != 0 || sticky || boost::multiprecision::bit_test(mant, 0U))) {
    mant += 1;
  }
  return std::ldexp(mant.convert_to<double>(), static_cast<int>(excess - shift));
}

}  // namespace pantr::exact
