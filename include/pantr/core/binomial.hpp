/// @file binomial.hpp
/// @brief Binomial coefficients as `double`: the exact-integer kernel and a correctly rounded
///        general form.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/exact_rational.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace pantr {

/// Largest upper index `n` for which `bincoeff` honours its contract.
///
/// The multiplicative recurrence `C(m, i) = C(m - 1, i - 1) * m / i` is run over `kk = min(k, n -
/// k)` in unsigned 64-bit arithmetic. Its running value after step `i` is exactly
/// `C(n - kk + i, i)`, an integer, so every division is exact; the largest intermediate is the
/// product before the last division, `C(n, kk) * kk`. That product is `6.98e18` at `n = 61` and
/// `1.44e19` at `n = 62`, both below the unsigned ceiling `1.84e19`, and `2.84e19` at `n = 63`,
/// above it: the algorithm as written is exact through `n = 62`. The limit is kept at 61, the
/// value of the Python kernel this replaces (where the signed ceiling `9.22e18` binds), so that a
/// request is accepted or refused identically in both implementations and the refusal message
/// is unchanged. Unsigned overflow is defined to wrap in C++, so past the limit the result is a
/// wrong number, not undefined behaviour.
///
/// This is the integer limit, not the tighter `n = 56` beyond which the `double` result can no
/// longer hold `C(n, k)` exactly (`C(57, 28)` is the first past `2^53`). Every caller consumes the
/// value inside a floating-point ratio of binomials, so a correctly rounded operand is all it can
/// use; for `57 <= n <= 61` the integer is exact and the conversion rounds once.
inline constexpr long long bincoeff_max_n = 61;

/// Compute the binomial coefficient `C(n, k)` in exact integer arithmetic.
///
/// Args:
///     n (long long): Upper index, `0 <= n <= bincoeff_max_n`.
///     k (long long): Lower index.
///
/// Returns:
///     double: `C(n, k)` rounded once from the exact integer, or `0.0` when `k` lies outside
///     `[0, n]`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): from `n = 63` an intermediate
///     wraps and the result is wrong. For general use, call `binomial_double`; a caller that
///     takes `n` from user input calls `check_bincoeff_envelope` first.
[[nodiscard]] inline double bincoeff(long long n, long long k) noexcept {
  if (k < 0 || k > n) {
    return 0.0;
  }
  const long long kk = (std::min)(k, n - k);
  std::uint64_t result = 1;
  for (long long i = 1; i <= kk; ++i) {
    result = result * static_cast<std::uint64_t>(n - kk + i) / static_cast<std::uint64_t>(i);
  }
  return static_cast<double>(result);
}

/// Refuse an upper index beyond the exactness envelope of `bincoeff`.
///
/// Args:
///     n (long long): Largest upper index the caller will pass to `bincoeff`.
///     what (std::string_view): Description of the requested operation, for the message.
///
/// Raises:
///     value_error: `n > bincoeff_max_n`.
inline void check_bincoeff_envelope(long long n, std::string_view what) {
  if (n > bincoeff_max_n) {
    throw value_error(std::string(what) + " needs binomial coefficients up to C(" +
                      std::to_string(n) + ", k), beyond the largest upper index " +
                      std::to_string(bincoeff_max_n) +
                      " that pantr's exact-integer binomial kernel can compute without an int64 "
                      "overflow. Past that the coefficients wrap silently and the result is "
                      "corrupted rather than merely inaccurate.");
  }
}

/// Compute `C(n, k)` rounded once to the nearest `double`, for any `n >= 0`.
///
/// The recurrence of `bincoeff` runs in unsigned 64-bit arithmetic while the next product fits
/// (tested as `result > UINT64_MAX / m` before multiplying, so nothing wraps and no compiler
/// intrinsic is needed); on the first product that would not fit, the running value continues in
/// `boost::multiprecision::cpp_int`, whose divisions are exact for the same reason. A value of at
/// most `2^53` converts exactly. Any other value is rounded by
/// `exact::correctly_rounded_quotient(value, 1)`, which shifts the integer to 55 or 56 significant
/// bits and rounds to nearest, ties to even, by hand with a sticky bit, so correctness of the
/// rounding rests on that code and its tests and not on how Boost converts a `cpp_int` (the only
/// conversion Boost performs there is of a mantissa of at most `2^53`, which is exact). The
/// rounding is therefore that of Python's `float(math.comb(n, k))`, independent of the
/// floating-point rounding mode.
///
/// Args:
///     n (long long): Upper index, `>= 0`.
///     k (long long): Lower index, `>= 0`.
///
/// Returns:
///     double: `C(n, k)`, or `0.0` when `k > n`.
///
/// Raises:
///     value_error: `n < 0` or `k < 0`.
///     std::overflow_error: `C(n, k)` rounds beyond the largest `double` (Python's `OverflowError`
///         on `float(math.comb(n, k))`).
[[nodiscard]] inline double binomial_double(long long n, long long k) {
  if (n < 0) {
    throw value_error("n must be a non-negative integer");
  }
  if (k < 0) {
    throw value_error("k must be a non-negative integer");
  }
  if (k > n) {
    return 0.0;
  }
  const long long kk = (std::min)(k, n - k);
  std::uint64_t result = 1;
  long long i = 1;
  for (; i <= kk; ++i) {
    const auto m = static_cast<std::uint64_t>(n - kk + i);
    if (result > (std::numeric_limits<std::uint64_t>::max)() / m) {
      break;
    }
    result = result * m / static_cast<std::uint64_t>(i);
  }
  if (i > kk) {  // the whole recurrence fit in 64 bits
    constexpr std::uint64_t two_pow_53 = std::uint64_t{1} << 53;
    if (result <= two_pow_53) {
      return static_cast<double>(result);
    }
    return exact::correctly_rounded_quotient(exact::Integer(result), exact::Integer(1));
  }
  exact::Integer big(result);
  for (; i <= kk; ++i) {
    big *= n - kk + i;
    big /= i;  // exact: the product is C(n - kk + i, i) * i
  }
  const double rounded = exact::correctly_rounded_quotient(big, exact::Integer(1));
  if (!(rounded <= (std::numeric_limits<double>::max)())) {
    throw std::overflow_error("int too large to convert to float");
  }
  return rounded;
}

}  // namespace pantr
