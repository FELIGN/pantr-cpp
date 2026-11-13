// Binomial coefficients as doubles: the exact-integer kernel and the correctly rounded general
// form. Oracles: Pascal's triangle in exact unsigned arithmetic, hand-computed values, and values
// computed independently with Python's math.comb (exact integers, and float(...), which rounds to
// nearest with ties to even).
#include <doctest/doctest.h>

#include <pantr/core/binomial.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

// Pascal's triangle up to row 62 in exact unsigned arithmetic: C(62, 31) = 465428353255261088
// is below 2^64, so no entry overflows.
std::vector<std::vector<std::uint64_t>> pascal_up_to_62() {
  std::vector<std::vector<std::uint64_t>> rows(63);
  for (std::size_t n = 0; n < rows.size(); ++n) {
    rows[n].assign(n + 1, 1);
    for (std::size_t k = 1; k < n; ++k) {
      rows[n][k] = rows[n - 1][k - 1] + rows[n - 1][k];
    }
  }
  return rows;
}

}  // namespace

TEST_CASE("binomial: bincoeff matches Pascal's triangle exactly through the envelope") {
  const auto rows = pascal_up_to_62();
  for (long long n = 0; n <= pantr::bincoeff_max_n; ++n) {
    for (long long k = 0; k <= n; ++k) {
      CAPTURE(n);
      CAPTURE(k);
      // Both sides round the same exact integer once to double.
      const double expected =
          static_cast<double>(rows[static_cast<std::size_t>(n)][static_cast<std::size_t>(k)]);
      const double got = pantr::bincoeff(n, k);
      CHECK(got == expected);
    }
  }
}

TEST_CASE("binomial: bincoeff hand values, the extremes of the envelope, and out-of-range k") {
  CHECK(pantr::bincoeff(0, 0) == 1.0);
  CHECK(pantr::bincoeff(5, 2) == 10.0);
  CHECK(pantr::bincoeff(10, 5) == 252.0);
  // Math.comb values, exact integers: the largest intermediate C(n, k) * min(k, n - k) is
  // 6.98e18 at n = 61, below the unsigned ceiling.
  CHECK(pantr::bincoeff(61, 30) == static_cast<double>(232714176627630544ULL));
  CHECK(pantr::bincoeff(62, 31) == static_cast<double>(465428353255261088ULL));
  CHECK(pantr::bincoeff(5, -1) == 0.0);
  CHECK(pantr::bincoeff(5, 6) == 0.0);
}

TEST_CASE("binomial: the envelope check passes at the limit and refuses past it, verbatim") {
  CHECK_NOTHROW(pantr::check_bincoeff_envelope(pantr::bincoeff_max_n, "x"));
  CHECK_NOTHROW(pantr::check_bincoeff_envelope(0, "x"));
  CHECK_THROWS_WITH_AS(
      pantr::check_bincoeff_envelope(62, "Degree elevation to degree 62 in direction 0"),
      "Degree elevation to degree 62 in direction 0 needs binomial coefficients up to C(62, k), "
      "beyond the largest upper index 61 that pantr's exact-integer binomial kernel can compute "
      "without an int64 overflow. Past that the coefficients wrap silently and the result is "
      "corrupted rather than merely inaccurate.",
      pantr::value_error);
}

TEST_CASE("binomial: binomial_double equals bincoeff inside the envelope") {
  // Above 2^53 (from C(57, 28)) this compares the hand-written correct rounding against the
  // hardware conversion of the same integer, which rounds to nearest even.
  for (long long n = 0; n <= pantr::bincoeff_max_n; ++n) {
    for (long long k = 0; k <= n; ++k) {
      CAPTURE(n);
      CAPTURE(k);
      const double got = pantr::binomial_double(n, k);
      const double expected = pantr::bincoeff(n, k);
      CHECK(got == expected);
    }
  }
}

TEST_CASE("binomial: binomial_double past 64 bits is the correctly rounded exact value") {
  // Each literal is float(math.comb(n, k)) from Python, computed independently of this library;
  // the comments give the exact integer's bit length, which is where the uint64 path ends.
  // C(67, 33) = 14226520737620288370 fits in 64 bits but its recurrence does not.
  CHECK(pantr::binomial_double(67, 33) == 1.4226520737620287e+19);
  // C(68, 34) = 28453041475240576740 has 65 bits.
  CHECK(pantr::binomial_double(68, 34) == 2.8453041475240575e+19);
  // 97 bits.
  CHECK(pantr::binomial_double(100, 50) == 1.008913445455642e+29);
  CHECK(pantr::binomial_double(90, 45) == 1.0382742128755342e+26);
  CHECK(pantr::binomial_double(200, 100) == 9.054851465610328e+58);
  // 79 bits, small k with a large n.
  CHECK(pantr::binomial_double(1030, 10) == 3.544676281344928e+23);
  // 988 bits: near the double range but finite.
  CHECK(pantr::binomial_double(1030, 400) == 1.6607886949053574e+297);
  // Symmetry C(n, k) = C(n, n - k).
  CHECK(pantr::binomial_double(1030, 630) == pantr::binomial_double(1030, 400));
}

TEST_CASE("binomial: binomial_double rounds an exact tie to even") {
  // C(73, 19) = 159186450151978480 has 58 bits and its 5 discarded bits are exactly 10000: a
  // tie with an odd kept mantissa, which rounds up to even. C(76, 18) = 125288932441604200 has 57
  // bits and its 4 discarded bits are exactly 1000: a tie with an even kept mantissa, which
  // rounds down. The literals are float(math.comb(...)).
  CHECK(pantr::binomial_double(73, 19) == 1.591864501519785e+17);
  CHECK(pantr::binomial_double(76, 18) == 1.252889324416042e+17);
}

TEST_CASE("binomial: binomial_double edge cases and errors") {
  CHECK(pantr::binomial_double(0, 0) == 1.0);
  CHECK(pantr::binomial_double(100, 0) == 1.0);
  CHECK(pantr::binomial_double(100, 100) == 1.0);
  CHECK(pantr::binomial_double(100, 3) == 161700.0);
  CHECK(pantr::binomial_double(5, 6) == 0.0);
  CHECK_THROWS_WITH_AS(static_cast<void>(pantr::binomial_double(-1, 0)),
                       "n must be a non-negative integer", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pantr::binomial_double(5, -1)),
                       "k must be a non-negative integer", pantr::value_error);
  // C(1030, 515) is about 2.9e308, past the largest double (float(math.comb(1030, 515)) raises
  // OverflowError).
  CHECK_THROWS_WITH_AS(static_cast<void>(pantr::binomial_double(1030, 515)),
                       "int too large to convert to float", std::overflow_error);
}
