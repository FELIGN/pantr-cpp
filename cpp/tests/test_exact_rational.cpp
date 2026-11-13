// Exact rationals. Oracles: closed-form identities that floating point cannot satisfy exactly.
#include <doctest/doctest.h>

#include <pantr/core/exact_rational.hpp>

#include <cstdint>

using pantr::exact::make_rational;
using pantr::exact::Rational;

TEST_CASE("exact: telescoping sum of 1/(k(k+1)) is exactly 1 - 1/(n+1)") {
  // A sum whose float evaluation rounds at every term; the exact one must hit the closed form.
  constexpr std::int64_t n = 200;
  Rational s = 0;
  for (std::int64_t k = 1; k <= n; ++k) {
    s += make_rational(1, k * (k + 1));
  }
  CHECK(s == 1 - make_rational(1, n + 1));
}

TEST_CASE("exact: values are reduced and keep a positive denominator") {
  const Rational q = make_rational(6, -4);
  CHECK(boost::multiprecision::numerator(q) == -3);
  CHECK(boost::multiprecision::denominator(q) == 2);
}

TEST_CASE("exact: products stay exact beyond 64-bit denominators") {
  // prod_{k=1}^{40} 1/k = 1/40!, whose denominator exceeds 2^64; multiplying back gives 1 exactly.
  Rational p = 1;
  for (std::int64_t k = 1; k <= 40; ++k) {
    p *= make_rational(1, k);
  }
  Rational back = p;
  for (std::int64_t k = 1; k <= 40; ++k) {
    back *= k;
  }
  CHECK(back == 1);
  CHECK(boost::multiprecision::denominator(p) > pantr::exact::Integer(UINT64_MAX));
}

TEST_CASE("exact: to_double of a dyadic rational is exact") {
  // A rational with a power-of-two denominator and a small numerator is representable, so any
  // conversion that is not wrong must return it exactly.
  CHECK(pantr::exact::to_double(make_rational(3, 8)) == 0.375);
  CHECK(pantr::exact::to_double(make_rational(-5, 1024)) == -5.0 / 1024.0);
}

TEST_CASE("exact: a zero denominator is a value_error") {
  CHECK_THROWS_WITH_AS(static_cast<void>(make_rational(1, 0)), "rational with a zero denominator",
                       pantr::value_error);
}
