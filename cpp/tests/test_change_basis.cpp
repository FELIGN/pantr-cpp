// Change-of-basis matrices and their process tables. Oracles: matrices derived by hand in exact
// arithmetic at low degree, round trips in the bounded direction, the closed-form monomial matrix,
// and the tables' identity and immutability under concurrent first use (M2, a TSan target).
#include <doctest/doctest.h>

#include <pantr/change_basis.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <set>
#include <thread>
#include <vector>

namespace cb = pantr::change_basis;
using pantr::basis::LagrangeVariant;
constexpr double eps = std::numeric_limits<double>::epsilon();

TEST_CASE("change_basis: Bernstein-to-cardinal at degree 2 is the hand-derived matrix") {
  // N0 = b0 / 2, N1 = (b0 + 2 b1 + b2) / 2, N2 = b2 / 2 (Bernstein b, cardinal N on [0, 1]).
  const auto a = cb::bernstein_to_cardinal<double>(2);
  const double expected[9] = {0.5, 0, 0, 0.5, 1, 0.5, 0, 0, 0.5};
  for (int i = 0; i < 9; ++i) {
    // A Gram solve with kappa(G_bern) about 30 at degree 2: 64 eps is ample.
    CHECK(std::fabs(a.data()[i] - expected[i]) <= 64 * eps);
  }
  const auto b = cb::cardinal_to_bernstein<double>(2);  // its inverse: [[2,0,0],[-1,1,-1],[0,0,2]]
  const double inv[9] = {2, 0, 0, -1, 1, -1, 0, 0, 2};
  for (int i = 0; i < 9; ++i) {
    CHECK(std::fabs(b.data()[i] - inv[i]) <= 256 * eps);
  }
}

TEST_CASE("change_basis: Legendre-to-cardinal at degree 1 is the hand-derived matrix") {
  // N0 = 1 - t = p0 / 2 - p1 / (2 sqrt 3), N1 = t = p0 / 2 + p1 / (2 sqrt 3).
  const auto a = cb::legendre_to_cardinal<double>(1);
  const double r = 0.5 / std::sqrt(3.0);
  CHECK(a.data()[0] == doctest::Approx(0.5).epsilon(16 * eps));
  CHECK(a.data()[1] == doctest::Approx(-r).epsilon(16 * eps));
  CHECK(a.data()[2] == doctest::Approx(0.5).epsilon(16 * eps));
  CHECK(a.data()[3] == doctest::Approx(r).epsilon(16 * eps));
}

TEST_CASE("change_basis: inverses round trip in the bounded direction A X = I") {
  for (int d = 1; d <= 8; ++d) {
    const auto c = cb::lagrange_to_bernstein<double>(d, LagrangeVariant::gauss_legendre);
    const auto l = cb::bernstein_to_lagrange<double>(d, LagrangeVariant::gauss_legendre);
    const std::size_t m = static_cast<std::size_t>(d) + 1;
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = 0; j < m; ++j) {
        double s = 0.0;
        for (std::size_t k = 0; k < m; ++k) {
          s += c.data()[i * m + k] * l.data()[k * m + j];
        }
        // kappa(C) is below 1e3 through degree 8 for these nodes; 1e3 * 64 eps.
        CHECK(std::fabs(s - (i == j ? 1.0 : 0.0)) <= 64e3 * eps);
      }
    }
  }
}

TEST_CASE("change_basis: monomial-to-Bernstein is C(i, j) / C(d, j), first column exactly 1") {
  const auto m = cb::monomial_to_bernstein<double>(4);
  CHECK(m.data()[1 * 5 + 1] == 0.25);  // C(1,1)/C(4,1)
  CHECK(m.data()[2 * 5 + 1] == 0.5);   // C(2,1)/C(4,1)
  CHECK(m.data()[3 * 5 + 2] == 0.5);   // C(3,2)/C(4,2) = 3/6
  CHECK(m.data()[0 * 5 + 1] == 0.0);
  for (const int d : {0, 10, 56, 57, 80}) {  // both the double path and the exact-integer path
    const auto big = cb::monomial_to_bernstein<double>(d);
    for (int i = 0; i <= d; ++i) {
      CHECK(big.data()[i * (d + 1)] == 1.0);
      CHECK(big.data()[i * (d + 1) + i] == doctest::Approx(1.0 / [&] {
              double c = 1;
              for (int k = 1; k <= i; ++k) c = c * (d - i + k) / k;
              return c;
            }()));
    }
  }
}

TEST_CASE("change_basis: correctly rounded quotient agrees with double division where exact") {
  for (std::uint64_t a = 1; a < 2000; a += 37) {
    for (std::uint64_t b = 1; b < 3000; b += 41) {
      CHECK(pantr::exact::correctly_rounded_quotient(pantr::exact::Integer(a),
                                                     pantr::exact::Integer(b)) ==
            static_cast<double>(a) / static_cast<double>(b));
    }
  }
  // Ties to even at 2^53 + 1 over 1: 9007199254740993 rounds to 9007199254740992.
  const pantr::exact::Integer odd = (pantr::exact::Integer(1) << 53) + 1;
  CHECK(pantr::exact::correctly_rounded_quotient(odd, 1) == 9007199254740992.0);
  // The smallest subnormal and below it.
  const pantr::exact::Integer huge = pantr::exact::Integer(1) << 1074;
  CHECK(pantr::exact::correctly_rounded_quotient(1, huge) ==
        std::numeric_limits<double>::denorm_min());
  CHECK(pantr::exact::correctly_rounded_quotient(1, huge << 2) == 0.0);
}

TEST_CASE("change_basis: domain refusals carry the builder, the degree and the dtype") {
  CHECK_THROWS_WITH_AS(static_cast<void>(cb::cardinal_to_bernstein<float>(9)),
                       doctest::Contains("compute_cardinal_to_bernstein_1d: degree 9 is outside "
                                         "the supported domain for float32"),
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(cb::bernstein_to_lagrange<double>(38, LagrangeVariant::equispaces)),
      doctest::Contains("compute_bernstein_to_lagrange_1d with EQUISPACES nodes: degree 38"),
      pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(cb::lagrange_to_bernstein<double>(0, LagrangeVariant::equispaces)),
      "Degree must at least 1", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(cb::legendre_to_cardinal<double>(-1)),
                       "Degree must be non-negative", pantr::value_error);
}

TEST_CASE("change_basis: process tables return one immutable matrix per key, bounded") {
  cb::cache_clear<double>();
  const auto a =
      cb::cached<double>(cb::Kind::cardinal_to_bernstein, 3, LagrangeVariant::equispaces);
  const auto b = cb::cached<double>(cb::Kind::cardinal_to_bernstein, 3,
                                    LagrangeVariant::chebyshev_1st);  // variant not in this key
  CHECK(a.get() == b.get());
  const auto direct = cb::cardinal_to_bernstein<double>(3);
  for (std::size_t i = 0; i < direct.size(); ++i) {
    CHECK(a->data()[i] == direct.data()[i]);  // the same computation, so the same bits
  }
  const auto l1 =
      cb::cached<double>(cb::Kind::lagrange_to_bernstein, 3, LagrangeVariant::equispaces);
  const auto l2 =
      cb::cached<double>(cb::Kind::lagrange_to_bernstein, 3, LagrangeVariant::gauss_legendre);
  CHECK(l1.get() != l2.get());
  CHECK(cb::cache_size<double>() == 3);
  const auto past =
      cb::cached<double>(cb::Kind::legendre_to_cardinal, cb::change_basis_cache_max_degree + 1,
                         LagrangeVariant::equispaces);
  CHECK(cb::cache_size<double>() == 3);  // past the bound: computed, not kept
  CHECK(past->shape()[0] == static_cast<std::size_t>(cb::change_basis_cache_max_degree) + 2);
  CHECK_THROWS_AS(static_cast<void>(cb::cached<float>(cb::Kind::cardinal_to_bernstein, 9,
                                                      LagrangeVariant::equispaces)),
                  pantr::value_error);
  cb::cache_clear<double>();
  CHECK(cb::cache_size<double>() == 0);
  CHECK(a->data()[0] == direct.data()[0]);  // a handed-out matrix outlives the table entry
}

TEST_CASE("change_basis: concurrent first use of a key yields one shared matrix") {
  cb::cache_clear<double>();
  constexpr int n_threads = 8;
  std::vector<std::shared_ptr<const pantr::NdArray<double>>> got(n_threads);
  std::vector<std::thread> threads;
  for (int t = 0; t < n_threads; ++t) {
    threads.emplace_back([&got, t] {
      got[static_cast<std::size_t>(t)] =
          cb::cached<double>(cb::Kind::bernstein_to_cardinal, 6, LagrangeVariant::equispaces);
    });
  }
  for (auto& th : threads) {
    th.join();
  }
  std::set<const void*> distinct;
  for (const auto& g : got) {
    distinct.insert(g.get());
  }
  CHECK(distinct.size() == 1);
}
