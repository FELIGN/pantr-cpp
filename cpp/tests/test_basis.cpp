// The one-dimensional bases and their tensor products. Oracles: closed forms (binomial Bernstein,
// explicit low-degree polynomials), partition of unity, orthonormality by an exact quadrature, the
// Kronecker property, and products of one-dimensional values.
#include <doctest/doctest.h>

#include <pantr/basis/tabulate.hpp>
#include <pantr/parallel/pool.hpp>
#include <pantr/quad/rules_1d.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace basis = pantr::basis;
using basis::LagrangeVariant;
constexpr double eps = std::numeric_limits<double>::epsilon();

namespace {

double binomial(int n, int k) {
  double c = 1.0;
  for (int i = 1; i <= k; ++i) {
    c = c * (n - k + i) / i;
  }
  return c;
}

std::vector<double> sample_points() { return {0.0, 1e-3, 0.1, 0.25, 0.5, 0.7, 0.999, 1.0}; }

}  // namespace

TEST_CASE("basis: Bernstein matches C(n, i) u^i (1 - u)^(n - i) at every degree path") {
  const auto pts = sample_points();
  // Degrees on both sides of the unmirrored fast path's bound.
  for (const int n : {0, 1, 3, 20, 21, 40}) {
    const auto b = basis::tabulate_bernstein_1d<double>(n, pts);
    for (std::size_t j = 0; j < pts.size(); ++j) {
      double sum = 0.0;
      for (int i = 0; i <= n; ++i) {
        const double u = pts[j];
        const double expected = binomial(n, i) * std::pow(u, i) * std::pow(1.0 - u, n - i);
        const double got =
            b.data()[j * static_cast<std::size_t>(n + 1) + static_cast<std::size_t>(i)];
        CAPTURE(n);
        CAPTURE(u);
        CAPTURE(i);
        // Both sides are O(n) roundings of a non-negative product; 8 n eps relative, with the
        // absolute floor of the same size against the largest term (<= 1).
        CHECK(std::fabs(got - expected) <= 8.0 * (n + 1) * eps * (expected + eps));
        sum += got;
      }
      CHECK(std::fabs(sum - 1.0) <= 8.0 * (n + 1) * eps);
    }
  }
}

TEST_CASE("basis: Bernstein derivatives of degree 2 match the closed form, order 3 vanishes") {
  // B0 = (1-u)^2, B1 = 2u(1-u), B2 = u^2: d/du = (-2(1-u), 2 - 4u, 2u), d2 = (2, -4, 2).
  const std::vector<double> pts = {0.0, 0.3, 1.0};
  pantr::NdArray<double> out({3, 4, 3});
  basis::tabulate_bernstein_derivatives_1d<double>(2, pts, 3, out.view<3>());
  for (std::size_t j = 0; j < 3; ++j) {
    const double u = pts[j];
    const double* d = out.data() + j * 12;
    CHECK(d[3] == doctest::Approx(-2.0 * (1.0 - u)));
    CHECK(d[4] == doctest::Approx(2.0 - 4.0 * u));
    CHECK(d[5] == doctest::Approx(2.0 * u));
    CHECK(d[6] == 2.0);
    CHECK(d[7] == -4.0);
    CHECK(d[8] == 2.0);
    CHECK(d[9] == 0.0);
    CHECK(d[10] == 0.0);
    CHECK(d[11] == 0.0);
  }
}

TEST_CASE("basis: Bernstein derivatives of order n are n! (-1)^(n-i) C(n, i), past 64 bits") {
  // The n-th derivative of B_{i,n} is the constant n! (-1)^(n-i) C(n, i). At degree 22 the
  // scaling n!/(n-k)! passes 2^64 at k = 21, so the double fallback of the scaling runs (the
  // Numba kernel wrapped its int64 there); 1e-12 relative leaves room for the O(n^2)
  // recurrence's roundings, whose values are bounded by C(n, i) 2^n.
  const int n = 22;
  const std::vector<double> pts = {0.3};
  pantr::NdArray<double> out({1, static_cast<std::size_t>(n) + 1, static_cast<std::size_t>(n) + 1});
  basis::tabulate_bernstein_derivatives_1d<double>(n, pts, n, out.view<3>());
  double fact = 1.0;
  for (int k = 2; k <= n; ++k) {
    fact *= k;
  }
  for (int i = 0; i <= n; ++i) {
    const double expected = fact * binomial(n, i) * (((n - i) % 2 == 0) ? 1.0 : -1.0);
    const double got = out.data()[static_cast<std::size_t>(n) * static_cast<std::size_t>(n + 1) +
                                  static_cast<std::size_t>(i)];
    CAPTURE(i);
    CHECK(std::fabs(got - expected) <= 1e-12 * fact * binomial(n, n / 2));
  }
}

TEST_CASE("basis: float tabulations are the double-intermediate values stored in float") {
  // Bernstein, cardinal and Legendre compute in double and store each value in float, reading it
  // back for the next step; agreement with the double tabulation is a few float roundings.
  const std::vector<double> pd = {0.0, 0.1, 0.45, 0.5, 0.77, 1.0};
  std::vector<float> pf(pd.size());
  for (std::size_t i = 0; i < pd.size(); ++i) {
    pf[i] = static_cast<float>(pd[i]);
  }
  const double feps = std::numeric_limits<float>::epsilon();
  for (const int p : {0, 3, 7}) {  // 7 is past float's unmirrored bound (6)
    const auto bd = basis::tabulate_bernstein_1d<double>(p, pd);
    const auto bf = basis::tabulate_bernstein_1d<float>(p, pf);
    const auto cd = basis::tabulate_cardinal_bspline_1d<double>(p, pd);
    const auto cf = basis::tabulate_cardinal_bspline_1d<float>(p, pf);
    const auto ld = basis::tabulate_legendre_1d<double>(p, pd);
    const auto lf = basis::tabulate_legendre_1d<float>(p, pf);
    for (std::size_t i = 0; i < bd.size(); ++i) {
      CAPTURE(p);
      CAPTURE(i);
      CHECK(std::fabs(static_cast<double>(bf.data()[i]) - bd.data()[i]) <= 4.0 * (p + 1) * feps);
      CHECK(std::fabs(static_cast<double>(cf.data()[i]) - cd.data()[i]) <= 4.0 * (p + 1) * feps);
      // Legendre values grow like sqrt(2p + 1) <= 4 here.
      CHECK(std::fabs(static_cast<double>(lf.data()[i]) - ld.data()[i]) <= 16.0 * (p + 1) * feps);
    }
  }
  // Derivatives run in the storage type: float within float roundings of double.
  pantr::NdArray<double> dd({pd.size(), 3, 4});
  pantr::NdArray<float> df({pf.size(), 3, 4});
  basis::tabulate_bernstein_derivatives_1d<double>(3, pd, 2, dd.view<3>());
  basis::tabulate_bernstein_derivatives_1d<float>(3, pf, 2, df.view<3>());
  for (std::size_t i = 0; i < dd.size(); ++i) {
    CHECK(std::fabs(static_cast<double>(df.data()[i]) - dd.data()[i]) <= 64.0 * feps * 6.0);  // |values| <= 6
  }
}

TEST_CASE("basis: cardinal B-splines of degree 2 match the explicit polynomials") {
  const auto pts = sample_points();
  const auto b = basis::tabulate_cardinal_bspline_1d<double>(2, pts);
  for (std::size_t j = 0; j < pts.size(); ++j) {
    const double t = pts[j];
    CHECK(b.data()[3 * j] == doctest::Approx((1 - t) * (1 - t) / 2).epsilon(8 * eps));
    CHECK(b.data()[3 * j + 1] == doctest::Approx((1 + 2 * t - 2 * t * t) / 2).epsilon(8 * eps));
    CHECK(b.data()[3 * j + 2] == doctest::Approx(t * t / 2).epsilon(8 * eps));
  }
}

TEST_CASE("basis: Legendre is orthonormal under an exact Gauss-Legendre rule") {
  const int p = 12;
  const auto rule = pantr::quad::gauss_legendre_1d<double>(p + 1);  // exact to degree 2p + 1
  const auto v = basis::tabulate_legendre_1d<double>(p, rule.nodes.flat());
  for (int a = 0; a <= p; ++a) {
    for (int c = 0; c <= p; ++c) {
      double s = 0.0;
      for (int q = 0; q <= p; ++q) {
        s += rule.weights.data()[q] * v.data()[q * (p + 1) + a] * v.data()[q * (p + 1) + c];
      }
      // Values grow like sqrt(2p + 1); the recurrence and sum cost O(p) roundings each.
      CHECK(std::fabs(s - (a == c ? 1.0 : 0.0)) <= 64.0 * (p + 1) * eps);
    }
  }
}

TEST_CASE("basis: Lagrange is the Kronecker delta at its nodes and a partition of unity") {
  for (const auto v : {LagrangeVariant::equispaces, LagrangeVariant::gauss_legendre,
                       LagrangeVariant::gauss_lobatto_legendre, LagrangeVariant::chebyshev_1st,
                       LagrangeVariant::chebyshev_2nd}) {
    const int p = 6;
    const auto nodes = basis::lagrange_points<double>(v, p + 1);
    const auto at_nodes = basis::tabulate_lagrange_1d<double>(p, v, nodes.flat());
    for (int j = 0; j <= p; ++j) {
      for (int i = 0; i <= p; ++i) {
        CHECK(at_nodes.data()[j * (p + 1) + i] == (i == j ? 1.0 : 0.0));
      }
    }
    const auto pts = std::vector<double>{0.05, 0.33, 0.61, 0.97};
    const auto vals = basis::tabulate_lagrange_1d<double>(p, v, pts);
    for (std::size_t j = 0; j < pts.size(); ++j) {
      double s = 0.0;
      for (int i = 0; i <= p; ++i) {
        s += vals.data()[j * static_cast<std::size_t>(p + 1) + static_cast<std::size_t>(i)];
      }
      CHECK(s == doctest::Approx(1.0).epsilon(1e-13));
    }
  }
  CHECK_THROWS_WITH_AS(static_cast<void>(basis::tabulate_lagrange_1d<double>(
                           -1, LagrangeVariant::equispaces, std::vector<double>{0.5})),
                       "degree must be non-negative", pantr::value_error);
}

TEST_CASE("basis: Lagrange of degree 30 on Chebyshev nodes reproduces t^3") {
  // sum_i x_i^3 L_i(t) = t^3: the barycentric weights' common scaling must cancel.
  const int p = 30;
  const auto nodes = basis::lagrange_points<double>(LagrangeVariant::chebyshev_2nd, p + 1);
  const std::vector<double> pts = {0.123, 0.5, 0.875};
  const auto vals = basis::tabulate_lagrange_1d<double>(p, LagrangeVariant::chebyshev_2nd, pts);
  for (std::size_t j = 0; j < pts.size(); ++j) {
    double s = 0.0;
    for (int i = 0; i <= p; ++i) {
      s += std::pow(nodes.data()[i], 3) *
           vals.data()[j * static_cast<std::size_t>(p + 1) + static_cast<std::size_t>(i)];
    }
    // Chebyshev Lebesgue constant at degree 30 is below 4; 1e3 eps leaves room for the rest.
    CHECK(std::fabs(s - std::pow(pts[j], 3)) <= 1e3 * eps);
  }
}

TEST_CASE("basis: out extents and negative degrees are value_errors with the Python messages") {
  const std::vector<double> pts = {0.1, 0.2};
  pantr::NdArray<double> out({2, 3});
  CHECK_THROWS_WITH_AS(basis::tabulate_bernstein_1d<double>(3, pts, out.view<2>()),
                       "Output array has shape (2, 3), but expected shape (2, 4)",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(basis::tabulate_legendre_1d<double>(-2, pts)),
                       "degree must be non-negative", pantr::value_error);
}

TEST_CASE("basis: tensor tabulation is the product of 1D values in C and F order") {
  const pantr::quad::PointsLattice<double> lat({{0.2, 0.7}, {0.1, 0.4, 0.9}});
  const std::vector<int> degrees = {1, 2};
  const auto bx = basis::tabulate_bernstein_1d<double>(1, lat.pts(0));
  const auto by = basis::tabulate_bernstein_1d<double>(2, lat.pts(1));
  for (const auto order : {pantr::Order::C, pantr::Order::F}) {
    pantr::NdArray<double> out({6, 6});
    basis::tabulate_tensor<double>(basis::Family::bernstein, degrees, LagrangeVariant::equispaces,
                                   lat, order, out.view<2>());
    for (std::size_t px = 0; px < 2; ++px) {
      for (std::size_t py = 0; py < 3; ++py) {
        for (std::size_t fx = 0; fx < 2; ++fx) {
          for (std::size_t fy = 0; fy < 3; ++fy) {
            const bool c = order == pantr::Order::C;
            const std::size_t p = c ? px * 3 + py : py * 2 + px;
            const std::size_t f = c ? fx * 3 + fy : fy * 2 + fx;
            CHECK(out.data()[p * 6 + f] == bx.data()[px * 2 + fx] * by.data()[py * 3 + fy]);
          }
        }
      }
    }
  }
  // Scattered points: row p uses coordinate p of each direction.
  const std::vector<double> pts = {0.2, 0.1, 0.7, 0.4};  // (2, 2)
  const pantr::mdview<const double, 2> pv(pts.data(), 2, 2);
  pantr::NdArray<double> out({2, 6});
  basis::tabulate_tensor<double>(basis::Family::bernstein, degrees, LagrangeVariant::equispaces, pv,
                                 pantr::Order::C, out.view<2>());
  CHECK(out.data()[6 + 4] == bx.data()[2 + 1] * by.data()[3 * 1 + 1]);
  CHECK_THROWS_WITH_AS(basis::tabulate_tensor<double>(basis::Family::bernstein, std::vector<int>{1},
                                                      LagrangeVariant::equispaces, lat,
                                                      pantr::Order::C, out.view<2>()),
                       "The number of evaluators must be equal to the dimension of the points.",
                       pantr::value_error);
}

TEST_CASE("basis: a 3D tensor tabulation is the product of 1D values, in C and F order") {
  const pantr::quad::PointsLattice<double> lat({{0.2, 0.7}, {0.1, 0.4, 0.9}, {0.35, 0.6}});
  const std::vector<int> degrees = {1, 2, 1};
  const auto b0 = basis::tabulate_bernstein_1d<double>(1, lat.pts(0));
  const auto b1 = basis::tabulate_bernstein_1d<double>(2, lat.pts(1));
  const auto b2 = basis::tabulate_bernstein_1d<double>(1, lat.pts(2));
  for (const auto order : {pantr::Order::C, pantr::Order::F}) {
    pantr::NdArray<double> out({12, 12});
    basis::tabulate_tensor<double>(basis::Family::bernstein, degrees, LagrangeVariant::equispaces,
                                   lat, order, out.view<2>());
    const bool c = order == pantr::Order::C;
    for (std::size_t p0 = 0; p0 < 2; ++p0) {
      for (std::size_t p1 = 0; p1 < 3; ++p1) {
        for (std::size_t p2 = 0; p2 < 2; ++p2) {
          const std::size_t p = c ? (p0 * 3 + p1) * 2 + p2 : (p2 * 3 + p1) * 2 + p0;
          for (std::size_t f0 = 0; f0 < 2; ++f0) {
            for (std::size_t f1 = 0; f1 < 3; ++f1) {
              for (std::size_t f2 = 0; f2 < 2; ++f2) {
                const std::size_t f = c ? (f0 * 3 + f1) * 2 + f2 : (f2 * 3 + f1) * 2 + f0;
                // Products in direction order, ((r0 r1) r2), as the einsum chain did.
                const double expected =
                    (b0.data()[p0 * 2 + f0] * b1.data()[p1 * 3 + f1]) * b2.data()[p2 * 2 + f2];
                CHECK(out.data()[p * 12 + f] == expected);
              }
            }
          }
        }
      }
    }
  }
}

TEST_CASE("basis: results do not depend on the thread count (M7)") {
  // Enough points to cross the serial threshold and several grains.
  std::vector<double> pts(200000);
  for (std::size_t i = 0; i < pts.size(); ++i) {
    pts[i] = static_cast<double>(i) / static_cast<double>(pts.size() - 1);
  }
  const std::size_t saved = pantr::parallel::get_num_threads();
  pantr::parallel::set_num_threads(1);
  const auto one = basis::tabulate_bernstein_1d<double>(25, pts);
  pantr::parallel::set_num_threads(static_cast<std::int64_t>(pantr::parallel::max_threads()));
  const auto many = basis::tabulate_bernstein_1d<double>(25, pts);
  pantr::parallel::set_num_threads(static_cast<std::int64_t>(saved));
  // Each point is computed by the same code on one thread whatever the split, so the values
  // agree to the build's determinism; asserted within one rounding rather than bitwise.
  for (std::size_t i = 0; i < one.size(); ++i) {
    REQUIRE(std::fabs(one.data()[i] - many.data()[i]) <= eps * std::fabs(one.data()[i]));
  }
}
