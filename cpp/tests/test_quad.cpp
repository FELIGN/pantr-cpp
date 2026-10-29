// Quadrature rules, Lambert W and the tolerance presets. Oracles: closed-form moments, the
// defining equation of W, symmetry, and the exact constants of the presets.
#include <doctest/doctest.h>

#include <pantr/core/lambert_w.hpp>
#include <pantr/quad/points_lattice.hpp>
#include <pantr/quad/quadrature_rule.hpp>
#include <pantr/quad/rules_1d.hpp>
#include <pantr/tolerance.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <vector>

namespace quad = pantr::quad;
constexpr double eps = std::numeric_limits<double>::epsilon();

TEST_CASE("tolerance: presets are K eps with K = 4, 64, 4096, exactly") {
  static_assert(pantr::tolerance::get_strict<double>() == 4.0 * eps);
  static_assert(pantr::tolerance::get_default<double>() == 64.0 * eps);
  static_assert(pantr::tolerance::get_conservative<double>() == 4096.0 * eps);
  static_assert(pantr::tolerance::get_default<float>() ==
                64.0F * std::numeric_limits<float>::epsilon());
  CHECK(pantr::tolerance::get_default<double>() == 1.4210854715202004e-14);
}

TEST_CASE("lambert_w0: solves w e^w = x, with W(0) = 0 and W(e) = 1") {
  CHECK(pantr::lambert_w0(0.0) == 0.0);
  CHECK(pantr::lambert_w0(std::numbers::e) == doctest::Approx(1.0).epsilon(4 * eps));
  for (const double x : {1e-300, 1e-8, 0.5, 1.0, 1.884955592153876, 10.0, 1e3, 1e9, 1e300}) {
    CAPTURE(x);
    const double w = pantr::lambert_w0(x);
    // Residual of the defining equation, relative: |w e^w - x| / x is a few roundings of the
    // product, amplified by w + 1 through the error in w (the condition number of x -> W(x) is
    // 1 / (1 + W)); 16 (w + 1) eps covers both.
    CHECK(std::fabs(w * std::exp(w) - x) <= 16.0 * (w + 1.0) * eps * x);
  }
  CHECK_THROWS_AS(static_cast<void>(pantr::lambert_w0(-1.0)), pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(pantr::lambert_w0(std::nan(""))), pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(pantr::lambert_w0(INFINITY)), pantr::value_error);
}

namespace {

// Integral of x^k over [0, 1] by the rule, against 1 / (k + 1). Rounding: n products and an n-term
// sum of non-negative terms bounded by the weights (sum 1), plus the nodes' own rounding (each node
// off by eps, moving x^k by k eps), so (2 n + k + 2) eps.
template <class Rule>
void check_moments(const Rule& r, int max_k) {
  const std::size_t n = r.nodes.size();
  for (int k = 0; k <= max_k; ++k) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      s += r.weights.data()[i] * std::pow(r.nodes.data()[i], k);
    }
    CAPTURE(n);
    CAPTURE(k);
    CHECK(std::fabs(s - 1.0 / (k + 1)) <= (2.0 * static_cast<double>(n) + k + 2) * eps);
  }
}

}  // namespace

TEST_CASE("quad: Gauss-Legendre integrates degree 2n - 1 exactly, symmetric, ascending") {
  for (int n = 1; n <= 40; ++n) {
    const auto r = quad::gauss_legendre_1d<double>(n);
    check_moments(r, 2 * n - 1);
    for (int i = 0; i < n; ++i) {
      // (x + 1) / 2 of exact negatives: x_i + x_{n-1-i} = 1 up to one rounding each.
      CHECK(std::fabs(r.nodes.data()[i] + r.nodes.data()[n - 1 - i] - 1.0) <= 2 * eps);
      if (i > 0) {
        CHECK(r.nodes.data()[i] > r.nodes.data()[i - 1]);
      }
    }
  }
}

TEST_CASE("quad: Gauss-Lobatto-Legendre integrates degree 2n - 3, endpoints exact") {
  for (int n = 2; n <= 40; ++n) {
    const auto r = quad::gauss_lobatto_legendre_1d<double>(n);
    CHECK(r.nodes.data()[0] == 0.0);
    CHECK(r.nodes.data()[n - 1] == 1.0);
    check_moments(r, 2 * n - 3);
  }
}

TEST_CASE("quad: Chebyshev rules, trapezoid and the Lobatto nodes") {
  const auto c1 = quad::chebyshev_gauss_1st_kind_1d<double>(5);
  CHECK(c1.nodes.data()[2] == doctest::Approx(0.5));
  CHECK(c1.weights.data()[0] == doctest::Approx(std::numbers::pi / 10.0));
  const auto c2 = quad::chebyshev_gauss_2nd_kind_1d<double>(6);
  for (int i = 1; i < 6; ++i) {
    CHECK(c2.nodes.data()[i] > c2.nodes.data()[i - 1]);
  }
  const auto t = quad::trapezoidal_1d<double>(5);
  CHECK(t.nodes.data()[4] == 1.0);
  CHECK(t.weights.data()[0] == 0.125);
  const auto t1 = quad::trapezoidal_1d<float>(1);
  CHECK(t1.nodes.data()[0] == 0.5F);
  const auto m = quad::modified_chebyshev_nodes_1d<double>(3);
  CHECK(m.data()[0] == 0.0);
  CHECK(m.data()[1] == doctest::Approx(0.5));
  CHECK(m.data()[2] == 1.0);
}

TEST_CASE("quad: n_pts below the minimum is a value_error with the Python message") {
  CHECK_THROWS_WITH_AS(static_cast<void>(quad::gauss_legendre_1d<double>(0)),
                       "n_pts must be at least 1", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(quad::gauss_lobatto_legendre_1d<double>(1)),
                       "n_pts must be at least 2", pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(quad::chebyshev_gauss_2nd_kind_1d<float>(1)),
                  pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(quad::tanh_sinh_1d<double>(0)), pantr::value_error);
}

TEST_CASE("quad: tanh-sinh nodes are interior, weights sum to 1, float keeps fewer nodes") {
  for (const int n : {1, 2, 3, 7, 50, 200, 600}) {
    CAPTURE(n);
    const auto r = quad::tanh_sinh_1d<double>(n);
    double s = 0.0;
    for (std::size_t i = 0; i < r.nodes.size(); ++i) {
      CHECK(r.nodes.data()[i] > 0.0);
      CHECK(r.nodes.data()[i] < 1.0);
      s += r.weights.data()[i];
    }
    // The weights were rescaled onto 2 by their own (rounded) sum, then halved: m roundings.
    CHECK(std::fabs(s - 1.0) <= 2.0 * static_cast<double>(r.nodes.size()) * eps);
    CHECK(r.nodes.size() <= static_cast<std::size_t>(n));
  }
  CHECK(quad::tanh_sinh_1d<float>(200).nodes.size() < quad::tanh_sinh_1d<double>(200).nodes.size());
}

TEST_CASE("quad: PointsLattice validates, shares its rep, and orders all_points") {
  CHECK_THROWS_WITH_AS(quad::PointsLattice<double>({}),
                       "Points lattice must have at least 1 dimension", pantr::value_error);
  CHECK_THROWS_WITH_AS(quad::PointsLattice<double>({{0.0}, {}}),
                       "All points must have at least 1 point", pantr::value_error);
  const quad::PointsLattice<double> lat({{0.0, 1.0}, {10.0, 20.0, 30.0}});
  const auto copy = lat;  // NOLINT(performance-unnecessary-copy-initialization): the point
  CHECK(copy.rep() == lat.rep());
  CHECK(lat.num_points() == 6);
  const auto c = lat.all_points(pantr::Order::C);
  const std::vector<double> c_expected = {0, 10, 0, 20, 0, 30, 1, 10, 1, 20, 1, 30};
  const auto f = lat.all_points(pantr::Order::F);
  const std::vector<double> f_expected = {0, 10, 1, 10, 0, 20, 1, 20, 0, 30, 1, 30};
  for (std::size_t i = 0; i < 12; ++i) {
    CHECK(c.data()[i] == c_expected[i]);
    CHECK(f.data()[i] == f_expected[i]);
  }
  CHECK_THROWS_AS(static_cast<void>(lat.pts(2)), pantr::index_error);
}

TEST_CASE("quad: QuadratureRule validation and the tensor-product Gauss-Legendre rule") {
  const std::vector<double> p = {0.5, 1.5};
  const std::size_t ps[2] = {1, 2};
  const std::vector<double> w = {1.0};
  const std::size_t ws[1] = {1};
  CHECK_THROWS_WITH_AS(quad::QuadratureRule({p, ps}, {w, ws}),
                       "points must lie in the unit cube [0, 1]^ndim.", pantr::value_error);
  const std::size_t bad[1] = {2};
  CHECK_THROWS_WITH_AS(quad::QuadratureRule({p, bad}, {w, ws}),
                       "points must be 2D (num_points, ndim); got shape (2,).", pantr::value_error);
  const std::vector<long long> npts = {2, 3};
  const auto rule = quad::gauss_legendre_quadrature(2, npts);
  CHECK(rule.ndim() == 2);
  CHECK(rule.num_points() == 6);
  double s = 0.0;
  for (std::size_t i = 0; i < 6; ++i) {
    s += rule.weights().data()[i];
  }
  CHECK(std::fabs(s - 1.0) <= 8 * eps);
  const std::vector<long long> zero = {0, 3};
  CHECK_THROWS_WITH_AS(static_cast<void>(quad::gauss_legendre_quadrature(2, zero)),
                       "gauss_legendre_quadrature: every npts entry must be >= 1; got (0, 3).",
                       pantr::value_error);
}
