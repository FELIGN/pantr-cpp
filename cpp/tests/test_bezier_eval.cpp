// Bezier evaluation and partial derivatives. Oracles: closed forms (t, t^2, t^3, u v, u^2 v,
// a rational quarter circle whose points satisfy x^2 + y^2 = 1, 1 / (1 + t), 1 / ((1 + u)(1 + v))),
// the direct Bernstein sum with powers and binomials (a different algorithm from the recurrences
// under test), the forward-difference formula for the derivatives of the Bernstein basis, and the
// agreement of the points, lattice and `out` overloads. Tolerances are derived in comments from
// the unit roundoff of the storage type.
#include <doctest/doctest.h>

#include <pantr/bezier/evaluate.hpp>
#include <pantr/parallel/pool.hpp>
#include <pantr/quad/points_lattice.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pb = pantr::bezier;
using pantr::NdArray;

namespace {

template <class T>
double eps_of() {
  return static_cast<double>(std::numeric_limits<T>::epsilon());
}

template <class T>
pb::Bezier<T> make_bezier(const std::vector<T>& ctrl, std::vector<std::size_t> shape,
                          bool rational = false) {
  return pb::Bezier<T>(std::span<const T>(ctrl), std::move(shape), rational);
}

template <class T>
pantr::mdview<const T, 2> view2(const std::vector<T>& v, std::size_t rows, std::size_t cols) {
  return pantr::mdview<const T, 2>(v.data(), rows, cols);
}

template <class T>
std::vector<T> cast_all(const std::vector<double>& v) {
  std::vector<T> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    out[i] = static_cast<T>(v[i]);
  }
  return out;
}

double binom_ref(int n, int k) {
  if (k < 0 || k > n) {
    return 0.0;
  }
  double c = 1.0;
  for (int i = 1; i <= k; ++i) {
    c = c * (n - k + i) / i;
  }
  return c;
}

// B_{i,n}(t) as the textbook product C(n, i) t^i (1 - t)^(n - i); zero outside 0 <= i <= n.
double bernstein_ref(int n, int i, double t) {
  if (i < 0 || i > n) {
    return 0.0;
  }
  return binom_ref(n, i) * std::pow(t, i) * std::pow(1.0 - t, n - i);
}

// d^k/dt^k B_{i,n} = n! / (n - k)! sum_j (-1)^(k - j) C(k, j) B_{i - j, n - k}: forward
// differences of the lower-degree basis. Zero for k > n.
double bernstein_deriv_ref(int n, int i, int k, double t) {
  if (k > n) {
    return 0.0;
  }
  double scale = 1.0;
  for (int m = 0; m < k; ++m) {
    scale *= (n - m);
  }
  double sum = 0.0;
  for (int j = 0; j <= k; ++j) {
    sum += (((k - j) % 2 == 0) ? 1.0 : -1.0) * binom_ref(k, j) * bernstein_ref(n - k, i - j, t);
  }
  return scale * sum;
}

// A deterministic stream in [-1, 1) (a linear congruential generator; the quality is irrelevant).
struct Lcg {
  std::uint64_t s = 12345;
  double next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(s >> 11) / 4503599627370496.0 - 1.0;  // 2^52
  }
};

// Points in [0, 1] covering both sides of the mirror at 0.5 and both endpoints.
std::vector<double> sample_params() {
  return {0.0, 1e-3, 0.1, 0.3, 0.5, 0.5000001, 0.7, 0.999, 1.0};
}

}  // namespace

TEST_CASE_TEMPLATE("bezier evaluate: 1D polynomials match their closed forms", T, float, double) {
  // Bernstein coefficients of t, t^2 and t^3. All coefficients and basis values are non-negative,
  // so sum_i B_i |c_i| equals the value f(t) and the evaluation error is relative: the basis
  // values carry at most 4 (p + 1) eps (the ratio recurrence, four roundings per step) and the
  // contraction adds p + 1 roundings, so 8 (p + 1) eps f(t) bounds the error.
  const double eps = eps_of<T>();
  struct Case {
    std::vector<double> ctrl;
    int power;
  };
  const std::vector<Case> cases = {
      {{0.0, 1.0}, 1}, {{0.0, 0.0, 1.0}, 2}, {{0.0, 0.0, 0.0, 1.0}, 3}};
  for (const Case& c : cases) {
    const std::size_t p = c.ctrl.size() - 1;
    const pb::Bezier<T> f = make_bezier<T>(cast_all<T>(c.ctrl), {p + 1, 1});
    const std::vector<T> ts = cast_all<T>(sample_params());
    const NdArray<T> v = pb::evaluate<T>(f, view2(ts, ts.size(), 1));
    REQUIRE(v.shape().size() == 2);
    CHECK(v.shape()[0] == ts.size());
    CHECK(v.shape()[1] == 1);
    for (std::size_t j = 0; j < ts.size(); ++j) {
      const double t = static_cast<double>(ts[j]);
      const double expected = std::pow(t, c.power);
      CAPTURE(c.power);
      CAPTURE(t);
      CHECK(std::fabs(static_cast<double>(v.data()[j]) - expected) <=
            8.0 * static_cast<double>(p + 1) * eps * expected);
    }
  }
}

TEST_CASE_TEMPLATE("bezier evaluate: a vector-valued 1D curve and the direct Bernstein sum", T,
                   float, double) {
  // (x, y) = (t, t^2) has coefficients x: (0, 1/2, 1), y: (0, 0, 1); then a random cubic
  // checked against sum_i C(3, i) t^i (1 - t)^(3 - i) c_i, whose error bound 8 (p + 1) eps
  // sum_i B_i |c_i| is the one above (now with mixed signs, so the sum of absolute terms).
  const double eps = eps_of<T>();
  const pb::Bezier<T> curve = make_bezier<T>(cast_all<T>({0.0, 0.0, 0.5, 0.0, 1.0, 1.0}), {3, 2});
  const std::vector<T> ts = cast_all<T>(sample_params());
  const NdArray<T> v = pb::evaluate<T>(curve, view2(ts, ts.size(), 1));
  CHECK(v.shape()[1] == 2);
  for (std::size_t j = 0; j < ts.size(); ++j) {
    const double t = static_cast<double>(ts[j]);
    CHECK(std::fabs(static_cast<double>(v.data()[2 * j]) - t) <= 24.0 * eps * t);
    CHECK(std::fabs(static_cast<double>(v.data()[2 * j + 1]) - t * t) <= 24.0 * eps * t * t);
  }
  Lcg rng;
  std::vector<T> ctrl(4 * 3);
  for (T& c : ctrl) {
    c = static_cast<T>(rng.next());
  }
  const pb::Bezier<T> cubic = make_bezier<T>(ctrl, {4, 3});
  const NdArray<T> w = pb::evaluate<T>(cubic, view2(ts, ts.size(), 1));
  for (std::size_t j = 0; j < ts.size(); ++j) {
    const double t = static_cast<double>(ts[j]);
    for (std::size_t r = 0; r < 3; ++r) {
      double sum = 0.0;
      double abs_sum = 0.0;
      for (std::size_t i = 0; i <= 3; ++i) {
        const double term =
            bernstein_ref(3, static_cast<int>(i), t) * static_cast<double>(ctrl[i * 3 + r]);
        sum += term;
        abs_sum += std::fabs(term);
      }
      CAPTURE(t);
      CHECK(std::fabs(static_cast<double>(w.data()[j * 3 + r]) - sum) <= 8.0 * 4.0 * eps * abs_sum);
    }
  }
}

TEST_CASE_TEMPLATE("bezier evaluate: surfaces and volumes match their closed forms", T, float,
                   double) {
  // u v: only the corner (1, 1) is nonzero. u^2 v: only [2][1] of a (3, 2) net. u v w: only the
  // corner of a (2, 2, 2) net. (x, y) = (u + v, u v): bilinear nets. The coefficients are
  // non-negative, so the value is sum_i prod_d B_{i_d}(t_d) c_i with no cancellation, and each
  // stage of the contraction adds (p_d + 1) roundings to the 4 (p_d + 1) of the basis values:
  // 8 sum_d (p_d + 1) eps f bounds the error.
  const double eps = eps_of<T>();
  const std::vector<double> grid = {0.0, 0.2, 0.5, 0.75, 1.0};
  std::vector<T> pts2;
  for (const double u : grid) {
    for (const double v : grid) {
      pts2.push_back(static_cast<T>(u));
      pts2.push_back(static_cast<T>(v));
    }
  }
  const std::size_t n2 = grid.size() * grid.size();
  const pb::Bezier<T> uv = make_bezier<T>(cast_all<T>({0.0, 0.0, 0.0, 1.0}), {2, 2, 1});
  const pb::Bezier<T> u2v = make_bezier<T>(cast_all<T>({0.0, 0.0, 0.0, 0.0, 0.0, 1.0}), {3, 2, 1});
  const pb::Bezier<T> sum_prod =
      make_bezier<T>(cast_all<T>({0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0}), {2, 2, 2});
  const NdArray<T> a = pb::evaluate<T>(uv, view2(pts2, n2, 2));
  const NdArray<T> b = pb::evaluate<T>(u2v, view2(pts2, n2, 2));
  const NdArray<T> c = pb::evaluate<T>(sum_prod, view2(pts2, n2, 2));
  for (std::size_t j = 0; j < n2; ++j) {
    const double u = static_cast<double>(pts2[2 * j]);
    const double v = static_cast<double>(pts2[2 * j + 1]);
    CAPTURE(u);
    CAPTURE(v);
    CHECK(std::fabs(static_cast<double>(a.data()[j]) - u * v) <= 8.0 * 4.0 * eps * u * v);
    CHECK(std::fabs(static_cast<double>(b.data()[j]) - u * u * v) <= 8.0 * 5.0 * eps * u * u * v);
    CHECK(std::fabs(static_cast<double>(c.data()[2 * j]) - (u + v)) <= 8.0 * 4.0 * eps * (u + v));
    CHECK(std::fabs(static_cast<double>(c.data()[2 * j + 1]) - u * v) <= 8.0 * 4.0 * eps * u * v);
  }
  std::vector<T> pts3;
  for (const double u : grid) {
    pts3.push_back(static_cast<T>(u));
    pts3.push_back(static_cast<T>(1.0 - u));
    pts3.push_back(static_cast<T>(0.5 * u + 0.25));
  }
  std::vector<T> corner(8, static_cast<T>(0));
  corner[7] = static_cast<T>(1);
  const pb::Bezier<T> uvw = make_bezier<T>(corner, {2, 2, 2, 1});
  const NdArray<T> d = pb::evaluate<T>(uvw, view2(pts3, grid.size(), 3));
  for (std::size_t j = 0; j < grid.size(); ++j) {
    const double f = static_cast<double>(pts3[3 * j]) * static_cast<double>(pts3[3 * j + 1]) *
                     static_cast<double>(pts3[3 * j + 2]);
    CHECK(std::fabs(static_cast<double>(d.data()[j]) - f) <= 8.0 * 6.0 * eps * f);
  }
}

TEST_CASE_TEMPLATE("bezier evaluate: a rational quarter circle stays on the unit circle", T, float,
                   double) {
  // Weights (1, 1/sqrt 2, 1) and points (1, 0), (1, 1), (0, 1), stored homogeneously (w x, w y, w).
  // Every point of the conic satisfies x^2 + y^2 = 1. Numerator and weight are sums of
  // non-negative terms, so each carries at most 8 (p + 1) eps relative error and the quotient one
  // more eps; x and y then have relative error at most delta = (16 (p + 1) + 1) eps, and
  // |x^2 + y^2 - 1| <= 2 max(x^2, y^2) ... <= 2 delta (x^2 + y^2) to first order, bounded by 4
  // delta with margin for the weight's own rounding to T (also below eps).
  const double eps = eps_of<T>();
  const double w = 1.0 / std::sqrt(2.0);
  const std::vector<T> ctrl = cast_all<T>({1.0, 0.0, 1.0, w, w, w, 0.0, 1.0, 1.0});
  const pb::Bezier<T> arc = make_bezier<T>(ctrl, {3, 3}, true);
  CHECK(arc.rank() == 2);
  const double tol = 4.0 * (16.0 * 3.0 + 1.0) * eps;
  const std::vector<T> ts = cast_all<T>(sample_params());
  const NdArray<T> v = pb::evaluate<T>(arc, view2(ts, ts.size(), 1));
  CHECK(v.shape()[1] == 2);
  for (std::size_t j = 0; j < ts.size(); ++j) {
    const double x = static_cast<double>(v.data()[2 * j]);
    const double y = static_cast<double>(v.data()[2 * j + 1]);
    CAPTURE(j);
    CHECK(std::fabs(x * x + y * y - 1.0) <= tol);
  }
  // At t = 1/2 the point is (1/sqrt 2, 1/sqrt 2): x_h = y_h = 1/4 + w/2, w_h = 1/2 + w/2.
  const std::vector<T> half = {static_cast<T>(0.5)};
  const NdArray<T> m = pb::evaluate<T>(arc, view2(half, 1, 1));
  CHECK(std::fabs(static_cast<double>(m.data()[0]) - w) <= tol);
  CHECK(std::fabs(static_cast<double>(m.data()[1]) - w) <= tol);
}

TEST_CASE_TEMPLATE("bezier evaluate: the lattice overload equals the points overload", T, float,
                   double) {
  // The same chain of contractions runs in both, so the results agree to the rounding of the
  // contraction order alone: at most (n_0 + n_1) eps sum |terms| each, with sum |terms| <= max|c|
  // (a partition of unity on [0, 1]); the bound below is twice that sum per side.
  const double eps = eps_of<T>();
  Lcg rng;
  std::vector<T> ctrl(4 * 3 * 2);
  for (T& c : ctrl) {
    c = static_cast<T>(rng.next());
  }
  const pb::Bezier<T> surf = make_bezier<T>(ctrl, {4, 3, 2});
  const std::vector<T> us = cast_all<T>({0.0, 0.15, 0.5, 0.8, 1.0});
  const std::vector<T> vs = cast_all<T>({0.05, 0.4, 0.6});
  const pantr::quad::PointsLattice<T> lattice({us, vs});
  const NdArray<T> grid = pb::evaluate<T>(surf, lattice);
  REQUIRE(grid.shape().size() == 3);
  CHECK(grid.shape()[0] == 5);
  CHECK(grid.shape()[1] == 3);
  CHECK(grid.shape()[2] == 2);
  const NdArray<T> all = lattice.all_points(pantr::Order::C);
  const NdArray<T> pointwise = pb::evaluate<T>(surf, all.template view<2>());
  REQUIRE(pointwise.size() == grid.size());
  for (std::size_t i = 0; i < grid.size(); ++i) {
    CAPTURE(i);
    CHECK(std::fabs(static_cast<double>(grid.data()[i]) -
                    static_cast<double>(pointwise.data()[i])) <= 4.0 * (4.0 + 3.0) * eps);
  }
  // The `out` overloads write the same values into a caller's array.
  NdArray<T> out1({all.shape()[0], 2});
  pb::evaluate<T>(surf, all.template view<2>(), out1.template view<2>());
  NdArray<T> out2({lattice.num_points(), 2});
  pb::evaluate<T>(surf, lattice, out2.template view<2>());
  for (std::size_t i = 0; i < grid.size(); ++i) {
    CHECK(out1.data()[i] == pointwise.data()[i]);
    CHECK(out2.data()[i] == grid.data()[i]);
  }
}

TEST_CASE_TEMPLATE("bezier evaluate: the 1D lattice, the out overloads and no points", T, float,
                   double) {
  const pb::Bezier<T> f = make_bezier<T>(cast_all<T>({0.0, 0.0, 1.0}), {3, 1});
  const std::vector<T> ts = cast_all<T>({0.25, 0.5, 0.75});
  const pantr::quad::PointsLattice<T> lattice({ts});
  const NdArray<T> a = pb::evaluate<T>(f, lattice);
  const NdArray<T> b = pb::evaluate<T>(f, view2(ts, 3, 1));
  REQUIRE(a.size() == 3);
  for (std::size_t i = 0; i < 3; ++i) {
    CHECK(a.data()[i] == b.data()[i]);
  }
  const std::vector<T> none;
  const NdArray<T> e = pb::evaluate<T>(f, view2(none, 0, 1));
  CHECK(e.shape()[0] == 0);
  CHECK(e.shape()[1] == 1);
}

TEST_CASE_TEMPLATE("bezier evaluate: every message, in the order the Python checks them", T, float,
                   double) {
  const pb::Bezier<T> curve = make_bezier<T>(cast_all<T>({0.0, 0.0, 1.0}), {3, 1});
  const pb::Bezier<T> surf = make_bezier<T>(cast_all<T>({0.0, 0.0, 0.0, 1.0}), {2, 2, 1});
  const std::vector<T> two_cols = cast_all<T>({0.1, 0.2, 0.3, 0.4});
  const std::vector<T> three_cols = cast_all<T>({0.1, 0.2, 0.3});
  const auto bad_1d = view2(two_cols, 2, 2);
  const auto bad_2d = view2(three_cols, 1, 3);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::evaluate<T>(curve, bad_1d)),
                       "Points must be a 1D array for a 1D Bézier.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::evaluate<T>(surf, bad_2d)),
                       "pts must be a 2D array with 2 columns.", pantr::value_error);
  const std::vector<T> ts = cast_all<T>({0.25, 0.5});
  const pantr::quad::PointsLattice<T> lat1({ts});
  const pantr::quad::PointsLattice<T> lat2({ts, ts});
  const pantr::quad::PointsLattice<T> lat3({ts, ts, ts});
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::evaluate<T>(curve, lat2)),
                       "PointsLattice must be 1D for a 1D Bézier.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::evaluate<T>(surf, lat3)),
                       "PointsLattice dim (3) must match Bézier dim (2).", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::evaluate<T>(surf, lat1)),
                       "PointsLattice dim (1) must match Bézier dim (2).", pantr::value_error);
  // `out` extents: both shapes spelled as Python tuples.
  NdArray<T> wrong({3, 1});
  CHECK_THROWS_WITH_AS(pb::evaluate<T>(curve, view2(ts, 2, 1), wrong.template view<2>()),
                       "Output array has shape (3, 1), but expected shape (2, 1)",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(pb::evaluate<T>(surf, lat2, wrong.template view<2>()),
                       "Output array has shape (3, 1), but expected shape (4, 1)",
                       pantr::value_error);
}

namespace {

// d^{k_0}/du^{k_0} d^{k_1}/dv^{k_1} of a tensor-product Bezier of degrees (p_0, p_1) and rank 1
// (c in row-major (p_0 + 1, p_1 + 1)), by the forward-difference formula of the basis. Also the
// sum of the absolute terms.
struct RefValue {
  double value;
  double abs_sum;
};

RefValue deriv_ref_2d(const std::vector<double>& c, int p0, int p1, int k0, int k1, double u,
                      double v) {
  RefValue r{0.0, 0.0};
  for (int i = 0; i <= p0; ++i) {
    for (int j = 0; j <= p1; ++j) {
      const double term = bernstein_deriv_ref(p0, i, k0, u) * bernstein_deriv_ref(p1, j, k1, v) *
                          c[static_cast<std::size_t>(i * (p1 + 1) + j)];
      r.value += term;
      r.abs_sum += std::fabs(term);
    }
  }
  return r;
}

}  // namespace

TEST_CASE_TEMPLATE(
    "bezier derivatives: 1D polynomials match the closed forms, zero past the degree", T, float,
    double) {
  // t^3: 3 t^2, 6 t, 6, 0, 0.  The derivative table is built by the O(p^2) recurrence in T; its
  // entries before the scaling by p! / (p - k)! are sums of at most 2^k products of numbers in
  // [0, 1], so each entry carries at most (p + 1)^2 eps 2^k, and the scaled, contracted value
  // (p + 1) terms of that, times p^k: tol = 8 (p + 1)^3 (2 p)^k eps max|c|, with max|c| = 1.
  const double eps = eps_of<T>();
  const pb::Bezier<T> cubic = make_bezier<T>(cast_all<T>({0.0, 0.0, 0.0, 1.0}), {4, 1});
  const std::vector<T> ts = cast_all<T>(sample_params());
  for (long long k = 0; k <= 4; ++k) {
    const std::vector<long long> orders = {k};
    const NdArray<T> v = pb::evaluate_derivatives<T>(cubic, view2(ts, ts.size(), 1), orders);
    CHECK(v.shape()[0] == ts.size());
    CHECK(v.shape()[1] == 1);
    for (std::size_t j = 0; j < ts.size(); ++j) {
      const double t = static_cast<double>(ts[j]);
      const double expected = k == 0   ? t * t * t
                              : k == 1 ? 3.0 * t * t
                              : k == 2 ? 6.0 * t
                              : k == 3 ? 6.0
                                       : 0.0;
      const double tol = 8.0 * 64.0 * std::pow(6.0, static_cast<double>(k)) * eps;
      CAPTURE(k);
      CAPTURE(t);
      CHECK(std::fabs(static_cast<double>(v.data()[j]) - expected) <= tol);
      if (k == 4) {
        CHECK(v.data()[j] == static_cast<T>(0));  // above the degree the table row is exactly zero
      }
    }
  }
}

TEST_CASE_TEMPLATE("bezier derivatives: 2D partial derivatives against the difference formula", T,
                   float, double) {
  // A random (3, 2)-degree net, every order pair up to (4, 3) (past both degrees). The reference
  // is the forward-difference formula of the Bernstein basis in double; its terms sum to S =
  // sum|terms|, and the evaluation in T errs by at most 8 (sum_d (p_d + 1)^2) eps prod_d (2
  // p_d)^{k_d} sum|c| (the 1D bound of the previous case per direction, multiplied).
  const double eps = eps_of<T>();
  Lcg rng;
  const int p0 = 3;
  const int p1 = 2;
  std::vector<double> c;
  double sum_abs_c = 0.0;
  for (int i = 0; i < (p0 + 1) * (p1 + 1); ++i) {
    c.push_back(static_cast<double>(static_cast<T>(rng.next())));
    sum_abs_c += std::fabs(c.back());
  }
  const pb::Bezier<T> surf = make_bezier<T>(cast_all<T>(c), {4, 3, 1});
  const std::vector<double> grid = {0.0, 0.3, 0.5, 0.9, 1.0};
  std::vector<T> pts;
  for (const double u : grid) {
    for (const double v : grid) {
      pts.push_back(static_cast<T>(u));
      pts.push_back(static_cast<T>(v));
    }
  }
  const std::size_t n = grid.size() * grid.size();
  for (long long k0 = 0; k0 <= 4; ++k0) {
    for (long long k1 = 0; k1 <= 3; ++k1) {
      const std::vector<long long> orders = {k0, k1};
      const NdArray<T> v = pb::evaluate_derivatives<T>(surf, view2(pts, n, 2), orders);
      const double scale =
          std::pow(2.0 * p0, static_cast<double>(k0)) * std::pow(2.0 * p1, static_cast<double>(k1));
      const double tol = 8.0 * (16.0 + 9.0) * eps * scale * sum_abs_c;
      for (std::size_t j = 0; j < n; ++j) {
        const double u = static_cast<double>(pts[2 * j]);
        const double w = static_cast<double>(pts[2 * j + 1]);
        const RefValue ref =
            deriv_ref_2d(c, p0, p1, static_cast<int>(k0), static_cast<int>(k1), u, w);
        CAPTURE(k0);
        CAPTURE(k1);
        CAPTURE(u);
        CAPTURE(w);
        CHECK(std::fabs(static_cast<double>(v.data()[j]) - ref.value) <= tol);
      }
    }
  }
  // d^2/du dv of u^2 v is 2 u, and the lattice overload agrees.
  const pb::Bezier<T> u2v = make_bezier<T>(cast_all<T>({0.0, 0.0, 0.0, 0.0, 0.0, 1.0}), {3, 2, 1});
  const std::vector<long long> mixed = {1, 1};
  const std::vector<T> us = cast_all<T>({0.0, 0.25, 0.5, 1.0});
  const pantr::quad::PointsLattice<T> lattice({us, us});
  const NdArray<T> m = pb::evaluate_derivatives<T>(u2v, lattice, mixed);
  REQUIRE(m.size() == 16);
  for (std::size_t a = 0; a < 4; ++a) {
    for (std::size_t b = 0; b < 4; ++b) {
      const double u = static_cast<double>(us[a]);
      CHECK(std::fabs(static_cast<double>(m.data()[a * 4 + b]) - 2.0 * u) <=
            8.0 * 25.0 * 4.0 * eps);
    }
  }
}

TEST_CASE_TEMPLATE("bezier derivatives: a rational curve against 1 / (1 + t)", T, float, double) {
  // N = 1, W = 1 + t: homogeneous control points (1, 1), (1, 2). f = 1 / (1 + t) and
  // f^(k) = (-1)^k k! / (1 + t)^(k + 1). The homogeneous derivatives are (1, 1 + t), (0, 1), (0, 0)
  // and the quotient rule adds, per order, a product, a product and a quotient with terms of one
  // sign (f^(k-1) alternates, the recurrence subtracts it): the relative error of f^(k) is at most
  // 8 (k + 1) eps.
  const double eps = eps_of<T>();
  const pb::Bezier<T> f = make_bezier<T>(cast_all<T>({1.0, 1.0, 1.0, 2.0}), {2, 2}, true);
  const std::vector<T> ts = cast_all<T>(sample_params());
  double fact = 1.0;
  for (long long k = 0; k <= 4; ++k) {
    if (k > 0) {
      fact *= static_cast<double>(k);
    }
    const std::vector<long long> orders = {k};
    const NdArray<T> v = pb::evaluate_derivatives<T>(f, view2(ts, ts.size(), 1), orders);
    CHECK(v.shape()[1] == 1);
    for (std::size_t j = 0; j < ts.size(); ++j) {
      const double t = static_cast<double>(ts[j]);
      const double expected =
          ((k % 2 == 0) ? 1.0 : -1.0) * fact / std::pow(1.0 + t, static_cast<double>(k + 1));
      CAPTURE(k);
      CAPTURE(t);
      CHECK(std::fabs(static_cast<double>(v.data()[j]) - expected) <=
            8.0 * static_cast<double>(k + 1) * eps * std::fabs(expected));
    }
  }
}

TEST_CASE_TEMPLATE("bezier derivatives: a rational surface against 1 / ((1 + u)(1 + v))", T, float,
                   double) {
  // N = 1, W = (1 + u)(1 + v): weights [[1, 2], [2, 4]]. With f_{ab} = (-1)^(a + b) a! b! /
  // ((1 + u)^(a + 1) (1 + v)^(b + 1)) and W_{ab} = (1 + u)^(1 - a) (1 + v)^(1 - b) for a, b in
  // {0, 1} (zero otherwise), the quotient rule gives f_{ab} = (N_{ab} - sum C(a, i) C(b, j) W_{ij}
  // f_{a-i, b-j}) / W. The terms of the sum can have different signs, so the error is bounded
  // by the sum of their magnitudes: each of the at most (a + 1)(b + 1) terms and the quotient
  // carries 4 roundings, and the weights and the homogeneous derivatives carry 4 more each,
  // tol = 8 ((a + 1)(b + 1) + 2) eps (sum|terms| + |f_ab|) / W.
  const double eps = eps_of<T>();
  const pb::Bezier<T> f =
      make_bezier<T>(cast_all<T>({1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 4.0}), {2, 2, 2}, true);
  const std::vector<double> grid = {0.0, 0.25, 0.5, 0.8, 1.0};
  std::vector<T> pts;
  for (const double u : grid) {
    for (const double v : grid) {
      pts.push_back(static_cast<T>(u));
      pts.push_back(static_cast<T>(v));
    }
  }
  const std::size_t n = grid.size() * grid.size();
  auto f_ab = [](int a, int b, double u, double v) {
    double fa = 1.0;
    for (int m = 2; m <= a; ++m) {
      fa *= m;
    }
    double fb = 1.0;
    for (int m = 2; m <= b; ++m) {
      fb *= m;
    }
    return (((a + b) % 2 == 0) ? 1.0 : -1.0) * fa * fb /
           (std::pow(1.0 + u, a + 1) * std::pow(1.0 + v, b + 1));
  };
  auto w_ab = [](int a, int b, double u, double v) {
    return (a > 1 || b > 1) ? 0.0 : std::pow(1.0 + u, 1 - a) * std::pow(1.0 + v, 1 - b);
  };
  for (int a = 0; a <= 2; ++a) {
    for (int b = 0; b <= 2; ++b) {
      const std::vector<long long> orders = {a, b};
      const NdArray<T> out = pb::evaluate_derivatives<T>(f, view2(pts, n, 2), orders);
      for (std::size_t j = 0; j < n; ++j) {
        const double u = static_cast<double>(pts[2 * j]);
        const double v = static_cast<double>(pts[2 * j + 1]);
        double terms = 0.0;
        for (int i = 0; i <= a; ++i) {
          for (int k = 0; k <= b; ++k) {
            if (i == 0 && k == 0) {
              continue;
            }
            terms += binom_ref(a, i) * binom_ref(b, k) * std::fabs(w_ab(i, k, u, v)) *
                     std::fabs(f_ab(a - i, b - k, u, v));
          }
        }
        const double expected = f_ab(a, b, u, v);
        const double w = w_ab(0, 0, u, v);
        const double tol = 8.0 * (static_cast<double>((a + 1) * (b + 1)) + 2.0) * eps *
                           (terms + std::fabs(expected) * w) / w;
        CAPTURE(a);
        CAPTURE(b);
        CAPTURE(u);
        CAPTURE(v);
        CHECK(std::fabs(static_cast<double>(out.data()[j]) - expected) <= tol);
      }
    }
  }
  // The lattice overload agrees with the points overload on the same points.
  const std::vector<long long> orders = {1, 1};
  const std::vector<T> gs = cast_all<T>(grid);
  const pantr::quad::PointsLattice<T> lattice({gs, gs});
  const NdArray<T> on_lattice = pb::evaluate_derivatives<T>(f, lattice, orders);
  const NdArray<T> on_points = pb::evaluate_derivatives<T>(
      f, lattice.all_points(pantr::Order::C).template view<2>(), orders);
  REQUIRE(on_lattice.size() == on_points.size());
  for (std::size_t i = 0; i < on_points.size(); ++i) {
    // The same sums in two orders (the lattice contracts the last direction first, the points
    // the first): each is n eps sum|terms| with n = (p + 1)^2 terms of magnitude <= 1 here, so the
    // two differ by at most twice that; 64 eps covers degree (3, 3).
    CHECK(std::fabs(static_cast<double>(on_lattice.data()[i]) -
                    static_cast<double>(on_points.data()[i])) <= 64.0 * eps);
  }
}

TEST_CASE_TEMPLATE("bezier derivatives: the out overloads agree with the returning ones", T, float,
                   double) {
  const pb::Bezier<T> curve = make_bezier<T>(cast_all<T>({0.0, 1.0, 0.0, 2.0, 1.0, 3.0}), {3, 2});
  const std::vector<T> ts = cast_all<T>({0.1, 0.6, 0.9});
  const std::vector<long long> orders = {1};
  const NdArray<T> a = pb::evaluate_derivatives<T>(curve, view2(ts, 3, 1), orders);
  NdArray<T> out({3, 2});
  pb::evaluate_derivatives<T>(curve, view2(ts, 3, 1), orders, out.template view<2>());
  for (std::size_t i = 0; i < a.size(); ++i) {
    CHECK(out.data()[i] == a.data()[i]);
  }
  const pantr::quad::PointsLattice<T> lattice({ts});
  NdArray<T> out2({3, 2});
  pb::evaluate_derivatives<T>(curve, lattice, orders, out2.template view<2>());
  for (std::size_t i = 0; i < a.size(); ++i) {
    CHECK(out2.data()[i] == a.data()[i]);
  }
  NdArray<T> wrong({2, 2});
  CHECK_THROWS_WITH_AS(
      pb::evaluate_derivatives<T>(curve, view2(ts, 3, 1), orders, wrong.template view<2>()),
      "Output array has shape (2, 2), but expected shape (3, 2)", pantr::value_error);
}

TEST_CASE_TEMPLATE("bezier derivatives: the order checks come first, with Python's messages", T,
                   float, double) {
  const pb::Bezier<T> curve = make_bezier<T>(cast_all<T>({0.0, 0.0, 1.0}), {3, 1});
  const pb::Bezier<T> surf = make_bezier<T>(cast_all<T>({0.0, 0.0, 0.0, 1.0}), {2, 2, 1});
  const std::vector<T> two_cols = cast_all<T>({0.1, 0.2, 0.3, 0.4});
  const std::vector<T> ts = cast_all<T>({0.25, 0.5});
  const std::vector<long long> one = {1};
  const std::vector<long long> two = {1, 1};
  const std::vector<long long> negative = {-1, 0};
  const std::vector<long long> huge = {2147483647LL};
  const pantr::quad::PointsLattice<T> lat2({ts, ts});
  const auto bad_1d = view2(two_cols, 2, 2);
  // Wrong length, with points that are also wrong: the length is reported.
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::evaluate_derivatives<T>(surf, view2(two_cols, 1, 4), one)),
      "len(orders) (1) must match dim (2).", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::evaluate_derivatives<T>(curve, bad_1d, two)),
                       "len(orders) (2) must match dim (1).", pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::evaluate_derivatives<T>(surf, view2(two_cols, 2, 2), negative)),
      "All derivative orders must be non-negative.", pantr::value_error);
  // Valid orders, then the point checks.
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::evaluate_derivatives<T>(curve, bad_1d, one)),
                       "Points must be a 1D array for a 1D Bézier.", pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::evaluate_derivatives<T>(surf, view2(two_cols, 1, 4), two)),
      "pts must be a 2D array with 2 columns.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::evaluate_derivatives<T>(curve, lat2, one)),
                       "PointsLattice must be 1D for a 1D Bézier.", pantr::value_error);
  const pantr::quad::PointsLattice<T> lat1({ts});
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::evaluate_derivatives<T>(surf, lat1, two)),
                       "PointsLattice dim (1) must match Bézier dim (2).", pantr::value_error);
  // An order that no table can hold is refused, not narrowed.
  CHECK_THROWS_AS(static_cast<void>(pb::evaluate_derivatives<T>(curve, view2(ts, 2, 1), huge)),
                  pantr::capacity_error);
}

TEST_CASE("bezier evaluate: the parallel result equals the serial one within the rounding bound") {
  // Every point (every output row of the lattice) is computed by the same instruction sequence
  // whatever the chunking, so the results are expected to coincide; the check is the rounding
  // bound of one evaluation, 8 (p + 1) eps max|c| per direction chain (a partition of unity
  // on [0, 1]), not equality of bits. Sizes are above `parallel_min_ops` (design/threading.md).
  using T = double;
  const double eps = eps_of<T>();
  Lcg rng;
  std::vector<T> ctrl1(6 * 3);
  for (T& c : ctrl1) {
    c = rng.next();
  }
  std::vector<T> ctrl2(4 * 4 * 2);
  for (T& c : ctrl2) {
    c = rng.next();
  }
  const pb::Bezier<T> curve = make_bezier<T>(ctrl1, {6, 3});
  const pb::Bezier<T> surf = make_bezier<T>(ctrl2, {4, 4, 2});
  const std::size_t n = 20000;
  std::vector<T> ts(n);
  for (T& t : ts) {
    t = 0.5 * (rng.next() + 1.0);
  }
  std::vector<T> ps(2 * 4000);
  for (T& t : ps) {
    t = 0.5 * (rng.next() + 1.0);
  }
  std::vector<T> axis(200);
  for (std::size_t i = 0; i < axis.size(); ++i) {
    axis[i] = static_cast<T>(i) / 199.0;
  }
  const pantr::quad::PointsLattice<T> lattice({axis, axis});
  const std::vector<long long> d1 = {2};
  const std::vector<long long> d2 = {1, 1};
  const std::size_t saved = pantr::parallel::get_num_threads();
  auto run_all = [&]() {
    std::vector<NdArray<T>> r;
    r.push_back(pb::evaluate<T>(curve, view2(ts, n, 1)));
    r.push_back(pb::evaluate<T>(surf, view2(ps, 4000, 2)));
    r.push_back(pb::evaluate<T>(surf, lattice));
    r.push_back(pb::evaluate_derivatives<T>(curve, view2(ts, n, 1), d1));
    r.push_back(pb::evaluate_derivatives<T>(surf, lattice, d2));
    return r;
  };
  pantr::parallel::set_num_threads(1);
  const std::vector<NdArray<T>> serial = run_all();
  pantr::parallel::set_num_threads(static_cast<std::int64_t>(pantr::parallel::max_threads()));
  const std::vector<NdArray<T>> threaded = run_all();
  pantr::parallel::set_num_threads(static_cast<std::int64_t>(saved));
  REQUIRE(serial.size() == threaded.size());
  const double bound[5] = {8.0 * 6.0 * eps, 8.0 * 8.0 * eps, 8.0 * 8.0 * eps,
                           8.0 * 64.0 * 25.0 * eps, 8.0 * 64.0 * 25.0 * eps};
  for (std::size_t k = 0; k < serial.size(); ++k) {
    REQUIRE(serial[k].size() == threaded[k].size());
    double worst = 0.0;
    for (std::size_t i = 0; i < serial[k].size(); ++i) {
      worst = (std::max)(worst, std::fabs(serial[k].data()[i] - threaded[k].data()[i]));
    }
    CAPTURE(k);
    CHECK(worst <= bound[k]);
  }
}

TEST_CASE_TEMPLATE("bezier: the constructor refuses a size that does not match the shape", T, float,
                   double) {
  const std::vector<T> five = cast_all<T>({1.0, 2.0, 3.0, 4.0, 5.0});
  const std::vector<T> three = cast_all<T>({1.0, 2.0, 3.0});
  CHECK_THROWS_WITH_AS(make_bezier<T>(five, {2, 2}),
                       "control points hold 5 values, but the shape (2, 2) needs 4",
                       pantr::value_error);
  CHECK_THROWS_AS(make_bezier<T>(three, {2, 2}), pantr::value_error);
}

TEST_CASE_TEMPLATE("bezier evaluate: a degree-0 curve is its control point everywhere", T, float,
                   double) {
  const std::vector<T> c = cast_all<T>({2.5, -1.0});
  const pb::Bezier<T> f = make_bezier<T>(c, {1, 2});
  const std::vector<T> ts = cast_all<T>({0.0, 0.3, 0.7, 1.0});
  const NdArray<T> v = pb::evaluate<T>(f, view2<T>(ts, ts.size(), 1));
  for (std::size_t j = 0; j < ts.size(); ++j) {
    const T x = v.data()[2 * j];
    const T y = v.data()[2 * j + 1];
    CHECK(x == c[0]);  // copied, not computed
    CHECK(y == c[1]);
  }
}

TEST_CASE_TEMPLATE("bezier derivatives: an order far above the degree is zero, with no table", T,
                   float, double) {
  // A table for this order would need gigabytes; the polynomial result is exactly zero.
  const std::vector<T> c = cast_all<T>({1.0, -2.0, 4.0});
  const pb::Bezier<T> f = make_bezier<T>(c, {3, 1});
  const std::vector<T> ts = cast_all<T>({0.25, 0.75});
  const std::vector<long long> huge = {1000000000};
  const NdArray<T> d = pb::evaluate_derivatives<T>(f, view2<T>(ts, ts.size(), 1), huge);
  for (std::size_t j = 0; j < ts.size(); ++j) {
    const T v = d.data()[j];
    CHECK(v == static_cast<T>(0));
  }
}

TEST_CASE_TEMPLATE("bezier derivatives: above the degree, a non-finite point gives NaN, not zero",
                   T, float, double) {
  const std::vector<T> c = cast_all<T>({1.0, 1.0, 1.0, 1.0});
  const pb::Bezier<T> f = make_bezier<T>(c, {2, 2, 1});
  const T nan = std::numeric_limits<T>::quiet_NaN();
  const T inf = std::numeric_limits<T>::infinity();
  const std::vector<T> pts = {
      nan, static_cast<T>(0.5), static_cast<T>(0.25), static_cast<T>(0.5), static_cast<T>(0.25),
      inf};
  const std::vector<long long> orders = {2, 0};
  const NdArray<T> d = pb::evaluate_derivatives<T>(f, view2<T>(pts, 3, 2), orders);
  const T first = d.data()[0];
  const T second = d.data()[1];
  const T third = d.data()[2];
  CHECK(std::isnan(first));
  CHECK(second == static_cast<T>(0));
  CHECK(std::isnan(third));
  const std::vector<T> u = {nan, static_cast<T>(0.5)};
  const std::vector<T> v = {static_cast<T>(0.5)};
  const pantr::quad::PointsLattice<T> lattice({u, v});
  const NdArray<T> dl = pb::evaluate_derivatives<T>(f, lattice, orders);
  const T l0 = dl.data()[0];
  const T l1 = dl.data()[1];
  CHECK(std::isnan(l0));
  CHECK(l1 == static_cast<T>(0));
}
