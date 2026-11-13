// Bezier degree elevation, product, derivative and the one-direction kernels. Oracles: closed
// forms (t -> (0, 1/2, 1), t^2 -> (0, 0, 1/3, 1), t * t = t^2, d/dt t^2 = 2 t, the rational
// quarter circle on x^2 + y^2 = 1 and its square h_0 + h_1 = 1), preservation of the evaluated
// values (elevation), the pointwise product of evaluations (multiplication), the evaluated
// derivative of the Bezier against the evaluation of its hodograph, and the direct Bernstein sum
// with powers and binomials for the de Casteljau kernels. Tolerances are derived in comments from
// the unit roundoff of the storage type.
#include <doctest/doctest.h>

#include <pantr/bezier/derivative.hpp>
#include <pantr/bezier/elevate.hpp>
#include <pantr/bezier/evaluate.hpp>
#include <pantr/bezier/product.hpp>
#include <pantr/parallel/pool.hpp>

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

// B_{i,n}(t) as the textbook product C(n, i) t^i (1 - t)^(n - i).
double bernstein_ref(std::size_t n, std::size_t i, double t) {
  return binom_ref(static_cast<int>(n), static_cast<int>(i)) * std::pow(t, static_cast<int>(i)) *
         std::pow(1.0 - t, static_cast<int>(n - i));
}

// Column `c` of a 1D control polygon (rows x cols, row-major) evaluated by the direct sum.
template <class T>
double direct_eval(const T* ctrl, std::size_t rows, std::size_t cols, std::size_t c, double t) {
  double s = 0.0;
  for (std::size_t i = 0; i < rows; ++i) {
    s += bernstein_ref(rows - 1, i, t) * static_cast<double>(ctrl[i * cols + c]);
  }
  return s;
}

struct Lcg {
  std::uint64_t s = 987654321;
  double next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(s >> 11) / 4503599627370496.0 - 1.0;  // [-1, 1)
  }
};

template <class T>
std::vector<T> random_vec(Lcg& rng, std::size_t n) {
  std::vector<T> v(n);
  for (T& x : v) {
    x = static_cast<T>(rng.next());
  }
  return v;
}

template <class T>
double max_abs(const std::vector<T>& v) {
  double m = 0.0;
  for (const T x : v) {
    m = (std::max)(m, std::fabs(static_cast<double>(x)));
  }
  return m;
}

std::vector<double> sample_params() { return {0.0, 0.1, 0.3, 0.5, 0.5000001, 0.8, 1.0}; }

template <class T>
NdArray<T> eval1(const pb::Bezier<T>& b, const std::vector<T>& ts) {
  return pb::evaluate<T>(b, view2(ts, ts.size(), 1));
}

}  // namespace

TEST_CASE_TEMPLATE("bezier elevate: t and t^2 elevate to their closed forms", T, float, double) {
  const double eps = eps_of<T>();
  // t = (0, 1) -> (0, 1/2, 1): every product and sum is exact.
  const pb::Bezier<T> lin = make_bezier<T>(cast_all<T>({0.0, 1.0}), {2, 1});
  const std::vector<long long> one = {1};
  const pb::Bezier<T> q = pb::elevate_degree<T>(lin, one);
  REQUIRE(q.dim() == 1);
  CHECK(q.degree(0) == 2);
  CHECK(q.cp_size() == 1);
  CHECK(q.control_points()[0] == static_cast<T>(0));
  CHECK(q.control_points()[1] == static_cast<T>(0.5));
  CHECK(q.control_points()[2] == static_cast<T>(1));
  // t^2 = (0, 0, 1) -> (0, 0, 1/3, 1). The coefficient 1/3 is C(2, 1) C(1, 1) / C(3, 2) formed in
  // double and rounded once to T: relative error at most 3 eps_double + eps_T, within 2 eps_T.
  const pb::Bezier<T> sq = make_bezier<T>(cast_all<T>({0.0, 0.0, 1.0}), {3, 1});
  const pb::Bezier<T> cube = pb::elevate_degree<T>(sq, one);
  CHECK(cube.degree(0) == 3);
  CHECK(cube.control_points()[0] == static_cast<T>(0));
  CHECK(cube.control_points()[1] == static_cast<T>(0));
  CHECK(std::fabs(static_cast<double>(cube.control_points()[2]) - 1.0 / 3.0) <= 2.0 * eps / 3.0);
  CHECK(cube.control_points()[3] == static_cast<T>(1));
}

TEST_CASE_TEMPLATE("bezier elevate: values are preserved, in 1D, 2D and rational", T, float,
                   double) {
  // Elevated and original evaluate to the same values. The error is the sum of three terms in
  // units of eps max|c|: the original's evaluation 8 (p + 1), the elevated's 8 (p + t + 1) (the
  // elevation coefficients are convex weights, so max|Q| <= max|c|), and the elevation's own
  // rounding of Q, at most (p + 4) per coefficient (three roundings of the weight, one rounding
  // per term). The sum is below 24 (p + t + 1) eps max|c|.
  const double eps = eps_of<T>();
  Lcg rng;
  const std::vector<T> ts = cast_all<T>(sample_params());
  for (const long long inc : {1LL, 2LL, 5LL}) {
    const std::size_t p = 3;
    const std::vector<T> ctrl = random_vec<T>(rng, (p + 1) * 2);
    const pb::Bezier<T> f = make_bezier<T>(ctrl, {p + 1, 2});
    const std::vector<long long> incs = {inc};
    const pb::Bezier<T> g = pb::elevate_degree<T>(f, incs);
    CHECK(g.degree(0) == p + static_cast<std::size_t>(inc));
    const NdArray<T> a = eval1(f, ts);
    const NdArray<T> b = eval1(g, ts);
    const double tol =
        24.0 * static_cast<double>(p + static_cast<std::size_t>(inc) + 1) * eps * max_abs(ctrl);
    for (std::size_t i = 0; i < a.size(); ++i) {
      CAPTURE(inc);
      CHECK(std::fabs(static_cast<double>(a.data()[i]) - static_cast<double>(b.data()[i])) <= tol);
    }
  }
  // A (2, 1)-degree surface elevated by (1, 2) in one call, and in two calls, direction by
  // direction.
  const std::vector<T> sctrl = random_vec<T>(rng, 3 * 2 * 3);
  const pb::Bezier<T> s = make_bezier<T>(sctrl, {3, 2, 3});
  const std::vector<long long> both = {1, 2};
  const pb::Bezier<T> s2 = pb::elevate_degree<T>(s, both);
  CHECK(s2.degree(0) == 3);
  CHECK(s2.degree(1) == 3);
  CHECK(s2.shape()[2] == 3);
  const std::vector<long long> first = {1, 0};
  const std::vector<long long> second = {0, 2};
  const pb::Bezier<T> s3 = pb::elevate_degree<T>(pb::elevate_degree<T>(s, first), second);
  std::vector<T> pts;
  for (const double u : sample_params()) {
    pts.push_back(static_cast<T>(u));
    pts.push_back(static_cast<T>(1.0 - u));
  }
  const std::size_t np = pts.size() / 2;
  const NdArray<T> va = pb::evaluate<T>(s, view2(pts, np, 2));
  const NdArray<T> vb = pb::evaluate<T>(s2, view2(pts, np, 2));
  const NdArray<T> vc = pb::evaluate<T>(s3, view2(pts, np, 2));
  const double stol = 24.0 * (4.0 + 4.0 + 2.0) * eps * max_abs(sctrl);
  for (std::size_t i = 0; i < va.size(); ++i) {
    CHECK(std::fabs(static_cast<double>(va.data()[i]) - static_cast<double>(vb.data()[i])) <= stol);
    CHECK(std::fabs(static_cast<double>(vb.data()[i]) - static_cast<double>(vc.data()[i])) <= stol);
  }
  // The rational quarter circle elevated by 2 stays on the unit circle. Its coordinates are
  // positive, the weights too, so the relative errors of the section above carry over: the bound
  // is that of the circle test of the evaluation suite, 4 (16 (p + t + 1) + 1) eps, doubled for
  // the elevated coefficients.
  const double w = 1.0 / std::sqrt(2.0);
  const pb::Bezier<T> arc =
      make_bezier<T>(cast_all<T>({1.0, 0.0, 1.0, w, w, w, 0.0, 1.0, 1.0}), {3, 3}, true);
  const std::vector<long long> two = {2};
  const pb::Bezier<T> arc4 = pb::elevate_degree<T>(arc, two);
  CHECK(arc4.rational());
  CHECK(arc4.rank() == 2);
  CHECK(arc4.degree(0) == 4);
  const NdArray<T> v = eval1(arc4, ts);
  for (std::size_t j = 0; j < ts.size(); ++j) {
    const double x = static_cast<double>(v.data()[2 * j]);
    const double y = static_cast<double>(v.data()[2 * j + 1]);
    CHECK(std::fabs(x * x + y * y - 1.0) <= 2.0 * 4.0 * (16.0 * 5.0 + 1.0) * eps);
  }
}

TEST_CASE_TEMPLATE("bezier elevate: the degree-61 end of the envelope is accepted", T, float,
                   double) {
  // Degree 60 to 61: C(61, k) is the largest coefficient the exact kernel supports. Values are
  // preserved to 24 (p + t + 1) eps max|c| as above.
  const double eps = eps_of<T>();
  Lcg rng;
  const std::vector<T> ctrl = random_vec<T>(rng, 61);
  const pb::Bezier<T> f = make_bezier<T>(ctrl, {61, 1});
  const std::vector<long long> one = {1};
  const pb::Bezier<T> g = pb::elevate_degree<T>(f, one);
  CHECK(g.degree(0) == 61);
  const std::vector<T> ts = cast_all<T>(sample_params());
  const NdArray<T> a = eval1(f, ts);
  const NdArray<T> b = eval1(g, ts);
  const double tol = 24.0 * 62.0 * eps * max_abs(ctrl);
  for (std::size_t i = 0; i < a.size(); ++i) {
    CHECK(std::fabs(static_cast<double>(a.data()[i]) - static_cast<double>(b.data()[i])) <= tol);
  }
}

TEST_CASE_TEMPLATE("bezier elevate: every message, in the order the Python checks them", T, float,
                   double) {
  const pb::Bezier<T> curve = make_bezier<T>(cast_all<T>({0.0, 0.0, 1.0}), {3, 1});
  const std::vector<long long> three = {1, 1, 1};
  const std::vector<long long> negative = {-1};
  const std::vector<long long> zero = {0};
  const std::vector<long long> to_60 = {59};
  const std::vector<long long> to_2 = {1};
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::elevate_degree<T>(curve, three)),
                       "Number of degree increments (3) must match dimension (1).",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::elevate_degree<T>(curve, negative)),
                       "Degree increments must be non-negative.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::elevate_degree<T>(curve, zero)),
                       "At least one degree increment must be positive.", pantr::value_error);
  // Degree 2 + 60 = 62, past the envelope; the message names the elevated degree and direction.
  const std::vector<long long> over = {60};
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::elevate_degree<T>(curve, over)),
      "Degree elevation to degree 62 in direction 0 needs binomial coefficients up to C(62, k), "
      "beyond the largest upper index 61 that pantr's exact-integer binomial kernel can compute "
      "without an int64 overflow. Past that the coefficients wrap silently and the result is "
      "corrupted rather than merely inaccurate.",
      pantr::value_error);
  CHECK_NOTHROW(static_cast<void>(pb::elevate_degree<T>(curve, to_60)));
  CHECK_NOTHROW(static_cast<void>(pb::elevate_degree<T>(curve, to_2)));
  // An increment so large that degree + increment would overflow still gives the envelope error.
  const std::vector<long long> huge = {(std::numeric_limits<long long>::max)()};
  CHECK_THROWS_AS(static_cast<void>(pb::elevate_degree<T>(curve, huge)), pantr::value_error);
  // The direction is named: a surface refused in direction 1.
  const pb::Bezier<T> surf = make_bezier<T>(cast_all<T>({0.0, 0.0, 0.0, 1.0}), {2, 2, 1});
  const std::vector<long long> second = {0, 61};
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::elevate_degree<T>(surf, second)),
                       "Degree elevation to degree 62 in direction 1 needs binomial coefficients "
                       "up to C(62, k), beyond the largest upper index 61 that pantr's "
                       "exact-integer binomial kernel can compute without an int64 overflow. Past "
                       "that the coefficients wrap silently and the result is corrupted rather "
                       "than merely inaccurate.",
                       pantr::value_error);
}

TEST_CASE_TEMPLATE("bezier elevate: elevate_degree_1d on columns, and its checks", T, float,
                   double) {
  const double eps = eps_of<T>();
  // Two independent columns, t^2 and t (as degree 2): (0, 0, 1) and (0, 1/2, 1).
  const std::vector<T> ctrl = cast_all<T>({0.0, 0.0, 0.0, 0.5, 1.0, 1.0});
  const NdArray<T> up = pb::elevate_degree_1d<T>(2, view2(ctrl, 3, 2), 1);
  REQUIRE(up.shape().size() == 2);
  CHECK(up.shape()[0] == 4);
  CHECK(up.shape()[1] == 2);
  // Column 0: (0, 0, 1/3, 1). Column 1: t as degree 3: (0, 1/3, 2/3, 1).
  const double expected[8] = {0.0, 0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0};
  for (std::size_t i = 0; i < 8; ++i) {
    CAPTURE(i);
    CHECK(std::fabs(static_cast<double>(up.data()[i]) - expected[i]) <= 2.0 * eps);
  }
  CHECK_THROWS_AS(static_cast<void>(pb::elevate_degree_1d<T>(3, view2(ctrl, 3, 2), 1)),
                  pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(pb::elevate_degree_1d<T>(2, view2(ctrl, 3, 2), 0)),
                  pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(pb::elevate_degree_1d<T>(2, view2(ctrl, 3, 2), 60)),
                  pantr::value_error);
}

TEST_CASE_TEMPLATE(
    "bezier multiply: t * t is t^2 exactly, and products match the pointwise product", T, float,
    double) {
  const double eps = eps_of<T>();
  const pb::Bezier<T> lin = make_bezier<T>(cast_all<T>({0.0, 1.0}), {2, 1});
  const pb::Bezier<T> sq = pb::multiply<T>(lin, lin);
  CHECK(sq.degree(0) == 2);
  CHECK(sq.control_points()[0] == static_cast<T>(0));
  CHECK(sq.control_points()[1] == static_cast<T>(0));
  CHECK(sq.control_points()[2] == static_cast<T>(1));
  // Pointwise product of two random vector-valued curves (degrees 3 and 2, rank 2), compared with
  // the componentwise product of the evaluations. With M = max|f| max|g| the product coefficients
  // are bounded by M (the binomial weights sum to one) and each carries at most (p + 6) eps M
  // (three roundings of the weight, two of the factors, one per accumulated term); evaluating h
  // costs 8 (p + q + 1) eps M, and the two factors 8 (p + 1) + 8 (q + 1) plus the product's own
  // rounding: all below 24 (p + q + 2) eps M.
  Lcg rng;
  const std::vector<T> fc = random_vec<T>(rng, 4 * 2);
  const std::vector<T> gc = random_vec<T>(rng, 3 * 2);
  const pb::Bezier<T> f = make_bezier<T>(fc, {4, 2});
  const pb::Bezier<T> g = make_bezier<T>(gc, {3, 2});
  const pb::Bezier<T> h = pb::multiply<T>(f, g);
  CHECK(h.degree(0) == 5);
  CHECK(h.rank() == 2);
  CHECK_FALSE(h.rational());
  const std::vector<T> ts = cast_all<T>(sample_params());
  const NdArray<T> a = eval1(f, ts);
  const NdArray<T> b = eval1(g, ts);
  const NdArray<T> c = eval1(h, ts);
  const double tol = 24.0 * (3.0 + 2.0 + 2.0) * eps * max_abs(fc) * max_abs(gc);
  for (std::size_t i = 0; i < c.size(); ++i) {
    CHECK(std::fabs(static_cast<double>(c.data()[i]) -
                    static_cast<double>(a.data()[i]) * static_cast<double>(b.data()[i])) <= tol);
  }
  // A surface times a surface: degrees (2, 1) and (1, 2), rank 1; then the tolerance counts the
  // (p_0 + 1)(p_1 + 1) accumulated terms in place of p + 1.
  const std::vector<T> sc = random_vec<T>(rng, 3 * 2);
  const std::vector<T> tc = random_vec<T>(rng, 2 * 3);
  const pb::Bezier<T> s = make_bezier<T>(sc, {3, 2, 1});
  const pb::Bezier<T> t = make_bezier<T>(tc, {2, 3, 1});
  const pb::Bezier<T> st = pb::multiply<T>(s, t);
  CHECK(st.degree(0) == 3);
  CHECK(st.degree(1) == 3);
  std::vector<T> pts;
  for (const double u : sample_params()) {
    pts.push_back(static_cast<T>(u));
    pts.push_back(static_cast<T>(1.0 - u * u));
  }
  const std::size_t np = pts.size() / 2;
  const NdArray<T> sa = pb::evaluate<T>(s, view2(pts, np, 2));
  const NdArray<T> sb = pb::evaluate<T>(t, view2(pts, np, 2));
  const NdArray<T> sc_v = pb::evaluate<T>(st, view2(pts, np, 2));
  const double stol = 24.0 * (6.0 + 6.0 + 4.0) * eps * max_abs(sc) * max_abs(tc);
  for (std::size_t i = 0; i < sc_v.size(); ++i) {
    CHECK(std::fabs(static_cast<double>(sc_v.data()[i]) -
                    static_cast<double>(sa.data()[i]) * static_cast<double>(sb.data()[i])) <= stol);
  }
}

TEST_CASE_TEMPLATE("bezier multiply: rational operands", T, float, double) {
  // The quarter circle (x, y) times itself is (x^2, y^2), and x^2 + y^2 = 1. Numerators and
  // weights are sums of non-negative terms, so relative errors apply: each product coefficient
  // carries (p + 6) eps, evaluating numerator and weight 8 (r + 1) eps each, the quotient one,
  // so 4 (16 (r + 1) + 2 (p + 6) + 1) eps bounds x^2 + y^2 - 1.
  const double eps = eps_of<T>();
  const double w = 1.0 / std::sqrt(2.0);
  const pb::Bezier<T> arc =
      make_bezier<T>(cast_all<T>({1.0, 0.0, 1.0, w, w, w, 0.0, 1.0, 1.0}), {3, 3}, true);
  const pb::Bezier<T> sq = pb::multiply<T>(arc, arc);
  CHECK(sq.rational());
  CHECK(sq.rank() == 2);
  CHECK(sq.degree(0) == 4);
  const std::vector<T> ts = cast_all<T>(sample_params());
  const NdArray<T> v = eval1(sq, ts);
  const double tol = 4.0 * (16.0 * 5.0 + 2.0 * 8.0 + 1.0) * eps;
  for (std::size_t j = 0; j < ts.size(); ++j) {
    CAPTURE(j);
    CHECK(std::fabs(static_cast<double>(v.data()[2 * j]) +
                    static_cast<double>(v.data()[2 * j + 1]) - 1.0) <= tol);
  }
  // Rational times non-rational, both orders: (2 x, 3 y) from the constant (2, 3), and the product
  // with a linear (1 + t, 2 t) against the componentwise product of the evaluations.
  const pb::Bezier<T> constant = make_bezier<T>(cast_all<T>({2.0, 3.0}), {1, 2});
  const pb::Bezier<T> scaled = pb::multiply<T>(arc, constant);
  const pb::Bezier<T> scaled2 = pb::multiply<T>(constant, arc);
  CHECK(scaled.rational());
  CHECK(scaled2.rational());
  CHECK(scaled.degree(0) == 2);
  const NdArray<T> base = eval1(arc, ts);
  const NdArray<T> s1 = eval1(scaled, ts);
  const NdArray<T> s2 = eval1(scaled2, ts);
  const pb::Bezier<T> lin = make_bezier<T>(cast_all<T>({1.0, 0.0, 2.0, 2.0}), {2, 2});
  const pb::Bezier<T> mixed = pb::multiply<T>(arc, lin);
  CHECK(mixed.degree(0) == 3);
  const NdArray<T> lv = eval1(lin, ts);
  const NdArray<T> mv = eval1(mixed, ts);
  // Each product coefficient is a sum of at most 5 binomial-weighted terms of magnitude <= 1, each
  // a few roundings (n eps sum|terms|); evaluating degree 3 adds 4 more: 64 eps per term, 5 terms.
  const double rel = 64.0 * 5.0 * eps;
  for (std::size_t j = 0; j < ts.size(); ++j) {
    const double x = static_cast<double>(base.data()[2 * j]);
    const double y = static_cast<double>(base.data()[2 * j + 1]);
    CHECK(std::fabs(static_cast<double>(s1.data()[2 * j]) - 2.0 * x) <= rel * 2.0 * x + rel);
    CHECK(std::fabs(static_cast<double>(s1.data()[2 * j + 1]) - 3.0 * y) <= rel * 3.0 * y + rel);
    CHECK(std::fabs(static_cast<double>(s2.data()[2 * j]) - 2.0 * x) <= rel * 2.0 * x + rel);
    CHECK(std::fabs(static_cast<double>(mv.data()[2 * j]) -
                    x * static_cast<double>(lv.data()[2 * j])) <= rel * 2.0);
    CHECK(std::fabs(static_cast<double>(mv.data()[2 * j + 1]) -
                    y * static_cast<double>(lv.data()[2 * j + 1])) <= rel * 2.0);
  }
}

TEST_CASE_TEMPLATE("bezier multiply: the dimension and rank checks, in order", T, float, double) {
  const pb::Bezier<T> curve = make_bezier<T>(cast_all<T>({0.0, 1.0}), {2, 1});
  const pb::Bezier<T> curve2 = make_bezier<T>(cast_all<T>({0.0, 1.0, 1.0, 0.0}), {2, 2});
  const pb::Bezier<T> surf2 =
      make_bezier<T>(cast_all<T>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}), {2, 2, 2});
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::multiply<T>(curve, surf2)),
                       "Operands must have the same dimension. Got 1 and 2.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::multiply<T>(curve, curve2)),
                       "Operands must have the same rank. Got 1 and 2.", pantr::value_error);
  // The rank excludes the weight: a rational rank-2 and a non-rational rank-3 differ in rank.
  const pb::Bezier<T> rat =
      make_bezier<T>(cast_all<T>({1.0, 0.0, 1.0, 0.0, 1.0, 1.0}), {2, 3}, true);
  const pb::Bezier<T> rank3 = make_bezier<T>(cast_all<T>({1.0, 0.0, 1.0, 0.0, 1.0, 1.0}), {2, 3});
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::multiply<T>(rat, rank3)),
                       "Operands must have the same rank. Got 2 and 3.", pantr::value_error);
  // Both wrong: the dimension is reported first.
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::multiply<T>(curve2, surf2)),
                       "Operands must have the same dimension. Got 1 and 2.", pantr::value_error);
}

TEST_CASE_TEMPLATE(
    "bezier bernstein_product: the 1D and tensor formulas agree, and reject bad shapes", T, float,
    double) {
  const double eps = eps_of<T>();
  Lcg rng;
  const std::vector<T> a = random_vec<T>(rng, 4 * 3);
  const std::vector<T> b = random_vec<T>(rng, 3 * 3);
  const std::size_t as[2] = {4, 3};
  const std::size_t bs[2] = {3, 3};
  const NdArray<T> p1 = pb::bernstein_product<T>(a, as, b, bs);
  const NdArray<T> pn = pb::bernstein_product_nd<T>(a, as, b, bs);
  REQUIRE(p1.shape().size() == 2);
  CHECK(p1.shape()[0] == 6);
  CHECK(p1.shape()[1] == 3);
  // Two formulas for the same exact numbers, each within (p + 6) eps M of it: 2 (p + 6) eps M.
  const double tol = 2.0 * (3.0 + 6.0) * eps * max_abs(a) * max_abs(b);
  for (std::size_t i = 0; i < p1.size(); ++i) {
    CHECK(std::fabs(static_cast<double>(p1.data()[i]) - static_cast<double>(pn.data()[i])) <= tol);
  }
  // Closed form: (1 + t)^2 = 1 + 2 t + t^2 has h(0) = 1, h(1) = 4 and h'(0) = 2 = 2 (c_1 - c_0),
  // so the Bernstein coefficients (1, 2, 4).
  const std::vector<T> one_plus_t = cast_all<T>({1.0, 2.0});
  const std::size_t s2[2] = {2, 1};
  const NdArray<T> sq = pb::bernstein_product<T>(one_plus_t, s2, one_plus_t, s2);
  CHECK(sq.shape()[0] == 3);
  CHECK(sq.data()[0] == static_cast<T>(1));
  CHECK(sq.data()[1] == static_cast<T>(2));
  CHECK(sq.data()[2] == static_cast<T>(4));
  // A product with the constant 1 returns the other factor, in the tensor formula too.
  const std::size_t two_dir[3] = {2, 2, 1};
  const std::size_t one_dir[3] = {1, 1, 1};
  const std::vector<T> f = cast_all<T>({3.0, -1.0, 2.0, 5.0});
  const std::vector<T> unit = cast_all<T>({1.0});
  const NdArray<T> fu = pb::bernstein_product<T>(f, two_dir, unit, one_dir);
  REQUIRE(fu.size() == 4);
  for (std::size_t i = 0; i < 4; ++i) {
    // A product with the constant 1 is f times binomial ratios equal to 1, a few roundings each:
    // 8 eps of the largest coefficient magnitude, 5.
    CHECK(std::fabs(static_cast<double>(fu.data()[i]) - static_cast<double>(f[i])) <=
          8.0 * eps * 5.0);
  }
  // Shapes that do not describe two operands of one rank, and a span of the wrong size.
  const std::size_t rank_differs[2] = {3, 2};
  const std::size_t one_extent[1] = {3};
  const std::size_t zero_extent[2] = {0, 3};
  CHECK_THROWS_AS(static_cast<void>(pb::bernstein_product<T>(a, as, b, rank_differs)),
                  pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(pb::bernstein_product<T>(a, as, b, two_dir)),
                  pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(pb::bernstein_product<T>(a, one_extent, b, one_extent)),
                  pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(pb::bernstein_product<T>(a, zero_extent, b, bs)),
                  pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(pb::bernstein_product<T>(a, bs, b, bs)), pantr::value_error);
}

TEST_CASE_TEMPLATE("bezier derivative: t^2 gives 2 t, with and without keeping the degree", T,
                   float, double) {
  const pb::Bezier<T> sq = make_bezier<T>(cast_all<T>({0.0, 0.0, 1.0}), {3, 1});
  // p (P_{i+1} - P_i) = (2 * 0, 2 * 1): exact.
  const pb::Bezier<T> d = pb::derivative<T>(sq, 0, false);
  CHECK(d.degree(0) == 1);
  CHECK(d.control_points()[0] == static_cast<T>(0));
  CHECK(d.control_points()[1] == static_cast<T>(2));
  // 2 t as a degree-2 polynomial: (0, 1, 2).
  const pb::Bezier<T> k = pb::derivative<T>(sq, 0, true);
  CHECK(k.degree(0) == 2);
  CHECK(k.control_points()[0] == static_cast<T>(0));
  CHECK(k.control_points()[1] == static_cast<T>(1));
  CHECK(k.control_points()[2] == static_cast<T>(2));
  // d/dv of u^2 v = u^2, in direction 1 of a (3, 2) net; d/du keeps the second direction.
  const pb::Bezier<T> u2v = make_bezier<T>(cast_all<T>({0.0, 0.0, 0.0, 0.0, 0.0, 1.0}), {3, 2, 1});
  const pb::Bezier<T> dv = pb::derivative<T>(u2v, 1, false);
  CHECK(dv.degree(0) == 2);
  CHECK(dv.degree(1) == 0);
  const std::vector<double> expect_dv = {0.0, 0.0, 1.0};
  for (std::size_t i = 0; i < 3; ++i) {
    CHECK(static_cast<double>(dv.control_points()[i]) == expect_dv[i]);
  }
  const pb::Bezier<T> du = pb::derivative<T>(u2v, 0, false);
  CHECK(du.degree(0) == 1);
  CHECK(du.degree(1) == 1);
}

TEST_CASE_TEMPLATE("bezier derivative: the hodograph evaluates to evaluate_derivatives", T, float,
                   double) {
  // Random non-rational surface, degrees (3, 2), rank 2, both directions, both degree options.
  // The hodograph's coefficients carry 3 eps (p Delta P, with Delta P < 2 max|c|) on the (2 p
  // max|c|) scale, and the evaluate_derivatives bound is 8 (sum_d (p_d + 1)^2) eps (2 p) sum|c|
  // (see the evaluation suite); the two are within twice the larger of them.
  const double eps = eps_of<T>();
  Lcg rng;
  const std::vector<T> ctrl = random_vec<T>(rng, 4 * 3 * 2);
  const pb::Bezier<T> surf = make_bezier<T>(ctrl, {4, 3, 2});
  std::vector<double> sumabs_c;
  double sum_abs = 0.0;
  for (const T c : ctrl) {
    sum_abs += std::fabs(static_cast<double>(c));
  }
  std::vector<T> pts;
  for (const double u : sample_params()) {
    for (const double v : {0.0, 0.4, 1.0}) {
      pts.push_back(static_cast<T>(u));
      pts.push_back(static_cast<T>(v));
    }
  }
  const std::size_t np = pts.size() / 2;
  for (long long dir = 0; dir < 2; ++dir) {
    for (const bool keep : {false, true}) {
      const pb::Bezier<T> h = pb::derivative<T>(surf, dir, keep);
      CHECK(h.degree(static_cast<std::size_t>(dir)) ==
            (keep ? surf.degree(static_cast<std::size_t>(dir))
                  : surf.degree(static_cast<std::size_t>(dir)) - 1));
      const NdArray<T> a = pb::evaluate<T>(h, view2(pts, np, 2));
      const std::vector<long long> orders = {dir == 0 ? 1LL : 0LL, dir == 1 ? 1LL : 0LL};
      const NdArray<T> b = pb::evaluate_derivatives<T>(surf, view2(pts, np, 2), orders);
      const double p_dir = static_cast<double>(surf.degree(static_cast<std::size_t>(dir)));
      const double tol = 2.0 * 8.0 * (16.0 + 9.0) * eps * 2.0 * p_dir * sum_abs;
      for (std::size_t i = 0; i < a.size(); ++i) {
        CAPTURE(dir);
        CAPTURE(keep);
        CHECK(std::fabs(static_cast<double>(a.data()[i]) - static_cast<double>(b.data()[i])) <=
              tol);
      }
    }
  }
}

TEST_CASE_TEMPLATE("bezier derivative: a rational curve against 1 / (1 + t) and the quarter circle",
                   T, float, double) {
  const double eps = eps_of<T>();
  // N = 1, W = 1 + t: f' = -1 / (1 + t)^2. A' = 0 exactly, so the numerator is -A w' with no
  // cancellation, and every quantity is positive: relative error at most 4 (16 (r + 1) + 2 (p + 6)
  // + 1) eps with r = 2 p = 2.
  const pb::Bezier<T> f = make_bezier<T>(cast_all<T>({1.0, 1.0, 1.0, 2.0}), {2, 2}, true);
  const pb::Bezier<T> df = pb::derivative<T>(f, 0, false);
  CHECK(df.rational());
  CHECK(df.rank() == 1);
  CHECK(df.degree(0) == 2);
  const std::vector<T> ts = cast_all<T>(sample_params());
  const NdArray<T> v = eval1(df, ts);
  for (std::size_t j = 0; j < ts.size(); ++j) {
    const double t = static_cast<double>(ts[j]);
    const double expected = -1.0 / ((1.0 + t) * (1.0 + t));
    CAPTURE(t);
    CHECK(std::fabs(static_cast<double>(v.data()[j]) - expected) <=
          4.0 * (16.0 * 3.0 + 2.0 * 7.0 + 1.0) * eps * std::fabs(expected));
  }
  // Quarter circle: the hodograph evaluates to evaluate_derivatives. The numerator A' w - A w' can
  // cancel, so the bound is absolute, on the scale of the products: |A'|, |w| <= 2 and the
  // weights >= 1/sqrt 2 (w^2 >= 1/2 divides the numerator): two products of (r + 6) eps on a scale
  // of 4 each, evaluation of numerator and weight 8 (r + 1) eps on that scale, doubled for the
  // division by w^2 and again for the other side's own error: below 1024 eps.
  const double w = 1.0 / std::sqrt(2.0);
  const pb::Bezier<T> arc =
      make_bezier<T>(cast_all<T>({1.0, 0.0, 1.0, w, w, w, 0.0, 1.0, 1.0}), {3, 3}, true);
  const pb::Bezier<T> tangent = pb::derivative<T>(arc, 0, false);
  CHECK(tangent.degree(0) == 4);
  const NdArray<T> a = eval1(tangent, ts);
  const std::vector<long long> orders = {1};
  const NdArray<T> b = pb::evaluate_derivatives<T>(arc, view2(ts, ts.size(), 1), orders);
  const NdArray<T> pos = eval1(arc, ts);
  for (std::size_t i = 0; i < a.size(); ++i) {
    CHECK(std::fabs(static_cast<double>(a.data()[i]) - static_cast<double>(b.data()[i])) <=
          1024.0 * eps);
  }
  // The tangent is orthogonal to the position of a point moving on a circle: x x' + y y' = 0.
  for (std::size_t j = 0; j < ts.size(); ++j) {
    const double dot =
        static_cast<double>(pos.data()[2 * j]) * static_cast<double>(a.data()[2 * j]) +
        static_cast<double>(pos.data()[2 * j + 1]) * static_cast<double>(a.data()[2 * j + 1]);
    CAPTURE(j);
    CHECK(std::fabs(dot) <= 2.0 * 1024.0 * eps);
  }
}

TEST_CASE_TEMPLATE("bezier derivative: the checks and their messages", T, float, double) {
  const pb::Bezier<T> surf = make_bezier<T>(cast_all<T>({0.0, 0.0, 1.0, 1.0}), {2, 2, 1});
  const pb::Bezier<T> flat = make_bezier<T>(cast_all<T>({0.0, 1.0}), {1, 2, 1});  // degrees (0, 1)
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::derivative<T>(surf, 2, false)),
                       "direction must be in [0, 2), got 2.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::derivative<T>(surf, -1, true)),
                       "direction must be in [0, 2), got -1.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::derivative<T>(flat, 0, false)),
                       "Derivative of a degree-0 Bézier is not defined.", pantr::value_error);
  CHECK_NOTHROW(static_cast<void>(pb::derivative<T>(flat, 1, false)));
  const pb::Bezier<T> rat_flat = make_bezier<T>(cast_all<T>({1.0, 1.0}), {1, 2}, true);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::derivative<T>(rat_flat, 0, true)),
                       "Derivative of a degree-0 Bézier is not defined.", pantr::value_error);
}

TEST_CASE_TEMPLATE("bezier kernels: slice, split and restrict agree with the direct Bernstein sum",
                   T, float, double) {
  // De Casteljau in a T workspace: each of the p levels is three operations and one rounding to T,
  // so a result carries at most (p + 1) eps_T max|c| from the workspace and 4 eps for the three
  // double operations; the reference (powers and binomials in double) adds 6 (p + 1) eps_double.
  // 12 (p + 1) eps_T max|c| covers both. A split or restriction is evaluated afterwards, which
  // doubles it.
  const double eps = eps_of<T>();
  Lcg rng;
  const std::size_t p = 4;
  const std::size_t cols = 3;
  const std::vector<T> ctrl = random_vec<T>(rng, (p + 1) * cols);
  const double m = max_abs(ctrl);
  const double tol = 12.0 * static_cast<double>(p + 1) * eps * m;
  const auto cview = view2(ctrl, p + 1, cols);
  for (const double u : {0.0, 0.3, 0.7, 1.0}) {
    std::vector<T> out(cols);
    pb::detail::slice_1d_kernel<T>(cview, u, std::span<T>(out));
    for (std::size_t c = 0; c < cols; ++c) {
      CAPTURE(u);
      CHECK(std::fabs(static_cast<double>(out[c]) - direct_eval(ctrl.data(), p + 1, cols, c, u)) <=
            tol);
      if (u == 0.0) {
        CHECK(out[c] == ctrl[c]);
      }
      if (u == 1.0) {
        CHECK(out[c] == ctrl[p * cols + c]);
      }
    }
  }
  const double split_at = 0.35;
  NdArray<T> left({p + 1, cols});
  NdArray<T> right({p + 1, cols});
  pb::detail::split_1d_kernel<T>(cview, split_at, left.template view<2>(),
                                 right.template view<2>());
  for (const double s : {0.0, 0.2, 0.5, 0.9, 1.0}) {
    for (std::size_t c = 0; c < cols; ++c) {
      CAPTURE(s);
      CHECK(std::fabs(direct_eval(left.data(), p + 1, cols, c, s) -
                      direct_eval(ctrl.data(), p + 1, cols, c, split_at * s)) <= 2.0 * tol);
      CHECK(std::fabs(direct_eval(right.data(), p + 1, cols, c, s) -
                      direct_eval(ctrl.data(), p + 1, cols, c, split_at + (1.0 - split_at) * s)) <=
            2.0 * tol);
    }
  }
  // Both orders of the two passes: |upper| >= |lower - 1| (0.9 against 0.8) and the other way
  // (0.5 against 0.9).
  const std::vector<std::pair<double, double>> ranges = {
      {0.2, 0.9}, {0.6, 0.8}, {0.1, 0.5}, {0.0, 1.0}};
  for (const auto& [lower, upper] : ranges) {
    NdArray<T> out({p + 1, cols});
    pb::detail::restrict_1d_kernel<T>(cview, lower, upper, out.template view<2>());
    for (const double s : {0.0, 0.25, 0.5, 1.0}) {
      for (std::size_t c = 0; c < cols; ++c) {
        CAPTURE(lower);
        CAPTURE(upper);
        CAPTURE(s);
        CHECK(std::fabs(direct_eval(out.data(), p + 1, cols, c, s) -
                        direct_eval(ctrl.data(), p + 1, cols, c, lower + (upper - lower) * s)) <=
              4.0 * tol);
      }
    }
  }
}

TEST_CASE_TEMPLATE("bezier kernels: the scalar product kernel and the degree-1 and degree-0 edges",
                   T, float, double) {
  const double eps = eps_of<T>();
  // t * t = t^2, exactly.
  const std::vector<T> lin = cast_all<T>({0.0, 1.0});
  std::vector<T> sq(3);
  pb::detail::scalar_bernstein_product_1d_kernel<T>(lin, lin, sq);
  CHECK(sq[0] == static_cast<T>(0));
  CHECK(sq[1] == static_cast<T>(0));
  CHECK(sq[2] == static_cast<T>(1));
  // Against the library formula on random operands: each is within (p + 6) eps M of the exact
  // coefficients, so 2 (p + 6) eps M apart.
  Lcg rng;
  const std::vector<T> a = random_vec<T>(rng, 4);
  const std::vector<T> b = random_vec<T>(rng, 3);
  std::vector<T> out(6);
  pb::detail::scalar_bernstein_product_1d_kernel<T>(a, b, out);
  const std::size_t as[2] = {4, 1};
  const std::size_t bs[2] = {3, 1};
  const NdArray<T> ref = pb::bernstein_product<T>(a, as, b, bs);
  for (std::size_t i = 0; i < 6; ++i) {
    CHECK(std::fabs(static_cast<double>(out[i]) - static_cast<double>(ref.data()[i])) <=
          2.0 * 9.0 * eps * max_abs(a) * max_abs(b));
  }
  // A degree-0 curve is its own slice, split and restriction.
  const std::vector<T> point = cast_all<T>({2.5, -1.0});
  const auto pv = view2(point, 1, 2);
  std::vector<T> s(2);
  pb::detail::slice_1d_kernel<T>(pv, 0.4, std::span<T>(s));
  CHECK(s[0] == point[0]);
  CHECK(s[1] == point[1]);
  NdArray<T> r({1, 2});
  pb::detail::restrict_1d_kernel<T>(pv, 0.1, 0.6, r.template view<2>());
  CHECK(r.data()[0] == point[0]);
  CHECK(r.data()[1] == point[1]);
}

TEST_CASE("bezier kernels: the threaded result equals the serial one within the rounding bound") {
  // One column per work item, each by the same instruction sequence whatever the chunking: equal
  // in practice, asserted to the bound of the previous case, 2 * 12 (p + 1) eps max|c|. The sizes
  // are above `parallel_min_ops` (design/threading.md).
  using T = double;
  const double eps = eps_of<T>();
  Lcg rng;
  const std::size_t p = 4;
  const std::size_t cols = 6000;
  const std::vector<T> ctrl = random_vec<T>(rng, (p + 1) * cols);
  const auto cview = view2(ctrl, p + 1, cols);
  const std::size_t saved = pantr::parallel::get_num_threads();
  auto run = [&]() {
    std::vector<NdArray<T>> r;
    NdArray<T> left({p + 1, cols});
    NdArray<T> right({p + 1, cols});
    pb::detail::split_1d_kernel<T>(cview, 0.37, left.template view<2>(), right.template view<2>());
    r.push_back(std::move(left));
    r.push_back(std::move(right));
    NdArray<T> res({p + 1, cols});
    pb::detail::restrict_1d_kernel<T>(cview, 0.15, 0.85, res.template view<2>());
    r.push_back(std::move(res));
    NdArray<T> sl({cols});
    pb::detail::slice_1d_kernel<T>(cview, 0.62, sl.flat());
    r.push_back(std::move(sl));
    return r;
  };
  pantr::parallel::set_num_threads(1);
  const std::vector<NdArray<T>> serial = run();
  pantr::parallel::set_num_threads(static_cast<std::int64_t>(pantr::parallel::max_threads()));
  const std::vector<NdArray<T>> threaded = run();
  pantr::parallel::set_num_threads(static_cast<std::int64_t>(saved));
  const double bound = 24.0 * static_cast<double>(p + 1) * eps * max_abs(ctrl);
  for (std::size_t k = 0; k < serial.size(); ++k) {
    REQUIRE(serial[k].size() == threaded[k].size());
    double worst = 0.0;
    for (std::size_t i = 0; i < serial[k].size(); ++i) {
      worst = (std::max)(worst, std::fabs(serial[k].data()[i] - threaded[k].data()[i]));
    }
    CAPTURE(k);
    CHECK(worst <= bound);
  }
}
