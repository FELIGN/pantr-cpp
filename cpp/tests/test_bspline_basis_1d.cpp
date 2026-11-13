// The B-spline basis of a 1D space at points: values, first indices and derivatives. Oracles: the
// basis from its recursive definition in exact rational arithmetic (bspline_oracle.hpp; not the
// triangle the library runs), closed forms of low-degree B-splines (hat functions, Bernstein
// polynomials with the chain rule, the cardinal quadratic), the partition of unity and the
// vanishing of the derivative rows' sum, the binomial values of the highest derivative of a
// cardinal B-spline, and the restricted space's functions being the parent's. Boundary cases are
// constructed (a knot a hair above a point), not sampled; the status of a zero divisor, the pool
// and the mixed knot and point types are exercised here.
#include <doctest/doctest.h>

#include "bspline_oracle.hpp"

#include <pantr/bspline/knot_factories.hpp>
#include <pantr/bspline/space_1d.hpp>
#include <pantr/parallel/parallel_for.hpp>
#include <pantr/parallel/work_model.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace bs = pantr::bspline;
namespace oracle = bspline_oracle;
using pantr::NdArray;

namespace {

template <class T>
double eps_of() {
  return static_cast<double>(std::numeric_limits<T>::epsilon());
}

/// The smallest denormal of the wider-resolution one of the two types, as a double.
template <class K, class P>
double denorm_of() {
  return (std::max)(static_cast<double>(std::numeric_limits<K>::denorm_min()),
                    static_cast<double>(std::numeric_limits<P>::denorm_min()));
}

template <class T>
std::vector<T> cast_all(const std::vector<double>& v) {
  std::vector<T> out;
  out.reserve(v.size());
  for (const double x : v) {
    out.push_back(static_cast<T>(x));
  }
  return out;
}

template <class T>
std::span<const T> sp(const std::vector<T>& v) {
  return std::span<const T>(v);
}

/// A knot type and a point type.
template <class K, class P>
struct KP {
  using Knot = K;
  using Point = P;
};
using DD = KP<double, double>;
using FF = KP<float, float>;
using FD = KP<float, double>;
using DF = KP<double, float>;

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// A space of the table every oracle test runs over. The knots are dyadic rationals of few bits,
/// so the same space exists in float and in double and every difference of two knots is exact.
struct Case {
  const char* name;
  std::vector<double> knots;
  int degree;
  bool periodic;
  bool can_extrapolate;  // the spans at both ends of the domain are non-empty
};

const std::vector<Case>& cases() {
  static const std::vector<Case> table = {
      {"degree 0", {0, 1, 2, 3}, 0, false, true},
      {"hat functions", {0, 0, 0.5, 1, 1}, 1, false, true},
      {"quadratic with a double knot", {0, 0, 0, 0.25, 0.75, 0.75, 1, 1, 1}, 2, false, true},
      {"cubic with C0 and C1 knots", {0, 0, 0, 0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4}, 3, false, true},
      {"quadratic unclamped", {0, 1, 2.5, 3, 4, 6, 7.5, 8}, 2, false, true},
      {"quadratic, discontinuous", {0, 0, 0, 1, 1, 1, 2, 2, 2}, 2, false, true},
      {"quartic", {0, 0, 0, 0, 0, 1, 2, 3, 3, 3, 3, 3}, 4, false, true},
      {"periodic cardinal cubic", {-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7}, 3, true, true},
      {"periodic C0 quadratic", {-0.5, 0, 0, 0.5, 0.5, 1, 1, 1.5}, 2, true, true},
      {"offset domain", {100, 100, 100, 100.5, 101, 101, 101}, 2, false, true},
      {"left end of multiplicity p + 2", {0, 0, 0, 0, 1, 1, 1}, 2, false, false},
      {"Bezier cubic on [2, 6]", {2, 2, 2, 2, 6, 6, 6, 6}, 3, false, true},
      {"Bezier quadratic", {0, 0, 0, 1, 1, 1}, 2, false, true},
      {"Bezier linear", {-1, -1, 3, 3}, 1, false, true},
      {"left clamped, right unclamped", {0, 0, 0, 1, 2, 3, 4, 5}, 2, false, true},
  };
  return table;
}

/// The smallest positive difference of two consecutive knots.
double min_gap(const Case& c) {
  double h = kInf;
  for (std::size_t i = 0; i + 1 < c.knots.size(); ++i) {
    const double g = c.knots[i + 1] - c.knots[i];
    if (g > 0.0) {
      h = (std::min)(h, g);
    }
  }
  return h;
}

/// Points of the domain: every distinct knot, the point a ulp inside it on each side, and in each
/// interval the midpoint and a point at 0.8 (not dyadic, so rounding is exercised).
template <class P>
std::vector<P> sample_points(const Case& c) {
  const auto d = static_cast<std::size_t>(c.degree);
  const double a = c.knots[d];
  const double b = c.knots[c.knots.size() - d - 1];
  std::vector<double> distinct;
  for (std::size_t i = d; i + d + 1 <= c.knots.size(); ++i) {
    if (distinct.empty() || c.knots[i] != distinct.back()) {
      distinct.push_back(c.knots[i]);
    }
  }
  const P up = std::numeric_limits<P>::infinity();
  std::vector<P> pts;
  for (const double k : distinct) {
    const auto pk = static_cast<P>(k);
    pts.push_back(pk);
    if (k > a) {
      pts.push_back(std::nextafter(pk, -up));
    }
    if (k < b) {
      pts.push_back(std::nextafter(pk, up));
    }
  }
  for (std::size_t i = 0; i + 1 < distinct.size(); ++i) {
    pts.push_back(static_cast<P>(0.5 * (distinct[i] + distinct[i + 1])));
    pts.push_back(static_cast<P>(distinct[i] + 0.8 * (distinct[i + 1] - distinct[i])));
  }
  return pts;
}

double to_double(const oracle::Rational& q) { return q.convert_to<double>(); }

/// Check `tabulate_basis` against the exact definition.
///
/// The error of A2.2 is relative, per value: the recurrence only adds and multiplies non-negative
/// numbers, each level contributes at most six roundings (two knot differences, their sum, the
/// quotient, the product, the sum), so the value of the function of a degree `p` is within
/// `6 p u = 3 p eps` of the exact one, relative; 8 p eps leaves a margin of about 2.7. The `eps` is
/// that of the narrower of the knot and point types, as the differences are held in the knot type
/// and the results in the point type. A zero value must be exactly zero. Outside the domain the
/// terms are no longer non-negative, so the bound is relative to the sum of the magnitudes.
/// Where a value underflows (a point a denormal away from a knot) the relative bound is void: each
/// rounding in the denormal range is absolute, at most half the smallest denormal of the wider
/// of the stored types, and a knot difference enters divided by at least `h`, so `4 p dmin / h`
/// is added (an admitted margin over the roundings of the p levels).
template <class K, class P>
void check_values(const bs::BsplineSpace1D<K>& s, const std::vector<P>& pts, bool validate,
                  bool outside, double h) {
  const int p = s.degree();
  const auto ord = static_cast<std::size_t>(p) + 1;
  const auto u = oracle::exact_knots<K>(s.knots());
  const double eps = (std::max)(eps_of<K>(), eps_of<P>());
  const double underflow_floor = 4.0 * p * denorm_of<K, P>() / h;
  const auto [vals, first] = s.template tabulate_basis<P>(pts, validate);
  REQUIRE(vals.size() == pts.size() * ord);
  for (std::size_t i = 0; i < pts.size(); ++i) {
    oracle::BasisAt b(u, p, oracle::exact(static_cast<double>(pts[i])));
    const auto want_first = static_cast<std::int64_t>(b.span()) - p;
    CAPTURE(i);
    CAPTURE(static_cast<double>(pts[i]));
    CHECK(first.data()[i] == want_first);
    std::vector<oracle::Rational> want;
    double sum_abs = 0.0;
    for (std::size_t j = 0; j < ord; ++j) {
      want.push_back(b.deriv(static_cast<std::size_t>(want_first) + j, 0));
      sum_abs += std::fabs(to_double(want.back()));
    }
    // A Bezier piece takes the Bernstein path, which forms s = (x - a) / (b - a) first: its
    // rounding is relative to 1, so a value is only accurate to an absolute 8 (p + 1) eps (the
    // value of the end function at the point a ulp inside b has no correct digit, correctly).
    const bool bernstein = s.has_bezier_like_knots();
    double sum = 0.0;
    for (std::size_t j = 0; j < ord; ++j) {
      const double mag = outside ? sum_abs : std::fabs(to_double(want[j]));
      const double tol =
          (bernstein ? 8.0 * static_cast<double>(ord) * eps * (std::max)(1.0, sum_abs)
                     : 8.0 * p * eps * mag) +
          underflow_floor;
      CAPTURE(j);
      CHECK(oracle::within(static_cast<double>(vals.data()[i * ord + j]), want[j], tol));
      sum += static_cast<double>(vals.data()[i * ord + j]);
    }
    if (!outside) {
      CHECK(std::fabs(sum - 1.0) <= 4.0 * static_cast<double>(ord) * eps);  // partition of unity
    }
  }
}

/// Check `tabulate_basis_derivatives` of orders 0..n_deriv against the exact definition.
///
/// Order 0 is bounded as the values are. For `1 <= k <= p`, each differentiation divides by a knot
/// difference of at least `h` and combines at most two terms of weight at most `p`, so the entries
/// are of size `(2 p / h)^k` and the rounding error of a sum of `p + 1` of them is that times
/// `(p + 1) eps`; the factor 16 is an admitted margin, not a derivation. Rows past the degree are
/// exactly zero. The rows `k >= 1` must also sum to zero, the derivative of the partition of unity.
template <class K, class P>
void check_derivatives(const bs::BsplineSpace1D<K>& s, const std::vector<P>& pts, int n_deriv,
                       double h) {
  const int p = s.degree();
  const auto ord = static_cast<std::size_t>(p) + 1;
  const auto rows = static_cast<std::size_t>(n_deriv) + 1;
  const auto u = oracle::exact_knots<K>(s.knots());
  const double eps = (std::max)(eps_of<K>(), eps_of<P>());
  const double underflow_floor = 4.0 * p * denorm_of<K, P>() / h;
  const auto d = s.template tabulate_basis_derivatives<P>(pts, n_deriv, true);
  CHECK(d.complete);
  for (std::size_t i = 0; i < pts.size(); ++i) {
    oracle::BasisAt b(u, p, oracle::exact(static_cast<double>(pts[i])));
    const auto f = static_cast<std::size_t>(b.span()) - static_cast<std::size_t>(p);
    CAPTURE(i);
    CAPTURE(static_cast<double>(pts[i]));
    CHECK(d.first.data()[i] == static_cast<std::int64_t>(f));
    for (std::size_t k = 0; k < rows; ++k) {
      const double tol_k =
          16.0 * static_cast<double>(ord) * eps * std::pow(2.0 * p / h, static_cast<double>(k));
      double row_sum = 0.0;
      for (std::size_t j = 0; j < ord; ++j) {
        const oracle::Rational want = b.deriv(f + j, static_cast<int>(k));
        const double got = static_cast<double>(d.values.data()[(i * rows + k) * ord + j]);
        CAPTURE(k);
        CAPTURE(j);
        if (k == 0) {
          const double tol0 =
              (s.has_bezier_like_knots() ? 8.0 * static_cast<double>(ord) * eps
                                         : 8.0 * p * eps * std::fabs(to_double(want))) +
              underflow_floor;
          CHECK(oracle::within(got, want, tol0));
        } else if (static_cast<int>(k) > p) {
          CHECK(got == 0.0);
        } else {
          CHECK(oracle::within(got, want, tol_k));
        }
        row_sum += got;
      }
      if (k >= 1) {
        CHECK(std::fabs(row_sum) <= tol_k * static_cast<double>(ord));
      }
    }
  }
}

/// Restores the calling thread's parallelism setting on scope exit.
class ThreadsGuard {
 public:
  explicit ThreadsGuard(std::size_t n) : previous_(pantr::parallel::get_num_threads()) {
    pantr::parallel::set_num_threads(static_cast<std::int64_t>(n));
  }
  ThreadsGuard(const ThreadsGuard&) = delete;
  ThreadsGuard& operator=(const ThreadsGuard&) = delete;
  ~ThreadsGuard() { pantr::parallel::set_num_threads(static_cast<std::int64_t>(previous_)); }

 private:
  std::size_t previous_;
};

}  // namespace

// ---- the basis against the definition ---------------------------------------------------------

TEST_CASE_TEMPLATE(
    "space1d: values, indices and derivatives equal the exact Cox-de Boor definition", Pair, DD, FF,
    FD, DF) {
  using K = typename Pair::Knot;
  using P = typename Pair::Point;
  for (const Case& c : cases()) {
    CAPTURE(std::string(c.name));
    const std::vector<K> knots = cast_all<K>(c.knots);
    const bs::BsplineSpace1D<K> s(sp(knots), c.degree, c.periodic);
    const std::vector<P> pts = sample_points<P>(c);
    REQUIRE(pts.size() >= 4);  // the loops below run over something
    check_values<K, P>(s, pts, true, false, min_gap(c));
    check_derivatives<K, P>(s, pts, c.degree + 2, min_gap(c));
  }
}

TEST_CASE_TEMPLATE("space1d: outside the domain the end span's polynomial is evaluated", Pair, DD,
                   FF, FD, DF) {
  // With validate == false a point left of the domain is evaluated on the first span and a point
  // right of it on the last. The header promises no extrapolation ("no extrapolation guarantee is
  // made"); this pins the polynomial of the nearest span, which a caller relying on the band of
  // the knot tolerance (a point a hair outside the domain) depends on.
  using K = typename Pair::Knot;
  using P = typename Pair::Point;
  for (const Case& c : cases()) {
    if (!c.can_extrapolate) {
      continue;
    }
    CAPTURE(std::string(c.name));
    const std::vector<K> knots = cast_all<K>(c.knots);
    const bs::BsplineSpace1D<K> s(sp(knots), c.degree, c.periodic);
    const auto d = static_cast<std::size_t>(c.degree);
    const double a = c.knots[d];
    const double b = c.knots[c.knots.size() - d - 1];
    const std::vector<P> pts = {static_cast<P>(a - 0.5), static_cast<P>(a - 0.0625),
                                static_cast<P>(b + 0.0625), static_cast<P>(b + 0.5)};
    check_values<K, P>(s, pts, false, true, min_gap(c));
  }
}

// ---- closed forms -----------------------------------------------------------------------------

namespace {

void expect_near(double got, double want, double tol) {
  CAPTURE(got);
  CAPTURE(want);
  CHECK(std::fabs(got - want) <= tol);
}

}  // namespace

TEST_CASE("space1d: hat functions, worked out by hand") {
  // Knots [0, 0, 1/2, 1, 1], degree 1: N0 = 1 - 2x, N1 = 2x then 2 - 2x, N2 = 2x - 1.
  const std::vector<double> knots = {0, 0, 0.5, 1, 1};
  const bs::BsplineSpace1D<double> s(sp(knots), 1);
  const std::vector<double> pts = {0.25, 0.5, 0.75, 1.0};
  const auto d = s.tabulate_basis_derivatives<double>(pts, 1);
  // At a knot the span to its right is used, and at the end the last span.
  CHECK(std::vector<std::int64_t>(d.first.data(), d.first.data() + 4) ==
        std::vector<std::int64_t>{0, 1, 1, 1});
  const double want[4][2][2] = {
      {{0.5, 0.5}, {-2, 2}}, {{1, 0}, {-2, 2}}, {{0.5, 0.5}, {-2, 2}}, {{0, 1}, {-2, 2}}};
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t k = 0; k < 2; ++k) {
      for (std::size_t j = 0; j < 2; ++j) {
        expect_near(static_cast<double>(d.values.data()[(i * 2 + k) * 2 + j]), want[i][k][j],
                    8.0 * eps_of<double>());
      }
    }
  }
}

TEST_CASE("space1d: a Bezier piece is the Bernstein basis, chain rule included") {
  // [0, 0, 0, 1, 1, 1]: B = ((1-x)^2, 2x(1-x), x^2), B' = (-2(1-x), 2 - 4x, 2x), B'' = (2, -4, 2),
  // B''' = 0. On [2, 6] the derivative of order k is the same divided by 4^k (s = (x - 2) / 4).
  const double eps = eps_of<double>();
  {
    const std::vector<double> knots = {0, 0, 0, 1, 1, 1};
    const bs::BsplineSpace1D<double> s(sp(knots), 2);
    REQUIRE(s.has_bezier_like_knots());
    const std::vector<double> pts = {0.25};
    const auto d = s.tabulate_basis_derivatives<double>(pts, 3);
    const double want[4][3] = {{0.5625, 0.375, 0.0625}, {-1.5, 1, 0.5}, {2, -4, 2}, {0, 0, 0}};
    for (std::size_t k = 0; k < 4; ++k) {
      for (std::size_t j = 0; j < 3; ++j) {
        expect_near(static_cast<double>(d.values.data()[k * 3 + j]), want[k][j], 16.0 * eps);
      }
    }
  }
  {
    const std::vector<double> knots = {2, 2, 2, 6, 6, 6};
    const bs::BsplineSpace1D<double> s(sp(knots), 2);
    const std::vector<double> pts = {3.0, 4.0};
    const auto d = s.tabulate_basis_derivatives<double>(pts, 3);
    const double want[2][4][3] = {
        {{0.5625, 0.375, 0.0625}, {-0.375, 0.25, 0.125}, {0.125, -0.25, 0.125}, {0, 0, 0}},
        {{0.25, 0.5, 0.25}, {-0.25, 0, 0.25}, {0.125, -0.25, 0.125}, {0, 0, 0}}};
    for (std::size_t i = 0; i < 2; ++i) {
      for (std::size_t k = 0; k < 4; ++k) {
        for (std::size_t j = 0; j < 3; ++j) {
          expect_near(static_cast<double>(d.values.data()[(i * 4 + k) * 3 + j]), want[i][k][j],
                      16.0 * eps);
        }
      }
    }
  }
}

TEST_CASE("space1d: the cardinal quadratic B-spline pieces, worked out by hand") {
  // Unit knots, degree 2: on [0, 1) the three non-zero functions are (1-x)^2/2, (-2x^2+2x+1)/2
  // and x^2/2 (the pieces of the cardinal quadratic shifted by 2, 1 and 0), with derivatives
  // (-(1-x), 1-2x, x) and (1, -2, 1). The same on every unit interval of a periodic space.
  const auto knots = bs::cardinal_knots<double>(4, 2);
  const bs::BsplineSpace1D<double> s(sp(knots), 2, true);
  for (const double x0 : {0.25, 0.75}) {
    for (const std::int64_t shift : {0, 1, 3}) {
      const double x = x0 + static_cast<double>(shift);
      const std::vector<double> pts = {x};
      const auto d = s.tabulate_basis_derivatives<double>(pts, 2);
      CHECK(d.first.data()[0] == shift);
      const double want[3][3] = {
          {(1 - x0) * (1 - x0) / 2, (-2 * x0 * x0 + 2 * x0 + 1) / 2, x0 * x0 / 2},
          {-(1 - x0), 1 - 2 * x0, x0},
          {1, -2, 1}};
      for (std::size_t k = 0; k < 3; ++k) {
        for (std::size_t j = 0; j < 3; ++j) {
          expect_near(static_cast<double>(d.values.data()[k * 3 + j]), want[k][j],
                      16.0 * eps_of<double>());
        }
      }
    }
  }
}

TEST_CASE("space1d: Bezier extrapolation is the Bernstein polynomial outside the unit interval") {
  const std::vector<double> knots = {0, 0, 0, 1, 1, 1};
  const bs::BsplineSpace1D<double> s(sp(knots), 2);
  const std::vector<double> pts = {-1.0, 2.0};
  const auto [v, f] = s.tabulate_basis<double>(pts, false);
  // s = -1: ((1-s)^2, 2s(1-s), s^2) = (4, -4, 1); s = 2: (1, -4, 4). Both sum to one.
  const double want[2][3] = {{4, -4, 1}, {1, -4, 4}};
  for (std::size_t i = 0; i < 2; ++i) {
    CHECK(f.data()[i] == 0);
    for (std::size_t j = 0; j < 3; ++j) {
      expect_near(v.data()[i * 3 + j], want[i][j], 8.0 * eps_of<double>());
    }
  }
}

TEST_CASE("space1d: a NaN or infinite point with validate == false is evaluated, not undefined") {
  // No result is promised for these points. The span they land on is: a NaN and +inf on the last
  // span, -inf on the first, and nothing may read outside the knots (the sanitizer builds run it).
  const std::vector<double> knots = {0, 0, 0, 1, 2, 2, 3, 3, 3};
  const bs::BsplineSpace1D<double> s(sp(knots), 2);
  const std::vector<double> pts = {kNaN, kInf, -kInf};
  const auto [v, f] = s.tabulate_basis<double>(pts, false);
  CHECK(f.data()[0] == 3);
  CHECK(f.data()[1] == 3);
  CHECK(f.data()[2] == 0);
  CHECK(std::isnan(v.data()[0]));
  const auto d = s.tabulate_basis_derivatives<double>(pts, 2, false);
  CHECK(d.first.data()[0] == 3);
  CHECK(d.first.data()[2] == 0);
  CHECK_THROWS_AS(static_cast<void>(s.tabulate_basis<double>(pts, true)), pantr::value_error);
}

// ---- mixed knot and point types ---------------------------------------------------------------

TEST_CASE("space1d: a tie between a point and a knot is decided in the wider type") {
  // Float knots, double points: the point is one double below the knot 0.3f, so its span is the
  // one left of the knot; compared as floats it would equal the knot and take the span to its
  // right (first index 1 instead of 0).
  {
    const float u = 0.3F;
    const std::vector<float> knots = {0.0F, 0.0F, u, 1.0F, 1.0F};
    const bs::BsplineSpace1D<float> s(sp(knots), 1);
    const double below = std::nextafter(static_cast<double>(u), -kInf);
    const std::vector<double> pts = {below, static_cast<double>(u)};
    const auto [v, f] = s.tabulate_basis<double>(pts, false);
    CHECK(f.data()[0] == 0);
    CHECK(f.data()[1] == 1);
  }
  // Double knots, float points: the knot is 0.3f plus 2^-40, which is 0.3f when rounded to float
  // but above the float point in double; the point is left of the knot (first index 0).
  {
    const float x = 0.3F;
    const double u = static_cast<double>(x) + 0x1p-40;
    REQUIRE(static_cast<float>(u) == x);
    const std::vector<double> knots = {0.0, 0.0, u, 1.0, 1.0};
    const bs::BsplineSpace1D<double> s(sp(knots), 1);
    const std::vector<float> pts = {x};
    const auto [v, f] = s.tabulate_basis<float>(pts, false);
    CHECK(f.data()[0] == 0);
  }
}

// ---- the status of a zero divisor -------------------------------------------------------------

TEST_CASE("space1d: a zero divisor in the derivative recurrence is reported, not returned") {
  // [0, 0, 0, 0, 1, 1, 1] at degree 2 has a left end of multiplicity 4 = p + 2, so the span just
  // left of 0 is empty and the recurrence divides by a zero knot difference. The point -1e-17 is
  // within the knot tolerance of the domain, so it passes validation: the status is reachable
  // without validate == false. (Its values are all zero there, a separate matter.)
  const std::vector<double> knots = {0, 0, 0, 0, 1, 1, 1};
  const bs::BsplineSpace1D<double> s(sp(knots), 2);
  const std::vector<double> bad = {-1e-17};
  REQUIRE(s.contains<double>(-1e-17));
  CHECK_FALSE(s.tabulate_basis_derivatives<double>(bad, 1, false).complete);
  CHECK_FALSE(s.tabulate_basis_derivatives<double>(bad, 1, true).complete);
  CHECK_FALSE(s.tabulate_basis_derivatives<double>(bad, 3, false).complete);
  // No division happens for order 0, so nothing to report.
  CHECK(s.tabulate_basis_derivatives<double>(bad, 0, false).complete);
  // The value recurrence drops its zero denominators (the empty span) instead of dividing: its
  // result there is finite and within [0, 1], whatever else it is (it is all zeros today, which
  // is not pinned: the partition of unity does not hold at such a point).
  const auto [v, f] = s.tabulate_basis<double>(bad, false);
  for (std::size_t j = 0; j < 3; ++j) {
    CAPTURE(j);
    CHECK(std::isfinite(v.data()[j]));
    CHECK(v.data()[j] >= 0.0);
    CHECK(v.data()[j] <= 1.0);
  }
  // The out overload returns the same status.
  NdArray<double> out({1, 2, 3});
  NdArray<std::int64_t> first({1});
  CHECK_FALSE(s.tabulate_basis_derivatives<double>(bad, 1, out.view<3>(), first.flat(), false));
  // In-domain points are complete, up to and including the left end where the span starts.
  const std::vector<double> good = {0.0, 0.25, 1.0};
  CHECK(s.tabulate_basis_derivatives<double>(good, 3, true).complete);
}

TEST_CASE("space1d: the incomplete status survives a batch run on the pool") {
  // 20000 points at degree 2 and order 1 cost more than the serial threshold, so the batch is
  // chunked over the pool; one bad point in the middle must still make the status false, and a
  // batch without one must be complete.
  constexpr std::size_t n = 20000;
  const std::vector<double> knots = {0, 0, 0, 0, 1, 1, 1};
  const bs::BsplineSpace1D<double> s(sp(knots), 2);
  REQUIRE(pantr::parallel::worth_parallel(
      n, pantr::parallel::WorkEstimate{bs::detail::derivs_ops_per_point(2, 1)}));
  std::vector<double> pts(n);
  for (std::size_t i = 0; i < n; ++i) {
    pts[i] = static_cast<double>(i) / static_cast<double>(n - 1);
  }
  CHECK(s.tabulate_basis_derivatives<double>(pts, 1, true).complete);
  for (const std::size_t at : {std::size_t{0}, n / 2, n - 1}) {
    std::vector<double> with_bad = pts;
    with_bad[at] = -1e-17;
    CAPTURE(at);
    CHECK_FALSE(s.tabulate_basis_derivatives<double>(with_bad, 1, false).complete);
  }
}

// ---- thread-count invariance and concurrent use -----------------------------------------------

TEST_CASE("space1d: the result does not depend on the thread count") {
  // Chunks are a function of the problem only (design/threading.md) and no chunk reduces across
  // points, so one thread and the whole pool must agree: first indices exactly, values within the
  // rounding bound of two evaluations (each within 8 p eps of the exact value, relative to 1).
  constexpr std::size_t n = 30000;
  const std::vector<double> knots = {0, 0, 0, 0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4};
  const bs::BsplineSpace1D<double> s(sp(knots), 3);
  const int p = 3;
  REQUIRE(pantr::parallel::worth_parallel(
      n, pantr::parallel::WorkEstimate{bs::detail::derivs_ops_per_point(p, 2)}));
  std::vector<double> pts(n);
  std::uint64_t state = 12345;
  for (double& x : pts) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    x = 4.0 * static_cast<double>(state >> 11) / 9007199254740992.0;
  }
  const std::size_t all = pantr::parallel::max_threads();
  WARN_MESSAGE(all >= 2, "one CPU: the pooled run is the serial one, so this case checks nothing");
  const auto run = [&](std::size_t threads) {
    const ThreadsGuard guard(threads);
    return std::make_pair(s.tabulate_basis<double>(pts, true),
                          s.tabulate_basis_derivatives<double>(pts, 2, true));
  };
  const auto serial = run(1);
  const auto pooled = run(all);
  REQUIRE(serial.first.first.size() == pooled.first.first.size());
  CHECK(std::equal(serial.first.second.data(), serial.first.second.data() + n,
                   pooled.first.second.data()));
  CHECK(std::equal(serial.second.first.data(), serial.second.first.data() + n,
                   pooled.second.first.data()));
  CHECK(serial.second.complete);
  CHECK(pooled.second.complete);
  double worst_value = 0.0;
  for (std::size_t i = 0; i < serial.first.first.size(); ++i) {
    worst_value = (std::max)(
        worst_value, std::fabs(serial.first.first.data()[i] - pooled.first.first.data()[i]));
  }
  CHECK(worst_value <= 16.0 * (p + 1) * eps_of<double>());
  // Derivative rows: absolute bound (2 p / h)^k (p + 1) eps with h = 1, doubled for two runs.
  const std::size_t rows = 3;
  const std::size_t ord = 4;
  for (std::size_t k = 0; k < rows; ++k) {
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < ord; ++j) {
        const std::size_t at = (i * rows + k) * ord + j;
        worst = (std::max)(
            worst, std::fabs(serial.second.values.data()[at] - pooled.second.values.data()[at]));
      }
    }
    CAPTURE(k);
    CHECK(worst <= 32.0 * (p + 1) * eps_of<double>() * std::pow(2.0 * p, static_cast<double>(k)));
  }
}

TEST_CASE("space1d: copies share one immutable representation, safe to use from several threads") {
  const auto knots = bs::cardinal_knots<double>(6, 3);
  const bs::BsplineSpace1D<double> s(sp(knots), 3, true);
  const bs::BsplineSpace1D<double> copy = s;
  CHECK(copy.knots().data() == s.knots().data());  // a reference-count increment, not a copy
  std::vector<double> pts(2000);
  for (std::size_t i = 0; i < pts.size(); ++i) {
    pts[i] = 6.0 * static_cast<double>(i) / static_cast<double>(pts.size() - 1);
  }
  const auto ref = s.tabulate_basis<double>(pts);
  std::vector<int> ok(4, 0);
  std::vector<std::thread> threads;
  for (std::size_t t = 0; t < ok.size(); ++t) {
    threads.emplace_back([&, t] {
      const bs::BsplineSpace1D<double> mine = s;  // copies and destroys concurrently
      const auto got = mine.tabulate_basis<double>(pts);
      ok[t] =
          std::equal(ref.first.data(), ref.first.data() + ref.first.size(), got.first.data()) &&
          std::equal(ref.second.data(), ref.second.data() + ref.second.size(), got.second.data());
    });
  }
  for (std::thread& th : threads) {
    th.join();
  }
  for (const int v : ok) {
    CHECK(v == 1);
  }
}

// ---- the highest derivative of a high-degree cardinal B-spline --------------------------------

namespace {

/// The value of the `p`-th derivative of the cardinal B-spline of degree `p` on its piece `j`:
/// `(-1)^j C(p, j)` (the `p`-th derivative is the `p`-th difference of the indicator of a unit
/// interval). Local function `m` at a point of a unit interval starts `p - m` pieces before it.
double cardinal_top_derivative(int p, int m) {
  double c = 1.0;
  for (int i = 1; i <= m; ++i) {
    c = c * (p - m + i) / i;
  }
  return (((p - m) % 2) == 0 ? 1.0 : -1.0) * c;
}

void check_top_derivative_of_cardinal(int p) {
  // Absolute bound: the coefficients of the recurrence are at most 2^p with unit knot spacing and
  // the sum has p + 1 terms, so the rounding error is (p + 1) 2^p eps, about 2e-8 at p = 22;
  // 1e-6 is that with a margin of 50.
  const auto knots = bs::cardinal_knots<double>(2 * p, p);
  const bs::BsplineSpace1D<double> s(sp(knots), p, true);
  const std::vector<double> pts = {static_cast<double>(p) + 0.5};
  const auto d = s.tabulate_basis_derivatives<double>(pts, p, false);
  REQUIRE(d.complete);
  const auto ord = static_cast<std::size_t>(p) + 1;
  for (int m = 0; m <= p; ++m) {
    CAPTURE(p);
    CAPTURE(m);
    CHECK(
        std::fabs(d.values.data()[static_cast<std::size_t>(p) * ord + static_cast<std::size_t>(m)] -
                  cardinal_top_derivative(p, m)) <= 1e-6);
  }
}

}  // namespace

TEST_CASE(
    "space1d: the p-th derivative of a cardinal B-spline is the binomial differences (p = 20)") {
  check_top_derivative_of_cardinal(20);
}

// Bug (present in the previous release as well): the row of order k is scaled by the 64-bit
// integer p! / (p - k)!, which wraps past 2^63 (from p = 21 on, for the top rows), so the rows of
// the orders that overflow are wrong by a large factor without any error. Marked as an expected
// failure so the suite stays green; when the scaling is fixed this case fails and must be unmarked.
TEST_CASE("space1d: the p-th derivative of a cardinal B-spline at p = 22" *
          doctest::should_fail()) {
  check_top_derivative_of_cardinal(22);
}

TEST_CASE("space1d: restriction keeps the functions, for every interval range of the table") {
  // The windowed space's functions on its domain are the parent's functions of the mapped
  // indices: same values, first index shifted by the window's first function. Compared at the left
  // end, the midpoint and a point at 3/4 of each of the window's intervals; the window's right end
  // is excluded, where the parent takes the span to its right.
  for (const Case& c : cases()) {
    if (c.periodic) {
      continue;
    }
    CAPTURE(std::string(c.name));
    const bs::BsplineSpace1D<double> s(sp(c.knots), c.degree);
    const long long n_int = s.num_intervals();
    for (long long lo = 0; lo < n_int; ++lo) {
      for (long long hi = lo + 1; hi <= n_int; ++hi) {
        CAPTURE(lo);
        CAPTURE(hi);
        const auto [w, map] = s.restrict(lo, hi);
        REQUIRE(map.size() == static_cast<std::size_t>(w.num_basis()));
        CHECK(w.num_intervals() == hi - lo);
        const auto wk = w.unique_knots_and_multiplicity(true).first;
        std::vector<double> pts;
        for (std::size_t i = 0; i + 1 < wk.size(); ++i) {
          pts.push_back(wk[i]);
          pts.push_back(0.5 * (wk[i] + wk[i + 1]));
          pts.push_back(wk[i] + 0.75 * (wk[i + 1] - wk[i]));
        }
        const auto [vw, fw] = w.tabulate_basis<double>(pts);
        const auto [vp, fp] = s.tabulate_basis<double>(pts);
        for (std::size_t i = 0; i < pts.size(); ++i) {
          CHECK(fw.data()[i] + map[0] == fp.data()[i]);
          for (std::size_t j = 0; j < static_cast<std::size_t>(c.degree) + 1; ++j) {
            const std::size_t at = i * (static_cast<std::size_t>(c.degree) + 1) + j;
            // Absolute 8 (p + 1) eps: a one-interval window takes the Bernstein path (see
            // check_values), the parent perhaps not.
            CHECK(std::fabs(vw.data()[at] - vp.data()[at]) <=
                  8.0 * (c.degree + 1) * eps_of<double>());
          }
        }
      }
    }
  }
}

// ---- changes of units -------------------------------------------------------------------------

namespace {

/// The knots of a case mapped by `x -> 2^e x + t`.
std::vector<double> mapped(const std::vector<double>& knots, int e, double t) {
  std::vector<double> out;
  out.reserve(knots.size());
  for (const double k : knots) {
    out.push_back(std::ldexp(k, e) + t);
  }
  return out;
}

}  // namespace

TEST_CASE("space1d: a power-of-two scaling and an exact shift change the units and nothing else") {
  // Every knot and point below has few significant bits, so x -> 2^e x + t is exact, and so is
  // every difference the recurrence forms and every quotient by a power of two: the first indices
  // and values are bit for bit the original's, the derivative of order k is the original's times
  // 2^(-e k), the counts and flags are equal. Scaled by 2^-30 the domain has length 1e-9, where a
  // tolerance floored at 1 (or an absolute one) would take every knot for one knot; shifted by
  // 256 the tolerance grows (it follows the largest coordinate) but stays far below the gaps.
  for (const Case& c : cases()) {
    if (c.periodic) {
      continue;
    }
    for (const int e : {-30, 20}) {
      for (const double t : {0.0, 256.0}) {
        CAPTURE(std::string(c.name));
        CAPTURE(e);
        CAPTURE(t);
        const bs::BsplineSpace1D<double> base(sp(c.knots), c.degree);
        const std::vector<double> moved_knots = mapped(c.knots, e, t);
        const bs::BsplineSpace1D<double> moved(sp(moved_knots), c.degree);
        CHECK(moved.num_basis() == base.num_basis());
        CHECK(moved.num_intervals() == base.num_intervals());
        CHECK(moved.has_open_knots() == base.has_open_knots());
        CHECK(moved.has_bezier_like_knots() == base.has_bezier_like_knots());
        if (t == 0.0) {
          CHECK(moved.tolerance() == std::ldexp(base.tolerance(), e));
        }
        // The domain's distinct knots and the midpoints between them (exact in both).
        const auto uk = base.unique_knots_and_multiplicity(true).first;
        std::vector<double> pts;
        for (std::size_t i = 0; i < uk.size(); ++i) {
          pts.push_back(uk[i]);
          if (i + 1 < uk.size()) {
            pts.push_back(0.5 * (uk[i] + uk[i + 1]));
          }
        }
        std::vector<double> moved_pts = mapped(pts, e, t);
        const int n_deriv = c.degree + 1;
        const auto d0 = base.tabulate_basis_derivatives<double>(pts, n_deriv);
        const auto d1 = moved.tabulate_basis_derivatives<double>(moved_pts, n_deriv);
        REQUIRE(d0.values.size() == d1.values.size());
        CHECK(d0.complete);
        CHECK(d1.complete);
        CHECK(std::equal(d0.first.data(), d0.first.data() + d0.first.size(), d1.first.data()));
        const std::size_t block = static_cast<std::size_t>(c.degree) + 1;
        const std::size_t rows = static_cast<std::size_t>(n_deriv) + 1;
        for (std::size_t i = 0; i < pts.size(); ++i) {
          for (std::size_t k = 0; k < rows; ++k) {
            const double factor = std::ldexp(1.0, -e * static_cast<int>(k));
            for (std::size_t j = 0; j < block; ++j) {
              const std::size_t at = (i * rows + k) * block + j;
              CHECK(d1.values.data()[at] == d0.values.data()[at] * factor);
            }
          }
        }
      }
    }
  }
}
