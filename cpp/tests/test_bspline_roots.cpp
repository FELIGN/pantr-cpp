// Root finding for scalar B-splines (Morken-Reimers knot insertion): the kernels, the tolerances
// and the validation of `find_roots`.
//
// Oracles. Each section names its own; the ones that recur are:
//
//  * Encoded truth. A polynomial is built from its roots, `P(t) = prod (t - r_i)`, in exact
//    rational arithmetic, and its B-spline coefficients on any knot vector are read off by the
//    blossom (`c_j = sum_i a_i e_i(t_{j+1}..t_{j+p}) / C(p, i)`, `a_i` the power coefficients and
//    `e_i` the elementary symmetric polynomial of the knot window), then rounded ONCE to the
//    storage type. No root finder and no library evaluation is involved, so the roots are known
//    before the call. Rounding the coefficients once perturbs the spline from `P` by at most
//    `u_T * max|c|` in the supremum norm (the basis is a partition of unity), `u_T` the unit
//    roundoff of the storage type `T`.
//  * Exact evaluation. The exact rational value of a stored spline at a double
//    (bspline_exact_eval.hpp, from the recursive definition of the basis, not the triangle the
//    library runs), against which the library's own evaluations are compared.
//  * The residual certificate. `find_roots` returns a value only if `|f(x)| <= root_tol` on its
//    refined working window. So `|P(x)|` of every returned `x` is bounded by
//        tau = RS * root_tol + E_stored + E_eval (+ E_open for an input that was converted),
//    with `E_stored = 1.01 u_T max|c|` (the one rounding of the coefficients above), `E_eval =
//    1.5 p (p + 1) eps max|c|` (de Boor evaluation in double: `p (p + 1) / 2` steps, each a convex
//    combination whose weight `alpha` and lerp carry six roundings of at most `u` against operands
//    of size at most `max|c|`, and a convex combination does not amplify what it is given, so
//    `3 p (p + 1) u max|c|` with `u = eps / 2`), `E_open = 6 u_T k max|c|` for the `k` knots a
//    conversion to open form inserts (the same six roundings per Boehm step, summed over the
//    steps), and `RS = 8` the factor between `root_tol`, which bounds the residual on the refined
//    window, and the stored spline: the window carries the rounding of up to a few hundred Boehm
//    insertions, damped by the smallness of the coefficients near a zero. `RS` and the `6` of
//    `E_open` are ADMITTED HEURISTICS, not derived; `RS = 8` is the factor `_RESIDUAL_SAFETY` of
//    the Python suite for the same gap. A simple root `r` then lies within `2 tau / |P'(r)|` of the
//    returned `x` (first order: the quadratic remainder is `O(tau^2 |P''| / P'^2)` of it and
//    negligible while `tau << |P'| / |P''|`, the factor 2 covering it); a root of multiplicity `m`
//    of `(t - r)^m` within `tau^(1/m)` (no remainder at all: `|P(x)| = |x - r|^m`).
//  * The Python implementation (`tests/test_bspline_roots.py` before the port) as a second
//    implementation: the budget cases take their inputs and outcomes from it, and the cases
//    ported from it say so by name.
#include <doctest/doctest.h>

#include "bspline_test_support.hpp"

#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/roots.hpp>
#include <pantr/bspline/space.hpp>
#include <pantr/bspline/space_1d.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/core/exact_rational.hpp>
#include <pantr/tolerance.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace bs = pantr::bspline;
namespace bsd = pantr::bspline::detail;
namespace oracle = bspline_oracle;
using bspline_test::cast_all;
using bspline_test::clamped;
using bspline_test::Dir;
using bspline_test::make_spline;
using bspline_test::sp;
using oracle::Integer;
using oracle::Rational;

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

/// The factor between the residual threshold on the refined window and the stored spline
/// (admitted heuristic, see the file header).
constexpr double kWindowSafety = 8.0;

Rational ex(double x) { return oracle::exact(x); }

double to_dbl(const Rational& q) { return pantr::exact::to_double(q); }

Rational abs_of(const Rational& q) { return q < 0 ? Rational(-q) : q; }

Rational rat(std::int64_t num, std::int64_t den) { return pantr::exact::make_rational(num, den); }

/// `numpy.linspace(0, 1, n)`: `i * (1 / (n - 1))`, the last element exactly 1.
std::vector<double> linspace01(int n) {
  std::vector<double> v(static_cast<std::size_t>(n));
  const double step = 1.0 / static_cast<double>(n - 1);
  for (std::size_t i = 0; i < v.size(); ++i) {
    v[i] = static_cast<double>(i) * step;
  }
  v.back() = 1.0;
  return v;
}

/// An open knot vector from breakpoints: `degree + 1` copies of the ends, each interior
/// breakpoint repeated `mult[i]` times (once if `mult` is empty).
std::vector<double> open_knots(int degree, const std::vector<double>& breaks,
                               const std::vector<int>& mult = {}) {
  std::vector<double> k(static_cast<std::size_t>(degree) + 1, breaks.front());
  for (std::size_t i = 1; i + 1 < breaks.size(); ++i) {
    const int reps = mult.empty() ? 1 : mult[i - 1];
    k.insert(k.end(), static_cast<std::size_t>(reps), breaks[i]);
  }
  k.insert(k.end(), static_cast<std::size_t>(degree) + 1, breaks.back());
  return k;
}

// --- Exact polynomials ---------------------------------------------------------------------

/// Power-basis coefficients, ascending.
using Poly = std::vector<Rational>;

/// `prod (t - r)` over the roots, padded with zero leading coefficients to `degree + 1` entries.
Poly poly_from_roots(const std::vector<Rational>& roots, std::size_t degree) {
  Poly p{Rational(1)};
  for (const Rational& r : roots) {
    Poly q(p.size() + 1, Rational(0));
    for (std::size_t i = 0; i < p.size(); ++i) {
      q[i + 1] += p[i];
      q[i] -= r * p[i];
    }
    p = std::move(q);
  }
  p.resize(degree + 1, Rational(0));
  return p;
}

Rational poly_eval(const Poly& p, const Rational& x) {
  Rational acc = 0;
  for (std::size_t i = p.size(); i-- > 0;) {
    acc = acc * x + p[i];
  }
  return acc;
}

Poly poly_derivative(const Poly& p) {
  Poly d;
  for (std::size_t i = 1; i < p.size(); ++i) {
    d.push_back(Rational(static_cast<std::int64_t>(i)) * p[i]);
  }
  return d;
}

Integer binom(unsigned n, unsigned k) {
  Integer r = 1;
  for (unsigned i = 1; i <= k; ++i) {
    r = r * (n - k + i) / i;
  }
  return r;
}

/// The blossom of the polynomial at a window of `degree` knots.
Rational blossom(const Poly& a, std::span<const double> window) {
  const std::size_t p = a.size() - 1;
  std::vector<Rational> e(p + 1, Rational(0));
  e[0] = 1;
  for (const double u : window) {
    const Rational ue = ex(u);
    for (std::size_t i = p; i >= 1; --i) {
      e[i] += e[i - 1] * ue;
    }
  }
  Rational acc = 0;
  for (std::size_t i = 0; i <= p; ++i) {
    acc += a[i] * e[i] / Rational(binom(static_cast<unsigned>(p), static_cast<unsigned>(i)));
  }
  return acc;
}

/// The double nearest to a rational (ties to even).
double round_to_double(const Rational& q) {
  if (q == 0) {
    return 0.0;
  }
  const bool negative = q < 0;
  const Rational a = abs_of(q);
  const double v = pantr::exact::correctly_rounded_quotient(boost::multiprecision::numerator(a),
                                                            boost::multiprecision::denominator(a));
  return negative ? -v : v;
}

/// The B-spline coefficients of the polynomial on a knot vector, each rounded once to double.
std::vector<double> blossom_coefficients(const Poly& a, const std::vector<double>& knots,
                                         int degree) {
  const auto p = static_cast<std::size_t>(degree);
  std::vector<double> c(knots.size() - p - 1);
  for (std::size_t j = 0; j < c.size(); ++j) {
    c[j] = round_to_double(blossom(a, std::span<const double>(knots).subspan(j + 1, p)));
  }
  return c;
}

/// A scalar spline with the given knots (not snapped) and coefficients, rational if weights are
/// given (then the control points are `(coefficient, weight)`).
template <class T>
bs::Bspline<T> make_scalar(const std::vector<double>& knots, int degree,
                           const std::vector<double>& coeffs,
                           const std::vector<double>& weights = {}) {
  const std::vector<T> k = cast_all<T>(knots);
  std::vector<bs::BsplineSpace1D<T>> directions;
  directions.emplace_back(sp(k), degree, false, false);
  bs::BsplineSpace<T> space(std::move(directions));
  std::vector<T> cp;
  for (std::size_t i = 0; i < coeffs.size(); ++i) {
    cp.push_back(static_cast<T>(coeffs[i]));
    if (!weights.empty()) {
      cp.push_back(static_cast<T>(weights[i]));
    }
  }
  return bs::Bspline<T>(std::move(space), sp(cp), !weights.empty());
}

template <class T>
double unit_roundoff() {
  return 0.5 * static_cast<double>(std::numeric_limits<T>::epsilon());
}

/// The largest absolute coefficient as stored in `T`.
template <class T>
double stored_max(const std::vector<double>& coeffs) {
  double m = 0.0;
  for (const double c : coeffs) {
    m = (std::max)(m, std::abs(static_cast<double>(static_cast<T>(c))));
  }
  return m;
}

/// An exactly known function: `P((t - origin) / span)` with the roots of `P` listed.
struct Truth {
  Poly power;
  std::vector<Rational> roots;  // of P, ascending and distinct
  Rational origin = 0;
  Rational span = 1;

  Rational value(const Rational& t) const { return poly_eval(power, (t - origin) / span); }
  Rational root(std::size_t i) const { return origin + span * roots[i]; }
  double slope_at_root(std::size_t i) const {
    return std::abs(to_dbl(poly_eval(poly_derivative(power), roots[i]) / span));
  }
};

/// `tau` of the file header.
template <class T>
double tau_of(const bs::BsplineRootSet& got, double cmax, int degree, double extra = 0.0) {
  const double e_stored = 1.01 * unit_roundoff<T>() * cmax;
  const double e_eval = 1.5 * degree * (degree + 1) * kEps * cmax;
  return kWindowSafety * got.root_tol + e_stored + e_eval + extra;
}

/// Check that `got` holds exactly the simple roots of `truth`, each within the bound of the file
/// header, and that each returned value satisfies the residual certificate against the exact `P`.
void expect_simple_roots(const bs::BsplineRootSet& got, const Truth& truth, double tau) {
  REQUIRE(got.roots.size() == truth.roots.size());
  CHECK(got.abandoned == 0);
  for (std::size_t i = 0; i < got.roots.size(); ++i) {
    CAPTURE(i);
    if (i > 0) {
      CHECK(got.roots[i - 1] < got.roots[i]);
    }
    CHECK(std::abs(to_dbl(truth.value(ex(got.roots[i])))) <= tau);
    const double bound = 2.0 * tau / truth.slope_at_root(i);
    CHECK(std::abs(got.roots[i] - to_dbl(truth.root(i))) <= bound);
  }
}

/// One encoded-truth case: roots, interior knots, and the affine image of the domain.
struct PolyCase {
  int degree;
  std::vector<Rational> roots;
  std::vector<double> interior;
  double origin = 0.0;
  double span = 1.0;
};

const std::vector<PolyCase>& poly_cases() {
  static const std::vector<PolyCase> cases = {
      // A single Bezier segment: the roots of the polynomial itself.
      {3, {rat(1, 4), rat(1, 2), rat(7, 8)}, {}},
      // Linear pieces; a root between knots.
      {1, {rat(1, 2)}, {0.25, 0.75}},
      {2, {rat(1, 4), rat(3, 4)}, {0.5}},
      // A root exactly on an interior knot (1/2), and the knot set refined around it.
      {3, {rat(1, 4), rat(1, 2), rat(7, 8)}, {0.125, 0.375, 0.5, 0.625}},
      {4, {rat(1, 8), rat(3, 8), rat(5, 8), rat(7, 8)}, {0.25, 0.5, 0.75}},
      {5, {rat(1, 8), rat(1, 4), rat(1, 2), rat(3, 4), rat(7, 8)}, {0.375, 0.625}},
      // Both domain endpoints are roots: an open knot vector interpolates its end coefficients.
      {2, {rat(0, 1), rat(1, 1)}, {0.5}},
      // The knots moved and scaled, the coefficients untouched: the affinely reparametrized
      // function, so the roots are known at every scale (the tolerances are scale-covariant).
      {3, {rat(1, 4), rat(1, 2), rat(7, 8)}, {0.125, 0.375, 0.625}, 1024.0, 8.0},
      {3, {rat(1, 4), rat(1, 2), rat(7, 8)}, {0.125, 0.375, 0.625}, -5.0, 7.0},
      {3, {rat(1, 4), rat(1, 2), rat(7, 8)}, {0.125, 0.375, 0.625}, 0.0, 1.0 / 1024.0},
  };
  return cases;
}

struct Built {
  std::vector<double> knots;
  std::vector<double> coeffs;
  Truth truth;
};

/// The knots, exact coefficients and truth of a case.
Built build(const PolyCase& c) {
  const std::vector<double> unit = clamped(c.degree, c.interior);
  Built out;
  out.truth.power = poly_from_roots(c.roots, static_cast<std::size_t>(c.degree));
  out.truth.roots = c.roots;
  out.truth.origin = ex(c.origin);
  out.truth.span = ex(c.span);
  out.coeffs = blossom_coefficients(out.truth.power, unit, c.degree);
  out.knots = unit;
  for (double& k : out.knots) {
    k = c.origin + c.span * k;
  }
  return out;
}

/// The coefficients of `(t - 0.2)(t - 0.5)(t - 0.9)` on the knots `open_knots(3, linspace01(6))`
/// as the blossom of the Python suite's `_polynomial_coefficients` produced them in floating
/// point (hex floats), the input of its budget tests.
const std::vector<double>& budget_coefficients() {
  static const std::vector<double> c = {
      -0x1.70a3d70a3d70bp-4, -0x1.529a485cd7b92p-5, 0x1.b4e81b4e81b4cp-7, 0x1.f671529a485c0p-7,
      -0x1.e098ead65b7c0p-7, -0x1.d5acb6f4650e0p-6, 0x1.31d5acb6f4600p-8, 0x1.47ae147ae1470p-5};
  return c;
}

std::vector<double> budget_knots() { return open_knots(3, linspace01(6)); }

}  // namespace

// --- The public result, against encoded truth ------------------------------------------------

// Oracle: encoded truth. The roots are chosen first (multiples of 1/8), the coefficients follow.
TEST_CASE_TEMPLATE("find_roots: returns the roots of an exactly encoded polynomial", T, double,
                   float) {
  for (const PolyCase& c : poly_cases()) {
    CAPTURE(c.degree);
    CAPTURE(c.origin);
    const Built b = build(c);
    const bs::Bspline<T> spline = make_scalar<T>(b.knots, c.degree, b.coeffs);
    const bs::BsplineRootSet got = bs::find_roots<T>(spline);
    CHECK(got.status == bs::RootStatus::complete);
    CHECK(got.truncated == 0);
    expect_simple_roots(got, b.truth, tau_of<T>(got, stored_max<T>(b.coeffs), c.degree));
    CHECK(got.budget == bs::max_insertions(c.degree));
  }
}

// Oracle: encoded truth, on a domain moved to 2^20. A root located to the coordinate's own
// resolution (`ulp(2^20) = 2^-32`) leaves `|f'|` times that behind however exactly the spline is
// evaluated, and `zero_tol` alone, which is the evaluation error, would reject it: the
// `stop_slack * slope * tol * scale` term of `root_tol` is what accepts it. Double only: in float
// the resolution `tol * scale` would exceed the spacing of the roots.
// ported from tests/test_bspline_roots.py::test_roots_follow_an_affine_reparametrization
TEST_CASE("find_roots: a domain far from the origin keeps its roots") {
  const PolyCase c{3, {rat(1, 4), rat(1, 2), rat(7, 8)}, {0.125, 0.375, 0.625}, 1048576.0, 1.0};
  const Built b = build(c);
  const bs::BsplineRootSet got = bs::find_roots<double>(make_scalar<double>(b.knots, 3, b.coeffs));
  CHECK(got.status == bs::RootStatus::complete);
  expect_simple_roots(got, b.truth, tau_of<double>(got, stored_max<double>(b.coeffs), 3));
  // The stop term is what made room: `root_tol` exceeds the evaluation error by far.
  CHECK(got.root_tol > 1e3 * 16.0 * kEps * stored_max<double>(b.coeffs));
}

// Oracle: encoded truth. A rational spline reports the zeros of its numerator, the weights being
// positive: the numerator is the first control-point column, the weights are arbitrary.
TEST_CASE_TEMPLATE("find_roots: a rational spline reports the zeros of its numerator", T, double,
                   float) {
  const PolyCase c{3, {rat(1, 4), rat(1, 2), rat(7, 8)}, {0.125, 0.375, 0.5, 0.625}};
  const Built b = build(c);
  std::vector<double> weights;
  for (std::size_t j = 0; j < b.coeffs.size(); ++j) {
    weights.push_back(1.0 + static_cast<double>(j) / 8.0);
  }
  const bs::Bspline<T> spline = make_scalar<T>(b.knots, c.degree, b.coeffs, weights);
  REQUIRE(spline.rational());
  const bs::BsplineRootSet got = bs::find_roots<T>(spline);
  expect_simple_roots(got, b.truth, tau_of<T>(got, stored_max<T>(b.coeffs), c.degree));
}

// Oracle: encoded truth, on an unclamped knot vector (the blossom represents the polynomial on any
// knot vector). The input is converted to open form first, by inserting `degree` knots at each end
// of the domain `[3, 5]`; the conversion's rounding is `E_open`.
TEST_CASE_TEMPLATE("find_roots: an unclamped input is converted to open form first", T, double,
                   float) {
  const int degree = 3;
  const std::vector<double> knots = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  Truth truth;
  truth.roots = {rat(13, 4), rat(4, 1), rat(19, 4)};
  truth.power = poly_from_roots(truth.roots, static_cast<std::size_t>(degree));
  const std::vector<double> coeffs = blossom_coefficients(truth.power, knots, degree);
  const bs::Bspline<T> spline = make_scalar<T>(knots, degree, coeffs);
  REQUIRE_FALSE(spline.space().spaces()[0].has_open_knots());

  const bs::BsplineRootSet got = bs::find_roots<T>(spline);
  const double cmax = stored_max<T>(coeffs);
  const double e_open = 6.0 * unit_roundoff<T>() * (2.0 * degree) * cmax;
  expect_simple_roots(got, truth, tau_of<T>(got, cmax, degree, e_open));
}

// Oracle: analytic. A spline whose coefficients have no sign change has no zero; the doc example
// of the Python `find_roots`, a linear spline through (-1, 1, 3) on [0, 2], has its zero at 1/2.
TEST_CASE("find_roots: no sign change means no zero, and a linear spline has its zero") {
  const std::vector<double> knots = open_knots(3, linspace01(6));
  const bs::Bspline<double> positive =
      make_scalar<double>(knots, 3, std::vector<double>(knots.size() - 4, 2.0));
  const bs::BsplineRootSet none = bs::find_roots<double>(positive);
  CHECK(none.roots.empty());
  CHECK(none.candidates == 0);
  CHECK(none.status == bs::RootStatus::complete);

  const bs::Bspline<double> linear =
      make_scalar<double>({0.0, 0.0, 1.0, 2.0, 2.0}, 1, {-1.0, 1.0, 3.0});
  const bs::BsplineRootSet one = bs::find_roots<double>(linear);
  REQUIRE(one.roots.size() == 1);
  // The spline has slope 2, so `|x - 1/2| <= 2 tau / 2`.
  CHECK(std::abs(one.roots[0] - 0.5) <= tau_of<double>(one, 3.0, 1));
}

// Oracle: encoded truth, multiplicity m. `P = (t - 1/2)^m`, so a value with `|P(x)| <= tau` has
// `|x - 1/2| <= tau^(1/m)` exactly. A zero of even multiplicity has no sign change at all and is
// reported through the residual test alone; one zero is reported, not two.
// ported from tests/test_bspline_roots.py::test_a_multiple_root_is_reported_once
TEST_CASE("find_roots: a multiple root is reported once") {
  for (const int m : {2, 3, 4}) {
    CAPTURE(m);
    Truth truth;
    truth.roots = std::vector<Rational>(static_cast<std::size_t>(m), rat(1, 2));
    truth.power = poly_from_roots(truth.roots, static_cast<std::size_t>(m));
    const std::vector<double> knots = open_knots(m, {0.0, 0.25, 0.75, 1.0});
    const std::vector<double> coeffs = blossom_coefficients(truth.power, knots, m);
    const bs::BsplineRootSet got = bs::find_roots<double>(make_scalar<double>(knots, m, coeffs));
    REQUIRE(got.roots.size() == 1);
    const double tau = tau_of<double>(got, stored_max<double>(coeffs), m);
    CHECK(std::abs(to_dbl(truth.value(ex(got.roots[0])))) <= tau);
    CHECK(std::abs(got.roots[0] - 0.5) <= std::pow(tau, 1.0 / static_cast<double>(m)));
  }
}

// Oracle: encoded truth, `P = (t - 1/4)(t - 1/2)^2`. Merging collapses a repeated report of one
// zero, never two distinct zeros. Near the double root `P(x) = (x - 1/4)(x - 1/2)^2`, so `|P(x)|
// <= tau` gives `|x - 1/2| <= sqrt(tau / |x - 1/4|)`, and `|x - 1/4| >= 1/4 - delta` for the
// tiny `delta` of the bound: the 1.1 below covers it (the problem allows no more accuracy, the
// interval where the spline is indistinguishable from zero having this half-width).
// ported from tests/test_bspline_roots.py::test_a_simple_and_a_double_root_stay_separate
TEST_CASE("find_roots: a simple and a double root stay separate") {
  Truth truth;
  truth.roots = {rat(1, 4), rat(1, 2), rat(1, 2)};
  truth.power = poly_from_roots(truth.roots, 3);
  const std::vector<double> knots = open_knots(3, {0.0, 0.25, 0.5, 0.75, 1.0});
  const std::vector<double> coeffs = blossom_coefficients(truth.power, knots, 3);
  const bs::BsplineRootSet got = bs::find_roots<double>(make_scalar<double>(knots, 3, coeffs));
  REQUIRE(got.roots.size() == 2);
  const double tau = tau_of<double>(got, stored_max<double>(coeffs), 3);
  // P'(1/4) = (1/4 - 1/2)^2 = 1/16.
  CHECK(std::abs(got.roots[0] - 0.25) <= 2.0 * tau * 16.0);
  CHECK(std::abs(got.roots[1] - 0.5) <= 1.1 * std::sqrt(tau / 0.25));
}

// Oracle: the residual certificate against the exact stored spline. A clamped degree-15 spline
// whose single interior knot is C^0 and whose coefficients alternate: the kernel reports three
// zeros, and the merge radius cap `domain_length * (degree! * zero_tol / coeff_scale)^(1/degree)`
// passes half the domain, so an uncertified merge would return the midpoint of the three where
// the spline is far from zero. Certifying the midpoint the way the reports are certified keeps
// them apart. Two sign changes do not converge inside the budget, which is a separate matter and
// is what `abandoned` reports.
// ported from tests/test_bspline_roots.py::
//   test_regression_merging_never_invents_a_root_between_two_real_ones
TEST_CASE("find_roots: merging never invents a root between two real ones") {
  const int degree = 15;
  const std::vector<double> knots = open_knots(degree, {0.0, 0.5, 1.0}, {degree});
  std::vector<double> coeffs(knots.size() - static_cast<std::size_t>(degree) - 1);
  for (std::size_t i = 0; i < coeffs.size(); ++i) {
    coeffs[i] = (i % 2 == 1) ? -1.0 : 1.0;
  }
  const bs::Bspline<double> spline = make_scalar<double>(knots, degree, coeffs);

  const bs::BsplineRootSet got = bs::find_roots<double>(spline);

  // Precondition: the cap really is wider than the gaps between the zeros it must not merge.
  double factorial = 1.0;
  for (int k = 2; k <= degree; ++k) {
    factorial *= static_cast<double>(k);
  }
  const double zero_tol = 1.0 * (degree + 1) * 4.0 * kEps;
  CHECK(std::pow(factorial * zero_tol / 1.0, 1.0 / degree) > 0.5);

  CHECK(got.roots.size() == 3);
  CHECK(got.candidates >= 3);
  CHECK(got.abandoned > 0);
  CHECK(got.status == bs::RootStatus::budget_exhausted);
  const double tau = tau_of<double>(got, 1.0, degree);
  for (const double x : got.roots) {
    CAPTURE(x);
    const std::vector<Rational> point = {ex(x)};
    const std::vector<Rational> value = oracle::exact_value<double>(spline, std::span(point));
    CHECK(std::abs(to_dbl(value[0])) <= tau);
  }
}

// Oracle: the exact stored spline, and the knot. A zero of the left piece at a knot of multiplicity
// `degree + 1` (a C^-1 knot) is reported at the knot, though evaluating there returns the right
// limit: the left piece reaches zero at the knot, the right piece jumps to -2. The report is at
// the knot to within the rounding of the Greville average of `degree` equal knots (a running sum
// of `degree` equal terms, then one division: `degree * eps * |knot|`, a heuristic upper bound).
// ported from tests/test_bspline_roots.py::test_a_one_sided_zero_at_a_jump_is_reported_at_the_knot
TEST_CASE("find_roots: a one-sided zero at a jump is reported at the knot") {
  const int degree = 5;
  const double jump_knot = 0.338252602594;
  const std::vector<double> knots = open_knots(
      degree, {0.0, 0.123541192295, 0.338252602594, 0.477533677605, 1.0}, {1, degree + 1, 3});
  const std::vector<double> coeffs = {1.0,  0.5,  -1.0, 1.5, -0.5, 2.0, 0.0,  -2.0,
                                      -1.5, -1.5, -0.5, 0.5, -0.0, 1.0, -1.0, -3.0};
  REQUIRE(coeffs.size() == knots.size() - static_cast<std::size_t>(degree) - 1);
  std::int64_t at_knot = 0;
  for (const double k : knots) {
    at_knot += (k == jump_knot) ? 1 : 0;
  }
  REQUIRE(at_knot == degree + 1);

  const bs::BsplineRootSet got = bs::find_roots<double>(make_scalar<double>(knots, degree, coeffs));
  double nearest = std::numeric_limits<double>::infinity();
  for (const double x : got.roots) {
    nearest = (std::min)(nearest, std::abs(x - jump_knot));
  }
  CHECK(nearest <= degree * kEps * jump_knot);
}

// --- Exhaustion of the insertion budget -------------------------------------------------------

// Oracle: the Python implementation's inputs and outcomes. The budget of eight insertions is
// enough for the residual to certify all three zeros of this spline but not for the iterates of
// the first to stagnate, which is the case `stalled_at_last_iterate` is for. The budget at which
// the two stop coinciding is a property of this spline, found by running it, not reasoned.
// ported from tests/test_bspline_roots.py::test_budget_exhaustion_warns
TEST_CASE("find_roots: a zero whose budget ran out is reported at the last iterate") {
  const bs::Bspline<double> spline = make_scalar<double>(budget_knots(), 3, budget_coefficients());
  const bs::BsplineRootSet got = bs::find_roots<double>(spline, std::nullopt, 8);
  CHECK(got.status == bs::RootStatus::stalled_at_last_iterate);
  CHECK(got.truncated > 0);
  CHECK(got.abandoned == 0);
  CHECK(got.roots.size() == 3);
  CHECK(got.candidates == 3);
  CHECK(got.budget == 8);

  // The budget is the only difference to the default call, which ends complete on the same roots.
  const bs::BsplineRootSet full = bs::find_roots<double>(spline);
  CHECK(full.status == bs::RootStatus::complete);
  CHECK(full.truncated == 0);
  CHECK(full.budget == bs::max_insertions(3));
  CHECK(full.roots.size() == 3);
}

// Oracle: the Python implementation's inputs and outcomes. A budget of one exhausts before any of
// the three zeros is located at all: a sign change whose tracking never converges is dropped
// rather than reported at whatever it reached. That is the sound direction, but it is a *missing*
// root, so it is the one rejection the status names.
// ported from tests/test_bspline_roots.py::
//   test_a_sign_change_the_budget_never_certified_is_not_silently_dropped
TEST_CASE("find_roots: a sign change the budget never certified is not silently dropped") {
  const bs::Bspline<double> spline = make_scalar<double>(budget_knots(), 3, budget_coefficients());
  const bs::BsplineRootSet got = bs::find_roots<double>(spline, std::nullopt, 1);
  CHECK(got.status == bs::RootStatus::budget_exhausted);
  CHECK(got.abandoned > 0);
  CHECK(got.roots.empty());
  CHECK(got.budget == 1);
}

// Oracle: the Python implementation's inputs and outcomes, at kernel level (the budget enters the
// kernel as an argument). Cutting the insertion budget to one is reported rather than silently
// accepted, and the roots vector holds exactly `count` entries. With the budget of the public
// call the kernel agrees with it on the number of candidates.
// ported from tests/test_bspline_roots.py::test_kernel_reports_a_zero_whose_budget_ran_out
TEST_CASE("morken_reimers_roots: reports a zero whose budget ran out") {
  const std::vector<double> knots = budget_knots();
  const std::vector<double> coeffs = budget_coefficients();
  const bs::BsplineRootSet reference =
      bs::find_roots<double>(make_scalar<double>(knots, 3, coeffs));
  // The scales `find_roots` derives for these coefficients: p = 3, so `(p + 1) * 4 eps` beats the
  // strict tolerance `4 eps`.
  const double tol = pantr::tolerance::get_strict<double>();
  const double zero_tol = stored_max<double>(coeffs) * 16.0 * kEps;

  const bsd::MorkenReimersResult one =
      bsd::morken_reimers_roots(knots, 3, coeffs, tol, zero_tol, reference.root_tol, 1);
  CHECK(one.truncated + one.abandoned > 0);
  CHECK(static_cast<std::int64_t>(one.roots.size()) == one.count);

  const bsd::MorkenReimersResult full = bsd::morken_reimers_roots(
      knots, 3, coeffs, tol, zero_tol, reference.root_tol, bs::max_insertions(3));
  CHECK(full.truncated == 0);
  CHECK(full.abandoned == 0);
  CHECK(static_cast<std::int64_t>(full.roots.size()) == full.count);
  CHECK(full.count == reference.candidates);
}

// Oracle: analytic. A budget of zero is allowed (every sign change stops at its first iterate); a
// negative one is refused (the working arrays are sized from it).
TEST_CASE("find_roots: the budget override is checked") {
  const bs::Bspline<double> spline = make_scalar<double>(budget_knots(), 3, budget_coefficients());
  CHECK_NOTHROW(static_cast<void>(bs::find_roots<double>(spline, std::nullopt, 0)));
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::find_roots<double>(spline, std::nullopt, -1)),
                       "max_insertions must be non-negative, got -1", pantr::value_error);
}

// Oracle: analytic. The status codes are wire values, and the budget is `64 + 8 (degree - 1)`.
// ported from tests/test_bspline_roots.py::test_insertion_budget_grows_with_degree
TEST_CASE("RootStatus and max_insertions: the pinned values") {
  static_assert(static_cast<std::int32_t>(bs::RootStatus::complete) == 0);
  static_assert(static_cast<std::int32_t>(bs::RootStatus::budget_exhausted) == 1);
  static_assert(static_cast<std::int32_t>(bs::RootStatus::stalled_at_last_iterate) == 2);
  CHECK(static_cast<std::int32_t>(bs::RootStatus::complete) == 0);
  CHECK(static_cast<std::int32_t>(bs::RootStatus::budget_exhausted) == 1);
  CHECK(static_cast<std::int32_t>(bs::RootStatus::stalled_at_last_iterate) == 2);

  static_assert(bs::max_insertions(1) == 64);
  static_assert(bs::max_insertions(2) == 72);
  static_assert(bs::max_insertions(25) == 256);
  CHECK(bs::max_insertions(25) > 4 * bs::max_insertions(1) / 2);
  CHECK(bs::max_insertions(5) > bs::max_insertions(3));
  CHECK(bs::max_insertions(3) > bs::max_insertions(1));
}

// --- Validation -------------------------------------------------------------------------------

// Oracle: the Python messages, verbatim. The checks run in the order dim, rank, degree, weights,
// then the conversion to open form, then `tol`, then the identically-zero and disconnected tests.
TEST_CASE_TEMPLATE("find_roots: refuses what is not a scalar univariate spline", T, double, float) {
  const bs::Bspline<T> surface =
      make_spline<T>({Dir{clamped(2, {}), 2}, Dir{clamped(2, {}), 2}}, 1, false, 1U);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::find_roots<T>(surface)),
                       "Bspline must be univariate (dim == 1), got dim=2", pantr::value_error);

  const bs::Bspline<T> vector_valued = make_spline<T>({Dir{clamped(2, {0.5}), 2}}, 2, false, 1U);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::find_roots<T>(vector_valued)),
                       "Bspline must be scalar valued (rank == 1), got rank=2", pantr::value_error);
  // A rational spline of rank 2 has three columns.
  const bs::Bspline<T> rational_vector = make_spline<T>({Dir{clamped(2, {0.5}), 2}}, 2, true, 1U);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::find_roots<T>(rational_vector)),
                       "Bspline must be scalar valued (rank == 1), got rank=2", pantr::value_error);

  const bs::Bspline<T> constant = make_spline<T>({Dir{{0.0, 0.5, 1.0}, 0}}, 1, false, 1U);
  const std::string degree_zero =
      "Root finding needs degree >= 1: a degree-zero spline is piecewise constant, so its zero "
      "set is a union of knot intervals, not a set of points";
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::find_roots<T>(constant)), degree_zero.c_str(),
                       pantr::value_error);

  // The earlier check wins over a later one.
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::find_roots<T>(surface, -1.0)),
                       "Bspline must be univariate (dim == 1), got dim=2", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::find_roots<T>(constant, -1.0)), degree_zero.c_str(),
                       pantr::value_error);
}

TEST_CASE_TEMPLATE("find_roots: refuses a rational spline without positive weights", T, double,
                   float) {
  const std::vector<double> knots = clamped(2, {0.5});
  const std::vector<double> coeffs = {1.0, -1.0, 1.0, -1.0};
  const auto message = [](const std::string& min_weight) {
    return "Rational B-splines must have strictly positive weights, so that the zeros of the "
           "numerator are the zeros of the mapping. Got min weight " +
           min_weight;
  };
  const auto with_weights = [&](const std::vector<double>& w) {
    return make_scalar<T>(knots, 2, coeffs, w);
  };
  const std::string zero = message("0.0");
  const std::string negative = message("-0.5");
  const std::string not_a_number = message("nan");
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::find_roots<T>(with_weights({1.0, 0.0, 2.0, 1.0}))),
                       zero.c_str(), pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::find_roots<T>(with_weights({1.0, 2.0, -0.5, 1.0}))),
                       negative.c_str(), pantr::value_error);
  // The minimum propagates a NaN, as `numpy.min` does, and a NaN weight is not positive.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::find_roots<T>(with_weights({1.0, nan, 2.0, 1.0}))),
                       not_a_number.c_str(), pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::find_roots<T>(with_weights({nan, 1.0, 2.0, 1.0}))),
                       not_a_number.c_str(), pantr::value_error);
  // The weights are checked before `tol`.
  CHECK_THROWS_WITH_AS(
      static_cast<void>(bs::find_roots<T>(with_weights({1.0, 0.0, 2.0, 1.0}), -1.0)), zero.c_str(),
      pantr::value_error);
}

TEST_CASE_TEMPLATE("find_roots: refuses a tolerance that is not positive", T, double, float) {
  const bs::Bspline<T> spline = make_scalar<T>(clamped(2, {0.5}), 2, {1.0, -1.0, 1.0, -1.0});
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::find_roots<T>(spline, -1.0)),
                       "tol must be positive, got -1.0", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::find_roots<T>(spline, 0.0)),
                       "tol must be positive, got 0.0", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::find_roots<T>(spline, -0.0)),
                       "tol must be positive, got -0.0", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::find_roots<T>(spline, -1e-05)),
                       "tol must be positive, got -1e-05", pantr::value_error);
  // Only `tol <= 0` raises, as in Python, so a NaN passes; it certifies nothing, since every
  // comparison with it is false, and the threshold it makes is NaN.
  const bs::BsplineRootSet nan_tol =
      bs::find_roots<T>(spline, std::numeric_limits<double>::quiet_NaN());
  CHECK(std::isnan(nan_tol.root_tol));
  // A positive tolerance is used as given.
  CHECK_NOTHROW(static_cast<void>(bs::find_roots<T>(spline, 1e-8)));
}

TEST_CASE_TEMPLATE("find_roots: refuses a spline with no isolated zero set", T, double, float) {
  const std::vector<double> knots = clamped(2, {0.5});
  // Identically zero, either sign of zero; `tol` is checked first.
  CHECK_THROWS_WITH_AS(
      static_cast<void>(bs::find_roots<T>(make_scalar<T>(knots, 2, {0.0, -0.0, 0.0, 0.0}))),
      "The spline is identically zero, so every point of its domain is a root", pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(bs::find_roots<T>(make_scalar<T>(knots, 2, {0.0, 0.0, 0.0, 0.0}), -2.0)),
      "tol must be positive, got -2.0", pantr::value_error);

  // Vanishing on the first knot interval [0, 0.5]: the three coefficients active there are zero.
  CHECK_THROWS_WITH_AS(
      static_cast<void>(bs::find_roots<T>(make_scalar<T>(knots, 2, {0.0, 0.0, 0.0, 1.0}))),
      "The spline vanishes identically on the knot interval [0.0, 0.5], so its zero set is not a "
      "set of isolated points. Split the spline to exclude that interval.",
      pantr::value_error);
  // ... and on the last one, [0.5, 1], after a first interval that does not vanish.
  CHECK_THROWS_WITH_AS(
      static_cast<void>(bs::find_roots<T>(make_scalar<T>(knots, 2, {1.0, 0.0, 0.0, 0.0}))),
      "The spline vanishes identically on the knot interval [0.5, 1.0], so its zero set is not a "
      "set of isolated points. Split the spline to exclude that interval.",
      pantr::value_error);
  // A zero-length interval (the double knot 0.5) is not an interval: the three coefficients that
  // vanish on it are no reason to refuse the spline, whose other two intervals do not vanish.
  CHECK_NOTHROW(static_cast<void>(bs::find_roots<T>(
      make_scalar<T>(open_knots(2, {0.0, 0.5, 1.0}, {2}), 2, {1.0, 0.0, 0.0, 0.0, -1.0}))));
}

// Oracle: analytic. A NaN coefficient makes every comparison false and the scales NaN
// (`numpy.max` propagates it); nothing is reported and nothing is thrown.
TEST_CASE("find_roots: a NaN coefficient reports no root") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const bs::BsplineRootSet got =
      bs::find_roots<double>(make_scalar<double>(clamped(2, {0.5}), 2, {1.0, nan, -1.0, 1.0}));
  CHECK(got.roots.empty());
  CHECK(std::isnan(got.root_tol));
}

// --- Kernel helpers ---------------------------------------------------------------------------

// Oracle: exact rational arithmetic. The Greville abscissa is the mean of the `degree` knots after
// the coefficient's own; their sum of dyadic knots is exact in double, so the one division is the
// only rounding and the result is within `u` of the exact mean.
// ported from tests/test_bspline_roots.py::test_knot_average_is_the_greville_abscissa
TEST_CASE("knot_average: is the Greville abscissa") {
  const int degree = 3;
  const std::vector<double> knots = open_knots(degree, linspace01(5));
  for (std::size_t index = 0; index + degree + 1 < knots.size(); ++index) {
    CAPTURE(index);
    Rational sum = 0;
    for (std::size_t j = index + 1; j <= index + degree; ++j) {
      sum += ex(knots[j]);
    }
    const Rational mean = sum / Rational(degree);
    const double got = bsd::knot_average(knots, degree, static_cast<std::int64_t>(index));
    CHECK(oracle::within(got, mean, 0.5 * kEps * std::abs(to_dbl(mean))));
  }
}

// Oracle: the defining inequality `knots[span] <= point < knots[span + 1]`, and a table.
// ported from tests/test_bspline_roots.py::test_span_at_stays_inside_the_last_span_of_the_domain
TEST_CASE("span_at: stays inside the last span of the domain") {
  const int degree = 2;
  const std::vector<double> knots = open_knots(degree, linspace01(5));
  const auto num_coeffs = static_cast<std::int64_t>(knots.size()) - degree - 1;
  const std::pair<double, std::int64_t> table[] = {
      {0.0, 2}, {0.1, 2}, {0.25, 3}, {0.6, 4}, {0.99, 5}};
  for (const auto& [point, expected] : table) {
    CAPTURE(point);
    CHECK(bsd::span_at(knots, num_coeffs, degree, point) == expected);
    CHECK(knots[static_cast<std::size_t>(expected)] <= point);
    CHECK(point < knots[static_cast<std::size_t>(expected) + 1]);
  }
  // The one point no bracket contains: an open knot vector's last span is half open, so the search
  // stops at it rather than walking into the clamped tail.
  CHECK(bsd::span_at(knots, num_coeffs, degree, 1.0) == num_coeffs - 1);
}

// Oracle: exact evaluation of the stored spline, within `E_eval`.
// ported from tests/test_bspline_roots.py::test_residual_at_evaluates_the_spline_it_is_given
TEST_CASE("residual_at: evaluates the spline it is given") {
  const int degree = 3;
  const std::vector<double> knots = open_knots(degree, linspace01(5));
  Truth truth;
  truth.roots = {rat(1, 4), rat(1, 2), rat(7, 8)};
  truth.power = poly_from_roots(truth.roots, 3);
  const std::vector<double> coeffs = blossom_coefficients(truth.power, knots, degree);
  const bs::Bspline<double> spline = make_scalar<double>(knots, degree, coeffs);
  const auto num_coeffs = static_cast<std::int64_t>(coeffs.size());
  std::vector<double> work(static_cast<std::size_t>(degree) + 1);
  const double e_eval = 1.5 * degree * (degree + 1) * kEps * stored_max<double>(coeffs);

  for (const double x : {0.0, 0.2, 0.37, 0.5, 0.75, 0.9, 1.0}) {
    CAPTURE(x);
    const std::vector<Rational> point = {ex(x)};
    const std::vector<Rational> value = oracle::exact_value<double>(spline, std::span(point));
    const double got = bsd::residual_at(knots, coeffs, num_coeffs, degree, x, work);
    CHECK(got >= 0.0);
    CHECK(oracle::within(got, abs_of(value[0]), e_eval));
  }
}

// Oracle: analytic polygon. A pinned coefficient is a barrier: no sign change is reported across
// it, and the first test index is clamped to 1.
// ported from tests/test_bspline_roots.py::test_zero_index_needs_a_non_zero_left_neighbour
TEST_CASE("zero_index: needs a non-zero left neighbour") {
  const std::vector<double> coeffs = {0.0, -1.0, 1.0, 1.0, -1.0};
  const auto n = static_cast<std::int64_t>(coeffs.size());
  CHECK(bsd::zero_index(coeffs, n, 1) == 2);
  CHECK(bsd::zero_index(coeffs, n, 3) == 4);
  CHECK(bsd::zero_index(coeffs, n, -7) == 2);  // clamped to 1 from below
  CHECK(bsd::zero_index(coeffs, n, 5) == -1);  // nothing left
  const std::vector<double> flat = {0.0, 0.0, 1.0};
  CHECK(bsd::zero_index(flat, 3, 1) == -1);
}

// ported from tests/test_bspline_roots.py::test_is_zero_index_rejects_an_index_outside_the_polygon
TEST_CASE("is_zero_index: rejects an index outside the polygon") {
  const std::vector<double> coeffs = {-1.0, 1.0, 2.0};
  CHECK(bsd::is_zero_index(coeffs, 3, 1));
  CHECK_FALSE(bsd::is_zero_index(coeffs, 3, 0));
  CHECK_FALSE(bsd::is_zero_index(coeffs, 3, 3));
  CHECK_FALSE(bsd::is_zero_index(coeffs, 3, 2));  // 1 * 2 > 0: no sign change
}

// Oracle: exact evaluation of the spline before and after (the function is unchanged); each new
// coefficient is a convex combination with weight `alpha = (point - t_i) / (t_{i+p} - t_i)`: five
// roundings (the differences of dyadic knots are exact, then `alpha`, `1 - alpha`, two products,
// the sum), `alpha`'s own error amplified by `|c_i - c_{i-1}| <= 2 max|c|`, in all at most
// `8 u max|c|`, and the basis being a partition of unity so is the spline.
// ported from tests/test_bspline_roots.py::test_insert_knot_leaves_the_spline_unchanged
TEST_CASE("insert_knot: leaves the spline unchanged") {
  const int degree = 3;
  const std::vector<double> knots = open_knots(degree, linspace01(5));
  const std::size_t n = knots.size() - degree - 1;
  std::vector<double> coeffs(n);
  const std::vector<double> grid = {-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};  // linspace(-1, 2, 7)
  for (std::size_t i = 0; i < n; ++i) {
    coeffs[i] = grid[i] * grid[i] - 0.5;
  }
  const bs::Bspline<double> before = make_scalar<double>(knots, degree, coeffs);

  std::vector<double> buffer_knots = knots;
  buffer_knots.push_back(0.0);
  std::vector<double> buffer_coeffs = coeffs;
  buffer_coeffs.push_back(0.0);
  const double point = 0.375;
  std::int64_t span = 0;
  while (point >= knots[static_cast<std::size_t>(span) + 1]) {
    ++span;
  }
  bsd::insert_knot(buffer_knots, buffer_coeffs, static_cast<std::int64_t>(n), degree, point, span);

  CHECK(buffer_knots == std::vector<double>({0, 0, 0, 0, 0.25, 0.375, 0.5, 0.75, 1, 1, 1, 1}));
  const bs::Bspline<double> after = make_scalar<double>(buffer_knots, degree, buffer_coeffs);
  CHECK(bspline_test::exact_difference<double>(before, after) <=
        8.0 * 0.5 * kEps * stored_max<double>(coeffs));
}

// Oracle: the structure the split must leave, analytic: the split point at multiplicity
// `degree + 1`, the knot before the run smaller, the first remaining coefficient pinned to zero.
// ported from tests/test_bspline_roots.py::test_split_at_root_leaves_an_open_knot_vector
TEST_CASE("split_at_root: leaves an open knot vector") {
  const int degree = 3;
  const std::vector<double> knots = budget_knots();
  const std::vector<double> coeffs = budget_coefficients();
  const auto num_coeffs = static_cast<std::int64_t>(coeffs.size());
  const std::size_t capacity = coeffs.size() + 4 * (degree + 1);
  std::vector<double> buffer_coeffs(capacity);
  std::copy(coeffs.begin(), coeffs.end(), buffer_coeffs.begin());
  std::vector<double> buffer_knots(capacity + degree + 1);
  std::copy(knots.begin(), knots.end(), buffer_knots.begin());

  const double root = 0.5;
  const double zero_tol = stored_max<double>(coeffs) * 16.0 * kEps;
  const bsd::SplitResult split =
      bsd::split_at_root(buffer_knots, buffer_coeffs, num_coeffs, degree, root, zero_tol);

  const auto offset = static_cast<std::size_t>(split.offset);
  for (std::size_t i = 0; i <= static_cast<std::size_t>(degree); ++i) {
    CHECK(buffer_knots[offset + i] == root);
  }
  CHECK(buffer_knots[offset - 1] < root);
  CHECK(buffer_coeffs[offset] == 0.0);
  CHECK(split.num_coeffs == num_coeffs + degree + 1);  // 0.5 is not a breakpoint here

  // Dropping the head leaves the open knot vector on [root, end]: the coefficients and knots
  // move down by `offset`.
  const std::int64_t remaining =
      bsd::drop_window_head(buffer_knots, buffer_coeffs, split.num_coeffs, degree, split.offset);
  CHECK(remaining == split.num_coeffs - split.offset);
  for (std::size_t i = 0; i <= static_cast<std::size_t>(degree); ++i) {
    CHECK(buffer_knots[i] == root);
    CHECK(buffer_knots[static_cast<std::size_t>(remaining) + i] == 1.0);
  }
  CHECK(buffer_coeffs[0] == 0.0);
}

// Oracle: analytic. A shift of the arrays by `offset`.
TEST_CASE("drop_window_head: shifts both windows down") {
  std::vector<double> knots = {10, 11, 12, 13, 14, 15, 16};
  std::vector<double> coeffs = {1, 2, 3, 4};
  const std::int64_t remaining = bsd::drop_window_head(knots, coeffs, 4, 2, 1);
  CHECK(remaining == 3);
  CHECK(std::vector<double>(coeffs.begin(), coeffs.begin() + 3) == std::vector<double>({2, 3, 4}));
  CHECK(std::vector<double>(knots.begin(), knots.begin() + 6) ==
        std::vector<double>({11, 12, 13, 14, 15, 16}));
}

// Oracle: analytic. A linear spline through (-1, 1, 3) on knots 0, 0, 1, 2, 2: the control polygon
// is the function, its zero is 1/2 and the first iterate is that exactly (lam = 1/2 exactly), the
// stopping rule is met at once (one iterate spans nothing) and the residual there is exactly 0.
TEST_CASE("track_zero: a linear spline stops at its zero") {
  std::vector<double> knots = {0.0, 0.0, 1.0, 2.0, 2.0};
  knots.resize(20, 0.0);
  std::vector<double> coeffs = {-1.0, 1.0, 3.0};
  coeffs.resize(16, 0.0);
  std::vector<double> iterates(1);
  std::vector<double> work(2);
  const bsd::TrackResult t =
      bsd::track_zero(knots, coeffs, 3, 1, 1, 1e-15, 2.0, 64, iterates, work);
  CHECK(t.x == 0.5);
  CHECK(t.status == bsd::TrackStatus::stagnated);
  CHECK(t.residual == 0.0);
  CHECK(t.num_coeffs == 3);
  CHECK(t.index == 1);
}

// Oracle: analytic derivatives. The slope bound is the largest hodograph quotient `degree *
// |c[i] - c[i-1]| / (t[i+degree] - t[i])`: for the Bernstein polynomial (0, 0, 1) of degree 2,
// which is `t^2`, it is 2, the slope at 1; a zero-length denominator (a C^-1 knot) is skipped.
TEST_CASE("slope_bound: the largest hodograph quotient, skipping a jump") {
  CHECK(bsd::slope_bound(std::vector<double>{0.0, 0.0, 1.0}, std::vector<double>{0, 0, 0, 1, 1, 1},
                         2) == 2.0);
  // Two linear pieces of slopes 1 and 2 joined by a jump of 4 at the double knot 1: the jump's
  // quotient has denominator zero and is not a slope.
  CHECK(bsd::slope_bound(std::vector<double>{0.0, 1.0, 5.0, 7.0},
                         std::vector<double>{0, 0, 1, 1, 2, 2}, 1) == 2.0);
  // A constant has slope zero.
  CHECK(bsd::slope_bound(std::vector<double>{3.0, 3.0, 3.0}, std::vector<double>{0, 0, 1, 2, 2},
                         1) == 0.0);
}

// --- Merging ----------------------------------------------------------------------------------

// ported from tests/test_bspline_roots.py::test_merge_roots_on_an_empty_input
// (the C++ result is trimmed to the runs: `merged` is empty, where Python returns a one-slot
// buffer with a count of zero)
TEST_CASE("merge_roots: on an empty input") {
  const bsd::MergedRoots m = bsd::merge_roots(std::vector<double>{}, std::vector<double>{});
  CHECK(m.merged.empty());
  CHECK(m.labels.empty());
}

// Oracle: analytic. The midpoint is `0.5 * fl(a + b)`: one rounding of the sum, then an exact
// halving, so within `u |a + b| / 2` of the exact midpoint.
// ported from tests/test_bspline_roots.py::test_merge_roots_collapses_a_run_but_not_a_gap
TEST_CASE("merge_roots: collapses a run but not a gap") {
  const std::vector<double> roots = {0.1, 0.1 + 1e-9, 0.5, 0.9};
  const std::vector<double> radii = {1e-8, 1e-8, 1e-12, 1e-12};
  const bsd::MergedRoots m = bsd::merge_roots(roots, radii);

  REQUIRE(m.merged.size() == 3);
  const Rational first_mid = (ex(roots[0]) + ex(roots[1])) / 2;
  CHECK(oracle::within(m.merged[0], first_mid, 0.5 * (kEps / 2.0) * (roots[0] + roots[1])));
  CHECK(m.merged[1] == 0.5);
  CHECK(m.merged[2] == 0.9);
  // Every input names the run it joined, so a caller can put a run back the way it found it when
  // the midpoint turns out not to be a zero.
  CHECK(m.labels == std::vector<std::int64_t>({0, 0, 1, 2}));

  // The larger of the two radii decides: a wide neighbour on the right joins the run.
  const bsd::MergedRoots wide =
      bsd::merge_roots(std::vector<double>{0.1, 0.2}, std::vector<double>{1e-12, 0.5});
  CHECK(wide.merged.size() == 1);
  CHECK(wide.labels == std::vector<std::int64_t>({0, 0}));
}
