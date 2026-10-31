// The pointwise product of two B-splines (`multiply`), the quotient-rule derivative of a rational
// B-spline (`derivative_rational`), and the minimum-norm least-squares solve both rest on
// (`linalg::lstsq_min_norm`).
//
// Oracles; each test case names its own again.
//  * Exact-rational evaluation. `bspline_exact_eval.hpp` evaluates a spline in exact rational
//    arithmetic from the recursive definition of the basis (no formula of the library) at exact
//    rational parameters. A product is compared with the product of the operands' exact values,
//    homogeneous coordinate by homogeneous coordinate: a rational spline stores `(w P, w)`, the
//    product of two splines is `(w_f w_g) (P_f P_g)` coordinate-wise, and a non-rational operand
//    takes the weight 1. The derivative of a rational spline is compared with the quotient rule
//    `(A' w - A w') / w^2`, assembled from the exact value and the exact first derivative of the
//    numerator `A` and the weight `w`.
//  * Closed forms. Polynomial identities in the Bernstein basis, and the stereographic circle
//    `x = (1 - t^2) / (1 + t^2)`, `y = 2 t / (1 + t^2)`, whose Bernstein form on `[0, 1]` has
//    integer homogeneous control points and whose products and derivatives are known functions.
//  * Tabulated knot structure. The interior multiplicity of a product at a breakpoint is
//    `max(m_f + q, m_g + p)` over the operands that carry it (Curry-Schoenberg, as the header of
//    product.hpp states); the multiplicities below are worked by hand, one comment each, not
//    computed by that formula.
//  * Constructed solutions. `lstsq_min_norm` is checked on systems whose answer is fixed by
//    construction or known in closed form (the normal equations, the pseudoinverse of a rank-one
//    matrix, a diagonal matrix whose singular values are its entries).
//
// Tolerance of a product value (`product_units`, `product_bound`). Write `u` for the unit
// roundoff of `T` and `S = max(1, M_f) max(1, M_g)`, `M` the largest stored coordinate of an
// operand (a rational operand's weight included; a non-rational operand in a rational product has
// the weight 1). At a point the computed `h` differs from `f g` by at most `K u S`, to first order
// in `u`, where `K` accumulates the stages of `multiply`:
//  1. The operands to Bezier form, by the Oslo algorithm. A Bezier control point is a convex
//     combination of the operand's (the refinement rows are nonnegative and sum to 1), so its
//     modulus is at most `M`. A row entry is built over `p` levels of a difference, a quotient, a
//     product and a sum, five roundings a level, and the `p + 1` entries are then summed with
//     products: at most `6 p + 1` roundings relative to `M`. The directions refine one after the
//     other, so each operand carries `e_ref = sum_d (6 p_d + 1)`.
//  2. The Bernstein product formula. A term is `C(p, i) C(q, j) / C(p + q, i + j) f_i g_j`: in one
//     direction the inverse binomial is rounded once and met by four products, five roundings;
//     in `D` directions the same count is `3 D + 2` (outer products of the weights, the
//     weighting, the product, the inverse binomials). The coefficients of the terms falling on one
//     control point sum to 1 exactly (Vandermonde's identity), so the sum of their moduli is at
//     most `M_f M_g`, and up to `n_t = prod_d (min(p_d, q_d) + 1)` terms are added:
//     `e_bern = 3 D + 2 + n_t`. A control point of the Bezier-form product is therefore within
//     `e = e_ref(f) + e_ref(g) + e_bern` units of `u S` of exact; the Bezier basis is a partition
//     of unity, so the value of the spline is no further.
//  3. A change of representation by least squares in double: onto the minimal knots when a
//     product multiplicity is below its Bezier value, and to a periodic or an unclamped direction.
//     The matrix `A` is the refinement matrix of the target knots, rounded to `T` (`6 r + 1`
//     roundings an entry, `r` the product degree). The minimiser leaves a residual no larger than
//     the incoming error, so the value of the result is that of the incoming control points
//     within twice that error, plus the rounding of `A` times the size of the solution, plus the
//     rounding of the solution to `T`. The size of the solution is that of the same function in
//     the target basis, `S` times the condition number of that basis. That number has no closed
//     form for general knots and none was derived: it is admitted as 2, a heuristic, for the
//     meshes used here (the knot spacings of every case below differ by factors of at most 4,
//     and the product degrees are at most 6). One pass is `K -> 2 K + c` with
//     `c = 2 (6 r + 1) + 1`. A multi-direction product passes every direction when any needs it,
//     a converted direction passes once more, and the conversion of the operands to open form
//     repeats stage 1.
// The derived `K` is under the conservative tier of `pantr::tolerance` (`4096 eps(T) = 8192 u`)
// for every case below. `derivative_bounds` adds the tier itself for the degree-keeping
// derivatives, the bound the derivative tests assert for that quantity.
#include <doctest/doctest.h>

#include "bspline_test_support.hpp"

#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/derivative.hpp>
#include <pantr/bspline/derivative_rational.hpp>
#include <pantr/bspline/product.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/core/ndarray.hpp>
#include <pantr/linalg/dense.hpp>
#include <pantr/tolerance.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace bs = pantr::bspline;
namespace la = pantr::linalg;
namespace oracle = bspline_oracle;
using bspline_test::cast_all;
using bspline_test::clamped;
using bspline_test::Dir;
using bspline_test::insert_in;
using bspline_test::knots_of;
using bspline_test::make_spline;
using bspline_test::message_of;
using bspline_test::periodic_knots;
using bspline_test::sp;
using oracle::Rational;

namespace {

template <class T>
double unit_roundoff() {
  return 0.5 * static_cast<double>(std::numeric_limits<T>::epsilon());
}

Rational abs_of(Rational v) {
  if (v < 0) {
    v = -v;
  }
  return v;
}

/// A periodic knot vector on `[0, 1]` with the given seam multiplicity and interior breakpoints
/// (repeated for multiplicity), with the `degree + 1 - seam` ghost knots on each side that continue
/// the pattern by the period 1.
std::vector<double> periodic_seam_knots(int degree, int seam, const std::vector<double>& interior) {
  std::vector<double> tile(static_cast<std::size_t>(seam), 0.0);
  tile.insert(tile.end(), interior.begin(), interior.end());
  std::vector<double> seq;
  for (int shift = -4; shift <= 4; ++shift) {
    for (const double k : tile) {
      seq.push_back(k + static_cast<double>(shift));
    }
  }
  const auto first = std::find(seq.begin(), seq.end(), 0.0) - seq.begin();
  const auto last = std::find(seq.begin(), seq.end(), 1.0) - seq.begin();
  const std::ptrdiff_t ghost = degree + 1 - seam;
  return std::vector<double>(seq.begin() + (first - ghost), seq.begin() + (last + seam + ghost));
}

/// `count + 1` equally spaced knots `lo, lo + h, ...`, an unclamped vector.
std::vector<double> equispaced(double lo, double h, int count) {
  std::vector<double> k;
  for (int i = 0; i <= count; ++i) {
    k.push_back(lo + h * static_cast<double>(i));
  }
  return k;
}

/// A B-spline of the given control points, flat and row-major, over one direction.
template <class T>
bs::Bspline<T> curve(const Dir& dir, const std::vector<double>& cps, bool rational) {
  const std::vector<T> flat = cast_all<T>(cps);
  return bs::Bspline<T>(bspline_test::space_of<T>({dir}), sp(flat), rational);
}

/// An interior breakpoint of a direction with its multiplicity.
struct Break {
  double knot;
  std::int64_t mult;
};

template <class T>
std::vector<Break> interior_structure(const bs::BsplineSpace1D<T>& s) {
  const auto [knots, mults] = s.unique_knots_and_multiplicity(true);
  std::vector<Break> out;
  for (std::size_t i = 1; i + 1 < knots.size(); ++i) {
    out.push_back({static_cast<double>(knots[i]), mults[i]});
  }
  return out;
}

/// The multiplicities of the first and of the last domain end of a direction.
template <class T>
std::pair<std::int64_t, std::int64_t> end_multiplicities(const bs::BsplineSpace1D<T>& s) {
  const auto mults = s.unique_knots_and_multiplicity(true).second;
  return {mults.front(), mults.back()};
}

void expect_structure(const std::vector<Break>& got, const std::vector<Break>& want,
                      double knot_tol) {
  REQUIRE(got.size() == want.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    CAPTURE(i);
    CHECK(std::abs(got[i].knot - want[i].knot) <= knot_tol);
    CHECK(got[i].mult == want[i].mult);
  }
}

using Domain = std::pair<double, double>;

/// The space of one direction, by value (a handle on a shared representation).
template <class T>
bs::BsplineSpace1D<T> dir_space(const bs::Bspline<T>& b, std::size_t d) {
  return b.space().spaces()[d];
}

template <class T>
std::vector<Domain> domains_of(const bs::Bspline<T>& b) {
  std::vector<Domain> out;
  for (std::size_t d = 0; d < b.dim(); ++d) {
    const auto dom = dir_space<T>(b, d).domain();
    out.emplace_back(static_cast<double>(dom.first), static_cast<double>(dom.second));
  }
  return out;
}

/// The tensor grid of every `step`-th of the eight sample points (`bspline_test::sample_points`:
/// none is a knot of the dyadic meshes used here) of each direction's domain.
std::vector<std::vector<Rational>> grid_of(const std::vector<Domain>& doms, std::size_t step) {
  std::vector<std::vector<Rational>> points(1);
  for (const Domain& dom : doms) {
    const std::vector<Rational> all =
        bspline_test::sample_points(oracle::exact(dom.first), oracle::exact(dom.second));
    std::vector<std::vector<Rational>> next;
    for (const auto& prefix : points) {
      for (std::size_t i = 0; i < all.size(); i += step) {
        std::vector<Rational> x = prefix;
        x.push_back(all[i]);
        next.push_back(std::move(x));
      }
    }
    points = std::move(next);
  }
  return points;
}

/// The homogeneous coordinates of the product of two splines at a point, from their exact values:
/// the coordinates multiply pairwise and the weights multiply, an absent weight being 1.
std::vector<Rational> product_of(const std::vector<Rational>& fv, bool f_rational,
                                 const std::vector<Rational>& gv, bool g_rational) {
  const bool rational = f_rational || g_rational;
  const std::size_t rank = fv.size() - (f_rational ? 1 : 0);
  std::vector<Rational> out;
  for (std::size_t c = 0; c < rank; ++c) {
    out.push_back(fv[c] * gv[c]);
  }
  if (rational) {
    out.push_back((f_rational ? fv[rank] : Rational(1)) * (g_rational ? gv[rank] : Rational(1)));
  }
  return out;
}

/// The largest difference, over the grid and the homogeneous coordinates, between the exact value
/// of `h` and the product of the exact values of `f` and `g`.
template <class T>
double product_error(const bs::Bspline<T>& f, const bs::Bspline<T>& g, const bs::Bspline<T>& h,
                     std::size_t step) {
  REQUIRE(h.dim() == f.dim());
  double worst = 0.0;
  for (const std::vector<Rational>& x : grid_of(domains_of(h), step)) {
    const std::vector<Rational> want = product_of(oracle::exact_value<T>(f, x), f.rational(),
                                                  oracle::exact_value<T>(g, x), g.rational());
    const std::vector<Rational> got = oracle::exact_value<T>(h, x);
    REQUIRE(got.size() == want.size());
    for (std::size_t c = 0; c < got.size(); ++c) {
      worst = (std::max)(worst, static_cast<double>(abs_of(got[c] - want[c])));
    }
  }
  return worst;
}

/// How the product's control points are reached from the Bezier form (stage 3 of the header).
enum class Route : std::uint8_t {
  bezier,     ///< No change of representation: every multiplicity is its Bezier value.
  projected,  ///< A least-squares change of representation onto the minimal knots.
  converted,  ///< Also a periodic or unclamped direction.
};

/// `K` of the file header for operands of the given degrees, in units of `u S`: the count `e` of
/// stages 1 and 2, then one pass of stage 3 per least-squares change of representation,
/// `K -> 2 K + c` with `c = 12 r + 3` (`(6 r + 1)` roundings of a refinement-matrix entry times the
/// admitted basis-condition constant 2, plus the final rounding to `T`; `r` the largest product
/// degree). A multi-direction product projects every direction when any needs it (`dim` passes);
/// a converted direction costs one more, and its operands' conversion to open form doubles the
/// refinement count of stage 1.
double product_units(const std::vector<std::size_t>& p, const std::vector<std::size_t>& q,
                     Route route, std::size_t converted_dirs) {
  const std::size_t dim = p.size();
  double refine = 0.0;
  double terms = 1.0;
  std::size_t top = 0;
  for (std::size_t d = 0; d < dim; ++d) {
    refine += (6.0 * static_cast<double>(p[d]) + 1.0) + (6.0 * static_cast<double>(q[d]) + 1.0);
    terms *= static_cast<double>((std::min)(p[d], q[d]) + 1);
    top = (std::max)(top, p[d] + q[d]);
  }
  if (route == Route::converted) {
    refine *= 2.0;
  }
  const double c = 12.0 * static_cast<double>(top) + 3.0;
  std::size_t passes = 0;
  if (route != Route::bezier) {
    passes = dim + (route == Route::converted ? converted_dirs : 0);
  }
  double units = refine + 3.0 * static_cast<double>(dim) + 2.0 + terms;
  for (std::size_t i = 0; i < passes; ++i) {
    units = 2.0 * units + c;
  }
  return units;
}

/// The number of directions of a spline that are not open (periodic or unclamped).
template <class T>
std::size_t nonopen_directions(const bs::Bspline<T>& b) {
  std::size_t count = 0;
  for (std::size_t d = 0; d < b.dim(); ++d) {
    if (!dir_space<T>(b, d).has_open_knots()) {
      ++count;
    }
  }
  return count;
}

/// The bound `K u S` of the file header on the value error of `h = multiply(f, g)`.
template <class T>
double product_bound(const bs::Bspline<T>& f, const bs::Bspline<T>& g, const bs::Bspline<T>& h,
                     Route route) {
  const double s = (std::max)(1.0, oracle::max_abs_control_point<T>(f)) *
                   (std::max)(1.0, oracle::max_abs_control_point<T>(g));
  return product_units(f.degrees(), g.degrees(), route, nonopen_directions<T>(h)) *
         unit_roundoff<T>() * s;
}

template <class T>
void expect_product_values(const bs::Bspline<T>& f, const bs::Bspline<T>& g,
                           const bs::Bspline<T>& h, Route route, std::size_t step = 1) {
  const double err = product_error<T>(f, g, h, step);
  const double bound = product_bound<T>(f, g, h, route);
  CAPTURE(err);
  CAPTURE(bound);
  CHECK(err <= bound);
}

struct Kind {
  bool f_rational;
  bool g_rational;
};

const std::array<Kind, 4> kKinds = {Kind{false, false}, Kind{true, true}, Kind{true, false},
                                    Kind{false, true}};

/// A 1D case: two directions and the interior structure of their product, worked by hand with
/// `m_h = max(m_f + q, m_g + p)` over the operands carrying the breakpoint (`r = p + q`).
struct Case1 {
  const char* name;
  Dir f;
  Dir g;
  std::vector<Break> want;
  Route route;
};

const std::vector<Case1>& open_cases() {
  static const std::vector<Case1> cases = {
      // r = 4. 0.25 only in f: 1 + 2; 0.5 in both: max(1 + 2, 1 + 2); 0.75 only in g: 1 + 2.
      {"two meshes, one shared knot",
       {clamped(2, {0.25, 0.5}), 2},
       {clamped(2, {0.5, 0.75}), 2},
       {{0.25, 3}, {0.5, 3}, {0.75, 3}},
       Route::projected},
      // r = 4, p = 3, q = 1. 0.5 only in f: m_f + q = 2. 0.75 only in g: m_g + p = 4.
      {"cubic by linear, a knot each",
       {clamped(3, {0.5}), 3},
       {clamped(1, {0.75}), 1},
       {{0.5, 2}, {0.75, 4}},
       Route::projected},
      // r = 4. 0.5 in both: max(2 + 2, 1 + 2) = 4, a C^0 product of C^0 and C^1.
      {"C0 by C1 at one knot",
       {clamped(2, {0.5, 0.5}), 2},
       {clamped(2, {0.5}), 2},
       {{0.5, 4}},
       Route::bezier},
      // r = 4. 0.5 only in f, with m_f = 3 = p + 1 (C^-1): 3 + 2 = 5 = r + 1, a C^-1 product.
      {"C-1 by smooth",
       {clamped(2, {0.5, 0.5, 0.5}), 2},
       {clamped(2, {}), 2},
       {{0.5, 5}},
       Route::bezier},
      // r = 4, p = 1, q = 3. 0.5 only in f: 1 + 3 = 4 = r, a C^0 product.
      {"linear kink by cubic",
       {clamped(1, {0.5}), 1},
       {clamped(3, {}), 3},
       {{0.5, 4}},
       Route::bezier},
      // r = 2, p = 0, q = 2. 0.5 only in f with m_f = 1 = p + 1: 1 + 2 = 3 = r + 1 (C^-1).
      // 0.25 only in g: m_g + p = 1 + 0 = 1, a C^1 breakpoint of a quadratic.
      {"degree 0 by quadratic",
       {clamped(0, {0.5}), 0},
       {clamped(2, {0.25}), 2},
       {{0.25, 1}, {0.5, 3}},
       Route::projected},
      // r = 4, one Bezier element each.
      {"no interior knot", {clamped(2, {}), 2}, {clamped(2, {}), 2}, {}, Route::bezier},
      // r = 5. 0.25: max(2 + 2, 1 + 3) = 4. 0.5: max(1 + 2, 2 + 3) = 5 (C^0).
      {"different multiplicities at two shared knots",
       {clamped(3, {0.25, 0.25, 0.5}), 3},
       {clamped(2, {0.25, 0.5, 0.5}), 2},
       {{0.25, 4}, {0.5, 5}},
       Route::projected},
      // r = 5. 0.25 and 0.75 only in f: 1 + 3. 0.5 in both: max(1 + 3, 1 + 2).
      {"quadratic by cubic",
       {clamped(2, {0.25, 0.5, 0.75}), 2},
       {clamped(3, {0.5}), 3},
       {{0.25, 4}, {0.5, 4}, {0.75, 4}},
       Route::projected},
  };
  return cases;
}

}  // namespace

TEST_CASE_TEMPLATE("product: polynomial identities in the Bernstein basis", T, double, float) {
  // Oracle: closed forms. The data are small dyadic numbers and the inverse binomials involved
  // (1 and 1/2) are exact, so the control points are exact in either precision and are compared
  // for equality.
  const Dir lin{clamped(1, {}), 1};
  const Dir quad{clamped(2, {}), 2};
  const auto control_points = [](const bs::Bspline<T>& b) {
    return std::vector<T>(b.control_points().begin(), b.control_points().end());
  };
  // t * t = t^2, whose Bernstein coefficients are (0, 0, 1).
  const bs::Bspline<T> t = curve<T>(lin, {0.0, 1.0}, false);
  CHECK(control_points(bs::multiply<T>(t, t)) == cast_all<T>({0.0, 0.0, 1.0}));
  // t * (1 - t) = (0, 1/2, 0).
  const bs::Bspline<T> one_minus_t = curve<T>(lin, {1.0, 0.0}, false);
  CHECK(control_points(bs::multiply<T>(t, one_minus_t)) == cast_all<T>({0.0, 0.5, 0.0}));
  // (1 + t)(1 - t) = 1 - t^2 = (1, 1, 0).
  const bs::Bspline<T> one_plus_t = curve<T>(lin, {1.0, 2.0}, false);
  CHECK(control_points(bs::multiply<T>(one_plus_t, one_minus_t)) == cast_all<T>({1.0, 1.0, 0.0}));
  // t * t^2 = t^3: the degrees add, (0, 0, 0, 1).
  const bs::Bspline<T> t_squared = curve<T>(quad, {0.0, 0.0, 1.0}, false);
  const bs::Bspline<T> cube = bs::multiply<T>(t, t_squared);
  CHECK(cube.degrees()[0] == 3);
  CHECK(control_points(cube) == cast_all<T>({0.0, 0.0, 0.0, 1.0}));
  // The product commutes.
  CHECK(control_points(bs::multiply<T>(t_squared, t)) == control_points(cube));
}

TEST_CASE_TEMPLATE("product: the constant one multiplies to the operand itself", T, double, float) {
  // Oracle: the multiplicative identity. 1 has degree 0 and no interior knot, so the minimal
  // product space is the operand's own: same degree, same knot vector (a larger multiplicity
  // rule would give a larger one), and the control points are the operand's up to rounding.
  const Dir dir{clamped(3, {0.25, 0.5, 0.5}), 3};
  const bs::Bspline<T> f = make_spline<T>({dir}, 2, false, 5U);
  const bs::Bspline<T> one = curve<T>({clamped(0, {}), 0}, {1.0, 1.0}, false);  // one row, rank 2
  for (const bool one_first : {false, true}) {
    CAPTURE(one_first);
    const bs::Bspline<T> h = one_first ? bs::multiply<T>(one, f) : bs::multiply<T>(f, one);
    CHECK(h.degrees()[0] == 3);
    CHECK(knots_of<T>(h, 0) == knots_of<T>(f, 0));
    REQUIRE(h.control_points().size() == f.control_points().size());
    const double bound = product_bound<T>(f, one, h, Route::projected);
    for (std::size_t i = 0; i < h.control_points().size(); ++i) {
      CHECK(std::abs(static_cast<double>(h.control_points()[i]) -
                     static_cast<double>(f.control_points()[i])) <= bound);
    }
  }
}

TEST_CASE_TEMPLATE("product: exact-rational values, one direction", T, double, float) {
  // Oracle: exact-rational evaluation of f, g and h at exact rational points; h(x) = f(x) g(x).
  // Bound: product_bound (file header). Non-rational and rational operands, in every pairing.
  for (const Case1& c : open_cases()) {
    for (const Kind kind : kKinds) {
      CAPTURE(c.name);
      CAPTURE(kind.f_rational);
      CAPTURE(kind.g_rational);
      const bs::Bspline<T> f = make_spline<T>({c.f}, 2, kind.f_rational, 11U);
      const bs::Bspline<T> g = make_spline<T>({c.g}, 2, kind.g_rational, 23U);
      const bs::Bspline<T> h = bs::multiply<T>(f, g);
      CHECK(h.rational() == (kind.f_rational || kind.g_rational));
      CHECK(h.rank() == 2);
      expect_product_values<T>(f, g, h, c.route);
    }
  }
}

TEST_CASE_TEMPLATE("product: the product knot vector is the minimal one", T, double, float) {
  // Oracle: the tabulated multiplicities of open_cases() (worked by hand from the Curry-Schoenberg
  // rule), degree p + q, open ends. The values are also checked: a smaller multiplicity than the
  // minimal one cannot carry the product.
  for (const Case1& c : open_cases()) {
    CAPTURE(c.name);
    const bs::Bspline<T> f = make_spline<T>({c.f}, 1, false, 3U);
    const bs::Bspline<T> g = make_spline<T>({c.g}, 1, false, 4U);
    const bs::Bspline<T> h = bs::multiply<T>(f, g);
    const bs::BsplineSpace1D<T> s = dir_space<T>(h, 0);
    CHECK(s.degree() == c.f.degree + c.g.degree);
    CHECK(s.has_open_knots());
    CHECK_FALSE(s.periodic());
    expect_structure(interior_structure<T>(s), c.want, 0.0);
    CHECK(end_multiplicities<T>(s) ==
          std::make_pair(std::int64_t{s.degree()} + 1, std::int64_t{s.degree()} + 1));
    CHECK(domains_of(h) == domains_of(f));
    expect_product_values<T>(f, g, h, c.route);
  }
}

/// An N-D case: one `Case1`-like pair of directions per parametric direction.
struct CaseN {
  const char* name;
  std::vector<Dir> f;
  std::vector<Dir> g;
  std::vector<std::vector<Break>> want;
  Route route;
  std::size_t step;  ///< Every `step`-th sample point per direction is compared.
};

const std::vector<CaseN>& surface_cases() {
  static const std::vector<CaseN> cases = {
      // Direction 0: r = 3. 0.25 only in g: m_g + p = 1 + 2 = 3; 0.5 in both: max(1 + 1, 1 + 2).
      // Direction 1: r = 3. 0.5 only in g: m_g + p = 1 + 1 = 2.
      {"knots one operand lacks, in both directions",
       {{clamped(2, {0.5}), 2}, {clamped(1, {}), 1}},
       {{clamped(1, {0.25, 0.5}), 1}, {clamped(2, {0.5}), 2}},
       {{{0.25, 3}, {0.5, 3}}, {{0.5, 2}}},
       Route::projected,
       2},
      // Direction 0: f is C^-1 at 0.5 (m = 3 = p + 1): 3 + 2 = 5 = r + 1. Direction 1: r = 3.
      // 0.25 only in f: 1 + 1 = 2; 0.75 only in g: 1 + 2 = 3.
      {"a discontinuous direction beside a smooth one",
       {{clamped(2, {0.5, 0.5, 0.5}), 2}, {clamped(2, {0.25}), 2}},
       {{clamped(2, {}), 2}, {clamped(1, {0.75}), 1}},
       {{{0.5, 5}}, {{0.25, 2}, {0.75, 3}}},
       Route::projected,
       2},
      // Direction 0: r = 3, 0.5 in both: max(1 + 2, 1 + 1) = 3 = r. Direction 1: single elements.
      // Nothing is projected.
      {"every direction already at its Bezier multiplicity",
       {{clamped(1, {0.5}), 1}, {clamped(3, {}), 3}},
       {{clamped(2, {0.5}), 2}, {clamped(3, {}), 3}},
       {{{0.5, 3}}, {}},
       Route::bezier,
       2},
  };
  return cases;
}

TEST_CASE_TEMPLATE("product: exact-rational values, two directions", T, double, float) {
  // Oracle: exact-rational evaluation of f, g and h on a tensor grid of exact points. The
  // multi-direction path of `multiply` is its own code (the tensor-product Bernstein formula, a
  // projection in every direction when any needs it).
  for (const CaseN& c : surface_cases()) {
    for (const Kind kind : kKinds) {
      CAPTURE(c.name);
      CAPTURE(kind.f_rational);
      CAPTURE(kind.g_rational);
      const bs::Bspline<T> f = make_spline<T>(c.f, 1, kind.f_rational, 31U);
      const bs::Bspline<T> g = make_spline<T>(c.g, 1, kind.g_rational, 47U);
      const bs::Bspline<T> h = bs::multiply<T>(f, g);
      CHECK(h.rational() == (kind.f_rational || kind.g_rational));
      expect_product_values<T>(f, g, h, c.route, c.step);
    }
  }
}

TEST_CASE_TEMPLATE("product: the product knot vectors are minimal in every direction", T, double,
                   float) {
  // Oracle: the multiplicities tabulated by hand in surface_cases(), per direction.
  for (const CaseN& c : surface_cases()) {
    CAPTURE(c.name);
    const bs::Bspline<T> f = make_spline<T>(c.f, 1, false, 5U);
    const bs::Bspline<T> g = make_spline<T>(c.g, 1, false, 6U);
    const bs::Bspline<T> h = bs::multiply<T>(f, g);
    REQUIRE(h.dim() == c.f.size());
    for (std::size_t d = 0; d < h.dim(); ++d) {
      CAPTURE(d);
      const bs::BsplineSpace1D<T> s = dir_space<T>(h, d);
      CHECK(s.degree() == c.f[d].degree + c.g[d].degree);
      CHECK(s.has_open_knots());
      expect_structure(interior_structure<T>(s), c.want[d], 0.0);
    }
  }
}

TEST_CASE_TEMPLATE("product: exact-rational values, three directions", T, double, float) {
  // Oracle: exact-rational evaluation, on three points per direction. Three directions exercise
  // the element odometer and the box copies beyond what a surface does.
  const std::vector<Dir> df = {{clamped(1, {0.5}), 1}, {clamped(1, {}), 1}, {clamped(2, {}), 2}};
  const std::vector<Dir> dg = {{clamped(1, {}), 1}, {clamped(2, {0.5}), 2}, {clamped(1, {0.5}), 1}};
  const bs::Bspline<T> f = make_spline<T>(df, 1, false, 9U);
  const bs::Bspline<T> g = make_spline<T>(dg, 1, false, 10U);
  const bs::Bspline<T> h = bs::multiply<T>(f, g);
  // Direction 0: 0.5 only in f, 1 + 1 = 2 = r. Direction 1: 0.5 only in g, 1 + 1 = 2 < r = 3.
  // Direction 2: 0.5 only in g, 1 + 2 = 3 = r.
  expect_structure(interior_structure<T>(dir_space<T>(h, 0)), {{0.5, 2}}, 0.0);
  expect_structure(interior_structure<T>(dir_space<T>(h, 1)), {{0.5, 2}}, 0.0);
  expect_structure(interior_structure<T>(dir_space<T>(h, 2)), {{0.5, 3}}, 0.0);
  expect_product_values<T>(f, g, h, Route::projected, 3);
}

enum class Boundary : std::uint8_t { open, periodic, unclamped };

/// A boundary case: the product of two directions must have the given boundary form, the interior
/// multiplicities worked by hand (`r = p + q`), and, for a periodic or an unclamped result, the
/// multiplicity of the domain ends: `max(m_f + q, m_g + p)` over the end multiplicities of the
/// operands (1 for the simple ends used here; the seam multiplicity for a periodic operand).
struct BoundaryCase {
  const char* name;
  Dir f;
  Dir g;
  Boundary want;
  std::vector<Break> interior;
  std::int64_t ends;
  Route route;
};

const Dir kOpen2{clamped(2, {0.25, 0.5, 0.75}), 2};
const Dir kOpen3{clamped(3, {0.5}), 3};
const Dir kPer2{periodic_knots(4, 2), 2, true};
const Dir kPer3{periodic_knots(4, 3), 3, true};
const Dir kUnc2{equispaced(-0.5, 0.25, 8), 2};
const Dir kUnc3{equispaced(-0.75, 0.25, 10), 3};
// Domain [0, 1], the left end a double knot.
const Dir kUncAsym{{-0.5, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}, 2};
// Domain [0, 1], both ends double knots.
const Dir kUncBoth{{-0.25, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.25}, 2};

const std::vector<BoundaryCase>& boundary_cases() {
  static const Dir per2_c0_seam{periodic_seam_knots(2, 2, {0.25, 0.5, 0.75}), 2, true};
  static const Dir per1_coarse{periodic_seam_knots(1, 1, {0.5}), 1, true};
  static const Dir per2_cminus1_seam{periodic_seam_knots(2, 3, {0.25, 0.5, 0.75}), 2, true};
  static const std::vector<BoundaryCase> cases = {
      // r = 5. 0.25 only in f: 1 + 3; 0.5 in both: max(1 + 3, 1 + 2); 0.75 only in f: 1 + 3.
      {"open by open",
       kOpen2,
       kOpen3,
       Boundary::open,
       {{0.25, 4}, {0.5, 4}, {0.75, 4}},
       6,
       Route::projected},
      // r = 5. Every breakpoint in both: max(1 + 3, 1 + 2) = 4. Seam: max(1 + 3, 1 + 2) = 4.
      {"periodic by periodic",
       kPer2,
       kPer3,
       Boundary::periodic,
       {{0.25, 4}, {0.5, 4}, {0.75, 4}},
       4,
       Route::converted},
      // r = 5, p = 3, q = 2. Breakpoints: max(1 + 2, 1 + 3) = 4. Seam: max(1 + 2, 2 + 3) = 5, a
      // C^0 seam of the product.
      {"periodic by periodic with a C0 seam in one",
       kPer3,
       per2_c0_seam,
       Boundary::periodic,
       {{0.25, 4}, {0.5, 4}, {0.75, 4}},
       5,
       Route::converted},
      // r = 4. Breakpoints in both: max(1 + 2, 1 + 2) = 3. Seam: max(3 + 2, 1 + 2) = 5 = r + 1, a
      // discontinuous seam, with no ghost knots at all.
      {"periodic by periodic with a C-1 seam in one",
       per2_cminus1_seam,
       kPer2,
       Boundary::periodic,
       {{0.25, 3}, {0.5, 3}, {0.75, 3}},
       5,
       Route::converted},
      // r = 3, p = 2, q = 1. 0.25 and 0.75 only in f: 1 + 1 = 2; 0.5 in both: max(1 + 1, 1 + 2).
      // Seam: max(1 + 1, 1 + 2) = 3.
      {"periodic by periodic on different meshes",
       kPer2,
       per1_coarse,
       Boundary::periodic,
       {{0.25, 2}, {0.5, 3}, {0.75, 2}},
       3,
       Route::converted},
      // r = 5. Breakpoints max(1 + 3, 1 + 2) = 4. Ends: max(1 + 3, 1 + 2) = 4 on both sides.
      {"periodic by unclamped",
       kPer2,
       kUnc3,
       Boundary::unclamped,
       {{0.25, 4}, {0.5, 4}, {0.75, 4}},
       4,
       Route::converted},
      {"unclamped by periodic",
       kUnc3,
       kPer2,
       Boundary::unclamped,
       {{0.25, 4}, {0.5, 4}, {0.75, 4}},
       4,
       Route::converted},
      {"unclamped by unclamped",
       kUnc2,
       kUnc3,
       Boundary::unclamped,
       {{0.25, 4}, {0.5, 4}, {0.75, 4}},
       4,
       Route::converted},
      // The ends of the unclamped operand kUncAsym have multiplicity 2 on the left and 1 on the
      // right (kUnc3: 1 and 1). p = 2, q = 3. Left: max(2 + 3, 1 + 2) = 5. Right: max(1 + 3, 1 + 2)
      // = 4. The product takes the smaller of the two on both ends (the convention of the previous
      // release; the function on the domain does not depend on it).
      {"unclamped by unclamped, a double knot at one end",
       kUncAsym,
       kUnc3,
       Boundary::unclamped,
       {{0.25, 4}, {0.5, 4}, {0.75, 4}},
       4,
       Route::converted},
      // kUncBoth has multiplicity 2 on both ends: left max(2 + 3, 1 + 2) = 5, right the same.
      {"unclamped by unclamped, double knots at both ends",
       kUncBoth,
       kUnc3,
       Boundary::unclamped,
       {{0.25, 4}, {0.5, 4}, {0.75, 4}},
       5,
       Route::converted},
      {"unclamped, a double knot at one end, by unclamped",
       kUnc3,
       kUncAsym,
       Boundary::unclamped,
       {{0.25, 4}, {0.5, 4}, {0.75, 4}},
       4,
       Route::converted},
      // Either operand open gives an open product. r = 5. 0.25 and 0.75 only in the non-open
      // operand: 1 + 3 (its partner's degree); 0.5 in both: max(1 + 3, 1 + 2) = 4.
      {"open by periodic",
       kOpen3,
       kPer2,
       Boundary::open,
       {{0.25, 4}, {0.5, 4}, {0.75, 4}},
       6,
       Route::converted},
      {"periodic by open",
       kPer2,
       kOpen3,
       Boundary::open,
       {{0.25, 4}, {0.5, 4}, {0.75, 4}},
       6,
       Route::converted},
      {"open by unclamped",
       kOpen3,
       kUnc2,
       Boundary::open,
       {{0.25, 4}, {0.5, 4}, {0.75, 4}},
       6,
       Route::converted},
      {"unclamped by open",
       kUnc2,
       kOpen3,
       Boundary::open,
       {{0.25, 4}, {0.5, 4}, {0.75, 4}},
       6,
       Route::converted},
  };
  return cases;
}

/// The knots, with repetition, of a list of breakpoints.
std::vector<double> repeated(const std::vector<Break>& interior) {
  std::vector<double> out;
  for (const Break& b : interior) {
    out.insert(out.end(), static_cast<std::size_t>(b.mult), b.knot);
  }
  return out;
}

/// Where the unit-interval cases above are placed: `[a, a + length]`. The second placement has a
/// period other than 1 and an origin other than 0, so that a period read as the right end, or a
/// shift measured from 0, shows.
struct Placement {
  double a;
  double length;
};

const std::array<Placement, 2> kPlacements = {Placement{0.0, 1.0}, Placement{2.0, 4.0}};

/// The direction with every knot `k` sent to `a + length k` (integers or dyadic numbers here).
Dir placed(const Dir& d, const Placement& pl) {
  Dir out = d;
  for (double& k : out.knots) {
    k = pl.a + pl.length * k;
  }
  return out;
}

std::vector<double> placed_knots(std::vector<double> knots, const Placement& pl) {
  for (double& k : knots) {
    k = pl.a + pl.length * k;
  }
  return knots;
}

std::vector<Break> placed_breaks(std::vector<Break> breaks, const Placement& pl) {
  for (Break& b : breaks) {
    b.knot = pl.a + pl.length * b.knot;
  }
  return breaks;
}

template <class T>
void expect_boundary(const bs::BsplineSpace1D<T>& s, Boundary want) {
  switch (want) {
    case Boundary::open:
      CHECK(s.has_open_knots());
      CHECK_FALSE(s.periodic());
      return;
    case Boundary::periodic:
      CHECK(s.periodic());
      CHECK_FALSE(s.has_open_knots());
      return;
    case Boundary::unclamped:
      CHECK_FALSE(s.periodic());
      CHECK_FALSE(s.has_open_knots());
      CHECK_FALSE(s.has_left_end_open());
      CHECK_FALSE(s.has_right_end_open());
      return;
  }
}

TEST_CASE_TEMPLATE("product: each boundary combination gives its boundary form", T, double, float) {
  // Oracle: the tabulated form and multiplicities of boundary_cases() (worked by hand), then
  // exact-rational evaluation of f, g and h over the domain. Open by anything is open; periodic by
  // periodic periodic; every other mix, unclamped. The domain of the product is the operands'.
  for (const Placement pl : kPlacements) {
    for (const BoundaryCase& c : boundary_cases()) {
      CAPTURE(c.name);
      CAPTURE(pl.a);
      const bs::Bspline<T> f = make_spline<T>({placed(c.f, pl)}, 1, false, 13U);
      const bs::Bspline<T> g = make_spline<T>({placed(c.g, pl)}, 1, false, 14U);
      const bs::Bspline<T> h = bs::multiply<T>(f, g);
      const bs::BsplineSpace1D<T> s = dir_space<T>(h, 0);
      const std::int64_t r = c.f.degree + c.g.degree;
      CHECK(s.degree() == r);
      expect_boundary<T>(s, c.want);
      expect_structure(interior_structure<T>(s), placed_breaks(c.interior, pl), 0.0);
      const std::int64_t ends = c.want == Boundary::open ? r + 1 : c.ends;
      CHECK(end_multiplicities<T>(s) == std::make_pair(ends, ends));
      CHECK(domains_of(h) == domains_of(f));
      expect_product_values<T>(f, g, h, c.route);
    }
  }
}

TEST_CASE_TEMPLATE("product: a periodic product has the periodic knot vector of its structure", T,
                   double, float) {
  // Oracle: the knot vector rebuilt by hand from the tabulated seam and breakpoints: one period
  // repeated, with `r + 1 - seam` ghost knots each side (integers here, so exact), and the control
  // point count of a periodic space, the sum of the multiplicities over one period. On the unit
  // interval and on `[2, 6]`.
  std::size_t checked = 0;
  for (const Placement pl : kPlacements) {
    for (const BoundaryCase& c : boundary_cases()) {
      if (c.want != Boundary::periodic) {
        continue;
      }
      ++checked;
      CAPTURE(c.name);
      CAPTURE(pl.a);
      const bs::Bspline<T> f = make_spline<T>({placed(c.f, pl)}, 1, false, 15U);
      const bs::Bspline<T> g = make_spline<T>({placed(c.g, pl)}, 1, false, 16U);
      const bs::Bspline<T> h = bs::multiply<T>(f, g);
      const bs::BsplineSpace1D<T> s = dir_space<T>(h, 0);
      const int r = c.f.degree + c.g.degree;
      CHECK(knots_of<T>(h, 0) ==
            cast_all<T>(placed_knots(
                periodic_seam_knots(r, static_cast<int>(c.ends), repeated(c.interior)), pl)));
      std::int64_t count = c.ends;
      for (const Break& b : c.interior) {
        count += b.mult;
      }
      CHECK(s.num_basis() == count);
      CHECK(h.control_points().size() == static_cast<std::size_t>(count));
    }
  }
  CHECK(checked == 8);  // four periodic cases, two placements
}

TEST_CASE_TEMPLATE("product: periodic operands give a product that closes at the seam", T, double,
                   float) {
  // Oracle: exact-rational values at the two domain ends, where a periodic operand has f(a) =
  // f(b) exactly (the stored control points repeat), so h(a) = f(a) g(a) = h(b) up to the bound.
  // A seam closed in the knot vector but not in the control points, or the reverse, opens here.
  std::size_t checked = 0;
  for (const Placement pl : kPlacements) {
    for (const BoundaryCase& c : boundary_cases()) {
      // A discontinuous seam (multiplicity r + 1) has no f(a) = f(b) to meet.
      if (c.want != Boundary::periodic || c.ends == c.f.degree + c.g.degree + 1) {
        continue;
      }
      ++checked;
      CAPTURE(c.name);
      CAPTURE(pl.a);
      const bs::Bspline<T> f = make_spline<T>({placed(c.f, pl)}, 2, true, 17U);
      const bs::Bspline<T> g = make_spline<T>({placed(c.g, pl)}, 2, true, 18U);
      const bs::Bspline<T> h = bs::multiply<T>(f, g);
      const std::vector<Rational> a = {oracle::exact(domains_of(f)[0].first)};
      const std::vector<Rational> b = {oracle::exact(domains_of(f)[0].second)};
      REQUIRE(oracle::exact_value<T>(f, a) == oracle::exact_value<T>(f, b));
      REQUIRE(oracle::exact_value<T>(g, a) == oracle::exact_value<T>(g, b));
      const double bound = product_bound<T>(f, g, h, Route::converted);
      const std::vector<Rational> ha = oracle::exact_value<T>(h, a);
      const std::vector<Rational> hb = oracle::exact_value<T>(h, b);
      const std::vector<Rational> want =
          product_of(oracle::exact_value<T>(f, a), true, oracle::exact_value<T>(g, a), true);
      REQUIRE(ha.size() == want.size());
      for (std::size_t k = 0; k < ha.size(); ++k) {
        CAPTURE(k);
        CHECK(static_cast<double>(abs_of(ha[k] - hb[k])) <= 2.0 * bound);
        CHECK(static_cast<double>(abs_of(ha[k] - want[k])) <= bound);
      }
    }
  }
  CHECK(checked == 6);  // three continuous-seam cases, two placements
}

TEST_CASE_TEMPLATE("product: boundary forms in two directions", T, double, float) {
  // Oracle: the tabulated forms per direction and exact-rational evaluation on a tensor grid.
  // Direction 0 periodic by periodic and direction 1 open by open; then direction 0 unclamped by
  // periodic and direction 1 periodic by open, whose product is unclamped in direction 0 and open
  // in direction 1. The directions convert independently.
  struct Surface {
    const char* name;
    Dir f0;
    Dir g0;
    Dir f1;
    Dir g1;
    Boundary want0;
    Boundary want1;
    std::int64_t ends0;
  };
  const std::vector<Surface> surfaces = {
      {"periodic by periodic, open by open", kPer2, kPer3, kOpen2, kOpen3, Boundary::periodic,
       Boundary::open, 4},
      {"unclamped by periodic, periodic by open", kUnc2, kPer2, kPer3, kOpen3, Boundary::unclamped,
       Boundary::open, 3},
  };
  for (const Surface& sf : surfaces) {
    for (const Kind kind : kKinds) {
      CAPTURE(sf.name);
      CAPTURE(kind.f_rational);
      CAPTURE(kind.g_rational);
      // Direction 0 on [2, 6], direction 1 on [-1, 3].
      const Placement p0{2.0, 4.0};
      const Placement p1{-1.0, 4.0};
      const bs::Bspline<T> f =
          make_spline<T>({placed(sf.f0, p0), placed(sf.f1, p1)}, 1, kind.f_rational, 19U);
      const bs::Bspline<T> g =
          make_spline<T>({placed(sf.g0, p0), placed(sf.g1, p1)}, 1, kind.g_rational, 20U);
      const bs::Bspline<T> h = bs::multiply<T>(f, g);
      expect_boundary<T>(dir_space<T>(h, 0), sf.want0);
      expect_boundary<T>(dir_space<T>(h, 1), sf.want1);
      CHECK(end_multiplicities<T>(dir_space<T>(h, 0)) == std::make_pair(sf.ends0, sf.ends0));
      CHECK(domains_of(h) == domains_of(f));
      expect_product_values<T>(f, g, h, Route::converted, 2);
    }
  }
}

TEST_CASE_TEMPLATE("product: knots closer than the tolerance are one breakpoint", T, double,
                   float) {
  // Oracle: tabulated structure. g's double knot sits a quarter of the tolerance from f's simple
  // knot at 0.5 (a few units in the last place): one breakpoint, near 0.5, of multiplicity
  // max(m_f + q, m_g + p) = max(1 + 2, 2 + 3) = 5. Four tolerances away they are two breakpoints:
  // 0.5 only in f, 1 + 2 = 3, and the other only in g, 2 + 3 = 5.
  const Dir fd{clamped(3, {0.5}), 3};
  const bs::Bspline<T> f = make_spline<T>({fd}, 1, false, 7U);
  const double tol = dir_space<T>(f, 0).tolerance();
  for (const double factor : {0.25, 4.0}) {
    CAPTURE(factor);
    const double knot = static_cast<double>(static_cast<T>(0.5 + factor * tol));
    REQUIRE(knot != 0.5);
    const Dir gd{clamped(2, {knot, knot}), 2};
    const bs::Bspline<T> g = make_spline<T>({gd}, 1, false, 8U);
    const bs::Bspline<T> h = bs::multiply<T>(f, g);
    const std::vector<Break> got = interior_structure<T>(dir_space<T>(h, 0));
    if (factor < 1.0) {
      expect_structure(got, {{0.5, 5}}, tol);
    } else {
      expect_structure(got, {{0.5, 3}, {knot, 5}}, tol);
    }
  }
}

TEST_CASE_TEMPLATE("product: the stereographic circle, closed forms", T, double, float) {
  // Oracle: closed forms. x = (1 - t^2) / (1 + t^2) and y = 2 t / (1 + t^2) have the homogeneous
  // Bernstein control points (1, 1), (1, 1), (0, 2) and (0, 1), (1, 1), (2, 2) on [0, 1], integers
  // all, so the operands are exact. At rational t the products are rational numbers known in
  // closed form: x y = 2 t (1 - t^2) / (1 + t^2)^2, x^2 = (1 - t^2)^2 / (1 + t^2)^2 and
  // y^2 = 4 t^2 / (1 + t^2)^2 (which add to 1). The comparison is of the quotient N / W of the
  // product's homogeneous value against the closed form c, exactly: |N / W - c| <= (B + |c| B) /
  // (W_c - B), with B the product bound on N and on W and W_c = (1 + t^2)^2 the weight of the
  // exact product.
  const Dir quad{clamped(2, {}), 2};
  const bs::Bspline<T> x0 = curve<T>(quad, {1, 1, 1, 1, 0, 2}, true);
  const bs::Bspline<T> y0 = curve<T>(quad, {0, 1, 1, 1, 2, 2}, true);
  const auto quotient_error = [](const bs::Bspline<T>& h, const Rational& t, const Rational& want,
                                 double bound) {
    const std::vector<Rational> v = oracle::exact_value<T>(h, std::vector<Rational>{t});
    const Rational weight = (1 + t * t) * (1 + t * t);
    const double q_err = static_cast<double>(abs_of(v[0] / v[1] - want));
    const double q_bound =
        (bound + static_cast<double>(abs_of(want)) * bound) / (static_cast<double>(weight) - bound);
    return q_err / q_bound;
  };
  for (const bool refine : {false, true}) {
    CAPTURE(refine);
    const bs::Bspline<T> x = refine ? insert_in<T>(x0, 0, {0.5}) : x0;
    const bs::Bspline<T> y = refine ? insert_in<T>(y0, 0, {0.25, 0.75}) : y0;
    const bs::Bspline<T> xy = bs::multiply<T>(x, y);
    const bs::Bspline<T> xx = bs::multiply<T>(x, x);
    const bs::Bspline<T> yy = bs::multiply<T>(y, y);
    CHECK(xy.degrees()[0] == 4);
    const double bound = product_bound<T>(x, y, xy, Route::projected);
    for (const Rational& t :
         {Rational(1, 5), Rational(1, 3), Rational(3, 8), Rational(2, 3), Rational(7, 8)}) {
      CAPTURE(static_cast<double>(t));
      const Rational d = 1 + t * t;
      CHECK(quotient_error(xy, t, 2 * t * (1 - t * t) / (d * d), bound) <= 1.0);
      CHECK(quotient_error(xx, t, (1 - t * t) * (1 - t * t) / (d * d), bound) <= 1.0);
      CHECK(quotient_error(yy, t, 4 * t * t / (d * d), bound) <= 1.0);
    }
  }
}

TEST_CASE("product: refusals and their messages") {
  using pantr::value_error;
  const Dir unit{clamped(2, {0.5}), 2};
  const Dir wide{{0.0, 0.0, 0.0, 2.0, 2.0, 2.0}, 2};
  const bs::Bspline<double> f1 = make_spline<double>({unit}, 1, false, 1U);
  const bs::Bspline<double> f1_rank2 = make_spline<double>({unit}, 2, false, 1U);
  const bs::Bspline<double> f2 = make_spline<double>({unit, unit}, 1, false, 2U);
  const bs::Bspline<double> f2_rank2 = make_spline<double>({unit, unit}, 2, false, 2U);
  const bs::Bspline<double> f3 = make_spline<double>({unit, unit, unit}, 1, false, 3U);
  const auto product = [](const bs::Bspline<double>& f, const bs::Bspline<double>& g) {
    return bs::multiply<double>(f, g);
  };

  // Dimension. The one-direction wording when f is 1D, the multi-direction one otherwise.
  CHECK_THROWS_WITH_AS(static_cast<void>(product(f1, f2)), "g must be a 1D B-spline, got dim=2",
                       value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(product(f2, f3)),
                       "f and g must have the same parametric dimension. Got f.dim=2 and g.dim=3.",
                       value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(product(f2, f1)),
                       "f and g must have the same parametric dimension. Got f.dim=2 and g.dim=1.",
                       value_error);

  // Rank, in the one-direction and the multi-direction wording. The rank, not the number of
  // stored columns: a rational rank-1 spline and a non-rational rank-2 one both store two.
  CHECK_THROWS_WITH_AS(static_cast<void>(product(f1, f1_rank2)),
                       "f and g must have the same rank, got 1 and 2", value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(product(f2, f2_rank2)),
                       "f and g must have the same rank. Got 1 and 2.", value_error);
  const bs::Bspline<double> rational_rank1 = make_spline<double>({unit}, 1, true, 4U);
  CHECK_THROWS_WITH_AS(static_cast<void>(product(rational_rank1, f1_rank2)),
                       "f and g must have the same rank, got 1 and 2", value_error);

  // Order of the checks: dimension before rank before domain.
  CHECK_THROWS_WITH_AS(static_cast<void>(product(f1, f2_rank2)),
                       "g must be a 1D B-spline, got dim=2", value_error);
  const bs::Bspline<double> wide_rank2 = make_spline<double>({wide}, 2, false, 5U);
  CHECK_THROWS_WITH_AS(static_cast<void>(product(f1, wide_rank2)),
                       "f and g must have the same rank, got 1 and 2", value_error);

  // Domain: the one-direction message, then the direction in the multi-direction one.
  const bs::Bspline<double> wide1 = make_spline<double>({wide}, 1, false, 6U);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(product(f1, wide1)),
      "f and g must share the same parametric domain. Got (0.0, 1.0) and (0.0, 2.0).", value_error);
  const bs::Bspline<double> wide_second = make_spline<double>({unit, wide}, 1, false, 7U);
  CHECK_THROWS_WITH_AS(static_cast<void>(product(f2, wide_second)),
                       "f and g must share the same parametric domain in direction 1. Got (0.0, "
                       "1.0) and (0.0, 2.0).",
                       value_error);
  // Only the left end differs.
  const Dir shifted{{-1.0, -1.0, -1.0, 1.0, 1.0, 1.0}, 2};
  CHECK_THROWS_WITH_AS(
      static_cast<void>(product(f1, make_spline<double>({shifted}, 1, false, 9U))),
      "f and g must share the same parametric domain. Got (0.0, 1.0) and (-1.0, 1.0).",
      value_error);
  const bs::Bspline<double> wide_first = make_spline<double>({wide, unit}, 1, false, 8U);
  CHECK_THROWS_WITH_AS(static_cast<void>(product(f2, wide_first)),
                       "f and g must share the same parametric domain in direction 0. Got (0.0, "
                       "1.0) and (0.0, 2.0).",
                       value_error);
}

TEST_CASE_TEMPLATE("product: the domains agree up to the tolerance and no further", T, double,
                   float) {
  // Oracle: the documented rule, domains equal if their ends differ by at most the larger
  // tolerance of the operands. An end 4 units in the last place away is the same end (the
  // tolerance is 8 epsilon); 32 away is not.
  const Dir unit{clamped(2, {0.5}), 2};
  const bs::Bspline<T> f = make_spline<T>({unit}, 1, false, 1U);
  const double eps = static_cast<double>(std::numeric_limits<T>::epsilon());
  REQUIRE(dir_space<T>(f, 0).tolerance() > 4.0 * eps);
  REQUIRE(dir_space<T>(f, 0).tolerance() < 32.0 * eps);
  const double close_end = 1.0 + 4.0 * eps;
  const double distant_end = 1.0 + 32.0 * eps;
  const Dir close_dir{{0.0, 0.0, 0.0, close_end, close_end, close_end}, 2};
  const Dir distant_dir{{0.0, 0.0, 0.0, distant_end, distant_end, distant_end}, 2};
  CHECK_NOTHROW(static_cast<void>(bs::multiply<T>(f, make_spline<T>({close_dir}, 1, false, 2U))));
  CHECK_THROWS_AS(
      static_cast<void>(bs::multiply<T>(f, make_spline<T>({distant_dir}, 1, false, 3U))),
      pantr::value_error);
}

// ---------------------------------------------------------------------------------------------
// derivative_rational
// ---------------------------------------------------------------------------------------------

namespace {

/// The columns `[first, first + count)` of a B-spline's control points, as a non-rational spline
/// over the same space.
template <class T>
bs::Bspline<T> columns_of(const bs::Bspline<T>& b, std::size_t first, std::size_t count) {
  const std::size_t cp = b.cp_size();
  const std::span<const T> src = b.control_points();
  std::vector<T> out;
  for (std::size_t i = 0; i < src.size() / cp; ++i) {
    for (std::size_t c = 0; c < count; ++c) {
      out.push_back(src[i * cp + first + c]);
    }
  }
  return bs::Bspline<T>(b.space(), sp(out), false);
}

/// Bounds on the error of the homogeneous numerator `N` and weight `W` of the derivative of a
/// rational spline `(A, w)` in one direction, against the quotient rule `N_c = A'_c w - A_c w'`,
/// `W_c = w^2`. `N` is `A' w - A w'` built from the degree-keeping derivatives (each within the
/// conservative tier of `pantr::tolerance` times its scale, the bound the derivative tests assert
/// for that quantity), two products (`product_units`, `Route` as given), a subtraction of control
/// points (one rounding of the larger term) and a refinement of both to a common knot vector (the
/// Oslo count `sum_d (6 (2 p_d) + 1)` of the file header); `W` is `w^2` and that refinement.
template <class T>
std::pair<double, double> derivative_bounds(const bs::Bspline<T>& b, std::size_t direction,
                                            Route route) {
  const std::size_t rank = b.rank();
  const bs::Bspline<T> a = columns_of<T>(b, 0, rank);
  const bs::Bspline<T> w = columns_of<T>(b, rank, 1);
  const auto scale = [](const bs::Bspline<T>& s) {
    return (std::max)(1.0, oracle::max_abs_control_point<T>(s));
  };
  const double m_a = scale(a);
  const double m_w = scale(w);
  const double m_da =
      scale(bs::derivative_keep_degree_nonrational<T>(a, static_cast<long long>(direction)));
  const double m_dw =
      scale(bs::derivative_keep_degree_nonrational<T>(w, static_cast<long long>(direction)));
  const double tier = static_cast<double>(pantr::tolerance::get_conservative<T>());
  const double u = unit_roundoff<T>();
  double refine = 0.0;
  for (const std::size_t p : b.degrees()) {
    refine += 12.0 * static_cast<double>(p) + 1.0;
  }
  const double units = product_units(b.degrees(), b.degrees(), route, nonopen_directions<T>(b));
  const double numerator =
      tier * (m_da * m_w + m_dw * m_a) + (units + 1.0 + refine) * u * (m_da * m_w + m_a * m_dw);
  const double weight = (units + refine) * u * m_w * m_w;
  return {numerator, weight};
}

/// The worst ratio, over a grid, of the error of the derivative `d` of the rational `b` to its
/// bound, for the homogeneous numerator, the weight and the quotient `N / W`.
struct QuotientRatios {
  double numerator = 0.0;
  double weight = 0.0;
  double quotient = 0.0;
};

template <class T>
QuotientRatios derivative_ratios(const bs::Bspline<T>& b, const bs::Bspline<T>& d,
                                 std::size_t direction, Route route,
                                 const std::vector<std::vector<Rational>>& points) {
  const std::size_t rank = b.rank();
  const auto [bound_n, bound_w] = derivative_bounds<T>(b, direction, route);
  std::vector<int> orders(b.dim(), 0);
  orders[direction] = 1;
  QuotientRatios worst;
  for (const std::vector<Rational>& x : points) {
    const std::vector<Rational> value = oracle::exact_value<T>(b, x);
    const std::vector<Rational> slope = oracle::exact_value<T>(b, x, orders);
    const std::vector<Rational> got = oracle::exact_value<T>(d, x);
    REQUIRE(got.size() == rank + 1);
    const Rational w = value[rank];
    const Rational w_prime = slope[rank];
    const Rational want_w = w * w;
    worst.weight =
        (std::max)(worst.weight, static_cast<double>(abs_of(got[rank] - want_w)) / bound_w);
    for (std::size_t c = 0; c < rank; ++c) {
      const Rational want_n = slope[c] * w - value[c] * w_prime;
      worst.numerator =
          (std::max)(worst.numerator, static_cast<double>(abs_of(got[c] - want_n)) / bound_n);
      const Rational want_q = want_n / want_w;
      const double q_err = static_cast<double>(abs_of(got[c] / got[rank] - want_q));
      const double q_bound = (bound_n + static_cast<double>(abs_of(want_q)) * bound_w) /
                             (static_cast<double>(want_w) - bound_w);
      worst.quotient = (std::max)(worst.quotient, q_err / q_bound);
    }
  }
  return worst;
}

template <class T>
void expect_derivative_at(const bs::Bspline<T>& b, const bs::Bspline<T>& d, std::size_t direction,
                          Route route, const std::vector<std::vector<Rational>>& points) {
  const QuotientRatios r = derivative_ratios<T>(b, d, direction, route, points);
  CAPTURE(r.numerator);
  CAPTURE(r.weight);
  CAPTURE(r.quotient);
  CHECK(r.numerator <= 1.0);
  CHECK(r.weight <= 1.0);
  CHECK(r.quotient <= 1.0);
}

template <class T>
void expect_derivative_values(const bs::Bspline<T>& b, const bs::Bspline<T>& d,
                              std::size_t direction, Route route, std::size_t step = 1) {
  expect_derivative_at<T>(b, d, direction, route, grid_of(domains_of(d), step));
}

struct DerivativeCase {
  const char* name;
  std::vector<Dir> dirs;
  std::size_t rank;
  std::size_t direction;
  Route route;
  /// Interior structure of the result in `direction`: a knot of multiplicity `m` at degree `p`
  /// becomes `m + p + 1` at degree `2 p` (numerator `A' w`: `max(m + 1 + p, m + p)`; `w^2`:
  /// `m + p`; the common refinement takes the larger), worked by hand per case.
  std::vector<Break> want;
  /// Multiplicity of the result's domain ends in `direction`: `2 p + 1` for an open direction,
  /// the seam multiplicity `p + m_bdy + 1` for a periodic one.
  std::int64_t ends;
};

const std::vector<DerivativeCase>& derivative_cases() {
  static const Dir per_c1{periodic_seam_knots(3, 2, {0.25, 0.5, 0.75}), 3, true};
  static const std::vector<DerivativeCase> cases = {
      // p = 2, m = 1: 1 + 2 + 1 = 4 at each knot.
      {"quadratic, two knots",
       {{clamped(2, {0.25, 0.5}), 2}},
       2,
       0,
       Route::projected,
       {{0.25, 4}, {0.5, 4}},
       5},
      // p = 3, m = 2: 2 + 3 + 1 = 6.
      {"cubic, a double knot",
       {{clamped(3, {0.5, 0.5}), 3}},
       1,
       0,
       Route::projected,
       {{0.5, 6}},
       7},
      // p = 2, m = 2 = p: the derivative jumps, 2 + 2 + 1 = 5 = 2 p + 1 (C^-1).
      {"quadratic, a C0 knot",
       {{clamped(2, {0.5, 0.5}), 2}},
       1,
       0,
       Route::projected,
       {{0.5, 5}},
       5},
      // p = 1, m = 1: 1 + 1 + 1 = 3 = 2 p + 1.
      {"linear", {{clamped(1, {0.5}), 1}}, 2, 0, Route::projected, {{0.5, 3}}, 3},
      {"cubic, no knot", {{clamped(3, {}), 3}}, 1, 0, Route::projected, {}, 7},
      // Surface, p = (2, 1). Direction 0: m = 1, 1 + 2 + 1 = 4.
      {"surface, direction 0",
       {{clamped(2, {0.5}), 2}, {clamped(1, {0.25}), 1}},
       1,
       0,
       Route::projected,
       {{0.5, 4}},
       5},
      // Direction 1 of the same surface: p = 1, 1 + 1 + 1 = 3.
      {"surface, direction 1",
       {{clamped(2, {0.5}), 2}, {clamped(1, {0.25}), 1}},
       1,
       1,
       Route::projected,
       {{0.25, 3}},
       3},
      // Periodic, p = 2, seam 1: every breakpoint 1 + 2 + 1 = 4.
      {"periodic quadratic",
       {{periodic_knots(4, 2), 2, true}},
       1,
       0,
       Route::converted,
       {{0.25, 4}, {0.5, 4}, {0.75, 4}},
       4},
      // p = 3, seam 1: 1 + 3 + 1 = 5.
      {"periodic cubic",
       {{periodic_knots(4, 3), 3, true}},
       2,
       0,
       Route::converted,
       {{0.25, 5}, {0.5, 5}, {0.75, 5}},
       5},
      // p = 3, seam 2 (C^1): the breakpoints 1 + 3 + 1 = 5, the seam 2 + 3 + 1 = 6.
      {"periodic cubic with a C1 seam",
       {per_c1},
       1,
       0,
       Route::converted,
       {{0.25, 5}, {0.5, 5}, {0.75, 5}},
       6},
      // The same on other domains, whose period is not 1 and whose origin is not 0.
      {"periodic quadratic on [2, 6]",
       {placed({periodic_knots(4, 2), 2, true}, kPlacements[1])},
       1,
       0,
       Route::converted,
       {{3.0, 4}, {4.0, 4}, {5.0, 4}},
       4},
      {"periodic cubic with a C1 seam on [-1, 3]",
       {placed(per_c1, Placement{-1.0, 4.0})},
       1,
       0,
       Route::converted,
       {{0.0, 5}, {1.0, 5}, {2.0, 5}},
       6},
      // A periodic direction beside an open one: differentiate either.
      {"surface periodic in direction 1, derivative in direction 0",
       {{clamped(2, {0.5}), 2}, {periodic_knots(4, 2), 2, true}},
       1,
       0,
       Route::converted,
       {{0.5, 4}},
       5},
      {"surface periodic in direction 1, derivative in direction 1",
       {{clamped(2, {0.5}), 2}, {periodic_knots(4, 2), 2, true}},
       1,
       1,
       Route::converted,
       {{0.25, 4}, {0.5, 4}, {0.75, 4}},
       4},
  };
  return cases;
}

}  // namespace

TEST_CASE_TEMPLATE("derivative_rational: the quotient rule at exact rational points", T, double,
                   float) {
  // Oracle: exact-rational evaluation of the numerator A, the weight w and their first
  // derivatives (bspline_exact_eval.hpp differentiates the recursive basis by the Leibniz rule),
  // assembled as (A' w - A w') / w^2 and compared with N / W of the result, which is rational.
  // The homogeneous N and W are compared as well, each with its own bound (derivative_bounds).
  for (const DerivativeCase& c : derivative_cases()) {
    CAPTURE(c.name);
    const bs::Bspline<T> b = make_spline<T>(c.dirs, c.rank, true, 41U);
    const bs::Bspline<T> d = bs::derivative_rational<T>(b, static_cast<long long>(c.direction));
    CHECK(d.rational());
    CHECK(d.rank() == c.rank);
    expect_derivative_values<T>(b, d, c.direction, c.route, c.dirs.size() > 1 ? 2 : 1);
  }
}

TEST_CASE_TEMPLATE("derivative_rational: the result has the doubled degree and the tabulated knots",
                   T, double, float) {
  // Oracle: tabulated structure (derivative_cases(), by hand). Degree 2 p in every direction
  // (`multiply` adds the operands' degrees in every direction); the differentiated direction keeps
  // a periodic form exactly when it had one, with the seam multiplicity p + m + 1 of the header of
  // derivative_rational.hpp; the other directions keep their forms.
  for (const DerivativeCase& c : derivative_cases()) {
    CAPTURE(c.name);
    const bs::Bspline<T> b = make_spline<T>(c.dirs, c.rank, true, 42U);
    const bs::Bspline<T> d = bs::derivative_rational<T>(b, static_cast<long long>(c.direction));
    REQUIRE(d.dim() == b.dim());
    for (std::size_t k = 0; k < d.dim(); ++k) {
      CAPTURE(k);
      CHECK(d.degrees()[k] == 2 * b.degrees()[k]);
      CHECK(dir_space<T>(d, k).periodic() == c.dirs[k].periodic);
    }
    const bs::BsplineSpace1D<T> s = dir_space<T>(d, c.direction);
    expect_structure(interior_structure<T>(s), c.want, 0.0);
    CHECK(end_multiplicities<T>(s) == std::make_pair(c.ends, c.ends));
  }
}

TEST_CASE_TEMPLATE("derivative_rational: the stereographic circle, closed forms", T, double,
                   float) {
  // Oracle: closed forms. The circle c(t) = (x, y) of the product tests has the homogeneous
  // control points (1, 0, 1), (1, 1, 1), (0, 2, 2); its derivative is x' = -4 t / (1 + t^2)^2 and
  // y' = 2 (1 - t^2) / (1 + t^2)^2, whose weight is (1 + t^2)^2. The quotient N / W of the result
  // is compared exactly, within (B_N + |c| B_W) / (W_c - B_W) (derivative_bounds), on the Bezier
  // curve and on the curve refined at 1/4 and 1/2.
  const Dir quad{clamped(2, {}), 2};
  const bs::Bspline<T> c0 = curve<T>(quad, {1, 0, 1, 1, 1, 1, 0, 2, 2}, true);
  for (const bool refine : {false, true}) {
    CAPTURE(refine);
    const bs::Bspline<T> c = refine ? insert_in<T>(c0, 0, {0.25, 0.5}) : c0;
    const bs::Bspline<T> d = bs::derivative_rational<T>(c, 0);
    const auto [bound_n, bound_w] = derivative_bounds<T>(c, 0, Route::projected);
    for (const Rational& t :
         {Rational(1, 5), Rational(1, 3), Rational(3, 8), Rational(2, 3), Rational(7, 8)}) {
      CAPTURE(static_cast<double>(t));
      const Rational w = (1 + t * t) * (1 + t * t);
      const std::vector<Rational> v = oracle::exact_value<T>(d, std::vector<Rational>{t});
      REQUIRE(v.size() == 3);
      const std::array<Rational, 2> want = {-4 * t / w, 2 * (1 - t * t) / w};
      for (std::size_t k = 0; k < 2; ++k) {
        CAPTURE(k);
        const double q_err = static_cast<double>(abs_of(v[k] / v[2] - want[k]));
        const double q_bound = (bound_n + static_cast<double>(abs_of(want[k])) * bound_w) /
                               (static_cast<double>(w) - bound_w);
        CHECK(q_err <= q_bound);
      }
    }
  }
}

TEST_CASE_TEMPLATE("derivative_rational: a periodic derivative closes at the seam", T, double,
                   float) {
  // Oracle: exact-rational evaluation at the two ends of the periodic direction. The operand's
  // seam is C^1 or smoother in every case, so its exact value and first derivative agree at the
  // two ends (checked, from the exact evaluation), and the quotient rule gives one value there;
  // the derivative must meet it from both sides.
  std::size_t checked = 0;
  for (const DerivativeCase& c : derivative_cases()) {
    if (!c.dirs[c.direction].periodic) {
      continue;
    }
    ++checked;
    CAPTURE(c.name);
    const bs::Bspline<T> b = make_spline<T>(c.dirs, c.rank, true, 43U);
    const bs::Bspline<T> d = bs::derivative_rational<T>(b, static_cast<long long>(c.direction));
    const std::vector<Domain> doms = domains_of(d);
    std::vector<std::vector<Rational>> ends(2);
    for (std::size_t k = 0; k < d.dim(); ++k) {
      const Rational lo = oracle::exact(doms[k].first);
      const Rational hi = oracle::exact(doms[k].second);
      ends[0].push_back(k == c.direction ? lo : lo + (hi - lo) * Rational(37, 100));
      ends[1].push_back(k == c.direction ? hi : lo + (hi - lo) * Rational(37, 100));
    }
    std::vector<int> orders(d.dim(), 0);
    orders[c.direction] = 1;
    REQUIRE(oracle::exact_value<T>(b, ends[0]) == oracle::exact_value<T>(b, ends[1]));
    REQUIRE(oracle::exact_value<T>(b, ends[0], orders) ==
            oracle::exact_value<T>(b, ends[1], orders));
    expect_derivative_at<T>(b, d, c.direction, c.route, ends);
  }
  CHECK(checked == 6);
}

TEST_CASE("derivative_rational: refusals and their messages") {
  using pantr::value_error;
  const Dir quad{clamped(2, {0.5}), 2};
  const Dir constant{clamped(0, {0.5}), 0};
  const bs::Bspline<double> rational = make_spline<double>({quad}, 1, true, 1U);
  const bs::Bspline<double> plain = make_spline<double>({quad}, 1, false, 1U);
  const bs::Bspline<double> surface = make_spline<double>({quad, constant}, 1, true, 2U);
  const auto derive = [](const bs::Bspline<double>& b, long long direction) {
    return bs::derivative_rational<double>(b, direction);
  };

  CHECK_THROWS_WITH_AS(static_cast<void>(derive(plain, 0)),
                       "derivative_rational is for rational B-splines.", value_error);
  // The kind of the spline is checked before the direction.
  CHECK_THROWS_WITH_AS(static_cast<void>(derive(plain, 7)),
                       "derivative_rational is for rational B-splines.", value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(derive(rational, 1)),
                       "direction must be in [0, 1), got 1.", value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(derive(rational, -1)),
                       "direction must be in [0, 1), got -1.", value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(derive(surface, 2)), "direction must be in [0, 2), got 2.",
                       value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(derive(surface, 1)),
                       "Derivative of a degree-0 B-spline is not defined.", value_error);

  // A periodic direction whose seam is C^0 (seam multiplicity = degree) has no periodic
  // derivative; one multiplicity less (C^1 at degree 2) is served (derivative_cases()).
  const Dir c0_quadratic{periodic_seam_knots(2, 2, {0.25, 0.5, 0.75}), 2, true};
  const Dir c0_cubic{periodic_seam_knots(3, 3, {0.25, 0.5, 0.75}), 3, true};
  CHECK_THROWS_WITH_AS(
      static_cast<void>(derive(make_spline<double>({c0_quadratic}, 1, true, 3U), 0)),
      "Cannot differentiate a periodic rational B-spline whose seam is C^0 (seam multiplicity 2 "
      "equals the degree 2): its derivative is discontinuous at the seam and has no periodic "
      "representation.",
      value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(derive(make_spline<double>({c0_cubic}, 1, true, 4U), 0)),
      "Cannot differentiate a periodic rational B-spline whose seam is C^0 (seam multiplicity 3 "
      "equals the degree 3): its derivative is discontinuous at the seam and has no periodic "
      "representation.",
      value_error);
  // The refusal concerns the direction differentiated: the same surface, differentiated along
  // its open direction, is served.
  const Dir open_dir{clamped(2, {0.5}), 2};
  CHECK_NOTHROW(
      static_cast<void>(derive(make_spline<double>({open_dir, c0_quadratic}, 1, true, 5U), 0)));
  CHECK_THROWS_WITH_AS(
      static_cast<void>(derive(make_spline<double>({open_dir, c0_quadratic}, 1, true, 5U), 1)),
      "Cannot differentiate a periodic rational B-spline whose seam is C^0 (seam multiplicity 2 "
      "equals the degree 2): its derivative is discontinuous at the seam and has no periodic "
      "representation.",
      value_error);

  // The degree-keeping derivative of the numerator refuses what it refuses for a non-rational
  // spline: an unclamped direction, and a C^-1 interior knot.
  const bs::Bspline<double> unclamped =
      make_spline<double>({{{0, 1, 2, 3, 4, 5, 6, 7, 8}, 2}}, 1, true, 6U);
  const std::string got = message_of([&] { static_cast<void>(derive(unclamped, 0)); });
  CHECK(got.find("Degree-preserving derivative in direction 0 (its degree-1 hodograph) needs a "
                 "clamped knot vector") == 0);
  const bs::Bspline<double> jump =
      make_spline<double>({{clamped(2, {0.5, 0.5, 0.5}), 2}}, 1, true, 7U);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(derive(jump, 0)),
      "Degree-preserving derivative in direction 0 (its degree-1 hodograph) cannot walk a knot "
      "vector that repeats the knot 0.5 3 times at degree 1: a knot may repeat at most degree + 1 "
      "= 2 times.",
      value_error);
}

// ---------------------------------------------------------------------------------------------
// linalg::lstsq_min_norm
// ---------------------------------------------------------------------------------------------

namespace {

template <class T>
la::RowMatrix<T> matrix_of(std::size_t rows, std::size_t cols, const std::vector<double>& v) {
  REQUIRE(v.size() == rows * cols);
  la::RowMatrix<T> m(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      m(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
          static_cast<T>(v[i * cols + j]);
    }
  }
  return m;
}

/// Whether the solution has the extents `(rows, cols)` and each entry is within `tol` of `want`
/// (row-major).
template <class T>
bool solution_matches(const pantr::NdArray<T>& got, std::size_t rows, std::size_t cols,
                      const std::vector<double>& want, double tol) {
  if (got.shape().size() != 2 || got.shape()[0] != rows || got.shape()[1] != cols) {
    return false;
  }
  bool ok = true;
  for (std::size_t i = 0; i < rows * cols; ++i) {
    const double err = std::abs(static_cast<double>(got.data()[i]) - want[i]);
    INFO("entry " << i << ": got " << static_cast<double>(got.data()[i]) << ", want " << want[i]
                  << ", error " << err << ", tolerance " << tol);
    ok = ok && err <= tol;
  }
  return ok;
}

constexpr double kEpsDouble = std::numeric_limits<double>::epsilon();

/// The error allowed for a solution of size `scale` from a backward-stable SVD solve in double of
/// a system of `n` unknowns and condition number `kappa_power` (the condition number, or its
/// square for the residual-dependent part of an inconsistent system), then rounded to `T`:
/// `32 n kappa eps_double scale + u_T scale`. The factor `32 n` is admitted as a heuristic: the
/// polynomial in `n` of an SVD solve's backward error is not derived here, only bounded generously
/// for the sizes below; the second term is the rounding of the result to `T`.
template <class T>
double solve_tolerance(std::size_t n, double kappa_power, double scale) {
  return (32.0 * static_cast<double>(n) * kappa_power * kEpsDouble + unit_roundoff<T>()) * scale;
}

}  // namespace

TEST_CASE_TEMPLATE("lstsq_min_norm: a square full-rank system returns its constructed solution", T,
                   double, float) {
  // Oracle: a solution fixed by construction. A is the tridiagonal (2, 1; 1, 2, 1; 1, 2), whose
  // eigenvalues 2 - sqrt 2, 2, 2 + sqrt 2 give the condition number (2 + sqrt 2) / (2 - sqrt 2) =
  // 5.83; X is chosen and B = A X formed by hand (small dyadic numbers, exact).
  const la::RowMatrix<T> a = matrix_of<T>(3, 3, {2, 1, 0, 1, 2, 1, 0, 1, 2});
  const std::vector<double> x = {1, 2, -1, 0.5, 3, -2};
  const la::RowMatrix<T> b = matrix_of<T>(3, 2, {1, 4.5, 2, 1, 5, -3.5});
  const pantr::NdArray<T> got = la::lstsq_min_norm<T>(a, b);
  CHECK(solution_matches<T>(got, 3, 2, x, solve_tolerance<T>(3, 5.83, 3.0)));
}

TEST_CASE_TEMPLATE(
    "lstsq_min_norm: an overdetermined inconsistent system gives the normal equations", T, double,
    float) {
  // Oracle: closed form. Fitting a line to (0, 1), (1, 3), (2, 2), (3, 5): with mean abscissa 3/2
  // and mean ordinate 11/4, slope = S_xy / S_xx = 5.5 / 5 = 1.1 and intercept = 11/4 - 1.1 (3/2) =
  // 1.1. A^T A = (4, 6; 6, 14) has eigenvalues 9 +- sqrt 61, so kappa(A)^2 = 14.1.
  const la::RowMatrix<T> a = matrix_of<T>(4, 2, {1, 0, 1, 1, 1, 2, 1, 3});
  const la::RowMatrix<T> b = matrix_of<T>(4, 1, {1, 3, 2, 5});
  const pantr::NdArray<T> got = la::lstsq_min_norm<T>(a, b);
  CHECK(solution_matches<T>(got, 2, 1, {1.1, 1.1}, solve_tolerance<T>(2, 14.2, 1.1)));
}

TEST_CASE_TEMPLATE("lstsq_min_norm: an underdetermined system returns the minimum-norm solution", T,
                   double, float) {
  // Oracle: x = A^T (A A^T)^-1 b for a full-row-rank A. A = (1, 0, 1; 0, 1, 1), A A^T = (2, 1; 1,
  // 2) with inverse (2, -1; -1, 2) / 3, b = (1, 2): (A A^T)^-1 b = (0, 1), x = (0, 1, 1), of norm
  // sqrt 2 where (1, 2, 0), another solution, has sqrt 5. kappa(A) = sqrt 3.
  const la::RowMatrix<T> a = matrix_of<T>(2, 3, {1, 0, 1, 0, 1, 1});
  const la::RowMatrix<T> b = matrix_of<T>(2, 1, {1, 2});
  const pantr::NdArray<T> got = la::lstsq_min_norm<T>(a, b);
  CHECK(solution_matches<T>(got, 3, 1, {0, 1, 1}, solve_tolerance<T>(3, 1.8, 1.0)));
}

TEST_CASE_TEMPLATE("lstsq_min_norm: a rank-deficient matrix gives the pseudoinverse solution", T,
                   double, float) {
  // Oracle: the pseudoinverse of a rank-one matrix in closed form, A = u v^T gives A^+ = v u^T /
  // (|u|^2 |v|^2). A is the 3 x 2 matrix of ones (u = (1, 1, 1), v = (1, 1), |u|^2 |v|^2 = 6).
  // b = (2, 2, 2) is consistent, A x = b, x = (1, 1) the minimum-norm solution of x_1 + x_2 = 2;
  // b = (1, 0, 0) is not, and the least-squares minimum-norm solution is v (u . b) / 6 = (1/6,
  // 1/6). The second singular value is rounding noise far below the rank cutoff; without the
  // cutoff its reciprocal would dominate the answer.
  const la::RowMatrix<T> a = matrix_of<T>(3, 2, {1, 1, 1, 1, 1, 1});
  const la::RowMatrix<T> b = matrix_of<T>(3, 2, {2, 1, 2, 0, 2, 0});
  const pantr::NdArray<T> got = la::lstsq_min_norm<T>(a, b);
  CHECK(solution_matches<T>(got, 2, 2, {1, 1.0 / 6.0, 1, 1.0 / 6.0},
                            solve_tolerance<T>(2, 1.0, 1.0)));
}

TEST_CASE_TEMPLATE("lstsq_min_norm: a null column of the matrix gets the entry zero", T, double,
                   float) {
  // Oracle: constructed. A = (1, 0, 2; 0, 0, 3) has a zero column, so the second unknown does not
  // enter A x and the minimum-norm solution sets it to 0. b = (5, 6): 3 x_3 = 6, x_3 = 2, x_1 =
  // 5 - 4 = 1.
  const la::RowMatrix<T> a = matrix_of<T>(2, 3, {1, 0, 2, 0, 0, 3});
  const la::RowMatrix<T> b = matrix_of<T>(2, 1, {5, 6});
  const pantr::NdArray<T> got = la::lstsq_min_norm<T>(a, b);
  CHECK(solution_matches<T>(got, 3, 1, {1, 0, 2}, solve_tolerance<T>(3, 4.0, 2.0)));
  CHECK(got.data()[1] == T(0));
}

TEST_CASE_TEMPLATE(
    "lstsq_min_norm: singular values at or below eps(double) max(m, n) s_max are zero", T, double,
    float) {
  // Oracle: the rule of numpy.linalg.lstsq with rcond = None: a singular value s_i <= eps max(m, n)
  // s_max is dropped. A = diag(s_1, s_2) (padded with zeros) has its entries as singular values,
  // exactly. For b = (1, 1), the kept answer is (1 / s_1, 1 / s_2) and the dropped one
  // (1 / s_1, 0). The cutoff is 3 eps s_1 for a 3 x 2 and a 2 x 3 matrix (the larger extent: 2 eps
  // would keep 2.5 eps), relative to the largest singular value (s_1 of 1, 1024 and 1/1024 tell
  // it from an absolute one), and the comparison is strict (s_2 = 3 eps s_1 is dropped).
  const double eps = kEpsDouble;
  struct Cutoff {
    double s1;
    double s2_over_eps_s1;
    bool kept;
  };
  const double big = 1024.0;
  const std::vector<Cutoff> cases = {
      {1.0, 2.5, false},       {1.0, 3.0, false},        {1.0, 3.5, true},
      {big, 2.5, false},       {big, 3.0, false},        {big, 3.5, true},
      {1.0 / big, 2.5, false}, {1.0 / big, 3.0, false},  {1.0 / big, 3.5, true},
      {1.0, 1e-9 / eps, true}, {1.0, 1e-17 / eps, false}};
  for (const Cutoff& c : cases) {
    for (const bool tall : {true, false}) {
      const double s2 = c.s2_over_eps_s1 * eps * c.s1;
      CAPTURE(c.s1);
      CAPTURE(c.s2_over_eps_s1);
      CAPTURE(tall);
      const la::RowMatrix<T> a = tall ? matrix_of<T>(3, 2, {c.s1, 0, 0, s2, 0, 0})
                                      : matrix_of<T>(2, 3, {c.s1, 0, 0, 0, s2, 0});
      const la::RowMatrix<T> b = tall ? matrix_of<T>(3, 1, {1, 1, 0}) : matrix_of<T>(2, 1, {1, 1});
      const pantr::NdArray<T> got = la::lstsq_min_norm<T>(a, b);
      const double first = 1.0 / static_cast<double>(static_cast<T>(c.s1));
      const double second = c.kept ? 1.0 / static_cast<double>(static_cast<T>(s2)) : 0.0;
      CHECK(solution_matches<T>(
          got, tall ? 2 : 3, 1,
          tall ? std::vector<double>{first, second} : std::vector<double>{first, second, 0},
          solve_tolerance<T>(2, 1.0, (std::max)(first, second))));
    }
  }
}

TEST_CASE("lstsq_min_norm: a single-precision matrix is decomposed in double") {
  // Oracle: numpy decomposes in double whatever the storage type. A = diag(1, 1e-9) held in
  // float has a second singular value, 1e-9, below float's eps(float) * 2 (2.4e-7) and so zero
  // for a single-precision decomposition, but far above eps(double) * 2: the answer for b = (1, 1)
  // is (1, 1 / stored 1e-9), not (1, 0).
  const la::RowMatrix<float> a = matrix_of<float>(2, 2, {1, 0, 0, 1e-9});
  const la::RowMatrix<float> b = matrix_of<float>(2, 1, {1, 1});
  const pantr::NdArray<float> got = la::lstsq_min_norm<float>(a, b);
  const double stored = static_cast<double>(static_cast<float>(1e-9));
  CHECK(
      solution_matches<float>(got, 2, 1, {1.0, 1.0 / stored}, solve_tolerance<float>(2, 1.0, 1e9)));
}

TEST_CASE_TEMPLATE("lstsq_min_norm: a non-finite entry of the matrix is refused", T, double,
                   float) {
  // Oracle: the documented refusal, with numpy's message. Eigen's SVD would return NaN singular
  // values that the rank cutoff silently drops. The entry may sit anywhere.
  const T not_a_number = std::numeric_limits<T>::quiet_NaN();
  const T infinity = std::numeric_limits<T>::infinity();
  const la::RowMatrix<T> b = matrix_of<T>(3, 1, {1, 2, 3});
  for (const T bad : {not_a_number, infinity, -infinity}) {
    for (const Eigen::Index at : {Eigen::Index{0}, Eigen::Index{4}, Eigen::Index{8}}) {
      la::RowMatrix<T> a = matrix_of<T>(3, 3, {2, 1, 0, 1, 2, 1, 0, 1, 2});
      a.data()[at] = bad;
      CAPTURE(static_cast<double>(bad));
      CAPTURE(at);
      CHECK_THROWS_WITH_AS(static_cast<void>(la::lstsq_min_norm<T>(a, b)),
                           "SVD did not converge in Linear Least Squares", pantr::value_error);
    }
  }
}

// Oracle: the knot vector itself. Its end runs are three copies of each domain end, the outer copy
// moved inside by a relative 1e-6 (far beyond the knot tolerance at every scale), so two copies
// of each end lie within the tolerance and the third does not. The count must stay two at every
// scale: a relative-tolerance leak (a fixed `rtol` window) would absorb the moved copy. Ported
// from the meaning of tests/test_knot_tolerance_absolute.py's boundary-multiplicity test, which
// read the same count through the product module's Python helper before the port. `double` only,
// as there: in `float` the 1e-6 move is inside the knot tolerance (8 eps(float) times the scale).
TEST_CASE_TEMPLATE("product: end-run multiplicities ignore a near-boundary knot at every scale", T,
                   double) {
  for (const double base : {1.0, 100.0, 1e6}) {
    std::vector<T> knots;
    for (const double k : {1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0}) {
      knots.push_back(static_cast<T>(k * base));
    }
    knots.front() = static_cast<T>(static_cast<double>(knots.front()) - 1e-6 * base);
    knots.back() = static_cast<T>(static_cast<double>(knots.back()) + 1e-6 * base);
    const pantr::bspline::BsplineSpace1D<T> space(std::span<const T>(knots), 2, false, false);
    const auto [left, right] =
        pantr::bspline::detail::boundary_multiplicities<T>(space, space.tolerance());
    CHECK(left == 2);
    CHECK(right == 2);
  }
}
