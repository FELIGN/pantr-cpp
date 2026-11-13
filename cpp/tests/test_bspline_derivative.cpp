// The partial derivative of a non-rational B-spline, and the common refinement of two B-splines.
//
// Oracle. The exact derivative of the exact-rational Cox-de Boor evaluation of the original
// (bspline_exact_eval.hpp differentiates the recursive definition of the basis by the Leibniz
// rule, so no hodograph formula of the library is reused), compared with the exact evaluation of
// the derivative spline's stored control points. The two differ by the rounding of those control
// points only. The derivative control points are `p (P[i+1] - P[i]) / (T[i+p+1] - T[i+1])`, formed
// in the storage type with four roundings, each correct: the difference of control points, the
// product by the integer `p`, the difference of knots in the denominator, and the quotient. Each
// stored point is therefore within `gamma_4 = 4 u / (1 - 4 u)` of its exact value (relative to
// itself, `u` the unit roundoff; the product of four factors `1 + delta`, `|delta| <= u`), and
// since the basis is a partition of unity the spline is within `gamma_4 max|Q|`. No statement of
// the kernel is contracted into a multiply-add, so no further term is owed. The degree-keeping
// derivative adds an elevation, whose tolerance is the conservative tier of `pantr::tolerance`
// (see test_bspline_degree.cpp).
#include <doctest/doctest.h>

#include "bspline_test_support.hpp"

#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/derivative.hpp>
#include <pantr/core/binomial.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/tolerance.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace bs = pantr::bspline;
namespace oracle = bspline_oracle;
using bspline_test::cast_all;
using bspline_test::clamped;
using bspline_test::Dir;
using bspline_test::exact_difference;
using bspline_test::knots_of;
using bspline_test::make_spline;
using bspline_test::message_of;
using bspline_test::periodic_knots;
using bspline_test::same_space;
using bspline_test::sp;
using bspline_test::tier_tolerance;
using oracle::Rational;

namespace {

template <class T>
double unit_roundoff() {
  return 0.5 * static_cast<double>(std::numeric_limits<T>::epsilon());
}

/// `gamma_4 = 4 u / (1 - 4 u)`: the relative error of a value formed by four correctly rounded
/// operations (the derivative control points, see the file header).
template <class T>
double gamma_4() {
  const double four_u = 4.0 * unit_roundoff<T>();
  return four_u / (1.0 - four_u);
}

/// The largest absolute difference, over the tensor grid of `bspline_test::sample_points` (eight
/// points spanning each direction's domain), between the exact value of `derived` and the exact
/// `direction`-derivative of `original`.
template <class T>
double derivative_error(const bs::Bspline<T>& original, const bs::Bspline<T>& derived,
                        std::size_t direction) {
  const std::size_t dim = original.dim();
  std::vector<std::vector<Rational>> axes(dim);
  for (std::size_t d = 0; d < dim; ++d) {
    const auto dom = original.space().spaces()[d].domain();
    axes[d] = bspline_test::sample_points(oracle::exact(static_cast<double>(dom.first)),
                                          oracle::exact(static_cast<double>(dom.second)));
  }
  std::vector<int> orders(dim, 0);
  orders[direction] = 1;
  double worst = 0.0;
  std::vector<std::size_t> idx(dim, 0);
  for (;;) {
    std::vector<Rational> x;
    for (std::size_t d = 0; d < dim; ++d) {
      x.push_back(axes[d][idx[d]]);
    }
    const auto want = oracle::exact_value<T>(original, x, orders);
    const auto got = oracle::exact_value<T>(derived, x);
    for (std::size_t c = 0; c < want.size(); ++c) {
      Rational diff = want[c] - got[c];
      if (diff < 0) {
        diff = -diff;
      }
      worst = (std::max)(worst, static_cast<double>(diff));
    }
    std::size_t d = dim;
    bool done = true;
    while (d-- > 0) {
      if (++idx[d] < axes[d].size()) {
        done = false;
        break;
      }
      idx[d] = 0;
    }
    if (done) {
      return worst;
    }
  }
}

template <class T>
double max_control_point(const bs::Bspline<T>& b) {
  return oracle::max_abs_control_point<T>(b);
}

struct Case {
  Dir dir;
};

// Clamped and unclamped vectors; smooth, C^0 and C^-1 interior knots.
const std::vector<Case> kCases = {
    {{clamped(1, {0.25, 0.75}), 1}},
    {{clamped(2, {0.5}), 2}},
    {{clamped(3, {0.25, 0.5, 0.75}), 3}},
    {{clamped(4, {0.5, 0.5}), 4}},
    {{clamped(3, {0.25, 0.25, 0.25, 0.75}), 3}},
    {{clamped(2, {0.5, 0.5, 0.5}), 2}},
    {{{0, 1, 2, 3, 4, 5, 6, 7, 8}, 2}},
    {{{-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5}, 3}},
};

}  // namespace

TEST_CASE_TEMPLATE("derivative: equals the exact derivative of the exact evaluation", T, double,
                   float) {
  for (const Case& c : kCases) {
    CAPTURE(c.dir.degree);
    const bs::Bspline<T> b =
        make_spline<T>({c.dir}, 2, false, 5U + static_cast<unsigned>(c.dir.knots.size()));
    const bs::Bspline<T> d = bs::derivative_nonrational<T>(b, 0);
    CHECK(d.degrees()[0] == static_cast<std::size_t>(c.dir.degree - 1));
    CHECK_FALSE(d.rational());
    // The knot vector without its first and last knots.
    const std::vector<T> all = knots_of<T>(b, 0);
    const std::vector<T> inner(all.begin() + 1, all.end() - 1);
    CHECK(knots_of<T>(d, 0) == inner);
    CHECK(derivative_error<T>(b, d, 0) <= gamma_4<T>() * max_control_point<T>(d));
  }
}

TEST_CASE_TEMPLATE("derivative: a periodic direction stays periodic", T, double, float) {
  for (const int degree : {2, 3}) {
    CAPTURE(degree);
    const std::vector<Dir> dirs = {{periodic_knots(4, degree), degree, true}};
    const bs::Bspline<T> b = make_spline<T>(dirs, 2, false, 19U + static_cast<unsigned>(degree));
    const bs::Bspline<T> d = bs::derivative_nonrational<T>(b, 0);
    CHECK(d.space().spaces()[0].periodic());
    CHECK(d.degrees()[0] == static_cast<std::size_t>(degree - 1));
    CHECK(derivative_error<T>(b, d, 0) <= gamma_4<T>() * max_control_point<T>(d));
  }
}

TEST_CASE_TEMPLATE("derivative: surfaces and volumes differentiate one direction", T, double,
                   float) {
  const std::vector<Dir> dirs = {
      {clamped(2, {0.5}), 2}, {clamped(3, {0.25, 0.75}), 3}, {clamped(2, {}), 2}};
  const bs::Bspline<T> b = make_spline<T>(dirs, 2, false, 33U);
  for (std::size_t direction = 0; direction < 3; ++direction) {
    CAPTURE(direction);
    const bs::Bspline<T> d = bs::derivative_nonrational<T>(b, static_cast<long long>(direction));
    for (std::size_t k = 0; k < 3; ++k) {
      if (k == direction) {
        CHECK(d.degrees()[k] == b.degrees()[k] - 1);
      } else {
        CHECK(d.degrees()[k] == b.degrees()[k]);
        // The directions left alone share their spaces.
        CHECK(same_space(d.space().spaces()[k], b.space().spaces()[k]));
      }
    }
    CHECK(derivative_error<T>(b, d, direction) <= gamma_4<T>() * max_control_point<T>(d));
  }
}

TEST_CASE_TEMPLATE("derivative: keeping the degree elevates the hodograph", T, double, float) {
  for (const Case& c : kCases) {
    if (c.dir.degree == 2 && c.dir.knots.size() == 9 && c.dir.knots[3] == 0.5 &&
        c.dir.knots[4] == 0.5) {
      continue;  // the C^-1 case: refused below
    }
    if (c.dir.knots.front() != 0.0 || c.dir.knots.back() != 1.0) {
      continue;  // unclamped: refused below
    }
    CAPTURE(c.dir.degree);
    const bs::Bspline<T> b =
        make_spline<T>({c.dir}, 2, false, 15U + static_cast<unsigned>(c.dir.knots.size()));
    const bs::Bspline<T> d = bs::derivative_keep_degree_nonrational<T>(b, 0);
    CHECK(d.degrees()[0] == static_cast<std::size_t>(c.dir.degree));
    CHECK(derivative_error<T>(b, d, 0) <=
          static_cast<double>(pantr::tolerance::get_conservative<T>()) *
              (std::max)(1.0, max_control_point<T>(d)));
  }
  // A C^0 interior knot (multiplicity = degree) is the ordinary case: the hodograph jumps there,
  // and the result carries multiplicity degree + 1.
  for (const int degree : {1, 2, 3, 4}) {
    CAPTURE(degree);
    const std::vector<double> knots =
        clamped(degree, std::vector<double>(static_cast<std::size_t>(degree), 0.5));
    const bs::Bspline<T> b = make_spline<T>({{knots, degree}}, 2, false, 3U);
    const bs::Bspline<T> d = bs::derivative_keep_degree_nonrational<T>(b, 0);
    long long at_half = 0;
    for (const T k : knots_of<T>(d, 0)) {
      at_half += (k == static_cast<T>(0.5)) ? 1 : 0;
    }
    CHECK(at_half == degree + 1);
    CHECK(derivative_error<T>(b, d, 0) <=
          static_cast<double>(pantr::tolerance::get_conservative<T>()) *
              (std::max)(1.0, max_control_point<T>(d)));
  }
}

TEST_CASE_TEMPLATE("derivative: keeping the degree of a periodic direction opens it", T, double,
                   float) {
  const std::vector<Dir> dirs = {{periodic_knots(4, 3), 3, true}, {clamped(2, {0.5}), 2}};
  const bs::Bspline<T> b = make_spline<T>(dirs, 1, false, 71U);
  const bs::Bspline<T> d = bs::derivative_keep_degree_nonrational<T>(b, 0);
  // Degree elevation does not support a periodic direction: the result is open there, and the
  // other direction keeps its space.
  CHECK_FALSE(d.space().spaces()[0].periodic());
  CHECK(d.space().spaces()[0].has_open_knots());
  CHECK(d.degrees()[0] == 3);
  CHECK(same_space(d.space().spaces()[1], b.space().spaces()[1]));
  CHECK(derivative_error<T>(b, d, 0) <=
        static_cast<double>(pantr::tolerance::get_conservative<T>()) *
            (std::max)(1.0, max_control_point<T>(d)));
}

TEST_CASE("derivative: refusals and their messages") {
  const std::vector<Dir> dirs = {{clamped(2, {0.5}), 2}, {clamped(0, {0.5}), 0}};
  const bs::Bspline<double> b = make_spline<double>(dirs, 1, false, 1U);
  const auto plain = [&](long long direction) {
    return message_of([&] { static_cast<void>(bs::derivative_nonrational<double>(b, direction)); });
  };
  CHECK(plain(2) == "direction must be in [0, 2), got 2.");
  CHECK(plain(-1) == "direction must be in [0, 2), got -1.");
  CHECK(plain(1) == "Derivative of a degree-0 B-spline is not defined.");

  // A rational B-spline is for the quotient rule, not for these.
  const std::vector<Dir> one = {{clamped(2, {0.5}), 2}};
  const bs::Bspline<double> rational = make_spline<double>(one, 1, true, 1U);
  CHECK(message_of([&] { static_cast<void>(bs::derivative_nonrational<double>(rational, 0)); }) ==
        "derivative_nonrational is for non-rational B-splines.");

  // The degree-keeping derivative of a degree past the binomial kernel's envelope.
  const std::vector<Dir> high = {{clamped(62, {}), 62}};
  const bs::Bspline<double> degree62 = make_spline<double>(high, 1, false, 1U);
  CHECK_NOTHROW(static_cast<void>(bs::derivative_nonrational<double>(degree62, 0)));
  CHECK(
      message_of([&] {
        static_cast<void>(bs::derivative_keep_degree_nonrational<double>(degree62, 0));
      }) ==
      "Degree-preserving derivative of a degree-62 B-spline needs binomial coefficients up to "
      "C(62, k), beyond the largest upper index 61 that pantr's exact-integer binomial kernel can "
      "compute without an int64 overflow. Past that the coefficients wrap silently and the result "
      "is corrupted rather than merely inaccurate.");

  // Re-elevating the hodograph needs it clamped; a C^-1 knot makes it over-multiple.
  const std::vector<Dir> unclamped = {{{0, 1, 2, 3, 4, 5, 6, 7, 8}, 2}};
  const bs::Bspline<double> u = make_spline<double>(unclamped, 1, false, 1U);
  CHECK_NOTHROW(static_cast<void>(bs::derivative_nonrational<double>(u, 0)));
  const std::string got =
      message_of([&] { static_cast<void>(bs::derivative_keep_degree_nonrational<double>(u, 0)); });
  CHECK(got.find("Degree-preserving derivative in direction 0 (its degree-1 hodograph) needs a "
                 "clamped knot vector") == 0);
  const std::vector<Dir> jump = {{clamped(2, {0.5, 0.5, 0.5}), 2}};
  const bs::Bspline<double> j = make_spline<double>(jump, 1, false, 1U);
  CHECK_NOTHROW(static_cast<void>(bs::derivative_nonrational<double>(j, 0)));
  CHECK(message_of(
            [&] { static_cast<void>(bs::derivative_keep_degree_nonrational<double>(j, 0)); }) ==
        "Degree-preserving derivative in direction 0 (its degree-1 hodograph) cannot walk a "
        "knot vector that repeats the knot 0.5 3 times at degree 1: a knot may repeat at most "
        "degree + 1 = 2 times.");
  // A direction of degree 1 has a degree-0 hodograph, which has no end runs to lack: any ends do.
  const std::vector<Dir> linear = {{{0, 1, 2, 3, 4}, 1}};
  const bs::Bspline<double> l = make_spline<double>(linear, 1, false, 1U);
  CHECK_NOTHROW(static_cast<void>(bs::derivative_keep_degree_nonrational<double>(l, 0)));
}

TEST_CASE("common refinement: the knots one lacks and the other has") {
  // Specification of the walk: for each unique knot of the other, the multiplicities it exceeds
  // by, and nothing for a knot within the tolerance.
  const std::vector<double> u_self = {0.0, 0.25, 1.0};
  const std::vector<std::int64_t> m_self = {3, 1, 3};
  const std::vector<double> u_other = {0.0, 0.25, 0.5, 0.75, 1.0};
  const std::vector<std::int64_t> m_other = {3, 2, 1, 2, 3};
  const std::vector<double> missing =
      bs::compute_missing_knots<double>(sp(u_self), std::span<const std::int64_t>(m_self),
                                        sp(u_other), std::span<const std::int64_t>(m_other), 1e-12);
  CHECK(missing == std::vector<double>{0.25, 0.5, 0.75, 0.75});
  // Within the tolerance a knot is the same knot.
  const std::vector<double> near_other = {0.0, 0.25 + 1e-14, 1.0};
  const std::vector<std::int64_t> m_near = {3, 1, 3};
  CHECK(bs::compute_missing_knots<double>(sp(u_self), std::span<const std::int64_t>(m_self),
                                          sp(near_other), std::span<const std::int64_t>(m_near),
                                          1e-12)
            .empty());
}

TEST_CASE_TEMPLATE("common refinement: both splines end on one knot vector, keeping their mappings",
                   T, double, float) {
  const std::vector<Dir> da = {{clamped(2, {0.25, 0.5}), 2}, {clamped(2, {0.5}), 2}};
  const std::vector<Dir> db = {{clamped(2, {0.5, 0.75}), 2}, {clamped(2, {0.5}), 2}};
  const bs::Bspline<T> a = make_spline<T>(da, 2, false, 3U);
  const bs::Bspline<T> b = make_spline<T>(db, 2, false, 4U);
  const auto [ra, rb] = bs::refine_to_common_space_1d<T>(a, b, 0);
  CHECK(knots_of<T>(ra, 0) == knots_of<T>(rb, 0));
  CHECK(knots_of<T>(ra, 0).size() == 9);  // 0 x3, 0.25, 0.5, 0.75 and 1 x3, each interior knot once
  CHECK(exact_difference<T>(ra, a) <= tier_tolerance(a));
  CHECK(exact_difference<T>(rb, b) <= tier_tolerance(b));
  // The other direction is untouched.
  CHECK(same_space(ra.space().spaces()[1], a.space().spaces()[1]));

  // Two splines already on one knot vector come back as they are, sharing their representation.
  const auto [sa, sb] = bs::refine_to_common_space_1d<T>(a, a, 0);
  CHECK(sa.same_rep(a));
  CHECK(sb.same_rep(a));
}
