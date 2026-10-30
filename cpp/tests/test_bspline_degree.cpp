// Degree elevation and reduction of a B-spline.
//
// Oracles. Elevation is exact: the elevated spline is the same mapping, so its exact-rational
// Cox-de Boor evaluation (bspline_exact_eval.hpp, the recursive definition of the basis and no
// formula of the library) must agree with the original's at points that are not knots, and the
// knot vector must be the original's with every multiplicity grown by the increment. Reduction is
// exact on a spline that is truly of the lower degree, so reducing an elevated spline must give
// the original mapping and knot vector back; on a spline that is not, the properties the
// construction guarantees are checked (the end values, the multiplicities, the interpolation of
// the breakpoints in the Bezier form), not a value. Tolerances are the conservative tier of
// `pantr::tolerance` times the control-point scale (a long accumulation, or a condition number up
// to about 1e3): A5.9 ends with inverse knot insertions whose weights leave [0, 1], and the knot
// spans of the cases below keep that amplification far under a thousand.
#include <doctest/doctest.h>

#include "bspline_test_support.hpp"

#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/degree.hpp>
#include <pantr/core/binomial.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/tolerance.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
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
using bspline_test::max_difference;
using bspline_test::message_of;
using bspline_test::periodic_knots;
using bspline_test::same_space;
using bspline_test::sp;
using bspline_test::tier_tolerance;

namespace {

/// The multiplicity of each distinct knot value (the knots of a space are snapped, so equal
/// within the tolerance means equal).
template <class T>
std::map<double, long long> multiplicities(const std::vector<T>& knots) {
  std::map<double, long long> m;
  for (const T k : knots) {
    ++m[static_cast<double>(k)];
  }
  return m;
}

template <class T>
bs::Bspline<T> elevate(const bs::Bspline<T>& b, const std::vector<long long>& inc) {
  return bs::elevate_degree<T>(b, std::span<const long long>(inc));
}

template <class T>
bs::Bspline<T> reduce(const bs::Bspline<T>& b, const std::vector<long long>& dec) {
  return bs::reduce_degree<T>(b, std::span<const long long>(dec));
}

struct Case1D {
  Dir dir;
  long long increment;
};

// Clamped vectors of several degrees; interior knots of every multiplicity from 1 to the degree
// + 1 (a C^-1 jump); a degree-0 step function; and a single Bezier piece.
const std::vector<Case1D> kCases = {
    {{clamped(2, {0.25, 0.75}), 2}, 1},
    {{clamped(3, {0.25, 0.5, 0.75}), 3}, 2},
    {{clamped(1, {0.5}), 1}, 3},
    {{clamped(4, {0.5, 0.5}), 4}, 1},
    {{clamped(3, {0.25, 0.25, 0.25, 0.75}), 3}, 1},
    {{clamped(2, {0.5, 0.5, 0.5}), 2}, 2},
    {{clamped(0, {0.5}), 0}, 1},
    {{clamped(3, {0.125, 0.5, 0.5625}), 3}, 1},
    {{clamped(3, {}), 3}, 2},
};

}  // namespace

TEST_CASE_TEMPLATE("degree elevation: the elevated spline is the same mapping", T, double, float) {
  for (const bool rational : {false, true}) {
    for (const Case1D& c : kCases) {
      CAPTURE(c.dir.degree);
      CAPTURE(c.increment);
      CAPTURE(rational);
      const bs::Bspline<T> b =
          make_spline<T>({c.dir}, 2, rational, 3U + static_cast<unsigned>(c.dir.knots.size()));
      const bs::Bspline<T> e = elevate<T>(b, {c.increment});

      REQUIRE(e.degrees()[0] == static_cast<std::size_t>(c.dir.degree + c.increment));
      CHECK(e.rational() == rational);
      CHECK(e.cp_size() == b.cp_size());

      // Every multiplicity grows by the increment, and nothing else changes.
      const auto before = multiplicities<T>(knots_of<T>(b, 0));
      const auto after = multiplicities<T>(knots_of<T>(e, 0));
      REQUIRE(after.size() == before.size());
      for (const auto& [value, mult] : before) {
        CHECK(after.at(value) == mult + c.increment);
      }

      CHECK(exact_difference<T>(b, e) <= tier_tolerance(b));
    }
  }
}

TEST_CASE_TEMPLATE("degree elevation: surfaces and volumes, one direction at a time", T, double,
                   float) {
  const std::vector<Dir> dirs = {{clamped(2, {0.5}), 2}, {clamped(3, {0.25, 0.75}), 3}};
  for (const bool rational : {false, true}) {
    const bs::Bspline<T> b = make_spline<T>(dirs, 2, rational, 41U);
    for (const std::vector<long long>& inc :
         {std::vector<long long>{1, 0}, std::vector<long long>{0, 2},
          std::vector<long long>{1, 1}}) {
      CAPTURE(inc[0]);
      CAPTURE(inc[1]);
      const bs::Bspline<T> e = elevate<T>(b, inc);
      CHECK(e.degrees()[0] == static_cast<std::size_t>(2 + inc[0]));
      CHECK(e.degrees()[1] == static_cast<std::size_t>(3 + inc[1]));
      // A direction left alone keeps its space, sharing the representation.
      for (std::size_t d = 0; d < 2; ++d) {
        if (inc[d] == 0) {
          CHECK(same_space(e.space().spaces()[d], b.space().spaces()[d]));
        }
      }
      CHECK(exact_difference<T>(b, e) <= tier_tolerance(b));
    }
  }
  const std::vector<Dir> vol = {
      {clamped(1, {0.5}), 1}, {clamped(2, {0.5}), 2}, {clamped(2, {}), 2}};
  const bs::Bspline<T> v = make_spline<T>(vol, 1, false, 5U);
  const bs::Bspline<T> ev = elevate<T>(v, {1, 1, 1});
  CHECK(exact_difference<T>(v, ev) <= tier_tolerance(v));
}

TEST_CASE_TEMPLATE("degree elevation: a periodic direction is elevated through its open form", T,
                   double, float) {
  for (const int degree : {2, 3}) {
    for (const long long inc : {1LL, 2LL}) {
      CAPTURE(degree);
      CAPTURE(inc);
      const std::vector<Dir> dirs = {{periodic_knots(4, degree), degree, true}};
      const bs::Bspline<T> b = make_spline<T>(dirs, 2, false, 17U + static_cast<unsigned>(degree));
      const bs::Bspline<T> e = elevate<T>(b, {inc});
      const auto space = e.space().spaces()[0];
      CHECK(space.periodic());
      CHECK(space.degree() == degree + inc);
      // Every in-domain multiplicity grows by the increment (the seam's included).
      const auto before = b.space().spaces()[0].unique_knots_and_multiplicity(true).second;
      const auto after = space.unique_knots_and_multiplicity(true).second;
      REQUIRE(after.size() == before.size());
      for (std::size_t i = 0; i < before.size(); ++i) {
        CHECK(after[i] == before[i] + inc);
      }
      CHECK(exact_difference<T>(b, e) <= tier_tolerance(b));
    }
  }
  // A mixed surface: the periodic direction elevated, the open one left alone.
  const std::vector<Dir> mixed = {{periodic_knots(4, 2), 2, true}, {clamped(2, {0.5}), 2}};
  const bs::Bspline<T> b = make_spline<T>(mixed, 1, true, 23U);
  const bs::Bspline<T> e = elevate<T>(b, {1, 0});
  CHECK(e.space().spaces()[0].periodic());
  CHECK(same_space(e.space().spaces()[1], b.space().spaces()[1]));
  CHECK(exact_difference<T>(b, e) <= tier_tolerance(b));
}

TEST_CASE("degree elevation: the largest degree the binomial kernel serves, and one past it") {
  // A quadratic elevated to the envelope of the exact-integer binomial kernel (61) is the same
  // mapping; one past it is refused, in the direction it names, with the kernel's message.
  const std::vector<Dir> dirs = {{clamped(2, {}), 2}};
  const bs::Bspline<double> b = make_spline<double>(dirs, 1, false, 2U);
  const bs::Bspline<double> e = elevate<double>(b, {pantr::bincoeff_max_n - 2});
  REQUIRE(e.degrees()[0] == static_cast<std::size_t>(pantr::bincoeff_max_n));
  CHECK(exact_difference<double>(b, e) <= tier_tolerance(b));

  const auto message = [](long long inc) {
    const std::vector<Dir> d = {{clamped(2, {}), 2}};
    const bs::Bspline<double> s = make_spline<double>(d, 1, false, 2U);
    return message_of([&] { static_cast<void>(elevate<double>(s, {inc})); });
  };
  const std::string tail =
      " needs binomial coefficients up to C(62, k), beyond the largest upper index 61 that pantr's "
      "exact-integer binomial kernel can compute without an int64 overflow. Past that the "
      "coefficients wrap silently and the result is corrupted rather than merely inaccurate.";
  CHECK(message(pantr::bincoeff_max_n - 1) ==
        "Degree elevation to degree 62 in direction 0" + tail);
  // An increment past what a degree can hold is refused the same way, not wrapped.
  CHECK(message((std::numeric_limits<long long>::max)())
            .find("Degree elevation to degree "
                  "9223372036854775807 in direction 0 "
                  "needs binomial coefficients") == 0);

  // The direction is named: the second of two.
  const std::vector<Dir> two = {{clamped(2, {}), 2}, {clamped(2, {}), 2}};
  const bs::Bspline<double> s = make_spline<double>(two, 1, false, 2U);
  CHECK(message_of([&] {
          static_cast<void>(elevate<double>(s, {0, pantr::bincoeff_max_n - 1}));
        }).find("Degree elevation to degree 62 in direction 1 needs") == 0);
}

TEST_CASE("degree elevation: refusals and their messages") {
  const std::vector<Dir> dirs = {{clamped(2, {0.5}), 2}, {clamped(2, {0.5}), 2}};
  const bs::Bspline<double> b = make_spline<double>(dirs, 1, false, 1U);
  const auto msg = [&](const std::vector<long long>& inc) {
    return message_of([&] { static_cast<void>(elevate<double>(b, inc)); });
  };
  CHECK(msg({1}) == "Number of degree increments (1) must match dimension (2).");
  CHECK(msg({1, -1}) == "Degree increments must be non-negative.");
  CHECK(msg({0, 0}) == "At least one degree increment must be positive.");

  // An unclamped direction is refused, naming the direction; so is an unclamped end at either side.
  const std::vector<Dir> open_first = {{clamped(2, {0.5}), 2}, {{0, 1, 2, 3, 4, 5, 6, 7}, 2}};
  const bs::Bspline<double> unclamped = make_spline<double>(open_first, 1, false, 1U);
  CHECK(message_of([&] { static_cast<void>(elevate<double>(unclamped, {1, 0})); }) ==
        "<no exception>");
  const std::string got =
      message_of([&] { static_cast<void>(elevate<double>(unclamped, {0, 1})); });
  CHECK(got.find("Degree elevation in direction 1 needs a clamped knot vector: A5.9 takes the "
                 "first 3 coefficients as the first Bézier segment") == 0);
  CHECK(got.find("this vector does not open or close with one.") != std::string::npos);
  const std::vector<Dir> half = {{{0, 0, 0, 0.5, 1, 2, 3}, 2}};
  const bs::Bspline<double> half_open = make_spline<double>(half, 1, false, 1U);
  CHECK(message_of([&] {
          static_cast<void>(elevate<double>(half_open, {1}));
        }).find("this vector does not close with one.") != std::string::npos);

  // A space accepts a knot that repeats more than degree + 1 times; the kernel cannot walk it.
  const std::vector<Dir> over = {{{0, 0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1, 1}, 2}};
  const bs::Bspline<double> too_many = make_spline<double>(over, 1, false, 1U);
  CHECK(message_of([&] { static_cast<void>(elevate<double>(too_many, {1})); }) ==
        "Degree elevation in direction 0 cannot walk a knot vector that repeats the knot 0.5 4 "
        "times at degree 2: a knot may repeat at most degree + 1 = 3 times.");
}

TEST_CASE_TEMPLATE("degree reduction: an elevated spline reduces back to the original", T, double,
                   float) {
  for (const bool rational : {false, true}) {
    for (const Case1D& c : kCases) {
      if (c.dir.degree == 0) {
        continue;
      }
      CAPTURE(c.dir.degree);
      CAPTURE(c.increment);
      CAPTURE(rational);
      const bs::Bspline<T> b =
          make_spline<T>({c.dir}, 2, rational, 7U + static_cast<unsigned>(c.dir.knots.size()));
      const bs::Bspline<T> back = reduce<T>(elevate<T>(b, {c.increment}), {c.increment});
      CHECK(knots_of<T>(back, 0) == knots_of<T>(b, 0));
      CHECK(back.degrees()[0] == b.degrees()[0]);
      REQUIRE(back.control_points().size() == b.control_points().size());
      CHECK(exact_difference<T>(back, b) <= tier_tolerance(b));
    }
  }
}

TEST_CASE_TEMPLATE("degree reduction: periodic and multi-dimensional round trips", T, double,
                   float) {
  for (const int degree : {2, 3}) {
    CAPTURE(degree);
    const std::vector<Dir> dirs = {{periodic_knots(4, degree), degree, true}};
    const bs::Bspline<T> b = make_spline<T>(dirs, 1, false, 29U + static_cast<unsigned>(degree));
    const bs::Bspline<T> back = reduce<T>(elevate<T>(b, {1}), {1});
    CHECK(back.space().spaces()[0].periodic());
    CHECK(back.degrees()[0] == b.degrees()[0]);
    CHECK(exact_difference<T>(back, b) <= tier_tolerance(b));
  }
  const std::vector<Dir> surf = {{clamped(3, {0.25, 0.75}), 3}, {clamped(2, {0.5}), 2}};
  const bs::Bspline<T> s = make_spline<T>(surf, 2, true, 13U);
  const bs::Bspline<T> back = reduce<T>(elevate<T>(s, {1, 1}), {1, 1});
  CHECK(knots_of<T>(back, 0) == knots_of<T>(s, 0));
  CHECK(knots_of<T>(back, 1) == knots_of<T>(s, 1));
  CHECK(exact_difference<T>(back, s) <= tier_tolerance(s));
  // One direction only; the other keeps its space.
  const bs::Bspline<T> one = reduce<T>(s, {0, 1});
  CHECK(one.degrees()[0] == 3);
  CHECK(one.degrees()[1] == 1);
  CHECK(same_space(one.space().spaces()[0], s.space().spaces()[0]));
}

TEST_CASE_TEMPLATE("degree reduction: what the construction guarantees of an inexact reduction", T,
                   double, float) {
  // A cubic that is not a quadratic, with a smooth, a C^1 and a C^-1 breakpoint (multiplicities 1,
  // 2 and 4). The reduced spline is an approximation, so no value is asserted beyond what the
  // construction fixes: the degree, the multiplicities (every breakpoint keeps its continuity,
  // `max(1, m - t)`), and the end values, which the operator's unit rows reproduce.
  const std::vector<double> knots = clamped(3, {0.25, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75});
  const bs::Bspline<T> b = make_spline<T>({{knots, 3}}, 2, false, 99U);
  const bs::Bspline<T> r = reduce<T>(b, {1});
  CHECK(r.degrees()[0] == 2);

  const auto before = multiplicities<T>(knots_of<T>(b, 0));
  const auto after = multiplicities<T>(knots_of<T>(r, 0));
  REQUIRE(after.size() == before.size());
  for (const auto& [value, mult] : before) {
    // The ends keep degree + 1; an interior knot of multiplicity m keeps `max(1, m - 1)`, and a
    // jump (m = degree + 1 = 4) keeps `new_degree + 1 = 3`: the same continuity `C^{p - m}`.
    const long long expected = (value == 0.0 || value == 1.0) ? 3 : (std::max)(1LL, mult - 1);
    CHECK(after.at(value) == expected);
  }

  // The end values of the mapping are unchanged: the first and last control points are.
  const std::size_t cols = b.cp_size();
  const auto cb = b.control_points();
  const auto cr = r.control_points();
  for (std::size_t c = 0; c < cols; ++c) {
    CHECK(cr[c] == cb[c]);
    CHECK(cr[cr.size() - cols + c] == cb[cb.size() - cols + c]);
  }
  // The result has as many control points as its space has functions.
  CHECK(r.control_points().size() ==
        static_cast<std::size_t>(r.space().spaces()[0].num_basis()) * cols);
}

TEST_CASE_TEMPLATE("degree reduction: the Bezier form meets the original at every breakpoint", T,
                   double, float) {
  // Before the knot coarsening, the reduced spline interpolates the original at each breakpoint:
  // the operator's unit rows reproduce the end values of every Bezier segment, so the junction
  // is written once and the stitch is exactly C0. The oracle is the exact evaluation of both.
  for (const int degree : {3, 4, 5}) {
    CAPTURE(degree);
    const std::vector<double> knots = clamped(degree, {0.25, 0.5, 0.75});
    const bs::Bspline<T> b =
        make_spline<T>({{knots, degree}}, 2, false, 61U + static_cast<unsigned>(degree));
    const std::size_t rows = b.shape()[0];
    const pantr::mdview<const T, 2> ctrl(b.control_points().data(), rows, b.cp_size());
    const bs::KnotsAndCtrl<T> red =
        bs::reduce_degree_1d<T>(degree, ctrl, b.space().spaces()[0].knots(), 1);

    // Seen as a spline of the lower degree, over the knots it came with.
    const std::vector<Dir> lower = {{[&] {
                                       std::vector<double> k(red.knots.begin(), red.knots.end());
                                       return k;
                                     }(),
                                     degree - 1}};
    const bs::BsplineSpace<T> space = bspline_test::space_of<T>(lower);
    const bs::Bspline<T> bezier_form(space, std::span<const T>(red.ctrl.data(), red.ctrl.size()),
                                     false);
    for (const double x : {0.25, 0.5, 0.75}) {
      const std::vector<oracle::Rational> pt = {oracle::exact(x)};
      const auto va = oracle::exact_value<T>(b, pt);
      const auto vb = oracle::exact_value<T>(bezier_form, pt);
      for (std::size_t c = 0; c < va.size(); ++c) {
        oracle::Rational d = va[c] - vb[c];
        if (d < 0) {
          d = -d;
        }
        CHECK(static_cast<double>(d) <= tier_tolerance(b));
      }
    }
  }
}

TEST_CASE("degree reduction: a forced removal restores the continuity whatever the error") {
  // The coarsening has no deviation criterion: the continuity of the original is structural. A
  // cubic whose single interior knot the reduction cannot honour is coarsened all the same.
  const std::vector<double> knots = {0, 0, 0, 0, 0.5, 1, 1, 1, 1};
  const std::vector<double> cp = {0, 1, -1, 1, 0};
  const std::vector<double> flat = cp;
  const bs::Bspline<double> b(bspline_test::space_of<double>({{knots, 3}}), sp(flat), false);
  const bs::Bspline<double> r = reduce<double>(b, {1});
  CHECK(r.degrees()[0] == 2);
  CHECK(multiplicities<double>(knots_of<double>(r, 0)).at(0.5) == 1);
}

TEST_CASE("degree reduction: a reduction to degree 0, and the weight of a rational result") {
  const std::vector<Dir> dirs = {{clamped(1, {0.5}), 1}};
  const bs::Bspline<double> b = make_spline<double>(dirs, 1, false, 4U);
  const bs::Bspline<double> r = reduce<double>(b, {1});
  CHECK(r.degrees()[0] == 0);
  CHECK(r.control_points().size() ==
        static_cast<std::size_t>(r.space().spaces()[0].num_basis()) * b.cp_size());

  // The cubic net whose reduction inverts a weight: the operator's most negative entry (-0.25)
  // sits in the column of the first control point, so a large weight there drives the reduced
  // weight negative (to -1.75).
  const std::vector<double> w = {12.0, 1.0, 1.0, 1.0};
  const std::vector<double> pts = {0.0, 0.0, 1.0, 2.0, 2.0, -1.0, 3.0, 0.0};
  std::vector<double> homogeneous;
  for (std::size_t i = 0; i < 4; ++i) {
    homogeneous.push_back(pts[2 * i] * w[i]);
    homogeneous.push_back(pts[2 * i + 1] * w[i]);
    homogeneous.push_back(w[i]);
  }
  const bs::Bspline<double> inverted(bspline_test::space_of<double>({{clamped(3, {}), 3}}),
                                     sp(homogeneous), true);
  const std::string msg = message_of([&] { static_cast<void>(reduce<double>(inverted, {1})); });
  CHECK(msg.find("Degree reduction drove a control weight to -1.750e+00, which is not strictly "
                 "positive, so the reduced rational spline is not a valid NURBS") == 0);
  CHECK(msg.find("Bspline has no minimize_degree; convert to Bezier form with to_bezier or "
                 "to_beziers") != std::string::npos);
}

TEST_CASE("degree reduction: refusals and their messages") {
  const std::vector<Dir> dirs = {{clamped(2, {0.5}), 2}, {clamped(3, {0.5}), 3}};
  const bs::Bspline<double> b = make_spline<double>(dirs, 1, false, 1U);
  const auto msg = [&](const std::vector<long long>& dec) {
    return message_of([&] { static_cast<void>(reduce<double>(b, dec)); });
  };
  CHECK(msg({1}) == "Number of degree decrements (1) must match dimension (2).");
  CHECK(msg({1, -1}) == "Degree decrements must be non-negative.");
  CHECK(msg({0, 0}) == "At least one degree decrement must be positive.");
  CHECK(msg({0, 4}) == "Degree decrement (4) in direction 1 exceeds current degree (3).");

  // An unclamped direction is refused before the kernel walks it.
  const std::vector<Dir> open = {{{0, 1, 2, 3, 4, 5, 6, 7}, 2}};
  const bs::Bspline<double> unclamped = make_spline<double>(open, 1, false, 1U);
  const std::string got = message_of([&] { static_cast<void>(reduce<double>(unclamped, {1})); });
  CHECK(got.find("Degree reduction in direction 0 needs a clamped knot vector: the decomposition "
                 "into Bézier segments takes the first 3 coefficients") == 0);
  CHECK(got.find("Reducing an unclamped direction is not supported;") != std::string::npos);
}

TEST_CASE("degree kernels: the clamped-ends test is by bit identity") {
  const std::vector<double> k = {0, 0, 0, 0.5, 1, 1, 1};
  const auto ends = bs::clamped_ends<double>(sp(k), 2);
  CHECK(ends.first);
  CHECK(ends.second);
  const std::vector<double> nearly = {0, 0, 1e-300, 0.5, 1, 1, 1};
  CHECK_FALSE(bs::clamped_ends<double>(sp(nearly), 2).first);
  // At degree 0 both runs are one knot long: any vector passes.
  const std::vector<double> any = {0.0, 0.25, 1.0};
  CHECK(bs::clamped_ends<double>(sp(any), 0).first);
  CHECK(bs::clamped_ends<double>(sp(any), 0).second);
  CHECK_NOTHROW(bs::check_clamped_knots<double>(sp(any), 0, "x"));
}

namespace {

/// A random clamped knot vector: degree 0 to 5, up to four distinct interior breakpoints on the
/// grid of sixteenths, each of a multiplicity from 1 to `degree + 1`; and an increment from 1 to 3.
/// The generator is a fixed xorshift, so a failure repeats.
struct RandomCase {
  std::vector<double> knots;
  int degree;
  long long increment;
};

RandomCase random_case(std::uint32_t& state) {
  const auto next = [&state](std::uint32_t bound) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state % bound;
  };
  RandomCase c;
  c.degree = static_cast<int>(next(6));
  c.increment = 1 + static_cast<long long>(next(3));
  std::vector<double> interior;
  std::uint32_t grid = 0;
  const std::uint32_t breaks = next(5);
  for (std::uint32_t b = 0; b < breaks; ++b) {
    grid += 1 + next(3);
    if (grid >= 16) {
      break;
    }
    const std::uint32_t mult = 1 + next(static_cast<std::uint32_t>(c.degree) + 1);
    interior.insert(interior.end(), mult, static_cast<double>(grid) / 16.0);
  }
  c.knots = clamped(c.degree, interior);
  return c;
}

}  // namespace

TEST_CASE_TEMPLATE("degree elevation and reduction: random knot vectors, every multiplicity", T,
                   double, float) {
  // The kernels' index arithmetic is intricate (the amended loop bound, the first coefficient a
  // segment contributes, the inverse insertions), so a sweep over random clamped vectors with
  // interior multiplicities from 1 to degree + 1 is the test the sanitizer builds are for: an
  // out-of-range read or write fails there. Each case also keeps the mapping, in exact arithmetic.
  std::uint32_t state = 0x9E3779B9U;
  for (int n = 0; n < 150; ++n) {
    const RandomCase c = random_case(state);
    CAPTURE(n);
    CAPTURE(c.degree);
    CAPTURE(c.increment);
    const bs::Bspline<T> b =
        make_spline<T>({{c.knots, c.degree}}, 2, (n % 3) == 0, static_cast<unsigned>(n));
    const bs::Bspline<T> e = elevate<T>(b, {c.increment});
    REQUIRE(e.control_points().size() ==
            static_cast<std::size_t>(e.space().spaces()[0].num_basis()) * e.cp_size());
    CHECK(exact_difference<T>(b, e) <= tier_tolerance(b));
    if (c.degree >= 1) {
      const bs::Bspline<T> back = reduce<T>(e, {c.increment});
      CHECK(knots_of<T>(back, 0) == knots_of<T>(b, 0));
      CHECK(exact_difference<T>(back, b) <= tier_tolerance(b));
      // And a reduction of the original, which is only an approximation, must still be a
      // well-formed spline: as many control points as functions. (A rational reduction may be
      // refused for a weight it drives through zero, which the weight test below pins.)
      if (!b.rational()) {
        const bs::Bspline<T> r = reduce<T>(b, {1});
        CHECK(r.control_points().size() ==
              static_cast<std::size_t>(r.space().spaces()[0].num_basis()) * r.cp_size());
      }
    }
  }
}
