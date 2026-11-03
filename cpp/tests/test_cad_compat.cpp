// `make_compat`: B-splines with one degree and one knot vector per axis.
//
// Oracles. The result must (1) have identical degrees and knot vectors along the chosen axes,
// which is checked against the knot vector built in the test from the inputs' interior knots (the
// union at the largest multiplicity, the multiplicity an elevation leaves written out by hand),
// and (2) represent the same geometry as each input. The geometry is compared with the exact
// rational evaluation of `bspline_exact_eval.hpp`, which sums the supported basis functions from
// their recursive definition in exact arithmetic, over a grid of parameters of the input's domain
// (so the only rounding in the difference is that of the stored control points). A domain remap is
// compared at the remapped parameter, `a_new + (u - a_old) * scale`, evaluated in the storage type.
// A degree elevation or a knot insertion is a change of basis with a few rounded steps; the
// difference is held to `tier_tolerance`, the conservative tier of `pantr::tolerance` times the
// control-point scale: a long accumulation, or a condition number up to about 1e3, which is what
// a chain of elevation and insertion is. Every message is matched by substring.
#include <doctest/doctest.h>

#include "bspline_test_support.hpp"
#include "cad_test_support.hpp"

#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/convert.hpp>
#include <pantr/cad/compat.hpp>
#include <pantr/core/errors.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace bs = pantr::bspline;
namespace cad = pantr::cad;
using bspline_test::clamped;
using bspline_test::Dir;
using bspline_test::exact_difference;
using bspline_test::make_spline;
using bspline_test::message_of;
using bspline_test::tier_tolerance;
using cad_test::evaluate_at;
using cad_test::knots_as_double;
using cad_test::linspace;

namespace {

/// A curve of the given degree and interior knots, with `rank` coordinates.
template <class T>
bs::Bspline<T> curve(int degree, const std::vector<double>& interior, std::size_t rank = 3,
                     unsigned seed = 1, bool rational = false) {
  return make_spline<T>({Dir{clamped(degree, interior), degree}}, rank, rational, seed);
}

template <class T>
std::vector<bs::Bspline<T>> compat(
    const std::vector<bs::Bspline<T>>& in,
    const std::optional<std::vector<long long>>& axes = std::nullopt) {
  return cad::make_compat<T>(std::span<const bs::Bspline<T>>(in), axes);
}

/// The knots a direction is expected to have, as the storage type rounds them, as doubles.
template <class T>
std::vector<double> as_stored(const std::vector<double>& knots) {
  std::vector<double> out;
  for (const double k : knots) {
    out.push_back(static_cast<double>(static_cast<T>(k)));
  }
  return out;
}

/// The result describes the same geometry as the input on the input's own domain (the domains are
/// equal, no remap): the exact difference of the stored coordinates is within the tier.
template <class T>
void check_same_geometry(const bs::Bspline<T>& in, const bs::Bspline<T>& out) {
  CHECK(exact_difference<T>(in, out) <= tier_tolerance(out));
}

}  // namespace

TEST_CASE_TEMPLATE("make_compat: fewer than two B-splines come back unchanged, unchecked", T, float,
                   double) {
  const bs::Bspline<T> c = curve<T>(2, {0.5});
  const std::vector<bs::Bspline<T>> one = {c};
  const std::vector<bs::Bspline<T>> got = compat(one);
  REQUIRE(got.size() == 1);
  CHECK(got[0].same_rep(c));
  // An out-of-range axis is not looked at: fewer than two is returned before any check.
  const std::vector<bs::Bspline<T>> got_axes = compat(one, std::vector<long long>{5});
  REQUIRE(got_axes.size() == 1);
  CHECK(got_axes[0].same_rep(c));
  CHECK(compat<T>({}).empty());
}

TEST_CASE_TEMPLATE("make_compat: different dimensions are refused with the set of dimensions", T,
                   float, double) {
  const bs::Bspline<T> crv = curve<T>(1, {});
  const bs::Bspline<T> srf =
      make_spline<T>({Dir{clamped(1, {}), 1}, Dir{clamped(1, {}), 1}}, 3, false, 2);
  const bs::Bspline<T> vol = make_spline<T>(
      {Dir{clamped(1, {}), 1}, Dir{clamped(1, {}), 1}, Dir{clamped(1, {}), 1}}, 3, false, 3);
  CHECK(message_of([&] {
          static_cast<void>(compat<T>({crv, srf}));
        }).find("All B-splines must have the same parametric dimension, got {1, 2}.") !=
        std::string::npos);
  CHECK(message_of([&] { static_cast<void>(compat<T>({vol, crv, srf})); }).find("got {1, 2, 3}.") !=
        std::string::npos);
  CHECK(message_of([&] { static_cast<void>(compat<T>({srf, vol})); }).find("got {2, 3}.") !=
        std::string::npos);
}

TEST_CASE_TEMPLATE("make_compat: an axis out of range is refused, an empty list changes nothing", T,
                   float, double) {
  const bs::Bspline<T> a = curve<T>(1, {});
  const bs::Bspline<T> b = curve<T>(2, {0.5});
  const auto message = [&](std::vector<long long> axes) {
    return message_of([&] { static_cast<void>(compat<T>({a, b}, axes)); });
  };
  CHECK(message({5}).find("Axis 5 out of range for dim=1.") != std::string::npos);
  CHECK(message({-1}).find("Axis -1 out of range for dim=1.") != std::string::npos);
  CHECK(message({0, 1}).find("Axis 1 out of range for dim=1.") != std::string::npos);

  const std::vector<bs::Bspline<T>> none = compat<T>({a, b}, std::vector<long long>{});
  REQUIRE(none.size() == 2);
  CHECK(none[0].same_rep(a));
  CHECK(none[1].same_rep(b));
}

TEST_CASE_TEMPLATE("make_compat: identical structure is returned as it was", T, float, double) {
  const bs::Bspline<T> a = curve<T>(2, {0.25, 0.5}, 3, 1);
  const bs::Bspline<T> b = curve<T>(2, {0.25, 0.5}, 3, 2);
  const std::vector<bs::Bspline<T>> got = compat<T>({a, b});
  REQUIRE(got.size() == 2);
  CHECK(got[0].same_rep(a));
  CHECK(got[1].same_rep(b));
}

TEST_CASE_TEMPLATE("make_compat: degrees are elevated to the largest, the geometry is kept", T,
                   float, double) {
  const bs::Bspline<T> lo = curve<T>(1, {});
  const bs::Bspline<T> hi = curve<T>(2, {}, 3, 2);
  const std::vector<bs::Bspline<T>> got = compat<T>({lo, hi});
  REQUIRE(got.size() == 2);
  CHECK(got[0].degrees() == std::vector<std::size_t>{2});
  CHECK(got[1].degrees() == std::vector<std::size_t>{2});
  CHECK(knots_as_double(got[0], 0) == knots_as_double(got[1], 0));
  check_same_geometry<T>(lo, got[0]);
  check_same_geometry<T>(hi, got[1]);
}

TEST_CASE_TEMPLATE("make_compat: knot vectors are the union at the largest multiplicity", T, float,
                   double) {
  SUBCASE("different interior knots") {
    const bs::Bspline<T> a = curve<T>(2, {0.5});
    const bs::Bspline<T> b = curve<T>(2, {0.3}, 3, 2);
    const std::vector<bs::Bspline<T>> got = compat<T>({a, b});
    const std::vector<double> expected = as_stored<T>(clamped(2, {0.3, 0.5}));
    CHECK(knots_as_double(got[0], 0) == expected);
    CHECK(knots_as_double(got[1], 0) == expected);
    check_same_geometry<T>(a, got[0]);
    check_same_geometry<T>(b, got[1]);
  }
  SUBCASE("the largest multiplicity wins") {
    const bs::Bspline<T> a = curve<T>(2, {0.5});
    const bs::Bspline<T> b = curve<T>(2, {0.5, 0.5}, 3, 2);
    const std::vector<bs::Bspline<T>> got = compat<T>({a, b});
    const std::vector<double> expected = as_stored<T>(clamped(2, {0.5, 0.5}));
    CHECK(knots_as_double(got[0], 0) == expected);
    CHECK(knots_as_double(got[1], 0) == expected);
    // The second was already at the target and is untouched.
    CHECK(got[1].same_rep(b));
    check_same_geometry<T>(a, got[0]);
  }
  SUBCASE("degree and knots together") {
    const bs::Bspline<T> a = curve<T>(1, {});
    const bs::Bspline<T> b = curve<T>(2, {0.5}, 3, 2);
    const std::vector<bs::Bspline<T>> got = compat<T>({a, b});
    CHECK(got[0].degrees() == got[1].degrees());
    CHECK(knots_as_double(got[0], 0) == knots_as_double(got[1], 0));
    check_same_geometry<T>(a, got[0]);
    check_same_geometry<T>(b, got[1]);
  }
}

TEST_CASE_TEMPLATE("make_compat: three curves of degrees 1, 2 and 3 meet at degree 3", T, float,
                   double) {
  const bs::Bspline<T> c1 = curve<T>(1, {}, 3, 1);
  const bs::Bspline<T> c2 = curve<T>(2, {0.5}, 3, 2);
  const bs::Bspline<T> c3 = curve<T>(3, {0.3, 0.7}, 3, 3);
  const std::vector<bs::Bspline<T>> got = compat<T>({c1, c2, c3});
  REQUIRE(got.size() == 3);
  // Elevating degree 2 to 3 raises the multiplicity of 0.5 from 1 to 2 (the continuity at the knot
  // is kept); the knots of the third have multiplicity 1: the union is 0.3, 0.5 twice, 0.7.
  const std::vector<double> expected = as_stored<T>(clamped(3, {0.3, 0.5, 0.5, 0.7}));
  for (const auto& g : got) {
    CHECK(g.degrees() == std::vector<std::size_t>{3});
    CHECK(knots_as_double(g, 0) == expected);
  }
  check_same_geometry<T>(c1, got[0]);
  check_same_geometry<T>(c2, got[1]);
  check_same_geometry<T>(c3, got[2]);
}

TEST_CASE_TEMPLATE(
    "make_compat: domains are remapped onto the envelope, the control points are kept", T, float,
    double) {
  // Domains [0, 1] and [2, 3] meet on [0, 3]; [-1, 1] and [0, 2] on [-1, 2].
  struct Row {
    double lo_a, hi_a, lo_b, hi_b;
  };
  const std::vector<Row> rows = {{0.0, 1.0, 2.0, 3.0}, {-1.0, 1.0, 0.0, 2.0}, {0.0, 4.0, 1.0, 2.0}};
  for (const Row& r : rows) {
    CAPTURE(r.lo_a);
    CAPTURE(r.lo_b);
    // Bezier pieces: no interior knot, so nothing is inserted and the control points are shared.
    const auto affine = [](double lo, double hi) { return std::vector<double>{lo, lo, hi, hi}; };
    const bs::Bspline<T> a = make_spline<T>({Dir{affine(r.lo_a, r.hi_a), 1}}, 3, false, 1);
    const bs::Bspline<T> b = make_spline<T>({Dir{affine(r.lo_b, r.hi_b), 1}}, 3, false, 2);
    const std::vector<bs::Bspline<T>> got = compat<T>({a, b});
    const double lo = (std::min)(r.lo_a, r.lo_b);
    const double hi = (std::max)(r.hi_a, r.hi_b);
    const std::array<const bs::Bspline<T>*, 2> in = {&a, &b};
    const std::array<std::pair<double, double>, 2> dom = {std::pair{r.lo_a, r.hi_a},
                                                          std::pair{r.lo_b, r.hi_b}};
    for (std::size_t i = 0; i < 2; ++i) {
      const auto d = got[i].space().spaces()[0].domain();
      CHECK(static_cast<double>(d.first) == lo);
      CHECK(static_cast<double>(d.second) == hi);
      CHECK(cad_test::control_points_of(got[i]) == cad_test::control_points_of(*in[i]));
      // The geometry at the remapped parameter is the input's at the original one.
      const double scale = (hi - lo) / (dom[i].second - dom[i].first);
      const std::vector<double> us = linspace(dom[i].first, dom[i].second, 9);
      std::vector<double> mapped;
      for (const double u : us) {
        mapped.push_back(lo + (u - dom[i].first) * scale);
      }
      const std::vector<double> before = evaluate_at(*in[i], us);
      const std::vector<double> after = evaluate_at(got[i], mapped);
      for (std::size_t k = 0; k < before.size(); ++k) {
        CHECK(std::abs(before[k] - after[k]) <= tier_tolerance(got[i]));
      }
    }
    // One knot vector for both, from the envelope's start to its end.
    CHECK(knots_as_double(got[0], 0) == knots_as_double(got[1], 0));
    CHECK(knots_as_double(got[0], 0).front() == lo);
    CHECK(knots_as_double(got[0], 0).back() == hi);
  }
}

TEST_CASE_TEMPLATE("make_compat: only the selected axes of a surface are made compatible", T, float,
                   double) {
  const bs::Bspline<T> s1 =
      make_spline<T>({Dir{clamped(1, {}), 1}, Dir{clamped(2, {}), 2}}, 3, false, 1);
  const bs::Bspline<T> s2 =
      make_spline<T>({Dir{clamped(2, {}), 2}, Dir{clamped(2, {0.4}), 2}}, 3, false, 2);
  SUBCASE("axis 0 only") {
    const auto got = compat<T>({s1, s2}, std::vector<long long>{0});
    CHECK(got[0].degrees() == std::vector<std::size_t>{2, 2});
    CHECK(got[1].degrees() == std::vector<std::size_t>{2, 2});
    CHECK(knots_as_double(got[0], 0) == knots_as_double(got[1], 0));
    // Axis 1 is left as it was.
    CHECK(knots_as_double(got[0], 1) == as_stored<T>(clamped(2, {})));
    CHECK(knots_as_double(got[1], 1) == as_stored<T>(clamped(2, {0.4})));
    check_same_geometry<T>(s1, got[0]);
    check_same_geometry<T>(s2, got[1]);
  }
  SUBCASE("axis 1 only") {
    const auto got = compat<T>({s1, s2}, std::vector<long long>{1});
    CHECK(got[0].degrees() == std::vector<std::size_t>{1, 2});
    CHECK(got[1].degrees() == std::vector<std::size_t>{2, 2});
    CHECK(knots_as_double(got[0], 1) == as_stored<T>(clamped(2, {0.4})));
    CHECK(knots_as_double(got[1], 1) == as_stored<T>(clamped(2, {0.4})));
    CHECK(got[1].same_rep(s2));
    check_same_geometry<T>(s1, got[0]);
  }
  SUBCASE("every axis, and a repeated axis is no different") {
    const auto all = compat<T>({s1, s2});
    const auto repeated = compat<T>({s1, s2}, std::vector<long long>{1, 0, 1, 0});
    for (std::size_t i = 0; i < 2; ++i) {
      CHECK(all[i].degrees() == std::vector<std::size_t>{2, 2});
      CHECK(knots_as_double(all[i], 0) == knots_as_double(all[0], 0));
      CHECK(knots_as_double(all[i], 1) == as_stored<T>(clamped(2, {0.4})));
      CHECK(cad_test::control_points_of(all[i]) == cad_test::control_points_of(repeated[i]));
    }
    check_same_geometry<T>(s1, all[0]);
    check_same_geometry<T>(s2, all[1]);
  }
}

TEST_CASE_TEMPLATE("make_compat: rational B-splines keep their weights through the stages", T,
                   float, double) {
  const bs::Bspline<T> a = curve<T>(1, {}, 2, 1, true);
  const bs::Bspline<T> b = curve<T>(2, {0.5}, 2, 2, true);
  const std::vector<bs::Bspline<T>> got = compat<T>({a, b});
  for (const auto& g : got) {
    CHECK(g.rational());
    CHECK(g.degrees() == std::vector<std::size_t>{2});
  }
  CHECK(knots_as_double(got[0], 0) == knots_as_double(got[1], 0));
  // The homogeneous coordinates, weight included, represent the same function.
  check_same_geometry<T>(a, got[0]);
  check_same_geometry<T>(b, got[1]);
}

TEST_CASE_TEMPLATE("make_compat: a periodic direction is opened and meets an open curve", T, float,
                   double) {
  const bs::Bspline<T> per =
      make_spline<T>({Dir{bspline_test::periodic_knots(4, 2), 2, true}}, 3, false, 1);
  const bs::Bspline<T> open = curve<T>(2, {0.5}, 3, 2);
  const std::vector<bs::Bspline<T>> got = compat<T>({per, open});
  const std::vector<double> expected = as_stored<T>(clamped(2, {0.25, 0.5, 0.75}));
  for (const auto& g : got) {
    CHECK_FALSE(g.space().spaces()[0].periodic());
    CHECK(knots_as_double(g, 0) == expected);
  }
  check_same_geometry<T>(per, got[0]);
  check_same_geometry<T>(open, got[1]);
}

TEST_CASE("make_compat: the mixed-precision overload keeps each input's storage type") {
  // A `float` degree-1 curve and a `double` degree-2 curve with an interior knot that `float`
  // cannot hold: both are widened, made compatible in `double`, and the first is cast back.
  const bs::Bspline<float> f = curve<float>(1, {}, 3, 1);
  const bs::Bspline<double> d = curve<double>(2, {0.3}, 3, 2);
  const std::vector<cad::AnyBspline> in = {f, d};
  const std::vector<cad::AnyBspline> got =
      cad::detail::make_compat(std::span<const cad::AnyBspline>(in), std::nullopt);
  REQUIRE(got.size() == 2);
  REQUIRE(std::holds_alternative<bs::Bspline<float>>(got[0]));
  REQUIRE(std::holds_alternative<bs::Bspline<double>>(got[1]));
  const auto& gf = std::get<bs::Bspline<float>>(got[0]);
  const auto& gd = std::get<bs::Bspline<double>>(got[1]);
  CHECK(gf.degrees() == std::vector<std::size_t>{2});
  CHECK(gd.degrees() == std::vector<std::size_t>{2});
  // The same knot vector up to the narrowing of the one stored in `float`.
  CHECK(knots_as_double(gf, 0) == as_stored<float>(knots_as_double(gd, 0)));
  CHECK(knots_as_double(gd, 0) == clamped(2, {0.3}));
  // The geometry, each in its own type and at its own tolerance tier.
  CHECK(exact_difference<float>(f, gf) <= tier_tolerance(gf));
  CHECK(exact_difference<double>(d, gd) <= tier_tolerance(gd));

  // The order of the types does not matter, and fewer than two are returned as they were.
  const std::vector<cad::AnyBspline> swapped = {d, f};
  const std::vector<cad::AnyBspline> got2 =
      cad::detail::make_compat(std::span<const cad::AnyBspline>(swapped), std::nullopt);
  CHECK(std::holds_alternative<bs::Bspline<double>>(got2[0]));
  CHECK(std::holds_alternative<bs::Bspline<float>>(got2[1]));
  const std::vector<cad::AnyBspline> single = {f};
  const std::vector<cad::AnyBspline> got3 =
      cad::detail::make_compat(std::span<const cad::AnyBspline>(single), std::vector<long long>{9});
  REQUIRE(got3.size() == 1);
  CHECK(std::get<bs::Bspline<float>>(got3[0]).same_rep(f));
  // The checks are those of `make_compat`.
  const bs::Bspline<float> surface =
      make_spline<float>({Dir{clamped(1, {}), 1}, Dir{clamped(1, {}), 1}}, 3, false, 3);
  const std::vector<cad::AnyBspline> mixed_dims = {f, surface};
  CHECK(message_of([&] {
          static_cast<void>(
              cad::detail::make_compat(std::span<const cad::AnyBspline>(mixed_dims), std::nullopt));
        }).find("got {1, 2}.") != std::string::npos);
}

TEST_CASE("remap_domain: a float end knot is the double remap rounded once to float") {
  // [0, 0.7f] onto [0.1, 0.9]. The remap is computed in double, `0.1 + 0.7f * (0.8 / 0.7f)`, which
  // is 0.9 to double precision, and rounded once: the end knot is `float(0.9)`. The Python
  // implementation's remap ran in float32 arithmetic and ended one float ulp above it, so the end
  // knot is a deliberate difference, not an oracle.
  const bs::Bspline<float> b = make_spline<float>({Dir{{0.0, 0.0, 0.7, 0.7}, 1}}, 3, false, 1);
  REQUIRE(knots_as_double(b, 0).back() == static_cast<double>(0.7f));
  const bs::Bspline<float> r = cad::detail::remap_domain<float>(b, 0, 0.1, 0.9);
  const std::vector<double> knots = knots_as_double(r, 0);
  CHECK(knots.front() == static_cast<double>(0.1f));
  CHECK(knots.back() == static_cast<double>(0.9f));
  CHECK(cad_test::control_points_of(r) == cad_test::control_points_of(b));
}

TEST_CASE("make_compat: a float knot and a double knot a float rounding apart get equal counts") {
  // The mixed-precision overload widens to double, so the float 0.3f and the double 0.3 are two
  // distinct knots of a double space, and each space gets the other's inserted: equal
  // numbers of knots. The Python implementation, which kept the float32 curve in float32,
  // returned unequal counts for the same pair.
  const bs::Bspline<float> f = curve<float>(2, {0.3}, 3, 1);
  const bs::Bspline<double> d = curve<double>(2, {0.3}, 3, 2);
  const std::vector<cad::AnyBspline> in = {f, d};
  const std::vector<cad::AnyBspline> got =
      cad::detail::make_compat(std::span<const cad::AnyBspline>(in), std::nullopt);
  REQUIRE(got.size() == 2);
  const auto& gf = std::get<bs::Bspline<float>>(got[0]);
  const auto& gd = std::get<bs::Bspline<double>>(got[1]);
  CHECK(knots_as_double(gf, 0).size() == knots_as_double(gd, 0).size());
}
