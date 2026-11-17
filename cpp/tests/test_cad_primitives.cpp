// The CAD primitives: `create_line`, `create_circle`, `create_bilinear`, `create_trilinear`.
//
// Oracles. The circle is checked against the geometry it represents, not against its own control
// points: the curve is evaluated at many parameters and each point must lie at the radius from the
// centre in the xy-plane; the points at the span junctions (`u = k / spans`, where the curve
// interpolates a control point) and at the two ends must lie at the angles the sweep names,
// `centre + r (cos a, sin a)` with the angle `a` formed from the sweep in the test. The bilinear
// and trilinear primitives reproduce their corners and are compared at interior parameters with
// the closed-form multilinear blend `sum_corner corner * prod_d (u_d or 1 - u_d)`. The span count
// is checked on the table of sweeps with the quadrant bounds. Messages are matched by substring.
//
// Tolerances. The primitives are `double` only. A rational quadratic evaluated at a parameter, its
// control points built from `sin`, `cos`, `tan` and a rotation, and compared by the test with a
// closed form, costs a few tens of roundings relative to `|r| + |centre|`: about 10 for the control
// points (three libm calls within an ulp each, `x`, `y`, `xm`, `wm * xm`, the rotation matrix and
// the three multiply-adds of the rotation), 2 for the centre, about 25 for the evaluation (three
// basis values of degree 2 at 4 eps each, the weighted sums, one division by a weight no smaller
// than cos(pi/4), which inflates by 1.4), 2 for the test's own `hypot` and subtraction, and for
// the angles at the junctions about 14 for the angle errors (`alpha + start`, one rounding per
// repeated rotation, and the test's `start + k delta / spans`, each of an angle of order 1 to 2
// pi). The tolerance is 128 eps (|r| + |centre|), twice the sum, which leaves room for another libm
// or contraction. The multilinear primitives cost, per corner term, `d` basis values of about 3 eps
// (the parameter map and `1 - t`, then the product) and `2^d - 1` additions, for the code and again
// for the closed form: `2 (3 d + 2^d) eps max|corner|`, doubled for build slack.
#include <doctest/doctest.h>

#include "bspline_test_support.hpp"
#include "cad_test_support.hpp"

#include <pantr/bspline/bspline.hpp>
#include <pantr/cad/primitives.hpp>
#include <pantr/core/errors.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace bs = pantr::bspline;
namespace cad = pantr::cad;
using bspline_test::message_of;
using cad_test::control_points_of;
using cad_test::eps;
using cad_test::evaluate_at;
using cad_test::knots_as_double;
using cad_test::linspace;

namespace {

constexpr double pi = std::numbers::pi;

std::span<const double> sp(const std::vector<double>& v) { return std::span<const double>(v); }

std::span<const std::size_t> sh(const std::vector<std::size_t>& v) {
  return std::span<const std::size_t>(v);
}

}  // namespace

// ----------------------------------------------------------------------------------------- line

TEST_CASE("create_line: two control points, the knots [0, 0, 1, 1], padded to three coordinates") {
  const std::vector<double> p0 = {1.0, 2.0, 3.0};
  const std::vector<double> p1 = {4.0, 5.0, 6.0};
  const bs::Bspline<double> c = cad::create_line(sp(p0), sp(p1));
  CHECK(c.dim() == 1);
  CHECK(c.degrees() == std::vector<std::size_t>{1});
  CHECK(c.rank() == 3);
  CHECK(c.cp_size() == 3);
  CHECK_FALSE(c.rational());
  CHECK(c.num_basis(0) == 2);
  CHECK(knots_as_double(c, 0) == std::vector<double>{0.0, 0.0, 1.0, 1.0});
  CHECK(control_points_of(c) == std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

  // Two coordinates, one coordinate and none are padded with zeros.
  const std::vector<double> a2 = {0.0, 0.0};
  const std::vector<double> b2 = {1.0, 1.0};
  CHECK(control_points_of(cad::create_line(sp(a2), sp(b2))) ==
        std::vector<double>{0.0, 0.0, 0.0, 1.0, 1.0, 0.0});
  const std::vector<double> a1 = {0.0};
  const std::vector<double> b1 = {5.0};
  CHECK(control_points_of(cad::create_line(sp(a1), sp(b1))) ==
        std::vector<double>{0.0, 0.0, 0.0, 5.0, 0.0, 0.0});
  const std::vector<double> none;
  CHECK(control_points_of(cad::create_line(sp(none), sp(none))) == std::vector<double>(6, 0.0));
}

TEST_CASE("create_line: the point at a parameter is the convex combination of the ends") {
  // `(1 - t) p0 + t p1` on the Bezier-like space: the parameter map, `1 - t`, two products and a
  // sum, 4 eps max|p| with a rounding of margin.
  const std::vector<double> p0 = {-1.0, 2.5, 3.0};
  const std::vector<double> p1 = {4.0, -5.0, 0.5};
  const bs::Bspline<double> c = cad::create_line(sp(p0), sp(p1));
  const std::vector<double> ts = {0.0, 0.1, 0.3, 0.5, 0.9, 1.0};
  const std::vector<double> got = evaluate_at(c, ts);
  const double tol = 4.0 * eps<double>() * 5.0;
  for (std::size_t i = 0; i < ts.size(); ++i) {
    for (std::size_t k = 0; k < 3; ++k) {
      const double expected = (1.0 - ts[i]) * p0[k] + ts[i] * p1[k];
      CHECK(std::abs(got[3 * i + k] - expected) <= tol);
    }
  }
}

TEST_CASE(
    "create_line: a point of more than three coordinates is refused, the first checked first") {
  const std::vector<double> four = {1.0, 2.0, 3.0, 4.0};
  const std::vector<double> five = {1.0, 2.0, 3.0, 4.0, 5.0};
  const std::vector<double> ok = {0.0, 0.0, 0.0};
  CHECK(message_of([&] {
          static_cast<void>(cad::create_line(sp(four), sp(ok)));
        }).find("Point must have at most 3 coordinates, got 4.") != std::string::npos);
  CHECK(message_of([&] {
          static_cast<void>(cad::create_line(sp(ok), sp(five)));
        }).find("Point must have at most 3 coordinates, got 5.") != std::string::npos);
  CHECK(message_of([&] {
          static_cast<void>(cad::create_line(sp(five), sp(four)));
        }).find("got 5.") != std::string::npos);
}

// --------------------------------------------------------------------------------------- circle

namespace {

struct ArcCase {
  double radius;
  std::vector<double> center;
  std::optional<std::array<double, 2>> sweep;
};

/// The derived tolerance of the header note, 128 eps (|r| + |centre|).
double circle_tolerance(const ArcCase& c) {
  double scale = std::abs(c.radius);
  double centre = 0.0;
  for (const double x : c.center) {
    centre = (std::max)(centre, std::abs(x));
  }
  return 128.0 * eps<double>() * (scale + centre);
}

std::array<double, 3> padded(const std::vector<double>& c) {
  std::array<double, 3> out = {0.0, 0.0, 0.0};
  std::copy(c.begin(), c.end(), out.begin());
  return out;
}

bs::Bspline<double> make_arc(const ArcCase& c) {
  if (c.center.empty()) {
    return cad::create_circle(c.radius, std::nullopt, c.sweep);
  }
  return cad::create_circle(c.radius, std::span<const double>(c.center), c.sweep);
}

}  // namespace

TEST_CASE(
    "create_circle: the points of every arc lie at the radius from the centre, in the plane") {
  const std::vector<ArcCase> cases = {
      {1.0, {}, std::nullopt},
      {3.5, {}, std::nullopt},
      {2.0, {1.0, -2.0, 0.5}, std::nullopt},
      {2.0, {}, std::array<double, 2>{0.0, pi}},
      {1.5, {2.0}, std::array<double, 2>{0.0, pi / 2.0}},
      {1.0, {}, std::array<double, 2>{0.0, 0.3}},
      {4.0, {-1.0, 3.0}, std::array<double, 2>{pi / 3.0, 3.0 * pi / 2.0}},
      {2.0, {}, std::array<double, 2>{pi / 2.0, -pi / 2.0}},
      {1.0, {}, std::array<double, 2>{0.0, -pi / 2.0}},
      {0.25, {10.0, 10.0, 10.0}, std::array<double, 2>{-1.0, 4.0}},
      {1.0, {}, std::array<double, 2>{0.0, 2.0 * pi}},
      {0.0, {1.0, 1.0}, std::nullopt},
      {-2.0, {}, std::array<double, 2>{0.0, 1.0}},
  };
  const std::vector<double> us = linspace(0.0, 1.0, 61);
  for (const ArcCase& c : cases) {
    const bs::Bspline<double> crv = make_arc(c);
    CAPTURE(c.radius);
    CAPTURE(c.center.size());
    REQUIRE(crv.dim() == 1);
    CHECK(crv.rational());
    CHECK(crv.rank() == 3);
    CHECK(crv.degrees() == std::vector<std::size_t>{2});
    const std::array<double, 3> ctr = padded(c.center);
    const double tol = circle_tolerance(c);
    const std::vector<double> pts = evaluate_at(crv, us);
    for (std::size_t i = 0; i < us.size(); ++i) {
      const double radial = std::hypot(pts[3 * i] - ctr[0], pts[3 * i + 1] - ctr[1]);
      CHECK(std::abs(radial - std::abs(c.radius)) <= tol);
      CHECK(std::abs(pts[3 * i + 2] - ctr[2]) <= tol);
    }
  }
}

TEST_CASE("create_circle: the span junctions and the ends are at the angles of the sweep") {
  struct Angles {
    double radius;
    std::vector<double> center;
    double start;
    double end;
  };
  const std::vector<Angles> cases = {
      {2.0, {}, 0.0, pi},
      {2.0, {1.0, 2.0}, 0.0, 2.0 * pi - 0.1},
      {3.0, {-1.0}, pi / 3.0, 3.0 * pi / 2.0},
      {1.0, {}, pi / 2.0, -pi / 2.0},
      {1.0, {}, 0.5, 0.5 + 1.3},
      {5.0, {0.0, 0.0, 2.0}, -1.0, 4.0},
  };
  for (const Angles& c : cases) {
    const ArcCase arc = {c.radius, c.center, std::array<double, 2>{c.start, c.end}};
    const bs::Bspline<double> crv = make_arc(arc);
    const std::size_t spans = (crv.num_basis(0) - 1) / 2;
    CAPTURE(c.start);
    CAPTURE(c.end);
    REQUIRE(spans >= 1);
    REQUIRE(spans <= 4);
    const std::array<double, 3> ctr = padded(c.center);
    const double tol = circle_tolerance(arc);
    const double delta = c.end - c.start;
    std::vector<double> us;
    for (std::size_t k = 0; k <= spans; ++k) {
      us.push_back(static_cast<double>(k) / static_cast<double>(spans));
    }
    const std::vector<double> pts = evaluate_at(crv, us);
    for (std::size_t k = 0; k <= spans; ++k) {
      const double angle = c.start + delta * static_cast<double>(k) / static_cast<double>(spans);
      CHECK(std::abs(pts[3 * k] - (ctr[0] + c.radius * std::cos(angle))) <= tol);
      CHECK(std::abs(pts[3 * k + 1] - (ctr[1] + c.radius * std::sin(angle))) <= tol);
      CHECK(std::abs(pts[3 * k + 2] - ctr[2]) <= tol);
    }
  }
}

TEST_CASE("create_circle: the full circle has nine control points and passes the quadrant points") {
  const double r = 3.0;
  const bs::Bspline<double> crv = cad::create_circle(r);
  CHECK(crv.num_basis(0) == 9);
  CHECK(knots_as_double(crv, 0) ==
        std::vector<double>{0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1});
  // The standard conic: corner points of the square of half-side r at weight 1, the points between
  // them at the weight cos(pi / 4) with the weighted coordinates (w r, w r).
  const std::vector<double> cp = control_points_of(crv);
  const double wm = std::cos(pi / 4.0);
  for (std::size_t i = 0; i < 9; ++i) {
    const double w = cp[4 * i + 3];
    CHECK(std::abs(w - (i % 2 == 0 ? 1.0 : wm)) <= 2.0 * eps<double>());
    CHECK(std::abs(cp[4 * i + 2]) == 0.0);
  }
  const std::vector<double> us = {0.0, 0.25, 0.5, 0.75, 1.0};
  const std::vector<double> pts = evaluate_at(crv, us);
  const std::vector<double> expected = {r, 0, 0, r, -r, 0, 0, -r, r, 0};
  const double tol = 128.0 * eps<double>() * r;
  for (std::size_t k = 0; k < 5; ++k) {
    CHECK(std::abs(pts[3 * k] - expected[2 * k]) <= tol);
    CHECK(std::abs(pts[3 * k + 1] - expected[2 * k + 1]) <= tol);
    CHECK(std::abs(pts[3 * k + 2]) <= tol);
  }
}

TEST_CASE("create_circle: the number of spans counts the quadrant bounds below the sweep") {
  // spans = number of the bounds 0, pi/2, pi, 3 pi/2 strictly below |sweep|, at least 1: a sweep
  // equal to a bound is in the span below it. 3 pi / 2 is formed as the implementation forms it so
  // that the bound is hit exactly.
  struct Row {
    double sweep;
    std::size_t spans;
  };
  const double three_halves = 3.0 * pi / 2.0;
  const std::vector<Row> rows = {
      {0.0, 1},
      {1e-9, 1},
      {pi / 2.0, 1},
      {pi / 2.0 * (1.0 + 1e-12), 2},
      {pi, 2},
      {pi * (1.0 + 1e-12), 3},
      {three_halves, 3},
      {three_halves * (1.0 + 1e-12), 4},
      {2.0 * pi, 4},
      {10.0, 4},
      {-pi / 2.0, 1},
      {-pi, 2},
      {-0.9 * three_halves, 3},
      {-10.0, 4},
  };
  for (const Row& row : rows) {
    CAPTURE(row.sweep);
    const bs::Bspline<double> crv =
        cad::create_circle(1.0, std::nullopt, std::array<double, 2>{0.0, row.sweep});
    CHECK(crv.num_basis(0) == 2 * row.spans + 1);
  }
}

TEST_CASE("create_circle: a zero sweep is one span of three coincident control points") {
  const bs::Bspline<double> crv =
      cad::create_circle(2.5, std::nullopt, std::array<double, 2>{0.0, 0.0});
  CHECK(crv.num_basis(0) == 3);
  CHECK(knots_as_double(crv, 0) == std::vector<double>{0, 0, 0, 1, 1, 1});
  const std::vector<double> cp = control_points_of(crv);
  for (std::size_t i = 0; i < 3; ++i) {
    // (r, 0, 0) at weight 1: alpha = 0, so tan = 0 and the template is rotated by exactly 0.
    CHECK(cp[4 * i] == 2.5);
    CHECK(cp[4 * i + 1] == 0.0);
    CHECK(cp[4 * i + 2] == 0.0);
    CHECK(cp[4 * i + 3] == 1.0);
  }
}

TEST_CASE("create_circle: the interior knots are i / spans, double, between three end knots") {
  for (std::size_t spans = 1; spans <= 4; ++spans) {
    const double sweep = static_cast<double>(spans) * (pi / 2.0) - 0.1;
    const bs::Bspline<double> crv =
        cad::create_circle(1.0, std::nullopt, std::array<double, 2>{0.0, sweep});
    CAPTURE(spans);
    REQUIRE(crv.num_basis(0) == 2 * spans + 1);
    const std::vector<double> knots = knots_as_double(crv, 0);
    REQUIRE(knots.size() == 2 * spans + 4);
    CHECK(knots[0] == 0.0);
    CHECK(knots[1] == 0.0);
    CHECK(knots[2 * spans + 2] == 1.0);
    CHECK(knots[2 * spans + 3] == 1.0);
    for (std::size_t i = 0; i <= spans; ++i) {
      // `i * (1 / spans)`: the rounding of the step and of the product, at most 2 eps for u <= 1.
      const double u = static_cast<double>(i) / static_cast<double>(spans);
      CHECK(std::abs(knots[1 + 2 * i] - u) <= 2.0 * eps<double>());
      CHECK(knots[1 + 2 * i] == knots[2 + 2 * i]);
    }
  }
}

TEST_CASE("create_circle: refusals and their messages") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const auto sweep_message = [&](double start, double end) {
    return message_of([&] {
      static_cast<void>(cad::create_circle(1.0, std::nullopt, std::array<double, 2>{start, end}));
    });
  };
  CHECK(sweep_message(0.0, nan).find("angle must be finite, got nan.") != std::string::npos);
  CHECK(sweep_message(nan, 1.0).find("angle must be finite, got nan.") != std::string::npos);
  CHECK(sweep_message(0.0, inf).find("angle must be finite, got inf.") != std::string::npos);
  // `alpha + start` is infinity minus infinity here, which the rotation reports as NaN.
  CHECK(sweep_message(-inf, 0.0).find("angle must be finite, got nan.") != std::string::npos);
  CHECK(sweep_message(inf, inf).find("angle must be finite, got nan.") != std::string::npos);
  const std::vector<double> four = {1.0, 2.0, 3.0, 4.0};
  CHECK(message_of([&] {
          static_cast<void>(cad::create_circle(1.0, sp(four)));
        }).find("Point must have at most 3 coordinates, got 4.") != std::string::npos);
}

// ------------------------------------------------------------------------- bilinear and trilinear

namespace {

/// The closed-form multilinear blend of corners given as `2^d` rows of three coordinates, the
/// first direction the slowest, at the parameters `t` (`d` values).
std::array<double, 3> blend(const std::vector<double>& corners3, std::size_t dim,
                            const std::vector<double>& t) {
  std::array<double, 3> out = {0.0, 0.0, 0.0};
  for (std::size_t c = 0; c < (std::size_t{1} << dim); ++c) {
    double weight = 1.0;
    for (std::size_t d = 0; d < dim; ++d) {
      const bool high = ((c >> (dim - 1 - d)) & 1U) != 0U;
      weight *= high ? t[d] : 1.0 - t[d];
    }
    for (std::size_t k = 0; k < 3; ++k) {
      out[k] += weight * corners3[3 * c + k];
    }
  }
  return out;
}

double max_abs(const std::vector<double>& v) {
  double m = 0.0;
  for (const double x : v) {
    m = (std::max)(m, std::abs(x));
  }
  return m;
}

/// The twisted corners of a quadrilateral, `corners[i, j]`, none coplanar with the rest.
const std::vector<double>& twisted_quad() {
  static const std::vector<double> v = {0.0, 0.0, 0.0,    // [0, 0]
                                        0.0, 2.0, 0.5,    // [0, 1]
                                        3.0, 0.5, 1.0,    // [1, 0]
                                        3.5, 2.5, -1.5};  // [1, 1]
  return v;
}

std::vector<double> twisted_hex() {
  std::vector<double> v;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        v.push_back(2.0 * i + 0.25 * j - 0.125 * k * i);
        v.push_back(1.5 * j - 0.5 * i * k + 0.125);
        v.push_back(1.0 * k + 0.5 * i * j - 0.75 * j * k * i);
      }
    }
  }
  return v;
}

}  // namespace

TEST_CASE("create_bilinear: the corners are the control points and the surface is the blend") {
  const std::vector<double>& corners = twisted_quad();
  const std::vector<std::size_t> shape = {2, 2, 3};
  const bs::Bspline<double> s = cad::create_bilinear(sp(corners), sh(shape));
  CHECK(s.dim() == 2);
  CHECK(s.degrees() == std::vector<std::size_t>{1, 1});
  CHECK(s.rank() == 3);
  CHECK_FALSE(s.rational());
  CHECK(s.num_basis(0) == 2);
  CHECK(s.num_basis(1) == 2);
  CHECK(knots_as_double(s, 0) == std::vector<double>{0, 0, 1, 1});
  CHECK(knots_as_double(s, 1) == std::vector<double>{0, 0, 1, 1});
  CHECK(control_points_of(s) == corners);

  const double tol = 4.0 * (3.0 * 2.0 + 4.0) * eps<double>() * max_abs(corners);
  const std::vector<double> us = linspace(0.0, 1.0, 7);
  const std::vector<double> pts = cad_test::grid({us, us});
  const std::vector<double> got = evaluate_at(s, pts);
  for (std::size_t i = 0; i < pts.size() / 2; ++i) {
    const std::array<double, 3> expected = blend(corners, 2, {pts[2 * i], pts[2 * i + 1]});
    for (std::size_t k = 0; k < 3; ++k) {
      CHECK(std::abs(got[3 * i + k] - expected[k]) <= tol);
    }
  }
  // The four parametric corners.
  const std::vector<double> at_corners = evaluate_at(s, {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0});
  CHECK(std::abs(at_corners[0] - corners[0]) <= tol);
  for (std::size_t k = 0; k < 3; ++k) {
    CHECK(std::abs(at_corners[3 + k] - corners[3 + k]) <= tol);
    CHECK(std::abs(at_corners[6 + k] - corners[6 + k]) <= tol);
    CHECK(std::abs(at_corners[9 + k] - corners[9 + k]) <= tol);
  }
}

TEST_CASE("create_bilinear: the default is the unit square of the xy-plane about the origin") {
  const bs::Bspline<double> s = cad::create_bilinear();
  CHECK(s.dim() == 2);
  CHECK(s.rank() == 3);
  CHECK(control_points_of(s) == std::vector<double>{-0.5, -0.5, 0.0,  // [0, 0]
                                                    -0.5, 0.5, 0.0,   // [0, 1]
                                                    0.5, -0.5, 0.0,   // [1, 0]
                                                    0.5, 0.5, 0.0});  // [1, 1]
  CHECK(knots_as_double(s, 0) == std::vector<double>{0, 0, 1, 1});
}

TEST_CASE(
    "create_bilinear: corners of fewer than three coordinates are padded, rank zero is zeros") {
  const std::vector<double> rank2 = {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0};
  const bs::Bspline<double> s2 = cad::create_bilinear(sp(rank2), sh({2, 2, 2}));
  CHECK(s2.rank() == 3);
  CHECK(control_points_of(s2) == std::vector<double>{0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0});
  const std::vector<double> rank1 = {1.0, 2.0, 3.0, 4.0};
  CHECK(control_points_of(cad::create_bilinear(sp(rank1), sh({2, 2, 1}))) ==
        std::vector<double>{1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0});
  const std::vector<double> rank0;
  const bs::Bspline<double> s0 = cad::create_bilinear(sp(rank0), sh({2, 2, 0}));
  CHECK(s0.rank() == 3);
  CHECK(control_points_of(s0) == std::vector<double>(12, 0.0));
}

TEST_CASE("create_bilinear: refusals and their messages, in order") {
  const std::vector<double> v(12, 1.0);
  const auto message = [&](const std::vector<double>& values,
                           const std::vector<std::size_t>& shape) {
    return message_of([&] { static_cast<void>(cad::create_bilinear(sp(values), sh(shape))); });
  };
  CHECK(message(v, {3, 2, 2}).find("corners must have shape (2, 2, rank), got (3, 2, 2).") !=
        std::string::npos);
  CHECK(message(v, {2, 3, 2}).find("got (2, 3, 2).") != std::string::npos);
  CHECK(message(v, {2, 2}).find("corners must have shape (2, 2, rank), got (2, 2).") !=
        std::string::npos);
  CHECK(message(v, {4}).find("got (4,).") != std::string::npos);
  CHECK(message(v, {2, 2, 2, 1}).find("got (2, 2, 2, 1).") != std::string::npos);
  CHECK(message(v, {2, 2, 4}).find("corners rank must be at most 3, got 4.") != std::string::npos);
  // The shape is checked before the rank.
  CHECK(message(v, {3, 2, 4}).find("corners must have shape") != std::string::npos);
  // The count of values is checked last, with this library's own message.
  CHECK(message(std::vector<double>(11, 1.0), {2, 2, 3})
            .find("corners holds 11 values, but its shape (2, 2, 3) needs 12.") !=
        std::string::npos);
}

TEST_CASE("create_trilinear: the corners are the control points and the volume is the blend") {
  const std::vector<double> corners = twisted_hex();
  const std::vector<std::size_t> shape = {2, 2, 2, 3};
  const bs::Bspline<double> s = cad::create_trilinear(sp(corners), sh(shape));
  CHECK(s.dim() == 3);
  CHECK(s.degrees() == std::vector<std::size_t>{1, 1, 1});
  CHECK(s.rank() == 3);
  CHECK_FALSE(s.rational());
  for (std::size_t d = 0; d < 3; ++d) {
    CHECK(s.num_basis(d) == 2);
    CHECK(knots_as_double(s, d) == std::vector<double>{0, 0, 1, 1});
  }
  CHECK(control_points_of(s) == corners);

  const double tol = 4.0 * (3.0 * 3.0 + 8.0) * eps<double>() * max_abs(corners);
  const std::vector<double> us = linspace(0.0, 1.0, 5);
  const std::vector<double> pts = cad_test::grid({us, us, us});
  const std::vector<double> got = evaluate_at(s, pts);
  for (std::size_t i = 0; i < pts.size() / 3; ++i) {
    const std::array<double, 3> expected =
        blend(corners, 3, {pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]});
    for (std::size_t k = 0; k < 3; ++k) {
      CHECK(std::abs(got[3 * i + k] - expected[k]) <= tol);
    }
  }
}

TEST_CASE("create_trilinear: the default unit cube, padding and refusals") {
  const bs::Bspline<double> cube = cad::create_trilinear();
  CHECK(cube.dim() == 3);
  std::vector<double> expected;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        expected.insert(expected.end(), {i - 0.5, j - 0.5, k - 0.5});
      }
    }
  }
  CHECK(control_points_of(cube) == expected);

  const std::vector<double> ones(16, 1.0);
  const bs::Bspline<double> padded2 = cad::create_trilinear(sp(ones), sh({2, 2, 2, 2}));
  CHECK(padded2.rank() == 3);
  CHECK(control_points_of(padded2)[2] == 0.0);
  CHECK(control_points_of(padded2)[0] == 1.0);
  const std::vector<double> none;
  CHECK(control_points_of(cad::create_trilinear(sp(none), sh({2, 2, 2, 0}))) ==
        std::vector<double>(24, 0.0));

  const auto message = [&](const std::vector<double>& values,
                           const std::vector<std::size_t>& shape) {
    return message_of([&] { static_cast<void>(cad::create_trilinear(sp(values), sh(shape))); });
  };
  const std::vector<double> v(32, 1.0);
  CHECK(
      message(v, {2, 2, 3, 3}).find("corners must have shape (2, 2, 2, rank), got (2, 2, 3, 3).") !=
      std::string::npos);
  CHECK(message(v, {2, 2, 2}).find("got (2, 2, 2).") != std::string::npos);
  CHECK(message(v, {2, 2, 2, 4}).find("corners rank must be at most 3, got 4.") !=
        std::string::npos);
  CHECK(message(std::vector<double>(23, 1.0), {2, 2, 2, 3})
            .find("corners holds 23 values, but its shape (2, 2, 2, 3) needs 24.") !=
        std::string::npos);
}
