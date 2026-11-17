// `create_ruled` and `create_revolution`.
//
// Oracles. A ruled surface or volume is checked at its two ends, `v = 0` and `v = 1`, where it must
// reproduce its two inputs: the end faces are extracted with `bspline::boundary` (a de Boor
// corner-cutting that at a knot of full multiplicity selects control points), and the geometry is
// evaluated at the grid of parameters and compared with the input evaluated directly; between the
// ends it is the average of the two. The rational annulus between two circles has the radius
// `(1 - v) r1 + v r2` at every parameter, a closed form. A revolution is checked by what it
// preserves: every point of the surface lies at the distance from the axis, and at the coordinate
// along the axis, of the input point it came from, `S(u, v) = rot_axis(v) C(u)`; at `v = 0` and
// `v = 1` it is the input rotated by the start and the end angle, by Rodrigues' formula in vector
// form (`p cos t + (a x p) sin t + a (a . p)(1 - cos t)` about the axis `a` through the point `q`),
// a route that shares nothing with the matrices of the implementation. Axes: z, y, x, an oblique
// one given unnormalized, -z (the flip branch), padded short axes; the NaN axis row of the
// reference implementation (accepted, revolves about z) is compared with the z axis. Messages are
// matched by substring.
//
// Tolerances. The ruled tests of two degree-1 inputs cost, per evaluated coordinate, the blend of
// the bilinear primitive: `4 (3 d + 2^d) eps max|cp|` for `d = 2` (each end face evaluated and
// averaged), and `tier_tolerance` where `make_compat` elevates or inserts (a change of basis, a
// chain of long accumulations). The annulus is two rational quadratics (about 55 eps r_max each,
// see test_cad_primitives.cpp), stacked by a convex combination (4 more), evaluated in a second
// direction (the product of the basis values and a sum, 6 more), and compared with a closed form:
// 256 eps r_max. A revolution costs, relative to the scale `S = 1 + max |control point| + |q|` (the
// radius of the ball the rotations act on, in the Euclidean norm of the three leading
// coordinates): the translation and the rotation of the input (about 8, the matrix entries being
// rounded trigonometric functions), the arc's control points (about 10 in the angle `alpha`), the
// product with `rho`, `cos theta`, `sin theta` and the sum of four terms (about 8), the weights'
// product (1), the transform back (about 8), evaluation of the rational tensor product (about 30),
// and the test's own Rodrigues rotation and dot products (about 10); about 75 in all, and the
// factor 256 is 3.4 times that, for the amplification by a weight no smaller than cos(pi/4) in the
// division and for another libm or contraction. Where the input is a rational circle the evaluation
// of the input itself adds its 25, still inside the factor. An axis near z adds a term of its own:
// the alignment angle is `acos(v_z)`, whose argument carries an absolute error of about 2 eps (the
// norm and the division that make the axis a unit vector), so the angle carries `2 eps /
// sin(gamma)`, and the same angle serves the transform and its inverse, which tilts the axis the
// surface is revolved about by that much: a position error of `S * 2 eps / sin(gamma)`. The tests
// of an axis near z hold the comparisons to `(256 + 4 / sin(gamma)) eps S`.
#include <doctest/doctest.h>

#include "bspline_test_support.hpp"
#include "cad_test_support.hpp"

#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/slice.hpp>
#include <pantr/cad/compat.hpp>
#include <pantr/cad/detail/validation.hpp>
#include <pantr/cad/operations.hpp>
#include <pantr/cad/primitives.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/tolerance.hpp>

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
using bspline_test::clamped;
using bspline_test::Dir;
using bspline_test::make_spline;
using bspline_test::message_of;
using bspline_test::tier_tolerance;
using cad_test::control_points_of;
using cad_test::eps;
using cad_test::evaluate_at;
using cad_test::grid;
using cad_test::knots_as_double;
using cad_test::linspace;

namespace {

constexpr double pi = std::numbers::pi;

std::span<const double> sp(const std::vector<double>& v) { return std::span<const double>(v); }

template <class T>
bs::Bspline<T> curve(int degree, const std::vector<double>& interior, std::size_t rank = 3,
                     unsigned seed = 1, bool rational = false) {
  return make_spline<T>({Dir{clamped(degree, interior), degree}}, rank, rational, seed);
}

template <class T>
bs::Bspline<T> surface(int deg_u, const std::vector<double>& in_u, int deg_v,
                       const std::vector<double>& in_v, unsigned seed, bool rational = false) {
  return make_spline<T>({Dir{clamped(deg_u, in_u), deg_u}, Dir{clamped(deg_v, in_v), deg_v}}, 3,
                        rational, seed);
}

double max_abs_of(const std::vector<double>& v) {
  double m = 0.0;
  for (const double x : v) {
    m = (std::max)(m, std::abs(x));
  }
  return m;
}

}  // namespace

// ----------------------------------------------------------------------------------------- ruled

TEST_CASE_TEMPLATE("create_ruled: two curves of degree 1 give a bilinear surface with them as ends",
                   T, float, double) {
  const bs::Bspline<T> a = curve<T>(1, {}, 3, 1);
  const bs::Bspline<T> b = curve<T>(1, {}, 3, 2);
  const bs::Bspline<T> r = cad::create_ruled<T>(a, b);
  CHECK(r.dim() == 2);
  CHECK(r.degrees() == std::vector<std::size_t>{1, 1});
  CHECK(r.rank() == 3);
  CHECK_FALSE(r.rational());
  CHECK(knots_as_double(r, 1) == std::vector<double>{0, 0, 1, 1});
  CHECK(knots_as_double(r, 0) == knots_as_double(a, 0));

  // The end faces are the inputs' control points, exactly.
  CHECK(control_points_of(bs::boundary<T>(r, 1, 0)) == control_points_of(a));
  CHECK(control_points_of(bs::boundary<T>(r, 1, 1)) == control_points_of(b));

  // The geometry, from the other side: the surface at v = 0, 1 is the curve, at v = 1/2 the mean.
  const std::vector<double> us = linspace(0.0, 1.0, 9);
  const double scale =
      (std::max)(max_abs_of(control_points_of(a)), max_abs_of(control_points_of(b)));
  const double tol = 4.0 * (3.0 * 2.0 + 4.0) * eps<T>() * scale;
  const std::vector<double> ca = evaluate_at(a, us);
  const std::vector<double> cb = evaluate_at(b, us);
  std::vector<double> at0;
  std::vector<double> at1;
  std::vector<double> at_half;
  for (const double u : us) {
    at0.insert(at0.end(), {u, 0.0});
    at1.insert(at1.end(), {u, 1.0});
    at_half.insert(at_half.end(), {u, 0.5});
  }
  const std::vector<double> s0 = evaluate_at(r, at0);
  const std::vector<double> s1 = evaluate_at(r, at1);
  const std::vector<double> sh = evaluate_at(r, at_half);
  for (std::size_t i = 0; i < ca.size(); ++i) {
    CHECK(std::abs(s0[i] - ca[i]) <= tol);
    CHECK(std::abs(s1[i] - cb[i]) <= tol);
    CHECK(std::abs(sh[i] - 0.5 * (ca[i] + cb[i])) <= tol);
  }
}

TEST_CASE_TEMPLATE("create_ruled: inputs of other degrees and knots are made compatible first", T,
                   float, double) {
  const bs::Bspline<T> a = curve<T>(1, {}, 3, 1);
  const bs::Bspline<T> b = curve<T>(2, {0.5}, 3, 2);
  const bs::Bspline<T> r = cad::create_ruled<T>(a, b);
  CHECK(r.degrees() == std::vector<std::size_t>{2, 1});
  const std::vector<double> expected_knots = [] {
    std::vector<double> k;
    for (const double x : clamped(2, {0.5})) {
      k.push_back(static_cast<double>(static_cast<T>(x)));
    }
    return k;
  }();
  CHECK(knots_as_double(r, 0) == expected_knots);
  const std::vector<double> us = linspace(0.0, 1.0, 11);
  std::vector<double> at0;
  std::vector<double> at1;
  for (const double u : us) {
    at0.insert(at0.end(), {u, 0.0});
    at1.insert(at1.end(), {u, 1.0});
  }
  // Elevation and insertion change the basis: the conservative tier, as in the compat tests.
  const double tol = tier_tolerance(r);
  const std::vector<double> s0 = evaluate_at(r, at0);
  const std::vector<double> s1 = evaluate_at(r, at1);
  const std::vector<double> ca = evaluate_at(a, us);
  const std::vector<double> cb = evaluate_at(b, us);
  for (std::size_t i = 0; i < ca.size(); ++i) {
    CHECK(std::abs(s0[i] - ca[i]) <= tol);
    CHECK(std::abs(s1[i] - cb[i]) <= tol);
  }
}

TEST_CASE_TEMPLATE("create_ruled: two surfaces give a volume with them as end faces", T, float,
                   double) {
  const bs::Bspline<T> a = surface<T>(1, {}, 2, {0.5}, 1);
  const bs::Bspline<T> b = surface<T>(1, {}, 2, {0.5}, 2);
  const bs::Bspline<T> r = cad::create_ruled<T>(a, b);
  CHECK(r.dim() == 3);
  CHECK(r.degrees() == std::vector<std::size_t>{1, 2, 1});
  CHECK(control_points_of(bs::boundary<T>(r, 2, 0)) == control_points_of(a));
  CHECK(control_points_of(bs::boundary<T>(r, 2, 1)) == control_points_of(b));
  const std::vector<double> us = linspace(0.0, 1.0, 5);
  const std::vector<double> pts = grid({us, us});
  std::vector<double> mid;
  for (std::size_t i = 0; i < pts.size() / 2; ++i) {
    mid.insert(mid.end(), {pts[2 * i], pts[2 * i + 1], 0.5});
  }
  const std::vector<double> sa = evaluate_at(a, pts);
  const std::vector<double> sb = evaluate_at(b, pts);
  const std::vector<double> sm = evaluate_at(r, mid);
  // A degree-(1, 2, 1) tensor product: per coordinate (3 + 1)(3 + 1)... bounded by the tier.
  const double tol = tier_tolerance(r);
  for (std::size_t i = 0; i < sa.size(); ++i) {
    CHECK(std::abs(sm[i] - 0.5 * (sa[i] + sb[i])) <= tol);
  }
}

TEST_CASE_TEMPLATE("create_ruled: a polynomial input meeting a rational one gets unit weights", T,
                   float, double) {
  const bs::Bspline<T> poly = curve<T>(2, {0.5}, 3, 1);
  const bs::Bspline<T> rat = curve<T>(2, {0.5}, 3, 2, true);
  const bs::Bspline<T> r = cad::create_ruled<T>(poly, rat);
  CHECK(r.rational());
  CHECK(r.cp_size() == 4);
  CHECK(r.rank() == 3);
  // The face at v = 0 is the polynomial curve's control points with a weight of exactly 1.
  const std::vector<double> face = control_points_of(bs::boundary<T>(r, 1, 0));
  const std::vector<double> cp = control_points_of(poly);
  REQUIRE(face.size() == cp.size() / 3 * 4);
  for (std::size_t p = 0; p < cp.size() / 3; ++p) {
    for (std::size_t k = 0; k < 3; ++k) {
      CHECK(face[4 * p + k] == cp[3 * p + k]);
    }
    CHECK(face[4 * p + 3] == 1.0);
  }
  CHECK(control_points_of(bs::boundary<T>(r, 1, 1)) == control_points_of(rat));

  // Either order promotes.
  const bs::Bspline<T> r2 = cad::create_ruled<T>(rat, poly);
  CHECK(r2.rational());
  CHECK(control_points_of(bs::boundary<T>(r2, 1, 0)) == control_points_of(rat));

  // The surface at v = 0 and 1 is the curve each came from.
  const std::vector<double> us = linspace(0.0, 1.0, 9);
  std::vector<double> at0;
  std::vector<double> at1;
  for (const double u : us) {
    at0.insert(at0.end(), {u, 0.0});
    at1.insert(at1.end(), {u, 1.0});
  }
  const double tol = tier_tolerance(r);
  const std::vector<double> s0 = evaluate_at(r, at0);
  const std::vector<double> s1 = evaluate_at(r, at1);
  const std::vector<double> cp_poly = evaluate_at(poly, us);
  const std::vector<double> cp_rat = evaluate_at(rat, us);
  for (std::size_t i = 0; i < cp_poly.size(); ++i) {
    CHECK(std::abs(s0[i] - cp_poly[i]) <= tol);
    CHECK(std::abs(s1[i] - cp_rat[i]) <= tol);
  }
}

TEST_CASE("create_ruled: the annulus between two circles has the interpolated radius") {
  const bs::Bspline<double> inner = cad::create_circle(1.0);
  const bs::Bspline<double> outer = cad::create_circle(2.0);
  const bs::Bspline<double> r = cad::create_ruled<double>(inner, outer);
  CHECK(r.dim() == 2);
  CHECK(r.rational());
  CHECK(r.degrees() == std::vector<std::size_t>{2, 1});
  const std::vector<double> pts = grid({linspace(0.0, 1.0, 31), {0.0, 0.25, 0.5, 1.0}});
  const std::vector<double> got = evaluate_at(r, pts);
  const double tol = 256.0 * eps<double>() * 2.0;
  for (std::size_t i = 0; i < pts.size() / 2; ++i) {
    const double v = pts[2 * i + 1];
    const double radius = std::hypot(got[3 * i], got[3 * i + 1]);
    CHECK(std::abs(radius - ((1.0 - v) * 1.0 + v * 2.0)) <= tol);
    CHECK(std::abs(got[3 * i + 2]) <= tol);
  }
}

TEST_CASE_TEMPLATE("create_ruled: a second B-spline of one coordinate is repeated over the columns",
                   T, float, double) {
  // Inherited from the NumPy implementation, which broadcast the width-1 second curve over the
  // three columns of the first; the reverse and every other mismatch of widths is refused.
  const bs::Bspline<T> wide = curve<T>(1, {}, 3, 1);
  const bs::Bspline<T> narrow = curve<T>(1, {}, 1, 2);
  const bs::Bspline<T> r = cad::create_ruled<T>(wide, narrow);
  CHECK(r.cp_size() == 3);
  CHECK(control_points_of(bs::boundary<T>(r, 1, 0)) == control_points_of(wide));
  const std::vector<double> face = control_points_of(bs::boundary<T>(r, 1, 1));
  const std::vector<double> one = control_points_of(narrow);
  REQUIRE(face.size() == 3 * one.size());
  for (std::size_t p = 0; p < one.size(); ++p) {
    for (std::size_t k = 0; k < 3; ++k) {
      CHECK(face[3 * p + k] == one[p]);
    }
  }
}

TEST_CASE_TEMPLATE("create_ruled: control points of different widths are refused", T, float,
                   double) {
  const auto message = [&](const bs::Bspline<T>& a, const bs::Bspline<T>& b) {
    return message_of([&] { static_cast<void>(cad::create_ruled<T>(a, b)); });
  };
  const bs::Bspline<T> r1 = curve<T>(1, {}, 1, 1);
  const bs::Bspline<T> r2 = curve<T>(1, {}, 2, 2);
  const bs::Bspline<T> r3 = curve<T>(1, {}, 3, 3);
  const bs::Bspline<T> rat3 = curve<T>(1, {}, 3, 4, true);
  const bs::Bspline<T> rat1 = curve<T>(1, {}, 1, 5, true);
  CHECK(message(r1, r3).find("ruled requires control points of the same width, got 1 and 3") !=
        std::string::npos);
  CHECK(message(r2, r3).find("got 2 and 3") != std::string::npos);
  CHECK(message(r3, r2).find("got 3 and 2") != std::string::npos);
  // After the promotion: a rational rank-3 first (4 columns) with a polynomial rank-1 second (2).
  CHECK(message(rat3, r1).find("got 4 and 2 after promotion") != std::string::npos);
  // A polynomial rank-3 first, promoted to 4 columns, with a rational rank-1 second (2 columns).
  CHECK(message(r3, rat1).find("got 4 and 2 after promotion") != std::string::npos);
}

TEST_CASE_TEMPLATE("create_ruled: knots a rounding apart leave unequal sizes, which are refused", T,
                   float, double) {
  // Inherited from the reference implementation, whose matching of knot classes looks for the
  // counterpart of a knot only at the first knot not below it: an interior knot a quarter of the
  // space tolerance above another is a counterpart of it in one space and not in the other, so
  // `make_compat` inserts a knot in one and none in the other and returns B-splines with 5 and 4
  // functions (reproduced here, and the reference's ruled surface then failed with a NumPy
  // broadcast error where this one raises a value_error of its own).
  const bs::Bspline<T> a = curve<T>(2, {0.5}, 3, 1);
  const double tol = a.space().spaces()[0].tolerance();
  const double shifted = static_cast<double>(static_cast<T>(0.5 + 0.25 * tol));
  REQUIRE(shifted > 0.5);
  REQUIRE(shifted - 0.5 < tol);
  const bs::Bspline<T> b = curve<T>(2, {shifted}, 3, 2);
  const std::array<bs::Bspline<T>, 2> pair = {a, b};
  const std::vector<bs::Bspline<T>> made =
      cad::make_compat<T>(std::span<const bs::Bspline<T>>(pair));
  REQUIRE(made[0].num_basis(0) != made[1].num_basis(0));
  CHECK(message_of([&] { static_cast<void>(cad::create_ruled<T>(a, b)); })
            .find("ruled requires B-splines with equal numbers of control points after making "
                  "them compatible, got") != std::string::npos);
  CHECK(message_of([&] {
          static_cast<void>(cad::create_ruled<T>(b, a));
        }).find("equal numbers of control points") != std::string::npos);
}

TEST_CASE_TEMPLATE("create_ruled: the dimensions are checked, equal first and then at most two", T,
                   float, double) {
  const bs::Bspline<T> crv = curve<T>(1, {}, 3, 1);
  const bs::Bspline<T> srf = surface<T>(1, {}, 1, {}, 2);
  const bs::Bspline<T> vol = make_spline<T>(
      {Dir{clamped(1, {}), 1}, Dir{clamped(1, {}), 1}, Dir{clamped(1, {}), 1}}, 3, false, 3);
  CHECK(message_of([&] {
          static_cast<void>(cad::create_ruled<T>(crv, srf));
        }).find("Both B-splines must have the same dim, got 1 and 2.") != std::string::npos);
  CHECK(message_of([&] {
          static_cast<void>(cad::create_ruled<T>(vol, srf));
        }).find("got 3 and 2.") != std::string::npos);
  CHECK(message_of([&] {
          static_cast<void>(cad::create_ruled<T>(vol, vol));
        }).find("ruled requires dim <= 2, got 3.") != std::string::npos);
}

// ------------------------------------------------------------------------------------ revolution

namespace {

using Vec3 = std::array<double, 3>;

Vec3 sub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Vec3 unit(const Vec3& a) {
  const double n = std::sqrt(dot(a, a));
  return {a[0] / n, a[1] / n, a[2] / n};
}

/// `q + rotation(p - q)` about the unit axis `a` through `q` by `theta`: Rodrigues' formula.
Vec3 rotate_about(const Vec3& p, const Vec3& q, const Vec3& a, double theta) {
  const Vec3 d = sub(p, q);
  const Vec3 axd = cross(a, d);
  const double along = dot(a, d) * (1.0 - std::cos(theta));
  Vec3 out{};
  for (std::size_t k = 0; k < 3; ++k) {
    out[k] = q[k] + (d[k] * std::cos(theta) + axd[k] * std::sin(theta) + a[k] * along);
  }
  return out;
}

Vec3 row(const std::vector<double>& v, std::size_t i) {
  return {v[3 * i], v[3 * i + 1], v[3 * i + 2]};
}

/// What a revolution is checked against.
struct Revolve {
  bs::Bspline<double> input;
  std::vector<double> point;  // up to three coordinates
  std::vector<double> axis;   // up to three coordinates, any length
  std::optional<std::array<double, 2>> sweep;
};

struct Frame {
  Vec3 q;
  Vec3 a;
};

Frame frame_of(const Revolve& r) {
  Frame f{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  std::copy(r.point.begin(), r.point.end(), f.q.begin());
  std::copy(r.axis.begin(), r.axis.end(), f.a.begin());
  f.a = unit(f.a);
  return f;
}

/// The scale `S` of the header note: the radius of the ball the rotations act on, `1 + max
/// |control point| + |q|` in the Euclidean norm of the three leading coordinates.
double revolve_scale(const Revolve& r, const Frame& f) {
  const std::size_t width = r.input.cp_size();
  const std::span<const double> cp = r.input.control_points();
  double largest = 0.0;
  for (std::size_t i = 0; i < cp.size() / width; ++i) {
    const Vec3 c = {cp[i * width], cp[i * width + 1], cp[i * width + 2]};
    largest = (std::max)(largest, std::sqrt(dot(c, c)));
  }
  return 1.0 + largest + std::sqrt(dot(f.q, f.q));
}

/// Check a revolution against the input: distances and axial coordinates preserved over the whole
/// angular range, and the ends at the input rotated by the start and the end angle.
///
/// Args:
///     r (const Revolve&): The revolution.
///     factor (double): The multiple of `eps * S` the comparisons are held to; 256 unless the
///         axis is near z (see the test of that name).
void check_revolution(const Revolve& r, double factor = 256.0) {
  const bs::Bspline<double> s = cad::create_revolution(r.input, sp(r.point), sp(r.axis), r.sweep);
  const std::size_t dim = r.input.dim();
  REQUIRE(s.dim() == dim + 1);
  CHECK(s.rational());
  CHECK(s.rank() == 3);
  const Frame f = frame_of(r);
  const double tol = factor * eps<double>() * revolve_scale(r, f);

  // The structure: the input's directions first, unchanged, then the arc's, as `create_circle`
  // builds them for the same sweep.
  const bs::Bspline<double> arc = cad::create_circle(1.0, std::nullopt, r.sweep);
  for (std::size_t d = 0; d < dim; ++d) {
    CHECK(knots_as_double(s, d) == knots_as_double(r.input, d));
    CHECK(s.degrees()[d] == r.input.degrees()[d]);
  }
  CHECK(knots_as_double(s, dim) == knots_as_double(arc, 0));
  CHECK(s.degrees()[dim] == 2);

  const std::vector<double> us = linspace(0.0, 1.0, 7);
  std::vector<std::vector<double>> axes(dim, us);
  const std::vector<double> in_pts = grid(axes);
  const std::size_t n_in = in_pts.size() / dim;
  const std::vector<double> base = evaluate_at(r.input, in_pts);

  const std::array<double, 2> angles = r.sweep ? *r.sweep : std::array<double, 2>{0.0, 2.0 * pi};
  const std::vector<double> vs = linspace(0.0, 1.0, 13);
  std::vector<double> all_pts;
  for (std::size_t i = 0; i < n_in; ++i) {
    for (const double v : vs) {
      for (std::size_t d = 0; d < dim; ++d) {
        all_pts.push_back(in_pts[dim * i + d]);
      }
      all_pts.push_back(v);
    }
  }
  const std::vector<double> got = evaluate_at(s, all_pts);
  for (std::size_t i = 0; i < n_in; ++i) {
    const Vec3 p = row(base, i);
    const double axial = dot(f.a, sub(p, f.q));
    const Vec3 along = {f.a[0] * axial, f.a[1] * axial, f.a[2] * axial};
    const Vec3 radial_in = sub(sub(p, f.q), along);
    const double radius = std::sqrt(dot(radial_in, radial_in));
    for (std::size_t j = 0; j < vs.size(); ++j) {
      const Vec3 x = row(got, i * vs.size() + j);
      const double x_axial = dot(f.a, sub(x, f.q));
      const Vec3 x_along = {f.a[0] * x_axial, f.a[1] * x_axial, f.a[2] * x_axial};
      const Vec3 radial = sub(sub(x, f.q), x_along);
      CHECK(std::abs(x_axial - axial) <= tol);
      CHECK(std::abs(std::sqrt(dot(radial, radial)) - radius) <= tol);
    }
    for (std::size_t end = 0; end < 2; ++end) {
      const Vec3 expected = rotate_about(p, f.q, f.a, angles[end]);
      const Vec3 x = row(got, i * vs.size() + (end == 0 ? 0 : vs.size() - 1));
      for (std::size_t k = 0; k < 3; ++k) {
        CHECK(std::abs(x[k] - expected[k]) <= tol);
      }
    }
  }
}

bs::Bspline<double> radial_line() {
  const std::vector<double> a = {1.0, 0.0, 0.0};
  const std::vector<double> b = {2.0, 0.0, 3.0};
  return cad::create_line(sp(a), sp(b));
}

bs::Bspline<double> generic_curve() {
  return make_spline<double>({Dir{clamped(2, {0.4}), 2}}, 3, false, 7);
}

bs::Bspline<double> offset_circle() {
  const std::vector<double> c = {2.0, 0.5, 0.0};
  return cad::create_circle(0.5, sp(c));
}

bs::Bspline<double> twisted_surface() {
  const std::vector<double> v = {1.0, 0.0, 0.0, 1.0, 2.0, 0.5, 3.0, 0.5, 1.0, 3.5, 2.5, -1.5};
  const std::vector<std::size_t> shape = {2, 2, 3};
  return cad::create_bilinear(sp(v), std::span<const std::size_t>(shape));
}

}  // namespace

TEST_CASE("create_revolution: a curve about the coordinate axes, full turn and partial sweeps") {
  const std::vector<std::optional<std::array<double, 2>>> sweeps = {
      std::nullopt,
      std::array<double, 2>{0.0, pi / 2.0},
      std::array<double, 2>{pi / 2.0, pi},
      std::array<double, 2>{0.0, -pi / 2.0},
      std::array<double, 2>{0.3, 5.0},
      std::array<double, 2>{-1.0, 0.7},
      std::array<double, 2>{0.0, 0.0},
  };
  const std::vector<std::vector<double>> axes = {{0, 0, 1}, {0, 1, 0}, {1, 0, 0}};
  for (const auto& sweep : sweeps) {
    for (const auto& axis : axes) {
      CAPTURE(axis[0]);
      CAPTURE(axis[1]);
      check_revolution({radial_line(), {0.0}, axis, sweep});
      check_revolution({generic_curve(), {0.5, -0.25, 1.0}, axis, sweep});
    }
  }
}

TEST_CASE("create_revolution: oblique axes, given unnormalized, and the flipped z axis") {
  const std::vector<std::vector<double>> axes = {{1, 2, 2},  {-1, -1, 1}, {0, 0, -1},
                                                 {0, 0, -5}, {3, 0, 0.1}, {1, 1, 0}};
  const std::vector<std::vector<double>> points = {{0.0}, {2.0, -1.0, 0.5}, {1.0, 1.0}};
  const std::vector<std::optional<std::array<double, 2>>> sweeps = {
      std::nullopt, std::array<double, 2>{0.0, pi / 2.0}, std::array<double, 2>{0.3, 5.0}};
  for (const auto& axis : axes) {
    for (const auto& point : points) {
      for (const auto& sweep : sweeps) {
        CAPTURE(axis[2]);
        CAPTURE(point.size());
        check_revolution({generic_curve(), point, axis, sweep});
        check_revolution({offset_circle(), point, axis, sweep});
      }
    }
  }
}

TEST_CASE("create_revolution: a surface gives a volume, a rational curve a torus-like surface") {
  check_revolution({twisted_surface(), {0.0}, {0, 0, 1}, std::nullopt});
  check_revolution(
      {twisted_surface(), {0.5, 0.0, -1.0}, {1, 2, 2}, std::array<double, 2>{0.0, pi / 3.0}});
  check_revolution({offset_circle(), {0.0}, {0, 1, 0}, std::nullopt});
  // A torus: the circle of radius 1/2 centred at (2, 0, 0) in the xy-plane about the y axis has
  // distance from the y axis |2 + cos(t) / 2| and height sin(t) / 2: its points satisfy
  // (rho - 2)^2 + y^2 = 1/4 with rho = hypot(x, z).
  const std::vector<double> c = {2.0, 0.0, 0.0};
  const bs::Bspline<double> circle = cad::create_circle(0.5, sp(c));
  const std::vector<double> origin = {0.0};
  const std::vector<double> y_axis = {0.0, 1.0, 0.0};
  const bs::Bspline<double> torus = cad::create_revolution(circle, sp(origin), sp(y_axis));
  const std::vector<double> pts = grid({linspace(0.0, 1.0, 41), linspace(0.0, 1.0, 17)});
  const std::vector<double> got = evaluate_at(torus, pts);
  const double tol = 256.0 * eps<double>() * 2.5;
  for (std::size_t i = 0; i < pts.size() / 2; ++i) {
    const double rho = std::hypot(got[3 * i], got[3 * i + 2]);
    const double y = got[3 * i + 1];
    CHECK(std::abs(std::hypot(rho - 2.0, y) - 0.5) <= tol);
  }
}

TEST_CASE("create_revolution: an axis near z is held to the bound that grows as 1 / sin(angle)") {
  // The alignment angle is `acos(v_z)`, which loses accuracy as the axis approaches z (see the
  // header note); the comparisons are held to `(256 + 4 / sin(gamma)) eps S`.
  for (const double x : {1e-1, 1e-2, 1e-3, 1e-4, 1e-6}) {
    const std::vector<double> axis = {x, 0.0, 1.0};
    const double sin_gamma = x / std::sqrt(1.0 + x * x);
    CAPTURE(x);
    check_revolution({generic_curve(), {0.5, -0.25, 1.0}, axis, std::nullopt},
                     256.0 + 4.0 / sin_gamma);
    check_revolution({generic_curve(), {0.0}, axis, std::array<double, 2>{0.3, 5.0}},
                     256.0 + 4.0 / sin_gamma);
  }
}

TEST_CASE("create_revolution: a NaN in the axis is accepted and revolves about z") {
  // Inherited from the reference implementation: the normalized axis is NaN, both comparisons on
  // it are false, and only the translation to the point is applied, which is what the z axis gets.
  const bs::Bspline<double> input = generic_curve();
  const std::vector<double> point = {0.5, -1.0, 0.25};
  const std::vector<double> z_axis = {0.0, 0.0, 1.0};
  const std::vector<double> nan_axis = {0.0, std::numeric_limits<double>::quiet_NaN(), 1.0};
  const bs::Bspline<double> about_z = cad::create_revolution(input, sp(point), sp(z_axis));
  const bs::Bspline<double> about_nan = cad::create_revolution(input, sp(point), sp(nan_axis));
  CHECK(control_points_of(about_nan) == control_points_of(about_z));
  CHECK(knots_as_double(about_nan, 1) == knots_as_double(about_z, 1));
}

TEST_CASE("create_revolution: a short axis is padded with zeros") {
  const bs::Bspline<double> input = generic_curve();
  const std::vector<double> origin = {0.0};
  const std::vector<double> short_axis = {1.0};
  const std::vector<double> x_axis = {1.0, 0.0, 0.0};
  const std::vector<double> two = {1.0, 0.0};
  const bs::Bspline<double> a = cad::create_revolution(input, sp(origin), sp(x_axis));
  CHECK(control_points_of(cad::create_revolution(input, sp(origin), sp(short_axis))) ==
        control_points_of(a));
  CHECK(control_points_of(cad::create_revolution(input, sp(origin), sp(two))) ==
        control_points_of(a));
}

TEST_CASE("create_revolution: refusals and their messages, in order") {
  const bs::Bspline<double> crv = generic_curve();
  const std::vector<double> origin = {0.0};
  const std::vector<double> z_axis = {0.0, 0.0, 1.0};
  const auto message = [&](const bs::Bspline<double>& b, const std::vector<double>& point,
                           const std::vector<double>& axis,
                           std::optional<std::array<double, 2>> sweep = std::nullopt) {
    return message_of(
        [&] { static_cast<void>(cad::create_revolution(b, sp(point), sp(axis), sweep)); });
  };
  const bs::Bspline<double> vol = make_spline<double>(
      {Dir{clamped(1, {}), 1}, Dir{clamped(1, {}), 1}, Dir{clamped(1, {}), 1}}, 3, false, 3);
  const bs::Bspline<double> rank2 = make_spline<double>({Dir{clamped(1, {}), 1}}, 2, false, 4);
  const bs::Bspline<double> vol_rank2 = make_spline<double>(
      {Dir{clamped(1, {}), 1}, Dir{clamped(1, {}), 1}, Dir{clamped(1, {}), 1}}, 2, false, 5);
  CHECK(message(vol, origin, z_axis).find("create_revolution requires dim <= 2, got 3.") !=
        std::string::npos);
  CHECK(message(rank2, origin, z_axis).find("create_revolution requires rank == 3, got 2.") !=
        std::string::npos);
  // The dimension is checked before the rank, the rank before the point, the point before the axis.
  CHECK(message(vol_rank2, origin, z_axis).find("requires dim <= 2") != std::string::npos);
  CHECK(message(rank2, {1.0, 2.0, 3.0, 4.0}, {0.0, 0.0, 0.0}).find("requires rank == 3") !=
        std::string::npos);
  CHECK(message(crv, {1.0, 2.0, 3.0, 4.0}, {0.0, 0.0, 0.0})
            .find("Point must have at most 3 coordinates, got 4.") != std::string::npos);
  CHECK(message(crv, origin, {0.0, 0.0, 0.0}).find("Rotation axis must be non-zero.") !=
        std::string::npos);
  CHECK(message(crv, origin, {}).find("Rotation axis must be non-zero.") != std::string::npos);
  CHECK(message(crv, origin, {0.0, 0.0, 1.0, 0.0})
            .find("Rotation axis must have at most 3 coordinates, got 4.") != std::string::npos);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  CHECK(message(crv, origin, z_axis, std::array<double, 2>{0.0, nan})
            .find("angle must be finite, got nan.") != std::string::npos);
}

// ------------------------------------------------------------------- shared helpers
// (validation.hpp)

TEST_CASE("cad::detail::pad_to_3d: zeros after the given coordinates, at most three") {
  using cad::detail::pad_to_3d;
  const std::vector<double> none;
  const std::vector<double> one = {5.0};
  const std::vector<double> three = {1.0, 2.0, 3.0};
  const std::vector<double> four = {1.0, 2.0, 3.0, 4.0};
  CHECK(pad_to_3d(sp(none)) == std::array<double, 3>{0.0, 0.0, 0.0});
  CHECK(pad_to_3d(sp(one)) == std::array<double, 3>{5.0, 0.0, 0.0});
  CHECK(pad_to_3d(sp(three)) == std::array<double, 3>{1.0, 2.0, 3.0});
  CHECK(message_of([&] {
          static_cast<void>(pad_to_3d(sp(four)));
        }).find("Point must have at most 3 coordinates, got 4.") != std::string::npos);
}

TEST_CASE_TEMPLATE("cad::detail::promote_to_rational: unit weights appended, rational kept", T,
                   float, double) {
  using cad::detail::promote_to_rational;
  const bs::Bspline<T> poly = curve<T>(2, {0.5}, 2, 1);
  const bs::Bspline<T> rat = promote_to_rational<T>(poly);
  CHECK(rat.rational());
  CHECK(rat.cp_size() == 3);
  CHECK(rat.rank() == 2);
  CHECK(rat.degrees() == poly.degrees());
  CHECK(knots_as_double(rat, 0) == knots_as_double(poly, 0));
  const std::vector<double> before = control_points_of(poly);
  const std::vector<double> after = control_points_of(rat);
  REQUIRE(after.size() == before.size() / 2 * 3);
  for (std::size_t p = 0; p < before.size() / 2; ++p) {
    CHECK(after[3 * p] == before[2 * p]);
    CHECK(after[3 * p + 1] == before[2 * p + 1]);
    CHECK(after[3 * p + 2] == 1.0);
  }
  // A rational B-spline is returned as it is.
  CHECK(promote_to_rational<T>(rat).same_rep(rat));
}

TEST_CASE("cad::detail::column_groups: coordinates first, the weight on its own") {
  using cad::detail::column_groups;
  const auto polynomial = column_groups(false, 3);
  REQUIRE(polynomial.size() == 1);
  CHECK(polynomial[0].name == "coordinate");
  CHECK(polynomial[0].begin == 0);
  CHECK(polynomial[0].end == 3);
  const auto rational = column_groups(true, 4);
  REQUIRE(rational.size() == 2);
  CHECK(rational[0].name == "coordinate");
  CHECK(rational[0].begin == 0);
  CHECK(rational[0].end == 3);
  CHECK(rational[1].name == "weight");
  CHECK(rational[1].begin == 3);
  CHECK(rational[1].end == 4);
}

TEST_CASE_TEMPLATE("cad::detail::homogeneous_endpoint: the end coefficient, not projected", T,
                   float, double) {
  using cad::detail::homogeneous_endpoint;
  const bs::Bspline<T> rat = curve<T>(2, {0.5}, 3, 2, true);
  const std::vector<double> cp = control_points_of(rat);
  const std::size_t width = rat.cp_size();
  // An open knot vector interpolates its end control points, so the coefficient at an end is the
  // end row of the control points (homogeneous: weighted coordinates and the weight).
  const pantr::NdArray<T> start = homogeneous_endpoint<T>(rat, 0);
  const pantr::NdArray<T> end = homogeneous_endpoint<T>(rat, 1);
  REQUIRE(start.size() == width);
  REQUIRE(end.size() == width);
  for (std::size_t c = 0; c < width; ++c) {
    CHECK(static_cast<double>(start.flat()[c]) == cp[c]);
    CHECK(static_cast<double>(end.flat()[c]) == cp[cp.size() - width + c]);
  }
  // The projected end point has one component fewer and is divided by the weight.
  const pantr::NdArray<T> projected = bs::boundary_point<T>(rat, 0, 0);
  CHECK(projected.size() == width - 1);
  // A polynomial curve is read as it is.
  const bs::Bspline<T> poly = curve<T>(1, {}, 3, 3);
  CHECK(homogeneous_endpoint<T>(poly, 1).size() == 3);
  CHECK(message_of([&] {
          static_cast<void>(homogeneous_endpoint<T>(poly, 2));
        }).find("side must be 0 or 1, got 2.") != std::string::npos);
}

TEST_CASE("cad::detail::conservative_tier: the coarsest of the types compared") {
  using cad::detail::conservative_tier;
  constexpr double fine = static_cast<double>(pantr::tolerance::get_conservative<double>());
  constexpr double coarse = static_cast<double>(pantr::tolerance::get_conservative<float>());
  static_assert(conservative_tier<double>() == fine);
  static_assert(conservative_tier<float>() == coarse);
  static_assert(conservative_tier<double, float>() == coarse);
  static_assert(conservative_tier<float, double, double>() == coarse);
  static_assert(conservative_tier<double, double>() == fine);
  CHECK(coarse > fine);
}
