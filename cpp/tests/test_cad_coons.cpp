// `create_coons_surface` and `create_coons_volume` (include/pantr/cad/coons.hpp).
//
// Oracles. Everything is compared **exactly** in homogeneous coordinates where it can be: the
// exact rational evaluator of bspline_exact_eval.hpp (the basis from its recursive definition, over
// the stored control points) evaluates the Coons result and the data it was built from, so the
// only rounding in a comparison is the one the construction made in the control points of the
// result, and the tolerance below is a bound on that and on nothing else.
//  - A patch or a volume restricted to each of its boundaries equals the curve or face it was
//    given: S(0, v) = C_v0(v), S(1, v) = C_v1(v), S(u, 0) = C_u0(u), S(u, 1) = C_u1(u), and the
//    six faces of a volume. For a rational blend this holds for the homogeneous coordinates (the
//    weight included), and the projected value follows.
//  - Four straight edges give the bilinear surface, and a hexahedron's six bilinear faces the
//    trilinear volume: `create_bilinear` and `create_trilinear` are the oracle (a second route to
//    the same geometry, built from the corners alone).
//  - A quarter of a cylinder, from a rational arc, its translate and two lines, is the surface
//    `(cos t, sin t, v)`: a closed form, in projected coordinates.
//  - Every refusal is matched by message, and the numbers in a mismatch message are compared with
//    the gap, tolerance and scale computed in the test from the data (`4096 eps` as a product of
//    two `numeric_limits` values, the printed form by `snprintf`, a route that shares nothing
//    with `py_format_exp`).
//
// Tolerances. Inputs that need no `make_compat` (equal degrees and knots) make every term of the
// blend a copy of input control points, so the only roundings are the additions of the sum, which
// starts from zero: with `u = eps / 2`, the sum of three terms rounds twice, at magnitudes at most
// `2 M` and `3 M` for `M` the largest input control-point magnitude, so each control point of the
// patch is off by at most `5 u M = 2.5 eps M`; seven terms round six times, at magnitudes at most
// `2 M` to `7 M`, so `27 u M = 13.5 eps M`. A basis is non-negative and sums to one, so the exact
// value of the result is off by at most that. The tests below use `4 eps M` and `16 eps M`
// (a factor 1.6 and 1.2 of slack, for a second-order term). Where `make_compat` elevates or
// inserts knots (a change of basis, a chain of convex combinations) the bound is the conservative
// tier of the type times the larger of 1 and the largest control-point magnitude of the result,
// as in test_cad_operations.cpp.
#include <doctest/doctest.h>

#include "bspline_test_support.hpp"
#include "cad_exact_support.hpp"
#include "cad_test_support.hpp"

#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/convert.hpp>
#include <pantr/bspline/degree.hpp>
#include <pantr/bspline/knot_insertion.hpp>
#include <pantr/bspline/reparametrize.hpp>
#include <pantr/bspline/slice.hpp>
#include <pantr/cad/compat.hpp>
#include <pantr/cad/coons.hpp>
#include <pantr/cad/detail/validation.hpp>
#include <pantr/cad/primitives.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/tolerance.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace bs = pantr::bspline;
namespace cad = pantr::cad;
namespace oracle = bspline_oracle;
using bspline_test::cast_all;
using bspline_test::clamped;
using bspline_test::Dir;
using bspline_test::make_spline;
using bspline_test::tier_tolerance;
using cad_test::control_points_of;
using cad_test::curve_flat;
using cad_test::curve_of;
using cad_test::eps;
using cad_test::evaluate_at;
using cad_test::flatten;
using cad_test::fmt3;
using cad_test::grid;
using cad_test::has;
using cad_test::hom_line;
using cad_test::line;
using cad_test::max_difference;
using cad_test::Pt;
using cad_test::refusal;
using cad_test::Reported;
using cad_test::reported_numbers;
using cad_test::surface_flat;
using cad_test::tier_of;
using cad_test::value_at;
using cad_test::with_control_points;
using oracle::Rational;

namespace {

template <class T>
using Pair = std::array<bs::Bspline<T>, 2>;
template <class T>
using Curves = std::array<Pair<T>, 2>;
template <class T>
using Faces = std::array<Pair<T>, 3>;

// ------------------------------------------------------------------------------------ builders

/// The corner coefficients of a hexahedron, `p[(i * 2 + j) * 2 + k]` at side `i` along u, `j`
/// along v and `k` along w.
using Hexahedron = std::array<Pt, 8>;

Hexahedron cube(double s) {
  Hexahedron p{};
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      for (std::size_t k = 0; k < 2; ++k) {
        p[(i * 2 + j) * 2 + k] = {s * static_cast<double>(i), s * static_cast<double>(j),
                                  s * static_cast<double>(k)};
      }
    }
  }
  return p;
}

/// Flat `(2, 2, 3)` corner arrays of the six faces, in the order `u0, u1, v0, v1, w0, w1`: each
/// face is indexed by its own two parameters in increasing volume-axis order, `u_i` by `(v, w)`,
/// `v_j` by `(u, w)`, `w_k` by `(u, v)`.
std::array<std::vector<double>, 6> face_corners(const Hexahedron& p) {
  const auto at = [&](std::size_t i, std::size_t j, std::size_t k) -> const Pt& {
    return p[(i * 2 + j) * 2 + k];
  };
  std::array<std::vector<double>, 6> out;
  for (std::size_t side = 0; side < 2; ++side) {
    for (std::size_t a = 0; a < 2; ++a) {
      for (std::size_t b = 0; b < 2; ++b) {
        for (const double c : at(side, a, b)) {
          out[side].push_back(c);  // u_side: (v, w)
        }
        for (const double c : at(a, side, b)) {
          out[2 + side].push_back(c);  // v_side: (u, w)
        }
        for (const double c : at(a, b, side)) {
          out[4 + side].push_back(c);  // w_side: (u, v)
        }
      }
    }
  }
  return out;
}

template <class T>
bs::Bspline<T> bilinear_face(const std::vector<double>& corners) {
  const std::array<std::size_t, 3> shape = {2, 2, 3};
  return bs::astype<T>(
      cad::create_bilinear(std::span<const double>(corners), std::span<const std::size_t>(shape)));
}

/// A rational bilinear face: the corner coordinates multiplied by the corner weights, then the
/// weights.
template <class T>
bs::Bspline<T> rational_bilinear_face(const std::vector<double>& corners,
                                      const std::array<double, 4>& weights) {
  std::vector<double> flat;
  for (std::size_t c = 0; c < 4; ++c) {
    for (std::size_t i = 0; i < 3; ++i) {
      flat.push_back(corners[c * 3 + i] * weights[c]);
    }
    flat.push_back(weights[c]);
  }
  return surface_flat<T>(1, {}, 1, {}, flat, true);
}

template <class T>
Faces<T> as_pairs(const std::array<bs::Bspline<T>, 6>& f) {
  return Faces<T>{{{f[0], f[1]}, {f[2], f[3]}, {f[4], f[5]}}};
}

template <class T>
Faces<T> bilinear_faces(const Hexahedron& p) {
  const auto corners = face_corners(p);
  return as_pairs<T>({bilinear_face<T>(corners[0]), bilinear_face<T>(corners[1]),
                      bilinear_face<T>(corners[2]), bilinear_face<T>(corners[3]),
                      bilinear_face<T>(corners[4]), bilinear_face<T>(corners[5])});
}

/// Multiply the weight of every control point by `f`'s pattern: `w' = 1 + (w - 1) f`, the
/// homogeneous coordinates kept as they are.
template <class T>
bs::Bspline<T> soften_weights(const bs::Bspline<T>& b, double f) {
  std::vector<double> flat = control_points_of(b);
  const std::size_t width = b.cp_size();
  for (std::size_t p = 0; p < flat.size() / width; ++p) {
    double& w = flat[p * width + width - 1];
    w = 1.0 + (w - 1.0) * f;
  }
  return with_control_points(b, flat);
}

// ------------------------------------------------------------------------------------- oracles

/// The largest exact gap between `whole` restricted to `axis = side` and `piece` (promoted to
/// rational when `whole` is), over a grid of the other parameters (all `[0, 1]`).
template <class T>
double restriction_gap(const bs::Bspline<T>& whole, std::size_t axis, double side,
                       const bs::Bspline<T>& piece) {
  const bs::Bspline<T> other =
      whole.rational() ? cad::detail::promote_to_rational<T>(piece) : piece;
  const std::vector<double> params =
      piece.dim() == 1 ? std::vector<double>{0.0, 0.125, 0.3125, 0.5, 0.6875, 0.875, 1.0}
                       : std::vector<double>{0.0, 0.25, 0.375, 0.75, 1.0};
  const std::vector<std::vector<double>> axes(piece.dim(), params);
  const std::vector<double> rows = grid(axes);
  double worst = 0.0;
  for (std::size_t r = 0; r < rows.size() / piece.dim(); ++r) {
    std::vector<double> inner(rows.begin() + static_cast<std::ptrdiff_t>(r * piece.dim()),
                              rows.begin() + static_cast<std::ptrdiff_t>((r + 1) * piece.dim()));
    std::vector<double> full = inner;
    full.insert(full.begin() + static_cast<std::ptrdiff_t>(axis), side);
    worst = (std::max)(worst, max_difference(value_at(whole, full), value_at(other, inner)));
  }
  return worst;
}

/// The largest exact gap between the value of `b` over a grid of `[0, 1]^dim` and that of `ref`.
template <class T, class U>
double grid_gap(const bs::Bspline<T>& b, const bs::Bspline<U>& ref) {
  const std::vector<double> params = {0.0, 0.25, 0.375, 0.5, 0.875, 1.0};
  const std::vector<std::vector<double>> axes(b.dim(), params);
  const std::vector<double> rows = grid(axes);
  double worst = 0.0;
  for (std::size_t r = 0; r < rows.size() / b.dim(); ++r) {
    const std::vector<double> x(rows.begin() + static_cast<std::ptrdiff_t>(r * b.dim()),
                                rows.begin() + static_cast<std::ptrdiff_t>((r + 1) * b.dim()));
    worst = (std::max)(worst, max_difference(value_at(b, x), value_at(ref, x)));
  }
  return worst;
}

template <class T>
bs::Bspline<T> quarter_arc() {
  return bs::astype<T>(
      cad::create_circle(1.0, std::nullopt, std::array<double, 2>{0.0, std::numbers::pi / 2.0}));
}

/// The arc raised by 1 along z, in homogeneous coordinates (`w z = w`).
template <class T>
bs::Bspline<T> raised_arc() {
  const bs::Bspline<T> arc = quarter_arc<T>();
  std::vector<double> cp = control_points_of(arc);
  for (std::size_t p = 0; p < cp.size() / 4; ++p) {
    cp[p * 4 + 2] = cp[p * 4 + 3];
  }
  return with_control_points(arc, cp);
}

}  // namespace

// ===================================================================================== surface

TEST_CASE_TEMPLATE("create_coons_surface: four straight edges give the bilinear surface", T, float,
                   double) {
  // p[i][j] at side i along u and side j along v; dyadic, so exact in float.
  const Pt p00 = {0.0, 0.0, 0.0};
  const Pt p10 = {4.0, 0.5, -0.25};
  const Pt p01 = {-0.5, 3.0, 1.0};
  const Pt p11 = {3.5, 3.25, 2.0};
  const Curves<T> curves = {
      {{line<T>(p00, p01), line<T>(p10, p11)}, {line<T>(p00, p10), line<T>(p01, p11)}}};
  const bs::Bspline<T> s = cad::create_coons_surface<T>(curves);
  CHECK(s.dim() == 2);
  CHECK(s.rank() == 3);
  CHECK_FALSE(s.rational());
  CHECK(s.degrees() == std::vector<std::size_t>{1, 1});

  const std::vector<double> values = flatten({p00, p01, p10, p11});  // corners[i][j]
  const std::array<std::size_t, 3> shape = {2, 2, 3};
  const bs::Bspline<double> bilinear =
      cad::create_bilinear(std::span<const double>(values), std::span<const std::size_t>(shape));
  const double m = 4.0;  // the largest coordinate magnitude of the data
  CHECK(grid_gap(s, bilinear) <= 4.0 * eps<T>() * m);
}

TEST_CASE_TEMPLATE("create_coons_surface: a patch restricts to each of its boundary curves", T,
                   float, double) {
  // Four curves of different degrees and knots that meet at the corners.
  const Pt p00 = {0.0, 0.0, 0.0};
  const Pt p10 = {4.0, 0.5, -0.25};
  const Pt p01 = {-0.5, 3.0, 1.0};
  const Pt p11 = {3.5, 3.25, 2.0};
  const bs::Bspline<T> c_u0 = curve_of<T>(3, {}, {p00, {1.0, -1.0, 0.5}, {3.0, 1.5, -1.0}, p10});
  const bs::Bspline<T> c_u1 = curve_of<T>(2, {0.5}, {p01, {1.0, 4.0, 0.0}, {2.0, 2.0, 2.5}, p11});
  const bs::Bspline<T> c_v0 = curve_of<T>(2, {}, {p00, {-1.5, 1.0, 0.75}, p01});
  const bs::Bspline<T> c_v1 = line<T>(p10, p11);
  const bs::Bspline<T> s = cad::create_coons_surface<T>(Curves<T>{{{c_v0, c_v1}, {c_u0, c_u1}}});
  CHECK(s.dim() == 2);
  CHECK(s.degrees() == std::vector<std::size_t>{3, 2});

  // make_compat elevated and inserted, so the conservative tier.
  const double tol = tier_tolerance(s);
  CHECK(restriction_gap(s, 0, 0.0, c_v0) <= tol);
  CHECK(restriction_gap(s, 0, 1.0, c_v1) <= tol);
  CHECK(restriction_gap(s, 1, 0.0, c_u0) <= tol);
  CHECK(restriction_gap(s, 1, 1.0, c_u1) <= tol);
}

TEST_CASE_TEMPLATE("create_coons_surface: planar curves of rank 2 give a patch of rank 2", T, float,
                   double) {
  // Accepted by the reference implementation, and kept: a rank-2 model is a planar patch.
  const auto seg = [](double x0, double y0, double x1, double y1) {
    return curve_flat<T>(1, {}, {x0, y0, x1, y1});
  };
  const Curves<T> curves = {
      {{seg(0, 0, 0, 2), seg(3, 0, 3, 2)}, {seg(0, 0, 3, 0), seg(0, 2, 3, 2)}}};
  const bs::Bspline<T> s = cad::create_coons_surface<T>(curves);
  CHECK(s.rank() == 2);
  CHECK(s.cp_size() == 2);
  CHECK(restriction_gap(s, 0, 0.0, curves[0][0]) <= 4.0 * eps<T>() * 3.0);
  CHECK(restriction_gap(s, 1, 1.0, curves[1][1]) <= 4.0 * eps<T>() * 3.0);
  // The interior is the rectangle's bilinear map.
  const std::vector<double> at = {0.25, 0.5};
  const std::vector<Rational> v = value_at(s, at);
  CHECK(static_cast<double>(v[0]) == doctest::Approx(0.75).epsilon(4.0 * eps<T>()));
  CHECK(static_cast<double>(v[1]) == doctest::Approx(1.0).epsilon(4.0 * eps<T>()));
}

TEST_CASE_TEMPLATE("create_coons_surface: a quarter of a cylinder from a rational arc", T, float,
                   double) {
  // C_u0 is the unit quarter circle, C_u1 its translate by 1 along z, C_v0 and C_v1 the vertical
  // lines at its ends, which the arcs' homogeneous end points meet (weight 1 at both ends).
  const bs::Bspline<T> arc = quarter_arc<T>();
  const bs::Bspline<T> top = raised_arc<T>();
  const bs::Bspline<T> c_v0 = line<T>({1, 0, 0}, {1, 0, 1});
  const bs::Bspline<T> c_v1 = line<T>({0, 1, 0}, {0, 1, 1});
  const bs::Bspline<T> s = cad::create_coons_surface<T>(Curves<T>{{{c_v0, c_v1}, {arc, top}}});
  CHECK(s.rational());
  CHECK(s.rank() == 3);

  // The boundary property holds in homogeneous coordinates, weights included.
  const double tol = tier_tolerance(s);
  CHECK(restriction_gap(s, 0, 0.0, c_v0) <= tol);
  CHECK(restriction_gap(s, 0, 1.0, c_v1) <= tol);
  CHECK(restriction_gap(s, 1, 0.0, arc) <= tol);
  CHECK(restriction_gap(s, 1, 1.0, top) <= tol);

  // The surface is the cylinder (cos t, sin t, v): radius 1 from the z axis, height v.
  const std::vector<double> pts = grid({cad_test::linspace(0.0, 1.0, 9), {0.0, 0.25, 0.5, 1.0}});
  const std::vector<double> got = evaluate_at(s, pts);
  for (std::size_t i = 0; i < pts.size() / 2; ++i) {
    CHECK(std::abs(std::hypot(got[3 * i], got[3 * i + 1]) - 1.0) <= tol);
    CHECK(std::abs(got[3 * i + 2] - pts[2 * i + 1]) <= tol);
  }
}

TEST_CASE_TEMPLATE("create_coons_surface: one rational curve among polynomial ones", T, float,
                   double) {
  // C_v0 is the arc from (1, 0, 0) to (0, 1, 0); the other three are lines (promoted with unit
  // weights, which the arc's end weights are).
  const bs::Bspline<T> arc = quarter_arc<T>();
  const bs::Bspline<T> c_v1 = line<T>({2, 0, 0}, {0, 2, 0});
  const bs::Bspline<T> c_u0 = line<T>({1, 0, 0}, {2, 0, 0});
  const bs::Bspline<T> c_u1 = line<T>({0, 1, 0}, {0, 2, 0});
  const bs::Bspline<T> s = cad::create_coons_surface<T>(Curves<T>{{{arc, c_v1}, {c_u0, c_u1}}});
  CHECK(s.rational());
  const double tol = tier_tolerance(s);
  CHECK(restriction_gap(s, 0, 0.0, arc) <= tol);
  CHECK(restriction_gap(s, 0, 1.0, c_v1) <= tol);
  CHECK(restriction_gap(s, 1, 0.0, c_u0) <= tol);
  CHECK(restriction_gap(s, 1, 1.0, c_u1) <= tol);
  // The arc side is on the unit circle, projected.
  const std::vector<double> at = {0.0, 0.0, 0.0, 0.25, 0.0, 0.5, 0.0, 0.75, 0.0, 1.0};
  const std::vector<double> got = evaluate_at(s, at);
  for (std::size_t i = 0; i < 5; ++i) {
    CHECK(std::abs(std::hypot(got[3 * i], got[3 * i + 1]) - 1.0) <= tol);
  }
}

TEST_CASE_TEMPLATE("create_coons_surface: a shared corner weight other than one is carried", T,
                   float, double) {
  // The unit square with every corner weight 2: homogeneous coordinates doubled, weights 2. A
  // consistent rational input; the blend is homogeneous, so each boundary is reproduced.
  const auto h = [](double x, double y) { return std::array<double, 4>{2 * x, 2 * y, 0.0, 2.0}; };
  const Curves<T> curves = {{{hom_line<T>(h(0, 0), h(0, 1)), hom_line<T>(h(1, 0), h(1, 1))},
                             {hom_line<T>(h(0, 0), h(1, 0)), hom_line<T>(h(0, 1), h(1, 1))}}};
  const bs::Bspline<T> s = cad::create_coons_surface<T>(curves);
  CHECK(s.rational());
  const double tol = 4.0 * eps<T>() * 2.0;
  CHECK(restriction_gap(s, 0, 0.0, curves[0][0]) <= tol);
  CHECK(restriction_gap(s, 0, 1.0, curves[0][1]) <= tol);
  CHECK(restriction_gap(s, 1, 0.0, curves[1][0]) <= tol);
  CHECK(restriction_gap(s, 1, 1.0, curves[1][1]) <= tol);
  // Rescaling every curve's homogeneous coefficients by one constant describes the same patch.
  const auto scaled = [&](const bs::Bspline<T>& c) {
    std::vector<double> cp = control_points_of(c);
    for (double& v : cp) {
      v *= 4.0;
    }
    return with_control_points(c, cp);
  };
  const Curves<T> big = {
      {{scaled(curves[0][0]), scaled(curves[0][1])}, {scaled(curves[1][0]), scaled(curves[1][1])}}};
  CHECK_NOTHROW(static_cast<void>(cad::create_coons_surface<T>(big)));
  // One weight of 1 where the others say 2 is a disagreement.
  Curves<T> off = curves;
  off[0][0] = hom_line<T>({0.0, 0.0, 0.0, 1.0}, h(0, 1));
  const std::string message =
      refusal([&] { static_cast<void>(cad::create_coons_surface<T>(off)); });
  CHECK_MESSAGE(has(message,
                    "Corner (0,0) mismatch: C_u0 gives [0.0, 0.0, 0.0, 2.0], C_v0 gives "
                    "[0.0, 0.0, 0.0, 1.0]"),
                message);
  const Reported r = reported_numbers(message);
  CHECK(r.population == "corner weight");
  CHECK(r.gap == doctest::Approx(1.0));
  CHECK(r.scale == doctest::Approx(2.0));
  CHECK(r.tol == doctest::Approx(tier_of<T>() * 2.0).epsilon(1e-3));
}

TEST_CASE_TEMPLATE("create_coons_surface: every curve must be one-dimensional", T, float, double) {
  const bs::Bspline<T> ok = line<T>({0, 0, 0}, {1, 0, 0});
  const bs::Bspline<T> surf = surface_flat<T>(1, {}, 1, {}, {0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0});
  const auto message = [&](std::size_t pair, std::size_t which) {
    Curves<T> curves = {{{ok, ok}, {ok, ok}}};
    curves[pair][which] = surf;
    return refusal([&] { static_cast<void>(cad::create_coons_surface<T>(curves)); });
  };
  for (std::size_t pair = 0; pair < 2; ++pair) {
    for (std::size_t which = 0; which < 2; ++which) {
      CHECK(message(pair, which) == "All curves must be 1D, got dim=2.");
    }
  }
}

namespace {

/// The unit-square-like patch whose `C_v0` starts displaced by `shift` in y: corner (0, 0) is then
/// read as `(0.5, 0, 0)` by `C_u0` and `(0.5, shift, 0)` by `C_v0`. The largest coordinate
/// over the eight readings is 2.
template <class T>
Curves<T> displaced_patch(double shift) {
  return Curves<T>{{{line<T>({0.5, shift, 0}, {0.5, 1, 0}), line<T>({2, 0, 0}, {2, 1, 0})},
                    {line<T>({0.5, 0, 0}, {2, 0, 0}), line<T>({0.5, 1, 0}, {2, 1, 0})}}};
}

}  // namespace

TEST_CASE_TEMPLATE("create_coons_surface: a corner mismatch is refused with the numbers it used", T,
                   float, double) {
  const double shift = 0.1;  // as T: float(0.1), not dyadic, so the array spelling is exercised
  const std::string message =
      refusal([&] { static_cast<void>(cad::create_coons_surface<T>(displaced_patch<T>(shift))); });
  const double got_shift = static_cast<double>(static_cast<T>(shift));
  const double tol = tier_of<T>() * 2.0;
  const std::string expect =
      "Corner (0,0) mismatch: C_u0 gives [0.5, 0.0, 0.0], C_v0 gives [0.5, 0.1, 0.0] (gap " +
      fmt3(got_shift) + ", above " + fmt3(tol) + " at corner coordinate scale " + fmt3(2.0) + ").";
  CHECK(message == expect);
}

TEST_CASE_TEMPLATE("create_coons_surface: the corner tolerance is the tier times the scale", T,
                   float, double) {
  // The gap is exact (dyadic) and the scale is 2, so the verdict sits exactly at the tolerance:
  // `gap > tol` refuses, so a gap equal to the tolerance is accepted and twice it is refused.
  const double tol = tier_of<T>() * 2.0;
  CHECK_NOTHROW(static_cast<void>(cad::create_coons_surface<T>(displaced_patch<T>(tol))));
  CHECK(has(refusal([&] {
              static_cast<void>(cad::create_coons_surface<T>(displaced_patch<T>(2.0 * tol)));
            }),
            "Corner (0,0) mismatch"));
}

TEST_CASE("create_coons_surface: the tier of the coarsest input decides, as a binding passes it") {
  // A double patch whose corner is off by 1e-5: refused at the double tier, accepted at the
  // float tier a binding passes when one of the inputs was a float.
  const double shift = 1.0e-5;
  const Curves<double> curves = displaced_patch<double>(shift);
  CHECK(has(refusal([&] { static_cast<void>(cad::create_coons_surface<double>(curves)); }),
            "Corner (0,0) mismatch"));
  const double float_tier = tier_of<float>();
  REQUIRE(shift < float_tier * 2.0);
  CHECK_NOTHROW(static_cast<void>(cad::detail::create_coons_surface<double>(curves, float_tier)));
  // The tolerance in the message is the tier handed in.
  const std::string message = refusal([&] {
    static_cast<void>(
        cad::detail::create_coons_surface<double>(displaced_patch<double>(0.5), float_tier));
  });
  CHECK(reported_numbers(message).tol == doctest::Approx(float_tier * 2.0).epsilon(1e-3));
}

TEST_CASE("create_coons_surface: the float tier is the only one a float corner can clear") {
  // A float patch whose corner is one float ulp off at magnitude 1, the closest two float readings
  // of one corner can be. Graded at the float tier it is accepted; graded at the double tier (what
  // a tier read at the wrong type would give) it is refused, by the factor
  // eps32 / (4096 eps64) = 2^17 = 131072 the derivation in `detail::verify_corners_2d` states.
  const float ulp = std::numeric_limits<float>::epsilon();
  const Curves<float> curves = {
      {{line<float>({static_cast<double>(1.0F + ulp), 0, 0}, {1, 1, 0}),
        line<float>({2, 0, 0}, {2, 1, 0})},
       {line<float>({1, 0, 0}, {2, 0, 0}), line<float>({1, 1, 0}, {2, 1, 0})}}};
  CHECK_NOTHROW(static_cast<void>(cad::create_coons_surface<float>(curves)));
  const std::string message = refusal([&] {
    static_cast<void>(cad::detail::create_coons_surface<float>(curves, tier_of<double>()));
  });
  CHECK_MESSAGE(has(message, "Corner (0,0) mismatch"), message);
  CHECK(static_cast<double>(ulp) / tier_of<double>() == 131072.0);
  CHECK(reported_numbers(message).gap == doctest::Approx(static_cast<double>(ulp)).epsilon(1e-3));
}

TEST_CASE_TEMPLATE("create_coons_surface: a weight-only corner mismatch names the weight group", T,
                   float, double) {
  // Unit square, rational, all weights 1 except C_v0's start (the origin: its homogeneous
  // coordinates are zero at any weight, so only the weight column disagrees).
  const double g = 4.0 * tier_of<T>();
  const auto h = [](double x, double y) { return std::array<double, 4>{x, y, 0.0, 1.0}; };
  const Curves<T> curves = {
      {{hom_line<T>({0.0, 0.0, 0.0, 1.0 + g}, h(0, 1)), hom_line<T>(h(1, 0), h(1, 1))},
       {hom_line<T>(h(0, 0), h(1, 0)), hom_line<T>(h(0, 1), h(1, 1))}}};
  const std::string message =
      refusal([&] { static_cast<void>(cad::create_coons_surface<T>(curves)); });
  REQUIRE_MESSAGE(has(message, "Corner (0,0) mismatch"), message);
  const Reported r = reported_numbers(message);
  CHECK(r.population == "corner weight");
  CHECK(r.gap == doctest::Approx(g).epsilon(1e-2));
  // The weight population's own maximum, which the displaced weight itself sets.
  CHECK(r.scale == doctest::Approx(1.0 + g).epsilon(1e-3));
  CHECK(r.tol == doctest::Approx(tier_of<T>() * (1.0 + g)).epsilon(1e-3));
}

TEST_CASE_TEMPLATE("create_coons_surface: curves of different ranks are refused", T, float,
                   double) {
  const auto seg2 = [](double x0, double y0, double x1, double y1) {
    return curve_flat<T>(1, {}, {x0, y0, x1, y1});
  };
  const bs::Bspline<T> a2 = seg2(0, 0, 1, 0);
  const bs::Bspline<T> a3 = line<T>({0, 0, 0}, {1, 0, 0});
  // The two u-curves differ.
  std::string message = refusal([&] {
    static_cast<void>(cad::create_coons_surface<T>(
        Curves<T>{{{line<T>({0, 0, 0}, {0, 1, 0}), line<T>({1, 0, 0}, {1, 1, 0})}, {a2, a3}}}));
  });
  CHECK(has(message,
            "create_coons_surface requires curves of one rank, got control points of 2 and "
            "3 coordinates (weights included) on C_u0 and C_u1."));
  // The u-curves agree and a v-curve differs: refused at the corner it meets.
  message = refusal([&] {
    static_cast<void>(cad::create_coons_surface<T>(Curves<T>{
        {{seg2(0, 0, 0, 1), line<T>({1, 0, 0}, {1, 1, 0})}, {a3, line<T>({0, 1, 0}, {1, 1, 0})}}}));
  });
  CHECK(has(message,
            "Corner (0,0): C_u0 carries 3 coordinates per control point and C_v0 2; "
            "the curves must have one rank (weights included)."));
}

TEST_CASE_TEMPLATE("create_coons_surface: knots a rounding apart leave unequal sizes, refused", T,
                   float, double) {
  // Inherited from make_compat (see test_cad_operations.cpp): an interior knot a quarter of the
  // space tolerance above another counterpart makes make_compat return B-splines with unequal
  // numbers of functions, which `create_ruled` refuses (the reference implementation failed with
  // a NumPy broadcast error at the same step).
  const bs::Bspline<T> a = curve_of<T>(2, {0.5}, {{0, 0, 0}, {1, 1, 0}, {2, 1, 0}, {3, 0, 0}});
  const double tol = a.space().spaces()[0].tolerance();
  const double shifted = static_cast<double>(static_cast<T>(0.5 + 0.25 * tol));
  REQUIRE(shifted > 0.5);
  const bs::Bspline<T> b = curve_of<T>(2, {shifted}, {{0, 2, 0}, {1, 3, 0}, {2, 3, 0}, {3, 2, 0}});
  const Curves<T> curves = {
      {{line<T>({0, 0, 0}, {0, 2, 0}), line<T>({3, 0, 0}, {3, 2, 0})}, {a, b}}};
  CHECK(has(refusal([&] { static_cast<void>(cad::create_coons_surface<T>(curves)); }),
            "ruled requires B-splines with equal numbers of control points after making them "
            "compatible, got"));
}

TEST_CASE("create_coons_surface: a NaN corner is not refused, as in the reference implementation") {
  // Unusual input, pinned as inherited (not endorsed): the corner comparison refuses `gap > tol`,
  // which a NaN gap does not satisfy, and the NaN scale makes the tolerance NaN too. The result
  // carries the NaN; nothing crashes.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Curves<double> curves = displaced_patch<double>(0.0);
  curves[0][0] = line<double>({0.5, nan, 0}, {0.5, 1, 0});
  bs::Bspline<double> s = cad::create_coons_surface<double>(curves);
  bool any_nan = false;
  for (const double v : s.control_points()) {
    any_nan = any_nan || std::isnan(v);
  }
  CHECK(any_nan);
}

// ====================================================================================== volume

TEST_CASE_TEMPLATE("create_coons_volume: a hexahedron from its six bilinear faces is trilinear", T,
                   float, double) {
  // Eight generic corners (dyadic), faces consistent by construction.
  Hexahedron p = cube(1.0);
  p[0] = {0.0, 0.0, 0.0};
  p[1] = {0.125, -0.25, 3.0};
  p[2] = {-0.375, 2.0, 0.25};
  p[3] = {0.5, 2.25, 2.75};
  p[4] = {4.0, 0.25, -0.125};
  p[5] = {3.5, 0.5, 3.125};
  p[6] = {4.25, 1.75, 0.5};
  p[7] = {3.75, 2.125, 2.875};
  const Faces<T> faces = bilinear_faces<T>(p);
  const bs::Bspline<T> v = cad::create_coons_volume<T>(faces);
  CHECK(v.dim() == 3);
  CHECK(v.degrees() == std::vector<std::size_t>{1, 1, 1});
  CHECK_FALSE(v.rational());

  const std::vector<double> values = flatten(std::vector<Pt>(p.begin(), p.end()));
  const std::array<std::size_t, 4> shape = {2, 2, 2, 3};
  const bs::Bspline<double> trilinear =
      cad::create_trilinear(std::span<const double>(values), std::span<const std::size_t>(shape));
  const double m = 4.25;
  CHECK(grid_gap(v, trilinear) <= 16.0 * eps<T>() * m);
  for (std::size_t axis = 0; axis < 3; ++axis) {
    for (std::size_t side = 0; side < 2; ++side) {
      CHECK(restriction_gap(v, axis, static_cast<double>(side), faces[axis][side]) <=
            16.0 * eps<T>() * m);
    }
  }
}

namespace {

/// The six boundary faces of one volume: consistent along every shared edge by construction,
/// with different degrees and knots from one pair to the next.
template <class T>
Faces<T> boundary_faces(const bs::Bspline<T>& v) {
  const auto face = [&](long long axis, long long side) { return bs::boundary<T>(v, axis, side); };
  return Faces<T>{{{face(0, 0), face(0, 1)}, {face(1, 0), face(1, 1)}, {face(2, 0), face(2, 1)}}};
}

template <class T>
void check_all_faces(const bs::Bspline<T>& v, const Faces<T>& faces, double tol) {
  for (std::size_t axis = 0; axis < 3; ++axis) {
    for (std::size_t side = 0; side < 2; ++side) {
      CHECK(restriction_gap(v, axis, static_cast<double>(side), faces[axis][side]) <= tol);
    }
  }
}

}  // namespace

TEST_CASE_TEMPLATE("create_coons_volume: curved faces of mixed degrees interpolate all six", T,
                   float, double) {
  const bs::Bspline<T> source = make_spline<T>(
      {Dir{clamped(2, {0.5}), 2}, Dir{clamped(2, {}), 2}, Dir{clamped(1, {0.25}), 1}}, 3, false, 7);
  const Faces<T> faces = boundary_faces<T>(source);
  const bs::Bspline<T> v = cad::create_coons_volume<T>(faces);
  CHECK(v.dim() == 3);
  CHECK(v.degrees() == std::vector<std::size_t>{2, 2, 1});
  check_all_faces(v, faces, tier_tolerance(v));
}

TEST_CASE_TEMPLATE("create_coons_volume: rational faces blend homogeneously and interpolate", T,
                   float, double) {
  // Weights in [1, 1.125]: the blend's control weights are then at least 4 - 3 * 1.125 > 0, so the
  // weight test passes, and the homogeneous boundary property can be tested.
  const bs::Bspline<T> source = soften_weights<T>(
      make_spline<T>(
          {Dir{clamped(2, {0.5}), 2}, Dir{clamped(2, {}), 2}, Dir{clamped(1, {0.25}), 1}}, 3, true,
          5),
      0.125);
  const Faces<T> faces = boundary_faces<T>(source);
  const bs::Bspline<T> v = cad::create_coons_volume<T>(faces);
  CHECK(v.rational());
  CHECK(v.cp_size() == 4);
  check_all_faces(v, faces, tier_tolerance(v));
}

TEST_CASE_TEMPLATE("create_coons_volume: a polynomial face among rational ones takes the same path",
                   T, float, double) {
  // Faces of a polynomial volume, three of them carried rationally with unit weights: the result is
  // rational, still interpolates every face, and (its weights all 1, as 1 + 1 + 1 - 1 - 1 - 1 + 1)
  // has the coordinates of the all-polynomial blend.
  const bs::Bspline<T> source = make_spline<T>(
      {Dir{clamped(1, {}), 1}, Dir{clamped(2, {0.5}), 2}, Dir{clamped(1, {}), 1}}, 3, false, 3);
  const Faces<T> poly = boundary_faces<T>(source);
  Faces<T> mixed = poly;
  mixed[0][1] = cad::detail::promote_to_rational<T>(poly[0][1]);
  mixed[1][0] = cad::detail::promote_to_rational<T>(poly[1][0]);
  mixed[2][1] = cad::detail::promote_to_rational<T>(poly[2][1]);
  const bs::Bspline<T> v_poly = cad::create_coons_volume<T>(poly);
  const bs::Bspline<T> v_mixed = cad::create_coons_volume<T>(mixed);
  CHECK_FALSE(v_poly.rational());
  REQUIRE(v_mixed.rational());
  check_all_faces(v_mixed, mixed, tier_tolerance(v_mixed));

  const std::vector<double> a = control_points_of(v_poly);
  const std::vector<double> b = control_points_of(v_mixed);
  REQUIRE(b.size() / 4 == a.size() / 3);
  const double tol = tier_tolerance(v_mixed);
  for (std::size_t p = 0; p < a.size() / 3; ++p) {
    for (std::size_t c = 0; c < 3; ++c) {
      CHECK(std::abs(a[3 * p + c] - b[4 * p + c]) <= tol);
    }
    CHECK(std::abs(b[4 * p + 3] - 1.0) <= tol);
  }
}

TEST_CASE_TEMPLATE("create_coons_volume: an edge reached at different degrees by its faces is ok",
                   T, float, double) {
  // face_w0 elevated in u: the shared edges with the v faces now meet at degree 2 on one side and
  // 1 on the other, which the comparison makes compatible; the geometry is unchanged.
  Faces<T> faces = bilinear_faces<T>(cube(1.0));
  const std::array<long long, 2> inc = {1, 0};
  faces[2][0] = bs::elevate_degree<T>(faces[2][0], std::span<const long long>(inc));
  const bs::Bspline<T> v = cad::create_coons_volume<T>(faces);
  check_all_faces(v, faces, tier_tolerance(v));
}

TEST_CASE_TEMPLATE("coons_edge_pairs: the twelve edges are read from the two faces that carry them",
                   T, float, double) {
  // The table of the labelling convention: an edge free in one direction is fixed in the other
  // two and is carried by the face pair of each. Expected here from that convention alone.
  Hexahedron p = cube(1.0);
  p[0] = {0.0, 0.0, 0.0};
  p[1] = {0.125, -0.25, 3.0};
  p[2] = {-0.375, 2.0, 0.25};
  p[3] = {0.5, 2.25, 2.75};
  p[4] = {4.0, 0.25, -0.125};
  p[5] = {3.5, 0.5, 3.125};
  p[6] = {4.25, 1.75, 0.5};
  p[7] = {3.75, 2.125, 2.875};
  const Faces<T> faces = bilinear_faces<T>(p);
  const std::vector<cad::detail::EdgeReadings<T>> readings =
      cad::detail::coons_edge_pairs<T>(faces);
  REQUIRE(readings.size() == 12);

  struct Expect {
    std::string label, face_a, face_b, low, high;
  };
  std::vector<Expect> expected;
  const auto s = [](int v) { return std::to_string(v); };
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      expected.push_back({"w_u" + s(a) + "_v" + s(b), "face_u" + s(a), "face_v" + s(b),
                          "(" + s(a) + "," + s(b) + ",0)", "(" + s(a) + "," + s(b) + ",1)"});
    }
  }
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      expected.push_back({"v_u" + s(a) + "_w" + s(b), "face_u" + s(a), "face_w" + s(b),
                          "(" + s(a) + ",0," + s(b) + ")", "(" + s(a) + ",1," + s(b) + ")"});
    }
  }
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      expected.push_back({"u_v" + s(a) + "_w" + s(b), "face_v" + s(a), "face_w" + s(b),
                          "(0," + s(a) + "," + s(b) + ")", "(1," + s(a) + "," + s(b) + ")"});
    }
  }
  // Ordered by free direction w, then v, then u.
  for (std::size_t e = 0; e < 12; ++e) {
    const auto& r = readings[e];
    CHECK(r.label == expected[e].label);
    CHECK(r.face_a == expected[e].face_a);
    CHECK(r.face_b == expected[e].face_b);
    CHECK(r.face_a != r.face_b);
    CHECK(r.corners[0] == expected[e].low);
    CHECK(r.corners[1] == expected[e].high);
    // Both readings really are that edge: their ends are the corners the label names.
    for (const bs::Bspline<T>* curve : {&r.curve_a, &r.curve_b}) {
      for (long long end = 0; end < 2; ++end) {
        const std::string& corner = r.corners[static_cast<std::size_t>(end)];
        const auto i = static_cast<std::size_t>(corner[1] - '0');
        const auto j = static_cast<std::size_t>(corner[3] - '0');
        const auto k = static_cast<std::size_t>(corner[5] - '0');
        const pantr::NdArray<T> point = bs::boundary_point<T>(*curve, 0, end);
        const Pt& want = p[(i * 2 + j) * 2 + k];
        for (std::size_t c = 0; c < 3; ++c) {
          CHECK(static_cast<double>(point.flat()[c]) == want[c]);
        }
      }
    }
  }
}

namespace {

template <class T>
std::string volume_refusal(const Faces<T>& faces) {
  return refusal([&] { static_cast<void>(cad::create_coons_volume<T>(faces)); });
}

/// The cube's faces with `face_w0`'s `(u = 1, v = 1)` corner lifted along z by `lift`.
template <class T>
Faces<T> lifted_w0_corner(double lift) {
  auto corners = face_corners(cube(1.0));
  corners[4][(1 * 2 + 1) * 3 + 2] += lift;
  return as_pairs<T>({bilinear_face<T>(corners[0]), bilinear_face<T>(corners[1]),
                      bilinear_face<T>(corners[2]), bilinear_face<T>(corners[3]),
                      bilinear_face<T>(corners[4]), bilinear_face<T>(corners[5])});
}

}  // namespace

TEST_CASE_TEMPLATE("create_coons_volume: every face must be a surface", T, float, double) {
  Faces<T> faces = bilinear_faces<T>(cube(1.0));
  const bs::Bspline<T> curve = line<T>({0, 0, 0}, {1, 0, 0});
  for (std::size_t pair = 0; pair < 3; ++pair) {
    for (std::size_t which = 0; which < 2; ++which) {
      Faces<T> bad = faces;
      bad[pair][which] = curve;
      CHECK(volume_refusal<T>(bad) == "All faces must be 2D B-splines, got dim=1.");
    }
  }
}

TEST_CASE_TEMPLATE("create_coons_volume: a corner mismatch is refused with the numbers it used", T,
                   float, double) {
  // The three faces meeting at (1, 1, 0) are face_u1, face_v1 and face_w0; the w-face is lifted,
  // so the first disagreeing pair in the order (u, v), (u, w), (v, w) is (face_u1, face_w0).
  const double lift = 0.25;
  const std::string message = volume_refusal<T>(lifted_w0_corner<T>(lift));
  const std::string expect =
      "Corner (1,1,0) mismatch: face_u1 gives [1.0, 1.0, 0.0], face_w0 gives "
      "[1.0, 1.0, 0.25] (gap " +
      fmt3(lift) + ", above " + fmt3(tier_of<T>() * 1.0) + " at corner scale " + fmt3(1.0) + ").";
  CHECK(message == expect);
}

TEST_CASE_TEMPLATE("create_coons_volume: a u or v face disagreeing at a corner is refused too", T,
                   float, double) {
  const double lift = 0.25;
  for (const std::size_t face : {std::size_t{1}, std::size_t{3}}) {  // face_u1, face_v1
    auto corners = face_corners(cube(1.0));
    // u1 is indexed (v, w): its (v = 1, w = 0) corner is [1][0]; v1 is indexed (u, w): (u = 1, w =
    // 0).
    corners[face][(1 * 2 + 0) * 3 + 2] += lift;
    const Faces<T> faces = as_pairs<T>(
        {bilinear_face<T>(corners[0]), bilinear_face<T>(corners[1]), bilinear_face<T>(corners[2]),
         bilinear_face<T>(corners[3]), bilinear_face<T>(corners[4]), bilinear_face<T>(corners[5])});
    const std::string message = volume_refusal<T>(faces);
    CHECK_MESSAGE(has(message, "Corner (1,1,0) mismatch"), message);
    CHECK(reported_numbers(message).gap == doctest::Approx(lift));
  }
}

TEST_CASE("create_coons_volume: the tier handed to the detail overload is the one used") {
  // A double cube with a corner lifted by 1e-5: refused at the double tier, accepted at the float
  // tier a binding passes when one of the faces was a float.
  const Faces<double> faces = lifted_w0_corner<double>(1.0e-5);
  CHECK(has(volume_refusal<double>(faces), "Corner (1,1,0) mismatch"));
  const double float_tier = tier_of<float>();
  CHECK_NOTHROW(static_cast<void>(cad::detail::create_coons_volume<double>(faces, float_tier)));
  const std::string message = refusal([&] {
    static_cast<void>(
        cad::detail::create_coons_volume<double>(lifted_w0_corner<double>(0.5), float_tier));
  });
  CHECK(reported_numbers(message).tol == doctest::Approx(float_tier).epsilon(1e-3));
}

TEST_CASE_TEMPLATE("create_coons_volume: the corner scale spans every reading, not the corner", T,
                   float, double) {
  // One corner of the hexahedron is far away (100 along z), which sets the largest coordinate over
  // all twenty-four readings; the corner under test, (1, 1, 0), is O(1). A lift of ten times the
  // tier is refused against its own scale (1) and accepted against the model's (100).
  const double lift = 10.0 * tier_of<T>();
  const auto faces_with_far_corner = [&](bool distant) {
    Hexahedron p = cube(1.0);
    if (distant) {
      p[1] = {0.0, 0.0, 100.0};  // corner (0, 0, 1)
    }
    auto corners = face_corners(p);
    corners[4][(1 * 2 + 1) * 3 + 2] += lift;  // face_w0's (u = 1, v = 1) corner
    return as_pairs<T>({bilinear_face<T>(corners[0]), bilinear_face<T>(corners[1]),
                        bilinear_face<T>(corners[2]), bilinear_face<T>(corners[3]),
                        bilinear_face<T>(corners[4]), bilinear_face<T>(corners[5])});
  };
  CHECK(has(volume_refusal<T>(faces_with_far_corner(false)), "Corner (1,1,0) mismatch"));
  // Every comparison of the volume (the corner and the edge ends) is graded against the model.
  CHECK_NOTHROW(static_cast<void>(cad::create_coons_volume<T>(faces_with_far_corner(true))));
}

namespace {

/// A Bezier surface of degree `(2, 1)` from a `(3, 2, 3)` net.
template <class T>
bs::Bspline<T> quadratic_linear(const std::vector<double>& net) {
  return surface_flat<T>(2, {}, 1, {}, net);
}

}  // namespace

TEST_CASE_TEMPLATE("create_coons_volume: an edge interior disagreement is refused at the edge", T,
                   float, double) {
  // face_w0 becomes a degree-(2, 1) patch whose middle control point at v = 1 is lifted: all eight
  // corners agree, and only an edge comparison sees it. The edge it shares with face_v1 (at
  // v = 1, w = 0, free in u) is read from face_v1 elevated to degree 2 (middle control point
  // (0.5, 1, 0)) and from face_w0 ((0.5, 1, 0.25)).
  const double bulge = 0.25;
  auto faces = bilinear_faces<T>(cube(1.0));
  faces[2][0] = quadratic_linear<T>({0, 0, 0, 0, 1, 0,          //
                                     0.5, 0, 0, 0.5, 1, bulge,  //
                                     1, 0, 0, 1, 1, 0});
  const std::string message = volume_refusal<T>(faces);
  const std::string expect =
      "Edge u_v1_w0 mismatch at control point 1: face_v1 gives [0.5, 1.0, 0.0], face_w0 gives "
      "[0.5, 1.0, 0.25], and they disagree along the edge they share (gap " +
      fmt3(bulge) + ", above " + fmt3(tier_of<T>() * 1.0) + " at edge coordinate scale " +
      fmt3(1.0) + ").";
  CHECK(message == expect);
}

TEST_CASE_TEMPLATE("create_coons_volume: the edge scale spans every edge, not the one compared", T,
                   float, double) {
  // face_v1 and face_w0 carry the same far interior control point (0.5, 1, reach) on the edge they
  // share (consistent: it widens the scale only), and face_w0's middle control point at v = 0 is
  // lifted by ten times the tier, a disagreement with face_v0 along edge u_v0_w0 whose own
  // magnitude is 1. Against the scale of the whole model (reach) it is within the tolerance;
  // against its own it would not be.
  const double gap = 10.0 * tier_of<T>();
  const auto faces_with_reach = [&](double reach) {
    auto faces = bilinear_faces<T>(cube(1.0));
    faces[1][1] = surface_flat<T>(2, {}, 1, {},
                                  {0, 1, 0, 0, 1, 1,          //
                                   0.5, 1, reach, 0.5, 1, 1,  //
                                   1, 1, 0, 1, 1, 1});
    faces[2][0] = quadratic_linear<T>({0, 0, 0, 0, 1, 0,            //
                                       0.5, 0, gap, 0.5, 1, reach,  //
                                       1, 0, 0, 1, 1, 0});
    return faces;
  };
  CHECK_NOTHROW(static_cast<void>(cad::create_coons_volume<T>(faces_with_reach(100.0))));
  const std::string message = volume_refusal<T>(faces_with_reach(1.0));
  CHECK_MESSAGE(has(message, "Edge u_v0_w0 mismatch at control point 1"), message);
  CHECK(reported_numbers(message).scale == doctest::Approx(1.0));
}

TEST_CASE_TEMPLATE("create_coons_volume: a weight disagreement names the weight group", T, float,
                   double) {
  // The cube with face_w0 and face_w1 rational (weights 1), and one weight of face_w0 raised by
  // four times the tier at the origin corner (its homogeneous coordinates stay zero, so the
  // projected corner check sees nothing and only the edge check can).
  const double g = 4.0 * tier_of<T>();
  const auto corners = face_corners(cube(1.0));
  auto faces = bilinear_faces<T>(cube(1.0));
  faces[2][0] = rational_bilinear_face<T>(corners[4], {1.0 + g, 1.0, 1.0, 1.0});
  faces[2][1] = rational_bilinear_face<T>(corners[5], {1.0, 1.0, 1.0, 1.0});
  const std::string message = volume_refusal<T>(faces);
  REQUIRE_MESSAGE(has(message, "Edge "), message);
  CHECK(has(message, "face_w0"));
  const Reported r = reported_numbers(message);
  CHECK(r.population == "edge weight");
  CHECK(r.gap == doctest::Approx(g).epsilon(1e-2));
  CHECK(r.scale == doctest::Approx(1.0 + g).epsilon(1e-3));
  CHECK(r.tol == doctest::Approx(tier_of<T>() * (1.0 + g)).epsilon(1e-3));
}

TEST_CASE_TEMPLATE(
    "create_coons_volume: a coordinate error on a rational edge names the coordinate", T, float,
    double) {
  // One control point of the rational face_w0 has its whole homogeneous row doubled: it projects
  // to the same point (so the corner check sees nothing) and disagrees with its neighbours by the
  // model's size in the coordinate columns beside 1 in the weight column.
  const double s = 100.0;
  const auto corners = face_corners(cube(s));
  auto faces = bilinear_faces<T>(cube(s));
  faces[2][0] = rational_bilinear_face<T>(corners[4], {1.0, 1.0, 1.0, 1.0});
  faces[2][1] = rational_bilinear_face<T>(corners[5], {1.0, 1.0, 1.0, 1.0});
  std::vector<double> cp = control_points_of(faces[2][0]);
  for (std::size_t c = 0; c < 4; ++c) {
    cp[3 * 4 + c] *= 2.0;  // the (u = 1, v = 1) control point
  }
  faces[2][0] = with_control_points(faces[2][0], cp);
  const std::string message = volume_refusal<T>(faces);
  REQUIRE_MESSAGE(has(message, "Edge "), message);
  const Reported r = reported_numbers(message);
  CHECK(r.population == "edge coordinate");
  CHECK(r.gap == doctest::Approx(s).epsilon(1e-3));
  CHECK(r.scale == doctest::Approx(2.0 * s).epsilon(1e-3));
  CHECK(r.tol == doctest::Approx(tier_of<T>() * 2.0 * s).epsilon(1e-3));
}

namespace {

/// The unit cube's faces, each a 3 x 3 net over `[0, 0.5, 1]^2` (degree 1, a knot at 0.5), the
/// boundary weights 1 and the middle weight `interior`. Every edge still agrees; only the face
/// interiors, which the checks leave free, carry the weight.
template <class T>
Faces<T> cube_with_interior_weight(double interior) {
  const auto corners = face_corners(cube(1.0));
  std::array<bs::Bspline<T>, 6> faces = {
      bilinear_face<T>(corners[0]), bilinear_face<T>(corners[1]), bilinear_face<T>(corners[2]),
      bilinear_face<T>(corners[3]), bilinear_face<T>(corners[4]), bilinear_face<T>(corners[5])};
  const std::array<double, 3> samples = {0.0, 0.5, 1.0};
  for (std::size_t f = 0; f < 6; ++f) {
    std::vector<double> net;
    for (std::size_t a = 0; a < 3; ++a) {
      for (std::size_t b = 0; b < 3; ++b) {
        const double u = samples[a];
        const double v = samples[b];
        const double w = (a == 1 && b == 1) ? interior : 1.0;
        for (std::size_t c = 0; c < 3; ++c) {
          const double x = (1 - u) * (1 - v) * corners[f][0 * 3 + c] +
                           (1 - u) * v * corners[f][1 * 3 + c] +
                           u * (1 - v) * corners[f][2 * 3 + c] + u * v * corners[f][3 * 3 + c];
          net.push_back(x * w);
        }
        net.push_back(w);
      }
    }
    faces[f] = surface_flat<T>(1, {0.5}, 1, {0.5}, net, true);
  }
  return as_pairs<T>(faces);
}

}  // namespace

TEST_CASE_TEMPLATE("create_coons_volume: positive face weights do not imply a positive blend", T,
                   float, double) {
  // The blend's middle control weight is 3 t - 3 + 1 = 3 t - 2 for interior weight t on every face
  // (three ruled terms carry t, the three bilinear terms and the corner term carry 1): -0.5 at
  // t = 0.5, with every input weight in [0.5, 1].
  const std::string message = volume_refusal<T>(cube_with_interior_weight<T>(0.5));
  CHECK(has(message,
            "Coons volume weight is not certified positive: control weight -5.000e-01 at "
            "index (1, 1, 1) is not above zero, so the bound w(u) >= min_i w_i no longer "
            "keeps the blended weight field away from zero and the volume may have a "
            "pole. The seven-term blend subtracts three of its terms, so positive weights "
            "on the six faces do not imply positive weights on the volume; supply faces "
            "whose weights vary less across the model. The test is sufficient, not "
            "necessary: a volume whose weight field never vanishes can still fail it."));
}

TEST_CASE_TEMPLATE("create_coons_volume: a positive blend of varying weights is accepted", T, float,
                   double) {
  // t = 0.7 lands on 3 t - 2 = 0.1: far from the faces' weights, still accepted, and the volume
  // still interpolates every face (without this, refusing all rational input would pass the test
  // above). The faces carry t as stored in T.
  const Faces<T> faces = cube_with_interior_weight<T>(0.7);
  const bs::Bspline<T> v = cad::create_coons_volume<T>(faces);
  const double t = static_cast<double>(static_cast<T>(0.7));
  double smallest = 1e300;
  const std::size_t width = v.cp_size();
  const std::span<const T> cp = v.control_points();
  for (std::size_t p = 0; p * width < cp.size(); ++p) {
    smallest = (std::min)(smallest, static_cast<double>(cp[p * width + width - 1]));
  }
  // Seven terms of magnitude at most 1: the aligned bound of the header comment for M = 1.
  CHECK(std::abs(smallest - (3.0 * t - 2.0)) <= 16.0 * eps<T>());
  check_all_faces(v, faces, tier_tolerance(v));
}

TEST_CASE_TEMPLATE("create_coons_volume: faces of different ranks are refused", T, float, double) {
  Faces<T> faces = bilinear_faces<T>(cube(1.0));
  // face_w1 in the plane: rank 2 (the corner it carries are the cube's z = 1 corners, but with two
  // coordinates only).
  faces[2][1] = surface_flat<T>(1, {}, 1, {}, {0, 0, 0, 1, 1, 0, 1, 1}, false);
  const std::string message = volume_refusal<T>(faces);
  CHECK_MESSAGE(has(message, "the faces must have one rank."), message);
  CHECK(has(message, "gives a point of"));
}

// ============================================================ NaN verdicts and the shape guard

namespace {

/// The cube's faces with the z coordinate of `face_w0`'s `(u = 1, v = 1)` corner set to NaN.
template <class T>
Faces<T> nan_w0_corner() {
  auto corners = face_corners(cube(1.0));
  corners[4][(1 * 2 + 1) * 3 + 2] = std::numeric_limits<double>::quiet_NaN();
  return as_pairs<T>({bilinear_face<T>(corners[0]), bilinear_face<T>(corners[1]),
                      bilinear_face<T>(corners[2]), bilinear_face<T>(corners[3]),
                      bilinear_face<T>(corners[4]), bilinear_face<T>(corners[5])});
}

}  // namespace

TEST_CASE_TEMPLATE("verify_corners_3d: a NaN corner is not refused, inherited from the reference",
                   T, float, double) {
  // Unusual input, pinned as inherited from the reference implementation (not endorsed): the
  // corner comparison refuses `gap > tol`, which a NaN gap does not satisfy, and the NaN scale
  // makes the tolerance NaN too. The corner check alone therefore accepts the corner. The edge
  // check that follows in `create_coons_volume` does not (see the next test), so the volume as a
  // whole is refused, at the edge and not at the corner.
  const Faces<T> faces = nan_w0_corner<T>();
  const std::vector<cad::detail::EdgeReadings<T>> readings =
      cad::detail::coons_edge_pairs<T>(faces);
  CHECK_NOTHROW(cad::detail::verify_corners_3d<T>(readings, tier_of<T>()));
  const std::string message = volume_refusal<T>(faces);
  CHECK_MESSAGE(has(message, "Edge "), message);
  CHECK_FALSE(has(message, "Corner "));
}

TEST_CASE_TEMPLATE("verify_edges_3d: a NaN edge control point is refused", T, float, double) {
  // face_w0 becomes a degree-(2, 1) patch whose middle control point at v = 1 has a NaN
  // coordinate: all eight corners are finite and agree, so only the edge comparison sees it. It is
  // refused because the comparison is `!(gap <= tol)`, which a NaN gap satisfies.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  auto faces = bilinear_faces<T>(cube(1.0));
  faces[2][0] = quadratic_linear<T>({0, 0, 0, 0, 1, 0,        //
                                     0.5, 0, 0, 0.5, 1, nan,  //
                                     1, 0, 0, 1, 1, 0});
  const std::string message = volume_refusal<T>(faces);
  CHECK_MESSAGE(has(message, "Edge "), message);
  CHECK_MESSAGE(has(message, "gap nan"), message);
}

TEST_CASE_TEMPLATE("verify_positive_weights: a NaN weight is refused", T, float, double) {
  // The smallest weight is taken as NumPy takes it: a NaN weight is the smallest, wherever it
  // sits (first, in the middle, or last in the control net), and is refused.
  const bs::Bspline<T> trilinear = make_spline<T>(
      {Dir{clamped(1, {}), 1}, Dir{clamped(1, {}), 1}, Dir{clamped(1, {}), 1}}, 3, true, 1);
  REQUIRE(trilinear.cp_size() == 4);
  CHECK_NOTHROW(cad::detail::verify_positive_weights<T>(trilinear));
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const std::size_t at : {std::size_t{0}, std::size_t{3}, std::size_t{7}}) {
    std::vector<double> flat = control_points_of(trilinear);
    flat[at * 4 + 3] = nan;
    const std::string message = refusal(
        [&] { cad::detail::verify_positive_weights<T>(with_control_points(trilinear, flat)); });
    CAPTURE(at);
    CHECK_MESSAGE(has(message, "weight is not certified positive"), message);
    CHECK_MESSAGE(has(message, "control weight nan at index"), message);
  }
}

TEST_CASE_TEMPLATE("create_coons_volume: a shared edge of unequal control-net sizes is refused", T,
                   float, double) {
  // An interior knot of the u direction a quarter of the space tolerance above another leaves
  // `make_compat` with unequal numbers of functions (see "knots a rounding apart" above). Here
  // face_v0 and face_v1 carry the knot at 0.5 and face_w0 and face_w1 the shifted one, so every
  // pair of opposite faces is compatible and the corners agree, while each u edge is read from a
  // v face and a w face and comes out with different control-point counts: the shape guard of
  // `verify_edges_3d` refuses before anything is built from them.
  const bs::Bspline<T> probe = line<T>({0, 0, 0}, {1, 0, 0});
  const double tol = probe.space().spaces()[0].tolerance();
  const double shifted = static_cast<double>(static_cast<T>(0.5 + 0.25 * tol));
  REQUIRE(shifted > 0.5);
  const auto with_u_knot = [](const bs::Bspline<T>& face, double knot) {
    const std::array<T, 1> k = {static_cast<T>(knot)};
    std::vector<std::span<const T>> per_dim(2);
    per_dim[0] = std::span<const T>(k);
    return bs::insert_knots<T>(face, per_dim);
  };
  auto faces = bilinear_faces<T>(cube(1.0));
  faces[1][0] = with_u_knot(faces[1][0], 0.5);
  faces[1][1] = with_u_knot(faces[1][1], 0.5);
  faces[2][0] = with_u_knot(faces[2][0], shifted);
  faces[2][1] = with_u_knot(faces[2][1], shifted);
  const std::string message = volume_refusal<T>(faces);
  CHECK_MESSAGE(has(message, "Edge u_v"), message);
  CHECK_MESSAGE(has(message, "give control nets of different shapes after being made compatible"),
                message);
}
