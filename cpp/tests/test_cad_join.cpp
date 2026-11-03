// `join` (include/pantr/cad/join.hpp).
//
// Oracles. A join evaluates equal to its parts: at a parameter of the first part's domain the
// result equals the first part, and at a parameter of the second the result equals the second
// evaluated at the matching point of its own domain (the join axis of the second part is shifted
// to start where the first ends, a translation of the parameter; the domains below are dyadic, so
// the shift is exact). Both sides are evaluated **exactly** (bspline_exact_eval.hpp: the basis
// from its recursive definition over the stored control points, in homogeneous coordinates for a
// rational one), so a tolerance bounds the construction's roundings and nothing else.
//  - Where nothing but the seam rows is touched (equal degrees and knots on every axis) every
//    control point of the result is a copy of an input control point, except the seam row, which
//    is `(x + x) / 2` of two equal values and is exact: the bound is zero.
//  - Where `make_compat` or the degree elevation or the conversion to open form runs, the bound is
//    the conservative tier of the type times the larger of 1 and the largest control-point
//    magnitude of the result, as in test_cad_operations.cpp.
//  - One smooth spline split in two (the knot inserted to full multiplicity and the two halves
//    taken apart) must rejoin to that spline: the junction knot is removed again.
//  - Two quarter circles join to a half circle: radius 1, ends at (1, 0) and (-1, 0).
//  - A join along axis 1 of two surfaces equals the join along axis 0 of the same surfaces with
//    their directions swapped, then swapped back (a metamorphic relation that exercises the
//    strided rows of a join axis other than the first).
// Every refusal is matched by message, and the gap, tolerance and scale it reports are compared
// with the values computed in the test from the data.
#include <doctest/doctest.h>

#include "bspline_test_support.hpp"
#include "cad_exact_support.hpp"
#include "cad_test_support.hpp"

#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/convert.hpp>
#include <pantr/bspline/knot_insertion.hpp>
#include <pantr/bspline/reparametrize.hpp>
#include <pantr/cad/compat.hpp>
#include <pantr/cad/detail/validation.hpp>
#include <pantr/cad/join.hpp>
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
using cad_test::fmt3;
using cad_test::has;
using cad_test::knots_as_double;
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

namespace {

/// A curve over the given knot vector (any domain).
template <class T>
bs::Bspline<T> curve_on(int degree, const std::vector<double>& knots,
                        const std::vector<double>& flat, bool rational = false) {
  const std::vector<T> k = cast_all<T>(knots);
  std::vector<bs::BsplineSpace1D<T>> spaces;
  spaces.emplace_back(std::span<const T>(k), degree);
  const std::vector<T> cp = cast_all<T>(flat);
  return bs::Bspline<T>(bs::BsplineSpace<T>(std::move(spaces)), std::span<const T>(cp), rational);
}

/// The largest exact gap between `result` and its two parts, over a grid on the other axes and five
/// parameters along the join axis of each part.
template <class T>
double parts_gap(const bs::Bspline<T>& result, const bs::Bspline<T>& first,
                 const bs::Bspline<T>& second, std::size_t axis) {
  const bs::Bspline<T> a = result.rational() ? cad::detail::promote_to_rational<T>(first) : first;
  const bs::Bspline<T> b = result.rational() ? cad::detail::promote_to_rational<T>(second) : second;
  const auto dom = [](const bs::Bspline<T>& x, std::size_t d) {
    const auto pair = x.space().spaces()[d].domain();
    return std::array<double, 2>{static_cast<double>(pair.first), static_cast<double>(pair.second)};
  };
  const std::array<double, 5> along = {0.0, 0.125, 0.5, 0.875, 1.0};
  const std::array<double, 4> across = {0.0, 0.25, 0.625, 1.0};
  const std::array<double, 2> da = dom(a, axis);
  const std::array<double, 2> db = dom(b, axis);
  double worst = 0.0;
  std::vector<std::size_t> idx(a.dim(), 0);
  for (;;) {
    for (const double f : along) {
      std::vector<double> xa(a.dim());
      std::vector<double> xb(a.dim());
      std::vector<double> xr(a.dim());
      for (std::size_t d = 0; d < a.dim(); ++d) {
        if (d == axis) {
          xa[d] = da[0] + f * (da[1] - da[0]);
          xr[d] = xa[d];
        } else {
          const std::array<double, 2> r = dom(a, d);
          xa[d] = r[0] + across[idx[d]] * (r[1] - r[0]);
          xb[d] = dom(b, d)[0] + across[idx[d]] * (dom(b, d)[1] - dom(b, d)[0]);
          xr[d] = xa[d];
        }
      }
      // First part: the result at the same parameters.
      worst = (std::max)(worst, max_difference(value_at(result, xr), value_at(a, xa)));
      // Second part: the result at the shifted parameter, the part at its own.
      xb[axis] = db[0] + f * (db[1] - db[0]);
      xr[axis] = da[1] + f * (db[1] - db[0]);
      worst = (std::max)(worst, max_difference(value_at(result, xr), value_at(b, xb)));
    }
    std::size_t d = a.dim();
    bool done = true;
    while (d-- > 0) {
      if (d == axis) {
        continue;
      }
      if (++idx[d] < across.size()) {
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

/// The values of the slab `index` of `b` along `axis`, the axis removed, row-major over the rest.
template <class T>
std::vector<double> slab_of(const bs::Bspline<T>& b, std::size_t axis, std::size_t index) {
  const std::span<const std::size_t> shape = b.shape();
  std::size_t outer = 1;
  std::size_t inner = 1;
  for (std::size_t d = 0; d < shape.size(); ++d) {
    if (d < axis) {
      outer *= shape[d];
    } else if (d > axis) {
      inner *= shape[d];
    }
  }
  const std::vector<double> cp = control_points_of(b);
  std::vector<double> out;
  for (std::size_t o = 0; o < outer; ++o) {
    const std::size_t at = (o * shape[axis] + index) * inner;
    out.insert(out.end(), cp.begin() + static_cast<std::ptrdiff_t>(at),
               cp.begin() + static_cast<std::ptrdiff_t>(at + inner));
  }
  return out;
}

/// `b` with the slab `index` along `axis` replaced.
template <class T>
bs::Bspline<T> with_slab(const bs::Bspline<T>& b, std::size_t axis, std::size_t index,
                         const std::vector<double>& slab) {
  const std::span<const std::size_t> shape = b.shape();
  std::size_t outer = 1;
  std::size_t inner = 1;
  for (std::size_t d = 0; d < shape.size(); ++d) {
    if (d < axis) {
      outer *= shape[d];
    } else if (d > axis) {
      inner *= shape[d];
    }
  }
  REQUIRE(slab.size() == outer * inner);
  std::vector<double> cp = control_points_of(b);
  for (std::size_t o = 0; o < outer; ++o) {
    const std::size_t at = (o * shape[axis] + index) * inner;
    std::copy(slab.begin() + static_cast<std::ptrdiff_t>(o * inner),
              slab.begin() + static_cast<std::ptrdiff_t>((o + 1) * inner),
              cp.begin() + static_cast<std::ptrdiff_t>(at));
  }
  return with_control_points(b, cp);
}

/// `second` with its first slab along `axis` made equal to the last slab of `first`.
template <class T>
bs::Bspline<T> meeting(const bs::Bspline<T>& first, const bs::Bspline<T>& second,
                       std::size_t axis) {
  return with_slab(second, axis, 0, slab_of(first, axis, first.shape()[axis] - 1));
}

std::string join_refusal_curves(const bs::Bspline<double>& a, const bs::Bspline<double>& b) {
  return refusal([&] { static_cast<void>(cad::join<double>(a, b, 0)); });
}

}  // namespace

// ================================================================================== evaluation

TEST_CASE_TEMPLATE("join: two curves of degree 1 concatenate and evaluate as their parts", T, float,
                   double) {
  const bs::Bspline<T> a = curve_on<T>(1, {0, 0, 0.5, 1, 1}, {0, 0, 0, 1, 2, 0.5, 3, 1, -1});
  // The second part lives on [3, 7] and starts where the first ends.
  const bs::Bspline<T> b = curve_on<T>(1, {3, 3, 5, 7, 7}, {3, 1, -1, 4, 2, 0, 6, -1, 2});
  const bs::Bspline<T> j = cad::join<T>(a, b, 0);
  CHECK(j.dim() == 1);
  CHECK(j.degrees() == std::vector<std::size_t>{1});
  // Knots: the first part's without its end clamp, the junction knot once (degree 1), the second
  // part's remapped to [1, 5] without its start clamp: 0 0 0.5 | 1 | 3 5 5.
  CHECK(knots_as_double(j, 0) == std::vector<double>{0, 0, 0.5, 1, 3, 5, 5});
  // Control points: the first part's, then the second's after the shared one (the average of two
  // equal values is that value).
  CHECK(control_points_of(j) ==
        std::vector<double>{0, 0, 0, 1, 2, 0.5, 3, 1, -1, 4, 2, 0, 6, -1, 2});
  // Nothing but copies and an exact average: the bound is zero.
  CHECK(parts_gap(j, a, b, 0) <= 0.0);
}

TEST_CASE_TEMPLATE("join: parts of different degrees are elevated and still evaluate as themselves",
                   T, float, double) {
  const bs::Bspline<T> a = curve_of<T>(2, {0.5}, {{0, 0, 0}, {1, 2, 0}, {2, 2, 1}, {3, 0, 1}});
  const bs::Bspline<T> b = curve_of<T>(3, {}, {{3, 0, 1}, {4, -1, 2}, {5, 1, 2}, {6, 0, 0}});
  const bs::Bspline<T> j = cad::join<T>(a, b, 0);
  CHECK(j.degrees() == std::vector<std::size_t>{3});
  // The elevation changes the basis: the conservative tier.
  CHECK(parts_gap(j, a, b, 0) <= tier_tolerance(j));
}

TEST_CASE_TEMPLATE("join: one smooth spline split in two rejoins with the junction knot removed", T,
                   float, double) {
  // A degree-2 spline whose knots and control points are dyadic, so the insertions below are exact.
  const bs::Bspline<T> whole =
      make_spline<T>({Dir{{0, 0, 0, 0.5, 1, 1.5, 2, 2, 2}, 2}}, 3, false, 11);
  // The knot 1 raised to multiplicity 3 = degree + 1 splits the spline into two clamped halves
  // that meet in one point (control points 3 and 4 of the refined spline).
  const bs::Bspline<T> refined = bspline_test::insert_in(whole, 0, {1.0, 1.0});
  REQUIRE(refined.num_basis(0) == 8);
  const std::vector<double> cp = control_points_of(refined);
  const std::vector<double> left_cp(cp.begin(), cp.begin() + 4 * 3);
  const std::vector<double> right_cp(cp.begin() + 4 * 3, cp.end());
  const bs::Bspline<T> left = curve_on<T>(2, {0, 0, 0, 0.5, 1, 1, 1}, left_cp);
  const bs::Bspline<T> right = curve_on<T>(2, {1, 1, 1, 1.5, 2, 2, 2}, right_cp);
  // The halves share their end point up to the rounding of the insertion.
  const bs::Bspline<T> j = cad::join<T>(left, right, 0);
  CHECK(knots_as_double(j, 0) == knots_as_double(whole, 0));
  CHECK(j.num_basis(0) == whole.num_basis(0));
  std::vector<double> grid_pts = cad_test::linspace(0.0, 2.0, 17);
  // Against the original spline, exactly: the removal is within the round-off floor.
  double worst = 0.0;
  for (const double t : grid_pts) {
    worst = (std::max)(worst, max_difference(value_at(j, {t}), value_at(whole, {t})));
  }
  CHECK(worst <= tier_tolerance(j));
}

TEST_CASE_TEMPLATE("join: a kink at the junction keeps the knot (C0)", T, float, double) {
  const bs::Bspline<T> a = curve_of<T>(2, {}, {{0, 0, 0}, {1, 2, 0}, {2, 0, 0}});
  // The second part leaves the junction in a different direction: not one smooth spline.
  const bs::Bspline<T> b = curve_of<T>(2, {}, {{2, 0, 0}, {2, -3, 1}, {4, 0, 0}});
  const bs::Bspline<T> j = cad::join<T>(a, b, 0);
  CHECK(j.num_basis(0) == 5);
  CHECK(knots_as_double(j, 0) == std::vector<double>{0, 0, 0, 1, 1, 2, 2, 2});
  CHECK(parts_gap(j, a, b, 0) <= tier_tolerance(j));
}

TEST_CASE("join: two quarter circles are a half circle") {
  const bs::Bspline<double> first =
      cad::create_circle(1.0, std::nullopt, std::array<double, 2>{0.0, std::numbers::pi / 2.0});
  const bs::Bspline<double> second = cad::create_circle(
      1.0, std::nullopt, std::array<double, 2>{std::numbers::pi / 2.0, std::numbers::pi});
  const bs::Bspline<double> j = cad::join<double>(first, second, 0);
  CHECK(j.rational());
  CHECK(j.num_basis(0) == 5);
  CHECK(knots_as_double(j, 0) == std::vector<double>{0, 0, 0, 1, 1, 2, 2, 2});
  const double tol = tier_tolerance(j);
  const std::vector<double> ts = cad_test::linspace(0.0, 2.0, 33);
  const std::vector<double> pts = evaluate_at(j, ts);
  for (std::size_t i = 0; i < ts.size(); ++i) {
    CHECK(std::abs(std::hypot(pts[3 * i], pts[3 * i + 1]) - 1.0) <= tol);
  }
  CHECK(std::abs(pts[0] - 1.0) <= tol);
  CHECK(std::abs(pts[1]) <= tol);
  CHECK(std::abs(pts[3 * 32] + 1.0) <= tol);
  CHECK(std::abs(pts[3 * 32 + 1]) <= tol);
  CHECK(parts_gap(j, first, second, 0) <= tol);
}

TEST_CASE_TEMPLATE("join: a polynomial part joined to a rational one is promoted", T, float,
                   double) {
  // The seam weight is 1, so the polynomial part's promoted seam (weight 1) agrees with it.
  const bs::Bspline<T> rat =
      curve_on<T>(2, {0, 0, 0, 1, 1, 1}, {0, 0, 0, 1, 2, 1, 0, 1.5, 3, 0, 0, 1}, true);
  const bs::Bspline<T> poly = curve_on<T>(2, {0, 0, 0, 1, 1, 1}, {3, 0, 0, 4, 1, 1, 5, -1, 2});
  const bs::Bspline<T> j = cad::join<T>(rat, poly, 0);
  CHECK(j.rational());
  CHECK(j.cp_size() == 4);
  CHECK(parts_gap(j, rat, poly, 0) <= tier_tolerance(j));
  // And the other way round: the rational part second.
  const bs::Bspline<T> rat2 =
      curve_on<T>(2, {0, 0, 0, 1, 1, 1}, {5, -1, 2, 1, 2, 1, 2, 1.5, 4, 0, 0, 1}, true);
  const bs::Bspline<T> j2 = cad::join<T>(poly, rat2, 0);
  CHECK(j2.rational());
  CHECK(parts_gap(j2, poly, rat2, 0) <= tier_tolerance(j2));
}

TEST_CASE_TEMPLATE("join: rational parts agreeing in homogeneous data, seam weight other than 1", T,
                   float, double) {
  const bs::Bspline<T> a =
      curve_on<T>(2, {0, 0, 0, 1, 1, 1}, {0, 0, 0, 1, 2, 2, 0, 2, 3, 3, 0, 2}, true);
  const bs::Bspline<T> b =
      curve_on<T>(2, {0, 0, 0, 1, 1, 1}, {3, 3, 0, 2, 4, 1, 1, 1.5, 6, 0, 2, 2}, true);
  const bs::Bspline<T> j = cad::join<T>(a, b, 0);
  CHECK(j.rational());
  CHECK(parts_gap(j, a, b, 0) <= tier_tolerance(j));
}

TEST_CASE_TEMPLATE("join: surfaces of different degrees and transverse knots, along each axis", T,
                   float, double) {
  // A surface in (u, v) with u of degree 2 (one knot) and v of degree 2; the second surface has u
  // of degree 1 and a v direction refined differently. Their seam rows agree once made compatible.
  const bs::Bspline<T> a0 =
      make_spline<T>({Dir{clamped(2, {0.5}), 2}, Dir{clamped(2, {}), 2}}, 3, false, 21);
  const bs::Bspline<T> a = bspline_test::insert_in(a0, 1, {0.25});
  const bs::Bspline<T> b0 =
      make_spline<T>({Dir{clamped(1, {}), 1}, Dir{clamped(2, {}), 2}}, 3, false, 22);
  // The seam row of `b0` is the last row of `a0`, whose v direction is `b0`'s own.
  const bs::Bspline<T> b = meeting(a0, b0, 0);
  const bs::Bspline<T> j = cad::join<T>(a, b, 0);
  CHECK(j.dim() == 2);
  CHECK(j.degrees() == std::vector<std::size_t>{2, 2});
  CHECK(parts_gap(j, a, b, 0) <= tier_tolerance(j));

  // Along axis 1: the same surfaces with their directions swapped.
  const std::array<long long, 2> swap_axes = {1, 0};
  const auto swapped = [&](const bs::Bspline<T>& x) {
    return bs::permuted<T>(x, std::span<const long long>(swap_axes));
  };
  const bs::Bspline<T> j1 = cad::join<T>(swapped(a), swapped(b), 1);
  CHECK(parts_gap(j1, swapped(a), swapped(b), 1) <= tier_tolerance(j1));
  // The metamorphic relation: the swap of the axis-1 join is the axis-0 join.
  const bs::Bspline<T> back = swapped(j1);
  CHECK(back.degrees() == j.degrees());
  CHECK(knots_as_double(back, 0) == knots_as_double(j, 0));
  CHECK(knots_as_double(back, 1) == knots_as_double(j, 1));
  const std::vector<double> c0 = control_points_of(back);
  const std::vector<double> c1 = control_points_of(j);
  REQUIRE(c0.size() == c1.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < c0.size(); ++i) {
    worst = (std::max)(worst, std::abs(c0[i] - c1[i]));
  }
  CHECK(worst <= tier_tolerance(j));
}

TEST_CASE_TEMPLATE("join: two volumes along the middle axis", T, float, double) {
  const std::vector<Dir> dirs = {Dir{clamped(2, {0.5}), 2}, Dir{clamped(1, {}), 1},
                                 Dir{clamped(1, {0.25}), 1}};
  const bs::Bspline<T> a = make_spline<T>(dirs, 3, false, 31);
  const bs::Bspline<T> b0 = make_spline<T>(dirs, 3, false, 32);
  const bs::Bspline<T> b = meeting(a, b0, 1);
  const bs::Bspline<T> j = cad::join<T>(a, b, 1);
  CHECK(j.dim() == 3);
  CHECK(j.num_basis(1) == 3);
  // Equal spaces except along the join axis: copies and an exact average only.
  CHECK(parts_gap(j, a, b, 1) <= 0.0);
}

TEST_CASE_TEMPLATE("join: an unclamped part is opened first", T, float, double) {
  // Uniform knots 0..5, degree 2: three control points, domain [2, 3] (not clamped).
  const bs::Bspline<T> a = curve_on<T>(2, {0, 1, 2, 3, 4, 5}, {0, 0, 0, 1, 2, 0, 3, 0, 1});
  const bs::Bspline<T> opened = bs::to_open_bspline<T>(a);
  const std::vector<double> end = slab_of(opened, 0, opened.num_basis(0) - 1);
  const bs::Bspline<T> b0 = curve_on<T>(2, {0, 0, 0, 1, 1, 1}, {0, 0, 0, 1, 1, 1, 2, 0, 0});
  const bs::Bspline<T> b = with_slab(b0, 0, 0, end);
  const bs::Bspline<T> j = cad::join<T>(a, b, 0);
  CHECK(j.space().spaces()[0].has_open_knots());
  CHECK(parts_gap(j, a, b, 0) <= tier_tolerance(j));
}

TEST_CASE("join: float32 stays float32, with its junction knot") {
  // Knots that are not dyadic: the first part ends at float(0.1), the second is 0.3 long.
  const bs::Bspline<float> a = curve_on<float>(1, {0, 0, 0.1, 0.1}, {0, 0, 0, 1, 1, 1});
  const bs::Bspline<float> b = curve_on<float>(1, {0, 0, 0.3, 0.3}, {1, 1, 1, 2, 2, 2});
  const bs::Bspline<float> j = cad::join<float>(a, b, 0);
  const double u1 = static_cast<double>(0.1F);
  const double end = u1 + (static_cast<double>(0.3F) - 0.0);  // formed in double, then rounded
  const std::vector<double> want = {0.0, 0.0, u1, static_cast<double>(static_cast<float>(end)),
                                    static_cast<double>(static_cast<float>(end))};
  CHECK(knots_as_double(j, 0) == want);
  CHECK(control_points_of(j) == std::vector<double>{0, 0, 0, 1, 1, 1, 2, 2, 2});
}

// ===================================================================================== refusals

TEST_CASE("join: the dimensions must agree and the axis must be one of them") {
  const bs::Bspline<double> curve = line<double>({0, 0, 0}, {1, 0, 0});
  const bs::Bspline<double> surface =
      surface_flat<double>(1, {}, 1, {}, {0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0});
  CHECK(refusal([&] { static_cast<void>(cad::join<double>(curve, surface, 0)); }) ==
        "Both B-splines must have the same dim, got 1 and 2.");
  CHECK(refusal([&] { static_cast<void>(cad::join<double>(surface, curve, 0)); }) ==
        "Both B-splines must have the same dim, got 2 and 1.");
  CHECK(refusal([&] { static_cast<void>(cad::join<double>(curve, curve, -1)); }) ==
        "axis must be in [0, 1), got -1.");
  CHECK(refusal([&] { static_cast<void>(cad::join<double>(curve, curve, 1)); }) ==
        "axis must be in [0, 1), got 1.");
  CHECK(refusal([&] { static_cast<void>(cad::join<double>(surface, surface, 2)); }) ==
        "axis must be in [0, 2), got 2.");
}

TEST_CASE_TEMPLATE("join: a curve seam that does not meet is refused with the numbers it used", T,
                   float, double) {
  // The first part ends at (1, 1, 0), the second starts at (1, 1.25, 0): gap 0.25, and the largest
  // magnitude over the two rows is 1.25. A curve's seam is one point, so no index is named.
  const bs::Bspline<T> a = curve_of<T>(1, {}, {{0, 0, 0}, {1, 1, 0}});
  const bs::Bspline<T> b = curve_of<T>(1, {}, {{1, 1.25, 0}, {2, 2, 0}});
  const std::string message = refusal([&] { static_cast<void>(cad::join<T>(a, b, 0)); });
  const std::string expect =
      "Shared boundary mismatch on axis 0: the first B-spline gives [1.0, 1.0, 0.0], "
      "the second gives [1.0, 1.25, 0.0] (gap " +
      fmt3(0.25) + ", above " + fmt3(tier_of<T>() * 1.25) + " at boundary coordinate scale " +
      fmt3(1.25) + ").";
  CHECK(message == expect);
}

namespace {

/// Two surfaces, bilinear in u and quadratic in v, `b` starting where `a` ends except along the
/// seam rows' second control point, moved by `shift` in y.
template <class T>
std::array<bs::Bspline<T>, 2> surfaces_with_seam_shift(double shift_y_at_1,
                                                       double shift_y_at_0 = 0.0) {
  // a[i][j] = (i, j, 0.25 i): the seam is the row i = 1.
  std::vector<double> ca;
  std::vector<double> cb;
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      const double di = static_cast<double>(i);
      const double dj = static_cast<double>(j);
      ca.insert(ca.end(), {di, dj, 0.25 * di});
      double y = dj;
      if (i == 0) {
        y += j == 1 ? shift_y_at_1 : (j == 0 ? shift_y_at_0 : 0.0);
      }
      cb.insert(cb.end(), {1.0 + di, y, 0.25 + 0.25 * di});
    }
  }
  return {surface_flat<T>(1, {}, 2, {}, ca), surface_flat<T>(1, {}, 2, {}, cb)};
}

}  // namespace

TEST_CASE_TEMPLATE("join: a surface seam names the control point that disagrees most", T, float,
                   double) {
  const auto parts = surfaces_with_seam_shift<T>(0.5);
  const std::string message =
      refusal([&] { static_cast<void>(cad::join<T>(parts[0], parts[1], 0)); });
  // The seam rows are (1, j, 0.25); the second part's j = 1 is (1, 1.5, 0.25); the largest
  // magnitude over both rows is 2 (the y of j = 2).
  const std::string expect =
      "Shared boundary mismatch on axis 0 at index (1,): the first B-spline gives "
      "[1.0, 1.0, 0.25], the second gives [1.0, 1.5, 0.25] (gap " +
      fmt3(0.5) + ", above " + fmt3(tier_of<T>() * 2.0) + " at boundary coordinate scale " +
      fmt3(2.0) + ").";
  CHECK(message == expect);
}

TEST_CASE_TEMPLATE("join: on equal gaps the first control point is named", T, float, double) {
  // j = 0 and j = 1 both disagree by 0.5 (the first of the two is the point named).
  const auto parts = surfaces_with_seam_shift<T>(0.5, 0.5);
  const std::string message =
      refusal([&] { static_cast<void>(cad::join<T>(parts[0], parts[1], 0)); });
  CHECK_MESSAGE(has(message, "on axis 0 at index (0,):"), message);
}

TEST_CASE_TEMPLATE("join: a volume seam names a two-index location", T, float, double) {
  // Joined along v (axis 1): the seam is the (u, w) face, of 2 x 2 control points. The second part
  // is the first translated one step in v, except at (u, w) = (1, 0), moved by 0.75 in x.
  std::vector<double> ca;
  std::vector<double> cb;
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      for (std::size_t k = 0; k < 2; ++k) {
        const std::array<double, 3> p = {static_cast<double>(i), static_cast<double>(j),
                                         static_cast<double>(k)};
        ca.insert(ca.end(), p.begin(), p.end());
        const double dx = (i == 1 && j == 0 && k == 0) ? 0.75 : 0.0;
        cb.insert(cb.end(), {p[0] + dx, p[1] + 1.0, p[2]});
      }
    }
  }
  const auto volume = [](const std::vector<double>& flat) {
    const std::vector<double> k = {0, 0, 1, 1};
    std::vector<bs::BsplineSpace1D<T>> spaces;
    const std::vector<T> kt = cast_all<T>(k);
    for (int d = 0; d < 3; ++d) {
      spaces.emplace_back(std::span<const T>(kt), 1);
    }
    const std::vector<T> cp = cast_all<T>(flat);
    return bs::Bspline<T>(bs::BsplineSpace<T>(std::move(spaces)), std::span<const T>(cp), false);
  };
  const bs::Bspline<T> a = volume(ca);
  const bs::Bspline<T> b = volume(cb);
  const std::string message = refusal([&] { static_cast<void>(cad::join<T>(a, b, 1)); });
  const std::string expect =
      "Shared boundary mismatch on axis 1 at index (1, 0): the first B-spline gives "
      "[1.0, 1.0, 0.0], the second gives [1.75, 1.0, 0.0] (gap " +
      fmt3(0.75) + ", above " + fmt3(tier_of<T>() * 1.75) + " at boundary coordinate scale " +
      fmt3(1.75) + ").";
  CHECK(message == expect);
}

TEST_CASE_TEMPLATE("join: a weight-only seam disagreement names the weight group", T, float,
                   double) {
  // Homogeneous seam rows (1, 1, 0, 1) and (1, 1, 0, 1 + g): the coordinate columns agree.
  const double g = 4.0 * tier_of<T>();
  const bs::Bspline<T> a = curve_flat<T>(1, {}, {0, 0, 0, 1, 1, 1, 0, 1}, true);
  const bs::Bspline<T> b = curve_flat<T>(1, {}, {1, 1, 0, 1.0 + g, 2, 2, 0, 1}, true);
  const std::string message = refusal([&] { static_cast<void>(cad::join<T>(a, b, 0)); });
  REQUIRE_MESSAGE(has(message, "Shared boundary mismatch on axis 0: the first B-spline gives"),
                  message);
  const Reported r = reported_numbers(message);
  CHECK(r.population == "boundary weight");
  CHECK(r.gap == doctest::Approx(g).epsilon(1e-2));
  // The weight population's own maximum, which the displaced weight itself sets.
  CHECK(r.scale == doctest::Approx(1.0 + g).epsilon(1e-3));
  CHECK(r.tol == doctest::Approx(tier_of<T>() * (1.0 + g)).epsilon(1e-3));
}

TEST_CASE_TEMPLATE(
    "join: a coordinate-only seam disagreement on rational data names the coordinates", T, float,
    double) {
  const bs::Bspline<T> a = curve_flat<T>(1, {}, {0, 0, 0, 1, 1, 1, 0, 1}, true);
  const bs::Bspline<T> b = curve_flat<T>(1, {}, {1.5, 1, 0, 1, 2, 2, 0, 1}, true);
  const std::string message = refusal([&] { static_cast<void>(cad::join<T>(a, b, 0)); });
  const Reported r = reported_numbers(message);
  CHECK(r.population == "boundary coordinate");
  CHECK(r.gap == doctest::Approx(0.5));
  CHECK(r.scale == doctest::Approx(1.5));
  CHECK(r.tol == doctest::Approx(tier_of<T>() * 1.5).epsilon(1e-3));
}

TEST_CASE_TEMPLATE("join: control points of different ranks are refused", T, float, double) {
  const bs::Bspline<T> a = curve_of<T>(1, {}, {{0, 0, 0}, {1, 1, 0}});
  const bs::Bspline<T> b = curve_flat<T>(1, {}, {1, 1, 2, 2});
  CHECK(refusal([&] { static_cast<void>(cad::join<T>(a, b, 0)); }) ==
        "Shared boundary mismatch on axis 0: the two B-splines have junction faces of different "
        "shapes after being made compatible, (3,) and (2,).");
}

TEST_CASE_TEMPLATE("join: knots a rounding apart leave unequal seams, which are refused", T, float,
                   double) {
  // Inherited from make_compat (see test_cad_operations.cpp): the transverse knots 0.5 and 0.5 +
  // a quarter of the space tolerance leave B-splines with unequal numbers of functions.
  const auto make = [](double interior) {
    return surface_flat<T>(1, {}, 2, {interior}, std::vector<double>(2 * 4 * 3, 0.0));
  };
  const bs::Bspline<T> a = make(0.5);
  const double tol = a.space().spaces()[1].tolerance();
  const double shifted = static_cast<double>(static_cast<T>(0.5 + 0.25 * tol));
  REQUIRE(shifted > 0.5);
  const bs::Bspline<T> b = make(shifted);
  const std::array<bs::Bspline<T>, 2> pair = {a, b};
  const std::vector<bs::Bspline<T>> made =
      cad::make_compat<T>(std::span<const bs::Bspline<T>>(pair),
                          std::optional<std::vector<long long>>(std::vector<long long>{1}));
  REQUIRE(made[0].num_basis(1) != made[1].num_basis(1));
  const std::string expect =
      "Shared boundary mismatch on axis 0: the two B-splines have junction faces of "
      "different shapes after being made compatible, (" +
      std::to_string(made[0].num_basis(1)) + ", 3) and (" + std::to_string(made[1].num_basis(1)) +
      ", 3).";
  CHECK(refusal([&] { static_cast<void>(cad::join<T>(a, b, 0)); }) == expect);
}

TEST_CASE("join: the tier handed to the detail overload is the one used") {
  // A double seam off by 1e-5: refused at the double tier, accepted at the float tier a binding
  // passes when one of the two inputs was a float; the average is then the mean of the two rows.
  const bs::Bspline<double> a = curve_of<double>(1, {}, {{0, 0, 0}, {1, 1, 0}});
  const bs::Bspline<double> b = curve_of<double>(1, {}, {{1, 1.00001, 0}, {2, 2, 0}});
  CHECK(has(join_refusal_curves(a, b), "Shared boundary mismatch on axis 0"));
  const double float_tier = tier_of<float>();
  const bs::Bspline<double> j = cad::detail::join<double>(a, b, 0, float_tier);
  const std::vector<double> cp = control_points_of(j);
  REQUIRE(cp.size() == 9);
  CHECK(cp[4] == (1.0 + 1.00001) / 2.0);
  // The tolerance printed is the tier handed in, times the scale.
  const bs::Bspline<double> c = curve_of<double>(1, {}, {{1, 1.5, 0}, {2, 2, 0}});
  const std::string message =
      refusal([&] { static_cast<void>(cad::detail::join<double>(a, c, 0, float_tier)); });
  CHECK(reported_numbers(message).tol == doctest::Approx(float_tier * 1.5).epsilon(1e-3));
}

TEST_CASE("join: a NaN seam is refused, unlike a NaN corner of a Coons patch") {
  // Unusual input, pinned as inherited (not endorsed): the seam comparison keeps a NaN gap
  // (`!(gap <= tol)` refuses it), where the Coons corner comparison (`gap > tol`) accepts it.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const bs::Bspline<double> a = curve_of<double>(1, {}, {{0, 0, 0}, {1, nan, 0}});
  const bs::Bspline<double> b = curve_of<double>(1, {}, {{1, 1, 0}, {2, 2, 0}});
  const std::string message = join_refusal_curves(a, b);
  CHECK_MESSAGE(has(message,
                    "the first B-spline gives [1.0, nan, 0.0], the second gives [1.0, 1.0, 0.0] "
                    "(gap nan, above nan at boundary coordinate scale nan)."),
                message);
}
