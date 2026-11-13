// Slicing a B-spline at a parameter value, and the boundary along a direction.
//
// Oracle. The exact-rational Cox-de Boor evaluation of the original (bspline_exact_eval.hpp),
// which sums the control points against the recursive definition of the basis, so no corner
// cutting of the library is reused: the slice at `v` in direction `a`, evaluated exactly at the
// remaining parameters, must equal the original evaluated exactly with `v` in direction `a`.
//
// Tolerance, derived. A corner-cutting step forms `alpha a + (1 - alpha) b` with `alpha` in
// [0, 1], from knot differences in double. The weights are rounded to the storage type (at most
// `u` each), the two products and the sum are rounded (at most `3 u` of the larger operand), and
// the weight itself carries the rounding of its quotient and of the two differences it is formed
// from (at most `3 u_double` of `alpha`, times the distance `2 max|P|` between the operands). A
// step therefore adds at most `10 u max|P|`, the slice takes at most `degree` steps, and since
// the basis is a partition of unity the slice, evaluated, is within `10 degree u max|P|` of the
// original. Data on a dyadic mesh at dyadic parameters is exact in the storage type when every
// knot difference a weight is formed from is a power of two (spacings that are powers of two and
// windows of at most two spans, or one interior knot at the middle): every weight is then a
// dyadic rational of a few bits, and the combination of the few bits of the control points needs
// fewer than the format holds, so those cases are compared for equality.
#include <doctest/doctest.h>

#include "bspline_test_support.hpp"

#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/slice.hpp>
#include <pantr/core/errors.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace bs = pantr::bspline;
namespace oracle = bspline_oracle;
using bspline_test::clamped;
using bspline_test::Dir;
using bspline_test::make_spline;
using bspline_test::message_of;
using bspline_test::periodic_knots;
using bspline_test::same_space;
using oracle::Rational;

namespace {

template <class T>
double unit_roundoff() {
  return 0.5 * static_cast<double>(std::numeric_limits<T>::epsilon());
}

/// The derived bound on the slice of `b` (see the file header).
template <class T>
double slice_tolerance(const bs::Bspline<T>& b, std::size_t axis) {
  const double degree = static_cast<double>(b.degrees()[axis]);
  return 10.0 * degree * unit_roundoff<T>() * oracle::max_abs_control_point<T>(b);
}

/// The largest difference of any coordinate between the exact value of `sliced` and the exact value
/// of `b` with `value` in direction `axis`, over the tensor grid of `sample_points` (eight points
/// spanning the whole domain) in each remaining direction.
template <class T>
double slice_error(const bs::Bspline<T>& b, const bs::Bspline<T>& sliced, std::size_t axis,
                   double value) {
  const std::size_t dim = b.dim();
  const bs::BsplineSpace<T> whole = b.space();
  std::vector<std::vector<Rational>> axes;
  for (std::size_t d = 0; d < dim; ++d) {
    if (d == axis) {
      continue;
    }
    const auto dom = whole.spaces()[d].domain();
    axes.push_back(bspline_test::sample_points(oracle::exact(static_cast<double>(dom.first)),
                                               oracle::exact(static_cast<double>(dom.second))));
  }
  double worst = 0.0;
  std::vector<std::size_t> idx(axes.size(), 0);
  for (;;) {
    std::vector<Rational> rest;
    std::vector<Rational> full;
    for (std::size_t d = 0, k = 0; d < dim; ++d) {
      if (d == axis) {
        full.push_back(oracle::exact(value));
      } else {
        rest.push_back(axes[k][idx[k]]);
        full.push_back(axes[k][idx[k]]);
        ++k;
      }
    }
    const auto got = oracle::exact_value<T>(sliced, rest);
    const auto want = oracle::exact_value<T>(b, full);
    REQUIRE(got.size() == want.size());
    for (std::size_t c = 0; c < got.size(); ++c) {
      Rational diff = got[c] - want[c];
      if (diff < 0) {
        diff = -diff;
      }
      worst = (std::max)(worst, static_cast<double>(diff));
    }
    std::size_t d = axes.size();
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

struct Config {
  const char* name;
  std::vector<Dir> dirs;
  bool rational;
  bool dyadic;  // data and parameters on a dyadic mesh: the slice is exact
  std::vector<std::vector<double>> values;  // the parameters to slice at, per direction
};

std::vector<Config> configs() {
  return {
      {"2D, interior knots",
       {{clamped(2, {0.5}), 2}, {clamped(3, {0.5}), 3}},
       false,
       true,
       {{0.0, 0.375, 0.5, 0.9375, 1.0}, {0.0, 0.125, 0.5, 0.75, 1.0}}},
      {"2D, rational",
       {{clamped(2, {0.5}), 2}, {clamped(3, {0.5}), 3}},
       true,
       true,
       {{0.0, 0.375, 0.5, 1.0}, {0.125, 0.5, 0.75, 1.0}}},
      {"3D",
       {{clamped(2, {0.5}), 2}, {clamped(1, {0.5}), 1}, {clamped(2, {}), 2}},
       false,
       true,
       {{0.25, 0.5, 1.0}, {0.0, 0.5, 0.75}, {0.0, 0.5, 0.875}}},
      {"3D, rational",
       {{clamped(2, {0.5}), 2}, {clamped(1, {0.5}), 1}, {clamped(2, {}), 2}},
       true,
       true,
       {{0.25, 1.0}, {0.5}, {0.125, 1.0}}},
      {"a knot of multiplicity degree (C0) and one below it",
       {{clamped(2, {0.5, 0.5}), 2}, {clamped(3, {0.5, 0.5}), 3}},
       false,
       true,
       {{0.25, 0.5, 0.75}, {0.25, 0.5, 0.75}}},
      {"unclamped knots",
       {{{-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}, 2}, {clamped(2, {0.5}), 2}},
       false,
       true,
       {{0.0, 0.125, 0.25, 0.5, 1.0}, {0.25, 0.5}}},
      {"a periodic direction beside a clamped one",
       {{periodic_knots(4, 2), 2, true}, {clamped(2, {0.5}), 2}},
       false,
       true,
       {{0.0, 0.125, 0.25, 0.5, 0.875, 1.0}, {0.25, 0.5, 1.0}}},
      {"a periodic direction, sliced away, beside a periodic one",
       {{periodic_knots(4, 2), 2, true}, {periodic_knots(8, 2), 2, true}},
       true,
       true,
       {{0.0, 0.25, 0.375, 1.0}, {0.0, 0.25, 0.75, 1.0}}},
      {"asymmetric knots with a periodic direction (rounding allowed)",
       {{{-0.5, -0.25, 0.0, 0.125, 0.5, 0.75, 1.0, 1.125, 1.5}, 2, true},
        {clamped(3, {0.125, 0.5, 0.5}), 3}},
       false,
       false,
       {{0.0, 0.125, 0.3, 0.75, 1.0}, {0.0, 0.125, 0.5, 0.9}}},
      {"off-mesh parameters and knots (rounding allowed)",
       {{clamped(2, {0.3}), 2}, {clamped(3, {0.1, 0.7}), 3}},
       false,
       false,
       {{0.1, 0.3, 0.55, 0.9}, {0.1, 0.4, 0.7, 0.95}}},
      {"off-mesh, rational",
       {{clamped(2, {0.3}), 2}, {clamped(3, {0.1, 0.7}), 3}},
       true,
       false,
       {{0.1, 0.3, 0.9}, {0.2, 0.7}}},
  };
}

}  // namespace

TEST_CASE_TEMPLATE(
    "slice: equals the original at the fixed parameter, across every whole direction", T, double,
    float) {
  for (const Config& c : configs()) {
    CAPTURE(c.name);
    const bs::Bspline<T> b =
        make_spline<T>(c.dirs, 2, c.rational, 7U + static_cast<unsigned>(c.dirs.size()));
    for (std::size_t axis = 0; axis < c.dirs.size(); ++axis) {
      for (const double v : c.values[axis]) {
        CAPTURE(axis);
        CAPTURE(v);
        const bs::Bspline<T> s = bs::slice<T>(b, static_cast<long long>(axis), v);
        CHECK(s.dim() == b.dim() - 1);
        CHECK(s.rational() == b.rational());
        CHECK(s.cp_size() == b.cp_size());
        const double err = slice_error<T>(b, s, axis, v);
        if (c.dyadic) {
          CHECK(err == 0.0);
        } else {
          CHECK(err <= slice_tolerance<T>(b, axis));
        }
        // The directions left alone keep their spaces.
        for (std::size_t d = 0, k = 0; d < b.dim(); ++d) {
          if (d != axis) {
            CHECK(same_space(s.space().spaces()[k], b.space().spaces()[d]));
            ++k;
          }
        }
      }
    }
  }
}

TEST_CASE_TEMPLATE("slice: a knot of multiplicity degree or more is a control point, copied", T,
                   double, float) {
  // clamped(2, {0.5, 0.5}): the knot 0.5 has multiplicity 2 = degree; the slice at it in
  // direction 0 is the row of control points of the function that is 1 there, with no rounding at
  // all, so it is compared bit for bit with the stored row.
  const bs::Bspline<T> b =
      make_spline<T>({{clamped(2, {0.5, 0.5}), 2}, {clamped(2, {}), 2}}, 3, false, 3U);
  const bs::Bspline<T> s = bs::slice<T>(b, 0, 0.5);
  const std::span<const T> all = b.control_points();
  const std::span<const T> got = s.control_points();
  REQUIRE(got.size() == b.shape()[1] * b.shape()[2]);
  // The basis functions 0, 1, 2 live on [0, 0.5]; function 2 is the one equal to 1 at 0.5.
  for (std::size_t i = 0; i < got.size(); ++i) {
    CHECK(got[i] == all[2 * got.size() + i]);
  }
}

TEST_CASE_TEMPLATE("slice: a curve slices to a point, projected when rational", T, double, float) {
  for (const bool rational : {false, true}) {
    for (const double v : {0.0, 0.125, 0.5, 0.75, 1.0}) {
      CAPTURE(rational);
      CAPTURE(v);
      const bs::Bspline<T> b = make_spline<T>({{clamped(3, {0.25, 0.5}), 3}}, 3, rational, 21U);
      const pantr::NdArray<T> pt = bs::slice_point<T>(b, 0, v);
      REQUIRE(pt.size() == 3);
      const std::vector<Rational> x = {oracle::exact(v)};
      const std::vector<Rational> exact = oracle::exact_value<T>(b, x);
      for (std::size_t i = 0; i < 3; ++i) {
        const Rational want = rational ? exact[i] / exact[3] : exact[i];
        // The homogeneous slice is within the derived bound `delta`. Dividing by the weight (at
        // least 1 once `delta` is below 1) gives `(delta + |x| delta) / (1 - delta) <= 3 delta`
        // for `|x| <= 1`, and rounding the quotient adds at most `u |x| <= u`.
        const double delta = slice_tolerance<T>(b, 0);
        const double tol = rational ? 3.0 * delta + unit_roundoff<T>() : delta;
        CHECK(oracle::within(static_cast<double>(pt.data()[i]), want, tol));
      }
    }
  }
}

TEST_CASE_TEMPLATE("boundary: the slice at the start or the end of the domain", T, double, float) {
  for (const Config& c : configs()) {
    CAPTURE(c.name);
    const bs::Bspline<T> b = make_spline<T>(c.dirs, 2, c.rational, 11U);
    for (std::size_t axis = 0; axis < c.dirs.size(); ++axis) {
      for (const long long side : {0LL, 1LL}) {
        const auto dom = b.space().spaces()[axis].domain();
        const double v = static_cast<double>(side == 0 ? dom.first : dom.second);
        const bs::Bspline<T> by_boundary = bs::boundary<T>(b, static_cast<long long>(axis), side);
        const bs::Bspline<T> by_slice = bs::slice<T>(b, static_cast<long long>(axis), v);
        const auto a = by_boundary.control_points();
        const auto s = by_slice.control_points();
        REQUIRE(a.size() == s.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
          CHECK(a[i] == s[i]);
        }
      }
    }
  }
}

TEST_CASE_TEMPLATE("boundary: the end points of a curve", T, double, float) {
  const bs::Bspline<T> b = make_spline<T>({{clamped(2, {0.5}), 2}}, 2, true, 5U);
  const pantr::NdArray<T> start = bs::boundary_point<T>(b, 0, 0);
  const pantr::NdArray<T> end = bs::boundary_point<T>(b, 0, 1);
  // A clamped end is the end control point, projected.
  const std::span<const T> cp = b.control_points();
  const std::size_t last = b.shape()[0] - 1;
  CHECK(start.data()[0] == cp[0] / cp[2]);
  CHECK(start.data()[1] == cp[1] / cp[2]);
  CHECK(end.data()[0] == cp[last * 3] / cp[last * 3 + 2]);
  CHECK(end.data()[1] == cp[last * 3 + 1] / cp[last * 3 + 2]);
}

TEST_CASE_TEMPLATE("slice: a value inside the tolerance band beyond an end is that end", T, double,
                   float) {
  const bs::Bspline<T> b =
      make_spline<T>({{clamped(2, {0.5}), 2}, {clamped(2, {}), 2}}, 2, false, 9U);
  const double tol = b.space().spaces()[0].tolerance();
  REQUIRE(tol > 0.0);
  const bs::Bspline<T> at_end = bs::slice<T>(b, 0, 1.0);
  const bs::Bspline<T> beyond = bs::slice<T>(b, 0, 1.0 + 0.5 * tol);
  const bs::Bspline<T> before = bs::slice<T>(b, 0, 0.0 - 0.5 * tol);
  const bs::Bspline<T> at_start = bs::slice<T>(b, 0, 0.0);
  for (std::size_t i = 0; i < at_end.control_points().size(); ++i) {
    CHECK(beyond.control_points()[i] == at_end.control_points()[i]);
    CHECK(before.control_points()[i] == at_start.control_points()[i]);
  }
}

TEST_CASE("slice: refusals and their messages") {
  const bs::Bspline<double> surface =
      make_spline<double>({{clamped(2, {0.5}), 2}, {clamped(2, {}), 2}}, 2, false, 3U);
  const bs::Bspline<double> curve = make_spline<double>({{clamped(2, {0.5}), 2}}, 2, false, 3U);
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();

  CHECK(message_of([&] { (void)bs::slice<double>(surface, 2, 0.5); }) ==
        "axis must be in [0, 2), got 2.");
  CHECK(message_of([&] { (void)bs::slice<double>(surface, -1, 0.5); }) ==
        "axis must be in [0, 2), got -1.");
  CHECK(message_of([&] { (void)bs::slice<double>(surface, 0, 2.0); }) ==
        "value 2.0 is outside the domain [0.0, 1.0] of direction 0.");
  CHECK(message_of([&] { (void)bs::slice<double>(surface, 1, -0.25); }) ==
        "value -0.25 is outside the domain [0.0, 1.0] of direction 1.");
  CHECK(message_of([&] { (void)bs::slice<double>(surface, 0, inf); }) ==
        "value inf is outside the domain [0.0, 1.0] of direction 0.");
  // A NaN is outside every domain, and must not reach the corner cutting.
  CHECK(message_of([&] { (void)bs::slice<double>(surface, 0, nan); }) ==
        "value nan is outside the domain [0.0, 1.0] of direction 0.");
  CHECK_THROWS_AS((void)bs::slice<double>(surface, 0, nan), pantr::value_error);
  CHECK_THROWS_AS((void)bs::slice_point<double>(curve, 0, nan), pantr::value_error);
  CHECK(message_of([&] { (void)bs::slice_point<double>(curve, 0, 1.5); }) ==
        "value 1.5 is outside the domain [0.0, 1.0] of direction 0.");
  // The return type is the dimension's.
  CHECK(message_of([&] { (void)bs::slice<double>(curve, 0, 0.5); }) ==
        "slice: a curve has no lower-dimensional B-spline; use slice_point.");
  CHECK(message_of([&] { (void)bs::slice_point<double>(surface, 0, 0.5); }) ==
        "slice_point: only a curve slices to a point; use slice.");
  // The side is checked before the axis.
  CHECK(message_of([&] { (void)bs::boundary<double>(surface, 5, 2); }) ==
        "side must be 0 or 1, got 2.");
  CHECK(message_of([&] { (void)bs::boundary<double>(surface, 5, 1); }) ==
        "axis must be in [0, 2), got 5.");
  CHECK(message_of([&] { (void)bs::boundary_point<double>(curve, 0, -1); }) ==
        "side must be 0 or 1, got -1.");
}
