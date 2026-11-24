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
#include <pantr/bspline/evaluate.hpp>
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
using bspline_test::cast_all;
using bspline_test::clamped;
using bspline_test::Dir;
using bspline_test::make_spline;
using bspline_test::message_of;
using bspline_test::periodic_knots;
using bspline_test::same_space;
using bspline_test::sp;
using bspline_test::values;
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

TEST_CASE_TEMPLATE(
    "boundary: the end beside empty last spans is the last non-empty span's end function", T,
    double, float) {
  // With `degree + 2` or more equal knots ending at the domain end `knots[n]`, the last in-domain
  // spans are empty and the function of row `n - degree - 1` has empty support. Oracle, analytic:
  // at the end of the last non-empty span exactly one function is non-zero, the one whose knots
  // `knots[i + 1 .. i + degree + 1]` all equal the end (its value there, taken from the left, is
  // 1), so the boundary is that function's slab of control points. Every corner-cutting weight is
  // 1 or the step is skipped: the right knot of each weight equals the end `u`, so
  // `(u - l) / (u - l)` is exactly 1, and `1 * b + 0 * a` is exact for finite `a`. The slab is
  // therefore compared bit for bit.
  struct Case {
    std::vector<double> knots;
    int degree;
    std::size_t end_function;
  };
  // The first three have `degree + 2` equal end knots, the fourth `degree + 3`; the last
  // `degree + 1` before a trailing knot, an empty last span whose end was already right and is
  // now reached through the corner cutting rather than a copy.
  const std::vector<Case> cases = {
      {{0, 1, 1}, 0, 0},
      {{0, 0, 1, 2, 2, 2, 3}, 1, 2},
      {{0, 0, 0, 1, 2, 3, 3, 3, 3, 4, 4}, 2, 4},
      {{0, 0, 1, 2, 2, 2, 2, 3}, 1, 2},
      {{0, 0, 1, 2, 2, 3}, 1, 2},
  };
  for (const Case& c : cases) {
    for (const bool rational : {false, true}) {
      // The direction at each position of a surface and of a volume.
      for (const std::size_t dim : {std::size_t{2}, std::size_t{3}}) {
        for (std::size_t axis = 0; axis < dim; ++axis) {
          CAPTURE(c.degree);
          CAPTURE(rational);
          CAPTURE(dim);
          CAPTURE(axis);
          std::vector<Dir> dirs(dim, Dir{clamped(2, {0.5}), 2});
          dirs[axis] = Dir{c.knots, c.degree};
          const bs::Bspline<T> b = make_spline<T>(dirs, 2, rational, 13U);
          const double end = static_cast<double>(b.space().spaces()[axis].domain().second);
          const bs::Bspline<T> by_boundary = bs::boundary<T>(b, static_cast<long long>(axis), 1);
          const bs::Bspline<T> by_slice = bs::slice<T>(b, static_cast<long long>(axis), end);
          // The slab `end_function` of `axis`, in the order of the remaining directions.
          const std::span<const T> all = b.control_points();
          const std::span<const std::size_t> shape = b.shape();
          std::size_t outer = 1;
          for (std::size_t d = 0; d < axis; ++d) {
            outer *= shape[d];
          }
          const std::size_t inner = all.size() / (outer * shape[axis]);
          const std::span<const T> got_b = by_boundary.control_points();
          const std::span<const T> got_s = by_slice.control_points();
          REQUIRE(got_b.size() == outer * inner);
          REQUIRE(got_s.size() == outer * inner);
          for (std::size_t o = 0; o < outer; ++o) {
            for (std::size_t i = 0; i < inner; ++i) {
              const T want = all[(o * shape[axis] + c.end_function) * inner + i];
              CHECK(got_b[o * inner + i] == want);
              CHECK(got_s[o * inner + i] == want);
            }
          }
        }
      }
      // A curve: the end point is that function's control point, projected when rational.
      const bs::Bspline<T> crv = make_spline<T>({{c.knots, c.degree}}, 2, rational, 17U);
      const double end = static_cast<double>(crv.space().spaces()[0].domain().second);
      const pantr::NdArray<T> by_boundary = bs::boundary_point<T>(crv, 0, 1);
      const pantr::NdArray<T> by_slice = bs::slice_point<T>(crv, 0, end);
      const std::size_t cols = crv.cp_size();
      const T* cp = crv.control_points().data() + c.end_function * cols;
      for (std::size_t i = 0; i < 2; ++i) {
        const T want = rational ? cp[i] / cp[2] : cp[i];
        CHECK(by_boundary.data()[i] == want);
        CHECK(by_slice.data()[i] == want);
      }
    }
  }
}

TEST_CASE("boundary: the end of an unsnapped space beside a sub-tolerance knot gap") {
  // Built without snapping, a knot `2.2e-15` from the end stays apart from it, closer than the
  // space's tolerance. The end still takes the last span of non-zero length, `k`, and no knot at
  // or left of `k` equals the end, so the corner cutting runs every round. Oracle, analytic: the
  // one function equal to 1 at the end from the left is `k`'s, since `knots[k] < u` and
  // `knots[k + 1 .. k + degree]` all equal the end `u`. Those are the right knots of every
  // corner-cutting weight, so each weight is `(u - l) / (u - l)`, exactly 1, and the result is
  // that function's control point bit for bit. Double only: a gap this small rounds away in float.
  constexpr double gap = 2.2e-15;
  struct Case {
    std::vector<double> knots;
    int degree;
    std::size_t end_function;
  };
  const std::vector<Case> cases = {
      // An empty last span, beside a span shorter than the tolerance.
      {{0, 0, 1, 2 - gap, 2, 2, 3}, 1, 3},
      {{0, 0, 0, 1, 2 - gap, 2, 2, 2, 3}, 2, 4},
      // A last span shorter than the tolerance.
      {{0, 0, 1, 1 + gap, 2}, 1, 2},
  };
  for (const Case& c : cases) {
    const bs::BsplineSpace1D<double> space(sp(c.knots), c.degree, false, false);
    const double end = space.domain().second;
    for (std::size_t j = 1; j <= static_cast<std::size_t>(c.degree); ++j) {
      REQUIRE(c.knots[c.end_function + j] == end);
    }
    REQUIRE(c.knots[c.end_function] < end);
    REQUIRE(end - c.knots[c.end_function] < space.tolerance());
    for (const bool rational : {false, true}) {
      CAPTURE(c.degree);
      CAPTURE(c.end_function);
      CAPTURE(rational);
      const std::size_t n = static_cast<std::size_t>(space.num_basis());
      const std::size_t cols = 2 + (rational ? 1 : 0);
      std::vector<double> flat = values(n * cols, 23U);
      if (rational) {
        for (std::size_t i = 0; i < n; ++i) {
          flat[i * cols + 2] = 1.5 + flat[i * cols + 2] / 2.0;
        }
      }
      const bs::Bspline<double> crv(
          bs::BsplineSpace<double>(std::vector<bs::BsplineSpace1D<double>>{space}), sp(flat),
          rational);
      const pantr::NdArray<double> by_boundary = bs::boundary_point<double>(crv, 0, 1);
      const pantr::NdArray<double> by_slice = bs::slice_point<double>(crv, 0, end);
      const double* cp = flat.data() + c.end_function * cols;
      for (std::size_t i = 0; i < 2; ++i) {
        const double want = rational ? cp[i] / cp[2] : cp[i];
        CHECK(by_boundary.data()[i] == want);
        CHECK(by_slice.data()[i] == want);
      }
    }
  }
}

TEST_CASE_TEMPLATE("boundary: the start of an unsnapped space beside a sub-tolerance knot gap", T,
                   double, float) {
  // Built without snapping, a knot `1e-15` from the start stays apart from it, closer than the
  // space's tolerance; evaluation takes it as a distinct knot, and so must the slice. Oracle,
  // analytic, from the knot vector. The span at the start `u` is `k`, the last knot equal to `u`
  // (`knots[k] == u < knots[k + 1]`). Where `degree` or more knots at or left of `k` equal `u`,
  // the one non-zero function there is `k - degree`'s and the slice is its control point, copied.
  // In the last case a single knot equals `u`, between `-gap` and `gap`: at degree 2 the functions
  // `0` and `1` are each 1/2 there, the one corner-cutting weight is `gap / (2 gap)`, exactly 1/2,
  // and the slice is `(P0 + P1) / 2`, exact for the dyadic control points. All are compared bit
  // for bit. Classed with the tolerance, each takes the knot at `gap` for a copy of the start and
  // copies row 1; with the span right but the multiplicity counted with the tolerance, the
  // `halfway` case takes `-gap` for a copy and copies row 0. The slices of a surface are also
  // compared with the exact evaluation of the surface at the start (`slice_error`), and the curve's
  // start point with `evaluate`, within the file's derived bound.
  constexpr double gap = 1e-15;
  struct Case {
    std::vector<double> knots;
    int degree;
    bool halfway;  // The slice is the mean of rows 0 and 1, not row 0.
  };
  const std::vector<Case> cases = {
      {{0, gap, 1}, 0, false},
      {{0, 0, gap, 1, 1}, 1, false},
      {{0, 0, 0, gap, 1, 2, 2, 2}, 2, false},
      {{-1, -gap, 0, gap, 1, 2, 2, 2}, 2, true},
  };
  const auto expected = [](T row0, T row1, bool halfway) {
    return halfway ? (row0 + row1) / T{2} : row0;
  };
  for (const Case& c : cases) {
    const std::vector<T> knots = cast_all<T>(c.knots);
    const bs::BsplineSpace1D<T> space(sp(knots), c.degree, false, false);
    const double start = static_cast<double>(space.domain().first);
    const double tol = space.tolerance();
    // The premise: a knot within the tolerance of the start and distinct from it (and in the last
    // case one on each side).
    REQUIRE(start == 0.0);
    REQUIRE(std::equal(space.knots().begin(), space.knots().end(), knots.begin(), knots.end()));
    const auto above = std::upper_bound(knots.begin(), knots.end(), T{0});
    REQUIRE(above != knots.end());
    REQUIRE(static_cast<double>(*above) > start);
    REQUIRE(static_cast<double>(*above) - start < tol);
    if (c.halfway) {
      const auto first = std::lower_bound(knots.begin(), knots.end(), T{0});
      REQUIRE(first != knots.begin());
      REQUIRE(static_cast<double>(*(first - 1)) < start);
      REQUIRE(start - static_cast<double>(*(first - 1)) < tol);
    }
    for (const bool rational : {false, true}) {
      CAPTURE(c.degree);
      CAPTURE(c.halfway);
      CAPTURE(rational);
      const std::size_t cols = 2 + (rational ? 1 : 0);
      const auto control_points = [&](std::size_t count, unsigned seed) {
        std::vector<double> flat = values(count * cols, seed);
        if (rational) {
          for (std::size_t i = 0; i < count; ++i) {
            flat[i * cols + 2] = 1.5 + flat[i * cols + 2] / 2.0;
          }
        }
        return cast_all<T>(flat);
      };
      // The direction at each position of a surface and of a volume, the others clamped.
      const std::vector<T> other = cast_all<T>(clamped(2, {0.5}));
      for (const std::size_t dim : {std::size_t{2}, std::size_t{3}}) {
        for (std::size_t axis = 0; axis < dim; ++axis) {
          CAPTURE(dim);
          CAPTURE(axis);
          std::vector<bs::BsplineSpace1D<T>> spaces;
          std::vector<std::size_t> shape;
          std::size_t total = 1;
          for (std::size_t d = 0; d < dim; ++d) {
            spaces.push_back(d == axis ? space : bs::BsplineSpace1D<T>(sp(other), 2));
            shape.push_back(static_cast<std::size_t>(spaces.back().num_basis()));
            total *= shape.back();
          }
          const std::vector<T> cp = control_points(total, 29U);
          const bs::Bspline<T> b(bs::BsplineSpace<T>(std::move(spaces)), sp(cp), rational);
          const auto ax = static_cast<long long>(axis);
          // A value within the tolerance of the start, either side, is the start.
          const std::vector<bs::Bspline<T>> got = {
              bs::boundary<T>(b, ax, 0), bs::slice<T>(b, ax, start),
              bs::slice<T>(b, ax, start + 0.5 * tol), bs::slice<T>(b, ax, start - 0.5 * tol)};
          std::size_t outer = 1;
          for (std::size_t d = 0; d < axis; ++d) {
            outer *= shape[d];
          }
          const std::size_t inner = cp.size() / (outer * shape[axis]);
          // Rows 0 and 1 along `axis` differ, so copying the wrong one cannot pass.
          bool rows_differ = false;
          for (std::size_t o = 0; o < outer; ++o) {
            for (std::size_t i = 0; i < inner; ++i) {
              const std::size_t row0 = o * shape[axis] * inner + i;
              rows_differ = rows_differ || cp[row0] != cp[row0 + inner];
            }
          }
          REQUIRE(rows_differ);
          for (std::size_t which = 0; which < got.size(); ++which) {
            CAPTURE(which);  // boundary, slice at the start, at start + tol / 2, at start - tol / 2
            const bs::Bspline<T>& g = got[which];
            REQUIRE(g.control_points().size() == outer * inner);
            for (std::size_t o = 0; o < outer; ++o) {
              for (std::size_t i = 0; i < inner; ++i) {
                const std::size_t row0 = o * shape[axis] * inner + i;
                CHECK(g.control_points()[o * inner + i] ==
                      expected(cp[row0], cp[row0 + inner], c.halfway));
              }
            }
            if (dim == 2) {
              CHECK(slice_error<T>(b, g, axis, start) <= slice_tolerance<T>(b, axis));
            }
          }
        }
      }
      // A curve: the start point, projected when rational.
      const std::vector<T> cp = control_points(static_cast<std::size_t>(space.num_basis()), 31U);
      // Rows 0 and 1 differ, so copying the wrong one cannot pass.
      REQUIRE(!std::equal(cp.begin(), cp.begin() + static_cast<std::ptrdiff_t>(cols),
                          cp.begin() + static_cast<std::ptrdiff_t>(cols)));
      const bs::Bspline<T> crv(bs::BsplineSpace<T>(std::vector<bs::BsplineSpace1D<T>>{space}),
                               sp(cp), rational);
      const pantr::NdArray<T> by_boundary = bs::boundary_point<T>(crv, 0, 0);
      const pantr::NdArray<T> by_slice = bs::slice_point<T>(crv, 0, start);
      std::vector<T> row(cols);
      for (std::size_t i = 0; i < cols; ++i) {
        row[i] = expected(cp[i], cp[cols + i], c.halfway);
      }
      const T at = static_cast<T>(start);
      const pantr::NdArray<T> by_evaluate =
          bs::evaluate<T, T>(crv, pantr::mdview<const T, 2>(&at, 1, 1));
      const double bound = slice_tolerance<T>(crv, 0);
      for (std::size_t i = 0; i < 2; ++i) {
        const T want = rational ? row[i] / row[2] : row[i];
        CHECK(by_boundary.data()[i] == want);
        CHECK(by_slice.data()[i] == want);
        CHECK(std::abs(static_cast<double>(by_evaluate.data()[i]) - static_cast<double>(want)) <=
              bound);
      }
    }
  }
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
