// The Bézier decomposition of a B-spline (`to_bezier`, `to_beziers`).
//
// Oracle. A decomposition is compared with the spline it decomposes, both evaluated exactly: each
// Bézier patch, in the exact Bernstein form of its stored control points, against the exact
// Cox-de Boor sum of the spline at the same point. The extraction along one direction forms convex
// combinations (the columns of an operator sum to one) of `p + 1` control points with operator
// entries that are off by `4 p (p + 1) u`, and rounds the sum of `p + 1` products (`(p + 1) u` of
// the terms), so each direction adds `(p + 1) (4 p (p + 1) + 2) u max|P|`.
#include <doctest/doctest.h>

#include "bspline_bezier_exact.hpp"
#include "bspline_test_support.hpp"

#include <pantr/bezier/bezier.hpp>
#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/to_beziers.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/parallel/pool.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace bs = pantr::bspline;
namespace oracle = bspline_oracle;
using bspline_test::bezier_exact;
using bspline_test::clamped;
using bspline_test::Dir;
using bspline_test::make_spline;
using bspline_test::message_of;
using bspline_test::periodic_knots;
using bspline_test::unit_roundoff;
using oracle::Rational;

namespace {

/// The largest exact difference between the Bézier of element `flat` (C-order index) and the
/// spline, over three samples across the whole reference interval per direction (none at an end),
/// mapped into the element.
template <class T>
double element_error(const bs::Bspline<T>& b, const bs::BezierDecomposition<T>& dec,
                     std::size_t flat) {
  const std::size_t dim = b.dim();
  const bs::BsplineSpace<T> whole = b.space();
  // The multi-index of the element.
  std::vector<std::size_t> e(dim);
  std::size_t rest = flat;
  for (std::size_t d = dim; d-- > 0;) {
    e[d] = rest % dec.extents[d];
    rest /= dec.extents[d];
  }
  std::vector<Rational> lo(dim);
  std::vector<Rational> hi(dim);
  for (std::size_t d = 0; d < dim; ++d) {
    const auto uk = whole.spaces()[d].unique_knots_and_multiplicity(true).first;
    lo[d] = oracle::exact(static_cast<double>(uk[e[d]]));
    hi[d] = oracle::exact(static_cast<double>(uk[e[d] + 1]));
  }
  const std::vector<Rational> xi = {Rational(1, 8), Rational(1, 2), Rational(7, 8)};
  double worst = 0.0;
  std::vector<std::size_t> idx(dim, 0);
  for (;;) {
    std::vector<Rational> t(dim);
    std::vector<Rational> x(dim);
    for (std::size_t d = 0; d < dim; ++d) {
      t[d] = xi[idx[d]];
      x[d] = lo[d] + (hi[d] - lo[d]) * t[d];
    }
    const auto got = bezier_exact<T>(dec.beziers[flat], t);
    const auto want = oracle::exact_value<T>(b, x);
    REQUIRE(got.size() == want.size());
    for (std::size_t c = 0; c < got.size(); ++c) {
      Rational diff = got[c] - want[c];
      if (diff < 0) {
        diff = -diff;
      }
      worst = (std::max)(worst, static_cast<double>(diff));
    }
    std::size_t d = dim;
    bool done = true;
    while (d-- > 0) {
      if (++idx[d] < xi.size()) {
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
double decomposition_tolerance(const bs::Bspline<T>& b) {
  double total = 0.0;
  for (const std::size_t p : b.degrees()) {
    const double pd = static_cast<double>(p);
    total += (pd + 1.0) * (4.0 * pd * (pd + 1.0) + 2.0) * unit_roundoff<T>();
  }
  return total * (std::max)(1.0, oracle::max_abs_control_point<T>(b));
}

struct Config {
  const char* name;
  std::vector<Dir> dirs;
  bool rational;
};

std::vector<Config> decomposition_configs() {
  return {
      {"curve", {{clamped(2, {0.25, 0.5, 0.75}), 2}}, false},
      {"rational curve with a double knot", {{clamped(3, {0.25, 0.5, 0.5}), 3}}, true},
      {"curve of a knot with multiplicity degree", {{clamped(2, {0.5, 0.5}), 2}}, false},
      {"unclamped curve", {{{-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}, 2}}, false},
      {"periodic curve", {{periodic_knots(4, 2), 2, true}}, false},
      {"surface", {{clamped(2, {0.5}), 2}, {clamped(3, {0.25, 0.75}), 3}}, false},
      {"rational surface, one direction unclamped",
       {{clamped(2, {0.5}), 2}, {{-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}, 2}},
       true},
      {"surface, a periodic and a clamped direction",
       {{periodic_knots(4, 2), 2, true}, {clamped(2, {0.5}), 2}},
       false},
      {"volume",
       {{clamped(2, {0.5}), 2}, {clamped(1, {0.5}), 1}, {clamped(2, {0.25, 0.75}), 2}},
       false},
      {"rational volume, off-mesh",
       {{clamped(2, {0.3}), 2}, {clamped(2, {0.6}), 2}, {clamped(1, {0.45}), 1}},
       true},
  };
}

}  // namespace

TEST_CASE_TEMPLATE("to_beziers: every patch equals the spline on its element", T, double, float) {
  for (const Config& c : decomposition_configs()) {
    CAPTURE(c.name);
    const bs::Bspline<T> b = make_spline<T>(c.dirs, 2, c.rational, 13U);
    const bs::BezierDecomposition<T> dec = bs::to_beziers<T>(b);
    const bs::BsplineSpace<T> whole = b.space();
    REQUIRE(dec.extents.size() == b.dim());
    std::size_t count = 1;
    for (std::size_t d = 0; d < b.dim(); ++d) {
      CHECK(dec.extents[d] == static_cast<std::size_t>(whole.spaces()[d].num_intervals()));
      count *= dec.extents[d];
    }
    REQUIRE(dec.beziers.size() == count);
    const double tol = decomposition_tolerance<T>(b);
    for (std::size_t n = 0; n < count; ++n) {
      const pantr::bezier::Bezier<T>& z = dec.beziers[n];
      CHECK(z.rational() == b.rational());
      CHECK(z.cp_size() == b.cp_size());
      for (std::size_t d = 0; d < b.dim(); ++d) {
        CHECK(z.degree(d) == b.degrees()[d]);
      }
      CHECK(element_error<T>(b, dec, n) <= tol);
    }
    // Each patch owns its control points: none points into the spline's, and none into another's.
    for (std::size_t n = 0; n < count; ++n) {
      CHECK(dec.beziers[n].control_points().data() != b.control_points().data());
      for (std::size_t m = n + 1; m < count; ++m) {
        CHECK(dec.beziers[n].control_points().data() != dec.beziers[m].control_points().data());
      }
    }
  }
}

TEST_CASE_TEMPLATE("to_beziers: a Bezier-like spline decomposes to itself, exactly", T, double,
                   float) {
  // The extraction operator is the identity: the patch is the spline's own control net, but for
  // the zeros and ones that multiplying by the identity leaves exact.
  const bs::Bspline<T> b = make_spline<T>({{clamped(3, {}), 3}, {clamped(2, {}), 2}}, 3, false, 5U);
  const bs::BezierDecomposition<T> dec = bs::to_beziers<T>(b);
  REQUIRE(dec.beziers.size() == 1);
  const auto got = dec.beziers[0].control_points();
  const auto want = b.control_points();
  REQUIRE(got.size() == want.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    CHECK(got[i] == want[i]);
  }
}

TEST_CASE("to_beziers: the result does not depend on the number of threads") {
  // 64 x 64 elements of degree (3, 2): each direction's application is far above the work that
  // makes the pool worth waking, so the multi-threaded run takes the parallel path.
  std::vector<double> knots3;
  std::vector<double> knots2;
  for (int i = 1; i < 64; ++i) {
    knots3.push_back(static_cast<double>(i) / 64.0);
    knots2.push_back(static_cast<double>(i) / 64.0);
  }
  const bs::Bspline<double> b =
      make_spline<double>({{clamped(3, knots3), 3}, {clamped(2, knots2), 2}}, 3, true, 17U);
  const std::size_t saved = pantr::parallel::get_num_threads();
  pantr::parallel::set_num_threads(1);
  const bs::BezierDecomposition<double> serial = bs::to_beziers<double>(b);
  pantr::parallel::set_num_threads(static_cast<std::int64_t>(pantr::parallel::max_threads()));
  const bs::BezierDecomposition<double> parallel = bs::to_beziers<double>(b);
  pantr::parallel::set_num_threads(static_cast<std::int64_t>(saved));
  REQUIRE(serial.beziers.size() == 64U * 64U);
  REQUIRE(parallel.beziers.size() == serial.beziers.size());
  bool same = true;
  for (std::size_t n = 0; n < serial.beziers.size(); ++n) {
    const auto a = serial.beziers[n].control_points();
    const auto c = parallel.beziers[n].control_points();
    same = same && std::equal(a.begin(), a.end(), c.begin(), c.end());
  }
  CHECK(same);
  // And the serial result is right: corner, middle and last elements, against the exact spline.
  const double tol = decomposition_tolerance<double>(b);
  for (const std::size_t n :
       {std::size_t{0}, std::size_t{17 * 64 + 40}, std::size_t{64 * 64 - 1}}) {
    CHECK(element_error<double>(b, serial, n) <= tol);
  }
}

TEST_CASE("to_beziers: a direction with one element and a long rest does not depend on threads") {
  // 1024 elements in the first direction and one in the second: along the second, the loop over
  // elements has a single item per slab, and the work is in the `4096` slabs of the first
  // direction's stacked points. The items are the (element, slab) pairs, so the pool has work to
  // share; the result must be bit for bit the single-thread one.
  std::vector<double> many;
  for (int i = 1; i < 1024; ++i) {
    many.push_back(static_cast<double>(i) / 1024.0);
  }
  const bs::Bspline<double> b =
      make_spline<double>({{clamped(3, many), 3}, {clamped(2, {}), 2}}, 2, true, 21U);
  const std::size_t saved = pantr::parallel::get_num_threads();
  pantr::parallel::set_num_threads(1);
  const bs::BezierDecomposition<double> serial = bs::to_beziers<double>(b);
  pantr::parallel::set_num_threads(static_cast<std::int64_t>(pantr::parallel::max_threads()));
  const bs::BezierDecomposition<double> parallel = bs::to_beziers<double>(b);
  pantr::parallel::set_num_threads(static_cast<std::int64_t>(saved));
  REQUIRE(serial.beziers.size() == 1024U);
  REQUIRE(parallel.beziers.size() == serial.beziers.size());
  bool same = true;
  for (std::size_t n = 0; n < serial.beziers.size(); ++n) {
    const auto x = serial.beziers[n].control_points();
    const auto y = parallel.beziers[n].control_points();
    same = same && std::equal(x.begin(), x.end(), y.begin(), y.end());
  }
  CHECK(same);
  const double tol = decomposition_tolerance<double>(b);
  for (const std::size_t n : {std::size_t{0}, std::size_t{511}, std::size_t{1023}}) {
    CHECK(element_error<double>(b, serial, n) <= tol);
  }
}

TEST_CASE_TEMPLATE(
    "to_bezier: a single-element spline converts, sharing its buffer only on request", T, double,
    float) {
  const bs::Bspline<T> b = make_spline<T>({{clamped(3, {}), 3}, {clamped(2, {}), 2}}, 2, true, 3U);
  // A copy owns its points.
  const pantr::bezier::Bezier<T> copied = bs::to_bezier<T>(b, true);
  CHECK(copied.control_points().data() != b.control_points().data());
  CHECK(std::equal(copied.control_points().begin(), copied.control_points().end(),
                   b.control_points().begin()));
  CHECK(copied.rational());
  CHECK(copied.degree(0) == 3);
  CHECK(copied.degree(1) == 2);
  // Without a copy the Bézier points into the spline's buffer and keeps its representation alive.
  const long before = b.rep().use_count();
  {
    const pantr::bezier::Bezier<T> shared = bs::to_bezier<T>(b, false);
    CHECK(shared.control_points().data() == b.control_points().data());
    CHECK(b.rep().use_count() == before + 1);
    const pantr::bezier::Bezier<T> twin = shared;
    CHECK(b.rep().use_count() == before + 1);  // copying the Bézier shares its representation
    CHECK(twin.control_points().data() == b.control_points().data());
  }
  CHECK(b.rep().use_count() == before);

  // The Bézier outlives the spline it shares with.
  std::vector<T> expected;
  const pantr::bezier::Bezier<T> survivor = [&] {
    const bs::Bspline<T> temporary =
        make_spline<T>({{clamped(3, {}), 3}, {clamped(2, {}), 2}}, 2, true, 3U);
    expected.assign(temporary.control_points().begin(), temporary.control_points().end());
    return bs::to_bezier<T>(temporary, false);
  }();
  const auto got = survivor.control_points();
  REQUIRE(got.size() == expected.size());
  CHECK(std::equal(got.begin(), got.end(), expected.begin()));
}

TEST_CASE_TEMPLATE("to_bezier: periodic and unclamped single elements go through open form", T,
                   double, float) {
  // One element with unclamped ends: the knots -1.. around [0, 1] of a quadratic.
  const bs::Bspline<T> unclamped =
      make_spline<T>({{{-1.0, 0.0, 1.0, 2.0, 3.0, 4.0}, 2}}, 2, false, 7U);
  for (const bool copy : {true, false}) {
    const pantr::bezier::Bezier<T> z = bs::to_bezier<T>(unclamped, copy);
    const bs::BezierDecomposition<T> dec = bs::to_beziers<T>(unclamped);
    REQUIRE(dec.beziers.size() == 1);
    CHECK(std::equal(z.control_points().begin(), z.control_points().end(),
                     dec.beziers[0].control_points().begin()));
    // Against the exact spline on its one element.
    CHECK(element_error<T>(unclamped, dec, 0) <= decomposition_tolerance<T>(unclamped));
  }
}

TEST_CASE_TEMPLATE("to_bezier: a periodic single element goes through open form, at any degree", T,
                   double, float) {
  // One interval, periodic: the knots hold `2 degree + 2` entries, so the space needs a closure of
  // multiplicity `degree + 1` to have `degree + 1` functions, and the stored control points are the
  // open form's in number. What this case pins is the path: the periodic spline is converted to
  // open form, and the Bézier shares the converted spline's points (its shape is the converted
  // spline's, which here equals the stored one, so a stored-shape mix-up cannot show).
  struct Case {
    const char* name;
    std::vector<double> knots;
    int degree;
  };
  const std::vector<Case> cases = {
      {"degree 0", {0.0, 1.0}, 0},
      {"degree 1", {0.0, 0.0, 1.0, 1.0}, 1},
      {"degree 2", {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}, 2},
      {"degree 3", clamped(3, {}), 3},
  };
  for (const Case& c : cases) {
    CAPTURE(c.name);
    const bs::Bspline<T> b = make_spline<T>({{c.knots, c.degree, true}}, 2, false, 5U);
    const bs::BezierDecomposition<T> dec = bs::to_beziers<T>(b);
    REQUIRE(dec.beziers.size() == 1);
    const std::size_t open_points = static_cast<std::size_t>(c.degree) + 1;
    CHECK(dec.beziers[0].shape()[0] == open_points);
    for (const bool copy : {true, false}) {
      CAPTURE(copy);
      const pantr::bezier::Bezier<T> z = bs::to_bezier<T>(b, copy);
      REQUIRE(z.shape().size() == 2);
      CHECK(z.shape()[0] == open_points);
      CHECK(z.shape()[1] == b.cp_size());
      CHECK(z.degree(0) == static_cast<std::size_t>(c.degree));
      REQUIRE(z.control_points().size() == dec.beziers[0].control_points().size());
      CHECK(std::equal(z.control_points().begin(), z.control_points().end(),
                       dec.beziers[0].control_points().begin()));
    }
    CHECK(element_error<T>(b, dec, 0) <= decomposition_tolerance<T>(b));
  }
}

TEST_CASE("to_bezier: more than one element is refused") {
  const bs::Bspline<double> open = make_spline<double>({{clamped(2, {0.5}), 2}}, 2, false, 3U);
  const std::string want =
      "B-spline has more than one element per direction and cannot be represented as a single "
      "B\xC3\xA9zier.";
  CHECK(message_of([&] { (void)bs::to_bezier<double>(open, true); }) == want);
  CHECK(message_of([&] { (void)bs::to_bezier<double>(open, false); }) == want);
  const bs::Bspline<double> periodic =
      make_spline<double>({{periodic_knots(4, 2), 2, true}}, 2, false, 3U);
  CHECK(message_of([&] { (void)bs::to_bezier<double>(periodic, true); }) == want);
  const bs::Bspline<double> unclamped =
      make_spline<double>({{{-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}, 2}}, 2, false, 3U);
  CHECK(message_of([&] { (void)bs::to_bezier<double>(unclamped, true); }) == want);
}
