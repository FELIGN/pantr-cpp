// Bspline point inversion (`locate`). Oracles, each from a route other than the solver:
//  - analytic inverses of maps built from their blossoms (u^2 and v^3 by their polar forms, an
//    affine map by its Greville interpolant, a degree-1 rational map by its closed-form inverse),
//    so the expected coordinate is `sqrt`, `cbrt` or a linear solve, never the Newton loop;
//  - points constructed on knot lines and at the domain ends, whose cell is fixed by the tie rule
//    (an interior breakpoint belongs to the lower cell) and checked against the cell boxes;
//  - points constructed outside the image, which must come back -1 and NaN;
//  - thread-count invariance, compared at one thread and min(8, capacity).
// A coordinate is compared within `accept / sigma_min + rounding`: the residual is at most the
// acceptance threshold, and the map stretches by at least `sigma_min` (stated per test).
#include <doctest/doctest.h>

#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/locate.hpp>
#include <pantr/bspline/space.hpp>
#include <pantr/bspline/space_1d.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/parallel/parallel_for.hpp>
#include <pantr/tolerance.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace bs = pantr::bspline;
namespace lk = pantr::bspline::detail::locate_kernels;
namespace par = pantr::parallel;

namespace {

constexpr double eps = std::numeric_limits<double>::epsilon();

/// Restores the calling thread's parallelism setting on scope exit.
class ThreadsGuard {
 public:
  explicit ThreadsGuard(std::size_t n) : previous_(par::get_num_threads()) {
    par::set_num_threads(static_cast<std::int64_t>(n));
  }
  ~ThreadsGuard() { par::set_num_threads(static_cast<std::int64_t>(previous_)); }
  ThreadsGuard(const ThreadsGuard&) = delete;
  ThreadsGuard& operator=(const ThreadsGuard&) = delete;

 private:
  std::size_t previous_;
};

template <class T>
bs::BsplineSpace1D<T> space1d(const std::vector<double>& knots, int degree, bool periodic = false) {
  const std::vector<T> k(knots.begin(), knots.end());
  return bs::BsplineSpace1D<T>(std::span<const T>(k), degree, periodic);
}

/// The control points of `u^p` on a clamped knot vector: the blossom of `u^p` is the product of
/// its `p` arguments, evaluated at the knots `t[i+1..i+p]` (Ramshaw's polar form).
std::vector<double> power_cps(const std::vector<double>& t, int p) {
  const std::size_t n = t.size() - static_cast<std::size_t>(p) - 1;
  std::vector<double> cp(n);
  for (std::size_t i = 0; i < n; ++i) {
    double v = 1.0;
    for (int j = 1; j <= p; ++j) {
      v *= t[i + static_cast<std::size_t>(j)];
    }
    cp[i] = v;
  }
  return cp;
}

/// The 1D map `F(u) = u^2` on `[0.5, 2]`, interior knots 1 and 1.5 (three cells).
bs::Bspline<double> square_map() {
  const std::vector<double> t{0.5, 0.5, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0};
  const std::vector<double> cp = power_cps(t, 2);
  return {bs::BsplineSpace<double>({space1d<double>(t, 2)}), std::span<const double>(cp), false};
}

bs::LocateResult run(const bs::LocateContext& ctx, const std::vector<double>& pts,
                     std::optional<double> tol = std::nullopt, std::int64_t max_iter = 30) {
  const std::size_t rank = ctx.spline().rank();
  return bs::locate(ctx, pts, pts.size() / rank, tol, max_iter);
}

}  // namespace

TEST_CASE("locate inverts u^2 by sqrt, with the tie rule on the knot lines") {
  const bs::LocateContext ctx(square_map());
  // Points: interior of each cell, the two interior knots (u = 1, 1.5), the two domain ends.
  const std::vector<double> u{0.7, 1.2, 1.8, 1.0, 1.5, 0.5, 2.0};
  std::vector<double> x(u.size());
  for (std::size_t i = 0; i < u.size(); ++i) {
    x[i] = u[i] * u[i];
  }
  const bs::LocateResult r = run(ctx, x);
  const bs::LocateThresholds th = ctx.default_thresholds();
  // sigma_min = min F' = 2 * 0.5 = 1; sqrt and the squaring above add a few ulps.
  const double bound = th.accept / 1.0 + 8.0 * eps * 2.0;
  for (std::size_t i = 0; i < u.size(); ++i) {
    CAPTURE(i);
    CHECK(std::abs(r.ref_coords.data()[i] - std::sqrt(x[i])) <= bound);
  }
  // Cells [0.5, 1], [1, 1.5], [1.5, 2]: an interior knot belongs to the lower cell, an end to its
  // own boundary cell. A coordinate within `bound` of a knot may land on either side of it, so the
  // tie is checked through the returned coordinate itself.
  const std::vector<std::int64_t> interior{0, 1, 2};
  for (std::size_t i = 0; i < 3; ++i) {
    CHECK(r.cell_ids.data()[i] == interior[i]);
  }
  for (std::size_t i = 3; i < 5; ++i) {
    const double got = r.ref_coords.data()[i];
    const double knot = u[i];
    const std::int64_t lower = i == 3 ? 0 : 1;
    CHECK(r.cell_ids.data()[i] == (got <= knot ? lower : lower + 1));
  }
  CHECK(r.cell_ids.data()[5] == 0);
  CHECK(r.cell_ids.data()[6] == 2);
}

TEST_CASE("locate pins the tie rule on a knot line the solver lands on exactly") {
  // F(u) = u on [0, 2] with an interior knot at 1, degree 1: the identity, so u = 1 is reached
  // exactly (the midpoint start of either cell is one Newton step from it).
  const std::vector<double> t{0.0, 0.0, 1.0, 2.0, 2.0};
  const std::vector<double> cp{0.0, 1.0, 2.0};
  const bs::Bspline<double> b(bs::BsplineSpace<double>({space1d<double>(t, 1)}),
                              std::span<const double>(cp), false);
  const bs::LocateResult r = run(bs::LocateContext(b), {1.0, 0.0, 2.0});
  CHECK(r.ref_coords.data()[0] == 1.0);
  CHECK(r.cell_ids.data()[0] == 0);  // the lower cell
  CHECK(r.cell_ids.data()[1] == 0);
  CHECK(r.cell_ids.data()[2] == 1);
}

TEST_CASE("locate reports points outside the image as -1 and NaN") {
  const bs::LocateContext ctx(square_map());
  // The image is [0.25, 4]; 0.2 and 4.5 are outside it by far more than any threshold.
  const bs::LocateResult r = run(ctx, {0.2, 4.5, -1.0, 100.0});
  for (std::size_t i = 0; i < 4; ++i) {
    CHECK(r.cell_ids.data()[i] == -1);
    CHECK(std::isnan(r.ref_coords.data()[i]));
  }
}

TEST_CASE("locate inverts a separable surface (u^2, v^3) by sqrt and cbrt") {
  const std::vector<double> tu{1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0};
  const std::vector<double> tv{0.5, 0.5, 0.5, 0.5, 1.0, 1.5, 1.5, 1.5, 1.5};
  const std::vector<double> cu = power_cps(tu, 2);
  const std::vector<double> cv = power_cps(tv, 3);
  std::vector<double> cp;
  for (const double a : cu) {
    for (const double c : cv) {
      cp.push_back(a);
      cp.push_back(c);
    }
  }
  const bs::Bspline<double> b(
      bs::BsplineSpace<double>({space1d<double>(tu, 2), space1d<double>(tv, 3)}),
      std::span<const double>(cp), false);
  const bs::LocateContext ctx(b);
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> du(1.0, 2.0);
  std::uniform_real_distribution<double> dv(0.5, 1.5);
  std::vector<double> x;
  for (int i = 0; i < 50; ++i) {
    const double u = du(rng);
    const double v = dv(rng);
    x.push_back(u * u);
    x.push_back(v * v * v);
  }
  // Corners and knot lines too.
  for (const double u : {1.0, 1.5, 2.0}) {
    for (const double v : {0.5, 1.0, 1.5}) {
      x.push_back(u * u);
      x.push_back(v * v * v);
    }
  }
  const bs::LocateResult r = run(ctx, x);
  const double accept = ctx.default_thresholds().accept;
  // sigma_min = min(2u, 3v^2) over the domain = min(2, 0.75); sqrt/cbrt and the powers add ulps.
  const double bound = accept / 0.75 + 16.0 * eps * 2.0;
  const std::size_t n = x.size() / 2;
  for (std::size_t i = 0; i < n; ++i) {
    CAPTURE(i);
    CHECK(std::abs(r.ref_coords.data()[2 * i] - std::sqrt(x[2 * i])) <= bound);
    CHECK(std::abs(r.ref_coords.data()[2 * i + 1] - std::cbrt(x[2 * i + 1])) <= bound);
    // The cell is the one whose box holds the returned coordinate (lower cell on a tie).
    const double u = r.ref_coords.data()[2 * i];
    const double v = r.ref_coords.data()[2 * i + 1];
    const std::int64_t iu = u <= 1.5 ? 0 : 1;
    const std::int64_t iv = v <= 1.0 ? 0 : 1;
    CHECK(r.cell_ids.data()[i] == iu * 2 + iv);
  }
}

TEST_CASE("locate inverts a degree-1 rational map by its closed form") {
  // F(u) = w1 u / (w0 (1 - u) + w1 u) on [0, 1], projected control points 0 and 1; inverse
  // u = w0 x / (w1 (1 - x) + w0 x). Weights 1 and 3.
  const double w0 = 1.0;
  const double w1 = 3.0;
  const std::vector<double> t{0.0, 0.0, 1.0, 1.0};
  const std::vector<double> cp{0.0 * w0, w0, 1.0 * w1, w1};
  const bs::Bspline<double> b(bs::BsplineSpace<double>({space1d<double>(t, 1)}),
                              std::span<const double>(cp), true);
  const bs::LocateContext ctx(b);
  const std::vector<double> x{0.0, 0.1, 0.5, 0.75, 0.99, 1.0};
  const bs::LocateResult r = run(ctx, x);
  // F'(u) = w0 w1 / (w0 (1 - u) + w1 u)^2 >= w0 w1 / w1^2 = 1/3.
  const double bound = ctx.default_thresholds().accept * 3.0 + 16.0 * eps;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double expect = w0 * x[i] / (w1 * (1.0 - x[i]) + w0 * x[i]);
    CHECK(std::abs(r.ref_coords.data()[i] - expect) <= bound);
    CHECK(r.cell_ids.data()[i] == 0);
  }
}

TEST_CASE("locate inverts an affine map far from both origins (the threshold split)") {
  // F(u, v) = A (u, v) + b on a knot vector at [1e6, 1e6 + 1]^2, degree 1: the Greville
  // interpolant of an affine map is exact, so the inverse is the linear solve.
  const double off = 1e6;
  const std::vector<double> t{off, off, off + 0.5, off + 1.0, off + 1.0};
  const double a00 = 2.0;
  const double a01 = 0.5;
  const double a10 = -0.25;
  const double a11 = 1.5;
  const double b0 = 3.0;
  const double b1 = -7.0;
  const std::vector<double> g{off, off + 0.5, off + 1.0};
  std::vector<double> cp;
  for (const double u : g) {
    for (const double v : g) {
      cp.push_back(a00 * (u - off) + a01 * (v - off) + b0);
      cp.push_back(a10 * (u - off) + a11 * (v - off) + b1);
    }
  }
  const bs::Bspline<double> b(
      bs::BsplineSpace<double>({space1d<double>(t, 1), space1d<double>(t, 1)}),
      std::span<const double>(cp), false);
  const bs::LocateContext ctx(b);
  const bs::LocateThresholds th = ctx.default_thresholds();
  CHECK(th.stop < th.accept);  // the parametric term is active at this reach
  CHECK(th.stop == ctx.tier() * ctx.physical());
  const double det = a00 * a11 - a01 * a10;
  std::vector<double> x;
  std::vector<double> expect;
  for (const double s : {0.1, 0.5, 0.9}) {
    for (const double q : {0.2, 0.5, 0.7}) {
      x.push_back(a00 * s + a01 * q + b0);
      x.push_back(a10 * s + a11 * q + b1);
      expect.push_back(off + s);
      expect.push_back(off + q);
    }
  }
  const bs::LocateResult r = run(ctx, x);
  // sigma_min of A is above 1; the parametric floor itself is one ulp of 1e6 plus the threshold.
  const double sigma_min = std::abs(det) / std::sqrt(a00 * a00 + a01 * a01 + a10 * a10 + a11 * a11);
  const double bound = th.accept / sigma_min + 4.0 * eps * off;
  for (std::size_t i = 0; i < expect.size(); ++i) {
    CAPTURE(i);
    CHECK(std::abs(r.ref_coords.data()[i] - expect[i]) <= bound);
  }
}

TEST_CASE("locate's default thresholds coincide for a knot vector from the origin") {
  const std::vector<double> t{0.0, 0.0, 1.0, 1.0};
  const std::vector<double> cp{0.0, 1.0};
  const bs::LocateContext unit(bs::Bspline<double>(
      bs::BsplineSpace<double>({space1d<double>(t, 1)}), std::span<const double>(cp), false));
  const bs::LocateThresholds th = unit.default_thresholds();
  CHECK(th.stop == th.accept);
  CHECK(th.stop == pantr::tolerance::get_default<double>() * 1.0);
}

TEST_CASE("locate refuses a default threshold past the parametric resolution") {
  // A unit span at 1.5e14: reach 1.5e14 > 1 / (64 eps) ~ 7.0e13, so tier * parametric (about 2.1)
  // exceeds the diagonal (1). Knot snapping there still resolves a unit mesh.
  const double off = 1.5e14;
  const std::vector<double> t{off, off, off + 1.0, off + 1.0};
  const std::vector<double> cp{0.0, 1.0};
  const bs::LocateContext ctx(bs::Bspline<double>(bs::BsplineSpace<double>({space1d<double>(t, 1)}),
                                                  std::span<const double>(cp), false));
  CHECK_THROWS_WITH_AS((void)run(ctx, {0.5}),
                       doctest::Contains("sits so far from the parametric origin"),
                       pantr::value_error);
  // An explicit tolerance is never refused.
  const bs::LocateResult r = run(ctx, {0.5}, 1e-3);
  CHECK(r.cell_ids.data()[0] == 0);
}

TEST_CASE("locate validates its spline and arguments") {
  const std::vector<double> t{0.0, 0.0, 1.0, 1.0};
  // rank 2 > dim 1: an embedded curve.
  const std::vector<double> curve{0.0, 0.0, 1.0, 1.0};
  CHECK_THROWS_AS(
      bs::LocateContext(bs::Bspline<double>(bs::BsplineSpace<double>({space1d<double>(t, 1)}),
                                            std::span<const double>(curve), false)),
      pantr::not_implemented);
  // rank 1 < dim 2.
  const std::vector<double> sheet{0.0, 1.0, 2.0, 3.0};
  CHECK_THROWS_WITH_AS(bs::LocateContext(bs::Bspline<double>(
                           bs::BsplineSpace<double>({space1d<double>(t, 1), space1d<double>(t, 1)}),
                           std::span<const double>(sheet), false)),
                       doctest::Contains("got rank 1 < dim 2"), pantr::value_error);
  // A non-positive weight.
  const std::vector<double> neg{0.0, 1.0, -1.0, -1.0};
  CHECK_THROWS_WITH_AS(
      bs::LocateContext(bs::Bspline<double>(bs::BsplineSpace<double>({space1d<double>(t, 1)}),
                                            std::span<const double>(neg), true)),
      doctest::Contains("strictly positive"), pantr::value_error);
  // A periodic direction.
  const std::vector<double> pk{0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75};
  const std::vector<double> pcp{0.0, 1.0, 2.0};
  CHECK_THROWS_WITH_AS(bs::LocateContext(bs::Bspline<double>(
                           bs::BsplineSpace<double>({space1d<double>(pk, 2, true)}),
                           std::span<const double>(pcp), false)),
                       doctest::Contains("periodic B-spline spaces are not supported (axis 0)"),
                       pantr::value_error);
  // A non-finite control point is refused by the hierarchy.
  const std::vector<double> nan_cp{0.0, std::numeric_limits<double>::infinity()};
  CHECK_THROWS_AS(
      bs::LocateContext(bs::Bspline<double>(bs::BsplineSpace<double>({space1d<double>(t, 1)}),
                                            std::span<const double>(nan_cp), false)),
      pantr::value_error);
  const bs::LocateContext ctx(square_map());
  CHECK_THROWS_WITH_AS((void)run(ctx, {1.0}, std::nullopt, 0),
                       "locate: max_iter must be >= 1; got 0.", pantr::value_error);
  CHECK_THROWS_WITH_AS((void)run(ctx, {1.0}, -1.0),
                       "locate: tol must be positive and finite; got -1.0.", pantr::value_error);
  CHECK_THROWS_AS((void)run(ctx, {1.0}, std::numeric_limits<double>::infinity()),
                  pantr::value_error);
  CHECK_THROWS_WITH_AS((void)run(ctx, {std::nan("")}),
                       "locate: points must be finite; got a NaN or infinite coordinate.",
                       pantr::value_error);
  CHECK_THROWS_AS((void)bs::locate(ctx, std::vector<double>{1.0, 2.0}, 3, std::nullopt, 30),
                  pantr::value_error);
}

TEST_CASE("locate on a float spline answers to the float tier") {
  const std::vector<double> t{0.5, 0.5, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0};
  const std::vector<double> cpd = power_cps(t, 2);
  const std::vector<float> cpf(cpd.begin(), cpd.end());
  const bs::Bspline<float> b(bs::BsplineSpace<float>({space1d<float>(t, 2)}),
                             std::span<const float>(cpf), false);
  const bs::LocateContext ctx(b);
  CHECK(ctx.tier() == static_cast<double>(pantr::tolerance::get_default<float>()));
  const std::vector<double> x{0.49, 1.21, 3.24};  // images of 0.7, 1.1, 1.8 in double
  const bs::LocateResult r = run(ctx, x);
  // The promoted map is u^2 up to the float rounding of its control points (relative eps32), so
  // the coordinate is within accept / sigma_min of a preimage of a point that far from x.
  const double ef = static_cast<double>(std::numeric_limits<float>::epsilon());
  const double bound = ctx.default_thresholds().accept + 4.0 * ef * 4.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    CHECK(std::abs(r.ref_coords.data()[i] - std::sqrt(x[i])) <= bound);
  }
}

TEST_CASE("locate gives the same answer at one thread and at min(8, capacity)") {
  const bs::LocateContext ctx(square_map());
  std::vector<double> x(2000);
  for (std::size_t i = 0; i < x.size(); ++i) {
    // Inside and outside the image [0.25, 4].
    x[i] = -0.5 + 5.0 * static_cast<double>(i) / static_cast<double>(x.size());
  }
  std::optional<bs::LocateResult> one;
  {
    const ThreadsGuard g(1);
    one.emplace(run(ctx, x));
  }
  const std::size_t many = (std::min)(std::size_t{8}, par::max_threads());
  if (many < 2) {
    MESSAGE("a single thread is available, so there is no parallel run to compare");
    return;
  }
  const ThreadsGuard g(many);
  const bs::LocateResult r = run(ctx, x);
  // Every point is solved alone from data that do not depend on the thread count, so the ids are
  // equal and the coordinates equal (compared with a zero bound, NaN to NaN).
  for (std::size_t i = 0; i < x.size(); ++i) {
    CHECK(r.cell_ids.data()[i] == one->cell_ids.data()[i]);
    const double a = r.ref_coords.data()[i];
    const double c = one->ref_coords.data()[i];
    CHECK(((std::isnan(a) && std::isnan(c)) || a == c));
  }
}

TEST_CASE("locate's scale helpers") {
  const std::vector<double> lo{0.0, -2.0, 10.0, 3.0};
  const std::vector<double> hi{4.0, 2.0, 12.0, 3.0};
  const std::vector<double> reach = lk::parametric_reach(lo, hi);
  CHECK(reach[0] == 1.0);
  CHECK(reach[1] == 0.5);
  CHECK(reach[2] == 6.0);
  CHECK(reach[3] == 0.0);  // no extent: contributes nothing
  CHECK(lk::physical_scale({0.0, 0.0}) == 1.0);
  CHECK(lk::physical_scale({3.0, 5.0}) == 5.0);
  CHECK(lk::geometric_scale(2.0, 3.0) == 3.0);
  CHECK(lk::sci3(1.5e-5) == "1.500e-05");
  CHECK(lk::sci3(std::numeric_limits<double>::infinity()) == "inf");
  // Per-cell boxes of u^2: cells [0.5,1], [1,1.5], [1.5,2] are supported by cps 0..2, 1..3, 2..4.
  const bs::Bspline<double> b = square_map();
  const auto [clo, chi] = lk::cell_physical_bounds(b);
  const std::vector<double> cp = power_cps({0.5, 0.5, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0}, 2);
  for (std::size_t c = 0; c < 3; ++c) {
    CHECK(clo[c] == *std::min_element(cp.begin() + static_cast<std::ptrdiff_t>(c),
                                      cp.begin() + static_cast<std::ptrdiff_t>(c + 3)));
    CHECK(chi[c] == *std::max_element(cp.begin() + static_cast<std::ptrdiff_t>(c),
                                      cp.begin() + static_cast<std::ptrdiff_t>(c + 3)));
  }
}

TEST_CASE("locate of a constant map finds only the constant's point") {
  // Every control point equal: J == 0, so only a start already inside `stop` is returned.
  const std::vector<double> t{0.0, 0.0, 1.0, 2.0, 2.0};
  const std::vector<double> cp{1.0, 1.0, 1.0};
  const bs::LocateContext ctx(bs::Bspline<double>(bs::BsplineSpace<double>({space1d<double>(t, 1)}),
                                                  std::span<const double>(cp), false));
  const bs::LocateResult r = run(ctx, {1.0, 1.5});
  CHECK(r.cell_ids.data()[0] == 0);
  CHECK(r.ref_coords.data()[0] == 0.5);  // the first candidate's midpoint, untouched
  CHECK(r.cell_ids.data()[1] == -1);
}

TEST_CASE("locate refuses a default threshold that overflows") {
  // Control points near 1e300: the box diagonal overflows, so the default acceptance threshold is
  // infinite and would report every point found. Refused (with the previous implementation's
  // message); an explicit tolerance still works, and an empty batch asks nothing.
  const std::vector<double> t{0.0, 0.0, 1.0, 1.0};
  const std::vector<double> cp{0.0, 4e300};
  const bs::LocateContext ctx(bs::Bspline<double>(bs::BsplineSpace<double>({space1d<double>(t, 1)}),
                                                  std::span<const double>(cp), false));
  CHECK_THROWS_WITH_AS((void)run(ctx, {1e300}), "pad(r) entries must be finite; got [inf].",
                       pantr::value_error);
  const bs::LocateResult r = run(ctx, {1e300}, 1e290);
  CHECK(std::abs(r.ref_coords.data()[0] - 0.25) <= 1e-12);
  CHECK(run(ctx, {}).cell_ids.size() == 0);
}

TEST_CASE("locate abandons a candidate whose Jacobian is undefined at the domain end") {
  // Knots [0,0,0,1,2,2,2,3], degree 2: the domain is [0, 2], and at its end 2 (a knot of full
  // multiplicity before a trailing knot) the first-derivative recurrence divides by zero. A
  // Newton step from cell 1 is clipped onto that end; the candidate is abandoned there instead
  // of raising. Oracle: whatever comes back satisfies the residual contract, or is not found.
  const std::vector<double> t{0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 3.0};
  const std::vector<double> cp{0.0, 4.0, -4.0, 1.0, 2.0};
  const bs::Bspline<double> b(bs::BsplineSpace<double>({space1d<double>(t, 2)}),
                              std::span<const double>(cp), false);
  const bs::LocateContext ctx(b);
  const std::vector<double> x{3.5, 1.0, -1.0};
  bs::LocateResult r{pantr::NdArray<std::int64_t>({0}), pantr::NdArray<double>({0, 1})};
  CHECK_NOTHROW(r = run(ctx, x));
  REQUIRE(r.cell_ids.size() == x.size());
  const double accept = ctx.default_thresholds().accept;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (r.cell_ids.data()[i] < 0) {
      CHECK(std::isnan(r.ref_coords.data()[i]));
      continue;
    }
    const std::vector<double> xi{r.ref_coords.data()[i]};
    const pantr::NdArray<double> f =
        bs::evaluate(b, pantr::mdview<const double, 2>(xi.data(), 1, 1));
    CHECK(std::abs(f.data()[0] - x[i]) <= accept);
  }
}
