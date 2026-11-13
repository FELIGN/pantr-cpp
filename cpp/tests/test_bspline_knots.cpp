// Knot vectors: the tolerance, knot classes, domain tests, the uniform factories and the Greville
// abscissae. Oracles: values worked out by hand from the definitions (a uniform grid, a mean of
// `degree` knots), closed forms of the counts (a periodic vector of `n` intervals and continuity
// `c` has `n (degree - c)` functions), exact-rational arithmetic for the means, NumPy's recorded
// results where a factory reproduces NumPy (the denormal `linspace` below), and the text the
// previous Python implementation formatted for the messages. Boundary cases are constructed (a step
// of exactly the tolerance, the next double past it), not sampled.
#include <doctest/doctest.h>

#include "bspline_oracle.hpp"

#include <pantr/bspline/knot_factories.hpp>
#include <pantr/bspline/knots.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace bs = pantr::bspline;
namespace oracle = bspline_oracle;

namespace {

template <class T>
double eps_of() {
  return static_cast<double>(std::numeric_limits<T>::epsilon());
}

template <class T>
std::vector<T> tv(std::initializer_list<double> values) {
  std::vector<T> out;
  for (const double v : values) {
    out.push_back(static_cast<T>(v));
  }
  return out;
}

template <class T>
std::span<const T> sp(const std::vector<T>& v) {
  return std::span<const T>(v);
}

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

}  // namespace

// ---- scale and tolerance ----------------------------------------------------------------------

TEST_CASE_TEMPLATE("knots: the scale is the largest of the span and the two end magnitudes", T,
                   float, double) {
  // A scale taken from the span alone would give 1 for the second and third rows, which is the
  // defect the scale exists to prevent (a difference of two coordinates of size 5 or 1e6 carries
  // an error of that size times eps, however short the span). The abs makes the second row 5.
  struct Row {
    std::vector<double> knots;
    double scale;
  };
  const std::vector<Row> rows = {
      {{0, 0, 0, 1, 1, 1}, 1.0}, {{-5, -5, -4, -4}, 5.0}, {{1e6, 1e6, 1e6 + 1, 1e6 + 1}, 1e6 + 1},
      {{-3, -3, 9, 9}, 12.0},    {{0, 0, 0, 0}, 0.0},
  };
  for (const Row& row : rows) {
    std::vector<T> knots;
    for (const double k : row.knots) {
      knots.push_back(static_cast<T>(k));
    }
    CAPTURE(row.scale);
    CHECK(bs::knot_scale<T>(sp(knots)) == row.scale);
    // 8 eps scale: the strict preset (4 eps) times the merge safety (2), one rounding by the scale.
    CHECK(bs::knot_tolerance<T>(sp(knots)) == 8.0 * eps_of<T>() * row.scale);
  }
}

TEST_CASE("knots: the tolerance follows a rescaling of the knot vector exactly") {
  // Multiplying every knot by a power of two multiplies the scale, hence the tolerance, by it:
  // the tolerance is covariant under a change of units.
  const std::vector<double> base = {0.0, 0.0, 0.25, 0.5, 1.0, 1.0};
  std::vector<double> scaled = base;
  for (double& k : scaled) {
    k *= 1048576.0;  // 2^20
  }
  CHECK(bs::knot_tolerance<double>(sp(scaled)) == 1048576.0 * bs::knot_tolerance<double>(sp(base)));
}

// ---- knot classes -----------------------------------------------------------------------------

TEST_CASE("knots: classes, multiplicities and the classes holding the domain ends") {
  struct Row {
    std::vector<double> knots;
    int degree;
    std::vector<double> reps;
    std::vector<std::int64_t> mult;
    std::size_t lo_class;
    std::size_t hi_class;
    std::vector<double> domain_reps;
    std::vector<std::int64_t> domain_mult;
  };
  const std::vector<Row> rows = {
      // Open: the first and last classes hold the clamped copies.
      {{0, 0, 0, 1, 2, 2, 3, 3, 3},
       2,
       {0, 1, 2, 3},
       {3, 1, 2, 3},
       0,
       3,
       {0, 1, 2, 3},
       {3, 1, 2, 3}},
      // Uniform and unclamped: the domain is the middle three of seven classes.
      {{-2, -1, 0, 1, 2, 3, 4},
       2,
       {-2, -1, 0, 1, 2, 3, 4},
       {1, 1, 1, 1, 1, 1, 1},
       2,
       4,
       {0, 1, 2},
       {1, 1, 1}},
      // Whole boundary classes are reported, the knots outside the domain included.
      {{0, 0, 0, 0, 1, 2, 2, 2, 2}, 2, {0, 1, 2}, {4, 1, 4}, 0, 2, {0, 1, 2}, {4, 1, 4}},
      // Unclamped on the left only: the class of knots[degree] starts the domain.
      {{0, 1, 1, 2, 3, 4, 5},
       2,
       {0, 1, 2, 3, 4, 5},
       {1, 2, 1, 1, 1, 1},
       1,
       3,
       {1, 2, 3},
       {2, 1, 1}},
  };
  for (const Row& row : rows) {
    const auto& k = row.knots;
    CAPTURE(k.size());
    const auto all = bs::classify_knots<double>(sp(k), row.degree, 1e-12);
    CHECK(all.classes.knots == row.reps);
    CHECK(all.classes.multiplicities == row.mult);
    CHECK(all.lo_class == row.lo_class);
    CHECK(all.hi_class == row.hi_class);
    const auto whole = bs::unique_knots_and_multiplicity<double>(sp(k), row.degree, 1e-12, false);
    CHECK(whole.knots == row.reps);
    const auto dom = bs::unique_knots_and_multiplicity<double>(sp(k), row.degree, 1e-12, true);
    CHECK(dom.knots == row.domain_reps);
    CHECK(dom.multiplicities == row.domain_mult);
  }
}

TEST_CASE("knots: a step of exactly the tolerance is the same knot, one double more is not") {
  // The grouping splits on `step > tol`, so a step equal to the tolerance stays in its class. The
  // tolerance 0.25 and the knots are dyadic, so every step is exact.
  const double tol = 0.25;
  const std::vector<double> equal = {0.0, 0.25, 0.5, 0.75, 1.0};
  const auto together = bs::classify_knots<double>(sp(equal), 1, tol);
  CHECK(together.classes.knots == std::vector<double>{0.0});
  CHECK(together.classes.multiplicities == std::vector<std::int64_t>{5});
  // The last step is one double past the tolerance: it splits there, and only there.
  const double next = std::nextafter(1.0, kInf);
  const std::vector<double> split = {0.0, 0.25, 0.5, 0.75, next};
  REQUIRE(next - 0.75 > tol);
  const auto apart = bs::classify_knots<double>(sp(split), 1, tol);
  CHECK(apart.classes.knots == std::vector<double>{0.0, next});
  CHECK(apart.classes.multiplicities == std::vector<std::int64_t>{4, 1});
}

TEST_CASE("knots: a chain of steps within the tolerance is one class although its ends are apart") {
  // The grouping chains (documented): three steps of 0.2 against a tolerance of 0.25 give one
  // class whose ends are 0.6 apart, and the representative is the class's first knot.
  const std::vector<double> knots = {1.0, 1.2, 1.4, 1.6, 3.0};
  const auto cls = bs::classify_knots<double>(sp(knots), 1, 0.25);
  CHECK(cls.classes.knots == std::vector<double>{1.0, 3.0});
  CHECK(cls.classes.multiplicities == std::vector<std::int64_t>{4, 1});
}

TEST_CASE("knots: regrouping the representatives reproduces the classes (idempotence)") {
  const std::vector<double> knots = {0.0, 1e-13, 2e-13, 1.0, 1.0 + 1e-13, 2.0, 2.0, 2.0 + 5e-14};
  const double tol = 1e-12;
  const auto first = bs::classify_knots<double>(sp(knots), 1, tol).classes;
  REQUIRE(first.knots == std::vector<double>{0.0, 1.0, 2.0});
  REQUIRE(first.multiplicities == std::vector<std::int64_t>{3, 2, 3});
  std::vector<double> expanded;
  for (std::size_t c = 0; c < first.knots.size(); ++c) {
    expanded.insert(expanded.end(), static_cast<std::size_t>(first.multiplicities[c]),
                    first.knots[c]);
  }
  const auto again = bs::classify_knots<double>(sp(expanded), 1, tol).classes;
  CHECK(again.knots == first.knots);
  CHECK(again.multiplicities == first.multiplicities);
}

TEST_CASE(
    "knots: the multiplicity of the first domain knot counts the knots within the tolerance") {
  const std::vector<double> clamped = {0, 0, 0, 1, 1, 1};
  CHECK(bs::multiplicity_of_first_knot_in_domain<double>(sp(clamped), 2, 1e-12) == 3);
  const std::vector<double> simple = {-1, -0.5, 0, 0.5, 1};
  CHECK(bs::multiplicity_of_first_knot_in_domain<double>(sp(simple), 2, 1e-12) == 1);
  // A difference of exactly the tolerance counts (`<=`); one double more does not.
  const std::vector<double> at_tol = {-0.25, 0.0, 0.0, 1.0};
  CHECK(bs::multiplicity_of_first_knot_in_domain<double>(sp(at_tol), 2, 0.25) == 3);
  const std::vector<double> past_tol = {std::nextafter(-0.25, -kInf), 0.0, 0.0, 1.0};
  CHECK(bs::multiplicity_of_first_knot_in_domain<double>(sp(past_tol), 2, 0.25) == 2);
}

// ---- the number of basis functions ------------------------------------------------------------

TEST_CASE_TEMPLATE("knots: the count of basis functions of a uniform vector is n (degree - c)", T,
                   float, double) {
  // Closed forms from a different route than the code: an open vector of n intervals and interior
  // continuity c has (n - 1) (p - c) + p + 1 functions, a periodic one n (p - c).
  for (const long long n : {1LL, 2LL, 5LL}) {
    for (const long long p : {0LL, 1LL, 2LL, 3LL, 4LL}) {
      for (long long c = -1; c < p; ++c) {
        const auto open = bs::uniform_open_knots<T>(n, p, c, T(0), T(1));
        const auto per = bs::uniform_periodic_knots<T>(n, p, c, T(0), T(1));
        const double tol_open = bs::knot_tolerance<T>(sp(open));
        const double tol_per = bs::knot_tolerance<T>(sp(per));
        CAPTURE(n);
        CAPTURE(p);
        CAPTURE(c);
        CHECK(bs::num_basis<T>(sp(open), static_cast<int>(p), false, tol_open) ==
              (n - 1) * (p - c) + p + 1);
        CHECK(bs::num_basis<T>(sp(per), static_cast<int>(p), true, tol_per) == n * (p - c));
      }
    }
  }
}

TEST_CASE("knots: the periodic count subtracts regularity + 1 of the first domain knot") {
  // [-2, ..., 4] at degree 2: 7 - 2 - 1 = 4 functions, regularity 2 - 1 = 1, so 4 - 2 = 2.
  const std::vector<double> simple = {-2, -1, 0, 1, 2, 3, 4};
  CHECK(bs::num_basis<double>(sp(simple), 2, false, 1e-12) == 4);
  CHECK(bs::num_basis<double>(sp(simple), 2, true, 1e-12) == 2);
  // A knot of multiplicity degree + 1 has regularity -1: nothing is subtracted.
  const std::vector<double> flat = {0, 0, 0, 1, 2, 3, 3, 3};
  CHECK(bs::num_basis<double>(sp(flat), 2, true, 1e-12) == 5);
}

// ---- the domain test --------------------------------------------------------------------------

TEST_CASE("knots: a point is in the domain up to exactly the tolerance, at each end") {
  const std::vector<double> knots = {0, 0, 1, 1};
  const double tol = 0.25;
  const auto in = [&](double x) { return bs::is_in_domain<double, double>(sp(knots), 1, x, tol); };
  CHECK(in(0.0));
  CHECK(in(1.0));
  CHECK(in(0.5));
  CHECK(in(-0.25));
  CHECK_FALSE(in(std::nextafter(-0.25, -kInf)));
  CHECK(in(1.25));
  CHECK_FALSE(in(std::nextafter(1.25, kInf)));
  CHECK_FALSE(in(kNaN));
  CHECK_FALSE(in(kInf));
  CHECK_FALSE(in(-kInf));
}

TEST_CASE("knots: the band is tol wide even where tol is below one ulp of the end") {
  // At 1e6 an ulp is 1.164e-10, so a bound shifted by `tol = 1e-10` (end + tol) rounds up to the
  // next double and admits one ulp, which is more than tol. The difference formed first does not.
  const std::vector<double> knots = {0, 0, 1e6, 1e6};
  const double tol = 1e-10;
  REQUIRE(std::nextafter(1e6, kInf) - 1e6 > tol);
  CHECK(bs::is_in_domain<double, double>(sp(knots), 1, 1e6, tol));
  CHECK_FALSE(bs::is_in_domain<double, double>(sp(knots), 1, std::nextafter(1e6, kInf), tol));
}

TEST_CASE("knots: a mixed knot and point type compares in the wider type") {
  // Knots float, point double: 1 + 1e-12 is not 1 in double, so it is outside a tolerance of
  // 1e-13; compared as floats it would round to 1 and be inside.
  const std::vector<float> fk = {0, 0, 1, 1};
  CHECK_FALSE(bs::is_in_domain<float, double>(sp(fk), 1, 1.0 + 1e-12, 1e-13));
  CHECK(bs::is_in_domain<float, double>(sp(fk), 1, 1.0, 1e-13));
  // Knots double, point float: the end 1 - 1e-9 is not representable in float, where it would
  // round to 1, which the point 1.0f then equals; in double the point is 1e-9 above it.
  const double end = 1.0 - 1e-9;
  const std::vector<double> dk = {0, 0, end, end};
  CHECK_FALSE(bs::is_in_domain<double, float>(sp(dk), 1, 1.0F, 1e-10));
  CHECK(bs::is_in_domain<double, float>(sp(dk), 1, 1.0F, 2e-9));
}

// ---- cardinal intervals -----------------------------------------------------------------------

namespace {

template <std::size_t N>
std::array<bool, N> cardinal(const std::vector<double>& knots, int degree, double tol,
                             bool prefill = false) {
  std::array<bool, N> out;
  out.fill(prefill);
  bs::cardinal_intervals<double>(sp(knots), degree, tol, std::span<bool>(out));
  return out;
}

}  // namespace

TEST_CASE("knots: cardinal intervals of uniform vectors, open and unclamped") {
  const std::vector<double> unclamped = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  CHECK(cardinal<4>(unclamped, 2, 1e-12) == std::array<bool, 4>{true, true, true, true});
  // [0,0,0,1,2,3,4,4,4]: the end intervals touch a knot of multiplicity 3, the two inner ones see
  // three unit lengths on each side of the window.
  const auto open = bs::uniform_open_knots<double>(4, 2, 1, 0.0, 4.0);
  const std::vector<double> ov(open.begin(), open.end());
  CHECK(cardinal<4>(ov, 2, 1e-12) == std::array<bool, 4>{false, true, true, false});
}

TEST_CASE(
    "knots: two lengths differing by exactly the tolerance are equal, by one double more not") {
  // Knot 3 of 0..8 is moved up by d: the lengths around it become 1 + d and 1 - d, and the windows
  // of the intervals starting at knots 2 and 3 compare them (difference 2 d). With tol = 0.5 and
  // d = 0.25 the difference is exactly the tolerance; with d one 2^-30 more it is past it.
  const double tol = 0.5;
  std::vector<double> at = {0, 1, 2, 3.25, 4, 5, 6, 7, 8};
  CHECK(cardinal<4>(at, 2, tol) == std::array<bool, 4>{true, true, true, true});
  std::vector<double> past = {0, 1, 2, 3.25 + 0x1p-30, 4, 5, 6, 7, 8};
  CHECK(cardinal<4>(past, 2, tol) == std::array<bool, 4>{false, false, true, true});
}

TEST_CASE("knots: the window of an interval follows the knots past a repeated one") {
  // 0, 1, 2, 3, 3, 4, ..., 8 at degree 2: the domain classes are 2, 3 (twice), 4, 5, 6. The two
  // intervals touching the double knot are not cardinal; the next two are, with a window of unit
  // lengths. The window of the third starts at knot index 4, two past the start of the second:
  // stepping the knot index by one per interval instead of by the multiplicity reads a window
  // holding the zero length between the two 3s.
  const std::vector<double> knots = {0, 1, 2, 3, 3, 4, 5, 6, 7, 8};
  CHECK(cardinal<4>(knots, 2, 1e-12) == std::array<bool, 4>{false, false, true, true});
}

TEST_CASE("knots: at degree 0 only the multiplicity decides, and the output is overwritten") {
  const std::vector<double> simple = {0, 1, 2, 3};
  CHECK(cardinal<3>(simple, 0, 1e-12) == std::array<bool, 3>{true, true, true});
  // A repeated interior knot makes both of its intervals non-cardinal.
  const std::vector<double> repeated = {0, 1, 1, 2, 3};
  CHECK(cardinal<3>(repeated, 0, 1e-12) == std::array<bool, 3>{false, false, true});
  // Every in-domain knot repeated: the early exit still clears a pre-filled output.
  const std::vector<double> clamped = {0, 0, 1, 1};
  CHECK(cardinal<1>(clamped, 1, 1e-12, true) == std::array<bool, 1>{false});
}

// ---- messages ---------------------------------------------------------------------------------

TEST_CASE("knots: snapping that collapses a float32 mesh is refused with the previous message") {
  // Provenance: the text is the f-string of the previous Python implementation evaluated at these
  // numbers (tolerance 0.954 = 8 eps_f32 1000001; an ulp of float32 at 1e6 is 0.0625, so 15 ulp).
  const std::vector<float> raw = tv<float>({1e6, 1e6, 1e6, 1e6 + 0.5, 1e6 + 1, 1e6 + 1, 1e6 + 1});
  const std::vector<float> snapped(7, 1e6F);
  const double tol = bs::knot_tolerance<float>(sp(raw));
  CHECK_THROWS_WITH_AS(
      bs::check_snapping_kept_an_interval<float>(sp(raw), sp(snapped), tol),
      "knot snapping collapsed every knot onto 1000000.0: in float32 at |coordinate| ~ 1e+06 two "
      "knots are the same knot unless they differ by more than 0.954 (15 ulp there), and the "
      "closest pair in [1000000.0, 1000001.0] is 0.5 apart. This mesh is finer than float32 "
      "resolves at that magnitude. Use float64, move the domain nearer the origin, or coarsen "
      "the mesh.",
      pantr::value_error);
}

TEST_CASE("knots: the float64 collapse message offers no float64 remedy") {
  const std::vector<double> raw = {1e15, 1e15, 1e15, 1e15 + 0.5, 1e15 + 1, 1e15 + 1, 1e15 + 1};
  const std::vector<double> snapped(7, 1e15);
  CHECK_THROWS_WITH_AS(
      bs::check_snapping_kept_an_interval<double>(sp(raw), sp(snapped),
                                                  bs::knot_tolerance<double>(sp(raw))),
      "knot snapping collapsed every knot onto 1000000000000000.0: in float64 at |coordinate| ~ "
      "1e+15 two knots are the same knot unless they differ by more than 1.78 (14 ulp there), and "
      "the closest pair in [1000000000000000.0, 1000000000000001.0] is 0.5 apart. This mesh is "
      "finer than float64 resolves at that magnitude. Move the domain nearer the origin, or "
      "coarsen the mesh.",
      pantr::value_error);
}

TEST_CASE("knots: the snapping check refuses only a collapse snapping caused") {
  const std::vector<double> flat(6, 5.0);
  const std::vector<double> distinct = {0, 0, 0.5, 1, 1};
  // Supplied flat: not this check's case (the interval check owns it).
  CHECK_NOTHROW(bs::check_snapping_kept_an_interval<double>(sp(flat), sp(flat), 1e-12));
  // Distinct before and after.
  CHECK_NOTHROW(bs::check_snapping_kept_an_interval<double>(sp(distinct), sp(distinct), 1e-12));
}

TEST_CASE("knots: a space with no interval is refused with the previous message") {
  // Provenance: the f-string of the previous implementation at degree 1, flat knots 5.0, whose
  // tolerance is 8 eps 5 = 8.88e-15.
  CHECK_NOTHROW(bs::check_space_has_an_interval<double>(sp(tv<double>({0, 1})), 0, 1, 1e-12));
  const std::vector<double> flat(4, 5.0);
  CHECK_THROWS_WITH_AS(
      bs::check_space_has_an_interval<double>(sp(flat), 1, 0, bs::knot_tolerance<double>(sp(flat))),
      "knot vector spans no interval: at degree 1 the domain runs from knots[1] = 5.0 to "
      "knots[2] = 5.0, and every step between them is at most this vector's tolerance of "
      "8.88e-15, so its in-domain knots are all one knot, the space has no cell, and nothing can "
      "be evaluated, tabulated or located on it. The domain needs two consecutive knots more than "
      "8.88e-15 apart.",
      pantr::value_error);
  // An interior knot of high multiplicity swallows the domain: the end indices are the domain's.
  const std::vector<double> swallowed = {0, 1, 1, 2};
  CHECK_THROWS_WITH_AS(
      bs::check_space_has_an_interval<double>(sp(swallowed), 1, 0,
                                              bs::knot_tolerance<double>(sp(swallowed))),
      doctest::Contains("at degree 1 the domain runs from knots[1] = 1.0 to knots[2] = 1.0, and "
                        "every step between them is at most this vector's tolerance of 3.55e-15"),
      pantr::value_error);
}

// ---- linspace and the factories ---------------------------------------------------------------

TEST_CASE("factories: linspace has the sizes, the endpoints and NumPy's values") {
  CHECK(bs::detail::linspace<double>(0.0, 1.0, 0).empty());
  CHECK(bs::detail::linspace<double>(0.7, 1.0, 1) == std::vector<double>{0.7});
  CHECK(bs::detail::linspace<double>(0.7, 1.0, 2) == std::vector<double>{0.7, 1.0});
  // 2 * 0.35 + 0.2 is 0.8999999999999999 in double: the last point is set to `stop`, not computed.
  // Provenance: np.linspace(0.2, 0.9, 3) == [0.2, 0.55, 0.9].
  const auto v = bs::detail::linspace<double>(0.2, 0.9, 3);
  REQUIRE(2.0 * ((0.9 - 0.2) / 2.0) + 0.2 != 0.9);
  CHECK(v == std::vector<double>{0.2, 0.55, 0.9});
  // Provenance: np.linspace(np.float32(0), np.float32(1), 4, dtype=np.float32).
  CHECK(bs::detail::linspace<float>(0.0F, 1.0F, 4) ==
        std::vector<float>{0.0F, 0.33333334F, 0.6666667F, 1.0F});
}

TEST_CASE("factories: linspace takes NumPy's branch where the step underflows to zero") {
  // The span is the smallest denormal, so (stop - start) / 4 rounds to zero and NumPy computes
  // (i / 4) * delta instead: 0.75 * denorm_min rounds to denorm_min, the rest to zero. Without
  // the branch every interior point is `start`. Provenance: np.linspace(0.0, 5e-324, 5) ==
  // [0, 0, 0, 5e-324, 5e-324] (NumPy 2.4).
  const double d = std::numeric_limits<double>::denorm_min();
  REQUIRE(d / 4.0 == 0.0);
  CHECK(bs::detail::linspace<double>(0.0, d, 5) == std::vector<double>{0.0, 0.0, 0.0, d, d});
  // Through a factory: degree 0, continuity -1, four intervals, one knot per unique value.
  CHECK(bs::uniform_open_knots<double>(4, 0, -1, 0.0, d) ==
        std::vector<double>{0.0, 0.0, 0.0, d, d});
}

TEST_CASE("factories: uniform open knots, worked out by hand") {
  using V = std::vector<double>;
  CHECK(bs::uniform_open_knots<double>(2, 2, 1, 0.0, 1.0) == V{0, 0, 0, 0.5, 1, 1, 1});
  // C^0 at degree 2: each interior knot twice.
  CHECK(bs::uniform_open_knots<double>(3, 2, 0, 0.0, 3.0) == V{0, 0, 0, 1, 1, 2, 2, 3, 3, 3});
  // C^-1: each interior knot degree + 1 times.
  CHECK(bs::uniform_open_knots<double>(2, 2, -1, 0.0, 1.0) == V{0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1});
  // One interval: only the clamped ends.
  CHECK(bs::uniform_open_knots<double>(1, 3, 2, -1.0, 2.0) == V{-1, -1, -1, -1, 2, 2, 2, 2});
  // Degree 0: the unique knots themselves.
  CHECK(bs::uniform_open_knots<double>(3, 0, -1, 0.0, 3.0) == V{0, 1, 2, 3});
  // An offset domain whose grid is exact: linspace(-3, 5, 3) = [-3, 1, 5].
  CHECK(bs::uniform_open_knots<double>(2, 2, 1, -3.0, 5.0) == V{-3, -3, -3, 1, 5, 5, 5});
  using F = std::vector<float>;
  CHECK(bs::uniform_open_knots<float>(4, 1, 0, 0.0F, 1.0F) == F{0, 0, 0.25F, 0.5F, 0.75F, 1, 1});
  CHECK(bs::uniform_open_knots<float>(3, 1, 0, 0.0F, 1.0F) ==
        F{0, 0, 0.33333334F, 0.6666667F, 1, 1});
}

TEST_CASE("factories: uniform periodic knots, worked out by hand") {
  using V = std::vector<double>;
  // C^1 at degree 2: one extra knot... two past each end at the interval length.
  CHECK(bs::uniform_periodic_knots<double>(2, 2, 1, 0.0, 1.0) == V{-1, -0.5, 0, 0.5, 1, 1.5, 2});
  // C^0: the knots doubled, one extension knot at each end.
  CHECK(bs::uniform_periodic_knots<double>(2, 2, 0, 0.0, 1.0) ==
        V{-0.5, 0, 0, 0.5, 0.5, 1, 1, 1.5});
  // C^-1: no extension at all.
  CHECK(bs::uniform_periodic_knots<double>(2, 1, -1, 0.0, 1.0) == V{0, 0, 0.5, 0.5, 1, 1});
  CHECK(bs::uniform_periodic_knots<double>(3, 3, 2, 0.0, 3.0) ==
        V{-3, -2, -1, 0, 1, 2, 3, 4, 5, 6});
  CHECK(bs::uniform_periodic_knots<double>(2, 3, 1, 0.0, 1.0) ==
        V{-1, -0.5, 0, 0, 0.5, 0.5, 1, 1, 1.5, 2});
  CHECK(bs::uniform_periodic_knots<float>(1, 1, 0, 0.0F, 1.0F) == std::vector<float>{-1, 0, 1, 2});
}

TEST_CASE_TEMPLATE("factories: the cardinal vector is the integers from -degree to n + degree", T,
                   float, double) {
  for (const long long n : {1LL, 2LL, 7LL}) {
    for (const long long p : {0LL, 1LL, 2LL, 5LL}) {
      const auto k = bs::cardinal_knots<T>(n, p);
      REQUIRE(k.size() == static_cast<std::size_t>(n + 2 * p + 1));
      for (std::size_t i = 0; i < k.size(); ++i) {
        CAPTURE(n);
        CAPTURE(p);
        CAPTURE(i);
        CHECK(k[i] == static_cast<T>(static_cast<long long>(i) - p));
      }
    }
  }
}

TEST_CASE_TEMPLATE("factories: the vector sizes follow the closed forms and the ends are exact", T,
                   float, double) {
  // Open: 2 (p + 1) + (n - 1) (p - c); periodic: 2 (c + 1) + (n + 1) (p - c). The ends of an open
  // vector are `start` and `end` bit for bit; a periodic vector holds both among its knots.
  const T start = static_cast<T>(-0.3);
  const T end = static_cast<T>(2.9);
  for (const long long n : {1LL, 3LL, 10LL}) {
    for (const long long p : {0LL, 1LL, 3LL}) {
      for (long long c = -1; c < p; ++c) {
        const auto open = bs::uniform_open_knots<T>(n, p, c, start, end);
        const auto per = bs::uniform_periodic_knots<T>(n, p, c, start, end);
        CAPTURE(n);
        CAPTURE(p);
        CAPTURE(c);
        CHECK(static_cast<long long>(open.size()) == 2 * (p + 1) + (n - 1) * (p - c));
        CHECK(static_cast<long long>(per.size()) == 2 * (c + 1) + (n + 1) * (p - c));
        CHECK(open.front() == start);
        CHECK(open.back() == end);
        CHECK(std::is_sorted(open.begin(), open.end()));
        CHECK(std::is_sorted(per.begin(), per.end()));
        const std::size_t ext = static_cast<std::size_t>(c + 1);
        CHECK(per[ext] == start);
        CHECK(per[per.size() - ext - 1] == end);
      }
    }
  }
}

TEST_CASE("factories: arguments are checked in the documented order with the matched messages") {
  // start >= end comes first, whatever else is wrong.
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::uniform_open_knots<double>(0, -1, 9, 1.0, 0.0)),
                       "domain[0] must be less than domain[1]", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::uniform_open_knots<double>(0, -1, 9, 1.0, 1.0)),
                       "domain[0] must be less than domain[1]", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::uniform_open_knots<double>(0, -1, 9, 0.0, 1.0)),
                       "num_intervals must be at least 1", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::uniform_open_knots<double>(1, -1, 9, 0.0, 1.0)),
                       "degree must be non-negative", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::uniform_open_knots<double>(1, 2, 2, 0.0, 1.0)),
                       "Continuity must be between -1 and 1 for degree 2.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::uniform_open_knots<double>(1, 2, -2, 0.0, 1.0)),
                       "Continuity must be between -1 and 1 for degree 2.", pantr::value_error);
  // Degree 0 admits only continuity -1, and the message says "-1 and -1".
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::uniform_periodic_knots<float>(1, 0, 0, 0.0F, 1.0F)),
                       "Continuity must be between -1 and -1 for degree 0.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::uniform_periodic_knots<double>(0, 1, 0, 0.0, 1.0)),
                       "num_intervals must be at least 1", pantr::value_error);
  // The cardinal factory reports a zero count as such, not as an empty domain.
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::cardinal_knots<double>(0, 2)),
                       "num_intervals must be at least 1", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::cardinal_knots<double>(2, -1)),
                       "degree must be non-negative", pantr::value_error);
}

TEST_CASE("factories: a count past the supported size is a capacity_error before any allocation") {
  const long long largest = (std::numeric_limits<long long>::max)() / 4;
  // Each of these would otherwise form a count or an allocation beyond memory.
  CHECK_THROWS_AS(static_cast<void>(bs::uniform_open_knots<double>(largest + 1, 1, 0, 0.0, 1.0)),
                  pantr::capacity_error);
  CHECK_THROWS_AS(static_cast<void>(bs::uniform_open_knots<double>(1, largest + 1, 0, 0.0, 1.0)),
                  pantr::capacity_error);
  CHECK_THROWS_AS(
      static_cast<void>(bs::uniform_periodic_knots<double>(largest + 1, 1, 0, 0.0, 1.0)),
      pantr::capacity_error);
  // 2^60 intervals of multiplicity 8: the product of the two overflows the supported count.
  CHECK_THROWS_AS(static_cast<void>(bs::uniform_open_knots<double>(1LL << 60, 7, -1, 0.0, 1.0)),
                  pantr::capacity_error);
  CHECK_THROWS_AS(
      static_cast<void>(bs::uniform_periodic_knots<float>(1LL << 60, 7, -1, 0.0F, 1.0F)),
      pantr::capacity_error);
  CHECK_THROWS_AS(static_cast<void>(bs::cardinal_knots<double>(largest + 1, 2)),
                  pantr::capacity_error);
}

// ---- Greville abscissae -----------------------------------------------------------------------

namespace {

template <class T>
std::vector<T> greville(const std::vector<T>& knots, int degree, bool periodic) {
  const double tol = bs::knot_tolerance<T>(sp(knots));
  const auto nb = static_cast<std::size_t>(bs::num_basis<T>(sp(knots), degree, periodic, tol));
  const T a = knots[static_cast<std::size_t>(degree)];
  const T b = knots[knots.size() - static_cast<std::size_t>(degree) - 1];
  return bs::greville_abscissae<T>(sp(knots), degree, nb, periodic, a, b);
}

}  // namespace

TEST_CASE("greville: the mean of degree knots, worked out by hand") {
  using V = std::vector<double>;
  // Open quadratic: g_i = (u_{i+1} + u_{i+2}) / 2.
  CHECK(greville<double>({0, 0, 0, 1, 2, 2, 3, 3, 3}, 2, false) == V{0, 0.5, 1.5, 2, 2.5, 3});
  // Degree 1: the interior knots themselves.
  CHECK(greville<double>({0, 0, 0.5, 1, 1}, 1, false) == V{0, 0.5, 1});
  // Degree 0: the midpoints of the spans.
  CHECK(greville<double>({0, 1, 3}, 0, false) == V{0.5, 2});
  // Cubic, four uniform intervals: [0, 1/12, 1/4, 1/2, 3/4, 11/12, 1], each within a few roundings.
  const auto g = greville<double>({0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1}, 3, false);
  const double want[7] = {0.0, 1.0 / 12, 0.25, 0.5, 0.75, 11.0 / 12, 1.0};
  REQUIRE(g.size() == 7);
  for (std::size_t i = 0; i < 7; ++i) {
    CAPTURE(i);
    CHECK(std::fabs(g[i] - want[i]) <= 4.0 * eps_of<double>());
  }
}

TEST_CASE("greville: a periodic space wraps into the half-open domain and sorts") {
  using V = std::vector<double>;
  // Cardinal quadratic on [0, 4]: the means are -0.5, 0.5, 1.5, 2.5; -0.5 wraps to 3.5 and moves
  // to the end of the sorted list.
  const auto card = bs::cardinal_knots<double>(4, 2);
  CHECK(greville<double>(card, 2, true) == V{0.5, 1.5, 2.5, 3.5});
  // A mean landing exactly on the end wraps to the start, since the domain is [a, b): degree 1,
  // knots [0, 0, 1, 2, 2] have the abscissae 0, 1, 2 over the domain [0, 2].
  CHECK(greville<double>({0, 0, 1, 2, 2}, 1, true) == V{0, 0, 1});
  // Cubic cardinal on [0, 4]: the means are -1, 0, 1, 2 and -1 wraps to 3.
  CHECK(greville<double>(bs::cardinal_knots<double>(4, 3), 3, true) == V{0, 1, 2, 3});
}

TEST_CASE("greville: a mean of negative zeros is +0.0, as NumPy's reduction returns it") {
  // Degree 2 knots [-1, -1, -1, -0.0, -0.0, 1, 1, 1]: the abscissa for i = 2 is the mean of two
  // -0.0, which NumPy returns as +0.0 (its sum starts from +0). The sign is observable.
  const std::vector<double> knots = {-1, -1, -1, -0.0, -0.0, 1, 1, 1};
  const auto g = greville<double>(knots, 2, false);
  REQUIRE(g.size() == 5);
  CHECK(g[0] == -1.0);
  CHECK(g[1] == -0.5);
  CHECK(g[2] == 0.0);
  CHECK_FALSE(std::signbit(g[2]));
  CHECK(g[3] == 0.5);
  CHECK(g[4] == 1.0);
}

TEST_CASE_TEMPLATE("greville: the means of a long run match the exact mean to the summation bound",
                   T, float, double) {
  // Degrees 1..140 cross every branch of the pairwise sum (under 8 terms, up to 128, the
  // recursion) and its remainders. The data alternate in sign over several magnitudes, so a
  // dropped or doubled term, or a wrong remainder loop, moves the result by far more than the
  // bound (n - 1) eps Σ|x| (Higham eq. 4.4 with u = eps / 2, loosened by 2), plus the division.
  constexpr std::size_t count = 300;
  std::vector<T> knots(count);
  for (std::size_t i = 0; i < count; ++i) {
    const double mag = 0.125 * static_cast<double>(1 + (i * 7) % 13);
    knots[i] = static_cast<T>((i % 2 == 0 ? mag : -mag) + 8.0);
  }
  std::sort(knots.begin(), knots.end());
  for (const int degree : {1, 2, 7, 8, 9, 15, 16, 17, 100, 127, 128, 129, 130, 140}) {
    CAPTURE(degree);
    const auto d = static_cast<std::size_t>(degree);
    const std::vector<T> g =
        bs::greville_abscissae<T>(sp(knots), degree, 3, false, knots[0], knots[1]);
    for (std::size_t i = 0; i < 3; ++i) {
      oracle::Rational sum = 0;
      double abs_sum = 0.0;
      for (std::size_t j = 0; j < d; ++j) {
        sum += oracle::exact(static_cast<double>(knots[i + 1 + j]));
        abs_sum += std::fabs(static_cast<double>(knots[i + 1 + j]));
      }
      const oracle::Rational mean = sum / static_cast<long long>(d);
      const double bound =
          (static_cast<double>(d) * eps_of<T>() * abs_sum) / static_cast<double>(d) +
          eps_of<T>() * std::fabs(static_cast<double>(g[i]));
      CAPTURE(i);
      CHECK(oracle::within(static_cast<double>(g[i]), mean, bound));
    }
  }
}

TEST_CASE("greville: numpy_mod moves a result onto the divisor's sign and keeps its zero's sign") {
  const auto mod = [](double x, double m) { return bs::detail::numpy_mod<double>(x, m); };
  CHECK(mod(5.0, 3.0) == 2.0);
  CHECK(mod(-0.5, 4.0) == 3.5);
  CHECK(mod(1.0, -3.0) == -2.0);
  CHECK(mod(-1.0, -3.0) == -1.0);
  // A zero result carries the sign of the divisor, as NumPy's does.
  CHECK_FALSE(std::signbit(mod(-6.0, 3.0)));
  CHECK(std::signbit(mod(6.0, -3.0)));
}
