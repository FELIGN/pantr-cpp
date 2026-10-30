// BsplineSpace1D as a value: construction and its refusals, derived state, snapping, from_state,
// the domain test, the call forms and their extents, cardinal intervals, restriction and Greville
// abscissae. Oracles: values worked out by hand from the definitions, the previous Python
// implementation's message text (the Python suite matches it), and closed forms of the counts.
// Boundary cases are constructed (a point exactly the tolerance outside), not sampled. The basis
// itself is tested in test_bspline_basis_1d.cpp.
#include <doctest/doctest.h>

#include <pantr/bspline/knot_factories.hpp>
#include <pantr/bspline/space_1d.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace bs = pantr::bspline;
using pantr::NdArray;

namespace {

template <class T>
double eps_of() {
  return static_cast<double>(std::numeric_limits<T>::epsilon());
}

template <class T>
std::vector<T> cast_all(const std::vector<double>& v) {
  std::vector<T> out;
  out.reserve(v.size());
  for (const double x : v) {
    out.push_back(static_cast<T>(x));
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

// ---- construction ------------------------------------------------------------------------------

TEST_CASE("space1d: derived state of a table of spaces, worked out by hand") {
  struct Row {
    const char* name;
    std::vector<double> knots;
    int degree;
    long long num_basis;
    long long num_intervals;
    std::vector<std::int64_t> first;
    bool left_open;
    bool right_open;
    bool bezier_like;
    double a;
    double b;
  };
  const std::vector<Row> rows = {
      {"open quadratic", {0, 0, 0, 1, 2, 2, 3, 3, 3}, 2, 6, 3, {0, 1, 3}, true, true, false, 0, 3},
      {"Bezier", {0, 0, 0, 1, 1, 1}, 2, 3, 1, {0}, true, true, true, 0, 1},
      {"unclamped", {0, 1, 2, 3, 4, 5, 6, 7}, 2, 5, 3, {0, 1, 2}, false, false, false, 2, 5},
      {"clamped left", {0, 0, 0, 1, 2, 3, 4, 5}, 2, 5, 3, {0, 1, 2}, true, false, false, 0, 3},
      {"interior double knot", {0, 0, 0, 1, 1, 2, 2, 2}, 2, 5, 2, {0, 2}, true, true, false, 0, 2},
      {"degree 0", {0, 1, 3}, 0, 2, 2, {0, 1}, true, true, false, 0, 3},
      {"degree 0, one cell", {0, 1}, 0, 1, 1, {0}, true, true, true, 0, 1},
      {"left multiplicity p + 2", {0, 0, 0, 0, 1, 1, 1}, 2, 4, 1, {1}, true, true, false, 0, 1},
  };
  for (const Row& r : rows) {
    CAPTURE(std::string(r.name));
    const bs::BsplineSpace1D<double> s(sp(r.knots), r.degree);
    CHECK(s.degree() == r.degree);
    CHECK_FALSE(s.periodic());
    CHECK(s.num_basis() == r.num_basis);
    CHECK(s.num_intervals() == r.num_intervals);
    const auto first = s.first_basis_per_interval();
    CHECK(std::vector<std::int64_t>(first.begin(), first.end()) == r.first);
    CHECK(s.has_left_end_open() == r.left_open);
    CHECK(s.has_right_end_open() == r.right_open);
    CHECK(s.has_open_knots() == (r.left_open && r.right_open));
    CHECK(s.has_bezier_like_knots() == r.bezier_like);
    CHECK(s.domain() == std::make_pair(r.a, r.b));
    CHECK(std::equal(s.knots().begin(), s.knots().end(), r.knots.begin(), r.knots.end()));
  }
}

TEST_CASE("space1d: a periodic space has no first_basis_per_interval and no open ends") {
  const auto knots = bs::cardinal_knots<double>(4, 2);
  const bs::BsplineSpace1D<double> s(sp(knots), 2, true);
  CHECK(s.periodic());
  CHECK(s.num_basis() == 4);
  CHECK(s.num_intervals() == 4);
  CHECK_FALSE(s.has_left_end_open());
  CHECK_FALSE(s.has_right_end_open());
  CHECK_FALSE(s.has_bezier_like_knots());
  CHECK(s.domain() == std::make_pair(0.0, 4.0));
  CHECK_THROWS_WITH_AS(static_cast<void>(s.first_basis_per_interval()),
                       "first_basis_per_interval: periodic B-spline spaces are not supported.",
                       pantr::value_error);
}

TEST_CASE("space1d: unique knots with multiplicities, all or in the domain") {
  const std::vector<double> knots = {0, 0, 0, 0, 1, 2, 2};
  const bs::BsplineSpace1D<double> s(sp(knots), 1);
  // Degree 1: the classes are 0 (x4, the clamped copies included), 1 and 2 (x2).
  const auto all = s.unique_knots_and_multiplicity(false);
  CHECK(std::vector<double>(all.first.begin(), all.first.end()) == std::vector<double>{0, 1, 2});
  CHECK(std::vector<std::int64_t>(all.second.begin(), all.second.end()) ==
        std::vector<std::int64_t>{4, 1, 2});
  // Periodic cardinal knots -2..6 at degree 2: the domain is 0..4, the classes outside it are not
  // reported with in_domain.
  const auto periodic_knots = bs::cardinal_knots<double>(4, 2);
  const bs::BsplineSpace1D<double> p(sp(periodic_knots), 2, true);
  const auto in = p.unique_knots_and_multiplicity(true);
  CHECK(std::vector<double>(in.first.begin(), in.first.end()) ==
        std::vector<double>{0, 1, 2, 3, 4});
  CHECK(std::vector<std::int64_t>(in.second.begin(), in.second.end()) ==
        std::vector<std::int64_t>{1, 1, 1, 1, 1});
  CHECK(p.unique_knots_and_multiplicity(false).first.size() == 9);
}

TEST_CASE("space1d: construction refuses what defines no space, in order, with the matched text") {
  const auto fails = [](const std::vector<double>& k, int degree, bool periodic,
                        const char* message) {
    CAPTURE(message);
    CHECK_THROWS_WITH_AS(static_cast<void>(bs::BsplineSpace1D<double>(sp(k), degree, periodic)),
                         doctest::Contains(message), pantr::value_error);
  };
  fails({}, -1, false, "degree must be non-negative");
  // The size comes before the order: three knots that decrease at degree 2 are too few first.
  fails({0, 1, 2, 3, 4}, 2, false, "knots must have at least 2*degree+2 elements");
  fails({1, 0, 2}, 2, false, "knots must have at least 2*degree+2 elements");
  fails({0, 0, 0, 1, 0.5, 1, 1, 1}, 2, false, "knots must be non-decreasing");
  fails({0, 0, 0, kNaN, 1, 1, 1}, 2, false, "knots must be non-decreasing");
  fails({0, 0, 0, 1, 1, kInf, kInf}, 2, false, "knots must be non-decreasing");
  // Periodic: 6 knots at degree 2 leave 6 - 3 - (1 + 1) = 1 function, fewer than degree + 1.
  fails({0, 1, 2, 3, 4, 5}, 2, true, "Not enough knots for the specified degree");
  // The boundary: 7 knots leave 2 = degree functions, still too few; 8 leave 3 = degree + 1.
  fails({-2, -1, 0, 1, 2, 3, 4}, 2, true, "Not enough knots for the specified degree");
  CHECK_NOTHROW(
      bs::BsplineSpace1D<double>(sp(std::vector<double>{-2, -1, 0, 1, 2, 3, 4, 5}), 2, true));
}

TEST_CASE("space1d: a domain of no extent is refused at every degree, type and magnitude") {
  for (const int degree : {0, 1, 2, 3}) {
    for (const double value : {0.0, 1.0, -3.5, 1e6}) {
      const auto n = static_cast<std::size_t>(2 * degree + 2);
      CAPTURE(degree);
      CAPTURE(value);
      CHECK_THROWS_WITH_AS(
          static_cast<void>(bs::BsplineSpace1D<double>(sp(std::vector<double>(n, value)), degree)),
          doctest::Contains("knot vector spans no interval"), pantr::value_error);
      CHECK_THROWS_WITH_AS(static_cast<void>(bs::BsplineSpace1D<float>(
                               sp(std::vector<float>(n, static_cast<float>(value))), degree)),
                           doctest::Contains("knot vector spans no interval"), pantr::value_error);
    }
  }
  // The flat vector is refused whatever else is set.
  const std::vector<double> flat(4, 5.0);
  CHECK_THROWS_AS(static_cast<void>(bs::BsplineSpace1D<double>(sp(flat), 1, true, true)),
                  pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(bs::BsplineSpace1D<double>(sp(flat), 1, false, false)),
                  pantr::value_error);
  // Distinct knots, and still no interval: an interior knot swallows the domain.
  const auto swallowed = [](const std::vector<double>& k, int degree) {
    CAPTURE(degree);
    CHECK_THROWS_WITH_AS(static_cast<void>(bs::BsplineSpace1D<double>(sp(k), degree)),
                         doctest::Contains("knot vector spans no interval"), pantr::value_error);
  };
  swallowed({0, 1, 1, 2}, 1);
  swallowed({0, 1, 1, 1, 2}, 1);
  swallowed({0, 0, 1, 1, 1, 2, 2}, 2);
}

TEST_CASE("space1d: the last knot may repeat degree + 1 times, not more, at positive degree") {
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::BsplineSpace1D<double>(
                           sp(std::vector<double>{0, 0, 0, 1, 1, 1, 1}), 2)),
                       "the last knot may repeat at most 3 times", pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(bs::BsplineSpace1D<double>(sp(std::vector<double>{0, 0, 1, 1, 1}), 1)),
      "the last knot may repeat at most 2 times", pantr::value_error);
  // A last knot that merges with the one before it (1e-15 is within the tolerance 1.8e-15) counts
  // the merged copies, with or without snapping.
  const std::vector<double> near_last = {0, 0, 0, 1, 1, 1, 1 + 1e-15};
  CHECK_THROWS_AS(static_cast<void>(bs::BsplineSpace1D<double>(sp(near_last), 2, false, true)),
                  pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(bs::BsplineSpace1D<double>(sp(near_last), 2, false, false)),
                  pantr::value_error);
  // Exactly degree + 1 is a Bezier piece; degree 0 has no denominator and is not limited; a left
  // end repeated more is allowed (it adds an empty function).
  CHECK_NOTHROW(bs::BsplineSpace1D<double>(sp(std::vector<double>{0, 0, 0, 1, 1, 1}), 2));
  CHECK_NOTHROW(bs::BsplineSpace1D<double>(sp(std::vector<double>{0, 1, 1, 1}), 0));
  CHECK_NOTHROW(bs::BsplineSpace1D<double>(sp(std::vector<double>{0, 0, 0, 0, 1, 1, 1}), 2));
}

TEST_CASE(
    "space1d: snapping merges a class onto its first knot, and a mesh it collapses is refused") {
  // Tolerance 8 eps 1 = 1.8e-15; the second 0.5 is 6 ulp (6.7e-16) above the first.
  const double near_half = 0.5 + 6.0 * 0x1p-53;
  const std::vector<double> raw = {0, 0, 0, 0.5, near_half, 1, 1, 1};
  REQUIRE(near_half != 0.5);
  const bs::BsplineSpace1D<double> snapped(sp(raw), 2);
  CHECK(snapped.knots()[3] == 0.5);
  CHECK(snapped.knots()[4] == 0.5);  // the first knot of the class, not an average
  CHECK(snapped.tolerance() == bs::knot_tolerance<double>(sp(raw)));
  CHECK(snapped.num_intervals() == 2);
  // Idempotent: a space built from snapped knots keeps them.
  const std::vector<double> again(snapped.knots().begin(), snapped.knots().end());
  const bs::BsplineSpace1D<double> twice(sp(again), 2);
  CHECK(std::equal(twice.knots().begin(), twice.knots().end(), snapped.knots().begin()));
  // Without snapping the stored knots are the supplied ones, and the classes are the same.
  const bs::BsplineSpace1D<double> kept(sp(raw), 2, false, false);
  CHECK(kept.knots()[4] == near_half);
  CHECK(kept.num_intervals() == 2);
  const auto uk = kept.unique_knots_and_multiplicity(true);
  CHECK(std::vector<std::int64_t>(uk.second.begin(), uk.second.end()) ==
        std::vector<std::int64_t>{3, 2, 3});
  // A float32 mesh finer than the type resolves at 1e6 is refused with the diagnosis, not as flat.
  const std::vector<float> fine = {1e6F,        1e6F,        1e6F,       1e6F + 0.5F,
                                   1e6F + 1.0F, 1e6F + 1.0F, 1e6F + 1.0F};
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::BsplineSpace1D<float>(sp(fine), 2)),
                       doctest::Contains("knot snapping collapsed every knot onto 1000000.0"),
                       pantr::value_error);
  // Not snapping, the same mesh is only flat by the tolerance.
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::BsplineSpace1D<float>(sp(fine), 2, false, false)),
                       doctest::Contains("knot vector spans no interval"), pantr::value_error);
}

TEST_CASE("space1d: from_state keeps the stored knots and tolerance, and does no snapping") {
  const std::vector<double> raw = {0, 0, 0, 1, 1.25, 2, 2, 2};
  const bs::BsplineSpace1D<double> built(sp(raw), 2);
  REQUIRE(built.num_intervals() == 3);
  // The stored tolerance wins: at 0.3 the knots 1 and 1.25 (0.25 apart) are one knot.
  const auto widened = bs::BsplineSpace1D<double>::from_state(sp(raw), 2, false, 0.3);
  CHECK(widened.tolerance() == 0.3);
  CHECK(widened.num_intervals() == 2);
  CHECK(std::equal(widened.knots().begin(), widened.knots().end(), raw.begin()));  // not snapped
  // Round trip: the state of a space rebuilds an equal space.
  const std::vector<double> state(built.knots().begin(), built.knots().end());
  const auto back = bs::BsplineSpace1D<double>::from_state(sp(state), built.degree(),
                                                           built.periodic(), built.tolerance());
  CHECK(back.tolerance() == built.tolerance());
  CHECK(back.num_basis() == built.num_basis());
  CHECK(back.num_intervals() == built.num_intervals());
  CHECK(std::equal(back.knots().begin(), back.knots().end(), built.knots().begin()));
  // It runs the checks of the constructor except the snapping diagnosis: the collapsed float32
  // mesh is "no interval" here.
  const std::vector<float> fine = {1e6F,        1e6F,        1e6F,       1e6F + 0.5F,
                                   1e6F + 1.0F, 1e6F + 1.0F, 1e6F + 1.0F};
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::BsplineSpace1D<float>::from_state(
                           sp(fine), 2, false, bs::knot_tolerance<float>(sp(fine)))),
                       doctest::Contains("knot vector spans no interval"), pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::BsplineSpace1D<double>::from_state(
                           sp(std::vector<double>{0, 1, 0.5, 1}), 1, false, 1e-12)),
                       "knots must be non-decreasing", pantr::value_error);
}

// ---- domain tests -----------------------------------------------------------------------------

TEST_CASE("space1d: a point is inside up to exactly the space's tolerance") {
  const std::vector<double> knots = {0, 0, 0, 1, 1, 1};
  const bs::BsplineSpace1D<double> s(sp(knots), 2);
  const double tol = s.tolerance();
  REQUIRE(tol == 0x1p-49);  // 8 eps scale with scale 1: a power of two, so the bounds are exact
  CHECK(s.contains<double>(-tol));
  CHECK_FALSE(s.contains<double>(std::nextafter(-tol, -kInf)));
  CHECK(s.contains<double>(1.0 + tol));
  CHECK_FALSE(s.contains<double>(std::nextafter(1.0 + tol, kInf)));
  CHECK(s.contains<float>(0.5F));
  CHECK_FALSE(s.contains<float>(1.001F));
  CHECK_FALSE(s.contains<double>(kNaN));
  const std::vector<double> mixed = {0.0, 0.5, kNaN};
  CHECK_FALSE(s.all_in_domain<double>(mixed));
  const std::vector<double> fine = {0.0, 0.5, 1.0};
  CHECK(s.all_in_domain<double>(fine));
  CHECK(s.all_in_domain<double>(std::span<const double>{}));  // nothing to refuse
}

TEST_CASE("space1d: check_in_domain names the points and the domain as Python spelled them") {
  const std::vector<double> knots = {0, 0, 0, 1, 2, 2, 3, 3, 3};
  const bs::BsplineSpace1D<double> s(sp(knots), 2);
  const std::vector<double> out = {1.0, 3.5};
  CHECK_THROWS_WITH_AS(s.check_in_domain<double>(out),
                       "One or more values in pts are outside the knot vector domain (0.0, 3.0)",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(s.check_in_domain<double>(out, "u"),
                       "One or more values in u are outside the knot vector domain (0.0, 3.0)",
                       pantr::value_error);
  // A float space spells its float knots as the doubles they are.
  const std::vector<float> fk = {0.1F, 0.1F, 0.1F, 0.9F, 0.9F, 0.9F};
  const bs::BsplineSpace1D<float> fs(sp(fk), 2);
  CHECK_THROWS_WITH_AS(
      fs.check_in_domain<float>(std::vector<float>{2.0F}),
      "One or more values in pts are outside the knot vector domain (0.10000000149011612, "
      "0.8999999761581421)",
      pantr::value_error);
}

// ---- the call forms ---------------------------------------------------------------------------

TEST_CASE("space1d: the out and convenience forms agree, and every size is checked") {
  const std::vector<double> knots = {0, 0, 0, 1, 2, 2, 3, 3, 3};
  const bs::BsplineSpace1D<double> s(sp(knots), 2);
  const std::vector<double> pts = {0.1, 1.5, 2.9};
  const auto [v, f] = s.tabulate_basis<double>(pts);
  CHECK(v.shape()[0] == 3);
  CHECK(v.shape()[1] == 3);
  CHECK(f.shape().size() == 1);
  CHECK(f.shape()[0] == 3);
  NdArray<double> out({3, 3});
  NdArray<std::int64_t> first({3});
  s.tabulate_basis<double>(pts, out.view<2>(), first.flat());
  CHECK(std::equal(out.data(), out.data() + 9, v.data()));
  CHECK(std::equal(first.data(), first.data() + 3, f.data()));
  const auto d = s.tabulate_basis_derivatives<double>(pts, 2);
  CHECK(d.values.shape()[1] == 3);
  NdArray<double> dout({3, 3, 3});
  NdArray<std::int64_t> dfirst({3});
  CHECK(s.tabulate_basis_derivatives<double>(pts, 2, dout.view<3>(), dfirst.flat()));
  CHECK(std::equal(dout.data(), dout.data() + 27, d.values.data()));
  // Wrong extents are value_errors spelled as Python tuples.
  NdArray<double> few({2, 3});
  NdArray<std::int64_t> short_first({2});
  CHECK_THROWS_WITH_AS(s.tabulate_basis<double>(pts, few.view<2>(), first.flat()),
                       "Output array has shape (2, 3), but expected shape (3, 3)",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(s.tabulate_basis<double>(pts, out.view<2>(), short_first.flat()),
                       "Output array has shape (2,), but expected shape (3,)", pantr::value_error);
  NdArray<double> dwrong({3, 2, 3});
  CHECK_THROWS_WITH_AS(static_cast<void>(s.tabulate_basis_derivatives<double>(
                           pts, 2, dwrong.view<3>(), dfirst.flat())),
                       "Output array has shape (3, 2, 3), but expected shape (3, 3, 3)",
                       pantr::value_error);
  // A negative order is refused first, before any extents are looked at.
  CHECK_THROWS_WITH_AS(static_cast<void>(s.tabulate_basis_derivatives<double>(
                           pts, -1, dwrong.view<3>(), dfirst.flat())),
                       "n_deriv must be non-negative, got -1", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(s.tabulate_basis_derivatives<double>(pts, -1)),
                       "n_deriv must be non-negative, got -1", pantr::value_error);
}

TEST_CASE("space1d: a refused call writes nothing") {
  const std::vector<double> knots = {0, 0, 0, 1, 1, 1};
  const bs::BsplineSpace1D<double> s(sp(knots), 2);
  const std::vector<double> bad = {0.5, 1.5};
  NdArray<double> out({2, 3});
  NdArray<std::int64_t> first({2});
  std::fill(out.data(), out.data() + 6, 7.0);
  std::fill(first.data(), first.data() + 2, std::int64_t{7});
  CHECK_THROWS_AS(s.tabulate_basis<double>(bad, out.view<2>(), first.flat()), pantr::value_error);
  CHECK(std::all_of(out.data(), out.data() + 6, [](double x) { return x == 7.0; }));
  CHECK(std::all_of(first.data(), first.data() + 2, [](std::int64_t x) { return x == 7; }));
  NdArray<double> dout({2, 2, 3});
  std::fill(dout.data(), dout.data() + 12, 7.0);
  CHECK_THROWS_AS(
      static_cast<void>(s.tabulate_basis_derivatives<double>(bad, 1, dout.view<3>(), first.flat())),
      pantr::value_error);
  CHECK(std::all_of(dout.data(), dout.data() + 12, [](double x) { return x == 7.0; }));
}

TEST_CASE_TEMPLATE("space1d: no points give empty arrays of the right shape", T, float, double) {
  for (const bool bezier : {true, false}) {
    const std::vector<T> knots =
        bezier ? cast_all<T>({0, 0, 0, 1, 1, 1}) : cast_all<T>({0, 0, 0, 1, 2, 2, 2});
    const bs::BsplineSpace1D<T> s(sp(knots), 2);
    const std::vector<T> none;
    const auto [v, f] = s.template tabulate_basis<T>(none);
    CHECK(v.size() == 0);
    CHECK(v.shape()[1] == 3);
    CHECK(f.size() == 0);
    const auto d = s.template tabulate_basis_derivatives<T>(none, 2);
    CHECK(d.values.size() == 0);
    CHECK(d.values.shape()[1] == 3);
    CHECK(d.complete);
  }
}

// ---- cardinal intervals, restriction and Greville ---------------------------------------------

TEST_CASE("space1d: cardinal intervals are flagged into an output of exactly the interval count") {
  const auto knots = bs::uniform_open_knots<double>(4, 2, 1, 0.0, 4.0);
  const bs::BsplineSpace1D<double> s(sp(knots), 2);
  std::array<bool, 4> flags{};
  s.cardinal_intervals(std::span<bool>(flags));
  // The two end intervals touch a knot of multiplicity 3; the inner two see unit lengths around.
  CHECK(flags == std::array<bool, 4>{false, true, true, false});
  std::array<bool, 3> few{};
  CHECK_THROWS_WITH_AS(s.cardinal_intervals(std::span<bool>(few)),
                       "Output array has shape (3,), but expected shape (4,)", pantr::value_error);
}

TEST_CASE("space1d: restricting to intervals gives the parent's functions on them") {
  const std::vector<double> knots = {0, 0, 0, 1, 2, 2, 3, 3, 3};
  const bs::BsplineSpace1D<double> s(sp(knots), 2);
  // Worked out by hand: the intervals [1, 3) have the functions 1..5 and the knots 1..8.
  {
    const auto [w, map] = s.restrict(1, 3);
    CHECK(map == std::vector<std::int64_t>{1, 2, 3, 4, 5});
    CHECK(std::vector<double>(w.knots().begin(), w.knots().end()) ==
          std::vector<double>{0, 0, 1, 2, 2, 3, 3, 3});
    CHECK(w.domain() == std::make_pair(1.0, 3.0));
  }
  {
    const auto [w, map] = s.restrict(0, 1);
    CHECK(map == std::vector<std::int64_t>{0, 1, 2});
    CHECK(std::vector<double>(w.knots().begin(), w.knots().end()) ==
          std::vector<double>{0, 0, 0, 1, 2, 2});
  }
  {
    const auto [w, map] = s.restrict(0, 3);
    CHECK(map == std::vector<std::int64_t>{0, 1, 2, 3, 4, 5});
    CHECK(std::equal(w.knots().begin(), w.knots().end(), knots.begin(), knots.end()));
  }
  {
    const auto [w, map] = s.restrict(2, 3);
    CHECK(map == std::vector<std::int64_t>{3, 4, 5});
    CHECK(std::vector<double>(w.knots().begin(), w.knots().end()) ==
          std::vector<double>{1, 2, 2, 3, 3, 3});
  }
}

// Bug (present in the previous release as well): restrict builds the window from the slice of
// knots, so its tolerance is derived from the slice's magnitude and not the parent's. A parent
// built with snap_knots == false that holds knots 1 and 1 + 1e-12 (one knot at the parent's
// tolerance of 1.8e-9, because of the 1e6 at its far end) gives a window whose tolerance is
// 7e-15, where they are two knots: three intervals for a range of two. Marked as an expected
// failure so the suite stays green; when the window takes the parent's tolerance this case fails
// and must be unmarked.
TEST_CASE("space1d: restricting an unsnapped space keeps the interval count" *
          doctest::should_fail()) {
  const std::vector<double> knots = {0, 0, 0, 1, 1 + 1e-12, 2, 3, 4, 1e6, 1e6, 1e6};
  const bs::BsplineSpace1D<double> s(sp(knots), 2, false, false);
  REQUIRE(s.num_intervals() == 5);
  const auto [w, map] = s.restrict(0, 2);
  CHECK(w.num_intervals() == 2);
}

TEST_CASE("space1d: restriction refuses a periodic space and a range outside the intervals") {
  const std::vector<double> knots = {0, 0, 0, 1, 2, 2, 3, 3, 3};
  const bs::BsplineSpace1D<double> s(sp(knots), 2);
  const auto card = bs::cardinal_knots<double>(4, 2);
  const bs::BsplineSpace1D<double> p(sp(card), 2, true);
  CHECK_THROWS_WITH_AS(static_cast<void>(p.restrict(0, 1)),
                       "restrict: periodic B-spline spaces are not supported.", pantr::value_error);
  const auto refused = [&](long long lo, long long hi, const char* message) {
    CAPTURE(lo);
    CAPTURE(hi);
    CHECK_THROWS_WITH_AS(static_cast<void>(s.restrict(lo, hi)), message, pantr::value_error);
  };
  refused(-1, 1, "restrict: require 0 <= interval_lo < interval_hi <= 3; got [-1, 1).");
  refused(1, 1, "restrict: require 0 <= interval_lo < interval_hi <= 3; got [1, 1).");
  refused(2, 1, "restrict: require 0 <= interval_lo < interval_hi <= 3; got [2, 1).");
  refused(0, 4, "restrict: require 0 <= interval_lo < interval_hi <= 3; got [0, 4).");
}

TEST_CASE("space1d: the space's Greville abscissae use its domain to wrap a periodic space") {
  using V = std::vector<double>;
  const V open = {0, 0, 0, 1, 2, 2, 3, 3, 3};
  CHECK(bs::BsplineSpace1D<double>(sp(open), 2).greville() == V{0, 0.5, 1.5, 2, 2.5, 3});
  const auto card = bs::cardinal_knots<double>(4, 2);
  CHECK(bs::BsplineSpace1D<double>(sp(card), 2, true).greville() == V{0.5, 1.5, 2.5, 3.5});
  const std::vector<float> f = {0, 0, 0, 0.5F, 1, 1, 1};
  CHECK(bs::BsplineSpace1D<float>(sp(f), 2).greville() == std::vector<float>{0, 0.25F, 0.75F, 1});
}
