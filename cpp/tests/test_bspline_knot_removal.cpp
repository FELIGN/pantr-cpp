// Knot removal of a B-spline: what the default removes, what a tolerance removes, the tolerance
// helpers, and the refusals with their messages.
//
// Oracles. A knot just inserted (by the exact Oslo refinement the library already tests against
// Boehm's insertion) is redundant by construction, so removing it must give back the knot vector
// bit for bit and the control points to round-off; and the removed spline must be the same
// mapping, which the exact-rational Cox-de Boor evaluation of bspline_exact_eval.hpp decides with
// no rounding of its own. A knot the curve needs must stay under the default and go under a
// budget larger than the geometry. The tolerance helpers are checked against the closed forms of
// their derivations (specification pins, not a second implementation of the algorithm).
//
// Tolerances are the conservative tier of `pantr::tolerance` (a long accumulation, or a
// condition number up to about 1e3) times the control-point scale: a removal runs the one-sided
// reconstructions of A5.8 over at most `degree + 1` points from each side, and the knot spans of
// the cases below keep the `1 / alpha` amplification of those steps far under a thousand.
#include <doctest/doctest.h>

#include "bspline_test_support.hpp"

#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/knot_insertion.hpp>
#include <pantr/bspline/knot_removal.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/core/format.hpp>
#include <pantr/tolerance.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace bs = pantr::bspline;
namespace oracle = bspline_oracle;
using bspline_test::cast_all;
using bspline_test::clamped;
using bspline_test::Dir;
using bspline_test::exact_difference;
using bspline_test::insert_in;
using bspline_test::knots_of;
using bspline_test::make_spline;
using bspline_test::max_difference;
using bspline_test::message_of;
using bspline_test::remove_in;
using bspline_test::sp;
using bspline_test::tier_tolerance;
using bspline_test::values;

TEST_CASE_TEMPLATE("knot removal: a knot just inserted comes out, knots bit for bit", T, double,
                   float) {
  // Oracle: the original spline. The knot vector must come back identical; the control points to
  // the conservative tier (the insertion and the removal each run a short convex or affine chain);
  // and the mapping is the same one, decided in exact arithmetic.
  for (int p = 1; p <= 5; ++p) {
    CAPTURE(p);
    const auto dirs = std::vector<Dir>{{clamped(p, {0.25, 0.5, 0.75}), p}};
    const bs::Bspline<T> original = make_spline<T>(dirs, 3, false, 11U + static_cast<unsigned>(p));
    const bs::Bspline<T> inserted = insert_in<T>(original, 0, {0.375});
    REQUIRE(inserted.space().spaces()[0].knots().size() ==
            original.space().spaces()[0].knots().size() + 1);

    const bs::Bspline<T> back = remove_in<T>(inserted, 0, {0.375}, std::nullopt, std::nullopt);
    CHECK(knots_of<T>(back, 0) == knots_of<T>(original, 0));
    REQUIRE(back.control_points().size() == original.control_points().size());
    CHECK(max_difference<T>(back.control_points(), original.control_points()) <=
          tier_tolerance(original));
    CHECK(exact_difference<T>(back, inserted) <= tier_tolerance(original));
  }
}

TEST_CASE_TEMPLATE("knot removal: a repeated knot comes out as often as it was inserted", T, double,
                   float) {
  // A knot of multiplicity s is removed s times by the default (the count is capped at the degree,
  // which is also the largest multiplicity an interior knot of a continuous spline has), and
  // `num` limits it. Every (degree, multiplicity) pair up to degree 6 reaches the kernel's
  // scratch buffer at its bound, which is what the sanitizer builds of this test are for.
  for (int p = 1; p <= 6; ++p) {
    for (int s = 1; s <= p; ++s) {
      CAPTURE(p);
      CAPTURE(s);
      const auto dirs = std::vector<Dir>{{clamped(p, {0.25, 0.75}), p}};
      const bs::Bspline<T> original =
          make_spline<T>(dirs, 2, false, 31U * static_cast<unsigned>(p) + static_cast<unsigned>(s));
      const bs::Bspline<T> inserted =
          insert_in<T>(original, 0, std::vector<double>(static_cast<std::size_t>(s), 0.5));

      const bs::Bspline<T> all = remove_in<T>(inserted, 0, {0.5}, std::nullopt, std::nullopt);
      CHECK(knots_of<T>(all, 0) == knots_of<T>(original, 0));
      CHECK(max_difference<T>(all.control_points(), original.control_points()) <=
            tier_tolerance(original));

      const bs::Bspline<T> one = remove_in<T>(inserted, 0, {0.5}, 1, std::nullopt);
      CHECK(knots_of<T>(one, 0).size() == knots_of<T>(inserted, 0).size() - 1);
      // A request for more than the multiplicity is capped, not refused.
      const bs::Bspline<T> capped = remove_in<T>(inserted, 0, {0.5}, 1000, std::nullopt);
      CHECK(knots_of<T>(capped, 0) == knots_of<T>(original, 0));
    }
  }
}

TEST_CASE_TEMPLATE(
    "knot removal: a knot the curve needs stays under the default and goes under a "
    "large budget",
    T, double, float) {
  for (int p = 2; p <= 4; ++p) {
    CAPTURE(p);
    const auto dirs = std::vector<Dir>{{clamped(p, {0.25, 0.5, 0.75}), p}};
    const bs::Bspline<T> b = make_spline<T>(dirs, 2, false, 5U + static_cast<unsigned>(p));

    // The default is lossless: nothing is removed, and the control points are returned unchanged.
    const bs::Bspline<T> kept = remove_in<T>(b, 0, {0.5}, std::nullopt, std::nullopt);
    CHECK(knots_of<T>(kept, 0) == knots_of<T>(b, 0));
    CHECK(max_difference<T>(kept.control_points(), b.control_points()) == 0.0);

    // A budget larger than the geometry removes it (one removal: the knot has multiplicity one).
    const bs::Bspline<T> gone = remove_in<T>(b, 0, {0.5}, std::nullopt, 1.0e6);
    CHECK(knots_of<T>(gone, 0).size() == knots_of<T>(b, 0).size() - 1);

    // A tolerance of zero accepts only a removal that reconstructs the points exactly.
    const bs::Bspline<T> zero = remove_in<T>(b, 0, {0.5}, std::nullopt, 0.0);
    CHECK(knots_of<T>(zero, 0) == knots_of<T>(b, 0));
    // An infinite budget accepts any, a NaN none.
    CHECK(knots_of<T>(
              remove_in<T>(b, 0, {0.5}, std::nullopt, std::numeric_limits<double>::infinity()), 0)
              .size() == knots_of<T>(b, 0).size() - 1);
    CHECK(knots_of<T>(
              remove_in<T>(b, 0, {0.5}, std::nullopt, std::numeric_limits<double>::quiet_NaN()),
              0) == knots_of<T>(b, 0));
  }
}

TEST_CASE_TEMPLATE(
    "knot removal: surfaces and volumes remove per direction, the floor spanning the "
    "stacked points",
    T, double, float) {
  const std::vector<Dir> dirs = {
      {clamped(3, {0.25, 0.5, 0.75}), 3}, {clamped(2, {0.5}), 2}, {clamped(2, {0.25, 0.75}), 2}};
  const bs::Bspline<T> original = make_spline<T>(dirs, 2, false, 77U);

  bs::Bspline<T> refined = insert_in<T>(original, 0, {0.375, 0.375});
  refined = insert_in<T>(refined, 1, {0.25});
  refined = insert_in<T>(refined, 2, {0.5});

  const std::vector<T> v0 = cast_all<T>({0.375});
  const std::vector<T> v1 = cast_all<T>({0.25});
  const std::vector<T> v2 = cast_all<T>({0.5});
  const std::vector<std::span<const T>> per = {sp(v0), sp(v1), sp(v2)};
  const bs::Bspline<T> back = bs::remove_knots<T>(refined, per, std::nullopt, std::nullopt);

  for (std::size_t d = 0; d < 3; ++d) {
    CHECK(knots_of<T>(back, d) == knots_of<T>(original, d));
  }
  CHECK(max_difference<T>(back.control_points(), original.control_points()) <=
        tier_tolerance(original));
  CHECK(exact_difference<T>(back, refined) <= tier_tolerance(original));

  // A direction with no values is left as it was, sharing its space.
  const bs::Bspline<T> only_one = remove_in<T>(refined, 1, {0.25}, std::nullopt, std::nullopt);
  CHECK(knots_of<T>(only_one, 0) == knots_of<T>(refined, 0));
  CHECK(knots_of<T>(only_one, 2) == knots_of<T>(refined, 2));
  CHECK(knots_of<T>(only_one, 1) == knots_of<T>(original, 1));
}

TEST_CASE_TEMPLATE("knot removal: a rational spline is graded in projected space", T, double,
                   float) {
  const auto dirs = std::vector<Dir>{{clamped(3, {0.25, 0.5, 0.75}), 3}};
  const bs::Bspline<T> original = make_spline<T>(dirs, 2, true, 9U);
  const bs::Bspline<T> inserted = insert_in<T>(original, 0, {0.375});

  // Exactly redundant: out under the default floor, and under an explicit tolerance.
  for (const std::optional<double> tol : {std::optional<double>{}, std::optional<double>{1.0e-3}}) {
    const bs::Bspline<T> back = remove_in<T>(inserted, 0, {0.375}, std::nullopt, tol);
    CHECK(knots_of<T>(back, 0) == knots_of<T>(original, 0));
    CHECK(exact_difference<T>(back, inserted) <= tier_tolerance(original));
  }

  // A weight that is not strictly positive cannot be converted to a homogeneous budget.
  std::vector<double> cp(original.control_points().begin(), original.control_points().end());
  cp[2] = 0.0;  // the weight of the first control point
  const std::vector<T> cpt = cast_all<T>(cp);
  const bs::Bspline<T> degenerate(original.space(), sp(cpt), true);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(remove_in<T>(degenerate, 0, {0.5}, std::nullopt, 1.0e-3)),
      "Knot removal with an explicit tolerance needs strictly positive weights to convert the "
      "tolerance into homogeneous coordinates; the smallest is 0.0.",
      pantr::value_error);
  // The default floor needs no conversion: it is a statement about the arithmetic.
  CHECK_NOTHROW(static_cast<void>(remove_in<T>(degenerate, 0, {0.5}, std::nullopt, std::nullopt)));
}

TEST_CASE("knot removal: the round-off floor is the derivation's closed form") {
  // Specification pin: floor = 2 * strict * (degree + 1) * scale * sqrt(num_stacked), and the
  // scale is max(bounding-box diagonal, largest norm) of the rows.
  const std::vector<double> flat = {3.0, 4.0, -1.0, 0.0, 0.5, -6.0};  // three rows of two
  const std::span<const double> ctrl(flat);
  // Diagonal: extents (4, 10) -> sqrt(116); largest norm: sqrt(37).
  const double diagonal = std::sqrt(4.0 * 4.0 + 10.0 * 10.0);
  CHECK(bs::control_point_scale<double>(ctrl, 2) == diagonal);
  const double strict = static_cast<double>(pantr::tolerance::get_strict<double>());
  for (int degree : {1, 3, 7}) {
    CHECK(bs::roundoff_deviation_floor<double>(ctrl, 2, degree, 1) ==
          2.0 * strict * static_cast<double>(degree + 1) * diagonal);
  }
  // The stacking factor is a square root: 10000 stacked points is a factor of 100.
  CHECK(
      bs::roundoff_deviation_floor<double>(ctrl, 2, 3, 10000) ==
      doctest::Approx(100.0 * bs::roundoff_deviation_floor<double>(ctrl, 2, 3, 1)).epsilon(1e-14));
  // A row larger than the diagonal wins; a NaN coordinate makes the scale NaN, as NumPy's max does.
  const std::vector<double> one_row = {3.0, 4.0};
  CHECK(bs::control_point_scale<double>(std::span<const double>(one_row), 2) == 5.0);
  const std::vector<double> with_nan = {3.0, std::numeric_limits<double>::quiet_NaN(), 1.0, 2.0};
  CHECK(std::isnan(bs::control_point_scale<double>(std::span<const double>(with_nan), 2)));
}

TEST_CASE("knot removal: the stacked point count is the product of the other extents") {
  // Asserted on the count itself, because no behavioural test can catch this: the floor carries a
  // large margin, so any positive multiplier still admits an exactly redundant knot. A lopsided
  // net separates the two natural mistakes by three orders of magnitude.
  const std::vector<std::size_t> lopsided = {7, 4003, 3};
  CHECK(bs::num_stacked_control_points(lopsided, 0) == 4003);
  CHECK(bs::num_stacked_control_points(lopsided, 1) == 7);
  CHECK(bs::num_stacked_control_points(std::vector<std::size_t>{9, 3}, 0) == 1);
  const std::vector<std::size_t> volume = {5, 6, 7, 3};
  CHECK(bs::num_stacked_control_points(volume, 0) == 42);
  CHECK(bs::num_stacked_control_points(volume, 1) == 35);
  CHECK(bs::num_stacked_control_points(volume, 2) == 30);
}

TEST_CASE("knot removal: the homogeneous tolerance is Eq. (5.30) of The NURBS Book") {
  // Specification pin: TOL = d * w_min / (1 + |P|_max), |P|_max the largest projected norm.
  const std::vector<double> flat = {2.0, 4.0, 2.0,   // P = (1, 2), w = 2
                                    3.0, 0.0, 0.5,   // P = (6, 0), w = 0.5
                                    1.0, 1.0, 1.0};  // P = (1, 1), w = 1
  const double p_max = 6.0;
  CHECK(bs::homogeneous_deviation_tolerance<double>(std::span<const double>(flat), 3, 1.0e-3) ==
        1.0e-3 * 0.5 / (1.0 + p_max));
  const std::vector<double> bad = {1.0, 1.0, 0.0};
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::homogeneous_deviation_tolerance<double>(
                           std::span<const double>(bad), 3, 1.0)),
                       "Knot removal with an explicit tolerance needs strictly positive weights to "
                       "convert the tolerance into homogeneous coordinates; the smallest is 0.0.",
                       pantr::value_error);
}

TEST_CASE("knot removal: the knot lookup is absolute and compares in the storage type") {
  const std::vector<double> kd = {0, 0, 0, 1, 2, 3, 3, 3};
  // An exact hit, a near miss relative 1e-6 away, and a value within the absolute tolerance.
  for (const double base : {1.0, 100.0, 1.0e6}) {
    std::vector<double> scaled = kd;
    for (double& v : scaled) {
      v *= base;
    }
    const std::pair<std::size_t, std::int64_t> hit =
        bs::detail::find_knot_index_and_multiplicity<double>(sp(scaled), 2, base, 1.0e-15);
    CHECK(hit.first == 3);
    CHECK(hit.second == 1);
    CHECK_THROWS_AS(static_cast<void>(bs::detail::find_knot_index_and_multiplicity<double>(
                        sp(scaled), 2, base * (1.0 + 1.0e-6), 1.0e-15)),
                    pantr::value_error);
  }
  // The last index of a repeated knot, and the message with the Python spellings.
  const std::pair<std::size_t, std::int64_t> end =
      bs::detail::find_knot_index_and_multiplicity<double>(sp(kd), 2, 3.0, 1.0e-12);
  CHECK(end.first == 7);
  CHECK(end.second == 3);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::detail::find_knot_index_and_multiplicity<double>(
                           sp(kd), 2, 1.5, 1.0e-12)),
                       "Knot value 1.5 not found in knot vector (tolerance=1e-12).",
                       pantr::value_error);
  // A float tolerance is rounded to float before the comparison, as NumPy compares a float32 array
  // with a Python float. The two float knots below are 2^-23 apart; a tolerance a hair under that
  // gap rounds to it as a float, so the lookup finds the knot, where a comparison made in double
  // would not; and a tolerance of half the gap finds nothing under either.
  const float one = 1.0F;
  const float next = std::nextafter(one, 2.0F);
  const double gap = static_cast<double>(next - one);
  const std::vector<float> kf = {0.0F, 0.0F, 0.0F, one, 5.0F, 5.0F, 5.0F};
  CHECK_NOTHROW(static_cast<void>(bs::detail::find_knot_index_and_multiplicity<float>(
      std::span<const float>(kf), 2, static_cast<double>(next), gap * (1.0 - 1.0e-8))));
  CHECK_THROWS_AS(static_cast<void>(bs::detail::find_knot_index_and_multiplicity<float>(
                      std::span<const float>(kf), 2, static_cast<double>(next), gap / 2.0)),
                  pantr::value_error);
}

TEST_CASE("knot removal: refusals and their messages") {
  const std::vector<double> k = clamped(3, {0.25, 0.5, 0.75});
  const auto dirs = std::vector<Dir>{{k, 3}};
  const bs::Bspline<double> b = make_spline<double>(dirs, 2, false, 3U);
  const double tol_space = b.space().spaces()[0].tolerance();
  const auto call = [&](const std::vector<double>& drop,
                        std::optional<long long> num = std::nullopt) {
    return [&b, drop, num] { static_cast<void>(remove_in<double>(b, 0, drop, num, std::nullopt)); };
  };
  const auto what = [](const std::function<void()>& f) { return message_of(f); };

  CHECK(what(call({0.5}, 0)) == "num must be a positive integer or None, got 0.");
  CHECK(what(call({0.5}, -3)) == "num must be a positive integer or None, got -3.");
  CHECK(what(call({0.3})) ==
        "Knot value 0.3 not found in knot vector (tolerance=" + pantr::py_repr(tol_space) + ").");
  CHECK(what(call({0.0})) == "Cannot remove boundary knot 0.0 (domain start).");
  CHECK(what(call({1.0})) == "Cannot remove boundary knot 1.0 (domain end).");

  // A periodic direction is refused first, with its direction.
  const std::vector<double> per_knots = {-0.75, -0.5, -0.25, 0.0, 0.25, 0.5,
                                         0.75,  1.0,  1.25,  1.5, 1.75};
  std::vector<bs::BsplineSpace1D<double>> spaces;
  spaces.emplace_back(sp(per_knots), 2, true);
  const std::vector<double> cp = values(static_cast<std::size_t>(spaces[0].num_basis()), 5U);
  const bs::Bspline<double> periodic(bs::BsplineSpace<double>(std::move(spaces)), sp(cp), false);
  CHECK(message_of([&] {
          static_cast<void>(remove_in<double>(periodic, 0, {0.25}, std::nullopt, std::nullopt));
        }) == "Knot removal is not supported for periodic B-splines (direction 0 is periodic).");

  // The dimension, and a request that names nothing.
  const std::vector<double> v = {0.5};
  const std::vector<std::span<const double>> two = {sp(v), sp(v)};
  CHECK(message_of([&] {
          static_cast<void>(bs::remove_knots<double>(b, two, std::nullopt, std::nullopt));
        }) == "knot_values sequence length (2) must match dim (1).");
  const std::vector<std::span<const double>> none = {std::span<const double>()};
  CHECK(message_of([&] {
          static_cast<void>(bs::remove_knots<double>(b, none, std::nullopt, std::nullopt));
        }) == "At least one direction must have a non-empty array of knot values to remove.");
}

TEST_CASE("knot removal: a knot outside the domain is refused, not read out of bounds") {
  // The ghost knots of an unclamped vector are knots of the vector, and the previous
  // implementation removed them by reading and writing outside its arrays. They are refused now.
  const std::vector<double> k = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  const auto dirs = std::vector<Dir>{{k, 2}};
  const bs::Bspline<double> b = make_spline<double>(dirs, 1, false, 8U);
  for (const double v : {0.0, 1.0, 7.0, 8.0}) {
    CAPTURE(v);
    const std::string msg =
        message_of([&] { static_cast<void>(remove_in<double>(b, 0, {v}, std::nullopt, 1.0e9)); });
    CHECK(msg.find("is not an interior knot of the domain [2.0, 6.0].") != std::string::npos);
  }
  // The interior knots of the same vector are fine: they are the ordinary case.
  for (const double v : {3.0, 4.0, 5.0}) {
    CAPTURE(v);
    const bs::Bspline<double> r = remove_in<double>(b, 0, {v}, std::nullopt, 1.0e9);
    CHECK(knots_of<double>(r, 0).size() == k.size() - 1);
  }
}

TEST_CASE_TEMPLATE("knot removal: random meshes, every multiplicity, are removed as inserted", T,
                   double, float) {
  // Insert a random multiset of grid knots, then remove them under a budget of the conservative
  // tier: the knot vector must be the original again, whatever the multiplicities and the degree
  // (the sanitizer builds run this sweep over every index the kernel touches). The budget is not
  // the default floor: the floor is an operation count, and on a mesh like this one (several
  // copies of knots close to the end of the domain at degree 5) the removal of the last copy sits
  // within a few percent of it, so a verdict there reports the rounding of the insertion that made
  // the input and not the algorithm.
  std::uint32_t state = 0x85EBCA6BU;
  const auto next = [&state](std::uint32_t bound) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state % bound;
  };
  for (int n = 0; n < 120; ++n) {
    const int p = 1 + static_cast<int>(next(5));
    CAPTURE(n);
    CAPTURE(p);
    const std::vector<double> base_interior = {0.25, 0.75};
    const bs::Bspline<T> original =
        make_spline<T>({{clamped(p, base_interior), p}}, 2, (n % 4) == 0, static_cast<unsigned>(n));
    // Up to three new distinct knots on the grid of sixteenths, each p or fewer times.
    std::vector<double> added;
    std::vector<double> distinct;
    for (std::uint32_t k = 0; k < 1 + next(3); ++k) {
      const double v = static_cast<double>(1 + next(15)) / 16.0;
      if (v == 0.25 || v == 0.75 ||
          std::find(distinct.begin(), distinct.end(), v) != distinct.end()) {
        continue;
      }
      distinct.push_back(v);
      added.insert(added.end(), 1 + next(static_cast<std::uint32_t>(p)), v);
    }
    if (added.empty()) {
      continue;
    }
    std::sort(added.begin(), added.end());
    const bs::Bspline<T> inserted = insert_in<T>(original, 0, added);
    const bs::Bspline<T> back =
        remove_in<T>(inserted, 0, distinct, std::nullopt, tier_tolerance(original));
    CHECK(knots_of<T>(back, 0) == knots_of<T>(original, 0));
    CHECK(max_difference<T>(back.control_points(), original.control_points()) <=
          tier_tolerance(original));
  }
}
