// AABB. Oracles: hand-computed boxes (exact in binary: small integers and halves), the identity
// and absorbing elements of union and intersection, and the exact image of a box under a map
// whose entries are signed integers or zero against infinite bounds.
#include <doctest/doctest.h>

#include <pantr/geometry/aabb.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

using pantr::geometry::AABB;
constexpr double inf = std::numeric_limits<double>::infinity();

namespace {

AABB box(const std::vector<double>& lo, const std::vector<double>& hi) { return {lo, hi}; }

}  // namespace

TEST_CASE("AABB: construction refuses mismatched, empty and NaN corners") {
  const std::vector<double> two = {0.0, 1.0};
  const std::vector<double> three = {0.0, 1.0, 2.0};
  const std::vector<double> none;
  const std::vector<double> with_nan = {0.0, std::nan("")};
  CHECK_THROWS_WITH_AS(AABB(two, three), "AABB.lo and AABB.hi must share shape; got (2,) vs (3,).",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(AABB(none, none), "AABB ndim must be >= 1; got 0.", pantr::value_error);
  CHECK_THROWS_WITH_AS(AABB(with_nan, two),
                       "AABB bounds must not contain NaN; got lo=[0.0, nan], hi=[0.0, 1.0].",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(AABB::empty(0)), "AABB.empty: ndim must be >= 1; got 0.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(AABB::unbounded(-1)),
                       "AABB.unbounded: ndim must be >= 1; got -1.", pantr::value_error);
}

TEST_CASE("AABB: emptiness, containment and overlap on the closed box") {
  const AABB a = box({0.0, 0.0}, {1.0, 2.0});
  CHECK_FALSE(a.is_empty());
  CHECK(AABB::empty(2).is_empty());
  CHECK_FALSE(AABB::unbounded(2).is_empty());
  const std::vector<double> corner = {1.0, 2.0};
  const std::vector<double> outside = {1.5, 0.0};
  CHECK(a.contains_point(corner));
  CHECK_FALSE(a.contains_point(outside));
  CHECK_FALSE(AABB::empty(2).contains_point(corner));
  const std::vector<double> short_x = {0.5};
  const std::vector<double> nan_x = {0.5, std::nan("")};
  CHECK_THROWS_WITH_AS(static_cast<void>(a.contains_point(short_x)),
                       "contains_point: x must have length 2; got (1,).", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(a.contains_point(nan_x)),
                       "contains_point: x must not contain NaN.", pantr::value_error);
  const AABB touching = box({1.0, 2.0}, {3.0, 3.0});
  const AABB apart = box({1.5, 0.0}, {3.0, 3.0});
  CHECK(a.overlaps(touching));
  CHECK_FALSE(a.overlaps(apart));
  CHECK_FALSE(a.overlaps(AABB::empty(2)));
  CHECK_THROWS_WITH_AS(static_cast<void>(a.overlaps(AABB::empty(3))),
                       "AABB.overlaps: dimension mismatch (a.ndim=2 vs b.ndim=3).",
                       pantr::value_error);
}

TEST_CASE("AABB: union has the empty box as identity and returns that operand itself") {
  const AABB a = box({0.0, 0.0}, {1.0, 2.0});
  const AABB e = AABB::empty(2);
  CHECK(a.union_with(e).same_rep(a));
  CHECK(e.union_with(a).same_rep(a));
  const AABB u = a.union_with(box({-1.0, 1.0}, {0.5, 3.0}));
  CHECK(u == box({-1.0, 0.0}, {1.0, 3.0}));
  CHECK_THROWS_AS(static_cast<void>(a.union_with(AABB::empty(1))), pantr::value_error);
}

TEST_CASE("AABB: intersection, empty when disjoint or with an empty operand") {
  const AABB a = box({0.0, 0.0}, {1.0, 2.0});
  const std::optional<AABB> i = a.intersect(box({0.5, -1.0}, {3.0, 1.0}));
  REQUIRE(i.has_value());
  CHECK(*i == box({0.5, 0.0}, {1.0, 1.0}));
  CHECK_FALSE(a.intersect(box({2.0, 0.0}, {3.0, 1.0})).has_value());
  CHECK_FALSE(a.intersect(AABB::empty(2)).has_value());
}

TEST_CASE("AABB: pad grows each axis, keeps infinite bounds and refuses non-finite padding") {
  const AABB a = box({0.0, -inf}, {1.0, 2.0});
  CHECK(a.pad(0.5) == box({-0.5, -inf}, {1.5, 2.5}));
  const std::vector<double> r = {1.0, 0.25};
  CHECK(a.pad(r) == box({-1.0, -inf}, {2.0, 2.25}));
  const std::vector<double> bad = {1.0, inf};
  const std::vector<double> wrong = {1.0};
  CHECK_THROWS_WITH_AS(static_cast<void>(a.pad(bad)),
                       "pad(r) entries must be finite; got [1.0, inf].", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(a.pad(wrong)),
                       "pad(r) requires r scalar or shape (2,); got (1,).", pantr::value_error);
}

TEST_CASE("AABB: transform is the exact image box, zeros contribute nothing against inf") {
  const AABB a = box({0.0, -inf}, {1.0, 2.0});
  // A = [[2, 0], [-1, 1]], b = [1, 0]: x' = 2x + 1 in [1, 3]; y' = -x + y in [-inf, 2].
  const std::vector<double> m = {2.0, 0.0, -1.0, 1.0};
  const std::size_t shape[2] = {2, 2};
  const std::vector<double> b = {1.0, 0.0};
  CHECK(a.transform(2, m, shape, b) == box({1.0, -inf}, {3.0, 2.0}));
  // An empty box maps to the canonical empty box before any check of the map.
  CHECK(AABB::empty(2).transform(5, m, shape, b) == AABB::empty(2));
  CHECK_THROWS_WITH_AS(static_cast<void>(a.transform(3, m, shape, b)),
                       "transform(): affine dim (3) must match AABB ndim (2).", pantr::value_error);
  const std::size_t bad_shape[2] = {1, 4};
  CHECK_THROWS_WITH_AS(static_cast<void>(a.transform(2, m, bad_shape, b)),
                       "transform(): affine.matrix must be (2, 2); got (1, 4).",
                       pantr::value_error);
  // Unbounded in every direction stays unbounded under an invertible map.
  const AABB wide = box({-inf, -inf}, {inf, inf});
  const std::vector<double> diff = {1.0, -1.0, 0.0, 1.0};
  CHECK(wide.transform(2, diff, shape, b) == AABB::unbounded(2));
  // x' = x + y + inf with y in [-inf, inf]: lo = (0 - inf) + inf is NaN, which is refused.
  const std::vector<double> sing = {1.0, 1.0, 0.0, 0.0};
  const AABB half = box({0.0, -inf}, {1.0, inf});
  const std::vector<double> nb = {inf, 0.0};
  CHECK_THROWS_AS(static_cast<void>(half.transform(2, sing, shape, nb)), pantr::value_error);
}

TEST_CASE("AABB: a row summed from its first term keeps a -0.0 bound, as numpy") {
  // A = [[-1]] on [0, 0] with b = [-0.0]: the one term is -1 * 0.0 = -0.0, and -0.0 + -0.0 is
  // -0.0. A sum seeded with +0.0 would give +0.0 instead (numpy reduces from the first element).
  const AABB z = box({0.0}, {0.0});
  const std::vector<double> m = {-1.0};
  const std::size_t shape[2] = {1, 1};
  const std::vector<double> b = {-0.0};
  const AABB img = z.transform(1, m, shape, b);
  CHECK(std::signbit(img.lo()[0]));
  CHECK(std::signbit(img.hi()[0]));
}

TEST_CASE("AABB: from_bounds and as_bounds are inverse, equality is IEEE") {
  const std::vector<double> rows = {0.0, 1.0, -2.0, 3.0};
  const std::size_t shape[2] = {2, 2};
  const AABB a = AABB::from_bounds(rows, shape);
  CHECK(a == box({0.0, -2.0}, {1.0, 3.0}));
  const auto back = a.as_bounds();
  for (std::size_t i = 0; i < 4; ++i) {
    CHECK(back.data()[i] == rows[i]);
  }
  const std::size_t bad[2] = {2, 3};
  CHECK_THROWS_WITH_AS(static_cast<void>(AABB::from_bounds(rows, bad)),
                       "from_bounds: bounds must have shape (ndim, 2); got (2, 3).",
                       pantr::value_error);
  CHECK(box({-0.0}, {1.0}) == box({0.0}, {1.0}));
  CHECK_FALSE(box({0.0}, {1.0}) == box({0.0, 0.0}, {1.0, 1.0}));
}

TEST_CASE("AABB: spans that do not hold what their shapes say are refused") {
  const std::vector<double> two = {0.0, 1.0};
  const std::size_t four_rows[2] = {4, 2};
  CHECK_THROWS_WITH_AS(static_cast<void>(AABB::from_bounds(two, four_rows)),
                       "from_bounds: bounds hold 2 values, but the shape (4, 2) needs 8.",
                       pantr::value_error);
  const AABB a = box({0.0, 0.0}, {1.0, 1.0});
  const std::vector<double> m3 = {1.0, 0.0, 0.0};
  const std::size_t shape[2] = {2, 2};
  const std::vector<double> b = {0.0, 0.0};
  CHECK_THROWS_WITH_AS(static_cast<void>(a.transform(2, m3, shape, b)),
                       "transform(): affine.matrix holds 3 values, but its shape needs 4.",
                       pantr::value_error);
}

TEST_CASE("AABB: an infinite matrix entry against a zero bound is NaN, refused as NumPy's was") {
  // [[inf]] on [-1, 0]: the terms are (-inf, inf * 0 = NaN); np.minimum propagates the NaN.
  const AABB a = box({-1.0}, {0.0});
  const std::vector<double> m = {std::numeric_limits<double>::infinity()};
  const std::size_t shape[2] = {1, 1};
  const std::vector<double> b = {0.0};
  CHECK_THROWS_AS(static_cast<void>(a.transform(1, m, shape, b)), pantr::value_error);
}
