// AffineTransform. Oracles: maps whose action is known in closed form (quarter turns, axis
// mirrors, shears with integer factors, all exact in binary), the group laws (a map composed with
// its inverse is the identity, a center is a fixed point), and the orthogonality of rotations.
#include <doctest/doctest.h>

#include <pantr/transform/affine_transform.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <vector>

using pantr::transform::AffineTransform;
constexpr double eps = std::numeric_limits<double>::epsilon();

namespace {

std::vector<double> apply(const AffineTransform& t, const std::vector<double>& x) {
  const auto y = t.apply(x);
  return {y.data(), y.data() + y.size()};
}

}  // namespace

TEST_CASE("AffineTransform: construction validates the shapes") {
  const std::vector<double> m = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  const std::size_t rect[2] = {2, 3};
  CHECK_THROWS_WITH_AS(AffineTransform(m, rect),
                       "matrix must be a square 2-D array, got shape (2, 3).", pantr::value_error);
  const std::vector<double> sq = {1.0, 0.0, 0.0, 1.0};
  const std::size_t shape[2] = {2, 2};
  const std::vector<double> t3 = {1.0, 2.0, 3.0};
  const std::size_t t3s[1] = {3};
  CHECK_THROWS_WITH_AS(AffineTransform(sq, shape, std::span<const double>(t3), t3s),
                       "translation must have shape (2,), got (3,).", pantr::value_error);
  const AffineTransform id(sq, shape);
  CHECK(id.dim() == 2);
  CHECK(id.offset()[0] == 0.0);
  CHECK(AffineTransform::identity(0).dim() == 0);
}

TEST_CASE("AffineTransform: a quarter turn about a center fixes the center") {
  const std::vector<double> c = {1.0, 2.0};
  const AffineTransform r = AffineTransform::rotation_2d(std::numbers::pi / 2, c);
  const auto fixed = apply(r, c);
  CHECK(std::fabs(fixed[0] - 1.0) <= 4 * eps);
  CHECK(std::fabs(fixed[1] - 2.0) <= 4 * eps);
  // (2, 2) is c + (1, 0); a quarter turn maps it to c + (0, 1) = (1, 3). cos(pi/2) is 6e-17,
  // not 0, so the error is one rounding of that against unit-sized terms: a few eps.
  const auto y = apply(r, {2.0, 2.0});
  CHECK(std::fabs(y[0] - 1.0) <= 8 * eps);
  CHECK(std::fabs(y[1] - 3.0) <= 8 * eps);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(AffineTransform::rotation_2d(std::numeric_limits<double>::infinity())),
      "angle must be finite, got inf.", pantr::value_error);
  const std::vector<double> c3 = {1.0, 2.0, 3.0};
  CHECK_THROWS_WITH_AS(static_cast<void>(AffineTransform::rotation_2d(0.0, c3)),
                       "center must have shape (2,), got (3,).", pantr::value_error);
}

TEST_CASE("AffineTransform: Rodrigues rotations are orthogonal and fix their axis") {
  const std::vector<double> axis = {1.0, 2.0, 2.0};  // norm 3: u = (1, 2, 2) / 3
  const AffineTransform r = AffineTransform::rotation_3d(0.7, axis);
  const auto m = r.matrix();
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      double dot = 0.0;
      for (std::size_t k = 0; k < 3; ++k) {
        dot += m[i * 3 + k] * m[j * 3 + k];
      }
      // R R^T = I: each entry is three products of entries bounded by 1, each from about ten
      // roundings: 32 eps is a safe bound.
      CHECK(std::fabs(dot - (i == j ? 1.0 : 0.0)) <= 32 * eps);
    }
  }
  const auto on_axis = apply(r, axis);
  for (std::size_t i = 0; i < 3; ++i) {
    CHECK(std::fabs(on_axis[i] - axis[i]) <= 32 * eps);
  }
  const std::vector<double> zero = {0.0, 0.0, 0.0};
  CHECK_THROWS_AS(static_cast<void>(AffineTransform::rotation_3d(0.1, zero)), pantr::value_error);
  const std::vector<double> two = {1.0, 0.0};
  CHECK_THROWS_WITH_AS(static_cast<void>(AffineTransform::rotation_3d(0.1, two)),
                       "Rotation axis must have shape (3,), got (2,).", pantr::value_error);
}

TEST_CASE("AffineTransform: mirror, shear and scaling act as their closed forms") {
  const std::vector<double> nx = {2.0, 0.0};  // reflect x
  const auto mx = apply(AffineTransform::mirror(nx), {3.0, 4.0});
  CHECK(mx[0] == -3.0);
  CHECK(mx[1] == 4.0);
  const auto sh = apply(AffineTransform::shear(2, 0, 1, 2.0), {1.0, 3.0});  // x += 2 y
  CHECK(sh[0] == 7.0);
  CHECK(sh[1] == 3.0);
  CHECK_THROWS_WITH_AS(static_cast<void>(AffineTransform::shear(2, 1, 1, 1.0)),
                       "component and direction must differ.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(AffineTransform::shear(2, 2, 0, 1.0)),
                       "component must be in [0, 2), got 2.", pantr::value_error);
  const std::vector<double> f = {2.0, 0.5};
  const std::vector<double> c = {1.0, 1.0};
  const auto sc = apply(AffineTransform::scaling(f, c), {3.0, 3.0});  // c + diag(f) (2, 2)
  CHECK(sc[0] == 5.0);
  CHECK(sc[1] == 2.0);
  const std::vector<double> zf = {1.0, 0.0};
  CHECK_THROWS_AS(static_cast<void>(AffineTransform::scaling(zf)), pantr::value_error);
  // Finiteness is checked over every factor before any zero, as the previous release did.
  const std::vector<double> zero_then_inf = {0.0, std::numeric_limits<double>::infinity()};
  CHECK_THROWS_WITH_AS(static_cast<void>(AffineTransform::scaling(zero_then_inf)),
                       "scaling factors must be finite.", pantr::value_error);
}

TEST_CASE("AffineTransform: composition order and the inverse") {
  const std::vector<double> t = {1.0, -2.0};
  const AffineTransform a = AffineTransform::translation(t);
  const AffineTransform b = AffineTransform::shear(2, 0, 1, 3.0);
  // (a o b)(x) = b(x) + t: (1, 1) -> (4, 1) -> (5, -1).
  const auto y = apply(a.compose(b), {1.0, 1.0});
  CHECK(y[0] == 5.0);
  CHECK(y[1] == -1.0);
  const AffineTransform ab = a.compose(b);
  const auto back = apply(ab.inverse().compose(ab), {0.25, 7.0});
  // An LU inverse of a matrix with entries <= 3 and condition number about 10: 64 eps of the
  // point's magnitude 7.
  CHECK(std::fabs(back[0] - 0.25) <= 64 * 7 * eps);
  CHECK(std::fabs(back[1] - 7.0) <= 64 * 7 * eps);
  const std::vector<double> sing = {1.0, 2.0, 2.0, 4.0};
  const std::size_t shape[2] = {2, 2};
  CHECK_THROWS_WITH_AS(static_cast<void>(AffineTransform(sing, shape).inverse()),
                       "Cannot invert a singular affine transformation.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(a.compose(AffineTransform::identity(3))),
                       "Cannot compose transforms of different dimensions (2 and 3).",
                       pantr::value_error);
}

TEST_CASE("AffineTransform: apply checks the point count") {
  const AffineTransform id = AffineTransform::identity(2);
  const std::vector<double> three = {1.0, 2.0, 3.0};
  CHECK_THROWS_AS(static_cast<void>(id.apply(three)), pantr::value_error);
  const std::vector<double> none;
  CHECK(id.apply(none).size() == 0);
}
