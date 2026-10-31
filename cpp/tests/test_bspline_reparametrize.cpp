// Reversing a direction, permuting the directions and applying an affine map to a B-spline.
//
// Oracle. The exact-rational Cox-de Boor evaluation of the original (bspline_exact_eval.hpp), at
// the point the operation maps each sample point of the result to: the reflected parameter for a
// reversal, the permuted parameters for a permutation, and, for the affine map, the map applied
// to the exact homogeneous value (`A x_h + w b`, with the weight `w` of a rational spline, 1
// otherwise). The sample points span each whole direction. The data is dyadic and small, so every
// operation (a copy, a product by a small dyadic matrix, a sum of a few terms) is exact in the
// storage type and the comparisons are for equality, not within a tolerance.
#include <doctest/doctest.h>

#include "bspline_test_support.hpp"

#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/reparametrize.hpp>
#include <pantr/core/errors.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
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

/// The largest exact difference of any coordinate between `got` at the sample points of its own
/// domain and `want` at the points `where` maps them to (and mapped through `coords`).
template <class T>
double mapped_difference(
    const bs::Bspline<T>& got, const bs::Bspline<T>& want,
    const std::function<std::vector<Rational>(const std::vector<Rational>&)>& where,
    const std::function<std::vector<Rational>(const std::vector<Rational>&,
                                              const std::vector<Rational>&)>& expected) {
  const std::size_t dim = got.dim();
  const bs::BsplineSpace<T> whole = got.space();
  std::vector<std::vector<Rational>> axes(dim);
  for (std::size_t d = 0; d < dim; ++d) {
    const auto dom = whole.spaces()[d].domain();
    axes[d] = bspline_test::sample_points(oracle::exact(static_cast<double>(dom.first)),
                                          oracle::exact(static_cast<double>(dom.second)));
  }
  double worst = 0.0;
  std::vector<std::size_t> idx(dim, 0);
  for (;;) {
    std::vector<Rational> x;
    for (std::size_t d = 0; d < dim; ++d) {
      x.push_back(axes[d][idx[d]]);
    }
    const auto g = oracle::exact_value<T>(got, x);
    const auto w = oracle::exact_value<T>(want, where(x));
    const auto e = expected(w, x);
    REQUIRE(g.size() == e.size());
    for (std::size_t c = 0; c < g.size(); ++c) {
      Rational diff = g[c] - e[c];
      if (diff < 0) {
        diff = -diff;
      }
      worst = (std::max)(worst, static_cast<double>(diff));
    }
    std::size_t d = dim;
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

const auto identity_values = [](const std::vector<Rational>& v, const std::vector<Rational>&) {
  return v;
};

/// All the permutations of `0..dim - 1`, in lexicographic order.
std::vector<std::vector<long long>> all_permutations(std::size_t dim) {
  std::vector<long long> perm(dim);
  std::iota(perm.begin(), perm.end(), 0LL);
  std::vector<std::vector<long long>> out;
  do {
    out.push_back(perm);
  } while (std::next_permutation(perm.begin(), perm.end()));
  return out;
}

struct Config {
  const char* name;
  std::vector<Dir> dirs;
  bool rational;
};

std::vector<Config> configs() {
  return {
      {"curve", {{clamped(2, {0.5}), 2}}, false},
      {"rational curve", {{clamped(3, {0.25, 0.75}), 3}}, true},
      {"surface", {{clamped(2, {0.5}), 2}, {clamped(3, {0.25, 0.75}), 3}}, false},
      {"rational volume",
       {{clamped(2, {0.5}), 2}, {clamped(1, {0.5}), 1}, {clamped(2, {}), 2}},
       true},
      {"unclamped",
       {{{-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}, 2}, {clamped(2, {}), 2}},
       false},
      {"periodic beside clamped", {{periodic_knots(4, 2), 2, true}, {clamped(2, {0.5}), 2}}, false},
      {"periodic, rational", {{periodic_knots(4, 3), 3, true}}, true},
      {"two periodic directions",
       {{periodic_knots(4, 2), 2, true}, {periodic_knots(4, 3), 3, true}},
       false},
      // Knot vectors that are not their own reflections (every one above is), so that a reversal
      // that left the knots alone would change the spline and be seen.
      {"asymmetric curve", {{clamped(3, {0.125, 0.5, 0.5}), 3}}, false},
      {"asymmetric rational surface",
       {{clamped(2, {0.125}), 2}, {clamped(2, {0.5, 0.75}), 2}},
       true},
      {"asymmetric periodic direction, beside a clamped one",
       {{{-0.5, -0.25, 0.0, 0.125, 0.5, 0.75, 1.0, 1.125, 1.5}, 2, true}, {clamped(2, {0.25}), 2}},
       false},
  };
}

}  // namespace

TEST_CASE_TEMPLATE("reversed: evaluates the original at the reflected parameter", T, double,
                   float) {
  for (const Config& c : configs()) {
    CAPTURE(c.name);
    const bs::Bspline<T> b = make_spline<T>(c.dirs, 2, c.rational, 4U);
    for (std::size_t d = 0; d < c.dirs.size(); ++d) {
      CAPTURE(d);
      const bs::Bspline<T> r = bs::reversed<T>(b, static_cast<long long>(d));
      CHECK(r.rational() == b.rational());
      CHECK(r.shape()[d] == b.shape()[d]);
      const auto dom = b.space().spaces()[d].domain();
      const Rational sum = oracle::exact(static_cast<double>(dom.first)) +
                           oracle::exact(static_cast<double>(dom.second));
      const auto reflect = [&](const std::vector<Rational>& x) {
        std::vector<Rational> y = x;
        y[d] = sum - x[d];
        return y;
      };
      CHECK(mapped_difference<T>(r, b, reflect, identity_values) == 0.0);
      // The reflected direction's domain is unchanged, the others keep their spaces.
      const auto rdom = r.space().spaces()[d].domain();
      CHECK(rdom.first == dom.first);
      CHECK(rdom.second == dom.second);
      for (std::size_t k = 0; k < c.dirs.size(); ++k) {
        if (k != d) {
          CHECK(same_space(r.space().spaces()[k], b.space().spaces()[k]));
        }
      }
      // The knots are `a + b - knots` read backwards, one subtraction in the storage type.
      const std::vector<T> old = bspline_test::knots_of<T>(b, d);
      const std::vector<T> got = bspline_test::knots_of<T>(r, d);
      const T total = dom.first + dom.second;
      REQUIRE(got.size() == old.size());
      for (std::size_t i = 0; i < old.size(); ++i) {
        CHECK(got[i] == total - old[old.size() - 1 - i]);
      }
      // Reversing twice restores the spline, bit for bit (the data is dyadic).
      const bs::Bspline<T> rr = bs::reversed<T>(r, static_cast<long long>(d));
      CHECK(bspline_test::knots_of<T>(rr, d) == old);
      const auto a = rr.control_points();
      const auto o = b.control_points();
      REQUIRE(a.size() == o.size());
      for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i] == o[i]);
      }
    }
  }
}

TEST_CASE("reversed: refusals and their messages") {
  const bs::Bspline<double> b =
      make_spline<double>({{clamped(2, {0.5}), 2}, {clamped(2, {}), 2}}, 2, false, 3U);
  CHECK(message_of([&] { (void)bs::reversed<double>(b, 2); }) ==
        "direction must be in [0, 2), got 2.");
  CHECK(message_of([&] { (void)bs::reversed<double>(b, -1); }) ==
        "direction must be in [0, 2), got -1.");
}

TEST_CASE_TEMPLATE("permuted: evaluates the original at the permuted parameters", T, double,
                   float) {
  for (const Config& c : configs()) {
    CAPTURE(c.name);
    const bs::Bspline<T> b = make_spline<T>(c.dirs, 2, c.rational, 8U);
    const std::size_t dim = c.dirs.size();
    // Every permutation of the directions: 1, 2 and 6 of them for a curve, a surface, a volume.
    const std::vector<std::vector<long long>> perms = all_permutations(dim);
    std::size_t factorial = 1;
    for (std::size_t k = 2; k <= dim; ++k) {
      factorial *= k;
    }
    CHECK(perms.size() == factorial);
    for (const std::vector<long long>& perm : perms) {
      const bs::Bspline<T> r = bs::permuted<T>(b, perm);
      CHECK(r.rational() == b.rational());
      // r(t)_k = the old direction perm[k]: the point of b is s with s[perm[k]] = t[k].
      const auto where = [&](const std::vector<Rational>& t) {
        std::vector<Rational> s(t.size());
        for (std::size_t k = 0; k < t.size(); ++k) {
          s[static_cast<std::size_t>(perm[k])] = t[k];
        }
        return s;
      };
      CHECK(mapped_difference<T>(r, b, where, identity_values) == 0.0);
      for (std::size_t k = 0; k < dim; ++k) {
        CHECK(same_space(r.space().spaces()[k],
                         b.space().spaces()[static_cast<std::size_t>(perm[k])]));
        CHECK(r.shape()[k] == b.shape()[static_cast<std::size_t>(perm[k])]);
      }
      // The inverse permutation restores the control points, bit for bit.
      std::vector<long long> inverse(dim);
      for (std::size_t k = 0; k < dim; ++k) {
        inverse[static_cast<std::size_t>(perm[k])] = static_cast<long long>(k);
      }
      const bs::Bspline<T> back = bs::permuted<T>(r, inverse);
      const auto a = back.control_points();
      const auto o = b.control_points();
      REQUIRE(a.size() == o.size());
      for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i] == o[i]);
      }
    }
  }
}

TEST_CASE("permuted: refusals and their messages") {
  const bs::Bspline<double> b =
      make_spline<double>({{clamped(2, {0.5}), 2}, {clamped(2, {}), 2}}, 2, false, 3U);
  const std::string want = "permutation must be a permutation of range(2)";
  for (const std::vector<long long>& bad :
       std::vector<std::vector<long long>>{{0, 0}, {0}, {0, 1, 2}, {-1, 0}, {0, 2}, {}}) {
    CAPTURE(bad.size());
    CHECK(message_of([&] { (void)bs::permuted<double>(b, bad); }) == want);
  }
}

TEST_CASE_TEMPLATE("transformed: applies x -> A x + b to the projected geometry", T, double,
                   float) {
  for (const Config& c : configs()) {
    CAPTURE(c.name);
    const std::size_t rank = 2;
    const bs::Bspline<T> b = make_spline<T>(c.dirs, rank, c.rational, 6U);
    // A is a dyadic matrix that mixes the coordinates; the offset is dyadic.
    const std::vector<double> a = {2.0, -0.5, 0.25, 1.0};
    const std::vector<double> t = {0.5, -1.0};
    const std::size_t shape[2] = {2, 2};
    const bs::Bspline<T> r = bs::transformed<T>(b, a, shape, t);
    CHECK(r.rational() == b.rational());
    CHECK(r.shape()[r.dim()] == b.cp_size());
    for (std::size_t d = 0; d < c.dirs.size(); ++d) {
      CHECK(same_space(r.space().spaces()[d], b.space().spaces()[d]));
    }
    const auto where = [](const std::vector<Rational>& x) { return x; };
    // y_h[i] = sum_k A[i][k] x_h[k] + w t[i], w the weight of a rational spline, else 1; the
    // weight is copied.
    const auto image = [&](const std::vector<Rational>& v, const std::vector<Rational>&) {
      std::vector<Rational> out = v;
      const Rational w = c.rational ? v[rank] : Rational(1);
      for (std::size_t i = 0; i < rank; ++i) {
        Rational sum = w * oracle::exact(t[i]);
        for (std::size_t k = 0; k < rank; ++k) {
          sum += oracle::exact(a[i * rank + k]) * v[k];
        }
        out[i] = sum;
      }
      return out;
    };
    CHECK(mapped_difference<T>(r, b, where, image) == 0.0);
    if (c.rational) {
      // The weights are copied unchanged.
      const auto cp = b.control_points();
      const auto rp = r.control_points();
      for (std::size_t i = rank; i < cp.size(); i += rank + 1) {
        CHECK(rp[i] == cp[i]);
      }
    }
  }
}

TEST_CASE("transformed: float32 data off the dyadic grid is rounded in float at every step") {
  // The dyadic test above cannot tell rounding in `T` at every operation from one rounding of an
  // exact (or double) accumulation: every product and sum there is exact. Here `A` and `b` are
  // not dyadic (rounded to float, each product of a 24-bit entry and a 7-bit control point needs
  // 31 bits), and the expected value is computed in float one operation at a time:
  // `acc = x0 a_i0; acc = acc + x1 a_i1; y_i = acc + b_i` (`+ w b_i` for a rational spline, `w` the
  // weight). Each statement below is one rounding (no contraction across statements). The library
  // may contract the sum with its product (the build allows one multiply-add per expression), so
  // the value of either grouping is accepted; the single rounding of a double accumulation is not,
  // and the data are checked to tell it apart.
  const std::vector<double> a = {0.1, -0.7, 0.3, 1.1};
  const std::vector<double> t = {0.37, -0.9};
  const float af[4] = {static_cast<float>(a[0]), static_cast<float>(a[1]), static_cast<float>(a[2]),
                       static_cast<float>(a[3])};
  const float tf[2] = {static_cast<float>(t[0]), static_cast<float>(t[1])};
  const std::size_t shape[2] = {2, 2};
  bool tells_double_apart = false;
  for (const Config& c : configs()) {
    CAPTURE(c.name);
    const bs::Bspline<float> b = make_spline<float>(c.dirs, 2, c.rational, 6U);
    const bs::Bspline<float> r = bs::transformed<float>(b, a, shape, t);
    const auto cp = b.control_points();
    const auto rp = r.control_points();
    const std::size_t cols = b.cp_size();
    REQUIRE(rp.size() == cp.size());
    for (std::size_t row = 0; row < cp.size() / cols; ++row) {
      const float x0 = cp[row * cols];
      const float x1 = cp[row * cols + 1];
      const float w = c.rational ? cp[row * cols + 2] : 1.0F;
      for (std::size_t i = 0; i < 2; ++i) {
        const float p0 = x0 * af[i * 2];
        const float p1 = x1 * af[i * 2 + 1];
        const float acc = p0 + p1;                                // two roundings, then a third
        const float acc_fused = std::fma(x1, af[i * 2 + 1], p0);  // the second term contracted
        const float offset = c.rational ? w * tf[i] : tf[i];
        std::vector<float> accepted = {acc + offset, acc_fused + offset};
        if (c.rational) {
          accepted.push_back(std::fma(w, tf[i], acc));
          accepted.push_back(std::fma(w, tf[i], acc_fused));
        }
        const float got = rp[row * cols + i];
        CHECK(std::find(accepted.begin(), accepted.end(), got) != accepted.end());
        // One rounding of the exact-enough double value: differs from every float grouping at
        // some entry, or this test could not tell the two apart.
        const double wide = static_cast<double>(x0) * static_cast<double>(af[i * 2]) +
                            static_cast<double>(x1) * static_cast<double>(af[i * 2 + 1]) +
                            (c.rational ? static_cast<double>(w) * static_cast<double>(tf[i])
                                        : static_cast<double>(tf[i]));
        const auto once = static_cast<float>(wide);
        if (std::find(accepted.begin(), accepted.end(), once) == accepted.end()) {
          tells_double_apart = true;
        }
      }
    }
  }
  CHECK(tells_double_apart);
}

TEST_CASE_TEMPLATE("transformed: the identity map keeps the control points", T, double, float) {
  const bs::Bspline<T> b =
      make_spline<T>({{clamped(2, {0.5}), 2}, {clamped(2, {}), 2}}, 3, true, 2U);
  const std::vector<double> eye = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const std::vector<double> zero = {0.0, 0.0, 0.0};
  const std::size_t shape[2] = {3, 3};
  const bs::Bspline<T> r = bs::transformed<T>(b, eye, shape, zero);
  const auto a = r.control_points();
  const auto o = b.control_points();
  REQUIRE(a.size() == o.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    CHECK(a[i] == o[i]);
  }
}

TEST_CASE("transformed: refusals and their messages") {
  const bs::Bspline<double> b = make_spline<double>({{clamped(2, {0.5}), 2}}, 2, true, 3U);
  const std::vector<double> m2 = {1.0, 0.0, 0.0, 1.0};
  const std::vector<double> m3(9, 0.0);
  const std::vector<double> off2 = {0.0, 0.0};
  const std::size_t s22[2] = {2, 2};
  const std::size_t s33[2] = {3, 3};
  const std::size_t s4[1] = {4};
  CHECK(message_of([&] { (void)bs::transformed<double>(b, m3, s33, std::vector<double>(3)); }) ==
        "Transform dimension (3) does not match the geometric rank (2) of the control points.");
  CHECK(message_of([&] { (void)bs::transformed<double>(b, m2, s4, off2); }) ==
        "matrix must be a 2D array, got shape (4,).");
  CHECK(message_of([&] { (void)bs::transformed<double>(b, std::vector<double>(3), s22, off2); }) ==
        "matrix holds 3 values, but the shape (2, 2) needs 4.");
  CHECK(message_of([&] { (void)bs::transformed<double>(b, m2, s22, std::vector<double>(3)); }) ==
        "offset must have length 2, got 3.");
}
