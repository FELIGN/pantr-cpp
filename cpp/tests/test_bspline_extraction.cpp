// The mode-wise contraction primitive, the Bézier, Lagrange and cardinal extraction operators of a
// 1D space, and the Bézier decomposition of a B-spline (`to_bezier`, `to_beziers`).
//
// Oracles.
//  * The mode product is compared with a naive loop over multi-indices in exact integer
//    arithmetic (the data are small integers, so every sum is exact in the storage type too).
//  * The Bézier extraction operators are compared with their defining relation `N = C B`: the
//    exact value of each B-spline basis function that is nonzero on an interval (the recursive
//    definition of bspline_oracle.hpp) against the exact Bernstein form with the stored operator.
//    Tolerance, derived: an operator entry is reached from the identity by at most `p (p + 1) / 2`
//    knot-insertion steps, each `alpha a + (1 - alpha) b` with `alpha` in [0, 1], rounded to at
//    most `4 u` of the larger operand (the weight and its complement rounded, two products, one
//    sum), the operands being entries of [0, 1]; the Bernstein weights are non-negative and sum to
//    one. So the relation holds within `4 p (p + 1) u`.
//  * The Lagrange and cardinal operators are the Bézier operator times a change-of-basis matrix
//    taken from the library's process table, so they are compared with an independent product of
//    the two in double, within the bound of a sum of `p + 1` products, `(p + 1) u sum |terms|`
//    (Higham, eq. (4.4)), plus the rounding of the stored Bézier operator the product starts from.
//  * A decomposition is compared with the spline it decomposes, both evaluated exactly: each
//    Bézier patch, in the exact Bernstein form of its stored control points, against the exact
//    Cox-de Boor sum of the spline at the same point. The extraction along one direction forms
//    convex combinations (the columns of an operator sum to one) of `p + 1` control points with
//    operator entries that are off by `4 p (p + 1) u`, and rounds the sum of `p + 1` products
//    (`(p + 1) u` of the terms), so each direction adds `(p + 1) (4 p (p + 1) + 2) u max|P|`.
#include <doctest/doctest.h>

#include "bspline_test_support.hpp"

#include <pantr/basis/lagrange_variant.hpp>
#include <pantr/bezier/bezier.hpp>
#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/detail/mode_product.hpp>
#include <pantr/bspline/extraction.hpp>
#include <pantr/bspline/to_beziers.hpp>
#include <pantr/change_basis.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/parallel/pool.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
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
using oracle::Rational;
using pantr::NdArray;

namespace {

template <class T>
double unit_roundoff() {
  return 0.5 * static_cast<double>(std::numeric_limits<T>::epsilon());
}

// ---- the mode product ------------------------------------------------------------------------

/// Row-major linear index of a multi-index.
std::size_t linear(const std::vector<std::size_t>& extents, const std::vector<std::size_t>& idx) {
  std::size_t n = 0;
  for (std::size_t d = 0; d < extents.size(); ++d) {
    n = n * extents[d] + idx[d];
  }
  return n;
}

/// Small deterministic integers in [-lo, lo].
std::vector<int> small_ints(std::size_t count, unsigned seed, int lo) {
  std::vector<int> v(count);
  std::uint32_t s = 2463534242U + seed;
  for (int& x : v) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    x = static_cast<int>(s % static_cast<std::uint32_t>(2 * lo + 1)) - lo;
  }
  return v;
}

/// Apply the stored `op` (or its transpose) along `mode` of `x` by the definition, a loop over the
/// multi-indices of the result in exact integers. `y` is filled with `sentinel` outside the window.
std::vector<std::int64_t> naive_mode_product(const std::vector<int>& op, std::size_t rows,
                                             std::size_t cols, bool transpose,
                                             const std::vector<int>& x,
                                             const std::vector<std::size_t>& x_extents,
                                             std::size_t mode, std::size_t x_begin,
                                             const std::vector<std::size_t>& y_extents,
                                             std::size_t y_begin, std::int64_t sentinel) {
  const std::size_t n_in = transpose ? rows : cols;
  const std::size_t n_out = transpose ? cols : rows;
  std::size_t total = 1;
  for (const std::size_t e : y_extents) {
    total *= e;
  }
  std::vector<std::int64_t> y(total, sentinel);
  std::vector<std::size_t> idx(y_extents.size(), 0);
  for (std::size_t n = 0; n < total; ++n) {
    if (idx[mode] >= y_begin && idx[mode] < y_begin + n_out) {
      const std::size_t i = idx[mode] - y_begin;
      std::int64_t sum = 0;
      std::vector<std::size_t> src = idx;
      for (std::size_t j = 0; j < n_in; ++j) {
        src[mode] = x_begin + j;
        const int a = transpose ? op[j * cols + i] : op[i * cols + j];
        sum += static_cast<std::int64_t>(a) * x[linear(x_extents, src)];
      }
      y[n] = sum;
    }
    for (std::size_t d = idx.size(); d-- > 0;) {
      if (++idx[d] < y_extents[d]) {
        break;
      }
      idx[d] = 0;
    }
  }
  return y;
}

}  // namespace

TEST_CASE_TEMPLATE("mode product: equals the definition, in every mode, window and transpose", T,
                   double, float) {
  const std::vector<std::vector<std::size_t>> shapes = {{5},       {3, 4},       {4, 1, 3},
                                                        {2, 3, 4}, {2, 3, 2, 3}, {1, 2, 3, 2, 2}};
  unsigned seed = 1;
  for (const std::vector<std::size_t>& x_extents : shapes) {
    for (std::size_t mode = 0; mode < x_extents.size(); ++mode) {
      for (const bool transpose : {false, true}) {
        for (const std::size_t n_in_want : {std::size_t{1}, std::size_t{3}}) {
          CAPTURE(x_extents.size());
          CAPTURE(mode);
          CAPTURE(transpose);
          const std::size_t n_in = (std::min)(n_in_want, x_extents[mode]);
          const std::size_t n_out = 2;
          // The stored operator is (n_out x n_in), or (n_in x n_out) when applied transposed.
          const std::size_t rows = transpose ? n_in : n_out;
          const std::size_t cols = transpose ? n_out : n_in;
          const std::size_t x_begin = x_extents[mode] - n_in;  // the last window: nonzero if it can
          const std::size_t y_begin = 1;
          std::vector<std::size_t> y_extents = x_extents;
          y_extents[mode] = n_out + y_begin + 1;  // room on both sides of the window

          std::size_t x_total = 1;
          for (const std::size_t e : x_extents) {
            x_total *= e;
          }
          const std::vector<int> op_i = small_ints(rows * cols, seed++, 4);
          const std::vector<int> x_i = small_ints(x_total, seed++, 8);
          const std::vector<T> op(op_i.begin(), op_i.end());
          const std::vector<T> x(x_i.begin(), x_i.end());
          const std::int64_t sentinel = -77;
          std::size_t y_total = 1;
          for (const std::size_t e : y_extents) {
            y_total *= e;
          }
          std::vector<T> y(y_total, static_cast<T>(sentinel));

          pantr::bspline::detail::mode_product<T>(op, rows, cols, transpose, x, x_extents, mode,
                                                  x_begin, y, y_extents, y_begin);
          const std::vector<std::int64_t> want =
              naive_mode_product(op_i, rows, cols, transpose, x_i, x_extents, mode, x_begin,
                                 y_extents, y_begin, sentinel);
          REQUIRE(want.size() == y.size());
          bool all = true;
          for (std::size_t i = 0; i < y.size(); ++i) {
            all = all && static_cast<std::int64_t>(y[i]) == want[i] &&
                  y[i] == static_cast<T>(want[i]);
          }
          CHECK(all);
        }
      }
    }
  }
}

TEST_CASE_TEMPLATE("mode product: the whole-mode overload maps the extent of the operator", T,
                   double, float) {
  // (2 x 3) applied to a 3-long mode gives 2, and transposed to a 2-long mode gives 3.
  const std::vector<T> op = {1, 2, 3, 4, 5, 6};
  const std::vector<std::size_t> ex = {2, 3};
  const std::vector<T> x = {1, 1, 1, 2, 2, 2};
  std::vector<T> y(4);
  const std::vector<std::size_t> ey = {2, 2};
  pantr::bspline::detail::mode_product<T>(op, 2, 3, false, x, ex, 1, y, ey);
  // Row r of x is (r+1) (1,1,1): applied to the columns gives (r+1) (6, 15).
  CHECK(y == std::vector<T>{6, 15, 12, 30});
  std::vector<T> z(6);
  const std::vector<std::size_t> ez = {2, 3};
  const std::vector<T> w = {1, 10, 2, 20};
  const std::vector<std::size_t> ew = {2, 2};
  pantr::bspline::detail::mode_product<T>(op, 2, 3, true, w, ew, 1, z, ez);
  // Transposed: columns of op are (1,4), (2,5), (3,6); row (1,10) gives 1*(1,2,3) + 10*(4,5,6).
  CHECK(z == std::vector<T>{41, 52, 63, 82, 104, 126});
}

TEST_CASE("mode product: adjacent buffers do not overlap, overlapping ones are refused") {
  // x is the first half of one allocation and y the second: the end of x is the start of y.
  std::vector<double> both(24, 0.0);
  for (std::size_t i = 0; i < 12; ++i) {
    both[i] = static_cast<double>(i);
  }
  const std::vector<double> op = {1.0, 1.0, 1.0, 1.0};
  const std::vector<std::size_t> e34 = {3, 4};
  namespace d = pantr::bspline::detail;
  const std::span<const double> x(both.data(), 12);
  const std::span<double> y(both.data() + 12, 12);
  CHECK_NOTHROW(d::mode_product<double>(op, 2, 2, false, x, e34, 1, 1, y, e34, 0));
  // Row r of x is (4r, 4r+1, 4r+2, 4r+3); the window [1, 3) summed is 8r + 3, written at y[., 0].
  CHECK(both[12] == 3.0);
  CHECK(both[16] == 11.0);
  CHECK(both[20] == 19.0);
  // y one element into x overlaps it, and so does x one element into y.
  const std::span<double> shifted(both.data() + 11, 12);
  CHECK_THROWS_AS(d::mode_product<double>(op, 2, 2, false, x, e34, 1, 1, shifted, e34, 0),
                  pantr::value_error);
  const std::span<const double> x_late(both.data() + 1, 12);
  const std::span<double> y_early(both.data() + 12, 12);
  CHECK_THROWS_AS(d::mode_product<double>(op, 2, 2, false, x_late, e34, 1, 1, y_early, e34, 0),
                  pantr::value_error);
}

TEST_CASE("mode product: refusals and their messages") {
  const std::vector<double> op(4, 1.0);
  const std::vector<double> x(12, 1.0);
  std::vector<double> y(12, 0.0);
  const std::vector<std::size_t> e34 = {3, 4};
  namespace d = pantr::bspline::detail;
  const auto call = [&](std::span<const double> o, std::size_t r, std::size_t c, bool t,
                        std::span<const double> in, std::vector<std::size_t> ie, std::size_t mode,
                        std::size_t ib, std::span<double> out, std::vector<std::size_t> oe,
                        std::size_t ob) {
    return message_of([&] { d::mode_product<double>(o, r, c, t, in, ie, mode, ib, out, oe, ob); });
  };
  CHECK(call(op, 2, 2, false, x, e34, 2, 0, y, e34, 0) ==
        "mode_product: mode 2 is out of range for 2 extents.");
  CHECK(call(op, 2, 2, false, x, {}, 0, 0, y, {}, 0) ==
        "mode_product: x and y need the same, nonzero, number of extents, got 0 and 0.");
  CHECK(call(op, 2, 2, false, x, e34, 0, 0, y, {3}, 0) ==
        "mode_product: x and y need the same, nonzero, number of extents, got 2 and 1.");
  CHECK(call(op, 2, 2, false, x, e34, 0, 0, y, {3, 3}, 0) ==
        "mode_product: x and y differ in extent 1, 4 and 3.");
  CHECK(call(op, 2, 2, false, x, {3, 0}, 0, 0, y, {3, 0}, 0) ==
        "mode_product: the extents must be at least one.");
  CHECK(call(std::span<const double>(op).first(3), 2, 2, false, x, e34, 0, 0, y, e34, 0) ==
        "mode_product: the operator holds 3 values, but its shape needs 4.");
  CHECK(call(op, 2, 2, false, std::span<const double>(x).first(11), e34, 0, 0, y, e34, 0) ==
        "mode_product: a buffer does not hold the product of its extents.");
  // A window of 2 entries from 2 of 3 leaves the extent.
  CHECK(call(op, 2, 2, false, x, e34, 0, 2, y, e34, 0) ==
        "mode_product: the input window [2, +2) leaves the extent 3.");
  CHECK(call(op, 2, 2, false, x, e34, 0, 0, y, e34, 2) ==
        "mode_product: the output window [2, +2) leaves the extent 3.");
  // A begin past the extent does not wrap the sum.
  CHECK(call(op, 2, 2, false, x, e34, 0, (std::numeric_limits<std::size_t>::max)(), y, e34, 0) ==
        "mode_product: the input window [" +
            std::to_string((std::numeric_limits<std::size_t>::max)()) +
            ", +2) leaves the extent 3.");
  // The input and the output must not overlap.
  CHECK(call(op, 2, 2, false, std::span<const double>(y), e34, 0, 0, y, e34, 0) ==
        "mode_product: x and y overlap.");
  // The whole-mode overload wants the operator's extents.
  CHECK(message_of([&] {
          d::mode_product<double>(op, 2, 2, false, x, std::vector<std::size_t>{3, 4}, 0, y,
                                  std::vector<std::size_t>{3, 4});
        }) ==
        "mode_product: the extents along the mode are 3 and 3, but the operator maps 2 to 2.");
}

namespace {

// ---- the exact Bernstein form ----------------------------------------------------------------

Rational binomial_exact(int n, int k) {
  Rational r = 1;
  for (int i = 1; i <= k; ++i) {
    r = r * (n - k + i) / i;
  }
  return r;
}

Rational bernstein_exact(int n, int k, const Rational& x) {
  Rational r = binomial_exact(n, k);
  for (int i = 0; i < k; ++i) {
    r *= x;
  }
  for (int i = 0; i < n - k; ++i) {
    r *= (1 - x);
  }
  return r;
}

/// The exact value (all coordinates) of a Bézier at `x`, from its stored control points.
template <class T>
std::vector<Rational> bezier_exact(const pantr::bezier::Bezier<T>& b,
                                   const std::vector<Rational>& x) {
  const std::size_t dim = b.dim();
  const std::size_t cols = b.cp_size();
  std::vector<std::vector<Rational>> basis(dim);
  for (std::size_t d = 0; d < dim; ++d) {
    const int p = static_cast<int>(b.degree(d));
    for (int k = 0; k <= p; ++k) {
      basis[d].push_back(bernstein_exact(p, k, x[d]));
    }
  }
  std::vector<Rational> out(cols, Rational(0));
  std::vector<std::size_t> idx(dim, 0);
  const auto cp = b.control_points();
  for (;;) {
    Rational w = 1;
    std::size_t row = 0;
    for (std::size_t d = 0; d < dim; ++d) {
      w *= basis[d][idx[d]];
      row = row * b.shape()[d] + idx[d];
    }
    for (std::size_t c = 0; c < cols; ++c) {
      out[c] += w * oracle::exact(static_cast<double>(cp[row * cols + c]));
    }
    std::size_t d = dim;
    bool done = true;
    while (d-- > 0) {
      if (++idx[d] < b.shape()[d]) {
        done = false;
        break;
      }
      idx[d] = 0;
    }
    if (done) {
      return out;
    }
  }
}

// ---- the extraction operators ----------------------------------------------------------------

struct Space1 {
  const char* name;
  std::vector<double> knots;
  int degree;
};

std::vector<Space1> spaces_1d() {
  return {
      {"degree 0", {0.0, 0.25, 0.5, 1.0}, 0},
      {"degree 1", clamped(1, {0.25, 0.5}), 1},
      {"Bezier-like", clamped(3, {}), 3},
      {"open, simple knots", clamped(2, {0.25, 0.5, 0.75}), 2},
      {"open, a double knot in a cubic", clamped(3, {0.25, 0.5, 0.5, 0.75}), 3},
      {"a knot of multiplicity degree", clamped(2, {0.5, 0.5}), 2},
      {"a knot of multiplicity degree + 1", clamped(2, {0.5, 0.5, 0.5}), 2},
      {"unclamped", {-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}, 2},
      {"unclamped cubic, uneven",
       {-0.75, -0.5, -0.25, 0.0, 0.125, 0.5, 0.875, 1.0, 1.25, 1.5, 1.75},
       3},
      {"periodic-style ghost knots", periodic_knots(4, 3), 3},
      {"degree 4, off-mesh", clamped(4, {0.1, 0.35, 0.6}), 4},
  };
}

template <class T>
bs::BsplineSpace1D<T> make_space(const Space1& s) {
  const std::vector<T> k = bspline_test::cast_all<T>(s.knots);
  return bs::BsplineSpace1D<T>(std::span<const T>(k), s.degree);
}

}  // namespace

TEST_CASE_TEMPLATE("Bezier extraction operators: N = C B on every interval", T, double, float) {
  for (const Space1& s : spaces_1d()) {
    CAPTURE(s.name);
    const bs::BsplineSpace1D<T> space = make_space<T>(s);
    const NdArray<T> ops = bs::bezier_extraction_operators<T>(space);
    const std::size_t order = static_cast<std::size_t>(s.degree) + 1;
    const std::size_t n_int = static_cast<std::size_t>(space.num_intervals());
    REQUIRE(ops.shape().size() == 3);
    CHECK(ops.shape()[0] == n_int);
    CHECK(ops.shape()[1] == order);
    CHECK(ops.shape()[2] == order);
    const auto [uk, um] = space.unique_knots_and_multiplicity(true);
    const auto first = space.first_basis_per_interval();
    const double p = static_cast<double>(s.degree);
    const double tol = 4.0 * p * (p + 1.0) * unit_roundoff<T>();
    const std::vector<Rational> knots = oracle::exact_knots<T>(space.knots());
    for (std::size_t i = 0; i < n_int; ++i) {
      const Rational a = oracle::exact(static_cast<double>(uk[i]));
      const Rational b = oracle::exact(static_cast<double>(uk[i + 1]));
      // Samples across the whole interval, none at its ends.
      for (const Rational& q :
           {Rational(1, 16), Rational(3, 8), Rational(5, 8), Rational(15, 16)}) {
        const Rational x = a + (b - a) * q;
        oracle::BasisAt at(knots, s.degree, x);
        for (std::size_t j = 0; j < order; ++j) {
          Rational got = 0;
          for (std::size_t k = 0; k < order; ++k) {
            got += oracle::exact(static_cast<double>(ops.data()[(i * order + j) * order + k])) *
                   bernstein_exact(s.degree, static_cast<int>(k), q);
          }
          const Rational want = at.deriv(static_cast<std::size_t>(first[i]) + j, 0);
          CHECK(oracle::within(static_cast<double>(got), want, tol));
        }
      }
      // The columns of an operator sum to one: the basis is a partition of unity.
      for (std::size_t k = 0; k < order; ++k) {
        double sum = 0.0;
        for (std::size_t j = 0; j < order; ++j) {
          sum += static_cast<double>(ops.data()[(i * order + j) * order + k]);
        }
        CHECK(std::abs(sum - 1.0) <= tol);
      }
    }
  }
}

TEST_CASE_TEMPLATE("Bezier extraction operators: a raw knot vector agrees with a space", T, double,
                   float) {
  for (const Space1& s : spaces_1d()) {
    CAPTURE(s.name);
    const bs::BsplineSpace1D<T> space = make_space<T>(s);
    const NdArray<T> by_space = bs::bezier_extraction_operators<T>(space);
    const NdArray<T> by_raw =
        bs::bezier_extraction_operators<T>(space.knots(), s.degree, space.tolerance());
    REQUIRE(by_space.size() == by_raw.size());
    CHECK(std::equal(by_space.flat().begin(), by_space.flat().end(), by_raw.flat().begin()));
    // Into a caller's array.
    NdArray<T> into(std::vector<std::size_t>(by_space.shape().begin(), by_space.shape().end()));
    bs::bezier_extraction_operators<T>(space.knots(), s.degree, space.tolerance(),
                                       into.template view<3>());
    CHECK(std::equal(by_space.flat().begin(), by_space.flat().end(), into.flat().begin()));
  }
}

TEST_CASE("extraction operators: refusals and their messages") {
  const std::vector<double> ok = {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
  const auto sp = [](const std::vector<double>& v) { return std::span<const double>(v); };
  CHECK(message_of([&] { (void)bs::bezier_extraction_operators<double>(sp(ok), 2, -1.0); }) ==
        "tol must be non-negative");
  CHECK(message_of([&] { (void)bs::bezier_extraction_operators<double>(sp(ok), -1, 1e-10); }) ==
        "degree must be non-negative");
  CHECK(message_of([&] {
          (void)bs::bezier_extraction_operators<double>(sp({0.0, 1.0}), 2, 1e-10);
        }) == "knots must have at least 2*degree+2 elements");
  CHECK(message_of([&] {
          (void)bs::bezier_extraction_operators<double>(sp({0.0, 1.0, 0.5, 1.0, 1.0, 1.0}), 2,
                                                        1e-10);
        }) == "knots must be non-decreasing");
  CHECK(message_of([&] {
          (void)bs::bezier_extraction_operators<double>(
              sp({0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 1.0, 1.0, 1.0}), 2, 1e-10);
        }) == "knots must be non-decreasing");
  // The output extents are checked after the input.
  NdArray<double> wrong({1, 3, 3});
  CHECK(message_of([&] {
          bs::bezier_extraction_operators<double>(sp(ok), 2, 1e-10, wrong.view<3>());
        }) == "Output array has shape (1, 3, 3), but expected shape (2, 3, 3)");
  CHECK(message_of([&] {
          bs::lagrange_extraction_operators<double>(
              sp(ok), 2, -1.0, pantr::basis::LagrangeVariant::equispaces, wrong.view<3>());
        }) == "tol must be non-negative");
  CHECK(message_of([&] {
          bs::cardinal_extraction_operators<double>(sp(ok), 2, 1e-10, wrong.view<3>());
        }) == "Output array has shape (1, 3, 3), but expected shape (2, 3, 3)");
  // A degree beyond the change-of-basis matrix is refused before anything is written.
  const std::vector<double> huge = bspline_test::clamped(40, {0.5});
  NdArray<double> big({2, 41, 41});
  std::fill(big.flat().begin(), big.flat().end(), -5.0);
  CHECK_THROWS_AS(bs::cardinal_extraction_operators<double>(sp(huge), 40, 1e-10, big.view<3>()),
                  pantr::value_error);
  CHECK(std::all_of(big.flat().begin(), big.flat().end(), [](double v) { return v == -5.0; }));
}

TEST_CASE_TEMPLATE(
    "Lagrange and cardinal extraction operators: the Bezier operator times the "
    "change of basis",
    T, double, float) {
  using pantr::basis::LagrangeVariant;
  for (const Space1& s : spaces_1d()) {
    if (s.degree > 4) {
      continue;
    }
    CAPTURE(s.name);
    const bs::BsplineSpace1D<T> space = make_space<T>(s);
    const std::size_t order = static_cast<std::size_t>(s.degree) + 1;
    const std::size_t n_int = static_cast<std::size_t>(space.num_intervals());
    const NdArray<T> bez = bs::bezier_extraction_operators<T>(space);
    const double u = unit_roundoff<T>();

    // The product `bez[i] M` in double, and the bound on the stored result against it.
    const auto check_product = [&](const NdArray<T>& got, const NdArray<T>& matrix,
                                   const std::vector<bool>& identity) {
      for (std::size_t i = 0; i < n_int; ++i) {
        for (std::size_t r = 0; r < order; ++r) {
          for (std::size_t c = 0; c < order; ++c) {
            double want = 0.0;
            double terms = 0.0;
            for (std::size_t k = 0; k < order; ++k) {
              const double t = static_cast<double>(bez.data()[(i * order + r) * order + k]) *
                               static_cast<double>(matrix.data()[k * order + c]);
              want += t;
              terms += std::abs(t);
            }
            const double have = static_cast<double>(got.data()[(i * order + r) * order + c]);
            if (identity[i]) {
              CHECK(have == (r == c ? 1.0 : 0.0));
            } else {
              CHECK(std::abs(have - want) <= 2.0 * static_cast<double>(order) * u * terms);
            }
          }
        }
      }
    };

    std::vector<LagrangeVariant> variants = {LagrangeVariant::equispaces,
                                             LagrangeVariant::gauss_legendre,
                                             LagrangeVariant::chebyshev_1st};
    if (s.degree >= 1) {  // the Lobatto families need both ends
      variants.push_back(LagrangeVariant::gauss_lobatto_legendre);
      variants.push_back(LagrangeVariant::chebyshev_2nd);
    }
    if (s.degree == 0) {
      // No Lagrange basis of degree 0 is built (the change-of-basis matrix is refused).
      NdArray<T> none({n_int, order, order}, pantr::for_overwrite);
      CHECK(message_of([&] {
              bs::lagrange_extraction_operators<T>(space, LagrangeVariant::equispaces,
                                                   none.template view<3>());
            }) == "Degree must at least 1");
      variants.clear();
    }
    for (const LagrangeVariant variant : variants) {
      NdArray<T> got({n_int, order, order}, pantr::for_overwrite);
      bs::lagrange_extraction_operators<T>(space, variant, got.template view<3>());
      const auto matrix = pantr::change_basis::cached<T>(
          pantr::change_basis::Kind::lagrange_to_bernstein, s.degree, variant);
      check_product(got, *matrix, std::vector<bool>(n_int, false));
    }

    NdArray<T> card({n_int, order, order}, pantr::for_overwrite);
    bs::cardinal_extraction_operators<T>(space, card.template view<3>());
    const auto cmatrix = pantr::change_basis::cached<T>(
        pantr::change_basis::Kind::cardinal_to_bernstein, s.degree, LagrangeVariant::equispaces);
    std::unique_ptr<bool[]> flags(new bool[n_int]);
    space.cardinal_intervals(std::span<bool>(flags.get(), n_int));
    check_product(card, *cmatrix, std::vector<bool>(flags.get(), flags.get() + n_int));
  }
}

TEST_CASE(
    "cardinal extraction operators: a uniform knot vector is the identity on its cardinal "
    "intervals, and nowhere else is") {
  const std::vector<double> knots = {0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0};
  const bs::BsplineSpace1D<double> space(std::span<const double>(knots), 2);
  const NdArray<double> ops = [&] {
    NdArray<double> o({6, 3, 3}, pantr::for_overwrite);
    bs::cardinal_extraction_operators<double>(space, o.view<3>());
    return o;
  }();
  // Six unit intervals of degree 2: only the two end intervals see the repeated end knots.
  const std::vector<bool> cardinal = {false, true, true, true, true, false};
  for (std::size_t i = 0; i < 6; ++i) {
    bool is_identity = true;
    for (std::size_t r = 0; r < 3; ++r) {
      for (std::size_t c = 0; c < 3; ++c) {
        is_identity = is_identity && ops.data()[(i * 3 + r) * 3 + c] == (r == c ? 1.0 : 0.0);
      }
    }
    CHECK(is_identity == cardinal[i]);
  }
}

TEST_CASE_TEMPLATE(
    "Bezier structural identity mask: both bounding knots have multiplicity "
    "degree + 1",
    T, double, float) {
  struct Case {
    std::vector<double> knots;
    int degree;
    std::vector<bool> mask;
  };
  const std::vector<Case> cases = {
      {clamped(2, {0.5, 0.5, 0.5}), 2, {true, true}},
      {clamped(2, {0.5, 0.5}), 2, {false, false}},
      {clamped(2, {0.25, 0.5, 0.5, 0.5}), 2, {false, false, true}},
      {clamped(3, {}), 3, {true}},
      {{0.0, 0.5, 1.0}, 0, {true, true}},  // degree 0: multiplicity 1 = degree + 1 at every knot
      {clamped(1, {0.5, 0.5}), 1, {true, true}},
  };
  for (const Case& c : cases) {
    const std::vector<T> k = bspline_test::cast_all<T>(c.knots);
    const bs::BsplineSpace1D<T> space(std::span<const T>(k), c.degree);
    const std::size_t n = static_cast<std::size_t>(space.num_intervals());
    REQUIRE(n == c.mask.size());
    std::unique_ptr<bool[]> got(new bool[n]);
    bs::bezier_structural_identity_mask<T>(space, std::span<bool>(got.get(), n));
    for (std::size_t i = 0; i < n; ++i) {
      CHECK(got[i] == c.mask[i]);
    }
    // An identity-flagged interval has the identity as its Bézier operator.
    const NdArray<T> ops = bs::bezier_extraction_operators<T>(space);
    const std::size_t order = static_cast<std::size_t>(c.degree) + 1;
    for (std::size_t i = 0; i < n; ++i) {
      if (!got[i]) {
        continue;
      }
      for (std::size_t r = 0; r < order; ++r) {
        for (std::size_t q = 0; q < order; ++q) {
          CHECK(ops.data()[(i * order + r) * order + q] == (r == q ? T(1) : T(0)));
        }
      }
    }
  }
  // The flags array must hold one flag per interval.
  const std::vector<double> kd = clamped(2, {0.5});
  const bs::BsplineSpace1D<double> two(std::span<const double>(kd), 2);
  std::unique_ptr<bool[]> wrong(new bool[3]);
  CHECK(message_of([&] {
          bs::bezier_structural_identity_mask<double>(two, std::span<bool>(wrong.get(), 3));
        }) == "Output array has shape (3,), but expected shape (2,)");
}

// ---- the decomposition -----------------------------------------------------------------------

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
