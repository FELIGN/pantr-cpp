// The mode-wise contraction primitive (`bspline::detail::mode_product`).
//
// Oracle. The mode product is compared with a naive loop over multi-indices in exact integer
// arithmetic (the data are small integers, so every sum is exact in the storage type too).
#include <doctest/doctest.h>

#include "bspline_test_support.hpp"

#include <pantr/bspline/detail/mode_product.hpp>
#include <pantr/core/errors.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

using bspline_test::message_of;

namespace bsd = pantr::bspline::detail;

namespace {

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

          bsd::mode_product<T>({op, rows, cols, transpose}, {x, x_extents, x_begin}, mode,
                               {y, y_extents, y_begin});
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
  bsd::mode_product_whole<T>({op, 2, 3, false}, {x, ex}, 1, {y, ey});
  // Row r of x is (r+1) (1,1,1): applied to the columns gives (r+1) (6, 15).
  CHECK(y == std::vector<T>{6, 15, 12, 30});
  std::vector<T> z(6);
  const std::vector<std::size_t> ez = {2, 3};
  const std::vector<T> w = {1, 10, 2, 20};
  const std::vector<std::size_t> ew = {2, 2};
  bsd::mode_product_whole<T>({op, 2, 3, true}, {w, ew}, 1, {z, ez});
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
  const std::span<const double> x(both.data(), 12);
  const std::span<double> y(both.data() + 12, 12);
  CHECK_NOTHROW(bsd::mode_product<double>({op, 2, 2}, {x, e34, 1}, 1, {y, e34, 0}));
  // Row r of x is (4r, 4r+1, 4r+2, 4r+3); the window [1, 3) summed is 8r + 3, written at y[., 0].
  CHECK(both[12] == 3.0);
  CHECK(both[16] == 11.0);
  CHECK(both[20] == 19.0);
  // y one element into x overlaps it, and so does x one element into y.
  const std::span<double> shifted(both.data() + 11, 12);
  CHECK_THROWS_AS(bsd::mode_product<double>({op, 2, 2}, {x, e34, 1}, 1, {shifted, e34, 0}),
                  pantr::value_error);
  const std::span<const double> x_late(both.data() + 1, 12);
  const std::span<double> y_early(both.data() + 12, 12);
  CHECK_THROWS_AS(bsd::mode_product<double>({op, 2, 2}, {x_late, e34, 1}, 1, {y_early, e34, 0}),
                  pantr::value_error);
}

TEST_CASE("mode product: refusals and their messages") {
  const std::vector<double> op(4, 1.0);
  const std::vector<double> x(12, 1.0);
  std::vector<double> y(12, 0.0);
  const std::vector<std::size_t> e34 = {3, 4};
  const auto call = [&](std::span<const double> o, std::size_t r, std::size_t c, bool t,
                        std::span<const double> in, std::vector<std::size_t> ie, std::size_t mode,
                        std::size_t ib, std::span<double> out, std::vector<std::size_t> oe,
                        std::size_t ob) {
    return message_of(
        [&] { bsd::mode_product<double>({o, r, c, t}, {in, ie, ib}, mode, {out, oe, ob}); });
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
          bsd::mode_product_whole<double>({op, 2, 2}, {x, std::vector<std::size_t>{3, 4}}, 0,
                                          {y, std::vector<std::size_t>{3, 4}});
        }) ==
        "mode_product: the extents along the mode are 3 and 3, but the operator maps 2 to 2.");
}
