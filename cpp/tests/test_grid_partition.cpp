// Partition and partition_grid. Oracles: the owner arrays, factorizations and exactly rounded sums
// the Python implementation produced (recorded from it, with integer-valued weights so the C++ data
// are bit-identical to the Python data), hand-derived small cases, the structural contract (every
// active cell owned, every rank non-empty, inactive cells -1), and the box shape of the block
// subdomains. A wrapper type that is a GridLike but not a TensorProductGrid stands for the
// hierarchical grid, to exercise the generic path and the block backend's refusal.
#include <doctest/doctest.h>

#include <pantr/grid/detail/fsum.hpp>
#include <pantr/grid/partition.hpp>
#include <pantr/tolerance.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

using pantr::grid::Partition;
using pantr::grid::partition_grid;
using pantr::grid::PartitionBackend;
using pantr::grid::TensorProductGrid;
using pantr::grid::detail::divisors;
using pantr::grid::detail::factor_blocks;
using pantr::grid::detail::fsum;
using Bp = std::vector<std::vector<double>>;

namespace {

using I64 = std::int64_t;
using I32 = std::int32_t;

/// A tensor-product grid with the breakpoints 0, 1, ..., n on each axis.
TensorProductGrid unit_grid(const std::vector<I64>& cells) {
  Bp bp;
  for (const I64 n : cells) {
    std::vector<double> axis;
    for (I64 i = 0; i <= n; ++i) {
      axis.push_back(static_cast<double>(i));
    }
    bp.push_back(axis);
  }
  return TensorProductGrid(std::move(bp));
}

/// A GridLike that is not a TensorProductGrid, as the hierarchical grid will be.
class OtherGrid {
 public:
  static constexpr std::string_view type_name = "HierarchicalGrid";
  explicit OtherGrid(TensorProductGrid g) : g_(std::move(g)) {}
  [[nodiscard]] std::size_t ndim() const { return g_.ndim(); }
  [[nodiscard]] I64 num_cells() const { return g_.num_cells(); }
  [[nodiscard]] std::pair<std::vector<double>, std::vector<double>> cell_bounds(I64 cid) const {
    return g_.cell_bounds(cid);
  }
  [[nodiscard]] std::pair<pantr::NdArray<double>, pantr::NdArray<double>> collect_cell_bounds()
      const {
    return g_.collect_cell_bounds();
  }

 private:
  TensorProductGrid g_;
};

static_assert(pantr::grid::GridLike<TensorProductGrid>);
static_assert(pantr::grid::GridLike<OtherGrid>);

std::vector<I32> owners(const Partition& p) {
  return {p.cell_owner().begin(), p.cell_owner().end()};
}

/// Weights (i * a) mod m, integer-valued so that they are the same numbers in the Python data.
std::vector<double> int_weights(std::size_t n, std::size_t a, std::size_t m) {
  std::vector<double> w(n);
  for (std::size_t i = 0; i < n; ++i) {
    w[i] = static_cast<double>((i * a) % m);
  }
  return w;
}

std::vector<std::uint8_t> active_not_multiple_of_3(std::size_t n) {
  std::vector<std::uint8_t> a(n);
  for (std::size_t i = 0; i < n; ++i) {
    a[i] = (i % 3) != 0 ? 1 : 0;
  }
  return a;
}

}  // namespace

TEST_CASE("grid: Partition: owners, counts, masks and owned cells") {
  const Partition p({0, 1, -1, 1, 0, 2}, 3);
  CHECK(p.n_parts() == 3);
  CHECK(p.n_cells() == 6);
  CHECK(owners(p) == std::vector<I32>{0, 1, -1, 1, 0, 2});
  CHECK(p.active_mask() == std::vector<std::uint8_t>{1, 1, 0, 1, 1, 1});
  CHECK(p.owned_cells(1) == std::vector<I64>{1, 3});
  CHECK(p.owned_cells(2) == std::vector<I64>{5});
  const Partition empty({}, 1);
  CHECK(empty.n_cells() == 0);
  CHECK(empty.owned_cells(0).empty());
}

TEST_CASE("grid: Partition: validation messages") {
  CHECK_THROWS_WITH_AS(Partition({0}, 0), "n_parts must be >= 1; got 0.", pantr::value_error);
  CHECK_THROWS_WITH_AS(Partition({0, 2}, 2),
                       "cell_owner values must lie in [-1, 2); got range [0, 2].",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(Partition({-2, 0}, 2),
                       "cell_owner values must lie in [-1, 2); got range [-2, 0].",
                       pantr::value_error);
  // n_parts is checked first, then the range.
  CHECK_THROWS_WITH_AS(Partition({5}, -1), "n_parts must be >= 1; got -1.", pantr::value_error);
  const Partition p({0, 1}, 2);
  CHECK_THROWS_WITH_AS(static_cast<void>(p.owned_cells(2)), "rank must be in [0, 2); got 2.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(p.owned_cells(-1)), "rank must be in [0, 2); got -1.",
                       pantr::value_error);
}

TEST_CASE("grid: fsum: exactly rounded sums, equal bit for bit to math.fsum") {
  CHECK(fsum(std::vector<double>{1e100, 1.0, -1e100}) == 1.0);
  CHECK(fsum(std::vector<double>(10, 0.1)) == 1.0);  // the naive sum is 0.9999999999999999
  CHECK(fsum(std::vector<double>{1e16, 1.0, 1.0}) == 0x1.1c37937e08001p+53);
  // Half-even rounding across partials.
  CHECK(fsum(std::vector<double>{1.0, 0x1p-53, 0x1p-105}) == 0x1.0000000000001p+0);
  CHECK(fsum(std::vector<double>{1.0, 0x1p-53, 0x1p-53}) == 0x1.0000000000001p+0);
  CHECK(fsum(std::vector<double>{-1.0, 0x1p-53, 0x1p-105}) == -0x1.fffffffffffffp-1);
  CHECK(fsum(std::vector<double>{3.0, 1e-20, -3.0, 1e-20}) == 0x1.79ca10c924223p-66);
  // Cases where the correction across partials decides the last bit (found by comparing
  // math.fsum with the same algorithm minus the correction).
  CHECK(fsum(std::vector<double>{-0x1p-62, 0x1p-39, 0x1p-10, -0x1p-116, 0x1p-9}) ==
        0x1.80000003fffffp-9);
  CHECK(fsum(std::vector<double>{0x1.4p-11, -0x1p-57, -0x1p-3, -0x1.8p-92, 0x1.8p-9}) ==
        -0x1.f180000000001p-4);
  CHECK(fsum(std::vector<double>{0x1.4p-29, -0x1p-110, 0x1.4p-80, 0x1.8p-27}) ==
        0x1.d000000000001p-27);
  CHECK(fsum(std::vector<double>{}) == 0.0);
  CHECK(fsum(std::vector<double>{-0.5}) == -0.5);
}

TEST_CASE("grid: fsum: the result does not depend on the order of the terms") {
  std::vector<double> terms;
  for (int i = 0; i < 200; ++i) {
    terms.push_back(std::ldexp(static_cast<double>((i * 37) % 101 + 1), (i % 9) * 11 - 40) *
                    (i % 2 == 0 ? 1.0 : -1.0));
  }
  const double forward = fsum(terms);
  std::reverse(terms.begin(), terms.end());
  CHECK(fsum(terms) == forward);
  std::rotate(terms.begin(), terms.begin() + 77, terms.end());
  CHECK(fsum(terms) == forward);
}

TEST_CASE("grid: divisors: ascending, with a perfect square counted once") {
  CHECK(divisors(1) == std::vector<I64>{1});
  CHECK(divisors(4) == std::vector<I64>{1, 2, 4});
  CHECK(divisors(9) == std::vector<I64>{1, 3, 9});
  CHECK(divisors(12) == std::vector<I64>{1, 2, 3, 4, 6, 12});
  CHECK(divisors(13) == std::vector<I64>{1, 13});
  CHECK(divisors(36) == std::vector<I64>{1, 2, 3, 4, 6, 9, 12, 18, 36});
  // Brute force for every n up to 400.
  for (I64 n = 1; n <= 400; ++n) {
    std::vector<I64> expect;
    for (I64 d = 1; d <= n; ++d) {
      if (n % d == 0) {
        expect.push_back(d);
      }
    }
    CHECK(divisors(n) == expect);
  }
  // 2^40 has 41 divisors.
  CHECK(divisors((I64{1} << 20) * (I64{1} << 20)).size() == 41);
}

TEST_CASE("grid: factor_blocks: the most cube-like factorization, ties to the first enumerated") {
  struct Case {
    std::vector<I64> cells;
    I64 n_parts;
    std::vector<I64> expected;
  };
  const std::vector<Case> cases = {
      {{8}, 4, {4}},
      {{4, 4}, 4, {2, 2}},
      {{8, 2}, 4, {4, 1}},
      {{6, 6}, 6, {2, 3}},
      {{12}, 12, {12}},
      {{3, 3, 3}, 9, {1, 3, 3}},
      {{4, 3, 2}, 6, {3, 2, 1}},
      {{10, 10}, 8, {2, 4}},
      {{5, 7}, 6, {2, 3}},
      {{2, 2, 2}, 8, {2, 2, 2}},
      {{16, 4}, 8, {4, 2}},
  };
  for (const Case& c : cases) {
    CHECK(factor_blocks(c.cells, c.n_parts) == c.expected);
    I64 product = 1;
    for (std::size_t d = 0; d < c.cells.size(); ++d) {
      CHECK(c.expected[d] <= c.cells[d]);
      product *= c.expected[d];
    }
    CHECK(product == c.n_parts);
  }
}

TEST_CASE("grid: extent_imbalance: the variance of the log block extents") {
  using pantr::grid::detail::extent_imbalance;
  // Equal extents: the deviations are exactly zero.
  const std::vector<I64> c1 = {8, 2};
  const std::vector<I64> b1 = {4, 1};
  CHECK(extent_imbalance(c1, b1) == 0.0);
  // Values recorded from the Python implementation. libm may differ from CPython's in the last
  // bits of a logarithm: each log is within an ulp, the deviations then carry an absolute error
  // of a few units of roundoff times max|log| ~ 2, against deviations of 0.2 and 1, so the
  // relative error of the variance is a few units of roundoff times 10; the default tolerance,
  // 64 units of roundoff, covers it.
  const double tol = pantr::tolerance::get_default<double>();
  const std::vector<I64> c2 = {6, 6};
  const std::vector<I64> b2 = {2, 3};
  const double v2 = 0x1.50b1f8baf508ep-5;
  CHECK(std::fabs(extent_imbalance(c2, b2) - v2) <= tol * v2);
  const std::vector<I64> c3 = {5, 7};
  const std::vector<I64> b3 = {1, 6};
  const double v3 = 0x1.0f161122998f6p-1;
  CHECK(std::fabs(extent_imbalance(c3, b3) - v3) <= tol * v3);
}

TEST_CASE("grid: factor_blocks: an infeasible part count names the grid") {
  const std::vector<I64> cells = {4, 4};
  CHECK_THROWS_WITH_AS(static_cast<void>(factor_blocks(cells, 7)),
                       "cannot factor n_parts=7 across axes with cells_per_axis=(4, 4) without "
                       "leaving a rank empty. Use a part count that divides the grid better, or "
                       "the 'rcb' backend.",
                       pantr::value_error);
  const std::vector<I64> one = {3};
  CHECK_THROWS_WITH_AS(static_cast<void>(factor_blocks(one, 4)),
                       "cannot factor n_parts=4 across axes with cells_per_axis=(3,) without "
                       "leaving a rank empty. Use a part count that divides the grid better, or "
                       "the 'rcb' backend.",
                       pantr::value_error);
}

TEST_CASE("grid: partition_grid block: the balanced contiguous Cartesian split") {
  struct Case {
    std::vector<I64> cells;
    I64 n_parts;
    std::vector<I32> expected;
  };
  const std::vector<Case> cases = {
      {{4, 4}, 4, {0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3}},
      {{6, 4}, 6, {0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 5, 5}},
      {{7, 5}, 6, {0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 2, 2, 2,
                   3, 3, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 4, 4, 4, 5, 5}},
      {{4, 3, 2}, 6, {0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5}},
      {{5, 3}, 4, {0, 0, 1, 0, 0, 1, 0, 0, 1, 2, 2, 3, 2, 2, 3}},
      {{9}, 4, {0, 0, 0, 1, 1, 2, 2, 3, 3}},
  };
  for (const Case& c : cases) {
    const TensorProductGrid g = unit_grid(c.cells);
    const Partition p = partition_grid(g, c.n_parts, PartitionBackend::block);
    CHECK(owners(p) == c.expected);
    CHECK(p.n_parts() == c.n_parts);
  }
}

TEST_CASE("grid: partition_grid block: every subdomain is a box of cells, none is empty") {
  const std::vector<I64> cells = {12, 8, 6};
  const TensorProductGrid g = unit_grid(cells);
  const Partition p = partition_grid(g, 12, PartitionBackend::block);
  for (I64 rank = 0; rank < 12; ++rank) {
    const std::vector<I64> owned = p.owned_cells(rank);
    REQUIRE_FALSE(owned.empty());
    // The multi-index bounding box of the rank's cells holds exactly its cells.
    std::vector<I64> lo(3, (std::numeric_limits<I64>::max)());
    std::vector<I64> hi(3, -1);
    for (const I64 cid : owned) {
      const std::vector<I64> m = g.cell_multi_index(cid);
      for (std::size_t d = 0; d < 3; ++d) {
        lo[d] = (std::min)(lo[d], m[d]);
        hi[d] = (std::max)(hi[d], m[d]);
      }
    }
    CHECK((hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) * (hi[2] - lo[2] + 1) ==
          static_cast<I64>(owned.size()));
    // Balanced to within the integer split: a block spans floor or ceil of cells / blocks.
    CHECK(owned.size() == static_cast<std::size_t>(12 * 8 * 6 / 12));
  }
}

TEST_CASE("grid: partition_grid rcb: recorded owner arrays, uniform and weighted and masked") {
  const TensorProductGrid g = unit_grid({4, 4});
  const auto rcb = [&](I64 n, std::optional<std::span<const double>> w = std::nullopt,
                       std::optional<std::span<const std::uint8_t>> a = std::nullopt) {
    return owners(partition_grid(g, n, PartitionBackend::rcb, w, a));
  };
  CHECK(rcb(3) == std::vector<I32>{0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2});
  CHECK(rcb(5) == std::vector<I32>{0, 0, 1, 1, 0, 0, 1, 3, 2, 2, 3, 3, 2, 4, 4, 4});
  const std::vector<double> w = int_weights(16, 37, 11);
  CHECK(rcb(3, w) == std::vector<I32>{0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2});
  const std::vector<std::uint8_t> act = active_not_multiple_of_3(16);
  CHECK(rcb(3, std::nullopt, act) ==
        std::vector<I32>{-1, 0, 0, -1, 0, 0, -1, 2, 1, -1, 1, 2, -1, 1, 2, -1});
  CHECK(rcb(4, w, act) == std::vector<I32>{-1, 0, 1, -1, 0, 0, -1, 1, 2, -1, 3, 3, -1, 2, 3, -1});
  // All-zero weights: the cumulative weight is 0 everywhere, the split point clamps.
  const std::vector<double> zero(16, 0.0);
  CHECK(rcb(3, zero) == std::vector<I32>{0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
}

TEST_CASE("grid: partition_grid rcb: degenerate weights are clamped so that no rank is empty") {
  // With all weight on one cell, or none, the weighted split point lands at an end; the clamp
  // keeps at least as many cells on each side as there are parts on it.
  const TensorProductGrid g = unit_grid({4, 4});
  const auto rcb = [&](I64 n, const std::vector<double>& w) {
    return owners(partition_grid(g, n, PartitionBackend::rcb, w));
  };
  const std::vector<double> zero(16, 0.0);
  CHECK(rcb(4, zero) == std::vector<I32>{0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3});
  CHECK(rcb(5, zero) == std::vector<I32>{0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
  CHECK(rcb(8, zero) == std::vector<I32>{0, 1, 2, 3, 4, 7, 7, 7, 5, 7, 7, 7, 6, 7, 7, 7});
  std::vector<double> first(16, 0.0);
  first[0] = 100.0;
  CHECK(rcb(4, first) == std::vector<I32>{0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3});
  CHECK(rcb(3, first) == std::vector<I32>{0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
  std::vector<double> last(16, 0.0);
  last[15] = 100.0;
  CHECK(rcb(4, last) == std::vector<I32>{0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3});
  CHECK(rcb(3, last) == std::vector<I32>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2});
  const TensorProductGrid line = unit_grid({10});
  std::vector<double> w1(10, 0.0);
  w1[9] = 5.0;
  CHECK(owners(partition_grid(line, 5, PartitionBackend::rcb, w1)) ==
        std::vector<I32>{0, 1, 1, 1, 1, 1, 1, 2, 3, 4});
  std::vector<double> w0(10, 0.0);
  w0[0] = 5.0;
  CHECK(owners(partition_grid(line, 5, PartitionBackend::rcb, w0)) ==
        std::vector<I32>{0, 1, 2, 3, 4, 4, 4, 4, 4, 4});
}

TEST_CASE("grid: partition_grid rcb: a three-dimensional non-uniform weighted grid") {
  const TensorProductGrid g(Bp{{0, 1, 1.5, 3, 3.25, 7.0}, {0, 2, 3.0, 4, 10.0}, {0, 1.0, 5.0}});
  const std::vector<double> w = int_weights(static_cast<std::size_t>(g.num_cells()), 53, 17);
  const Partition p = partition_grid(g, 5, PartitionBackend::rcb, w);
  CHECK(owners(p) == std::vector<I32>{0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 3, 3, 0, 0, 0, 0,
                                      3, 3, 3, 3, 0, 1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4});
}

TEST_CASE("grid: partition_grid rcb: the structural contract on a larger weighted grid") {
  const TensorProductGrid g = unit_grid({40, 30});
  const std::size_t n = 1200;
  const std::vector<double> w = int_weights(n, 91, 23);
  const std::vector<std::uint8_t> act = active_not_multiple_of_3(n);
  for (const I64 parts : {1, 2, 7, 16, 63}) {
    const Partition p = partition_grid(g, parts, PartitionBackend::rcb, w, act);
    std::vector<double> load(static_cast<std::size_t>(parts), 0.0);
    std::vector<I64> count(static_cast<std::size_t>(parts), 0);
    for (std::size_t c = 0; c < n; ++c) {
      const I32 o = p.cell_owner()[c];
      if (act[c] == 0) {
        CHECK(o == -1);
      } else {
        REQUIRE(o >= 0);
        REQUIRE(o < parts);
        load[static_cast<std::size_t>(o)] += w[c];
        ++count[static_cast<std::size_t>(o)];
      }
    }
    for (I64 r = 0; r < parts; ++r) {
      CHECK(count[static_cast<std::size_t>(r)] > 0);
    }
    CHECK(p.active_mask() == act);
  }
}

TEST_CASE(
    "grid: partition_grid auto: block when the grid is tensor-product and the count factors, rcb "
    "otherwise") {
  const TensorProductGrid g = unit_grid({4, 4});
  CHECK(owners(partition_grid(g, 4)) == owners(partition_grid(g, 4, PartitionBackend::block)));
  // 7 does not factor onto 4 x 4: auto falls back to rcb, where block throws.
  CHECK(owners(partition_grid(g, 7)) ==
        std::vector<I32>{0, 0, 1, 2, 0, 1, 2, 5, 3, 3, 4, 5, 3, 4, 6, 6});
  CHECK(owners(partition_grid(g, 7)) == owners(partition_grid(g, 7, PartitionBackend::rcb)));
  CHECK_THROWS_AS(static_cast<void>(partition_grid(g, 7, PartitionBackend::block)),
                  pantr::value_error);
  // Weights or activity send auto to rcb even when the count factors.
  const std::vector<double> w = int_weights(16, 37, 11);
  CHECK(owners(partition_grid(g, 4, PartitionBackend::automatic, w)) ==
        owners(partition_grid(g, 4, PartitionBackend::rcb, w)));
  // One part: everything is rank 0.
  CHECK(owners(partition_grid(g, 1)) == std::vector<I32>(16, 0));
}

TEST_CASE("grid: partition_grid: argument errors carry the Python messages") {
  const TensorProductGrid g = unit_grid({4, 4});
  CHECK_THROWS_WITH_AS(static_cast<void>(partition_grid(g, 0)), "n_parts must be >= 1; got 0.",
                       pantr::value_error);
  const std::vector<double> short_w(15, 1.0);
  CHECK_THROWS_WITH_AS(static_cast<void>(partition_grid(g, 2, PartitionBackend::rcb, short_w)),
                       "cell_weights must have shape (16,); got (15,).", pantr::value_error);
  std::vector<double> neg(16, 1.0);
  neg[3] = -0.5;
  CHECK_THROWS_WITH_AS(static_cast<void>(partition_grid(g, 2, PartitionBackend::rcb, neg)),
                       "cell_weights must be finite and non-negative.", pantr::value_error);
  std::vector<double> nan_w(16, 1.0);
  nan_w[3] = std::nan("");
  CHECK_THROWS_WITH_AS(static_cast<void>(partition_grid(g, 2, PartitionBackend::rcb, nan_w)),
                       "cell_weights must be finite and non-negative.", pantr::value_error);
  std::vector<double> inf_w(16, 1.0);
  inf_w[3] = std::numeric_limits<double>::infinity();
  CHECK_THROWS_AS(static_cast<void>(partition_grid(g, 2, PartitionBackend::rcb, inf_w)),
                  pantr::value_error);
  const std::vector<std::uint8_t> short_a(15, 1);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(partition_grid(g, 2, PartitionBackend::rcb, std::nullopt, short_a)),
      "cell_active must have shape (16,); got (15,).", pantr::value_error);
  const std::vector<std::uint8_t> none(16, 0);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(partition_grid(g, 2, PartitionBackend::rcb, std::nullopt, none)),
      "cell_active must mark at least one cell active.", pantr::value_error);
  // Validated before the backend is looked at, as in Python.
  CHECK_THROWS_WITH_AS(static_cast<void>(partition_grid(g, 2, PartitionBackend::block, short_w)),
                       "cell_weights must have shape (16,); got (15,).", pantr::value_error);
  const std::vector<double> ok_w(16, 1.0);
  CHECK_THROWS_WITH_AS(static_cast<void>(partition_grid(g, 2, PartitionBackend::block, ok_w)),
                       "the 'block' backend does not support cell_weights or cell_active; use "
                       "backend='rcb' (or 'auto').",
                       pantr::value_error);
  const std::vector<std::uint8_t> ok_a(16, 1);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(partition_grid(g, 2, PartitionBackend::block, std::nullopt, ok_a)),
      "the 'block' backend does not support cell_weights or cell_active; use backend='rcb' (or "
      "'auto').",
      pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(partition_grid(g, 17, PartitionBackend::rcb)),
                       "n_parts=17 exceeds the number of active cells (16); cannot assign every "
                       "rank a cell.",
                       pantr::value_error);
  const std::vector<std::uint8_t> two_active = {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  CHECK_THROWS_WITH_AS(
      static_cast<void>(
          partition_grid(g, 3, PartitionBackend::automatic, std::nullopt, two_active)),
      "n_parts=3 exceeds the number of active cells (2); cannot assign every rank a cell.",
      pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(partition_grid(g, 7, PartitionBackend::block)),
                       "cannot factor n_parts=7 across axes with cells_per_axis=(4, 4) without "
                       "leaving a rank empty. Use a part count that divides the grid better, or "
                       "the 'rcb' backend.",
                       pantr::value_error);
}

TEST_CASE("grid: partition_grid: a grid that is not a TensorProductGrid has no block backend") {
  const OtherGrid h(unit_grid({4, 4}));
  CHECK_THROWS_WITH_AS(static_cast<void>(partition_grid(h, 4, PartitionBackend::block)),
                       "the 'block' backend requires a TensorProductGrid; got HierarchicalGrid.",
                       pantr::value_error);
  // rcb and auto work, and auto is rcb for such a grid whatever the count.
  const TensorProductGrid g = unit_grid({4, 4});
  CHECK(owners(partition_grid(h, 4, PartitionBackend::rcb)) ==
        owners(partition_grid(g, 4, PartitionBackend::rcb)));
  CHECK(owners(partition_grid(h, 4)) == owners(partition_grid(g, 4, PartitionBackend::rcb)));
}
