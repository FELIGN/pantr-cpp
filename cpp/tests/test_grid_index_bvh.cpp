// Cell-index helpers and the BVH. Oracles: enumeration of the multi-indices by nested loops with a
// running counter (against the stride formula), brute force over every box (against the tree
// traversal), the second implementation `AABB::overlaps` (against the separating-axis test), and a
// tree derived by hand for four unit squares.
#include <doctest/doctest.h>

#include <pantr/geometry/aabb.hpp>
#include <pantr/grid/bvh.hpp>
#include <pantr/grid/cell_index.hpp>
#include <pantr/grid/detail/checked.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

using pantr::geometry::AABB;
using pantr::grid::BVH;
using pantr::grid::c_order_strides;
using pantr::grid::flat_to_multi;
using pantr::grid::multi_to_flat;

namespace {

using I64 = std::int64_t;

/// A small deterministic generator (a 64-bit LCG); the tests need reproducible, not random, data.
class Lcg {
 public:
  explicit Lcg(std::uint64_t seed) : state_(seed) {}
  /// Uniform in [0, 1).
  double next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state_ >> 11) / 9007199254740992.0;
  }

 private:
  std::uint64_t state_;
};

struct Boxes {
  std::vector<double> lo;
  std::vector<double> hi;
  std::vector<std::size_t> shape;
};

Boxes random_boxes(std::size_t n, std::size_t ndim, std::uint64_t seed) {
  Lcg rng(seed);
  Boxes b;
  b.shape = {n, ndim};
  for (std::size_t i = 0; i < n * ndim; ++i) {
    const double l = 10.0 * rng.next();
    b.lo.push_back(l);
    b.hi.push_back(l + 0.5 * rng.next());
  }
  return b;
}

BVH build(const Boxes& b) { return BVH::from_cell_bounds(b.lo, b.hi, b.shape, b.shape); }

/// The cells whose closed box is not separated from the query on some axis, by brute force.
std::vector<I64> brute_force_query(const Boxes& b, const std::vector<double>& qlo,
                                   const std::vector<double>& qhi) {
  const std::size_t ndim = b.shape[1];
  std::vector<I64> out;
  for (std::size_t c = 0; c < b.shape[0]; ++c) {
    bool separated = false;
    for (std::size_t d = 0; d < ndim; ++d) {
      if (qhi[d] < b.lo[c * ndim + d] || qlo[d] > b.hi[c * ndim + d]) {
        separated = true;
      }
    }
    if (!separated) {
      out.push_back(static_cast<I64>(c));
    }
  }
  return out;
}

std::vector<I64> sorted(std::vector<I64> v) {
  std::sort(v.begin(), v.end());
  return v;
}

/// A chain tree: internal node k (k < m) is node 2k, with leaf 2k + 1 on the left and node 2k + 2
/// on the right; the last right child is a leaf. Depth m + 1 with m + 1 leaves.
struct Chain {
  std::vector<double> lo;
  std::vector<double> hi;
  std::vector<I64> left;
  std::vector<I64> right;
  std::vector<I64> cell;
  std::vector<std::size_t> shape;
  I64 n_cells;
};

Chain make_chain(I64 m) {
  Chain c;
  const auto n_nodes = static_cast<std::size_t>(2 * m + 1);
  c.lo.assign(n_nodes, 0.0);
  c.hi.assign(n_nodes, 1.0);
  c.left.assign(n_nodes, -1);
  c.right.assign(n_nodes, -1);
  c.cell.assign(n_nodes, -1);
  for (I64 k = 0; k < m; ++k) {
    c.left[static_cast<std::size_t>(2 * k)] = 2 * k + 1;
    c.right[static_cast<std::size_t>(2 * k)] = 2 * k + 2;
    c.cell[static_cast<std::size_t>(2 * k + 1)] = k;
  }
  c.cell[static_cast<std::size_t>(2 * m)] = m;
  c.shape = {n_nodes, 1};
  c.n_cells = m + 1;
  return c;
}

BVH from_chain(const Chain& c) {
  return {c.lo, c.hi, c.shape, c.shape, c.left, c.right, c.cell, c.n_cells};
}

}  // namespace

TEST_CASE("grid: cell index: strides and the round trip agree with enumeration by nested loops") {
  const I64 cells[3] = {3, 4, 2};
  const std::vector<I64> strides = c_order_strides(cells);
  CHECK(strides == std::vector<I64>{8, 2, 1});
  const I64 one[1] = {5};
  CHECK(c_order_strides(one) == std::vector<I64>{1});

  I64 expected_flat = 0;
  for (I64 i = 0; i < 3; ++i) {
    for (I64 j = 0; j < 4; ++j) {
      for (I64 k = 0; k < 2; ++k) {
        const std::vector<I64> multi = {i, j, k};
        CHECK(multi_to_flat(multi, cells) == expected_flat);
        CHECK(flat_to_multi(expected_flat, cells) == multi);
        ++expected_flat;
      }
    }
  }
  CHECK(expected_flat == 24);
}

TEST_CASE("grid: cell index: errors carry the Python messages") {
  const I64 cells[3] = {3, 4, 2};
  CHECK_THROWS_WITH_AS(static_cast<void>(flat_to_multi(24, cells)),
                       "flat cell id 24 is out of range [0, 24).", pantr::index_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(flat_to_multi(-1, cells)),
                       "flat cell id -1 is out of range [0, 24).", pantr::index_error);
  const std::vector<I64> short_multi = {1, 2};
  CHECK_THROWS_WITH_AS(static_cast<void>(multi_to_flat(short_multi, cells)),
                       "multi-index has length 2; expected 3.", pantr::value_error);
  const std::vector<I64> bad_axis = {1, 4, 0};
  CHECK_THROWS_WITH_AS(static_cast<void>(multi_to_flat(bad_axis, cells)),
                       "axis 1 index 4 out of range [0, 4).", pantr::index_error);
  const std::vector<I64> negative = {-1, 0, 0};
  CHECK_THROWS_WITH_AS(static_cast<void>(multi_to_flat(negative, cells)),
                       "axis 0 index -1 out of range [0, 3).", pantr::index_error);
}

TEST_CASE("grid: cell index: a negative cell count is refused before checked_mul sees it") {
  // checked_mul assumes non-negative operands, so each entry point states the precondition.
  const I64 negative_first[2] = {-3, 4};
  const I64 negative_last[2] = {3, -4};
  const std::vector<I64> multi = {0, 0};
  CHECK_THROWS_WITH_AS(static_cast<void>(c_order_strides(negative_first)),
                       "cells_per_axis entries must be >= 0; got -3.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(flat_to_multi(0, negative_last)),
                       "cells_per_axis entries must be >= 0; got -4.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(multi_to_flat(multi, negative_first)),
                       "cells_per_axis entries must be >= 0; got -3.", pantr::value_error);
  // Zero is a legal count: an empty axis has no valid multi-index, and the check does not mask it.
  const I64 empty_axis[2] = {3, 0};
  CHECK(c_order_strides(empty_axis) == std::vector<I64>{0, 1});
  CHECK_THROWS_AS(static_cast<void>(flat_to_multi(0, empty_axis)), pantr::index_error);
  CHECK_THROWS_AS(static_cast<void>(multi_to_flat(multi, empty_axis)), pantr::index_error);
}

TEST_CASE(
    "grid: cell index: multi_to_flat refuses a cell count that overflows, like flat_to_multi") {
  // 3037000500^2 > INT64_MAX, so the total does not fit. The old loop's multiplication
  // 3037000499 * 3037000500 still fit; only the following `+ 3037000499` overflowed, which the
  // checked multiply did not see. The product check refuses first, as flat_to_multi does.
  const I64 cells[2] = {3037000500, 3037000500};
  const I64 multi[2] = {3037000499, 3037000499};
  CHECK_THROWS_AS(static_cast<void>(multi_to_flat(multi, cells)), std::overflow_error);
  CHECK_THROWS_AS(static_cast<void>(flat_to_multi(0, cells)), std::overflow_error);
  // The largest total that fits: its last cell round-trips.
  const I64 fits[2] = {3037000499, 3037000499};
  const I64 last[2] = {3037000498, 3037000498};
  const I64 flat = multi_to_flat(last, fits);
  CHECK(flat == 3037000499LL * 3037000499LL - 1);
  CHECK(flat_to_multi(flat, fits) == std::vector<I64>{3037000498, 3037000498});
}

TEST_CASE("grid: checked_mul: exact at the int64 boundary, refusing the first product past it") {
  using pantr::grid::detail::checked_mul;
  constexpr I64 hi = std::numeric_limits<I64>::max();
  // hi = 7 * 1317624576693539401, so the factorization sits exactly on the boundary.
  constexpr I64 seventh = 1317624576693539401;
  CHECK(checked_mul(7, seventh) == hi);
  CHECK(checked_mul(seventh, 7) == hi);
  CHECK_THROWS_AS(static_cast<void>(checked_mul(8, seventh)), std::overflow_error);
  CHECK_THROWS_AS(static_cast<void>(checked_mul(seventh, 8)), std::overflow_error);
  // Equal factors: 3037000499^2 is the largest square below 2^63.
  CHECK(checked_mul(3037000499, 3037000499) == 9223372030926249001);
  CHECK_THROWS_AS(static_cast<void>(checked_mul(3037000500, 3037000500)), std::overflow_error);
  // Powers of two: 2^31 * 2^31 = 2^62 fits, 2^32 * 2^31 = 2^63 does not.
  CHECK(checked_mul(I64{1} << 31, I64{1} << 31) == I64{1} << 62);
  CHECK_THROWS_AS(static_cast<void>(checked_mul(I64{1} << 32, I64{1} << 31)), std::overflow_error);
  CHECK_THROWS_AS(static_cast<void>(checked_mul(hi, 2)), std::overflow_error);
  // The identity and the zero, which need no division.
  CHECK(checked_mul(hi, 1) == hi);
  CHECK(checked_mul(1, hi) == hi);
  CHECK(checked_mul(0, hi) == 0);
  CHECK(checked_mul(hi, 0) == 0);
  CHECK(checked_mul(0, 0) == 0);
}

TEST_CASE("grid: cell index: a stride or a cell count beyond int64 is refused") {
  const I64 big = I64{1} << 32;
  const I64 cells[4] = {big, big, big, 2};
  CHECK_THROWS_AS(static_cast<void>(c_order_strides(cells)), std::overflow_error);
  // The last stride is the product of all but the first axis; the first axis alone may be large.
  const I64 fits[3] = {I64{1} << 40, I64{1} << 20, I64{1} << 20};
  CHECK(c_order_strides(fits) == std::vector<I64>{I64{1} << 40, I64{1} << 20, 1});
  const I64 total_overflows[2] = {big, big * 4};
  CHECK_THROWS_AS(static_cast<void>(flat_to_multi(0, total_overflows)), std::overflow_error);
}

TEST_CASE("grid: BVH: four unit squares give the tree derived by hand") {
  // Squares [i, i+1] x [j, j+1] for (i, j) = (0,0), (0,1), (1,0), (1,1), cells 0..3.
  // Root box [0,2]^2: equal extents, so axis 0 (the lowest on a tie); stable sort by x centroid
  // keeps the order 0, 1 | 2, 3. The left node (box [0,1] x [0,2]) splits on axis 1 into cells 0
  // and 1, the right node into cells 2 and 3. Nodes are claimed in preorder, left first: the root
  // claims 1 and 2, node 1 claims 3 and 4, node 2 claims 5 and 6.
  const Boxes b{{0, 0, 0, 1, 1, 0, 1, 1}, {1, 1, 1, 2, 2, 1, 2, 2}, {4, 2}};
  const BVH t = build(b);
  CHECK(t.ndim() == 2);
  CHECK(t.n_cells() == 4);
  CHECK(t.n_nodes() == 7);
  CHECK(std::vector<I64>(t.node_cell().begin(), t.node_cell().end()) ==
        std::vector<I64>{-1, -1, -1, 0, 1, 2, 3});
  CHECK(std::vector<I64>(t.node_left().begin(), t.node_left().end()) ==
        std::vector<I64>{1, 3, 5, -1, -1, -1, -1});
  CHECK(std::vector<I64>(t.node_right().begin(), t.node_right().end()) ==
        std::vector<I64>{2, 4, 6, -1, -1, -1, -1});
  CHECK(std::vector<double>(t.node_lo().begin(), t.node_lo().end()) ==
        std::vector<double>{0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1});
  CHECK(std::vector<double>(t.node_hi().begin(), t.node_hi().end()) ==
        std::vector<double>{2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2});
  // The traversal pushes left then right, so it visits the right subtree first.
  const std::vector<double> qlo = {0.5, 0.5};
  const std::vector<double> qhi = {1.5, 1.5};
  CHECK(t.query_aabb(qlo, qhi) == std::vector<I64>{3, 2, 1, 0});
  // A query touching only the corner of cell 3 reports it (inclusive comparison).
  const std::vector<double> corner = {2.0, 2.0};
  const std::vector<double> beyond = {3.0, 3.0};
  CHECK(t.query_aabb(corner, beyond) == std::vector<I64>{3});
}

TEST_CASE("grid: BVH: queries match brute force over random boxes in two and three dimensions") {
  for (const std::size_t ndim : {std::size_t{1}, std::size_t{2}, std::size_t{3}}) {
    const Boxes b = random_boxes(700, ndim, 11 + ndim);
    const BVH t = build(b);
    CHECK(t.n_nodes() == 2 * 700 - 1);
    Lcg rng(99 + ndim);
    for (int q = 0; q < 200; ++q) {
      std::vector<double> qlo(ndim);
      std::vector<double> qhi(ndim);
      for (std::size_t d = 0; d < ndim; ++d) {
        qlo[d] = 11.0 * rng.next() - 0.5;
        qhi[d] = qlo[d] + 2.0 * rng.next();
      }
      const std::vector<I64> got = sorted(t.query_aabb(qlo, qhi));
      CHECK(got == brute_force_query(b, qlo, qhi));
      // Second implementation: for a non-empty query the separating-axis test is AABB::overlaps.
      const AABB query(qlo, qhi);
      std::vector<I64> via_aabb;
      for (std::size_t c = 0; c < 700; ++c) {
        const std::vector<double> clo(b.lo.begin() + static_cast<std::ptrdiff_t>(c * ndim),
                                      b.lo.begin() + static_cast<std::ptrdiff_t>((c + 1) * ndim));
        const std::vector<double> chi(b.hi.begin() + static_cast<std::ptrdiff_t>(c * ndim),
                                      b.hi.begin() + static_cast<std::ptrdiff_t>((c + 1) * ndim));
        if (AABB(clo, chi).overlaps(query)) {
          via_aabb.push_back(static_cast<I64>(c));
        }
      }
      CHECK(got == via_aabb);
      CHECK(sorted(t.query_aabb(query)) == got);
    }
  }
}

TEST_CASE("grid: BVH: the tree is a tight preorder tree over a permutation of the cells") {
  const Boxes b = random_boxes(257, 3, 5);
  const BVH t = build(b);
  const std::size_t n = static_cast<std::size_t>(t.n_nodes());
  REQUIRE(n == 513);
  std::set<I64> seen;
  for (std::size_t i = 0; i < n; ++i) {
    const bool leaf = t.node_cell()[i] >= 0;
    CHECK(leaf == (t.node_left()[i] == -1));
    CHECK(leaf == (t.node_right()[i] == -1));
    if (leaf) {
      seen.insert(t.node_cell()[i]);
      for (std::size_t d = 0; d < 3; ++d) {
        const auto c = static_cast<std::size_t>(t.node_cell()[i]);
        CHECK(t.node_lo()[i * 3 + d] == b.lo[c * 3 + d]);
        CHECK(t.node_hi()[i * 3 + d] == b.hi[c * 3 + d]);
      }
    } else {
      const auto l = static_cast<std::size_t>(t.node_left()[i]);
      const auto r = static_cast<std::size_t>(t.node_right()[i]);
      CHECK(l > i);
      CHECK(r > i);
      for (std::size_t d = 0; d < 3; ++d) {
        CHECK(t.node_lo()[i * 3 + d] == (std::min)(t.node_lo()[l * 3 + d], t.node_lo()[r * 3 + d]));
        CHECK(t.node_hi()[i * 3 + d] == (std::max)(t.node_hi()[l * 3 + d], t.node_hi()[r * 3 + d]));
      }
    }
  }
  CHECK(seen.size() == 257);
  CHECK(*seen.begin() == 0);
  CHECK(*seen.rbegin() == 256);
}

TEST_CASE("grid: BVH: cells with equal centroids keep their input order (the sort is stable)") {
  // 100 identical boxes: every centroid ties, so each split leaves the order as it was and the
  // leaves, read in preorder, are the cells in input order. An unstable sort scrambles them.
  const std::size_t n = 100;
  const Boxes b{std::vector<double>(2 * n, 0.0), std::vector<double>(2 * n, 1.0), {n, 2}};
  const BVH t = build(b);
  std::vector<I64> leaves;
  for (const I64 c : t.node_cell()) {
    if (c >= 0) {
      leaves.push_back(c);
    }
  }
  REQUIRE(leaves.size() == n);
  for (std::size_t i = 0; i < n; ++i) {
    CHECK(leaves[i] == static_cast<I64>(i));
  }
}

TEST_CASE("grid: BVH: the median split keeps the depth at ceil(log2 n) + 1") {
  const Boxes b = random_boxes(5000, 2, 3);
  const BVH t = build(b);
  const auto n = static_cast<std::size_t>(t.n_nodes());
  std::vector<int> depth(n, 0);
  depth[0] = 1;
  int max_depth = 1;
  for (std::size_t i = 0; i < n; ++i) {
    if (t.node_left()[i] >= 0) {
      depth[static_cast<std::size_t>(t.node_left()[i])] = depth[i] + 1;
      depth[static_cast<std::size_t>(t.node_right()[i])] = depth[i] + 1;
    }
    max_depth = (std::max)(max_depth, depth[i]);
  }
  CHECK(max_depth == 14);  // ceil(log2 5000) = 13, plus the root
}

TEST_CASE("grid: BVH: a reversed query interval is reported against a cell that contains it") {
  const Boxes b{{0.0}, {10.0}, {1, 1}};
  const BVH t = build(b);
  const std::vector<double> qlo = {5.0};
  const std::vector<double> qhi = {3.0};
  // The separating-axis test has no emptiness branch ...
  CHECK(t.query_aabb(qlo, qhi) == std::vector<I64>{0});
  // ... where the box says an empty box overlaps nothing.
  const std::vector<double> clo = {0.0};
  const std::vector<double> chi = {10.0};
  CHECK_FALSE(AABB(clo, chi).overlaps(AABB(qlo, qhi)));
}

TEST_CASE("grid: BVH: no cells give the empty tree, one cell a single leaf") {
  const std::vector<double> none;
  const std::vector<std::size_t> shape0 = {0, 2};
  const BVH empty = BVH::from_cell_bounds(none, none, shape0, shape0);
  CHECK(empty.ndim() == 2);
  CHECK(empty.n_cells() == 0);
  CHECK(empty.n_nodes() == 0);
  const std::vector<double> q = {0.0, 0.0};
  CHECK(empty.query_aabb(q, q).empty());
  CHECK(empty.query_aabb(AABB::unbounded(2)).empty());

  const Boxes one{{0.0, 0.0}, {1.0, 1.0}, {1, 2}};
  const BVH t = build(one);
  CHECK(t.n_nodes() == 1);
  CHECK(t.node_cell()[0] == 0);
  CHECK(t.query_aabb(AABB::unbounded(2)) == std::vector<I64>{0});
}

TEST_CASE("grid: BVH: a raw tree with no node queries to empty whatever n_cells says") {
  // The constructor accepts n_cells <= 0 with empty arrays (as the Python one does); the query
  // must test the node count, since reading node 0 of an empty array is out of bounds.
  const std::vector<double> none;
  const std::vector<I64> no_ids;
  const std::vector<std::size_t> shape0 = {0, 2};
  for (const I64 n_cells : {I64{0}, I64{-1}, I64{-5}}) {
    const BVH t(none, none, shape0, shape0, no_ids, no_ids, no_ids, n_cells);
    CHECK(t.n_nodes() == 0);
    const std::vector<double> q = {0.0, 0.0};
    CHECK(t.query_aabb(q, q).empty());
    CHECK(t.query_aabb(AABB::unbounded(2)).empty());
  }
}

TEST_CASE("grid: BVH: a copy shares the arrays") {
  const Boxes b = random_boxes(10, 2, 1);
  const BVH t = build(b);
  const BVH copy = t;  // NOLINT(performance-unnecessary-copy-initialization)
  CHECK(copy.rep() == t.rep());
  CHECK(copy.node_lo().data() == t.node_lo().data());
}

TEST_CASE("grid: BVH: the query refuses a box of another dimension") {
  const BVH t = build(random_boxes(10, 2, 1));
  const std::vector<double> lo3 = {0.0, 0.0, 0.0};
  CHECK_THROWS_WITH_AS(static_cast<void>(t.query_aabb(AABB(lo3, lo3))),
                       "BVH.query_aabb: aabb.ndim (3) must match self.ndim (2).",
                       pantr::value_error);
  const std::vector<double> one = {0.0};
  CHECK_THROWS_AS(static_cast<void>(t.query_aabb(one, one)), pantr::value_error);
}

TEST_CASE("grid: BVH: the corner query refuses a NaN corner, as the box does") {
  const BVH t = build(random_boxes(10, 2, 1));
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> ok = {0.0, 0.0};
  const std::vector<double> with_nan = {0.0, nan};
  const char* const msg = "BVH.query_aabb: query corners must not contain NaN.";
  CHECK_THROWS_WITH_AS(static_cast<void>(t.query_aabb(with_nan, ok)), msg, pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(t.query_aabb(ok, with_nan)), msg, pantr::value_error);
  // The empty tree refuses it too: the policy does not depend on what the tree holds.
  const std::vector<double> none;
  const std::vector<std::size_t> shape0 = {0, 2};
  const BVH empty = BVH::from_cell_bounds(none, none, shape0, shape0);
  CHECK_THROWS_WITH_AS(static_cast<void>(empty.query_aabb(with_nan, ok)), msg, pantr::value_error);
  // The box route refuses it earlier, in the AABB constructor.
  CHECK_THROWS_AS(static_cast<void>(AABB(with_nan, ok)), pantr::value_error);
}

TEST_CASE("grid: BVH: a shape whose product overflows size_t is refused with capacity_error") {
  const std::size_t huge = std::size_t{1} << 40;
  const std::vector<std::size_t> shape = {huge, huge};
  const std::vector<double> none;
  const std::vector<I64> no_ids;
  CHECK_THROWS_WITH_AS(static_cast<void>(BVH::from_cell_bounds(none, none, shape, shape)),
                       "a product of extents (1099511627776, 1099511627776) has more entries than "
                       "fit in memory",
                       pantr::capacity_error);
  CHECK_THROWS_AS(
      static_cast<void>(BVH(none, none, shape, shape, no_ids, no_ids, no_ids, std::int64_t{1})),
      pantr::capacity_error);
}

TEST_CASE("grid: BVH::from_cell_bounds: validation messages") {
  const std::vector<double> v4 = {0.0, 1.0, 2.0, 3.0};
  const std::vector<std::size_t> flat = {4};
  const std::vector<std::size_t> s22 = {2, 2};
  const std::vector<std::size_t> s31 = {3, 1};
  CHECK_THROWS_WITH_AS(static_cast<void>(BVH::from_cell_bounds(v4, v4, flat, flat)),
                       "cell_lo must be 2-D (n_cells, ndim); got shape (4,).", pantr::value_error);
  const std::vector<double> v3 = {0.0, 1.0, 2.0};
  CHECK_THROWS_WITH_AS(static_cast<void>(BVH::from_cell_bounds(v4, v3, s22, s31)),
                       "cell_hi shape (3, 1) must match cell_lo shape (2, 2).", pantr::value_error);
  const std::vector<double> none;
  const std::vector<std::size_t> s30 = {3, 0};
  CHECK_THROWS_WITH_AS(static_cast<void>(BVH::from_cell_bounds(none, none, s30, s30)),
                       "BVH ndim must be >= 1; got 0.", pantr::value_error);
  const std::vector<double> with_nan = {0.0, std::numeric_limits<double>::quiet_NaN()};
  const std::vector<double> ones = {1.0, 1.0};
  const std::vector<std::size_t> s12 = {1, 2};
  CHECK_THROWS_WITH_AS(
      static_cast<void>(BVH::from_cell_bounds(with_nan, ones, s12, s12)),
      "BVH.from_cell_bounds: cell_lo and cell_hi must contain only finite values; got NaN or Inf.",
      pantr::value_error);
  const std::vector<double> inf = {0.0, std::numeric_limits<double>::infinity()};
  CHECK_THROWS_AS(static_cast<void>(BVH::from_cell_bounds(ones, inf, s12, s12)),
                  pantr::value_error);
  const std::vector<double> lo = {1.0, 1.0};
  const std::vector<double> hi = {1.0, 0.5};
  CHECK_THROWS_WITH_AS(static_cast<void>(BVH::from_cell_bounds(lo, hi, s12, s12)),
                       "Every cell must satisfy cell_hi >= cell_lo on every axis; at least one "
                       "cell violates this.",
                       pantr::value_error);
  // A degenerate (hi == lo) cell is a valid box.
  const std::vector<double> flat_hi = {1.0, 1.0};
  CHECK(BVH::from_cell_bounds(lo, flat_hi, s12, s12).n_cells() == 1);
}

TEST_CASE("grid: BVH constructor: a tree round-trips through its raw arrays") {
  const Boxes b = random_boxes(40, 2, 8);
  const BVH t = build(b);
  const std::vector<std::size_t> shape = {static_cast<std::size_t>(t.n_nodes()), 2};
  const BVH copy(t.node_lo(), t.node_hi(), shape, shape, t.node_left(), t.node_right(),
                 t.node_cell(), t.n_cells());
  const std::vector<double> qlo = {2.0, 2.0};
  const std::vector<double> qhi = {6.0, 6.0};
  CHECK(copy.query_aabb(qlo, qhi) == t.query_aabb(qlo, qhi));
  CHECK(copy.rep() != t.rep());
}

TEST_CASE("grid: BVH constructor: shape and structure errors carry the Python messages") {
  const BVH t = build(random_boxes(3, 2, 8));  // 5 nodes
  const std::vector<double> lo(t.node_lo().begin(), t.node_lo().end());
  const std::vector<I64> left(t.node_left().begin(), t.node_left().end());
  const std::vector<I64> right(t.node_right().begin(), t.node_right().end());
  const std::vector<I64> cell(t.node_cell().begin(), t.node_cell().end());
  const std::vector<std::size_t> shape = {5, 2};
  const auto make = [&](const std::vector<std::size_t>& lo_shape,
                        const std::vector<std::size_t>& hi_shape, const std::vector<I64>& l,
                        const std::vector<I64>& r, const std::vector<I64>& c,
                        I64 n_cells) { return BVH(lo, lo, lo_shape, hi_shape, l, r, c, n_cells); };
  CHECK_THROWS_WITH_AS(static_cast<void>(make({10}, {10}, left, right, cell, 3)),
                       "node_lo must be 2-D (n_nodes, ndim); got shape (10,).", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(make(shape, {10, 1}, left, right, cell, 3)),
                       "node_hi shape (10, 1) must match node_lo shape (5, 2).",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(make({10, 0}, {10, 0}, left, right, cell, 3)),
                       "BVH ndim must be >= 1; got 0.", pantr::value_error);
  const std::vector<I64> short_left(left.begin(), left.end() - 1);
  CHECK_THROWS_WITH_AS(static_cast<void>(make(shape, shape, short_left, right, cell, 3)),
                       "node_left must have shape (5,); got (4,).", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(make(shape, shape, left, short_left, cell, 3)),
                       "node_right must have shape (5,); got (4,).", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(make(shape, shape, left, right, short_left, 3)),
                       "node_cell must have shape (5,); got (4,).", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(make(shape, shape, left, right, cell, 4)),
                       "BVH: n_cells=4 implies n_nodes=7; got node arrays with 5 rows.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(make(shape, shape, left, right, cell, 0)),
                       "BVH: n_cells=0 implies n_nodes=0; got node arrays with 5 rows.",
                       pantr::value_error);

  // A leaf whose node_cell says internal, and an internal node whose cell says leaf.
  std::vector<I64> bad_cell = cell;
  for (std::size_t i = 0; i < bad_cell.size(); ++i) {
    if (bad_cell[i] >= 0) {
      bad_cell[i] = -1;
      break;
    }
  }
  CHECK_THROWS_WITH_AS(static_cast<void>(make(shape, shape, left, right, bad_cell, 3)),
                       "BVH: node_cell and the child pointers disagree about which nodes are "
                       "leaves. A node is a leaf iff node_cell >= 0, and exactly then must "
                       "node_left and node_right both be -1.",
                       pantr::value_error);
  // A child index past the node array.
  std::vector<I64> far_left = left;
  far_left[0] = 5;
  CHECK_THROWS_WITH_AS(
      static_cast<void>(make(shape, shape, far_left, right, cell, 3)),
      "BVH: node_left contains values outside [0, 5) on internal nodes: range is [3, 5].",
      pantr::value_error);
  std::vector<I64> neg_right = right;
  neg_right[0] = -2;
  CHECK_THROWS_AS(static_cast<void>(make(shape, shape, left, neg_right, cell, 3)),
                  pantr::value_error);
  // A leaf cell id at or past n_cells.
  std::vector<I64> big_cell = cell;
  for (std::size_t i = 0; i < big_cell.size(); ++i) {
    if (big_cell[i] == 2) {
      big_cell[i] = 3;
    }
  }
  CHECK_THROWS_WITH_AS(static_cast<void>(make(shape, shape, left, right, big_cell, 3)),
                       "BVH: node_cell contains values outside [0, 3) on leaves: maximum is 3.",
                       pantr::value_error);
}

TEST_CASE("grid: BVH constructor: a tree of depth 128 fits the traversal stack, 129 does not") {
  // 127 internal nodes in a chain: 128 leaves, depth 128.
  const Chain ok = make_chain(127);
  const BVH t = from_chain(ok);
  CHECK(t.n_nodes() == 255);
  const std::vector<double> q = {0.5};
  const std::vector<I64> all = sorted(t.query_aabb(q, q));
  CHECK(all.size() == 128);
  CHECK(all.front() == 0);
  CHECK(all.back() == 127);

  const Chain deep = make_chain(128);  // depth 129
  CHECK_THROWS_WITH_AS(
      static_cast<void>(from_chain(deep)),
      "BVH: the given node arrays encode a tree of depth 129 or more, exceeding the stack depth "
      "128 that the traversal kernels allow. Build the tree more evenly, or use "
      "BVH.from_cell_bounds, whose median split keeps the depth logarithmic in the cell count.",
      pantr::value_error);
}

TEST_CASE("grid: BVH constructor: a node that is its own child is not a tree") {
  // One internal node whose children are itself: structurally in range, but not a tree.
  const std::vector<double> lo = {0.0};
  const std::vector<double> hi = {1.0};
  const std::vector<std::size_t> shape = {1, 1};
  const std::vector<I64> self = {0};
  const std::vector<I64> internal = {-1};
  CHECK_THROWS_WITH_AS(
      static_cast<void>(BVH(lo, hi, shape, shape, self, self, internal, 1)),
      "BVH: the node arrays do not encode a tree: every node must be reachable from the root "
      "exactly once.",
      pantr::value_error);
}

TEST_CASE("grid: BVH constructor: internal nodes sharing children are refused without a blow-up") {
  // 31 cells, 61 nodes: nodes 0..29 are internal with both children the next node, nodes 30..60
  // are leaves. Every index is in range and the depth is only 31, but there are 2^30 root-to-leaf
  // paths, so a walk that enumerates paths never returns; a walk that marks nodes refuses at the
  // first node popped a second time.
  constexpr I64 n_cells = 31;
  constexpr std::size_t n_nodes = 61;
  std::vector<I64> left(n_nodes, -1);
  std::vector<I64> right(n_nodes, -1);
  std::vector<I64> cell(n_nodes, -1);
  for (std::size_t i = 0; i < 30; ++i) {
    left[i] = static_cast<I64>(i) + 1;
    right[i] = static_cast<I64>(i) + 1;
  }
  for (std::size_t i = 30; i < n_nodes; ++i) {
    cell[i] = static_cast<I64>(i) - 30;
  }
  const std::vector<double> lo(n_nodes, 0.0);
  const std::vector<double> hi(n_nodes, 1.0);
  const std::vector<std::size_t> shape = {n_nodes, 1};
  CHECK_THROWS_WITH_AS(
      static_cast<void>(BVH(lo, hi, shape, shape, left, right, cell, n_cells)),
      "BVH: the node arrays do not encode a tree: every node must be reachable from the root "
      "exactly once.",
      pantr::value_error);
}

TEST_CASE("grid: BVH constructor: a shared child with an orphan node is not a tree") {
  // Nodes 0:(1,2), 1:(3,3); leaves 2, 3, 4 hold cells 0, 1, 2. Node 3 is popped twice and node 4
  // never, so the walk makes exactly n_nodes = 5 pops: a pop count alone accepts it, and a query
  // would then return cell 1 twice. Only the visited marks refuse it.
  constexpr I64 n_cells = 3;
  const std::vector<double> lo = {0.0, 0.0, 1.0, 2.0, 3.0};
  const std::vector<double> hi = {4.0, 4.0, 1.5, 2.5, 3.5};
  const std::vector<std::size_t> shape = {5, 1};
  const std::vector<I64> left = {1, 3, -1, -1, -1};
  const std::vector<I64> right = {2, 3, -1, -1, -1};
  const std::vector<I64> cell = {-1, -1, 0, 1, 2};
  CHECK_THROWS_WITH_AS(
      static_cast<void>(BVH(lo, hi, shape, shape, left, right, cell, n_cells)),
      "BVH: the node arrays do not encode a tree: every node must be reachable from the root "
      "exactly once.",
      pantr::value_error);
}

TEST_CASE("grid: BVH constructor: an unreachable subtree is not a tree") {
  // Control: 0:(1,2), 1:(3,4) is a tree over leaves 2, 3, 4. Re-pointing node 0 at leaves 2 and 3
  // leaves nodes 1 and 4 unreached. No node is popped twice, so only the final "every node was
  // reached" check refuses it.
  constexpr I64 n_cells = 3;
  const std::vector<double> lo = {0.0, 0.0, 1.0, 2.0, 3.0};
  const std::vector<double> hi = {4.0, 4.0, 1.5, 2.5, 3.5};
  const std::vector<std::size_t> shape = {5, 1};
  const std::vector<I64> cell = {-1, -1, 0, 1, 2};
  const std::vector<I64> left = {1, 3, -1, -1, -1};
  const std::vector<I64> right = {2, 4, -1, -1, -1};
  CHECK_NOTHROW(static_cast<void>(BVH(lo, hi, shape, shape, left, right, cell, n_cells)));
  const std::vector<I64> left_cut = {2, 3, -1, -1, -1};
  const std::vector<I64> right_cut = {3, 4, -1, -1, -1};
  CHECK_THROWS_AS(static_cast<void>(BVH(lo, hi, shape, shape, left_cut, right_cut, cell, n_cells)),
                  pantr::value_error);
}
