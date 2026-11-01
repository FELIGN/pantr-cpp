// HierarchicalGrid and its block helpers. Oracles, each from a route other than the code under
// test: cell boxes by brute force (locate against every box, facet neighbors against geometric
// adjacency of the boxes, boundary facets against the domain bounds), analytic bounds of a dyadic
// grid (exact in binary), counts and ids worked out by hand for small refinement patterns, the
// exported mesh against the set of distinct cell corners, and values recorded from the Python
// HierarchicalGrid of main (named where used; hierarchical_golden.hpp says how). Parallel results
// are compared at 1 and 8 threads.
#include <doctest/doctest.h>

#include "hierarchical_golden.hpp"

#include <pantr/geometry/aabb.hpp>
#include <pantr/grid/cell_index.hpp>
#include <pantr/grid/cell_quadrature.hpp>
#include <pantr/grid/detail/hier_blocks.hpp>
#include <pantr/grid/hierarchical_grid.hpp>
#include <pantr/grid/partition.hpp>
#include <pantr/parallel/parallel_for.hpp>
#include <pantr/parallel/work_model.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using pantr::geometry::AABB;
using pantr::grid::HierarchicalGrid;
using pantr::grid::TensorProductGrid;
using Bp = std::vector<std::vector<double>>;

namespace par = pantr::parallel;
namespace gd = pantr::grid::detail;

static_assert(pantr::grid::GridLike<HierarchicalGrid>);

namespace {

using I64 = std::int64_t;
using Ivec = std::vector<I64>;
using Levels = std::vector<std::vector<gd::Block>>;
constexpr double inf = std::numeric_limits<double>::infinity();

/// Restores the calling thread's parallelism setting on scope exit.
class ThreadsGuard {
 public:
  explicit ThreadsGuard(std::size_t n) : previous_(par::get_num_threads()) {
    par::set_num_threads(static_cast<std::int64_t>(n));
  }
  ThreadsGuard(const ThreadsGuard&) = delete;
  ThreadsGuard& operator=(const ThreadsGuard&) = delete;
  ~ThreadsGuard() { par::set_num_threads(static_cast<std::int64_t>(previous_)); }

 private:
  std::size_t previous_;
};

std::size_t eight_threads() { return (std::min)(std::size_t{8}, par::max_threads()); }

class Lcg {
 public:
  explicit Lcg(std::uint64_t seed) : state_(seed) {}
  double next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state_ >> 11) / 9007199254740992.0;
  }
  /// An integer in `[0, n)`.
  I64 below(I64 n) { return static_cast<I64>(next() * static_cast<double>(n)); }

 private:
  std::uint64_t state_;
};

HierarchicalGrid hg(Bp bp, Ivec factor) {
  return HierarchicalGrid(TensorProductGrid(std::move(bp)), std::move(factor));
}

/// Equal breakpoints i / cells on [0, 1] along every axis.
Bp uniform_bp(std::size_t cells, std::size_t ndim) {
  Bp bp(ndim);
  for (auto& axis : bp) {
    for (std::size_t i = 0; i <= cells; ++i) {
      axis.push_back(static_cast<double>(i) / static_cast<double>(cells));
    }
  }
  return bp;
}

struct Op {
  I64 level;
  Ivec lo;
  Ivec hi;
};

HierarchicalGrid apply(HierarchicalGrid g, const std::vector<Op>& ops) {
  for (const Op& op : ops) {
    g = g.refined(op.level, op.lo, op.hi);
  }
  return g;
}

Bp dyadic_root_2d() { return {{0.0, 0.25, 0.5, 1.0}, {0.0, 0.5, 1.0}}; }

/// A dyadic 2D grid with three levels: hanging faces across one and across two levels.
HierarchicalGrid grid2d() {
  return apply(hg(dyadic_root_2d(), {2, 2}),
               {{0, {1, 0}, {3, 2}}, {1, {2, 1}, {4, 3}}, {2, {5, 3}, {7, 5}}});
}

/// A dyadic 3D grid with an axis of factor 1, so a refinement does not split along it.
HierarchicalGrid grid3d() {
  return apply(hg({{0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}}, {2, 1, 2}),
               {{0, {0, 0, 0}, {1, 2, 1}}, {1, {0, 0, 0}, {1, 1, 1}}});
}

/// The golden grid of hierarchical_golden.hpp.
HierarchicalGrid golden_grid() {
  return apply(hg({{0.0, 0.3, 1.1, 1.7}, {-0.5, 0.1, 0.7}}, {2, 3}),
               {{0, {0, 0}, {2, 1}}, {1, {1, 0}, {3, 2}}, {2, {2, 0}, {4, 3}}});
}

/// Random refinements of `g`: regions of up to three cells per axis at random levels.
HierarchicalGrid random_refined(HierarchicalGrid g, std::uint64_t seed, int rounds) {
  Lcg rng(seed);
  for (int r = 0; r < rounds; ++r) {
    const I64 level = rng.below(g.max_level() + 1);
    const Ivec n = g.level_cells_per_axis(level);
    Ivec lo(n.size());
    Ivec hi(n.size());
    for (std::size_t k = 0; k < n.size(); ++k) {
      lo[k] = rng.below(n[k]);
      hi[k] = lo[k] + 1 + rng.below((std::min)(I64{3}, n[k] - lo[k]));
    }
    g = g.refined(level, lo, hi);
  }
  return g;
}

struct Box {
  std::vector<double> lo;
  std::vector<double> hi;
};

std::vector<Box> all_boxes(const HierarchicalGrid& g) {
  std::vector<Box> boxes;
  for (I64 cid = 0; cid < g.num_cells(); ++cid) {
    auto [lo, hi] = g.cell_bounds(cid);
    boxes.push_back({std::move(lo), std::move(hi)});
  }
  return boxes;
}

/// Every cell whose closed box contains `pt`, by brute force.
Ivec brute_containing(const std::vector<Box>& boxes, const std::vector<double>& pt) {
  Ivec out;
  for (std::size_t c = 0; c < boxes.size(); ++c) {
    bool inside = true;
    for (std::size_t d = 0; d < pt.size(); ++d) {
      inside = inside && boxes[c].lo[d] <= pt[d] && pt[d] <= boxes[c].hi[d];
    }
    if (inside) {
      out.push_back(static_cast<I64>(c));
    }
  }
  return out;
}

/// The cells sharing a facet (a positive-measure piece of it) with facet `(axis, side)` of `c`.
Ivec geometric_neighbors(const std::vector<Box>& boxes, I64 c, std::size_t axis, int side) {
  const Box& b = boxes[static_cast<std::size_t>(c)];
  Ivec out;
  for (std::size_t n = 0; n < boxes.size(); ++n) {
    if (static_cast<I64>(n) == c) {
      continue;
    }
    const Box& o = boxes[n];
    const bool plane = side == 0 ? o.hi[axis] == b.lo[axis] : o.lo[axis] == b.hi[axis];
    bool overlap = plane;
    for (std::size_t d = 0; d < b.lo.size() && overlap; ++d) {
      if (d != axis) {
        overlap = (std::max)(b.lo[d], o.lo[d]) < (std::min)(b.hi[d], o.hi[d]);
      }
    }
    if (overlap) {
      out.push_back(static_cast<I64>(n));
    }
  }
  return out;
}

Ivec sorted(Ivec v) {
  std::sort(v.begin(), v.end());
  return v;
}

/// Query points that stress the descent: every cell's corners, center and half-way points, then
/// random points over the domain enlarged on every side.
std::vector<double> stress_points(const HierarchicalGrid& g, const std::vector<Box>& boxes,
                                  std::uint64_t seed, std::size_t n_random) {
  const std::size_t nd = g.ndim();
  std::vector<double> pts;
  const auto push = [&](const std::vector<double>& p) {
    pts.insert(pts.end(), p.begin(), p.end());
  };
  for (const Box& b : boxes) {
    std::vector<double> center(nd);
    for (std::size_t d = 0; d < nd; ++d) {
      center[d] = b.lo[d] + (b.hi[d] - b.lo[d]) / 2.0;
    }
    push(b.lo);
    push(b.hi);
    push(center);
    for (std::size_t a = 0; a < nd; ++a) {
      std::vector<double> p = b.lo;
      p[a] = center[a];
      push(p);
      p = b.hi;
      p[a] = center[a];
      push(p);
    }
  }
  Lcg rng(seed);
  for (std::size_t i = 0; i < n_random; ++i) {
    for (std::size_t d = 0; d < nd; ++d) {
      const double lo = g.root().bounds()[2 * d] - 0.25;
      const double hi = g.root().bounds()[2 * d + 1] + 0.25;
      pts.push_back(lo + (hi - lo) * rng.next());
    }
  }
  return pts;
}

/// locate and locate_many against the boxes: a point of the domain is assigned to a cell whose
/// closed box contains it, a point outside every box to none.
void check_locate_against_brute_force(const HierarchicalGrid& g, std::uint64_t seed) {
  const std::vector<Box> boxes = all_boxes(g);
  const std::vector<double> pts = stress_points(g, boxes, seed, 200);
  const std::size_t nd = g.ndim();
  const std::size_t npts = pts.size() / nd;
  const auto got = g.locate_many(pts, npts);
  std::size_t inside = 0;
  for (std::size_t p = 0; p < npts; ++p) {
    const std::vector<double> pt(pts.begin() + static_cast<std::ptrdiff_t>(p * nd),
                                 pts.begin() + static_cast<std::ptrdiff_t>((p + 1) * nd));
    const Ivec candidates = brute_containing(boxes, pt);
    const I64 id = got.data()[p];
    const std::optional<I64> single = g.locate(pt);
    if (candidates.empty()) {
      CHECK(id == -1);
      CHECK_FALSE(single.has_value());
    } else {
      ++inside;
      if (std::find(candidates.begin(), candidates.end(), id) == candidates.end()) {
        FAIL_CHECK("point " << p << " located in " << id << ", which does not contain it");
      }
      CHECK(single == std::optional<I64>(id));
    }
  }
  CHECK(inside > npts / 2);
}

/// neighbor_across_facet and hanging_neighbors against the geometric adjacency of the boxes.
void check_adjacency(const HierarchicalGrid& g) {
  const std::vector<Box> boxes = all_boxes(g);
  const auto n_facets = static_cast<I64>(2 * g.ndim());
  for (I64 c = 0; c < g.num_cells(); ++c) {
    for (I64 f = 0; f < n_facets; ++f) {
      const Ivec expected = sorted(
          geometric_neighbors(boxes, c, static_cast<std::size_t>(f / 2), static_cast<int>(f % 2)));
      const Ivec hanging = g.hanging_neighbors(c, f);
      const std::optional<I64> one = g.neighbor_across_facet(c, f);
      if (sorted(hanging) != expected) {
        FAIL_CHECK("cell " << c << " facet " << f << ": hanging neighbors differ from the boxes");
      }
      if (expected.empty()) {
        CHECK_FALSE(one.has_value());
        CHECK(g.is_mesh_boundary_facet(c, f));
      } else {
        REQUIRE(one.has_value());
        CHECK(*one == hanging.front());
        CHECK(std::binary_search(expected.begin(), expected.end(), *one));
        CHECK_FALSE(g.is_mesh_boundary_facet(c, f));
      }
    }
  }
}

/// boundary_facets against the domain bounds: a facet is on the boundary iff it lies on a face of
/// the root domain.
void check_boundary_facets(const HierarchicalGrid& g) {
  const std::vector<Box> boxes = all_boxes(g);
  Ivec expected;
  for (std::size_t c = 0; c < boxes.size(); ++c) {
    for (std::size_t axis = 0; axis < g.ndim(); ++axis) {
      for (int side = 0; side < 2; ++side) {
        const double plane = side == 0 ? boxes[c].lo[axis] : boxes[c].hi[axis];
        if (plane == g.root().bounds()[2 * axis + static_cast<std::size_t>(side)]) {
          expected.push_back(static_cast<I64>(c));
          expected.push_back(static_cast<I64>(2 * axis) + side);
        }
      }
    }
  }
  const auto got = g.boundary_facets();
  REQUIRE(got.rank() == 2);
  CHECK(got.shape()[1] == 2);
  CHECK(got.shape()[0] * 2 == expected.size());
  CHECK(std::equal(got.data(), got.data() + got.size(), expected.begin(), expected.end()));
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Block helpers
// ---------------------------------------------------------------------------------------------

TEST_CASE("grid: hierarchical block helpers: size, membership and intersection") {
  CHECK(gd::block_size({2}, {5}) == 3);
  CHECK(gd::block_size({0, 0}, {3, 4}) == 12);
  CHECK(gd::block_size({1, 1}, {1, 9}) == 0);
  // A zero extent makes the product 0 however large the others are.
  const I64 big = I64{1} << 40;
  CHECK(gd::block_size({0, 0, 0}, {big, big, 0}) == 0);
  CHECK_THROWS_AS(static_cast<void>(gd::block_size({0, 0, 0}, {big, big, big})),
                  std::overflow_error);
  CHECK_FALSE(gd::try_block_size({0, 0, 0}, {big, big, big}).has_value());
  CHECK_FALSE(gd::try_block_size({0, 0}, {big, big << 23}).has_value());
  CHECK(gd::try_block_size({0, 0}, {big, I64{1} << 22}) == std::optional<I64>(I64{1} << 62));
}

TEST_CASE("grid: hierarchical block helpers: in_block is half-open on every axis") {
  CHECK(gd::in_block({1, 2}, {0, 0}, {3, 4}));
  CHECK(gd::in_block({0, 0}, {0, 0}, {2, 2}));
  CHECK_FALSE(gd::in_block({2, 2}, {0, 0}, {2, 2}));
  CHECK_FALSE(gd::in_block({1, 2}, {0, 0}, {3, 2}));
  CHECK_FALSE(gd::in_block({-1, 0}, {0, 0}, {3, 3}));
}

TEST_CASE("grid: hierarchical block helpers: rect_intersect") {
  const auto overlap = gd::rect_intersect({0, 0}, {3, 3}, {1, 1}, {4, 4});
  REQUIRE(overlap.has_value());
  CHECK(overlap->first == Ivec{1, 1});
  CHECK(overlap->second == Ivec{3, 3});
  CHECK_FALSE(gd::rect_intersect({0, 0}, {2, 2}, {3, 3}, {5, 5}).has_value());
  // Rectangles that only touch share no cell.
  CHECK_FALSE(gd::rect_intersect({0, 0}, {2, 2}, {2, 0}, {4, 2}).has_value());
}

TEST_CASE("grid: hierarchical block helpers: peel returns the slabs in axis order") {
  // Outer [0,5)^2 minus inner [1,4)^2: axis 0 gives the slab below, then the slab above, of the
  // full height; axis 1 then does the same for the remaining core of width 3.
  const auto frame = gd::peel({0, 0}, {5, 5}, {1, 1}, {4, 4});
  REQUIRE(frame.size() == 4);
  CHECK(frame[0] == gd::Block{{0, 0}, {1, 5}});
  CHECK(frame[1] == gd::Block{{4, 0}, {5, 5}});
  CHECK(frame[2] == gd::Block{{1, 0}, {4, 1}});
  CHECK(frame[3] == gd::Block{{1, 4}, {4, 5}});

  const auto line = gd::peel({0}, {10}, {2}, {7});
  REQUIRE(line.size() == 2);
  CHECK(line[0] == gd::Block{{0}, {2}});
  CHECK(line[1] == gd::Block{{7}, {10}});

  CHECK(gd::peel({0, 0}, {3, 3}, {0, 0}, {3, 3}).empty());
  // An inner block flush with three sides leaves one slab.
  const auto one = gd::peel({0, 0}, {3, 3}, {0, 0}, {3, 1});
  REQUIRE(one.size() == 1);
  CHECK(one[0] == gd::Block{{0, 1}, {3, 3}});
}

TEST_CASE("grid: hierarchical block helpers: try_merge joins aligned neighbors only") {
  CHECK(gd::try_merge({0}, {3}, {3}, {5}) == std::optional<gd::Block>(gd::Block{{0}, {5}}));
  CHECK(gd::try_merge({3}, {5}, {0}, {3}) == std::optional<gd::Block>(gd::Block{{0}, {5}}));
  CHECK_FALSE(gd::try_merge({0}, {2}, {3}, {5}).has_value());              // a gap
  CHECK_FALSE(gd::try_merge({0}, {3}, {2}, {5}).has_value());              // overlapping
  CHECK_FALSE(gd::try_merge({0, 0}, {2, 2}, {2, 1}, {4, 3}).has_value());  // misaligned
  CHECK_FALSE(gd::try_merge({0, 0}, {2, 2}, {2, 2}, {4, 4}).has_value());  // diagonal
  CHECK_FALSE(gd::try_merge({0, 0}, {2, 2}, {0, 0}, {2, 2}).has_value());  // equal
  CHECK(gd::try_merge({0, 0}, {2, 2}, {2, 0}, {4, 2}) ==
        std::optional<gd::Block>(gd::Block{{0, 0}, {4, 2}}));
}

TEST_CASE("grid: hierarchical block helpers: normalize_blocks merges and sorts") {
  CHECK(gd::normalize_blocks({{{0}, {3}}, {{3}, {7}}}) == std::vector<gd::Block>{{{0}, {7}}});
  CHECK(gd::normalize_blocks({{{4}, {5}}, {{0}, {2}}}) ==
        std::vector<gd::Block>{{{0}, {2}}, {{4}, {5}}});
  CHECK(gd::normalize_blocks({}).empty());

  // Four unit blocks tiling a 2x2 square collapse to one block whatever their order.
  std::vector<gd::Block> square = {
      {{0, 0}, {1, 1}}, {{0, 1}, {1, 2}}, {{1, 0}, {2, 1}}, {{1, 1}, {2, 2}}};
  std::sort(square.begin(), square.end());
  do {
    CHECK(gd::normalize_blocks(square) == std::vector<gd::Block>{{{0, 0}, {2, 2}}});
  } while (std::next_permutation(square.begin(), square.end()));

  // A pinwheel has no mergeable pair: the five blocks come back sorted and unmerged.
  const std::vector<gd::Block> pinwheel = {
      {{1, 1}, {2, 2}}, {{1, 2}, {3, 3}}, {{0, 0}, {2, 1}}, {{2, 0}, {3, 2}}, {{0, 1}, {1, 3}}};
  std::vector<gd::Block> expected = pinwheel;
  std::sort(expected.begin(), expected.end());
  CHECK(gd::normalize_blocks(pinwheel) == expected);
}

TEST_CASE("grid: hierarchical block helpers: mark_region_cells writes only the overlap") {
  // A 4x3 region at [2,6)x[1,4); the block [4,9)x[0,3) overlaps it in cells [4,6)x[1,3), i.e.
  // region rows 2-3 and columns 0-1.
  std::vector<std::uint8_t> mask(12, 0);
  gd::mark_region_cells(mask, {2, 1}, {6, 4}, {{4, 0}, {9, 3}});
  std::vector<std::uint8_t> expected(12, 0);
  for (const std::size_t row : {2U, 3U}) {
    for (const std::size_t col : {0U, 1U}) {
      expected[row * 3 + col] = 1;
    }
  }
  CHECK(mask == expected);

  std::vector<std::uint8_t> line(3, 0);
  gd::mark_region_cells(line, {0}, {3}, {{5}, {8}});
  CHECK(line == std::vector<std::uint8_t>{0, 0, 0});
  gd::mark_region_cells(line, {0}, {3}, {{1}, {8}});
  CHECK(line == std::vector<std::uint8_t>{0, 1, 1});

  // In a 2x2x2 region the block [1,2)x[0,2)x[1,2) is cells (1,0,1) and (1,1,1): offsets 5 and 7.
  std::vector<std::uint8_t> cube(8, 0);
  gd::mark_region_cells(cube, {0, 0, 0}, {2, 2, 2}, {{1, 0, 1}, {2, 2, 2}});
  CHECK(cube == std::vector<std::uint8_t>{0, 0, 0, 0, 0, 1, 0, 1});
}

TEST_CASE("grid: hierarchical block helpers: name_marked_cells") {
  CHECK(gd::max_named_cells == 6);
  CHECK(gd::max_diagnosed_cells == (I64{1} << 20));
  CHECK(gd::name_marked_cells(std::vector<std::uint8_t>{0, 1, 0, 1}, {4}, {10}) == "(11,), (13,)");
  CHECK(gd::name_marked_cells(std::vector<std::uint8_t>{0, 0, 1, 0}, {2, 2}, {3, 7}) == "(4, 7)");
  CHECK(gd::name_marked_cells(std::vector<std::uint8_t>{0, 0, 0}, {3}, {0}).empty());
  // Exactly the cap spells every cell and claims no remainder.
  CHECK(gd::name_marked_cells(std::vector<std::uint8_t>(6, 1), {6}, {0}) ==
        "(0,), (1,), (2,), (3,), (4,), (5,)");
  // One more is counted, not named.
  CHECK(gd::name_marked_cells(std::vector<std::uint8_t>(7, 1), {7}, {0}) ==
        "(0,), (1,), (2,), (3,), (4,), (5,) and 1 more");
  CHECK(gd::name_marked_cells(std::vector<std::uint8_t>(8, 1), {8}, {0}).ends_with("and 2 more"));
}

TEST_CASE("grid: hierarchical block helpers: checked powers refuse what int64 cannot hold") {
  CHECK(gd::try_pow(2, 62) == std::optional<I64>(I64{1} << 62));
  CHECK_FALSE(gd::try_pow(2, 63).has_value());
  CHECK(gd::try_pow(3, 39) == std::optional<I64>(4052555153018976267LL));
  CHECK_FALSE(gd::try_pow(3, 40).has_value());
  // A base of 1 is 1 at any exponent, and returns without looping that many times.
  CHECK(gd::try_pow(1, (std::numeric_limits<I64>::max)()) == std::optional<I64>(1));
  CHECK(gd::try_pow(7, 0) == std::optional<I64>(1));
  CHECK_FALSE(gd::try_pow(2, (std::numeric_limits<I64>::max)()).has_value());
  CHECK(gd::checked_pow(10, 18) == 1000000000000000000LL);
  CHECK_THROWS_AS(static_cast<void>(gd::checked_pow(10, 19)), std::overflow_error);
}

// ---------------------------------------------------------------------------------------------
// Construction, ids and bounds
// ---------------------------------------------------------------------------------------------

TEST_CASE("grid: hierarchical construction: every root cell is active at level 0, version 1") {
  const HierarchicalGrid g = hg(dyadic_root_2d(), {2, 3});
  CHECK(g.ndim() == 2);
  CHECK(g.num_cells() == 6);
  CHECK(g.max_level() == 0);
  CHECK(g.version() == 1);
  CHECK(Ivec(g.factor().begin(), g.factor().end()) == Ivec{2, 3});
  REQUIRE(g.blocks().size() == 1);
  CHECK(g.active_blocks(0) == std::vector<gd::Block>{{{0, 0}, {3, 2}}});
  CHECK(g.root().num_cells() == 6);
  for (I64 cid = 0; cid < 6; ++cid) {
    CHECK(g.cell_level(cid) == 0);
    CHECK(g.cell_multi_index(cid) == pantr::grid::flat_to_multi(cid, g.root().cells_per_axis()));
    CHECK(g.cell_bounds(cid) == g.root().cell_bounds(cid));
  }
  CHECK(g.child_cells(0).empty());
  CHECK(g.num_local_facets(0) == 4);
}

TEST_CASE("grid: hierarchical construction: factor errors carry the Python messages") {
  CHECK_THROWS_WITH_AS(hg(dyadic_root_2d(), {2, 0}),
                       "every factor entry must be >= 1 (1 = no subdivision on that axis); got "
                       "(2, 0).",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(hg(dyadic_root_2d(), {-1, 2}),
                       "every factor entry must be >= 1 (1 = no subdivision on that axis); got "
                       "(-1, 2).",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(hg({{0.0, 1.0}}, {0}),
                       "every factor entry must be >= 1 (1 = no subdivision on that axis); got "
                       "(0,).",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(hg(dyadic_root_2d(), {2, 2, 2}),
                       "factor must be a scalar or length-2 sequence; got length 3.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(hg(dyadic_root_2d(), {}),
                       "factor must be a scalar or length-2 sequence; got length 0.",
                       pantr::value_error);
}

TEST_CASE("grid: hierarchical ids: a 1D grid with the middle refined, worked out by hand") {
  // Root {0, 1/4, 1/2, 3/4, 1}, factor 2; refine cells 1 and 2. Level 0 keeps cells 0 and 3
  // (ids 0, 1); level 1 holds cells 2..5 (ids 2..5), each of width 1/8.
  const HierarchicalGrid g0 = hg({{0.0, 0.25, 0.5, 0.75, 1.0}}, {2});
  const HierarchicalGrid g = g0.refined(0, Ivec{1}, Ivec{3});
  CHECK(g.num_cells() == 6);
  CHECK(g.version() == 2);
  CHECK(g.max_level() == 1);
  CHECK(g.active_blocks(0) == std::vector<gd::Block>{{{0}, {1}}, {{3}, {4}}});
  CHECK(g.active_blocks(1) == std::vector<gd::Block>{{{2}, {6}}});
  const Ivec levels = {0, 0, 1, 1, 1, 1};
  const Ivec midx = {0, 3, 2, 3, 4, 5};
  const std::vector<std::pair<double, double>> bounds = {{0.0, 0.25},  {0.75, 1.0},  {0.25, 0.375},
                                                         {0.375, 0.5}, {0.5, 0.625}, {0.625, 0.75}};
  for (I64 cid = 0; cid < 6; ++cid) {
    const auto i = static_cast<std::size_t>(cid);
    CHECK(g.cell_level(cid) == levels[i]);
    CHECK(g.cell_multi_index(cid) == Ivec{midx[i]});
    const auto [lo, hi] = g.cell_bounds(cid);
    CHECK(lo == std::vector<double>{bounds[i].first});
    CHECK(hi == std::vector<double>{bounds[i].second});
    CHECK(g.cell_id(levels[i], Ivec{midx[i]}) == std::optional<I64>(cid));
    CHECK(g.is_active_leaf(levels[i], Ivec{midx[i]}));
  }
  // Positions that are not leaves: refined away, not yet created, out of range, wrong shape.
  CHECK_FALSE(g.is_active_leaf(0, Ivec{1}));
  CHECK_FALSE(g.is_active_leaf(1, Ivec{0}));
  CHECK_FALSE(g.is_active_leaf(2, Ivec{0}));
  CHECK_FALSE(g.is_active_leaf(-1, Ivec{0}));
  CHECK_FALSE(g.is_active_leaf(1, Ivec{-2}));
  CHECK_FALSE(g.is_active_leaf(1, Ivec{8}));
  CHECK_FALSE(g.is_active_leaf(0, Ivec{0, 0}));
  CHECK(g.level_cells_per_axis(0) == Ivec{4});
  CHECK(g.level_cells_per_axis(1) == Ivec{8});
  CHECK(g.level_cells_per_axis(4) == Ivec{64});  // a pure formula, past max_level
}

TEST_CASE("grid: hierarchical bounds: analytic on a dyadic grid, every cell, every level") {
  // A 4 x 4 unit square, factor 2, the middle 2 x 2 refined: 12 + 16 cells; a level-l cell at
  // index m spans [m, m + 1] * 2^-(2 + l) on every axis, exact in binary.
  const HierarchicalGrid g = hg(uniform_bp(4, 2), {2, 2}).refined(0, Ivec{1, 1}, Ivec{3, 3});
  CHECK(g.num_cells() == 28);
  I64 level0 = 0;
  for (I64 cid = 0; cid < g.num_cells(); ++cid) {
    const I64 level = g.cell_level(cid);
    level0 += level == 0 ? 1 : 0;
    const double h = std::ldexp(1.0, -static_cast<int>(2 + level));
    const Ivec m = g.cell_multi_index(cid);
    const auto [lo, hi] = g.cell_bounds(cid);
    for (std::size_t d = 0; d < 2; ++d) {
      CHECK(lo[d] == static_cast<double>(m[d]) * h);
      CHECK(hi[d] == static_cast<double>(m[d] + 1) * h);
    }
    CHECK(g.cell_id(level, m) == std::optional<I64>(cid));
  }
  CHECK(level0 == 12);
}

TEST_CASE("grid: hierarchical index errors carry the Python messages") {
  const HierarchicalGrid g = grid2d();
  const I64 n = g.num_cells();
  const std::string out_of_range =
      "cell id " + std::to_string(n) + " is out of range [0, " + std::to_string(n) + ").";
  CHECK_THROWS_WITH_AS(static_cast<void>(g.cell_bounds(n)), out_of_range.c_str(),
                       pantr::index_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.cell_level(n)), out_of_range.c_str(),
                       pantr::index_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.cell_multi_index(-1)),
                       ("cell id -1 is out of range [0, " + std::to_string(n) + ").").c_str(),
                       pantr::index_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.neighbor_across_facet(0, 4)),
                       "local facet id 4 is out of range [0, 4).", pantr::index_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.hanging_neighbors(n, 0)), out_of_range.c_str(),
                       pantr::index_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.active_blocks(4)), "level must be in [0, 3]; got 4.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.active_leaf_mask(-1)),
                       "level must be in [0, 3]; got -1.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.subdomain_mask(9)), "level must be in [0, 3]; got 9.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.level_cells_per_axis(-1)), "level must be >= 0; got -1.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.locate(std::vector<double>{0.5})),
                       "pt must have shape (2,); got (1,).", pantr::value_error);
  const std::vector<double> three = {0.1, 0.2, 0.3};
  CHECK_THROWS_WITH_AS(static_cast<void>(g.locate_many(three, 2)),
                       "locate_many: points holds 3 values, but (npts, ndim) = (2, 2) needs 4.",
                       pantr::value_error);
  CHECK(g.locate_many(std::vector<double>{}, 0).size() == 0);
}

// ---------------------------------------------------------------------------------------------
// Location
// ---------------------------------------------------------------------------------------------

TEST_CASE("grid: hierarchical locate: ties go to the lower root cell and the upper child") {
  // Root {0, 1, 2}, both cells refined by 2: level 1 holds [0, .5], [.5, 1], [1, 1.5], [1.5, 2]
  // as ids 0..3. A point on the interior root breakpoint 1 belongs to the lower root cell, then
  // (child boundaries going up) to that cell's last child; a point on 0.5 goes to the upper child.
  const HierarchicalGrid g = hg({{0.0, 1.0, 2.0}}, {2}).refined(0, Ivec{0}, Ivec{2});
  const auto locate = [&](double x) { return g.locate(std::vector<double>{x}); };
  CHECK(locate(0.0) == std::optional<I64>(0));
  CHECK(locate(0.25) == std::optional<I64>(0));
  CHECK(locate(0.5) == std::optional<I64>(1));
  CHECK(locate(1.0) == std::optional<I64>(1));
  CHECK(locate(1.25) == std::optional<I64>(2));
  CHECK(locate(1.5) == std::optional<I64>(3));
  CHECK(locate(2.0) == std::optional<I64>(3));
  CHECK_FALSE(locate(-1e-300).has_value());
  CHECK_FALSE(locate(std::nextafter(2.0, 3.0)).has_value());
  CHECK_FALSE(locate(std::nan("")).has_value());
  CHECK_FALSE(locate(inf).has_value());
  CHECK_FALSE(locate(-inf).has_value());
  const std::vector<double> xs = {0.0, 0.5, 1.0, 1.5, 2.0, -1.0, 3.0, std::nan(""), inf, -inf};
  const auto got = g.locate_many(xs, xs.size());
  CHECK(Ivec(got.data(), got.data() + got.size()) == Ivec{0, 1, 1, 3, 3, -1, -1, -1, -1, -1});
}

TEST_CASE("grid: hierarchical locate: agrees with brute force over the boxes") {
  check_locate_against_brute_force(grid2d(), 11);
  check_locate_against_brute_force(grid3d(), 12);
  // Random refinements of a dyadic 2D root (factors 2 and 4) and a 3D one with a factor of 1.
  check_locate_against_brute_force(
      random_refined(hg({{0.0, 0.5, 1.5, 2.0}, {-1.0, 0.0, 1.0}}, {2, 4}), 21, 12), 22);
  check_locate_against_brute_force(
      random_refined(hg({{0.0, 1.0}, {0.0, 0.5, 1.0}, {0.0, 2.0}}, {2, 1, 4}), 31, 10), 32);
}

TEST_CASE("grid: hierarchical locate: the center of every cell is found in a non-dyadic grid") {
  // Factors 3 and 5 on a root with non-dyadic breakpoints: cell boxes are inexact, but a cell's
  // center is far from every boundary, so it must be located in that very cell.
  const HierarchicalGrid g =
      random_refined(hg({{0.0, 0.3, 1.1, 1.7}, {-0.5, 0.1, 0.7}}, {3, 5}), 41, 14);
  CHECK(g.max_level() >= 2);
  for (I64 cid = 0; cid < g.num_cells(); ++cid) {
    const auto [lo, hi] = g.cell_bounds(cid);
    const std::vector<double> center = {lo[0] + (hi[0] - lo[0]) / 2.0,
                                        lo[1] + (hi[1] - lo[1]) / 2.0};
    if (g.locate(center) != std::optional<I64>(cid)) {
      FAIL_CHECK("center of cell " << cid << " located elsewhere");
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Facet adjacency and boundary
// ---------------------------------------------------------------------------------------------

TEST_CASE("grid: hierarchical adjacency: matches the geometric adjacency of the boxes") {
  check_adjacency(grid2d());
  check_adjacency(grid3d());
  check_adjacency(random_refined(hg({{0.0, 0.5, 1.5, 2.0}, {-1.0, 0.0, 1.0}}, {2, 4}), 51, 10));
  check_adjacency(hg({{0.0, 1.0, 2.0}}, {2}).refined(0, Ivec{0}, Ivec{1}));
}

TEST_CASE("grid: hierarchical adjacency: a coarse cell's finer neighbors in face order") {
  // Two unit squares side by side; the right one is refined 2x2. Ids: 0 is the left square; the
  // level-1 block [2,4) x [0,2) gives 1 = (2,0), 2 = (2,1), 3 = (3,0), 4 = (3,1).
  const HierarchicalGrid g =
      hg({{0.0, 1.0, 2.0}, {0.0, 1.0}}, {2, 2}).refined(0, Ivec{1, 0}, Ivec{2, 1});
  REQUIRE(g.num_cells() == 5);
  CHECK(g.hanging_neighbors(0, 1) == Ivec{1, 2});
  CHECK(g.neighbor_across_facet(0, 1) == std::optional<I64>(1));
  CHECK(g.hanging_neighbors(2, 0) == Ivec{0});  // a coarser neighbor is a single cell
  CHECK(g.neighbor_across_facet(2, 0) == std::optional<I64>(0));
  CHECK(g.hanging_neighbors(1, 3) == Ivec{2});  // same level
  CHECK(g.hanging_neighbors(3, 3) == Ivec{4});
  CHECK(g.hanging_neighbors(4, 1).empty());  // the domain's far face
  CHECK_FALSE(g.neighbor_across_facet(4, 1).has_value());
  CHECK(g.neighbors(0) == Ivec{1});
  CHECK(g.neighbors(1) == Ivec{0, 3, 2});
}

TEST_CASE("grid: hierarchical boundary_facets: exactly the facets on the domain boundary") {
  check_boundary_facets(hg({{0.0, 1.0, 2.0}}, {2}));
  check_boundary_facets(grid2d());
  check_boundary_facets(grid3d());
  check_boundary_facets(
      random_refined(hg({{0.0, 0.5, 1.5, 2.0}, {-1.0, 0.0, 1.0}}, {2, 4}), 61, 10));
  // A single root cell has all 2 ndim facets on the boundary.
  const auto single = hg({{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}, {2, 2, 2}).boundary_facets();
  CHECK(single.shape()[0] == 6);
  CHECK(single.data()[0] == 0);
  CHECK(single.data()[11] == 5);
}

// ---------------------------------------------------------------------------------------------
// Values recorded from the Python implementation
// ---------------------------------------------------------------------------------------------

TEST_CASE("grid: hierarchical golden (recorded from Python): bounds, location, adjacency") {
  namespace gold = pantr_golden;
  const HierarchicalGrid g = golden_grid();
  REQUIRE(g.num_cells() == gold::cells);
  CHECK(g.version() == gold::version);
  CHECK(g.max_level() == 3);
  CHECK(g.active_blocks(0) == std::vector<gd::Block>{{{0, 1}, {2, 2}}, {{2, 0}, {3, 2}}});
  CHECK(g.active_blocks(1) ==
        std::vector<gd::Block>{{{0, 0}, {1, 3}}, {{1, 2}, {3, 3}}, {{3, 0}, {4, 3}}});
  CHECK(g.active_blocks(2) == std::vector<gd::Block>{{{2, 3}, {4, 6}}, {{4, 0}, {6, 6}}});
  CHECK(g.active_blocks(3) == std::vector<gd::Block>{{{4, 0}, {8, 9}}});

  // The bounds are compared exactly, which pins the floating-point expression order.
  const auto [all_lo, all_hi] = g.collect_cell_bounds();
  for (I64 c = 0; c < gold::cells; ++c) {
    const auto i = static_cast<std::size_t>(c);
    const auto [lo, hi] = g.cell_bounds(c);
    CHECK(lo[0] == gold::bounds[i][0]);
    CHECK(lo[1] == gold::bounds[i][1]);
    CHECK(hi[0] == gold::bounds[i][2]);
    CHECK(hi[1] == gold::bounds[i][3]);
    CHECK(all_lo.data()[2 * i] == gold::bounds[i][0]);
    CHECK(all_lo.data()[2 * i + 1] == gold::bounds[i][1]);
    CHECK(all_hi.data()[2 * i] == gold::bounds[i][2]);
    CHECK(all_hi.data()[2 * i + 1] == gold::bounds[i][3]);
  }

  // Corners, centers and root-breakpoint points: ties and the descent's clamps.
  std::vector<double> pts;
  for (const auto& p : gold::points) {
    pts.push_back(p[0]);
    pts.push_back(p[1]);
  }
  const auto got = g.locate_many(pts, static_cast<std::size_t>(gold::n_points));
  for (int p = 0; p < gold::n_points; ++p) {
    const auto i = static_cast<std::size_t>(p);
    CHECK(got.data()[i] == gold::located[i]);
    CHECK(g.locate(std::vector<double>{gold::points[i][0], gold::points[i][1]}) ==
          std::optional<I64>(gold::located[i]));
  }

  const auto& hanging = gold::hanging();
  for (I64 c = 0; c < gold::cells; ++c) {
    for (I64 f = 0; f < 4; ++f) {
      const auto k = static_cast<std::size_t>(c * 4 + f);
      CHECK(g.hanging_neighbors(c, f) == hanging[k]);
      const std::optional<I64> one = g.neighbor_across_facet(c, f);
      CHECK(one.value_or(-1) == gold::neighbor[k]);
    }
  }

  const auto bf = g.boundary_facets();
  REQUIRE(bf.shape()[0] == std::size(gold::boundary));
  for (std::size_t r = 0; r < bf.shape()[0]; ++r) {
    CHECK(bf.data()[2 * r] == gold::boundary[r][0]);
    CHECK(bf.data()[2 * r + 1] == gold::boundary[r][1]);
  }
}

TEST_CASE("grid: hierarchical golden (recorded from Python): export_cells") {
  namespace gold = pantr_golden;
  const HierarchicalGrid g = golden_grid();
  const auto [points, conn] = g.export_cells();
  REQUIRE(points.shape()[0] == static_cast<std::size_t>(gold::n_vertices));
  CHECK(points.shape()[1] == 2);
  REQUIRE(conn.shape()[0] == static_cast<std::size_t>(gold::cells));
  CHECK(conn.shape()[1] == 4);
  for (std::size_t v = 0; v < points.shape()[0]; ++v) {
    CHECK(points.data()[2 * v] == gold::vertices[v][0]);
    CHECK(points.data()[2 * v + 1] == gold::vertices[v][1]);
  }
  for (std::size_t c = 0; c < conn.shape()[0]; ++c) {
    for (std::size_t j = 0; j < 4; ++j) {
      CHECK(conn.data()[4 * c + j] == gold::conn[c][j]);
    }
  }
}

TEST_CASE("grid: hierarchical golden (recorded from Python): restrict") {
  namespace gold = pantr_golden;
  const HierarchicalGrid g = golden_grid();
  const auto r = g.restrict(gold::restrict_ids);
  CHECK(r.grid.num_cells() == gold::restrict_cells);
  CHECK(r.grid.max_level() + 1 == gold::restrict_levels);
  CHECK(r.grid.version() == 1);
  CHECK(r.grid.root().breakpoints(0).size() == std::size(gold::restrict_bp0));
  CHECK(r.grid.root().breakpoints(1).size() == std::size(gold::restrict_bp1));
  for (std::size_t i = 0; i < std::size(gold::restrict_bp0); ++i) {
    CHECK(r.grid.root().breakpoints(0)[i] == gold::restrict_bp0[i]);
  }
  for (std::size_t i = 0; i < std::size(gold::restrict_bp1); ++i) {
    CHECK(r.grid.root().breakpoints(1)[i] == gold::restrict_bp1[i]);
  }
  REQUIRE(r.local_to_global.size() == std::size(gold::restrict_l2g));
  REQUIRE(r.in_subset.size() == std::size(gold::restrict_in_subset));
  for (std::size_t i = 0; i < r.local_to_global.size(); ++i) {
    CHECK(r.local_to_global[i] == gold::restrict_l2g[i]);
    CHECK(r.in_subset[i] == gold::restrict_in_subset[i]);
    // The sub-grid's root is a pure slice, so its cells are the original cells bit for bit.
    CHECK(r.grid.cell_bounds(static_cast<I64>(i)) == g.cell_bounds(r.local_to_global[i]));
  }
}

TEST_CASE("grid: hierarchical golden (recorded from Python): 3D hanging faces across levels") {
  namespace gold = pantr_golden;
  // A coarse cell facing a face of 2 x 2 fine cells, two of which are refined again: the leaves
  // on the shared plane come in depth-first order with two free axes.
  const HierarchicalGrid g = apply(hg({{0.0, 1.0, 2.0}, {0.0, 1.0}, {0.0, 1.0}}, {2, 2, 2}),
                                   {{0, {1, 0, 0}, {2, 1, 1}}, {1, {2, 0, 0}, {3, 2, 1}}});
  REQUIRE(g.num_cells() == gold::cells3);
  const auto& hanging = gold::hanging3();
  for (I64 c = 0; c < gold::cells3; ++c) {
    for (I64 f = 0; f < 6; ++f) {
      const Ivec& expected = hanging[static_cast<std::size_t>(c * 6 + f)];
      CHECK(g.hanging_neighbors(c, f) == expected);
      CHECK(g.neighbor_across_facet(c, f).value_or(-1) == (expected.empty() ? -1 : expected[0]));
    }
  }
  // The plane x = 1 is covered by leaves of two levels.
  CHECK(g.hanging_neighbors(0, 1).size() > 4);
  check_adjacency(g);
}

TEST_CASE("grid: hierarchical golden (recorded from Python): coarsened_cells block lists") {
  namespace gold = pantr_golden;
  // The order the parents are demoted in shapes the block lists, through the greedy merge, and so
  // the flat ids; the result must be the one main produces.
  const HierarchicalGrid g = hg(uniform_bp(4, 2), {2, 2})
                                 .refined_cells(Ivec{5, 6, 9, 10})
                                 .refined(1, Ivec{2, 2}, Ivec{6, 6})
                                 .refined(2, Ivec{4, 4}, Ivec{8, 8});
  REQUIRE(g.version() == gold::coarsen_version_before);
  const Ivec ids(std::begin(gold::coarsen_ids), std::end(gold::coarsen_ids));
  const HierarchicalGrid c = g.coarsened_cells(ids);
  CHECK(c.version() == gold::coarsen_version_after);
  CHECK(c.num_cells() == gold::coarsen_cells_after);
  CHECK(c.max_level() + 1 == gold::coarsen_levels_after);
  std::size_t at = 0;
  for (I64 level = 0; level < gold::coarsen_levels_after; ++level) {
    const I64 n_blocks = gold::coarsen_blocks[at++];
    const auto& blocks = c.active_blocks(level);
    REQUIRE(static_cast<I64>(blocks.size()) == n_blocks);
    for (const gd::Block& block : blocks) {
      const gd::Block expected{{gold::coarsen_blocks[at], gold::coarsen_blocks[at + 1]},
                               {gold::coarsen_blocks[at + 2], gold::coarsen_blocks[at + 3]}};
      CHECK(block == expected);
      at += 4;
    }
  }
}

TEST_CASE("grid: hierarchical golden (recorded from Python): a grid with 2^60 cells per level") {
  namespace gold = pantr_golden;
  // Main keeps block coordinates as Python integers, so cell_bounds and locate stay right at
  // depths where a 64-bit kernel would wrap.
  const I64 f = I64{1} << 20;
  const HierarchicalGrid g =
      apply(hg({{0.0, 0.7}, {0.0, 0.9}, {0.0, 0.5}}, {f, f, f}),
            {{0, {0, 0, 0}, {1, 1, 1}}, {1, {0, 0, 0}, {1, 1, 1}}, {2, {0, 0, 0}, {1, 1, 1}}});
  CHECK(g.num_cells() == gold::deep_cells);
  CHECK(g.max_level() == 3);
  CHECK(g.version() == 4);
  CHECK(g.level_cells_per_axis(3) == Ivec{I64{1} << 60, I64{1} << 60, I64{1} << 60});
  CHECK(g.cell_level(gold::deep_base3) == 3);
  CHECK(g.cell_level(gold::deep_base3 - 1) == 2);
  CHECK(g.cell_multi_index(gold::deep_base3 + f) == Ivec{0, 1, 0});
  CHECK(g.cell_multi_index(gold::deep_base3 + (I64{1} << 40)) == Ivec{1, 0, 0});
  for (std::size_t i = 0; i < std::size(gold::deep_ids); ++i) {
    const auto [lo, hi] = g.cell_bounds(gold::deep_ids[i]);
    for (std::size_t d = 0; d < 3; ++d) {
      CHECK(lo[d] == gold::deep_bounds[i][d]);
      CHECK(hi[d] == gold::deep_bounds[i][3 + d]);
    }
  }
  const std::vector<double> pt(std::begin(gold::deep_point), std::end(gold::deep_point));
  CHECK(g.locate(pt) == std::optional<I64>(gold::deep_point_cell));
  CHECK(g.locate_many(pt, 1).data()[0] == gold::deep_point_cell);
  // The face of the deepest block towards +x abuts the level-2 cell (1, 0, 0): a coarser
  // neighbor, one level up.
  const I64 last_in_x = gold::deep_base3 + (f - 1) * (I64{1} << 40);
  const std::optional<I64> nbr = g.neighbor_across_facet(last_in_x, 1);
  REQUIRE(nbr.has_value());
  CHECK(g.cell_level(*nbr) == 2);
  CHECK(g.cell_multi_index(*nbr) == Ivec{1, 0, 0});
  CHECK_FALSE(g.neighbor_across_facet(gold::deep_base3, 0).has_value());

  // One refinement further would need 2^80 cells along an axis.
  CHECK_THROWS_WITH_AS(
      static_cast<void>(g.refined(3, Ivec{0, 0, 0}, Ivec{1, 1, 1})),
      "refining level 3 would give level 4 more than 9223372036854775807 cells along axis 0.",
      std::overflow_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.level_cells_per_axis(4)),
                       "level 4 would have more than 9223372036854775807 cells along axis 0.",
                       std::overflow_error);
}

// ---------------------------------------------------------------------------------------------
// Refinement, coarsening, versions
// ---------------------------------------------------------------------------------------------

TEST_CASE("grid: hierarchical refine: a change bumps the version, a no-op is the same value") {
  const HierarchicalGrid g0 = hg({{0.0, 0.25, 0.5, 0.75, 1.0}}, {2});
  CHECK(g0.version() == 1);
  const HierarchicalGrid g1 = g0.refined(0, Ivec{1}, Ivec{3});
  CHECK(g1.version() == 2);
  CHECK_FALSE(g1.same_rep(g0));
  CHECK(g0.num_cells() == 4);  // the original value is unchanged

  // The region is refined already: nothing in it is an active level-0 cell.
  const HierarchicalGrid again = g1.refined(0, Ivec{1}, Ivec{3});
  CHECK(again.same_rep(g1));
  CHECK(again.version() == 2);
  // A region partly refined refines only its active part (union semantics): of [0, 3) only
  // cell 0 is a level-0 leaf, so level 1 gains cells 0 and 1 and meets [2, 6) in [0, 6).
  const HierarchicalGrid g2 = g1.refined(0, Ivec{0}, Ivec{3});
  CHECK(g2.version() == 3);
  CHECK(g2.num_cells() == 1 + 6);
  CHECK(g2.active_blocks(0) == std::vector<gd::Block>{{{3}, {4}}});
  CHECK(g2.active_blocks(1) == std::vector<gd::Block>{{{0}, {6}}});

  CHECK(g1.refined_cells(Ivec{}).same_rep(g1));
  CHECK_THROWS_WITH_AS(static_cast<void>(g1.refined(2, Ivec{0}, Ivec{1})),
                       "level must be in [0, 1]; got 2.", pantr::value_error);
}

TEST_CASE("grid: hierarchical refine: a region at the deepest level hidden by coarser leaves") {
  // Level-1 cells 0 and 1 sit inside the level-0 leaf 0, so refining them at level 1 finds no
  // active cell and changes nothing.
  const HierarchicalGrid g = hg({{0.0, 0.25, 0.5, 0.75, 1.0}}, {2}).refined(0, Ivec{2}, Ivec{3});
  REQUIRE(g.max_level() == 1);
  const HierarchicalGrid same = g.refined(1, Ivec{0}, Ivec{2});
  CHECK(same.same_rep(g));
  CHECK(same.version() == g.version());
  CHECK(g.refined(1, Ivec{4}, Ivec{5}).version() == g.version() + 1);
}

TEST_CASE("grid: hierarchical refine: argument errors carry the Python messages") {
  // Level 0 keeps cells 1..3 and level 1 holds cells 0 and 1: five cells.
  const HierarchicalGrid g = hg({{0.0, 0.25, 0.5, 0.75, 1.0}}, {2}).refined(0, Ivec{0}, Ivec{1});
  REQUIRE(g.num_cells() == 5);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.refined(5, Ivec{0}, Ivec{1})),
                       "level must be in [0, 1]; got 5.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.refined(-1, Ivec{0}, Ivec{1})),
                       "level must be in [0, 1]; got -1.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.refined(0, Ivec{0, 0}, Ivec{1})),
                       "lo and hi must have length 1; got 2 and 1.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.refined(0, Ivec{1}, Ivec{1})),
                       "lo must be strictly less than hi in every dimension; got lo=(1,), "
                       "hi=(1,).",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.refined(0, Ivec{0}, Ivec{9})),
                       "[lo, hi) out of bounds at level 0: axis 0 needs [0, 4), got [0, 9).",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.refined(0, Ivec{-1}, Ivec{1})),
                       "[lo, hi) out of bounds at level 0: axis 0 needs [0, 4), got [-1, 1).",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.refined_cells(Ivec{99})),
                       "cell id 99 is out of range [0, 5).", pantr::index_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.coarsened_cells(Ivec{0, -3})),
                       "cell id -3 is out of range [0, 5).", pantr::index_error);
  // Coarsening takes a level strictly below the deepest.
  CHECK_THROWS_WITH_AS(static_cast<void>(g.coarsened(1, Ivec{0}, Ivec{1})),
                       "level must be in [0, 1); got 1.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(hg({{0.0, 1.0}}, {2}).coarsened(0, Ivec{0}, Ivec{1})),
                       "level must be in [0, 0); got 0.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.coarsened(0, Ivec{0, 0}, Ivec{1})),
                       "lo and hi must have length 1; got 2 and 1.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.coarsened(0, Ivec{2}, Ivec{2})),
                       "lo must be strictly less than hi in every dimension; got lo=(2,), "
                       "hi=(2,).",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.coarsened(0, Ivec{0}, Ivec{9})),
                       "[lo, hi) out of bounds at level 0: axis 0 needs [0, 4), got [0, 9).",
                       pantr::value_error);
}

TEST_CASE("grid: hierarchical refined_cells: levels in turn, one version step per level") {
  // 1D, 4 root cells, factor 2. After refine(0, [0,1)): ids 0 = (0,1), 1 = (0,2), 2 = (0,3),
  // 3 = (1,0), 4 = (1,1). Naming cell 1 (level 0) and cell 3 (level 1) refines level 0 at [2,3)
  // and then level 1 at [0,1).
  const HierarchicalGrid g = hg({{0.0, 1.0, 2.0, 3.0, 4.0}}, {2}).refined(0, Ivec{0}, Ivec{1});
  REQUIRE(g.version() == 2);
  REQUIRE(g.cell_level(1) == 0);
  REQUIRE(g.cell_level(3) == 1);
  const HierarchicalGrid r = g.refined_cells(Ivec{3, 1, 3});
  CHECK(r.version() == 4);  // one step for level 0 and one for level 1
  CHECK(r.active_blocks(0) == std::vector<gd::Block>{{{1}, {2}}, {{3}, {4}}});
  CHECK(r.active_blocks(1) == std::vector<gd::Block>{{{1}, {2}}, {{4}, {6}}});
  CHECK(r.active_blocks(2) == std::vector<gd::Block>{{{0}, {2}}});
  CHECK(r.num_cells() == 2 + 3 + 2);
  // The bounding box of two cells at one level also refines the cells between them.
  const HierarchicalGrid wide = hg({{0.0, 1.0, 2.0, 3.0, 4.0}}, {2}).refined_cells(Ivec{0, 3});
  CHECK(wide.version() == 2);
  CHECK(wide.num_cells() == 8);
  CHECK(wide.max_level() == 1);
}

TEST_CASE("grid: hierarchical coarsen: a full sibling set comes back, and versions advance") {
  const HierarchicalGrid base = hg({{0.0, 1.0, 2.0, 3.0, 4.0}}, {2});
  const HierarchicalGrid refined = base.refined(0, Ivec{0}, Ivec{2});
  const HierarchicalGrid back = refined.coarsened(0, Ivec{0}, Ivec{2});
  CHECK(back.version() == 3);  // a coarsening is a change like any other
  CHECK(back.num_cells() == 4);
  CHECK(back.max_level() == 0);  // the emptied level is dropped
  CHECK(back.active_blocks(0) == base.active_blocks(0));
  // Refining always undoes a coarsening of the same box.
  CHECK(back.refined(0, Ivec{0}, Ivec{2}).blocks() == refined.blocks());
}

TEST_CASE("grid: hierarchical coarsened_cells: only complete parents, one version step each") {
  const HierarchicalGrid base = hg({{0.0, 1.0, 2.0, 3.0, 4.0}}, {2});
  // Parents 0 and 2 refined: level 0 keeps cells 1 and 3 (ids 0, 1) and level 1 holds [0,2) and
  // [4,6) (ids 2..5).
  const HierarchicalGrid g = base.refined(0, Ivec{0}, Ivec{1}).refined(0, Ivec{2}, Ivec{3});
  REQUIRE(g.version() == 3);
  REQUIRE(g.num_cells() == 6);
  Ivec level1;
  for (I64 c = 0; c < g.num_cells(); ++c) {
    if (g.cell_level(c) == 1) {
      level1.push_back(c);
    }
  }
  REQUIRE(level1 == Ivec{2, 3, 4, 5});

  const HierarchicalGrid all = g.coarsened_cells(level1);
  CHECK(all.version() == 5);  // two parents demoted: two steps
  CHECK(all.num_cells() == 4);
  CHECK(all.max_level() == 0);

  // A parent with only some of its children named is skipped, and so are level-0 ids.
  CHECK(g.coarsened_cells(Ivec{level1[0]}).same_rep(g));
  CHECK(g.coarsened_cells(Ivec{0, 1}).same_rep(g));
  CHECK(g.coarsened_cells(Ivec{}).same_rep(g));
  // One complete parent among partial ones: a single step, and the other parent is untouched.
  const HierarchicalGrid one = g.coarsened_cells(Ivec{level1[0], level1[1], level1[2]});
  CHECK(one.version() == 4);
  CHECK(one.num_cells() == 5);
  CHECK(one.is_active_leaf(0, Ivec{0}));
  CHECK_FALSE(one.is_active_leaf(0, Ivec{2}));
}

TEST_CASE("grid: hierarchical coarsened_cells: deepest parents first and no cascade") {
  // refine(0, [0,1)) then refine(1, [0,2)): level 1 is left empty and level 2 holds the four
  // grandchildren [0,4), ids 3..6 after the level-0 cells 1..3 (ids 0..2).
  const HierarchicalGrid g = hg({{0.0, 1.0, 2.0, 3.0, 4.0}}, {2})
                                 .refined(0, Ivec{0}, Ivec{1})
                                 .refined(1, Ivec{0}, Ivec{2});
  REQUIRE(g.max_level() == 2);
  REQUIRE(g.num_cells() == 7);
  REQUIRE(g.cell_level(3) == 2);
  // Ids 3 and 4 are the children of level-1 cell 0: that parent alone is demoted.
  const HierarchicalGrid down = g.coarsened_cells(Ivec{3, 4});
  CHECK(down.version() == g.version() + 1);
  CHECK(down.is_active_leaf(1, Ivec{0}));
  CHECK(down.is_active_leaf(2, Ivec{2}));
  CHECK_FALSE(down.is_active_leaf(2, Ivec{0}));
  // All four named: both level-1 parents are demoted in one call, and the cascade stops there,
  // since the level-1 cells it creates were not leaves when the ids were chosen.
  const HierarchicalGrid both = g.coarsened_cells(Ivec{3, 4, 5, 6});
  CHECK(both.version() == g.version() + 2);
  CHECK(both.max_level() == 1);
  CHECK(both.is_active_leaf(1, Ivec{0}));
  CHECK(both.is_active_leaf(1, Ivec{1}));
  CHECK(both.num_cells() == 5);
}

TEST_CASE("grid: hierarchical coarsen refusals: the messages (recorded from Python)") {
  const auto refuse = [](const HierarchicalGrid& g, I64 level, const Ivec& lo, const Ivec& hi) {
    try {
      static_cast<void>(g.coarsened(level, lo, hi));
    } catch (const pantr::value_error& e) {
      return std::string(e.what());
    }
    return std::string("NO ERROR");
  };

  // Leaves, exactly at the cap of named cells, and one over it.
  const HierarchicalGrid a = hg(uniform_bp(8, 1), {2}).refined(0, Ivec{0}, Ivec{1});
  CHECK(refuse(a, 0, {0}, {7}) ==
        "cannot coarsen [(0,), (7,)) at level 0: the region is not fully refined to exactly level "
        "1. Offending cells in level-0 indices: still active leaves at level 0, with no children "
        "to remove: (1,), (2,), (3,), (4,), (5,), (6,).");
  CHECK(refuse(a, 0, {0}, {8}) ==
        "cannot coarsen [(0,), (8,)) at level 0: the region is not fully refined to exactly level "
        "1. Offending cells in level-0 indices: still active leaves at level 0, with no children "
        "to remove: (1,), (2,), (3,), (4,), (5,), (6,) and 1 more.");

  // A 2D region: the first six leaves in row-major order, and a count of the rest.
  const HierarchicalGrid b = hg(uniform_bp(6, 2), {2, 2}).refined(0, Ivec{0, 0}, Ivec{1, 1});
  CHECK(refuse(b, 0, {0, 0}, {6, 6}) ==
        "cannot coarsen [(0, 0), (6, 6)) at level 0: the region is not fully refined to exactly "
        "level 1. Offending cells in level-0 indices: still active leaves at level 0, with no "
        "children to remove: (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 0) and 29 more.");

  // Hidden by a coarser leaf.
  const HierarchicalGrid c =
      hg(uniform_bp(4, 1), {2}).refined(0, Ivec{0}, Ivec{2}).refined(1, Ivec{0}, Ivec{2});
  CHECK(refuse(c, 1, {4}, {6}) ==
        "cannot coarsen [(4,), (6,)) at level 1: the region is not fully refined to exactly level "
        "2. Offending cells in level-1 indices: covered by a coarser active leaf and absent at "
        "level 1: (4,), (5,).");

  // All three reasons in one box, in the order leaves, refined beyond, hidden.
  const HierarchicalGrid d = hg(uniform_bp(8, 1), {2})
                                 .refined(0, Ivec{0}, Ivec{4})
                                 .refined(1, Ivec{0}, Ivec{4})
                                 .refined(2, Ivec{0}, Ivec{2});
  CHECK(refuse(d, 1, {0}, {12}) ==
        "cannot coarsen [(0,), (12,)) at level 1: the region is not fully refined to exactly "
        "level 2. Offending cells in level-1 indices: still active leaves at level 1, with no "
        "children to remove: (4,), (5,), (6,), (7,); refined beyond level 2: (0,); covered by a "
        "coarser active leaf and absent at level 1: (8,), (9,), (10,), (11,).");

  // Anisotropic factor: only the cell that owns the level-2 cells is named.
  const HierarchicalGrid e = hg(uniform_bp(2, 2), {2, 3})
                                 .refined(0, Ivec{0, 0}, Ivec{2, 2})
                                 .refined(1, Ivec{0, 0}, Ivec{1, 1});
  CHECK(refuse(e, 0, {0, 0}, {2, 2}) ==
        "cannot coarsen [(0, 0), (2, 2)) at level 0: the region is not fully refined to exactly "
        "level 1. Offending cells in level-0 indices: refined beyond level 1: (0, 0).");

  // A region far larger than the grid is described, not enumerated.
  HierarchicalGrid f = hg(uniform_bp(2, 1), {2});
  for (I64 level = 0; level < 21; ++level) {
    f = f.refined(level, Ivec{0}, Ivec{1});
  }
  const I64 region = f.level_cells_per_axis(20)[0];
  CHECK(region == 2097152);
  CHECK(refuse(f, 20, {0}, {region}) ==
        "cannot coarsen [(0,), (2097152,)) at level 20: the region is not fully refined to "
        "exactly level 21. The region spans 2097152 cells, too many to name.");

  // A box refined to exactly the next level is accepted.
  CHECK(refuse(hg(uniform_bp(4, 1), {2}).refined(0, Ivec{0}, Ivec{2}), 0, {0}, {2}) == "NO ERROR");
}

TEST_CASE("grid: hierarchical coarsen refusals: a region beyond int64 cells is still a refusal") {
  // Three axes of factor 2^20: level 2 has 2^40 cells per axis, so a box over all of it holds
  // 2^120 cells, more than int64 counts. Main prints the exact count; here it is a bound.
  const I64 f = I64{1} << 20;
  const HierarchicalGrid g =
      apply(hg({{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}, {f, f, f}),
            {{0, {0, 0, 0}, {1, 1, 1}}, {1, {0, 0, 0}, {1, 1, 1}}, {2, {0, 0, 0}, {1, 1, 1}}});
  const I64 n = I64{1} << 40;
  CHECK_THROWS_WITH_AS(
      static_cast<void>(g.coarsened(2, Ivec{0, 0, 0}, Ivec{n, n, n})),
      "cannot coarsen [(0, 0, 0), (1099511627776, 1099511627776, 1099511627776)) at level 2: the "
      "region is not fully refined to exactly level 3. The region spans more than "
      "9223372036854775807 cells, too many to name.",
      pantr::value_error);
}

// ---------------------------------------------------------------------------------------------
// Masks, export, restrict, from_blocks
// ---------------------------------------------------------------------------------------------

TEST_CASE("grid: hierarchical masks: leaf and subdomain masks against locate at cell centers") {
  const HierarchicalGrid g = grid2d();
  for (I64 level = 0; level <= g.max_level(); ++level) {
    const Ivec n = g.level_cells_per_axis(level);
    const auto leaf = g.active_leaf_mask(level);
    const auto sub = g.subdomain_mask(level);
    REQUIRE(leaf.rank() == 2);
    REQUIRE(sub.rank() == 2);
    CHECK(leaf.shape()[0] == static_cast<std::size_t>(n[0]));
    CHECK(leaf.shape()[1] == static_cast<std::size_t>(n[1]));
    // The level-l cell (i, j) is the (i mod 2^l, j mod 2^l)-th sub-cell of root cell
    // (i >> l, j >> l); its center is that sub-cell's, found without the grid's own formulas.
    const double scale = std::ldexp(1.0, -static_cast<int>(level));
    const std::vector<double>& bx = g.root().breakpoints()[0];
    const std::vector<double>& by = g.root().breakpoints()[1];
    for (I64 i = 0; i < n[0]; ++i) {
      for (I64 j = 0; j < n[1]; ++j) {
        const std::array<I64, 2> m = {i, j};
        const double fx = static_cast<double>(i % (I64{1} << level)) + 0.5;
        const double fy = static_cast<double>(j % (I64{1} << level)) + 0.5;
        const auto ri = static_cast<std::size_t>(i >> level);
        const auto rj = static_cast<std::size_t>(j >> level);
        const std::vector<double> center = {bx[ri] + (bx[ri + 1] - bx[ri]) * (fx * scale),
                                            by[rj] + (by[rj + 1] - by[rj]) * (fy * scale)};
        const std::optional<I64> cid = g.locate(center);
        REQUIRE(cid.has_value());
        const auto at = static_cast<std::size_t>(i * n[1] + j);
        // An active leaf of this level iff the leaf containing the center is of this level; in
        // the subdomain iff that leaf is of this level or finer.
        CHECK((leaf.data()[at] != 0) == (g.cell_level(*cid) == level));
        CHECK((sub.data()[at] != 0) == (g.cell_level(*cid) >= level));
        CHECK((leaf.data()[at] != 0) == g.is_active_leaf(level, m));
      }
    }
  }
  CHECK(g.subdomain_mask(0).data()[0] == 1);  // level 0 is the whole domain
}

TEST_CASE("grid: hierarchical export_cells: distinct corners, corner order and the hand count") {
  // The 2 x 2 root with one cell refined: 3 + 4 cells and 9 + 5 distinct vertices (the refined
  // cell adds four edge midpoints and a center).
  const HierarchicalGrid quad =
      hg({{0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}}, {2, 2}).refined(0, Ivec{0, 0}, Ivec{1, 1});
  const auto [sp, sc] = quad.export_cells();
  CHECK(quad.num_cells() == 7);
  CHECK(sp.shape()[0] == 14);
  CHECK(sc.shape()[0] == 7);
  CHECK(sc.shape()[1] == 4);

  for (const HierarchicalGrid& g : {quad, grid2d()}) {
    const auto [points, conn] = g.export_cells();
    const std::vector<Box> boxes = all_boxes(g);
    std::set<std::vector<double>> distinct;
    for (const Box& b : boxes) {
      for (const double x : {b.lo[0], b.hi[0]}) {
        for (const double y : {b.lo[1], b.hi[1]}) {
          distinct.insert({x, y});
        }
      }
    }
    CHECK(points.shape()[0] == distinct.size());
    // Rows sorted lexicographically, no row repeated: the set in iteration order.
    auto it = distinct.begin();
    for (std::size_t v = 0; v < points.shape()[0] && it != distinct.end(); ++v, ++it) {
      CHECK(points.data()[2 * v] == (*it)[0]);
      CHECK(points.data()[2 * v + 1] == (*it)[1]);
    }
    // Corner c takes the upper bound on axis k iff bit k of c is set (axis 0 least significant).
    for (std::size_t c = 0; c < boxes.size(); ++c) {
      for (std::size_t j = 0; j < 4; ++j) {
        const auto v = static_cast<std::size_t>(conn.data()[4 * c + j]);
        CHECK(points.data()[2 * v] == ((j & 1U) != 0 ? boxes[c].hi[0] : boxes[c].lo[0]));
        CHECK(points.data()[2 * v + 1] == ((j & 2U) != 0 ? boxes[c].hi[1] : boxes[c].lo[1]));
      }
    }
  }
}

TEST_CASE("grid: hierarchical export_cells: in 3D the corners follow the bit order") {
  const HierarchicalGrid g = grid3d();
  const auto [points, conn] = g.export_cells();
  const std::vector<Box> boxes = all_boxes(g);
  REQUIRE(conn.shape()[1] == 8);
  std::set<std::vector<double>> distinct;
  for (std::size_t c = 0; c < boxes.size(); ++c) {
    for (std::size_t j = 0; j < 8; ++j) {
      std::vector<double> corner(3);
      for (std::size_t k = 0; k < 3; ++k) {
        corner[k] = ((j >> k) & 1U) != 0 ? boxes[c].hi[k] : boxes[c].lo[k];
      }
      distinct.insert(corner);
      const auto v = static_cast<std::size_t>(conn.data()[8 * c + j]);
      CHECK(std::vector<double>(points.data() + 3 * v, points.data() + 3 * v + 3) == corner);
    }
  }
  CHECK(points.shape()[0] == distinct.size());
}

TEST_CASE("grid: hierarchical export_cells: non-dyadic corners within the stated bound") {
  // Each vertex is b + s * (w / steps) while cell_bounds builds the corner from other
  // expressions; the documented bound is 2 g2 |x| + 2 g4 |x - b| with g_m = m u / (1 - m u),
  // u = eps / 2, and |x - b| <= w, the width of the root cell holding the corner (here bounded by
  // the widest root cell of the axis).
  const HierarchicalGrid g =
      random_refined(hg({{0.0, 0.3, 1.1, 1.7}, {-0.5, 0.1, 0.7}}, {3, 5}), 71, 10);
  const auto [points, conn] = g.export_cells();
  const double u = std::numeric_limits<double>::epsilon() / 2.0;
  const auto gamma = [u](double m) { return m * u / (1.0 - m * u); };
  const std::vector<Box> boxes = all_boxes(g);
  for (std::size_t c = 0; c < boxes.size(); ++c) {
    for (std::size_t j = 0; j < 4; ++j) {
      const auto v = static_cast<std::size_t>(conn.data()[4 * c + j]);
      for (std::size_t k = 0; k < 2; ++k) {
        const double x = ((j >> k) & 1U) != 0 ? boxes[c].hi[k] : boxes[c].lo[k];
        const std::vector<double>& bp = g.root().breakpoints()[k];
        double width = 0.0;
        for (std::size_t i = 1; i < bp.size(); ++i) {
          width = (std::max)(width, bp[i] - bp[i - 1]);
        }
        const double bound = 2.0 * gamma(2.0) * std::abs(x) + 2.0 * gamma(4.0) * width;
        if (!(std::abs(points.data()[2 * v + k] - x) <= bound)) {
          FAIL_CHECK("cell " << c << " corner " << j << " axis " << k << " is off by more than "
                             << bound);
        }
      }
    }
  }
}

TEST_CASE("grid: hierarchical restrict: the window's cells are the original cells") {
  const HierarchicalGrid g = grid2d();
  const Ivec ids = {g.cell_id(3, Ivec{10, 6}).value_or(-1), 0, g.num_cells() - 1};
  REQUIRE(ids[0] >= 0);
  const auto r = g.restrict(ids);
  REQUIRE(r.local_to_global.size() == static_cast<std::size_t>(r.grid.num_cells()));
  CHECK(r.grid.version() == 1);
  // Brute force: every sub cell has the bounds of its global counterpart, no global cell is
  // listed twice, and exactly the requested ids are flagged.
  std::set<I64> seen;
  for (I64 i = 0; i < r.grid.num_cells(); ++i) {
    const I64 global = r.local_to_global[static_cast<std::size_t>(i)];
    CHECK(r.grid.cell_bounds(i) == g.cell_bounds(global));
    CHECK(r.grid.cell_level(i) == g.cell_level(global));
    CHECK(seen.insert(global).second);
    const bool requested = std::find(ids.begin(), ids.end(), global) != ids.end();
    CHECK((r.in_subset[static_cast<std::size_t>(i)] != 0) == requested);
  }
  for (const I64 id : ids) {
    CHECK(seen.count(id) == 1);
  }
  // The window is root-cell aligned, so the sub-grid tiles whole root cells: its cells' areas
  // (dyadic, so the sum is exact) add up to the area of the sub-grid's root domain.
  double area = 0.0;
  for (I64 i = 0; i < r.grid.num_cells(); ++i) {
    const auto [lo, hi] = r.grid.cell_bounds(i);
    area += (hi[0] - lo[0]) * (hi[1] - lo[1]);
  }
  const auto bounds = r.grid.root().bounds();
  CHECK(area == (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]));

  CHECK_THROWS_WITH_AS(static_cast<void>(g.restrict(Ivec{})),
                       "restrict: cell_ids must be non-empty.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.restrict(Ivec{0, g.num_cells()})),
                       ("restrict: cell id out of range [0, " + std::to_string(g.num_cells()) +
                        "); got [0, " + std::to_string(g.num_cells()) + "].")
                           .c_str(),
                       pantr::index_error);
}

TEST_CASE("grid: hierarchical from_blocks: a value round-trips through its blocks and version") {
  const HierarchicalGrid g = grid2d();
  Levels blocks = g.blocks();
  for (auto& level : blocks) {
    std::reverse(level.begin(), level.end());  // normalization restores the order
  }
  blocks.emplace_back();  // a trailing empty level is dropped
  const Ivec factor(g.factor().begin(), g.factor().end());
  const HierarchicalGrid h = HierarchicalGrid::from_blocks(g.root(), factor, blocks, 7);
  CHECK(h.version() == 7);
  CHECK(h.max_level() == g.max_level());
  CHECK(h.num_cells() == g.num_cells());
  CHECK(h.blocks() == g.blocks());
  for (I64 c = 0; c < g.num_cells(); ++c) {
    CHECK(h.cell_bounds(c) == g.cell_bounds(c));
  }
  // Adjacent blocks of one level are merged.
  const TensorProductGrid line = hg({{0.0, 1.0, 2.0, 3.0, 4.0}}, {2}).root();
  const HierarchicalGrid merged =
      HierarchicalGrid::from_blocks(line, {2}, Levels{{{{0}, {2}}, {{2}, {4}}}}, 1);
  CHECK(merged.active_blocks(0) == std::vector<gd::Block>{{{0}, {4}}});
  // A middle level with no cells is kept; only trailing ones go. One root cell, factor 2: level
  // 2 has 4 cells, ids 0..3, and level 1 is empty.
  const TensorProductGrid one = hg({{0.0, 1.0}}, {2}).root();
  const HierarchicalGrid hole =
      HierarchicalGrid::from_blocks(one, {2}, Levels{{}, {}, {{{0}, {4}}}}, 1);
  CHECK(hole.max_level() == 2);
  CHECK(hole.num_cells() == 4);
  CHECK(hole.cell_level(0) == 2);
  CHECK(hole.cell_bounds(3).first == std::vector<double>{0.75});
}

TEST_CASE("grid: hierarchical from_blocks: malformed input is refused") {
  const TensorProductGrid root = hg({{0.0, 1.0, 2.0, 3.0, 4.0}}, {2}).root();
  CHECK_THROWS_WITH_AS(static_cast<void>(HierarchicalGrid::from_blocks(root, {2}, Levels{}, 1)),
                       "from_blocks: blocks must hold at least one level.", pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(HierarchicalGrid::from_blocks(root, {2}, Levels{{{{2}, {2}}}}, 1)),
      "from_blocks: the block (2,), (2,) at level 0 is not a non-empty rectangle inside the "
      "level's index range (4,).",
      pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(HierarchicalGrid::from_blocks(root, {2}, Levels{{{{0}, {5}}}}, 1)),
      "from_blocks: the block (0,), (5,) at level 0 is not a non-empty rectangle inside the "
      "level's index range (4,).",
      pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(HierarchicalGrid::from_blocks(root, {2}, Levels{{{{0, 0}, {1, 1}}}}, 1)),
      "from_blocks: the block (0, 0), (1, 1) at level 0 is not a non-empty rectangle inside the "
      "level's index range (4,).",
      pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(HierarchicalGrid::from_blocks(root, {2, 2}, Levels{{{{0}, {4}}}}, 1)),
      "factor must be a scalar or length-1 sequence; got length 2.", pantr::value_error);
}

// ---------------------------------------------------------------------------------------------
// Overflow
// ---------------------------------------------------------------------------------------------

TEST_CASE("grid: hierarchical overflow: a level past int64 cells along an axis is refused") {
  // One root cell, factor 2^21: level l has 2^(21 l) cells, so level 3 would have 2^63.
  const I64 f = I64{1} << 21;
  HierarchicalGrid g = hg({{0.0, 1.0}}, {f});
  g = g.refined(0, Ivec{0}, Ivec{1});
  g = g.refined(1, Ivec{0}, Ivec{1});
  CHECK(g.max_level() == 2);
  CHECK(g.level_cells_per_axis(2) == Ivec{I64{1} << 42});
  CHECK_THROWS_WITH_AS(
      static_cast<void>(g.refined(2, Ivec{0}, Ivec{1})),
      "refining level 2 would give level 3 more than 9223372036854775807 cells along axis 0.",
      std::overflow_error);
  // The refusal leaves the value as it was, and a region with no active cell never refuses:
  // level-2 cells from 2^21 on sit under level-1 leaves.
  CHECK(g.max_level() == 2);
  CHECK(g.version() == 3);
  CHECK(g.refined(2, Ivec{f}, Ivec{f + 1}).same_rep(g));
  CHECK_THROWS_WITH_AS(static_cast<void>(g.level_cells_per_axis(3)),
                       "level 3 would have more than 9223372036854775807 cells along axis 0.",
                       std::overflow_error);
  // refined_cells refuses the same way.
  CHECK_THROWS_AS(static_cast<void>(g.refined_cells(Ivec{g.cell_id(2, Ivec{0}).value()})),
                  std::overflow_error);
}

TEST_CASE("grid: hierarchical overflow: a cell count past int64 is refused") {
  // Every axis count fits (2^32), but the children of one cell number 2^64.
  const I64 f = I64{1} << 32;
  const HierarchicalGrid g = hg({{0.0, 1.0}, {0.0, 1.0}}, {f, f});
  CHECK_THROWS_WITH_AS(static_cast<void>(g.refined(0, Ivec{0, 0}, Ivec{1, 1})),
                       "the hierarchical grid would have more than 9223372036854775807 cells.",
                       std::overflow_error);
  // One axis of factor 1 keeps it in range: 2^32 children.
  const HierarchicalGrid h =
      hg({{0.0, 1.0}, {0.0, 1.0}}, {f, 1}).refined(0, Ivec{0, 0}, Ivec{1, 1});
  CHECK(h.num_cells() == f);
}

TEST_CASE("grid: hierarchical factor 1: levels deepen without subdividing") {
  // Factor 1 on the only axis: the child of a cell is the same interval one level down, so a
  // hundred levels cost nothing, and level_cells_per_axis does not loop over the exponent. Cell
  // 0 stays at level 0 (id 0), cell 1 is carried down to level 100 (id 1).
  HierarchicalGrid g = hg({{0.0, 1.0, 2.0}}, {1});
  for (I64 level = 0; level < 100; ++level) {
    g = g.refined(level, Ivec{0}, Ivec{1});
  }
  CHECK(g.max_level() == 100);
  CHECK(g.num_cells() == 2);
  CHECK(g.version() == 101);
  CHECK(g.cell_level(0) == 0);
  CHECK(g.cell_level(1) == 100);
  CHECK(g.cell_bounds(0) == std::make_pair(std::vector<double>{1.0}, std::vector<double>{2.0}));
  CHECK(g.cell_bounds(1) == std::make_pair(std::vector<double>{0.0}, std::vector<double>{1.0}));
  CHECK(g.level_cells_per_axis((std::numeric_limits<I64>::max)()) == Ivec{2});
  CHECK(g.locate(std::vector<double>{0.5}) == std::optional<I64>(1));
  CHECK(g.locate(std::vector<double>{1.5}) == std::optional<I64>(0));
}

// ---------------------------------------------------------------------------------------------
// GridBase, the generic consumers, and threads
// ---------------------------------------------------------------------------------------------

TEST_CASE("grid: hierarchical GridBase: the BVH is lazy, shared by copies, fresh per new value") {
  const HierarchicalGrid g = grid2d();
  CHECK_FALSE(g.bvh_built());
  const std::vector<Box> boxes = all_boxes(g);
  Lcg rng(81);
  for (int q = 0; q < 40; ++q) {
    const double x = -0.25 + 1.5 * rng.next();
    const double y = -0.25 + 1.5 * rng.next();
    const double w = 0.4 * rng.next();
    const double h = 0.4 * rng.next();
    const AABB query(std::vector<double>{x, y}, std::vector<double>{x + w, y + h});
    Ivec expected;
    for (std::size_t c = 0; c < boxes.size(); ++c) {
      if (AABB(boxes[c].lo, boxes[c].hi).overlaps(query)) {
        expected.push_back(static_cast<I64>(c));
      }
    }
    CHECK(sorted(g.query_aabb(query)) == expected);
  }
  CHECK(g.bvh_built());
  const HierarchicalGrid copy = g;  // NOLINT(performance-unnecessary-copy-initialization)
  CHECK(copy.bvh_built());
  // A no-op refinement is the same value, so the built BVH is still there; a change is a new
  // value with a new, unbuilt one. Level-3 cells [10,14) x [6,10) are leaves.
  CHECK(g.refined(0, Ivec{1, 0}, Ivec{3, 2}).bvh_built());
  const HierarchicalGrid changed = g.refined(3, Ivec{10, 6}, Ivec{11, 7});
  CHECK_FALSE(changed.same_rep(g));
  CHECK(changed.max_level() == 4);
  CHECK_FALSE(changed.bvh_built());
  CHECK(g.bvh_built());
  CHECK_THROWS_WITH_AS(static_cast<void>(g.query_aabb(AABB::unbounded(3))),
                       "BVH.query_aabb: aabb.ndim (3) must match self.ndim (2).",
                       pantr::value_error);
}

TEST_CASE("grid: hierarchical GridBase: boxes, reference maps and facet bounds") {
  const HierarchicalGrid g = grid2d();
  for (I64 c = 0; c < g.num_cells(); ++c) {
    const auto [lo, hi] = g.cell_bounds(c);
    CHECK(g.cell_aabb(c) == AABB(lo, hi));
    const pantr::transform::AffineTransform map = g.reference_map(c);
    CHECK(std::vector<double>(map.offset().begin(), map.offset().end()) == lo);
    CHECK(map.matrix()[0] == hi[0] - lo[0]);
    CHECK(map.matrix()[3] == hi[1] - lo[1]);
  }
  const auto [flo, fhi] = g.local_facet_bounds(0, 1);  // axis 0, high side
  const auto [lo, hi] = g.cell_bounds(0);
  CHECK(flo == std::vector<double>{hi[0], lo[1]});
  CHECK(fhi == std::vector<double>{hi[0], hi[1]});
}

TEST_CASE("grid: hierarchical partition_grid and cell_quadrature instantiate for the grid") {
  namespace gold = pantr_golden;
  const HierarchicalGrid g = golden_grid();
  // The default backend on a hierarchical grid is the recursive bisection (the owner arrays are
  // recorded from Python).
  for (const auto& [n_parts, expected] :
       {std::pair<I64, const int*>{4, gold::owner4}, std::pair<I64, const int*>{5, gold::owner5}}) {
    const pantr::grid::Partition p = pantr::grid::partition_grid(g, n_parts);
    REQUIRE(p.cell_owner().size() == static_cast<std::size_t>(g.num_cells()));
    for (std::size_t c = 0; c < p.cell_owner().size(); ++c) {
      CHECK(p.cell_owner()[c] == expected[c]);
    }
    CHECK(p.n_parts() == n_parts);
  }
  const pantr::grid::Partition rcb =
      pantr::grid::partition_grid(g, 3, pantr::grid::PartitionBackend::rcb);
  CHECK(rcb.n_cells() == g.num_cells());
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pantr::grid::partition_grid(g, 3, pantr::grid::PartitionBackend::block)),
      "the 'block' backend requires a TensorProductGrid; got HierarchicalGrid.",
      pantr::value_error);

  // A three-point rule on the reference square; the weights sum to 1, so the total weight over
  // the cells is the domain's area.
  const std::vector<double> rule_points = {0.25, 0.25, 0.5, 0.75, 1.0, 0.0};
  const std::vector<double> rule_weights = {0.25, 0.5, 0.25};
  const HierarchicalGrid d = grid2d();
  const auto [pts, wts] = pantr::grid::cell_quadrature(d, rule_points, 3, 2, rule_weights);
  REQUIRE(pts.rank() == 3);
  CHECK(pts.shape()[0] == static_cast<std::size_t>(d.num_cells()));
  CHECK(pts.shape()[1] == 3);
  CHECK(pts.shape()[2] == 2);
  double area = 0.0;
  for (I64 c = 0; c < d.num_cells(); ++c) {
    const auto [lo, hi] = d.cell_bounds(c);
    const auto cc = static_cast<std::size_t>(c);
    for (std::size_t q = 0; q < 3; ++q) {
      area += wts.data()[cc * 3 + q];
      for (std::size_t k = 0; k < 2; ++k) {
        CHECK(pts.data()[(cc * 3 + q) * 2 + k] == lo[k] + (hi[k] - lo[k]) * rule_points[q * 2 + k]);
      }
    }
  }
  // The weights are w * volume, so `area` is the sum of the cells' volumes: the unit square's.
  CHECK(area == 1.0);
  // A selection (with a repeat, out of order) is the matching rows.
  const Ivec pick = {5, 2, 5};
  const auto [spts, swts] =
      pantr::grid::cell_quadrature(d, rule_points, 3, 2, rule_weights, std::span<const I64>(pick));
  REQUIRE(spts.shape()[0] == 3);
  for (std::size_t s = 0; s < 3; ++s) {
    for (std::size_t q = 0; q < 3; ++q) {
      CHECK(swts.data()[s * 3 + q] == wts.data()[static_cast<std::size_t>(pick[s]) * 3 + q]);
    }
  }
}

TEST_CASE("grid: hierarchical threads: locate_many and collect_cell_bounds at 1 and 8 threads") {
  // A 256 x 256 dyadic root with its middle 192 x 192 refined: 65536 - 36864 + 147456 cells, above
  // the parallel threshold even at one operation per cell, so the pool runs several chunks.
  const std::size_t n_root = 256;
  const HierarchicalGrid g =
      hg(uniform_bp(n_root, 2), {2, 2}).refined(0, Ivec{32, 32}, Ivec{224, 224});
  REQUIRE(g.num_cells() == 176128);
  REQUIRE(par::worth_parallel(static_cast<std::size_t>(g.num_cells()), par::WorkEstimate{1.0}));

  const auto collect = [&](std::size_t threads) {
    const ThreadsGuard guard(threads);
    return g.collect_cell_bounds();
  };
  const auto [lo1, hi1] = collect(1);
  const auto [lo8, hi8] = collect(eight_threads());
  REQUIRE(lo1.size() == lo8.size());
  // No reduction is involved and every cell's bounds are a function of that cell alone, so the
  // arrays agree bit for bit whatever the chunking.
  CHECK(std::memcmp(lo1.data(), lo8.data(), lo1.size() * sizeof(double)) == 0);
  CHECK(std::memcmp(hi1.data(), hi8.data(), hi1.size() * sizeof(double)) == 0);
  // And they are the exact values: the root breakpoints i / 256 and every subdivision are dyadic,
  // so a level-l cell at index m spans [m, m + 1] * 2^-(8 + l) with no rounding anywhere.
  for (I64 c = 0; c < g.num_cells(); ++c) {
    const double h = std::ldexp(1.0, -static_cast<int>(8 + g.cell_level(c)));
    const Ivec m = g.cell_multi_index(c);
    const auto i = static_cast<std::size_t>(c);
    if (lo8.data()[2 * i] != static_cast<double>(m[0]) * h ||
        lo8.data()[2 * i + 1] != static_cast<double>(m[1]) * h ||
        hi8.data()[2 * i] != static_cast<double>(m[0] + 1) * h ||
        hi8.data()[2 * i + 1] != static_cast<double>(m[1] + 1) * h) {
      FAIL_CHECK("cell " << c << " has the wrong bounds");
      break;
    }
  }

  // Points: every cell's center (so the answer is known), then three that are outside or not
  // finite.
  const auto n_centers = static_cast<std::size_t>(g.num_cells());
  std::vector<double> pts;
  pts.reserve(2 * (n_centers + 3));
  for (std::size_t c = 0; c < n_centers; ++c) {
    pts.push_back(lo8.data()[2 * c] + (hi8.data()[2 * c] - lo8.data()[2 * c]) / 2.0);
    pts.push_back(lo8.data()[2 * c + 1] + (hi8.data()[2 * c + 1] - lo8.data()[2 * c + 1]) / 2.0);
  }
  for (const double v : {std::nan(""), inf, -0.5}) {
    pts.push_back(v);
    pts.push_back(0.5);
  }
  const std::size_t npts = pts.size() / 2;
  REQUIRE(par::worth_parallel(npts, par::WorkEstimate{1.0}));
  const auto locate = [&](std::size_t threads) {
    const ThreadsGuard guard(threads);
    return g.locate_many(pts, npts);
  };
  const auto serial = locate(1);
  const auto parallel = locate(eight_threads());
  CHECK(std::equal(serial.data(), serial.data() + npts, parallel.data()));
  for (std::size_t c = 0; c < n_centers; ++c) {
    if (serial.data()[c] != static_cast<I64>(c)) {
      FAIL_CHECK("center of cell " << c << " located in " << serial.data()[c]);
      break;
    }
  }
  CHECK(serial.data()[n_centers] == -1);
  CHECK(serial.data()[n_centers + 1] == -1);
  CHECK(serial.data()[n_centers + 2] == -1);
}
