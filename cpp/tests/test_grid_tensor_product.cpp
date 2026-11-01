// TensorProductGrid and the GridBase defaults. Oracles: cell boxes enumerated by nested loops over
// the breakpoints (dyadic values, exact in binary), brute force over every cell box (locate,
// neighbors, boundary facets, box queries), the second implementation AABB::overlaps (queries),
// and a geometric reading of the domain boundary (boundary facets), none of which uses the grid's
// own index arithmetic. Parallel results are compared at 1 and 8 threads for exact equality.
#include <doctest/doctest.h>

#include <pantr/geometry/aabb.hpp>
#include <pantr/grid/tensor_product_grid.hpp>
#include <pantr/parallel/parallel_for.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

using pantr::geometry::AABB;
using pantr::grid::TensorProductGrid;
using Bp = std::vector<std::vector<double>>;

namespace par = pantr::parallel;

namespace {

using I64 = std::int64_t;
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

class Lcg {
 public:
  explicit Lcg(std::uint64_t seed) : state_(seed) {}
  double next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state_ >> 11) / 9007199254740992.0;
  }

 private:
  std::uint64_t state_;
};

/// Build a grid from braced breakpoints (a named type keeps the braces from reading as a copy).
TensorProductGrid tpg(Bp bp) { return TensorProductGrid(std::move(bp)); }

/// Dyadic breakpoints: exact in binary, strictly increasing, non-uniform.
Bp dyadic3() { return {{0.0, 0.5, 1.0, 2.0, 2.25}, {-1.0, 0.0, 0.75}, {4.0, 5.0, 5.5, 8.0}}; }

/// Eight threads, or the pool capacity if that is lower.
std::size_t eight_threads() { return (std::min)(std::size_t{8}, par::max_threads()); }

struct Box {
  std::vector<double> lo;
  std::vector<double> hi;
};

/// Every cell box in C-order, by nested loops over the breakpoints.
std::vector<Box> enumerate_cells(const Bp& bp) {
  std::vector<Box> cells;
  const std::size_t nd = bp.size();
  std::vector<std::size_t> idx(nd, 0);
  while (true) {
    Box b;
    for (std::size_t d = 0; d < nd; ++d) {
      b.lo.push_back(bp[d][idx[d]]);
      b.hi.push_back(bp[d][idx[d] + 1]);
    }
    cells.push_back(b);
    std::size_t d = nd;
    while (d-- > 0) {
      if (++idx[d] + 1 < bp[d].size()) {
        break;
      }
      idx[d] = 0;
      if (d == 0) {
        return cells;
      }
    }
  }
}

/// The lowest id among the cells whose closed box contains `pt`, by brute force; -1 if none.
I64 brute_locate(const std::vector<Box>& cells, const std::vector<double>& pt) {
  for (std::size_t c = 0; c < cells.size(); ++c) {
    bool inside = true;
    for (std::size_t d = 0; d < pt.size(); ++d) {
      inside = inside && cells[c].lo[d] <= pt[d] && pt[d] <= cells[c].hi[d];
    }
    if (inside) {
      return static_cast<I64>(c);
    }
  }
  return -1;
}

}  // namespace

TEST_CASE("grid: TensorProductGrid: construction refuses malformed breakpoints") {
  CHECK_THROWS_WITH_AS(tpg(Bp{}), "TensorProductGrid needs at least one axis; got 0.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(tpg({{0.0, 1.0}, {2.0}}),
                       "breakpoints[1] must have at least 2 entries (>= 1 cell); got shape (1,).",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(tpg({{}}),
                       "breakpoints[0] must have at least 2 entries (>= 1 cell); got shape (0,).",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(tpg({{0.0, std::nan(""), 2.0}}),
                       "breakpoints[0] must contain only finite values.", pantr::value_error);
  CHECK_THROWS_WITH_AS(tpg({{0.0, 1.0}, {0.0, inf}}),
                       "breakpoints[1] must contain only finite values.", pantr::value_error);
  CHECK_THROWS_WITH_AS(tpg({{0.0, 1.0, 1.0}}), "breakpoints[0] must be strictly increasing.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(tpg({{1.0, 0.0}}), "breakpoints[0] must be strictly increasing.",
                       pantr::value_error);
  // The checks run axis by axis: axis 0's error wins over axis 1's.
  CHECK_THROWS_WITH_AS(tpg({{1.0, 1.0}, {1.0}}), "breakpoints[0] must be strictly increasing.",
                       pantr::value_error);
}

TEST_CASE("grid: TensorProductGrid: a cell count beyond int64 is refused with overflow_error") {
  // Five axes of 8192 = 2^13 cells give 2^65 cells; every axis is valid on its own, and the
  // largest stride, 2^52, still fits, so only the total overflows.
  std::vector<double> axis(8193);
  for (std::size_t i = 0; i < axis.size(); ++i) {
    axis[i] = static_cast<double>(i);
  }
  CHECK_THROWS_AS(static_cast<void>(tpg(Bp(5, axis))), std::overflow_error);
  // Four such axes (2^52 cells) are fine.
  CHECK(tpg(Bp(4, axis)).num_cells() == I64{1} << 52);
}

TEST_CASE("grid: TensorProductGrid: locate_work is one binary search per axis plus four") {
  // Oracle: ceil(log2(cells + 2)) comparisons per axis, written as a table (cells 1 -> 2, 2 -> 2,
  // 3 -> 3, 5 -> 3, 6 -> 3, 7 -> 4), plus about four operations per axis.
  CHECK(tpg(Bp{{0.0, 1.0}}).locate_work().ops_per_item == doctest::Approx(2.0 + 4.0));
  CHECK(tpg(Bp{{0.0, 1.0, 2.0, 3.0}, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}}).locate_work().ops_per_item ==
        doctest::Approx((3.0 + 4.0) + (3.0 + 4.0)));
  CHECK(tpg(Bp{{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}}).locate_work().ops_per_item ==
        doctest::Approx(4.0 + 4.0));
}

TEST_CASE("grid: TensorProductGrid: metadata of a three-dimensional grid") {
  const TensorProductGrid g(dyadic3());
  CHECK(g.ndim() == 3);
  CHECK(g.num_cells() == 4 * 2 * 3);
  CHECK(std::vector<I64>(g.cells_per_axis().begin(), g.cells_per_axis().end()) ==
        std::vector<I64>{4, 2, 3});
  CHECK(std::vector<I64>(g.strides().begin(), g.strides().end()) == std::vector<I64>{6, 3, 1});
  CHECK(std::vector<double>(g.bounds().begin(), g.bounds().end()) ==
        std::vector<double>{0.0, 2.25, -1.0, 0.75, 4.0, 8.0});
  CHECK(g.breakpoints(2).size() == 4);
  CHECK(g.breakpoints().size() == 3);
  CHECK_FALSE(g.is_uniform());
  CHECK_THROWS_WITH_AS(static_cast<void>(g.breakpoints(3)), "axis 3 is out of range [0, 3).",
                       pantr::index_error);
}

TEST_CASE(
    "grid: TensorProductGrid: uniform spacing is judged by an absolute tolerance on the spread") {
  CHECK(tpg({{0.0, 1.0, 2.0, 3.0}, {5.0, 6.0}}).is_uniform());
  // Two breakpoints are uniform whatever the width; spacings 1 and 3 are not.
  CHECK(tpg({{0.0, 1e6}}).is_uniform());
  CHECK_FALSE(tpg({{0.0, 1.0, 4.0}}).is_uniform());
  // Spread 5e-11 is below the absolute tolerance 1e-10; spread 2e-10 is not.
  CHECK(tpg({{0.0, 1.0, 2.00000000005}}).is_uniform());
  CHECK_FALSE(tpg({{0.0, 1.0, 2.0000000002}}).is_uniform());
  // One non-uniform axis makes the grid non-uniform.
  CHECK_FALSE(tpg({{0.0, 1.0, 2.0}, {0.0, 1.0, 5.0}}).is_uniform());
}

TEST_CASE(
    "grid: TensorProductGrid: cell bounds, multi-indices and the collected corners are the "
    "enumerated boxes") {
  const auto bp = dyadic3();
  const TensorProductGrid g(bp);
  const std::vector<Box> cells = enumerate_cells(bp);
  REQUIRE(static_cast<I64>(cells.size()) == g.num_cells());
  const auto [cell_lo, cell_hi] = g.collect_cell_bounds();
  REQUIRE(cell_lo.shape()[0] == cells.size());
  REQUIRE(cell_lo.shape()[1] == 3);
  for (std::size_t c = 0; c < cells.size(); ++c) {
    const auto [lo, hi] = g.cell_bounds(static_cast<I64>(c));
    CHECK(lo == cells[c].lo);
    CHECK(hi == cells[c].hi);
    for (std::size_t d = 0; d < 3; ++d) {
      CHECK(cell_lo.data()[c * 3 + d] == cells[c].lo[d]);
      CHECK(cell_hi.data()[c * 3 + d] == cells[c].hi[d]);
    }
    const std::vector<I64> multi = g.cell_multi_index(static_cast<I64>(c));
    CHECK(g.flat_cell_index(multi) == static_cast<I64>(c));
    // The multi-index names the breakpoints the box was built from.
    for (std::size_t d = 0; d < 3; ++d) {
      CHECK(bp[d][static_cast<std::size_t>(multi[d])] == cells[c].lo[d]);
    }
  }
}

TEST_CASE("grid: TensorProductGrid: index errors carry the Python messages") {
  const TensorProductGrid g(dyadic3());
  CHECK_THROWS_WITH_AS(static_cast<void>(g.cell_bounds(24)),
                       "flat cell id 24 is out of range [0, 24).", pantr::index_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.cell_multi_index(-1)),
                       "flat cell id -1 is out of range [0, 24).", pantr::index_error);
  const std::vector<I64> bad = {0, 2, 0};
  CHECK_THROWS_WITH_AS(static_cast<void>(g.flat_cell_index(bad)),
                       "axis 1 index 2 out of range [0, 2).", pantr::index_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.flat_cell_index(std::vector<I64>{0, 0})),
                       "multi-index has length 2; expected 3.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.check_cid(24)), "cell id 24 is out of range [0, 24).",
                       pantr::index_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.check_lfid(0, 6)),
                       "local facet id 6 is out of range [0, 6).", pantr::index_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.check_lfid(0, -1)),
                       "local facet id -1 is out of range [0, 6).", pantr::index_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.neighbor_across_facet(99, 0)),
                       "cell id 99 is out of range [0, 24).", pantr::index_error);
}

TEST_CASE(
    "grid: TensorProductGrid::locate: matches the lowest cell whose closed box contains the "
    "point") {
  const auto bp = dyadic3();
  const TensorProductGrid g(bp);
  const std::vector<Box> cells = enumerate_cells(bp);
  // Every combination of {each breakpoint, the midpoint of each interval, a point outside either
  // end} per axis: the breakpoints exercise ties, the ends the clamping.
  Bp candidates(3);
  for (std::size_t d = 0; d < 3; ++d) {
    candidates[d].push_back(bp[d].front() - 1.0);
    for (std::size_t i = 0; i < bp[d].size(); ++i) {
      candidates[d].push_back(bp[d][i]);
      if (i + 1 < bp[d].size()) {
        candidates[d].push_back(0.5 * (bp[d][i] + bp[d][i + 1]));
      }
    }
    candidates[d].push_back(bp[d].back() + 1.0);
  }
  std::size_t n_inside = 0;
  for (const double x : candidates[0]) {
    for (const double y : candidates[1]) {
      for (const double z : candidates[2]) {
        const std::vector<double> pt = {x, y, z};
        const I64 expect = brute_locate(cells, pt);
        const std::optional<I64> got = g.locate(pt);
        if (expect < 0) {
          CHECK_FALSE(got.has_value());
        } else {
          REQUIRE(got.has_value());
          CHECK(*got == expect);
          ++n_inside;
        }
      }
    }
  }
  // 9 x 5 x 7 candidates fall inside the domain.
  CHECK(n_inside == 9 * 5 * 7);
}

TEST_CASE(
    "grid: TensorProductGrid::locate: the domain ends, non-finite coordinates and the length "
    "check") {
  const TensorProductGrid g = tpg({{0.0, 1.0, 3.0}});
  CHECK(g.locate(std::vector<double>{0.0}) == 0);
  CHECK(g.locate(std::vector<double>{1.0}) == 0);  // interior breakpoint: the lower cell
  CHECK(g.locate(std::vector<double>{3.0}) == 1);  // the end: the adjacent boundary cell
  CHECK_FALSE(g.locate(std::vector<double>{-1e-300}).has_value());
  CHECK_FALSE(g.locate(std::vector<double>{3.0000000000000004}).has_value());
  CHECK_FALSE(g.locate(std::vector<double>{std::nan("")}).has_value());
  CHECK_FALSE(g.locate(std::vector<double>{inf}).has_value());
  CHECK_FALSE(g.locate(std::vector<double>{-inf}).has_value());
  CHECK_THROWS_WITH_AS(static_cast<void>(g.locate(std::vector<double>{0.5, 0.5})),
                       "pt must have shape (1,); got (2,).", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.locate(std::vector<double>{})),
                       "pt must have shape (1,); got (0,).", pantr::value_error);
}

TEST_CASE(
    "grid: TensorProductGrid::locate_many: agrees with brute force and with locate, at 1 and 8 "
    "threads") {
  // 12 x 10 x 9 cells: about 24 operations per point, so 30000 points exceed the parallel
  // threshold and run in several chunks.
  Bp bp(3);
  const std::size_t cells[3] = {12, 10, 9};
  for (std::size_t d = 0; d < 3; ++d) {
    double x = static_cast<double>(d);
    bp[d].push_back(x);
    for (std::size_t i = 0; i < cells[d]; ++i) {
      x += 0.25 * static_cast<double>(1 + (i * (d + 3)) % 5);  // dyadic, non-uniform
      bp[d].push_back(x);
    }
  }
  const TensorProductGrid g(bp);
  const std::vector<Box> boxes = enumerate_cells(bp);

  const std::size_t npts = 30000;
  std::vector<double> pts(npts * 3);
  Lcg rng(2024);
  for (std::size_t p = 0; p < npts; ++p) {
    for (std::size_t d = 0; d < 3; ++d) {
      const double lo = bp[d].front() - 0.5;
      const double hi = bp[d].back() + 0.5;
      double x = lo + (hi - lo) * rng.next();
      if (p % 7 == 0) {
        // Land exactly on a breakpoint (or an end) on this axis.
        x = bp[d][(p / 7 + d) % bp[d].size()];
      }
      pts[p * 3 + d] = x;
    }
  }
  // Non-finite coordinates in the first and last point.
  pts[1] = std::nan("");
  pts[(npts - 1) * 3 + 2] = inf;
  pts[5 * 3] = -inf;

  // eight_threads() is capped by the CPU mask: with fewer than two CPUs both runs are serial, and
  // agreement between them says nothing about the pool.
  const bool pool = par::max_threads() >= 2;
  if (pool) {
    const ThreadsGuard guard(eight_threads());
    CHECK(par::get_num_threads() > 1);
  } else {
    MESSAGE(
        "max_threads() < 2: the 8-thread run is serial, so the comparison does not exercise "
        "the pool");
  }
  const auto run = [&](std::size_t threads) {
    const ThreadsGuard guard(threads);
    return g.locate_many(pts, npts);
  };
  const auto serial = run(1);
  const auto parallel = run(eight_threads());
  REQUIRE(serial.size() == npts);
  CHECK(serial.shape()[0] == npts);
  if (pool) {
    CHECK(std::equal(serial.data(), serial.data() + npts, parallel.data()));
  }

  for (std::size_t p = 0; p < npts; ++p) {
    const std::vector<double> pt = {pts[p * 3], pts[p * 3 + 1], pts[p * 3 + 2]};
    const bool finite = std::isfinite(pt[0]) && std::isfinite(pt[1]) && std::isfinite(pt[2]);
    const I64 expect = finite ? brute_locate(boxes, pt) : -1;
    if (serial.data()[p] != expect) {
      FAIL_CHECK("point " << p << ": got " << serial.data()[p] << ", expected " << expect);
    }
    const std::optional<I64> single = g.locate(pt);
    CHECK(serial.data()[p] == (single ? *single : -1));
  }
  CHECK(serial.data()[0] == -1);
  CHECK(serial.data()[5] == -1);
  CHECK(serial.data()[npts - 1] == -1);
}

TEST_CASE("grid: TensorProductGrid::locate_many: empty batch and a wrong size") {
  const TensorProductGrid g = tpg({{0.0, 1.0}, {0.0, 1.0}});
  CHECK(g.locate_many(std::vector<double>{}, 0).size() == 0);
  const std::vector<double> pts = {0.5, 0.5, 0.25};
  CHECK_THROWS_AS(static_cast<void>(g.locate_many(pts, 2)), pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(g.locate_many(pts, 0)), pantr::value_error);
}

TEST_CASE(
    "grid: TensorProductGrid: facet neighbors agree with a geometric search, boundary facets "
    "with the domain bounds") {
  const auto bp = dyadic3();
  const TensorProductGrid g(bp);
  const std::vector<Box> cells = enumerate_cells(bp);
  const auto n = static_cast<I64>(cells.size());
  std::vector<I64> boundary_rows;
  for (I64 cid = 0; cid < n; ++cid) {
    CHECK(g.num_local_facets(cid) == 6);
    for (I64 lfid = 0; lfid < 6; ++lfid) {
      const auto axis = static_cast<std::size_t>(lfid / 2);
      const bool high = lfid % 2 == 1;
      const Box& me = cells[static_cast<std::size_t>(cid)];
      // A neighbor shares this facet: same extent on the other axes, touching along the axis.
      I64 expect = -1;
      for (I64 other = 0; other < n; ++other) {
        const Box& o = cells[static_cast<std::size_t>(other)];
        bool match = true;
        for (std::size_t d = 0; d < 3; ++d) {
          if (d == axis) {
            match = match && (high ? o.lo[d] == me.hi[d] : o.hi[d] == me.lo[d]);
          } else {
            match = match && o.lo[d] == me.lo[d] && o.hi[d] == me.hi[d];
          }
        }
        if (match) {
          expect = other;
        }
      }
      const std::optional<I64> got = g.neighbor_across_facet(cid, lfid);
      CHECK(got.has_value() == (expect >= 0));
      if (got.has_value()) {
        CHECK(*got == expect);
      }
      CHECK(g.is_mesh_boundary_facet(cid, lfid) == (expect < 0));
      // Geometric reading: the facet is on the boundary iff it lies on the domain's bounds.
      const double face = high ? me.hi[axis] : me.lo[axis];
      const double bound = high ? g.bounds()[2 * axis + 1] : g.bounds()[2 * axis];
      CHECK((face == bound) == (expect < 0));
      if (face == bound) {
        boundary_rows.push_back(cid);
        boundary_rows.push_back(lfid);
      }
      const auto [a, s] = g.local_facet_axis_side(cid, lfid);
      CHECK(a == static_cast<I64>(axis));
      CHECK(s == (high ? 1 : 0));
    }
    const std::vector<I64> nb = g.neighbors(cid);
    I64 count = 0;
    for (I64 lfid = 0; lfid < 6; ++lfid) {
      count += g.neighbor_across_facet(cid, lfid).has_value() ? 1 : 0;
    }
    CHECK(static_cast<I64>(nb.size()) == count);
  }
  const pantr::NdArray<I64> bf = g.boundary_facets();
  REQUIRE(bf.rank() == 2);
  CHECK(bf.shape()[1] == 2);
  CHECK(std::vector<I64>(bf.data(), bf.data() + bf.size()) == boundary_rows);
  // 2 * (2*3 + 4*3 + 4*2) faces on the outer boundary of a 4 x 2 x 3 grid.
  CHECK(bf.shape()[0] == 2 * (2 * 3 + 4 * 3 + 4 * 2));
}

TEST_CASE("grid: TensorProductGrid: neighbor counts run from ndim at a corner to 2 ndim inside") {
  const TensorProductGrid g = tpg({{0.0, 1.0, 2.0, 3.0}, {0.0, 1.0, 2.0, 3.0}});
  CHECK(g.neighbors(0).size() == 2);
  CHECK(g.neighbors(4).size() == 4);                      // the middle cell of the 3 x 3 grid
  CHECK(g.neighbors(1).size() == 3);                      // an edge cell
  CHECK(g.neighbors(4) == std::vector<I64>{1, 7, 3, 5});  // axis 0 low and high, then axis 1
}

TEST_CASE(
    "grid: GridBase defaults: level, children, facet bounds, hanging neighbors, boxes and maps") {
  const TensorProductGrid g(dyadic3());
  CHECK(g.cell_level(5) == 0);
  CHECK(g.child_cells(5).empty());
  CHECK_THROWS_AS(static_cast<void>(g.cell_level(24)), pantr::index_error);
  CHECK_THROWS_AS(static_cast<void>(g.child_cells(-1)), pantr::index_error);
  CHECK_THROWS_AS(static_cast<void>(g.num_local_facets(24)), pantr::index_error);

  // Cell 0 is [0, 0.5] x [-1, 0] x [4, 5]; its facets: lfid = 2 * axis + side.
  const auto [lo0, hi0] = g.local_facet_bounds(0, 0);
  CHECK(lo0 == std::vector<double>{0.0, -1.0, 4.0});
  CHECK(hi0 == std::vector<double>{0.0, 0.0, 5.0});
  const auto [lo3, hi3] = g.local_facet_bounds(0, 3);
  CHECK(lo3 == std::vector<double>{0.0, 0.0, 4.0});
  CHECK(hi3 == std::vector<double>{0.5, 0.0, 5.0});
  CHECK_THROWS_AS(static_cast<void>(g.local_facet_bounds(0, 6)), pantr::index_error);

  CHECK(g.hanging_neighbors(0, 0).empty());
  CHECK(g.hanging_neighbors(0, 1) == std::vector<I64>{6});
  const AABB box = g.cell_aabb(0);
  CHECK(box == AABB(lo0, std::vector<double>{0.5, 0.0, 5.0}));

  // Cell 0 maps the unit cube to its box: A = diag(0.5, 1, 1), b = lo.
  const pantr::transform::AffineTransform map = g.reference_map(0);
  CHECK(map.dim() == 3);
  CHECK(std::vector<double>(map.matrix().begin(), map.matrix().end()) ==
        std::vector<double>{0.5, 0, 0, 0, 1, 0, 0, 0, 1});
  CHECK(std::vector<double>(map.offset().begin(), map.offset().end()) ==
        std::vector<double>{0.0, -1.0, 4.0});
}

TEST_CASE(
    "grid: GridBase::query_aabb: matches AABB::overlaps over every cell, and builds the BVH once") {
  const auto bp = dyadic3();
  const TensorProductGrid g(bp);
  const std::vector<Box> cells = enumerate_cells(bp);
  CHECK_FALSE(g.bvh_built());
  Lcg rng(7);
  for (int q = 0; q < 100; ++q) {
    std::vector<double> lo(3);
    std::vector<double> hi(3);
    for (std::size_t d = 0; d < 3; ++d) {
      const double a = bp[d].front() - 1.0;
      const double b = bp[d].back() + 1.0;
      lo[d] = a + (b - a) * rng.next();
      if (q % 5 == 0) {
        lo[d] = bp[d][static_cast<std::size_t>(q) % bp[d].size()];  // touch a face
      }
      hi[d] = lo[d] + 1.5 * rng.next();
    }
    const AABB query(lo, hi);
    std::vector<I64> expect;
    for (std::size_t c = 0; c < cells.size(); ++c) {
      if (AABB(cells[c].lo, cells[c].hi).overlaps(query)) {
        expect.push_back(static_cast<I64>(c));
      }
    }
    std::vector<I64> got = g.query_aabb(query);
    std::sort(got.begin(), got.end());
    CHECK(got == expect);
    CHECK(g.bvh_built());
  }
  const pantr::grid::BVH& tree = g.cell_bvh();
  CHECK(&tree == &g.cell_bvh());
  CHECK(tree.n_cells() == g.num_cells());
  CHECK_THROWS_WITH_AS(static_cast<void>(g.query_aabb(AABB::unbounded(2))),
                       "BVH.query_aabb: aabb.ndim (2) must match self.ndim (3).",
                       pantr::value_error);
}

TEST_CASE("grid: GridBase::cell_bvh: threads racing on the first call all see the same BVH") {
  const TensorProductGrid g = tpg({{0.0, 1.0, 2.0, 3.0, 4.0}, {0.0, 1.0, 2.0, 3.0, 4.0}});
  CHECK_FALSE(g.bvh_built());
  constexpr int n_threads = 8;
  std::vector<const pantr::grid::BVH*> seen(n_threads, nullptr);
  std::atomic<int> ready{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < n_threads; ++t) {
    threads.emplace_back([&, t] {
      ready.fetch_add(1);
      while (ready.load() < n_threads) {
        std::this_thread::yield();
      }
      seen[static_cast<std::size_t>(t)] = &g.cell_bvh();
    });
  }
  for (std::thread& t : threads) {
    t.join();
  }
  for (const auto* p : seen) {
    CHECK(p == seen[0]);
  }
  CHECK(g.bvh_built());
}

TEST_CASE("grid: TensorProductGrid: a copy shares the representation and the built BVH") {
  const TensorProductGrid a = tpg({{0.0, 1.0, 2.0}});
  const TensorProductGrid b = a;  // NOLINT(performance-unnecessary-copy-initialization)
  CHECK(a.same_rep(b));
  CHECK_FALSE(a.bvh_built());
  CHECK_FALSE(b.bvh_built());
  static_cast<void>(a.cell_bvh());
  CHECK(b.bvh_built());
  const TensorProductGrid c = tpg({{0.0, 1.0, 2.0}});
  CHECK_FALSE(a.same_rep(c));
}

TEST_CASE(
    "grid: TensorProductGrid::restrict: a pure slice of the breakpoints with the requested "
    "cells flagged") {
  // 4 x 2 x 3 cells; cell id = 6 i + 3 j + k. Request cells (1,0,1) = 7 and (3,1,2) = 23 and the
  // duplicate 7: the window is i in [1,3], j in [0,1], k in [1,2], 3 x 2 x 2 = 12 cells.
  const auto bp = dyadic3();
  const TensorProductGrid g(bp);
  const std::vector<I64> ids = {7, 23, 7};
  const pantr::grid::TensorProductGridRestriction r = g.restrict(ids);
  CHECK(r.grid.num_cells() == 12);
  REQUIRE(r.grid.ndim() == 3);
  CHECK(std::vector<double>(r.grid.breakpoints(0).begin(), r.grid.breakpoints(0).end()) ==
        std::vector<double>{0.5, 1.0, 2.0, 2.25});
  CHECK(std::vector<double>(r.grid.breakpoints(1).begin(), r.grid.breakpoints(1).end()) ==
        std::vector<double>{-1.0, 0.0, 0.75});
  CHECK(std::vector<double>(r.grid.breakpoints(2).begin(), r.grid.breakpoints(2).end()) ==
        std::vector<double>{5.0, 5.5, 8.0});
  // Local cell (a, b, c) = 4 a + 2 b + c is global cell (a + 1, b, c + 1).
  std::vector<I64> expect_global;
  for (I64 a = 0; a < 3; ++a) {
    for (I64 b = 0; b < 2; ++b) {
      for (I64 c = 0; c < 2; ++c) {
        expect_global.push_back(6 * (a + 1) + 3 * b + (c + 1));
      }
    }
  }
  CHECK(r.local_to_global == expect_global);
  std::vector<std::uint8_t> expect_in(12, 0);
  for (std::size_t l = 0; l < 12; ++l) {
    expect_in[l] = (expect_global[l] == 7 || expect_global[l] == 23) ? 1 : 0;
  }
  CHECK(r.in_subset == expect_in);
  // The sub-grid's cells coincide with the matching cells of the original, bit for bit.
  for (std::size_t l = 0; l < 12; ++l) {
    const auto sub = r.grid.cell_bounds(static_cast<I64>(l));
    const auto glob = g.cell_bounds(expect_global[l]);
    CHECK(sub.first == glob.first);
    CHECK(sub.second == glob.second);
  }
}

TEST_CASE("grid: TensorProductGrid::restrict: one cell, the whole grid, and errors") {
  const TensorProductGrid g(dyadic3());
  const std::vector<I64> one = {13};
  const auto single = g.restrict(one);
  CHECK(single.grid.num_cells() == 1);
  CHECK(single.local_to_global == std::vector<I64>{13});
  CHECK(single.in_subset == std::vector<std::uint8_t>{1});
  CHECK(single.grid.cell_bounds(0) == g.cell_bounds(13));

  std::vector<I64> all(24);
  for (I64 i = 0; i < 24; ++i) {
    all[static_cast<std::size_t>(i)] = i;
  }
  const auto whole = g.restrict(all);
  CHECK(whole.grid.num_cells() == 24);
  CHECK(whole.local_to_global == all);
  CHECK(std::all_of(whole.in_subset.begin(), whole.in_subset.end(),
                    [](std::uint8_t f) { return f == 1; }));

  CHECK_THROWS_WITH_AS(static_cast<void>(g.restrict(std::vector<I64>{})),
                       "restrict: cell_ids must be non-empty.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.restrict(std::vector<I64>{-1, 3})),
                       "restrict: cell id out of range [0, 24); got [-1, 3].", pantr::index_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(g.restrict(std::vector<I64>{0, 24})),
                       "restrict: cell id out of range [0, 24); got [0, 24].", pantr::index_error);
}
