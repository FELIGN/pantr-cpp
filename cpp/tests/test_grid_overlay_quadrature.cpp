// overlay and cell_quadrature. Oracles: hand-derived breakpoint sets (inputs chosen so the merged
// values are the inputs' own doubles), the refinement property read through a second route
// (cell centers located in both inputs), dyadic grids and rules whose mapped points and weights
// are exact in binary (compared for equality), and the integral of a polynomial that Gauss
// quadrature integrates exactly (compared to a derived tolerance).
#include <doctest/doctest.h>

#include <pantr/grid/cell_quadrature.hpp>
#include <pantr/grid/overlay.hpp>

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

using pantr::grid::cell_quadrature;
using pantr::grid::overlay;
using pantr::grid::TensorProductGrid;
using Bp = std::vector<std::vector<double>>;

namespace {

using I64 = std::int64_t;

TensorProductGrid tpg(Bp bp) { return TensorProductGrid(std::move(bp)); }

std::vector<double> axis(const TensorProductGrid& g, std::size_t d) {
  const std::span<const double> b = g.breakpoints(d);
  return {b.begin(), b.end()};
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

/// A reference rule: row-major points and weights.
struct Rule {
  std::vector<double> points;
  std::vector<double> weights;
  std::size_t npts;
  std::size_t ndim;
};

/// The tensor product of the two-point Gauss rule on [0, 1] in `ndim` dimensions.
Rule gauss2(std::size_t ndim) {
  const double x0 = 0.5 - 0.5 / std::sqrt(3.0);
  const double x1 = 0.5 + 0.5 / std::sqrt(3.0);
  Rule r;
  r.ndim = ndim;
  r.npts = std::size_t{1} << ndim;
  for (std::size_t q = 0; q < r.npts; ++q) {
    double w = 1.0;
    for (std::size_t d = 0; d < ndim; ++d) {
      r.points.push_back(((q >> (ndim - 1 - d)) & 1U) != 0 ? x1 : x0);
      w *= 0.5;
    }
    r.weights.push_back(w);
  }
  return r;
}

}  // namespace

TEST_CASE("grid: overlay: the union of the breakpoints on the common window") {
  // Cells 2 and 3 on [0, 1]: the interior breakpoints 1/2 and 1/3, 2/3 interleave.
  const TensorProductGrid a = tpg({{0.0, 0.5, 1.0}});
  const TensorProductGrid b = tpg({{0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0}});
  const TensorProductGrid o = overlay(a, b);
  CHECK(axis(o, 0) == std::vector<double>{0.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0});
  CHECK(axis(overlay(b, a), 0) == axis(o, 0));
}

TEST_CASE("grid: overlay: the window is the intersection of the domains, in every dimension") {
  const TensorProductGrid a = tpg({{0.0, 1.0, 2.0, 4.0}, {-1.0, 0.0, 1.0}});
  const TensorProductGrid b = tpg({{1.5, 2.0, 3.0, 9.0}, {-4.0, -0.5, 0.5, 1.0}});
  const TensorProductGrid o = overlay(a, b);
  CHECK(o.ndim() == 2);
  // Axis 0: window [1.5, 4]; a contributes 2, b contributes 2 and 3 (merged with a's 2).
  CHECK(axis(o, 0) == std::vector<double>{1.5, 2.0, 3.0, 4.0});
  // Axis 1: window [-1, 1]; a contributes 0, b contributes -0.5 and 0.5.
  CHECK(axis(o, 1) == std::vector<double>{-1.0, -0.5, 0.0, 0.5, 1.0});
}

TEST_CASE("grid: overlay: breakpoints closer than the relative tolerance are merged") {
  // atol = 64 eps * max(window, |lo|, |hi|). On [0, 1] that is about 1.4e-14.
  const double eps = std::numeric_limits<double>::epsilon();
  const TensorProductGrid a = tpg({{0.0, 0.5, 1.0}});
  const TensorProductGrid close_by = tpg({{0.0, 0.5 + 8 * eps, 1.0}});
  const TensorProductGrid distant = tpg({{0.0, 0.5 + 1e-9, 1.0}});
  CHECK(axis(overlay(a, close_by), 0) == std::vector<double>{0.0, 0.5, 1.0});
  CHECK(axis(overlay(a, distant), 0) == std::vector<double>{0.0, 0.5, 0.5 + 1e-9, 1.0});
  // A breakpoint within atol of an end of the window is dropped, not merged into it.
  const TensorProductGrid at_end = tpg({{0.0, 8 * eps, 1.0}});
  CHECK(axis(overlay(a, at_end), 0) == std::vector<double>{0.0, 0.5, 1.0});
  // The same 1e-9 spacing sitting at 1e6 is below the relative tolerance (1.4e-8) and merges.
  const double base = 1e6;
  const TensorProductGrid shifted_a = tpg({{base, base + 0.5, base + 1.0}});
  const TensorProductGrid shifted_far = tpg({{base, base + 0.5 + 1e-9, base + 1.0}});
  CHECK(axis(overlay(shifted_a, shifted_far), 0).size() == 3);
}

TEST_CASE("grid: overlay: every overlay cell lies in one cell of each input, which it refines") {
  const TensorProductGrid a = tpg({{0.0, 0.7, 1.9, 3.0}, {0.0, 1.0, 1.25, 2.0}});
  const TensorProductGrid b = tpg({{0.5, 1.1, 2.4, 2.9, 5.0}, {-1.0, 0.4, 1.75}});
  const TensorProductGrid o = overlay(a, b);
  for (I64 c = 0; c < o.num_cells(); ++c) {
    const auto [lo, hi] = o.cell_bounds(c);
    std::vector<double> center(2);
    for (std::size_t d = 0; d < 2; ++d) {
      center[d] = 0.5 * (lo[d] + hi[d]);
    }
    for (const TensorProductGrid* in : {&a, &b}) {
      const std::optional<I64> id = in->locate(center);
      REQUIRE(id.has_value());
      const auto [ilo, ihi] = in->cell_bounds(*id);
      for (std::size_t d = 0; d < 2; ++d) {
        CHECK(ilo[d] <= lo[d]);
        CHECK(hi[d] <= ihi[d]);
      }
    }
  }
  // Each input breakpoint inside the window appears in the overlay.
  for (std::size_t d = 0; d < 2; ++d) {
    const std::vector<double> ob = axis(o, d);
    for (const TensorProductGrid* in : {&a, &b}) {
      for (const double x : in->breakpoints(d)) {
        if (x > ob.front() && x < ob.back()) {
          CHECK(std::binary_search(ob.begin(), ob.end(), x));
        }
      }
    }
  }
}

TEST_CASE("grid: overlay: errors carry the Python messages") {
  const TensorProductGrid one = tpg({{0.0, 1.0}});
  const TensorProductGrid two = tpg({{0.0, 1.0}, {0.0, 1.0}});
  CHECK_THROWS_WITH_AS(static_cast<void>(overlay(one, two)),
                       "overlay(): grids must share ndim; got 1 vs 2.", pantr::value_error);
  const TensorProductGrid apart = tpg({{2.0, 3.0}});
  CHECK_THROWS_WITH_AS(static_cast<void>(overlay(one, apart)),
                       "overlay(): domains do not overlap on axis 0; grid_a extent [0.0, 1.0] vs "
                       "grid_b extent [2.0, 3.0].",
                       pantr::value_error);
  // Domains that only touch have a degenerate window.
  const TensorProductGrid touching = tpg({{1.0, 2.0}});
  CHECK_THROWS_AS(static_cast<void>(overlay(one, touching)), pantr::value_error);
  // The axis number is the offending one.
  const TensorProductGrid mixed = tpg({{0.0, 1.0}, {5.0, 6.0}});
  CHECK_THROWS_WITH_AS(static_cast<void>(overlay(two, mixed)),
                       "overlay(): domains do not overlap on axis 1; grid_a extent [0.0, 1.0] vs "
                       "grid_b extent [5.0, 6.0].",
                       pantr::value_error);
}

TEST_CASE("grid: cell_quadrature: exact mapping on a dyadic grid") {
  // Cells [0, 0.5], [0.5, 2] by [0, 1]; the rule has points on the quarter lattice, so every
  // product and sum below is exact in binary.
  const TensorProductGrid g = tpg({{0.0, 0.5, 2.0}, {0.0, 1.0}});
  Rule r;
  r.ndim = 2;
  r.npts = 2;
  r.points = {0.25, 0.5, 0.75, 0.25};
  r.weights = {0.5, 0.5};
  const auto [pts, wts] = cell_quadrature(g, r.points, r.npts, r.ndim, r.weights);
  REQUIRE(pts.rank() == 3);
  CHECK(pts.shape()[0] == 2);
  CHECK(pts.shape()[1] == 2);
  CHECK(pts.shape()[2] == 2);
  REQUIRE(wts.rank() == 2);
  CHECK(wts.shape()[0] == 2);
  CHECK(wts.shape()[1] == 2);
  // Cell 0: lo (0, 0), span (0.5, 1): points (0.125, 0.5), (0.375, 0.25); volume 0.5.
  CHECK(std::vector<double>(pts.data(), pts.data() + 4) ==
        std::vector<double>{0.125, 0.5, 0.375, 0.25});
  CHECK(std::vector<double>(wts.data(), wts.data() + 2) == std::vector<double>{0.25, 0.25});
  // Cell 1: lo (0.5, 0), span (1.5, 1): points (0.875, 0.5), (1.625, 0.25); volume 1.5.
  CHECK(std::vector<double>(pts.data() + 4, pts.data() + 8) ==
        std::vector<double>{0.875, 0.5, 1.625, 0.25});
  CHECK(std::vector<double>(wts.data() + 2, wts.data() + 4) == std::vector<double>{0.75, 0.75});
}

TEST_CASE(
    "grid: cell_quadrature: the weights of a cell sum to its volume and a cubic is integrated "
    "exactly") {
  for (const std::size_t nd : {std::size_t{1}, std::size_t{2}, std::size_t{3}}) {
    Bp bp;
    const std::vector<double> ends = {0.0, 0.3, 1.1, 1.2, 2.5};
    for (std::size_t d = 0; d < nd; ++d) {
      std::vector<double> b;
      for (const double e : ends) {
        b.push_back(e + 0.5 * static_cast<double>(d));
      }
      bp.push_back(b);
    }
    const TensorProductGrid g = tpg(bp);
    const Rule r = gauss2(nd);
    const auto [pts, wts] = cell_quadrature(g, r.points, r.npts, r.ndim, r.weights);
    const auto n = static_cast<std::size_t>(g.num_cells());
    REQUIRE(wts.shape()[0] == n);
    // Derived tolerance: each weight is a product of nd rounded factors times 0.5^nd (exact), so a
    // relative error of about (nd + 1) u per weight; the sum of nq positive terms adds nq u. The
    // factor 4 covers the build's contraction choices.
    const double u = std::numeric_limits<double>::epsilon() / 2.0;
    const double rel = 4.0 * static_cast<double>(nd + 1 + r.npts) * u;
    double total = 0.0;
    double integral = 0.0;  // of x_0 * x_1 ... over the grid (degree 1 per variable)
    for (std::size_t c = 0; c < n; ++c) {
      const auto [lo, hi] = g.cell_bounds(static_cast<I64>(c));
      double volume = 1.0;
      for (std::size_t d = 0; d < nd; ++d) {
        volume *= hi[d] - lo[d];
      }
      double sum = 0.0;
      for (std::size_t q = 0; q < r.npts; ++q) {
        sum += wts.data()[c * r.npts + q];
        double f = 1.0;
        for (std::size_t d = 0; d < nd; ++d) {
          const double x = pts.data()[(c * r.npts + q) * nd + d];
          CHECK(x >= lo[d]);
          CHECK(x <= hi[d]);
          f *= x;
        }
        integral += wts.data()[c * r.npts + q] * f;
      }
      CHECK(std::fabs(sum - volume) <= rel * volume);
      total += sum;
    }
    // The integral of prod x_d over the box is prod (b_d^2 - a_d^2) / 2.
    double exact = 1.0;
    double volume_all = 1.0;
    for (std::size_t d = 0; d < nd; ++d) {
      const double a0 = bp[d].front();
      const double b0 = bp[d].back();
      exact *= (b0 * b0 - a0 * a0) / 2.0;
      volume_all *= b0 - a0;
    }
    // All terms are positive, so no cancellation: the relative error of a sequential sum of m
    // products of at most nd + 2 rounded factors is at most (m + nd + 2) u; the factor 8 covers
    // the build's contraction choices and the (b^2 - a^2) / 2 of the reference value.
    const double m = static_cast<double>(n * r.npts) + static_cast<double>(nd) + 2.0;
    CHECK(std::fabs(total - volume_all) <= 8.0 * m * u * volume_all);
    CHECK(std::fabs(integral - exact) <= 8.0 * m * u * exact);
  }
}

TEST_CASE(
    "grid: cell_quadrature: a selection keeps its order and duplicates, an empty one is empty") {
  const TensorProductGrid g = tpg({{0.0, 0.5, 2.0, 4.0}, {0.0, 1.0, 1.5}});
  const Rule r = gauss2(2);
  const auto [all_pts, all_wts] = cell_quadrature(g, r.points, r.npts, r.ndim, r.weights);
  const std::vector<I64> pick = {4, 0, 4, 5};
  const auto [pts, wts] =
      cell_quadrature(g, r.points, r.npts, r.ndim, r.weights, std::span<const I64>(pick));
  REQUIRE(pts.shape()[0] == 4);
  REQUIRE(wts.shape()[0] == 4);
  for (std::size_t s = 0; s < pick.size(); ++s) {
    const auto cell = static_cast<std::size_t>(pick[s]);
    for (std::size_t k = 0; k < r.npts * 2; ++k) {
      CHECK(pts.data()[s * r.npts * 2 + k] == all_pts.data()[cell * r.npts * 2 + k]);
    }
    for (std::size_t k = 0; k < r.npts; ++k) {
      CHECK(wts.data()[s * r.npts + k] == all_wts.data()[cell * r.npts + k]);
    }
  }
  const std::vector<I64> none;
  const auto [e_pts, e_wts] =
      cell_quadrature(g, r.points, r.npts, r.ndim, r.weights, std::span<const I64>(none));
  CHECK(e_pts.shape()[0] == 0);
  CHECK(e_pts.shape()[1] == 4);
  CHECK(e_pts.shape()[2] == 2);
  CHECK(e_wts.shape()[0] == 0);
  CHECK(e_pts.size() == 0);
}

TEST_CASE("grid: cell_quadrature: errors carry the Python messages") {
  const TensorProductGrid g = tpg({{0.0, 1.0, 2.0, 3.0}, {0.0, 1.0, 2.0}});  // 6 cells
  const Rule r = gauss2(2);
  const Rule r3 = gauss2(3);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(cell_quadrature(g, r3.points, r3.npts, r3.ndim, r3.weights)),
      "rule.ndim (3) must match grid.ndim (2).", pantr::value_error);
  const std::vector<I64> too_big = {0, 6};
  CHECK_THROWS_WITH_AS(static_cast<void>(cell_quadrature(g, r.points, r.npts, r.ndim, r.weights,
                                                         std::span<const I64>(too_big))),
                       "cells must lie in [0, 6); got range [0, 6].", pantr::value_error);
  const std::vector<I64> negative = {3, -1, 7};
  CHECK_THROWS_WITH_AS(static_cast<void>(cell_quadrature(g, r.points, r.npts, r.ndim, r.weights,
                                                         std::span<const I64>(negative))),
                       "cells must lie in [0, 6); got range [-1, 7].", pantr::value_error);
  const std::vector<double> short_weights(3, 0.25);
  CHECK_THROWS_AS(static_cast<void>(cell_quadrature(g, r.points, r.npts, r.ndim, short_weights)),
                  pantr::value_error);
}

TEST_CASE("grid: cell_quadrature: a GridLike that is not a TensorProductGrid maps the same cells") {
  const TensorProductGrid g = tpg({{0.0, 0.5, 2.0}, {0.0, 1.0}});
  const OtherGrid h(g);
  const Rule r = gauss2(2);
  const auto [pg, wg] = cell_quadrature(g, r.points, r.npts, r.ndim, r.weights);
  const auto [ph, wh] = cell_quadrature(h, r.points, r.npts, r.ndim, r.weights);
  CHECK(std::equal(pg.data(), pg.data() + pg.size(), ph.data()));
  CHECK(std::equal(wg.data(), wg.data() + wg.size(), wh.data()));
  const std::vector<I64> pick = {1, 0};
  const auto [ps, ws] =
      cell_quadrature(h, r.points, r.npts, r.ndim, r.weights, std::span<const I64>(pick));
  CHECK(ps.data()[0] == pg.data()[8]);
  CHECK(ws.data()[0] == wg.data()[4]);
}
