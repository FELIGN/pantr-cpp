// THBSplineSpace and its detail helpers. Oracles, each from a route other than the code under
// test:
//   - partition of unity of the truncated basis on every cell, at points that include the cell
//     corners, and non-negativity; the plain hierarchical basis (HB) as the control that does not
//     sum to one near a level interface, so the check can fail;
//   - the Kraft selection and the cell supports by brute force, from the knot vectors and the
//     grid's point location, never from the support tables the space builds;
//   - derivatives against central finite differences of the values, with a derived step and bound;
//   - the two-scale step against a matrix worked out by hand;
//   - values recorded from the Python implementation (thb_golden.hpp says how);
//   - contributions at 1 and 8 threads, which are integers and so compared exactly.
#include <doctest/doctest.h>

#include "thb_golden.hpp"
#include "thb_test_support.hpp"

#include <pantr/bspline/space.hpp>
#include <pantr/bspline/space_1d.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/core/ndarray.hpp>
#include <pantr/grid/hierarchical_grid.hpp>
#include <pantr/grid/tensor_product_grid.hpp>
#include <pantr/parallel/parallel_for.hpp>
#include <pantr/parallel/work_model.hpp>
#include <pantr/thb/detail/summed_area.hpp>
#include <pantr/thb/detail/truncation.hpp>
#include <pantr/thb/space.hpp>
#include <pantr/tolerance.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace bs = pantr::bspline;
namespace gd = pantr::grid;
namespace par = pantr::parallel;
namespace td = pantr::thb::detail;
namespace gold = pantr_thb_golden;
using pantr::mdview;
using pantr::thb::THBSplineSpace;

using thb_test::build_slack;
using thb_test::cell_points;
using thb_test::check_many_threads_are_used;
using thb_test::eight_threads;
using thb_test::eps;
using thb_test::I64;
using thb_test::Ivec;
using thb_test::k2;
using thb_test::k2_3d;
using thb_test::k2n;
using thb_test::k3;
using thb_test::Reg;
using thb_test::ThreadsGuard;
using thb_test::value_error_units;

// ---- Setups ----------------------------------------------------------------------------------

struct Refine {
  I64 level;
  Ivec lo;
  Ivec hi;
};

/// Everything that defines a space: per-direction knots (in double, narrowed to the storage type
/// of the root), degrees, the grid's factor, the regularity and the refinements.
struct Setup {
  std::vector<std::vector<double>> knots;
  std::vector<int> degrees;
  Ivec factor;
  Reg regularity;
  std::vector<Refine> refinements;
  bool truncate = true;
};

/// Degree 2 and not clamped: domain [0, 1], and the first two-scale rows start left of column 0.
const std::vector<double> k_unclamped = {-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5};

Setup setup_2d(bool truncate, int levels, std::vector<double> knots = k2, Ivec factor = {2, 2}) {
  Setup s{{knots, knots}, {2, 2}, std::move(factor), {std::nullopt, std::nullopt}, {}, truncate};
  s.refinements.push_back({0, {0, 0}, {2, 2}});
  if (levels == 3) {
    s.refinements.push_back({1, {0, 0}, {2, 2}});
  }
  return s;
}

Setup setup_1d(bool truncate, int levels) {
  Setup s{{k3}, {3}, {2}, {0}, {}, truncate};
  s.refinements.push_back({0, {0}, {2}});
  if (levels == 3) {
    s.refinements.push_back({1, {0}, {2}});
  }
  return s;
}

Setup setup_unclamped() {
  return Setup{{k_unclamped}, {2}, {2}, {std::nullopt}, {{0, {0}, {2}}}, true};
}

Setup setup_3d() {
  return Setup{
      {k2_3d, k2_3d, k2_3d},       {2, 2, 2}, {2, 2, 2}, {std::nullopt, std::nullopt, std::nullopt},
      {{0, {0, 0, 0}, {1, 1, 1}}}, true};
}

template <class T>
bs::BsplineSpace<T> make_root(const Setup& s) {
  std::vector<bs::BsplineSpace1D<T>> dirs;
  for (std::size_t k = 0; k < s.knots.size(); ++k) {
    std::vector<T> knots;
    for (const double v : s.knots[k]) {
      knots.push_back(static_cast<T>(v));
    }
    dirs.emplace_back(std::span<const T>(knots), s.degrees[k]);
  }
  return bs::BsplineSpace<T>(std::move(dirs));
}

/// The root grid whose cells are the root space's knot spans, then the setup's refinements.
template <class T>
gd::HierarchicalGrid make_grid(const bs::BsplineSpace<T>& root, const Setup& s) {
  std::vector<std::vector<double>> breakpoints;
  for (const auto& dir : root.spaces()) {
    std::vector<double> bp;
    for (const T v : dir.unique_knots_and_multiplicity(true).first) {
      bp.push_back(static_cast<double>(v));
    }
    breakpoints.push_back(std::move(bp));
  }
  gd::HierarchicalGrid grid(gd::TensorProductGrid(std::move(breakpoints)), s.factor);
  for (const Refine& r : s.refinements) {
    grid = grid.refined(r.level, r.lo, r.hi);
  }
  return grid;
}

template <class T = double>
THBSplineSpace<T> make_space(const Setup& s) {
  bs::BsplineSpace<T> root = make_root<T>(s);
  gd::HierarchicalGrid grid = make_grid<T>(root, s);
  return THBSplineSpace<T>(std::move(root), std::move(grid), s.truncate, s.regularity);
}

// ---- Helpers ---------------------------------------------------------------------------------

mdview<const double, 2> rows_of(const std::vector<double>& pts, std::size_t d) {
  return mdview<const double, 2>(pts.data(), pts.size() / d, d);
}

template <class T>
pantr::thb::TabulatedBasis tabulate(const THBSplineSpace<T>& space, I64 cid,
                                    const std::vector<double>& pts, const Ivec& orders) {
  return space.tabulate_basis(cid, rows_of(pts, space.dim()), orders);
}

/// The unit roundoff scale of the root's storage type, as a double.
template <class T>
double storage_eps(const THBSplineSpace<T>& /*space*/) {
  return static_cast<double>(std::numeric_limits<T>::epsilon());
}

// ---- Golden values from the Python implementation --------------------------------------------

struct GoldenCase {
  const char* name;
  const gold::Case* data;
  Setup setup;
  bool single_precision;
};

std::vector<GoldenCase> golden_cases() {
  Setup fac21 = setup_2d(true, 2, k2, {2, 1});
  return {{"a2t", &gold::a2t, setup_2d(true, 2), false},
          {"a2h", &gold::a2h, setup_2d(false, 2), false},
          {"a3t", &gold::a3t, setup_2d(true, 3), false},
          {"a3h", &gold::a3h, setup_2d(false, 3), false},
          {"b2t", &gold::b2t, setup_1d(true, 2), false},
          {"b3t", &gold::b3t, setup_1d(true, 3), false},
          {"b3h", &gold::b3h, setup_1d(false, 3), false},
          {"f32", &gold::f32, setup_2d(true, 2, k2n), true},
          {"fac21", &gold::fac21, std::move(fac21), false},
          {"d3t", &gold::d3t, setup_3d(), false},
          {"unc", &gold::unc, setup_unclamped(), false}};
}

/// Compare a space with the values the Python implementation produced for the same inputs.
///
/// Integers (active functions, box positions, dofs, knots, which are the same arithmetic on the
/// same inputs) are compared exactly. The coefficients and the tabulated values are compared
/// within `4 * m * eps * scale`: the two implementations run the same recurrences, so they differ
/// only in the order a sum is formed in (a BLAS contraction against a fixed-order one) and in
/// contraction of a multiply-add. `m` is `value_error_units`, the bound on the rounding error of a
/// value with non-negative terms, `4` is `build_slack`, and `scale` is the largest recorded
/// magnitude of the table, because a derivative's terms cancel and its error is relative to them,
/// not to the (smaller) result. `eps` is the root's storage epsilon: a float root forms its
/// two-scale coefficients in float, so a contracted multiply-add moves them by float ulps.
template <class T>
void check_against_golden(const THBSplineSpace<T>& space, const gold::Case& g) {
  const auto d = static_cast<std::size_t>(g.dim);
  REQUIRE(space.dim() == d);
  CHECK(space.truncate() == g.truncate);
  CHECK(space.num_total_basis() == g.num_total);
  CHECK(space.grid().num_cells() == g.num_cells);
  CHECK(space.max_active_per_cell() == g.max_active);
  CHECK(space.tolerance() == g.tolerance);
  REQUIRE(static_cast<std::size_t>(space.num_levels()) == g.per_level.size());
  const std::vector<I64> per_level = space.num_basis_per_level();
  for (std::size_t l = 0; l < g.per_level.size(); ++l) {
    CHECK(per_level[l] == g.per_level[l]);
    const std::span<const I64> active = space.active_function_indices(static_cast<I64>(l));
    CHECK(std::equal(active.begin(), active.end(), g.active[l].begin(), g.active[l].end()));
    const bs::BsplineSpace<T> level_space = space.level_space(static_cast<I64>(l));
    for (std::size_t k = 0; k < d; ++k) {
      const std::span<const T> knots = level_space.spaces()[k].knots();
      const std::span<const double> want = g.level_knots[l * d + k];
      REQUIRE(knots.size() == want.size());
      for (std::size_t i = 0; i < knots.size(); ++i) {
        CHECK(static_cast<double>(knots[i]) == want[i]);
      }
    }
  }

  const double unit = build_slack * value_error_units(space) * storage_eps(space);
  std::size_t n_trunc = 0;
  for (I64 dof = 0; dof < space.num_total_basis(); ++dof) {
    n_trunc += space.truncated(dof) != nullptr ? std::size_t{1} : std::size_t{0};
  }
  CHECK(n_trunc == g.trunc.size());
  for (const gold::TruncEntry& want : g.trunc) {
    const pantr::thb::TruncatedFunction* got = space.truncated(want.dof);
    REQUIRE(got != nullptr);
    CHECK(got->rep_level == want.rep);
    std::size_t size = 1;
    for (std::size_t k = 0; k < d; ++k) {
      CHECK(got->box_lo[k] == want.lo[k]);
      CHECK(static_cast<I64>(got->box_shape[k]) == want.shape[k]);
      size *= got->box_shape[k];
    }
    REQUIRE(got->coeffs.size() == size);
    for (std::size_t i = 0; i < size; ++i) {
      CHECK(std::abs(got->coeffs[i] - want.coeffs[i]) <= unit);
    }
  }

  for (const gold::EvalEntry& e : g.evals) {
    CAPTURE(e.cid);
    CAPTURE(e.order[0]);
    std::vector<double> pts;
    for (const auto& row : e.pts) {
      pts.insert(pts.end(), row, row + d);
    }
    const Ivec orders(e.order, e.order + d);
    const pantr::thb::TabulatedBasis got = tabulate(space, e.cid, pts, orders);
    REQUIRE(static_cast<I64>(got.dofs.size()) == e.k);
    REQUIRE(got.values.size() == static_cast<std::size_t>(3 * e.k));
    double scale = 1.0;
    for (I64 i = 0; i < 3 * e.k; ++i) {
      scale = (std::max)(scale, std::abs(e.values[i]));
    }
    for (I64 j = 0; j < e.k; ++j) {
      CHECK(got.dofs.data()[j] == e.dofs[j]);
    }
    for (I64 i = 0; i < 3 * e.k; ++i) {
      CHECK(std::abs(got.values.data()[i] - e.values[i]) <= unit * scale);
    }
  }
}

TEST_CASE("THBSplineSpace: spaces reproduce the Python implementation's values") {
  for (const GoldenCase& c : golden_cases()) {
    CAPTURE(c.name);
    if (c.single_precision) {
      check_against_golden(make_space<float>(c.setup), *c.data);
    } else {
      check_against_golden(make_space<double>(c.setup), *c.data);
    }
  }
}

TEST_CASE("THBSplineSpace: a float root keeps float level spaces and tabulates in double") {
  const THBSplineSpace<float> space = make_space<float>(setup_2d(true, 2, k2n));
  static_assert(std::is_same_v<decltype(space.level_space(0)), bs::BsplineSpace<float>>);
  static_assert(std::is_same_v<decltype(space.tabulate_basis(0, mdview<const double, 2>())),
                               pantr::thb::TabulatedBasis>);
  // The level-1 knots are the float32 root's, subdivided in double and rounded to float: not the
  // knots the same subdivision of the float64 root gives.
  const bs::BsplineSpace<float> level1 = space.level_space(1);
  const std::span<const float> knots = level1.spaces()[0].knots();
  const THBSplineSpace<double> wide = make_space<double>(setup_2d(true, 2, k2n));
  const bs::BsplineSpace<double> wide_level1 = wide.level_space(1);
  const std::span<const double> wide_knots = wide_level1.spaces()[0].knots();
  REQUIRE(knots.size() == wide_knots.size());
  double largest = 0.0;
  for (std::size_t i = 0; i < knots.size(); ++i) {
    largest = (std::max)(largest, std::abs(static_cast<double>(knots[i]) - wide_knots[i]));
  }
  CHECK(largest > 0.0);
  CHECK(largest <= static_cast<double>(pantr::tolerance::get_strict<float>()));
}

// ---- Brute-force oracles ----------------------------------------------------------------------

Ivec unravel(I64 flat, const Ivec& num_basis) {
  Ivec multi(num_basis.size());
  for (std::size_t k = num_basis.size(); k-- > 0;) {
    multi[k] = flat % num_basis[k];
    flat /= num_basis[k];
  }
  return multi;
}

I64 ravel(const std::span<const I64>& multi, const Ivec& num_basis) {
  I64 flat = 0;
  for (std::size_t k = 0; k < multi.size(); ++k) {
    flat = flat * num_basis[k] + multi[k];
  }
  return flat;
}

template <class T>
Ivec level_num_basis(const THBSplineSpace<T>& space, I64 level) {
  Ivec nb;
  const bs::BsplineSpace<T> level_space = space.level_space(level);
  for (const auto& dir : level_space.spaces()) {
    nb.push_back(static_cast<I64>(dir.num_basis()));
  }
  return nb;
}

/// The dofs of the active functions whose open support contains the midpoint of cell `cid`, from
/// the level spaces' knots alone: function `i` of a direction is non-zero on the open interval
/// `(t_i, t_{i + p + 1})`, and a cell of the grid lies inside one interval of every level space.
template <class T>
Ivec brute_active_basis(const THBSplineSpace<T>& space, I64 cid) {
  const auto [lo, hi] = space.grid().cell_bounds(cid);
  Ivec dofs;
  for (I64 level = 0; level < space.num_levels(); ++level) {
    const bs::BsplineSpace<T> ls = space.level_space(level);
    const Ivec nb = level_num_basis(space, level);
    const std::span<const I64> active = space.active_function_indices(level);
    for (std::size_t pos = 0; pos < active.size(); ++pos) {
      const Ivec multi = unravel(active[pos], nb);
      bool covers = true;
      for (std::size_t k = 0; k < multi.size(); ++k) {
        const std::span<const T> knots = ls.spaces()[k].knots();
        const auto i = static_cast<std::size_t>(multi[k]);
        const double mid = 0.5 * (lo[k] + hi[k]);
        covers = covers && static_cast<double>(knots[i]) < mid &&
                 mid < static_cast<double>(
                           knots[i + static_cast<std::size_t>(ls.spaces()[k].degree()) + 1]);
      }
      if (covers) {
        dofs.push_back(space.func_offset()[static_cast<std::size_t>(level)] +
                       static_cast<I64>(pos));
      }
    }
  }
  return dofs;
}

/// The Kraft selection from the definition and point location: a level-`l` function is active iff
/// every level-`l` cell of its support lies in a leaf of level `l` or finer (inside the level-`l`
/// subdomain) and at least one lies in a leaf of level exactly `l` (not inside the finer one).
template <class T>
std::vector<Ivec> brute_kraft(const THBSplineSpace<T>& space) {
  const gd::HierarchicalGrid grid = space.grid();
  const std::size_t d = space.dim();
  std::vector<Ivec> result;
  for (I64 level = 0; level < space.num_levels(); ++level) {
    const bs::BsplineSpace<T> ls = space.level_space(level);
    const Ivec nb = level_num_basis(space, level);
    I64 total = 1;
    for (const I64 n : nb) {
      total *= n;
    }
    Ivec selected;
    for (I64 flat = 0; flat < total; ++flat) {
      const Ivec multi = unravel(flat, nb);
      // The level-`level` intervals each direction's function covers, as (lo, hi) coordinates.
      std::vector<std::vector<std::pair<double, double>>> spans(d);
      for (std::size_t k = 0; k < d; ++k) {
        const auto& dir = ls.spaces()[k];
        const std::span<const T> knots = dir.knots();
        const std::span<const T> unique = dir.unique_knots_and_multiplicity(true).first;
        const auto i = static_cast<std::size_t>(multi[k]);
        const T first = knots[i];
        const T last = knots[i + static_cast<std::size_t>(dir.degree()) + 1];
        for (std::size_t j = 0; j + 1 < unique.size(); ++j) {
          if (first <= unique[j] && unique[j + 1] <= last) {
            spans[k].emplace_back(static_cast<double>(unique[j]),
                                  static_cast<double>(unique[j + 1]));
          }
        }
      }
      bool inside = true;
      bool at_level = false;
      std::vector<std::size_t> pos(d, 0);
      bool done = std::any_of(spans.begin(), spans.end(), [](const auto& s) { return s.empty(); });
      while (!done) {
        std::vector<double> mid(d);
        for (std::size_t k = 0; k < d; ++k) {
          mid[k] = 0.5 * (spans[k][pos[k]].first + spans[k][pos[k]].second);
        }
        const std::optional<I64> leaf = grid.locate(mid);
        REQUIRE(leaf.has_value());
        const I64 leaf_level = grid.cell_level(*leaf);
        inside = inside && leaf_level >= level;
        at_level = at_level || leaf_level == level;
        done = true;
        for (std::size_t k = d; k-- > 0;) {
          if (++pos[k] < spans[k].size()) {
            done = false;
            break;
          }
          pos[k] = 0;
        }
      }
      if (inside && at_level) {
        selected.push_back(flat);
      }
    }
    result.push_back(std::move(selected));
  }
  return result;
}

template <class T>
void check_selection_and_supports(const THBSplineSpace<T>& space) {
  const std::vector<Ivec> want = brute_kraft(space);
  I64 total = 0;
  REQUIRE(static_cast<I64>(want.size()) == space.num_levels());
  for (I64 level = 0; level < space.num_levels(); ++level) {
    const std::span<const I64> got = space.active_function_indices(level);
    const Ivec& expected = want[static_cast<std::size_t>(level)];
    CHECK(std::equal(got.begin(), got.end(), expected.begin(), expected.end()));
    total += static_cast<I64>(got.size());
    CHECK(space.func_offset()[static_cast<std::size_t>(level)] + static_cast<I64>(got.size()) ==
          space.func_offset()[static_cast<std::size_t>(level) + 1]);
  }
  CHECK(space.func_offset().front() == 0);
  CHECK(total == space.num_total_basis());
  const std::vector<I64> per_level = space.num_basis_per_level();
  CHECK(std::accumulate(per_level.begin(), per_level.end(), I64{0}) == space.num_total_basis());

  // The per-cell table: the support-overlap set by brute force, sorted, consistent with the
  // active lists, covering every function.
  const pantr::thb::CellIncidence inc = space.cell_dof_incidence();
  const I64 n_cells = space.grid().num_cells();
  REQUIRE(static_cast<I64>(inc.cell_ptr.size()) == n_cells + 1);
  CHECK(inc.cell_ptr.front() == 0);
  CHECK(static_cast<std::size_t>(inc.cell_ptr.back()) == inc.dofs.size());
  std::vector<unsigned char> seen(static_cast<std::size_t>(space.num_total_basis()), 0);
  I64 most = 0;
  for (I64 cid = 0; cid < n_cells; ++cid) {
    CAPTURE(cid);
    const pantr::thb::CellContributions c = space.contributions(cid);
    const Ivec brute = brute_active_basis(space, cid);
    CHECK(std::equal(c.dofs.begin(), c.dofs.end(), brute.begin(), brute.end()));
    const std::span<const I64> basis = space.active_basis(cid);
    CHECK(std::equal(basis.begin(), basis.end(), brute.begin(), brute.end()));
    CHECK(space.num_active_on(cid) == brute.size());
    CHECK(static_cast<std::size_t>(inc.cell_ptr[static_cast<std::size_t>(cid) + 1] -
                                   inc.cell_ptr[static_cast<std::size_t>(cid)]) == brute.size());
    most = (std::max)(most, static_cast<I64>(brute.size()));
    const I64 cell_level = space.grid().cell_level(cid);
    for (std::size_t i = 0; i < c.size(); ++i) {
      if (i > 0) {
        CHECK(c.dofs[i - 1] < c.dofs[i]);
      }
      const I64 level = c.levels[i];
      REQUIRE(0 <= level);
      REQUIRE(level <= cell_level);
      const Ivec nb = level_num_basis(space, level);
      const std::span<const I64> active = space.active_function_indices(level);
      const I64 pos = c.dofs[i] - space.func_offset()[static_cast<std::size_t>(level)];
      REQUIRE(0 <= pos);
      REQUIRE(pos < static_cast<I64>(active.size()));
      CHECK(active[static_cast<std::size_t>(pos)] == ravel(c.multi_index(i), nb));
      seen[static_cast<std::size_t>(c.dofs[i])] = 1;
    }
  }
  CHECK(space.max_active_per_cell() == most);
  CHECK(std::all_of(seen.begin(), seen.end(), [](unsigned char s) { return s != 0; }));
}

TEST_CASE("THBSplineSpace: Kraft selection and cell supports match a brute-force oracle") {
  SUBCASE("2D, two levels, THB") { check_selection_and_supports(make_space(setup_2d(true, 2))); }
  SUBCASE("2D, three levels, HB") { check_selection_and_supports(make_space(setup_2d(false, 3))); }
  SUBCASE("2D, factor (2, 1)") {
    check_selection_and_supports(make_space(setup_2d(true, 3, k2, {2, 1})));
  }
  SUBCASE("1D degree 3 with regularity 0") {
    check_selection_and_supports(make_space(setup_1d(true, 3)));
  }
  SUBCASE("3D") { check_selection_and_supports(make_space(setup_3d())); }
  SUBCASE("1D on an unclamped knot vector") {
    check_selection_and_supports(make_space(setup_unclamped()));
  }
  SUBCASE("float root") { check_selection_and_supports(make_space<float>(setup_2d(true, 3, k2n))); }
  SUBCASE("a refinement away from the corner, in the middle of the domain") {
    Setup s = setup_2d(true, 2);
    s.refinements = {{0, {1, 1}, {3, 3}}, {1, {3, 3}, {5, 5}}};
    check_selection_and_supports(make_space(s));
  }
  SUBCASE("two disjoint refinement blocks") {
    Setup s = setup_2d(true, 2);
    s.refinements = {{0, {0, 0}, {1, 1}}, {0, {3, 3}, {4, 4}}};
    check_selection_and_supports(make_space(s));
  }
}

TEST_CASE("THBSplineSpace: an unrefined grid is the tensor-product space") {
  Setup s = setup_2d(true, 2);
  s.refinements.clear();
  const THBSplineSpace<double> space = make_space(s);
  CHECK(space.num_levels() == 1);
  CHECK(space.num_total_basis() == 36);
  CHECK(space.max_active_per_cell() == 9);
  for (I64 dof = 0; dof < space.num_total_basis(); ++dof) {
    CHECK(space.truncated(dof) == nullptr);
  }
  check_selection_and_supports(space);
}

// ---- Partition of unity, non-negativity, derivatives ------------------------------------------

struct UnityReport {
  double worst_deviation = 0.0;  // max |sum - 1| over cells and points
  double smallest_value = 0.0;   // min over every value
};

/// The parameters of the points of a cell: its corners and a few interior ones.
std::vector<double> unity_parameters(std::size_t dim) {
  if (dim <= 2) {
    return {0.0, 0.17, 0.5, 0.83, 1.0};
  }
  return {0.0, 0.3, 1.0};
}

template <class T>
UnityReport measure_unity(const THBSplineSpace<T>& space) {
  UnityReport report;
  for (I64 cid = 0; cid < space.grid().num_cells(); ++cid) {
    const std::vector<double> pts = cell_points(space, cid, unity_parameters(space.dim()));
    const pantr::thb::TabulatedBasis tab = tabulate(space, cid, pts, Ivec(space.dim(), 0));
    const std::size_t k = tab.dofs.size();
    for (std::size_t i = 0; i < pts.size() / space.dim(); ++i) {
      double sum = 0.0;
      for (std::size_t j = 0; j < k; ++j) {
        const double v = tab.values.data()[i * k + j];
        sum += v;
        report.smallest_value = (std::min)(report.smallest_value, v);
      }
      report.worst_deviation = (std::max)(report.worst_deviation, std::abs(sum - 1.0));
    }
  }
  return report;
}

/// The bound on `|sum - 1|` for the sum of `n` values: every value has relative error at most
/// `value_error_units` (non-negative terms), so the sum, which is 1, has at most `units + n`.
///
/// The unit is the storage type's epsilon, not `double`'s: the two-scale coefficients of a
/// truncated function are computed in the root's type, so a float root's partition of unity holds
/// to float round-off, however accurate the double evaluation on top of it.
template <class T>
double unity_bound(const THBSplineSpace<T>& space) {
  return build_slack *
         (value_error_units(space) + static_cast<double>(space.max_active_per_cell())) *
         storage_eps(space);
}

TEST_CASE("THBSplineSpace: the truncated basis is a non-negative partition of unity") {
  const auto check = [](const auto& space) {
    const UnityReport r = measure_unity(space);
    CHECK(r.worst_deviation <= unity_bound(space));
    CHECK(r.smallest_value >= -build_slack * value_error_units(space) * storage_eps(space));
  };
  SUBCASE("2D, two levels") { check(make_space(setup_2d(true, 2))); }
  SUBCASE("2D, three levels") { check(make_space(setup_2d(true, 3))); }
  SUBCASE("2D, factor (2, 1)") { check(make_space(setup_2d(true, 3, k2, {2, 1}))); }
  SUBCASE("1D degree 3 with regularity 0") { check(make_space(setup_1d(true, 3))); }
  SUBCASE("3D") { check(make_space(setup_3d())); }
  SUBCASE("1D on an unclamped knot vector") { check(make_space(setup_unclamped())); }
  SUBCASE("float root") { check(make_space<float>(setup_2d(true, 2, k2n))); }
  SUBCASE("a refinement in the middle of the domain") {
    Setup s = setup_2d(true, 2);
    s.refinements = {{0, {1, 1}, {3, 3}}, {1, {3, 3}, {5, 5}}};
    check(make_space(s));
  }
}

TEST_CASE("THBSplineSpace: the plain hierarchical basis does not sum to one at an interface") {
  // The control of the test above: the same check, on the same grid without truncation, fails.
  const THBSplineSpace<double> thb = make_space(setup_2d(true, 2));
  const THBSplineSpace<double> hb = make_space(setup_2d(false, 2));
  CHECK(measure_unity(thb).worst_deviation <= unity_bound(thb));
  // A control threshold needs only to be far above the round-off level (`unity_bound`, a few
  // dozen eps) and far below the effect it detects (the weights of the coarse functions that
  // truncation would have removed, a sizeable fraction of one): 1e-2 is both, by orders of
  // magnitude on either side.
  CHECK(measure_unity(hb).worst_deviation > 1e-2);
  // Away from the interface, on the unrefined coarse corner, the two agree: the last cell of
  // level 0 is the farthest from the refined region.
  const I64 last = 11;
  CHECK(thb.grid().cell_level(last) == 0);
  const std::vector<double> pts = cell_points(hb, last, {0.5});
  const pantr::thb::TabulatedBasis h = tabulate(hb, last, pts, {0, 0});
  double sum = 0.0;
  for (std::size_t j = 0; j < h.dofs.size(); ++j) {
    sum += h.values.data()[j];
  }
  CHECK(std::abs(sum - 1.0) <= unity_bound(hb));
}

namespace {

/// The sum of the active functions of cell `cid` at the single point `pt`.
template <class T>
double basis_sum(const THBSplineSpace<T>& space, I64 cid, const std::vector<double>& pt) {
  const pantr::thb::TabulatedBasis tab = tabulate(space, cid, pt, Ivec(space.dim(), 0));
  double sum = 0.0;
  for (std::size_t j = 0; j < tab.dofs.size(); ++j) {
    sum += tab.values.data()[j];
  }
  return sum;
}

/// Check the partition of unity of a 1D space at both bounds of every cell.
template <class T>
void check_unity_at_every_edge_1d(const THBSplineSpace<T>& space) {
  for (I64 cid = 0; cid < space.grid().num_cells(); ++cid) {
    CAPTURE(cid);
    const auto [lo, hi] = space.grid().cell_bounds(cid);
    CHECK(std::abs(basis_sum(space, cid, {lo[0]}) - 1.0) <= unity_bound(space));
    CHECK(std::abs(basis_sum(space, cid, {hi[0]}) - 1.0) <= unity_bound(space));
  }
}

/// Count the bounds of a 1D space's cells that lie past the knot of multiplicity `degree + 1` they
/// stand for, on the neighboring cell's side: an upper bound above the knot ending the cell's
/// interval at its own level, or a lower bound below the knot starting it. The grid forms its
/// bounds and the level spaces their knots by different arithmetic, so the two can differ by
/// rounding; a test of such bounds needs at least one, or it does not exercise them.
template <class T>
I64 count_bounds_past_full_knots(const THBSplineSpace<T>& space) {
  I64 count = 0;
  const auto full = static_cast<I64>(space.level_space(0).spaces()[0].degree()) + 1;
  for (I64 cid = 0; cid < space.grid().num_cells(); ++cid) {
    const I64 level = space.grid().cell_level(cid);
    const auto at = static_cast<std::size_t>(space.grid().cell_multi_index(cid)[0]);
    const bs::BsplineSpace<T> level_space = space.level_space(level);
    const auto [knots, mult] = level_space.spaces()[0].unique_knots_and_multiplicity(true);
    const auto [lo, hi] = space.grid().cell_bounds(cid);
    // The domain's end knots have full multiplicity too, but the search keeps a point past them
    // on the end span, so only interior knots count.
    const bool interior_hi = at + 2 < knots.size();
    if (interior_hi && mult[at + 1] == full && hi[0] > static_cast<double>(knots[at + 1])) {
      ++count;
    }
    if (at > 0 && mult[at] == full && lo[0] < static_cast<double>(knots[at])) {
      ++count;
    }
  }
  return count;
}

}  // namespace

// Regression (present in the previous release as well): on a space of regularity -1 (a knot of full
// multiplicity at every breakpoint inserted by refinement) refined once, the basis tabulated at a
// cell's upper bound `hi` summed to 0 for the cells ending on a knot of multiplicity degree + 1:
// the 1D search, right-continuous at a knot, took the span to the right of the cell, whose
// functions are not the cell's. A point the search places in a span sharing no function with the
// cell's is now evaluated on the cell's span (the limit from inside).
TEST_CASE("inherited: regularity -1 basis vanishes at a cell's upper edge") {
  // Degree 2 on four equal spans, regularity -1, the two left root cells refined: cells 0 and 1
  // are the level-0 cells (indices 2 and 3), cells 2 to 5 the level-1 cells (indices 0 to 3). The
  // vanishing sum was at the upper edge of cells 2 and 4; every cell and both edges are checked.
  const THBSplineSpace<double> space =
      make_space(Setup{{k2}, {2}, {2}, {-1}, {{0, {0}, {2}}}, true});
  REQUIRE(space.grid().num_cells() == 6);
  check_unity_at_every_edge_1d(space);
}

TEST_CASE("THBSplineSpace: regularity -1 derivatives at a cell's upper edge are the left limit") {
  // On a cell every function is one polynomial piece, so its first derivative at `hi` is the limit
  // of the derivative at `hi - h`; with h = 1e-10 they differ by at most h |f''|. A quadratic
  // Bernstein function on a span of width w = 1/8 has |f''| <= 4 / w^2 = 256, and a truncated
  // function combines at most p + 1 = 3 of them with coefficients in [0, 1], so h |f''| <= 7.7e-8;
  // 1e-6 is that with an admitted margin. Before the fix the derivatives at `hi` of cells 2 and 4
  // were all zero.
  const THBSplineSpace<double> space =
      make_space(Setup{{k2}, {2}, {2}, {-1}, {{0, {0}, {2}}}, true});
  for (I64 cid = 0; cid < space.grid().num_cells(); ++cid) {
    CAPTURE(cid);
    const auto [lo, hi] = space.grid().cell_bounds(cid);
    // Order 0 as well, per function: |f'| <= 3 * 2 / w = 48, so h |f'| <= 4.8e-9.
    for (const I64 order : {I64{0}, I64{1}}) {
      CAPTURE(order);
      const pantr::thb::TabulatedBasis at = tabulate(space, cid, {hi[0]}, {order});
      const pantr::thb::TabulatedBasis in = tabulate(space, cid, {hi[0] - 1e-10}, {order});
      REQUIRE(at.dofs.size() == in.dofs.size());
      for (std::size_t j = 0; j < at.dofs.size(); ++j) {
        CAPTURE(j);
        CHECK(std::abs(at.values.data()[j] - in.values.data()[j]) <= 1e-6);
      }
    }
  }
}

TEST_CASE("THBSplineSpace: regularity -1 basis sums to one at every upper edge in 2D") {
  // A 2D space of regularity -1 in both directions with the lower-left 2 x 2 root cells refined:
  // at every cell, the upper edge in each direction (at the other's midpoint) and the upper corner.
  const THBSplineSpace<double> space =
      make_space(Setup{{k2, k2}, {2, 2}, {2, 2}, {-1, -1}, {{0, {0, 0}, {2, 2}}}, true});
  for (I64 cid = 0; cid < space.grid().num_cells(); ++cid) {
    CAPTURE(cid);
    const auto [lo, hi] = space.grid().cell_bounds(cid);
    const double m0 = 0.5 * (lo[0] + hi[0]);
    const double m1 = 0.5 * (lo[1] + hi[1]);
    CHECK(std::abs(basis_sum(space, cid, {hi[0], m1}) - 1.0) <= unity_bound(space));
    CHECK(std::abs(basis_sum(space, cid, {m0, hi[1]}) - 1.0) <= unity_bound(space));
    CHECK(std::abs(basis_sum(space, cid, {hi[0], hi[1]}) - 1.0) <= unity_bound(space));
  }
}

// The three cases below have cell bounds that differ from their knots by rounding: the grid forms
// a bound as `root_lo + i * size` (and the upper one as `lo + size`), the level spaces their knots
// by subdividing each span of the level above. The cell's span is found from its integer index, so
// a bound past its knot is evaluated on the cell's span like a bound equal to it. Their setups are
// shared with the derivative cases further down.

/// Degree 2, regularity -1, a subdivision factor of 3 (a third of a span is not a binary fraction).
Setup factor_3_setup() { return Setup{{k2}, {2}, {3}, {-1}, {{0, {2}, {4}}}, true}; }

/// Degree 2, regularity -1, unequal root spans refined twice (the second level's cells start at a
/// non-dyadic point).
Setup non_dyadic_twice_setup() {
  return Setup{{k2n}, {2}, {2}, {-1}, {{0, {0}, {4}}, {1, {2}, {7}}}, true};
}

/// Degree 2, regularity -1, a float root: the level spaces' knots are rounded to float while the
/// grid forms its bounds in double, so a bound differs from its knot by float round-off.
Setup float_root_setup() {
  return Setup{
      {{0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 1.0, 1.0, 1.0}}, {2}, {2}, {-1}, {{0, {1}, {4}}}, true};
}

TEST_CASE("THBSplineSpace: regularity -1 basis sums to one at every edge with a factor of 3") {
  const THBSplineSpace<double> space = make_space(factor_3_setup());
  REQUIRE(count_bounds_past_full_knots(space) > 0);
  check_unity_at_every_edge_1d(space);
}

TEST_CASE(
    "THBSplineSpace: regularity -1 basis sums to one at every edge of a non-dyadic root "
    "refined twice") {
  const THBSplineSpace<double> space = make_space(non_dyadic_twice_setup());
  REQUIRE(count_bounds_past_full_knots(space) > 0);
  check_unity_at_every_edge_1d(space);
}

TEST_CASE("THBSplineSpace: regularity -1 basis sums to one at every edge of a float root") {
  const THBSplineSpace<float> space = make_space<float>(float_root_setup());
  REQUIRE(count_bounds_past_full_knots(space) > 0);
  check_unity_at_every_edge_1d(space);
}

TEST_CASE(
    "THBSplineSpace: regularity -1 values at the edges of a cell below a finer "
    "representation are the limits from inside") {
  // Two levels of refinement on the left: the level-0 and the level-1 cells 2 and 3 are not
  // refined to level 2, but functions active on them are truncated against level 2 and evaluated
  // there, where the cell covers several knot intervals. At an upper bound on a knot of
  // multiplicity degree + 1 the search takes the level-2 span after the cell, where those
  // functions vanish; the point must be taken back into the cell. Each value at a bound is
  // compared with the value a step h = 1e-10 inside: the function is one polynomial piece on the
  // cell, so the two differ by at most h |f'|. On a level-2 span (width w = 1/16) at most
  // p + 1 = 3 quadratic B-splines are non-zero, each with |B'| <= 2 p / w = 64, and a truncated
  // function weighs them by coefficients in [0, 1], so h |f'| <= 1.9e-8; 1e-6 is that with an
  // admitted margin.
  const THBSplineSpace<double> space =
      make_space(Setup{{k2}, {2}, {2}, {-1}, {{0, {0}, {2}}, {1, {0}, {2}}}, true});
  bool finer = false;
  for (I64 cid = 0; cid < space.grid().num_cells(); ++cid) {
    for (const I64 dof : space.active_basis(cid)) {
      const pantr::thb::TruncatedFunction* f = space.truncated(dof);
      finer = finer || (f != nullptr && f->rep_level > space.grid().cell_level(cid));
    }
  }
  REQUIRE(finer);
  const double h = 1e-10;
  for (I64 cid = 0; cid < space.grid().num_cells(); ++cid) {
    CAPTURE(cid);
    const auto [lo, hi] = space.grid().cell_bounds(cid);
    for (const auto& [edge, inside] : {std::pair{lo[0], lo[0] + h}, std::pair{hi[0], hi[0] - h}}) {
      CAPTURE(edge);
      const pantr::thb::TabulatedBasis at = tabulate(space, cid, {edge}, {0});
      const pantr::thb::TabulatedBasis in = tabulate(space, cid, {inside}, {0});
      for (std::size_t j = 0; j < at.dofs.size(); ++j) {
        CAPTURE(j);
        CHECK(std::abs(at.values.data()[j] - in.values.data()[j]) <= 1e-6);
      }
    }
  }
  check_unity_at_every_edge_1d(space);
}

// ---- A bound is the cell's limit, order by order ---------------------------------------------

/// The rounding unit of one value of the space, `build_slack * value_error_units * eps`: the
/// `delta` of `check_derivatives`, whose comment derives it. The `eps` is the storage type's, as in
/// `unity_bound`: the recurrences hold their knot differences, values and coefficients in the
/// root's type (`bspline::detail::basis_derivs_point`), so a float root's derivatives carry float
/// round-off, which is the whole of the difference for a double root.
template <class T>
double value_delta(const THBSplineSpace<T>& space) {
  return build_slack * value_error_units(space) * storage_eps(space);
}

/// How the upper bound of a cell stands along one axis, at the cell's own level.
struct UpperBound {
  bool at_domain_end;  // the last knot span: the search keeps a point there on the cell's span
  bool full_knot;      // an interior knot of multiplicity `degree + 1`: the search leaves the cell
  /// Every derivative order at the bound is the cell's own: the search keeps the cell's span at the
  /// domain end, and the point is taken back to it at a knot of full multiplicity.
  [[nodiscard]] bool is_cells() const { return at_domain_end || full_knot; }
};

/// At a knot of lower multiplicity the search keeps the span to the right of the cell, so the
/// orders above the continuity there are that piece's (pinned in its own test below), and a grid
/// bound that differs from its knot by rounding moves the lower orders by the gap times the next
/// derivative: no bound of this file covers that, so such a bound is not compared order by order.
template <class T>
UpperBound upper_bound_of(const THBSplineSpace<T>& space, I64 cid, std::size_t axis) {
  const auto at = static_cast<std::size_t>(space.grid().cell_multi_index(cid)[axis]);
  const auto [knots, mult] = space.level_space(space.grid().cell_level(cid))
                                 .spaces()[axis]
                                 .unique_knots_and_multiplicity(true);
  const bool at_end = at + 2 >= knots.size();
  const I64 full = static_cast<I64>(space.degrees()[axis]) + 1;
  return UpperBound{at_end, !at_end && mult[at + 1] == full};
}

/// Check that the order-`order` derivative of every function of cell `cid` of a 1D space at `x` is
/// that of the cell's own polynomial piece, `x` being a bound of the cell.
///
/// Oracle: on a cell every function is a polynomial of degree at most `p`, so the derivatives at
/// the cell's midpoint `c`, which the 1D search places without doubt, fix it:
/// `f^(o)(x) = sum_{k=o}^{p} f^(k)(c) (x - c)^(k-o) / (k-o)!` exactly. Rounding: a derivative `a`
/// carries `delta (1 + |a|)`, the heuristic of `check_derivatives`, so the reference carries that
/// of each `f^(k)(c)` times its weight and the value under test its own.
template <class T>
void check_limit_from_inside(const THBSplineSpace<T>& space, I64 cid, double x, I64 order) {
  const double delta = value_delta(space);
  const auto [lo, hi] = space.grid().cell_bounds(cid);
  const double c = 0.5 * (lo[0] + hi[0]);
  const std::size_t n_fun = space.num_active_on(cid);
  std::vector<double> reference(n_fun, 0.0);
  std::vector<double> bound(n_fun, 0.0);
  double weight = 1.0;
  for (I64 k = order; k <= space.degrees()[0]; ++k) {
    weight *= k > order ? (x - c) / static_cast<double>(k - order) : 1.0;
    const pantr::thb::TabulatedBasis mid = tabulate(space, cid, {c}, {k});
    for (std::size_t j = 0; j < n_fun; ++j) {
      const double a = mid.values.data()[j];
      reference[j] += a * weight;
      bound[j] += delta * (1.0 + std::abs(a)) * std::abs(weight);
    }
  }
  const pantr::thb::TabulatedBasis at = tabulate(space, cid, {x}, {order});
  for (std::size_t j = 0; j < n_fun; ++j) {
    CAPTURE(cid);
    CAPTURE(x);
    CAPTURE(order);
    CAPTURE(j);
    const double got = at.values.data()[j];
    CAPTURE(got);
    CAPTURE(reference[j]);
    CHECK(std::abs(got - reference[j]) <= bound[j] + delta * (1.0 + std::abs(got)));
  }
}

/// Check every derivative order up to the degree at the bounds of every cell of a 1D space: the
/// lower bound always, the upper bound where `UpperBound::is_cells`. Returns how many upper bounds
/// on an interior knot of full multiplicity it compared, which a test requires to be positive.
template <class T>
I64 check_edge_derivatives_1d(const THBSplineSpace<T>& space) {
  I64 on_full_knots = 0;
  for (I64 cid = 0; cid < space.grid().num_cells(); ++cid) {
    const auto [lo, hi] = space.grid().cell_bounds(cid);
    const UpperBound upper = upper_bound_of(space, cid, 0);
    for (I64 order = 0; order <= space.degrees()[0]; ++order) {
      check_limit_from_inside(space, cid, lo[0], order);
      if (upper.is_cells()) {
        check_limit_from_inside(space, cid, hi[0], order);
      }
    }
    on_full_knots += upper.full_knot ? 1 : 0;
  }
  return on_full_knots;
}

TEST_CASE("THBSplineSpace: regularity -1 derivatives at every edge are the cell's, factor of 3") {
  const THBSplineSpace<double> space = make_space(factor_3_setup());
  REQUIRE(count_bounds_past_full_knots(space) > 0);
  CHECK(check_edge_derivatives_1d(space) > 0);
}

TEST_CASE(
    "THBSplineSpace: regularity -1 derivatives at every edge are the cell's, non-dyadic root "
    "refined twice") {
  const THBSplineSpace<double> space = make_space(non_dyadic_twice_setup());
  REQUIRE(count_bounds_past_full_knots(space) > 0);
  CHECK(check_edge_derivatives_1d(space) > 0);
}

TEST_CASE("THBSplineSpace: regularity -1 derivatives at every edge are the cell's, float root") {
  const THBSplineSpace<float> space = make_space<float>(float_root_setup());
  REQUIRE(count_bounds_past_full_knots(space) > 0);
  CHECK(check_edge_derivatives_1d(space) > 0);
}

/// Check the mixed partial `d^2 / dx dy` of every function of every cell of a 2D space at the
/// cell's upper corner, where the upper bound along each axis is an interior knot of full
/// multiplicity: the corner where the search leaves the cell in both directions, each taken back
/// separately. Returns how many cells that is, which a test requires to be positive.
///
/// Oracle: `check_limit_from_inside`'s, in two variables: the function is a polynomial of degree at
/// most `p_k` in direction `k` on the cell, so `d^{1,1} f(x, y) = sum_{a >= 1, b >= 1}
/// f^(a,b)(c) (x - c_0)^(a-1) (y - c_1)^(b-1) / ((a-1)! (b-1)!)`, from the derivatives at the
/// midpoint `c`. Rounding as there, with the weight of each term the product of its two factors.
template <class T>
I64 check_mixed_partial_at_upper_corners_2d(const THBSplineSpace<T>& space) {
  const double delta = value_delta(space);
  const auto p0 = static_cast<std::size_t>(space.degrees()[0]);
  const auto p1 = static_cast<std::size_t>(space.degrees()[1]);
  I64 corners = 0;
  for (I64 cid = 0; cid < space.grid().num_cells(); ++cid) {
    if (!upper_bound_of(space, cid, 0).full_knot || !upper_bound_of(space, cid, 1).full_knot) {
      continue;
    }
    ++corners;
    CAPTURE(cid);
    const auto [lo, hi] = space.grid().cell_bounds(cid);
    const double c0 = 0.5 * (lo[0] + hi[0]);
    const double c1 = 0.5 * (lo[1] + hi[1]);
    std::vector<double> wx(p0 + 1, 1.0);  // wx[a] = (hi[0] - c0)^(a-1) / (a-1)!
    std::vector<double> wy(p1 + 1, 1.0);
    for (std::size_t a = 2; a <= p0; ++a) {
      wx[a] = wx[a - 1] * (hi[0] - c0) / static_cast<double>(a - 1);
    }
    for (std::size_t b = 2; b <= p1; ++b) {
      wy[b] = wy[b - 1] * (hi[1] - c1) / static_cast<double>(b - 1);
    }
    const std::size_t n_fun = space.num_active_on(cid);
    std::vector<double> reference(n_fun, 0.0);
    std::vector<double> bound(n_fun, 0.0);
    for (std::size_t a = 1; a <= p0; ++a) {
      for (std::size_t b = 1; b <= p1; ++b) {
        const pantr::thb::TabulatedBasis mid =
            tabulate(space, cid, {c0, c1}, {static_cast<I64>(a), static_cast<I64>(b)});
        for (std::size_t j = 0; j < n_fun; ++j) {
          const double v = mid.values.data()[j];
          reference[j] += v * wx[a] * wy[b];
          bound[j] += delta * (1.0 + std::abs(v)) * std::abs(wx[a] * wy[b]);
        }
      }
    }
    const pantr::thb::TabulatedBasis at = tabulate(space, cid, {hi[0], hi[1]}, {1, 1});
    for (std::size_t j = 0; j < n_fun; ++j) {
      const double got = at.values.data()[j];
      CAPTURE(j);
      CAPTURE(got);
      CAPTURE(reference[j]);
      CHECK(std::abs(got - reference[j]) <= bound[j] + delta * (1.0 + std::abs(got)));
    }
  }
  return corners;
}

TEST_CASE("THBSplineSpace: regularity -1 mixed partial at an upper corner is the cell's") {
  // The 2D space of the unity test above. Before the fix the mixed partial at a corner where both
  // bounds are knots of full multiplicity was zero.
  const THBSplineSpace<double> space =
      make_space(Setup{{k2, k2}, {2, 2}, {2, 2}, {-1, -1}, {{0, {0, 0}, {2, 2}}}, true});
  CHECK(check_mixed_partial_at_upper_corners_2d(space) > 0);
}

/// The cells of the finest level of `space` whose upper bound is 0.5, a dyadic point of every
/// setup of the test below.
template <class T>
std::vector<I64> finest_cells_ending_at_half(const THBSplineSpace<T>& space) {
  std::vector<I64> cells;
  for (I64 cid = 0; cid < space.grid().num_cells(); ++cid) {
    if (space.grid().cell_level(cid) == space.num_levels() - 1 &&
        std::abs(space.grid().cell_bounds(cid).second[0] - 0.5) <= eps) {
      cells.push_back(cid);
    }
  }
  return cells;
}

/// Count the pairs of a cell and an untruncated function of a level coarser than the cell, active
/// on it, whose level has an interior knot of multiplicity `degree + 1` at the cell's upper bound:
/// the pairs for which `tabulate_1d` meets a coarser level whose search leaves the cell's span. The
/// bounds and knots of the spaces that use it are dyadic, so equal up to `eps`.
template <class T>
I64 count_coarser_full_knot_bounds(const THBSplineSpace<T>& space) {
  I64 count = 0;
  const auto full = static_cast<I64>(space.degrees()[0]) + 1;
  for (I64 cid = 0; cid < space.grid().num_cells(); ++cid) {
    const I64 cell_level = space.grid().cell_level(cid);
    const double hi = space.grid().cell_bounds(cid).second[0];
    const pantr::thb::CellContributions cell = space.contributions(cid);
    for (std::size_t j = 0; j < cell.size(); ++j) {
      if (cell.levels[j] >= cell_level || space.truncated(cell.dofs[j]) != nullptr) {
        continue;
      }
      const auto [knots, mult] =
          space.level_space(cell.levels[j]).spaces()[0].unique_knots_and_multiplicity(true);
      for (std::size_t i = 1; i + 1 < knots.size(); ++i) {
        count += (mult[i] == full && std::abs(static_cast<double>(knots[i]) - hi) <= eps) ? 1 : 0;
      }
    }
  }
  return count;
}

/// The column of cell `cid`'s tabulation that holds the function of level `level` and index
/// `index` (1D); the test requires it to exist.
template <class T>
std::size_t column_of(const THBSplineSpace<T>& space, I64 cid, I64 level, I64 index) {
  const pantr::thb::CellContributions cell = space.contributions(cid);
  for (std::size_t j = 0; j < cell.size(); ++j) {
    if (cell.levels[j] == level && cell.multi_index(j)[0] == index) {
      return j;
    }
  }
  return cell.size();
}

// A root whose knot 0.5 has multiplicity 3 = degree + 1: the root is discontinuous there, and its
// functions N1 (knots 0, 0, 1/4, 1/2) and N2 (0, 1/4, 1/2, 1/2) end at it from the left. With the
// root cell [1/4, 1/2] refined, N1 (and N2 in the plain hierarchical basis) is an untruncated
// function of level 0 active on the level-1 cell [3/8, 1/2], whose upper bound is that knot: a
// level coarser than the cell, where the search puts 1/2 on the root span [1/2, 3/4], which none of
// them is supported on, so every derivative would come out zero. The cell's ancestor span is
// [1/4, 1/2]; the Cox-de Boor recurrence worked by hand on it gives
//   N1 = 8 (1/2 - x)^2,   N2 = (1/2 - x) (24 x - 4),
// so at x = 1/2 the orders 0, 1, 2 are N1: 0, 0, 16 and N2: 0, -8, -48. The sum of the truncated
// basis is 1, so its first derivatives sum to zero there.
TEST_CASE("THBSplineSpace: a bound on a coarser level's full knot gets the cell's limit") {
  const std::vector<double> knots = {0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0};
  const auto check = [&](bool truncate, std::vector<Refine> refinements, bool plain_n2) {
    const THBSplineSpace<double> space =
        make_space(Setup{{knots}, {2}, {2}, {std::nullopt}, std::move(refinements), truncate});
    REQUIRE(count_coarser_full_knot_bounds(space) > 0);
    const std::vector<I64> cells = finest_cells_ending_at_half(space);
    REQUIRE(cells.size() == 1);
    const I64 cid = cells[0];
    const double delta = value_delta(space);
    const double hi = space.grid().cell_bounds(cid).second[0];
    // The analytic derivatives of the function of level 0 and index `index` at the bound.
    const auto expect = [&](I64 index, const std::vector<double>& want) {
      const std::size_t j = column_of(space, cid, 0, index);
      REQUIRE(j < space.num_active_on(cid));
      REQUIRE(space.truncated(space.active_basis(cid)[j]) == nullptr);
      for (I64 order = 0; order < 3; ++order) {
        CAPTURE(index);
        CAPTURE(order);
        const double got = tabulate(space, cid, {hi}, {order}).values.data()[j];
        CHECK(std::abs(got - want[static_cast<std::size_t>(order)]) <=
              delta * (1.0 + std::abs(want[static_cast<std::size_t>(order)])));
      }
    };
    expect(1, {0.0, 0.0, 16.0});
    if (plain_n2) {
      expect(2, {0.0, -8.0, -48.0});
    }
    if (truncate) {
      // The partition of unity and its derivative, at the bound and inside the cell, where a wrong
      // map from the cell to its level-0 span shows as well.
      const double mid = 0.5 * (space.grid().cell_bounds(cid).first[0] + hi);
      CHECK(std::abs(basis_sum(space, cid, {hi}) - 1.0) <= unity_bound(space));
      CHECK(std::abs(basis_sum(space, cid, {mid}) - 1.0) <= unity_bound(space));
      const pantr::thb::TabulatedBasis slope = tabulate(space, cid, {hi}, {1});
      double sum = 0.0;
      double magnitude = 0.0;
      for (std::size_t j = 0; j < slope.dofs.size(); ++j) {
        sum += slope.values.data()[j];
        magnitude += std::abs(slope.values.data()[j]);
      }
      CHECK(std::abs(sum) <= delta * (1.0 + magnitude));
    }
    CHECK(check_edge_derivatives_1d(space) > 0);
  };
  SUBCASE("THB, one level below") { check(true, {{0, {1}, {2}}}, false); }
  SUBCASE("HB, one level below") { check(false, {{0, {1}, {2}}}, true); }
  SUBCASE("THB, two levels below") { check(true, {{0, {1}, {2}}, {1, {3}, {4}}}, false); }
}

// An upper bound on a knot of lower multiplicity than degree + 1 stays where the search puts it,
// on the span to the right of the cell, so a derivative of an order above the continuity there is
// that piece's (documented at `THBSplineSpace::tabulate_basis`); a function of the cell the right
// piece lacks has derivative zero there, as the function itself has to the right of its support.
// A knot of multiplicity `m` puts the first functions of the two spans `m` apart, and the span is
// taken back to the cell's only above `degree`: the two cases below are the multiplicities just
// under it for degree 2, `m = 1` (C^1) and `m = 2` (C^0), so a threshold lowered to `degree` (which
// fires at `m = 2`) or to 1 (which fires at both) changes the answer.
TEST_CASE(
    "THBSplineSpace: a derivative above the continuity at an upper bound is the right piece's") {
  const auto check = [](const std::vector<double>& knots, I64 cid, I64 order,
                        const std::vector<double>& want) {
    const THBSplineSpace<double> space =
        make_space(Setup{{knots}, {2}, {2}, {std::nullopt}, {}, true});
    const double hi = space.grid().cell_bounds(cid).second[0];
    const pantr::thb::TabulatedBasis tab = tabulate(space, cid, {hi}, {order});
    REQUIRE(tab.dofs.size() == want.size());
    for (std::size_t j = 0; j < want.size(); ++j) {
      CAPTURE(j);
      CHECK(std::abs(tab.values.data()[j] - want[j]) <=
            value_delta(space) * (1.0 + std::abs(want[j])));
    }
  };
  SUBCASE("a simple knot, C^1: order 2 of cell 0 at 1/4") {
    // k2 = {0, 0, 0, 1/4, 1/2, ...}: cell 0 holds N0, N1, N2. On the cell (width w = 1/4)
    // N0 = (1 - x / w)^2 and N2 = (x / w)^2 / 2, so N0'' = 2 / w^2 = 32 and N2'' = 16, and N1 =
    // 1 - N0 - N2 gives -48: the cell's own piece is (32, -48, 16). On [1/4, 1/2], N0 is absent,
    // N1 = 8 (1/2 - x)^2 gives 16, and N2 = (1 + 2 s - 2 s^2) / 2 with s = 4 (x - 1/4) gives
    // -2 / w^2 = -32: the right piece is (0, 16, -32).
    check(k2, 0, 2, {0.0, 16.0, -32.0});
  }
  SUBCASE("a double knot, C^0: orders 0, 1 and 2 of cell 1 at 1/2") {
    // {0, 0, 0, 1/4, 1/2, 1/2, 3/4, 1, 1, 1}: cell 1 = [1/4, 1/2] holds N1, N2, N3 and its upper
    // bound is a knot of multiplicity 2, where the values are continuous: N1 = N2 = 0, N3 = 1
    // (the knot window 1/4, 1/2, 1/2, 3/4 peaks there). Order 1: the right piece of N3 is
    // 16 (3/4 - x)^2, slope -8, and N1, N2 vanish to the right; the cell's own piece of N3 is
    // 16 (x - 1/4)^2, slope +8, with N1 and N2 not zero. Order 2: N3'' = 32 on both pieces, and
    // N1, N2 absent to the right.
    const std::vector<double> knots = {0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0};
    check(knots, 1, 0, {0.0, 0.0, 1.0});
    check(knots, 1, 1, {0.0, 0.0, -8.0});
    check(knots, 1, 2, {0.0, 0.0, 32.0});
  }
}

TEST_CASE("THBSplineSpace: degree 0 basis is the cell's indicator at both bounds of every cell") {
  // Every knot of a degree-0 space has multiplicity 1 = degree + 1, so every upper bound of a cell
  // is a knot of full multiplicity: the search takes the span to the right, whose function is not
  // the cell's, and the basis summed to 0 at the upper bound of every cell short of the domain's
  // end. Three levels, so cells of every level.
  const THBSplineSpace<double> space = make_space(Setup{{{0.0, 0.25, 0.5, 0.75, 1.0}},
                                                        {0},
                                                        {2},
                                                        {std::nullopt},
                                                        {{0, {0}, {2}}, {1, {0}, {2}}},
                                                        true});
  REQUIRE(space.grid().num_cells() == 8);
  check_unity_at_every_edge_1d(space);
}

/// Derivatives against central finite differences of the values.
///
/// On a cell every function is a polynomial of degree at most `p` in each variable, so a central
/// difference with step `h` has the truncation error `h^2 / 6 * |f'''|` and the four-point mixed
/// difference is exact when `p <= 2`. `|f'''|` is read from the derivative of order three, which
/// is zero below degree 3 (and exercises an order above the degree). The rounding error of a value
/// is `delta = build_slack * value_error_units * eps` (every value is at most 1), so the
/// difference quotient carries `delta / h` (`delta / (h1 h2)` for the mixed one). The analytic
/// derivative's own rounding is taken as `delta * (1 + |a|)`, an admitted heuristic and not a bound
/// in `sum |terms|` (the terms of the derivative recurrence are not exposed, and instrumenting it
/// would be most of the implementation): each term is a value over a knot difference, so the
/// error scales with the derivative's size as `|a|` does, and the dominant `delta / h` is 64 /
/// width times larger than `delta` itself. With `h = width / 64` at parameters 0.3 and 0.6 the
/// points stay in the cell.
template <class T>
void check_derivatives(const THBSplineSpace<T>& space) {
  const std::size_t d = space.dim();
  const double delta = value_delta(space);
  const bool mixed_exact =
      d >= 2 && *std::max_element(space.degrees().begin(), space.degrees().end()) <= 2;
  for (I64 cid = 0; cid < space.grid().num_cells(); ++cid) {
    CAPTURE(cid);
    const auto [lo, hi] = space.grid().cell_bounds(cid);
    const std::vector<double> pts = cell_points(space, cid, {0.3, 0.6});
    const std::size_t n_pts = pts.size() / d;
    const std::size_t n_fun = space.num_active_on(cid);
    const auto shifted = [&](std::size_t axis_a, double sa, std::size_t axis_b, double sb) {
      std::vector<double> q = pts;
      for (std::size_t i = 0; i < n_pts; ++i) {
        q[i * d + axis_a] += sa * (hi[axis_a] - lo[axis_a]) / 64.0;
        if (axis_b != axis_a) {
          q[i * d + axis_b] += sb * (hi[axis_b] - lo[axis_b]) / 64.0;
        }
      }
      return q;
    };
    for (std::size_t k = 0; k < d; ++k) {
      const double h = (hi[k] - lo[k]) / 64.0;
      Ivec e(d, 0);
      e[k] = 1;
      Ivec e3(d, 0);
      e3[k] = 3;
      const auto analytic = tabulate(space, cid, pts, e);
      const auto third = tabulate(space, cid, pts, e3);
      const auto plus = tabulate(space, cid, shifted(k, 1.0, k, 0.0), Ivec(d, 0));
      const auto minus = tabulate(space, cid, shifted(k, -1.0, k, 0.0), Ivec(d, 0));
      for (std::size_t i = 0; i < n_pts; ++i) {
        double derivative_sum = 0.0;
        double magnitude_sum = 0.0;
        for (std::size_t j = 0; j < n_fun; ++j) {
          const std::size_t at = i * n_fun + j;
          const double fd = (plus.values.data()[at] - minus.values.data()[at]) / (2.0 * h);
          const double a = analytic.values.data()[at];
          const double bound = delta / h + h * h / 6.0 * std::abs(third.values.data()[at]) +
                               delta * (1.0 + std::abs(a));
          CHECK(std::abs(a - fd) <= bound);
          derivative_sum += a;
          magnitude_sum += std::abs(a);
        }
        // The truncated functions sum to one, so their derivatives sum to zero (the plain
        // hierarchical ones do not, near an interface).
        if (space.truncate()) {
          CHECK(std::abs(derivative_sum) <= delta * (1.0 + magnitude_sum));
        }
      }
    }
    if (mixed_exact) {
      const double h0 = (hi[0] - lo[0]) / 64.0;
      const double h1 = (hi[1] - lo[1]) / 64.0;
      Ivec e(d, 0);
      e[0] = 1;
      e[1] = 1;
      const auto analytic = tabulate(space, cid, pts, e);
      const auto pp = tabulate(space, cid, shifted(0, 1.0, 1, 1.0), Ivec(d, 0));
      const auto pm = tabulate(space, cid, shifted(0, 1.0, 1, -1.0), Ivec(d, 0));
      const auto mp = tabulate(space, cid, shifted(0, -1.0, 1, 1.0), Ivec(d, 0));
      const auto mm = tabulate(space, cid, shifted(0, -1.0, 1, -1.0), Ivec(d, 0));
      for (std::size_t at = 0; at < n_pts * n_fun; ++at) {
        const double fd = (pp.values.data()[at] - pm.values.data()[at] - mp.values.data()[at] +
                           mm.values.data()[at]) /
                          (4.0 * h0 * h1);
        const double a = analytic.values.data()[at];
        CHECK(std::abs(a - fd) <= delta / (h0 * h1) + delta * (1.0 + std::abs(a)));
      }
    }
  }
}

TEST_CASE("THBSplineSpace: derivatives match central finite differences of the values") {
  SUBCASE("2D THB, three levels") { check_derivatives(make_space(setup_2d(true, 3))); }
  SUBCASE("2D HB, two levels") { check_derivatives(make_space(setup_2d(false, 2))); }
  SUBCASE("1D degree 3 with regularity 0, THB") {
    check_derivatives(make_space(setup_1d(true, 3)));
  }
  SUBCASE("3D THB") { check_derivatives(make_space(setup_3d())); }
}

TEST_CASE("THBSplineSpace: an order above the degree gives zero, on every function") {
  const THBSplineSpace<double> space = make_space(setup_2d(true, 2));
  for (I64 cid = 0; cid < space.grid().num_cells(); ++cid) {
    const std::vector<double> pts = cell_points(space, cid, {0.2, 0.9});
    for (const Ivec& orders : {Ivec{3, 0}, Ivec{0, 3}, Ivec{1000000, 1}}) {
      const pantr::thb::TabulatedBasis tab = tabulate(space, cid, pts, orders);
      for (std::size_t i = 0; i < tab.values.size(); ++i) {
        CHECK(tab.values.data()[i] == 0.0);
      }
    }
  }
}

// ---- Per-cell table: threads, concurrency
// --------------------------------------------------------

/// A space with enough cells that the table's work (levels x prod(degree + 1) x (dim + log2
/// active) per cell, thousands of cells) is far above `par::parallel_min_ops`, so more than one
/// thread takes part.
Setup big_setup() {
  Setup s{{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}, {2, 2}, {2, 2},
          {std::nullopt, std::nullopt},       {},     true};
  std::vector<double> knots(3, 0.0);
  for (int i = 1; i < 48; ++i) {
    knots.push_back(static_cast<double>(i) / 48.0);
  }
  knots.insert(knots.end(), 3, 1.0);
  s.knots = {knots, knots};
  s.refinements = {{0, {0, 0}, {24, 24}}, {1, {0, 0}, {24, 24}}, {2, {10, 10}, {30, 30}}};
  return s;
}

TEST_CASE("THBSplineSpace: the per-cell table does not depend on the thread count") {
  const Setup s = big_setup();
  check_many_threads_are_used();
  const std::size_t threads = eight_threads();
  const THBSplineSpace<double> one = [&] {
    const ThreadsGuard guard(1);
    THBSplineSpace<double> space = make_space(s);
    static_cast<void>(space.max_active_per_cell());
    return space;
  }();
  const THBSplineSpace<double> many = [&] {
    const ThreadsGuard guard(threads);
    THBSplineSpace<double> space = make_space(s);
    static_cast<void>(space.max_active_per_cell());
    return space;
  }();
  REQUIRE(one.grid().num_cells() == many.grid().num_cells());
  REQUIRE(one.grid().num_cells() > 3000);
  const pantr::thb::CellIncidence a = one.cell_dof_incidence();
  const pantr::thb::CellIncidence b = many.cell_dof_incidence();
  CHECK(std::equal(a.cell_ptr.begin(), a.cell_ptr.end(), b.cell_ptr.begin(), b.cell_ptr.end()));
  CHECK(std::equal(a.dofs.begin(), a.dofs.end(), b.dofs.begin(), b.dofs.end()));
  CHECK(one.max_active_per_cell() == many.max_active_per_cell());
  bool same_levels_and_multi = true;
  for (I64 cid = 0; cid < one.grid().num_cells(); ++cid) {
    const pantr::thb::CellContributions x = one.contributions(cid);
    const pantr::thb::CellContributions y = many.contributions(cid);
    same_levels_and_multi =
        same_levels_and_multi &&
        std::equal(x.levels.begin(), x.levels.end(), y.levels.begin(), y.levels.end()) &&
        std::equal(x.multi.begin(), x.multi.end(), y.multi.begin(), y.multi.end());
  }
  CHECK(same_levels_and_multi);
  // The table of the largest case still agrees with the brute-force support overlap on a sample.
  for (I64 cid = 0; cid < one.grid().num_cells(); cid += 97) {
    const Ivec brute = brute_active_basis(one, cid);
    const std::span<const I64> got = one.active_basis(cid);
    CHECK(std::equal(got.begin(), got.end(), brute.begin(), brute.end()));
  }
}

TEST_CASE("THBSplineSpace: threads racing on the first use of the lazy table agree") {
  // A TSan target: the table is computed once, under the lazy block's lock, and every thread then
  // reads the same block.
  constexpr int n_threads = 8;
  const THBSplineSpace<double> reference = make_space(setup_2d(true, 3));
  const std::size_t n_cells = static_cast<std::size_t>(reference.grid().num_cells());
  std::vector<I64> expected(n_cells);
  for (std::size_t c = 0; c < n_cells; ++c) {
    expected[c] = static_cast<I64>(reference.num_active_on(static_cast<I64>(c)));
  }
  for (int round = 0; round < 10; ++round) {
    const THBSplineSpace<double> space = make_space(setup_2d(true, 3));
    std::atomic<int> ready{0};
    std::vector<std::vector<I64>> seen(n_threads, std::vector<I64>(n_cells, -1));
    std::vector<I64> maxima(n_threads, -1);
    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; ++t) {
      threads.emplace_back([&, t] {
        ready.fetch_add(1);
        while (ready.load() < n_threads) {
        }
        for (std::size_t c = 0; c < n_cells; ++c) {
          const std::size_t j = (c + static_cast<std::size_t>(t) * 7) % n_cells;
          seen[static_cast<std::size_t>(t)][j] =
              static_cast<I64>(space.active_basis(static_cast<I64>(j)).size());
        }
        maxima[static_cast<std::size_t>(t)] = space.max_active_per_cell();
      });
    }
    for (auto& th : threads) {
      th.join();
    }
    for (int t = 0; t < n_threads; ++t) {
      CHECK(seen[static_cast<std::size_t>(t)] == expected);
      CHECK(maxima[static_cast<std::size_t>(t)] == reference.max_active_per_cell());
    }
  }
}

TEST_CASE("THBSplineSpace: copies share the representation") {
  const THBSplineSpace<double> a = make_space(setup_2d(true, 2));
  const THBSplineSpace<double> b = a;  // NOLINT(performance-unnecessary-copy-initialization)
  CHECK(a.rep() == b.rep());
  CHECK(a.rep().use_count() >= 2);
  CHECK(a.grid().num_cells() == b.grid().num_cells());
  // A table computed through one copy is the table of the other.
  CHECK(a.active_basis(3).data() == b.active_basis(3).data());
}

// ---- Tabulation interface
// ------------------------------------------------------------------------

TEST_CASE("THBSplineSpace: the out overload writes what the returning overload returns") {
  const THBSplineSpace<double> space = make_space(setup_2d(true, 3));
  for (const I64 cid : {I64{0}, I64{15}, I64{39}}) {
    const std::vector<double> pts = cell_points(space, cid, {0.0, 0.4, 1.0});
    const Ivec orders = {1, 0};
    const pantr::thb::TabulatedBasis want = tabulate(space, cid, pts, orders);
    const std::size_t n = pts.size() / 2;
    const std::size_t k = space.num_active_on(cid);
    std::vector<double> values(n * k, -7.0);
    std::vector<I64> dofs(k, -7);
    space.tabulate_basis(cid, rows_of(pts, 2), orders, mdview<double, 2>(values.data(), n, k),
                         dofs);
    CHECK(std::equal(values.begin(), values.end(), want.values.data()));
    CHECK(std::equal(dofs.begin(), dofs.end(), want.dofs.data()));
    // Without orders: the values, as the orders-all-zero call.
    std::vector<double> plain(n * k, -7.0);
    space.tabulate_basis(cid, rows_of(pts, 2), mdview<double, 2>(plain.data(), n, k), dofs);
    const pantr::thb::TabulatedBasis zero = tabulate(space, cid, pts, {0, 0});
    CHECK(std::equal(plain.begin(), plain.end(), zero.values.data()));
    const pantr::thb::TabulatedBasis returned = space.tabulate_basis(cid, rows_of(pts, 2));
    CHECK(std::equal(returned.values.data(), returned.values.data() + returned.values.size(),
                     zero.values.data()));
    const std::span<const I64> active = space.active_basis(cid);
    CHECK(std::equal(dofs.begin(), dofs.end(), active.begin(), active.end()));
  }
  // No points: an empty table with the right width.
  const std::vector<double> none;
  const pantr::thb::TabulatedBasis empty = tabulate(space, 0, none, {0, 0});
  CHECK(empty.values.shape()[0] == 0);
  CHECK(empty.values.shape()[1] == space.num_active_on(0));
  CHECK(empty.dofs.size() == space.num_active_on(0));
}

TEST_CASE("THBSplineSpace: points at the slack edge of a boundary cell, on a one-piece root") {
  // The unusual inputs, for the sanitizer builds: points a fraction of the membership slack
  // outside the domain end (the 1D kernel is run without its own domain check there), on a root
  // that is a single Bezier piece (the Bernstein path), orders at and above the degree, and a cell
  // far from the origin where the slack is largest.
  const std::vector<double> bez = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
  Setup s{{bez, bez}, {2, 2}, {2, 2}, {std::nullopt, std::nullopt}, {{0, {0, 0}, {1, 1}}}, true};
  const THBSplineSpace<double> space = make_space(s);
  REQUIRE(space.num_levels() == 2);
  for (I64 cid = 0; cid < space.grid().num_cells(); ++cid) {
    const auto [lo, hi] = space.grid().cell_bounds(cid);
    const double slack0 = pantr::thb::cell_membership_tolerance(lo[0], hi[0]);
    const double slack1 = pantr::thb::cell_membership_tolerance(lo[1], hi[1]);
    for (const double sign : {-1.0, 1.0}) {
      const std::vector<double> pts = {
          sign < 0 ? lo[0] - 0.5 * slack0 : hi[0] + 0.5 * slack0,
          sign < 0 ? lo[1] - 0.5 * slack1 : hi[1] + 0.5 * slack1,
      };
      for (const Ivec& orders : {Ivec{0, 0}, Ivec{1, 1}, Ivec{2, 0}, Ivec{3, 3}}) {
        const pantr::thb::TabulatedBasis tab = tabulate(space, cid, pts, orders);
        for (std::size_t i = 0; i < tab.values.size(); ++i) {
          CHECK(std::isfinite(tab.values.data()[i]));
        }
        if (orders[0] == 0 && orders[1] == 0) {
          double sum = 0.0;
          for (std::size_t i = 0; i < tab.values.size(); ++i) {
            sum += tab.values.data()[i];
          }
          // Half a slack (4 eps of the cell's scale) outside the end: the piece's polynomial still
          // sums to one, and the point only moves each affine weight of the recurrence by that
          // much, so the sum errs by rounding, which `unity_bound` bounds.
          CHECK(std::abs(sum - 1.0) <= unity_bound(space));
        }
      }
    }
  }
  // A domain far from the origin, where the slack (8 eps of the cell's own magnitude) is largest.
  const std::vector<double> knots = {1e6, 1e6, 1e6, 1e6 + 1.0, 1e6 + 2.0, 1e6 + 2.0, 1e6 + 2.0};
  Setup shifted{{knots}, {2}, {2}, {std::nullopt}, {{0, {0}, {1}}}, true};
  const THBSplineSpace<double> remote = make_space(shifted);
  for (I64 cid = 0; cid < remote.grid().num_cells(); ++cid) {
    const auto [lo, hi] = remote.grid().cell_bounds(cid);
    const double slack = pantr::thb::cell_membership_tolerance(lo[0], hi[0]);
    CHECK(slack > 1e6 * eps);
    const std::vector<double> pts = {lo[0], 0.5 * (lo[0] + hi[0]), hi[0] + 0.5 * slack};
    const pantr::thb::TabulatedBasis tab = tabulate(remote, cid, pts, {0});
    for (std::size_t i = 0; i < tab.values.size(); ++i) {
      CHECK(std::isfinite(tab.values.data()[i]));
    }
  }
}

// ---- Validation --------------------------------------------------------------------------------

TEST_CASE("THBSplineSpace: the constructor refuses what the Python implementation refuses") {
  const Setup s = setup_2d(true, 2);
  const bs::BsplineSpace<double> root = make_root<double>(s);
  const gd::HierarchicalGrid grid = make_grid(root, s);
  const auto build = [&](const bs::BsplineSpace<double>& r, const gd::HierarchicalGrid& g,
                         Reg reg) { return THBSplineSpace<double>(r, g, true, std::move(reg)); };
  const Reg none = {std::nullopt, std::nullopt};

  SUBCASE("grid and root of another dimension") {
    const Setup one = setup_1d(true, 2);
    const bs::BsplineSpace<double> root1 = make_root<double>(one);
    THB_THROWS_WITH_AS(build(root1, grid, {std::nullopt}),
                       "grid.ndim (2) must equal root_space.dim (1).", pantr::value_error);
  }
  SUBCASE("a grid whose root has other cell counts") {
    Setup coarse = s;
    coarse.knots = {{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0}, k2};
    const bs::BsplineSpace<double> other = make_root<double>(coarse);
    THB_THROWS_WITH_AS(build(other, grid, none),
                       "grid root cells_per_axis (4, 4) must match root_space.num_intervals "
                       "(2, 4).",
                       pantr::value_error);
  }
  SUBCASE("a grid on another domain") {
    Setup moved = s;
    moved.knots = {{0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0}, k2};
    const bs::BsplineSpace<double> other = make_root<double>(moved);
    THB_THROWS_WITH_AS(build(other, grid, none), "grid root bounds must match root_space domain.",
                       pantr::value_error);
  }
  SUBCASE("a domain mismatch below the knot tolerance is accepted, one above is refused") {
    // The check is absolute against the root's own tolerance (8 eps at unit scale here).
    Setup nearly = s;
    nearly.knots = {{0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0 + 4 * eps, 1.0 + 4 * eps, 1.0 + 4 * eps},
                    k2};
    THB_NOTHROW(build(make_root<double>(nearly), grid, none));
    nearly.knots[0].assign({0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0 + 1e-6, 1.0 + 1e-6, 1.0 + 1e-6});
    THB_THROWS_WITH_AS(build(make_root<double>(nearly), grid, none),
                       "grid root bounds must match root_space domain.", pantr::value_error);
  }
  SUBCASE("regularity of the wrong length") {
    THB_THROWS_WITH_AS(build(root, grid, {1}),
                       "regularity must be a scalar or length-2 sequence; got length 1.",
                       pantr::value_error);
    THB_THROWS_WITH_AS(build(root, grid, {1, 1, 1}),
                       "regularity must be a scalar or length-2 sequence; got length 3.",
                       pantr::value_error);
  }
  SUBCASE("regularity out of range") {
    THB_THROWS_WITH_AS(build(root, grid, {std::nullopt, 2}),
                       "regularity[1]=2 must be in [-1, degree[1]-1=1]; got 2.",
                       pantr::value_error);
    THB_THROWS_WITH_AS(build(root, grid, {-2, 0}),
                       "regularity[0]=-2 must be in [-1, degree[0]-1=1]; got -2.",
                       pantr::value_error);
    THB_NOTHROW(build(root, grid, {-1, 1}));
  }
  SUBCASE("a root cell whose subdivision knots fall within the space tolerance") {
    // The interior cell (0.5, 0.5 + 3e-15) is wider than the knot tolerance (8 eps of the unit
    // domain), but its midpoint is within it of both ends: the three knots merge into one, the
    // level-1 space has 4 intervals where the grid has 6 cells, and the space is refused, where the
    // Python implementation builds it and misreads its own tables later.
    const double e = 0.5 + 3e-15;
    const std::vector<double> knots = {0.0, 0.0, 0.0, 0.5, e, 1.0, 1.0, 1.0};
    Setup tiny{{knots}, {2}, {2}, {std::nullopt}, {{0, {1}, {2}}}, true};
    THB_THROWS_WITH_AS(make_space(tiny),
                       doctest::Contains("level 1 has 4 knot intervals along axis 0 but the grid "
                                         "has 6 cells there"),
                       pantr::value_error);
  }
  SUBCASE("a root function with an empty support is refused") {
    // Knots the 1D space accepts, with a multiplicity of four at degree 2: the function whose
    // knots `t_i .. t_{i + 3}` are all equal has a zero-width support, which no interval holds.
    Setup inner{
        {{0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0}}, {2}, {2}, {std::nullopt}, {}, true};
    THB_THROWS_WITH_AS(make_space(inner),
                       doctest::Contains("B-spline function(s) with empty support detected at "
                                         "indices [3]. This indicates an invalid B-spline space."),
                       pantr::state_error);
    Setup first{{{0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0}}, {2}, {2}, {std::nullopt}, {}, true};
    THB_THROWS_WITH_AS(make_space(first),
                       doctest::Contains("B-spline function(s) with empty support detected at "
                                         "indices [0]."),
                       pantr::state_error);
  }
  SUBCASE("a subdivision the knot functions refuse is refused with their text") {
    // A cell narrower than the tolerance of its own subdivision, three levels down: the knots chain
    // into one class above the degree's multiplicity limit.
    const std::vector<double> knots = {0.0, 0.0, 0.0, 1e-14, 0.5, 1.0, 1.0, 1.0};
    Setup tiny{{knots}, {2}, {2}, {std::nullopt}, {{0, {0}, {1}}, {1, {0}, {2}}, {2, {0}, {4}}},
               true};
    THB_THROWS_WITH_AS(
        make_space(tiny),
        doctest::Contains("Inserting these knots would exceed the maximum multiplicity of 3"),
        pantr::value_error);
  }
}

TEST_CASE("THBSplineSpace: accessors refuse an out-of-range argument with the Python text") {
  const THBSplineSpace<double> space = make_space(setup_2d(true, 2));
  THB_THROWS_WITH_AS(space.level_space(2), "level must be in [0, 1]; got 2.", pantr::value_error);
  THB_THROWS_WITH_AS(space.level_space(-1), "level must be in [0, 1]; got -1.", pantr::value_error);
  THB_THROWS_WITH_AS(space.active_function_indices(5), "level must be in [0, 1]; got 5.",
                     pantr::value_error);
  THB_NOTHROW(space.level_space(1));
  const I64 n = space.grid().num_cells();
  THB_THROWS_WITH_AS(space.active_basis(n), "cell id 28 is out of range [0, 28).",
                     pantr::index_error);
  THB_THROWS_WITH_AS(space.active_basis(-1), "cell id -1 is out of range [0, 28).",
                     pantr::index_error);
  THB_THROWS_WITH_AS(space.contributions(n), "cell id 28 is out of range [0, 28).",
                     pantr::index_error);
  THB_THROWS_WITH_AS(space.num_active_on(n), "cell id 28 is out of range [0, 28).",
                     pantr::index_error);
  THB_THROWS_WITH_AS(space.truncated(48), "dof 48 is out of range [0, 48).", pantr::index_error);
  THB_THROWS_WITH_AS(space.truncated(-1), "dof -1 is out of range [0, 48).", pantr::index_error);
  THB_THROWS_AS(space.two_scale(1, 0), pantr::value_error);
  THB_THROWS_AS(space.two_scale(0, 2), pantr::index_error);
  THB_THROWS_AS(space.support(2, 0), pantr::value_error);
  THB_THROWS_AS(space.support(0, 2), pantr::index_error);
}

TEST_CASE("THBSplineSpace: tabulation refuses what the Python implementation refuses") {
  const THBSplineSpace<double> space = make_space(setup_2d(true, 2));
  const std::vector<double> pts = cell_points(space, 0, {0.5});  // one point of cell 0
  const std::vector<double> distant = {5.0, 5.0};
  const std::vector<double> three = {0.1, 0.1, 0.1};
  const std::size_t k = space.num_active_on(0);

  SUBCASE("orders of the wrong length") {
    THB_THROWS_WITH_AS(tabulate(space, 0, pts, {1}),
                       "orders must be a scalar or length-2 sequence; got length 1.",
                       pantr::value_error);
    THB_THROWS_WITH_AS(tabulate(space, 0, pts, {1, 0, 0}),
                       "orders must be a scalar or length-2 sequence; got length 3.",
                       pantr::value_error);
  }
  SUBCASE("a negative order") {
    THB_THROWS_WITH_AS(tabulate(space, 0, pts, {0, -1}),
                       "orders must be non-negative; got (0, -1).", pantr::value_error);
  }
  SUBCASE("the orders are checked before the cell") {
    THB_THROWS_AS(tabulate(space, 999, pts, {-1, 0}), pantr::value_error);
    THB_THROWS_AS(tabulate(space, 999, pts, {0, 0}), pantr::index_error);
  }
  SUBCASE("a cell out of range") {
    THB_THROWS_WITH_AS(tabulate(space, 28, pts, {0, 0}), "cell id 28 is out of range [0, 28).",
                       pantr::index_error);
  }
  SUBCASE("points of another dimension") {
    THB_THROWS_WITH_AS(space.tabulate_basis(0, mdview<const double, 2>(three.data(), 1, 3)),
                       "pts must have trailing dimension 2; got shape (1, 3).", pantr::value_error);
  }
  SUBCASE("a point outside the cell") {
    THB_THROWS_WITH_AS(tabulate(space, 0, distant, {0, 0}),
                       doctest::Contains("pts must lie inside cell 0 with bounds"),
                       pantr::value_error);
    // The cell above in the same level: inside the domain, not inside this cell.
    const std::vector<double> next = cell_points(space, 1, {0.5});
    THB_THROWS_WITH_AS(tabulate(space, 0, next, {0, 0}),
                       doctest::Contains("pts must lie inside cell 0"), pantr::value_error);
  }
  SUBCASE("a NaN coordinate") {
    const std::vector<double> nan = {std::numeric_limits<double>::quiet_NaN(), 0.1};
    THB_THROWS_WITH_AS(tabulate(space, 0, nan, {0, 0}),
                       doctest::Contains("pts must lie inside cell 0"), pantr::value_error);
  }
  SUBCASE("a point within the slack of the boundary is inside, one beyond it is not") {
    const auto [lo, hi] = space.grid().cell_bounds(0);
    const double slack = pantr::thb::cell_membership_tolerance(lo[0], hi[0]);
    const std::vector<double> edge = {hi[0] + slack, 0.5 * (lo[1] + hi[1])};
    THB_NOTHROW(tabulate(space, 0, edge, {0, 0}));
    const std::vector<double> beyond = {hi[0] + 4.0 * slack, 0.5 * (lo[1] + hi[1])};
    THB_THROWS_AS(tabulate(space, 0, beyond, {0, 0}), pantr::value_error);
  }
  SUBCASE("outputs of the wrong shape") {
    std::vector<double> values(k + 1, 0.0);
    std::vector<I64> dofs(k, 0);
    const std::string wide = "out_basis must have shape (1, " + std::to_string(k) + "); got (1, " +
                             std::to_string(k + 1) + ").";
    THB_THROWS_WITH_AS(
        space.tabulate_basis(0, rows_of(pts, 2), mdview<double, 2>(values.data(), 1, k + 1), dofs),
        wide.c_str(), pantr::value_error);
    THB_THROWS_WITH_AS(
        space.tabulate_basis(0, rows_of(pts, 2), mdview<double, 2>(values.data(), 2, k), dofs),
        doctest::Contains("out_basis must have shape"), pantr::value_error);
    std::vector<I64> short_dofs(k - 1, 0);
    const std::string few = "out_dofs must have shape (" + std::to_string(k) + ",); got (" +
                            std::to_string(k - 1) + ",).";
    THB_THROWS_WITH_AS(space.tabulate_basis(0, rows_of(pts, 2),
                                            mdview<double, 2>(values.data(), 1, k), short_dofs),
                       few.c_str(), pantr::value_error);
  }
}

// ---- Tolerances
// ----------------------------------------------------------------------------------

TEST_CASE("THBSplineSpace: cell membership tolerance is 8 eps of the cell's own magnitude") {
  using pantr::thb::cell_membership_safety;
  using pantr::thb::cell_membership_tolerance;
  CHECK(cell_membership_safety == 2.0);
  // The unit cell: scale 1, and 2 * 4 eps is exact.
  CHECK(cell_membership_tolerance(0.0, 1.0) == 8.0 * eps);
  // The scale is the largest of the coordinates and the width.
  CHECK(cell_membership_tolerance(-3.0, -1.0) == 8.0 * eps * 3.0);
  CHECK(cell_membership_tolerance(1.0, 5.0) == 8.0 * eps * 5.0);
  // A narrow cell far from the origin: the coordinates, not the width, set the slack.
  CHECK(cell_membership_tolerance(1e6, 1e6 + 1e-3) == 8.0 * eps * (1e6 + 1e-3));
  // No floor: a cell inside the unit interval has a slack below 8 eps (covariant under scaling).
  CHECK(cell_membership_tolerance(0.0, 1e-3) == 8.0 * eps * 1e-3);
  CHECK(cell_membership_tolerance(0.0, 1e-3) < 8.0 * eps);
}

TEST_CASE("THBSplineSpace: the prolongation residual tolerance is 4096 eps of one plus the size") {
  using pantr::thb::prolongation_residual_tolerance;
  CHECK(prolongation_residual_tolerance(0.0) == 4096.0 * eps);
  CHECK(prolongation_residual_tolerance(1.0) == 4096.0 * eps * 2.0);
  CHECK(prolongation_residual_tolerance(3.0) == 4096.0 * eps * 4.0);
}

// ---- Detail kernels ----------------------------------------------------------------------------

TEST_CASE("FalseCountTable: a table whose entries overflow size_t is refused") {
  // One cell on each of 64 axes is a one-cell mask, but its table has 2^64 entries: the extents
  // are one more than the mask's on every axis. The refusal is the implementation limit, raised
  // before any allocation (an unchecked product wraps and the table is built too small).
  const std::vector<std::int64_t> counts(64, 1);
  const std::vector<std::uint8_t> mask(1, 1);
  CHECK_THROWS_AS(pantr::thb::detail::FalseCountTable(mask, counts), pantr::capacity_error);
  // 63 axes fit the product (2^63 entries); the allocation itself is what would then fail, so
  // the table is not built here.
}

TEST_CASE("FalseCountTable: box queries equal a count over the box") {
  // A 3 x 4 x 2 mask from a fixed pattern: every box checked against a direct count.
  const Ivec counts = {3, 4, 2};
  std::vector<std::uint8_t> mask(24);
  for (std::size_t i = 0; i < mask.size(); ++i) {
    mask[i] = (i * 7 + i / 5) % 3 == 0 ? 0 : 1;
  }
  const td::FalseCountTable table(mask, counts);
  for (I64 a0 = 0; a0 < 3; ++a0) {
    for (I64 b0 = a0 + 1; b0 <= 3; ++b0) {
      for (I64 a1 = 0; a1 < 4; ++a1) {
        for (I64 b1 = a1 + 1; b1 <= 4; ++b1) {
          for (I64 a2 = 0; a2 < 2; ++a2) {
            for (I64 b2 = a2 + 1; b2 <= 2; ++b2) {
              I64 falses = 0;
              I64 volume = 0;
              for (I64 i = a0; i < b0; ++i) {
                for (I64 j = a1; j < b1; ++j) {
                  for (I64 k = a2; k < b2; ++k) {
                    falses += mask[static_cast<std::size_t>((i * 4 + j) * 2 + k)] == 0 ? 1 : 0;
                    ++volume;
                  }
                }
              }
              const Ivec lo = {a0, a1, a2};
              const Ivec hi = {b0, b1, b2};
              CHECK(table.count_false(lo, hi) == falses);
              CHECK(table.all_true(lo, hi) == (falses == 0));
              CHECK(table.any_true(lo, hi) == (falses < volume));
            }
          }
        }
      }
    }
  }
}

TEST_CASE("make_two_scale: the degree-2 Bezier piece halved, against the hand-worked matrix") {
  // New control points of a quadratic split at 1/2: Q = (P0, (P0 + P1) / 2, (P1 + P2) / 2, P2).
  const std::vector<double> coarse = {0, 0, 0, 1, 1, 1};
  const std::vector<double> fine = {0, 0, 0, 0.5, 1, 1, 1};
  const bs::BsplineSpace1D<double> c{std::span<const double>(coarse), 2};
  const bs::BsplineSpace1D<double> f{std::span<const double>(fine), 2};
  const td::TwoScale1D a = td::make_two_scale<double>(c, f);
  CHECK(a.num_fine == 4);
  CHECK(a.num_coarse == 3);
  CHECK(a.width == 3);
  // alpha(j, i): row j of the fine basis, column i of the coarse one.
  const double dense[4][3] = {{1, 0, 0}, {0.5, 0.5, 0}, {0, 0.5, 0.5}, {0, 0, 1}};
  for (std::int64_t j = 0; j < 4; ++j) {
    for (std::int64_t l = 0; l < a.width; ++l) {
      const std::int64_t col = a.first_col[static_cast<std::size_t>(j)] + l;
      if (0 <= col && col < 3) {
        CHECK(a.alphas[static_cast<std::size_t>(j * a.width + l)] == dense[j][col]);
      }
    }
  }

  // refine_box of the coarse function B_1 (box [1, 2), coefficient 1): its column of alpha, which
  // is (0, 1/2, 1/2, 0), trimmed to the rows 1 and 2.
  td::CoeffBox b1;
  b1.lo = {1};
  b1.shape = {1};
  b1.coeffs = {1.0};
  const std::vector<td::TwoScale1D> mats = {a};
  const td::CoeffBox r1 = td::refine_box(b1, mats);
  CHECK(r1.lo == Ivec{1});
  CHECK(r1.shape == std::vector<std::size_t>{2});
  CHECK(r1.coeffs == std::vector<double>{0.5, 0.5});
  CHECK(b1.coeffs == std::vector<double>{1.0});  // the input is untouched

  // B_0 refines to rows 0 and 1 with (1, 1/2); the whole box [0, 3) of ones (the partition of
  // unity) refines to ones on all four rows.
  td::CoeffBox b0;
  b0.lo = {0};
  b0.shape = {1};
  b0.coeffs = {1.0};
  const td::CoeffBox r0 = td::refine_box(b0, mats);
  CHECK(r0.lo == Ivec{0});
  CHECK(r0.coeffs == std::vector<double>{1.0, 0.5});
  td::CoeffBox all;
  all.lo = {0};
  all.shape = {3};
  all.coeffs = {1.0, 1.0, 1.0};
  const td::CoeffBox rall = td::refine_box(all, mats);
  CHECK(rall.lo == Ivec{0});
  CHECK(rall.coeffs == std::vector<double>{1.0, 1.0, 1.0, 1.0});

  // Two directions: the outer product of the two columns, direction by direction.
  const std::vector<td::TwoScale1D> both = {a, a};
  td::CoeffBox corner;
  corner.lo = {1, 0};
  corner.shape = {1, 1};
  corner.coeffs = {1.0};
  const td::CoeffBox r2 = td::refine_box(corner, both);
  CHECK(r2.lo == Ivec{1, 0});
  CHECK(r2.shape == std::vector<std::size_t>{2, 2});
  CHECK(r2.coeffs == std::vector<double>{0.5, 0.25, 0.5, 0.25});
}

TEST_CASE("refine_box: a zero two-scale slice is refused with the Python text") {
  td::TwoScale1D zero;
  zero.num_fine = 3;
  zero.num_coarse = 3;
  zero.width = 2;
  zero.first_col = {0, 1, 2};
  zero.alphas.assign(6, 0.0);
  td::CoeffBox box;
  box.lo = {1};
  box.shape = {1};
  box.coeffs = {1.0};
  const std::vector<td::TwoScale1D> mats = {zero};
  THB_THROWS_WITH_AS(td::refine_box(box, mats),
                     "_refine_box: Oslo matrix slice for direction 0 (columns [1:2]) is entirely "
                     "zero \xE2\x80\x94 degenerate or invalid knot refinement.",
                     pantr::value_error);
}

TEST_CASE("truncate_box: coefficients on active functions are zeroed and nothing else") {
  // A 3 x 3 box at lo = (1, 0) of a level with 4 x 5 functions; flat index = 5 * i + j.
  td::CoeffBox box;
  box.lo = {1, 0};
  box.shape = {3, 3};
  box.coeffs.assign(9, 1.0);
  const Ivec num_basis = {4, 5};
  const Ivec active = {6, 12, 14, 19};  // (1,1), (2,2), (2,4), (3,4)
  CHECK(td::truncate_box(box, active, num_basis));
  // The box covers rows 1..3, columns 0..2: (1,1) is box (0,1), (2,2) is box (1,2); (2,4) and
  // (3,4) are outside the columns.
  CHECK(box.coeffs == std::vector<double>{1, 0, 1, 1, 1, 0, 1, 1, 1});
  td::CoeffBox untouched;
  untouched.lo = {0, 0};
  untouched.shape = {1, 2};
  untouched.coeffs = {1.0, 1.0};
  CHECK_FALSE(td::truncate_box(untouched, active, num_basis));
  CHECK(untouched.coeffs == std::vector<double>{1.0, 1.0});
}

TEST_CASE("THBSplineSpace: truncated functions are stored consistently") {
  const THBSplineSpace<double> space = make_space(setup_2d(true, 3));
  std::size_t count = 0;
  for (I64 dof = 0; dof < space.num_total_basis(); ++dof) {
    const pantr::thb::TruncatedFunction* f = space.truncated(dof);
    if (f == nullptr) {
      continue;
    }
    ++count;
    CAPTURE(dof);
    // The function is finer than its own level, and no finer than the space.
    I64 level = 0;
    while (space.func_offset()[static_cast<std::size_t>(level) + 1] <= dof) {
      ++level;
    }
    CHECK(f->rep_level > level);
    CHECK(f->rep_level < space.num_levels());
    const Ivec nb = level_num_basis(space, f->rep_level);
    std::size_t size = 1;
    for (std::size_t k = 0; k < 2; ++k) {
      CHECK(f->box_lo[k] >= 0);
      CHECK(f->box_lo[k] + static_cast<I64>(f->box_shape[k]) <= nb[k]);
      size *= f->box_shape[k];
    }
    REQUIRE(f->coeffs.size() == size);
    // Refinement coefficients and zeroing: every coefficient is in [0, 1].
    for (const double c : f->coeffs) {
      CHECK(c >= 0.0);
      CHECK(c <= 1.0 + 4 * eps);
    }
    // Truncation zeroed the coefficients on the active functions of the representation level:
    // none of the box's functions is active there.
    const std::span<const I64> active = space.active_function_indices(f->rep_level);
    for (std::size_t i = 0; i < f->box_shape[0]; ++i) {
      for (std::size_t j = 0; j < f->box_shape[1]; ++j) {
        const I64 flat =
            (f->box_lo[0] + static_cast<I64>(i)) * nb[1] + f->box_lo[1] + static_cast<I64>(j);
        if (std::binary_search(active.begin(), active.end(), flat)) {
          CHECK(f->coeffs[i * f->box_shape[1] + j] == 0.0);
        }
      }
    }
  }
  CHECK(count == 10);
  // The plain hierarchical basis stores none.
  const THBSplineSpace<double> hb = make_space(setup_2d(false, 3));
  for (I64 dof = 0; dof < hb.num_total_basis(); ++dof) {
    CHECK(hb.truncated(dof) == nullptr);
  }
  CHECK(hb.num_total_basis() == space.num_total_basis());
}

TEST_CASE(
    "THBSplineSpace: the default regularity is degree - 1 and the accessors report the inputs") {
  Setup explicit_reg = setup_2d(true, 2);
  explicit_reg.regularity = {1, 1};
  const THBSplineSpace<double> a = make_space(setup_2d(true, 2));
  const THBSplineSpace<double> b = make_space(explicit_reg);
  CHECK(a.num_total_basis() == b.num_total_basis());
  for (I64 level = 0; level < a.num_levels(); ++level) {
    const std::span<const I64> x = a.active_function_indices(level);
    const std::span<const I64> y = b.active_function_indices(level);
    CHECK(std::equal(x.begin(), x.end(), y.begin(), y.end()));
    for (std::size_t k = 0; k < 2; ++k) {
      const bs::BsplineSpace<double> sa = a.level_space(level);
      const bs::BsplineSpace<double> sb = b.level_space(level);
      const std::span<const double> kx = sa.spaces()[k].knots();
      const std::span<const double> ky = sb.spaces()[k].knots();
      CHECK(std::equal(kx.begin(), kx.end(), ky.begin(), ky.end()));
    }
  }
  CHECK(a.regularity().size() == 2);
  CHECK_FALSE(a.regularity()[0].has_value());
  CHECK(*b.regularity()[1] == 1);
  CHECK(a.truncate());
  CHECK(a.dim() == 2);
  CHECK(a.degrees()[0] == 2);
  CHECK(a.num_levels() == 2);
  CHECK(a.root_space().dim() == 2);
  CHECK(a.grid().num_cells() == 28);
  CHECK(a.tolerance() == a.root_space().tolerance());
  const std::vector<I64> per_level = a.num_basis_per_level();
  CHECK(per_level == std::vector<I64>{32, 16});
  CHECK(std::vector<I64>(a.func_offset().begin(), a.func_offset().end()) ==
        std::vector<I64>{0, 32, 48});
  // Regularity 0 inserts each knot degree - 0 = 3 times: the level-1 space of the 1D case has
  // more functions than the maximally smooth one.
  Setup smooth = setup_1d(true, 2);
  smooth.regularity = {std::nullopt};
  CHECK(make_space(setup_1d(true, 2)).level_space(1).spaces()[0].num_basis() >
        make_space(smooth).level_space(1).spaces()[0].num_basis());
}

}  // namespace
