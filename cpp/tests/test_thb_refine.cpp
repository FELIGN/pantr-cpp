// The operations of THBSplineSpace that build another space or a matrix from it: refine,
// refine_region, coarsen, restrict and the CSR prolongation. Oracles, each from a route other than
// the code under test:
//   - the refinement relation: a coarse function's value at a point equals the fine functions'
//     combination with the prolongation's coefficients, the coarse cell containing the point found
//     by brute force over the coarse grid's cell bounds, never by the grid's point location;
//   - the partition of unity carried through the prolongation (THB), with the plain hierarchical
//     basis as the control that does not carry it;
//   - the admissibility of a graded mesh by its definition, brute force over the cells' functions,
//     with the ungraded mesh as the control that violates it;
//   - coarsening the children of an ungraded refinement, which has to give the original cells back;
//   - the windowed space's values against the global ones at the same points, over the cells a
//     geometric argument (a padding in root cells, not the code's support tables) makes interior;
//   - values recorded from the Python implementation (thb_refine_golden.hpp says how);
//   - the prolongation at 1 and 8 threads, which is exactly equal (each column is solved alone and
//     the matrix is gathered in column order).
#include <doctest/doctest.h>

#include "thb_refine_golden.hpp"
#include "thb_test_support.hpp"

#include <pantr/bspline/space.hpp>
#include <pantr/bspline/space_1d.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/core/mdspan.hpp>
#include <pantr/grid/hierarchical_grid.hpp>
#include <pantr/grid/tensor_product_grid.hpp>
#include <pantr/parallel/parallel_for.hpp>
#include <pantr/thb/space.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace bs = pantr::bspline;
namespace gd = pantr::grid;
namespace par = pantr::parallel;
namespace gold = pantr_thb_refine_golden;
using pantr::mdview;
using pantr::thb::ProlongationCsr;
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
using thb_test::Reg;
using thb_test::ThreadsGuard;
using thb_test::value_error_units;
using Space = THBSplineSpace<double>;

/// The size of an entry's error in a prolongation: the library's own tier for the residual of the
/// local solve that produces it (`prolongation_residual_tolerance`). The entries are refinement
/// coefficients in `[0, 1]`, so it is also their absolute error; it stands for the product of the
/// condition number of the local system and the backward error constant, neither of which is
/// derived per entry. An admitted heuristic, not a derived bound.
constexpr double entry_error = 4096.0 * eps;

// ---- Setups ----------------------------------------------------------------------------------

const std::vector<double> k3 = {0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0};

/// Degree-2 knots with `n` equal spans on [0, 1].
std::vector<double> uniform_knots(int n) {
  std::vector<double> knots = {0.0, 0.0, 0.0};
  for (int i = 1; i < n; ++i) {
    knots.push_back(static_cast<double>(i) / static_cast<double>(n));
  }
  knots.insert(knots.end(), {1.0, 1.0, 1.0});
  return knots;
}

/// A space on the root whose knots and degrees are given, an unrefined grid whose cells are the
/// root's knot spans, and the given factor.
Space make_space(const std::vector<std::vector<double>>& knots, const std::vector<int>& degrees,
                 const Ivec& factor, const Reg& regularity, bool truncate) {
  std::vector<bs::BsplineSpace1D<double>> dirs;
  std::vector<std::vector<double>> breakpoints;
  for (std::size_t k = 0; k < knots.size(); ++k) {
    dirs.emplace_back(std::span<const double>(knots[k]), degrees[k]);
    const std::span<const double> unique = dirs.back().unique_knots_and_multiplicity(true).first;
    breakpoints.emplace_back(unique.begin(), unique.end());
  }
  bs::BsplineSpace<double> root(std::move(dirs));
  gd::HierarchicalGrid grid(gd::TensorProductGrid(std::move(breakpoints)), factor);
  return Space(std::move(root), std::move(grid), truncate, regularity);
}

/// The 4 x 4 root cells, degree (2, 2), factor (2, 2) space the golden values were recorded on.
Space base_2d(bool truncate = true) {
  return make_space({k2, k2}, {2, 2}, {2, 2}, {std::nullopt, std::nullopt}, truncate);
}

Space base_1d(bool truncate = true) { return make_space({k3}, {3}, {2}, {0}, truncate); }

Space base_3d() {
  return make_space({k2_3d, k2_3d, k2_3d}, {2, 2, 2}, {2, 2, 2},
                    {std::nullopt, std::nullopt, std::nullopt}, true);
}

/// A space with `n x n` root cells, degree (2, 2), factor (2, 2).
Space base_uniform(int n) {
  return make_space({uniform_knots(n), uniform_knots(n)}, {2, 2}, {2, 2},
                    {std::nullopt, std::nullopt}, true);
}

/// Refine the cell `(l, 0, ..., 0)`, for `l = 0 .. steps - 1`, one `refine` call each.
Space corner_chain(Space space, std::optional<I64> m, int steps) {
  for (int level = 0; level < steps; ++level) {
    const Ivec origin(space.dim(), 0);
    const std::optional<I64> cid = space.grid().cell_id(level, origin);
    REQUIRE(cid.has_value());
    space = space.refine(Ivec{*cid}, m);
  }
  return space;
}

// ---- Helpers ---------------------------------------------------------------------------------

/// The cells of a grid as `(level, index...)` rows, in cell-id order.
Ivec cells_flat(const gd::HierarchicalGrid& g) {
  Ivec out;
  for (I64 cid = 0; cid < g.num_cells(); ++cid) {
    out.push_back(g.cell_level(cid));
    for (const I64 i : g.cell_multi_index(cid)) {
      out.push_back(i);
    }
  }
  return out;
}

/// The cells of a grid as sorted `(lo..., hi...)` bounds.
std::set<std::vector<double>> cell_bound_set(const gd::HierarchicalGrid& g) {
  std::set<std::vector<double>> out;
  for (I64 cid = 0; cid < g.num_cells(); ++cid) {
    const auto [lo, hi] = g.cell_bounds(cid);
    std::vector<double> row = lo;
    row.insert(row.end(), hi.begin(), hi.end());
    out.insert(std::move(row));
  }
  return out;
}

template <class A, class B>
bool same_ints(const A& a, const B& b) {
  return std::equal(a.begin(), a.end(), b.begin(), b.end());
}

void check_summary(const Space& s, const gold::Summary& g) {
  CHECK(s.num_levels() == g.num_levels);
  CHECK(s.num_total_basis() == g.num_total);
  CHECK(s.grid().num_cells() == g.num_cells);
  CHECK(same_ints(s.num_basis_per_level(), g.per_level));
  CHECK(same_ints(cells_flat(s.grid()), g.cells));
}

/// A reproducible stream of numbers in `[0, 1)`.
class Lcg {
 public:
  Lcg() = default;
  explicit Lcg(std::uint64_t seed) : state_(seed) {}

  double next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state_ >> 11) / 9007199254740992.0;
  }

 private:
  std::uint64_t state_ = 0x9E3779B97F4A7C15ULL;
};

/// `P u` for a CSR matrix.
std::vector<double> multiply(const ProlongationCsr& p, const std::vector<double>& u) {
  std::vector<double> out(static_cast<std::size_t>(p.rows), 0.0);
  for (std::size_t r = 0; r < out.size(); ++r) {
    for (I64 at = p.indptr[r]; at < p.indptr[r + 1]; ++at) {
      out[r] += p.data[static_cast<std::size_t>(at)] *
                u[static_cast<std::size_t>(p.indices[static_cast<std::size_t>(at)])];
    }
  }
  return out;
}

/// Check the CSR invariants: shape, monotone offsets, increasing in-range columns within a row.
void check_csr_form(const ProlongationCsr& p, const Space& coarse, const Space& fine) {
  CHECK(p.rows == fine.num_total_basis());
  CHECK(p.cols == coarse.num_total_basis());
  REQUIRE(p.indptr.size() == static_cast<std::size_t>(p.rows) + 1);
  CHECK(p.indptr.front() == 0);
  REQUIRE(static_cast<std::size_t>(p.indptr.back()) == p.indices.size());
  CHECK(p.data.size() == p.indices.size());
  for (std::size_t r = 0; r < static_cast<std::size_t>(p.rows); ++r) {
    CHECK(p.indptr[r] <= p.indptr[r + 1]);
    for (I64 at = p.indptr[r]; at < p.indptr[r + 1]; ++at) {
      const I64 col = p.indices[static_cast<std::size_t>(at)];
      CHECK(0 <= col);
      CHECK(col < p.cols);
      if (at > p.indptr[r]) {
        CHECK(p.indices[static_cast<std::size_t>(at) - 1] < col);
      }
    }
  }
}

std::size_t max_row_nnz(const ProlongationCsr& p) {
  std::size_t most = 0;
  for (std::size_t r = 0; r < static_cast<std::size_t>(p.rows); ++r) {
    most = (std::max)(most, static_cast<std::size_t>(p.indptr[r + 1] - p.indptr[r]));
  }
  return most;
}

/// Whether the truncated functions acting on every cell belong to at most `m` successive levels
/// (the definition of an admissible mesh of class `m`).
///
/// A function acts on a cell when it is not zero there. `contributions` lists every function whose
/// tensor-product support covers the cell, and truncation can make some of those exactly zero on
/// it, so the list over-counts the levels: the check evaluates the functions at the cell's centre,
/// where a B-spline of a supporting cell is positive and a truncated function (a sum of
/// non-negative coefficients times such B-splines) is positive unless every term is exactly zero.
bool is_admissible(const Space& s, I64 m) {
  const std::size_t d = s.dim();
  for (I64 cid = 0; cid < s.grid().num_cells(); ++cid) {
    const std::vector<double> centre = cell_points(s, cid, {0.5});
    const auto vals = s.tabulate_basis(cid, mdview<const double, 2>(centre.data(), 1, d));
    const auto levels = s.contributions(cid).levels;
    I64 lo = (std::numeric_limits<I64>::max)();
    I64 hi = -1;
    for (std::size_t j = 0; j < levels.size(); ++j) {
      if (vals.values.data()[j] > 0.0) {
        lo = (std::min)(lo, levels[j]);
        hi = (std::max)(hi, levels[j]);
      }
    }
    if (hi >= 0 && hi - lo + 1 > m) {
      return false;
    }
  }
  return true;
}

/// The cell of `space` holding `pt` (strictly inside it), found by testing every cell's bounds.
I64 brute_force_cell(const Space& space, const double* pt) {
  const std::size_t d = space.dim();
  for (I64 cid = 0; cid < space.grid().num_cells(); ++cid) {
    const auto [lo, hi] = space.grid().cell_bounds(cid);
    bool inside = true;
    for (std::size_t k = 0; k < d; ++k) {
      inside = inside && lo[k] <= pt[k] && pt[k] <= hi[k];
    }
    if (inside) {
      return cid;
    }
  }
  return -1;
}

/// The refinement relation of a pair of spaces at interior points of every fine cell, and the
/// largest `|P 1 - 1|` over the fine dofs.
///
/// For random `u` in `[-1, 1]` the two sides are `sum_i u_i B^c_i(x)` and `sum_j (P u)_j B^f_j(x)`.
/// With `S_c`, `S_f` the sums of the absolute values of the basis functions at the point, `U_c`,
/// `U_f` the models of `value_error_units`, and `m` the largest `|P u|`: the left side errs by at
/// most `U_c eps S_c`; the right side by at most `U_f eps max(1, m) S_f` for the values and
/// `nnz_max * entry_error * S_f` for the entries of `P`, each of whose at most `nnz_max` terms in a
/// row carries one entry's error against a coefficient `|u| <= 1`. The bound is their sum times
/// `build_slack`.
double check_refinement_relation(const Space& coarse, const Space& fine) {
  const ProlongationCsr p = coarse.prolongation_csr(fine);
  check_csr_form(p, coarse, fine);
  Lcg rng;
  std::vector<double> u(static_cast<std::size_t>(p.cols));
  for (double& v : u) {
    v = 2.0 * rng.next() - 1.0;
  }
  const std::vector<double> pu = multiply(p, u);
  double m = 1.0;
  for (const double v : pu) {
    m = (std::max)(m, std::abs(v));
  }
  const double uc = value_error_units(coarse);
  const double uf = value_error_units(fine);
  const auto nnz = static_cast<double>(max_row_nnz(p));
  const std::size_t d = fine.dim();
  const std::vector<double> ts = {0.13, 0.5, 0.87};
  for (I64 cf = 0; cf < fine.grid().num_cells(); ++cf) {
    const std::vector<double> pts = cell_points(fine, cf, ts);
    const std::size_t n_pts = pts.size() / d;
    const auto fine_vals = fine.tabulate_basis(cf, mdview<const double, 2>(pts.data(), n_pts, d));
    const std::size_t nf = fine_vals.dofs.size();
    for (std::size_t i = 0; i < n_pts; ++i) {
      double rhs = 0.0;
      double s_f = 0.0;
      for (std::size_t j = 0; j < nf; ++j) {
        const double b = fine_vals.values.data()[i * nf + j];
        rhs += b * pu[static_cast<std::size_t>(fine_vals.dofs.data()[j])];
        s_f += std::abs(b);
      }
      const I64 cc = brute_force_cell(coarse, pts.data() + i * d);
      REQUIRE(cc >= 0);
      const auto coarse_vals =
          coarse.tabulate_basis(cc, mdview<const double, 2>(pts.data() + i * d, 1, d));
      double lhs = 0.0;
      double s_c = 0.0;
      for (std::size_t j = 0; j < coarse_vals.dofs.size(); ++j) {
        const double b = coarse_vals.values.data()[j];
        lhs += b * u[static_cast<std::size_t>(coarse_vals.dofs.data()[j])];
        s_c += std::abs(b);
      }
      const double bound =
          build_slack * eps * (uc * s_c + (uf * m + entry_error / eps * nnz) * s_f);
      CAPTURE(cf);
      CAPTURE(i);
      CHECK(std::abs(lhs - rhs) <= bound);
    }
  }
  const std::vector<double> ones(static_cast<std::size_t>(p.cols), 1.0);
  const std::vector<double> pone = multiply(p, ones);
  double worst = 0.0;
  for (const double v : pone) {
    worst = (std::max)(worst, std::abs(v - 1.0));
  }
  return worst;
}

// ---- Golden values from the Python implementation --------------------------------------------

TEST_CASE("refine: graded corner refinements reproduce the Python implementation") {
  SUBCASE("class 2") { check_summary(corner_chain(base_2d(), 2, 4), gold::ref_2); }
  SUBCASE("class 3") { check_summary(corner_chain(base_2d(), 3, 4), gold::ref_3); }
  SUBCASE("ungraded") { check_summary(corner_chain(base_2d(), std::nullopt, 4), gold::ref_n); }
}

TEST_CASE("refine_region: graded regions reproduce the Python implementation") {
  Space s = base_2d();
  s = s.refine_region(0, Ivec{1, 1}, Ivec{3, 3}, 3);
  s = s.refine_region(1, Ivec{2, 2}, Ivec{4, 4}, 3);
  s = s.refine_region(2, Ivec{4, 4}, Ivec{6, 6}, 3);
  check_summary(s, gold::reg_3);
}

TEST_CASE("coarsen: the guard and the family rule reproduce the Python implementation") {
  const Space s = corner_chain(base_2d(), 2, 4);
  Ivec ids;
  for (I64 cid = 0; cid < s.grid().num_cells(); ++cid) {
    if (s.grid().cell_level(cid) == 2) {
      ids.push_back(cid);
    }
  }
  const auto check = [&](const gold::Coarsening& g, std::optional<I64> m) {
    CHECK(same_ints(ids, g.ids));
    check_summary(s.coarsen(ids, m), g.result);
  };
  SUBCASE("class 2 vetoes every family") { check(gold::coa_2, 2); }
  SUBCASE("class 3 vetoes every family") { check(gold::coa_3, 3); }
  SUBCASE("no class coarsens every complete family") { check(gold::coa_n, std::nullopt); }
}

TEST_CASE("restrict: the maps reproduce the Python implementation") {
  Space parent = base_2d();
  parent = parent.refine_region(0, Ivec{0, 0}, Ivec{2, 2}, std::nullopt);
  parent = parent.refine_region(1, Ivec{0, 0}, Ivec{2, 2}, std::nullopt);
  const auto check = [&](const gold::Restriction& g, const Ivec& ids) {
    CHECK(same_ints(ids, g.ids));
    const auto r = parent.restrict(ids);
    check_summary(r.space, g.space);
    CHECK(same_ints(r.local_to_global_dof, g.dof_map));
    CHECK(same_ints(r.local_to_global_cell, g.cell_map));
    for (std::size_t l = 0; l < g.active.size(); ++l) {
      CHECK(same_ints(r.space.active_function_indices(static_cast<I64>(l)), g.active[l]));
    }
  };
  SUBCASE("a box of cells") {
    Ivec ids;
    for (I64 cid = 0; cid < parent.grid().num_cells(); ++cid) {
      const auto [lo, hi] = parent.grid().cell_bounds(cid);
      bool in_box = true;
      for (std::size_t k = 0; k < 2; ++k) {
        const double centre = 0.5 * (lo[k] + hi[k]);
        in_box = in_box && centre >= 0.1 && centre <= 0.6;
      }
      if (in_box) {
        ids.push_back(cid);
      }
    }
    check(gold::res_box, ids);
  }
  SUBCASE("one deepest cell brings the tiling of its root cell") {
    I64 deepest = 0;
    for (I64 cid = 0; cid < parent.grid().num_cells(); ++cid) {
      if (parent.grid().cell_level(cid) > parent.grid().cell_level(deepest)) {
        deepest = cid;
      }
    }
    check(gold::res_deep, Ivec{deepest});
  }
}

/// Compare a CSR with the Python one: the structure exactly (it is made of integers that follow
/// from the supports), the values within `entry_error`, since the two run the same local solves
/// with another SVD and another order of summation.
void check_csr_against_golden(const ProlongationCsr& p, const gold::Csr& g) {
  CHECK(p.rows == g.rows);
  CHECK(p.cols == g.cols);
  CHECK(same_ints(p.indptr, g.indptr));
  CHECK(same_ints(p.indices, g.indices));
  REQUIRE(p.data.size() == g.data.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < p.data.size(); ++i) {
    worst = (std::max)(worst, std::abs(p.data[i] - g.data[i]));
  }
  CAPTURE(worst);
  CHECK(worst <= entry_error);
}

TEST_CASE("prolongation_csr: the matrix reproduces the Python implementation") {
  SUBCASE("2D THB, ungraded") {
    const Space coarse = base_2d().refine_region(0, Ivec{0, 0}, Ivec{2, 2}, std::nullopt);
    const Space fine = coarse.refine_region(1, Ivec{0, 0}, Ivec{2, 2}, std::nullopt);
    check_csr_against_golden(coarse.prolongation_csr(fine), gold::pro_a);
  }
  SUBCASE("1D degree 3 with regularity 0") {
    const Space coarse = base_1d().refine_region(0, Ivec{0}, Ivec{2}, std::nullopt);
    const Space fine = coarse.refine_region(1, Ivec{0}, Ivec{2}, std::nullopt);
    check_csr_against_golden(coarse.prolongation_csr(fine), gold::pro_b);
  }
}

// ---- Oracles ---------------------------------------------------------------------------------

TEST_CASE("prolongation_csr: the refinement relation holds at points of every fine cell") {
  const auto region = [](const Space& s, I64 level, Ivec lo, Ivec hi, std::optional<I64> m) {
    return s.refine_region(level, std::move(lo), std::move(hi), m);
  };
  SUBCASE("2D THB, ungraded") {
    const Space coarse = region(base_2d(), 0, {0, 0}, {2, 2}, std::nullopt);
    check_refinement_relation(coarse, region(coarse, 1, {0, 0}, {2, 2}, std::nullopt));
  }
  SUBCASE("2D HB, ungraded") {
    const Space coarse = region(base_2d(false), 0, {0, 0}, {2, 2}, std::nullopt);
    check_refinement_relation(coarse, region(coarse, 1, {0, 0}, {2, 2}, std::nullopt));
  }
  SUBCASE("2D THB, graded to class 3") {
    const Space coarse = region(base_2d(), 0, {1, 1}, {3, 3}, 2);
    check_refinement_relation(coarse, region(coarse, 1, {2, 2}, {4, 4}, 3));
  }
  SUBCASE("2D THB, a graded corner chain, one more step") {
    const Space coarse = corner_chain(base_2d(), 2, 2);
    const std::optional<I64> cid = coarse.grid().cell_id(2, Ivec{0, 0});
    REQUIRE(cid.has_value());
    check_refinement_relation(coarse, coarse.refine(Ivec{*cid}, 2));
  }
  SUBCASE("2D THB, factor (2, 3)") {
    const Space root = make_space({k2, k2}, {2, 2}, {2, 3}, {std::nullopt, std::nullopt}, true);
    const Space coarse = region(root, 0, {0, 0}, {2, 2}, std::nullopt);
    check_refinement_relation(coarse, region(coarse, 1, {0, 0}, {3, 3}, std::nullopt));
  }
  SUBCASE("1D degree 3 with regularity 0, THB") {
    const Space coarse = region(base_1d(), 0, {0}, {2}, std::nullopt);
    check_refinement_relation(coarse, region(coarse, 1, {0}, {2}, std::nullopt));
  }
  SUBCASE("3D THB") {
    const Space coarse = region(base_3d(), 0, {0, 0, 0}, {1, 1, 1}, std::nullopt);
    check_refinement_relation(coarse, region(coarse, 1, {0, 0, 0}, {1, 1, 1}, std::nullopt));
  }
}

TEST_CASE("prolongation_csr: a refinement whose local solve needs an accurate SVD is accepted") {
  // 3D HB, degree 2, factor (2, 2, 1): the local system of one column is full rank and well
  // conditioned, yet a divide-and-conquer SVD reconstructs it with an error far above eps, which
  // the residual gate (4096 eps of one plus the size) refuses; an LAPACK `gelsd` solve (the Python
  // implementation) passes it.
  const Space root = make_space({k2, k2, k2}, {2, 2, 2}, {2, 2, 1},
                                {std::nullopt, std::nullopt, std::nullopt}, false);
  const Space prev = root.refine(Ivec{49, 56}, std::nullopt);
  const Space fine = prev.refine(Ivec{7, 51}, 2);
  check_refinement_relation(prev, fine);
}

TEST_CASE("prolongation_csr: random 3D refine chains give valid prolongations") {
  // 20 chains on a 4 x 4 x 4 degree-2 root: the seed fixes two boxes of 2 or 3 cells per axis,
  // refined at level 0 one after the other (the second may overlap or extend the first). THB and
  // HB, graded (class 2) and ungraded, factor (2, 2, 2) and (2, 2, 1) all occur. Every step's
  // prolongation has to be accepted by the residual gate and satisfy the refinement relation. The
  // chains stay at two levels because the local solves of a 3D prolongation over three levels are
  // costly (seconds each, under the sanitizers minutes).
  constexpr std::uint64_t n_chains = 20;
  for (std::uint64_t seed = 0; seed < n_chains; ++seed) {
    const bool truncate = seed % 2 == 0;
    const std::optional<I64> grade = (seed / 2) % 2 == 0 ? std::optional<I64>(2) : std::nullopt;
    const Ivec factor = (seed / 4) % 2 == 0 ? Ivec{2, 2, 2} : Ivec{2, 2, 1};
    CAPTURE(seed);
    CAPTURE(truncate);
    CAPTURE(factor);
    Lcg rng(0x2545F4914F6CDD1DULL + seed);
    // A uniform draw in `0 .. n - 1`.
    const auto draw = [&rng](I64 n) {
      return static_cast<I64>(rng.next() * static_cast<double>(n));
    };
    Space space = make_space({k2, k2, k2}, {2, 2, 2}, factor,
                             {std::nullopt, std::nullopt, std::nullopt}, truncate);
    for (int step = 0; step < 2; ++step) {
      Ivec lo(3);
      Ivec hi(3);
      for (std::size_t k = 0; k < 3; ++k) {
        const I64 width = 2 + draw(2);
        lo[k] = draw(4 - width + 1);
        hi[k] = lo[k] + width;
      }
      const Space next = space.refine_region(0, lo, hi, grade);
      check_refinement_relation(space, next);
      space = next;
    }
  }
}

TEST_CASE("prolongation_csr: the partition of unity is carried through, and HB does not carry it") {
  const auto pairs = [](bool truncate) {
    const Space coarse = base_2d(truncate).refine_region(0, Ivec{0, 0}, Ivec{2, 2}, std::nullopt);
    return std::pair{coarse, coarse.refine_region(1, Ivec{0, 0}, Ivec{2, 2}, std::nullopt)};
  };
  const auto [coarse_t, fine_t] = pairs(true);
  const ProlongationCsr pt = coarse_t.prolongation_csr(fine_t);
  const std::vector<double> ones(static_cast<std::size_t>(pt.cols), 1.0);
  const std::vector<double> carried = multiply(pt, ones);
  double worst = 0.0;
  for (const double v : carried) {
    worst = (std::max)(worst, std::abs(v - 1.0));
  }
  // Every entry of a row carries `entry_error` (see `check_refinement_relation`).
  CHECK(worst <= build_slack * static_cast<double>(max_row_nnz(pt)) * entry_error);

  // The control: the plain hierarchical basis sums to something other than one at an interface,
  // so its prolongation of the ones is not the ones, and the check above can fail.
  const auto [coarse_h, fine_h] = pairs(false);
  const ProlongationCsr ph = coarse_h.prolongation_csr(fine_h);
  const std::vector<double> carried_h =
      multiply(ph, std::vector<double>(static_cast<std::size_t>(ph.cols), 1.0));
  double worst_h = 0.0;
  for (const double v : carried_h) {
    worst_h = (std::max)(worst_h, std::abs(v - 1.0));
  }
  // A control threshold needs only to be far above the round-off level (the bound above, a few
  // `entry_error`) and far below the effect it detects (the weight of the coarse functions that
  // truncation would have removed, a sizeable fraction of one): 0.01 is both, by orders of
  // magnitude on either side.
  CHECK(worst_h > 0.01);
}

TEST_CASE("refine: a graded mesh is admissible of its class, the ungraded one is not") {
  const auto check = [](const Space& s, I64 m) {
    CAPTURE(m);
    CHECK(is_admissible(s, m));
  };
  SUBCASE("2D corner chain, class 2") { check(corner_chain(base_2d(), 2, 4), 2); }
  SUBCASE("2D corner chain, class 3") { check(corner_chain(base_2d(), 3, 4), 3); }
  SUBCASE("1D corner chain, class 2") { check(corner_chain(base_1d(), 2, 4), 2); }
  SUBCASE("graded regions, class 3") {
    Space s = base_uniform(6);
    s = s.refine_region(0, Ivec{1, 1}, Ivec{4, 4}, 3);
    s = s.refine_region(1, Ivec{3, 3}, Ivec{7, 7}, 3);
    s = s.refine_region(2, Ivec{8, 8}, Ivec{12, 12}, 3);
    check(s, 3);
  }
  SUBCASE("graded refinement of scattered cells, class 2") {
    Space s = base_uniform(6);
    s = s.refine_region(0, Ivec{2, 2}, Ivec{4, 4}, 2);
    Ivec ids;
    for (I64 cid = 0; cid < s.grid().num_cells(); cid += 3) {
      ids.push_back(cid);
    }
    s = s.refine(ids, 2);
    check(s, 2);
  }
  SUBCASE("the ungraded chain violates class 2, which is what the check detects") {
    CHECK_FALSE(is_admissible(corner_chain(base_2d(), std::nullopt, 4), 2));
  }
}

TEST_CASE("coarsen: the children of an ungraded refinement coarsen back to the original cells") {
  const auto check = [](const Space& original, const Ivec& ids) {
    const Space refined = original.refine(ids, std::nullopt);
    Ivec children;
    const auto factor = original.grid().factor();
    for (const I64 cid : ids) {
      const I64 level = original.grid().cell_level(cid);
      const Ivec midx = original.grid().cell_multi_index(cid);
      // Every child: the index times the factor plus an offset in the factor box.
      Ivec child(midx.size());
      for (I64 flat = 0;; ++flat) {
        I64 rest = flat;
        for (std::size_t k = midx.size(); k-- > 0;) {
          child[k] = midx[k] * factor[k] + rest % factor[k];
          rest /= factor[k];
        }
        if (rest != 0) {
          break;
        }
        const std::optional<I64> id = refined.grid().cell_id(level + 1, child);
        REQUIRE(id.has_value());
        children.push_back(*id);
      }
    }
    const Space back = refined.coarsen(children, std::nullopt);
    CHECK(cell_bound_set(back.grid()) == cell_bound_set(original.grid()));
    CHECK(back.num_total_basis() == original.num_total_basis());
    CHECK(back.num_levels() == original.num_levels());
    for (I64 l = 0; l < original.num_levels(); ++l) {
      CHECK(same_ints(back.active_function_indices(l), original.active_function_indices(l)));
    }
  };
  SUBCASE("root cells") { check(base_2d(), Ivec{0, 5, 6, 15}); }
  SUBCASE("cells of two levels") {
    const Space s = base_2d().refine_region(0, Ivec{0, 0}, Ivec{2, 2}, std::nullopt);
    Ivec ids;
    for (I64 cid = 0; cid < s.grid().num_cells(); cid += 2) {
      ids.push_back(cid);
    }
    check(s, ids);
  }
  SUBCASE("3D") { check(base_3d(), Ivec{1, 6}); }
}

TEST_CASE("coarsen: an incomplete family is left alone") {
  const Space s = base_2d().refine_region(0, Ivec{0, 0}, Ivec{1, 1}, std::nullopt);
  // The four children of cell (0, 0) are the first four cells at level 1; name three of them.
  Ivec three;
  for (I64 cid = 0; cid < s.grid().num_cells() && three.size() < 3; ++cid) {
    if (s.grid().cell_level(cid) == 1) {
      three.push_back(cid);
    }
  }
  const Space same = s.coarsen(three, std::nullopt);
  CHECK(same.grid().num_cells() == s.grid().num_cells());
  CHECK(cell_bound_set(same.grid()) == cell_bound_set(s.grid()));
}

TEST_CASE("restrict: over interior cells the windowed space equals the global one") {
  // 12 x 12 root cells, degree 2; the window is the root cells [1, 11) on both axes, so the cells
  // of root index 4 to 7 are at least `degree + 1` root cells from both window edges. A function of
  // level `l` acting on a cell reaches at most `degree` root cells away, and a finer function that
  // truncates it lies in its support up to less than one more root cell: `degree + 1` in all.
  Space global = base_uniform(12);
  global = global.refine_region(0, Ivec{3, 3}, Ivec{9, 9}, std::nullopt);
  global = global.refine_region(1, Ivec{8, 8}, Ivec{14, 14}, std::nullopt);
  Ivec ids;
  for (I64 cid = 0; cid < global.grid().num_cells(); ++cid) {
    const auto [lo, hi] = global.grid().cell_bounds(cid);
    bool in_box = true;
    for (std::size_t k = 0; k < 2; ++k) {
      const double centre = 0.5 * (lo[k] + hi[k]);
      in_box = in_box && centre >= 0.1 && centre <= 0.9;
    }
    if (in_box) {
      ids.push_back(cid);
    }
  }
  const auto r = global.restrict(ids);
  const Space& local = r.space;
  REQUIRE(r.local_to_global_cell.size() == static_cast<std::size_t>(local.grid().num_cells()));
  REQUIRE(r.local_to_global_dof.size() == static_cast<std::size_t>(local.num_total_basis()));

  const std::vector<double> ts = {0.1, 0.45, 0.9};
  const double unit = build_slack * value_error_units(global) * eps;
  std::size_t interior = 0;
  for (I64 lc = 0; lc < local.grid().num_cells(); ++lc) {
    const I64 gc = r.local_to_global_cell[static_cast<std::size_t>(lc)];
    // The same cell, bound for bound.
    CHECK(local.grid().cell_bounds(lc) == global.grid().cell_bounds(gc));
    const auto [lo, hi] = global.grid().cell_bounds(gc);
    bool is_interior = true;
    for (std::size_t k = 0; k < 2; ++k) {
      const double root_index = std::floor(0.5 * (lo[k] + hi[k]) * 12.0);
      is_interior = is_interior && root_index >= 4.0 && root_index <= 7.0;
    }
    if (!is_interior) {
      continue;
    }
    ++interior;
    const std::vector<double> pts = cell_points(global, gc, ts);
    const std::size_t n_pts = pts.size() / 2;
    const auto want = global.tabulate_basis(gc, mdview<const double, 2>(pts.data(), n_pts, 2));
    const auto got = local.tabulate_basis(lc, mdview<const double, 2>(pts.data(), n_pts, 2));
    REQUIRE(got.dofs.size() == want.dofs.size());
    for (std::size_t j = 0; j < got.dofs.size(); ++j) {
      const I64 mapped = r.local_to_global_dof[static_cast<std::size_t>(got.dofs.data()[j])];
      REQUIRE(mapped >= 0);
      // The mapped dofs come out in the global order, so the columns pair up.
      CHECK(mapped == want.dofs.data()[j]);
      for (std::size_t i = 0; i < n_pts; ++i) {
        CHECK(std::abs(got.values.data()[i * got.dofs.size() + j] -
                       want.values.data()[i * want.dofs.size() + j]) <= unit);
      }
    }
  }
  CHECK(interior > 0);
}

TEST_CASE("restrict: every single-cell window builds and maps into the global dofs") {
  // The map has one failure it can raise, a local function beyond the global level's range, which
  // no window should reach; the sweep is the check that none does, on a clamped 2D root, a 1D root
  // of degree 3 with regularity 0 and an unclamped one.
  const std::vector<double> unclamped = {-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5};
  const auto sweep = [](const Space& s) {
    for (I64 cid = 0; cid < s.grid().num_cells(); ++cid) {
      const auto r = s.restrict(Ivec{cid});
      CHECK(r.local_to_global_dof.size() == static_cast<std::size_t>(r.space.num_total_basis()));
      for (const I64 g : r.local_to_global_dof) {
        CHECK(-1 <= g);
        CHECK(g < s.num_total_basis());
      }
      CHECK(std::count(r.local_to_global_cell.begin(), r.local_to_global_cell.end(), cid) == 1);
    }
  };
  SUBCASE("2D") {
    sweep(base_2d()
              .refine_region(0, Ivec{1, 1}, Ivec{3, 3}, 2)
              .refine_region(1, Ivec{2, 2}, Ivec{4, 4}, 2));
  }
  SUBCASE("1D degree 3, regularity 0") {
    sweep(base_1d().refine_region(0, Ivec{1}, Ivec{3}, 2).refine_region(1, Ivec{2}, Ivec{4}, 2));
  }
  SUBCASE("1D unclamped") {
    const Space s = make_space({unclamped}, {2}, {2}, {std::nullopt}, true);
    sweep(s.refine_region(0, Ivec{0}, Ivec{2}, 2));
  }
}

// ---- Determinism -----------------------------------------------------------------------------

TEST_CASE("prolongation_csr: the matrix does not depend on the thread count") {
  Space coarse = base_uniform(6);
  coarse = coarse.refine_region(0, Ivec{1, 1}, Ivec{4, 4}, 2);
  coarse = coarse.refine_region(1, Ivec{3, 3}, Ivec{7, 7}, 2);
  const Space fine = coarse.refine_region(2, Ivec{6, 6}, Ivec{10, 10}, 2);
  const auto build = [&](std::size_t threads) {
    const ThreadsGuard guard(threads);
    return coarse.prolongation_csr(fine);
  };
  check_many_threads_are_used();
  const ProlongationCsr one = build(1);
  const ProlongationCsr many = build(eight_threads());
  CHECK(one.rows == many.rows);
  CHECK(one.cols == many.cols);
  CHECK(one.indptr == many.indptr);
  CHECK(one.indices == many.indices);
  REQUIRE(one.data.size() == many.data.size());
  // The same columns, each solved alone and gathered in order: equal to the last bit, signed
  // zeros included.
  CHECK(std::memcmp(one.data.data(), many.data.data(), one.data.size() * sizeof(double)) == 0);
}

// ---- Validation ------------------------------------------------------------------------------

TEST_CASE("refine and coarsen refuse what the Python implementation refuses") {
  const Space s = base_2d().refine_region(0, Ivec{0, 0}, Ivec{2, 2}, std::nullopt);
  const std::string n = std::to_string(s.grid().num_cells());
  SUBCASE("a class below 2") {
    THB_THROWS_WITH_AS(s.refine(Ivec{0}, 1),
                       "admissible_class must be an integer >= 2 or None; got 1.",
                       pantr::value_error);
    THB_THROWS_WITH_AS(s.coarsen(Ivec{0}, 0),
                       "admissible_class must be an integer >= 2 or None; got 0.",
                       pantr::value_error);
    THB_THROWS_WITH_AS(s.refine_region(0, Ivec{0, 0}, Ivec{1, 1}, -3),
                       "admissible_class must be an integer >= 2 or None; got -3.",
                       pantr::value_error);
  }
  SUBCASE("the class is checked before the ids") {
    THB_THROWS_WITH_AS(s.refine(Ivec{9999}, 1),
                       "admissible_class must be an integer >= 2 or None; got 1.",
                       pantr::value_error);
  }
  SUBCASE("ids out of range are listed sorted and without repeats") {
    const std::string want =
        "cell_ids must lie in [0, " + n + "); got out-of-range id(s): [-1, 99, 100].";
    THB_THROWS_WITH_AS(s.refine(Ivec{100, 3, -1, 99, 100}, 2), want.c_str(), pantr::index_error);
    THB_THROWS_WITH_AS(s.refine(Ivec{100, 3, -1, 99, 100}, std::nullopt), want.c_str(),
                       pantr::index_error);
    THB_THROWS_WITH_AS(s.coarsen(Ivec{100, 3, -1, 99, 100}, 2), want.c_str(), pantr::index_error);
  }
  SUBCASE("a region: level, lengths, order, bounds, in that order") {
    THB_THROWS_WITH_AS(s.refine_region(5, Ivec{0, 0}, Ivec{1, 1}, 2),
                       "level must be in [0, 1]; got 5.", pantr::value_error);
    THB_THROWS_WITH_AS(s.refine_region(-1, Ivec{0, 0}, Ivec{1, 1}, 2),
                       "level must be in [0, 1]; got -1.", pantr::value_error);
    THB_THROWS_WITH_AS(s.refine_region(0, Ivec{0}, Ivec{1, 1}, 2),
                       "lo and hi must have length 2; got 1 and 2.", pantr::value_error);
    THB_THROWS_WITH_AS(s.refine_region(0, Ivec{0, 2}, Ivec{1, 2}, 2),
                       "lo must be strictly less than hi in every dimension; got lo=(0, 2), "
                       "hi=(1, 2).",
                       pantr::value_error);
    THB_THROWS_WITH_AS(s.refine_region(0, Ivec{0, 0}, Ivec{1, 5}, 2),
                       "[lo, hi) out of bounds at level 0: axis 1 needs [0, 4), got [0, 5).",
                       pantr::value_error);
    THB_THROWS_WITH_AS(s.refine_region(1, Ivec{-1, 0}, Ivec{1, 1}, 2),
                       "[lo, hi) out of bounds at level 1: axis 0 needs [0, 8), got [-1, 1).",
                       pantr::value_error);
  }
  SUBCASE("a region with no active leaf is valid and changes nothing") {
    // Level 0 cell (0, 0) is refined away, so the box holds no active leaf.
    const Space same = s.refine_region(0, Ivec{0, 0}, Ivec{1, 1}, 2);
    CHECK(cell_bound_set(same.grid()) == cell_bound_set(s.grid()));
    CHECK(same.num_total_basis() == s.num_total_basis());
  }
}

TEST_CASE("restrict refuses what the grid refuses") {
  const Space s = base_2d();
  THB_THROWS_WITH_AS(s.restrict(std::span<const I64>()), "restrict: cell_ids must be non-empty.",
                     pantr::value_error);
  THB_THROWS_WITH_AS(s.restrict(Ivec{3, 16}),
                     "restrict: cell id out of range [0, 16); got [3, 16].", pantr::index_error);
}

TEST_CASE("prolongation_csr refuses what the Python implementation refuses") {
  const Space coarse = base_2d().refine_region(0, Ivec{0, 0}, Ivec{2, 2}, std::nullopt);
  SUBCASE("every structural mismatch is listed") {
    const Space coarse_reg = make_space({k2, k2}, {2, 2}, {2, 2}, {std::nullopt, 1}, true)
                                 .refine_region(0, Ivec{0, 0}, Ivec{2, 2}, std::nullopt);
    const Space fine = make_space({k2n, k2}, {2, 2}, {2, 3}, {std::nullopt, std::nullopt}, false);
    THB_THROWS_WITH_AS(
        coarse_reg.prolongation_csr(fine),
        "fine must be a refinement of this space; mismatches: truncate: self=True vs "
        "fine=False; factor: self=(2, 2) vs fine=(2, 3); regularity: self=(None, 1) vs "
        "fine=(None, None); fine.num_levels=1 < self.num_levels=2; root knot vectors differ.",
        pantr::value_error);
  }
  SUBCASE("a dimension mismatch, with the one-entry tuple spelled as Python does") {
    const Space coarse_1d = make_space({k3}, {3}, {2}, {std::nullopt}, true);
    THB_THROWS_WITH_AS(coarse_1d.prolongation_csr(base_2d()),
                       "fine must be a refinement of this space; mismatches: dim: self=1 vs "
                       "fine=2; factor: self=(2,) vs fine=(2, 2); regularity: self=(None,) vs "
                       "fine=(None, None).",
                       pantr::value_error);
  }
  SUBCASE("a different root with the same structure") {
    const Space other = make_space({k2n, k2n}, {2, 2}, {2, 2}, {std::nullopt, std::nullopt}, true)
                            .refine_region(0, Ivec{0, 0}, Ivec{2, 2}, std::nullopt);
    THB_THROWS_WITH_AS(coarse.prolongation_csr(other),
                       "fine must be a refinement of this space; mismatches: root knot vectors "
                       "differ.",
                       pantr::value_error);
  }
  SUBCASE("a fine space that is a coarser one fails the residual check") {
    // The same structure (two levels, same root) but a smaller refined region: the coarse
    // functions on the level-1 region the fine space lacks cannot be reproduced.
    const Space smaller = base_2d().refine_region(0, Ivec{0, 0}, Ivec{1, 1}, std::nullopt);
    try {
      static_cast<void>(coarse.prolongation_csr(smaller));
      FAIL("expected a value_error");
    } catch (const pantr::value_error& e) {
      const std::string what = e.what();
      CAPTURE(what);
      const std::string head = "fine is not a refinement of this space (prolongation residual ";
      REQUIRE(what.compare(0, head.size(), head) == 0);
      CHECK(what.find(", above ") != std::string::npos);
      CHECK(what.back() == '.');
      // `{:.2e}`: one digit, a point, two digits, a signed two-digit exponent.
      const std::size_t e_at = what.find('e', head.size());
      REQUIRE(e_at != std::string::npos);
      CHECK(what[e_at - 3] == '.');
      CHECK((what[e_at + 1] == '-' || what[e_at + 1] == '+'));
    }
  }
}

// ---- Unusual inputs (run under ASAN and UBSAN) -----------------------------------------------

TEST_CASE("unusual inputs: repeats, unsorted ids, a huge class, empty ids and an identical space") {
  const Space s = corner_chain(base_2d(), 2, 3);
  SUBCASE("repeated and unsorted ids refine as the sorted distinct ids do") {
    const Space a = s.refine(Ivec{7, 2, 7, 2, 11}, 2);
    const Space b = s.refine(Ivec{2, 7, 11}, 2);
    CHECK(cells_flat(a.grid()) == cells_flat(b.grid()));
  }
  SUBCASE("a class above every level grades nothing, like no class") {
    const I64 huge = (std::numeric_limits<I64>::max)();
    const Space a = s.refine(Ivec{0, 3}, huge);
    const Space b = s.refine(Ivec{0, 3}, std::nullopt);
    CHECK(cells_flat(a.grid()) == cells_flat(b.grid()));
    // The guard of a coarsening reads the blocks `m` levels down, and no such level exists.
    Ivec leaves;
    for (I64 cid = 0; cid < s.grid().num_cells(); ++cid) {
      if (s.grid().cell_level(cid) == s.grid().max_level()) {
        leaves.push_back(cid);
      }
    }
    CHECK(cells_flat(s.coarsen(leaves, huge).grid()) ==
          cells_flat(s.coarsen(leaves, std::nullopt).grid()));
  }
  SUBCASE("empty ids give an unchanged copy") {
    const Space same = s.coarsen(std::span<const I64>(), 2);
    CHECK(cells_flat(same.grid()) == cells_flat(s.grid()));
    const Space again = s.refine(std::span<const I64>(), 2);
    CHECK(cells_flat(again.grid()) == cells_flat(s.grid()));
  }
  SUBCASE("a space is its own refinement: the prolongation is the identity") {
    // Every coarse function is matched against all the fine functions over its support, so the
    // matrix stores explicit zeros beside the diagonal, as solved.
    const ProlongationCsr p = s.prolongation_csr(s);
    check_csr_form(p, s, s);
    for (std::size_t r = 0; r < static_cast<std::size_t>(p.rows); ++r) {
      bool has_diagonal = false;
      for (I64 at = p.indptr[r]; at < p.indptr[r + 1]; ++at) {
        const bool diagonal = p.indices[static_cast<std::size_t>(at)] == static_cast<I64>(r);
        has_diagonal = has_diagonal || diagonal;
        CHECK(std::abs(p.data[static_cast<std::size_t>(at)] - (diagonal ? 1.0 : 0.0)) <=
              entry_error);
      }
      CHECK(has_diagonal);
    }
  }
  SUBCASE("an unrefined space to a refined one") {
    const Space coarse = base_2d();
    const Space fine = coarse.refine_region(0, Ivec{0, 0}, Ivec{2, 2}, std::nullopt);
    check_refinement_relation(coarse, fine);
  }
  SUBCASE("restricting the whole domain keeps every function") {
    Ivec all(static_cast<std::size_t>(s.grid().num_cells()));
    for (std::size_t i = 0; i < all.size(); ++i) {
      all[i] = static_cast<I64>(i);
    }
    const auto r = s.restrict(all);
    CHECK(r.space.num_total_basis() == s.num_total_basis());
    for (I64 d = 0; d < s.num_total_basis(); ++d) {
      CHECK(r.local_to_global_dof[static_cast<std::size_t>(d)] == d);
    }
  }
}

TEST_CASE("refine and coarsen leave the space they are called on unchanged") {
  const Space s = base_2d().refine_region(0, Ivec{0, 0}, Ivec{2, 2}, std::nullopt);
  const Ivec before = cells_flat(s.grid());
  const I64 total = s.num_total_basis();
  const Space a = s.refine(Ivec{0, 1, 2}, 2);
  // The four children of the level-0 cell (0, 0), a complete family.
  Ivec family;
  for (const I64 i : {0, 1}) {
    for (const I64 j : {0, 1}) {
      family.push_back(s.grid().cell_id(1, Ivec{i, j}).value());
    }
  }
  const Space b = s.coarsen(family, std::nullopt);
  CHECK(cells_flat(s.grid()) == before);
  CHECK(s.num_total_basis() == total);
  CHECK(a.grid().num_cells() != s.grid().num_cells());
  CHECK(b.grid().num_cells() != s.grid().num_cells());
}

}  // namespace
