// THBSpline: a THB space with control points, evaluated at points. Oracles, each from a route other
// than the code under test:
//   - evaluation against `sum_i c_i B_i(x)`, the point's cell found by testing every cell's bounds
//     (never the grid's point location) and the cell's functions tabulated one point at a time,
//     summed in dof order with the bound of recursive summation over the terms;
//   - derivatives against central finite differences of the spline's own values, with a step and a
//     bound derived as in test_thb_space.cpp (truncation `h^2 / 6 |f'''|`, rounding over the step);
//   - the refinement relation: the refined spline equals the original at the same points, within
//     the rounding model of the two evaluations and the error of the prolongation's entries;
//   - the partition of unity of the truncated basis (control points all 1);
//   - the evaluation at 1 and at up to 8 threads, which is bit-identical (every point's row is
//     computed alone and the blocks the loop cuts are a function of the problem).
#include <doctest/doctest.h>

#include "thb_test_support.hpp"

#include <pantr/core/errors.hpp>
#include <pantr/core/mdspan.hpp>
#include <pantr/core/ndarray.hpp>
#include <pantr/thb/space.hpp>
#include <pantr/thb/spline.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

namespace th = pantr::thb;
using pantr::mdview;
using pantr::NdArray;
using thb_test::build_slack;
using thb_test::check_many_threads_are_used;
using thb_test::eight_threads;
using thb_test::eps;
using thb_test::I64;
using thb_test::Ivec;
using thb_test::Lcg;
using thb_test::ThreadsGuard;
using thb_test::value_error_units;
using Space = th::THBSplineSpace<double>;
using Spline = th::THBSpline<double>;

/// Control points in `[-1, 1]`, shape `(n, rank)`.
NdArray<double> random_control_points(std::size_t n, std::size_t rank, std::uint64_t seed) {
  Lcg rng(seed);
  NdArray<double> cp({n, rank});
  for (std::size_t i = 0; i < cp.size(); ++i) {
    cp.data()[i] = 2.0 * rng.next() - 1.0;
  }
  return cp;
}

NdArray<double> constant_control_points(std::size_t n, std::size_t rank, double value) {
  NdArray<double> cp({n, rank});
  std::fill(cp.data(), cp.data() + cp.size(), value);
  return cp;
}

/// `m` points of the unit cube, row-major `(m, d)`, generic (never on a cell boundary).
std::vector<double> random_points(std::size_t m, std::size_t d, std::uint64_t seed) {
  Lcg rng(seed);
  std::vector<double> pts(m * d);
  for (double& x : pts) {
    x = 0.001 + 0.998 * rng.next();
  }
  return pts;
}

mdview<const double, 2> view_of(const std::vector<double>& pts, std::size_t d) {
  return mdview<const double, 2>(pts.data(), pts.size() / d, d);
}

/// The cell of `space` holding `pt`, found by testing every cell's bounds.
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

/// The value of the spline at one point by the definition, and the bound of its rounding error:
/// the functions of the point's cell tabulated alone, `sum_j b_j c_j` summed in dof order, whose
/// error is bounded by `build_slack (K + U) eps sum_j |b_j c_j|` (`K` terms in the sum, `U` the
/// error units of one value, `value_error_units`).
struct Reference {
  std::vector<double> value;
  double bound = 0.0;
  double abs_basis_sum = 0.0;  // sum_j |b_j|, the amplification of an error in the coefficients
};

Reference reference_value(const Spline& s, const double* pt, const Ivec& orders) {
  const std::size_t d = s.dim();
  const std::size_t rank = s.rank();
  const I64 cid = brute_force_cell(s.space(), pt);
  REQUIRE(cid >= 0);
  const auto tab = s.space().tabulate_basis(cid, mdview<const double, 2>(pt, 1, d), orders);
  const std::size_t k = tab.dofs.size();
  Reference ref;
  ref.value.assign(rank, 0.0);
  const double units = value_error_units(s.space()) + static_cast<double>(k);
  const auto cp = s.control_points().view<2>();
  for (std::size_t r = 0; r < rank; ++r) {
    double abs_terms = 0.0;
    for (std::size_t j = 0; j < k; ++j) {
      const double b = tab.values.data()[j];
      const double c = cp(static_cast<std::size_t>(tab.dofs.data()[j]), r);
      ref.value[r] += b * c;
      abs_terms += std::abs(b * c);
    }
    ref.bound = (std::max)(ref.bound, build_slack * units * eps * abs_terms);
  }
  for (std::size_t j = 0; j < k; ++j) {
    ref.abs_basis_sum += std::abs(tab.values.data()[j]);
  }
  return ref;
}

std::vector<Space> test_spaces() {
  std::vector<Space> out;
  out.push_back(thb_test::refined_2d(true));
  out.push_back(thb_test::refined_2d(false));
  const auto base =
      thb_test::make_thb_space({thb_test::k2_3d, thb_test::k2_3d, thb_test::k2_3d}, {2, 2, 2},
                               {2, 2, 2}, {std::nullopt, std::nullopt, std::nullopt}, true);
  out.push_back(base.refine(Ivec{0, 3}, std::nullopt));
  return out;
}

TEST_CASE("THBSpline: values equal the sum of the control points times the basis") {
  for (const Space& space : test_spaces()) {
    for (const std::size_t rank : {std::size_t{1}, std::size_t{3}}) {
      const Spline s(space, random_control_points(static_cast<std::size_t>(space.num_total_basis()),
                                                  rank, 7 + rank));
      const std::size_t d = space.dim();
      const std::vector<double> pts = random_points(200, d, 3);
      const NdArray<double> got = s.evaluate(view_of(pts, d));
      REQUIRE(got.shape()[0] == 200);
      REQUIRE(got.shape()[1] == rank);
      const Ivec zeros(d, 0);
      for (std::size_t i = 0; i < 200; ++i) {
        const Reference ref = reference_value(s, pts.data() + i * d, zeros);
        for (std::size_t r = 0; r < rank; ++r) {
          CHECK(std::abs(got.data()[i * rank + r] - ref.value[r]) <= ref.bound);
        }
      }
    }
  }
}

TEST_CASE("THBSpline: the truncated basis is a partition of unity") {
  const Space space = thb_test::refined_2d(true);
  const Spline s(
      space, constant_control_points(static_cast<std::size_t>(space.num_total_basis()), 1, 1.0));
  const std::vector<double> pts = random_points(500, 2, 5);
  const NdArray<double> got = s.evaluate(view_of(pts, 2));
  // Each value is a sum of at most K non-negative terms of total one; the bound is the one of
  // `reference_value` with every term non-negative.
  for (std::size_t i = 0; i < 500; ++i) {
    const Reference ref = reference_value(s, pts.data() + i * 2, Ivec{0, 0});
    CHECK(std::abs(got.data()[i] - 1.0) <= ref.bound + build_slack * eps);
  }
}

/// Derivatives against central finite differences of the spline's values, at generic points.
///
/// On a cell the spline is a polynomial of degree at most `p` per variable (here `p = 2`), so the
/// central difference with step `h` has the truncation error `h^2 / 6 |f'''|` (zero here, read from
/// the derivative of order three), and the four-point mixed difference is exact. The control
/// points lie in `[-1, 1]`, so the rounding error of a value is at most `delta = build_slack
/// value_error_units eps` (the basis is a partition of unity: `sum |b_j c_j| <= 1`), the quotient
/// carries `delta / h` (`delta / (h1 h2)` for the mixed one), and the analytic derivative's own
/// rounding is taken as `delta (1 + |a|)`, the admitted heuristic of test_thb_space.cpp for the
/// same quantity. The step is a 64th of the cell's width, and the points stay in their cell:
/// they are at parameters 0.3 and 0.6 of a cell.
void check_spline_derivatives(const Space& space) {
  const std::size_t d = space.dim();
  const Spline s(space,
                 random_control_points(static_cast<std::size_t>(space.num_total_basis()), 2, 11));
  const double delta = build_slack * value_error_units(space) * eps;
  for (I64 cid = 0; cid < space.grid().num_cells(); ++cid) {
    CAPTURE(cid);
    const auto [lo, hi] = space.grid().cell_bounds(cid);
    const std::vector<double> pts = thb_test::cell_points(space, cid, {0.3, 0.6});
    const std::size_t n = pts.size() / d;
    const auto shifted = [&](std::size_t a, double sa, std::size_t b, double sb) {
      std::vector<double> q = pts;
      for (std::size_t i = 0; i < n; ++i) {
        q[i * d + a] += sa * (hi[a] - lo[a]) / 64.0;
        if (b != a) {
          q[i * d + b] += sb * (hi[b] - lo[b]) / 64.0;
        }
      }
      return q;
    };
    const Ivec zeros(d, 0);
    for (std::size_t k = 0; k < d; ++k) {
      const double h = (hi[k] - lo[k]) / 64.0;
      Ivec e(d, 0);
      e[k] = 1;
      Ivec e3(d, 0);
      e3[k] = 3;
      const auto analytic = s.evaluate(view_of(pts, d), e);
      const auto third = s.evaluate(view_of(pts, d), e3);
      const auto qp = shifted(k, 1.0, k, 0.0);
      const auto qm = shifted(k, -1.0, k, 0.0);
      const auto plus = s.evaluate(view_of(qp, d), zeros);
      const auto minus = s.evaluate(view_of(qm, d), zeros);
      for (std::size_t at = 0; at < analytic.size(); ++at) {
        const double fd = (plus.data()[at] - minus.data()[at]) / (2.0 * h);
        const double a = analytic.data()[at];
        const double bound =
            delta / h + h * h / 6.0 * std::abs(third.data()[at]) + delta * (1.0 + std::abs(a));
        CHECK(std::abs(a - fd) <= bound);
      }
    }
    if (d >= 2) {
      const double h0 = (hi[0] - lo[0]) / 64.0;
      const double h1 = (hi[1] - lo[1]) / 64.0;
      Ivec e(d, 0);
      e[0] = 1;
      e[1] = 1;
      const auto analytic = s.evaluate(view_of(pts, d), e);
      const auto q1 = shifted(0, 1.0, 1, 1.0);
      const auto q2 = shifted(0, 1.0, 1, -1.0);
      const auto q3 = shifted(0, -1.0, 1, 1.0);
      const auto q4 = shifted(0, -1.0, 1, -1.0);
      const auto pp = s.evaluate(view_of(q1, d), zeros);
      const auto pm = s.evaluate(view_of(q2, d), zeros);
      const auto mp = s.evaluate(view_of(q3, d), zeros);
      const auto mm = s.evaluate(view_of(q4, d), zeros);
      for (std::size_t at = 0; at < analytic.size(); ++at) {
        const double fd =
            (pp.data()[at] - pm.data()[at] - mp.data()[at] + mm.data()[at]) / (4.0 * h0 * h1);
        const double a = analytic.data()[at];
        CHECK(std::abs(a - fd) <= delta / (h0 * h1) + delta * (1.0 + std::abs(a)));
      }
    }
  }
}

TEST_CASE("THBSpline: derivatives match central finite differences of the values") {
  const auto spaces = test_spaces();
  SUBCASE("2D THB, three levels") { check_spline_derivatives(spaces[0]); }
  SUBCASE("2D HB, three levels") { check_spline_derivatives(spaces[1]); }
  SUBCASE("3D THB") { check_spline_derivatives(spaces[2]); }
}

TEST_CASE("THBSpline: derivatives equal the control-point sum of the basis derivatives") {
  const Space space = thb_test::refined_2d(true);
  const Spline s(space,
                 random_control_points(static_cast<std::size_t>(space.num_total_basis()), 2, 13));
  const std::vector<double> pts = random_points(100, 2, 17);
  for (const Ivec& orders : {Ivec{1, 0}, Ivec{0, 2}, Ivec{1, 1}, Ivec{3, 0}}) {
    const auto got = s.evaluate(view_of(pts, 2), orders);
    for (std::size_t i = 0; i < 100; ++i) {
      const Reference ref = reference_value(s, pts.data() + i * 2, orders);
      for (std::size_t r = 0; r < 2; ++r) {
        CHECK(std::abs(got.data()[i * 2 + r] - ref.value[r]) <= ref.bound);
      }
    }
  }
}

/// The largest number of entries in a row of the prolongation.
std::size_t max_row_entries(const th::ProlongationCsr& p) {
  std::size_t most = 0;
  for (std::size_t r = 0; r < static_cast<std::size_t>(p.rows); ++r) {
    most = (std::max)(most, static_cast<std::size_t>(p.indptr[r + 1] - p.indptr[r]));
  }
  return most;
}

/// The refined spline against the original at generic points.
///
/// The two sides carry the rounding of their own evaluations (`Reference::bound`, the second
/// evaluated on the fine space) and, on the fine side, the error of the prolongation's entries:
/// each coefficient `(P c)_j` is a sum of at most `nnz_max` entries, each carrying
/// `entry_error` (the admitted `4096 eps` of test_thb_refine.cpp for the local solve's residual)
/// against a coefficient of at most 1 in magnitude, and enters the value weighted by the basis,
/// `sum |b|` of the fine functions (`abs_basis_sum`). The coefficients of the fine spline can
/// exceed 1, so its rounding is scaled by their largest magnitude. The bound is the sum times
/// `build_slack`.
void check_refined_equals_original(const Spline& coarse, const Spline& fine) {
  const std::size_t d = coarse.dim();
  const th::ProlongationCsr p = coarse.space().prolongation_csr(fine.space());
  const double entry_error = 4096.0 * eps;
  double cmax = 1.0;
  for (std::size_t i = 0; i < fine.control_points().size(); ++i) {
    cmax = (std::max)(cmax, std::abs(fine.control_points().data()[i]));
  }
  const std::vector<double> pts = random_points(300, d, 23);
  const auto vc = coarse.evaluate(view_of(pts, d));
  const auto vf = fine.evaluate(view_of(pts, d));
  const Ivec zeros(d, 0);
  for (std::size_t i = 0; i < 300; ++i) {
    const Reference rc = reference_value(coarse, pts.data() + i * d, zeros);
    const Reference rf = reference_value(fine, pts.data() + i * d, zeros);
    const double prolongation =
        static_cast<double>(max_row_entries(p)) * entry_error * rf.abs_basis_sum;
    const double bound = build_slack * (rc.bound + cmax * rf.bound + prolongation);
    for (std::size_t r = 0; r < coarse.rank(); ++r) {
      CHECK(std::abs(vc.data()[i * coarse.rank() + r] - vf.data()[i * coarse.rank() + r]) <= bound);
    }
  }
}

TEST_CASE("THBSpline: refinement keeps the function") {
  const Space base = thb_test::refined_2d(true);
  const auto n = static_cast<std::size_t>(base.num_total_basis());
  const Spline s(base, random_control_points(n, 2, 31));
  SUBCASE("refine the named cells, graded") {
    const I64 cells = base.grid().num_cells();
    const Spline fine = s.refine(Ivec{0, cells / 2, cells - 1}, 2);
    CHECK(fine.space().num_total_basis() > base.num_total_basis());
    check_refined_equals_original(s, fine);
  }
  SUBCASE("refine the named cells, ungraded") {
    const Spline fine = s.refine(Ivec{1, 2}, std::nullopt);
    check_refined_equals_original(s, fine);
  }
  SUBCASE("refine a region") {
    const Spline fine = s.refine_region(0, Ivec{2, 2}, Ivec{4, 4}, 2);
    CHECK(fine.space().grid().num_cells() != base.grid().num_cells());
    check_refined_equals_original(s, fine);
  }
  SUBCASE("refine twice") {
    const Spline once = s.refine_region(0, Ivec{2, 2}, Ivec{4, 4}, std::nullopt);
    const Spline twice = once.refine_region(1, Ivec{4, 4}, Ivec{6, 6}, std::nullopt);
    check_refined_equals_original(s, twice);
  }
}

TEST_CASE("THBSpline: an HB space keeps its function through refinement") {
  const Space base = thb_test::refined_2d(false);
  const Spline s(base,
                 random_control_points(static_cast<std::size_t>(base.num_total_basis()), 1, 37));
  check_refined_equals_original(s, s.refine_region(0, Ivec{2, 2}, Ivec{4, 4}, std::nullopt));
}

TEST_CASE("THBSpline: points on cell edges and the domain's corners") {
  const Space space = thb_test::refined_2d(true);
  const Spline s(space,
                 random_control_points(static_cast<std::size_t>(space.num_total_basis()), 1, 41));
  // Cell edges at every level (0.125 and 0.25 multiples), the corners, and the upper boundary.
  std::vector<double> pts;
  for (const double x : {0.0, 0.125, 0.25, 0.5, 1.0}) {
    for (const double y : {0.0, 0.0625, 0.25, 0.75, 1.0}) {
      pts.push_back(x);
      pts.push_back(y);
    }
  }
  const auto got = s.evaluate(view_of(pts, 2));
  for (std::size_t i = 0; i < pts.size() / 2; ++i) {
    // A value is continuous across an edge, so any cell that holds the point gives it.
    const Reference ref = reference_value(s, pts.data() + 2 * i, Ivec{0, 0});
    CHECK(std::abs(got.data()[i] - ref.value[0]) <= ref.bound);
  }
}

TEST_CASE("THBSpline: an empty batch and a spline without components") {
  const Space space = thb_test::refined_2d(true);
  const auto n = static_cast<std::size_t>(space.num_total_basis());
  const Spline s(space, random_control_points(n, 3, 43));
  const std::vector<double> none;
  const NdArray<double> empty = s.evaluate(mdview<const double, 2>(none.data(), 0, 2));
  CHECK(empty.shape()[0] == 0);
  CHECK(empty.shape()[1] == 3);
  CHECK(empty.size() == 0);

  const Spline flat(space, NdArray<double>({n, 0}));
  const std::vector<double> pts = random_points(10, 2, 47);
  const NdArray<double> got = flat.evaluate(view_of(pts, 2));
  CHECK(got.shape()[0] == 10);
  CHECK(got.shape()[1] == 0);
}

TEST_CASE("THBSpline: the refusals and their messages") {
  const Space space = thb_test::refined_2d(true);
  const auto n = static_cast<std::size_t>(space.num_total_basis());
  const std::string lead = std::to_string(n);
  THB_THROWS_WITH_AS(Spline(space, NdArray<double>({n + 1, 2})),
                     ("control_points leading dimension must be " + lead + "; got shape (" +
                      std::to_string(n + 1) + ", 2).")
                         .c_str(),
                     pantr::value_error);
  THB_THROWS_WITH_AS(Spline(space, NdArray<double>({n})), "control_points must be 2-D; got 1-D.",
                     pantr::value_error);

  const Spline s(space, random_control_points(n, 2, 53));
  // The first point outside, in input order, is the one named.
  const std::vector<double> outside = {0.5, 0.5, 2.5, 0.25, 0.5, -1.0};
  THB_THROWS_WITH_AS(s.evaluate(view_of(outside, 2)),
                     "point [2.5, 0.25] lies outside the grid domain.", pantr::value_error);
  const std::vector<double> not_a_number = {0.5, std::numeric_limits<double>::quiet_NaN()};
  THB_THROWS_WITH_AS(s.evaluate(view_of(not_a_number, 2)),
                     "point [0.5, nan] lies outside the grid domain.", pantr::value_error);

  const std::vector<double> three = {0.1, 0.2, 0.3};
  THB_THROWS_WITH_AS(s.evaluate(mdview<const double, 2>(three.data(), 1, 3)),
                     "pts must have trailing dimension 2; got shape (1, 3).", pantr::value_error);

  const std::vector<double> ok = {0.1, 0.2};
  std::vector<double> wrong(3);
  THB_THROWS_WITH_AS(s.evaluate(view_of(ok, 2), Ivec{0, 0}, mdview<double, 2>(wrong.data(), 1, 3)),
                     "out must have shape (1, 2); got (1, 3).", pantr::value_error);
  THB_THROWS_WITH_AS(s.evaluate(view_of(ok, 2), Ivec{0, -1}),
                     "orders must be non-negative; got (0, -1).", pantr::value_error);
  THB_THROWS_WITH_AS(s.evaluate(view_of(ok, 2), Ivec{0}),
                     "orders must be a scalar or length-2 sequence; got length 1.",
                     pantr::value_error);
}

TEST_CASE("THBSpline: one thread and several give bit-identical values") {
  check_many_threads_are_used();
  const Space space = thb_test::refined_2d(true);
  const Spline s(space,
                 random_control_points(static_cast<std::size_t>(space.num_total_basis()), 3, 59));
  // 2 x 10^4 points: far more than the 131072 operations below which the loop stays serial, at
  // about two hundred operations per point.
  const std::vector<double> pts = random_points(20000, 2, 61);
  const Ivec orders = {1, 0};
  std::vector<NdArray<double>> values;
  std::vector<NdArray<double>> derivatives;
  for (const std::size_t threads : {std::size_t{1}, eight_threads()}) {
    const ThreadsGuard guard(threads);
    values.push_back(s.evaluate(view_of(pts, 2)));
    derivatives.push_back(s.evaluate(view_of(pts, 2), orders));
  }
  CHECK(std::memcmp(values[0].data(), values[1].data(), values[0].size() * sizeof(double)) == 0);
  CHECK(std::memcmp(derivatives[0].data(), derivatives[1].data(),
                    derivatives[0].size() * sizeof(double)) == 0);
}

TEST_CASE("THBSpline: the prolongation of control points does not depend on the thread count") {
  check_many_threads_are_used();
  const Space base = thb_test::refined_2d(true);
  const Spline s(base,
                 random_control_points(static_cast<std::size_t>(base.num_total_basis()), 2, 67));
  std::vector<Spline> fine;
  for (const std::size_t threads : {std::size_t{1}, eight_threads()}) {
    const ThreadsGuard guard(threads);
    fine.push_back(s.refine_region(0, Ivec{2, 2}, Ivec{4, 4}, std::nullopt));
  }
  const auto& a = fine[0].control_points();
  const auto& b = fine[1].control_points();
  REQUIRE(a.size() == b.size());
  CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

}  // namespace
