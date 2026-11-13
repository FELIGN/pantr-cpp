// THBQuasiInterpolation: the staged quasi-interpolant onto a THB space, `points()` then
// `coefficients(values)`. Oracles, each from a route other than the code under test:
//   - the defining property: the quasi-interpolant of a THB spline's own values returns the
//     spline's coefficients (random coefficients; the property needs the truncated basis, and the
//     plain hierarchical basis is the control that does not have it), and the quasi-interpolant of
//     a polynomial of the space's degree evaluates back to the polynomial;
//   - the choice of each function's leaf cell by a brute-force scan written dof by dof over the
//     cells (the code works cell by cell), with the Greville abscissae from the knot formula, and
//     the tie rule on a configuration whose candidates are exactly equidistant;
//   - values recorded from the Python implementation (thb_qi_golden.hpp says how);
//   - the coefficients at 1 and at up to 8 threads, which are bit-identical (each function is
//     contracted alone, in a fixed order).
#include <doctest/doctest.h>

#include "thb_qi_golden.hpp"
#include "thb_test_support.hpp"

#include <pantr/core/errors.hpp>
#include <pantr/core/mdspan.hpp>
#include <pantr/core/ndarray.hpp>
#include <pantr/thb/quasi_interpolate.hpp>
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

namespace gold = pantr_thb_qi_golden;
using pantr::mdview;
using pantr::NdArray;
using pantr::thb::THBQuasiInterpolation;
using pantr::thb::THBSpline;
using pantr::thb::THBSplineSpace;
using thb_test::build_slack;
using thb_test::check_many_threads_are_used;
using thb_test::eight_threads;
using thb_test::eps;
using thb_test::I64;
using thb_test::Ivec;
using thb_test::k3;
using thb_test::Lcg;
using thb_test::ThreadsGuard;
using thb_test::value_error_units;
using Space = THBSplineSpace<double>;
using Qi = THBQuasiInterpolation<double>;

std::vector<Space> test_spaces() {
  std::vector<Space> out;
  out.push_back(thb_test::refined_2d(true));
  const auto b3 =
      thb_test::make_thb_space({thb_test::k2_3d, thb_test::k2_3d, thb_test::k2_3d}, {2, 2, 2},
                               {2, 2, 2}, {std::nullopt, std::nullopt, std::nullopt}, true);
  out.push_back(b3.refine(Ivec{0, 3}, std::nullopt));
  const auto b1 = thb_test::make_thb_space({k3}, {3}, {2}, {0}, true);
  out.push_back(b1.refine_region(0, Ivec{0}, Ivec{2}, std::nullopt)
                    .refine_region(1, Ivec{0}, Ivec{2}, std::nullopt));
  const auto mixed = thb_test::make_thb_space({thb_test::k2, k3}, {2, 3}, {2, 2},
                                              {std::nullopt, std::nullopt}, true);
  out.push_back(mixed.refine_region(0, Ivec{0, 0}, Ivec{2, 2}, std::nullopt));
  return out;
}

NdArray<double> random_matrix(std::size_t rows, std::size_t cols, std::uint64_t seed) {
  Lcg rng(seed);
  NdArray<double> a({rows, cols});
  for (std::size_t i = 0; i < a.size(); ++i) {
    a.data()[i] = 2.0 * rng.next() - 1.0;
  }
  return a;
}

/// The bound on the error of one coefficient `lambda(f) = sum_i w_i f(x_i)` when `|f| <= f_max` at
/// the sample points, for the functional of function `dof`.
///
/// With `Lambda_k = sum_i |w^k_i|` the 1-norm of direction `k`'s weight row, the tensor weights
/// have 1-norm `Lambda = prod_k Lambda_k`, and the error is bounded by `Lambda` times
/// `f_max` times the rounding in units of `eps` of
///   - the sampled values: `value_error_units` (the samples are values of a spline of `|c| <= 1`
///     when `f_max` is the coefficient bound; scaled to `f_max` otherwise),
///   - the contraction: one sum of `order_k` terms per direction, `sum_k order_k` in all,
///   - the weights themselves: an inverse of a local block of `order_k` collocation rows by
///     elimination carries a relative error of the order of `order_k^2 kappa eps`, and `kappa` of
///     a collocation block of non-negative rows summing to one is its inverse's norm, at least
///     the row norm `Lambda_k`. An admitted heuristic, not a derived bound: the inverse is not
///     exposed,
///   - `extra_units`, the caller's own (the construction of a float root's truncated coefficients).
/// Where the caller has shown the weights equal to those of a double root bit for bit
/// (`weights_exact`), the weights' term is dropped. The sum is multiplied by `build_slack`. `eps`
/// is that of the space's storage type `T`: a float root tabulates and builds its truncated
/// functions in float, and bounding the stages done in double by the same `eps` only loosens the
/// bound.
template <class T>
double coefficient_bound(const THBSplineSpace<T>& space, const THBQuasiInterpolation<T>& qi,
                         I64 dof, double f_max, double extra_units = 0.0,
                         bool weights_exact = false) {
  const double unit = static_cast<double>(std::numeric_limits<T>::epsilon());
  const std::span<const double> w = qi.weights(dof);
  const std::vector<int> degrees(space.degrees().begin(), space.degrees().end());
  double lambda = 1.0;
  double weight_units = 0.0;
  double orders = 0.0;
  std::size_t at = 0;
  for (const int p : degrees) {
    const std::size_t order = static_cast<std::size_t>(p) + 1;
    double lambda_k = 0.0;
    for (std::size_t i = 0; i < order; ++i) {
      lambda_k += std::abs(w[at + i]);
    }
    at += order;
    lambda *= lambda_k;
    orders += static_cast<double>(order);
    weight_units += weights_exact ? 0.0 : static_cast<double>(order * order) * lambda_k;
  }
  return build_slack * unit * lambda * f_max *
         (value_error_units(space) + orders + weight_units + extra_units);
}

TEST_CASE("THBQuasiInterpolation: the quasi-interpolant of a THB spline returns its coefficients") {
  for (const Space& space : test_spaces()) {
    const Qi qi(space);
    const auto n = static_cast<std::size_t>(space.num_total_basis());
    REQUIRE(qi.num_functions() == n);
    const NdArray<double> c = random_matrix(n, 2, 5);
    const THBSpline<double> spline(space, c);
    const NdArray<double> pts = qi.points();
    REQUIRE(pts.shape()[0] == n * qi.block());
    REQUIRE(pts.shape()[1] == space.dim());
    const NdArray<double> values = spline.evaluate(pts.view<2>());
    const NdArray<double> back = qi.coefficients(values.view<2>());
    REQUIRE(back.shape()[0] == n);
    REQUIRE(back.shape()[1] == 2);
    for (std::size_t dof = 0; dof < n; ++dof) {
      const double bound = coefficient_bound(space, qi, static_cast<I64>(dof), 1.0);
      // Not vacuous: the bound is a modest multiple of `eps` and must stay far below the error of
      // the plain basis (the control below, of the order of the coefficients); 1e-6 is both, by
      // orders of magnitude on either side.
      CHECK(bound < 1e-6);
      for (std::size_t r = 0; r < 2; ++r) {
        CHECK(std::abs(back.data()[dof * 2 + r] - c.data()[dof * 2 + r]) <= bound);
      }
    }
  }
}

/// The quasi-interpolant of a float root on dyadic knots equals the double root's, and returns the
/// coefficients of the float root's own spline to float round-off.
void check_float_root_reproduces(const THBSplineSpace<float>& sf, const Qi& qd) {
  using QiF = THBQuasiInterpolation<float>;
  const QiF qf(sf);
  const auto n = static_cast<std::size_t>(sf.num_total_basis());
  REQUIRE(qf.num_functions() == n);
  REQUIRE(qd.num_functions() == n);
  // The sample points come from the double grid's cell bounds, and the leaf cell from the Greville
  // abscissae, a mean of two knots of a quadratic, exact in float on dyadic knots: both agree
  // exactly. So do the weights: each is a row of the inverse, computed in double, of a collocation
  // block of basis values at dyadic points on dyadic knots, which the float tabulation also
  // produces exactly. (For a cubic the Greville abscissa is a mean of three knots, rounded in
  // float, and a near-tie could pick another leaf; either gives a valid quasi-interpolant, see the
  // class comment.)
  for (std::size_t dof = 0; dof < n; ++dof) {
    CHECK(qf.leaf_cell(static_cast<I64>(dof)) == qd.leaf_cell(static_cast<I64>(dof)));
    const std::span<const double> wf = qf.weights(static_cast<I64>(dof));
    const std::span<const double> wd = qd.weights(static_cast<I64>(dof));
    CHECK(std::equal(wf.begin(), wf.end(), wd.begin(), wd.end()));
  }
  const NdArray<double> pf = qf.points();
  const NdArray<double> pd = qd.points();
  REQUIRE(pf.size() == pd.size());
  CHECK(std::memcmp(pf.data(), pd.data(), pf.size() * sizeof(double)) == 0);
  const NdArray<double> c = random_matrix(n, 2, 11);
  const THBSpline<float> spline(sf, c);
  const NdArray<double> pts = qf.points();
  const NdArray<double> values = spline.evaluate(pts.view<2>());
  const NdArray<double> back = qf.coefficients(values.view<2>());
  // The float root's truncated coefficients are built in float: `levels - 1` refinements whose
  // two-scale entries carry `6 p + 1` roundings per direction (the count the B-spline product test
  // derives for an Oslo row entry).
  double alpha_units = 0.0;
  for (const int p : sf.degrees()) {
    alpha_units += 6.0 * p + 1.0;
  }
  const double extra = static_cast<double>(sf.num_levels() - 1) * alpha_units;
  for (std::size_t dof = 0; dof < n; ++dof) {
    const double bound = coefficient_bound(sf, qf, static_cast<I64>(dof), 1.0, extra, true);
    CHECK(bound < 0.05);  // not vacuous: the coefficients are of order one, the bound a few percent
    for (std::size_t r = 0; r < 2; ++r) {
      CHECK(std::abs(back.data()[dof * 2 + r] - c.data()[dof * 2 + r]) <= bound);
    }
  }
}

TEST_CASE("THBQuasiInterpolation: a float root on dyadic knots") {
  // Oracle: the double root's quasi-interpolant of the same space (exact agreement of leaf cells,
  // points and weights, where the data are dyadic), and the defining property on the float root.
  const Qi qd(thb_test::refined_2d<double>(true));
  check_float_root_reproduces(thb_test::refined_2d<float>(true), qd);
}

TEST_CASE("THBQuasiInterpolation: the plain hierarchical basis is not reproduced") {
  const Space hb = thb_test::refined_2d(false);
  const Qi qi(hb);
  const auto n = static_cast<std::size_t>(hb.num_total_basis());
  const NdArray<double> c = random_matrix(n, 1, 7);
  const THBSpline<double> spline(hb, c);
  const NdArray<double> pts = qi.points();
  const NdArray<double> back = qi.coefficients(spline.evaluate(pts.view<2>()).view<2>());
  double worst = 0.0;
  for (std::size_t dof = 0; dof < n; ++dof) {
    worst = (std::max)(worst, std::abs(back.data()[dof] - c.data()[dof]));
  }
  // A control threshold needs only to be far above the round-off level (a few `eps` times the
  // weights' size) and far below the effect it detects (the weight of the truncated-away coarse
  // components, of the order of the coefficients): 1e-3 is both, by orders of magnitude on either
  // side.
  CHECK(worst > 1e-3);
}

/// `prod_k q_{p_k}(x_k)` with `q_p(t) = sum_{i <= p} t^i / (i + 1)`: a polynomial of degree `p_k`
/// in each variable, so a member of the space.
double polynomial(const std::vector<int>& degrees, const double* x) {
  double value = 1.0;
  for (std::size_t k = 0; k < degrees.size(); ++k) {
    double q = 0.0;
    for (int i = degrees[k]; i >= 0; --i) {
      q = q * x[k] + 1.0 / static_cast<double>(i + 1);
    }
    value *= q;
  }
  return value;
}

TEST_CASE("THBQuasiInterpolation: a polynomial of the space's degree is reproduced") {
  for (const Space& space : test_spaces()) {
    const Qi qi(space);
    const std::size_t d = space.dim();
    const std::vector<int> degrees(space.degrees().begin(), space.degrees().end());
    const NdArray<double> pts = qi.points();
    const std::size_t m = pts.shape()[0];
    NdArray<double> values({m, 1});
    double f_max = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
      values.data()[i] = polynomial(degrees, pts.data() + i * d);
      f_max = (std::max)(f_max, std::abs(values.data()[i]));
    }
    const NdArray<double> c = qi.coefficients(values.view<2>());
    // The coefficients are those of the polynomial, so `|c| <= f_max` up to the bound's own size,
    // and the spline evaluates back to the polynomial: its basis is a non-negative partition of
    // unity, so the coefficients' error carries to a value unamplified, plus the value's own
    // rounding at `|c| <= f_max`.
    double c_bound = 0.0;
    for (std::size_t dof = 0; dof < qi.num_functions(); ++dof) {
      c_bound = (std::max)(c_bound, coefficient_bound(space, qi, static_cast<I64>(dof), f_max));
    }
    const THBSpline<double> spline(space, c);
    Lcg rng(9);
    std::vector<double> x(50 * d);
    for (double& v : x) {
      v = 0.001 + 0.998 * rng.next();
    }
    const NdArray<double> got = spline.evaluate(mdview<const double, 2>(x.data(), 50, d));
    const double bound = c_bound + build_slack * eps * value_error_units(space) * f_max;
    // Not vacuous: the values are of order one and the bound a modest multiple of `eps`; 1e-6 only
    // refuses a bound too loose for the comparison below to be able to fail.
    CHECK(bound < 1e-6);
    for (std::size_t i = 0; i < 50; ++i) {
      CHECK(std::abs(got.data()[i] - polynomial(degrees, x.data() + i * d)) <= bound);
    }
  }
}

/// The leaf cell of function `dof` by definition, scanned dof by dof: the candidates are the cells,
/// in increasing id, whose own level equals the function's level and on which it is listed; the
/// target is the Greville abscissa at that level of the first candidate's multi-index, from the
/// formula `(t_{i+1} + ... + t_{i+p}) / p` on the level space's knots; the nearest candidate by
/// the Euclidean distance of its centre to the target, the first of equals.
I64 brute_force_leaf(const Space& space, I64 dof) {
  const std::size_t d = space.dim();
  I64 leaf = -1;
  double best = std::numeric_limits<double>::infinity();
  std::vector<double> target;
  for (I64 cid = 0; cid < space.grid().num_cells(); ++cid) {
    const auto cell = space.contributions(cid);
    for (std::size_t j = 0; j < cell.size(); ++j) {
      if (cell.dofs[j] != dof || cell.levels[j] != space.grid().cell_level(cid)) {
        continue;
      }
      if (target.empty()) {
        const auto level_space = space.level_space(cell.levels[j]);
        const auto& dirs = level_space.spaces();
        for (std::size_t k = 0; k < d; ++k) {
          const auto knots = dirs[k].knots();
          const auto p = static_cast<std::size_t>(dirs[k].degree());
          const auto i = static_cast<std::size_t>(cell.multi_index(j)[k]);
          double sum = 0.0;
          for (std::size_t a = 1; a <= p; ++a) {
            sum += knots[i + a];
          }
          target.push_back(sum / static_cast<double>(p));
        }
      }
      const auto [lo, hi] = space.grid().cell_bounds(cid);
      double squares = 0.0;
      for (std::size_t k = 0; k < d; ++k) {
        const double diff = 0.5 * (lo[k] + hi[k]) - target[k];
        squares += diff * diff;
      }
      const double dist = std::sqrt(squares);
      if (dist < best) {
        best = dist;
        leaf = cid;
      }
    }
  }
  return leaf;
}

TEST_CASE("THBQuasiInterpolation: each function samples the leaf cell the definition names") {
  for (const Space& space : test_spaces()) {
    const Qi qi(space);
    for (I64 dof = 0; dof < space.num_total_basis(); ++dof) {
      CAPTURE(dof);
      CHECK(qi.leaf_cell(dof) == brute_force_leaf(space, dof));
    }
    // The sample points of a function lie inside its leaf cell.
    const NdArray<double> pts = qi.points();
    const std::size_t d = space.dim();
    for (I64 dof = 0; dof < space.num_total_basis(); ++dof) {
      const auto [lo, hi] = space.grid().cell_bounds(qi.leaf_cell(dof));
      for (std::size_t kk = 0; kk < qi.block(); ++kk) {
        const double* x = pts.data() + (static_cast<std::size_t>(dof) * qi.block() + kk) * d;
        for (std::size_t k = 0; k < d; ++k) {
          CHECK(lo[k] < x[k]);
          CHECK(x[k] < hi[k]);
        }
      }
    }
  }
}

TEST_CASE("THBQuasiInterpolation: equally near candidates give the smallest cell id") {
  // Degree 3 on four equal cells: the Greville abscissa of function 2 is 0.25, between cells 0 and
  // 1 (centres 0.125 and 0.375, both 0.125 away); that of function 3 is 0.5, between cells 1 and 2;
  // that of function 4 is 0.75, between cells 2 and 3. Every distance is computed exactly, so
  // each is a tie, and the first candidate in cell id wins.
  const Space space = thb_test::make_thb_space({k3}, {3}, {2}, {0}, true);
  const Qi qi(space);
  REQUIRE(space.num_total_basis() == 7);
  CHECK(qi.leaf_cell(2) == 0);
  CHECK(qi.leaf_cell(3) == 1);
  CHECK(qi.leaf_cell(4) == 2);
  // The functions with a unique nearest cell take it.
  CHECK(qi.leaf_cell(0) == 0);
  CHECK(qi.leaf_cell(6) == 3);
}

TEST_CASE("THBQuasiInterpolation: the recorded Python points and coefficients") {
  const Space space = thb_test::refined_2d(true);
  REQUIRE(space.num_total_basis() == gold::num_total);
  const Qi qi(space);
  const NdArray<double> pts = qi.points();
  REQUIRE(pts.size() == sizeof(gold::points) / sizeof(double));
  // The points are `lo + (hi - lo) i / (order + 1)` of dyadic cells: equal up to the contraction
  // of that expression, a few units of `eps` of the point.
  for (std::size_t i = 0; i < pts.size(); ++i) {
    CHECK(std::abs(pts.data()[i] - gold::points[i]) <=
          build_slack * eps * std::abs(gold::points[i]));
  }
  const std::size_t m = pts.shape()[0];
  NdArray<double> values({m, 2});
  double f_max = 0.0;
  for (std::size_t i = 0; i < m; ++i) {
    const double x = pts.data()[i * 2];
    const double y = pts.data()[i * 2 + 1];
    values.data()[i * 2] = x * x * x - 2.0 * x * y * y + 0.5 * y + 1.0 / (1.0 + x * y);
    values.data()[i * 2 + 1] = x * y + y * y * y;
    f_max = (std::max)({f_max, std::abs(values.data()[i * 2]), std::abs(values.data()[i * 2 + 1])});
  }
  const NdArray<double> c = qi.coefficients(values.view<2>());
  REQUIRE(c.size() == sizeof(gold::coeffs) / sizeof(double));
  // Two computations of one coefficient: twice the bound of the function-values' rounding.
  for (std::size_t dof = 0; dof < qi.num_functions(); ++dof) {
    const double bound = 2.0 * coefficient_bound(space, qi, static_cast<I64>(dof), f_max);
    for (std::size_t r = 0; r < 2; ++r) {
      CHECK(std::abs(c.data()[dof * 2 + r] - gold::coeffs[dof * 2 + r]) <= bound);
    }
  }
}

TEST_CASE("THBQuasiInterpolation: the refusals and their messages") {
  const Space space = thb_test::refined_2d(true);
  const Qi qi(space);
  const std::size_t rows = qi.num_functions() * qi.block();
  const NdArray<double> bad({rows - 1, 1});
  THB_THROWS_WITH_AS(qi.coefficients(bad.view<2>()),
                     ("THBQuasiInterpolation: expected " + std::to_string(rows) +
                      " rows of values, got " + std::to_string(rows - 1) + ".")
                         .c_str(),
                     pantr::value_error);
  const std::string range = " is out of range [0, " + std::to_string(qi.num_functions()) + ").";
  THB_THROWS_WITH_AS(qi.leaf_cell(-1), ("dof -1" + range).c_str(), pantr::index_error);
  THB_THROWS_WITH_AS(qi.weights(static_cast<I64>(qi.num_functions())),
                     ("dof " + std::to_string(qi.num_functions()) + range).c_str(),
                     pantr::index_error);
}

TEST_CASE("THBQuasiInterpolation: one thread and several give bit-identical coefficients") {
  check_many_threads_are_used();
  // 50 x 50 functions of three components: 72 operations each (2 x (27 + 9)), 2500 of them, above
  // the 131072 operations below which the loop stays serial.
  const auto knots = thb_test::uniform_knots(48);
  const Space space =
      thb_test::make_thb_space({knots, knots}, {2, 2}, {2, 2}, {std::nullopt, std::nullopt}, true);
  const Qi qi(space);
  const NdArray<double> values = random_matrix(qi.num_functions() * qi.block(), 3, 11);
  std::vector<NdArray<double>> results;
  for (const std::size_t threads : {std::size_t{1}, eight_threads()}) {
    const ThreadsGuard guard(threads);
    results.push_back(qi.coefficients(values.view<2>()));
  }
  REQUIRE(results[0].size() == results[1].size());
  CHECK(std::memcmp(results[0].data(), results[1].data(), results[0].size() * sizeof(double)) == 0);
}

TEST_CASE("THBQuasiInterpolation: a degree-0 space and a rank of zero") {
  // Degree 0: one sample per direction at the centre of the leaf cell, weight 1.
  const std::vector<double> knots = {0.0, 0.25, 0.5, 0.75, 1.0};
  const Space space = thb_test::make_thb_space({knots}, {0}, {2}, {-1}, true);
  const Qi qi(space);
  REQUIRE(qi.block() == 1);
  const NdArray<double> pts = qi.points();
  const NdArray<double> values = random_matrix(pts.shape()[0], 1, 13);
  const NdArray<double> c = qi.coefficients(values.view<2>());
  for (std::size_t dof = 0; dof < qi.num_functions(); ++dof) {
    CHECK(qi.weights(static_cast<I64>(dof)).size() == 1);
    CHECK(qi.weights(static_cast<I64>(dof))[0] == 1.0);
    CHECK(c.data()[dof] == values.data()[dof]);
  }
  const NdArray<double> none({pts.shape()[0], 0});
  const NdArray<double> empty = qi.coefficients(none.view<2>());
  CHECK(empty.shape()[0] == qi.num_functions());
  CHECK(empty.shape()[1] == 0);
}

}  // namespace
