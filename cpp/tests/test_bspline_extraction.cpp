// The Bézier, Lagrange and cardinal extraction operators of a 1D space, and the mask of the
// elements whose Bézier operator is structurally the identity.
//
// Oracles.
//  * The Bézier extraction operators are compared with their defining relation `N = C B`: the
//    exact value of each B-spline basis function that is nonzero on an interval (the recursive
//    definition of bspline_oracle.hpp) against the exact Bernstein form with the stored operator.
//    Tolerance, derived: an operator entry is reached from the identity by at most `p (p + 1) / 2`
//    knot-insertion steps, each `alpha a + (1 - alpha) b` with `alpha` in [0, 1], rounded to at
//    most `4 u` of the larger operand (the weight and its complement rounded, two products, one
//    sum), the operands being entries of [0, 1]; the Bernstein weights are non-negative and sum to
//    one. So the relation holds within `4 p (p + 1) u`.
//  * The Lagrange and cardinal operators are the Bézier operator times a change-of-basis matrix
//    taken from the library's process table, so a product test with that same matrix would pass for
//    a wrong kind or orientation of it. Two checks come first, independent of the table: the
//    Lagrange operator's column `k` is the exact B-spline basis at the element's `k`-th Lagrange
//    node (`C_i [Lagrange values] = [B-spline values]`, and the Lagrange basis is cardinal at its
//    nodes), and the cardinal operator reproduces the exact B-spline basis from the exact cardinal
//    basis (the B-splines of a unit uniform knot vector on its central interval). The product of
//    the stored Bézier operator with the table's matrix, in double, stays as a consistency check
//    within the bound of a sum of `p + 1` products, `(p + 1) u sum |terms|` (Higham, eq. (4.4)).
#include <doctest/doctest.h>

#include "bspline_bezier_exact.hpp"
#include "bspline_test_support.hpp"

#include <pantr/basis/lagrange_variant.hpp>
#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/extraction.hpp>
#include <pantr/change_basis.hpp>
#include <pantr/core/errors.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace bs = pantr::bspline;
namespace oracle = bspline_oracle;
using bspline_test::bernstein_exact;
using bspline_test::clamped;
using bspline_test::message_of;
using bspline_test::periodic_knots;
using bspline_test::unit_roundoff;
using oracle::Rational;
using pantr::NdArray;

namespace {

struct Space1 {
  const char* name;
  std::vector<double> knots;
  int degree;
};

std::vector<Space1> spaces_1d() {
  return {
      {"degree 0", {0.0, 0.25, 0.5, 1.0}, 0},
      {"degree 1", clamped(1, {0.25, 0.5}), 1},
      {"Bezier-like", clamped(3, {}), 3},
      {"open, simple knots", clamped(2, {0.25, 0.5, 0.75}), 2},
      {"open, a double knot in a cubic", clamped(3, {0.25, 0.5, 0.5, 0.75}), 3},
      {"a knot of multiplicity degree", clamped(2, {0.5, 0.5}), 2},
      {"a knot of multiplicity degree + 1", clamped(2, {0.5, 0.5, 0.5}), 2},
      {"unclamped", {-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}, 2},
      {"unclamped cubic, uneven",
       {-0.75, -0.5, -0.25, 0.0, 0.125, 0.5, 0.875, 1.0, 1.25, 1.5, 1.75},
       3},
      {"periodic-style ghost knots", periodic_knots(4, 3), 3},
      {"degree 4, off-mesh", clamped(4, {0.1, 0.35, 0.6}), 4},
  };
}

template <class T>
bs::BsplineSpace1D<T> make_space(const Space1& s) {
  const std::vector<T> k = bspline_test::cast_all<T>(s.knots);
  return bs::BsplineSpace1D<T>(std::span<const T>(k), s.degree);
}

}  // namespace

TEST_CASE_TEMPLATE("Bezier extraction operators: N = C B on every interval", T, double, float) {
  for (const Space1& s : spaces_1d()) {
    CAPTURE(s.name);
    const bs::BsplineSpace1D<T> space = make_space<T>(s);
    const NdArray<T> ops = bs::bezier_extraction_operators<T>(space);
    const std::size_t order = static_cast<std::size_t>(s.degree) + 1;
    const std::size_t n_int = static_cast<std::size_t>(space.num_intervals());
    REQUIRE(ops.shape().size() == 3);
    CHECK(ops.shape()[0] == n_int);
    CHECK(ops.shape()[1] == order);
    CHECK(ops.shape()[2] == order);
    const auto [uk, um] = space.unique_knots_and_multiplicity(true);
    const auto first = space.first_basis_per_interval();
    const double p = static_cast<double>(s.degree);
    const double tol = 4.0 * p * (p + 1.0) * unit_roundoff<T>();
    const std::vector<Rational> knots = oracle::exact_knots<T>(space.knots());
    for (std::size_t i = 0; i < n_int; ++i) {
      const Rational a = oracle::exact(static_cast<double>(uk[i]));
      const Rational b = oracle::exact(static_cast<double>(uk[i + 1]));
      // Samples across the whole interval, none at its ends.
      for (const Rational& q :
           {Rational(1, 16), Rational(3, 8), Rational(5, 8), Rational(15, 16)}) {
        const Rational x = a + (b - a) * q;
        oracle::BasisAt at(knots, s.degree, x);
        for (std::size_t j = 0; j < order; ++j) {
          Rational got = 0;
          for (std::size_t k = 0; k < order; ++k) {
            got += oracle::exact(static_cast<double>(ops.data()[(i * order + j) * order + k])) *
                   bernstein_exact(s.degree, static_cast<int>(k), q);
          }
          const Rational want = at.deriv(static_cast<std::size_t>(first[i]) + j, 0);
          CHECK(oracle::within(static_cast<double>(got), want, tol));
        }
      }
      // The columns of an operator sum to one: the basis is a partition of unity.
      for (std::size_t k = 0; k < order; ++k) {
        double sum = 0.0;
        for (std::size_t j = 0; j < order; ++j) {
          sum += static_cast<double>(ops.data()[(i * order + j) * order + k]);
        }
        CHECK(std::abs(sum - 1.0) <= tol);
      }
    }
  }
}

TEST_CASE_TEMPLATE("Bezier extraction operators: a raw knot vector agrees with a space", T, double,
                   float) {
  for (const Space1& s : spaces_1d()) {
    CAPTURE(s.name);
    const bs::BsplineSpace1D<T> space = make_space<T>(s);
    const NdArray<T> by_space = bs::bezier_extraction_operators<T>(space);
    const NdArray<T> by_raw =
        bs::bezier_extraction_operators<T>(space.knots(), s.degree, space.tolerance());
    REQUIRE(by_space.size() == by_raw.size());
    CHECK(std::equal(by_space.flat().begin(), by_space.flat().end(), by_raw.flat().begin()));
    // Into a caller's array.
    NdArray<T> into(std::vector<std::size_t>(by_space.shape().begin(), by_space.shape().end()));
    bs::bezier_extraction_operators<T>(space.knots(), s.degree, space.tolerance(),
                                       into.template view<3>());
    CHECK(std::equal(by_space.flat().begin(), by_space.flat().end(), into.flat().begin()));
  }
}

TEST_CASE("extraction operators: refusals and their messages") {
  const std::vector<double> ok = {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
  const auto sp = [](const std::vector<double>& v) { return std::span<const double>(v); };
  CHECK(message_of([&] { (void)bs::bezier_extraction_operators<double>(sp(ok), 2, -1.0); }) ==
        "tol must be non-negative");
  CHECK(message_of([&] { (void)bs::bezier_extraction_operators<double>(sp(ok), -1, 1e-10); }) ==
        "degree must be non-negative");
  CHECK(message_of([&] {
          (void)bs::bezier_extraction_operators<double>(sp({0.0, 1.0}), 2, 1e-10);
        }) == "knots must have at least 2*degree+2 elements");
  CHECK(message_of([&] {
          (void)bs::bezier_extraction_operators<double>(sp({0.0, 1.0, 0.5, 1.0, 1.0, 1.0}), 2,
                                                        1e-10);
        }) == "knots must be non-decreasing");
  CHECK(message_of([&] {
          (void)bs::bezier_extraction_operators<double>(
              sp({0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 1.0, 1.0, 1.0}), 2, 1e-10);
        }) == "knots must be non-decreasing");
  // A tolerance that is not a finite number is refused, whatever the knots: a NaN passes `tol < 0`.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  CHECK(message_of([&] { (void)bs::bezier_extraction_operators<double>(sp(ok), 2, nan); }) ==
        "tol must be finite, got nan");
  CHECK(message_of([&] { (void)bs::bezier_extraction_operators<double>(sp(ok), 2, inf); }) ==
        "tol must be finite, got inf");
  CHECK(message_of([&] { (void)bs::bezier_extraction_operators<double>(sp(ok), 2, -inf); }) ==
        "tol must be non-negative");
  // The output extents are checked after the input.
  NdArray<double> wrong({1, 3, 3});
  CHECK(message_of([&] {
          bs::bezier_extraction_operators<double>(sp(ok), 2, 1e-10, wrong.view<3>());
        }) == "Output array has shape (1, 3, 3), but expected shape (2, 3, 3)");
  CHECK(message_of([&] {
          bs::lagrange_extraction_operators<double>(
              sp(ok), 2, -1.0, pantr::basis::LagrangeVariant::equispaces, wrong.view<3>());
        }) == "tol must be non-negative");
  CHECK(message_of([&] {
          bs::cardinal_extraction_operators<double>(sp(ok), 2, 1e-10, wrong.view<3>());
        }) == "Output array has shape (1, 3, 3), but expected shape (2, 3, 3)");
  // A degree beyond the change-of-basis matrix is refused before anything is written.
  const std::vector<double> huge = bspline_test::clamped(40, {0.5});
  NdArray<double> big({2, 41, 41});
  std::fill(big.flat().begin(), big.flat().end(), -5.0);
  CHECK_THROWS_AS(bs::cardinal_extraction_operators<double>(sp(huge), 40, 1e-10, big.view<3>()),
                  pantr::value_error);
  CHECK(std::all_of(big.flat().begin(), big.flat().end(), [](double v) { return v == -5.0; }));
}

TEST_CASE("extraction operators: a tolerance that merges the domain into one knot is refused") {
  // Knots 0, 0.1, ..., 0.7 at degree 3: the domain [0.3, 0.4] is one class at a tolerance of 0.15
  // (every step is within it, and a class ends only where a step exceeds it), so there is no
  // interval. Before the check the start-insertion block wrote into an output of zero intervals.
  const std::vector<double> knots = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7};
  const auto sp = [](const std::vector<double>& v) { return std::span<const double>(v); };
  const int degree = 3;
  const double tol = 0.15;
  // The same words as the space that holds the same tolerance and knots refuses with.
  const std::string want = message_of(
      [&] { (void)bs::BsplineSpace1D<double>::from_state(sp(knots), degree, false, tol); });
  REQUIRE(want.rfind("knot vector spans no interval: at degree 3", 0) == 0);

  const std::vector<double> sentinel(4 * 4, -5.0);
  NdArray<double> out({1, 4, 4});
  std::copy(sentinel.begin(), sentinel.end(), out.flat().begin());
  CHECK(message_of([&] {
          (void)bs::bezier_extraction_operators<double>(sp(knots), degree, tol);
        }) == want);
  CHECK(message_of([&] {
          bs::bezier_extraction_operators<double>(sp(knots), degree, tol, out.view<3>());
        }) == want);
  CHECK(message_of([&] {
          bs::lagrange_extraction_operators<double>(
              sp(knots), degree, tol, pantr::basis::LagrangeVariant::equispaces, out.view<3>());
        }) == want);
  CHECK(message_of([&] {
          bs::cardinal_extraction_operators<double>(sp(knots), degree, tol, out.view<3>());
        }) == want);
  // Nothing was written, and the refusal is the tolerance's: the same knots at a tolerance below
  // every step are fine and give the one interval the domain has.
  CHECK(std::equal(sentinel.begin(), sentinel.end(), out.flat().begin()));
  CHECK(bs::bezier_extraction_operators<double>(sp(knots), degree, 1e-3).shape()[0] == 1);
}

TEST_CASE_TEMPLATE(
    "Lagrange and cardinal extraction operators: the Bezier operator times the "
    "change of basis",
    T, double, float) {
  using pantr::basis::LagrangeVariant;
  for (const Space1& s : spaces_1d()) {
    if (s.degree > 4) {
      continue;
    }
    CAPTURE(s.name);
    const bs::BsplineSpace1D<T> space = make_space<T>(s);
    const std::size_t order = static_cast<std::size_t>(s.degree) + 1;
    const std::size_t n_int = static_cast<std::size_t>(space.num_intervals());
    const NdArray<T> bez = bs::bezier_extraction_operators<T>(space);
    const double u = unit_roundoff<T>();

    // The product `bez[i] M` in double, and the bound on the stored result against it.
    const auto check_product = [&](const NdArray<T>& got, const NdArray<T>& matrix,
                                   const std::vector<bool>& identity) {
      for (std::size_t i = 0; i < n_int; ++i) {
        for (std::size_t r = 0; r < order; ++r) {
          for (std::size_t c = 0; c < order; ++c) {
            double want = 0.0;
            double terms = 0.0;
            for (std::size_t k = 0; k < order; ++k) {
              const double t = static_cast<double>(bez.data()[(i * order + r) * order + k]) *
                               static_cast<double>(matrix.data()[k * order + c]);
              want += t;
              terms += std::abs(t);
            }
            const double have = static_cast<double>(got.data()[(i * order + r) * order + c]);
            if (identity[i]) {
              CHECK(have == (r == c ? 1.0 : 0.0));
            } else {
              CHECK(std::abs(have - want) <= 2.0 * static_cast<double>(order) * u * terms);
            }
          }
        }
      }
    };

    std::vector<LagrangeVariant> variants = {LagrangeVariant::equispaces,
                                             LagrangeVariant::gauss_legendre,
                                             LagrangeVariant::chebyshev_1st};
    if (s.degree >= 1) {  // the Lobatto families need both ends
      variants.push_back(LagrangeVariant::gauss_lobatto_legendre);
      variants.push_back(LagrangeVariant::chebyshev_2nd);
    }
    if (s.degree == 0) {
      // No Lagrange basis of degree 0 is built (the change-of-basis matrix is refused).
      NdArray<T> none({n_int, order, order}, pantr::for_overwrite);
      CHECK(message_of([&] {
              bs::lagrange_extraction_operators<T>(space, LagrangeVariant::equispaces,
                                                   none.template view<3>());
            }) == "Degree must at least 1");
      variants.clear();
    }
    for (const LagrangeVariant variant : variants) {
      NdArray<T> got({n_int, order, order}, pantr::for_overwrite);
      bs::lagrange_extraction_operators<T>(space, variant, got.template view<3>());
      const auto matrix = pantr::change_basis::cached<T>(
          pantr::change_basis::Kind::lagrange_to_bernstein, s.degree, variant);
      check_product(got, *matrix, std::vector<bool>(n_int, false));
    }

    NdArray<T> card({n_int, order, order}, pantr::for_overwrite);
    bs::cardinal_extraction_operators<T>(space, card.template view<3>());
    const auto cmatrix = pantr::change_basis::cached<T>(
        pantr::change_basis::Kind::cardinal_to_bernstein, s.degree, LagrangeVariant::equispaces);
    std::unique_ptr<bool[]> flags(new bool[n_int]);
    space.cardinal_intervals(std::span<bool>(flags.get(), n_int));
    check_product(card, *cmatrix, std::vector<bool>(flags.get(), flags.get() + n_int));
  }
}

namespace {

/// The bound on the relative error of a stored Bernstein value of degree `p >= 1`, in units of the
/// unit roundoff `u`: the ratio recurrence takes `p` steps, each rounding the ratio (two
/// roundings), the product with the constant factor (two) and the stored value (one), `5 p`, the
/// seed by binary powering (`2 ceil(log2 p)` roundings, `1 - x` and the store, `2`) and the seed
/// `1 - x` once more, so at most `6 p + 2 ceil(log2 p) + 2`.
double bernstein_value_roundings(int p) {
  return 6.0 * p + 2.0 * std::ceil(std::log2(static_cast<double>(p))) + 2.0;
}

}  // namespace

TEST_CASE_TEMPLATE(
    "Lagrange extraction operators: column k is the B-spline basis at the element's k-th node", T,
    double, float) {
  // `C_i [Lagrange values] = [B-spline values]` and the Lagrange basis is cardinal at its own nodes
  // (`L_j(x_k) = delta_jk`), so column `k` of `C_i` is the vector of B-spline functions of the
  // interval at the interval's `k`-th node. Nothing of the change-of-basis table enters the
  // expected value: the nodes are the variant's, the basis is the exact Cox-de Boor recurrence on
  // the piece of this element (also at the element's right end, where the next piece would be
  // picked by position).
  //
  // Tolerance, derived (`p` the degree, `u` the unit roundoff of `T`): the stored Bézier operator
  // satisfies the relation `N = C B` within `4 p (p + 1) u` (the Bézier test above); the stored
  // Bernstein values at the nodes carry a relative error of at most `bernstein_value_roundings(p)
  // u` each, which the product weighs by the operator entries (in [0, 1]) and the sum of the
  // products is `N_j(x_k) <= 1`; the product itself is a sum of `p + 1` products, `(p + 1) u` of
  // `sum |terms| = N_j(x_k) <= 1`.
  using pantr::basis::LagrangeVariant;
  const LagrangeVariant variants[] = {LagrangeVariant::equispaces, LagrangeVariant::gauss_legendre,
                                      LagrangeVariant::gauss_lobatto_legendre,
                                      LagrangeVariant::chebyshev_1st,
                                      LagrangeVariant::chebyshev_2nd};
  for (const Space1& s : spaces_1d()) {
    if (s.degree < 1 || s.degree > 4) {
      continue;  // degree 0 has no Lagrange basis; degree > 4 only repeats the argument
    }
    CAPTURE(s.name);
    const bs::BsplineSpace1D<T> space = make_space<T>(s);
    const std::size_t order = static_cast<std::size_t>(s.degree) + 1;
    const std::size_t n_int = static_cast<std::size_t>(space.num_intervals());
    const auto [uk, um] = space.unique_knots_and_multiplicity(true);
    const auto first = space.first_basis_per_interval();
    const std::vector<Rational> knots = oracle::exact_knots<T>(space.knots());
    const double p = static_cast<double>(s.degree);
    const double tol = (4.0 * p * (p + 1.0) + bernstein_value_roundings(s.degree) + (p + 1.0)) *
                       unit_roundoff<T>();
    for (const LagrangeVariant variant : variants) {
      CAPTURE(pantr::basis::variant_name(variant));
      NdArray<T> got({n_int, order, order}, pantr::for_overwrite);
      bs::lagrange_extraction_operators<T>(space, variant, got.template view<3>());
      const NdArray<T> nodes =
          pantr::basis::lagrange_points<T>(variant, static_cast<long long>(order));
      for (std::size_t i = 0; i < n_int; ++i) {
        const Rational a = oracle::exact(static_cast<double>(uk[i]));
        const Rational b = oracle::exact(static_cast<double>(uk[i + 1]));
        const std::size_t piece = oracle::span_of(knots, s.degree, (a + b) / 2);
        for (std::size_t k = 0; k < order; ++k) {
          const Rational x = a + (b - a) * oracle::exact(static_cast<double>(nodes.data()[k]));
          oracle::BasisAt at(knots, s.degree, x, piece);
          for (std::size_t j = 0; j < order; ++j) {
            const Rational want = at.deriv(static_cast<std::size_t>(first[i]) + j, 0);
            CHECK(oracle::within(static_cast<double>(got.data()[(i * order + j) * order + k]), want,
                                 tol));
          }
        }
      }
    }
  }
}

namespace {

/// The largest absolute column sum of a square row-major matrix.
double max_column_sum(std::span<const double> m, std::size_t n) {
  double worst = 0.0;
  for (std::size_t c = 0; c < n; ++c) {
    double sum = 0.0;
    for (std::size_t r = 0; r < n; ++r) {
      sum += std::abs(m[r * n + c]);
    }
    worst = (std::max)(worst, sum);
  }
  return worst;
}

/// The largest absolute row sum of a square row-major matrix.
double max_row_sum(std::span<const double> m, std::size_t n) {
  double worst = 0.0;
  for (std::size_t r = 0; r < n; ++r) {
    double sum = 0.0;
    for (std::size_t c = 0; c < n; ++c) {
      sum += std::abs(m[r * n + c]);
    }
    worst = (std::max)(worst, sum);
  }
  return worst;
}

}  // namespace

TEST_CASE_TEMPLATE(
    "cardinal extraction operators: the exact B-spline basis from the exact cardinal basis", T,
    double, float) {
  // The cardinal basis of degree `p` on `[0, 1]` is the set of `p + 1` B-splines of the unit
  // uniform knot vector `-p, ..., p + 1` on its central interval, so the exact Cox-de Boor
  // recurrence gives it with no change-of-basis table. The operator maps it to the B-splines of the
  // interval: `N_{first + r}(a + (b - a) q) = sum_k C[r][k] cardinal_k(q)`. A cardinal interval
  // (uniform over its window) has the identity as its operator, which this holds to as well: the
  // knots of such an interval here are dyadic, so the identity is exact.
  //
  // Tolerance. `C = Bez M`, `M` the cardinal-to-Bernstein matrix. The Bézier part is within `4 p
  // (p + 1) u` as in the Bézier test (the cardinal basis is non-negative and sums to one). `M` is
  // a stored inverse: its entries are off by about `c kappa u ||M||`, `kappa` the condition number
  // of the matrix it inverts and `c` taken as `2 (p + 1)`, an admitted heuristic (the normwise
  // inverse error scale, no sharper constant derived); the sum over the `p + 1` Bézier entries
  // (in [0, 1]) multiplies it by `p + 1`, the cardinal basis weighing the columns as a convex
  // combination. The product is a sum of `p + 1` products, `(p + 1) u` of `sum |terms|`, which the
  // convex weights bound by the largest column sum of `|M|`. `kappa` and the norms are read from
  // the double matrices of the library: they only scale the tolerance.
  using pantr::basis::LagrangeVariant;
  const std::vector<double> sample = {1.0 / 16.0, 3.0 / 8.0, 5.0 / 8.0, 15.0 / 16.0};
  for (const Space1& s : spaces_1d()) {
    if (s.degree > 4) {
      continue;
    }
    CAPTURE(s.name);
    const bs::BsplineSpace1D<T> space = make_space<T>(s);
    const std::size_t order = static_cast<std::size_t>(s.degree) + 1;
    const std::size_t n_int = static_cast<std::size_t>(space.num_intervals());
    const auto [uk, um] = space.unique_knots_and_multiplicity(true);
    const auto first = space.first_basis_per_interval();
    const std::vector<Rational> knots = oracle::exact_knots<T>(space.knots());
    std::vector<Rational> unit;  // -p, ..., p + 1
    for (int k = -s.degree; k <= s.degree + 1; ++k) {
      unit.emplace_back(k);
    }

    const double p = static_cast<double>(s.degree);
    const double u = unit_roundoff<T>();
    const NdArray<double> to_card = pantr::change_basis::bernstein_to_cardinal<double>(s.degree);
    const NdArray<double> to_bern = pantr::change_basis::cardinal_to_bernstein<double>(s.degree);
    const double kappa = max_row_sum(to_card.flat(), order) * max_row_sum(to_bern.flat(), order);
    const double matrix_error = 2.0 * (p + 1.0) * kappa * u * max_row_sum(to_bern.flat(), order);
    const double tol = 4.0 * p * (p + 1.0) * u + (p + 1.0) * matrix_error +
                       (p + 1.0) * u * max_column_sum(to_bern.flat(), order);

    NdArray<T> got({n_int, order, order}, pantr::for_overwrite);
    bs::cardinal_extraction_operators<T>(space, got.template view<3>());
    for (std::size_t i = 0; i < n_int; ++i) {
      const Rational a = oracle::exact(static_cast<double>(uk[i]));
      const Rational b = oracle::exact(static_cast<double>(uk[i + 1]));
      for (const double q : sample) {
        const Rational qx = oracle::exact(q);
        oracle::BasisAt at(knots, s.degree, a + (b - a) * qx);
        oracle::BasisAt card(unit, s.degree, qx);
        for (std::size_t r = 0; r < order; ++r) {
          Rational sum = 0;
          for (std::size_t k = 0; k < order; ++k) {
            sum += oracle::exact(static_cast<double>(got.data()[(i * order + r) * order + k])) *
                   card.deriv(k, 0);
          }
          Rational diff = sum - at.deriv(static_cast<std::size_t>(first[i]) + r, 0);
          if (diff < 0) {
            diff = -diff;
          }
          CHECK(diff <= oracle::exact(tol));
        }
      }
    }
  }
}

TEST_CASE(
    "cardinal extraction operators: a uniform knot vector is the identity on its cardinal "
    "intervals, and nowhere else is") {
  const std::vector<double> knots = {0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0};
  const bs::BsplineSpace1D<double> space(std::span<const double>(knots), 2);
  const NdArray<double> ops = [&] {
    NdArray<double> o({6, 3, 3}, pantr::for_overwrite);
    bs::cardinal_extraction_operators<double>(space, o.view<3>());
    return o;
  }();
  // Six unit intervals of degree 2: only the two end intervals see the repeated end knots.
  const std::vector<bool> cardinal = {false, true, true, true, true, false};
  for (std::size_t i = 0; i < 6; ++i) {
    bool is_identity = true;
    for (std::size_t r = 0; r < 3; ++r) {
      for (std::size_t c = 0; c < 3; ++c) {
        is_identity = is_identity && ops.data()[(i * 3 + r) * 3 + c] == (r == c ? 1.0 : 0.0);
      }
    }
    CHECK(is_identity == cardinal[i]);
  }
}

TEST_CASE_TEMPLATE(
    "Bezier structural identity mask: both bounding knots have multiplicity "
    "degree + 1",
    T, double, float) {
  struct Case {
    std::vector<double> knots;
    int degree;
    std::vector<bool> mask;
  };
  const std::vector<Case> cases = {
      {clamped(2, {0.5, 0.5, 0.5}), 2, {true, true}},
      {clamped(2, {0.5, 0.5}), 2, {false, false}},
      {clamped(2, {0.25, 0.5, 0.5, 0.5}), 2, {false, false, true}},
      {clamped(3, {}), 3, {true}},
      {{0.0, 0.5, 1.0}, 0, {true, true}},  // degree 0: multiplicity 1 = degree + 1 at every knot
      {clamped(1, {0.5, 0.5}), 1, {true, true}},
  };
  for (const Case& c : cases) {
    const std::vector<T> k = bspline_test::cast_all<T>(c.knots);
    const bs::BsplineSpace1D<T> space(std::span<const T>(k), c.degree);
    const std::size_t n = static_cast<std::size_t>(space.num_intervals());
    REQUIRE(n == c.mask.size());
    std::unique_ptr<bool[]> got(new bool[n]);
    bs::bezier_structural_identity_mask<T>(space, std::span<bool>(got.get(), n));
    for (std::size_t i = 0; i < n; ++i) {
      CHECK(got[i] == c.mask[i]);
    }
    // An identity-flagged interval has the identity as its Bézier operator.
    const NdArray<T> ops = bs::bezier_extraction_operators<T>(space);
    const std::size_t order = static_cast<std::size_t>(c.degree) + 1;
    for (std::size_t i = 0; i < n; ++i) {
      if (!got[i]) {
        continue;
      }
      for (std::size_t r = 0; r < order; ++r) {
        for (std::size_t q = 0; q < order; ++q) {
          CHECK(ops.data()[(i * order + r) * order + q] == (r == q ? T(1) : T(0)));
        }
      }
    }
  }
  // The flags array must hold one flag per interval.
  const std::vector<double> kd = clamped(2, {0.5});
  const bs::BsplineSpace1D<double> two(std::span<const double>(kd), 2);
  std::unique_ptr<bool[]> wrong(new bool[3]);
  CHECK(message_of([&] {
          bs::bezier_structural_identity_mask<double>(two, std::span<bool>(wrong.get(), 3));
        }) == "Output array has shape (3,), but expected shape (2,)");
}
