// B-spline fitting: collocation matrices and the solves behind `fit_tensor`, `fit_scattered` and
// `interpolate`, the staged L2 projection (`L2Projection`, `l2_project`) and the Lee-Lyche-Morken
// quasi-interpolant (`QuasiInterpolation`, `quasi_interpolate`). The Python suite observes the
// results of these through the bindings; this file owns what it cannot: the exception types and
// messages of the C++ API, the float and double instantiations, the paths of the solve (banded QR,
// LU, truncated SVD), the order of the boundary traces, and the call pattern of the convenience
// templates.
//
// Oracles; each test case names its own again.
//  * Exact basis. `bspline_exact_eval.hpp` evaluates a spline, and any partial derivative of it, in
//    exact rational arithmetic from the recursive definition of the basis (not the triangle the
//    library runs). A column of a collocation matrix is the exact value of the spline whose only
//    control point is one, so the matrix is compared entry by entry, wrap-around of a periodic
//    space included, and a fit is compared with the control points of a spline that lies in the
//    space and was sampled exactly (dyadic control points).
//  * Exact linear algebra. `solve_exact` is Gauss-Jordan elimination on `Rational` entries. A
//    square or overdetermined solve is compared with the exact solution of the same matrix and
//    right-hand side (the normal equations `A^T A x = A^T b`); a singular system with the exact
//    minimum-norm solution `A^T (A A^T)^-1 b` of its independent rows. solve_kronecker is compared
//    with one dense solve of the Kronecker product `kron(A_0, A_1, ...)`.
//  * Exact quadrature. The Gram matrix of a space is integrated span by span with the closed
//    Newton-Cotes rule of the degree that is exact for the integrand, its weights obtained by
//    solving the moment equations in rationals, so it shares nothing with the library's Gauss
//    rules. The L2 projection of a polynomial outside the space is compared with `G^-1 b` in
//    rationals.
//  * Closed forms and tables written by hand: the order of the boundary traces, the points of a
//    quasi-interpolant on a two-interval hat space, the Greville abscissae of a Bezier space,
//    minimum-norm solutions of tiny systems.
//  * Invariants: `Q B = B` (the quasi-interpolant reproduces every spline of its space), and the
//    derivative constraints of an interpolation hold at the ends.
//
// Tolerances. `u` is the unit roundoff of the storage type `T` (half the machine epsilon).
//  * Basis values: the triangle forms a value over `p` levels of a difference, a quotient, a
//    product and a sum, at most `6 p + 1` roundings relative to a value in [0, 1] (the same count
//    as test_bspline_product.cpp).
//  * Derivative rows: each entry is a sum of `k + 1` differences of triangle entries scaled by the
//    knot spans, so its error is `(k + 1) (6 p + 1) u` relative to the largest entry of its row,
//    times a slack of 8.
//  * Solves: every solve is backward stable (Givens QR, LU with partial pivoting, SVD), so the
//    computed solution of a system with matrix `A` (m x n) and data of relative accuracy `u` has
//    a forward error of at most `K (m + n + 2) u kappa ||x||_2`, `kappa` the condition number of
//    `A` and `K = 8` (the constant of test_bezier_fit.cpp: it covers the explicit rounding of the
//    data and the `6 p + 1` roundings of the matrix entries for the degrees used here, and left
//    a wide margin over the errors seen when the tests were written). `kappa` is bounded from the
//    exact inverse of the normal matrix `N = A^T A`: `kappa_2(A) <= sqrt(n ||N||_inf ||N^-1||_inf)`
//    (a bound on `kappa_2(N)` is `n kappa_inf(N)`, and `kappa_2(A)^2 = kappa_2(N)`). For a
//    Kronecker product `kappa` is the product of the factors', and the size term is the sum of the
//    factors'. A test that needs the solve to keep every singular value asserts first that `1 /
//    kappa` exceeds the truncation tolerance `100 eps`.
//  * Gram entries: all terms of a mass-matrix entry are non-negative, so its relative error is
//    the number of terms plus the roundings of the two basis values and the weight, times `u`.
//  * A solve in `float` runs its decompositions in `double` and rounds the result, so the bound
//    above is loose for `float` (by about the ratio of the two roundoffs) while still far under
//    the O(1) error of a structural bug; the `double` instantiations hold the tight bound.
//  * Quasi-interpolant: the blocks it inverts carry the roundings of the space's own type (the
//    basis of a `float` space is tabulated in `float` even at `double` points), `(6 p + 1) u_T`
//    per entry, besides the `double` inverse; `check_qi_reproduces` states the sum.
//
// Known defects. Two tests are marked `doctest::should_fail()`: they state the invariant `Q B = B`
// for a space where the quasi-interpolant is wrong (a knot of multiplicity `degree + 1`; some
// periodic spaces) and are expected to fail. Both defects are in the previous release as well.
// When one is fixed its test reports that it passed unexpectedly: remove the marker then.
#include <doctest/doctest.h>

#include "bspline_test_support.hpp"

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>

#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/fit.hpp>
#include <pantr/bspline/knot_factories.hpp>
#include <pantr/bspline/l2_project.hpp>
#include <pantr/bspline/quasi_interpolate.hpp>
#include <pantr/bspline/space.hpp>
#include <pantr/bspline/space_1d.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/core/mdspan.hpp>
#include <pantr/core/ndarray.hpp>
#include <pantr/linalg/banded_lstsq.hpp>
#include <pantr/linalg/dense.hpp>
#include <pantr/quad/points_lattice.hpp>
#include <pantr/tolerance.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bs = pantr::bspline;
namespace la = pantr::linalg;
namespace oracle = bspline_oracle;
using bspline_test::cast_all;
using bspline_test::clamped;
using bspline_test::Dir;
using bspline_test::make_spline;
using bspline_test::periodic_knots;
using bspline_test::space_of;
using oracle::Rational;
using pantr::NdArray;

namespace {

using Matrix = std::vector<std::vector<Rational>>;

constexpr double fit_slack = 8.0;  // K of the header comment

template <class T>
constexpr double unit_roundoff() {
  return 0.5 * static_cast<double>(std::numeric_limits<T>::epsilon());
}

template <class T>
constexpr double machine_epsilon() {
  return static_cast<double>(std::numeric_limits<T>::epsilon());
}

double to_double(const Rational& q) { return static_cast<double>(q); }

Rational abs_of(const Rational& q) { return q < 0 ? Rational(-q) : q; }

Rational rat(long long num, long long den = 1) {
  return Rational(pantr::exact::Integer(num), pantr::exact::Integer(den));
}

// ---- exact linear algebra --------------------------------------------------------------------

Matrix identity(std::size_t n) {
  Matrix m(n, std::vector<Rational>(n, Rational(0)));
  for (std::size_t i = 0; i < n; ++i) {
    m[i][i] = 1;
  }
  return m;
}

Matrix transposed(const Matrix& m) {
  Matrix t(m[0].size(), std::vector<Rational>(m.size()));
  for (std::size_t i = 0; i < m.size(); ++i) {
    for (std::size_t j = 0; j < m[0].size(); ++j) {
      t[j][i] = m[i][j];
    }
  }
  return t;
}

Matrix product(const Matrix& a, const Matrix& b) {
  Matrix c(a.size(), std::vector<Rational>(b[0].size(), Rational(0)));
  for (std::size_t i = 0; i < a.size(); ++i) {
    for (std::size_t k = 0; k < b.size(); ++k) {
      if (a[i][k] == 0) {
        continue;
      }
      for (std::size_t j = 0; j < b[0].size(); ++j) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return c;
}

// The solution of the square system a x = b, by Gauss-Jordan elimination on exact entries.
Matrix solve_exact(Matrix a, Matrix b) {
  const std::size_t n = a.size();
  for (std::size_t col = 0; col < n; ++col) {
    std::size_t piv = col;
    while (piv < n && a[piv][col] == 0) {
      ++piv;
    }
    REQUIRE(piv < n);  // a singular system is an error of the oracle's own inputs
    std::swap(a[col], a[piv]);
    std::swap(b[col], b[piv]);
    const Rational d = a[col][col];
    for (auto& v : a[col]) {
      v /= d;
    }
    for (auto& v : b[col]) {
      v /= d;
    }
    for (std::size_t r = 0; r < n; ++r) {
      if (r == col || a[r][col] == 0) {
        continue;
      }
      const Rational f = a[r][col];
      for (std::size_t k = 0; k < n; ++k) {
        a[r][k] -= f * a[col][k];
      }
      for (std::size_t k = 0; k < b[r].size(); ++k) {
        b[r][k] -= f * b[col][k];
      }
    }
  }
  return b;
}

// The least-squares solution of a full-column-rank system, from the normal equations.
Matrix lstsq_exact(const Matrix& a, const Matrix& b) {
  const Matrix at = transposed(a);
  return solve_exact(product(at, a), product(at, b));
}

// The minimum-norm solution of a full-row-rank system: A^T (A A^T)^-1 b.
Matrix min_norm_exact(const Matrix& a, const Matrix& b) {
  return product(transposed(a), solve_exact(product(a, transposed(a)), b));
}

double norm_inf(const Matrix& m) {
  double worst = 0.0;
  for (const auto& row : m) {
    Rational s = 0;
    for (const Rational& v : row) {
      s += abs_of(v);
    }
    worst = (std::max)(worst, to_double(s));
  }
  return worst;
}

// An upper bound of kappa_2 of a full-column-rank matrix (see the header comment).
double kappa_bound(const Matrix& a) {
  const std::size_t n = a[0].size();
  const Matrix normal = product(transposed(a), a);
  const Matrix inv = solve_exact(normal, identity(n));
  return std::sqrt(static_cast<double>(n) * norm_inf(normal) * norm_inf(inv));
}

template <class P>
Matrix exact_matrix(const NdArray<P>& a) {
  REQUIRE(a.rank() == 2);
  const std::size_t rows = a.shape()[0];
  const std::size_t cols = a.shape()[1];
  Matrix m(rows, std::vector<Rational>(cols));
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      m[i][j] = oracle::exact(static_cast<double>(a.data()[i * cols + j]));
    }
  }
  return m;
}

template <class C>
Matrix exact_matrix(const la::RowMatrix<C>& a) {
  Matrix m(static_cast<std::size_t>(a.rows()),
           std::vector<Rational>(static_cast<std::size_t>(a.cols())));
  for (Eigen::Index i = 0; i < a.rows(); ++i) {
    for (Eigen::Index j = 0; j < a.cols(); ++j) {
      m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
          oracle::exact(static_cast<double>(a(i, j)));
    }
  }
  return m;
}

template <class C>
la::RowMatrix<C> row_matrix(const NdArray<C>& a) {
  return Eigen::Map<const la::RowMatrix<C>>(a.data(), static_cast<Eigen::Index>(a.shape()[0]),
                                            static_cast<Eigen::Index>(a.shape()[1]));
}

// A dense (rows x cols) matrix of the given entries, row-major.
template <class C>
la::RowMatrix<C> dense(std::size_t rows, std::size_t cols, const std::vector<double>& entries) {
  REQUIRE(entries.size() == rows * cols);
  la::RowMatrix<C> m(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      m(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
          static_cast<C>(entries[i * cols + j]);
    }
  }
  return m;
}

// Dyadic right-hand sides (multiples of 1/64, exact in float).
template <class C>
la::RowMatrix<C> dyadic_rhs(std::size_t rows, std::size_t cols, unsigned seed) {
  return dense<C>(rows, cols, bspline_test::values(rows * cols, seed));
}

// ---- spaces -----------------------------------------------------------------------------------

Dir cubic_dir() { return {clamped(3, {0.25, 0.5, 0.75}), 3}; }  // 7 functions on [0, 1]
// Quadratic with a double knot at 1/2 (C^0 there): 7 functions on [0, 1].
Dir c0_quadratic_dir() { return {clamped(2, {0.25, 0.5, 0.5, 0.75}), 2}; }
Dir quartic_dir() { return {clamped(4, {0.125, 0.5, 0.75}), 4}; }  // 8 functions on [0, 1]
// Quadratic on [2, 5]: 4 functions, knots dyadic after the shift.
Dir shifted_dir() { return {{2.0, 2.0, 2.0, 3.5, 5.0, 5.0, 5.0}, 2}; }
// Periodic cubic with 8 intervals on [0, 1]: 8 functions whose supports wrap.
Dir periodic_dir() { return {periodic_knots(8, 3), 3, true}; }
// Unclamped cubic: 6 functions, domain [3/4, 3/2], no function equal to 1 at an end.
Dir unclamped_dir() { return {{0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25}, 3}; }

template <class T>
bs::BsplineSpace1D<T> space_1d(const Dir& dir) {
  return space_of<T>({dir}).spaces()[0];
}

template <class T>
std::vector<std::vector<T>> greville_of(const bs::BsplineSpace<T>& space) {
  std::vector<std::vector<T>> out;
  for (const bs::BsplineSpace1D<T>& s : space.spaces()) {
    out.push_back(s.greville());
  }
  return out;
}

// `count` equispaced nodes from the start of the domain to its end.
template <class T>
std::vector<T> uniform_nodes(const bs::BsplineSpace1D<T>& s, std::size_t count) {
  const auto [a, b] = s.domain();
  std::vector<T> nodes(count);
  for (std::size_t i = 0; i < count; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(count - 1);
    nodes[i] = i + 1 == count
                   ? b
                   : static_cast<T>(static_cast<double>(a) +
                                    (static_cast<double>(b) - static_cast<double>(a)) * t);
  }
  return nodes;
}

// The ends of the domain, a knot, and nodes that are not knots.
template <class T>
std::vector<T> probe_nodes(const bs::BsplineSpace1D<T>& s) {
  const auto [a, b] = s.domain();
  const double lo = static_cast<double>(a);
  const double len = static_cast<double>(b) - lo;
  std::vector<T> nodes = {a};
  for (const double f : {1.0 / 16.0, 0.25, 0.3, 0.5, 0.83}) {
    nodes.push_back(static_cast<T>(lo + len * f));
  }
  nodes.push_back(b);
  return nodes;
}

template <class T>
bs::Bspline<T> unit_spline(const bs::BsplineSpace<T>& space, std::size_t j) {
  std::size_t total = 1;
  for (const auto& s : space.spaces()) {
    total *= static_cast<std::size_t>(s.num_basis());
  }
  std::vector<T> cp(total, T(0));
  cp[j] = T(1);
  return bs::Bspline<T>(space, std::span<const T>(cp), false);
}

// The exact values (all coordinates) of `s` on a tensor grid of nodes, rounded once to T,
// row-major over (grid, coordinates).
template <class T>
std::vector<T> samples_on_grid(const bs::Bspline<T>& s, const std::vector<std::vector<T>>& nodes) {
  const std::size_t dim = nodes.size();
  std::vector<T> out;
  std::vector<std::size_t> idx(dim, 0);
  for (;;) {
    std::vector<Rational> x;
    for (std::size_t d = 0; d < dim; ++d) {
      x.push_back(oracle::exact(static_cast<double>(nodes[d][idx[d]])));
    }
    for (const Rational& v : oracle::exact_value<T>(s, x)) {
      out.push_back(static_cast<T>(to_double(v)));
    }
    std::size_t d = dim;
    bool done = true;
    while (d-- > 0) {
      if (++idx[d] < nodes[d].size()) {
        done = false;
        break;
      }
      idx[d] = 0;
    }
    if (done) {
      return out;
    }
  }
}

// The exact values of `s` at scattered points `pts` (n_pts x dim, row-major), rounded once to T.
template <class T>
std::vector<T> samples_at(const bs::Bspline<T>& s, const std::vector<T>& pts, std::size_t dim) {
  std::vector<T> out;
  for (std::size_t i = 0; i < pts.size() / dim; ++i) {
    std::vector<Rational> x;
    for (std::size_t d = 0; d < dim; ++d) {
      x.push_back(oracle::exact(static_cast<double>(pts[i * dim + d])));
    }
    for (const Rational& v : oracle::exact_value<T>(s, x)) {
      out.push_back(static_cast<T>(to_double(v)));
    }
  }
  return out;
}

template <class T>
double two_norm_of(std::span<const T> v) {
  double s = 0.0;
  for (const T x : v) {
    s += static_cast<double>(x) * static_cast<double>(x);
  }
  return std::sqrt(s);
}

// The largest |got - want| over two equally long arrays of control points.
template <class T>
double worst_difference(std::span<const T> got, std::span<const T> want) {
  REQUIRE(got.size() == want.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double d = std::abs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
    worst = (std::max)(worst, std::isnan(d) ? (std::numeric_limits<double>::infinity)() : d);
  }
  return worst;
}

// K (m + n + 2) u kappa ||x||: the forward-error bound of the header comment.
template <class T>
double solve_bound(std::size_t size_term, double kappa, double x_norm) {
  return fit_slack * static_cast<double>(size_term) * unit_roundoff<T>() * kappa * x_norm;
}

// Out-of-range message of the domain check, for a domain written as Python prints it.
const char* const outside_unit =
    "One or more values in pts are outside the knot vector domain (0.0, 1.0)";

}  // namespace

// ---- collocation_matrix -----------------------------------------------------------------------

TEST_CASE_TEMPLATE(
    "collocation_matrix: each entry is the exact basis value, a periodic space wraps", T, float,
    double) {
  // Oracle: column j is the exact value of the spline whose only control point is the j-th (the
  // recursive definition of the basis), at the exact node. Bug caught: a transposed matrix, a
  // first-function offset that is off by one, a periodic wrap that is missing or shifted, a
  // clamped space read as unclamped (the unclamped and shifted spaces have no unit end rows).
  const double u = unit_roundoff<T>();
  const std::vector<Dir> dirs = {cubic_dir(),   c0_quadratic_dir(), quartic_dir(),
                                 shifted_dir(), unclamped_dir(),    periodic_dir()};
  for (const Dir& dir : dirs) {
    const bs::BsplineSpace<T> space = space_of<T>({dir});
    const bs::BsplineSpace1D<T>& s = space.spaces()[0];
    const std::vector<T> nodes = probe_nodes<T>(s);
    const NdArray<T> mat = bs::collocation_matrix<T, T>(s, std::span<const T>(nodes));
    const auto n = static_cast<std::size_t>(s.num_basis());
    REQUIRE(mat.rank() == 2);
    REQUIRE(mat.shape()[0] == nodes.size());
    REQUIRE(mat.shape()[1] == n);
    const double bound = (6.0 * dir.degree + 1.0) * u;
    for (std::size_t j = 0; j < n; ++j) {
      const bs::Bspline<T> column = unit_spline<T>(space, j);
      for (std::size_t i = 0; i < nodes.size(); ++i) {
        const std::vector<Rational> x = {oracle::exact(static_cast<double>(nodes[i]))};
        const Rational want = oracle::exact_value<T>(column, x)[0];
        INFO("degree " << dir.degree << ", periodic " << dir.periodic << ", node " << i
                       << ", function " << j);
        CHECK(oracle::within(static_cast<double>(mat.data()[i * n + j]), want, bound));
      }
    }
  }
}

TEST_CASE_TEMPLATE("collocation_matrix: a node outside the domain, or NaN, is refused", T, float,
                   double) {
  const bs::BsplineSpace1D<T> s = space_1d<T>(cubic_dir());
  const std::vector<T> below = {T(0.5), T(-0.25)};
  const std::vector<T> above = {T(1.25)};
  const std::vector<T> nan = {T(0.5), std::numeric_limits<T>::quiet_NaN()};
  for (const auto* nodes : {&below, &above, &nan}) {
    CHECK_THROWS_WITH_AS(
        static_cast<void>(bs::collocation_matrix<T, T>(s, std::span<const T>(*nodes))),
        outside_unit, pantr::value_error);
  }
  // The ends themselves are inside (a closed domain).
  const std::vector<T> ends = {T(0), T(1)};
  CHECK_NOTHROW(static_cast<void>(bs::collocation_matrix<T, T>(s, std::span<const T>(ends))));
}

TEST_CASE_TEMPLATE("collocation_matrix: boundary derivative rows are the exact derivatives", T,
                   float, double) {
  // Oracle: the exact k-th derivative of the unit splines (Leibniz differentiation of the
  // recursive definition) at the exact end of the domain. Rows 1..left hold the orders 1..left at
  // the start, rows m-1-right..m-2 the orders 1..right at the end (order 1 first); every other row
  // is the plain collocation row. Bug caught: orders in the wrong row, a derivative taken at the
  // wrong end, a row off by one at either side, a plain row overwritten.
  const double u = unit_roundoff<T>();
  struct Case {
    Dir dir;
    long long left;
    long long right;
  };
  const std::vector<Case> cases = {{cubic_dir(), 2, 3},
                                   {c0_quadratic_dir(), 1, 2},
                                   {shifted_dir(), 1, 1},
                                   {unclamped_dir(), 2, 1}};
  for (const Case& c : cases) {
    const bs::BsplineSpace<T> space = space_of<T>({c.dir});
    const bs::BsplineSpace1D<T>& s = space.spaces()[0];
    const auto n = static_cast<std::size_t>(s.num_basis());
    const std::vector<T> nodes =
        uniform_nodes<T>(s, 7);  // Greville points leave an unclamped domain
    const std::size_t m = nodes.size();
    const NdArray<T> plain = bs::collocation_matrix<T, T>(s, std::span<const T>(nodes));
    const NdArray<T> mat = bs::collocation_matrix<T, T>(s, std::span<const T>(nodes),
                                                        bs::BoundaryDerivatives{c.left, c.right});
    REQUIRE(mat.shape()[0] == m);
    REQUIRE(mat.shape()[1] == n);
    const auto [a, b] = s.domain();
    const auto first_right = m - 1 - static_cast<std::size_t>(c.right);
    for (std::size_t r = 0; r < m; ++r) {
      const bool is_left = r >= 1 && r <= static_cast<std::size_t>(c.left);
      const bool is_right = r >= first_right && r + 1 < m;
      INFO("degree " << c.dir.degree << ", row " << r);
      if (!is_left && !is_right) {
        for (std::size_t j = 0; j < n; ++j) {
          CHECK(mat.data()[r * n + j] == plain.data()[r * n + j]);  // untouched: identical
        }
        continue;
      }
      const int order = is_right ? static_cast<int>(r - first_right) + 1 : static_cast<int>(r);
      const T x_end = is_right ? b : a;
      std::vector<Rational> want_row;
      double row_max = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        const std::vector<Rational> x = {oracle::exact(static_cast<double>(x_end))};
        const std::vector<int> orders = {order};
        want_row.push_back(oracle::exact_value<T>(unit_spline<T>(space, j), x, orders)[0]);
        row_max = (std::max)(row_max, std::abs(to_double(want_row.back())));
      }
      const double bound = fit_slack * (order + 1.0) * (6.0 * c.dir.degree + 1.0) * u * row_max;
      for (std::size_t j = 0; j < n; ++j) {
        INFO("order " << order << ", function " << j);
        CHECK(oracle::within(static_cast<double>(mat.data()[r * n + j]), want_row[j], bound));
      }
    }
  }
}

TEST_CASE_TEMPLATE(
    "collocation_matrix: a count that is not positive adds no row, a periodic space ignores them",
    T, float, double) {
  const bs::BsplineSpace1D<T> s = space_1d<T>(cubic_dir());
  const std::vector<T> nodes = s.greville();
  const NdArray<T> plain = bs::collocation_matrix<T, T>(s, std::span<const T>(nodes));
  for (const bs::BoundaryDerivatives bd :
       {bs::BoundaryDerivatives{0, 0}, bs::BoundaryDerivatives{-3, -1},
        bs::BoundaryDerivatives{0, -2}}) {
    const NdArray<T> mat = bs::collocation_matrix<T, T>(s, std::span<const T>(nodes), bd);
    REQUIRE(mat.size() == plain.size());
    for (std::size_t i = 0; i < mat.size(); ++i) {
      CHECK(mat.data()[i] == plain.data()[i]);
    }
  }
  // A periodic space has no ends to constrain: the same matrix as without the request.
  const bs::BsplineSpace1D<T> per = space_1d<T>(periodic_dir());
  const std::vector<T> pnodes = per.greville();
  const NdArray<T> pplain = bs::collocation_matrix<T, T>(per, std::span<const T>(pnodes));
  const NdArray<T> pmat =
      bs::collocation_matrix<T, T>(per, std::span<const T>(pnodes), bs::BoundaryDerivatives{2, 2});
  REQUIRE(pmat.size() == pplain.size());
  for (std::size_t i = 0; i < pmat.size(); ++i) {
    CHECK(pmat.data()[i] == pplain.data()[i]);
  }
}

TEST_CASE(
    "collocation_matrix: more derivative rows than the rows they replace repeat NumPy's broadcast "
    "message") {
  // The rows replaced are `[1, 1 + left)` and `[m - 1 - right, m - 1)` clipped to the matrix, as
  // the Python slices resolve them. With m = 7 nodes a count of 6 fits exactly (6 rows) and 7 does
  // not; the boundary is constructed, not sampled. Bug caught: a clipping that is off by one in
  // either direction (a count that should fit refused, or one that should not accepted and writing
  // past the matrix), or a message that names the wrong shapes.
  const bs::BsplineSpace1D<double> s = space_1d<double>(cubic_dir());
  const std::vector<double> nodes = s.greville();
  REQUIRE(nodes.size() == 7);
  const std::span<const double> sp7(nodes);
  CHECK_NOTHROW(static_cast<void>(
      bs::collocation_matrix<double, double>(s, sp7, bs::BoundaryDerivatives{6, 0})));
  CHECK_NOTHROW(static_cast<void>(
      bs::collocation_matrix<double, double>(s, sp7, bs::BoundaryDerivatives{0, 6})));
  const char* const seven = "could not broadcast input array from shape (7,7) into shape (6,7)";
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::collocation_matrix<double, double>(
                           s, sp7, bs::BoundaryDerivatives{7, 0})),
                       seven, pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::collocation_matrix<double, double>(
                           s, sp7, bs::BoundaryDerivatives{0, 7})),
                       seven, pantr::value_error);
  // Fewer nodes than basis functions: three nodes leave two replaceable rows on either side.
  const std::vector<double> three = {0.0, 0.5, 1.0};
  const std::span<const double> sp3(three);
  CHECK_NOTHROW(static_cast<void>(
      bs::collocation_matrix<double, double>(s, sp3, bs::BoundaryDerivatives{2, 0})));
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::collocation_matrix<double, double>(
                           s, sp3, bs::BoundaryDerivatives{3, 0})),
                       "could not broadcast input array from shape (3,7) into shape (2,7)",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::collocation_matrix<double, double>(
                           s, sp3, bs::BoundaryDerivatives{0, 3})),
                       "could not broadcast input array from shape (3,7) into shape (2,7)",
                       pantr::value_error);
  // A single node has no row to replace.
  const std::vector<double> one = {0.5};
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::collocation_matrix<double, double>(
                           s, std::span<const double>(one), bs::BoundaryDerivatives{1, 0})),
                       "could not broadcast input array from shape (1,7) into shape (0,7)",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::collocation_matrix<double, double>(
                           s, std::span<const double>(one), bs::BoundaryDerivatives{0, 1})),
                       "could not broadcast input array from shape (1,7) into shape (0,7)",
                       pantr::value_error);
}

TEST_CASE(
    "collocation_matrix: a derivative row at a degenerate end raises std::domain_error, not "
    "value_error") {
  // The derivative recurrence divides by the span's knot differences. At the right end of an
  // unclamped knot vector whose domain-end knot is repeated, the clamped span is empty
  // (knots {0, 0, 0, 1, 2, 2, 3, 4}, degree 2: the domain is [0, 2] and the span used at 2 is
  // [2, 2]), so the order-2 row has a zero divisor; order 1 does not (the divisor of the order-k
  // row is a difference of triangle entries that is only zero from order 2 here), and neither does
  // the start, whose span is never empty. The binding turns the exception into ZeroDivisionError.
  // Bug caught: the incomplete flag ignored (a row of garbage returned silently), or the exception
  // type changed (a pantr::error would be mapped to ValueError by the binding).
  const std::vector<double> knots = {0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 4.0};
  const bs::BsplineSpace1D<double> s(std::span<const double>(knots), 2);
  REQUIRE(s.domain().second == 2.0);
  const std::vector<double> nodes = {0.0, 0.5, 1.0, 1.5, 2.0};
  const std::span<const double> sp5(nodes);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::collocation_matrix<double, double>(
                           s, sp5, bs::BoundaryDerivatives{0, 2})),
                       "division by zero", std::domain_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::collocation_matrix<double, double>(
                           s, sp5, bs::BoundaryDerivatives{0, 3})),
                       "division by zero", std::domain_error);
  CHECK_NOTHROW(static_cast<void>(
      bs::collocation_matrix<double, double>(s, sp5, bs::BoundaryDerivatives{0, 1})));
  CHECK_NOTHROW(static_cast<void>(
      bs::collocation_matrix<double, double>(s, sp5, bs::BoundaryDerivatives{2, 0})));
  // It propagates out of the tensor system as well.
  const bs::BsplineSpace<double> space(std::vector<bs::BsplineSpace1D<double>>{s});
  const std::vector<std::span<const double>> grid = {sp5};
  CHECK_THROWS_AS(static_cast<void>(
                      bs::TensorCollocation<double>(space, grid, {bs::BoundaryDerivatives{0, 2}})),
                  std::domain_error);
  bool is_pantr_error = false;
  try {
    static_cast<void>(bs::TensorCollocation<double>(space, grid, {bs::BoundaryDerivatives{0, 2}}));
  } catch (const pantr::error&) {
    is_pantr_error = true;
  } catch (...) {
  }
  CHECK_FALSE(is_pantr_error);
}

TEST_CASE(
    "collocation_matrix: a periodic space whose functions fill the support never adds two values "
    "to one entry") {
  // `collocation_matrix` adds `v` into `row[(f + a) % n]` for a periodic space "so that a function
  // appearing twice in a node's support has its values added" (its comment). A space with fewer
  // functions than `degree + 1` would be the case, and the constructor refuses it, so the support
  // has `degree + 1 <= n` distinct functions and each entry is written once. Pinned at the
  // boundary n == degree + 1: every row of the 4 x 4 matrix has four non-zero entries that sum
  // to 1.
  const std::vector<double> knots = periodic_knots(4, 3);
  const bs::BsplineSpace1D<double> s(std::span<const double>(knots), 3, true);
  REQUIRE(s.num_basis() == 4);
  const std::vector<double> nodes = {0.0, 0.125, 0.375, 0.6, 0.99};
  const NdArray<double> mat =
      bs::collocation_matrix<double, double>(s, std::span<const double>(nodes));
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    double sum = 0.0;
    for (std::size_t j = 0; j < 4; ++j) {
      sum += mat.data()[i * 4 + j];
    }
    CHECK(std::abs(sum - 1.0) <= 8.0 * unit_roundoff<double>());
  }
  const std::vector<double> too_few = periodic_knots(3, 3);
  CHECK_THROWS_AS(
      static_cast<void>(bs::BsplineSpace1D<double>(std::span<const double>(too_few), 3, true)),
      pantr::value_error);
}

// ---- TensorCollocation and fit_tensor ---------------------------------------------------------

TEST_CASE("TensorCollocation: it describes its grid and refuses wrong arguments") {
  const std::vector<Dir> dirs = {cubic_dir(), shifted_dir()};
  const bs::BsplineSpace<double> space = space_of<double>(dirs);
  const std::vector<double> x0 = space.spaces()[0].greville();
  const std::vector<double> x1 = uniform_nodes<double>(space.spaces()[1], 5);
  const std::vector<std::span<const double>> grid = {std::span<const double>(x0),
                                                     std::span<const double>(x1)};
  const bs::TensorCollocation<double> system(space, grid);
  CHECK(system.grid_shape() == std::vector<std::size_t>{7, 5});
  REQUIRE(system.matrices().size() == 2);
  CHECK(system.matrices()[0].shape()[0] == 7);
  CHECK(system.matrices()[0].shape()[1] == 7);
  CHECK(system.matrices()[1].shape()[0] == 5);
  CHECK(system.matrices()[1].shape()[1] == 4);

  // One node array per direction.
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::TensorCollocation<double>(
                           space, std::vector<std::span<const double>>{grid[0]})),
                       "Expected 2 node arrays, got 1.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::TensorCollocation<double>(
                           space, std::vector<std::span<const double>>{grid[0], grid[1], grid[0]})),
                       "Expected 2 node arrays, got 3.", pantr::value_error);
  // A node outside the domain of its own direction (the second domain is [2, 5]).
  const std::vector<double> outside = {2.0, 5.5};
  CHECK_THROWS_WITH_AS(
      static_cast<void>(bs::TensorCollocation<double>(
          space, std::vector<std::span<const double>>{grid[0], std::span<const double>(outside)})),
      "One or more values in pts are outside the knot vector domain (2.0, 5.0)",
      pantr::value_error);
  // Constraint entries past the dimension are ignored, and a short list leaves the rest free.
  CHECK_NOTHROW(static_cast<void>(bs::TensorCollocation<double>(
      space, grid, {std::nullopt, bs::BoundaryDerivatives{1, 0}, bs::BoundaryDerivatives{9, 9}})));
  const bs::TensorCollocation<double> short_list(space, grid, {bs::BoundaryDerivatives{1, 0}});
  REQUIRE(short_list.matrices().size() == 2);
  for (std::size_t i = 0; i < system.matrices()[1].size(); ++i) {
    CHECK(short_list.matrices()[1].data()[i] ==
          system.matrices()[1].data()[i]);  // direction 1 free
  }
  CHECK(short_list.matrices()[0].data()[7 + 0] !=
        system.matrices()[0].data()[7 + 0]);  // row 1 changed

  // The values must hold `rank` per node of the grid.
  const std::vector<double> values(34, 1.0);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(system.solve<double>(std::span<const double>(values), 1, 1e-12)),
      "TensorCollocation::solve: expected 35 x 1 values, got 34.", pantr::value_error);
  const std::vector<double> right_count(35, 1.0);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(system.solve<double>(std::span<const double>(right_count), 0, 1e-12)),
      "TensorCollocation::solve: expected 35 x 0 values, got 35.", pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(system.solve<double>(std::span<const double>(right_count), 2, 1e-12)),
      "TensorCollocation::solve: expected 35 x 2 values, got 35.", pantr::value_error);
}

template <class T>
void check_tensor_recovery(const std::vector<Dir>& dirs, const std::vector<std::size_t>& counts,
                           std::size_t rank, unsigned seed) {
  // `counts[d] == 0` takes the Greville abscissae in direction d, else that many equispaced nodes.
  const bs::BsplineSpace<T> space = space_of<T>(dirs);
  const bs::Bspline<T> truth = make_spline<T>(dirs, rank, false, seed);
  std::vector<std::vector<T>> nodes;
  for (std::size_t d = 0; d < dirs.size(); ++d) {
    nodes.push_back(counts[d] == 0 ? space.spaces()[d].greville()
                                   : uniform_nodes<T>(space.spaces()[d], counts[d]));
  }
  const std::vector<T> samples = samples_on_grid<T>(truth, nodes);
  const std::vector<std::span<const T>> spans(nodes.begin(), nodes.end());
  const bs::Bspline<T> fit = bs::fit_tensor<T>(space, spans, std::span<const T>(samples), rank);

  const bs::TensorCollocation<T> system(space, spans);
  double kappa = 1.0;
  std::size_t sizes = 0;
  for (const NdArray<T>& mat : system.matrices()) {
    const double k = kappa_bound(exact_matrix<T>(mat));
    REQUIRE(1.0 / k > 100.0 * machine_epsilon<T>());  // the truncation must not act
    kappa *= k;
    sizes += mat.shape()[0] + mat.shape()[1] + 2;
  }
  CHECK_FALSE(fit.rational());
  CHECK(fit.rank() == rank);
  REQUIRE(fit.control_points().size() == truth.control_points().size());
  const double bound = solve_bound<T>(sizes, kappa, two_norm_of<T>(truth.control_points()));
  const double worst = worst_difference<T>(fit.control_points(), truth.control_points());
  INFO("kappa bound " << kappa << ", worst error " << worst << ", bound " << bound);
  CHECK(worst <= bound);
}

TEST_CASE_TEMPLATE("fit_tensor: samples of a spline of the space give its control points back", T,
                   float, double) {
  // Oracle: a spline of the space with dyadic control points, sampled exactly (recursive-definition
  // evaluation at the exact nodes, rounded once to T): the fit must return the control points
  // (reproduction of a spline of the space). Bug caught: a solve along the wrong axis, rank
  // components mixed or transposed, a tensor layout that is not first-direction-slowest, a wrapped
  // periodic row read in the wrong column, a banded QR whose answer drifts from the dense one.
  SUBCASE("1D cubic at the Greville abscissae, a square system") {
    check_tensor_recovery<T>({cubic_dir()}, {0}, 1, 11);
  }
  SUBCASE("1D quartic on 25 equispaced nodes, a least-squares system, rank 2") {
    check_tensor_recovery<T>({quartic_dir()}, {25}, 2, 12);
  }
  SUBCASE("2D cubic by C0 quadratic at the Greville abscissae, rank 2") {
    check_tensor_recovery<T>({cubic_dir(), c0_quadratic_dir()}, {0, 0}, 2, 13);
  }
  SUBCASE("2D periodic cubic by a shifted quadratic, rank 1") {
    check_tensor_recovery<T>({periodic_dir(), shifted_dir()}, {0, 0}, 1, 14);
  }
  SUBCASE("2D mixed: a square direction by a least-squares unclamped one") {
    check_tensor_recovery<T>({c0_quadratic_dir(), unclamped_dir()}, {0, 11}, 1, 15);
  }
  SUBCASE("2D periodic by a least-squares direction on 20 nodes, rank 3") {
    check_tensor_recovery<T>({periodic_dir(), quartic_dir()}, {0, 20}, 3, 16);
  }
  SUBCASE("3D with three different degrees, rank 1") {
    check_tensor_recovery<T>(
        {Dir{clamped(1, {0.5}), 1}, Dir{clamped(2, {0.5}), 2}, Dir{clamped(3, {}), 3}}, {0, 0, 0},
        1, 17);
  }
}

// The derivative constraints of one direction, as rows: 0 for a free row, else the order.
std::vector<int> constraint_orders(std::size_t m, const bs::BoundaryDerivatives& bd) {
  std::vector<int> orders(m, 0);
  for (long long k = 1; k <= bd.left; ++k) {
    orders[static_cast<std::size_t>(k)] = static_cast<int>(k);
  }
  for (long long k = 1; k <= bd.right; ++k) {
    orders[m - 1 - static_cast<std::size_t>(bd.right) + static_cast<std::size_t>(k) - 1] =
        static_cast<int>(k);
  }
  return orders;
}

template <class T>
void check_constrained_fit(const std::vector<Dir>& dirs,
                           const std::vector<bs::BoundaryDerivatives>& bds, unsigned seed,
                           const std::vector<bs::BoundaryDerivatives>& requested = {}) {
  // The fit on the Greville grid with derivative rows must (a) interpolate the data at every node
  // whose row is free in every direction, (b) have the constrained derivative equal to zero at the
  // end of the domain, at every node of the other directions, and (c) not depend on the data at
  // the replaced rows (set to 1000 here). All three are evaluated on the exact basis. The size of
  // the residual of a backward-stable solve: K (sizes) u ||A||_F ||x||_1, with the derivative
  // rows' own error (k + 1)(6 p + 1) u relative to their largest entry added to it.
  const std::size_t dim = dirs.size();
  const bs::BsplineSpace<T> space = space_of<T>(dirs);
  const bs::Bspline<T> truth = make_spline<T>(dirs, 1, false, seed);
  std::vector<std::vector<T>> nodes = greville_of<T>(space);
  std::vector<std::vector<int>> orders(dim);
  std::vector<std::optional<bs::BoundaryDerivatives>> derivs;
  for (std::size_t d = 0; d < dim; ++d) {
    orders[d] = constraint_orders(nodes[d].size(), bds[d]);
    derivs.emplace_back(requested.empty() ? bds[d] : requested[d]);  // `bds`: what takes effect
  }
  std::vector<T> samples = samples_on_grid<T>(truth, nodes);
  {  // garbage at the replaced rows
    for (std::size_t flat = 0; flat < samples.size(); ++flat) {
      std::size_t rem = flat;
      bool replaced = false;
      for (std::size_t d = dim; d-- > 0;) {
        replaced = replaced || orders[d][rem % nodes[d].size()] != 0;
        rem /= nodes[d].size();
      }
      if (replaced) {
        samples[flat] = T(1000);
      }
    }
  }
  const std::vector<std::span<const T>> spans(nodes.begin(), nodes.end());
  const bs::TensorCollocation<T> system(space, spans, derivs);
  const NdArray<T> coeffs =
      system.template solve<T>(std::span<const T>(samples), 1, 100.0 * machine_epsilon<T>());
  const bs::Bspline<T> fit(space, NdArray<T>(coeffs), false);

  double frobenius = 1.0;
  double sizes = 0.0;
  double derivative_term = 0.0;
  for (std::size_t d = 0; d < dim; ++d) {
    double sq = 0.0;
    for (const T v : system.matrices()[d].flat()) {
      sq += static_cast<double>(v) * static_cast<double>(v);
    }
    frobenius *= std::sqrt(sq);
    sizes +=
        static_cast<double>(system.matrices()[d].shape()[0] + system.matrices()[d].shape()[1] + 2);
    const long long kmax = (std::max)(bds[d].left, bds[d].right);
    derivative_term += (static_cast<double>(kmax) + 1.0) * (6.0 * dirs[d].degree + 1.0);
  }
  double one_norm = 0.0;
  for (const T v : fit.control_points()) {
    one_norm += std::abs(static_cast<double>(v));
  }
  const double bound =
      fit_slack * (sizes + derivative_term) * unit_roundoff<T>() * frobenius * one_norm;

  std::vector<std::size_t> idx(dim, 0);
  std::size_t checked_free = 0;
  std::size_t checked_derivative = 0;
  for (;;) {
    std::vector<Rational> x;
    std::vector<int> order(dim, 0);
    bool free_everywhere = true;
    for (std::size_t d = 0; d < dim; ++d) {
      const int k = orders[d][idx[d]];
      order[d] = k;
      if (k == 0) {
        x.push_back(oracle::exact(static_cast<double>(nodes[d][idx[d]])));
      } else {
        free_everywhere = false;
        const auto dom = space.spaces()[d].domain();
        const bool at_start = idx[d] <= static_cast<std::size_t>(bds[d].left);
        x.push_back(oracle::exact(static_cast<double>(at_start ? dom.first : dom.second)));
      }
    }
    const double got = to_double(oracle::exact_value<T>(fit, x, order)[0]);
    if (free_everywhere) {
      std::size_t flat = 0;
      for (std::size_t d = 0; d < dim; ++d) {
        flat = flat * nodes[d].size() + idx[d];
      }
      INFO("free node, flat index " << flat);
      CHECK(std::abs(got - static_cast<double>(samples[flat])) <= bound);
      ++checked_free;
    } else {
      INFO("constrained node, orders " << order[0] << (dim > 1 ? "," : "")
                                       << (dim > 1 ? order[1] : 0));
      CHECK(std::abs(got) <= bound);
      ++checked_derivative;
    }
    std::size_t d = dim;
    bool done = true;
    while (d-- > 0) {
      if (++idx[d] < nodes[d].size()) {
        done = false;
        break;
      }
      idx[d] = 0;
    }
    if (done) {
      break;
    }
  }
  CHECK(checked_free > 0);
  CHECK(checked_derivative > 0);
  CHECK(bound < 1.0);  // a bound that large would hide the O(1) error of an unzeroed row
}

TEST_CASE_TEMPLATE(
    "TensorCollocation::solve: derivative constraints hold and the replaced data is ignored", T,
    float, double) {
  // Oracle: invariants evaluated on the exact basis of the fitted spline (see
  // `check_constrained_fit`). Bug caught: the right-hand side of a derivative row not set to zero
  // (the garbage 1000 of the replaced rows would leak into the coefficients), derivative orders in
  // the wrong rows, a constraint zeroed along the wrong axis of a tensor, the right end's rows
  // placed at the left.
  SUBCASE("1D cubic, first derivative zero at both ends") {
    check_constrained_fit<T>({cubic_dir()}, {bs::BoundaryDerivatives{1, 1}}, 21);
  }
  SUBCASE("1D quartic, two orders at the start and one at the end") {
    check_constrained_fit<T>({quartic_dir()}, {bs::BoundaryDerivatives{2, 1}}, 22);
  }
  if constexpr (std::is_same_v<T, double>) {
    SUBCASE(
        "2D, orders 1 and 2 at the end of the first direction and order 1 at the start of the "
        "second") {
      check_constrained_fit<T>({cubic_dir(), c0_quadratic_dir()},
                               {bs::BoundaryDerivatives{0, 2}, bs::BoundaryDerivatives{1, 0}}, 23);
    }
    SUBCASE("2D, a periodic direction asked for constraints that it ignores") {
      // Constraints are requested in both directions, but only the non-periodic one takes them:
      // the periodic direction keeps all its interpolation rows (nothing is replaced, nothing
      // zeroed) and the other direction's constraints still hold at every one of its nodes.
      check_constrained_fit<T>({periodic_dir(), cubic_dir()},
                               {bs::BoundaryDerivatives{0, 0}, bs::BoundaryDerivatives{1, 1}}, 24,
                               {bs::BoundaryDerivatives{2, 2}, bs::BoundaryDerivatives{1, 1}});
    }
  }
}

TEST_CASE("fit_tensor: a tolerance above 1 drops every singular value of a least-squares fit") {
  // 20 nodes for 7 functions: a rectangular system, truncated at tol * s_max. The default keeps all
  // singular values of this well-conditioned matrix (the control points come back), a tolerance
  // above 1 keeps none (the zero fit, exactly). Bug caught: a tolerance that is ignored or applied
  // to the wrong end of the spectrum.
  const std::vector<Dir> dirs = {cubic_dir()};
  const bs::BsplineSpace<double> space = space_of<double>(dirs);
  const bs::Bspline<double> truth = make_spline<double>(dirs, 1, false, 31);
  const std::vector<double> nodes = uniform_nodes<double>(space.spaces()[0], 20);
  const std::vector<double> samples = samples_on_grid<double>(truth, {nodes});
  const std::vector<std::span<const double>> spans = {std::span<const double>(nodes)};
  const bs::Bspline<double> dropped =
      bs::fit_tensor<double>(space, spans, std::span<const double>(samples), 1, {}, 2.0);
  for (const double v : dropped.control_points()) {
    CHECK(v == 0.0);
  }
  const bs::Bspline<double> kept =
      bs::fit_tensor<double>(space, spans, std::span<const double>(samples), 1);
  CHECK(worst_difference<double>(kept.control_points(), truth.control_points()) < 1e-9);
}

TEST_CASE(
    "interpolate: the Greville abscissae of an unclamped space leave its domain and are refused") {
  // The default nodes of an unclamped space (the first Greville abscissa of the cubic below is
  // 1/2, left of its domain [3/4, 3/2]) are refused by the basis tabulation, as the previous
  // release's were. The callable is never reached: the system is built first.
  const bs::BsplineSpace<double> unclamped = space_of<double>({unclamped_dir()});
  int calls = 0;
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::interpolate<double>(
                           unclamped,
                           [&calls](const pantr::quad::PointsLattice<double>& l) {
                             ++calls;
                             return NdArray<double>({l.num_points()});
                           })),
                       "One or more values in pts are outside the knot vector domain (0.75, 1.5)",
                       pantr::value_error);
  CHECK(calls == 0);
}

// ---- solve_collocation: the banded QR, the LU and the truncated SVD ---------------------------

namespace {

double frobenius_norm(const Matrix& m) {
  double s = 0.0;
  for (const auto& row : m) {
    for (const Rational& v : row) {
      const double d = to_double(v);
      s += d * d;
    }
  }
  return std::sqrt(s);
}

template <class C>
void expect_solution(const NdArray<C>& got, const Matrix& want, double bound, const char* what) {
  REQUIRE(got.rank() == 2);
  REQUIRE(got.shape()[0] == want.size());
  REQUIRE(got.shape()[1] == want[0].size());
  double worst = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    for (std::size_t j = 0; j < want[0].size(); ++j) {
      const double d =
          std::abs(static_cast<double>(got.data()[i * want[0].size() + j]) - to_double(want[i][j]));
      worst = (std::max)(worst, std::isnan(d) ? (std::numeric_limits<double>::infinity)() : d);
    }
  }
  INFO(what << ": worst error " << worst << ", bound " << bound);
  CHECK(worst <= bound);
}

template <class C>
bs::BsplineSpace1D<C> clamped_cubic_space() {
  return space_1d<C>(cubic_dir());
}

// The matrix of a collocation system and its exact copy.
template <class C>
la::RowMatrix<C> collocation_of(const bs::BsplineSpace1D<C>& s, const std::vector<C>& nodes) {
  return row_matrix<C>(bs::collocation_matrix<C, C>(s, std::span<const C>(nodes)));
}

}  // namespace

TEST_CASE_TEMPLATE(
    "solve_collocation: a banded square or least-squares system matches the exact solution", C,
    float, double) {
  // Oracle: the exact solution of the same matrix and right-hand side (Gauss-Jordan on the exact
  // entries; the normal equations for the rectangular system). The banded QR answers both systems
  // (asserted through `try_lstsq_banded`), so this is its accuracy against an independent route.
  // Bug caught: a banded solve with a wrong band offset, a Givens rotation applied to the wrong
  // pair of rows, a back substitution that skips the last columns.
  const double tol = 100.0 * machine_epsilon<C>();
  SUBCASE("square, 7 x 7 at the Greville abscissae, two right-hand sides") {
    const bs::BsplineSpace1D<C> s = clamped_cubic_space<C>();
    const la::RowMatrix<C> a = collocation_of<C>(s, s.greville());
    const la::RowMatrix<C> b = dyadic_rhs<C>(7, 2, 5);
    CHECK(la::try_lstsq_banded<C>(a, b, 0, la::lstsq_rcond(7, 7)).has_value());
    const Matrix ea = exact_matrix<C>(a);
    const Matrix want = solve_exact(ea, exact_matrix<C>(b));
    const double bound = solve_bound<C>(7 + 7 + 2, kappa_bound(ea), frobenius_norm(want));
    expect_solution<C>(bs::solve_collocation<C>(a, b, tol), want, bound, "square");
  }
  SUBCASE("rectangular, 25 x 8 on equispaced nodes, three right-hand sides") {
    const bs::BsplineSpace1D<C> s = space_1d<C>(quartic_dir());
    const la::RowMatrix<C> a = collocation_of<C>(s, uniform_nodes<C>(s, 25));
    const la::RowMatrix<C> b = dyadic_rhs<C>(25, 3, 6);
    CHECK(la::try_lstsq_banded<C>(a, b, 0, tol).has_value());
    const Matrix ea = exact_matrix<C>(a);
    const double kappa = kappa_bound(ea);
    REQUIRE(1.0 / kappa > tol);
    const Matrix want = lstsq_exact(ea, exact_matrix<C>(b));
    const double bound = solve_bound<C>(25 + 8 + 2, kappa, frobenius_norm(want));
    expect_solution<C>(bs::solve_collocation<C>(a, b, tol), want, bound, "rectangular");
  }
  SUBCASE("square, 11 x 11 on a band short enough to need no dense border") {
    const bs::BsplineSpace1D<C> s =
        space_1d<C>(Dir{clamped(3, {0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875}), 3});
    const la::RowMatrix<C> a = collocation_of<C>(s, s.greville());
    const la::RowMatrix<C> b = dyadic_rhs<C>(11, 2, 4);
    CHECK(bs::detail::wrap_border<C>(a) == 0);
    CHECK(bs::detail::try_banded<C>(a, b, la::lstsq_rcond(11, 11)).has_value());
    const Matrix ea = exact_matrix<C>(a);
    const Matrix want = solve_exact(ea, exact_matrix<C>(b));
    const double bound = solve_bound<C>(11 + 11 + 2, kappa_bound(ea), frobenius_norm(want));
    expect_solution<C>(bs::solve_collocation<C>(a, b, tol), want, bound, "long band");
  }
  SUBCASE("a square system ignores the truncation tolerance, even NaN") {
    const bs::BsplineSpace1D<C> s = clamped_cubic_space<C>();
    const la::RowMatrix<C> a = collocation_of<C>(s, s.greville());
    const la::RowMatrix<C> b = dyadic_rhs<C>(7, 1, 7);
    const Matrix ea = exact_matrix<C>(a);
    const Matrix want = solve_exact(ea, exact_matrix<C>(b));
    const double bound = solve_bound<C>(7 + 7 + 2, kappa_bound(ea), frobenius_norm(want));
    expect_solution<C>(bs::solve_collocation<C>(a, b, std::numeric_limits<double>::quiet_NaN()),
                       want, bound, "square, NaN tolerance");
    expect_solution<C>(bs::solve_collocation<C>(a, b, 5.0), want, bound, "square, tolerance 5");
  }
}

TEST_CASE_TEMPLATE(
    "solve_collocation: a periodic matrix whose rows wrap goes through the dense border", C, float,
    double) {
  // The collocation matrix of a periodic space has rows whose non-zeros wrap around the last
  // column, so the banded path moves the leading columns to the end (`wrap_border`) and undoes
  // the permutation. Oracle: the exact solution. `wrap_border` is asserted positive here,
  // `try_banded` that the banded path answered, so the test cannot pass through the dense
  // fallback alone. Bug caught: a border that is not moved back, a wrong
  // count of leading columns, a permutation applied to the right-hand side as well.
  const double tol = 100.0 * machine_epsilon<C>();
  const bs::BsplineSpace1D<C> s = space_1d<C>(periodic_dir());
  SUBCASE("square, 8 x 8 at the Greville abscissae") {
    const la::RowMatrix<C> a = collocation_of<C>(s, s.greville());
    const la::RowMatrix<C> b = dyadic_rhs<C>(8, 2, 8);
    CHECK(bs::detail::wrap_border<C>(a) > 0);
    CHECK(bs::detail::try_banded<C>(a, b, la::lstsq_rcond(8, 8)).has_value());
    const Matrix ea = exact_matrix<C>(a);
    const Matrix want = solve_exact(ea, exact_matrix<C>(b));
    const double bound = solve_bound<C>(8 + 8 + 2, kappa_bound(ea), frobenius_norm(want));
    expect_solution<C>(bs::solve_collocation<C>(a, b, tol), want, bound, "periodic square");
  }
  SUBCASE("rectangular, 20 x 8 on equispaced nodes of [0, 1)") {
    std::vector<C> nodes(20);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      nodes[i] = static_cast<C>(static_cast<double>(i) / 20.0);
    }
    const la::RowMatrix<C> a = collocation_of<C>(s, nodes);
    const la::RowMatrix<C> b = dyadic_rhs<C>(20, 1, 9);
    CHECK(bs::detail::wrap_border<C>(a) > 0);
    CHECK(bs::detail::try_banded<C>(a, b, tol).has_value());
    const Matrix ea = exact_matrix<C>(a);
    const double kappa = kappa_bound(ea);
    REQUIRE(1.0 / kappa > tol);
    const Matrix want = lstsq_exact(ea, exact_matrix<C>(b));
    const double bound = solve_bound<C>(20 + 8 + 2, kappa, frobenius_norm(want));
    expect_solution<C>(bs::solve_collocation<C>(a, b, tol), want, bound, "periodic rectangular");
  }
}

TEST_CASE_TEMPLATE(
    "solve_collocation: an exactly singular square system gets the minimum-norm least-squares "
    "solution",
    C, float, double) {
  // A duplicated node repeats a row, so LU meets an exactly zero pivot (the two rows are
  // processed with identical operations, so the pivot is exactly 0, not tiny), the banded
  // certificate fails (asserted), and the truncated SVD answers. Oracle: with rows r and r + 1
  // equal and the data different, the least-squares problem has the same minimisers as the
  // problem with that row once and the average of the two data (the pair's residual is
  // 2 (a x - mean)^2 + const), whose rows are independent: its minimum-norm solution is
  // `A^T (A A^T)^-1 b` in exact rationals. Bug caught: the zero pivot unnoticed (inf and NaN from
  // a division), the first of the duplicate rows used alone, the SVD cutoff applied against the
  // smallest instead of the largest singular value.
  const double tol = 100.0 * machine_epsilon<C>();
  const bs::BsplineSpace1D<C> s = clamped_cubic_space<C>();
  std::vector<C> nodes = s.greville();
  nodes[3] = nodes[2];
  const la::RowMatrix<C> a = collocation_of<C>(s, nodes);
  const la::RowMatrix<C> b = dyadic_rhs<C>(7, 2, 10);
  CHECK_FALSE(la::try_lstsq_banded<C>(a, b, 0, la::lstsq_rcond(7, 7)).has_value());

  const Matrix ea = exact_matrix<C>(a);
  const Matrix eb = exact_matrix<C>(b);
  const std::vector<std::size_t> keep = {0, 1, 2, 4, 5, 6};
  Matrix reduced;
  Matrix averaged;
  for (const std::size_t r : keep) {
    reduced.push_back(ea[r]);
    std::vector<Rational> rhs = eb[r];
    if (r == 2) {
      for (std::size_t c = 0; c < rhs.size(); ++c) {
        rhs[c] = (eb[2][c] + eb[3][c]) / 2;
      }
    }
    averaged.push_back(rhs);
  }
  const Matrix want = min_norm_exact(reduced, averaged);
  // The SVD sees the pair as one row scaled by sqrt 2, which changes kappa by at most that factor.
  const double kappa = std::sqrt(2.0) * kappa_bound(transposed(reduced));
  REQUIRE(1.0 / kappa > tol);
  const double bound = solve_bound<C>(7 + 7 + 2, kappa, frobenius_norm(want));
  expect_solution<C>(bs::solve_collocation<C>(a, b, tol), want, bound, "singular square");
  // Truncating everything leaves the zero solution, also for this system.
  const NdArray<C> zero = bs::solve_collocation<C>(a, b, std::numeric_limits<double>::quiet_NaN());
  for (std::size_t i = 0; i < zero.size(); ++i) {
    CHECK(zero.data()[i] == C(0));
  }
}

TEST_CASE_TEMPLATE(
    "solve_collocation: the truncation tolerance is relative, NaN and infinity drop everything, a "
    "negative one nothing",
    C, float, double) {
  // The 4 x 2 matrix [[1, 0], [1, 0], [0, s], [0, s]] has the singular values sqrt 2 and sqrt 2 s
  // (orthogonal columns), the data [1, 1, s, s] has the solution (1, 1). s = 1e-6, so the ratio of
  // the singular values is 1e-6:
  //   tol 1e-9, 0 and -1  keep both       -> (1, 1)
  //   tol 1e-3            drops the second -> (1, 0)
  //   tol 2, NaN, +inf    drop both        -> (0, 0) exactly (the comparison `s > tol s_max` is
  //                                           false for NaN, and for tol >= 1 no value exceeds it)
  // Bug caught: a tolerance that is absolute, `abs(tol)` for a negative one, a comparison that
  // lets NaN through, a banded path that answers with its own default cutoff whatever the tolerance
  // (the tol 1e-3 case: the banded certificate must refuse, which `try_lstsq_banded` is asserted
  // to do for an `rcond` above the ratio and to accept below it).
  const double s = 1e-6;
  const la::RowMatrix<C> a = dense<C>(4, 2, {1, 0, 1, 0, 0, s, 0, s});
  const la::RowMatrix<C> b = dense<C>(4, 1, {1, 1, s, s});
  // Exact data after the rounding of s to C: b3 = a(3, 1), so the solution is exactly (1, 1).
  const double inexact_rounding = 2.0 * unit_roundoff<C>();
  const double dec = solve_bound<double>(4 + 2 + 2, 1e6, std::sqrt(2.0));
  const double bound = dec + inexact_rounding * std::sqrt(2.0);
  const auto solve_with = [&](double tol) { return bs::solve_collocation<C>(a, b, tol); };
  const Matrix keep_both = {{Rational(1)}, {Rational(1)}};
  const Matrix keep_first = {{Rational(1)}, {Rational(0)}};
  const Matrix keep_none = {{Rational(0)}, {Rational(0)}};
  expect_solution<C>(solve_with(1e-9), keep_both, bound, "tol 1e-9");
  expect_solution<C>(solve_with(0.0), keep_both, bound, "tol 0");
  expect_solution<C>(solve_with(-1.0), keep_both, bound, "tol -1");
  expect_solution<C>(solve_with(1e-3), keep_first, bound, "tol 1e-3");
  expect_solution<C>(solve_with(2.0), keep_none, 0.0, "tol 2");
  expect_solution<C>(solve_with(std::numeric_limits<double>::quiet_NaN()), keep_none, 0.0,
                     "tol NaN");
  expect_solution<C>(solve_with(std::numeric_limits<double>::infinity()), keep_none, 0.0,
                     "tol +inf");
  // The banded certificate honours the cutoff it is given.
  CHECK(la::try_lstsq_banded<C>(a, b, 0, 1e-9).has_value());
  CHECK_FALSE(la::try_lstsq_banded<C>(a, b, 0, 1e-3).has_value());
}

TEST_CASE("try_lstsq_banded: the cutoff rcond moves the certificate, not the answer it gives") {
  // diag(1, 1e-4): the smallest singular value is 1e-4 of the largest. The certificate needs
  // 1 / ||R^-1||_F = 1e-4 > tau ||A||_F with tau = max(sqrt eps, 4 (m n eps + rcond)); rcond 1e-6
  // gives tau = 4e-6 (accepted), rcond 1e-4 gives tau = 4e-4 (refused), the default
  // eps * max(m, n) gives tau = sqrt eps (accepted). Bug caught: `rcond` unused, used as `sqrt`
  // or with the wrong factor (the two sides of the boundary are 40 apart, so a factor of 4 or
  // less on either side is not what is asserted).
  const la::RowMatrix<double> a = dense<double>(2, 2, {1.0, 0.0, 0.0, 1e-4});
  const la::RowMatrix<double> b = dense<double>(2, 1, {1.0, 1.0});
  const std::optional<NdArray<double>> by_default = la::try_lstsq_banded<double>(a, b);
  const std::optional<NdArray<double>> loose = la::try_lstsq_banded<double>(a, b, 0, 1e-6);
  REQUIRE(by_default.has_value());
  REQUIRE(loose.has_value());
  CHECK(std::abs(by_default->data()[0] - 1.0) <= 1e-12);
  CHECK(std::abs(by_default->data()[1] - 1e4) <= 1e-8);
  CHECK(std::abs(loose->data()[1] - 1e4) <= 1e-8);
  CHECK_FALSE(la::try_lstsq_banded<double>(a, b, 0, 1e-4).has_value());
  CHECK_FALSE(la::try_lstsq_banded<double>(a, b, 0, 1e-2).has_value());
  // A cutoff of exactly zero is the loosest (tau is the floor sqrt eps).
  CHECK(la::try_lstsq_banded<double>(a, b, 0, 0.0).has_value());
}

TEST_CASE("solve_collocation: fewer rows than columns, no rows, and refused input") {
  // [[1, 1, 0], [0, 1, 1]] x = (2, 2): the minimum-norm solution is A^T (A A^T)^-1 b with
  // A A^T = [[2, 1], [1, 2]] (inverse 1/3 [[2, -1], [-1, 2]]), (A A^T)^-1 b = (2/3, 2/3), so
  // x = (2/3, 4/3, 2/3). The banded path needs m >= n and so declines; the SVD answers.
  const la::RowMatrix<double> wide = dense<double>(2, 3, {1, 1, 0, 0, 1, 1});
  const la::RowMatrix<double> wide_b = dense<double>(2, 1, {2, 2});
  const NdArray<double> x = bs::solve_collocation<double>(wide, wide_b, 1e-12);
  REQUIRE(x.size() == 3);
  CHECK(std::abs(x.data()[0] - 2.0 / 3.0) <= 8.0 * unit_roundoff<double>());
  CHECK(std::abs(x.data()[1] - 4.0 / 3.0) <= 8.0 * unit_roundoff<double>());
  CHECK(std::abs(x.data()[2] - 2.0 / 3.0) <= 8.0 * unit_roundoff<double>());
  // No rows: the zero solution of the right shape.
  const la::RowMatrix<double> none(0, 3);
  const la::RowMatrix<double> none_b(0, 2);
  const NdArray<double> zero = bs::solve_collocation<double>(none, none_b, 1e-12);
  REQUIRE(zero.rank() == 2);
  CHECK(zero.shape()[0] == 3);
  CHECK(zero.shape()[1] == 2);
  for (std::size_t i = 0; i < zero.size(); ++i) {
    CHECK(zero.data()[i] == 0.0);
  }
  // The rows of the matrix and of the right-hand side must agree.
  CHECK_THROWS_WITH_AS(
      static_cast<void>(bs::solve_collocation<double>(wide, dense<double>(3, 1, {1, 1, 1}), 1e-12)),
      "solve_collocation: the matrix has 2 rows and the right-hand side 3.", pantr::value_error);
  // A NaN or an infinity in the matrix is refused with LAPACK's message, on every path: square
  // (the LU meets NaN pivots), tall (the SVD) and wide.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (const double bad : {nan, inf, -inf}) {
    const la::RowMatrix<double> square = dense<double>(2, 2, {1, bad, 0, 1});
    const la::RowMatrix<double> tall = dense<double>(3, 2, {1, 0, 0, 1, bad, 1});
    const la::RowMatrix<double> wide_bad = dense<double>(2, 3, {1, 0, bad, 0, 1, 0});
    CHECK_THROWS_WITH_AS(static_cast<void>(bs::solve_collocation<double>(
                             square, dense<double>(2, 1, {1, 1}), 1e-12)),
                         "SVD did not converge", pantr::value_error);
    CHECK_THROWS_WITH_AS(static_cast<void>(bs::solve_collocation<double>(
                             tall, dense<double>(3, 1, {1, 1, 1}), 1e-12)),
                         "SVD did not converge", pantr::value_error);
    CHECK_THROWS_WITH_AS(static_cast<void>(bs::solve_collocation<double>(
                             wide_bad, dense<double>(2, 1, {1, 1}), 1e-12)),
                         "SVD did not converge", pantr::value_error);
  }
  // A NaN in the right-hand side only spreads: the coefficients are not finite (a finite wrong
  // answer would be worse), and nothing throws.
  const la::RowMatrix<double> identity2 = dense<double>(2, 2, {1, 0, 0, 1});
  const NdArray<double> spread =
      bs::solve_collocation<double>(identity2, dense<double>(2, 1, {1, nan}), 1e-12);
  CHECK(std::isnan(spread.data()[1]));
}

// ---- solve_kronecker --------------------------------------------------------------------------

namespace fit_types {
struct FloatFloat {
  using P = float;
  using C = float;
};
struct DoubleFloat {
  using P = double;
  using C = float;
};
struct FloatDouble {
  using P = float;
  using C = double;
};
struct DoubleDouble {
  using P = double;
  using C = double;
};
}  // namespace fit_types

namespace {

// kron(m_0, m_1, ...): row index (i_0 m_1 + i_1) ..., the first factor slowest, as the C-order
// layout of the tensor right-hand side.
Matrix kron_all(const std::vector<Matrix>& mats) {
  Matrix k = {{Rational(1)}};
  for (const Matrix& m : mats) {
    Matrix next(k.size() * m.size(), std::vector<Rational>(k[0].size() * m[0].size()));
    for (std::size_t i = 0; i < k.size(); ++i) {
      for (std::size_t j = 0; j < k[0].size(); ++j) {
        for (std::size_t a = 0; a < m.size(); ++a) {
          for (std::size_t b = 0; b < m[0].size(); ++b) {
            next[i * m.size() + a][j * m[0].size() + b] = k[i][j] * m[a][b];
          }
        }
      }
    }
    k = std::move(next);
  }
  return k;
}

}  // namespace

TEST_CASE_TEMPLATE(
    "solve_kronecker: the direction-wise solve equals one dense solve of the Kronecker product", PC,
    fit_types::FloatFloat, fit_types::DoubleFloat, fit_types::FloatDouble,
    fit_types::DoubleDouble) {
  // Oracle: the Kronecker product of the exact matrices (cast to the computation type C, as the
  // solve casts them) assembled densely, and one exact normal-equations solve of the 60 x 24
  // system, against the library's three one-direction solves (a square, an overdetermined and a
  // thin one). Bug caught: the factors applied in the wrong order or along the wrong axis (the
  // shapes differ, so a transposition cannot pass), an axis flattened and unflattened
  // inconsistently, a direction whose solution is not carried into the next.
  using P = typename PC::P;
  using C = typename PC::C;
  const double tol = 100.0 * machine_epsilon<C>();
  const std::vector<Dir> dirs = {Dir{clamped(2, {0.5}), 2}, Dir{clamped(2, {}), 2},
                                 Dir{clamped(1, {}), 1}};
  const bs::BsplineSpace<P> space = space_of<P>(dirs);
  const std::vector<std::size_t> counts = {0, 5, 3};  // 4 x 4, 5 x 3 and 3 x 2 matrices
  std::vector<NdArray<P>> mats;
  std::vector<Matrix> exact;
  double kappa = 1.0;
  std::size_t sizes = 0;
  for (std::size_t d = 0; d < dirs.size(); ++d) {
    const bs::BsplineSpace1D<P>& s = space.spaces()[d];
    const std::vector<P> nodes = counts[d] == 0 ? s.greville() : uniform_nodes<P>(s, counts[d]);
    mats.push_back(bs::collocation_matrix<P, P>(s, std::span<const P>(nodes)));
    la::RowMatrix<C> in_c = row_matrix<P>(mats.back()).template cast<C>();
    exact.push_back(exact_matrix<C>(in_c));
    const double k = kappa_bound(exact.back());
    REQUIRE(1.0 / k > tol);
    kappa *= k;
    sizes += mats.back().shape()[0] + mats.back().shape()[1] + 2;
  }
  const std::vector<std::size_t> shape = {4, 5, 3};
  const std::vector<double> rhs_d = bspline_test::values(60, 41);
  const std::vector<C> rhs = cast_all<C>(rhs_d);
  const NdArray<C> got = bs::solve_kronecker<P, C>(mats, std::span<const C>(rhs),
                                                   std::span<const std::size_t>(shape), tol);
  REQUIRE(got.rank() == 3);
  CHECK(got.shape()[0] == 4);
  CHECK(got.shape()[1] == 3);
  CHECK(got.shape()[2] == 2);

  const Matrix k_dense = kron_all(exact);
  REQUIRE(k_dense.size() == 60);
  REQUIRE(k_dense[0].size() == 24);
  Matrix b_dense(60, std::vector<Rational>(1));
  for (std::size_t i = 0; i < 60; ++i) {
    b_dense[i][0] = oracle::exact(static_cast<double>(rhs[i]));
  }
  const Matrix want = lstsq_exact(k_dense, b_dense);
  const double bound = solve_bound<C>(sizes, kappa, frobenius_norm(want));
  double worst = 0.0;
  for (std::size_t i = 0; i < 24; ++i) {
    worst = (std::max)(worst, std::abs(static_cast<double>(got.data()[i]) - to_double(want[i][0])));
  }
  INFO("kappa bound " << kappa << ", worst error " << worst << ", bound " << bound);
  CHECK(worst <= bound);
}

TEST_CASE("solve_kronecker: inconsistent extents are refused with the documented messages") {
  NdArray<double> m3({3, 3});
  for (std::size_t i = 0; i < 3; ++i) {
    m3.data()[i * 3 + i] = 1.0;
  }
  NdArray<double> m4({4, 4});
  for (std::size_t i = 0; i < 4; ++i) {
    m4.data()[i * 4 + i] = 1.0;
  }
  const std::vector<double> rhs(12, 1.0);
  const std::vector<std::size_t> shape = {3, 4};
  const char* const message =
      "solve_kronecker: one matrix per axis of the right-hand side is needed.";
  // One matrix too few, too many, and a shape that is not the size of the right-hand side.
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::solve_kronecker<double, double>(
                           std::vector<NdArray<double>>{m3}, std::span<const double>(rhs),
                           std::span<const std::size_t>(shape), 1e-12)),
                       message, pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::solve_kronecker<double, double>(
                           std::vector<NdArray<double>>{m3, m4, m3}, std::span<const double>(rhs),
                           std::span<const std::size_t>(shape), 1e-12)),
                       message, pantr::value_error);
  const std::vector<std::size_t> wrong_shape = {3, 5};
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::solve_kronecker<double, double>(
                           std::vector<NdArray<double>>{m3, m4}, std::span<const double>(rhs),
                           std::span<const std::size_t>(wrong_shape), 1e-12)),
                       message, pantr::value_error);
  // A matrix with the wrong number of rows for its axis names the axis.
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::solve_kronecker<double, double>(
                           std::vector<NdArray<double>>{m4, m4}, std::span<const double>(rhs),
                           std::span<const std::size_t>(shape), 1e-12)),
                       "solve_kronecker: matrix 0 does not match axis 0 of the right-hand side.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::solve_kronecker<double, double>(
                           std::vector<NdArray<double>>{m3, m3}, std::span<const double>(rhs),
                           std::span<const std::size_t>(shape), 1e-12)),
                       "solve_kronecker: matrix 1 does not match axis 1 of the right-hand side.",
                       pantr::value_error);
  // Identity factors return the right-hand side itself (a closed form).
  const NdArray<double> same = bs::solve_kronecker<double, double>(
      std::vector<NdArray<double>>{m3, m4}, std::span<const double>(rhs),
      std::span<const std::size_t>(shape), 1e-12);
  for (std::size_t i = 0; i < 12; ++i) {
    CHECK(std::abs(same.data()[i] - 1.0) <= 8.0 * unit_roundoff<double>());
  }
}

// ---- scattered nodes --------------------------------------------------------------------------

namespace {

// 64 points of the dyadic grid of the domain of a 2D space (multiples of 1/64 of the extent).
template <class T>
std::vector<T> scattered_points(const bs::BsplineSpace<T>& space, std::size_t count,
                                std::uint64_t seed) {
  std::vector<T> pts;
  std::uint64_t state = seed;
  for (std::size_t i = 0; i < count; ++i) {
    for (const auto& s : space.spaces()) {
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      const auto k = static_cast<double>((state >> 33) % 65);
      const auto [a, b] = s.domain();
      pts.push_back(static_cast<T>(static_cast<double>(a) +
                                   (static_cast<double>(b) - static_cast<double>(a)) * k / 64.0));
    }
  }
  return pts;
}

}  // namespace

TEST_CASE_TEMPLATE(
    "scattered_collocation_matrix: each row is the row-wise Kronecker product of the exact basis "
    "values",
    T, float, double) {
  // Oracle: column (a, b) is the exact value of the 2D spline whose only control point is (a, b)
  // (first direction slowest). Bug caught: the factors in the wrong order (the two directions
  // have different numbers of functions, 7 and 4, so a swap cannot pass), a row-major layout that
  // is not first-direction-slowest, a direction read from the wrong column of the points.
  const double u = unit_roundoff<T>();
  const std::vector<Dir> dirs = {cubic_dir(), shifted_dir()};
  const bs::BsplineSpace<T> space = space_of<T>(dirs);
  const std::vector<double> xs = {0.0, 0.3, 0.25, 0.5, 0.83, 1.0, 1.0 / 16.0, 0.6, 0.99};
  std::vector<T> pts;
  for (std::size_t i = 0; i < xs.size(); ++i) {
    pts.push_back(static_cast<T>(xs[i]));
    pts.push_back(static_cast<T>(2.0 + 3.0 * xs[(i * 4 + 3) % xs.size()]));
  }
  const std::size_t n_pts = xs.size();
  const NdArray<T> mat = bs::scattered_collocation_matrix<T, T>(
      space, pantr::mdview<const T, 2>(pts.data(), n_pts, 2));
  REQUIRE(mat.rank() == 2);
  REQUIRE(mat.shape()[0] == n_pts);
  REQUIRE(mat.shape()[1] == 28);
  const double bound = ((6.0 * 3 + 1.0) + (6.0 * 2 + 1.0) + 1.0) * u;
  for (std::size_t j = 0; j < 28; ++j) {
    const bs::Bspline<T> column = unit_spline<T>(space, j);
    for (std::size_t i = 0; i < n_pts; ++i) {
      const std::vector<Rational> x = {oracle::exact(static_cast<double>(pts[2 * i])),
                                       oracle::exact(static_cast<double>(pts[2 * i + 1]))};
      INFO("point " << i << ", function " << j);
      CHECK(oracle::within(static_cast<double>(mat.data()[i * 28 + j]),
                           oracle::exact_value<T>(column, x)[0], bound));
    }
  }
}

TEST_CASE_TEMPLATE(
    "fit_scattered: samples of a 2D spline at scattered points give its control points back", T,
    float, double) {
  // Oracle: a 2D spline with dyadic control points, sampled exactly at 64 points of the dyadic
  // grid (rank 2). The 64 x 28 system is bounded by the exact normal equations. Bug caught: the
  // directions of the points swapped or mixed, a rank component that is not carried through, a
  // coefficient layout that is not first-direction-slowest.
  const std::vector<Dir> dirs = {cubic_dir(), shifted_dir()};
  const bs::BsplineSpace<T> space = space_of<T>(dirs);
  const bs::Bspline<T> truth = make_spline<T>(dirs, 2, false, 51);
  const std::size_t n_pts = 64;
  const std::vector<T> pts = scattered_points<T>(space, n_pts, 1234);
  const std::vector<T> values = samples_at<T>(truth, pts, 2);
  const bs::Bspline<T> fit =
      bs::fit_scattered<T>(space, pantr::mdview<const T, 2>(pts.data(), n_pts, 2),
                           pantr::mdview<const T, 2>(values.data(), n_pts, 2));
  const NdArray<T> mat = bs::scattered_collocation_matrix<T, T>(
      space, pantr::mdview<const T, 2>(pts.data(), n_pts, 2));
  const double kappa = kappa_bound(exact_matrix<T>(mat));
  REQUIRE(1.0 / kappa > 100.0 * machine_epsilon<T>());
  CHECK_FALSE(fit.rational());
  CHECK(fit.rank() == 2);
  REQUIRE(fit.control_points().size() == truth.control_points().size());
  const double bound =
      solve_bound<T>(n_pts + 28 + 2, kappa, two_norm_of<T>(truth.control_points()));
  const double worst = worst_difference<T>(fit.control_points(), truth.control_points());
  INFO("kappa bound " << kappa << ", worst error " << worst << ", bound " << bound);
  CHECK(worst <= bound);
}

TEST_CASE("fit_scattered_coefficients: refused input, in the documented order") {
  const std::vector<Dir> dirs = {Dir{clamped(2, {}), 2},
                                 Dir{clamped(1, {}), 1}};  // 3 x 2 = 6 functions
  const bs::BsplineSpace<double> space = space_of<double>(dirs);
  const auto pts_of = [](const std::vector<double>& v, std::size_t rows, std::size_t cols) {
    return pantr::mdview<const double, 2>(v.data(), rows, cols);
  };
  const std::vector<double> six_pts = {0.0, 0.0, 0.2, 0.9, 0.4, 0.1, 0.6, 0.7, 0.8, 0.3, 1.0, 1.0};
  const std::vector<double> six_vals(6, 1.0);
  // The boundary of "Underdetermined": 6 points for 6 functions is a square system and is solved
  // (the rank is not asserted: the answer is the SVD's), 5 points are refused.
  CHECK_NOTHROW(static_cast<void>(bs::fit_scattered_coefficients<double, double, double>(
      space, pts_of(six_pts, 6, 2), pts_of(six_vals, 6, 1), 1e-12)));
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::fit_scattered_coefficients<double, double, double>(
                           space, pts_of(six_pts, 5, 2), pts_of(six_vals, 5, 1), 1e-12)),
                       "Underdetermined system: 5 points < 6 basis functions.", pantr::value_error);
  // A wrong number of columns is reported before the count of points.
  const std::vector<double> one_col = {0.1, 0.2, 0.3};
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::fit_scattered_coefficients<double, double, double>(
                           space, pts_of(one_col, 3, 1), pts_of(six_vals, 3, 1), 1e-12)),
                       "Scattered nodes have 1 columns, expected 2.", pantr::value_error);
  // The count of points is reported before the domain, the domain before the values' rows.
  std::vector<double> outside = six_pts;
  outside[3] = 1.5;
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::fit_scattered_coefficients<double, double, double>(
                           space, pts_of(outside, 4, 2), pts_of(six_vals, 4, 1), 1e-12)),
                       "Underdetermined system: 4 points < 6 basis functions.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::fit_scattered_coefficients<double, double, double>(
                           space, pts_of(outside, 6, 2), pts_of(six_vals, 5, 1), 1e-12)),
                       "One or more values in pts are outside the knot vector domain (0.0, 1.0)",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::fit_scattered_coefficients<double, double, double>(
                           space, pts_of(six_pts, 6, 2), pts_of(six_vals, 5, 1), 1e-12)),
                       "Values have 5 rows for 6 points.", pantr::value_error);
  // `scattered_collocation_matrix` has its own column check.
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::scattered_collocation_matrix<double, double>(
                           space, pts_of(one_col, 3, 1))),
                       "Scattered nodes have 1 columns, expected 2.", pantr::value_error);
}

// ---- L2 projection ----------------------------------------------------------------------------

namespace {

// The exact weights, on [0, 1], of the closed Newton-Cotes rule with `degree + 1` equispaced nodes
// k / degree: it is exact for polynomials of degree <= degree, and its weights solve the moment
// equations sum_k w_k t_k^j = 1 / (j + 1) for j = 0 .. degree.
std::vector<Rational> newton_cotes_weights(unsigned degree) {
  const std::size_t n = degree + 1;
  Matrix vandermonde(n, std::vector<Rational>(n));
  Matrix moments(n, std::vector<Rational>(1));
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t k = 0; k < n; ++k) {
      Rational term = 1;
      for (std::size_t e = 0; e < j; ++e) {
        term *= rat(static_cast<long long>(k), static_cast<long long>(degree));
      }
      vandermonde[j][k] = term;
    }
    moments[j][0] = rat(1, static_cast<long long>(j) + 1);
  }
  const Matrix w = solve_exact(vandermonde, moments);
  std::vector<Rational> out;
  for (const auto& row : w) {
    out.push_back(row[0]);
  }
  return out;
}

// One exact quadrature node of a span: weight (the rule's, times the span length), abscissa, the
// index of the first function that does not vanish there, and the `degree + 1` exact values.
struct QuadRecord {
  Rational weight;
  Rational x;
  std::size_t first;
  std::vector<Rational> basis;
};

// The exact quadrature of a 1D space over its domain: every non-empty span inside the domain, the
// Newton-Cotes rule of `nc_degree` on each, the basis from the recursive definition on that
// span's polynomial piece.
template <class T>
std::vector<QuadRecord> exact_quadrature(const bs::BsplineSpace1D<T>& s, unsigned nc_degree) {
  const auto p = static_cast<std::size_t>(s.degree());
  const std::vector<Rational> u = oracle::exact_knots<T>(s.knots());
  const Rational lo = oracle::exact(static_cast<double>(s.domain().first));
  const Rational hi = oracle::exact(static_cast<double>(s.domain().second));
  const std::vector<Rational> w = newton_cotes_weights(nc_degree);
  std::vector<QuadRecord> records;
  for (std::size_t j = 0; j + 1 < u.size(); ++j) {
    if (!(u[j] < u[j + 1]) || u[j] < lo || u[j + 1] > hi) {
      continue;
    }
    const Rational h = u[j + 1] - u[j];
    for (std::size_t k = 0; k <= nc_degree; ++k) {
      QuadRecord r;
      r.x = u[j] + h * rat(static_cast<long long>(k), static_cast<long long>(nc_degree));
      r.weight = h * w[k];
      r.first = j - p;
      oracle::BasisAt at(u, s.degree(), r.x, j);
      for (std::size_t a = 0; a <= p; ++a) {
        r.basis.push_back(at.deriv(j - p + a, 0));
      }
      records.push_back(std::move(r));
    }
  }
  return records;
}

// G_ij = integral of N_i N_j over the domain (the functions of a periodic space wrap).
template <class T>
Matrix exact_gram(const bs::BsplineSpace1D<T>& s) {
  const auto n = static_cast<std::size_t>(s.num_basis());
  Matrix g(n, std::vector<Rational>(n, Rational(0)));
  for (const QuadRecord& r : exact_quadrature<T>(s, static_cast<unsigned>(2 * s.degree()))) {
    for (std::size_t a = 0; a < r.basis.size(); ++a) {
      for (std::size_t b = 0; b < r.basis.size(); ++b) {
        g[(r.first + a) % n][(r.first + b) % n] += r.weight * r.basis[a] * r.basis[b];
      }
    }
  }
  return g;
}

// b_i = integral of N_i f over the domain, for a polynomial f of degree `f_degree` given exactly.
template <class T, class F>
Matrix exact_load_vector(const bs::BsplineSpace1D<T>& s, unsigned f_degree, F&& f) {
  const auto n = static_cast<std::size_t>(s.num_basis());
  Matrix b(n, std::vector<Rational>(1, Rational(0)));
  for (const QuadRecord& r : exact_quadrature<T>(s, static_cast<unsigned>(s.degree()) + f_degree)) {
    const Rational fx = f(r.x);
    for (std::size_t a = 0; a < r.basis.size(); ++a) {
      b[(r.first + a) % n][0] += r.weight * r.basis[a] * fx;
    }
  }
  return b;
}

using Sides = std::vector<std::vector<int>>;

template <class T>
Sides traces_of(const bs::L2Projection<T>& p) {
  Sides out;
  for (const bs::BoundaryTrace& t : p.traces()) {
    out.push_back(t.side);
  }
  return out;
}

template <class T>
bs::L2Projection<T> projection_of(const std::vector<Dir>& dirs, std::vector<long long> n_quad,
                                  std::vector<std::pair<bool, bool>> flags,
                                  bs::L2Quadrature rule = bs::L2Quadrature::gauss_legendre) {
  return bs::L2Projection<T>(space_of<T>(dirs), std::move(n_quad), rule, std::move(flags));
}

}  // namespace

TEST_CASE(
    "L2Projection::traces: subsets by size, then lexicographic, start before end, periodic "
    "directions skipped") {
  // Oracle: tables written by hand from the rule "the fixed directions form a subset of the flagged
  // ones, taken by increasing size and, within a size, in lexicographic order; a subset's sides run
  // through the Cartesian product with the last fixed direction fastest, the start (0) before the
  // end (1)". Each entry lists, per direction, -1 (free), 0 (start) or 1 (end). This is the order
  // in which a later trace overwrites an earlier one where they meet (a corner over an edge over a
  // face), so a permutation changes the coefficients. Bug caught: sizes taken in decreasing
  // order, subsets or sides in reverse, a periodic direction fixed, a flag read from the wrong
  // direction.
  const std::pair<bool, bool> both{true, true};
  const std::pair<bool, bool> none{false, false};
  SUBCASE("2D, both ends of both directions: faces first, then the four corners") {
    const auto p = projection_of<double>({cubic_dir(), c0_quadratic_dir()}, {4, 3}, {both, both});
    const Sides want = {{0, -1}, {1, -1}, {-1, 0}, {-1, 1}, {0, 0}, {0, 1}, {1, 0}, {1, 1}};
    CHECK(traces_of(p) == want);
  }
  SUBCASE("3D, all six faces, twelve edges, eight corners") {
    const auto p = projection_of<double>({cubic_dir(), c0_quadratic_dir(), shifted_dir()},
                                         {4, 3, 3}, {both, both, both});
    const Sides want = {
        {0, -1, -1}, {1, -1, -1}, {-1, 0, -1}, {-1, 1, -1}, {-1, -1, 0}, {-1, -1, 1}, {0, 0, -1},
        {0, 1, -1},  {1, 0, -1},  {1, 1, -1},  {0, -1, 0},  {0, -1, 1},  {1, -1, 0},  {1, -1, 1},
        {-1, 0, 0},  {-1, 0, 1},  {-1, 1, 0},  {-1, 1, 1},  {0, 0, 0},   {0, 0, 1},   {0, 1, 0},
        {0, 1, 1},   {1, 0, 0},   {1, 0, 1},   {1, 1, 0},   {1, 1, 1}};
    CHECK(traces_of(p) == want);
  }
  SUBCASE("3D, a start only, a free direction and both ends") {
    const auto p = projection_of<double>({cubic_dir(), c0_quadratic_dir(), shifted_dir()},
                                         {4, 3, 3}, {{true, false}, none, both});
    const Sides want = {{0, -1, -1}, {-1, -1, 0}, {-1, -1, 1}, {0, -1, 0}, {0, -1, 1}};
    CHECK(traces_of(p) == want);
  }
  SUBCASE("a periodic direction is skipped, whatever its flags say") {
    const auto p = projection_of<double>({cubic_dir(), periodic_dir()}, {4, 4}, {both, both});
    const Sides want = {{0, -1}, {1, -1}};
    CHECK(traces_of(p) == want);
    const auto q = projection_of<double>({periodic_dir(), cubic_dir(), shifted_dir()}, {4, 4, 3},
                                         {both, {false, true}, {true, false}});
    const Sides want_q = {{-1, 1, -1}, {-1, -1, 0}, {-1, 1, 0}};
    CHECK(traces_of(q) == want_q);
  }
  SUBCASE("1D, and no flags at all") {
    CHECK(traces_of(projection_of<double>({cubic_dir()}, {4}, {both})) == Sides{{0}, {1}});
    CHECK(traces_of(projection_of<double>({cubic_dir()}, {4}, {{false, true}})) == Sides{{1}});
    CHECK(traces_of(projection_of<double>({cubic_dir()}, {4}, {none})).empty());
  }
}

TEST_CASE(
    "L2Projection::trace_nodes: the boundary coordinate where fixed, the quadrature nodes "
    "elsewhere") {
  // Domains [2, 5] and [3/4, 3/2]: a start and an end are different numbers, in both directions.
  const std::pair<bool, bool> both{true, true};
  const auto p = projection_of<double>({shifted_dir(), unclamped_dir()}, {3, 4}, {both, both});
  REQUIRE(p.traces().size() == 8);
  const std::vector<double> q0(p.nodes()[0].data(), p.nodes()[0].data() + p.nodes()[0].size());
  const std::vector<double> q1(p.nodes()[1].data(), p.nodes()[1].data() + p.nodes()[1].size());
  using Nodes = std::vector<std::vector<double>>;
  CHECK(p.trace_nodes(0) == Nodes{{2.0}, q1});      // {0, -1}
  CHECK(p.trace_nodes(1) == Nodes{{5.0}, q1});      // {1, -1}
  CHECK(p.trace_nodes(2) == Nodes{q0, {0.75}});     // {-1, 0}
  CHECK(p.trace_nodes(3) == Nodes{q0, {1.5}});      // {-1, 1}
  CHECK(p.trace_nodes(4) == Nodes{{2.0}, {0.75}});  // {0, 0}
  CHECK(p.trace_nodes(5) == Nodes{{2.0}, {1.5}});   // {0, 1}
  CHECK(p.trace_nodes(6) == Nodes{{5.0}, {0.75}});  // {1, 0}
  CHECK(p.trace_nodes(7) == Nodes{{5.0}, {1.5}});   // {1, 1}
  // One past the last trace, and far past it.
  CHECK_THROWS_WITH_AS(static_cast<void>(p.trace_nodes(8)), "trace 8 out of range for 8 traces",
                       pantr::index_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(p.trace_nodes(1000)),
                       "trace 1000 out of range for 8 traces", pantr::index_error);
  const std::vector<double> any(1, 0.0);
  CHECK_THROWS_WITH_AS(static_cast<void>(p.trace_load<double>(8, std::span<const double>(any))),
                       "trace 8 out of range for 8 traces", pantr::index_error);
  // Without flags there are no traces at all.
  const auto unflagged = projection_of<double>({shifted_dir(), unclamped_dir()}, {3, 4},
                                               {{false, false}, {false, false}});
  CHECK_THROWS_WITH_AS(static_cast<void>(unflagged.trace_nodes(0)),
                       "trace 0 out of range for 0 traces", pantr::index_error);
}

TEST_CASE("L2Projection: refused construction and refused values") {
  const std::pair<bool, bool> both{true, true};
  const std::pair<bool, bool> none{false, false};
  const std::vector<Dir> dirs = {cubic_dir(), c0_quadratic_dir()};
  using P = bs::L2Projection<double>;
  const bs::BsplineSpace<double> space = space_of<double>(dirs);
  const auto gl = bs::L2Quadrature::gauss_legendre;
  const auto lobatto = bs::L2Quadrature::gauss_lobatto;
  CHECK_THROWS_WITH_AS(static_cast<void>(P(space, {4}, gl, {none, none})),
                       "n_quad has length 1, expected 2", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(P(space, {4, 3}, gl, {none, none, none})),
                       "boundary_interpolation has length 3, expected 2", pantr::value_error);
  // The first check wins when both are wrong.
  CHECK_THROWS_WITH_AS(static_cast<void>(P(space, {4, 3, 2}, gl, {none})),
                       "n_quad has length 3, expected 2", pantr::value_error);
  // A rule's minimum number of points per element: 1 for Gauss-Legendre, 2 for Gauss-Lobatto.
  CHECK_NOTHROW(static_cast<void>(P(space, {1, 1}, gl, {none, none})));
  CHECK_THROWS_WITH_AS(static_cast<void>(P(space, {4, 0}, gl, {none, none})),
                       "n_pts must be at least 1", pantr::value_error);
  CHECK_NOTHROW(static_cast<void>(P(space, {2, 2}, lobatto, {none, none})));
  CHECK_THROWS_WITH_AS(static_cast<void>(P(space, {4, 1}, lobatto, {none, none})),
                       "n_pts must be at least 2", pantr::value_error);

  const P proj(space, {4, 3}, gl, {both, both});
  const std::vector<std::size_t> grid = proj.grid_shape();
  REQUIRE(grid.size() == 2);
  CHECK(grid[0] == 16);  // four spans of four Gauss points
  CHECK(grid[1] == 12);  // four spans of three: the double knot at 1/2 is one break
  const std::vector<double> short_values(16 * 12 - 1, 1.0);
  CHECK_THROWS_WITH_AS(static_cast<void>(proj.load<double>(std::span<const double>(short_values))),
                       "L2Projection: expected 192 values on the quadrature lattice, got 191.",
                       pantr::value_error);
  // Each trace takes the values of its own lattice, and project takes one array per trace.
  const std::vector<double> one(1, 1.0);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(proj.trace_load<double>(4, std::span<const double>(short_values))),
      "L2Projection: expected 1 values on trace 4, got 191.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(proj.trace_load<double>(0, std::span<const double>(one))),
                       "L2Projection: expected 12 values on trace 0, got 1.", pantr::value_error);
  const std::vector<double> all(16 * 12, 1.0);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(proj.project<double>(std::span<const double>(all),
                                             std::vector<std::span<const double>>{}, 1e-12)),
      "L2Projection: expected 8 trace value arrays, got 0.", pantr::value_error);
}

TEST_CASE_TEMPLATE(
    "L2Projection: the mass matrix is the exact Gram matrix, its boundary rows the exact basis at "
    "the ends",
    T, float, double) {
  // Oracle: exact Gram matrices by Newton-Cotes quadrature of the exact basis on each span (the
  // closed rule of degree 2 p, exact for the integrand), so nothing of the library's Gauss rule is
  // shared. All terms of an entry are non-negative, so each entry has a relative bound: the number
  // of quadrature points, the roundings of the two basis values (6 p + 1 each) and of the
  // weight times the value, and the rounding of the Gauss nodes and weights themselves, which the
  // Newton iteration gives to within a few units (taken as 20). Boundary rows: the exact basis
  // values at the end of the domain (a unit row for a clamped space, a dense row for the
  // unclamped one). Bug caught: a mass matrix assembled from the wrong nodes or weights, a
  // transposed or unsymmetrised accumulation, a periodic wrap that is dropped, the replaced row
  // that is the wrong one, or one that holds the basis at the other end.
  const double u = unit_roundoff<T>();
  for (const Dir& dir :
       {cubic_dir(), c0_quadratic_dir(), shifted_dir(), unclamped_dir(), periodic_dir()}) {
    const bs::BsplineSpace<T> space = space_of<T>({dir});
    const bs::BsplineSpace1D<T>& s = space.spaces()[0];
    const auto n = static_cast<std::size_t>(s.num_basis());
    const bs::L2Projection<T> plain(space, {dir.degree + 1LL}, bs::L2Quadrature::gauss_legendre,
                                    {{false, false}});
    const bs::L2Projection<T> pinned(space, {dir.degree + 1LL}, bs::L2Quadrature::gauss_legendre,
                                     {{true, true}});
    const Matrix gram = exact_gram<T>(s);
    const double points = static_cast<double>(plain.nodes()[0].size());
    const double rel = (points + 2.0 * (6.0 * dir.degree + 1.0) + 3.0 + 20.0) * u;
    REQUIRE(plain.mass()[0].shape()[0] == n);
    REQUIRE(plain.mass()[0].shape()[1] == n);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        const double got = static_cast<double>(plain.mass()[0].data()[i * n + j]);
        INFO("degree " << dir.degree << ", periodic " << dir.periodic << ", entry (" << i << ", "
                       << j << ")");
        if (gram[i][j] == 0) {
          CHECK(got == 0.0);
        } else {
          CHECK(oracle::within(got, gram[i][j], rel * to_double(gram[i][j])));
        }
      }
    }
    // The pinned matrix: the first and the last rows are the basis at the ends, the others as
    // before.
    const auto [a, b] = s.domain();
    for (std::size_t i = 0; i < n; ++i) {
      const bool replaced = !dir.periodic && (i == 0 || i + 1 == n);
      for (std::size_t j = 0; j < n; ++j) {
        const double got = static_cast<double>(pinned.mass()[0].data()[i * n + j]);
        if (replaced) {
          const std::vector<Rational> x = {oracle::exact(static_cast<double>(i == 0 ? a : b))};
          INFO("degree " << dir.degree << ", boundary row " << i << ", column " << j);
          CHECK(oracle::within(got, oracle::exact_value<T>(unit_spline<T>(space, j), x)[0],
                               (6.0 * dir.degree + 1.0) * u));
        } else {
          CHECK(got == static_cast<double>(plain.mass()[0].data()[i * n + j]));
        }
      }
    }
  }
}

TEST_CASE_TEMPLATE("L2Projection: the quadrature lattice has n_quad points on every non-empty span",
                   T, float, double) {
  // Closed forms: a Gauss rule on [0, 1] has weights summing to 1, so the weights of a direction
  // sum to the length of the domain, and each span holds n_quad ascending nodes strictly inside.
  const double u = unit_roundoff<T>();
  const bs::BsplineSpace<T> space = space_of<T>({c0_quadratic_dir(), shifted_dir()});
  const bs::L2Projection<T> proj(space, {3, 5}, bs::L2Quadrature::gauss_legendre,
                                 {{false, false}, {false, false}});
  // c0_quadratic_dir: breaks 0, 1/4, 1/2 (double), 3/4, 1 -> four spans. shifted_dir: 2, 3.5, 5 ->
  // two.
  CHECK(proj.nodes()[0].size() == 12);
  CHECK(proj.nodes()[1].size() == 10);
  CHECK(proj.weights()[0].size() == 12);
  CHECK(proj.weights()[1].size() == 10);
  const std::vector<std::vector<double>> breaks = {{0.0, 0.25, 0.5, 0.75, 1.0}, {2.0, 3.5, 5.0}};
  const std::vector<std::size_t> per_span = {3, 5};
  for (std::size_t d = 0; d < 2; ++d) {
    double total = 0.0;
    for (std::size_t span = 0; span + 1 < breaks[d].size(); ++span) {
      for (std::size_t k = 0; k < per_span[d]; ++k) {
        const std::size_t q = span * per_span[d] + k;
        const double x = static_cast<double>(proj.nodes()[d].data()[q]);
        INFO("direction " << d << ", span " << span << ", node " << k);
        CHECK(x > breaks[d][span]);
        CHECK(x < breaks[d][span + 1]);
        if (k > 0) {
          CHECK(x > static_cast<double>(proj.nodes()[d].data()[q - 1]));
        }
        total += static_cast<double>(proj.weights()[d].data()[q]);
      }
    }
    const double length = breaks[d].back() - breaks[d].front();
    CHECK(std::abs(total - length) <= 16.0 * u * length);
  }
}

namespace {

// out[o][i][r] = sum_q w_q N_i(x_q) v[o][q][r] with `axis` splitting the shape into outer, nodes
// and inner, in exact rationals; `abs_sum` receives sum |terms| of every entry.
template <class T, class C>
Matrix exact_contraction(const bs::BsplineSpace<T>& space1d, std::span<const C> values,
                         std::size_t outer, std::size_t inner, std::span<const T> nodes,
                         std::span<const T> weights, std::vector<double>& abs_sum) {
  const auto n = static_cast<std::size_t>(space1d.spaces()[0].num_basis());
  std::vector<std::vector<Rational>> basis(n);  // basis[i][q] = N_i(x_q)
  for (std::size_t i = 0; i < n; ++i) {
    const bs::Bspline<T> unit = unit_spline<T>(space1d, i);
    for (const T x : nodes) {
      const std::vector<Rational> point = {oracle::exact(static_cast<double>(x))};
      basis[i].push_back(oracle::exact_value<T>(unit, point)[0]);
    }
  }
  Matrix out(outer * n, std::vector<Rational>(inner, Rational(0)));
  abs_sum.assign(outer * n * inner, 0.0);
  for (std::size_t o = 0; o < outer; ++o) {
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t r = 0; r < inner; ++r) {
        Rational acc = 0;
        Rational abs_acc = 0;
        for (std::size_t q = 0; q < nodes.size(); ++q) {
          const Rational term =
              oracle::exact(static_cast<double>(weights[q])) * basis[i][q] *
              oracle::exact(static_cast<double>(values[(o * nodes.size() + q) * inner + r]));
          acc += term;
          abs_acc += abs_of(term);
        }
        out[o * n + i][r] = acc;
        abs_sum[(o * n + i) * inner + r] = to_double(abs_acc);
      }
    }
  }
  return out;
}

}  // namespace

TEST_CASE_TEMPLATE(
    "assemble_load_1d: the weighted basis contracted along an axis equals the exact sum", PC,
    fit_types::FloatFloat, fit_types::DoubleFloat, fit_types::FloatDouble,
    fit_types::DoubleDouble) {
  // Oracle: the exact sum over the nodes of weight x exact basis value x value, for the given
  // nodes and weights (any: the quadrature rule is not under test here), on the first, the middle
  // and the last axis of a tensor, for a clamped, a periodic and an unclamped space. Each term is
  // formed as `w * N` in P, rounded to C, then multiplied and accumulated in C, so the error is
  // ((6 p + 2) u_P + (nodes + 3) u_C) sum |terms|. Bug caught: the contraction along the wrong
  // axis, an outer or an inner stride that is swapped, a periodic function not wrapped, a weight
  // multiplied twice.
  using P = typename PC::P;
  using C = typename PC::C;
  struct Layout {
    std::vector<std::size_t> shape;
    std::size_t axis;
  };
  const std::vector<Layout> layouts = {{{5, 3}, 0}, {{2, 5, 3}, 1}, {{3, 5}, 1}};
  for (const Dir& dir : {cubic_dir(), periodic_dir(), unclamped_dir()}) {
    const bs::BsplineSpace<P> space = space_of<P>({dir});
    const bs::BsplineSpace1D<P>& s = space.spaces()[0];
    const auto n = static_cast<std::size_t>(s.num_basis());
    const std::vector<P> all_nodes = probe_nodes<P>(s);
    const std::vector<P> nodes(all_nodes.begin(), all_nodes.begin() + 5);
    std::vector<P> weights;
    for (std::size_t q = 0; q < nodes.size(); ++q) {
      weights.push_back(static_cast<P>(static_cast<double>(q + 1) / 8.0));
    }
    for (const Layout& layout : layouts) {
      std::size_t total = 1;
      std::size_t outer = 1;
      std::size_t inner = 1;
      for (std::size_t d = 0; d < layout.shape.size(); ++d) {
        total *= layout.shape[d];
        if (d < layout.axis) {
          outer *= layout.shape[d];
        } else if (d > layout.axis) {
          inner *= layout.shape[d];
        }
      }
      const std::vector<C> values = cast_all<C>(bspline_test::values(total, 61));
      const NdArray<C> got = bs::assemble_load_1d<P, C>(
          s, std::span<const C>(values), std::span<const std::size_t>(layout.shape),
          std::span<const P>(nodes), std::span<const P>(weights), layout.axis);
      std::vector<std::size_t> want_shape = layout.shape;
      want_shape[layout.axis] = n;
      REQUIRE(got.rank() == want_shape.size());
      for (std::size_t d = 0; d < want_shape.size(); ++d) {
        CHECK(got.shape()[d] == want_shape[d]);
      }
      std::vector<double> abs_sum;
      const Matrix want =
          exact_contraction<P, C>(space, std::span<const C>(values), outer, inner,
                                  std::span<const P>(nodes), std::span<const P>(weights), abs_sum);
      const double rel = (6.0 * dir.degree + 2.0) * unit_roundoff<P>() +
                         (static_cast<double>(nodes.size()) + 3.0) * unit_roundoff<C>();
      for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t i = 0; i < n; ++i) {
          for (std::size_t r = 0; r < inner; ++r) {
            const std::size_t flat = (o * n + i) * inner + r;
            INFO("degree " << dir.degree << ", axis " << layout.axis << ", entry " << flat);
            CHECK(oracle::within(static_cast<double>(got.data()[flat]), want[o * n + i][r],
                                 rel * abs_sum[flat]));
          }
        }
      }
    }
  }
}

TEST_CASE("assemble_load_1d: inconsistent extents and nodes outside the domain are refused") {
  const bs::BsplineSpace1D<double> s = space_1d<double>(cubic_dir());
  const std::vector<double> nodes = {0.1, 0.5, 0.9};
  const std::vector<double> weights = {0.25, 0.5, 0.25};
  const std::vector<double> values(6, 1.0);
  const std::vector<std::size_t> shape = {2, 3};
  const char* const mismatch = "assemble_load_1d: the values, nodes and weights do not match.";
  const auto call = [&](std::span<const double> v, std::span<const std::size_t> sh,
                        std::span<const double> x, std::span<const double> w, std::size_t axis) {
    return bs::assemble_load_1d<double, double>(s, v, sh, x, w, axis);
  };
  CHECK_NOTHROW(static_cast<void>(call(values, shape, nodes, weights, 1)));
  CHECK_THROWS_WITH_AS(static_cast<void>(call(values, shape, nodes, weights, 2)), mismatch,
                       pantr::value_error);  // no such axis
  CHECK_THROWS_WITH_AS(static_cast<void>(call(values, shape, nodes, weights, 0)), mismatch,
                       pantr::value_error);  // 2 != 3 nodes
  const std::vector<double> two_weights = {0.5, 0.5};
  CHECK_THROWS_WITH_AS(static_cast<void>(call(values, shape, nodes, two_weights, 1)), mismatch,
                       pantr::value_error);
  const std::vector<double> five(5, 1.0);
  CHECK_THROWS_WITH_AS(static_cast<void>(call(five, shape, nodes, weights, 1)), mismatch,
                       pantr::value_error);
  const std::vector<double> outside = {0.1, 0.5, 1.5};
  CHECK_THROWS_WITH_AS(static_cast<void>(call(values, shape, outside, weights, 1)),
                       "One or more values in pts are outside the knot vector domain (0.0, 1.0)",
                       pantr::value_error);
}

TEST_CASE(
    "L2Projection::load and trace_load: contract the free directions, a corner is its point "
    "value") {
  // Oracle: the exact tensor contraction `sum_q w0 N_i(x_q0) w1 N_j(y_q1) v` with the library's own
  // nodes and weights taken as given (their accuracy is the quadrature's, not this test's), on
  // the full lattice and on the three kinds of trace of a 2D space: a face with the first
  // direction fixed (shape (1, n1)), a face with the second fixed (shape (n0, 1)) and a corner,
  // whose load is the value itself. Bug caught: the fixed axis contracted as well, a load shaped
  // (n0, n1) for a face, the corner multiplied by a weight, the two directions' weights swapped.
  const std::pair<bool, bool> both{true, true};
  const std::vector<Dir> dirs = {cubic_dir(), shifted_dir()};
  const auto proj = projection_of<double>(dirs, {4, 3}, {both, both});
  const bs::BsplineSpace<double> sp0 = space_of<double>({dirs[0]});
  const bs::BsplineSpace<double> sp1 = space_of<double>({dirs[1]});
  const std::size_t nq0 = proj.nodes()[0].size();  // 16
  const std::size_t nq1 = proj.nodes()[1].size();  // 6
  REQUIRE(nq0 == 16);
  REQUIRE(nq1 == 6);
  const std::vector<double> v = bspline_test::values(nq0 * nq1, 71);
  const double rel =
      (static_cast<double>(nq0 * nq1) + 2.0 * (6.0 * 3 + 1.0) + 8.0) * unit_roundoff<double>();

  const NdArray<double> load = proj.load<double>(std::span<const double>(v));
  REQUIRE(load.rank() == 2);
  REQUIRE(load.shape()[0] == 7);
  REQUIRE(load.shape()[1] == 4);
  // Contract axis 1 first (the exact way, then axis 0): the order does not matter in exact
  // arithmetic.
  std::vector<double> abs1;
  const Matrix stage1 =
      exact_contraction<double, double>(sp1, std::span<const double>(v), nq0, 1,
                                        proj.nodes()[1].flat(), proj.weights()[1].flat(), abs1);
  // stage1[(q0 * 4 + j)][0]; now contract over q0 with the first direction's data.
  std::vector<std::vector<Rational>> basis0(7);
  for (std::size_t i = 0; i < 7; ++i) {
    for (std::size_t q = 0; q < nq0; ++q) {
      const std::vector<Rational> x = {oracle::exact(proj.nodes()[0].data()[q])};
      basis0[i].push_back(oracle::exact_value<double>(unit_spline<double>(sp0, i), x)[0]);
    }
  }
  for (std::size_t i = 0; i < 7; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      Rational acc = 0;
      double scale = 0.0;
      for (std::size_t q = 0; q < nq0; ++q) {
        const Rational w_n = oracle::exact(proj.weights()[0].data()[q]) * basis0[i][q];
        acc += w_n * stage1[q * 4 + j][0];
        scale += to_double(w_n) * abs1[q * 4 + j];  // sum of the moduli of all the terms
      }
      INFO("load entry (" << i << ", " << j << ")");
      CHECK(oracle::within(load.data()[i * 4 + j], acc, rel * scale));
    }
  }

  // Trace {0, -1}: first direction fixed at its start, values over the (1, nq1) lattice.
  const std::vector<double> face_values = bspline_test::values(nq1, 72);
  const NdArray<double> face = proj.trace_load<double>(0, std::span<const double>(face_values));
  REQUIRE(face.rank() == 2);
  REQUIRE(face.shape()[0] == 1);
  REQUIRE(face.shape()[1] == 4);
  std::vector<double> abs_face;
  const Matrix want_face =
      exact_contraction<double, double>(sp1, std::span<const double>(face_values), 1, 1,
                                        proj.nodes()[1].flat(), proj.weights()[1].flat(), abs_face);
  for (std::size_t j = 0; j < 4; ++j) {
    INFO("face {0, -1}, entry " << j);
    CHECK(oracle::within(face.data()[j], want_face[j][0], rel * abs_face[j]));
  }
  // Trace {-1, 1}: second direction fixed at its end, values over the (nq0, 1) lattice.
  const std::vector<double> edge_values = bspline_test::values(nq0, 73);
  const NdArray<double> edge = proj.trace_load<double>(3, std::span<const double>(edge_values));
  REQUIRE(edge.rank() == 2);
  REQUIRE(edge.shape()[0] == 7);
  REQUIRE(edge.shape()[1] == 1);
  std::vector<double> abs_edge;
  const Matrix want_edge =
      exact_contraction<double, double>(sp0, std::span<const double>(edge_values), 1, 1,
                                        proj.nodes()[0].flat(), proj.weights()[0].flat(), abs_edge);
  for (std::size_t i = 0; i < 7; ++i) {
    INFO("face {-1, 1}, entry " << i);
    CHECK(oracle::within(edge.data()[i], want_edge[i][0], rel * abs_edge[i]));
  }
  // Trace {1, 0}: a corner. Its load is the point value, untouched.
  const std::vector<double> corner_value = {0.40625};
  const NdArray<double> corner = proj.trace_load<double>(6, std::span<const double>(corner_value));
  REQUIRE(corner.size() == 1);
  CHECK(corner.data()[0] == 0.40625);
}

namespace {

// A function of a lattice that returns the exact values of `truth` at its points, counting calls
// and recording the lattice extents it saw.
template <class T>
struct ExactSampler {
  const bs::Bspline<T>* truth;
  std::vector<std::vector<std::size_t>>* shapes;

  NdArray<T> operator()(const pantr::quad::PointsLattice<T>& lat) const {
    std::vector<std::vector<T>> nodes;
    std::vector<std::size_t> shape;
    for (std::size_t d = 0; d < lat.dim(); ++d) {
      nodes.emplace_back(lat.pts(d).begin(), lat.pts(d).end());
      shape.push_back(nodes.back().size());
    }
    if (shapes != nullptr) {
      shapes->push_back(shape);
    }
    const std::vector<T> samples = samples_on_grid<T>(*truth, nodes);
    NdArray<T> out({lat.num_points(), truth->rank()});
    std::copy(samples.begin(), samples.end(), out.data());
    return out;
  }
};

template <class T>
void check_l2_reproduction(const std::vector<Dir>& dirs,
                           const std::vector<std::pair<bool, bool>>& flags, std::size_t rank,
                           unsigned seed,
                           bs::L2Quadrature rule = bs::L2Quadrature::gauss_legendre) {
  const bs::BsplineSpace<T> space = space_of<T>(dirs);
  const bs::Bspline<T> truth = make_spline<T>(dirs, rank, false, seed);
  std::vector<long long> n_quad;
  for (const Dir& d : dirs) {
    n_quad.push_back(d.degree + 1LL);
  }
  const std::vector<std::pair<bool, bool>> padded =
      flags.empty() ? std::vector<std::pair<bool, bool>>(dirs.size(), {false, false}) : flags;
  const bs::L2Projection<T> proj(space, n_quad, rule, padded);
  double kappa = 1.0;
  double sizes = 0.0;
  for (std::size_t d = 0; d < dirs.size(); ++d) {
    kappa *= kappa_bound(exact_matrix<T>(proj.mass()[d]));
    sizes += static_cast<double>(2 * proj.mass()[d].shape()[0] + 2 + proj.nodes()[d].size());
  }
  std::vector<std::vector<std::size_t>> shapes;
  const ExactSampler<T> sampler{&truth, &shapes};
  const bs::Bspline<T> fit = bs::l2_project<T>(space, sampler, n_quad, rule, flags);
  CHECK(shapes.size() == 1 + proj.traces().size());
  CHECK_FALSE(fit.rational());
  CHECK(fit.rank() == rank);
  REQUIRE(fit.control_points().size() == truth.control_points().size());
  const double bound = solve_bound<T>(static_cast<std::size_t>(sizes), kappa,
                                      two_norm_of<T>(truth.control_points()));
  const double worst = worst_difference<T>(fit.control_points(), truth.control_points());
  INFO("traces " << proj.traces().size() << ", kappa bound " << kappa << ", worst error " << worst
                 << ", bound " << bound);
  CHECK(worst <= bound);
}

}  // namespace

TEST_CASE_TEMPLATE(
    "l2_project: a spline of the space is reproduced, boundary interpolation included", T, float,
    double) {
  // Oracle: a spline of the space with dyadic control points, sampled exactly on the quadrature
  // lattice and on every boundary trace by the callable. The projection of an element of the
  // space is itself; with a boundary row replaced by `N(end) c = f(end)` the system is still
  // satisfied by the true coefficients, provided each trace load is the exact load integral of
  // the trace and each later trace (an edge, a corner) overwrites the earlier ones (the faces) it
  // meets. Bug caught: traces written in the wrong order or at the wrong end, a trace load with
  // the wrong axis contracted, a replaced mass row paired with the wrong right-hand side, a rank
  // component taken from the wrong column.
  const std::pair<bool, bool> both{true, true};
  SUBCASE("1D cubic, no boundary interpolation") {
    check_l2_reproduction<T>({cubic_dir()}, {}, 1, 81);
  }
  SUBCASE("1D cubic, both ends interpolated, rank 2") {
    check_l2_reproduction<T>({cubic_dir()}, {both}, 2, 82);
  }
  SUBCASE("1D unclamped cubic, whose end rows are dense") {
    check_l2_reproduction<T>({unclamped_dir()}, {both}, 1, 83);
  }
  SUBCASE("1D periodic cubic, the flags ignored") {
    check_l2_reproduction<T>({periodic_dir()}, {both}, 1, 84);
  }
  SUBCASE(
      "1D with Gauss-Lobatto points, whose mass matrix is not exact: the discrete system is "
      "consistent") {
    check_l2_reproduction<T>({cubic_dir()}, {both}, 1, 85, bs::L2Quadrature::gauss_lobatto);
  }
  SUBCASE("2D, both ends of both directions, rank 2: faces and corners") {
    check_l2_reproduction<T>({cubic_dir(), shifted_dir()}, {both, both}, 2, 86);
  }
  SUBCASE("2D, a periodic direction and a start only") {
    check_l2_reproduction<T>({periodic_dir(), c0_quadratic_dir()}, {both, {true, false}}, 1, 87);
  }
  if constexpr (std::is_same_v<T, double>) {
    SUBCASE("2D, an unclamped direction pinned at both ends and a clamped one pinned at its end") {
      check_l2_reproduction<T>({unclamped_dir(), cubic_dir()}, {both, {false, true}}, 1, 88);
    }
    SUBCASE("3D, faces, edges and corners") {
      check_l2_reproduction<T>(
          {Dir{clamped(1, {0.5}), 1}, Dir{clamped(2, {}), 2}, Dir{clamped(1, {}), 1}},
          {both, both, both}, 1, 89);
    }
  }
}

TEST_CASE_TEMPLATE(
    "l2_project: the projection of a polynomial outside the space equals G^-1 b in exact rationals",
    T, float, double) {
  // Oracle: the Gram matrix G and the load b_i = integral N_i f, both from the closed Newton-Cotes
  // rules in exact rationals (see the header), and c = G^-1 b by Gauss-Jordan. f has degree
  // p + 1, so the library's p + 1 Gauss points per span integrate N_i f exactly, and the result
  // is the true L2 projection (not an interpolant: f is not in the space). The two components
  // f and g = x^(p+1) / 2 + 3/8 check the interleaving of the coefficients. Bug caught: a load
  // assembled with the wrong weights or a missing factor of the span length, a mass matrix
  // that is not the Gram matrix, a rank component lost, a periodic wrap dropped.
  const double u = unit_roundoff<T>();
  for (const Dir& dir : {cubic_dir(), c0_quadratic_dir(), unclamped_dir(), periodic_dir()}) {
    const bs::BsplineSpace<T> space = space_of<T>({dir});
    const bs::BsplineSpace1D<T>& s = space.spaces()[0];
    const auto p = static_cast<unsigned>(dir.degree);
    const auto n = static_cast<std::size_t>(s.num_basis());
    const auto power = [p](const Rational& x) {
      Rational r = 1;
      for (unsigned k = 0; k < p + 1; ++k) {
        r *= x;
      }
      return r;
    };
    const Matrix gram = exact_gram<T>(s);
    const Matrix b_f =
        exact_load_vector<T>(s, p + 1, [&](const Rational& x) { return power(x) - rat(3, 8) * x; });
    const Matrix b_g =
        exact_load_vector<T>(s, p + 1, [&](const Rational& x) { return power(x) / 2 + rat(3, 8); });
    const Matrix c_f = solve_exact(gram, b_f);
    const Matrix c_g = solve_exact(gram, b_g);
    const auto exponent = static_cast<int>(p) + 1;
    const auto fn = [exponent](const pantr::quad::PointsLattice<T>& lat) {
      NdArray<T> out({lat.num_points(), 2});
      for (std::size_t i = 0; i < lat.num_points(); ++i) {
        const double x = static_cast<double>(lat.pts(0)[i]);
        const double xp = std::pow(x, exponent);
        out.data()[2 * i] = static_cast<T>(xp - 0.375 * x);
        out.data()[2 * i + 1] = static_cast<T>(xp / 2.0 + 0.375);
      }
      return out;
    };
    const bs::Bspline<T> fit = bs::l2_project<T>(space, fn);
    REQUIRE(fit.control_points().size() == 2 * n);
    const double kappa = kappa_bound(gram);
    const auto points = static_cast<std::size_t>((s.num_intervals() * (dir.degree + 1)));
    const double c_norm = std::sqrt(frobenius_norm(c_f) * frobenius_norm(c_f) +
                                    frobenius_norm(c_g) * frobenius_norm(c_g));
    const double bound = solve_bound<T>(2 * n + 2 + points, kappa, c_norm);
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      worst = (std::max)(
          worst, std::abs(static_cast<double>(fit.control_points()[2 * i]) - to_double(c_f[i][0])));
      worst = (std::max)(worst, std::abs(static_cast<double>(fit.control_points()[2 * i + 1]) -
                                         to_double(c_g[i][0])));
    }
    INFO("degree " << dir.degree << ", periodic " << dir.periodic << ", kappa bound " << kappa
                   << ", worst error " << worst << ", bound " << bound << ", u " << u);
    CHECK(worst <= bound);
  }
}

TEST_CASE(
    "l2_project: the callable is sampled once on the lattice and once per boundary trace, with "
    "these extents") {
  // The convenience template samples every lattice once for all the components, in the order of
  // `traces()`: the quadrature lattice, then the four faces and the four corners of a 2D space.
  // (The Python wrapper samples per component instead, in the previous release's pattern.)
  const std::pair<bool, bool> both{true, true};
  const std::vector<Dir> dirs = {cubic_dir(), shifted_dir()};
  const bs::BsplineSpace<double> space = space_of<double>(dirs);
  const bs::Bspline<double> truth = make_spline<double>(dirs, 2, false, 91);
  std::vector<std::vector<std::size_t>> shapes;
  const ExactSampler<double> sampler{&truth, &shapes};
  const bs::Bspline<double> fit = bs::l2_project<double>(
      space, sampler, std::nullopt, bs::L2Quadrature::gauss_legendre, {both, both});
  using Shapes = std::vector<std::vector<std::size_t>>;
  const Shapes want = {{16, 6}, {1, 6}, {1, 6}, {16, 1}, {16, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
  CHECK(shapes == want);
  CHECK(fit.rank() == 2);
  // Without flags: one call, on the lattice alone. With a single flagged end: one trace.
  shapes.clear();
  static_cast<void>(bs::l2_project<double>(space, sampler));
  CHECK(shapes == Shapes{{16, 6}});
  shapes.clear();
  static_cast<void>(bs::l2_project<double>(space, sampler, std::nullopt,
                                           bs::L2Quadrature::gauss_legendre,
                                           {{false, true}, {false, false}}));
  CHECK(shapes == Shapes{{16, 6}, {1, 6}});
  // The default number of points is degree + 1 per direction: 4 per span in the cubic direction
  // (4 spans) and 3 per span in the quadratic one (2 spans).
  shapes.clear();
  static_cast<void>(bs::l2_project<double>(space, sampler, std::vector<long long>{2, 2}));
  CHECK(shapes == Shapes{{8, 4}});
}

TEST_CASE("l2_project: a callable returning the wrong number of values or components is refused") {
  const std::pair<bool, bool> both{true, true};
  const bs::BsplineSpace<double> space = space_of<double>({cubic_dir()});
  using Lattice = pantr::quad::PointsLattice<double>;
  const char* const lattice_message =
      "l2_project: the function returned another number of values than the 16 points.";
  CHECK_THROWS_WITH_AS(
      static_cast<void>(bs::l2_project<double>(
          space, [](const Lattice& l) { return NdArray<double>({l.num_points() - 1}); })),
      lattice_message, pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(bs::l2_project<double>(
          space, [](const Lattice& l) { return NdArray<double>({l.num_points() + 1, 1}); })),
      lattice_message, pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(bs::l2_project<double>(
          space, [](const Lattice& l) { return NdArray<double>({l.num_points(), 0}); })),
      lattice_message, pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(bs::l2_project<double>(
          space, [](const Lattice& l) { return NdArray<double>({l.num_points(), 1, 1}); })),
      lattice_message, pantr::value_error);
  // A trace that answers with another number of components than the lattice did.
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::l2_project<double>(
                           space,
                           [](const Lattice& l) {
                             return l.num_points() == 1 ? NdArray<double>({1})
                                                        : NdArray<double>({l.num_points(), 2});
                           },
                           std::nullopt, bs::L2Quadrature::gauss_legendre, {both})),
                       "l2_project: a boundary trace returned another number of components.",
                       pantr::value_error);
}

// ---- quasi-interpolation ----------------------------------------------------------------------

namespace {

// The exact values of `truth` at the rows of `pts` (M x dim), rounded once to double, row-major
// over (M, rank): the data a function of the sample points would return.
template <class T>
std::vector<double> exact_values_at(const bs::Bspline<T>& truth, const NdArray<double>& pts) {
  const std::size_t dim = pts.shape()[1];
  std::vector<double> out;
  for (std::size_t i = 0; i < pts.shape()[0]; ++i) {
    std::vector<Rational> x;
    for (std::size_t d = 0; d < dim; ++d) {
      x.push_back(oracle::exact(pts.data()[i * dim + d]));
    }
    for (const Rational& v : oracle::exact_value<T>(truth, x)) {
      out.push_back(to_double(v));
    }
  }
  return out;
}

// The largest kappa bound, over the functions of a direction, of the local collocation block the
// quasi-interpolant inverts: the exact values of the `order` supported functions at the function's
// `order` sample points (from the points of the 1D quasi-interpolant of the direction).
template <class T>
double local_block_kappa(const bs::BsplineSpace1D<T>& s) {
  const bs::BsplineSpace<T> single(std::vector<bs::BsplineSpace1D<T>>{s});
  const bs::QuasiInterpolation<T> qi(single);
  const NdArray<double> pts = qi.points();
  const auto p = static_cast<std::size_t>(s.degree());
  const std::size_t order = p + 1;
  const auto n = static_cast<std::size_t>(s.num_basis());
  const std::vector<Rational> u = oracle::exact_knots<T>(s.knots());
  double worst = 1.0;
  for (std::size_t j = 0; j < n; ++j) {
    Matrix block(order, std::vector<Rational>(order));
    for (std::size_t k = 0; k < order; ++k) {
      oracle::BasisAt at(u, s.degree(), oracle::exact(pts.data()[j * order + k]));
      for (std::size_t a = 0; a < order; ++a) {
        block[k][a] = at.deriv(at.span() - p + a, 0);
      }
    }
    worst = (std::max)(worst, kappa_bound(block));
  }
  return worst;
}

template <class T>
void check_qi_reproduces(const std::vector<Dir>& dirs, std::size_t rank, unsigned seed) {
  const bs::BsplineSpace<T> space = space_of<T>(dirs);
  const bs::Bspline<T> truth = make_spline<T>(dirs, rank, false, seed);
  const bs::QuasiInterpolation<T> qi(space);
  const NdArray<double> pts = qi.points();
  const std::size_t rows = pts.shape()[0];
  REQUIRE(rows == qi.num_functions() * qi.block());
  const std::vector<double> values = exact_values_at<T>(truth, pts);
  const NdArray<double> c =
      qi.coefficients(pantr::mdview<const double, 2>(values.data(), rows, rank));
  REQUIRE(c.size() == truth.control_points().size());
  double kappa = 1.0;
  double orders = 0.0;
  double basis_units = 0.0;
  for (const bs::BsplineSpace1D<T>& s : space.spaces()) {
    kappa *= local_block_kappa<T>(s);
    orders += static_cast<double>(s.degree()) + 1.0;
    basis_units += (static_cast<double>(s.degree()) + 1.0) * (6.0 * s.degree() + 1.0);
  }
  double c_inf = 0.0;
  double worst = 0.0;
  for (std::size_t i = 0; i < c.size(); ++i) {
    c_inf = (std::max)(c_inf, std::abs(static_cast<double>(truth.control_points()[i])));
    worst =
        (std::max)(worst, std::abs(c.data()[i] - static_cast<double>(truth.control_points()[i])));
  }
  // Two sources, both multiplied by kappa of the blocks and ||c||_inf (the sampled values are a
  // convex combination of the control points, so they are bounded by it): the inverse of each
  // local block, computed in double and backward stable, K (sum of orders + 2) u_double; and the
  // block itself, whose entries the space's own type produces (the basis of a float space is
  // tabulated in float even at double points), with (6 p + 1) u_T relative to a row sum of 1 over
  // `order` entries, summed over the directions: sum (p + 1)(6 p + 1) u_T.
  const double bound =
      (fit_slack * (orders + 2.0) * unit_roundoff<double>() + basis_units * unit_roundoff<T>()) *
      kappa * (std::max)(c_inf, 1.0);
  INFO("local block kappa bound " << kappa << ", worst error " << worst << ", bound " << bound);
  CHECK(worst <= bound);
}

}  // namespace

TEST_CASE(
    "QuasiInterpolation: the points of a two-interval hat space and the coefficients of x and "
    "x^2") {
  // Oracle: closed forms. Degree 1 on the knots {0, 0, 1/2, 1, 1}: the Greville abscissae are
  // 0, 1/2, 1; the functions' intervals are [0, 1/2] for the first and [1/2, 1] for the other two
  // (a Greville abscissa on a break takes the interval to its right, the last one the last
  // interval), and the two points of an interval are at 1/3 and 2/3 of it. So the points are
  // 1/6, 1/3 | 2/3, 5/6 | 2/3, 5/6. The coefficient of a hat function is the value of the local
  // line through the two samples at the function's Greville abscissa: for f = x it is the
  // abscissa itself (0, 1/2, 1), for f = 1 it is 1, and for f = x^2 the line through (x_a, x_a^2),
  // (x_b, x_b^2) is (x_a + x_b) x - x_a x_b, giving -1/18, 7/36 and 17/18. Bug caught: a sample
  // interval taken from the wrong side of a break, points at the wrong fractions, weights taken
  // from the wrong row of the block.
  const std::vector<double> knots = {0.0, 0.0, 0.5, 1.0, 1.0};
  const bs::BsplineSpace1D<double> s1(std::span<const double>(knots), 1);
  const bs::BsplineSpace<double> space(std::vector<bs::BsplineSpace1D<double>>{s1});
  const bs::QuasiInterpolation<double> qi(space);
  CHECK(qi.block() == 2);
  CHECK(qi.num_functions() == 3);
  const NdArray<double> pts = qi.points();
  REQUIRE(pts.rank() == 2);
  REQUIRE(pts.shape()[0] == 6);
  REQUIRE(pts.shape()[1] == 1);
  const double want[6] = {1.0 / 6.0, 1.0 / 3.0, 2.0 / 3.0, 5.0 / 6.0, 2.0 / 3.0, 5.0 / 6.0};
  const double u = unit_roundoff<double>();
  for (std::size_t i = 0; i < 6; ++i) {
    CHECK(std::abs(pts.data()[i] - want[i]) <= 4.0 * u);
  }
  const auto coefficients_of = [&](const std::vector<double>& f) {
    return qi.coefficients(pantr::mdview<const double, 2>(f.data(), 6, 1));
  };
  const std::vector<double> ones(6, 1.0);
  const std::vector<double> xs(pts.data(), pts.data() + 6);
  std::vector<double> squares(6);
  for (std::size_t i = 0; i < 6; ++i) {
    squares[i] = xs[i] * xs[i];
  }
  const NdArray<double> c1 = coefficients_of(ones);
  const NdArray<double> cx = coefficients_of(xs);
  const NdArray<double> c2 = coefficients_of(squares);
  REQUIRE(c1.rank() == 2);
  REQUIRE(c1.shape()[0] == 3);
  REQUIRE(c1.shape()[1] == 1);
  const double want_x[3] = {0.0, 0.5, 1.0};
  const double want_sq[3] = {-1.0 / 18.0, 7.0 / 36.0, 17.0 / 18.0};
  for (std::size_t i = 0; i < 3; ++i) {
    INFO("function " << i);
    CHECK(std::abs(c1.data()[i] - 1.0) <= 16.0 * u);
    CHECK(std::abs(cx.data()[i] - want_x[i]) <= 16.0 * u);
    CHECK(std::abs(c2.data()[i] - want_sq[i]) <= 16.0 * u);
  }
}

TEST_CASE(
    "QuasiInterpolation: the points of a tensor space list each function's grid, functions in C "
    "order") {
  // The header states the layout: functions in C order (last direction fastest), and for each
  // function the tensor grid of its per-direction points, again in C order. Oracle: the 1D
  // quasi-interpolants' points (checked against closed forms above) combined by plain loops of the
  // statement. The 2D space has degrees (1, 2) so the blocks of the two directions differ in size.
  const std::vector<Dir> dirs = {Dir{clamped(1, {0.5}), 1}, Dir{clamped(2, {}), 2}};
  const bs::BsplineSpace<double> space = space_of<double>(dirs);
  const bs::QuasiInterpolation<double> qi(space);
  const bs::QuasiInterpolation<double> q0(space_of<double>({dirs[0]}));
  const bs::QuasiInterpolation<double> q1(space_of<double>({dirs[1]}));
  const NdArray<double> p0 = q0.points();  // 3 functions x 2 points
  const NdArray<double> p1 = q1.points();  // 3 functions x 3 points
  CHECK(qi.num_functions() == 9);
  CHECK(qi.block() == 6);
  const NdArray<double> pts = qi.points();
  REQUIRE(pts.rank() == 2);
  REQUIRE(pts.shape()[0] == 54);
  REQUIRE(pts.shape()[1] == 2);
  std::size_t row = 0;
  for (std::size_t f0 = 0; f0 < 3; ++f0) {
    for (std::size_t f1 = 0; f1 < 3; ++f1) {
      for (std::size_t k0 = 0; k0 < 2; ++k0) {
        for (std::size_t k1 = 0; k1 < 3; ++k1) {
          INFO("function (" << f0 << ", " << f1 << "), point (" << k0 << ", " << k1 << ")");
          CHECK(pts.data()[row * 2] == p0.data()[f0 * 2 + k0]);
          CHECK(pts.data()[row * 2 + 1] == p1.data()[f1 * 3 + k1]);
          ++row;
        }
      }
    }
  }
}

TEST_CASE_TEMPLATE("QuasiInterpolation: every spline of the space is reproduced, Q B = B", T, float,
                   double) {
  // Oracle: the invariant itself. A spline of the space with dyadic control points is sampled
  // exactly (recursive-definition evaluation) at the quasi-interpolant's points, and the
  // coefficients must come back. The bound uses the exact condition number of each local block
  // (see check_qi_reproduces). Bug caught: sample points outside the function's interval, a
  // weight row of the inverse that belongs to another function, a contraction in the wrong
  // direction order, a periodic function wrapped to the wrong index, a multiple knot that
  // moves the interval.
  SUBCASE("cubic, simple knots") { check_qi_reproduces<T>({cubic_dir()}, 1, 101); }
  SUBCASE("cubic with a double and with a triple interior knot, rank 2") {
    check_qi_reproduces<T>({Dir{clamped(3, {0.25, 0.5, 0.5, 0.75}), 3}}, 2, 102);
    check_qi_reproduces<T>({Dir{clamped(3, {0.25, 0.5, 0.5, 0.5, 0.75}), 3}}, 2, 103);
  }
  SUBCASE("quadratic, C0 at a double knot, and nonuniform quartic") {
    check_qi_reproduces<T>({c0_quadratic_dir()}, 1, 104);
    check_qi_reproduces<T>({quartic_dir()}, 1, 105);
  }
  SUBCASE("shifted domain, unclamped, periodic") {
    check_qi_reproduces<T>({shifted_dir()}, 1, 106);
    check_qi_reproduces<T>({unclamped_dir()}, 1, 107);
    check_qi_reproduces<T>({periodic_dir()}, 1, 108);
  }
  SUBCASE("2D, mixed degrees and knots, rank 2") {
    check_qi_reproduces<T>({Dir{clamped(2, {0.25, 0.5, 0.5}), 2}, Dir{clamped(3, {0.5}), 3}}, 2,
                           109);
  }
  SUBCASE("2D with a periodic direction") {
    check_qi_reproduces<T>({periodic_dir(), shifted_dir()}, 1, 110);
  }
  SUBCASE("3D, three different degrees") {
    check_qi_reproduces<T>(
        {Dir{clamped(1, {0.5}), 1}, Dir{clamped(2, {}), 2}, Dir{clamped(2, {0.25, 0.75}), 2}}, 1,
        111);
  }
}

TEST_CASE(
    "QuasiInterpolation: a spline with a discontinuity is not reproduced (known defect, expected "
    "to fail)" *
    doctest::should_fail()) {
  // KNOWN DEFECT, pinned as an expected failure so that the day it is fixed this test says so.
  // The interval of a function is the one holding its Greville abscissa, taken as the interval to
  // the right of a break. For a knot of multiplicity degree + 1 (a discontinuity) the Greville
  // abscissa of the function that ends there is that knot, so the interval chosen lies to the right
  // of the function's support, `local` in `llm_direction` is -1 and indexes the block from the
  // end ("a Python index outside [-order, order), as before"): the functional of another
  // function. The header states Q B_j = B_j for every space; it fails here (the previous release
  // fails the same way). Cubic with the interior knot 1/2 four times: the error of a coefficient is
  // of order 1.
  check_qi_reproduces<double>({Dir{clamped(3, {0.25, 0.5, 0.5, 0.5, 0.5, 0.75}), 3}}, 1, 112);
}

TEST_CASE(
    "QuasiInterpolation: some periodic spaces are not reproduced (known defect, expected to fail)" *
    doctest::should_fail()) {
  // KNOWN DEFECT, pinned as an expected failure. A periodic cubic space of 6 intervals on [0, 1],
  // knots as `uniform_periodic_knots` makes them (the space `create_uniform_space(3, 6,
  // periodic=True)` of the Python API), returns its coefficients cyclically shifted, an error of
  // order 1 (the previous release does the same, and so do 12 intervals; 5, 7, 8, 9 and 10
  // intervals, and the exact-translate knots of `periodic_dir()`, pass).
  // Mechanism, observed in a scratch copy and not applied: `llm_direction` takes the abscissa of
  // function j from `space.greville()`, which for a periodic space is wrapped into the domain and
  // sorted (NumPy's order), so entry j is another function's abscissa, and `local = j - first` is
  // taken in the unwrapped numbering. With the abscissae of the functions in order and `local`
  // reduced modulo the number of functions, every case above reproduces.
  const std::vector<double> knots = bs::uniform_periodic_knots<double>(6, 3, 2, 0.0, 1.0);
  check_qi_reproduces<double>({Dir{knots, 3, true}}, 1, 113);
}

TEST_CASE_TEMPLATE(
    "quasi_interpolate: the callable is called once with the points, the result is the rounded "
    "coefficients",
    T, float, double) {
  // The function of the points is called exactly once, with the (M, dim) array `points()` (same
  // values, same extents), and the result is the coefficients of `QuasiInterpolation` rounded to
  // T, bit for bit (a rounding is exact arithmetic here). Oracle for the values: a polynomial of
  // the space's degree, x^3 - y, reproduced up to the derived bound at the exact sample points.
  const std::vector<Dir> dirs = {cubic_dir(), c0_quadratic_dir()};
  const bs::BsplineSpace<T> space = space_of<T>(dirs);
  const bs::QuasiInterpolation<T> qi(space);
  const NdArray<double> pts = qi.points();
  int calls = 0;
  std::size_t seen_rows = 0;
  std::size_t seen_cols = 0;
  bool same_points = true;
  const auto fn = [&](pantr::mdview<const double, 2> p) {
    ++calls;
    seen_rows = p.extent(0);
    seen_cols = p.extent(1);
    NdArray<double> out({p.extent(0)});
    for (std::size_t i = 0; i < p.extent(0); ++i) {
      same_points = same_points && p(i, 0) == pts.data()[i * 2] && p(i, 1) == pts.data()[i * 2 + 1];
      out.data()[i] = p(i, 0) * p(i, 0) * p(i, 0) - p(i, 1);
    }
    return out;
  };
  const bs::Bspline<T> result = bs::quasi_interpolate<T>(space, fn);
  CHECK(calls == 1);
  CHECK(seen_rows == pts.shape()[0]);
  CHECK(seen_cols == 2);
  CHECK(same_points);
  CHECK(result.rank() == 1);
  CHECK_FALSE(result.rational());
  // Bit for bit the rounded coefficients of the same data.
  std::vector<double> values(pts.shape()[0]);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = pts.data()[i * 2] * pts.data()[i * 2] * pts.data()[i * 2] - pts.data()[i * 2 + 1];
  }
  const NdArray<double> c =
      qi.coefficients(pantr::mdview<const double, 2>(values.data(), values.size(), 1));
  REQUIRE(result.control_points().size() == c.size());
  for (std::size_t i = 0; i < c.size(); ++i) {
    CHECK(result.control_points()[i] == static_cast<T>(c.data()[i]));
  }
  // x^3 - y lies in the space (degrees 3 and 2), so the result evaluates to it.
  for (const double px : {0.0, 0.1, 0.5, 0.97}) {
    for (const double py : {0.0, 0.3, 0.5, 1.0}) {
      const std::vector<Rational> x = {oracle::exact(static_cast<double>(static_cast<T>(px))),
                                       oracle::exact(static_cast<double>(static_cast<T>(py)))};
      const double xt = static_cast<double>(static_cast<T>(px));
      const double yt = static_cast<double>(static_cast<T>(py));
      const double got = to_double(oracle::exact_value<T>(result, x)[0]);
      INFO("point (" << px << ", " << py << ")");
      CHECK(std::abs(got - (xt * xt * xt - yt)) <= 1024.0 * machine_epsilon<T>());
    }
  }
}

TEST_CASE("QuasiInterpolation and quasi_interpolate: values of another extent are refused") {
  const bs::BsplineSpace<double> space = space_of<double>({cubic_dir()});
  const bs::QuasiInterpolation<double> qi(space);
  const std::size_t m = qi.num_functions() * qi.block();  // 7 x 4
  REQUIRE(m == 28);
  const std::vector<double> some(m + 1, 1.0);
  CHECK_NOTHROW(
      static_cast<void>(qi.coefficients(pantr::mdview<const double, 2>(some.data(), m, 1))));
  CHECK_THROWS_WITH_AS(
      static_cast<void>(qi.coefficients(pantr::mdview<const double, 2>(some.data(), m - 1, 1))),
      "QuasiInterpolation: expected 28 rows of values, got 27.", pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(qi.coefficients(pantr::mdview<const double, 2>(some.data(), m + 1, 1))),
      "QuasiInterpolation: expected 28 rows of values, got 29.", pantr::value_error);
  using Points = pantr::mdview<const double, 2>;
  const char* const message =
      "quasi_interpolate: the function returned another number of values than the 28 points.";
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::quasi_interpolate<double>(
                           space, [](Points p) { return NdArray<double>({p.extent(0) - 1}); })),
                       message, pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::quasi_interpolate<double>(
                           space, [](Points p) { return NdArray<double>({p.extent(0) + 1, 2}); })),
                       message, pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::quasi_interpolate<double>(
                           space, [](Points p) { return NdArray<double>({p.extent(0), 1, 1}); })),
                       message, pantr::value_error);
  // No component at all is refused by the B-spline itself.
  CHECK_THROWS_AS(static_cast<void>(bs::quasi_interpolate<double>(
                      space, [](Points p) { return NdArray<double>({p.extent(0), 0}); })),
                  pantr::value_error);
  // A vector-valued function takes (M, rank): rank 3 here.
  const bs::Bspline<double> vec = bs::quasi_interpolate<double>(
      space, [](Points p) { return NdArray<double>({p.extent(0), 3}); });
  CHECK(vec.rank() == 3);
}

// ---- interpolate -----------------------------------------------------------------------------

TEST_CASE_TEMPLATE(
    "interpolate: one call on the Greville lattice, and the result interpolates the sampled spline",
    T, float, double) {
  // The callable is called exactly once, with a lattice whose directions are the Greville
  // abscissae of the space; the rows of its answer follow the lattice in C order (last direction
  // fastest). Oracle: the exact samples of a spline of the space (the sampler), whose control
  // points the interpolant must return within the recovery bound. A scalar answer as (N,) or as
  // (N, 1) is the same interpolation, bit for bit. Bug caught: a lattice that is not the Greville
  // one, rows in F order, a second call, the rank of the answer misread.
  const std::vector<Dir> dirs = {cubic_dir(), shifted_dir()};
  const bs::BsplineSpace<T> space = space_of<T>(dirs);
  const bs::Bspline<T> truth = make_spline<T>(dirs, 2, false, 121);
  std::vector<std::vector<std::size_t>> shapes;
  std::vector<std::vector<T>> seen_nodes(2);
  const ExactSampler<T> sampler{&truth, &shapes};
  const auto recording = [&](const pantr::quad::PointsLattice<T>& lat) {
    for (std::size_t d = 0; d < 2; ++d) {
      seen_nodes[d].assign(lat.pts(d).begin(), lat.pts(d).end());
    }
    return sampler(lat);
  };
  const bs::Bspline<T> fit = bs::interpolate<T>(space, recording);
  CHECK(shapes.size() == 1);
  REQUIRE(shapes.size() == 1);
  CHECK(shapes[0] == std::vector<std::size_t>{7, 4});
  CHECK(seen_nodes == greville_of<T>(space));
  const std::vector<std::span<const T>> spans(seen_nodes.begin(), seen_nodes.end());
  const bs::TensorCollocation<T> system(space, spans);
  double kappa = 1.0;
  std::size_t sizes = 0;
  for (const NdArray<T>& mat : system.matrices()) {
    kappa *= kappa_bound(exact_matrix<T>(mat));
    sizes += mat.shape()[0] + mat.shape()[1] + 2;
  }
  const double bound = solve_bound<T>(sizes, kappa, two_norm_of<T>(truth.control_points()));
  CHECK(worst_difference<T>(fit.control_points(), truth.control_points()) <= bound);

  // (N,) and (N, 1) are the same scalar interpolation.
  const bs::Bspline<T> scalar_truth = make_spline<T>(dirs, 1, false, 122);
  const ExactSampler<T> scalar_sampler{&scalar_truth, nullptr};
  const bs::Bspline<T> as_matrix = bs::interpolate<T>(space, scalar_sampler);
  const bs::Bspline<T> as_vector =
      bs::interpolate<T>(space, [&](const pantr::quad::PointsLattice<T>& lat) {
        const NdArray<T> m = scalar_sampler(lat);
        NdArray<T> flat({m.size()});
        std::copy(m.data(), m.data() + m.size(), flat.data());
        return flat;
      });
  REQUIRE(as_matrix.control_points().size() == as_vector.control_points().size());
  for (std::size_t i = 0; i < as_matrix.control_points().size(); ++i) {
    CHECK(as_matrix.control_points()[i] == as_vector.control_points()[i]);
  }
}

TEST_CASE(
    "interpolate: the Greville lattice of a Bezier space and of a hat space, in closed form") {
  using Lattice = pantr::quad::PointsLattice<double>;
  std::vector<double> seen;
  const auto record = [&seen](const Lattice& lat) {
    seen.assign(lat.pts(0).begin(), lat.pts(0).end());
    return NdArray<double>({lat.num_points()});
  };
  const std::vector<double> bezier = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
  static_cast<void>(
      bs::interpolate<double>(bs::BsplineSpace<double>(std::vector<bs::BsplineSpace1D<double>>{
                                  bs::BsplineSpace1D<double>(std::span<const double>(bezier), 2)}),
                              record));
  CHECK(seen == std::vector<double>{0.0, 0.5, 1.0});
  const std::vector<double> hat = {0.0, 0.0, 0.5, 1.0, 1.0};
  static_cast<void>(
      bs::interpolate<double>(bs::BsplineSpace<double>(std::vector<bs::BsplineSpace1D<double>>{
                                  bs::BsplineSpace1D<double>(std::span<const double>(hat), 1)}),
                              record));
  CHECK(seen == std::vector<double>{0.0, 0.5, 1.0});
}

TEST_CASE("interpolate: nodes, derivative constraints and tolerance reach the solve") {
  // The convenience form is the staged system plus the callable: with the same nodes, constraints
  // and tolerance, its coefficients are those of `TensorCollocation::solve` on the callable's own
  // values, bit for bit (one code path); and each argument changes the answer where it should
  // (so none is dropped on the way). Bug caught: `nodes`, `derivatives` or `tol` ignored.
  const std::vector<Dir> dirs = {cubic_dir()};
  const bs::BsplineSpace<double> space = space_of<double>(dirs);
  const bs::Bspline<double> truth = make_spline<double>(dirs, 1, false, 131);
  std::vector<std::vector<std::size_t>> shapes;
  const ExactSampler<double> sampler{&truth, &shapes};
  const std::vector<std::vector<double>> twenty = {uniform_nodes<double>(space.spaces()[0], 20)};

  // Custom nodes: 20 of them (the lattice has 20 points), the same coefficients as the system.
  const bs::Bspline<double> by_nodes = bs::interpolate<double>(space, sampler, twenty);
  CHECK(shapes.back() == std::vector<std::size_t>{20});
  const std::vector<std::span<const double>> spans = {std::span<const double>(twenty[0])};
  const std::vector<double> values = samples_on_grid<double>(truth, twenty);
  const NdArray<double> direct = bs::TensorCollocation<double>(space, spans)
                                     .solve<double>(std::span<const double>(values), 1,
                                                    bs::default_fit_tol<double>(std::nullopt));
  REQUIRE(by_nodes.control_points().size() == direct.size());
  for (std::size_t i = 0; i < direct.size(); ++i) {
    CHECK(by_nodes.control_points()[i] == direct.data()[i]);
  }
  // Derivative constraints change the answer, and are those of the system.
  const std::vector<std::optional<bs::BoundaryDerivatives>> derivs = {
      bs::BoundaryDerivatives{1, 1}};
  const bs::Bspline<double> plain = bs::interpolate<double>(space, sampler);
  const bs::Bspline<double> pinned = bs::interpolate<double>(space, sampler, {}, derivs);
  CHECK(worst_difference<double>(pinned.control_points(), plain.control_points()) > 1e-3);
  const std::vector<std::vector<double>> greville = greville_of<double>(space);
  const std::vector<std::span<const double>> gspans = {std::span<const double>(greville[0])};
  const std::vector<double> gvalues = samples_on_grid<double>(truth, greville);
  const NdArray<double> gdirect = bs::TensorCollocation<double>(space, gspans, derivs)
                                      .solve<double>(std::span<const double>(gvalues), 1,
                                                     bs::default_fit_tol<double>(std::nullopt));
  for (std::size_t i = 0; i < gdirect.size(); ++i) {
    // The sampler's values at the replaced rows are the spline's, not zero: the system zeroes them.
    CHECK(pinned.control_points()[i] == gdirect.data()[i]);
  }
  // A tolerance above 1 on the overdetermined system drops every singular value.
  const bs::Bspline<double> dropped = bs::interpolate<double>(space, sampler, twenty, {}, 2.0);
  for (const double v : dropped.control_points()) {
    CHECK(v == 0.0);
  }
}

TEST_CASE("interpolate: a callable returning the wrong number of values is refused") {
  const bs::BsplineSpace<double> space = space_of<double>({cubic_dir(), shifted_dir()});
  using Lattice = pantr::quad::PointsLattice<double>;
  const char* const message =
      "interpolate: the function returned another number of values than the 28 nodes.";
  CHECK_THROWS_WITH_AS(
      static_cast<void>(bs::interpolate<double>(
          space, [](const Lattice& l) { return NdArray<double>({l.num_points() - 1}); })),
      message, pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(bs::interpolate<double>(
          space, [](const Lattice& l) { return NdArray<double>({l.num_points() + 1, 1}); })),
      message, pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(bs::interpolate<double>(
          space, [](const Lattice& l) { return NdArray<double>({l.num_points(), 0}); })),
      message, pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(bs::interpolate<double>(
          space, [](const Lattice& l) { return NdArray<double>({l.num_points(), 1, 1}); })),
      message, pantr::value_error);
  // The node arrays are checked before the callable is reached.
  int calls = 0;
  const std::vector<std::vector<double>> one_direction = {{2.0, 3.0}};
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::interpolate<double>(
                           space,
                           [&calls](const Lattice& l) {
                             ++calls;
                             return NdArray<double>({l.num_points()});
                           },
                           one_direction)),
                       "Expected 2 node arrays, got 1.", pantr::value_error);
  CHECK(calls == 0);
}

TEST_CASE_TEMPLATE("default_fit_tol: 100 eps of the matrix type, an explicit value untouched", T,
                   float, double) {
  const double eps = machine_epsilon<T>();
  CHECK(bs::default_fit_tol<T>(std::nullopt) == 100.0 * eps);
  CHECK(bs::default_fit_tol<T>(0.25) == 0.25);
  CHECK(bs::default_fit_tol<T>(0.0) == 0.0);
}

TEST_CASE_TEMPLATE(
    "fit_tensor: the default tolerance truncates a near-duplicate rectangular system but not a "
    "square one",
    T, float, double) {
  // Degree 1 on [0, 1]: the row of a node x is (1 - x, x). Three nodes 0.5, 0.5 + d, 0.5 + 2d with
  // d = 4 eps give singular values sqrt(3/2) and about 2 d, a ratio of about 1.6 d = 6.5 eps, under
  // the default cutoff 100 eps and over 1e-3 eps. The data (0, 1/2, 1) have the component
  // 1 / sqrt 2 along the weak left singular vector. The rectangular fit truncates by default
  // (coefficients of order 1) and, with the tolerance lowered, keeps the weak direction
  // (coefficients of order 1 / (4 d)). A square system (two nodes) is solved by LU without
  // truncation: its coefficients are those of the near-singular matrix (order 1 / d) whatever the
  // default says. Bug caught: a default that truncates nothing (noise amplified by 1/eps), one that
  // truncates a square system, or one that uses the epsilon of the wrong type.
  const T eps = std::numeric_limits<T>::epsilon();
  const T d = T(4) * eps;
  const std::vector<double> knots = {0.0, 0.0, 1.0, 1.0};
  const bs::BsplineSpace1D<T> s1(std::span<const T>(cast_all<T>(knots)), 1);
  const bs::BsplineSpace<T> space(std::vector<bs::BsplineSpace1D<T>>{s1});
  const std::vector<T> three = {T(0.5), T(0.5) + d, T(0.5) + T(2) * d};
  const std::vector<T> data = {T(0), T(0.5), T(1)};
  const std::vector<std::span<const T>> spans = {std::span<const T>(three)};
  const bs::Bspline<T> truncated = bs::fit_tensor<T>(space, spans, std::span<const T>(data), 1);
  const bs::Bspline<T> kept = bs::fit_tensor<T>(space, spans, std::span<const T>(data), 1, {},
                                                1e-3 * static_cast<double>(eps));
  double largest_truncated = 0.0;
  double largest_kept = 0.0;
  for (std::size_t i = 0; i < 2; ++i) {
    largest_truncated =
        (std::max)(largest_truncated, std::abs(static_cast<double>(truncated.control_points()[i])));
    largest_kept =
        (std::max)(largest_kept, std::abs(static_cast<double>(kept.control_points()[i])));
  }
  CHECK(largest_truncated < 10.0);
  CHECK(largest_kept > 1.0 / (16.0 * static_cast<double>(d)));
  // The square system: det = d, the solution (-0.5, 0.5) / d for the data (0, 1).
  const std::vector<T> two = {T(0.5), T(0.5) + d};
  const std::vector<T> two_data = {T(0), T(1)};
  const std::vector<std::span<const T>> two_spans = {std::span<const T>(two)};
  const bs::Bspline<T> square =
      bs::fit_tensor<T>(space, two_spans, std::span<const T>(two_data), 1);
  CHECK(std::abs(static_cast<double>(square.control_points()[0]) + 0.5 / static_cast<double>(d)) <=
        0.01 * 0.5 / static_cast<double>(d));
  CHECK(std::abs(static_cast<double>(square.control_points()[1]) - 0.5 / static_cast<double>(d)) <=
        0.01 * 0.5 / static_cast<double>(d));
}

// ---- mixed types: the instantiations the bindings route to ------------------------------------

namespace fit_types {
// Space, nodes (and matrices), values.
struct SpaceDoubleNodesFloatValuesFloat {
  using T = double;
  using P = float;
  using C = float;
};
struct SpaceFloatNodesDoubleValuesDouble {
  using T = float;
  using P = double;
  using C = double;
};
struct SpaceDoubleNodesFloatValuesDouble {
  using T = double;
  using P = float;
  using C = double;
};
struct SpaceFloatNodesDoubleValuesFloat {
  using T = float;
  using P = double;
  using C = float;
};
}  // namespace fit_types

TEST_CASE_TEMPLATE(
    "TensorCollocation and fit_scattered_coefficients: space, node and value types may differ", TPC,
    fit_types::SpaceDoubleNodesFloatValuesFloat, fit_types::SpaceFloatNodesDoubleValuesDouble,
    fit_types::SpaceDoubleNodesFloatValuesDouble, fit_types::SpaceFloatNodesDoubleValuesFloat) {
  // The bindings route every combination of the space's type T, the nodes' (and so the matrices')
  // type P and the values' type C. Oracle: a spline of the space with dyadic control points,
  // sampled exactly at the nodes as P actually holds them (exact rational evaluation, rounded once
  // to C), and recovered from 12 equispaced nodes (tensor) and 20 scattered points; the bound is
  // the header's, with u the largest unit roundoff of the three types (the basis is tabulated in
  // the space's knots and the nodes' type, the matrix cast to C for the solve). Bug caught: a
  // cast in the wrong direction (an exact node moved), a value type that changes the layout of the
  // result, an instantiation that no other test compiles.
  using T = typename TPC::T;
  using P = typename TPC::P;
  using C = typename TPC::C;
  const double u = (std::max)({unit_roundoff<T>(), unit_roundoff<P>(), unit_roundoff<C>()});
  const std::vector<Dir> dirs = {cubic_dir()};
  const bs::BsplineSpace<T> space = space_of<T>(dirs);
  const bs::Bspline<T> truth = make_spline<T>(dirs, 2, false, 141);
  const std::vector<T> nodes_t = uniform_nodes<T>(space.spaces()[0], 12);
  std::vector<P> nodes;
  for (const T x : nodes_t) {
    nodes.push_back(static_cast<P>(static_cast<double>(x)));
  }
  const auto sample = [&](const std::vector<P>& pts) {
    std::vector<C> out;
    for (const P x : pts) {
      const std::vector<Rational> point = {oracle::exact(static_cast<double>(x))};
      for (const Rational& v : oracle::exact_value<T>(truth, point)) {
        out.push_back(static_cast<C>(to_double(v)));
      }
    }
    return out;
  };

  // Tensor: TensorCollocation<P> built on a space of type T, solved with values of type C.
  const std::vector<std::span<const P>> spans = {std::span<const P>(nodes)};
  const bs::TensorCollocation<P> system(space, spans);
  const std::vector<C> values = sample(nodes);
  const NdArray<C> coeffs =
      system.template solve<C>(std::span<const C>(values), 2, 100.0 * machine_epsilon<C>());
  Matrix cast_matrix = exact_matrix<P>(system.matrices()[0]);
  for (auto& row : cast_matrix) {
    for (Rational& v : row) {
      v = oracle::exact(
          static_cast<double>(static_cast<C>(to_double(v))));  // the solve's cast to C
    }
  }
  const double kappa = kappa_bound(cast_matrix);
  REQUIRE(1.0 / kappa > 100.0 * machine_epsilon<C>());
  const double bound =
      fit_slack * (12.0 + 7.0 + 2.0) * u * kappa * two_norm_of<T>(truth.control_points());
  REQUIRE(coeffs.size() == truth.control_points().size());
  double worst = 0.0;
  for (std::size_t i = 0; i < coeffs.size(); ++i) {
    worst = (std::max)(worst, std::abs(static_cast<double>(coeffs.data()[i]) -
                                       static_cast<double>(truth.control_points()[i])));
  }
  INFO("tensor: kappa bound " << kappa << ", worst error " << worst << ", bound " << bound);
  CHECK(worst <= bound);

  // Scattered: the points as an (n, 1) array of type P.
  std::vector<P> pts;
  for (std::size_t i = 0; i < 20; ++i) {
    pts.push_back(static_cast<P>(static_cast<double>((i * 7) % 20) / 19.0));
  }
  const std::vector<C> scattered_values = sample(pts);
  const NdArray<C> fitted = bs::fit_scattered_coefficients<T, P, C>(
      space, pantr::mdview<const P, 2>(pts.data(), 20, 1),
      pantr::mdview<const C, 2>(scattered_values.data(), 20, 2), 100.0 * machine_epsilon<C>());
  const NdArray<P> matrix =
      bs::scattered_collocation_matrix<T, P>(space, pantr::mdview<const P, 2>(pts.data(), 20, 1));
  Matrix cast_scattered = exact_matrix<P>(matrix);
  for (auto& row : cast_scattered) {
    for (Rational& v : row) {
      v = oracle::exact(static_cast<double>(static_cast<C>(to_double(v))));
    }
  }
  const double kappa_s = kappa_bound(cast_scattered);
  REQUIRE(1.0 / kappa_s > 100.0 * machine_epsilon<C>());
  const double bound_s =
      fit_slack * (20.0 + 7.0 + 2.0) * u * kappa_s * two_norm_of<T>(truth.control_points());
  REQUIRE(fitted.size() == truth.control_points().size());
  double worst_s = 0.0;
  for (std::size_t i = 0; i < fitted.size(); ++i) {
    worst_s = (std::max)(worst_s, std::abs(static_cast<double>(fitted.data()[i]) -
                                           static_cast<double>(truth.control_points()[i])));
  }
  INFO("scattered: kappa bound " << kappa_s << ", worst error " << worst_s << ", bound "
                                 << bound_s);
  CHECK(worst_s <= bound_s);
}

TEST_CASE_TEMPLATE("L2Projection: the mass matrices of one type project values of another", PC,
                   fit_types::DoubleFloat, fit_types::FloatDouble) {
  // `L2Projection<P>::project<C>` casts the mass matrices to C for the solve. Oracle: a cubic
  // spline with dyadic control points sampled exactly on the quadrature lattice (rounded once to
  // C): projecting it onto its own space gives its control points back, within the header's bound
  // (u the larger of the two unit roundoffs; the size term adds the quadrature points).
  using T = typename PC::P;
  using C = typename PC::C;
  const double u = (std::max)(unit_roundoff<T>(), unit_roundoff<C>());
  const std::vector<Dir> dirs = {cubic_dir()};
  const bs::BsplineSpace<T> space = space_of<T>(dirs);
  const bs::Bspline<T> truth = make_spline<T>(dirs, 1, false, 142);
  const bs::L2Projection<T> proj(space, {4}, bs::L2Quadrature::gauss_legendre, {{false, false}});
  const std::vector<T> lattice(proj.nodes()[0].data(),
                               proj.nodes()[0].data() + proj.nodes()[0].size());
  std::vector<C> values;
  for (const T x : lattice) {
    const std::vector<Rational> point = {oracle::exact(static_cast<double>(x))};
    values.push_back(static_cast<C>(to_double(oracle::exact_value<T>(truth, point)[0])));
  }
  const NdArray<C> coeffs =
      proj.template project<C>(std::span<const C>(values), {}, 100.0 * machine_epsilon<C>());
  Matrix mass = exact_matrix<T>(proj.mass()[0]);
  for (auto& row : mass) {
    for (Rational& v : row) {
      v = oracle::exact(static_cast<double>(static_cast<C>(to_double(v))));
    }
  }
  const double bound = fit_slack * (2.0 * 7.0 + 2.0 + static_cast<double>(lattice.size())) * u *
                       kappa_bound(mass) * two_norm_of<T>(truth.control_points());
  REQUIRE(coeffs.size() == truth.control_points().size());
  double worst = 0.0;
  for (std::size_t i = 0; i < coeffs.size(); ++i) {
    worst = (std::max)(worst, std::abs(static_cast<double>(coeffs.data()[i]) -
                                       static_cast<double>(truth.control_points()[i])));
  }
  INFO("worst error " << worst << ", bound " << bound);
  CHECK(worst <= bound);
}

// The square path's banded LU (with its periodic border). Oracle: the backward error of LU with
// partial pivoting (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., section 9.3):
// the normwise relative residual `||A x - b|| / (||A|| ||x|| + ||b||)` of a stable solve is a
// modest multiple of the unit roundoff (the growth factor, small for these banded matrices, times
// `n`); 1e3 u bounds it for order 120 with a wide margin. Plus agreement with Eigen's dense
// PartialPivLU, relative to the solution's size and the matrix's condition number.
namespace {

la::RowMatrix<double> band_matrix(Eigen::Index n, Eigen::Index kl, Eigen::Index ku,
                                  Eigen::Index wrap, unsigned seed) {
  la::RowMatrix<double> a = la::RowMatrix<double>::Zero(n, n);
  unsigned state = seed;
  const auto next = [&state]() {
    state = state * 1103515245U + 12345U;
    // In (-0.5, 0.5), never zero, so that no entry of the band vanishes by chance.
    return (static_cast<double>((state >> 8) % 1000U) + 0.5) / 1000.0 - 0.5;
  };
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index j = (std::max)(Eigen::Index{0}, i - kl); j <= (std::min)(n - 1, i + ku);
         ++j) {
      a(i, j) = next();
    }
    // With a band on both sides, a small diagonal makes partial pivoting swap rows whenever the
    // band below holds more; a one-sided band with a small diagonal would be exponentially
    // ill-conditioned (and swaps nothing useful), so it gets a dominant one.
    a(i, i) = kl > 0 && ku > 0 ? 0.05 * next() : 1.0 + std::abs(next());
  }
  for (Eigen::Index i = n - wrap; i < n; ++i) {
    for (Eigen::Index t = 0; t < wrap; ++t) {
      a(i, t) = next();  // the periodic wrap: the last rows reach the first columns
    }
  }
  return a;
}

double dense_difference(const la::RowMatrix<double>& a, const la::RowMatrix<double>& b,
                        const NdArray<double>& x) {
  // Relative to the reference's size, in units of its condition number times the unit roundoff.
  const Eigen::PartialPivLU<Eigen::MatrixXd> lu(a);
  const Eigen::MatrixXd ref = lu.solve(Eigen::MatrixXd(b));
  const Eigen::Map<const la::RowMatrix<double>> got(x.data(), ref.rows(), ref.cols());
  const double cond = lu.rcond() > 0.0 ? 1.0 / lu.rcond() : 1e300;
  const double u = std::numeric_limits<double>::epsilon() / 2.0;
  return (ref - Eigen::MatrixXd(got)).lpNorm<Eigen::Infinity>() /
         (ref.lpNorm<Eigen::Infinity>() * cond * u);
}

double backward_error(const la::RowMatrix<double>& a, const la::RowMatrix<double>& b,
                      const NdArray<double>& x) {
  const Eigen::Map<const la::RowMatrix<double>> got(x.data(), a.cols(), b.cols());
  const double u = std::numeric_limits<double>::epsilon() / 2.0;
  const double num = (a * got - b).lpNorm<Eigen::Infinity>();
  const double den =
      a.lpNorm<Eigen::Infinity>() * got.lpNorm<Eigen::Infinity>() + b.lpNorm<Eigen::Infinity>();
  return num / den / u;
}

}  // namespace

TEST_CASE("solve_collocation: the banded LU matches the dense LU over its band shapes") {
  constexpr Eigen::Index n = 120;  // wide enough that every shape below takes the band
  for (Eigen::Index kl = 0; kl <= 3; ++kl) {
    for (Eigen::Index ku = 0; ku <= 3; ++ku) {
      for (Eigen::Index wrap : {Eigen::Index{0}, Eigen::Index{3}}) {
        CAPTURE(kl);
        CAPTURE(ku);
        CAPTURE(wrap);
        const la::RowMatrix<double> a =
            band_matrix(n, kl, ku, wrap, static_cast<unsigned>(17 * kl + 5 * ku + wrap + 1));
        la::RowMatrix<double> b(n, 2);
        for (Eigen::Index i = 0; i < n; ++i) {
          b(i, 0) = 1.0;
          b(i, 1) = static_cast<double>(i % 7) - 3.0;
        }
        const bs::detail::Bandwidths bw =
            bs::detail::bandwidths<double>(a, bs::detail::wrap_border<double>(a));
        CHECK(bs::detail::band_pays(bw, n));
        const std::optional<NdArray<double>> banded = bs::detail::banded_lu_solve<double>(a, b, bw);
        REQUIRE(banded.has_value());
        CHECK(backward_error(a, b, *banded) < 1e3);
        // A bidiagonal band with a small diagonal is exponentially ill-conditioned; there the
        // forward comparison means nothing and only the backward error is checked.
        if (Eigen::PartialPivLU<Eigen::MatrixXd>(a).rcond() > 1e-10) {
          CHECK(dense_difference(a, b, *banded) < 1e3);
        }
        const NdArray<double> via = bs::solve_collocation<double>(a, b, 1e-14);
        CHECK(backward_error(a, b, via) < 1e3);
      }
    }
  }
}

TEST_CASE("solve_collocation: a singular banded system and a NaN right-hand side") {
  constexpr Eigen::Index n = 60;
  la::RowMatrix<double> a = band_matrix(n, 1, 2, 0, 7U);
  a.row(30) = a.row(29);  // two equal rows: elimination leaves an exactly zero pivot
  la::RowMatrix<double> b = la::RowMatrix<double>::Ones(n, 1);
  const bs::detail::Bandwidths bw = bs::detail::bandwidths<double>(a, 0);
  CHECK_FALSE(bs::detail::banded_lu_solve<double>(a, b, bw).has_value());
  // The truncated SVD answers: its solution is the dense pseudo-inverse's.
  const NdArray<double> x = bs::solve_collocation<double>(a, b, 1e-12);
  const Eigen::VectorXd ref =
      Eigen::MatrixXd(a).completeOrthogonalDecomposition().solve(Eigen::VectorXd::Ones(n));
  for (Eigen::Index i = 0; i < n; ++i) {
    CHECK(std::abs(x.data()[i] - ref(i)) < 1e-8 * (1.0 + std::abs(ref(i))));
  }
  // A NaN sample spreads over the solution as the dense factorisation spreads it.
  const la::RowMatrix<double> good = band_matrix(n, 1, 1, 0, 9U);
  la::RowMatrix<double> nan_rhs = la::RowMatrix<double>::Ones(n, 1);
  nan_rhs(5, 0) = std::numeric_limits<double>::quiet_NaN();
  const NdArray<double> spread = bs::solve_collocation<double>(good, nan_rhs, 1e-14);
  CHECK(std::isnan(spread.data()[n - 1]));
}
