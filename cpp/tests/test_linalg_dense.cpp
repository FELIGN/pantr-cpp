// The dense least-squares solves of `linalg/dense.hpp`: `lstsq_min_norm_jacobi` (the Jacobi-SVD
// solve) and `try_lstsq_full_rank_qr` (Householder QR where a full-rank certificate holds).
//
// Oracles, each named again in its test case: constructed solutions (data chosen dyadic so that
// `b = A x` is formed exactly), the minimum-norm solution of a rank-deficient system worked by
// hand, and the rank rule of the certificate itself, whose boundary `s_min = sqrt(eps) ||A||_F` is
// derived in the header.
//
// Tolerance of a solution. A backward-stable solve of a consistent full-rank system `A x = b`
// gives `||x_hat - x|| <= 2 kappa gamma ||x||` for `kappa kappa gamma <= 1/2`, with `gamma` the
// backward error relative to `||A||`; both solves are held to `gamma = 24 m n eps`, the constant
// `try_lstsq_full_rank_qr` assumes for its factorization (the SVD's own constant is not quantified
// by Eigen, so for it the same bound is an admitted heuristic). The solution is then rounded to
// `T`, which adds `u_T ||x||`. Two solutions are compared within the sum of their distances to
// `x`.
#include <doctest/doctest.h>

#include <pantr/core/errors.hpp>
#include <pantr/core/ndarray.hpp>
#include <pantr/linalg/dense.hpp>

#include <Eigen/Core>
#include <Eigen/SVD>

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

namespace {

namespace la = pantr::linalg;
using pantr::NdArray;

constexpr double eps = std::numeric_limits<double>::epsilon();

template <class T>
la::RowMatrix<T> matrix_of(Eigen::Index rows, Eigen::Index cols, const std::vector<double>& v) {
  REQUIRE(v.size() == static_cast<std::size_t>(rows * cols));
  la::RowMatrix<T> m(rows, cols);
  for (Eigen::Index i = 0; i < rows; ++i) {
    for (Eigen::Index j = 0; j < cols; ++j) {
      m(i, j) = static_cast<T>(v[static_cast<std::size_t>(i * cols + j)]);
    }
  }
  return m;
}

/// The solution as a double matrix.
template <class T>
Eigen::MatrixXd as_double(const NdArray<T>& x) {
  REQUIRE(x.shape().size() == 2);
  Eigen::MatrixXd out(static_cast<Eigen::Index>(x.shape()[0]),
                      static_cast<Eigen::Index>(x.shape()[1]));
  for (Eigen::Index i = 0; i < out.rows(); ++i) {
    for (Eigen::Index j = 0; j < out.cols(); ++j) {
      out(i, j) = static_cast<double>(x.data()[static_cast<std::size_t>(i * out.cols() + j)]);
    }
  }
  return out;
}

/// `u_T`, the unit roundoff of `T`.
template <class T>
constexpr double unit_roundoff() {
  return static_cast<double>(std::numeric_limits<T>::epsilon()) / 2.0;
}

/// A dyadic `m x n` matrix, entries multiples of 1/16 in `[-1/2, 1/2]`, the column `j` scaled by
/// `scales[j]` (a power of two, so still dyadic).
Eigen::MatrixXd dyadic_matrix(Eigen::Index m, Eigen::Index n, const std::vector<double>& scales) {
  Eigen::MatrixXd a(m, n);
  for (Eigen::Index i = 0; i < m; ++i) {
    for (Eigen::Index j = 0; j < n; ++j) {
      const long code = (7 * i + 13 * j + 3 * i * j + i * i) % 17;
      a(i, j) = static_cast<double>(code - 8) / 16.0 * scales[static_cast<std::size_t>(j)];
    }
  }
  return a;
}

template <class T>
la::RowMatrix<T> to_storage(const Eigen::MatrixXd& a) {
  return a.cast<T>();
}

}  // namespace

TEST_CASE("lstsq_min_norm_jacobi: a non-finite matrix raises numpy's message") {
  const char* const msg = "SVD did not converge in Linear Least Squares";
  const la::RowMatrix<double> b = matrix_of<double>(2, 1, {1, 1});
  for (const double bad :
       {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()}) {
    la::RowMatrix<double> a = matrix_of<double>(2, 2, {1, 0, 0, 1});
    a(1, 0) = bad;
    CHECK_THROWS_WITH_AS(static_cast<void>(la::lstsq_min_norm_jacobi<double>(a, b)), msg,
                         pantr::value_error);
    CHECK_THROWS_WITH_AS(static_cast<void>(la::lstsq_min_norm<double>(a, b)), msg,
                         pantr::value_error);
  }
}

TEST_CASE(
    "try_lstsq_full_rank_qr: a non-finite input, a wide matrix or a bad cutoff gets no answer") {
  const la::RowMatrix<double> good = matrix_of<double>(3, 2, {1, 0, 0, 1, 1, 1});
  const la::RowMatrix<double> b = matrix_of<double>(3, 1, {1, 1, 2});
  REQUIRE(la::try_lstsq_full_rank_qr<double>(good, b).has_value());
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  la::RowMatrix<double> bad_a = good;
  bad_a(0, 1) = nan;
  CHECK_FALSE(la::try_lstsq_full_rank_qr<double>(bad_a, b).has_value());
  bad_a(0, 1) = inf;
  CHECK_FALSE(la::try_lstsq_full_rank_qr<double>(bad_a, b).has_value());
  la::RowMatrix<double> bad_b = b;
  bad_b(2, 0) = nan;
  CHECK_FALSE(la::try_lstsq_full_rank_qr<double>(good, bad_b).has_value());
  bad_b(2, 0) = -inf;
  CHECK_FALSE(la::try_lstsq_full_rank_qr<double>(good, bad_b).has_value());
  // Fewer rows than columns: the minimum-norm solution is the SVD's to give.
  const la::RowMatrix<double> wide = matrix_of<double>(2, 3, {1, 0, 1, 0, 1, 1});
  CHECK_FALSE(
      la::try_lstsq_full_rank_qr<double>(wide, matrix_of<double>(2, 1, {1, 2})).has_value());
  // No certificate stands for a NaN, infinite or negative cutoff.
  CHECK_FALSE(la::try_lstsq_full_rank_qr<double>(good, b, nan).has_value());
  CHECK_FALSE(la::try_lstsq_full_rank_qr<double>(good, b, inf).has_value());
  CHECK_FALSE(la::try_lstsq_full_rank_qr<double>(good, b, -1.0).has_value());
  CHECK(la::try_lstsq_full_rank_qr<double>(good, b, 0.0).has_value());
}

TEST_CASE_TEMPLATE(
    "a rank-deficient system: QR declines and Jacobi gives the minimum-norm solution", T, double,
    float) {
  // Oracle, worked by hand. A = [c1, c2, c1 + c2] with c1 = (1, 0, 1, 1), c2 = (0, 1, 1, -1), so
  // the null space is spanned by n = (1, 1, -1) / sqrt 3. b = A x0 = c1 + c2 = (1, 1, 2, 0) for
  // x0 = (1, 1, 0); the minimum-norm solution is the projection of x0 on the row space, x0 - (x0 .
  // n) n = (1, 1, 0) - (2 / 3) (1, 1, -1) = (1/3, 1/3, 2/3). The nonzero singular values of A are
  // 3 and sqrt 3 (the eigenvalues 9 and 3 of A^T A on the row space), so kappa = sqrt 3; the third
  // is rounding noise below the rank cutoff `4 eps s_max`.
  const la::RowMatrix<T> a = matrix_of<T>(4, 3, {1, 0, 1, 0, 1, 1, 1, 1, 2, 1, -1, 0});
  const la::RowMatrix<T> b = matrix_of<T>(4, 1, {1, 1, 2, 0});
  CHECK_FALSE(la::try_lstsq_full_rank_qr<T>(a, b).has_value());
  const NdArray<T> got = la::lstsq_min_norm_jacobi<T>(a, b);
  const Eigen::MatrixXd x = as_double(got);
  REQUIRE(x.rows() == 3);
  REQUIRE(x.cols() == 1);
  const Eigen::Vector3d want(1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0);
  const double kappa = std::sqrt(3.0);
  const double gamma = 24.0 * 4.0 * 3.0 * eps;
  const double tol = 2.0 * kappa * gamma * want.norm() + unit_roundoff<T>() * want.norm();
  CHECK((x.col(0) - want).norm() <= tol);
  // The all-ones matrix of two columns, whose rank cutoff drops the second singular value: the
  // minimum-norm solution of x1 + x2 = 2 is (1, 1), and of the inconsistent b = (1, 0, 0) it is
  // v (u . b) / 6 = (1/6, 1/6).
  const la::RowMatrix<T> ones = matrix_of<T>(3, 2, {1, 1, 1, 1, 1, 1});
  const la::RowMatrix<T> rhs = matrix_of<T>(3, 2, {2, 1, 2, 0, 2, 0});
  CHECK_FALSE(la::try_lstsq_full_rank_qr<T>(ones, rhs).has_value());
  const Eigen::MatrixXd y = as_double(la::lstsq_min_norm_jacobi<T>(ones, rhs));
  const double tol2 = 2.0 * 1.0 * 24.0 * 3.0 * 2.0 * eps * 1.5 + unit_roundoff<T>() * 1.5;
  CHECK(std::abs(y(0, 0) - 1.0) <= tol2);
  CHECK(std::abs(y(1, 0) - 1.0) <= tol2);
  CHECK(std::abs(y(0, 1) - 1.0 / 6.0) <= tol2);
  CHECK(std::abs(y(1, 1) - 1.0 / 6.0) <= tol2);
}

TEST_CASE_TEMPLATE("a full-rank system: QR and Jacobi agree within the stated bound", T, double,
                   float) {
  // Oracle: a constructed solution. A is dyadic with columns scaled by powers of two (so column
  // pivoting reorders them), x is dyadic, b = A x is exact in double and in float, and the system
  // is consistent: the bound of the header comment applies with the condition number `kappa` of A,
  // measured here by an SVD of A.
  struct Shape {
    Eigen::Index m;
    Eigen::Index n;
  };
  for (const Shape s : {Shape{3, 3}, Shape{9, 4}, Shape{40, 7}, Shape{60, 12}}) {
    CAPTURE(s.m);
    CAPTURE(s.n);
    std::vector<double> scales(static_cast<std::size_t>(s.n));
    for (Eigen::Index j = 0; j < s.n; ++j) {
      scales[static_cast<std::size_t>(j)] = std::ldexp(1.0, static_cast<int>(j % 3) * 2 - 2);
    }
    const Eigen::MatrixXd ad = dyadic_matrix(s.m, s.n, scales);
    Eigen::MatrixXd xd(s.n, 2);
    for (Eigen::Index j = 0; j < s.n; ++j) {
      xd(j, 0) = static_cast<double>(j % 5 - 2) / 4.0;
      xd(j, 1) = static_cast<double>(3 - j % 4) / 2.0;
    }
    const Eigen::MatrixXd bd = ad * xd;
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(ad);
    const double kappa = svd.singularValues()(0) / svd.singularValues()(s.n - 1);
    REQUIRE(kappa < 200.0);
    const double gamma = 24.0 * static_cast<double>(s.m * s.n) * eps;
    REQUIRE(kappa * gamma <= 0.5);
    const double dist = 2.0 * kappa * gamma + unit_roundoff<T>();
    const la::RowMatrix<T> a = to_storage<T>(ad);
    const la::RowMatrix<T> b = to_storage<T>(bd);
    const std::optional<NdArray<T>> qr = la::try_lstsq_full_rank_qr<T>(a, b);
    REQUIRE(qr.has_value());
    const Eigen::MatrixXd xq = as_double(*qr);
    const Eigen::MatrixXd xj = as_double(la::lstsq_min_norm_jacobi<T>(a, b));
    REQUIRE(xq.rows() == s.n);
    REQUIRE(xq.cols() == 2);
    for (Eigen::Index c = 0; c < 2; ++c) {
      const double scale = xd.col(c).norm();
      CHECK((xq.col(c) - xd.col(c)).norm() <= dist * scale);
      CHECK((xj.col(c) - xd.col(c)).norm() <= dist * scale);
      CHECK((xq.col(c) - xj.col(c)).norm() <= 2.0 * dist * scale);
    }
  }
}

TEST_CASE("try_lstsq_full_rank_qr: the certificate declines below s_min = sqrt(eps) ||A||_F") {
  // Oracle: the rank rule. A = diag(1, s) has singular values 1 and s and ||A||_F ~ 1, and
  // ||R^-1||_F ~ 1 / s, so the certificate `||R^-1||_F tau ||A||_F < 1` with `tau = sqrt(eps)` (the
  // floor, which exceeds `4 (24 m n eps + rcond)` for a 2 x 2 matrix) holds iff `s` is above
  // about `sqrt(eps) = 1.5e-8`: s = 1e-6 is answered, s = 1e-10 is declined although the SVD rule
  // (cutoff `2 eps`) would keep it.
  const la::RowMatrix<double> b = matrix_of<double>(2, 1, {1, 1});
  const la::RowMatrix<double> kept = matrix_of<double>(2, 2, {1, 0, 0, 1e-6});
  const std::optional<NdArray<double>> got = la::try_lstsq_full_rank_qr<double>(kept, b);
  REQUIRE(got.has_value());
  CHECK(std::abs(got->data()[0] - 1.0) <= 1e-12);
  CHECK(std::abs(got->data()[1] - 1e6) <= 1e-9 * 1e6);
  const la::RowMatrix<double> small = matrix_of<double>(2, 2, {1, 0, 0, 1e-10});
  CHECK_FALSE(la::try_lstsq_full_rank_qr<double>(small, b).has_value());
  const Eigen::MatrixXd svd_answer = as_double(la::lstsq_min_norm_jacobi<double>(small, b));
  CHECK(std::abs(svd_answer(1, 0) - 1e10) <= 1e-9 * 1e10);
}

TEST_CASE("try_lstsq_full_rank_qr: an exactly rank-1 matrix below the normal range is declined") {
  // Oracle: exact rank. A = [[a, a], [b, b]] with a = 2^-500, b = 1.5 * 2^-512 has rank 1. Eigen
  // skips a Householder reflector whose tail's squared norm is at most DBL_MIN (here b^2 = 0.5625
  // DBL_MIN), leaving R = [[a, a], [0, b]], whose certificate then held: the QR answer for the
  // right-hand side (a, b) was (0, 1), where the minimum-norm one is (1/2, 1/2). The same for s
  // times the 2 x 2 all-ones matrix at s = 2^-511, where the squared tail is DBL_MIN exactly.
  const double a = std::ldexp(1.0, -500);
  const double b = 1.5 * std::ldexp(1.0, -512);
  const la::RowMatrix<double> skew = matrix_of<double>(2, 2, {a, a, b, b});
  CHECK_FALSE(
      la::try_lstsq_full_rank_qr<double>(skew, matrix_of<double>(2, 1, {a, b})).has_value());
  const double s = std::ldexp(1.0, -511);
  const la::RowMatrix<double> ones = matrix_of<double>(2, 2, {s, s, s, s});
  CHECK_FALSE(
      la::try_lstsq_full_rank_qr<double>(ones, matrix_of<double>(2, 1, {s, s})).has_value());
}

TEST_CASE(
    "try_lstsq_full_rank_qr: a well-conditioned matrix is certified at any power-of-two scale") {
  // Oracle: a metamorphic relation. Scaling A by 2^k scales the solution by 2^-k exactly, and the
  // certificate, which measures kappa_F(A), does not depend on k. Bug caught: certification only
  // for |k| <= 511 (a sum of squares overflowing or underflowing past it), and the one-column case
  // near sqrt(DBL_MAX), whose squared norm overflows.
  const la::RowMatrix<double> base = matrix_of<double>(3, 2, {2, 1, 1, 3, 0, 1});
  const la::RowMatrix<double> rhs = matrix_of<double>(3, 1, {1, 2, 3});
  const std::optional<NdArray<double>> ref = la::try_lstsq_full_rank_qr<double>(base, rhs);
  REQUIRE(ref.has_value());
  for (const int k : {-1000, -600, -511, 0, 511, 600, 1000}) {
    INFO("k = " << k);
    const la::RowMatrix<double> scaled = base.unaryExpr([k](double v) { return std::ldexp(v, k); });
    const std::optional<NdArray<double>> got = la::try_lstsq_full_rank_qr<double>(scaled, rhs);
    REQUIRE(got.has_value());
    for (std::size_t i = 0; i < 2; ++i) {
      CHECK(std::ldexp(got->data()[i], k) == ref->data()[i]);
    }
  }
  // One column whose norm is about sqrt(DBL_MAX), the same relation against the unit column.
  const auto one_column = [](double c) {
    return la::try_lstsq_full_rank_qr<double>(matrix_of<double>(2, 1, {c, c}),
                                              matrix_of<double>(2, 1, {1, 1}));
  };
  const std::optional<NdArray<double>> unit = one_column(1.0);
  const std::optional<NdArray<double>> huge = one_column(std::ldexp(1.0, 511));
  REQUIRE(unit.has_value());
  REQUIRE(huge.has_value());
  CHECK(std::ldexp(huge->data()[0], 511) == unit->data()[0]);
}

TEST_CASE("try_lstsq_full_rank_qr: a matrix with no columns has an empty solution") {
  const la::RowMatrix<double> a(3, 0);
  const la::RowMatrix<double> b = matrix_of<double>(3, 2, {1, 2, 3, 4, 5, 6});
  const std::optional<NdArray<double>> got = la::try_lstsq_full_rank_qr<double>(a, b);
  REQUIRE(got.has_value());
  CHECK(got->shape()[0] == 0);
  CHECK(got->shape()[1] == 2);
}
