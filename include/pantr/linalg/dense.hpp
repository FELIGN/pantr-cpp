/// @file dense.hpp
/// @brief A thin front over Eigen for the dense linear algebra pantr needs (design §6).
///
/// Each function reproduces the numpy decision rule its Python predecessor relied on, because
/// the outcomes the suite asserts are discrete (raised or not), not merely numeric.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>

namespace pantr::linalg {

/// A dense row-major Eigen matrix of `T`.
template <StorageFloat T>
using RowMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// View a rank-2 `NdArray` as an Eigen matrix (no copy).
///
/// Args:
///     a (const NdArray<T>&): Extents `(rows, cols)`.
///
/// Returns:
///     Eigen::Map<const RowMatrix<T>>: A read-only map over `a`'s elements.
template <StorageFloat T>
[[nodiscard]] Eigen::Map<const RowMatrix<T>> as_matrix(const NdArray<T>& a) {
  return {a.data(), static_cast<Eigen::Index>(a.shape()[0]),
          static_cast<Eigen::Index>(a.shape()[1])};
}

/// Invert a square matrix by LU with partial pivoting, as numpy.linalg.inv and solve do (gesv).
///
/// The explicit inverse solves `A X = I` (Du Croz and Higham's Method A), which bounds the
/// residual `A X - I`, not `X A - I`.
///
/// Args:
///     a (const NdArray<T>&): Square, extents `(m, m)`.
///
/// Returns:
///     std::optional<NdArray<T>>: The inverse, or no value when a pivot of the factorisation is
///     exactly zero (the condition under which gesv reports a singular matrix and numpy raises
///     LinAlgError) or NaN (where gesv raises only in some positions; refusing it everywhere
///     keeps the outcome independent of where the NaN sits).
///
/// Note:
///     `a` is assumed square (no validation performed).
template <StorageFloat T>
[[nodiscard]] std::optional<NdArray<T>> try_inverse(const NdArray<T>& a) {
  const auto m = static_cast<Eigen::Index>(a.shape()[0]);
  const Eigen::PartialPivLU<RowMatrix<T>> lu(as_matrix(a));
  const auto& factors = lu.matrixLU();
  for (Eigen::Index i = 0; i < m; ++i) {
    const T pivot = factors(i, i);
    if (pivot == static_cast<T>(0) || std::isnan(pivot)) {
      return std::nullopt;
    }
  }
  NdArray<T> out({a.shape()[0], a.shape()[1]}, for_overwrite);
  Eigen::Map<RowMatrix<T>>(out.data(), m, m) = lu.solve(RowMatrix<T>::Identity(m, m));
  return out;
}

/// The rank cutoff of `numpy.linalg.lstsq(rcond=None)` relative to the largest singular value.
///
/// Args:
///     rows (Eigen::Index): Rows of the matrix.
///     cols (Eigen::Index): Columns.
///
/// Returns:
///     double: `eps(double) * max(rows, cols)` (numpy `_linalg.py`, `lstsq`).
[[nodiscard]] inline double lstsq_rcond(Eigen::Index rows, Eigen::Index cols) {
  return std::numeric_limits<double>::epsilon() * static_cast<double>((std::max)(rows, cols));
}

namespace detail {

/// The message numpy raises when `gelsd` does not converge, which `lstsq_min_norm` and
/// `lstsq_min_norm_jacobi` raise for a non-finite input or a failed SVD.
inline constexpr const char* lstsq_failure = "SVD did not converge in Linear Least Squares";

/// Solve `A X = B` over the singular values above numpy's cutoff, given the thin SVD of `A`.
///
/// Args:
///     svd (const Svd&): A converged thin SVD of `A` (`Eigen::BDCSVD` or `Eigen::JacobiSVD` of a
///         `double` matrix), singular values in decreasing order.
///     b (const RowMatrix<T>&): Extents `(m, k)`.
///     rows (Eigen::Index): `m`, the rows of `A`.
///     cols (Eigen::Index): `n`, the columns of `A`.
///
/// Returns:
///     NdArray<T>: Extents `(n, k)`: `V diag(1 / s_i) U^T B` over the kept values, `s_i > rcond *
///     s_max` with `rcond = lstsq_rcond(m, n)`, rounded to `T`.
template <StorageFloat T, class Svd>
[[nodiscard]] NdArray<T> lstsq_from_svd(const Svd& svd, const RowMatrix<T>& b, Eigen::Index rows,
                                        Eigen::Index cols) {
  using MatD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  const auto& s = svd.singularValues();
  // numpy's default rcond, then gelsd's comparison against the largest singular value.
  const double rcond = lstsq_rcond(rows, cols);
  const double cutoff = s.size() > 0 ? rcond * s(0) : 0.0;
  Eigen::Index rank = 0;
  while (rank < s.size() && s(rank) > cutoff) {
    ++rank;
  }
  MatD coeffs = svd.matrixU().leftCols(rank).transpose() * b.template cast<double>();
  for (Eigen::Index i = 0; i < rank; ++i) {
    coeffs.row(i) /= s(i);
  }
  const MatD x = svd.matrixV().leftCols(rank) * coeffs;
  NdArray<T> out({static_cast<std::size_t>(cols), static_cast<std::size_t>(b.cols())},
                 for_overwrite);
  Eigen::Map<RowMatrix<T>>(out.data(), cols, b.cols()) = x.template cast<T>();
  return out;
}

/// The relative margin `tau` of the rank certificate of a QR factorization, for the factorization's
/// backward error.
///
/// The certificate (the argument is made at `try_lstsq_banded`) keeps the QR path when `1 /
/// ||R^-1||_F > tau ||A||_F`. `tau = max(sqrt(eps), 4 (growth eps + cutoff))`: where `growth eps`
/// bounds the factorization's backward error `gamma` (`||E||_2 <= gamma ||A||_F`), the second term
/// gives the argument's hypothesis `gamma + cutoff <= tau / 4`; the floor `sqrt(eps)` gives it
/// where `growth` does not bound `gamma` (see `try_lstsq_banded` for where that holds).
///
/// Args:
///     growth (double): The backward-error bound in units of `eps`.
///     cutoff (double): The relative singular-value cutoff of the dense path the answer stands for.
///
/// Returns:
///     std::optional<double>: `tau`, or no value for a NaN, infinite or negative `cutoff` (no
///     certificate stands for it).
[[nodiscard]] inline std::optional<double> rank_certificate_tau(double growth, double cutoff) {
  if (!(cutoff >= 0.0) || !std::isfinite(cutoff)) {
    return std::nullopt;
  }
  constexpr double eps = std::numeric_limits<double>::epsilon();
  return (std::max)(std::sqrt(eps), 4.0 * (growth * eps) + 4.0 * cutoff);
}

/// The test of the rank certificate: `||R^-1||_F tau ||A||_F < 1`, written so that a NaN or an
/// overflow to infinity fails it.
///
/// Args:
///     inv_sq (double): `||R^-1||_F^2` of the computed factor.
///     a_sq (double): `||A||_F^2`.
///     tau (double): `rank_certificate_tau`.
///
/// Returns:
///     bool: Whether the certificate holds.
[[nodiscard]] inline bool rank_certificate_holds(double inv_sq, double a_sq, double tau) {
  return std::sqrt(inv_sq) * tau * std::sqrt(a_sq) < 1.0;
}

}  // namespace detail

/// Solve `A X = B` in the least-squares sense with the minimum-norm solution, as
/// `numpy.linalg.lstsq(a, b, rcond=None)` does (LAPACK `gelsd`).
///
/// numpy decomposes in `double` whatever the storage type and rounds the solution to it, with the
/// default cutoff `rcond = eps(double) * max(m, n)`: singular values `s_i <= rcond * s_max` count
/// as zero (`gelsd`'s rule; numpy `_linalg.py`, `lstsq`). Eigen's `BDCSVD` stands for the
/// divide-and-conquer SVD, and the solution is `V diag(1 / s_i) U^T B` over the kept values, which
/// is `gelsd`'s in exact arithmetic. The rank decision at the cutoff can flip with last bits; the
/// callers in this library solve full-rank changes of representation, far from it.
///
/// Args:
///     a (const RowMatrix<T>&): Extents `(m, n)`.
///     b (const RowMatrix<T>&): Extents `(m, k)`.
///
/// Returns:
///     NdArray<T>: Extents `(n, k)`.
///
/// Raises:
///     value_error: `a` has a NaN or infinite entry, or the SVD fails, with numpy's message ("SVD
///         did not converge in Linear Least Squares"); Eigen would return NaN singular values
///         that the cutoff would then drop, a silent wrong answer.
///
/// Note:
///     `a` and `b` are assumed to have the same number of rows (no validation performed). Use
///     `lstsq_min_norm_jacobi` where the caller's check assumes a `gelsd`-class backward error.
template <StorageFloat T>
[[nodiscard]] NdArray<T> lstsq_min_norm(const RowMatrix<T>& a, const RowMatrix<T>& b) {
  using MatD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  if (!a.allFinite()) {
    throw value_error(detail::lstsq_failure);
  }
  const Eigen::BDCSVD<MatD, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(
      a.template cast<double>());
  if (svd.info() != Eigen::Success) {
    throw value_error(detail::lstsq_failure);
  }
  return detail::lstsq_from_svd<T>(svd, b, a.rows(), a.cols());
}

/// Solve `A X = B` as `lstsq_min_norm` does, with a one-sided Jacobi SVD in place of the
/// divide-and-conquer one.
///
/// It exists for a caller that checks the residual of the solve against a bound derived for a
/// `gelsd`-class backward error (a few `eps` times the size of the data). `BDCSVD`'s
/// divide-and-conquer path, used above its switch to the plain bidiagonal SVD for small matrices,
/// does not give that on every well-conditioned system: its reconstruction `A - U S V^T` can be
/// orders of magnitude above `eps`, so a solution `gelsd` computes to rounding is refused by such a
/// gate. `JacobiSVD` gives it on those systems. It is slower on large matrices (a QR
/// preconditioner, then sweeps of rotations), so `lstsq_min_norm` stays the default.
///
/// The cutoff, the error contract and the message are `lstsq_min_norm`'s; the code is shared
/// (`detail::lstsq_from_svd`).
///
/// Args:
///     a (const RowMatrix<T>&): Extents `(m, n)`.
///     b (const RowMatrix<T>&): Extents `(m, k)`.
///
/// Returns:
///     NdArray<T>: Extents `(n, k)`.
///
/// Raises:
///     value_error: As `lstsq_min_norm`.
///
/// Note:
///     `a` and `b` are assumed to have the same number of rows (no validation performed).
template <StorageFloat T>
[[nodiscard]] NdArray<T> lstsq_min_norm_jacobi(const RowMatrix<T>& a, const RowMatrix<T>& b) {
  using MatD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  if (!a.allFinite()) {
    throw value_error(detail::lstsq_failure);
  }
  const Eigen::JacobiSVD<MatD, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(
      a.template cast<double>());
  if (svd.info() != Eigen::Success) {
    throw value_error(detail::lstsq_failure);
  }
  return detail::lstsq_from_svd<T>(svd, b, a.rows(), a.cols());
}

/// Solve `A X = B` by Householder QR with column pivoting where `A` is shown to have full column
/// rank, which makes the answer `lstsq_min_norm`'s up to rounding.
///
/// The cost is `2 m n^2 - 2 n^3 / 3` operations (Golub and Van Loan, Matrix Computations, 4th ed.,
/// section 5.2.1), against the several sweeps of `lstsq_min_norm_jacobi`. The answer is given when
/// `A` and `B` are finite, `m >= n`, and the certificate of `try_lstsq_banded` holds, `||R^-1||_F
/// tau ||A'||_F < 1` with `tau` from `detail::rank_certificate_tau` and `growth = 24 m n`, for the
/// computed factor `R` of `A' = 2^-e A`. That is `A` scaled exactly by the power of two bringing
/// its largest entry into `[1/2, 1)`; the solution is scaled back by `2^-e`. In the normal range
/// every operation of the factorization commutes with that scaling, so the answer is the unscaled
/// one bit for bit; what the scaling changes is below it.
///
/// The argument is `try_lstsq_banded`'s, with Householder QR's backward error for `gamma`. Higham,
/// Accuracy and Stability of Numerical Algorithms, 2nd ed., Theorem 19.4, gives `||da_j||_2 <=
/// gamma~_{m n} ||a_j||_2` columnwise; it is stated for unpivoted QR and carries over to the
/// pivoted `A' P` because its Lemma 19.3 holds for Householder matrices that "need have no
/// connection to the matrix to which [they are] applied". The theorem leaves the constant
/// unspecified; a first-order count of Eigen's own construction and update gives `(6 m - 3 n + 34)
/// n u <= 24 m n eps` for all `m >= n >= 1` (`u = eps / 2`), whence `growth`: a hand count checked
/// against an emulation of Eigen's sequence, not a published constant, so this is argued and not
/// proved. So is the inverse's backward error: Eigen's triangular solve multiplies by reciprocals,
/// where Higham's Lemma 8.4 assumes division.
///
/// Higham's model ignores underflow, and Eigen's Householder construction does not: a reflector
/// whose tail has a squared norm at most `DBL_MIN` is skipped, the tail deleted rather than
/// annihilated. That perturbs `A` by up to `sqrt(DBL_MIN)` per column, outside `gamma ||A||_F`, and
/// without the scaling the exactly rank-1 `[[2^-500, 2^-500], [1.5 2^-512, 1.5 2^-512]]` was
/// certified. After it `||A'||_F >= 1/2`, and the skip's perturbation (at most `sqrt(n DBL_MIN)`)
/// and the scaling's own rounding of an entry it carries below the normal range (at most `2^-1075`
/// each) lie far inside the argument's margin `(tau / 2) ||A'||_F`. The scaling also keeps the
/// squared norms of the test from overflowing or underflowing, which declined well-conditioned
/// matrices beyond `2^511` or below `2^-511`.
///
/// Everything is computed in `double` and the solution rounded to `T`, as `lstsq_min_norm` does.
/// No answer is given for a non-finite `A` or `B`, for fewer rows than columns, or where the test
/// fails (a rank-deficient or nearly rank-deficient `A`): the caller then runs the SVD path.
///
/// Args:
///     a (const RowMatrix<T>&): Extents `(m, n)`.
///     b (const RowMatrix<T>&): Extents `(m, k)`.
///     rcond (std::optional<double>): The relative cutoff of the dense path the answer stands for;
///         `lstsq_rcond(m, n)` when absent. A NaN, infinite or negative one gets no answer.
///
/// Returns:
///     std::optional<NdArray<T>>: Extents `(n, k)`, or no value where the QR's answer is not shown
///     to be `lstsq_min_norm`'s.
///
/// Note:
///     `a` and `b` are assumed to have the same number of rows (no validation performed).
template <StorageFloat T>
[[nodiscard]] std::optional<NdArray<T>> try_lstsq_full_rank_qr(
    const RowMatrix<T>& a, const RowMatrix<T>& b, std::optional<double> rcond = std::nullopt) {
  using MatD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  const Eigen::Index m = a.rows();
  const Eigen::Index n = a.cols();
  const std::optional<double> tau = detail::rank_certificate_tau(
      24.0 * static_cast<double>(m) * static_cast<double>(n), rcond ? *rcond : lstsq_rcond(m, n));
  if (!tau || m < n || !a.allFinite() || !b.allFinite()) {
    return std::nullopt;
  }
  if (n == 0) {
    // No unknowns: the empty solution, trivially of full column rank. Eigen's pivoted QR reads the
    // first pivot of an empty matrix.
    return NdArray<T>({std::size_t{0}, static_cast<std::size_t>(b.cols())}, for_overwrite);
  }
  // `A' = 2^-e A` with the largest `|a'_ij|` in `[1/2, 1)` (exact, `ldexp`); see above.
  int e = 0;
  static_cast<void>(std::frexp(static_cast<double>(a.cwiseAbs().maxCoeff()), &e));
  const MatD ad = a.template cast<double>().unaryExpr([e](double v) { return std::ldexp(v, -e); });
  const Eigen::ColPivHouseholderQR<MatD> qr(ad);
  const auto r = qr.matrixQR().topLeftCorner(n, n).template triangularView<Eigen::Upper>();
  const MatD r_inv = r.solve(MatD::Identity(n, n));
  if (!detail::rank_certificate_holds(r_inv.squaredNorm(), ad.squaredNorm(), *tau)) {
    return std::nullopt;
  }
  // `A P = Q R`, so `x = P R^-1 (Q^T b)` over the leading `n` rows.
  MatD y = qr.householderQ().adjoint() * b.template cast<double>();
  MatD top = y.topRows(n);
  r.solveInPlace(top);
  const MatD x =
      (qr.colsPermutation() * top).unaryExpr([e](double v) { return std::ldexp(v, -e); });
  NdArray<T> out({static_cast<std::size_t>(n), static_cast<std::size_t>(b.cols())}, for_overwrite);
  Eigen::Map<RowMatrix<T>>(out.data(), n, b.cols()) = x.template cast<T>();
  return out;
}

}  // namespace pantr::linalg
