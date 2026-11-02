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

}  // namespace pantr::linalg
