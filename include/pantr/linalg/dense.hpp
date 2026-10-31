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
///     `a` and `b` are assumed to have the same number of rows (no validation performed).
template <StorageFloat T>
[[nodiscard]] NdArray<T> lstsq_min_norm(const RowMatrix<T>& a, const RowMatrix<T>& b) {
  using MatD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  constexpr const char* failure = "SVD did not converge in Linear Least Squares";
  if (!a.allFinite()) {
    throw value_error(failure);
  }
  const Eigen::BDCSVD<MatD, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(
      a.template cast<double>());
  if (svd.info() != Eigen::Success) {
    throw value_error(failure);
  }
  const auto& s = svd.singularValues();
  const Eigen::Index cols = a.cols();
  // numpy's default rcond, then gelsd's comparison against the largest singular value.
  const double rcond =
      std::numeric_limits<double>::epsilon() * static_cast<double>((std::max)(a.rows(), cols));
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

}  // namespace pantr::linalg
