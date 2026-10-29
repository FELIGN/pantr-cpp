/// @file dense.hpp
/// @brief A thin front over Eigen for the dense linear algebra pantr needs (design §6).
///
/// Each function reproduces the numpy decision rule its Python predecessor relied on, because
/// the outcomes the suite asserts are discrete (raised or not), not merely numeric.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <Eigen/Core>
#include <Eigen/LU>

#include <cmath>
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

}  // namespace pantr::linalg
