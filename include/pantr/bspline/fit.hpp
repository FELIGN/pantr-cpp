/// @file fit.hpp
/// @brief B-spline fitting by collocation: interpolation and least-squares fits on tensor-product
/// and scattered nodes.
///
/// On a tensor-product grid of nodes the collocation system is a Kronecker product of one matrix
/// per direction, so it is solved one direction at a time (each solve along one axis of the value
/// tensor). At scattered points the full collocation matrix (a row-wise Kronecker product of the
/// univariate ones) is solved once.
///
/// Every solve follows the previous release's rule (NumPy's `linalg.solve`, then a truncated SVD
/// pseudo-inverse): a square system is solved by LU with partial pivoting, a rectangular one, or a
/// square one whose LU meets an exactly zero pivot, by the truncated SVD that drops the singular
/// values `s <= tol * s_max`. The decompositions run in `double` and their results are rounded to
/// the computation type, as NumPy's did. A collocation matrix is banded (each row holds the
/// `degree + 1` functions supported at its node), so a square system is LU-factored within its
/// band (LAPACK's `gbtrf` algorithm, whose pivots are the dense algorithm's; a periodic wrap is
/// held as dense trailing columns) where the band is narrow, in `O(n w (w + c))`, and a
/// rectangular system is first offered to
/// `linalg::try_lstsq_banded`, which answers only where its certificate shows full column rank
/// well clear of the cutoff the dense path would apply; there the dense path has the same unique
/// solution in exact arithmetic, and the banded QR computes it in `O(m w (w + c))` instead of
/// `O(m n^2)`, plus `O(n^2 w)` for its certificate. The matrices are stored dense, as they were,
/// so every solve still scans its matrix once or more (`O(m n)`).
#pragma once

#include "pantr/bezier/detail/along_axis.hpp"
#include "pantr/bezier/fit.hpp"
#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/linalg/banded_lstsq.hpp"
#include "pantr/linalg/dense.hpp"
#include "pantr/quad/points_lattice.hpp"

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bspline {

/// Boundary derivative constraints of one direction of an interpolation: the derivatives of
/// orders `1..left` at the domain start and `1..right` at its end are set to zero, each replacing
/// one collocation row (rows `1..left` and `m - 1 - right..m - 2`). A count `<= 0` adds none.
///
/// Attributes:
///     left (long long): Derivative orders constrained at the domain start.
///     right (long long): Derivative orders constrained at the domain end.
struct BoundaryDerivatives {
  long long left = 0;
  long long right = 0;
};

namespace detail {

/// The rows `[start, stop)` of `m` that the Python slices `[1 : 1 + left]` and
/// `[-1 - right : -1]` select, as NumPy resolves them.
///
/// Attributes:
///     start (std::size_t): First row.
///     stop (std::size_t): One past the last row, `>= start`.
struct RowRange {
  std::size_t start = 0;
  std::size_t stop = 0;
};

/// The rows replaced by the `left` derivative rows: Python's `[1 : 1 + left]` on `m` rows.
[[nodiscard]] inline RowRange left_rows(std::size_t m, long long left) {
  const std::size_t start = (std::min)(std::size_t{1}, m);
  const auto want = static_cast<unsigned long long>(left) + 1ULL;
  const std::size_t stop = (std::max)(
      start, static_cast<std::size_t>((std::min)(want, static_cast<unsigned long long>(m))));
  return {start, stop};
}

/// The rows replaced by the `right` derivative rows: Python's `[-1 - right : -1]` on `m` rows.
[[nodiscard]] inline RowRange right_rows(std::size_t m, long long right) {
  if (m == 0) {
    return {0, 0};
  }
  const std::size_t stop = m - 1;
  const auto back = static_cast<unsigned long long>(right) + 1ULL;
  const std::size_t start =
      back >= static_cast<unsigned long long>(m) ? 0 : m - static_cast<std::size_t>(back);
  return {(std::min)(start, stop), stop};
}

/// NumPy's error for assigning `rows` rows to a slice of `slots` rows of a `cols`-wide matrix.
[[noreturn]] inline void broadcast_rows_error(unsigned long long rows, std::size_t slots,
                                              std::size_t cols) {
  throw value_error("could not broadcast input array from shape (" + std::to_string(rows) + "," +
                    std::to_string(cols) + ") into shape (" + std::to_string(slots) + "," +
                    std::to_string(cols) + ")");
}

/// Overwrite rows `range` of `mat` with the derivatives of orders `1..count` of the basis at `x`.
///
/// One derivative row into an empty range (a direction with a single node) writes nothing, as
/// NumPy's assignment broadcasts one row onto zero.
///
/// Raises:
///     value_error: `count` differs from the number of rows in `range`, and is not one row
///         assigned to none (NumPy's broadcast message).
///     std::domain_error: The derivative recurrence met a zero divisor (an empty clamped span at
///         that domain end, as an unclamped knot vector repeating its end knot has); the binding
///         raises `ZeroDivisionError`, as the previous implementation's float division did.
template <StorageFloat T, StorageFloat P>
void put_derivative_rows(const BsplineSpace1D<T>& space, P x, long long count, RowRange range,
                         NdArray<P>& mat) {
  const std::size_t n = mat.shape()[1];
  const std::size_t slots = range.stop - range.start;
  const bool one_onto_none = count == 1 && slots == 0;
  if (static_cast<unsigned long long>(count) != slots && !one_onto_none) {
    broadcast_rows_error(static_cast<unsigned long long>(count), slots, n);
  }
  // Evaluated even when nothing is written: the previous release formed the row before assigning
  // it, so a zero divisor raised there too.
  const BasisDerivatives<P> d = space.template tabulate_basis_derivatives<P>(
      std::span<const P>(&x, 1), static_cast<int>(count));
  if (!d.complete) {
    throw std::domain_error("division by zero");
  }
  const auto order = static_cast<std::size_t>(space.degree()) + 1;
  const auto first = static_cast<std::size_t>(d.first.data()[0]);
  // A non-periodic row's `order` functions start at `first` and end inside the matrix.
  for (std::size_t k = 0; k < slots; ++k) {
    P* row = mat.data() + (range.start + k) * n;
    std::fill(row, row + n, P(0));
    const P* src = d.values.data() + (k + 1) * order;  // point 0, order k + 1
    std::copy(src, src + order, row + first);
  }
}

/// The trailing border a periodic wrap needs: where some row's nonzeros span more than half the
/// columns, the leading columns that row reaches below half the width, moved to the end of the
/// matrix, turn its run into a short one plus border columns.
///
/// The choice only affects the cost of `linalg::try_lstsq_banded`, never its answer (a column
/// permutation leaves the least-squares solution and the singular values unchanged); it is a
/// function of the nonzero pattern alone, so every caller handing the same matrix gets the same
/// rounding.
template <StorageFloat C>
[[nodiscard]] Eigen::Index wrap_border(const linalg::RowMatrix<C>& a) {
  const Eigen::Index n = a.cols();
  const Eigen::Index half = n / 2;
  Eigen::Index border = 0;
  for (Eigen::Index i = 0; i < a.rows(); ++i) {
    Eigen::Index first = n;
    Eigen::Index last = -1;
    Eigen::Index lead = 0;
    for (Eigen::Index j = 0; j < n; ++j) {
      if (a(i, j) != C(0)) {
        first = (std::min)(first, j);
        last = j;
        if (j < half) {
          lead = j + 1;
        }
      }
    }
    if (last >= 0 && last - first + 1 > half) {
      border = (std::max)(border, lead);
    }
  }
  return border;
}

/// `linalg::try_lstsq_banded` with the first `border` columns of `a` held dense at the end.
template <StorageFloat C>
[[nodiscard]] std::optional<NdArray<C>> try_banded(const linalg::RowMatrix<C>& a,
                                                   const linalg::RowMatrix<C>& b, double rcond) {
  const Eigen::Index border = wrap_border<C>(a);
  if (border == 0) {
    return linalg::try_lstsq_banded<C>(a, b, 0, rcond);
  }
  const Eigen::Index n = a.cols();
  const Eigen::Index rest = n - border;
  linalg::RowMatrix<C> moved(a.rows(), n);
  moved.leftCols(rest) = a.rightCols(rest);
  moved.rightCols(border) = a.leftCols(border);
  std::optional<NdArray<C>> y = linalg::try_lstsq_banded<C>(moved, b, border, rcond);
  if (!y) {
    return std::nullopt;
  }
  // Undo the permutation: unknown `j >= border` sits at row `j - border`, `j < border` at the end.
  const std::size_t cols = y->shape()[1];
  NdArray<C> x({static_cast<std::size_t>(n), cols}, for_overwrite);
  const auto nb = static_cast<std::size_t>(border);
  const auto nr = static_cast<std::size_t>(rest);
  std::copy(y->data() + nr * cols, y->data() + (nr + nb) * cols, x.data());
  std::copy(y->data(), y->data() + nr * cols, x.data() + nb * cols);
  return x;
}

/// The band of a square matrix whose first `border` columns are moved to the end (the periodic
/// wrap, see `wrap_border`): over the other columns, renumbered from 0, `kl = max(i - first_i)`
/// and `ku = max(last_i - i)` over the rows' nonzeros.
///
/// Attributes:
///     kl (Eigen::Index): Lower bandwidth.
///     ku (Eigen::Index): Upper bandwidth.
///     border (Eigen::Index): The columns held dense at the end.
struct Bandwidths {
  Eigen::Index kl = 0;
  Eigen::Index ku = 0;
  Eigen::Index border = 0;
};

/// Measure the band of `a` with `border` leading columns moved to the end.
///
/// Args:
///     a (const linalg::RowMatrix<C>&): Square.
///     border (Eigen::Index): Leading columns held dense, `< a.cols()`.
///
/// Returns:
///     Bandwidths: The band of the other columns.
template <StorageFloat C>
[[nodiscard]] Bandwidths bandwidths(const linalg::RowMatrix<C>& a, Eigen::Index border) {
  Bandwidths bw;
  bw.border = border;
  const Eigen::Index n = a.cols();
  for (Eigen::Index i = 0; i < a.rows(); ++i) {
    Eigen::Index first = border;
    while (first < n && a(i, first) == C(0)) {
      ++first;
    }
    if (first == n) {
      continue;  // nonzeros in the border only, or none: no constraint on the band
    }
    Eigen::Index last = n - 1;
    while (a(i, last) == C(0)) {
      --last;
    }
    bw.kl = (std::max)(bw.kl, i - (first - border));
    bw.ku = (std::max)(bw.ku, (last - border) - i);
  }
  return bw;
}

/// Whether the banded LU is cheaper than the dense one: the band, with the fill-in partial
/// pivoting adds (`kl` more columns above the diagonal), and the border cover well under the
/// matrix's width.
///
/// Args:
///     bw (Bandwidths): The band.
///     n (Eigen::Index): The matrix's order.
///
/// Returns:
///     bool: Whether `4 (2 kl + ku + 1 + border) < n`. The factor 4 is an admitted heuristic (not
///     measured): it keeps the banded loops to a quarter of a dense row, where the dense
///     factorisation's blocked kernels have no row length left to amortise.
[[nodiscard]] inline bool band_pays(Bandwidths bw, Eigen::Index n) {
  return 4 * (2 * bw.kl + bw.ku + 1 + bw.border) < n;
}

/// LU with partial pivoting of a banded square matrix whose first `bw.border` columns are moved to
/// the end and held dense (LAPACK's `gbtrf` algorithm on the band, then a dense trailing block):
/// the row chosen at step `k` is the first one of largest magnitude in column `k` among the rows
/// that can hold it, which is the dense algorithm's choice on the permuted matrix, since the other
/// rows hold zeros there. Computed in `double`; a non-finite entry of `a` sends the system to the
/// dense path, as the factorisation would spread it there differently.
///
/// Args:
///     a (const linalg::RowMatrix<C>&): Square, `(n, n)`.
///     b (const linalg::RowMatrix<C>&): `(n, cols)`.
///     bw (Bandwidths): The band of `a` (see `bandwidths`).
///
/// Returns:
///     std::optional<NdArray<C>>: `(n, cols)`, or no value where `a` is not finite or a pivot is
///     exactly zero or NaN (the condition under which `gesv` reports a singular matrix).
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): `a` square with the band `bw`.
template <StorageFloat C>
[[nodiscard]] std::optional<NdArray<C>> banded_lu_solve(const linalg::RowMatrix<C>& a,
                                                        const linalg::RowMatrix<C>& b,
                                                        Bandwidths bw) {
  if (!a.allFinite()) {
    return std::nullopt;
  }
  const Eigen::Index n = a.cols();
  const Eigen::Index nb = bw.border;
  const Eigen::Index n0 = n - nb;  // banded columns, renumbered `j - nb`
  const Eigen::Index kl = bw.kl;
  const Eigen::Index reach = bw.ku + kl;  // U's upper bandwidth after pivoting
  const Eigen::Index width = kl + reach + 1;
  Eigen::MatrixXd band = Eigen::MatrixXd::Zero(n, width);
  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(n, nb);
  const auto at = [&band, kl](Eigen::Index i, Eigen::Index j) -> double& {
    return band(i, j - i + kl);
  };
  for (Eigen::Index i = 0; i < n; ++i) {
    const Eigen::Index lo = (std::max)(Eigen::Index{0}, i - kl);
    const Eigen::Index hi = (std::min)(n0 - 1, i + bw.ku);
    for (Eigen::Index j = lo; j <= hi; ++j) {
      at(i, j) = static_cast<double>(a(i, j + nb));
    }
    for (Eigen::Index t = 0; t < nb; ++t) {
      dense(i, t) = static_cast<double>(a(i, t));
    }
  }
  Eigen::MatrixXd x = b.template cast<double>();
  const Eigen::Index cols = x.cols();
  // The banded columns.
  for (Eigen::Index k = 0; k < n0; ++k) {
    const Eigen::Index last_row = (std::min)(n - 1, k + kl);
    const Eigen::Index last_col = (std::min)(n0 - 1, k + reach);
    Eigen::Index piv = k;
    double best = std::abs(at(k, k));
    for (Eigen::Index r = k + 1; r <= last_row; ++r) {
      const double v = std::abs(at(r, k));
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (piv != k) {
      for (Eigen::Index j = k; j <= last_col; ++j) {
        std::swap(at(k, j), at(piv, j));
      }
      dense.row(k).swap(dense.row(piv));
      x.row(k).swap(x.row(piv));
    }
    const double pivot = at(k, k);
    if (pivot == 0.0 || std::isnan(pivot)) {
      return std::nullopt;  // singular for gesv; the caller takes the SVD
    }
    for (Eigen::Index r = k + 1; r <= last_row; ++r) {
      const double l = at(r, k) / pivot;
      for (Eigen::Index j = k + 1; j <= last_col; ++j) {
        at(r, j) -= l * at(k, j);
      }
      for (Eigen::Index t = 0; t < nb; ++t) {
        dense(r, t) -= l * dense(k, t);
      }
      for (Eigen::Index c = 0; c < cols; ++c) {
        x(r, c) -= l * x(k, c);
      }
    }
  }
  // The dense trailing block: rows `n0..n-1` of the border columns.
  for (Eigen::Index t = 0; t < nb; ++t) {
    const Eigen::Index k = n0 + t;
    Eigen::Index piv = k;
    double best = std::abs(dense(k, t));
    for (Eigen::Index r = k + 1; r < n; ++r) {
      const double v = std::abs(dense(r, t));
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (piv != k) {
      dense.row(k).swap(dense.row(piv));
      x.row(k).swap(x.row(piv));
    }
    const double pivot = dense(k, t);
    if (pivot == 0.0 || std::isnan(pivot)) {
      return std::nullopt;
    }
    for (Eigen::Index r = k + 1; r < n; ++r) {
      const double l = dense(r, t) / pivot;
      for (Eigen::Index u = t + 1; u < nb; ++u) {
        dense(r, u) -= l * dense(k, u);
      }
      for (Eigen::Index c = 0; c < cols; ++c) {
        x(r, c) -= l * x(k, c);
      }
    }
  }
  // Back-substitution, the border unknowns first (they are the last rows of the permuted system).
  for (Eigen::Index t = nb; t-- > 0;) {
    const Eigen::Index i = n0 + t;
    for (Eigen::Index c = 0; c < cols; ++c) {
      double s = x(i, c);
      for (Eigen::Index u = t + 1; u < nb; ++u) {
        s -= dense(i, u) * x(n0 + u, c);
      }
      x(i, c) = s / dense(i, t);
    }
  }
  for (Eigen::Index i = n0; i-- > 0;) {
    const Eigen::Index last_col = (std::min)(n0 - 1, i + reach);
    for (Eigen::Index c = 0; c < cols; ++c) {
      double s = x(i, c);
      for (Eigen::Index j = i + 1; j <= last_col; ++j) {
        s -= at(i, j) * x(j, c);
      }
      for (Eigen::Index u = 0; u < nb; ++u) {
        s -= dense(i, u) * x(n0 + u, c);
      }
      x(i, c) = s / at(i, i);
    }
  }
  // Undo the column permutation: unknown `j >= nb` is row `j - nb`, unknown `j < nb` row `n0 + j`.
  NdArray<C> out({static_cast<std::size_t>(n), static_cast<std::size_t>(cols)}, for_overwrite);
  Eigen::Map<linalg::RowMatrix<C>> o(out.data(), n, cols);
  o.topRows(nb) = x.bottomRows(nb).template cast<C>();
  o.bottomRows(n0) = x.topRows(n0).template cast<C>();
  return out;
}

/// The truncated-SVD solve of the previous release: `V diag(s_inv) U^T b` with `s_inv = 1 / s`
/// where `s > tol * s_max` and zero elsewhere, the factors computed in `double` and rounded to
/// `C`, the rest in `C`.
///
/// Raises:
///     value_error: The SVD fails ("SVD did not converge"), which NumPy reports for a NaN or
///         infinite entry.
template <StorageFloat C>
[[nodiscard]] NdArray<C> truncated_svd_solve(const linalg::RowMatrix<C>& a,
                                             const linalg::RowMatrix<C>& b, double tol) {
  using MatC = Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic>;
  using MatD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  const Eigen::Index n = a.cols();
  NdArray<C> out({static_cast<std::size_t>(n), static_cast<std::size_t>(b.cols())});
  if (a.rows() == 0 || n == 0) {
    return out;  // nothing to keep: the zero solution
  }
  if (!a.allFinite()) {
    throw value_error("SVD did not converge");
  }
  const Eigen::BDCSVD<MatD, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(
      a.template cast<double>());
  if (svd.info() != Eigen::Success) {
    throw value_error("SVD did not converge");
  }
  const Eigen::Matrix<C, Eigen::Dynamic, 1> s = svd.singularValues().template cast<C>();
  const MatC u = svd.matrixU().template cast<C>();
  const MatC v = svd.matrixV().template cast<C>();
  const C threshold = static_cast<C>(tol) * s(0);
  Eigen::Matrix<C, Eigen::Dynamic, 1> s_inv(s.size());
  for (Eigen::Index i = 0; i < s.size(); ++i) {
    s_inv(i) = s(i) > threshold ? C(1) / s(i) : C(0);
  }
  const MatC ub = u.transpose() * b;
  const MatC x = v * (s_inv.asDiagonal() * ub);
  Eigen::Map<linalg::RowMatrix<C>>(out.data(), n, b.cols()) = x;
  return out;
}

}  // namespace detail

/// Solve one collocation system `A X = B` by the previous release's rule (see the file comment).
///
/// Args:
///     a (const linalg::RowMatrix<C>&): Extents `(m, n)`, finite.
///     b (const linalg::RowMatrix<C>&): Extents `(m, k)`.
///     tol (double): Relative truncation tolerance of the SVD path, compared as `s > tol * s_max`,
///         as before: NaN drops every singular value, and a negative value keeps every one, an
///         exactly zero one included (its reciprocal is then infinite).
///
/// Returns:
///     NdArray<C>: Extents `(n, k)`.
///
/// Raises:
///     value_error: The rows of `a` and `b` differ, or the SVD fails.
template <StorageFloat C>
[[nodiscard]] NdArray<C> solve_collocation(const linalg::RowMatrix<C>& a,
                                           const linalg::RowMatrix<C>& b, double tol) {
  const Eigen::Index m = a.rows();
  const Eigen::Index n = a.cols();
  if (b.rows() != m) {
    throw value_error("solve_collocation: the matrix has " + std::to_string(m) +
                      " rows and the right-hand side " + std::to_string(b.rows()) + ".");
  }
  if (m == n && n > 0) {
    // numpy.linalg.solve: LU with partial pivoting in double (gesv), the solution rounded to the
    // result type; an exactly zero or NaN pivot sends the system to the truncated SVD.
    // A non-finite right-hand side takes the dense factorisation, which spreads it over the
    // solution as LAPACK's does (through `0 * NaN`); the band would confine it to the band.
    const detail::Bandwidths bw = detail::bandwidths<C>(a, detail::wrap_border<C>(a));
    if (detail::band_pays(bw, n) && b.allFinite()) {
      if (std::optional<NdArray<C>> x = detail::banded_lu_solve<C>(a, b, bw)) {
        return std::move(*x);
      }
    } else {
      using MatD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
      const Eigen::PartialPivLU<MatD> lu(a.template cast<double>());
      const auto& factors = lu.matrixLU();
      bool singular = false;
      for (Eigen::Index i = 0; i < n && !singular; ++i) {
        const double pivot = factors(i, i);
        singular = pivot == 0.0 || std::isnan(pivot);
      }
      if (!singular) {
        NdArray<C> out({static_cast<std::size_t>(n), static_cast<std::size_t>(b.cols())},
                       for_overwrite);
        Eigen::Map<linalg::RowMatrix<C>>(out.data(), n, b.cols()) =
            lu.solve(b.template cast<double>()).template cast<C>();
        return out;
      }
    }
  } else if (m > n && n > 0 && std::isfinite(tol)) {
    // A rectangular system keeps every singular value above `tol * s_max` (a negative `tol` keeps
    // all), so the QR's certificate must clear that cutoff; a NaN or infinite one drops every
    // singular value, which no full-rank certificate can stand for.
    if (std::optional<NdArray<C>> x = detail::try_banded<C>(a, b, (std::max)(tol, 0.0))) {
      return std::move(*x);
    }
  }
  return detail::truncated_svd_solve<C>(a, b, tol);
}

/// Solve the Kronecker system `(A_{d-1} ⊗ ... ⊗ A_0) vec(X) = vec(R)` one direction at a time:
/// direction `k`'s system is solved along axis `k` of the current tensor.
///
/// Args:
///     matrices (const std::vector<NdArray<P>>&): `d` matrices, `(m_k, n_k)`.
///     rhs (std::span<const C>): Row-major over `shape`.
///     shape (std::span<const std::size_t>): `(m_0, ..., m_{d-1})`.
///     tol (double): Truncation tolerance of the SVD path (see `solve_collocation`).
///
/// Returns:
///     NdArray<C>: Extents `(n_0, ..., n_{d-1})`.
///
/// Raises:
///     value_error: Inconsistent extents, an empty right-hand side, or an SVD failure.
template <StorageFloat P, StorageFloat C>
[[nodiscard]] NdArray<C> solve_kronecker(const std::vector<NdArray<P>>& matrices,
                                         std::span<const C> rhs, std::span<const std::size_t> shape,
                                         double tol) {
  if (shape.size() != matrices.size() || checked_product(shape) != rhs.size()) {
    throw value_error("solve_kronecker: one matrix per axis of the right-hand side is needed.");
  }
  if (rhs.empty()) {
    throw value_error("solve_kronecker: the right-hand side is empty (a direction has no node).");
  }
  std::vector<std::size_t> cur(shape.begin(), shape.end());
  NdArray<C> current(std::vector<std::size_t>(cur), for_overwrite);
  std::copy(rhs.begin(), rhs.end(), current.data());
  for (std::size_t d = 0; d < matrices.size(); ++d) {
    const NdArray<P>& mat = matrices[d];
    if (mat.rank() != 2 || mat.shape()[0] != cur[d]) {
      throw value_error("solve_kronecker: matrix " + std::to_string(d) + " does not match axis " +
                        std::to_string(d) + " of the right-hand side.");
    }
    const auto m = static_cast<Eigen::Index>(mat.shape()[0]);
    const auto n = static_cast<Eigen::Index>(mat.shape()[1]);
    const linalg::RowMatrix<C> a =
        Eigen::Map<const linalg::RowMatrix<P>>(mat.data(), m, n).template cast<C>();
    const NdArray<C> flat = bezier::detail::flatten_along_axis<C>(current.flat(), cur, d);
    const auto cols = static_cast<Eigen::Index>(flat.shape()[1]);
    const linalg::RowMatrix<C> b = Eigen::Map<const linalg::RowMatrix<C>>(flat.data(), m, cols);
    const NdArray<C> solved = solve_collocation<C>(a, b, tol);
    current = bezier::detail::unflatten_along_axis<C>(solved.template view<2>(), cur, d);
    cur[d] = static_cast<std::size_t>(n);
  }
  return current;
}

/// The collocation matrix `C[i, j] = N_j(nodes[i])` of a 1D space, dense, in the nodes' type.
///
/// Periodic functions wrap around (`j mod num_basis`), and a function that appears twice in a
/// node's support (fewer functions than `degree + 1`) has its two values added.
///
/// Args:
///     space (const BsplineSpace1D<T>&): The space.
///     nodes (std::span<const P>): The nodes, in the domain.
///
/// Returns:
///     NdArray<P>: Extents `(len(nodes), num_basis)`.
///
/// Raises:
///     value_error: A node (NaN included) outside the domain.
template <StorageFloat T, StorageFloat P>
[[nodiscard]] NdArray<P> collocation_matrix(const BsplineSpace1D<T>& space,
                                            std::span<const P> nodes) {
  const auto [values, first] = space.template tabulate_basis<P>(nodes);
  const std::size_t m = nodes.size();
  const auto n = static_cast<std::size_t>(space.num_basis());
  const auto order = static_cast<std::size_t>(space.degree()) + 1;
  NdArray<P> mat({m, n});
  for (std::size_t i = 0; i < m; ++i) {
    const auto f = static_cast<std::size_t>(first.data()[i]);
    P* row = mat.data() + i * n;
    const P* v = values.data() + i * order;
    for (std::size_t a = 0; a < order; ++a) {
      if (space.periodic()) {
        row[(f + a) % n] += v[a];
      } else {
        row[f + a] = v[a];
      }
    }
  }
  return mat;
}

/// The collocation matrix with boundary derivative rows (see `BoundaryDerivatives`); a periodic
/// space ignores them.
///
/// Raises:
///     value_error: A node outside the domain, or more derivative rows than the rows they replace
///         (NumPy's broadcast message, as before).
///     std::domain_error: The derivative recurrence met a zero divisor (see `put_derivative_rows`).
template <StorageFloat T, StorageFloat P>
[[nodiscard]] NdArray<P> collocation_matrix(const BsplineSpace1D<T>& space,
                                            std::span<const P> nodes, BoundaryDerivatives bd) {
  NdArray<P> mat = collocation_matrix<T, P>(space, nodes);
  if (space.periodic()) {
    return mat;
  }
  const auto [a, b] = space.domain();
  const std::size_t m = nodes.size();
  if (bd.left > 0) {
    detail::put_derivative_rows<T, P>(space, static_cast<P>(a), bd.left,
                                      detail::left_rows(m, bd.left), mat);
  }
  if (bd.right > 0) {
    detail::put_derivative_rows<T, P>(space, static_cast<P>(b), bd.right,
                                      detail::right_rows(m, bd.right), mat);
  }
  return mat;
}

/// A tensor-product collocation system, built from the nodes before any value is known: the
/// staged half of `interpolate` and `fit_tensor` (design §3.6). Immutable.
///
/// Attributes:
///     matrices (std::vector<NdArray<P>>): One collocation matrix per direction, `(m_k, n_k)`,
///         with its boundary derivative rows.
///     zeroed (std::vector<std::vector<detail::RowRange>>): Per direction, the rows whose
///         right-hand side the derivative constraints set to zero.
template <StorageFloat P>
class TensorCollocation {
 public:
  /// Build the per-direction matrices.
  ///
  /// Args:
  ///     space (const BsplineSpace<T>&): The target space.
  ///     nodes (const std::vector<std::span<const P>>&): One node array per direction.
  ///     derivatives (const std::vector<std::optional<BoundaryDerivatives>>&): Per direction,
  ///         the constraints or none; directions past its end have none, and entries past the
  ///         dimension are ignored.
  ///
  /// Raises:
  ///     value_error: Not one node array per direction, a node outside the domain, or more
  ///         derivative rows than nodes allow.
  ///     std::domain_error: See `collocation_matrix`.
  template <StorageFloat T>
  TensorCollocation(const BsplineSpace<T>& space, const std::vector<std::span<const P>>& nodes,
                    const std::vector<std::optional<BoundaryDerivatives>>& derivatives = {}) {
    const std::size_t dim = space.dim();
    if (nodes.size() != dim) {
      throw value_error("Expected " + std::to_string(dim) + " node arrays, got " +
                        std::to_string(nodes.size()) + ".");
    }
    matrices_.reserve(dim);
    zeroed_.resize(dim);
    for (std::size_t d = 0; d < dim; ++d) {
      const BsplineSpace1D<T>& s = space.spaces()[d];
      const std::optional<BoundaryDerivatives> bd =
          d < derivatives.size() ? derivatives[d] : std::nullopt;
      if (bd && !s.periodic()) {
        matrices_.push_back(collocation_matrix<T, P>(s, nodes[d], *bd));
        const std::size_t m = nodes[d].size();
        if (bd->left > 0) {
          zeroed_[d].push_back(detail::left_rows(m, bd->left));
        }
        if (bd->right > 0) {
          zeroed_[d].push_back(detail::right_rows(m, bd->right));
        }
      } else {
        matrices_.push_back(collocation_matrix<T, P>(s, nodes[d]));
      }
    }
  }

  /// The per-direction matrices (numeric view of the held value).
  [[nodiscard]] const std::vector<NdArray<P>>& matrices() const noexcept { return matrices_; }

  /// The node grid's extents `(m_0, ..., m_{d-1})`.
  [[nodiscard]] std::vector<std::size_t> grid_shape() const {
    std::vector<std::size_t> g(matrices_.size());
    for (std::size_t d = 0; d < g.size(); ++d) {
      g[d] = matrices_[d].shape()[0];
    }
    return g;
  }

  /// Solve for the coefficients of values given on the node grid, one component at a time.
  ///
  /// Each component's right-hand side has the rows of its derivative constraints set to zero
  /// and is solved by `solve_kronecker` on its own, as the distributed fits in `pantr.mpi` solve
  /// them, so that both give the same bits. The price is that each component repeats the per-solve
  /// work (scans, QR, certificate).
  ///
  /// Args:
  ///     values (std::span<const C>): Row-major over `(m_0, ..., m_{d-1}, rank)`.
  ///     rank (std::size_t): Components, `>= 1`.
  ///     tol (double): Truncation tolerance of the SVD path (see `solve_collocation`).
  ///
  /// Returns:
  ///     NdArray<C>: Extents `(n_0, ..., n_{d-1}, rank)`.
  ///
  /// Raises:
  ///     value_error: `values` does not hold `rank` values per node, or an SVD fails.
  template <StorageFloat C>
  [[nodiscard]] NdArray<C> solve(std::span<const C> values, std::size_t rank, double tol) const {
    const std::vector<std::size_t> grid = grid_shape();
    const std::size_t points = checked_product(grid);
    // Divided rather than multiplied: `points * rank` can wrap for sizes past the `size_t` range.
    if (rank == 0 || values.size() % rank != 0 || values.size() / rank != points) {
      throw value_error("TensorCollocation::solve: expected " + std::to_string(points) + " x " +
                        std::to_string(rank) + " values, got " + std::to_string(values.size()) +
                        ".");
    }
    std::vector<std::size_t> out_shape(matrices_.size() + 1);
    for (std::size_t d = 0; d < matrices_.size(); ++d) {
      out_shape[d] = matrices_[d].shape()[1];
    }
    out_shape.back() = rank;
    NdArray<C> out(std::vector<std::size_t>(out_shape), for_overwrite);
    const std::size_t coeffs = out.size() / rank;
    std::vector<C> comp(points);
    for (std::size_t r = 0; r < rank; ++r) {
      for (std::size_t i = 0; i < points; ++i) {
        comp[i] = values[i * rank + r];
      }
      zero_constrained(comp, grid);
      const NdArray<C> x = solve_kronecker<P, C>(matrices_, std::span<const C>(comp), grid, tol);
      for (std::size_t i = 0; i < coeffs; ++i) {
        out.data()[i * rank + r] = x.data()[i];
      }
    }
    return out;
  }

 private:
  /// Zero the right-hand side on the rows of every derivative constraint.
  template <StorageFloat C>
  void zero_constrained(std::vector<C>& comp, const std::vector<std::size_t>& grid) const {
    for (std::size_t d = 0; d < zeroed_.size(); ++d) {
      for (const detail::RowRange& range : zeroed_[d]) {
        const bezier::detail::AxisSplit split = bezier::detail::split_around(grid, d);
        for (std::size_t o = 0; o < split.outer; ++o) {
          for (std::size_t a = range.start; a < range.stop; ++a) {
            C* row = comp.data() + (o * grid[d] + a) * split.inner;
            std::fill(row, row + split.inner, C(0));
          }
        }
      }
    }
  }

  std::vector<NdArray<P>> matrices_;
  std::vector<std::vector<detail::RowRange>> zeroed_;
};

/// The collocation matrix of a tensor-product space at scattered points: row `i` is the row-wise
/// Kronecker product of the univariate rows at `pts(i, :)`, the first direction slowest.
///
/// Args:
///     space (const BsplineSpace<T>&): The space.
///     pts (mdview<const P, 2>): `(n_pts, dim)` points in the domain.
///
/// Returns:
///     NdArray<P>: Extents `(n_pts, num_total_basis)`.
///
/// Raises:
///     value_error: The points have another number of columns, or one is outside the domain.
template <StorageFloat T, StorageFloat P>
[[nodiscard]] NdArray<P> scattered_collocation_matrix(const BsplineSpace<T>& space,
                                                      mdview<const P, 2> pts) {
  const std::size_t n_pts = pts.extent(0);
  const std::size_t dim = space.dim();
  if (pts.extent(1) != dim) {
    throw value_error("Scattered nodes have " + std::to_string(pts.extent(1)) +
                      " columns, expected " + std::to_string(dim) + ".");
  }
  NdArray<P> result({n_pts, 1});
  std::fill(result.data(), result.data() + n_pts, P(1));
  std::size_t width = 1;
  std::vector<P> col(n_pts);
  for (std::size_t d = 0; d < dim; ++d) {
    for (std::size_t i = 0; i < n_pts; ++i) {
      col[i] = pts(i, d);
    }
    const NdArray<P> mat = collocation_matrix<T, P>(space.spaces()[d], std::span<const P>(col));
    const std::size_t nd = mat.shape()[1];
    NdArray<P> next({n_pts, width * nd}, for_overwrite);
    for (std::size_t i = 0; i < n_pts; ++i) {
      const P* r = result.data() + i * width;
      const P* b = mat.data() + i * nd;
      P* o = next.data() + i * width * nd;
      for (std::size_t a = 0; a < width; ++a) {
        for (std::size_t c = 0; c < nd; ++c) {
          o[a * nd + c] = r[a] * b[c];
        }
      }
    }
    result = std::move(next);
    width *= nd;
  }
  return result;
}

/// Fit the coefficients of a tensor-product space to values at scattered points.
///
/// Args:
///     space (const BsplineSpace<T>&): The target space.
///     pts (mdview<const P, 2>): `(n_pts, dim)`, `n_pts >= num_total_basis`.
///     values (mdview<const C, 2>): `(n_pts, rank)`.
///     tol (double): Truncation tolerance of the SVD path (see `solve_collocation`).
///
/// Returns:
///     NdArray<C>: Extents `(n_0, ..., n_{d-1}, rank)`.
///
/// Raises:
///     value_error: Another number of columns, fewer points than basis functions (checked in
///         that order, before the domain), a point outside the domain, values with another number
///         of rows, or an SVD failure.
template <StorageFloat T, StorageFloat P, StorageFloat C>
[[nodiscard]] NdArray<C> fit_scattered_coefficients(const BsplineSpace<T>& space,
                                                    mdview<const P, 2> pts,
                                                    mdview<const C, 2> values, double tol) {
  const std::size_t n_pts = pts.extent(0);
  if (pts.extent(1) != space.dim()) {
    throw value_error("Scattered nodes have " + std::to_string(pts.extent(1)) +
                      " columns, expected " + std::to_string(space.dim()) + ".");
  }
  std::vector<std::size_t> shape;
  std::size_t total = 1;
  for (const BsplineSpace1D<T>& s : space.spaces()) {
    shape.push_back(static_cast<std::size_t>(s.num_basis()));
    total *= shape.back();
  }
  if (n_pts < total) {
    throw value_error("Underdetermined system: " + std::to_string(n_pts) + " points < " +
                      std::to_string(total) + " basis functions.");
  }
  const NdArray<P> colloc = scattered_collocation_matrix<T, P>(space, pts);
  if (values.extent(0) != n_pts) {
    throw value_error("Values have " + std::to_string(values.extent(0)) + " rows for " +
                      std::to_string(n_pts) + " points.");
  }
  const auto m = static_cast<Eigen::Index>(n_pts);
  const auto n = static_cast<Eigen::Index>(total);
  const auto rank = static_cast<Eigen::Index>(values.extent(1));
  const linalg::RowMatrix<C> a =
      Eigen::Map<const linalg::RowMatrix<P>>(colloc.data(), m, n).template cast<C>();
  const linalg::RowMatrix<C> b =
      Eigen::Map<const linalg::RowMatrix<C>>(values.data_handle(), m, rank);
  NdArray<C> x = solve_collocation<C>(a, b, tol);
  shape.push_back(values.extent(1));
  NdArray<C> out(std::move(shape), for_overwrite);
  std::copy(x.data(), x.data() + x.size(), out.data());
  return out;
}

/// The default truncation tolerance of a fit whose matrix has type `P`: `100 * eps(P)`.
template <StorageFloat P>
[[nodiscard]] double default_fit_tol(std::optional<double> tol) {
  return bezier::detail::resolve_svd_tol(tol,
                                         static_cast<double>(std::numeric_limits<P>::epsilon()));
}

/// Fit a B-spline to values on a tensor-product grid of nodes.
///
/// Args:
///     space (const BsplineSpace<T>&): The target space.
///     nodes (const std::vector<std::span<const T>>&): One node array per direction, in the domain.
///     values (std::span<const T>): Row-major over `(m_0, ..., m_{d-1}, rank)`.
///     rank (std::size_t): Components.
///     derivatives (const std::vector<std::optional<BoundaryDerivatives>>&): See
///         `TensorCollocation`. Defaults to none.
///     tol (std::optional<double>): Truncation tolerance; `100 * eps(T)` when absent.
///
/// Returns:
///     Bspline<T>: A non-rational B-spline on `space`.
///
/// Raises:
///     value_error: See `TensorCollocation` and its `solve`.
template <StorageFloat T>
[[nodiscard]] Bspline<T> fit_tensor(
    const BsplineSpace<T>& space, const std::vector<std::span<const T>>& nodes,
    std::span<const T> values, std::size_t rank,
    const std::vector<std::optional<BoundaryDerivatives>>& derivatives = {},
    std::optional<double> tol = std::nullopt) {
  const TensorCollocation<T> system(space, nodes, derivatives);
  return Bspline<T>(space, system.template solve<T>(values, rank, default_fit_tol<T>(tol)), false);
}

/// Fit a B-spline to values at scattered points (see `fit_scattered_coefficients`).
///
/// Args:
///     space (const BsplineSpace<T>&): The target space.
///     pts (mdview<const T, 2>): `(n_pts, dim)`.
///     values (mdview<const T, 2>): `(n_pts, rank)`.
///     tol (std::optional<double>): Truncation tolerance; `100 * eps(T)` when absent.
///
/// Returns:
///     Bspline<T>: A non-rational B-spline on `space`.
template <StorageFloat T>
[[nodiscard]] Bspline<T> fit_scattered(const BsplineSpace<T>& space, mdview<const T, 2> pts,
                                       mdview<const T, 2> values,
                                       std::optional<double> tol = std::nullopt) {
  return Bspline<T>(
      space, fit_scattered_coefficients<T, T, T>(space, pts, values, default_fit_tol<T>(tol)),
      false);
}

/// Interpolate a function at a tensor-product grid of nodes (by default the Greville abscissae):
/// the convenience form taking a C++ callable.
///
/// Args:
///     space (const BsplineSpace<T>&): The target space.
///     f (F&&): Called once as `f(lattice)` with the node lattice
///         (`const quad::PointsLattice<T>&`); returns an `NdArray<T>` of `num_points()` values
///         (scalar) or `(num_points(), rank)`, rows in the lattice's C order (last direction
///         fastest).
///     nodes (std::vector<std::vector<T>>): One node array per direction; the Greville
///         abscissae when empty.
///     derivatives (const std::vector<std::optional<BoundaryDerivatives>>&): See
///         `TensorCollocation`.
///     tol (std::optional<double>): Truncation tolerance; `100 * eps(T)` when absent.
///
/// Returns:
///     Bspline<T>: A non-rational B-spline on `space`.
///
/// Raises:
///     value_error: See `TensorCollocation`; `f` returned another number of values.
template <StorageFloat T, class F>
  requires std::invocable<F&, const quad::PointsLattice<T>&>
[[nodiscard]] Bspline<T> interpolate(
    const BsplineSpace<T>& space, F&& f, std::vector<std::vector<T>> nodes = {},
    const std::vector<std::optional<BoundaryDerivatives>>& derivatives = {},
    std::optional<double> tol = std::nullopt) {
  std::vector<std::vector<T>> per_dir = std::move(nodes);
  if (per_dir.empty()) {
    for (const BsplineSpace1D<T>& s : space.spaces()) {
      per_dir.push_back(s.greville());
    }
  }
  std::vector<std::span<const T>> spans(per_dir.begin(), per_dir.end());
  const TensorCollocation<T> system(space, spans, derivatives);
  const quad::PointsLattice<T> lattice(per_dir);
  const NdArray<T> values = f(lattice);
  const std::size_t points = lattice.num_points();
  const bool shaped = values.rank() == 1 || values.rank() == 2;
  if (!shaped || values.shape()[0] != points || (values.rank() == 2 && values.shape()[1] == 0)) {
    throw value_error("interpolate: the function returned another number of values than the " +
                      std::to_string(points) + " nodes.");
  }
  const std::size_t rank = values.rank() == 1 ? 1 : values.shape()[1];
  return Bspline<T>(space, system.template solve<T>(values.flat(), rank, default_fit_tol<T>(tol)),
                    false);
}

}  // namespace pantr::bspline
