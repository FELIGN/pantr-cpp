/// @file banded_lstsq.hpp
/// @brief `lstsq_min_norm` for a banded matrix of full column rank, by a banded Givens QR.
///
/// The changes of representation in `bspline/product.hpp` solve least-squares systems whose matrix
/// is a knot-insertion (Oslo) matrix or a product of two: each row's nonzeros sit in a run of
/// consecutive columns about the degree long, and a periodic wrap adds a few dense columns. A
/// dense SVD of such a matrix costs `O(m n^2)`. The row-sequential Givens QR here costs
/// `O(m (w + k) (w + k + c))` for a band of width `w`, `k` dense trailing columns and `c`
/// right-hand sides, and the rank certificate below `O(n^2 (w + k))`.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/linalg/dense.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

namespace pantr::linalg {

namespace detail {

/// An upper-triangular `R` stored as a band plus dense trailing columns, with `Q^T B` beside it.
///
/// Row `j` holds `R(j, j + t)` for `t < w` and `j + t < n0` in `band`, and `R(j, n0 + t)` for
/// `t < k` in `border`, where `n0 = n - k`. Rows of `A` are rotated in one at a time.
///
/// Attributes:
///     n (Eigen::Index): Columns of `A`.
///     n0 (Eigen::Index): Columns of the banded part, `n - k`.
///     w (Eigen::Index): Band width.
///     k (Eigen::Index): Dense trailing columns.
///     band (Eigen::MatrixXd): `(n, w)`.
///     border (Eigen::MatrixXd): `(n, k)`.
///     rhs (Eigen::MatrixXd): `(n, cols of B)`, the leading rows of `Q^T B`.
///     filled (std::vector<char>): Whether row `j` of `R` has been set.
struct BandedQr {
  Eigen::Index n;
  Eigen::Index n0;
  Eigen::Index w;
  Eigen::Index k;
  Eigen::MatrixXd band;
  Eigen::MatrixXd border;
  Eigen::MatrixXd rhs;
  std::vector<char> filled;

  /// Diagonal entry `R(j, j)`.
  [[nodiscard]] double diag(Eigen::Index j) const {
    return j < n0 ? band(j, 0) : border(j, j - n0);
  }

  /// Back-substitute `R x = y` in place for the rows `[0, last]` (the rows below are zero).
  ///
  /// Note:
  ///     Inputs are assumed to be correct (no validation performed): every row is filled.
  void back_substitute(Eigen::Ref<Eigen::MatrixXd> y, Eigen::Index last) const {
    for (Eigen::Index i = last; i >= 0; --i) {
      for (Eigen::Index c = 0; c < y.cols(); ++c) {
        double s = y(i, c);
        if (i < n0) {
          const Eigen::Index top = (std::min)(w, n0 - i);
          for (Eigen::Index t = 1; t < top; ++t) {
            s -= band(i, t) * y(i + t, c);
          }
        }
        for (Eigen::Index t = (std::max)(Eigen::Index{0}, i + 1 - n0); t < k; ++t) {
          s -= border(i, t) * y(n0 + t, c);
        }
        y(i, c) = s / diag(i);
      }
    }
  }
};

/// A Givens rotation `(c, s)`, applied to a pair of rows `(u, v)` as `(c u + s v, c v - s u)`.
///
/// Attributes:
///     c (double): Cosine.
///     s (double): Sine.
struct Rotation {
  double c;
  double s;
};

/// The Givens rotation that zeros `b` against `a`, for `(a, b)` not both zero.
[[nodiscard]] inline Rotation givens(double a, double b) {
  const double r = std::hypot(a, b);
  return {a / r, b / r};
}

}  // namespace detail

/// Solve `A X = B` by a banded Givens QR where that gives `lstsq_min_norm`'s answer up to rounding.
///
/// The last `border` columns of `A` are treated as dense; in the others, the band width is the
/// widest run, over the rows, from a row's first to its last nonzero. Any matrix is solved
/// correctly; the cost is low only when that run is short.
///
/// The QR path is kept when `A` and `B` are finite, `A` has at least as many rows as columns,
/// every column reaches a nonzero pivot, and `||R^-1||_F tau ||A||_F < 1` for the computed factor
/// `R`, with `tau = max(sqrt(eps), 4 (m n eps + rcond))` (`detail::rank_certificate_tau`) and
/// `rcond = lstsq_rcond(m, n)` by default, the cutoff below which `lstsq_min_norm` drops a singular
/// value (a caller whose dense path truncates at another relative cutoff passes that one).
///
/// The argument (argued, not proved formally). Hypotheses: IEEE binary64 with round to nearest
/// and gradual underflow, and no fast-math or flush-to-zero (the test relies on a NaN or an
/// infinity failing it); the computed `R` is the exact factor of `A + E` (`A P + E` for the dense
/// path's pivoting) with `||E||_2 <= gamma ||A||_F` and **`gamma + rcond <= tau / 4`**; each column
/// of the computed `R^-1` is the exact solution of a perturbed system `(R + dR_j) x_j = e_j`,
/// `|dR_j| <= gamma_{n+1} |R|`; and `n + 1 <= 1e7`, `m n + n^2 <= 1e13`. The size bounds keep the
/// test's own relative rounding `delta` and the factor `beta = gamma_{n+1} (1 + ||E||_F / ||A||_F)`
/// small enough that `delta tau + beta <= tau / 4`. The inverse's rounding is not small by itself:
/// it enters as a relative error of at most `beta X` in `||R^-1||_F`, `X = ||R^-1||_F ||A||_F` the
/// certificate's own product, and the passing test then bounds `X` by self-consistency, `X (tau (1
/// - delta) - beta) < 1`. Then the test passing gives `s_min(R) >
/// (3 tau / 4) ||A||_F` and, by Weyl's inequality, `s_min(A) > (tau / 2 + rcond) ||A||_F >= rcond
/// s_max(A)` plus a margin of `(tau / 2) ||A||_F`: the SVD would keep every singular value, its
/// minimum-norm solution is the unique least-squares solution the QR computes, and the two differ
/// by rounding, not by rank. The margin also covers the SVD's thresholding of computed rather than
/// exact singular values (its own error constant is not quantified by LAPACK; observed, not proved,
/// to fit). A zero pivot makes `||R^-1||_F` infinite or NaN, which fails the test.
///
/// Here `gamma` is that of Givens QR, below `4.3 (m + n - 2) eps` for any standard ordering of the
/// rotations (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., Theorem 19.10 and
/// eq. 19.25); the row-sequential order of this code is one, its rotations falling in at most `m +
/// n - 2` disjoint stages. The `4 m n eps` term of `tau` gives `gamma + rcond <= tau / 4` once `m n
/// >= 4.3 (m + n - 2)`, that is `(m - 4.3) (n - 4.3) >= 9.9`, which holds for all `m >= n >= 8`
/// and for `n <= 4` only at `m = n = 1`. Wherever it does not, the floor `sqrt(eps)` does, while
/// `4.3 (m + n - 2) eps + rcond <= sqrt(eps) / 4`: with the default `rcond = max(m, n) eps`, for
/// `m` up to about `3.1e6` (no fit or product in this library reaches such a system). The rotations
/// come from `std::hypot`, and a rotation is skipped only on an exact zero, so no part of a column
/// is dropped below the normal range; that `hypot` is accurate enough for Higham's rotation model
/// is assumed, not checked.
///
/// No answer is given for a non-finite `A` or `B` (the SVD path spreads a NaN or an infinity in
/// its own way), for fewer rows than columns, or where the test fails. Everything is computed in
/// `double` and the solution rounded to `T`, as `lstsq_min_norm` does.
///
/// Args:
///     a (const RowMatrix<T>&): Extents `(m, n)`.
///     b (const RowMatrix<T>&): Extents `(m, cols)`.
///     border (Eigen::Index): Trailing columns of `a` held dense, in `[0, n]`. Defaults to 0.
///     rcond (std::optional<double>): The relative cutoff of the dense path the answer stands for;
///         `lstsq_rcond(m, n)` when absent. A NaN, infinite or negative one gets no answer.
///
/// Returns:
///     std::optional<NdArray<T>>: Extents `(n, cols)`, or no value where the QR's answer is not
///     shown to be `lstsq_min_norm`'s.
///
/// Note:
///     `a` and `b` are assumed to have the same number of rows and `border` to be in range (no
///     validation performed).
template <StorageFloat T>
[[nodiscard]] std::optional<NdArray<T>> try_lstsq_banded(
    const RowMatrix<T>& a, const RowMatrix<T>& b, Eigen::Index border = 0,
    std::optional<double> rcond = std::nullopt) {
  const Eigen::Index m = a.rows();
  const Eigen::Index n = a.cols();
  const Eigen::Index nb = b.cols();
  const auto md = static_cast<double>(m);
  const auto nd = static_cast<double>(n);
  const std::optional<double> tau =
      detail::rank_certificate_tau(md * nd, rcond ? *rcond : lstsq_rcond(m, n));
  if (!tau) {
    return std::nullopt;  // no certificate stands for a NaN, infinite or negative cutoff
  }
  if (m < n || !a.allFinite() || !b.allFinite()) {
    return std::nullopt;
  }

  detail::BandedQr qr;
  qr.n = n;
  qr.k = border;
  qr.n0 = n - border;
  qr.w = 1;
  for (Eigen::Index i = 0; i < m; ++i) {
    Eigen::Index first = qr.n0;
    Eigen::Index last = -1;
    for (Eigen::Index j = 0; j < qr.n0; ++j) {
      if (a(i, j) != T(0)) {
        first = (std::min)(first, j);
        last = j;
      }
    }
    qr.w = (std::max)(qr.w, last - first + 1);
  }
  qr.band = Eigen::MatrixXd::Zero(n, qr.w);
  qr.border = Eigen::MatrixXd::Zero(n, qr.k);
  qr.rhs = Eigen::MatrixXd::Zero(n, nb);
  qr.filled.assign(static_cast<std::size_t>(n), 0);

  Eigen::VectorXd wb(qr.w);
  Eigen::VectorXd wk(qr.k);
  Eigen::VectorXd wr(nb);
  double a_norm_sq = 0.0;
  for (Eigen::Index i = 0; i < m; ++i) {
    Eigen::Index j = 0;
    while (j < qr.n0 && a(i, j) == T(0)) {
      ++j;
    }
    wb.setZero();
    for (Eigen::Index t = 0; t < qr.w && j + t < qr.n0; ++t) {
      wb(t) = static_cast<double>(a(i, j + t));
    }
    for (Eigen::Index t = 0; t < qr.k; ++t) {
      wk(t) = static_cast<double>(a(i, qr.n0 + t));
    }
    for (Eigen::Index c = 0; c < nb; ++c) {
      wr(c) = static_cast<double>(b(i, c));
    }
    a_norm_sq += wb.squaredNorm() + wk.squaredNorm();
    // The banded columns: rotate the row into `R` row by row until it is consumed.
    bool consumed = false;
    for (; j < qr.n0; ++j) {
      if (wb(0) != 0.0) {
        const auto row = static_cast<std::size_t>(j);
        if (qr.filled[row] == 0) {
          qr.band.row(j) = wb.transpose();
          qr.border.row(j) = wk.transpose();
          qr.rhs.row(j) = wr.transpose();
          qr.filled[row] = 1;
          consumed = true;
          break;
        }
        const auto [c, s] = detail::givens(qr.band(j, 0), wb(0));
        for (Eigen::Index t = 0; t < qr.w; ++t) {
          const double u = qr.band(j, t);
          qr.band(j, t) = c * u + s * wb(t);
          wb(t) = c * wb(t) - s * u;
        }
        for (Eigen::Index t = 0; t < qr.k; ++t) {
          const double u = qr.border(j, t);
          qr.border(j, t) = c * u + s * wk(t);
          wk(t) = c * wk(t) - s * u;
        }
        for (Eigen::Index t = 0; t < nb; ++t) {
          const double u = qr.rhs(j, t);
          qr.rhs(j, t) = c * u + s * wr(t);
          wr(t) = c * wr(t) - s * u;
        }
      }
      // Shift the band window one column right; the entry leaving it is zero in exact arithmetic.
      for (Eigen::Index t = 1; t < qr.w; ++t) {
        wb(t - 1) = wb(t);
      }
      wb(qr.w - 1) = 0.0;
      if ((wb.array() == 0.0).all()) {
        break;
      }
    }
    // The dense trailing columns.
    for (Eigen::Index t0 = 0; t0 < qr.k && !consumed; ++t0) {
      const Eigen::Index jj = qr.n0 + t0;
      if (wk(t0) == 0.0) {
        continue;
      }
      const auto row = static_cast<std::size_t>(jj);
      if (qr.filled[row] == 0) {
        qr.border.row(jj) = wk.transpose();
        qr.rhs.row(jj) = wr.transpose();
        qr.filled[row] = 1;
        consumed = true;
        break;
      }
      const auto [c, s] = detail::givens(qr.border(jj, t0), wk(t0));
      for (Eigen::Index t = t0; t < qr.k; ++t) {
        const double u = qr.border(jj, t);
        qr.border(jj, t) = c * u + s * wk(t);
        wk(t) = c * wk(t) - s * u;
      }
      wk(t0) = 0.0;  // zero in exact arithmetic; below the diagonal of any later row
      for (Eigen::Index t = 0; t < nb; ++t) {
        const double u = qr.rhs(jj, t);
        qr.rhs(jj, t) = c * u + s * wr(t);
        wr(t) = c * wr(t) - s * u;
      }
    }
  }
  if (std::any_of(qr.filled.begin(), qr.filled.end(), [](char f) { return f == 0; })) {
    return std::nullopt;
  }

  // ||R^-1||_F, one column at a time: column `j` of `R^-1` solves `R x = e_j`, zero below `j`.
  double inv_sq = 0.0;
  Eigen::MatrixXd col(n, 1);
  for (Eigen::Index j = 0; j < n; ++j) {
    col.setZero();
    col(j, 0) = 1.0;
    qr.back_substitute(col, j);
    inv_sq += col.topRows(j + 1).squaredNorm();
  }
  if (!detail::rank_certificate_holds(inv_sq, a_norm_sq, *tau)) {
    return std::nullopt;
  }

  qr.back_substitute(qr.rhs, n - 1);
  NdArray<T> out({static_cast<std::size_t>(n), static_cast<std::size_t>(nb)}, for_overwrite);
  Eigen::Map<RowMatrix<T>>(out.data(), n, nb) = qr.rhs.template cast<T>();
  return out;
}

/// Solve `A X = B` as `lstsq_min_norm` does: by `try_lstsq_banded` where it gives an answer, else
/// by `lstsq_min_norm` itself.
///
/// Args:
///     a (const RowMatrix<T>&): Extents `(m, n)`.
///     b (const RowMatrix<T>&): Extents `(m, cols)`.
///     border (Eigen::Index): Trailing columns of `a` held dense, in `[0, n]`. Defaults to 0.
///
/// Returns:
///     NdArray<T>: Extents `(n, cols)`.
///
/// Raises:
///     value_error: As `lstsq_min_norm`, on the inputs that reach it.
///
/// Note:
///     `a` and `b` are assumed to have the same number of rows and `border` to be in range (no
///     validation performed).
template <StorageFloat T>
[[nodiscard]] NdArray<T> lstsq_min_norm_banded(const RowMatrix<T>& a, const RowMatrix<T>& b,
                                               Eigen::Index border = 0) {
  std::optional<NdArray<T>> x = try_lstsq_banded<T>(a, b, border);
  return x ? std::move(*x) : lstsq_min_norm<T>(a, b);
}

}  // namespace pantr::linalg
