/// @file oslo.hpp
/// @brief The Oslo algorithm's kernels: the banded rows of the knot-refinement matrix, the dense
/// matrix, and the refined control points.
///
/// Implements the discrete B-spline recurrence of Cohen, Lyche and Riesenfeld (1980), which maps
/// old control points to new ones in one pass. It is run row by row over a band of `degree + 1`
/// entries (Lyche and Morken, *Spline Methods*, section 4.2.3): row `i` of the refinement matrix
/// is a discrete B-spline supported on the columns `[mu(i) - degree, mu(i)]`, so everything a dense
/// sweep computes outside that window is exactly zero, and one row costs `O(degree^2)` instead of
/// `O(n * degree)`.
///
/// Storage type. The recurrence is carried in `T` except for the two running values `saved` and
/// the entry written from it, which the Numba kernel this replaces held in `double` (its
/// accumulator was a `float64` literal, which Numba unifies with the `float32` terms): the sum of
/// the carried value and the left term is formed in `double` and rounded once when stored. For
/// `double` the two spellings coincide; for `float` the stored values differ from an all-`float`
/// recurrence in the last place, and these kernels reproduce the Numba ones.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed). For general use, call
///     `pantr::bspline::insert_knots` or `pantr::bspline::oslo_matrix`.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace pantr::bspline::detail {

/// The banded rows of a refinement matrix.
///
/// Attributes:
///     alphas (NdArray<T>): Extents `(m + 1, degree + 1)`. Entry `(i, l)` is the coefficient of old
///         control point `first_col[i] + l` in new control point `i`.
///     first_col (std::vector<std::int64_t>): `m + 1` entries. Negative when the span sits within
///         the first `degree` knots (non-clamped knot vectors); the leading entries of such a row
///         are meaningless and callers skip columns outside `[0, n]`.
template <StorageFloat T>
struct OsloRows {
  NdArray<T> alphas;
  std::vector<std::int64_t> first_col;
};

/// Find, for increasing points, the span of the old knot vector that contains each, clamped to
/// `[0, n]`.
///
/// The span of `x` is `upper_bound(x) - 1`. Consecutive points of a sorted refined knot vector
/// land on non-decreasing spans, so the search resumes from the previous result and advances
/// linearly (amortised constant time per point); a point that falls before the previous one (an
/// unsorted vector) is searched from scratch, so the result is `upper_bound` in every case.
template <StorageFloat T>
class SpanFinder {
 public:
  SpanFinder(std::span<const T> old_knots, std::ptrdiff_t n) : knots_(old_knots), n_(n) {}

  /// Get the span containing `x`, in `[0, n]`.
  [[nodiscard]] std::ptrdiff_t operator()(T x) {
    const auto size = static_cast<std::ptrdiff_t>(knots_.size());
    std::ptrdiff_t c = count_;
    if (c > 0 && x < knots_[static_cast<std::size_t>(c - 1)]) {
      c = std::upper_bound(knots_.begin(), knots_.end(), x) - knots_.begin();
    } else {
      while (c < size && !(x < knots_[static_cast<std::size_t>(c)])) {
        ++c;
      }
    }
    count_ = c;
    return (std::max)((std::min)(c - 1, n_), std::ptrdiff_t{0});
  }

 private:
  std::span<const T> knots_;
  std::ptrdiff_t n_;
  std::ptrdiff_t count_ = 0;
};

/// Compute one banded row of the Oslo refinement matrix, in place in `band`.
///
/// Row `i` holds the discrete B-splines `alpha_{j, p+1}(i)`, which vanish outside `j` in
/// `[mu - degree, mu]`, `mu` being the old-knot span containing `new_knots[i]` (the right end of
/// the knot vector lands past the last span and is clamped back onto it). Only that window is
/// computed, by
///
///     alpha_j^(k+1) = (x_k - t_j) / (t_{j+k} - t_j) alpha_j^(k)
///                   + (t_{j+k+1} - x_k) / (t_{j+k+1} - t_{j+1}) alpha_{j+1}^(k),   x_k =
///                   tau_{i+k},
///
/// for `k = 1 .. degree` from the seed `alpha_mu = 1`. A term whose denominator vanishes is
/// dropped, and since the two terms that consume `alpha_j^(k)` share the denominator `t_{j+k} -
/// t_j`, one test per entry settles both, which lets the level run in place with a single carried
/// value. Unlike the Cox-de Boor triangle for basis functions the evaluation point changes with the
/// level, so no knot differences can be reused across levels.
///
/// Args:
///     degree (int): Degree of the B-spline, `>= 0`.
///     mu (std::ptrdiff_t): The span of `new_knots[i]`, in `[0, n]`.
///     old_knots (std::span<const T>): Original knot vector.
///     new_knots (std::span<const T>): Refined knot vector.
///     i (std::size_t): The row.
///     band (T*): `degree + 1` entries, zero on entry; receives the row.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
void oslo_band(int degree, std::ptrdiff_t mu, std::span<const T> old_knots,
               std::span<const T> new_knots, std::size_t i, T* band) {
  const auto p = static_cast<std::ptrdiff_t>(degree);
  band[p] = static_cast<T>(1);
  for (std::ptrdiff_t k = 1; k <= p; ++k) {
    const T x = new_knots[i + static_cast<std::size_t>(k)];
    double saved = 0.0;
    // The band grows one entry to the left per level; entries left of column 0 only ever feed
    // columns further left, so they are skipped.
    const std::ptrdiff_t l_start = (std::max)(p - k + 1, p - mu);
    for (std::ptrdiff_t level = l_start; level <= p; ++level) {
      const auto j = static_cast<std::size_t>(mu - p + level);
      const auto ku = static_cast<std::size_t>(k);
      const T denom = old_knots[j + ku] - old_knots[j];
      double to_left = 0.0;
      double to_right = 0.0;
      if (denom > static_cast<T>(0)) {
        const T b = band[level];
        const T left_term = (old_knots[j + ku] - x) / denom * b;
        const T right_term = (x - old_knots[j]) / denom * b;
        to_left = static_cast<double>(left_term);
        to_right = static_cast<double>(right_term);
      }
      const double entry = saved + to_left;
      band[level - 1] = static_cast<T>(entry);
      saved = to_right;
    }
    band[p] = static_cast<T>(saved);
  }
}

/// Compute the Oslo refinement matrix one banded row at a time.
///
/// See `oslo_band` for the recurrence.
///
/// Args:
///     degree (int): Degree of the B-spline, `>= 0`.
///     old_knots (std::span<const T>): Original knot vector, `n + degree + 2` entries.
///     new_knots (std::span<const T>): Refined knot vector, `m + degree + 2` entries; a superset of
///         `old_knots`.
///
/// Returns:
///     OsloRows<T>: The `m + 1` banded rows.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): both vectors hold at least
///     `degree + 2` knots.
template <StorageFloat T>
[[nodiscard]] OsloRows<T> oslo_rows(int degree, std::span<const T> old_knots,
                                    std::span<const T> new_knots) {
  const auto p = static_cast<std::ptrdiff_t>(degree);
  const auto n = static_cast<std::ptrdiff_t>(old_knots.size()) - p - 2;  // old control points - 1
  const auto m = static_cast<std::ptrdiff_t>(new_knots.size()) - p - 2;  // new control points - 1
  const auto width = static_cast<std::size_t>(p) + 1;

  OsloRows<T> out{NdArray<T>({static_cast<std::size_t>(m) + 1, width}),
                  std::vector<std::int64_t>(static_cast<std::size_t>(m) + 1)};
  T* alphas = out.alphas.data();
  SpanFinder<T> span_of(old_knots, n);
  for (std::size_t i = 0; i <= static_cast<std::size_t>(m); ++i) {
    const std::ptrdiff_t mu = span_of(new_knots[i]);
    out.first_col[i] = mu - p;
    oslo_band<T>(degree, mu, old_knots, new_knots, i, alphas + i * width);
  }
  return out;
}

/// Compute the dense Oslo refinement matrix by scattering the banded rows.
///
/// The matrix `alpha` has extents `(m + 1, n + 1)`: the new control points `Q = alpha P` reproduce
/// the original geometry exactly. It equals the dense recurrence entry for entry, since outside the
/// band the dense sweep computes exact zeros.
///
/// Args:
///     degree (int): Degree of the B-spline, `>= 0`.
///     old_knots (std::span<const T>): Original knot vector, `n + degree + 2` entries.
///     new_knots (std::span<const T>): Refined knot vector, `m + degree + 2` entries.
///
/// Returns:
///     NdArray<T>: Extents `(m + 1, n + 1)`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] NdArray<T> oslo_matrix_dense(int degree, std::span<const T> old_knots,
                                           std::span<const T> new_knots) {
  const auto n1 = old_knots.size() - static_cast<std::size_t>(degree) - 1;  // n + 1
  const auto m1 = new_knots.size() - static_cast<std::size_t>(degree) - 1;  // m + 1
  const OsloRows<T> rows = oslo_rows<T>(degree, old_knots, new_knots);
  NdArray<T> out({m1, n1});
  const auto width = static_cast<std::size_t>(degree) + 1;
  for (std::size_t i = 0; i < m1; ++i) {
    const std::int64_t base = rows.first_col[i];
    for (std::size_t l = 0; l < width; ++l) {
      const std::int64_t col = base + static_cast<std::int64_t>(l);
      if (0 <= col && col < static_cast<std::int64_t>(n1)) {
        out.data()[i * n1 + static_cast<std::size_t>(col)] = rows.alphas.data()[i * width + l];
      }
    }
  }
  return out;
}

/// Apply the Oslo algorithm: the control points of the refined space.
///
/// The refinement matrix is never materialised: each new control point is the combination of the
/// `degree + 1` old ones its band selects. The summation order therefore differs from a dense
/// matrix product, so results move in the last bits.
///
/// Args:
///     degree (int): Degree of the B-spline, `>= 0`.
///     old_knots (std::span<const T>): Original knot vector, `n + degree + 2` entries.
///     ctrl (mdview<const T, 2>): Control points, extents `(n + 1, cols)`.
///     new_knots (std::span<const T>): Refined knot vector, `m + degree + 2` entries; a superset of
///         `old_knots`.
///
/// Returns:
///     NdArray<T>: Extents `(m + 1, cols)`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): `ctrl.extent(0)` is `n + 1`.
template <StorageFloat T>
[[nodiscard]] NdArray<T> insert_knots_core(int degree, std::span<const T> old_knots,
                                           mdview<const T, 2> ctrl, std::span<const T> new_knots) {
  const auto p = static_cast<std::ptrdiff_t>(degree);
  const auto n = static_cast<std::ptrdiff_t>(old_knots.size()) - p - 2;
  const auto m1 = new_knots.size() - static_cast<std::size_t>(degree) - 1;  // m + 1
  const std::size_t cols = ctrl.extent(1);
  const auto n1 = static_cast<std::int64_t>(ctrl.extent(0));
  const auto width = static_cast<std::size_t>(degree) + 1;

  NdArray<T> out({m1, cols});
  T* dst = out.data();
  const T* src = ctrl.data_handle();
  std::vector<T> band(width);
  SpanFinder<T> span_of(old_knots, n);
  for (std::size_t i = 0; i < m1; ++i) {
    // The band of this row, formed here and applied at once: it is not stored.
    const std::ptrdiff_t mu = span_of(new_knots[i]);
    std::fill(band.begin(), band.end(), static_cast<T>(0));
    oslo_band<T>(degree, mu, old_knots, new_knots, i, band.data());
    const std::int64_t base = mu - p;
    T* row = dst + i * cols;
    for (std::size_t l = 0; l < width; ++l) {
      const std::int64_t col = base + static_cast<std::int64_t>(l);
      if (0 <= col && col < n1) {
        const T weight = band[l];
        const T* in = src + static_cast<std::size_t>(col) * cols;
        for (std::size_t r = 0; r < cols; ++r) {
          row[r] += weight * in[r];
        }
      }
    }
  }
  return out;
}

}  // namespace pantr::bspline::detail
