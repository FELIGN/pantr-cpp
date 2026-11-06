/// @file truncation.hpp
/// @brief Two-scale matrices between levels, and the box operations that build truncated
/// functions: refinement of a coefficient box and truncation against the active functions.
///
/// A truncated function is stored as a dense box of coefficients in the tensor-product basis of
/// one level. `refine_box` expresses such a box in the next finer level (one banded matrix per
/// direction) and `truncate_box` zeroes the coefficients that sit on active functions of that
/// level. They are pure functions of their inputs, so the refinement of a space and the multilevel
/// extraction can reuse them.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed). For general use, call
///     `pantr::thb::THBSplineSpace`.
#pragma once

#include "pantr/bspline/detail/oslo.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/scalar.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace pantr::thb::detail {

/// The two-scale (Oslo) matrix of one direction between two consecutive levels, stored by rows.
///
/// A level-`m` function `B_i` equals `sum_j alpha(j, i) B_j` in the level-`(m + 1)` basis. Row `j`
/// of `alpha` is non-zero on at most `degree + 1` consecutive columns, which are the only ones
/// stored; entries inside that window can still be exactly zero.
///
/// The values are the ones the 1D refinement matrix has in the levels' storage type, widened to
/// `double`.
///
/// Attributes:
///     num_fine (std::int64_t): Rows, the function count of the finer level.
///     num_coarse (std::int64_t): Columns, the function count of the coarser level.
///     width (std::int64_t): `degree + 1`, entries stored per row.
///     first_col (std::vector<std::int64_t>): Column of the first stored entry of each row;
///         non-decreasing in the row, and negative only on a non-clamped knot vector.
///     alphas (std::vector<double>): Row-major `(num_fine, width)`; entry `(j, l)` is `alpha(j,
///         first_col[j] + l)`.
struct TwoScale1D {
  std::int64_t num_fine = 0;
  std::int64_t num_coarse = 0;
  std::int64_t width = 0;
  std::vector<std::int64_t> first_col;
  std::vector<double> alphas;
};

/// Build the two-scale matrix between two nested 1D spaces.
///
/// The matrix is computed from the spaces' knots in their storage type `T`, as the Python
/// implementation computes it, and widened to `double`.
///
/// Args:
///     coarse (const bspline::BsplineSpace1D<T>&): The coarser space.
///     fine (const bspline::BsplineSpace1D<T>&): A refinement of it, of the same degree.
///
/// Returns:
///     TwoScale1D: The matrix.
///
/// Raises:
///     std::overflow_error: An entry is not finite. A space's knots are finite, so the overflow is
///         the ratios' own: a knot difference beyond the range of `T` (a float32 knot vector
///         spanning more than `FLT_MAX`) gives `inf / inf`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): `fine`'s knots contain
///     `coarse`'s.
template <StorageFloat T>
[[nodiscard]] TwoScale1D make_two_scale(const bspline::BsplineSpace1D<T>& coarse,
                                        const bspline::BsplineSpace1D<T>& fine) {
  const bspline::detail::OsloRows<T> rows =
      bspline::detail::oslo_rows<T>(coarse.degree(), coarse.knots(), fine.knots());
  TwoScale1D out;
  out.num_fine = fine.num_basis();
  out.num_coarse = coarse.num_basis();
  out.width = static_cast<std::int64_t>(coarse.degree()) + 1;
  out.first_col = rows.first_col;
  const std::size_t count = static_cast<std::size_t>(out.num_fine * out.width);
  out.alphas.resize(count);
  const T* src = rows.alphas.data();
  for (std::size_t i = 0; i < count; ++i) {
    out.alphas[i] = static_cast<double>(src[i]);
  }
  if (!std::all_of(out.alphas.begin(), out.alphas.end(),
                   [](double v) { return std::isfinite(v); })) {
    throw std::overflow_error(
        "the two-scale matrix between two levels overflows the knots' dtype (a knot span beyond "
        "its range)");
  }
  return out;
}

/// A dense box of coefficients in a level's tensor-product basis.
///
/// Attributes:
///     lo (std::vector<std::int64_t>): Per-direction index of the box's first function.
///     shape (std::vector<std::size_t>): Per-direction function count.
///     coeffs (std::vector<double>): The coefficients, row-major over `shape`.
struct CoeffBox {
  std::vector<std::int64_t> lo;
  std::vector<std::size_t> shape;
  std::vector<double> coeffs;
};

/// `dst[i] += weight * src[i]` for `i` in `[0, n)`.
///
/// A `noexcept` function of its own so that GCC vectorises the loop: written inside a body that may
/// throw, GCC 14 with `-ffp-contract=on` treats the contracted `fma` as a statement that can throw
/// and leaves the loop scalar.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
inline void accumulate_scaled(double* dst, const double* src, double weight,
                              std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) {
    dst[i] += weight * src[i];
  }
}

/// Express a coefficient box in the next finer level.
///
/// Applies, direction by direction, the two-scale matrix restricted to the box's columns, and the
/// box grows to the rows that have a non-zero entry there: the first and the last such row bound
/// the new range (a zero row between them stays in it). The contraction of one direction sums over
/// the box's columns in increasing order, so the result does not depend on a thread count or a
/// BLAS.
///
/// Args:
///     in (const CoeffBox&): The box, in the coarse level's basis.
///     two_scale (std::span<const TwoScale1D>): The level transition's matrix of each direction.
///
/// Returns:
///     CoeffBox: The box in the fine level's basis; `in` is untouched.
///
/// Raises:
///     value_error: The matrix has only zeros on the box's columns in some direction (a degenerate
///         box or an invalid knot refinement).
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): the box lies inside the coarse
///     level's function range and `two_scale` has one matrix per direction.
[[nodiscard]] inline CoeffBox refine_box(const CoeffBox& in,
                                         std::span<const TwoScale1D> two_scale) {
  const std::size_t dim = in.shape.size();
  CoeffBox cur = in;
  for (std::size_t k = 0; k < dim; ++k) {
    const TwoScale1D& a = two_scale[k];
    const std::int64_t lo = in.lo[k];
    const std::int64_t hi = lo + static_cast<std::int64_t>(in.shape[k]);
    // Rows whose stored window `[first_col, first_col + width)` meets `[lo, hi)`: first_col is
    // non-decreasing, so they are one contiguous range, found by two binary searches. Rows
    // outside it are zero on the box's columns.
    const auto first_it =
        std::partition_point(a.first_col.begin(), a.first_col.end(),
                             [&](std::int64_t first_col) { return first_col + a.width <= lo; });
    const auto last_it = std::partition_point(
        first_it, a.first_col.end(), [&](std::int64_t first_col) { return first_col < hi; });
    const auto row_begin = static_cast<std::int64_t>(first_it - a.first_col.begin());
    const auto row_end = static_cast<std::int64_t>(last_it - a.first_col.begin());
    std::int64_t new_lo = -1;
    std::int64_t new_hi = -1;
    for (std::int64_t row = row_begin; row < row_end; ++row) {
      const std::int64_t base = a.first_col[static_cast<std::size_t>(row)];
      const double* band = a.alphas.data() + static_cast<std::size_t>(row * a.width);
      const std::int64_t col_begin = (std::max)(lo, base);
      const std::int64_t col_end = (std::min)(hi, base + a.width);
      bool any = false;
      for (std::int64_t col = col_begin; col < col_end; ++col) {
        any = any || band[col - base] != 0.0;
      }
      if (any) {
        if (new_lo < 0) {
          new_lo = row;
        }
        new_hi = row + 1;
      }
    }
    if (new_lo < 0) {
      // Python's text, em dash included (written as its UTF-8 bytes: the sources are ASCII).
      throw value_error("_refine_box: Oslo matrix slice for direction " + std::to_string(k) +
                        " (columns [" + std::to_string(lo) + ":" + std::to_string(hi) +
                        "]) is entirely zero \xE2\x80\x94 degenerate or invalid knot refinement.");
    }
    std::size_t outer = 1;
    std::size_t inner = 1;
    for (std::size_t j = 0; j < k; ++j) {
      outer *= cur.shape[j];
    }
    for (std::size_t j = k + 1; j < dim; ++j) {
      inner *= cur.shape[j];
    }
    const std::size_t n_old = cur.shape[k];
    const auto n_new = static_cast<std::size_t>(new_hi - new_lo);
    std::vector<double> next(outer * n_new * inner, 0.0);
    for (std::size_t o = 0; o < outer; ++o) {
      for (std::int64_t row = new_lo; row < new_hi; ++row) {
        const std::int64_t base = a.first_col[static_cast<std::size_t>(row)];
        const double* band = a.alphas.data() + static_cast<std::size_t>(row * a.width);
        double* dst = next.data() + (o * n_new + static_cast<std::size_t>(row - new_lo)) * inner;
        const std::int64_t col_begin = (std::max)(lo, base);
        const std::int64_t col_end = (std::min)(hi, base + a.width);
        for (std::int64_t col = col_begin; col < col_end; ++col) {
          const double* src =
              cur.coeffs.data() + (o * n_old + static_cast<std::size_t>(col - lo)) * inner;
          accumulate_scaled(dst, src, band[col - base], inner);
        }
      }
    }
    cur.coeffs = std::move(next);
    cur.shape[k] = n_new;
    cur.lo[k] = new_lo;
  }
  return cur;
}

/// Zero the coefficients of a box that sit on active functions of its level.
///
/// Args:
///     box (CoeffBox&): The box, in the basis of the level whose active functions are given;
///         modified in place.
///     active (std::span<const std::int64_t>): Sorted flat (row-major) indices of that level's
///         active functions.
///     num_basis (std::span<const std::int64_t>): Per-direction function count of that level.
///
/// Returns:
///     bool: True when at least one coefficient was zeroed.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
inline bool truncate_box(CoeffBox& box, std::span<const std::int64_t> active,
                         std::span<const std::int64_t> num_basis) {
  const std::size_t dim = box.shape.size();
  bool zeroed = false;
  std::vector<std::size_t> idx(dim, 0);
  for (std::size_t at = 0; at < box.coeffs.size(); ++at) {
    std::int64_t flat = 0;
    for (std::size_t k = 0; k < dim; ++k) {
      flat = flat * num_basis[k] + box.lo[k] + static_cast<std::int64_t>(idx[k]);
    }
    if (std::binary_search(active.begin(), active.end(), flat)) {
      box.coeffs[at] = 0.0;
      zeroed = true;
    }
    for (std::size_t k = dim; k-- > 0;) {
      if (++idx[k] < box.shape[k]) {
        break;
      }
      idx[k] = 0;
    }
  }
  return zeroed;
}

}  // namespace pantr::thb::detail
