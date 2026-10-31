/// @file mode_product.hpp
/// @brief The mode-wise contraction primitive: a small dense matrix applied along one mode of a
///        row-major tensor.
///
/// Every tensor-product application of a per-direction operator is a sequence of these: a Bézier
/// extraction of control points, the per-element operators `M v`, `M^T v`, `M^T K M` and
/// `M K M^T` of a spanwise extraction (one product per direction, each on the result of the
/// previous). The primitive has runtime extents and rank, allocates nothing and writes into the
/// caller's buffer.
///
/// A product acts on a **window** of the contracted mode: `x_begin` and `y_begin` pick the run of
/// entries along the mode that is read and written, so one tensor can hold the results of many
/// elements side by side (the extraction of a spline stacks the Bézier control points of the
/// elements along the mode, and reads each element's run of B-spline control points from
/// `first_basis[e]`). A product of the whole mode has both windows at 0 and the extents of the
/// operator.
#pragma once

#include "pantr/bezier/detail/along_axis.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <string>

namespace pantr::bspline::detail {

/// The raw mode-product kernel on a window of one mode.
///
/// Computes `y[o, i, t] = sum_j A(i, j) x[o, j, t]` for `o < outer`, `i < n_out`, `t < inner`,
/// where `x` and `y` point at the first entry of their windows and the strides are those of the
/// whole tensors (`x_extent * inner` per step of `o`), and `A(i, j)` is `op[i * ld + j]` or, when
/// `transpose`, `op[j * ld + i]`. Each entry is summed in the order of `j`, starting from zero, in
/// `T`; with contraction on the sum is a chain of multiply-adds.
///
/// Args:
///     op (const T*): The operator, row-major with `ld` columns.
///     ld (std::size_t): Columns of the stored operator.
///     n_out (std::size_t): Rows of `A`: the entries written along the mode.
///     n_in (std::size_t): Columns of `A`: the entries read along the mode.
///     transpose (bool): Whether `A` is the transpose of the stored operator.
///     x (const T*): The first entry of the input window.
///     x_extent (std::size_t): The full extent of the input along the mode.
///     y (T*): The first entry of the output window.
///     y_extent (std::size_t): The full extent of the output along the mode.
///     outer (std::size_t): The product of the extents before the mode.
///     inner (std::size_t): The product of the extents after the mode.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed). `x` and `y` must not overlap.
///     `noexcept` and free of throwing code, so the compiler may contract the sums into fused
///     multiply-adds even when a caller invokes it inside a `parallel_for` body.
template <StorageFloat T>
void mode_product_kernel(const T* op, std::size_t ld, std::size_t n_out, std::size_t n_in,
                         bool transpose, const T* x, std::size_t x_extent, T* y,
                         std::size_t y_extent, std::size_t outer, std::size_t inner) noexcept {
  // `A(i, j)` is `op[i * a_row + j * a_col]`.
  const std::size_t a_row = transpose ? 1 : ld;
  const std::size_t a_col = transpose ? ld : 1;
  for (std::size_t o = 0; o < outer; ++o) {
    const T* xo = x + o * x_extent * inner;
    T* yo = y + o * y_extent * inner;
    if (inner == 1) {
      for (std::size_t i = 0; i < n_out; ++i) {
        T acc = static_cast<T>(0);
        for (std::size_t j = 0; j < n_in; ++j) {
          acc += op[i * a_row + j * a_col] * xo[j];
        }
        yo[i] = acc;
      }
      continue;
    }
    for (std::size_t i = 0; i < n_out; ++i) {
      T* yrow = yo + i * inner;
      for (std::size_t t = 0; t < inner; ++t) {
        yrow[t] = static_cast<T>(0);
      }
      for (std::size_t j = 0; j < n_in; ++j) {
        const T c = op[i * a_row + j * a_col];
        const T* xrow = xo + j * inner;
        for (std::size_t t = 0; t < inner; ++t) {
          yrow[t] += c * xrow[t];
        }
      }
    }
  }
}

/// Apply a dense matrix along one mode of a row-major tensor, into a caller buffer.
///
/// With `A` the stored operator (`op_rows x op_cols`, row-major) or its transpose, `n_in` its
/// columns and `n_out` its rows, the result is, for every index of the other modes,
/// `y[..., y_begin + i, ...] = sum_j A(i, j) x[..., x_begin + j, ...]`, `i < n_out`, `j < n_in`.
/// Entries of `y` outside the window are not touched. `x` and `y` have the same extents except
/// along `mode`; the window of each must lie within its extent.
///
/// Args:
///     op (std::span<const T>): The operator, `op_rows * op_cols` values, row-major.
///     op_rows (std::size_t): Rows of the stored operator.
///     op_cols (std::size_t): Columns of the stored operator.
///     transpose (bool): Apply the transpose: `n_in = op_rows`, `n_out = op_cols`.
///     x (std::span<const T>): The input tensor, row-major over `x_extents`.
///     x_extents (std::span<const std::size_t>): Its extents (at least one).
///     mode (std::size_t): The mode to contract, `< x_extents.size()`.
///     x_begin (std::size_t): First entry of the input window along `mode`.
///     y (std::span<T>): The output tensor, row-major over `y_extents`; must not overlap `x`.
///     y_extents (std::span<const std::size_t>): Its extents.
///     y_begin (std::size_t): First entry of the output window along `mode`.
///
/// Raises:
///     value_error: A buffer's size is not the product of its extents; the extents differ off the
///         mode or are not at least one; the operator has the wrong size; a window leaves its
///         extent; or `x` and `y` overlap.
///     capacity_error: A product of extents overflows `std::size_t`.
template <StorageFloat T>
void mode_product(std::span<const T> op, std::size_t op_rows, std::size_t op_cols, bool transpose,
                  std::span<const T> x, std::span<const std::size_t> x_extents, std::size_t mode,
                  std::size_t x_begin, std::span<T> y, std::span<const std::size_t> y_extents,
                  std::size_t y_begin) {
  if (x_extents.empty() || x_extents.size() != y_extents.size()) {
    throw value_error("mode_product: x and y need the same, nonzero, number of extents, got " +
                      std::to_string(x_extents.size()) + " and " +
                      std::to_string(y_extents.size()) + ".");
  }
  if (mode >= x_extents.size()) {
    throw value_error("mode_product: mode " + std::to_string(mode) + " is out of range for " +
                      std::to_string(x_extents.size()) + " extents.");
  }
  for (std::size_t d = 0; d < x_extents.size(); ++d) {
    if (x_extents[d] == 0 || y_extents[d] == 0) {
      throw value_error("mode_product: the extents must be at least one.");
    }
    if (d != mode && x_extents[d] != y_extents[d]) {
      throw value_error("mode_product: x and y differ in extent " + std::to_string(d) + ", " +
                        std::to_string(x_extents[d]) + " and " + std::to_string(y_extents[d]) +
                        ".");
    }
  }
  if (op_rows != 0 && op_cols > (std::numeric_limits<std::size_t>::max)() / op_rows) {
    throw capacity_error("mode_product: the operator is too large.");
  }
  if (op.size() != op_rows * op_cols) {
    throw value_error("mode_product: the operator holds " + std::to_string(op.size()) +
                      " values, but its shape needs " + std::to_string(op_rows * op_cols) + ".");
  }
  if (x.size() != checked_product(x_extents) || y.size() != checked_product(y_extents)) {
    throw value_error("mode_product: a buffer does not hold the product of its extents.");
  }
  const std::size_t n_in = transpose ? op_rows : op_cols;
  const std::size_t n_out = transpose ? op_cols : op_rows;
  // A window is `[begin, begin + n)`: formed as `begin <= extent && n <= extent - begin` so the
  // sum cannot wrap.
  if (x_begin > x_extents[mode] || n_in > x_extents[mode] - x_begin) {
    throw value_error("mode_product: the input window [" + std::to_string(x_begin) + ", +" +
                      std::to_string(n_in) + ") leaves the extent " +
                      std::to_string(x_extents[mode]) + ".");
  }
  if (y_begin > y_extents[mode] || n_out > y_extents[mode] - y_begin) {
    throw value_error("mode_product: the output window [" + std::to_string(y_begin) + ", +" +
                      std::to_string(n_out) + ") leaves the extent " +
                      std::to_string(y_extents[mode]) + ".");
  }
  // Two ranges are disjoint when one starts at or after the other's end (adjacent buffers are
  // disjoint). `std::less` is a total order on pointers into unrelated allocations.
  const std::less<const T*> before;
  const bool disjoint =
      !before(y.data(), x.data() + x.size()) || !before(x.data(), y.data() + y.size());
  if (!disjoint) {
    throw value_error("mode_product: x and y overlap.");
  }
  const bezier::detail::AxisSplit s = bezier::detail::split_around(x_extents, mode);
  mode_product_kernel<T>(op.data(), op_cols, n_out, n_in, transpose, x.data() + x_begin * s.inner,
                         x_extents[mode], y.data() + y_begin * s.inner, y_extents[mode], s.outer,
                         s.inner);
}

/// Apply a dense matrix to the whole of one mode (both windows start at 0 and cover the mode).
///
/// `x` and `y` then have the extents of `x_extents` and `y_extents`, equal off the mode, and the
/// extent of `x` along the mode is `n_in`, that of `y` is `n_out`. See the windowed overload.
///
/// Args:
///     op (std::span<const T>): The operator, `op_rows * op_cols` values, row-major.
///     op_rows (std::size_t): Rows of the stored operator.
///     op_cols (std::size_t): Columns of the stored operator.
///     transpose (bool): Apply the transpose.
///     x (std::span<const T>): The input tensor.
///     x_extents (std::span<const std::size_t>): Its extents.
///     mode (std::size_t): The mode to contract.
///     y (std::span<T>): The output tensor; must not overlap `x`.
///     y_extents (std::span<const std::size_t>): Its extents.
///
/// Raises:
///     value_error: As the windowed overload, or a mode extent that is not exactly the operator's.
template <StorageFloat T>
void mode_product(std::span<const T> op, std::size_t op_rows, std::size_t op_cols, bool transpose,
                  std::span<const T> x, std::span<const std::size_t> x_extents, std::size_t mode,
                  std::span<T> y, std::span<const std::size_t> y_extents) {
  const std::size_t n_in = transpose ? op_rows : op_cols;
  const std::size_t n_out = transpose ? op_cols : op_rows;
  if (mode < x_extents.size() && mode < y_extents.size() &&
      (x_extents[mode] != n_in || y_extents[mode] != n_out)) {
    throw value_error("mode_product: the extents along the mode are " +
                      std::to_string(x_extents[mode]) + " and " + std::to_string(y_extents[mode]) +
                      ", but the operator maps " + std::to_string(n_in) + " to " +
                      std::to_string(n_out) + ".");
  }
  mode_product<T>(op, op_rows, op_cols, transpose, x, x_extents, mode, 0, y, y_extents, 0);
}

}  // namespace pantr::bspline::detail
