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
/// A product acts on a **window** of the contracted mode: the `begin` of the input and of the
/// output `ModeView` pick the run of entries along the mode that is read and written, so one
/// tensor can hold the results of many elements side by side (the extraction of a spline stacks
/// the Bézier control points of the elements along the mode, and reads each element's run of
/// B-spline control points from `first_basis[e]`). A product of the whole mode has both windows at
/// 0 and the extents of the operator (`mode_product_whole`).
///
/// The checked entry points take three records, `ModeOperator` and two `ModeView`s, instead of
/// eleven positional arguments, seven of them adjacent `std::size_t`. The raw kernel, which hot
/// loops call inside a `parallel_for` body, takes the geometry as one `ModeShape` record for the
/// same reason; its fields are copied to locals first, so the loops run on loop-invariant locals.
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

/// The geometry of one raw mode product: the operator's stored width, the entries read and written
/// along the mode, the full extents of the two tensors along it, and the products of the extents
/// before and after it.
///
/// Attributes:
///     ld (std::size_t): Columns of the stored operator (its row stride).
///     n_out (std::size_t): Rows of `A`: the entries written along the mode.
///     n_in (std::size_t): Columns of `A`: the entries read along the mode.
///     x_extent (std::size_t): The full extent of the input along the mode.
///     y_extent (std::size_t): The full extent of the output along the mode.
///     outer (std::size_t): The product of the extents before the mode.
///     inner (std::size_t): The product of the extents after the mode.
///     transpose (bool): Whether `A` is the transpose of the stored operator.
struct ModeShape {
  std::size_t ld;
  std::size_t n_out;
  std::size_t n_in;
  std::size_t x_extent;
  std::size_t y_extent;
  std::size_t outer;
  std::size_t inner;
  bool transpose;
};

/// The raw mode-product kernel on a window of one mode.
///
/// Computes `y[o, i, t] = sum_j A(i, j) x[o, j, t]` for `o < outer`, `i < n_out`, `t < inner`,
/// where `x` and `y` point at the first entry of their windows and the strides are those of the
/// whole tensors (`x_extent * inner` per step of `o`), and `A(i, j)` is `op[i * ld + j]` or, when
/// `transpose`, `op[j * ld + i]`. Each entry is summed in the order of `j`, starting from zero, in
/// `T`; with contraction on the sum is a chain of multiply-adds.
///
/// Args:
///     op (const T*): The operator, row-major with `shape.ld` columns.
///     x (const T*): The first entry of the input window.
///     y (T*): The first entry of the output window.
///     shape (const ModeShape&): The geometry. Its fields are copied to locals before the loops, so
///         the loop bounds and strides are loop-invariant locals, not loads through the record.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed). `x` and `y` must not overlap.
///     `noexcept` and free of throwing code, so the compiler may contract the sums into fused
///     multiply-adds even when a caller invokes it inside a `parallel_for` body.
template <StorageFloat T>
void mode_product_kernel(const T* op, const T* x, T* y, const ModeShape& shape) noexcept {
  const std::size_t n_out = shape.n_out;
  const std::size_t n_in = shape.n_in;
  const std::size_t x_extent = shape.x_extent;
  const std::size_t y_extent = shape.y_extent;
  const std::size_t outer = shape.outer;
  const std::size_t inner = shape.inner;
  // `A(i, j)` is `op[i * a_row + j * a_col]`.
  const std::size_t a_row = shape.transpose ? 1 : shape.ld;
  const std::size_t a_col = shape.transpose ? shape.ld : 1;
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

/// A dense matrix as stored, to be applied as itself or transposed.
///
/// Attributes:
///     values (std::span<const T>): `rows * cols` values, row-major.
///     rows (std::size_t): Rows of the stored matrix.
///     cols (std::size_t): Columns of the stored matrix.
///     transpose (bool): Apply the transpose: `n_in = rows`, `n_out = cols`. Defaults to false.
template <StorageFloat T>
struct ModeOperator {
  std::span<const T> values;
  std::size_t rows;
  std::size_t cols;
  bool transpose = false;
};

/// A row-major tensor and the window of one mode that a product reads or writes.
///
/// Attributes:
///     data (std::span<E>): The elements, row-major over `extents`; `E` is `const T` for an input
///         and `T` for an output.
///     extents (std::span<const std::size_t>): The extents (at least one of them).
///     begin (std::size_t): First entry of the window along the contracted mode. Defaults to 0.
template <class E>
struct ModeView {
  std::span<E> data;
  std::span<const std::size_t> extents;
  std::size_t begin = 0;
};

/// Apply a dense matrix along one mode of a row-major tensor, into a caller buffer.
///
/// With `A` the stored operator or its transpose, `n_in` its columns and `n_out` its rows, the
/// result is, for every index of the other modes,
/// `y[..., y.begin + i, ...] = sum_j A(i, j) x[..., x.begin + j, ...]`, `i < n_out`, `j < n_in`.
/// Entries of `y` outside the window are not touched. `x` and `y` have the same extents except
/// along `mode`; the window of each must lie within its extent.
///
/// Args:
///     op (const ModeOperator<T>&): The operator.
///     x (ModeView<const T>): The input tensor and its window.
///     mode (std::size_t): The mode to contract, `< x.extents.size()`.
///     y (ModeView<T>): The output tensor and its window; must not overlap `x`.
///
/// Raises:
///     value_error: A buffer's size is not the product of its extents; the extents differ off the
///         mode or are not at least one; the operator has the wrong size; a window leaves its
///         extent; or `x` and `y` overlap.
///     capacity_error: A product of extents overflows `std::size_t`.
template <StorageFloat T>
void mode_product(const ModeOperator<T>& op, ModeView<const T> x, std::size_t mode, ModeView<T> y) {
  if (x.extents.empty() || x.extents.size() != y.extents.size()) {
    throw value_error("mode_product: x and y need the same, nonzero, number of extents, got " +
                      std::to_string(x.extents.size()) + " and " +
                      std::to_string(y.extents.size()) + ".");
  }
  if (mode >= x.extents.size()) {
    throw value_error("mode_product: mode " + std::to_string(mode) + " is out of range for " +
                      std::to_string(x.extents.size()) + " extents.");
  }
  for (std::size_t d = 0; d < x.extents.size(); ++d) {
    if (x.extents[d] == 0 || y.extents[d] == 0) {
      throw value_error("mode_product: the extents must be at least one.");
    }
    if (d != mode && x.extents[d] != y.extents[d]) {
      throw value_error("mode_product: x and y differ in extent " + std::to_string(d) + ", " +
                        std::to_string(x.extents[d]) + " and " + std::to_string(y.extents[d]) +
                        ".");
    }
  }
  if (op.rows != 0 && op.cols > (std::numeric_limits<std::size_t>::max)() / op.rows) {
    throw capacity_error("mode_product: the operator is too large.");
  }
  if (op.values.size() != op.rows * op.cols) {
    throw value_error("mode_product: the operator holds " + std::to_string(op.values.size()) +
                      " values, but its shape needs " + std::to_string(op.rows * op.cols) + ".");
  }
  if (x.data.size() != checked_product(x.extents) || y.data.size() != checked_product(y.extents)) {
    throw value_error("mode_product: a buffer does not hold the product of its extents.");
  }
  const std::size_t n_in = op.transpose ? op.rows : op.cols;
  const std::size_t n_out = op.transpose ? op.cols : op.rows;
  // A window is `[begin, begin + n)`: formed as `begin <= extent && n <= extent - begin` so the
  // sum cannot wrap.
  if (x.begin > x.extents[mode] || n_in > x.extents[mode] - x.begin) {
    throw value_error("mode_product: the input window [" + std::to_string(x.begin) + ", +" +
                      std::to_string(n_in) + ") leaves the extent " +
                      std::to_string(x.extents[mode]) + ".");
  }
  if (y.begin > y.extents[mode] || n_out > y.extents[mode] - y.begin) {
    throw value_error("mode_product: the output window [" + std::to_string(y.begin) + ", +" +
                      std::to_string(n_out) + ") leaves the extent " +
                      std::to_string(y.extents[mode]) + ".");
  }
  // Two ranges are disjoint when one starts at or after the other's end (adjacent buffers are
  // disjoint). `std::less` is a total order on pointers into unrelated allocations.
  const std::less<const T*> before;
  const bool disjoint = !before(y.data.data(), x.data.data() + x.data.size()) ||
                        !before(x.data.data(), y.data.data() + y.data.size());
  if (!disjoint) {
    throw value_error("mode_product: x and y overlap.");
  }
  const bezier::detail::AxisSplit s = bezier::detail::split_around(x.extents, mode);
  const ModeShape shape{.ld = op.cols,
                        .n_out = n_out,
                        .n_in = n_in,
                        .x_extent = x.extents[mode],
                        .y_extent = y.extents[mode],
                        .outer = s.outer,
                        .inner = s.inner,
                        .transpose = op.transpose};
  mode_product_kernel<T>(op.values.data(), x.data.data() + x.begin * s.inner,
                         y.data.data() + y.begin * s.inner, shape);
}

/// Apply a dense matrix to the whole of one mode (both windows start at 0 and cover the mode).
///
/// `x` and `y` then have the extents of `x.extents` and `y.extents`, equal off the mode, and the
/// extent of `x` along the mode is `n_in`, that of `y` is `n_out`. See `mode_product`.
///
/// Args:
///     op (const ModeOperator<T>&): The operator.
///     x (ModeView<const T>): The input tensor; its window must be the whole mode.
///     mode (std::size_t): The mode to contract.
///     y (ModeView<T>): The output tensor, same; must not overlap `x`.
///
/// Raises:
///     value_error: As `mode_product`, or a mode extent that is not exactly the operator's.
template <StorageFloat T>
void mode_product_whole(const ModeOperator<T>& op, ModeView<const T> x, std::size_t mode,
                        ModeView<T> y) {
  const std::size_t n_in = op.transpose ? op.rows : op.cols;
  const std::size_t n_out = op.transpose ? op.cols : op.rows;
  if (mode < x.extents.size() && mode < y.extents.size() &&
      (x.extents[mode] != n_in || y.extents[mode] != n_out)) {
    throw value_error("mode_product: the extents along the mode are " +
                      std::to_string(x.extents[mode]) + " and " + std::to_string(y.extents[mode]) +
                      ", but the operator maps " + std::to_string(n_in) + " to " +
                      std::to_string(n_out) + ".");
  }
  mode_product<T>(op, x, mode, y);
}

}  // namespace pantr::bspline::detail
