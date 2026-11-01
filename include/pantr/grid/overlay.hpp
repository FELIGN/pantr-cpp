/// @file overlay.hpp
/// @brief `overlay`: the coarsest tensor-product grid refining two tensor-product grids.
///
/// The overlay's per-axis breakpoints are the sorted union of both inputs' breakpoints restricted
/// to the intersection of their domains. This is the background-grid bridge for immersed or
/// unfitted quadrature: when a B-spline (one knot structure) is integrated over an immersion grid
/// (another), each overlay cell falls inside one polynomial piece of the B-spline *and* one
/// immersion cell. The operation is symmetric up to floating-point noise and is defined for any
/// dimension.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/grid/tensor_product_grid.hpp"
#include "pantr/tolerance.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace pantr::grid {

namespace detail {

/// Merge two breakpoint vectors into their sorted, deduplicated union.
///
/// Only breakpoints strictly inside `(lo + atol, hi - atol)` participate; the `lo` and `hi`
/// intersection bounds are always emitted as the first and last entries. Breakpoints closer than
/// `atol` are folded into one (the first of a run is kept).
///
/// Args:
///     ba (std::span<const double>): First input's breakpoints.
///     bb (std::span<const double>): Second input's breakpoints.
///     lo (double): Lower intersection bound; always emitted first.
///     hi (double): Upper intersection bound; always emitted last.
///     atol (double): Absolute tolerance for merging near-coincident entries.
///
/// Returns:
///     std::vector<double>: A strictly increasing vector starting with `lo` and ending with `hi`
///     (at least two entries).
[[nodiscard]] inline std::vector<double> merge_axis_breakpoints(std::span<const double> ba,
                                                                std::span<const double> bb,
                                                                double lo, double hi, double atol) {
  std::vector<double> candidates = {lo, hi};
  for (const std::span<const double> b : {ba, bb}) {
    for (const double x : b) {
      if (x > lo + atol && x < hi - atol) {
        candidates.push_back(x);
      }
    }
  }
  std::stable_sort(candidates.begin(), candidates.end());
  std::vector<double> merged;
  merged.reserve(candidates.size());
  merged.push_back(candidates.front());
  for (std::size_t i = 1; i < candidates.size(); ++i) {
    if (candidates[i] - candidates[i - 1] > atol) {
      merged.push_back(candidates[i]);
    }
  }
  // Re-pin lo and hi, as the Python version does after its sort and concatenation.
  merged.front() = lo;
  merged.back() = hi;
  return merged;
}

}  // namespace detail

/// Get the tensor-product overlay of `grid_a` and `grid_b`.
///
/// The overlay's per-axis breakpoints are the sorted union of both inputs' breakpoints restricted
/// to the intersection of their domains. Breakpoints closer than the default *relative* tolerance
/// (`tolerance::get_default<double>()`) times the axis's own magnitude, the intersection window
/// together with the coordinates bounding it, are merged into one, so the merge behaves the same
/// on `[0, 1]` as on `[1e6, 1e6 + 1]`. The result is the coarsest grid that refines both inputs.
///
/// The `TypeError` for an argument that is not a `TensorProductGrid` lives in the Python wrapper.
///
/// Args:
///     grid_a (const TensorProductGrid&): First input grid.
///     grid_b (const TensorProductGrid&): Second input grid; must share `ndim()` with `grid_a` and
///         have a non-degenerate domain intersection on every axis.
///
/// Returns:
///     TensorProductGrid: The overlay grid.
///
/// Raises:
///     value_error: The grids have different `ndim()`, or their domains do not overlap on some
///         axis (the per-axis intersection is empty or degenerate).
[[nodiscard]] inline TensorProductGrid overlay(const TensorProductGrid& grid_a,
                                               const TensorProductGrid& grid_b) {
  if (grid_a.ndim() != grid_b.ndim()) {
    throw value_error("overlay(): grids must share ndim; got " + std::to_string(grid_a.ndim()) +
                      " vs " + std::to_string(grid_b.ndim()) + ".");
  }
  const double relative = tolerance::get_default<double>();
  std::vector<std::vector<double>> merged;
  merged.reserve(grid_a.ndim());
  for (std::size_t d = 0; d < grid_a.ndim(); ++d) {
    const std::span<const double> ba = grid_a.breakpoints(d);
    const std::span<const double> bb = grid_b.breakpoints(d);
    const double lo = (std::max)(ba.front(), bb.front());
    const double hi = (std::min)(ba.back(), bb.back());
    // `relative` is dimensionless; the magnitude a breakpoint comparison on this axis is relative
    // to is the intersection window, taken together with the coordinates bounding it: a window of
    // length 1 sitting at 1e6 resolves its breakpoints no better than eps * 1e6, however short the
    // window is.
    const double atol = relative * (std::max)({hi - lo, std::fabs(lo), std::fabs(hi)});
    if (hi - lo <= atol) {
      throw value_error("overlay(): domains do not overlap on axis " + std::to_string(d) +
                        "; grid_a extent [" + py_repr(ba.front()) + ", " + py_repr(ba.back()) +
                        "] vs grid_b extent [" + py_repr(bb.front()) + ", " + py_repr(bb.back()) +
                        "].");
    }
    merged.push_back(detail::merge_axis_breakpoints(ba, bb, lo, hi, atol));
  }
  return TensorProductGrid(std::move(merged));
}

}  // namespace pantr::grid
