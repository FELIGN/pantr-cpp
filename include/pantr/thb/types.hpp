/// @file types.hpp
/// @brief The value types `THBSplineSpace` hands out: a truncated function, a cell's active
/// functions, the cell-to-dof incidence and a tabulation.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/ndarray.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace pantr::thb {

/// A truncated function: a box of coefficients in the tensor-product basis of one level.
///
/// The function equals `sum_multi coeffs[multi] * prod_k B^{rep_level}_{box_lo[k] + multi[k]}`
/// over the box, where `B^l_i` is the level-`l` B-spline of direction `k`.
///
/// Attributes:
///     rep_level (std::int64_t): The finest level the function is expressed in.
///     box_lo (std::vector<std::int64_t>): Per-direction index of the box's first function.
///     box_shape (std::vector<std::size_t>): Per-direction function count of the box.
///     coeffs (std::vector<double>): The coefficients, row-major (C order) over `box_shape`.
struct TruncatedFunction {
  std::int64_t rep_level = 0;
  std::vector<std::int64_t> box_lo;
  std::vector<std::size_t> box_shape;
  std::vector<double> coeffs;
};

/// The active functions that are non-zero on one cell, in increasing order of global dof.
///
/// A view into the space's per-cell table, valid while the space (or a copy of it) lives.
///
/// Attributes:
///     dofs (std::span<const std::int64_t>): Global dof of each function.
///     levels (std::span<const std::int64_t>): Level of each function.
///     multi (std::span<const std::int64_t>): Row-major `(size(), dim)` per-direction function
///         index of each function in its level's space.
///     dim (std::size_t): Number of parametric directions.
struct CellContributions {
  std::span<const std::int64_t> dofs;
  std::span<const std::int64_t> levels;
  std::span<const std::int64_t> multi;
  std::size_t dim = 0;

  /// Get the number of functions.
  [[nodiscard]] std::size_t size() const noexcept { return dofs.size(); }

  /// Get the per-direction function index of the `i`-th function.
  [[nodiscard]] std::span<const std::int64_t> multi_index(std::size_t i) const noexcept {
    return multi.subspan(i * dim, dim);
  }
};

/// The incidence of cells and global dofs, in compressed-row form.
///
/// Attributes:
///     cell_ptr (std::span<const std::int64_t>): `num_cells + 1` offsets; cell `c` owns
///         `dofs[cell_ptr[c] .. cell_ptr[c + 1])`.
///     dofs (std::span<const std::int64_t>): Global dofs, increasing within a cell.
struct CellIncidence {
  std::span<const std::int64_t> cell_ptr;
  std::span<const std::int64_t> dofs;
};

/// A tabulation of the active functions on a cell.
///
/// Attributes:
///     values (NdArray<double>): Extents `(n, K)`: function `dofs[j]` at point `i`.
///     dofs (NdArray<std::int64_t>): Extents `(K,)`: the global dofs, increasing.
struct [[nodiscard]] TabulatedBasis {
  NdArray<double> values;
  NdArray<std::int64_t> dofs;
};

}  // namespace pantr::thb
