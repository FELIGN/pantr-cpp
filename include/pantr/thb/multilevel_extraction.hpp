/// @file multilevel_extraction.hpp
/// @brief `MultiLevelExtraction`: the per-cell multi-level (Bezier) extraction operators of a THB
/// space.
///
/// On an active cell of level `L` the active hierarchical functions `H` are a linear combination of
/// the level-`L` tensor-product B-splines `N` with support on the cell, `H = M N` (the multi-level
/// extraction operator `M`, D'Angella et al.), and composing with the single-level extraction `E`
/// of the cell, `N = E B`, gives `H = (M E) B` on the reference basis `B`. `M` is built from the
/// space's own truncation (Giannelli, Juettler and Speleers), which keeps and refines forward the
/// passive functions that straddle a refinement boundary; it does not use the activeness-restricted
/// local truncation of the 2017 paper.
#pragma once

#include "pantr/basis/detail/tabulate_1d.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/spanwise_extraction.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/grid/detail/py_text.hpp"
#include "pantr/thb/detail/truncation.hpp"
#include "pantr/thb/space.hpp"
#include "pantr/thb/types.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::thb {

namespace detail {

/// `out(i, :) = sum_j m(i, j) * e(j, :)` for the rows `[0, rows)`, summing over `j` in increasing
/// order.
///
/// A `noexcept` function of its own so that GCC vectorises it, as `contract_cell`.
///
/// Args:
///     m (const double*): Row-major `(rows, inner)`.
///     e (const double*): Row-major `(inner, cols)`.
///     out (double*): Row-major `(rows, cols)`, written in full.
///     rows (std::size_t): Rows of `m`.
///     inner (std::size_t): Columns of `m` and rows of `e`.
///     cols (std::size_t): Columns of `e`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
inline void multiply_rows(const double* m, const double* e, double* out, std::size_t rows,
                          std::size_t inner, std::size_t cols) noexcept {
  for (std::size_t i = 0; i < rows; ++i) {
    double* row = out + i * cols;
    for (std::size_t c = 0; c < cols; ++c) {
      row[c] = 0.0;
    }
    for (std::size_t j = 0; j < inner; ++j) {
      const double w = m[i * inner + j];
      const double* src = e + j * cols;
      for (std::size_t c = 0; c < cols; ++c) {
        row[c] += w * src[c];
      }
    }
  }
}

}  // namespace detail

/// Per-cell multi-level (Bezier) extraction for a `THBSplineSpace`.
///
/// Mirrors `bspline::SpanwiseElementExtraction`: built from a space and a target reference basis,
/// it hands out per-cell operators. The number of active functions `K` differs between cells and
/// the hierarchical basis has no tensor-product structure, so the operators are ragged across
/// cells. With `d` directions of degrees `p_k`, `n = prod(p_k + 1)`:
///   - `multilevel_operator(cid)` is `M`, `(K, n)`: the level-`L` tensor-product B-splines of the
///     cell to the active functions (independent of the target);
///   - `extraction_operator(cid)` is `M E`, `(K, n_in)`: the target basis of the cell to the active
///     functions, `n_in = prod(input_shape_per_dir)` (equal to `n` for every target).
/// Rows follow `active_basis(cid)`, the sorted global dofs.
///
/// An immutable value (design section 2.2): the per-level single-level extractions are built once,
/// eagerly, in the constructor, so a failure of theirs (a degree beyond the target's table)
/// surfaces there and not at the first operator; the Python class builds them lazily. There is no
/// memo of the refined coefficient boxes (Python's `_coeffs_cache`): a cached entry would be
/// mutable state of a value, and a function's box is recomputed per cell row by at most `levels -
/// 1` refinements.
///
/// Attributes:
///     space_ (THBSplineSpace<T>): The space.
///     target_ (bspline::ExtractionTarget): The single-level reference basis.
///     levels_ (std::vector<bspline::SpanwiseElementExtraction<T>>): The extraction of each
///         level's space.
template <StorageFloat T>
class MultiLevelExtraction {
 public:
  /// Build the extraction of every level.
  ///
  /// Args:
  ///     space (THBSplineSpace<T>): The space, truncated or not.
  ///     target (bspline::ExtractionTarget): The single-level reference basis. Defaults to Bezier.
  ///
  /// Raises:
  ///     value_error, capacity_error: As `bspline::SpanwiseElementExtraction` for a level space.
  explicit MultiLevelExtraction(
      THBSplineSpace<T> space, bspline::ExtractionTarget target = bspline::ExtractionTarget::bezier)
      : space_(std::move(space)), target_(target) {
    for (std::int64_t level = 0; level < space_.num_levels(); ++level) {
      levels_.emplace_back(space_.level_space(level), target_);
    }
  }

  /// Get the space. Holds: a reference valid while the object lives.
  [[nodiscard]] const THBSplineSpace<T>& space() const noexcept { return space_; }

  /// Get the single-level reference basis.
  [[nodiscard]] bspline::ExtractionTarget target() const noexcept { return target_; }

  /// Get the number of parametric directions.
  [[nodiscard]] std::size_t dim() const noexcept { return space_.dim(); }

  /// Get the number of active cells (elements).
  [[nodiscard]] std::int64_t num_elements() const { return space_.grid().num_cells(); }

  /// Get the global dofs labelling the rows of the operators on cell `cid`.
  ///
  /// Returns:
  ///     std::span<const std::int64_t>: Sorted global dofs, `THBSplineSpace::active_basis`.
  ///
  /// Raises:
  ///     index_error: `cid` is out of range.
  [[nodiscard]] std::span<const std::int64_t> active_basis(std::int64_t cid) const {
    return space_.active_basis(cid);
  }

  /// Write the multi-level extraction operator `M` of cell `cid` into `out`.
  ///
  /// Row `j` holds the coefficients of active function `j` in the level-`L` B-splines supported on
  /// the cell, in C order over the `(p_k + 1)` functions per direction. A function's coefficients
  /// are its level-`origin` B-spline refined to the cell's level `L` one level at a time, each step
  /// followed by the truncation against the next level's active functions when the space truncates
  /// (`refine_box`, `truncate_box`), restricted to the cell's window. The refinement and the
  /// copy are exact arithmetic of a fixed order, so the result does not depend on a thread count.
  ///
  /// Args:
  ///     cid (std::int64_t): Active cell id in `[0, num_elements())`.
  ///     out (mdview<double, 2>): Extents `(K, n)`, `K = active_basis(cid).size()`; overwritten.
  ///
  /// Raises:
  ///     index_error: `cid` is out of range.
  ///     value_error: `out` has another shape; a refinement step meets a zero two-scale slice.
  ///     state_error: A refined coefficient box does not overlap the cell's window (an
  ///         invariant of a valid space).
  void multilevel_operator(std::int64_t cid, mdview<double, 2> out) const {
    const CellContributions cell = space_.contributions(cid);
    const std::size_t d = dim();
    const std::size_t k = cell.size();
    const grid::HierarchicalGrid grid = space_.grid();
    const std::int64_t level = grid.cell_level(cid);
    const std::vector<std::int64_t> midx = grid.cell_multi_index(cid);
    std::vector<std::size_t> n_per(d);
    std::vector<std::int64_t> first_basis(d);
    std::size_t n_single = 1;
    for (std::size_t a = 0; a < d; ++a) {
      n_per[a] = static_cast<std::size_t>(space_.degrees()[a]) + 1;
      n_single *= n_per[a];
      first_basis[a] = space_.support(level, a).first_basis[static_cast<std::size_t>(midx[a])];
    }
    basis::detail::check_out<double, 2>(out, {k, n_single});
    std::fill(out.data_handle(), out.data_handle() + k * n_single, 0.0);

    std::vector<std::size_t> at(d);
    for (std::size_t row = 0; row < k; ++row) {
      const std::int64_t origin = cell.levels[row];
      const std::span<const std::int64_t> multi = cell.multi_index(row);
      const detail::CoeffBox box = element_coeffs(origin, multi, level);
      // The overlap of the box with the cell's window `[first_basis, first_basis + n_per)`, per
      // direction: window indices `[j0, j1)` read the box at `offset + j`.
      std::vector<std::int64_t> j0(d);
      std::vector<std::int64_t> j1(d);
      std::vector<std::int64_t> offset(d);
      for (std::size_t a = 0; a < d; ++a) {
        offset[a] = first_basis[a] - box.lo[a];
        j0[a] = (std::max)(std::int64_t{0}, -offset[a]);
        j1[a] = (std::min)(static_cast<std::int64_t>(n_per[a]),
                           static_cast<std::int64_t>(box.shape[a]) - offset[a]);
        if (j1[a] <= j0[a]) {
          throw state_error("multilevel_operator: cell " + std::to_string(cid) + " row " +
                            std::to_string(row) + " (origin_level=" + std::to_string(origin) +
                            ", multi=" + grid::detail::py_repr_ints(multi) +
                            ") \xE2\x80\x94 refined coefficient box does not overlap the cell "
                            "window. This is a bug; please report it with the space and grid "
                            "configuration.");
        }
      }
      // Odometer over the window's overlap, C order on both sides.
      std::vector<std::int64_t> idx(j0);
      for (;;) {
        std::size_t dst = 0;
        std::size_t src = 0;
        for (std::size_t a = 0; a < d; ++a) {
          dst = dst * n_per[a] + static_cast<std::size_t>(idx[a]);
          src = src * box.shape[a] + static_cast<std::size_t>(offset[a] + idx[a]);
        }
        out(row, dst) = box.coeffs[src];
        std::size_t a = d;
        while (a-- > 0) {
          if (++idx[a] < j1[a]) {
            break;
          }
          idx[a] = j0[a];
        }
        if (a == static_cast<std::size_t>(-1)) {
          break;
        }
      }
    }
  }

  /// Get the multi-level extraction operator `M` of cell `cid`.
  ///
  /// Returns:
  ///     NdArray<double>: Extents `(K, n)`.
  ///
  /// Raises:
  ///     index_error, value_error, state_error: As the overload with `out`.
  [[nodiscard]] NdArray<double> multilevel_operator(std::int64_t cid) const {
    std::size_t n_single = 1;
    for (const int p : space_.degrees()) {
      n_single *= static_cast<std::size_t>(p) + 1;
    }
    NdArray<double> out({space_.num_active_on(cid), n_single}, for_overwrite);
    multilevel_operator(cid, out.template view<2>());
    return out;
  }

  /// Write the multi-level extraction `M E` of cell `cid` into `out`.
  ///
  /// `E` is the cell's single-level operator (`SpanwiseElementExtraction::kron_operator` of the
  /// cell's level), widened to `double`; the product sums over the `n` inner indices in increasing
  /// order (the Python class multiplies with a BLAS, so the last bits differ).
  ///
  /// Args:
  ///     cid (std::int64_t): Active cell id in `[0, num_elements())`.
  ///     out (mdview<double, 2>): Extents `(K, n_in)`; overwritten.
  ///
  /// Raises:
  ///     index_error, value_error, state_error: As `multilevel_operator`.
  void extraction_operator(std::int64_t cid, mdview<double, 2> out) const {
    const grid::HierarchicalGrid grid = space_.grid();
    const std::int64_t level = grid.cell_level(cid);
    const bspline::SpanwiseElementExtraction<T>& ext = levels_[static_cast<std::size_t>(level)];
    std::size_t n_in = 1;
    std::size_t n_out = 1;
    for (const std::size_t c : ext.input_shape_per_dir()) {
      n_in *= c;
    }
    for (const std::size_t r : ext.output_shape_per_dir()) {
      n_out *= r;
    }
    const std::size_t k = space_.num_active_on(cid);
    basis::detail::check_out<double, 2>(out, {k, n_in});
    // `E` has `n_out = prod(p_k + 1)` rows, the width of `M`, for every target (each direction's
    // operator is `(p + 1) x (p + 1)`).
    const std::vector<std::int64_t> midx = grid.cell_multi_index(cid);
    const std::vector<std::size_t> cell(midx.begin(), midx.end());
    std::vector<T> single(n_out * n_in);
    ext.kron_operator(cell, mdview<T, 2>(single.data(), n_out, n_in));
    std::vector<double> widened(single.begin(), single.end());
    NdArray<double> multilevel = multilevel_operator(cid);
    detail::multiply_rows(multilevel.data(), widened.data(), out.data_handle(), k, n_out, n_in);
  }

  /// Get the multi-level extraction `M E` of cell `cid`.
  ///
  /// Returns:
  ///     NdArray<double>: Extents `(K, n_in)`.
  ///
  /// Raises:
  ///     index_error, value_error, state_error: As `multilevel_operator`.
  [[nodiscard]] NdArray<double> extraction_operator(std::int64_t cid) const {
    const bspline::SpanwiseElementExtraction<T>& ext =
        levels_[static_cast<std::size_t>(space_.grid().cell_level(cid))];
    std::size_t n_in = 1;
    for (const std::size_t c : ext.input_shape_per_dir()) {
      n_in *= c;
    }
    NdArray<double> out({space_.num_active_on(cid), n_in}, for_overwrite);
    extraction_operator(cid, out.template view<2>());
    return out;
  }

 private:
  // The function with multi-index `multi` of level `origin` in the level-`target_level` basis: the
  // box of coefficients after the refinement from `origin` to `target_level`, truncated at each
  // level when the space truncates.
  [[nodiscard]] detail::CoeffBox element_coeffs(std::int64_t origin,
                                                std::span<const std::int64_t> multi,
                                                std::int64_t target_level) const {
    const std::size_t d = dim();
    detail::CoeffBox box;
    box.lo.assign(multi.begin(), multi.end());
    box.shape.assign(d, 1);
    box.coeffs.assign(1, 1.0);
    const Rep& rep = *space_.rep();
    for (std::int64_t l = origin; l < target_level; ++l) {
      const auto lv = static_cast<std::size_t>(l);
      box = detail::refine_box(box, rep.two_scale[lv]);
      if (rep.truncate) {
        detail::truncate_box(box, rep.active[lv + 1], rep.num_basis[lv + 1]);
      }
    }
    return box;
  }

  using Rep = typename THBSplineSpace<T>::Rep;

  THBSplineSpace<T> space_;
  bspline::ExtractionTarget target_;
  std::vector<bspline::SpanwiseElementExtraction<T>> levels_;
};

}  // namespace pantr::thb
