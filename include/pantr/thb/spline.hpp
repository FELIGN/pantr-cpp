/// @file spline.hpp
/// @brief `THBSpline`: a `THBSplineSpace` paired with control points, an evaluable function
/// `f = sum_i c_i phi_i`.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/parallel/parallel_for.hpp"
#include "pantr/parallel/work_model.hpp"
#include "pantr/thb/space.hpp"
#include "pantr/thb/types.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::thb {

namespace detail {

/// Accumulate `out(i, :) = sum_j values(i, j) * cp(dofs[j], :)` for `n` points, summing over `j`
/// in increasing order.
///
/// A `noexcept` function of its own so that GCC vectorises it: written inside a chunk body that may
/// throw, a contracted `fma` leaves the loop scalar.
///
/// Args:
///     values (const double*): Row-major `(n, k)` basis values.
///     dofs (const std::int64_t*): `k` global dofs, the columns of `values`.
///     cp (const double*): Row-major `(num_total_basis, rank)` control points.
///     out (double* const*): `n` row pointers, each to `rank` doubles, written in full.
///     n (std::size_t): Points.
///     k (std::size_t): Functions.
///     rank (std::size_t): Components.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
inline void contract_cell(const double* values, const std::int64_t* dofs, const double* cp,
                          double* const* out, std::size_t n, std::size_t k,
                          std::size_t rank) noexcept {
  for (std::size_t i = 0; i < n; ++i) {
    double* row = out[i];
    for (std::size_t r = 0; r < rank; ++r) {
      row[r] = 0.0;
    }
    for (std::size_t j = 0; j < k; ++j) {
      const double b = values[i * k + j];
      const double* c = cp + static_cast<std::size_t>(dofs[j]) * rank;
      for (std::size_t r = 0; r < rank; ++r) {
        row[r] += b * c[r];
      }
    }
  }
}

/// One entry of the prolongation product: `out(r, :) = sum_e data[e] * cp(indices[e], :)`, over the
/// entries `e` of row `r` in index order, for the rows `[begin, end)`.
///
/// A `noexcept` function of its own, as `contract_cell`.
///
/// Args:
///     p (const ProlongationCsr&): The matrix.
///     cp (const double*): Row-major `(p.cols, rank)` control points.
///     out (double*): Row-major `(p.rows, rank)` result; rows `[begin, end)` are written in full.
///     begin (std::size_t): First row.
///     end (std::size_t): One past the last row.
///     rank (std::size_t): Components.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
inline void prolong_rows(const ProlongationCsr& p, const double* cp, double* out, std::size_t begin,
                         std::size_t end, std::size_t rank) noexcept {
  for (std::size_t r = begin; r < end; ++r) {
    double* row = out + r * rank;
    for (std::size_t c = 0; c < rank; ++c) {
      row[c] = 0.0;
    }
    for (auto e = static_cast<std::size_t>(p.indptr[r]);
         e < static_cast<std::size_t>(p.indptr[r + 1]); ++e) {
      const double w = p.data[e];
      const double* src = cp + static_cast<std::size_t>(p.indices[e]) * rank;
      for (std::size_t c = 0; c < rank; ++c) {
        row[c] += w * src[c];
      }
    }
  }
}

/// A run of the points of one cell: `order[begin, end)` all lie in cell `cid`.
struct CellTask {
  std::int64_t cid = 0;
  std::size_t begin = 0;
  std::size_t end = 0;
};

/// The shared representation of a `THBSpline`: its space and its control points.
///
/// Attributes:
///     space (THBSplineSpace<T>): The hierarchical space (a handle sharing its representation).
///     data (std::shared_ptr<const double[]>): The control points, row-major over `shape`.
///     shape (std::array<std::size_t, 2>): `(num_total_basis, rank)`.
template <StorageFloat T>
struct THBSplineRep {
  THBSplineSpace<T> space;
  std::shared_ptr<const double[]> data;
  std::array<std::size_t, 2> shape;
};

}  // namespace detail

/// An evaluable THB spline function `f = sum_i c_i phi_i`.
///
/// The hierarchical analogue of `bspline::Bspline`: a `THBSplineSpace` with one control point per
/// active function, `(num_total_basis, rank)` row-major (the Python wrapper keeps the scalar and
/// the vector-valued spellings apart). An immutable value (design section 2.2): construction
/// validates the shape and nothing changes afterwards; the space and the control points are shared,
/// read-only, by every copy (a copy is a reference-count increment), so a numeric view of the
/// control points stays valid as long as any copy, or the representation `rep()` returns, lives.
///
/// Evaluation locates each point's active leaf cell, groups the points by cell, and per cell
/// combines the active functions' values with their control points. A point on a cell boundary
/// belongs to the cell the grid's `locate_many` gives it.
///
/// Attributes:
///     rep_ (std::shared_ptr<const detail::THBSplineRep<T>>): The shared representation.
template <StorageFloat T>
class THBSpline {
 public:
  /// The shared representation type.
  using Rep = detail::THBSplineRep<T>;

  /// Build the function.
  ///
  /// Args:
  ///     space (THBSplineSpace<T>): The hierarchical space.
  ///     control_points (NdArray<double>): Extents `(space.num_total_basis(), rank)`, ordered by
  ///         global dof; consumed (an lvalue is copied first).
  ///
  /// Raises:
  ///     value_error: `control_points` is not 2-D, or its leading dimension is not
  ///         `space.num_total_basis()`.
  THBSpline(THBSplineSpace<T> space, NdArray<double> control_points)
      : rep_(make_rep(std::move(space), std::move(control_points))) {}

  /// Get the space. Holds: a reference valid while any copy of the spline lives.
  [[nodiscard]] const THBSplineSpace<T>& space() const noexcept { return rep_->space; }

  /// Get the control points (a numeric view into the shared representation).
  ///
  /// Returns:
  ///     std::span<const double>: Row-major over `shape()`, valid while any copy of the value
  ///     lives.
  [[nodiscard]] std::span<const double> control_points() const noexcept {
    return {rep_->data.get(), rep_->shape[0] * rep_->shape[1]};
  }

  /// Get the extents of the control points.
  ///
  /// Returns:
  ///     std::span<const std::size_t>: `(num_total_basis, rank)`, valid while any copy of the value
  ///     lives.
  [[nodiscard]] std::span<const std::size_t> shape() const noexcept { return rep_->shape; }

  /// Get the shared representation, for an owner that must keep the control points alive (a view
  /// handed to another language).
  ///
  /// Returns:
  ///     const std::shared_ptr<const Rep>&: The representation.
  [[nodiscard]] const std::shared_ptr<const Rep>& rep() const noexcept { return rep_; }

  /// Report whether two values share one representation.
  ///
  /// Args:
  ///     other (const THBSpline&): The other value.
  ///
  /// Returns:
  ///     bool: Whether the representations are the same object.
  [[nodiscard]] bool same_rep(const THBSpline& other) const noexcept { return rep_ == other.rep_; }

  /// Get the number of parametric directions.
  [[nodiscard]] std::size_t dim() const noexcept { return space().dim(); }

  /// Get the number of value components.
  [[nodiscard]] std::size_t rank() const noexcept { return rep_->shape[1]; }

  /// Evaluate one mixed partial derivative of the spline at `pts`, writing into `out`.
  ///
  /// `orders[k]` is the derivative order in direction `k` (an order above the degree gives zero).
  /// A point's value is, per component, the sum over the functions of its cell, in increasing dof
  /// order, of the function's value times its control point. The orders are validated by the cell
  /// tabulation, so an empty batch does not check them (as the Python implementation).
  ///
  /// The points are grouped by cell with a stable sort, and each cell's run is cut into blocks of
  /// `grain_for(ops per point)` points, the points one chunk of `parallel_for` carries, a function
  /// of the problem only. The blocks run in parallel, each computing its points' rows alone and
  /// writing them, so the result is bit-identical for every thread count. The work of one point is
  /// the cost of tabulating the one-dimensional bases of every level in every direction, about
  /// `2 (p + 1)^2` operations each for degree `p`, plus for each of the at most
  /// `max_active_per_cell()` functions of a cell `dim` multiplications for the product and `2 rank`
  /// for the contraction. The sum over the coefficient box of a truncated function is not counted,
  /// so the estimate errs low there (larger blocks), which costs balance and not correctness.
  ///
  /// Args:
  ///     pts (mdview<const double, 2>): Extents `(m, dim())`.
  ///     orders (std::span<const std::int64_t>): `dim()` non-negative derivative orders.
  ///     out (mdview<double, 2>): Extents `(m, rank())`; unspecified when the call throws.
  ///
  /// Raises:
  ///     value_error: `pts` or `out` has another shape; a point lies outside the grid domain (the
  ///         first in input order is named); `orders` has the wrong length or a negative entry, or
  ///         a point lies outside its cell up to the membership tolerance.
  ///     state_error: A cell has no active function; a 1D derivative recurrence met a zero
  ///         divisor.
  void evaluate(mdview<const double, 2> pts, std::span<const std::int64_t> orders,
                mdview<double, 2> out) const {
    const std::size_t d = dim();
    const std::size_t m = pts.extent(0);
    const std::size_t rk = rank();
    if (pts.extent(1) != d) {
      const std::size_t got[2] = {pts.extent(0), pts.extent(1)};
      throw value_error("pts must have trailing dimension " + std::to_string(d) + "; got shape " +
                        py_repr(std::span<const std::size_t>(got)) + ".");
    }
    if (out.extent(0) != m || out.extent(1) != rk) {
      const std::size_t want[2] = {m, rk};
      const std::size_t got[2] = {out.extent(0), out.extent(1)};
      throw value_error("out must have shape " + py_repr(std::span<const std::size_t>(want)) +
                        "; got " + py_repr(std::span<const std::size_t>(got)) + ".");
    }
    if (m == 0) {
      return;
    }
    const grid::HierarchicalGrid grid = space().grid();
    const NdArray<std::int64_t> cids =
        grid.locate_many(std::span<const double>(pts.data_handle(), m * d), m);
    const std::int64_t* cell_of = cids.data();
    for (std::size_t i = 0; i < m; ++i) {
      if (cell_of[i] < 0) {
        std::string point = "[";
        for (std::size_t k = 0; k < d; ++k) {
          point += (k > 0 ? ", " : "") + py_repr(pts(i, k));
        }
        throw value_error("point " + point + "] lies outside the grid domain.");
      }
    }

    // Group the points by cell, keeping the input order within a cell.
    std::vector<std::size_t> order(m);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [cell_of](std::size_t a, std::size_t b) { return cell_of[a] < cell_of[b]; });
    const parallel::WorkEstimate per_point = point_work(rk);
    const std::size_t block = parallel::grain_for(per_point);
    std::vector<detail::CellTask> tasks;
    for (std::size_t s = 0; s < m;) {
      std::size_t e = s;
      while (e < m && cell_of[order[e]] == cell_of[order[s]]) {
        ++e;
      }
      for (std::size_t b = s; b < e; b += block) {
        tasks.push_back({cell_of[order[s]], b, (std::min)(e, b + block)});
      }
      s = e;
    }
    const parallel::WorkEstimate per_task{(std::max)(
        1.0, per_point.ops_per_item * static_cast<double>(m) / static_cast<double>(tasks.size()))};

    const double* cp = rep_->data.get();
    parallel::parallel_for(tasks.size(), per_task, [&](std::size_t begin, std::size_t end) {
      std::vector<double> local;
      std::vector<double> values;
      std::vector<std::int64_t> dofs;
      std::vector<double*> rows;
      for (std::size_t t = begin; t < end; ++t) {
        const detail::CellTask& task = tasks[t];
        const std::size_t n = task.end - task.begin;
        const std::size_t k = space().num_active_on(task.cid);
        if (k == 0) {
          throw state_error("cell " + std::to_string(task.cid) +
                            " has no active basis functions; the THBSplineSpace may be "
                            "inconsistent.");
        }
        local.resize(n * d);
        rows.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
          const std::size_t at = order[task.begin + i];
          for (std::size_t c = 0; c < d; ++c) {
            local[i * d + c] = pts(at, c);
          }
          rows[i] = out.data_handle() + at * rk;
        }
        values.resize(n * k);
        dofs.resize(k);
        space().tabulate_basis(task.cid, mdview<const double, 2>(local.data(), n, d), orders,
                               mdview<double, 2>(values.data(), n, k), dofs);
        detail::contract_cell(values.data(), dofs.data(), cp, rows.data(), n, k, rk);
      }
    });
  }

  /// Evaluate one mixed partial derivative of the spline at `pts`.
  ///
  /// Returns:
  ///     NdArray<double>: Extents `(m, rank())`.
  ///
  /// Raises:
  ///     value_error, state_error: As the overload with `out`.
  [[nodiscard]] NdArray<double> evaluate(mdview<const double, 2> pts,
                                         std::span<const std::int64_t> orders) const {
    // Every row is written by `evaluate`, or the call throws.
    NdArray<double> out({pts.extent(0), rank()}, for_overwrite);
    evaluate(pts, orders, out.template view<2>());
    return out;
  }

  /// Evaluate the spline at `pts` (all orders zero).
  ///
  /// Returns:
  ///     NdArray<double>: Extents `(m, rank())`.
  ///
  /// Raises:
  ///     value_error, state_error: As the overload with `out`.
  [[nodiscard]] NdArray<double> evaluate(mdview<const double, 2> pts) const {
    const std::vector<std::int64_t> zeros(dim(), 0);
    return evaluate(pts, zeros);
  }

  /// Get the same function on the space with the named cells refined.
  ///
  /// Refines the space (`THBSplineSpace::refine`) and prolongs the control points onto it; the
  /// spaces are nested, so the function is the same. This spline is not changed.
  ///
  /// Args:
  ///     cell_ids (std::span<const std::int64_t>): Active cell ids to refine; repeats are ignored.
  ///     admissible_class (std::optional<std::int64_t>): The class `m >= 2` to maintain, or no
  ///         value for ungraded refinement.
  ///
  /// Returns:
  ///     THBSpline: The spline on the refined space.
  ///
  /// Raises:
  ///     value_error, index_error, state_error: As `THBSplineSpace::refine` and `prolong_to`.
  [[nodiscard]] THBSpline refine(std::span<const std::int64_t> cell_ids,
                                 std::optional<std::int64_t> admissible_class) const {
    return prolong_to(space().refine(cell_ids, admissible_class));
  }

  /// Get the same function on the space with a region refined.
  ///
  /// Args:
  ///     level (std::int64_t): Level the box lives at.
  ///     lo (std::span<const std::int64_t>): Per-direction start (inclusive).
  ///     hi (std::span<const std::int64_t>): Per-direction end (exclusive).
  ///     admissible_class (std::optional<std::int64_t>): The class `m >= 2` to maintain, or no
  ///         value for ungraded refinement.
  ///
  /// Returns:
  ///     THBSpline: The spline on the refined space.
  ///
  /// Raises:
  ///     value_error, index_error, state_error: As `THBSplineSpace::refine_region` and
  ///         `prolong_to`.
  [[nodiscard]] THBSpline refine_region(std::int64_t level, std::span<const std::int64_t> lo,
                                        std::span<const std::int64_t> hi,
                                        std::optional<std::int64_t> admissible_class) const {
    return prolong_to(space().refine_region(level, lo, hi, admissible_class));
  }

  /// Get the same function on a refinement of the space.
  ///
  /// The control points are multiplied by the CSR prolongation (`prolongation_csr`), each fine
  /// control point summing its row's entries in column order; the rows are independent and run in
  /// parallel, so the result does not depend on the thread count. (The Python implementation
  /// multiplies by a dense matrix; the two differ in the last bits.) A row costs `2 rank`
  /// operations per stored entry, so the work per row is `2 rank` times the mean row length.
  ///
  /// Args:
  ///     fine (const THBSplineSpace<T>&): A refinement of this spline's space.
  ///
  /// Returns:
  ///     THBSpline: The spline on `fine`.
  ///
  /// Raises:
  ///     value_error: `fine` is not a refinement of the space (see `prolongation_csr`).
  [[nodiscard]] THBSpline prolong_to(const THBSplineSpace<T>& fine) const {
    const ProlongationCsr p = space().prolongation_csr(fine);
    const std::size_t rk = rank();
    const auto rows = static_cast<std::size_t>(p.rows);
    NdArray<double> out({rows, rk}, for_overwrite);  // every row is zeroed, then accumulated
    const double mean_row =
        rows == 0 ? 1.0 : static_cast<double>(p.indices.size()) / static_cast<double>(rows);
    const parallel::WorkEstimate work{(std::max)(1.0, 2.0 * static_cast<double>(rk) * mean_row)};
    const double* cp = rep_->data.get();
    double* dst = out.data();
    parallel::parallel_for(rows, work, [&p, cp, dst, rk](std::size_t begin, std::size_t end) {
      detail::prolong_rows(p, cp, dst, begin, end, rk);
    });
    return THBSpline(fine, std::move(out));
  }

 private:
  // Operations to evaluate one point; see `evaluate`.
  [[nodiscard]] parallel::WorkEstimate point_work(std::size_t rk) const {
    const auto d = static_cast<double>(dim());
    int p_max = 0;
    for (const int p : space().degrees()) {
      p_max = (std::max)(p_max, p);
    }
    const auto width = static_cast<double>(p_max) + 1.0;
    const double tabulation = static_cast<double>(space().num_levels()) * d * 2.0 * width * width;
    const double per_function = d + 2.0 * static_cast<double>(rk);
    return parallel::WorkEstimate{tabulation + static_cast<double>(space().max_active_per_cell()) *
                                                   per_function};
  }

  static std::shared_ptr<const Rep> make_rep(THBSplineSpace<T> space,
                                             NdArray<double> control_points) {
    if (control_points.rank() != 2) {
      throw value_error("control_points must be 2-D; got " + std::to_string(control_points.rank()) +
                        "-D.");
    }
    const auto n = static_cast<std::size_t>(space.num_total_basis());
    if (control_points.shape()[0] != n) {
      throw value_error("control_points leading dimension must be " + std::to_string(n) +
                        "; got shape " + py_repr(control_points.shape()) + ".");
    }
    const std::array<std::size_t, 2> shape = {control_points.shape()[0], control_points.shape()[1]};
    return std::make_shared<const Rep>(
        Rep{std::move(space),
            std::shared_ptr<const double[]>(std::move(control_points).into_buffer()), shape});
  }

  std::shared_ptr<const Rep> rep_;
};

}  // namespace pantr::thb
