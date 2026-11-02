/// @file space.hpp
/// @brief `THBSplineSpace`: a truncated hierarchical B-spline space on a `HierarchicalGrid`.
///
/// The space follows the self-evaluating model: per level it stores the Kraft selection of active
/// tensor-product B-splines, and a coefficient box only for truncated functions; an untruncated
/// function is a plain tensor-product B-spline. With `truncate` true the basis is the truncated
/// hierarchical basis (THB, which keeps the partition of unity), otherwise the hierarchical basis
/// (HB).
#pragma once

#include "pantr/bspline/detail/support_1d.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/grid/detail/py_text.hpp"
#include "pantr/grid/hierarchical_grid.hpp"
#include "pantr/thb/detail/evaluate.hpp"
#include "pantr/thb/detail/prolongation.hpp"
#include "pantr/thb/detail/refine.hpp"
#include "pantr/thb/detail/rep.hpp"
#include "pantr/thb/detail/restrict.hpp"
#include "pantr/thb/detail/truncation.hpp"
#include "pantr/thb/tolerances.hpp"
#include "pantr/thb/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::thb {

// Defined after the space, which it holds by value.
template <StorageFloat T>
struct THBSplineSpaceRestriction;

/// A hierarchical B-spline space on a `HierarchicalGrid`, truncated or not.
///
/// Built from a root `BsplineSpace` (level 0), the grid carrying the active-cell hierarchy, and a
/// per-direction regularity. The level spaces are the root subdivided uniformly by the grid's
/// per-direction factor. The active basis is the Kraft selection: a level-`l` tensor-product
/// B-spline is active iff its support lies in the level-`l` subdomain but not entirely in the
/// finer one. With `truncate()` the active functions that straddle a refinement boundary have
/// their components on finer active functions removed (Giannelli, Juettler and Speleers), which
/// restores the partition of unity.
///
/// An immutable value (design section 2.2): everything is computed at construction except the
/// per-cell table of active functions, one lazy block computed for every cell at its first use,
/// thread-safely. Copies share one representation. The grid is itself an immutable value, so there
/// is no stale space: a refined grid makes a new space.
///
/// `active_basis` lists the functions whose untruncated support covers a cell; under truncation a
/// few may evaluate to exactly zero there. `tabulate_basis` always returns the correct (possibly
/// zero) values.
///
/// Attributes:
///     rep_ (std::shared_ptr<const detail::THBSpaceRep<T>>): The shared representation.
template <StorageFloat T>
class THBSplineSpace {
 public:
  /// The shared representation type.
  using Rep = detail::THBSpaceRep<T>;

  /// Build the space, computing the level spaces, the active functions and the truncated
  /// coefficients.
  ///
  /// Args:
  ///     root (bspline::BsplineSpace<T>): The level-0 tensor-product space.
  ///     hierarchy (grid::HierarchicalGrid): A grid whose root cells and bounds match `root`.
  ///     truncate (bool): Build the truncated basis (THB) rather than the plain one (HB).
  ///     regularity (std::vector<std::optional<int>>): Per-direction continuity at the knots
  ///         inserted when subdividing, `dim` entries (a scalar is broadcast by the caller); no
  ///         value means maximal smoothness, `degree - 1`. A value must lie in `[-1, degree)`.
  ///
  /// Raises:
  ///     value_error: `hierarchy` and `root` disagree on the dimension, on the root knot-span
  ///         counts or on the domain; `regularity` has the wrong length or a value out of range;
  ///         the subdivision merges knots (a level has fewer intervals than the grid has cells);
  ///         a refinement of a truncated function meets a zero two-scale slice. A periodic root
  ///         is refused with the `value_error` of `BsplineSpace1D::first_basis_per_interval`.
  ///     state_error: A function of a level space has an empty support (an invalid space).
  THBSplineSpace(bspline::BsplineSpace<T> root, grid::HierarchicalGrid hierarchy, bool truncate,
                 std::vector<std::optional<int>> regularity)
      : rep_(build(std::move(root), std::move(hierarchy), truncate, std::move(regularity))) {}

  /// Get the underlying hierarchical grid. Holds: a copy of the handle, sharing the grid.
  [[nodiscard]] grid::HierarchicalGrid grid() const { return rep_->hier; }

  /// Get the level-0 tensor-product space. Holds: a copy of the handle.
  [[nodiscard]] bspline::BsplineSpace<T> root_space() const { return rep_->root_space; }

  /// Get the number of parametric directions.
  [[nodiscard]] std::size_t dim() const noexcept { return rep_->root_space.dim(); }

  /// Get the per-direction polynomial degrees (the same at every level).
  [[nodiscard]] std::span<const int> degrees() const noexcept { return rep_->degrees; }

  /// Get the number of hierarchy levels.
  [[nodiscard]] std::int64_t num_levels() const noexcept {
    return static_cast<std::int64_t>(rep_->level_spaces.size());
  }

  /// Report whether the basis is truncated (THB) rather than the plain hierarchical one (HB).
  [[nodiscard]] bool truncate() const noexcept { return rep_->truncate; }

  /// Get the per-direction continuity used when subdividing to build the finer levels.
  ///
  /// Returns:
  ///     std::span<const std::optional<int>>: `dim()` entries; no value means maximal smoothness.
  [[nodiscard]] std::span<const std::optional<int>> regularity() const noexcept {
    return rep_->regularity;
  }

  /// Get the total number of active hierarchical basis functions.
  [[nodiscard]] std::int64_t num_total_basis() const noexcept { return rep_->func_offset.back(); }

  /// Get the number of active basis functions at each level, `num_levels()` entries.
  [[nodiscard]] std::vector<std::int64_t> num_basis_per_level() const {
    std::vector<std::int64_t> counts(rep_->active.size());
    for (std::size_t l = 0; l < counts.size(); ++l) {
      counts[l] = static_cast<std::int64_t>(rep_->active[l].size());
    }
    return counts;
  }

  /// Get the global dof of each level's first active function.
  ///
  /// Returns:
  ///     std::span<const std::int64_t>: `num_levels() + 1` cumulative active counts: level `l`
  ///     owns the dofs `[func_offset()[l], func_offset()[l + 1])`.
  [[nodiscard]] std::span<const std::int64_t> func_offset() const noexcept {
    return rep_->func_offset;
  }

  /// Get the numerical tolerance, the root space's.
  [[nodiscard]] double tolerance() const noexcept { return rep_->root_space.tolerance(); }

  /// Get the tensor-product space at `level`: the root subdivided to that level. Holds: a copy of
  /// the handle.
  ///
  /// Args:
  ///     level (std::int64_t): Hierarchy level in `[0, num_levels())`.
  ///
  /// Raises:
  ///     value_error: `level` is out of range.
  [[nodiscard]] bspline::BsplineSpace<T> level_space(std::int64_t level) const {
    return rep_->level_spaces[check_level(level)];
  }

  /// Get the flat indices of the active functions at `level`.
  ///
  /// Args:
  ///     level (std::int64_t): Hierarchy level in `[0, num_levels())`.
  ///
  /// Returns:
  ///     std::span<const std::int64_t>: Sorted flat (row-major) level-`level` function indices
  ///     selected by the Kraft rule; a view into the space.
  ///
  /// Raises:
  ///     value_error: `level` is out of range.
  [[nodiscard]] std::span<const std::int64_t> active_function_indices(std::int64_t level) const {
    return rep_->active[check_level(level)];
  }

  /// Get the global dofs of the active functions whose support meets cell `cid`.
  ///
  /// The first call computes the per-cell table of every cell.
  ///
  /// Args:
  ///     cid (std::int64_t): Active cell id in `[0, grid().num_cells())`.
  ///
  /// Returns:
  ///     std::span<const std::int64_t>: Sorted global dofs; a view into the space's table.
  ///
  /// Raises:
  ///     index_error: `cid` is out of range.
  [[nodiscard]] std::span<const std::int64_t> active_basis(std::int64_t cid) const {
    return contributions(cid).dofs;
  }

  /// Get the number of active functions whose support meets cell `cid`: the width an output of
  /// `tabulate_basis` needs.
  ///
  /// Raises:
  ///     index_error: `cid` is out of range.
  [[nodiscard]] std::size_t num_active_on(std::int64_t cid) const {
    return contributions(cid).size();
  }

  /// Get the active functions non-zero on cell `cid`, with their levels and multi-indices.
  ///
  /// Args:
  ///     cid (std::int64_t): Active cell id in `[0, grid().num_cells())`.
  ///
  /// Returns:
  ///     CellContributions: Entries sorted by global dof; a view into the space's table.
  ///
  /// Raises:
  ///     index_error: `cid` is out of range.
  [[nodiscard]] CellContributions contributions(std::int64_t cid) const {
    rep_->hier.check_cid(cid);
    const detail::CellBlock& block = cell_block();
    const auto c = static_cast<std::size_t>(cid);
    const auto begin = static_cast<std::size_t>(block.cell_ptr[c]);
    const auto count = static_cast<std::size_t>(block.cell_ptr[c + 1]) - begin;
    const std::size_t d = dim();
    return CellContributions{
        std::span<const std::int64_t>(block.dofs).subspan(begin, count),
        std::span<const std::int64_t>(block.levels).subspan(begin, count),
        std::span<const std::int64_t>(block.multi).subspan(begin * d, count * d), d};
  }

  /// Get the largest number of active functions on any single cell.
  ///
  /// The width a fixed-size dofmap needs. On an unrefined space this is `prod(degree + 1)`; near a
  /// level interface a cell also sees the coarser functions overlapping it, so the count grows
  /// there. Counts exactly what `active_basis` returns, which selects on tensor-product support:
  /// truncation can only annihilate a function on a cell it supports, never add one, so with
  /// truncation this is an upper bound on the number of non-zero functions, and the value is the
  /// same for the THB and HB bases.
  ///
  /// The first call computes the per-cell table of every cell.
  [[nodiscard]] std::int64_t max_active_per_cell() const { return cell_block().max_active; }

  /// Get the incidence of cells and global dofs, for a coupling graph.
  ///
  /// The first call computes the per-cell table of every cell.
  ///
  /// Returns:
  ///     CellIncidence: Views into the space's table.
  [[nodiscard]] CellIncidence cell_dof_incidence() const {
    const detail::CellBlock& block = cell_block();
    return CellIncidence{block.cell_ptr, block.dofs};
  }

  /// Get the stored coefficients of global dof `dof`, if the function is truncated.
  ///
  /// Args:
  ///     dof (std::int64_t): Global dof in `[0, num_total_basis())`.
  ///
  /// Returns:
  ///     const TruncatedFunction*: The function's box, valid while the space lives; null when the
  ///     function is a plain tensor-product B-spline (always for an HB space).
  ///
  /// Raises:
  ///     index_error: `dof` is out of range.
  [[nodiscard]] const TruncatedFunction* truncated(std::int64_t dof) const {
    if (!(0 <= dof && dof < num_total_basis())) {
      throw index_error("dof " + std::to_string(dof) + " is out of range [0, " +
                        std::to_string(num_total_basis()) + ").");
    }
    const std::int64_t at = rep_->trunc_index[static_cast<std::size_t>(dof)];
    return at < 0 ? nullptr : &rep_->trunc[static_cast<std::size_t>(at)];
  }

  /// Get the two-scale matrix of one direction from level `level` to level `level + 1`.
  ///
  /// Args:
  ///     level (std::int64_t): Coarse level in `[0, num_levels() - 1)`.
  ///     axis (std::size_t): Direction in `[0, dim())`.
  ///
  /// Returns:
  ///     const detail::TwoScale1D&: The banded matrix, valid while the space lives.
  ///
  /// Raises:
  ///     value_error: `level` is out of range.
  ///     index_error: `axis` is out of range.
  [[nodiscard]] const detail::TwoScale1D& two_scale(std::int64_t level, std::size_t axis) const {
    if (!(0 <= level && level + 1 < num_levels())) {
      throw value_error("level must be in [0, " + std::to_string(num_levels() - 2) + "]; got " +
                        std::to_string(level) + ".");
    }
    if (axis >= dim()) {
      throw index_error("axis " + std::to_string(axis) + " is out of range [0, " +
                        std::to_string(dim()) + ").");
    }
    return rep_->two_scale[static_cast<std::size_t>(level)][axis];
  }

  /// Get the cell support of every function of one direction at one level.
  ///
  /// Args:
  ///     level (std::int64_t): Hierarchy level in `[0, num_levels())`.
  ///     axis (std::size_t): Direction in `[0, dim())`.
  ///
  /// Returns:
  ///     const bspline::detail::Support1D&: The support arrays, valid while the space lives.
  ///
  /// Raises:
  ///     value_error: `level` is out of range.
  ///     index_error: `axis` is out of range.
  [[nodiscard]] const bspline::detail::Support1D& support(std::int64_t level,
                                                          std::size_t axis) const {
    const std::size_t lv = check_level(level);
    if (axis >= dim()) {
      throw index_error("axis " + std::to_string(axis) + " is out of range [0, " +
                        std::to_string(dim()) + ").");
    }
    return rep_->support[lv][axis];
  }

  /// Evaluate the active functions on cell `cid`, or one mixed partial derivative of them.
  ///
  /// Entry `(i, j)` of `out_basis` is `d^{orders} N_{dofs[j]}` at point `i`, `orders[k]` being the
  /// derivative order in direction `k` (an order above the degree gives zero); `out_dofs` receives
  /// the sorted global dofs, equal to `active_basis(cid)`. An untruncated function is the product
  /// over the directions `k = 0, 1, ...` of its 1D B-spline values, which stops at the first
  /// direction where the function is not supported at the point. A truncated function is the sum
  /// over its coefficient box of the coefficient times the same product at its representation
  /// level, contracted one direction at a time from the last direction in increasing index, a fixed
  /// order. The 1D values are computed in `double` whatever the root's storage type `T`, on the
  /// level spaces' knots in `T`.
  ///
  /// No dimension limit applies (the Python implementation refused more than 24 directions, a
  /// limit of its einsum subscripts).
  ///
  /// Only scratch is allocated: the 1D tabulations of the levels the cell's functions use.
  ///
  /// Args:
  ///     cid (std::int64_t): Active cell id in `[0, grid().num_cells())`.
  ///     pts (mdview<const double, 2>): Extents `(n, dim())`, every row inside the cell up to
  ///         `cell_membership_tolerance` per axis.
  ///     orders (std::span<const std::int64_t>): `dim()` non-negative derivative orders.
  ///     out_basis (mdview<double, 2>): Extents `(n, num_active_on(cid))`.
  ///     out_dofs (std::span<std::int64_t>): `num_active_on(cid)` entries.
  ///
  /// Raises:
  ///     value_error: `orders` has the wrong length or a negative entry; `pts` has another column
  ///         count or a point outside the cell; `out_basis` or `out_dofs` has another shape.
  ///     index_error: `cid` is out of range.
  ///     state_error: A 1D derivative recurrence met a zero divisor.
  void tabulate_basis(std::int64_t cid, mdview<const double, 2> pts,
                      std::span<const std::int64_t> orders, mdview<double, 2> out_basis,
                      std::span<std::int64_t> out_dofs) const {
    check_orders(orders);
    const CellContributions cell = contributions(cid);
    check_points(cid, pts);
    const std::size_t n_active = cell.size();
    if (out_basis.extent(0) != pts.extent(0) || out_basis.extent(1) != n_active) {
      const std::size_t got[2] = {out_basis.extent(0), out_basis.extent(1)};
      const std::size_t want[2] = {pts.extent(0), n_active};
      throw value_error("out_basis must have shape " + py_repr(std::span<const std::size_t>(want)) +
                        "; got " + py_repr(std::span<const std::size_t>(got)) + ".");
    }
    if (out_dofs.size() != n_active) {
      const std::size_t got[1] = {out_dofs.size()};
      const std::size_t want[1] = {n_active};
      throw value_error("out_dofs must have shape " + py_repr(std::span<const std::size_t>(want)) +
                        "; got " + py_repr(std::span<const std::size_t>(got)) + ".");
    }
    std::copy(cell.dofs.begin(), cell.dofs.end(), out_dofs.begin());
    evaluate(cell, pts, orders, out_basis);
  }

  /// Evaluate the active functions on cell `cid` at `pts` (all orders zero; see the `out_basis`
  /// overload taking orders).
  ///
  /// Raises:
  ///     value_error, index_error, state_error: As the overload taking orders.
  void tabulate_basis(std::int64_t cid, mdview<const double, 2> pts, mdview<double, 2> out_basis,
                      std::span<std::int64_t> out_dofs) const {
    const std::vector<std::int64_t> zeros(dim(), 0);
    tabulate_basis(cid, pts, zeros, out_basis, out_dofs);
  }

  /// Evaluate one mixed partial derivative of the active functions on cell `cid`.
  ///
  /// Returns:
  ///     TabulatedBasis: Values `(n, K)` and the `K` sorted global dofs, as the overload with `out`
  ///     arrays writes them.
  ///
  /// Raises:
  ///     value_error, index_error, state_error: As the overload with `out` arrays.
  [[nodiscard]] TabulatedBasis tabulate_basis(std::int64_t cid, mdview<const double, 2> pts,
                                              std::span<const std::int64_t> orders) const {
    // The output is sized after `cid` is known good; the checks of `orders` and `pts` run in the
    // `out` overload, which this one then only has to feed.
    check_orders(orders);
    const std::size_t n_active = num_active_on(cid);
    NdArray<double> values({pts.extent(0), n_active}, for_overwrite);
    NdArray<std::int64_t> dofs({n_active}, for_overwrite);
    tabulate_basis(cid, pts, orders, values.template view<2>(),
                   std::span<std::int64_t>(dofs.data(), dofs.size()));
    return TabulatedBasis{std::move(values), std::move(dofs)};
  }

  /// Evaluate the active functions on cell `cid` at `pts` (all orders zero).
  ///
  /// Returns:
  ///     TabulatedBasis: Values `(n, K)` and the `K` sorted global dofs.
  ///
  /// Raises:
  ///     value_error, index_error, state_error: As the overload taking orders.
  [[nodiscard]] TabulatedBasis tabulate_basis(std::int64_t cid, mdview<const double, 2> pts) const {
    const std::vector<std::int64_t> zeros(dim(), 0);
    return tabulate_basis(cid, pts, zeros);
  }

  /// Get the windowed sub-space over a subset of the active cells.
  ///
  /// The grid is restricted (`HierarchicalGrid::restrict`) to the root-cell-aligned bounding box of
  /// `cell_ids`, the root space is windowed to the same root cells (`BsplineSpace1D::restrict` per
  /// direction), and a space is rebuilt on the sub-grid, re-running the Kraft selection and the
  /// truncation. The windowed basis equals the global one only over the interior cells, those whose
  /// cross-level function-support closure lies inside the window, because the selection and the
  /// truncation depend on the subdomain near the window boundary; pad `cell_ids` with a support
  /// halo to make the cells of interest interior.
  ///
  /// Args:
  ///     cell_ids (std::span<const std::int64_t>): Active cell ids to span; repeats are ignored.
  ///
  /// Returns:
  ///     THBSplineSpaceRestriction<T>: The windowed space and the maps of its dofs and cells to
  ///     the global ones.
  ///
  /// Raises:
  ///     value_error: `cell_ids` is empty.
  ///     index_error: An id is outside `[0, grid().num_cells())`.
  ///     state_error: A windowed function falls outside the global level's function range (an
  ///         invariant; see `detail::restriction_dof_map`).
  [[nodiscard]] THBSplineSpaceRestriction<T> restrict(
      std::span<const std::int64_t> cell_ids) const {
    grid::HierarchicalGridRestriction grid_restr = rep_->hier.restrict(cell_ids);
    const grid::HierarchicalGrid& sub_grid = grid_restr.grid;
    const std::size_t d = dim();
    // The sub-grid's root is a pure slice of this one's breakpoints, so the first slice breakpoint
    // is found among them exactly.
    std::vector<std::int64_t> r_lo(d);
    std::vector<bspline::BsplineSpace1D<T>> windowed;
    windowed.reserve(d);
    for (std::size_t k = 0; k < d; ++k) {
      const std::vector<double>& bp = rep_->hier.root().breakpoints()[k];
      const double first = sub_grid.root().breakpoints()[k][0];
      r_lo[k] = std::lower_bound(bp.begin(), bp.end(), first) - bp.begin();
      const std::int64_t r_hi = r_lo[k] + sub_grid.root().cells_per_axis()[k];
      const auto window = rep_->root_space.spaces()[k].restrict(static_cast<long long>(r_lo[k]),
                                                                static_cast<long long>(r_hi));
      windowed.push_back(window.first);
    }
    THBSplineSpace sub(bspline::BsplineSpace<T>(std::move(windowed)), sub_grid, rep_->truncate,
                       rep_->regularity);
    std::vector<std::int64_t> dof_map = detail::restriction_dof_map(*rep_, *sub.rep_, r_lo);
    return THBSplineSpaceRestriction<T>{std::move(sub), std::move(dof_map),
                                        std::move(grid_restr.local_to_global)};
  }

  /// Get a new space with the named cells refined.
  ///
  /// With a class `m` the refinement is graded so that the mesh stays admissible of class `m` (the
  /// truncated functions acting on any cell span at most `m` successive levels), by the recursive
  /// refinement-neighborhood algorithm of Carraturo et al. (2019); the current mesh is assumed
  /// admissible of that class already, as the root and any mesh built by graded refinement are.
  /// Without a class exactly the named cells that are active leaves are refined. This space and its
  /// grid are not changed.
  ///
  /// Args:
  ///     cell_ids (std::span<const std::int64_t>): Active cell ids to refine; repeats are ignored.
  ///     admissible_class (std::optional<std::int64_t>): The class `m >= 2` to maintain, or no
  ///     value
  ///         for ungraded refinement.
  ///
  /// Returns:
  ///     THBSplineSpace: A space on the refined grid with the same root, truncation and
  ///     regularity.
  ///
  /// Raises:
  ///     value_error: `admissible_class` is below 2.
  ///     index_error: An id is outside `[0, grid().num_cells())`; the message lists every one.
  ///     std::overflow_error: A level would have more cells along an axis than `int64` holds.
  ///     value_error, state_error: As the constructor, for the refined grid.
  [[nodiscard]] THBSplineSpace refine(std::span<const std::int64_t> cell_ids,
                                      std::optional<std::int64_t> admissible_class) const {
    detail::check_admissible_class(admissible_class);
    const std::vector<std::int64_t> ids =
        detail::sorted_valid_ids(cell_ids, rep_->hier.num_cells());
    return on_grid(
        detail::refine_marked(*rep_, detail::named_cells(rep_->hier, ids), admissible_class));
  }

  /// Get a new space with the active cells of a rectangular region refined.
  ///
  /// The region is the cell-index box `[lo, hi)` at `level`, in level coordinates, as
  /// `HierarchicalGrid::refined` takes it. Only the active leaves inside the box are refined; a box
  /// with none gives a space equivalent to this one. See `refine` for the grading.
  ///
  /// Args:
  ///     level (std::int64_t): Level the box lives at, in `[0, grid().max_level()]`.
  ///     lo (std::span<const std::int64_t>): Per-direction start (inclusive).
  ///     hi (std::span<const std::int64_t>): Per-direction end (exclusive).
  ///     admissible_class (std::optional<std::int64_t>): The class `m >= 2` to maintain, or no
  ///     value
  ///         for ungraded refinement.
  ///
  /// Returns:
  ///     THBSplineSpace: A space on the refined grid with the same root, truncation and
  ///     regularity.
  ///
  /// Raises:
  ///     value_error: `admissible_class` is below 2, `level` is out of range, `lo` or `hi` has the
  ///         wrong length, some `lo[k] >= hi[k]`, or part of the box lies outside the level.
  ///     std::overflow_error, value_error, state_error: As `refine`.
  [[nodiscard]] THBSplineSpace refine_region(std::int64_t level, std::span<const std::int64_t> lo,
                                             std::span<const std::int64_t> hi,
                                             std::optional<std::int64_t> admissible_class) const {
    detail::check_admissible_class(admissible_class);
    detail::validate_region(rep_->hier, level, lo, hi);
    return on_grid(detail::refine_marked(
        *rep_, detail::active_leaves_in_box(rep_->hier, level, lo, hi), admissible_class));
  }

  /// Get a new space with the named cells coarsened away.
  ///
  /// A parent is reactivated, its children removed, only when all its children are named active
  /// leaves (Algorithm 5 of Carraturo et al.), taking the deepest parents first. With a class `m` a
  /// parent is reactivated only when its coarsening neighborhood (Definition 3.5) is empty, so the
  /// mesh stays admissible of class `m`; without one that guard is skipped and, for the cells an
  /// ungraded `refine` created, this inverts it. An empty `cell_ids` gives a copy of this space.
  ///
  /// Args:
  ///     cell_ids (std::span<const std::int64_t>): Active leaf ids to coarsen away; repeats are
  ///         ignored.
  ///     admissible_class (std::optional<std::int64_t>): The class `m >= 2` to guard, or no value
  ///         to skip the guard.
  ///
  /// Returns:
  ///     THBSplineSpace: A space on the coarsened grid with the same root, truncation and
  ///     regularity.
  ///
  /// Raises:
  ///     value_error: `admissible_class` is below 2.
  ///     index_error: An id is outside `[0, grid().num_cells())`; the message lists every one.
  ///     value_error, state_error: As the constructor, for the coarsened grid.
  [[nodiscard]] THBSplineSpace coarsen(std::span<const std::int64_t> cell_ids,
                                       std::optional<std::int64_t> admissible_class) const {
    detail::check_admissible_class(admissible_class);
    const std::vector<std::int64_t> ids =
        detail::sorted_valid_ids(cell_ids, rep_->hier.num_cells());
    return on_grid(detail::coarsen_marked(*rep_, ids, admissible_class));
  }

  /// Get the prolongation from this space to a refinement of it, in CSR form.
  ///
  /// The spaces are nested, so every function of this (coarse) space lies in `fine`; the matrix `P`
  /// maps coefficients in this basis to the coefficients of the same function in `fine`'s basis.
  /// It is built column by column by the local two-scale construction (Garau and Vazquez 2018;
  /// D'Angella et al. 2018): each coarse function is matched against the fine functions over its
  /// support, expressed in the deepest level present there, and reproduced by a small
  /// least-squares solve. A column is the fine functions' coefficients; away from the refinement it
  /// is one entry of value 1. The entries are stored as solved, including exact zeros and negative
  /// zeros. The columns are solved in parallel and gathered in order, so the matrix does not depend
  /// on the thread count.
  ///
  /// Args:
  ///     fine (const THBSplineSpace&): A refinement of this space: same root knots, factor,
  ///         regularity and truncation, and at least as many levels.
  ///
  /// Returns:
  ///     ProlongationCsr: The matrix of shape `(fine.num_total_basis(), num_total_basis())`, column
  ///     indices increasing within a row.
  ///
  /// Raises:
  ///     value_error: `fine` is not a refinement of this space: a structural mismatch (the message
  ///         lists every one), or a prolongation residual above
  ///         `prolongation_residual_tolerance`; also a zero two-scale slice (see `refine_box`).
  [[nodiscard]] ProlongationCsr prolongation_csr(const THBSplineSpace& fine) const {
    detail::check_is_refinement(*rep_, *fine.rep_);
    return detail::assemble_prolongation(*rep_, *fine.rep_);
  }

  /// The shared representation (for a binding's read-only views and for the operations that
  /// rebuild a space).
  [[nodiscard]] const std::shared_ptr<const Rep>& rep() const noexcept { return rep_; }

 private:
  static std::shared_ptr<const Rep> build(bspline::BsplineSpace<T> root,
                                          grid::HierarchicalGrid hierarchy, bool truncate,
                                          std::vector<std::optional<int>> regularity) {
    check_arguments(root, hierarchy, regularity);
    return detail::build_rep<T>(std::move(root), std::move(hierarchy), truncate,
                                std::move(regularity));
  }

  // The value checks of the constructor, in Python's order and with its messages.
  static void check_arguments(const bspline::BsplineSpace<T>& root,
                              const grid::HierarchicalGrid& hierarchy,
                              const std::vector<std::optional<int>>& regularity) {
    const std::size_t d = root.dim();
    if (hierarchy.ndim() != d) {
      throw value_error("grid.ndim (" + std::to_string(hierarchy.ndim()) +
                        ") must equal root_space.dim (" + std::to_string(d) + ").");
    }
    std::vector<std::size_t> intervals;
    for (const auto& s : root.spaces()) {
      intervals.push_back(static_cast<std::size_t>(s.num_intervals()));
    }
    const std::span<const std::int64_t> cells = hierarchy.root().cells_per_axis();
    bool same = true;
    for (std::size_t k = 0; k < d; ++k) {
      same = same && static_cast<std::size_t>(cells[k]) == intervals[k];
    }
    if (!same) {
      std::vector<std::size_t> cells_as_size;
      for (const std::int64_t c : cells) {
        cells_as_size.push_back(static_cast<std::size_t>(c));
      }
      throw value_error("grid root cells_per_axis " +
                        py_repr(std::span<const std::size_t>(cells_as_size)) +
                        " must match root_space.num_intervals " +
                        py_repr(std::span<const std::size_t>(intervals)) + ".");
    }
    // Absolute comparison against the space's own knot tolerance. A relative-and-absolute test
    // with library defaults would accept a few-percent-of-a-millionth relative domain mismatch at
    // every scale and, through its absolute leg, `[0, 1e-9]` against `[0, 2e-9]`.
    const std::span<const double> bounds = hierarchy.root().bounds();
    for (std::size_t k = 0; k < d; ++k) {
      const auto [lo, hi] = root.spaces()[k].domain();
      if (!(std::abs(bounds[2 * k] - static_cast<double>(lo)) <= root.tolerance()) ||
          !(std::abs(bounds[2 * k + 1] - static_cast<double>(hi)) <= root.tolerance())) {
        throw value_error("grid root bounds must match root_space domain.");
      }
    }
    if (regularity.size() != d) {
      throw value_error("regularity must be a scalar or length-" + std::to_string(d) +
                        " sequence; got length " + std::to_string(regularity.size()) + ".");
    }
    for (std::size_t k = 0; k < d; ++k) {
      const int degree = root.spaces()[k].degree();
      if (regularity[k].has_value() && !(-1 <= *regularity[k] && *regularity[k] < degree)) {
        const std::string r = std::to_string(*regularity[k]);
        throw value_error("regularity[" + std::to_string(k) + "]=" + r +
                          " must be in [-1, degree[" + std::to_string(k) +
                          "]-1=" + std::to_string(degree - 1) + "]; got " + r + ".");
      }
    }
  }

  // A space of the same root, truncation and regularity on another grid.
  [[nodiscard]] THBSplineSpace on_grid(grid::HierarchicalGrid hierarchy) const {
    return THBSplineSpace(rep_->root_space, std::move(hierarchy), rep_->truncate, rep_->regularity);
  }

  [[nodiscard]] std::size_t check_level(std::int64_t level) const {
    if (!(0 <= level && level < num_levels())) {
      throw value_error("level must be in [0, " + std::to_string(num_levels() - 1) + "]; got " +
                        std::to_string(level) + ".");
    }
    return static_cast<std::size_t>(level);
  }

  [[nodiscard]] const detail::CellBlock& cell_block() const {
    return rep_->cells.get([this] { return detail::build_cell_block<T>(*rep_); });
  }

  void check_orders(std::span<const std::int64_t> orders) const {
    if (orders.size() != dim()) {
      throw value_error("orders must be a scalar or length-" + std::to_string(dim()) +
                        " sequence; got length " + std::to_string(orders.size()) + ".");
    }
    if (std::any_of(orders.begin(), orders.end(), [](std::int64_t o) { return o < 0; })) {
      throw value_error("orders must be non-negative; got " + grid::detail::py_repr_ints(orders) +
                        ".");
    }
  }

  // The column count, then membership of every point in the cell up to the per-axis slack. A NaN
  // coordinate fails both comparisons and is refused.
  void check_points(std::int64_t cid, mdview<const double, 2> pts) const {
    const std::size_t d = dim();
    if (pts.extent(1) != d) {
      const std::size_t got[2] = {pts.extent(0), pts.extent(1)};
      throw value_error("pts must have trailing dimension " + std::to_string(d) + "; got shape " +
                        py_repr(std::span<const std::size_t>(got)) + ".");
    }
    const auto [lo, hi] = rep_->hier.cell_bounds(cid);
    std::vector<double> slack(d);
    for (std::size_t k = 0; k < d; ++k) {
      slack[k] = cell_membership_tolerance(lo[k], hi[k]);
    }
    for (std::size_t i = 0; i < pts.extent(0); ++i) {
      for (std::size_t k = 0; k < d; ++k) {
        const double x = pts(i, k);
        if (!(x >= lo[k] - slack[k] && x <= hi[k] + slack[k])) {
          throw value_error("pts must lie inside cell " + std::to_string(cid) +
                            " with bounds lo=" + list_repr(lo) + ", hi=" + list_repr(hi) +
                            " (slack " + list_repr(slack) + ").");
        }
      }
    }
  }

  // `[a, b]` with Python float spellings (NumPy's array text is not reproduced).
  static std::string list_repr(const std::vector<double>& xs) {
    std::string s = "[";
    for (std::size_t i = 0; i < xs.size(); ++i) {
      s += (i > 0 ? ", " : "") + py_repr(xs[i]);
    }
    return s + "]";
  }

  // Tabulate the 1D basis (order 0) or its `order`-th derivative of one level and direction at
  // the points' coordinates along that direction.
  //
  // The points were checked against the cell, which lies inside the domain up to the membership
  // slack, so the 1D kernel is run without its own domain check; a point within the slack of an
  // end is evaluated on the nearest in-domain span. An order above the degree is zero: the kernel
  // is asked for `degree + 1`, whose slice is exactly zero, rather than for an order that sizes
  // an array by the caller's number.
  void tabulate_1d(std::int64_t level, std::size_t k, std::int64_t order,
                   mdview<const double, 2> pts, detail::Basis1D& out) const {
    const auto& sp1d = rep_->level_spaces[static_cast<std::size_t>(level)].spaces()[k];
    const std::size_t n = pts.extent(0);
    const auto width = static_cast<std::size_t>(sp1d.degree()) + 1;
    std::vector<double> coord(n);
    for (std::size_t i = 0; i < n; ++i) {
      coord[i] = pts(i, k);
    }
    out.first.assign(n, 0);
    out.degree = sp1d.degree();
    const std::span<const double> coords(coord);
    if (order == 0) {
      out.data.assign(n * width, 0.0);
      out.stride = width;
      out.offset = 0;
      sp1d.template tabulate_basis<double>(coords, mdview<double, 2>(out.data.data(), n, width),
                                           std::span<std::int64_t>(out.first), false);
      return;
    }
    const int n_deriv = static_cast<int>((std::min)(order, static_cast<std::int64_t>(width)));
    const auto rows = static_cast<std::size_t>(n_deriv) + 1;
    out.data.assign(n * rows * width, 0.0);
    out.stride = rows * width;
    out.offset = static_cast<std::size_t>(n_deriv) * width;
    const bool complete = sp1d.template tabulate_basis_derivatives<double>(
        coords, n_deriv, mdview<double, 3>(out.data.data(), n, rows, width),
        std::span<std::int64_t>(out.first), false);
    if (!complete) {
      throw state_error("the 1D derivative recurrence met a zero divisor at level " +
                        std::to_string(level) + ", axis " + std::to_string(k) +
                        ": a basis function of empty support.");
    }
  }

  // Evaluate every function of `cell` at every point. The 1D tabulations are made once per
  // (level, direction) the functions use, as the orders are fixed for the call.
  void evaluate(const CellContributions& cell, mdview<const double, 2> pts,
                std::span<const std::int64_t> orders, mdview<double, 2> out) const {
    const std::size_t d = dim();
    const std::size_t levels = rep_->level_spaces.size();
    std::vector<unsigned char> needed(levels, 0);
    std::size_t work_size = 0;
    for (std::size_t j = 0; j < cell.size(); ++j) {
      const std::int64_t at = rep_->trunc_index[static_cast<std::size_t>(cell.dofs[j])];
      if (at < 0) {
        needed[static_cast<std::size_t>(cell.levels[j])] = 1;
      } else {
        const TruncatedFunction& f = rep_->trunc[static_cast<std::size_t>(at)];
        needed[static_cast<std::size_t>(f.rep_level)] = 1;
        std::size_t widths = 0;
        for (const std::size_t w : f.box_shape) {
          widths += w;
        }
        work_size = (std::max)(work_size, widths + f.coeffs.size());
      }
    }
    std::vector<detail::Basis1D> tables(levels * d);
    for (std::size_t l = 0; l < levels; ++l) {
      if (needed[l] != 0) {
        for (std::size_t k = 0; k < d; ++k) {
          tabulate_1d(static_cast<std::int64_t>(l), k, orders[k], pts, tables[l * d + k]);
        }
      }
    }
    std::vector<const detail::Basis1D*> bases(levels * d);
    for (std::size_t i = 0; i < bases.size(); ++i) {
      bases[i] = &tables[i];
    }
    std::vector<double> work(work_size);
    for (std::size_t i = 0; i < pts.extent(0); ++i) {
      for (std::size_t j = 0; j < cell.size(); ++j) {
        const std::int64_t at = rep_->trunc_index[static_cast<std::size_t>(cell.dofs[j])];
        if (at < 0) {
          out(i, j) = detail::tensor_product_value(
              bases.data() + static_cast<std::size_t>(cell.levels[j]) * d, d, i,
              cell.multi_index(j));
        } else {
          const TruncatedFunction& f = rep_->trunc[static_cast<std::size_t>(at)];
          out(i, j) = detail::truncated_value(
              f, bases.data() + static_cast<std::size_t>(f.rep_level) * d, i, work.data());
        }
      }
    }
  }

  std::shared_ptr<const Rep> rep_;
};

/// The windowed sub-space `THBSplineSpace::restrict` returns, with its index maps.
///
/// Attributes:
///     space (THBSplineSpace<T>): The space rebuilt on the restricted grid; its basis equals the
///         global one pointwise over the interior cells (those whose cross-level function-support
///         closure lies inside the window).
///     local_to_global_dof (std::vector<std::int64_t>): Per local dof, the global dof of the
///         function of the same level and multi-index when the global space has it active, else
///         -1 (near the window boundary the Kraft selection can differ). Exact over interior cells.
///     local_to_global_cell (std::vector<std::int64_t>): Per local cell, its id in the global grid.
template <StorageFloat T>
struct THBSplineSpaceRestriction {
  THBSplineSpace<T> space;
  std::vector<std::int64_t> local_to_global_dof;
  std::vector<std::int64_t> local_to_global_cell;
};

}  // namespace pantr::thb
