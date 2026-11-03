/// @file local_space.hpp
/// @brief Serial windowing kernels for distributing a tensor-product B-spline space: the support
/// closure of a set of owned cells (the halo) and the owner rank of each function, and their
/// composition into a rank's local space (`build_local`).
///
/// Cells are flat-indexed in C order over the per-direction interval counts and functions in C
/// order over the per-direction function counts, as `coupling_graph` and the spanwise extraction
/// index them. The owner of a function is the owner of the active cell (`cell_owner >= 0`) with the
/// smallest flat id in the function's support: a flat C-order id is monotone in the lexicographic
/// order of the multi-index, so a C-order walk over a support box visits cells in increasing id and
/// the first active cell met is the smallest.
#pragma once

#include "pantr/bspline/detail/support_1d.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/grid/partition.hpp"
#include "pantr/parallel/parallel_for.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <span>
#include <string>
#include <vector>

namespace pantr::bspline {

namespace detail {

/// The per-direction support ranges and strides of a non-periodic tensor-product space.
///
/// Attributes:
///     num_intervals (std::vector<std::int64_t>): Cells per direction.
///     num_basis (std::vector<std::int64_t>): Functions per direction.
///     cell_stride (std::vector<std::int64_t>): C-order strides of the cell grid.
///     first_cell (std::vector<std::vector<std::int64_t>>): Per direction, the first interval of
///         each function's support.
///     last_cell (std::vector<std::vector<std::int64_t>>): Per direction, the last interval of
///         each function's support (inclusive).
///     closure_first (std::vector<std::vector<std::int64_t>>): Per direction, the first interval
///         of the support closure of each interval: the union of the supports of the functions
///         non-zero on it, `first_cell[first_basis[j]]`.
///     closure_last (std::vector<std::vector<std::int64_t>>): The last interval of that closure,
///         `last_cell[first_basis[j] + degree]`.
///     n_cells (std::int64_t): Total number of cells.
///     n_dofs (std::int64_t): Total number of functions.
struct LocalSpaceAxes {
  std::vector<std::int64_t> num_intervals;
  std::vector<std::int64_t> num_basis;
  std::vector<std::int64_t> cell_stride;
  std::vector<std::vector<std::int64_t>> first_cell;
  std::vector<std::vector<std::int64_t>> last_cell;
  std::vector<std::vector<std::int64_t>> closure_first;
  std::vector<std::vector<std::int64_t>> closure_last;
  std::int64_t n_cells = 1;
  std::int64_t n_dofs = 1;
};

/// Tabulate the supports of a tensor-product space.
///
/// Args:
///     space (const BsplineSpace<T>&): The space; no direction may be periodic.
///
/// Returns:
///     LocalSpaceAxes: The tables.
///
/// Raises:
///     value_error: A direction is periodic.
///     state_error: A function has an empty support (an invalid space).
template <StorageFloat T>
[[nodiscard]] LocalSpaceAxes local_space_axes(const BsplineSpace<T>& space) {
  const auto& spaces = space.spaces();
  if (std::any_of(spaces.begin(), spaces.end(), [](const auto& s) { return s.periodic(); })) {
    throw value_error("periodic B-spline spaces are not supported.");
  }
  const std::size_t dim = spaces.size();
  LocalSpaceAxes out;
  out.cell_stride.assign(dim, 1);
  for (std::size_t k = 0; k < dim; ++k) {
    const auto& sp = spaces[k];
    const detail::Support1D sup = detail::func_support_1d(sp);
    out.num_intervals.push_back(sp.num_intervals());
    out.num_basis.push_back(sp.num_basis());
    out.n_cells *= sp.num_intervals();  // the space's constructor refuses an overflowing product
    out.n_dofs *= sp.num_basis();
    std::vector<std::int64_t> cf(sup.first_basis.size());
    std::vector<std::int64_t> cl(sup.first_basis.size());
    for (std::size_t j = 0; j < cf.size(); ++j) {
      const std::int64_t fb = sup.first_basis[j];
      cf[j] = sup.first_cell[static_cast<std::size_t>(fb)];
      cl[j] = sup.last_cell[static_cast<std::size_t>(fb + sp.degree())];
    }
    out.first_cell.push_back(sup.first_cell);
    out.last_cell.push_back(sup.last_cell);
    out.closure_first.push_back(std::move(cf));
    out.closure_last.push_back(std::move(cl));
  }
  for (std::size_t k = dim; k-- > 1;) {
    out.cell_stride[k - 1] = out.cell_stride[k] * out.num_intervals[k];
  }
  return out;
}

/// Walk a box of cells in C order by an odometer, calling `visit(flat)` on each until it returns
/// true.
///
/// The last axis advances fastest, so the flat id increases monotonically.
///
/// Args:
///     span (std::span<const std::int64_t>): Cells per direction of the box, each at least 1.
///     strides (std::span<const std::int64_t>): C-order strides of the cell grid.
///     counter (std::span<std::int64_t>): Scratch, `dim` entries; overwritten.
///     flat (std::int64_t): Flat id of the box's first cell.
///     visit (Visit&&): Callable as `bool visit(std::int64_t)`; true stops the walk.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <class Visit>
void walk_box(std::span<const std::int64_t> span, std::span<const std::int64_t> strides,
              std::span<std::int64_t> counter, std::int64_t flat, Visit&& visit) {
  const std::size_t dim = span.size();
  std::int64_t total = 1;
  for (const std::int64_t s : span) {
    total *= s;
  }
  std::fill(counter.begin(), counter.end(), std::int64_t{0});
  for (std::int64_t step = 0; step < total; ++step) {
    if (visit(flat)) {
      return;
    }
    for (std::size_t axis = dim; axis-- > 0;) {
      ++counter[axis];
      flat += strides[axis];
      if (counter[axis] < span[axis]) {
        break;
      }
      flat -= counter[axis] * strides[axis];
      counter[axis] = 0;
    }
  }
}

/// The largest extent, over the intervals or functions, of one direction's support ranges,
/// multiplied over the directions: the longest box walk of a kernel.
///
/// Args:
///     first (const std::vector<std::vector<std::int64_t>>&): Per-direction first cells.
///     last (const std::vector<std::vector<std::int64_t>>&): Per-direction last cells.
///
/// Returns:
///     double: `prod_k max_j (last[k][j] - first[k][j] + 1)`, at least 1.
[[nodiscard]] inline double longest_box_walk(const std::vector<std::vector<std::int64_t>>& first,
                                             const std::vector<std::vector<std::int64_t>>& last) {
  double cells = 1.0;
  for (std::size_t k = 0; k < first.size(); ++k) {
    std::int64_t widest = 1;
    for (std::size_t j = 0; j < first[k].size(); ++j) {
      widest = (std::max)(widest, last[k][j] - first[k][j] + 1);
    }
    cells *= static_cast<double>(widest);
  }
  return cells;
}

}  // namespace detail

/// Get the support-closure halo of a set of owned cells.
///
/// The halo is the set of cells, excluding the owned ones, covered by the support of any function
/// non-zero on an owned cell: what a rank owning `owned_cells` needs so that every function
/// touching its cells is fully represented over the owned and halo cells. General and repeated
/// knots are handled exactly, through the per-direction supports.
///
/// The owned cells are walked in parallel; every write stores 1 into a byte, so the closure does
/// not depend on the order or the thread count. One cell costs a box walk, one store per cell of
/// the largest support closure of a cell (the odometer step amortizing to a constant), so the work
/// per owned cell is the product over the directions of the widest closure.
///
/// Args:
///     space (const BsplineSpace<T>&): The space; no direction may be periodic.
///     owned_cells (std::span<const std::int64_t>): Flat cell ids; repeats are ignored.
///
/// Returns:
///     std::vector<std::int64_t>: The halo cell ids, ascending; empty when `owned_cells` is.
///
/// Raises:
///     value_error: A direction is periodic.
///     index_error: An owned id is outside `[0, num_total_intervals)`.
template <StorageFloat T>
[[nodiscard]] std::vector<std::int64_t> compute_halo(const BsplineSpace<T>& space,
                                                     std::span<const std::int64_t> owned_cells) {
  const detail::LocalSpaceAxes ax = detail::local_space_axes(space);
  if (owned_cells.empty()) {
    return {};
  }
  if (std::any_of(owned_cells.begin(), owned_cells.end(),
                  [&](std::int64_t c) { return c < 0 || c >= ax.n_cells; })) {
    throw index_error("owned cell id out of range [0, " + std::to_string(ax.n_cells) + ").");
  }
  const std::size_t dim = ax.num_intervals.size();
  const auto n_cells = static_cast<std::size_t>(ax.n_cells);

  std::vector<unsigned char> owned(n_cells, 0);
  for (const std::int64_t c : owned_cells) {
    owned[static_cast<std::size_t>(c)] = 1;
  }
  // Concurrent stores of 1 to one byte are a data race unless atomic; relaxed order suffices, as
  // nothing is read until the loop has joined. C++20 value-initializes each atomic to 0.
  std::vector<std::atomic<unsigned char>> mask(n_cells);

  const parallel::WorkEstimate work{detail::longest_box_walk(ax.closure_first, ax.closure_last)};
  parallel::parallel_for(owned_cells.size(), work, [&](std::size_t begin, std::size_t end) {
    std::vector<std::int64_t> span(dim);
    std::vector<std::int64_t> counter(dim);
    for (std::size_t i = begin; i < end; ++i) {
      std::int64_t rest = owned_cells[i];
      std::int64_t base = 0;
      for (std::size_t k = dim; k-- > 0;) {
        const auto index = static_cast<std::size_t>(rest % ax.num_intervals[k]);
        rest /= ax.num_intervals[k];
        base += ax.closure_first[k][index] * ax.cell_stride[k];
        span[k] = ax.closure_last[k][index] - ax.closure_first[k][index] + 1;
      }
      detail::walk_box(span, ax.cell_stride, counter, base, [&](std::int64_t flat) {
        mask[static_cast<std::size_t>(flat)].store(1, std::memory_order_relaxed);
        return false;
      });
    }
  });

  std::vector<std::int64_t> halo;
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (mask[c].load(std::memory_order_relaxed) != 0 && owned[c] == 0) {
      halo.push_back(static_cast<std::int64_t>(c));
    }
  }
  return halo;
}

/// Get the owner rank of the requested functions (lex-first-active-cell rule).
///
/// The subset counterpart of `dof_owner`, for a consumer resolving ghosts that needs a handful of
/// owners. Each function is owned by the owner of the active cell with the smallest flat id in its
/// support; a function whose support holds no active cell is dead and gets -1.
///
/// The requested functions are resolved in parallel, each writing its own result, so the result is
/// independent of the thread count. One function costs its decode (one division per direction) and
/// a box walk over its support, in the worst case (every cell inactive) one read per cell of the
/// widest support (the odometer step amortizing to a constant), so the work per function is the
/// number of directions plus the product over the directions of the widest support.
///
/// Args:
///     space (const BsplineSpace<T>&): The space; no direction may be periodic.
///     cell_owner (std::span<const std::int32_t>): Owner rank of every cell, -1 if inactive;
///         `num_total_intervals` entries.
///     dofs (std::span<const std::int64_t>): Flat function ids; repeats and any order allowed.
///
/// Returns:
///     std::vector<std::int32_t>: The owner of each requested function, in request order.
///
/// Raises:
///     value_error: A direction is periodic, or `cell_owner` has the wrong length.
///     index_error: A function id is outside `[0, num_total_basis)`.
template <StorageFloat T>
[[nodiscard]] std::vector<std::int32_t> dof_owner_windowed(const BsplineSpace<T>& space,
                                                           std::span<const std::int32_t> cell_owner,
                                                           std::span<const std::int64_t> dofs) {
  const detail::LocalSpaceAxes ax = detail::local_space_axes(space);
  if (cell_owner.size() != static_cast<std::size_t>(ax.n_cells)) {
    throw value_error("partition has " + std::to_string(cell_owner.size()) + " cells; expected " +
                      std::to_string(ax.n_cells) + " (space.num_total_intervals).");
  }
  if (std::any_of(dofs.begin(), dofs.end(),
                  [&](std::int64_t d) { return d < 0 || d >= ax.n_dofs; })) {
    throw index_error("DOF id out of range [0, " + std::to_string(ax.n_dofs) + ").");
  }
  std::vector<std::int32_t> owners(dofs.size(), -1);
  if (dofs.empty()) {
    return owners;
  }
  const std::size_t dim = ax.num_basis.size();
  const parallel::WorkEstimate work{static_cast<double>(dim) +
                                    detail::longest_box_walk(ax.first_cell, ax.last_cell)};
  parallel::parallel_for(dofs.size(), work, [&](std::size_t begin, std::size_t end) {
    std::vector<std::int64_t> span(dim);
    std::vector<std::int64_t> counter(dim);
    for (std::size_t i = begin; i < end; ++i) {
      std::int64_t rest = dofs[i];
      std::int64_t base = 0;
      for (std::size_t k = dim; k-- > 0;) {
        const auto index = static_cast<std::size_t>(rest % ax.num_basis[k]);
        rest /= ax.num_basis[k];
        base += ax.first_cell[k][index] * ax.cell_stride[k];
        span[k] = ax.last_cell[k][index] - ax.first_cell[k][index] + 1;
      }
      std::int32_t owner = -1;
      detail::walk_box(span, ax.cell_stride, counter, base, [&](std::int64_t flat) {
        const std::int32_t o = cell_owner[static_cast<std::size_t>(flat)];
        if (o >= 0) {
          owner = o;
          return true;
        }
        return false;
      });
      owners[i] = owner;
    }
  });
  return owners;
}

/// Get the owner rank of every function (lex-first-active-cell rule).
///
/// Args:
///     space (const BsplineSpace<T>&): The space; no direction may be periodic.
///     cell_owner (std::span<const std::int32_t>): Owner rank of every cell, -1 if inactive;
///         `num_total_intervals` entries.
///
/// Returns:
///     std::vector<std::int32_t>: `num_total_basis` owners, -1 for a dead function.
///
/// Raises:
///     value_error: A direction is periodic, or `cell_owner` has the wrong length.
template <StorageFloat T>
[[nodiscard]] std::vector<std::int32_t> dof_owner(const BsplineSpace<T>& space,
                                                  std::span<const std::int32_t> cell_owner) {
  std::int64_t n_dofs = 1;
  for (const auto& s : space.spaces()) {
    n_dofs *= s.num_basis();
  }
  std::vector<std::int64_t> all(static_cast<std::size_t>(n_dofs));
  std::iota(all.begin(), all.end(), std::int64_t{0});
  return dof_owner_windowed(space, cell_owner, all);
}

/// The rank-local windowed view of a distributed B-spline or THB space.
///
/// Produced by `build_local`. Bundles a windowed space with the maps and masks relating it to the
/// global space.
///
/// Attributes:
///     space (Space): The windowed space over the bounding box of the rank's owned cells and their
///         support-closure halo; its basis equals the global basis pointwise over the rank's owned
///         cells.
///     local_to_global_cell (std::vector<std::int64_t>): One entry per local cell: its global cell
///         id.
///     local_to_global_dof (std::vector<std::int64_t>): One entry per local function (the space's
///         total basis count): its global function id, -1 for a THB function with no global
///         counterpart.
///     owned_cell_mask (std::vector<std::uint8_t>): One entry per local cell; 1 for the cells this
///         rank owns (the rest are halo or bounding-box fill).
///     owned_dof_mask (std::vector<std::uint8_t>): One entry per local function; 1 for the
///         functions this rank owns.
///     n_global_cells (std::int64_t): Total cells of the global space's grid.
///     n_global_dofs (std::int64_t): Total basis functions of the global space.
template <class Space>
struct LocalSpace {
  Space space;
  std::vector<std::int64_t> local_to_global_cell;
  std::vector<std::int64_t> local_to_global_dof;
  std::vector<std::uint8_t> owned_cell_mask;
  std::vector<std::uint8_t> owned_dof_mask;
  std::int64_t n_global_cells;
  std::int64_t n_global_dofs;
};

namespace detail {

/// Check that `rank` is a part of `partition`.
///
/// Args:
///     partition (const grid::Partition&): The partition.
///     rank (std::int64_t): The rank asked for.
///
/// Raises:
///     value_error: `rank` is outside `[0, partition.n_parts())`.
inline void check_rank_in_partition(const grid::Partition& partition, std::int64_t rank) {
  if (!(0 <= rank && rank < partition.n_parts())) {
    throw value_error("rank must be in [0, " + std::to_string(partition.n_parts()) + "); got " +
                      std::to_string(rank) + ".");
  }
}

/// Get the cells a rank owns, refusing a rank that owns none.
///
/// Args:
///     partition (const grid::Partition&): The partition; `rank` must be one of its parts.
///     rank (std::int64_t): The rank.
///
/// Returns:
///     std::vector<std::int64_t>: The owned cell ids, ascending, not empty.
///
/// Raises:
///     value_error: `rank` owns no cell.
[[nodiscard]] inline std::vector<std::int64_t> owned_cells_or_throw(
    const grid::Partition& partition, std::int64_t rank) {
  std::vector<std::int64_t> owned = partition.owned_cells(rank);
  if (owned.empty()) {
    throw value_error("rank " + std::to_string(rank) +
                      " owns no cells; cannot build a local space.");
  }
  return owned;
}

/// Get the sorted union of two sorted, disjoint id lists.
///
/// Args:
///     a (std::span<const std::int64_t>): Ascending ids.
///     b (std::span<const std::int64_t>): Ascending ids, none in `a`.
///
/// Returns:
///     std::vector<std::int64_t>: The ids of both, ascending.
[[nodiscard]] inline std::vector<std::int64_t> sorted_union(std::span<const std::int64_t> a,
                                                            std::span<const std::int64_t> b) {
  std::vector<std::int64_t> out;
  out.reserve(a.size() + b.size());
  std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
  return out;
}

/// Get the C-order ravel of a product of per-axis index lists.
///
/// Entry `(i_0, ..., i_{d-1})`, the last axis fastest, is `sum_k axes[k][i_k] * stride_k` with
/// `stride_k` the C-order stride of `extents`.
///
/// Args:
///     axes (const std::vector<std::vector<std::int64_t>>&): Per axis, the ids along it, each in
///         `[0, extents[k])`; none empty.
///     extents (std::span<const std::int64_t>): The size of the global index space per axis.
///
/// Returns:
///     std::vector<std::int64_t>: `prod_k axes[k].size()` flat ids.
[[nodiscard]] inline std::vector<std::int64_t> ravel_axes(
    const std::vector<std::vector<std::int64_t>>& axes, std::span<const std::int64_t> extents) {
  const std::size_t dim = axes.size();
  std::vector<std::int64_t> stride(dim, 1);
  for (std::size_t k = dim; k-- > 1;) {
    stride[k - 1] = stride[k] * extents[k];
  }
  std::size_t total = 1;
  for (const auto& axis : axes) {
    total *= axis.size();
  }
  std::vector<std::int64_t> out;
  out.reserve(total);
  std::vector<std::size_t> counter(dim, 0);
  for (std::size_t step = 0; step < total; ++step) {
    std::int64_t flat = 0;
    for (std::size_t k = 0; k < dim; ++k) {
      flat += axes[k][counter[k]] * stride[k];
    }
    out.push_back(flat);
    for (std::size_t k = dim; k-- > 0;) {
      if (++counter[k] < axes[k].size()) {
        break;
      }
      counter[k] = 0;
    }
  }
  return out;
}

}  // namespace detail

/// Build the rank-local windowed space of a distributed tensor-product B-spline space.
///
/// Windows `space` to the cells `rank` owns together with their support-closure halo
/// (`compute_halo`), so the local basis equals the global basis pointwise over the rank's owned
/// cells. Each direction is restricted (`BsplineSpace1D::restrict`) to the window's per-direction
/// bounding box of intervals, so the local cells are that box in C order and the local functions
/// the product of the per-direction function ranges, in C order; cells of the box outside the
/// window are fill, not owned by `rank`. A function is owned by `rank` iff `dof_owner` says so.
///
/// Args:
///     space (const BsplineSpace<T>&): The global space; no direction may be periodic.
///     partition (const grid::Partition&): Owner of every cell of the space's knot-span grid;
///         `num_total_intervals` cells.
///     rank (std::int64_t): The rank whose local space is built, in `[0, partition.n_parts())`.
///
/// Returns:
///     LocalSpace<BsplineSpace<T>>: The windowed space, its maps to the global cells and
///     functions, and the ownership masks.
///
/// Raises:
///     value_error: A direction is periodic; `rank` is out of range; `partition` has another cell
///         count than the space; `rank` owns no cell.
///     state_error: The restricted directions and the window's bounding box disagree on the cell
///         count (an invariant of `BsplineSpace1D::restrict`).
template <StorageFloat T>
[[nodiscard]] LocalSpace<BsplineSpace<T>> build_local(const BsplineSpace<T>& space,
                                                      const grid::Partition& partition,
                                                      std::int64_t rank) {
  const detail::LocalSpaceAxes ax = detail::local_space_axes(space);
  detail::check_rank_in_partition(partition, rank);
  if (partition.n_cells() != ax.n_cells) {
    throw value_error("partition has " + std::to_string(partition.n_cells()) + " cells; expected " +
                      std::to_string(ax.n_cells) + " (space.num_total_intervals).");
  }
  const std::vector<std::int64_t> owned = detail::owned_cells_or_throw(partition, rank);
  const std::vector<std::int64_t> halo = compute_halo(space, owned);
  const std::vector<std::int64_t> window = detail::sorted_union(owned, halo);

  // The per-direction bounding box of the window's intervals; the window is not empty and every
  // interval index is below its direction's count, so these initial values are always replaced.
  const std::size_t dim = ax.num_intervals.size();
  std::vector<std::int64_t> box_lo = ax.num_intervals;
  std::vector<std::int64_t> box_hi(dim, -1);
  for (const std::int64_t id : window) {
    std::int64_t rest = id;
    for (std::size_t k = dim; k-- > 0;) {
      const std::int64_t index = rest % ax.num_intervals[k];
      rest /= ax.num_intervals[k];
      box_lo[k] = (std::min)(box_lo[k], index);
      box_hi[k] = (std::max)(box_hi[k], index);
    }
  }

  std::vector<BsplineSpace1D<T>> windowed;
  windowed.reserve(dim);
  std::vector<std::vector<std::int64_t>> dof_axes;
  std::vector<std::vector<std::int64_t>> cell_axes;
  std::int64_t local_intervals = 1;
  for (std::size_t k = 0; k < dim; ++k) {
    auto restricted = space.spaces()[k].restrict(static_cast<long long>(box_lo[k]),
                                                 static_cast<long long>(box_hi[k] + 1));
    local_intervals *= restricted.first.num_intervals();
    windowed.push_back(std::move(restricted.first));
    dof_axes.push_back(std::move(restricted.second));
    std::vector<std::int64_t> cells(static_cast<std::size_t>(box_hi[k] - box_lo[k] + 1));
    std::iota(cells.begin(), cells.end(), box_lo[k]);
    cell_axes.push_back(std::move(cells));
  }
  std::vector<std::int64_t> local_to_global_cell = detail::ravel_axes(cell_axes, ax.num_intervals);
  std::vector<std::int64_t> local_to_global_dof = detail::ravel_axes(dof_axes, ax.num_basis);
  if (static_cast<std::int64_t>(local_to_global_cell.size()) != local_intervals) {
    throw state_error("Grid restriction (" + std::to_string(local_to_global_cell.size()) +
                      " cells) and space restriction (" + std::to_string(local_intervals) +
                      " cells) disagree after restricting to the same window of " +
                      std::to_string(window.size()) + " cells. This is a bug in restrict().");
  }

  const std::span<const std::int32_t> cell_owner = partition.cell_owner();
  std::vector<std::uint8_t> owned_cell_mask(local_to_global_cell.size());
  for (std::size_t i = 0; i < owned_cell_mask.size(); ++i) {
    const auto owner =
        static_cast<std::int64_t>(cell_owner[static_cast<std::size_t>(local_to_global_cell[i])]);
    owned_cell_mask[i] = owner == rank ? 1 : 0;
  }
  // The subset counterpart of `dof_owner`: the same rule, resolved only for the local functions.
  const std::vector<std::int32_t> dof_owners =
      dof_owner_windowed(space, cell_owner, local_to_global_dof);
  std::vector<std::uint8_t> owned_dof_mask(dof_owners.size());
  for (std::size_t i = 0; i < owned_dof_mask.size(); ++i) {
    owned_dof_mask[i] = static_cast<std::int64_t>(dof_owners[i]) == rank ? 1 : 0;
  }
  return LocalSpace<BsplineSpace<T>>{BsplineSpace<T>(std::move(windowed)),
                                     std::move(local_to_global_cell),
                                     std::move(local_to_global_dof),
                                     std::move(owned_cell_mask),
                                     std::move(owned_dof_mask),
                                     ax.n_cells,
                                     ax.n_dofs};
}

}  // namespace pantr::bspline
