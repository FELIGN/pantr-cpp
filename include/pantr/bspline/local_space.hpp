/// @file local_space.hpp
/// @brief Serial windowing kernels for distributing a tensor-product B-spline space: the support
/// closure of a set of owned cells (the halo) and the owner rank of each function.
///
/// Cells are flat-indexed in C order over the per-direction interval counts and functions in C
/// order over the per-direction function counts, as `coupling_graph` and the spanwise extraction
/// index them. The owner of a function is the owner of the active cell (`cell_owner >= 0`) with the
/// smallest flat id in the function's support: a flat C-order id is monotone in the lexicographic
/// order of the multi-index, so a C-order walk over a support box visits cells in increasing id and
/// the first active cell met is the smallest.
#pragma once

#include "pantr/bspline/space.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/parallel/parallel_for.hpp"
#include "pantr/thb/detail/support.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
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
    const thb::detail::Support1D sup = thb::detail::func_support_1d(sp);
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

}  // namespace pantr::bspline
