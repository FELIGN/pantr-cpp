/// @file local_space.hpp
/// @brief Serial windowing kernels for distributing a THB space: the cross-level support closure of
/// a set of owned cells (the halo) and the owner rank of every function, and their composition into
/// a rank's local space (`build_local`).
///
/// Both read the space's cell-to-function incidence (`THBSplineSpace::cell_dof_incidence`) rather
/// than walking supports, since the supports of a hierarchical function span several levels.
#pragma once

#include "pantr/bspline/local_space.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/grid/partition.hpp"
#include "pantr/parallel/parallel_for.hpp"
#include "pantr/thb/space.hpp"
#include "pantr/thb/types.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace pantr::thb {

/// Get the cross-level support-closure halo of a set of owned cells.
///
/// A cell is in the halo iff it shares an active function with an owned cell, so every function
/// touching an owned cell has its whole support inside the owned and halo cells. The functions of
/// the owned cells are marked, then every cell touching a marked function is found, in parallel
/// over the cells; each writes its own flag, so the result is independent of the thread count. A
/// cell costs one read per function it holds, so the work per cell is the mean incidence length.
///
/// Args:
///     space (const THBSplineSpace<T>&): The space.
///     owned_cells (std::span<const std::int64_t>): Active cell ids; repeats are ignored.
///
/// Returns:
///     std::vector<std::int64_t>: The halo cell ids, ascending, the owned cells excluded; empty
///     when `owned_cells` is.
///
/// Raises:
///     index_error: An owned id is outside `[0, grid().num_cells())`, as `active_basis` raises it.
template <StorageFloat T>
[[nodiscard]] std::vector<std::int64_t> compute_halo(const THBSplineSpace<T>& space,
                                                     std::span<const std::int64_t> owned_cells) {
  if (owned_cells.empty()) {
    return {};
  }
  const auto n_cells = static_cast<std::size_t>(space.grid().num_cells());
  std::vector<unsigned char> owned(n_cells, 0);
  std::vector<unsigned char> marked(static_cast<std::size_t>(space.num_total_basis()), 0);
  for (const std::int64_t c : owned_cells) {
    // The id is validated by `active_basis`, so it indexes `owned` afterwards.
    for (const std::int64_t d : space.active_basis(c)) {
      marked[static_cast<std::size_t>(d)] = 1;
    }
    owned[static_cast<std::size_t>(c)] = 1;
  }
  const CellIncidence inc = space.cell_dof_incidence();
  std::vector<unsigned char> touches(n_cells, 0);
  const parallel::WorkEstimate work{
      (std::max)(1.0, static_cast<double>(inc.dofs.size()) / static_cast<double>(n_cells))};
  parallel::parallel_for(n_cells, work, [&](std::size_t begin, std::size_t end) {
    for (std::size_t c = begin; c < end; ++c) {
      const auto first = static_cast<std::size_t>(inc.cell_ptr[c]);
      const auto last = static_cast<std::size_t>(inc.cell_ptr[c + 1]);
      for (std::size_t j = first; j < last; ++j) {
        if (marked[static_cast<std::size_t>(inc.dofs[j])] != 0) {
          touches[c] = 1;
          break;
        }
      }
    }
  });
  std::vector<std::int64_t> halo;
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (touches[c] != 0 && owned[c] == 0) {
      halo.push_back(static_cast<std::int64_t>(c));
    }
  }
  return halo;
}

/// Get the owner rank of every function (lex-first-active-cell rule).
///
/// A function is owned by the owner of the active cell (`cell_owner >= 0`) with the smallest id
/// among the cells whose incidence holds it; -1 for a function none of whose cells is active.
/// One serial pass over the incidence in increasing cell id, the first writer winning: the rule is
/// order-dependent, so the pass is not split.
///
/// Args:
///     space (const THBSplineSpace<T>&): The space.
///     cell_owner (std::span<const std::int32_t>): Owner rank of every cell, -1 if inactive;
///         `grid().num_cells()` entries.
///
/// Returns:
///     std::vector<std::int32_t>: `num_total_basis()` owners.
///
/// Raises:
///     value_error: `cell_owner` has the wrong length.
template <StorageFloat T>
[[nodiscard]] std::vector<std::int32_t> dof_owner(const THBSplineSpace<T>& space,
                                                  std::span<const std::int32_t> cell_owner) {
  const auto n_cells = static_cast<std::size_t>(space.grid().num_cells());
  if (cell_owner.size() != n_cells) {
    throw value_error("partition has " + std::to_string(cell_owner.size()) + " cells; expected " +
                      std::to_string(n_cells) + " (grid.num_cells).");
  }
  std::vector<std::int32_t> owners(static_cast<std::size_t>(space.num_total_basis()), -1);
  const CellIncidence inc = space.cell_dof_incidence();
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (cell_owner[c] < 0) {
      continue;
    }
    const auto first = static_cast<std::size_t>(inc.cell_ptr[c]);
    const auto last = static_cast<std::size_t>(inc.cell_ptr[c + 1]);
    for (std::size_t j = first; j < last; ++j) {
      std::int32_t& slot = owners[static_cast<std::size_t>(inc.dofs[j])];
      if (slot < 0) {
        slot = cell_owner[c];
      }
    }
  }
  return owners;
}

/// Build the rank-local windowed space of a distributed THB space.
///
/// Windows `space` to the cells `rank` owns together with their cross-level support-closure halo
/// (`compute_halo`), with `THBSplineSpace::restrict`, which also gives the maps of the local cells
/// and functions to the global ones. The halo makes the owned cells interior, so the local basis
/// equals the global basis pointwise over them. A cell is owned iff `partition` says so; a function
/// iff `dof_owner` says so, a local function with no global counterpart (-1 in the map) never.
///
/// Args:
///     space (const THBSplineSpace<T>&): The global hierarchical space.
///     partition (const grid::Partition&): Owner of every cell of `space.grid()`;
///         `grid().num_cells()` cells.
///     rank (std::int64_t): The rank whose local space is built, in `[0, partition.n_parts())`.
///
/// Returns:
///     bspline::LocalSpace<THBSplineSpace<T>>: The windowed space, its maps to the global cells and
///     functions, and the ownership masks.
///
/// Raises:
///     value_error: `rank` is out of range; `partition` has another cell count than the grid;
///         `rank` owns no cell.
///     state_error: `THBSplineSpace::restrict` returned inconsistent maps (a cell count other than
///         its grid's, a cell id outside `[0, grid().num_cells())`, or a function id at or above
///         `num_total_basis()`), an invariant of `restrict`.
template <StorageFloat T>
[[nodiscard]] bspline::LocalSpace<THBSplineSpace<T>> build_local(const THBSplineSpace<T>& space,
                                                                 const grid::Partition& partition,
                                                                 std::int64_t rank) {
  bspline::detail::check_rank_in_partition(partition, rank);
  const std::int64_t n_global_cells = space.grid().num_cells();
  if (partition.n_cells() != n_global_cells) {
    throw value_error("partition has " + std::to_string(partition.n_cells()) + " cells; expected " +
                      std::to_string(n_global_cells) + " (grid.num_cells).");
  }
  const std::vector<std::int64_t> owned = bspline::detail::owned_cells_or_throw(partition, rank);
  const std::vector<std::int64_t> halo = compute_halo(space, owned);
  const std::vector<std::int64_t> window = bspline::detail::sorted_union(owned, halo);

  THBSplineSpaceRestriction<T> restr = space.restrict(window);
  const std::int64_t n_local_cells = restr.space.grid().num_cells();
  if (static_cast<std::int64_t>(restr.local_to_global_cell.size()) != n_local_cells) {
    throw state_error("THBSplineSpace.restrict returned " +
                      std::to_string(restr.local_to_global_cell.size()) +
                      " cells in local_to_global_cell but " + std::to_string(n_local_cells) +
                      " cells in space.grid for a window of " + std::to_string(window.size()) +
                      " cells. This is a bug in restrict().");
  }
  if (std::any_of(restr.local_to_global_cell.begin(), restr.local_to_global_cell.end(),
                  [](std::int64_t c) { return c < 0; })) {
    throw state_error(
        "THBSplineSpace.restrict returned a negative value in local_to_global_cell. "
        "This is a bug in restrict().");
  }
  // Python's numpy indexing raises on an id past the end; here it would be an out-of-bounds read.
  if (std::any_of(restr.local_to_global_cell.begin(), restr.local_to_global_cell.end(),
                  [&](std::int64_t c) { return c >= n_global_cells; })) {
    throw state_error(
        "THBSplineSpace.restrict returned a value >= grid.num_cells in local_to_global_cell. "
        "This is a bug in restrict().");
  }
  const std::int64_t n_global_dofs = space.num_total_basis();
  std::int64_t largest_dof = -1;
  for (const std::int64_t d : restr.local_to_global_dof) {
    largest_dof = (std::max)(largest_dof, d);
  }
  if (largest_dof >= n_global_dofs) {
    throw state_error("THBSplineSpace.restrict returned a local_to_global_dof value " +
                      std::to_string(largest_dof) + " >= num_total_basis " +
                      std::to_string(n_global_dofs) + ". This is a bug in restrict().");
  }

  const std::span<const std::int32_t> cell_owner = partition.cell_owner();
  std::vector<std::uint8_t> owned_cell_mask(restr.local_to_global_cell.size());
  for (std::size_t i = 0; i < owned_cell_mask.size(); ++i) {
    const auto owner = static_cast<std::int64_t>(
        cell_owner[static_cast<std::size_t>(restr.local_to_global_cell[i])]);
    owned_cell_mask[i] = owner == rank ? 1 : 0;
  }
  const std::vector<std::int32_t> global_dof_owner = dof_owner(space, cell_owner);
  std::vector<std::uint8_t> owned_dof_mask(restr.local_to_global_dof.size(), 0);
  for (std::size_t i = 0; i < owned_dof_mask.size(); ++i) {
    const std::int64_t g = restr.local_to_global_dof[i];
    if (g >= 0) {
      owned_dof_mask[i] =
          static_cast<std::int64_t>(global_dof_owner[static_cast<std::size_t>(g)]) == rank ? 1 : 0;
    }
  }
  return bspline::LocalSpace<THBSplineSpace<T>>{std::move(restr.space),
                                                std::move(restr.local_to_global_cell),
                                                std::move(restr.local_to_global_dof),
                                                std::move(owned_cell_mask),
                                                std::move(owned_dof_mask),
                                                n_global_cells,
                                                n_global_dofs};
}

}  // namespace pantr::thb
