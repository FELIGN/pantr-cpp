/// @file refine.hpp
/// @brief The grading algorithms behind `THBSplineSpace::refine`, `refine_region` and `coarsen`:
/// the recursive refinement neighborhood and the coarsening neighborhood of Carraturo et al.,
/// driven over a grid that is rebound after every step.
///
/// The grid is an immutable value, so where the Python implementation mutates a deep copy these
/// functions keep a local grid and replace it by the grid `refined` / `coarsened_cells` returns.
/// Every neighborhood reads the supports (and the factor) of the space being refined, not of the
/// grid under construction, and queries only the active set of the grid under construction.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed) unless a function says so. For
///     general use, call `pantr::thb::THBSplineSpace`.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/grid/detail/py_text.hpp"
#include "pantr/grid/hierarchical_grid.hpp"
#include "pantr/thb/detail/rep.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::thb::detail {

/// A cell named by its level and its per-axis index at that level: the form that survives a
/// change of the grid, since flat ids are reassigned by every refinement and coarsening.
using LevelCell = std::pair<std::int64_t, std::vector<std::int64_t>>;

/// Spell a list of ints as Python spells `repr(list)`: `[]`, `[3]`, `[3, 4]`.
///
/// Args:
///     xs (std::span<const std::int64_t>): The entries.
///
/// Returns:
///     std::string: The spelling.
[[nodiscard]] inline std::string py_repr_int_list(std::span<const std::int64_t> xs) {
  std::string out = "[";
  for (std::size_t i = 0; i < xs.size(); ++i) {
    out += (i > 0 ? ", " : "") + std::to_string(xs[i]);
  }
  return out + "]";
}

/// Check the `admissible_class` argument of the refine and coarsen operations.
///
/// Args:
///     admissible_class (std::optional<std::int64_t>): The class, or no value for none.
///
/// Raises:
///     value_error: A value below 2.
inline void check_admissible_class(std::optional<std::int64_t> admissible_class) {
  if (admissible_class.has_value() && *admissible_class < 2) {
    throw value_error("admissible_class must be an integer >= 2 or None; got " +
                      std::to_string(*admissible_class) + ".");
  }
}

/// Sort cell ids and drop repeats, then check they name cells of a grid with `num_cells` cells.
///
/// Args:
///     ids (std::span<const std::int64_t>): The ids.
///     num_cells (std::int64_t): The grid's cell count.
///
/// Returns:
///     std::vector<std::int64_t>: The distinct ids, increasing.
///
/// Raises:
///     index_error: Some id is outside `[0, num_cells)`; the message lists every one, increasing.
[[nodiscard]] inline std::vector<std::int64_t> sorted_valid_ids(std::span<const std::int64_t> ids,
                                                                std::int64_t num_cells) {
  std::vector<std::int64_t> sorted(ids.begin(), ids.end());
  std::sort(sorted.begin(), sorted.end());
  sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
  std::vector<std::int64_t> bad;
  std::copy_if(sorted.begin(), sorted.end(), std::back_inserter(bad),
               [num_cells](std::int64_t id) { return id < 0 || id >= num_cells; });
  if (!bad.empty()) {
    throw index_error("cell_ids must lie in [0, " + std::to_string(num_cells) +
                      "); got out-of-range id(s): " + py_repr_int_list(bad) + ".");
  }
  return sorted;
}

/// Name the cells with the given ids by level and index, on the grid the ids belong to.
///
/// Args:
///     grid (const grid::HierarchicalGrid&): The grid.
///     ids (std::span<const std::int64_t>): Valid cell ids.
///
/// Returns:
///     std::vector<LevelCell>: One entry per id, in the order of `ids`.
[[nodiscard]] inline std::vector<LevelCell> named_cells(const grid::HierarchicalGrid& grid,
                                                        std::span<const std::int64_t> ids) {
  std::vector<LevelCell> cells;
  cells.reserve(ids.size());
  for (const std::int64_t cid : ids) {
    cells.emplace_back(grid.cell_level(cid), grid.cell_multi_index(cid));
  }
  return cells;
}

/// Check a `[lo, hi)` cell-index box at `level`: the four checks `HierarchicalGrid::refined`
/// makes, in its order and with its text, made before any cell is looked at (the box may hold no
/// active cell, in which case no grid operation would ever report a bad one).
///
/// Args:
///     grid (const grid::HierarchicalGrid&): The grid.
///     level (std::int64_t): Level the box lives at.
///     lo (std::span<const std::int64_t>): Per-axis start (inclusive).
///     hi (std::span<const std::int64_t>): Per-axis end (exclusive).
///
/// Raises:
///     value_error: `level` is outside `[0, max_level()]`, `lo` or `hi` has the wrong length, some
///         `lo[k] >= hi[k]`, or the box leaves `[0, level_cells_per_axis(level))` on some axis.
inline void validate_region(const grid::HierarchicalGrid& grid, std::int64_t level,
                            std::span<const std::int64_t> lo, std::span<const std::int64_t> hi) {
  const std::size_t nd = grid.ndim();
  if (!(0 <= level && level <= grid.max_level())) {
    throw value_error("level must be in [0, " + std::to_string(grid.max_level()) + "]; got " +
                      std::to_string(level) + ".");
  }
  if (lo.size() != nd || hi.size() != nd) {
    throw value_error("lo and hi must have length " + std::to_string(nd) + "; got " +
                      std::to_string(lo.size()) + " and " + std::to_string(hi.size()) + ".");
  }
  for (std::size_t k = 0; k < nd; ++k) {
    if (lo[k] >= hi[k]) {
      throw value_error("lo must be strictly less than hi in every dimension; got lo=" +
                        grid::detail::py_repr_ints(lo) + ", hi=" + grid::detail::py_repr_ints(hi) +
                        ".");
    }
  }
  const std::vector<std::int64_t> counts = grid.level_cells_per_axis(level);
  for (std::size_t k = 0; k < nd; ++k) {
    if (lo[k] < 0 || hi[k] > counts[k]) {
      throw value_error("[lo, hi) out of bounds at level " + std::to_string(level) + ": axis " +
                        std::to_string(k) + " needs [0, " + std::to_string(counts[k]) + "), got [" +
                        std::to_string(lo[k]) + ", " + std::to_string(hi[k]) + ").");
    }
  }
}

/// List the active leaves of `level` inside a validated box, in lexicographic order of their index.
///
/// The leaves are found by intersecting the level's active blocks with the box, so the cost follows
/// the number of leaves found and not the volume of the box.
///
/// Args:
///     grid (const grid::HierarchicalGrid&): The grid.
///     level (std::int64_t): A level of the grid.
///     lo (std::span<const std::int64_t>): Per-axis start (inclusive), validated.
///     hi (std::span<const std::int64_t>): Per-axis end (exclusive), validated.
///
/// Returns:
///     std::vector<LevelCell>: The leaves, increasing in index (the last axis varies fastest).
[[nodiscard]] inline std::vector<LevelCell> active_leaves_in_box(const grid::HierarchicalGrid& grid,
                                                                 std::int64_t level,
                                                                 std::span<const std::int64_t> lo,
                                                                 std::span<const std::int64_t> hi) {
  const std::size_t nd = grid.ndim();
  const std::vector<std::int64_t> box_lo(lo.begin(), lo.end());
  const std::vector<std::int64_t> box_hi(hi.begin(), hi.end());
  std::vector<std::vector<std::int64_t>> found;
  for (const grid::detail::Block& block : grid.active_blocks(level)) {
    const std::optional<grid::detail::Block> inter =
        grid::detail::rect_intersect(block.first, block.second, box_lo, box_hi);
    if (!inter.has_value()) {
      continue;
    }
    std::vector<std::int64_t> idx = inter->first;
    bool done = false;
    while (!done) {
      found.push_back(idx);
      done = true;
      for (std::size_t k = nd; k-- > 0;) {
        if (++idx[k] < inter->second[k]) {
          done = false;
          break;
        }
        idx[k] = inter->first[k];
      }
    }
  }
  std::sort(found.begin(), found.end());
  std::vector<LevelCell> cells;
  cells.reserve(found.size());
  for (std::vector<std::int64_t>& idx : found) {
    cells.emplace_back(level, std::move(idx));
  }
  return cells;
}

/// Get the refinement neighborhood of a cell for class `m` (Definition 3.4 of Carraturo et al.).
///
/// The cells at level `level - m + 1` that are parents of a level-`level - m + 2` cell touched by
/// any B-spline whose support covers the cell's ancestor at level `level - m + 2`, and that are
/// currently active leaves of `current`. The supports are the space's own (`rep`), the active set
/// is `current`'s.
///
/// Args:
///     rep (const THBSpaceRep<T>&): The space being refined.
///     level (std::int64_t): Level of the cell, at most the space's deepest level.
///     midx (std::span<const std::int64_t>): Per-axis index of the cell at `level`.
///     m (std::int64_t): The admissibility class, at least 2.
///     current (const grid::HierarchicalGrid&): The grid whose active set is queried.
///
/// Returns:
///     std::vector<std::vector<std::int64_t>>: The per-axis indices, at level `level - m + 1`, of
///     the neighbors, in lexicographic order; empty when that level is below 0.
template <StorageFloat T>
[[nodiscard]] std::vector<std::vector<std::int64_t>> refinement_neighborhood(
    const THBSpaceRep<T>& rep, std::int64_t level, std::span<const std::int64_t> midx,
    std::int64_t m, const grid::HierarchicalGrid& current) {
  const std::size_t dim = midx.size();
  const std::span<const std::int64_t> factor = rep.hier.factor();
  const std::int64_t k_nbr = level - m + 1;
  if (k_nbr < 0) {
    return {};
  }
  // `m >= 2` gives `k_ext <= level`, and `level` is at most the deepest level of `rep`, so the
  // level `k_ext` has a support table.
  const std::int64_t k_ext = k_nbr + 1;
  const auto& support = rep.support[static_cast<std::size_t>(k_ext)];
  std::vector<std::int64_t> range_lo(dim);
  std::vector<std::int64_t> range_hi(dim);  // inclusive
  for (std::size_t d = 0; d < dim; ++d) {
    // The ancestor of the cell at level `k_ext`.
    const std::int64_t q = midx[d] / grid::detail::checked_pow(factor[d], level - k_ext);
    const detail::Support1D& s = support[d];
    const std::int64_t fb = s.first_basis[static_cast<std::size_t>(q)];
    const std::int64_t s_lo = s.first_cell[static_cast<std::size_t>(fb)];
    const std::int64_t s_hi = s.last_cell[static_cast<std::size_t>(fb + rep.degrees[d])] + 1;
    range_lo[d] = s_lo / factor[d];
    range_hi[d] = (s_hi - 1) / factor[d];
  }
  std::vector<std::vector<std::int64_t>> out;
  std::vector<std::int64_t> p = range_lo;
  bool done = false;
  while (!done) {
    if (current.is_active_leaf(k_nbr, p)) {
      out.push_back(p);
    }
    done = true;
    for (std::size_t d = dim; d-- > 0;) {
      if (++p[d] <= range_hi[d]) {
        done = false;
        break;
      }
      p[d] = range_lo[d];
    }
  }
  return out;
}

/// Refine the cell `(level, midx)` of `current`, grading for class-`m` admissibility.
///
/// Refines every cell of the refinement neighborhood, recursively and at the coarser level `level -
/// m + 1`, before subdividing the cell itself (Algorithm 4 of Carraturo et al.). The neighborhood
/// is listed once, before any of its cells is refined, and each listed cell is visited even if an
/// earlier visit has already refined it: the visit then only recurses into its own neighborhood.
///
/// The recursion depth is bounded by the level count: each call is at a level `m - 1 >= 1` below
/// its caller and levels are not negative, so at most `level + 1` calls are nested.
///
/// Args:
///     rep (const THBSpaceRep<T>&): The space being refined.
///     current (grid::HierarchicalGrid&): The grid under construction; replaced by each refinement.
///     level (std::int64_t): Level of the cell.
///     midx (std::span<const std::int64_t>): Per-axis index of the cell at `level`.
///     m (std::int64_t): The admissibility class, at least 2.
template <StorageFloat T>
void refine_recursive(const THBSpaceRep<T>& rep, grid::HierarchicalGrid& current,
                      std::int64_t level, std::span<const std::int64_t> midx, std::int64_t m) {
  const std::vector<std::vector<std::int64_t>> neighbors =
      refinement_neighborhood(rep, level, midx, m, current);
  for (const std::vector<std::int64_t>& nmidx : neighbors) {
    refine_recursive(rep, current, level - m + 1, nmidx, m);
  }
  if (current.is_active_leaf(level, midx)) {
    std::vector<std::int64_t> hi(midx.begin(), midx.end());
    for (std::int64_t& h : hi) {
      h += 1;
    }
    current = current.refined(level, midx, hi);
  }
}

/// Refine the named cells of a space's grid, in order.
///
/// Args:
///     rep (const THBSpaceRep<T>&): The space being refined.
///     marked (const std::vector<LevelCell>&): Cells named on the space's own grid.
///     admissible_class (std::optional<std::int64_t>): The class (at least 2) to grade for, or no
///         value to refine exactly the named cells that are still active leaves.
///
/// Returns:
///     grid::HierarchicalGrid: The refined grid.
///
/// Raises:
///     std::overflow_error: As `HierarchicalGrid::refined`.
template <StorageFloat T>
[[nodiscard]] grid::HierarchicalGrid refine_marked(const THBSpaceRep<T>& rep,
                                                   const std::vector<LevelCell>& marked,
                                                   std::optional<std::int64_t> admissible_class) {
  grid::HierarchicalGrid current = rep.hier;
  for (const auto& [level, midx] : marked) {
    if (admissible_class.has_value()) {
      refine_recursive(rep, current, level, midx, *admissible_class);
    } else if (current.is_active_leaf(level, midx)) {
      std::vector<std::int64_t> hi = midx;
      for (std::int64_t& h : hi) {
        h += 1;
      }
      current = current.refined(level, midx, hi);
    }
  }
  return current;
}

/// Report whether the coarsening neighborhood of a parent is empty (Definition 3.5 of Carraturo et
/// al.).
///
/// The neighborhood is the set of active cells at level `parent_level + m` inside the multilevel
/// support extension, at level `parent_level + 1`, of the parent's children. Empty means that
/// reactivating the parent keeps the mesh admissible of class `m`. The supports are the space's
/// own (`rep`), the active set is `current`'s.
///
/// Args:
///     rep (const THBSpaceRep<T>&): The space being coarsened.
///     parent_level (std::int64_t): Level of the parent, with `parent_level + 1` a level of `rep`.
///     pmidx (std::span<const std::int64_t>): Per-axis index of the parent at `parent_level`.
///     m (std::int64_t): The admissibility class, at least 2.
///     current (const grid::HierarchicalGrid&): The grid whose active blocks are queried.
///
/// Returns:
///     bool: True iff no active cell at level `parent_level + m` lies in the extension.
template <StorageFloat T>
[[nodiscard]] bool coarsening_neighborhood_empty(const THBSpaceRep<T>& rep,
                                                 std::int64_t parent_level,
                                                 std::span<const std::int64_t> pmidx,
                                                 std::int64_t m,
                                                 const grid::HierarchicalGrid& current) {
  const std::size_t dim = pmidx.size();
  const std::span<const std::int64_t> factor = rep.hier.factor();
  const auto& support = rep.support[static_cast<std::size_t>(parent_level) + 1];
  std::vector<std::int64_t> ext_lo(dim);
  std::vector<std::int64_t> ext_hi(dim);
  for (std::size_t d = 0; d < dim; ++d) {
    const detail::Support1D& s = support[d];
    const std::int64_t c_lo = pmidx[d] * factor[d];
    const std::int64_t c_hi = (pmidx[d] + 1) * factor[d];
    const std::int64_t fmin = s.first_basis[static_cast<std::size_t>(c_lo)];
    const std::int64_t fmax = s.first_basis[static_cast<std::size_t>(c_hi - 1)] + rep.degrees[d];
    ext_lo[d] = s.first_cell[static_cast<std::size_t>(fmin)];
    ext_hi[d] = s.last_cell[static_cast<std::size_t>(fmax)] + 1;
  }
  // `parent_level + m > max_level`, written so that a huge class cannot overflow the sum (the
  // parent's children are active leaves of `current`, so `max_level >= parent_level + 1`).
  if (m > current.max_level() - parent_level) {
    return true;
  }
  const std::int64_t target = parent_level + m;
  std::vector<std::int64_t> box_lo(dim);
  std::vector<std::int64_t> box_hi(dim);
  for (std::size_t d = 0; d < dim; ++d) {
    const std::int64_t scale = grid::detail::checked_pow(factor[d], m - 1);
    box_lo[d] = ext_lo[d] * scale;
    box_hi[d] = ext_hi[d] * scale;
  }
  for (const grid::detail::Block& block : current.active_blocks(target)) {
    bool overlaps = true;
    for (std::size_t d = 0; d < dim; ++d) {
      overlaps = overlaps &&
                 (std::max)(box_lo[d], block.first[d]) < (std::min)(box_hi[d], block.second[d]);
    }
    if (overlaps) {
      return false;
    }
  }
  return true;
}

/// Coarsen away the named cells of a space's grid, deepest parent first.
///
/// A parent is reactivated only when all its children are named active leaves of the grid under
/// construction (an incomplete family is skipped), and, with a class, only when its coarsening
/// neighborhood is empty (the veto). Parents are taken deepest level first and then in
/// lexicographic order of their index; parents of one level are independent of each other, since a
/// veto reads the blocks `m >= 2` levels below the parent's.
///
/// Args:
///     rep (const THBSpaceRep<T>&): The space being coarsened.
///     ids (const std::vector<std::int64_t>&): Valid cell ids of the space's own grid.
///     admissible_class (std::optional<std::int64_t>): The class (at least 2) to guard, or no
///         value to skip the veto.
///
/// Returns:
///     grid::HierarchicalGrid: The coarsened grid.
template <StorageFloat T>
[[nodiscard]] grid::HierarchicalGrid coarsen_marked(const THBSpaceRep<T>& rep,
                                                    const std::vector<std::int64_t>& ids,
                                                    std::optional<std::int64_t> admissible_class) {
  const std::size_t dim = rep.root_space.dim();
  const std::span<const std::int64_t> factor = rep.hier.factor();
  const std::set<LevelCell> marked = [&] {
    const std::vector<LevelCell> named = named_cells(rep.hier, ids);
    return std::set<LevelCell>(named.begin(), named.end());
  }();
  // Keyed (-level, index): deepest first, then lexicographic.
  std::set<LevelCell> parents;
  for (const auto& [level, midx] : marked) {
    if (level < 1) {
      continue;
    }
    std::vector<std::int64_t> parent(dim);
    for (std::size_t d = 0; d < dim; ++d) {
      parent[d] = midx[d] / factor[d];
    }
    parents.emplace(-(level - 1), std::move(parent));
  }
  grid::HierarchicalGrid current = rep.hier;
  for (const auto& [neg_level, pmidx] : parents) {
    const std::int64_t parent_level = -neg_level;
    // The family's marked children as they are on the current grid: coarsening reassigns flat ids,
    // so they are resolved afresh. One that is not marked, or not a leaf, makes the family
    // incomplete, which `coarsened_cells` would skip anyway.
    std::vector<std::int64_t> child_ids;
    std::vector<std::int64_t> child(dim);
    for (std::size_t d = 0; d < dim; ++d) {
      child[d] = pmidx[d] * factor[d];
    }
    bool complete = true;
    bool more = true;
    while (more && complete) {
      const std::optional<std::int64_t> cid = current.cell_id(parent_level + 1, child);
      if (!cid.has_value() || !marked.contains({parent_level + 1, child})) {
        complete = false;
        break;
      }
      child_ids.push_back(*cid);
      more = false;
      for (std::size_t d = dim; d-- > 0;) {
        if (++child[d] < (pmidx[d] + 1) * factor[d]) {
          more = true;
          break;
        }
        child[d] = pmidx[d] * factor[d];
      }
    }
    if (!complete) {
      continue;
    }
    if (admissible_class.has_value() &&
        !coarsening_neighborhood_empty(rep, parent_level, pmidx, *admissible_class, current)) {
      continue;
    }
    current = current.coarsened_cells(child_ids);
  }
  return current;
}

}  // namespace pantr::thb::detail
