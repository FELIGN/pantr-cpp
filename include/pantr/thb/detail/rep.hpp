/// @file rep.hpp
/// @brief The shared representation of a `THBSplineSpace` and its construction: level spaces,
/// supports, the Kraft selection, the two-scale matrices and the truncated coefficients, then the
/// lazy per-cell table.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed). For general use, call
///     `pantr::thb::THBSplineSpace`.
#pragma once

#include "pantr/bspline/detail/support_1d.hpp"
#include "pantr/bspline/knot_insertion.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/lazy.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/grid/hierarchical_grid.hpp"
#include "pantr/parallel/parallel_for.hpp"
#include "pantr/parallel/work_model.hpp"
#include "pantr/thb/detail/summed_area.hpp"
#include "pantr/thb/detail/truncation.hpp"
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

namespace pantr::thb::detail {

/// The per-cell table: the active functions non-zero on every cell, in compressed-row form.
///
/// Attributes:
///     cell_ptr (std::vector<std::int64_t>): `num_cells + 1` offsets.
///     dofs (std::vector<std::int64_t>): Global dof of each entry, increasing within a cell.
///     levels (std::vector<std::int64_t>): Level of each entry.
///     multi (std::vector<std::int64_t>): Row-major `(entries, dim)` per-direction function index
///         of each entry in its level's space.
///     max_active (std::int64_t): The largest entry count of any cell.
struct CellBlock {
  std::vector<std::int64_t> cell_ptr;
  std::vector<std::int64_t> dofs;
  std::vector<std::int64_t> levels;
  std::vector<std::int64_t> multi;
  std::int64_t max_active = 0;
};

/// The shared representation of a `THBSplineSpace`.
///
/// Everything but `cells` is computed at construction (the space is eager, as the Python one is);
/// `cells` is the one lazy block, computed on first use for every cell at once.
///
/// Attributes:
///     root_space (bspline::BsplineSpace<T>): The level-0 tensor-product space.
///     hier (grid::HierarchicalGrid): The active-cell hierarchy.
///     truncate (bool): Whether the basis is truncated (THB) or not (HB).
///     regularity (std::vector<std::optional<int>>): Per-direction continuity at the inserted
///         knots, one entry per direction.
///     degrees (std::vector<int>): Per-direction degrees.
///     level_spaces (std::vector<bspline::BsplineSpace<T>>): Per-level tensor-product spaces.
///     num_basis (std::vector<std::vector<std::int64_t>>): `[level][direction]` function count.
///     support (std::vector<std::vector<bspline::detail::Support1D>>): `[level][direction]`
///         function supports.
///     active (std::vector<std::vector<std::int64_t>>): Per-level sorted flat (row-major) indices
///         of the active tensor-product functions.
///     func_offset (std::vector<std::int64_t>): Per-level global dof base, `num_levels + 1`
///         cumulative active counts.
///     two_scale (std::vector<std::vector<TwoScale1D>>): `[m][direction]` two-scale matrix from
///         level `m` to level `m + 1`.
///     trunc_index (std::vector<std::int64_t>): Per global dof, the index into `trunc` or -1 when
///         the function is untruncated (every dof of an HB space).
///     trunc (std::vector<TruncatedFunction>): The truncated functions, in increasing dof order.
///     cells (pantr::detail::Lazy<CellBlock>): The lazy per-cell table.
template <StorageFloat T>
struct THBSpaceRep {
  THBSpaceRep(bspline::BsplineSpace<T> root, grid::HierarchicalGrid hierarchy, bool truncated,
              std::vector<std::optional<int>> reg)
      : root_space(std::move(root)),
        hier(std::move(hierarchy)),
        truncate(truncated),
        regularity(std::move(reg)) {}

  bspline::BsplineSpace<T> root_space;
  grid::HierarchicalGrid hier;
  bool truncate;
  std::vector<std::optional<int>> regularity;
  std::vector<int> degrees;
  std::vector<bspline::BsplineSpace<T>> level_spaces;
  std::vector<std::vector<std::int64_t>> num_basis;
  std::vector<std::vector<bspline::detail::Support1D>> support;
  std::vector<std::vector<std::int64_t>> active;
  std::vector<std::int64_t> func_offset;
  std::vector<std::vector<TwoScale1D>> two_scale;
  std::vector<std::int64_t> trunc_index;
  std::vector<TruncatedFunction> trunc;
  pantr::detail::Lazy<CellBlock> cells;
};

/// Build the nested per-level tensor-product spaces.
///
/// Level `l + 1` is level `l` subdivided in every direction by the grid's factor through the same
/// knot functions the 1D space's `subdivide` uses (uniform subdivision knots, then insertion); an
/// axis whose factor is 1 keeps its space.
///
/// Args:
///     rep (THBSpaceRep<T>&): The representation; its root, grid and regularity are read, its
///         `level_spaces` and `num_basis` filled.
///
/// Raises:
///     value_error: The knot functions refuse a subdivision, or a level space has fewer knot
///         intervals along an axis than the grid has cells there (subdivision knots closer than
///         the space's tolerance were merged).
template <StorageFloat T>
void build_level_spaces(THBSpaceRep<T>& rep) {
  const std::span<const std::int64_t> factor = rep.hier.factor();
  const std::size_t dim = rep.root_space.dim();
  std::vector<bspline::BsplineSpace1D<T>> current = rep.root_space.spaces();
  rep.level_spaces.push_back(rep.root_space);
  for (std::int64_t level = 1; level <= rep.hier.max_level(); ++level) {
    for (std::size_t k = 0; k < dim; ++k) {
      if (factor[k] == 1) {
        continue;
      }
      const bspline::BsplineSpace1D<T>& sp = current[k];
      std::optional<long long> reg;
      if (rep.regularity[k].has_value()) {
        reg = static_cast<long long>(*rep.regularity[k]);
      }
      const std::vector<T> inserted = bspline::compute_uniform_subdivision_knots<T>(
          sp.knots(), sp.degree(), sp.tolerance(), factor[k], reg);
      const std::vector<T> merged = bspline::compute_inserted_knot_vector<T>(
          sp.knots(), sp.degree(), std::span<const T>(inserted), sp.tolerance());
      current[k] = bspline::BsplineSpace1D<T>(std::span<const T>(merged), sp.degree());
    }
    rep.level_spaces.emplace_back(current);
  }
  // The masks and the Kraft selection index the level's cells by the grid's count, so the space
  // must have exactly as many intervals; fewer happens when subdivision knots fall within the
  // space's tolerance of each other and are merged.
  for (std::int64_t level = 0; level <= rep.hier.max_level(); ++level) {
    const std::vector<std::int64_t> cells = rep.hier.level_cells_per_axis(level);
    const auto& spaces = rep.level_spaces[static_cast<std::size_t>(level)].spaces();
    for (std::size_t k = 0; k < dim; ++k) {
      const auto intervals = static_cast<std::int64_t>(spaces[k].num_intervals());
      if (intervals != cells[k]) {
        throw value_error("level " + std::to_string(level) + " has " + std::to_string(intervals) +
                          " knot intervals along axis " + std::to_string(k) + " but the grid has " +
                          std::to_string(cells[k]) +
                          " cells there; the subdivision knots are closer than the space "
                          "tolerance.");
      }
    }
    std::vector<std::int64_t> nb;
    for (std::size_t k = 0; k < dim; ++k) {
      nb.push_back(static_cast<std::int64_t>(spaces[k].num_basis()));
    }
    rep.num_basis.push_back(std::move(nb));
  }
}

/// Compute the Kraft selection of active functions per level.
///
/// A level-`l` tensor-product function is selected iff its support lies entirely in the level-`l`
/// subdomain but not entirely in the part of it refined further (`subdomain & ~active_leaf`).
/// Candidates are the functions whose support meets the bounding box of the subdomain, enumerated
/// in row-major order of their multi-index, which is increasing flat index, so each level's list
/// comes out sorted.
///
/// Args:
///     rep (THBSpaceRep<T>&): The representation, with level spaces and supports built; its
///         `active` is filled.
///
/// Returns:
///     std::vector<FalseCountTable>: Per level, the table of the refined mask, which the
///     truncation uses to decide whether a support meets a finer region.
template <StorageFloat T>
[[nodiscard]] std::vector<FalseCountTable> select_active_functions(THBSpaceRep<T>& rep) {
  const std::size_t dim = rep.root_space.dim();
  std::vector<FalseCountTable> refined_tables;
  for (std::int64_t level = 0; level <= rep.hier.max_level(); ++level) {
    const auto lv = static_cast<std::size_t>(level);
    const std::vector<std::int64_t> counts = rep.hier.level_cells_per_axis(level);
    const NdArray<std::uint8_t> subdomain = rep.hier.subdomain_mask(level);
    const NdArray<std::uint8_t> leaves = rep.hier.active_leaf_mask(level);
    std::vector<std::uint8_t> refined(subdomain.size());
    for (std::size_t i = 0; i < refined.size(); ++i) {
      refined[i] = (subdomain.data()[i] != 0 && leaves.data()[i] == 0) ? 1 : 0;
    }
    refined_tables.emplace_back(std::span<const std::uint8_t>(refined),
                                std::span<const std::int64_t>(counts));

    // Bounding box of the subdomain.
    std::vector<std::int64_t> bbox_lo(counts.size(), 0);
    std::vector<std::int64_t> bbox_hi(counts.size(), 0);
    bool any = false;
    std::vector<std::int64_t> idx(dim, 0);
    for (std::size_t cell = 0; cell < subdomain.size(); ++cell) {
      if (subdomain.data()[cell] != 0) {
        for (std::size_t k = 0; k < dim; ++k) {
          bbox_lo[k] = any ? (std::min)(bbox_lo[k], idx[k]) : idx[k];
          bbox_hi[k] = any ? (std::max)(bbox_hi[k], idx[k] + 1) : idx[k] + 1;
        }
        any = true;
      }
      for (std::size_t k = dim; k-- > 0;) {
        if (++idx[k] < counts[k]) {
          break;
        }
        idx[k] = 0;
      }
    }
    std::vector<std::int64_t>& active = rep.active.emplace_back();

    std::vector<std::vector<std::int64_t>> candidates(dim);
    for (std::size_t k = 0; k < dim; ++k) {
      const bspline::detail::Support1D& s = rep.support[lv][k];
      for (std::size_t i = 0; i < s.first_cell.size(); ++i) {
        if (s.last_cell[i] >= bbox_lo[k] && s.first_cell[i] < bbox_hi[k]) {
          candidates[k].push_back(static_cast<std::int64_t>(i));
        }
      }
    }
    const FalseCountTable subdomain_table(subdomain.flat(), counts);
    const FalseCountTable& refined_table = refined_tables.back();
    std::vector<std::size_t> pos(dim, 0);
    std::vector<std::int64_t> box_lo(dim);
    std::vector<std::int64_t> box_hi(dim);
    // A level whose subdomain is empty has no candidates and so no active function. A valid grid
    // has no such level, but `HierarchicalGrid::from_blocks` takes its blocks as a valid tiling
    // without checking it, and the Python implementation selects nothing there.
    bool done = std::any_of(candidates.begin(), candidates.end(),
                            [](const std::vector<std::int64_t>& c) { return c.empty(); });
    while (!done) {
      std::int64_t flat = 0;
      for (std::size_t k = 0; k < dim; ++k) {
        const std::int64_t fn = candidates[k][pos[k]];
        const bspline::detail::Support1D& s = rep.support[lv][k];
        box_lo[k] = s.first_cell[static_cast<std::size_t>(fn)];
        box_hi[k] = s.last_cell[static_cast<std::size_t>(fn)] + 1;
        flat = flat * rep.num_basis[lv][k] + fn;
      }
      if (subdomain_table.all_true(box_lo, box_hi) && !refined_table.all_true(box_lo, box_hi)) {
        active.push_back(flat);
      }
      done = true;
      for (std::size_t k = dim; k-- > 0;) {
        if (++pos[k] < candidates[k].size()) {
          done = false;
          break;
        }
        pos[k] = 0;
      }
    }
  }
  return refined_tables;
}

/// Build the truncated functions of the THB basis.
///
/// Each active function whose support meets a finer refinement is expressed in successively finer
/// bases (two-scale refinement), zeroing the coefficients on the active functions of each finer
/// level (truncation), until its support no longer meets a refined region. A function that no
/// truncation changed stays a plain tensor-product B-spline and is not stored.
///
/// Args:
///     rep (THBSpaceRep<T>&): The representation, with active functions and two-scale matrices
///         built; its `trunc_index` and `trunc` are filled.
///     refined_tables (const std::vector<FalseCountTable>&): Per level, the table of the refined
///         mask.
///
/// Raises:
///     value_error: A refinement step meets a zero two-scale slice (see `refine_box`).
template <StorageFloat T>
void compute_truncated_coeffs(THBSpaceRep<T>& rep,
                              const std::vector<FalseCountTable>& refined_tables) {
  const std::size_t dim = rep.root_space.dim();
  const auto num_levels = static_cast<std::int64_t>(rep.level_spaces.size());
  std::vector<std::int64_t> cell_lo(dim);
  std::vector<std::int64_t> cell_hi(dim);
  for (std::int64_t level = 0; level + 1 < num_levels; ++level) {
    const auto lv = static_cast<std::size_t>(level);
    const std::vector<std::int64_t>& nb = rep.num_basis[lv];
    const std::int64_t offset = rep.func_offset[lv];
    for (std::size_t pos = 0; pos < rep.active[lv].size(); ++pos) {
      // Unravel the flat index, row-major.
      CoeffBox box;
      box.lo.resize(dim);
      box.shape.assign(dim, 1);
      std::int64_t rest = rep.active[lv][pos];
      for (std::size_t k = dim; k-- > 0;) {
        box.lo[k] = rest % nb[k];
        rest /= nb[k];
      }
      box.coeffs.assign(1, 1.0);
      std::int64_t rep_level = level;
      bool any_zeroed = false;
      for (std::int64_t m = level; m + 1 < num_levels; ++m) {
        const auto mv = static_cast<std::size_t>(m);
        for (std::size_t k = 0; k < dim; ++k) {
          const bspline::detail::Support1D& s = rep.support[mv][k];
          cell_lo[k] = s.first_cell[static_cast<std::size_t>(box.lo[k])];
          cell_hi[k] = s.last_cell[static_cast<std::size_t>(box.lo[k]) + box.shape[k] - 1] + 1;
        }
        if (!refined_tables[mv].any_true(cell_lo, cell_hi)) {
          break;
        }
        box = refine_box(box, rep.two_scale[mv]);
        rep_level = m + 1;
        const bool zeroed = truncate_box(box, rep.active[mv + 1], rep.num_basis[mv + 1]);
        any_zeroed = any_zeroed || zeroed;
      }
      if (any_zeroed) {
        rep.trunc_index[static_cast<std::size_t>(offset) + pos] =
            static_cast<std::int64_t>(rep.trunc.size());
        rep.trunc.push_back(TruncatedFunction{rep_level, std::move(box.lo), std::move(box.shape),
                                              std::move(box.coeffs)});
      }
    }
  }
}

/// Build the whole representation of a space: everything but the lazy per-cell table.
///
/// Args:
///     root (bspline::BsplineSpace<T>): The level-0 space.
///     hier_grid (grid::HierarchicalGrid): The hierarchy.
///     truncate (bool): Whether to truncate.
///     regularity (std::vector<std::optional<int>>): One entry per direction.
///
/// Returns:
///     std::shared_ptr<const THBSpaceRep<T>>: The representation.
///
/// Raises:
///     value_error, state_error: See the steps above.
///     std::overflow_error: A two-scale matrix overflows `T` (see `make_two_scale`).
template <StorageFloat T>
[[nodiscard]] std::shared_ptr<const THBSpaceRep<T>> build_rep(
    bspline::BsplineSpace<T> root, grid::HierarchicalGrid hier_grid, bool truncate,
    std::vector<std::optional<int>> regularity) {
  auto rep = std::make_shared<THBSpaceRep<T>>(std::move(root), std::move(hier_grid), truncate,
                                              std::move(regularity));
  const std::size_t dim = rep->root_space.dim();
  for (const auto& s : rep->root_space.spaces()) {
    rep->degrees.push_back(s.degree());
  }
  build_level_spaces(*rep);
  for (const auto& level_space : rep->level_spaces) {
    std::vector<bspline::detail::Support1D>& per_dir = rep->support.emplace_back();
    for (const auto& sp1d : level_space.spaces()) {
      per_dir.push_back(bspline::detail::func_support_1d<T>(sp1d));
    }
  }
  const std::vector<FalseCountTable> refined_tables = select_active_functions(*rep);
  rep->func_offset.assign(1, 0);
  for (const auto& per_level : rep->active) {
    rep->func_offset.push_back(rep->func_offset.back() +
                               static_cast<std::int64_t>(per_level.size()));
  }
  for (std::size_t m = 0; m + 1 < rep->level_spaces.size(); ++m) {
    std::vector<TwoScale1D>& per_dir = rep->two_scale.emplace_back();
    for (std::size_t k = 0; k < dim; ++k) {
      per_dir.push_back(make_two_scale<T>(rep->level_spaces[m].spaces()[k],
                                          rep->level_spaces[m + 1].spaces()[k]));
    }
  }
  rep->trunc_index.assign(static_cast<std::size_t>(rep->func_offset.back()), -1);
  if (truncate && rep->level_spaces.size() > 1) {
    compute_truncated_coeffs(*rep, refined_tables);
  }
  return rep;
}

/// Per-thread scratch of `for_each_cell_function`.
///
/// Attributes:
///     first (std::vector<std::int64_t>): The first supported function per direction.
///     multi (std::vector<std::int64_t>): The multi-index being visited.
struct CellScratch {
  explicit CellScratch(std::size_t dim) : first(dim), multi(dim) {}
  std::vector<std::int64_t> first;
  std::vector<std::int64_t> multi;
};

/// Visit the active functions non-zero on one cell, in increasing order of global dof.
///
/// For every level `l <= cell_level`, the cell's ancestor at level `l` has `degree + 1` supported
/// functions per direction (`first_basis` of the ancestor's interval), and each of their
/// multi-indices, taken in row-major order, is looked up in the level's sorted active list.
/// Levels come in increasing order, the level offsets increase with the level, and the position in
/// a level's list increases with the flat index, which row-major order of the multi-index makes
/// increasing: so the visits are already sorted by dof and nothing is sorted afterwards.
///
/// Args:
///     rep (const THBSpaceRep<T>&): The representation.
///     factor_pow (const std::vector<std::vector<std::int64_t>>&): `[e][k]` is `factor[k]^e`, for
///         `e` up to the grid's deepest level.
///     cell_level (std::int64_t): The cell's level.
///     midx (std::span<const std::int64_t>): The cell's per-direction index at its level.
///     scratch (CellScratch&): Scratch; `scratch.multi` holds the visited function's multi-index
///         during a call of `visit`.
///     visit (F&&): Called as `visit(dof, level, multi)` once per function.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T, class F>
void for_each_cell_function(const THBSpaceRep<T>& rep,
                            const std::vector<std::vector<std::int64_t>>& factor_pow,
                            std::int64_t cell_level, std::span<const std::int64_t> midx,
                            CellScratch& scratch, F&& visit) {
  const std::size_t dim = midx.size();
  for (std::int64_t level = 0; level <= cell_level; ++level) {
    const auto lv = static_cast<std::size_t>(level);
    const std::vector<std::int64_t>& divisor =
        factor_pow[static_cast<std::size_t>(cell_level - level)];
    for (std::size_t k = 0; k < dim; ++k) {
      const std::int64_t ancestor = midx[k] / divisor[k];
      scratch.first[k] = rep.support[lv][k].first_basis[static_cast<std::size_t>(ancestor)];
      scratch.multi[k] = scratch.first[k];
    }
    const std::vector<std::int64_t>& active = rep.active[lv];
    const std::int64_t offset = rep.func_offset[lv];
    bool done = false;
    while (!done) {
      std::int64_t flat = 0;
      for (std::size_t k = 0; k < dim; ++k) {
        flat = flat * rep.num_basis[lv][k] + scratch.multi[k];
      }
      const auto at = std::lower_bound(active.begin(), active.end(), flat);
      if (at != active.end() && *at == flat) {
        visit(offset + static_cast<std::int64_t>(at - active.begin()), level,
              std::span<const std::int64_t>(scratch.multi));
      }
      done = true;
      for (std::size_t k = dim; k-- > 0;) {
        if (++scratch.multi[k] <= scratch.first[k] + rep.degrees[k]) {
          done = false;
          break;
        }
        scratch.multi[k] = scratch.first[k];
      }
    }
  }
}

/// Build the per-cell table of every cell, in parallel over cells.
///
/// A count pass gives each cell's number of functions, a prefix sum turns the counts into offsets,
/// and a fill pass writes each cell's entries into its own disjoint range, so the table is the
/// same for every thread count.
///
/// The work of a cell is, per level it has an ancestor at (at most `num_levels`), `prod(degree +
/// 1)` candidate functions, each costing a flat-index formation (`dim` multiply-adds) and a binary
/// search of the level's active list (`log2(active + 1)` comparisons). The count and the fill pass
/// run the same loops, so both carry that estimate.
///
/// Args:
///     rep (const THBSpaceRep<T>&): The representation.
///
/// Returns:
///     CellBlock: The table.
template <StorageFloat T>
[[nodiscard]] CellBlock build_cell_block(const THBSpaceRep<T>& rep) {
  const grid::HierarchicalGrid& hier = rep.hier;
  const auto n_cells = static_cast<std::size_t>(hier.num_cells());
  const std::size_t dim = rep.root_space.dim();
  std::vector<std::vector<std::int64_t>> factor_pow(static_cast<std::size_t>(hier.max_level()) + 1);
  for (std::size_t e = 0; e < factor_pow.size(); ++e) {
    factor_pow[e].assign(dim, 1);
    for (std::size_t k = 0; k < dim; ++k) {
      factor_pow[e][k] = e == 0 ? 1 : factor_pow[e - 1][k] * hier.factor()[k];
    }
  }
  double local_functions = 1.0;
  for (const int p : rep.degrees) {
    local_functions *= static_cast<double>(p) + 1.0;
  }
  std::size_t most_active = 0;
  for (const auto& per_level : rep.active) {
    most_active = (std::max)(most_active, per_level.size());
  }
  const double per_candidate =
      static_cast<double>(dim) + std::log2(static_cast<double>(most_active) + 1.0);
  const parallel::WorkEstimate work{static_cast<double>(rep.level_spaces.size()) * local_functions *
                                    per_candidate};

  CellBlock block;
  block.cell_ptr.assign(n_cells + 1, 0);
  parallel::parallel_for(n_cells, work, [&](std::size_t begin, std::size_t end) {
    CellScratch scratch(dim);
    for (std::size_t c = begin; c < end; ++c) {
      const auto cid = static_cast<std::int64_t>(c);
      const std::int64_t level = hier.cell_level(cid);
      const std::vector<std::int64_t> midx = hier.cell_multi_index(cid);
      std::int64_t count = 0;
      for_each_cell_function(
          rep, factor_pow, level, midx, scratch,
          [&count](std::int64_t, std::int64_t, std::span<const std::int64_t>) { ++count; });
      block.cell_ptr[c + 1] = count;
    }
  });
  for (std::size_t c = 0; c < n_cells; ++c) {
    block.max_active = (std::max)(block.max_active, block.cell_ptr[c + 1]);
    block.cell_ptr[c + 1] += block.cell_ptr[c];
  }
  const auto entries = static_cast<std::size_t>(block.cell_ptr[n_cells]);
  block.dofs.assign(entries, 0);
  block.levels.assign(entries, 0);
  block.multi.assign(entries * dim, 0);
  parallel::parallel_for(n_cells, work, [&](std::size_t begin, std::size_t end) {
    CellScratch scratch(dim);
    for (std::size_t c = begin; c < end; ++c) {
      const auto cid = static_cast<std::int64_t>(c);
      const std::int64_t level = hier.cell_level(cid);
      const std::vector<std::int64_t> midx = hier.cell_multi_index(cid);
      auto at = static_cast<std::size_t>(block.cell_ptr[c]);
      for_each_cell_function(
          rep, factor_pow, level, midx, scratch,
          [&](std::int64_t dof, std::int64_t fn_level, std::span<const std::int64_t> multi) {
            block.dofs[at] = dof;
            block.levels[at] = fn_level;
            std::copy(multi.begin(), multi.end(),
                      block.multi.begin() + static_cast<std::ptrdiff_t>(at * dim));
            ++at;
          });
    }
  });
  return block;
}

}  // namespace pantr::thb::detail
