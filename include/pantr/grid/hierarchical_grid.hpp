/// @file hierarchical_grid.hpp
/// @brief `HierarchicalGrid`: a root `TensorProductGrid` plus levels of uniform rectangular
/// refinement.
///
/// Each refinement promotes a rectangular region of currently active cells to the next level,
/// splitting every cell of it into `prod(factor)` equal children. Only rectangular blocks are
/// stored, at most a short list per level, so memory is independent of the cell count. Flat cell
/// ids run level by level (level 0 first), block by block within a level (blocks sorted by their
/// lower corner), row-major within a block.
///
/// A grid is an immutable value. Refining or coarsening returns a new value, or the same value
/// (same representation, same version) when nothing changes; `version()` counts the structural
/// changes behind a value, so a consumer holding a snapshot can detect a different grid even when
/// the cell count and level count agree.
///
/// Refinement imposes no 2:1 grading: cells of any two levels may share a facet, and facet
/// adjacency walks as many levels up or down as the interface needs.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/lazy.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/grid/bvh.hpp"
#include "pantr/grid/detail/checked.hpp"
#include "pantr/grid/detail/hier_blocks.hpp"
#include "pantr/grid/detail/hier_core.hpp"
#include "pantr/grid/detail/py_text.hpp"
#include "pantr/grid/grid_base.hpp"
#include "pantr/grid/tensor_product_grid.hpp"
#include "pantr/parallel/parallel_for.hpp"
#include "pantr/parallel/work_model.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pantr::grid {

struct HierarchicalGridRestriction;

/// A hierarchical grid with a fixed per-direction subdivision factor.
///
/// Attributes:
///     rep_ (std::shared_ptr<const detail::HierarchicalGridRep>): The root, factor, per-level
///         blocks, packed kernel arrays, version and the lazy spatial index.
class HierarchicalGrid : public GridBase<HierarchicalGrid> {
 public:
  /// The name of the grid type, as messages that name a grid type spell it.
  static constexpr std::string_view type_name = "HierarchicalGrid";

  /// Build a grid with every root cell active at level 0 (version 1).
  ///
  /// Args:
  ///     root (TensorProductGrid): The level-0 grid.
  ///     factor (std::vector<std::int64_t>): Per-axis subdivision factor, `ndim` entries each at
  ///         least 1; 1 means no subdivision along that axis.
  ///
  /// Raises:
  ///     value_error: `factor` has the wrong length or an entry below 1.
  HierarchicalGrid(TensorProductGrid root, std::vector<std::int64_t> factor)
      : rep_(make_initial(std::move(root), std::move(factor))) {}

  /// Build a grid directly from per-level block lists (the pickling and windowing constructor).
  ///
  /// Each level's list is normalized (greedily merged, then sorted) and empty trailing levels are
  /// dropped. The blocks are otherwise taken as a valid decomposition: non-overlapping, and
  /// together tiling the root consistently with `factor`, which this does not verify.
  ///
  /// Args:
  ///     root (TensorProductGrid): The level-0 grid.
  ///     factor (std::vector<std::int64_t>): Per-axis subdivision factor.
  ///     blocks (std::vector<std::vector<detail::Block>>): `blocks[l]` lists the active
  ///         rectangles of level `l` in level-`l` coordinates; at least one level.
  ///     grid_version (std::int64_t): The version the value reports, at least 1.
  ///
  /// Returns:
  ///     HierarchicalGrid: The grid.
  ///
  /// Raises:
  ///     value_error: `factor` is invalid, `grid_version < 1`, there is no level, or a block is
  ///         not a non-empty rectangle of `ndim` bounds inside its level's index range.
  ///     std::overflow_error: A level's cell count along an axis, or the total, exceeds `int64`.
  [[nodiscard]] static HierarchicalGrid from_blocks(TensorProductGrid root,
                                                    std::vector<std::int64_t> factor,
                                                    std::vector<std::vector<detail::Block>> blocks,
                                                    std::int64_t grid_version) {
    check_factor(root.ndim(), factor);
    if (grid_version < 1) {
      throw value_error("version must be >= 1; got " + std::to_string(grid_version) + ".");
    }
    if (blocks.empty()) {
      throw value_error("from_blocks: blocks must hold at least one level.");
    }
    const std::size_t nd = root.ndim();
    for (std::size_t level = 0; level < blocks.size(); ++level) {
      std::vector<std::int64_t> counts(nd);
      for (std::size_t k = 0; k < nd; ++k) {
        const std::optional<std::int64_t> n =
            detail::try_cells_at_level(root, factor, k, static_cast<std::int64_t>(level));
        if (!n.has_value()) {
          throw std::overflow_error(level_overflow_message(static_cast<std::int64_t>(level), k));
        }
        counts[k] = *n;
      }
      for (const detail::Block& block : blocks[level]) {
        bool ok = block.first.size() == nd && block.second.size() == nd;
        for (std::size_t k = 0; ok && k < nd; ++k) {
          ok = 0 <= block.first[k] && block.first[k] < block.second[k] &&
               block.second[k] <= counts[k];
        }
        if (!ok) {
          throw value_error("from_blocks: the block " + detail::py_repr_ints(block.first) + ", " +
                            detail::py_repr_ints(block.second) + " at level " +
                            std::to_string(level) +
                            " is not a non-empty rectangle inside the level's index range " +
                            detail::py_repr_ints(counts) + ".");
        }
      }
      blocks[level] = detail::normalize_blocks(std::move(blocks[level]));
    }
    while (blocks.size() > 1 && blocks.back().empty()) {
      blocks.pop_back();
    }
    return HierarchicalGrid(
        detail::make_hier_rep(std::move(root), std::move(factor), std::move(blocks), grid_version));
  }

  /// Get the spatial dimension.
  ///
  /// Returns:
  ///     std::size_t: The number of axes, at least 1.
  [[nodiscard]] std::size_t ndim() const noexcept { return rep_->factor.size(); }

  /// Get the total number of active cells across all levels.
  ///
  /// Returns:
  ///     std::int64_t: The cell count.
  [[nodiscard]] std::int64_t num_cells() const noexcept { return rep_->num_cells; }

  /// Get the level-0 grid.
  ///
  /// Returns:
  ///     const TensorProductGrid&: The root, valid while this grid lives.
  [[nodiscard]] const TensorProductGrid& root() const noexcept { return rep_->root; }

  /// Get the per-axis subdivision factor.
  ///
  /// Returns:
  ///     std::span<const std::int64_t>: `ndim()` entries, each at least 1.
  [[nodiscard]] std::span<const std::int64_t> factor() const noexcept { return rep_->factor; }

  /// Get the index of the deepest level.
  ///
  /// Returns:
  ///     std::int64_t: 0 before any refinement. A middle level can hold no cells, when every cell
  ///     that reached it was refined again.
  [[nodiscard]] std::int64_t max_level() const noexcept {
    return static_cast<std::int64_t>(rep_->blocks.size()) - 1;
  }

  /// Get the structural-change counter.
  ///
  /// Returns:
  ///     std::int64_t: 1 for a new grid; each refinement or coarsening that changes the grid adds
  ///     one. `refined_cells` and `coarsened_cells` add one per level refined or per parent
  ///     demoted.
  [[nodiscard]] std::int64_t version() const noexcept { return rep_->version; }

  /// Get cell `cid`'s axis-aligned `(lo, hi)` corners.
  ///
  /// Args:
  ///     cid (std::int64_t): Cell identifier.
  ///
  /// Returns:
  ///     std::pair<std::vector<double>, std::vector<double>>: `ndim()` entries each.
  ///
  /// Raises:
  ///     index_error: `cid` is out of range.
  [[nodiscard]] std::pair<std::vector<double>, std::vector<double>> cell_bounds(
      std::int64_t cid) const {
    check_cid(cid);
    detail::Index midx(ndim());
    const std::int64_t level = detail::decode_flat_id(*rep_, cid, midx.data());
    return detail::level_cell_bounds(*rep_, level, midx.data());
  }

  /// Get the active leaf containing `pt`, if any.
  ///
  /// A top-down descent: the root cell is found by binary search, then each level either holds the
  /// position as a leaf or the containing child is computed from the cell's floating-point bounds.
  /// A point on an interior root breakpoint belongs to the lower root cell, one on a boundary
  /// between children to the upper child.
  ///
  /// Args:
  ///     pt (std::span<const double>): The point, `ndim()` coordinates.
  ///
  /// Returns:
  ///     std::optional<std::int64_t>: The leaf's flat id, or no value for a point outside the
  ///     domain or with a NaN coordinate.
  ///
  /// Raises:
  ///     value_error: `pt` does not have `ndim()` coordinates.
  [[nodiscard]] std::optional<std::int64_t> locate(std::span<const double> pt) const {
    if (pt.size() != ndim()) {
      const std::size_t got[1] = {pt.size()};
      throw value_error("pt must have shape (" + std::to_string(ndim()) + ",); got " +
                        py_repr(got) + ".");
    }
    detail::LocateScratch scratch(ndim());
    const std::int64_t cid = detail::locate_point(*rep_, pt.data(), scratch);
    if (cid < 0) {
      return std::nullopt;
    }
    return cid;
  }

  /// Get the work of locating one point, which sizes `locate_many`'s chunks.
  ///
  /// One item is one point. The root search is a binary search per axis over `cells + 1`
  /// breakpoints (the root grid's `locate_work`); then, at worst, every level costs a scan of its
  /// blocks (`ndim()` comparisons each) and about `4 * ndim()` operations to pick the child. An
  /// estimate of the algorithm's work, not a timing; the Python binding also uses it to decide
  /// whether to release the GIL.
  ///
  /// Returns:
  ///     pantr::parallel::WorkEstimate: Operations to locate one point.
  [[nodiscard]] pantr::parallel::WorkEstimate locate_work() const {
    const auto nd = static_cast<double>(ndim());
    double ops = rep_->root.locate_work().ops_per_item;
    for (std::size_t level = 0; level < rep_->blocks.size(); ++level) {
      ops += static_cast<double>(rep_->level_start[level + 1] - rep_->level_start[level]) * nd +
             4.0 * nd;
    }
    return pantr::parallel::WorkEstimate{ops};
  }

  /// Locate a batch of points, in parallel over the points.
  ///
  /// Args:
  ///     points (std::span<const double>): Row-major `(npts, ndim())` points; the Python wrapper
  ///         has checked their shape.
  ///     npts (std::size_t): Number of points.
  ///
  /// Returns:
  ///     NdArray<std::int64_t>: Shape `(npts,)`; the leaf of each point, -1 for a point outside
  ///     the grid, including one with a NaN or infinite coordinate.
  ///
  /// Raises:
  ///     value_error: `points.size() != npts * ndim()`.
  [[nodiscard]] NdArray<std::int64_t> locate_many(std::span<const double> points,
                                                  std::size_t npts) const {
    const std::size_t nd = ndim();
    if (points.size() != npts * nd) {
      throw value_error("locate_many: points holds " + std::to_string(points.size()) +
                        " values, but (npts, ndim) = (" + std::to_string(npts) + ", " +
                        std::to_string(nd) + ") needs " + std::to_string(npts * nd) + ".");
    }
    NdArray<std::int64_t> out({npts}, for_overwrite);
    const detail::HierarchicalGridRep& rep = *rep_;
    std::int64_t* const dst = out.data();
    const double* const src = points.data();
    pantr::parallel::parallel_for(npts, locate_work(),
                                  [&rep, dst, src, nd](std::size_t begin, std::size_t end) {
                                    detail::LocateScratch scratch(nd);
                                    for (std::size_t p = begin; p < end; ++p) {
                                      dst[p] = detail::locate_point(rep, src + p * nd, scratch);
                                    }
                                  });
    return out;
  }

  /// Get the cell across local facet `lfid` of `cid`, if any.
  ///
  /// Handles hanging interfaces across any level difference. When the neighbor is coarser, the
  /// active leaf covering the position, however many levels up, is returned. When it is finer, the
  /// first active leaf touching the face (first in depth-first order along the face) is returned;
  /// `hanging_neighbors` gives all of them.
  ///
  /// Args:
  ///     cid (std::int64_t): Cell identifier.
  ///     lfid (std::int64_t): Local facet identifier `2 * axis + side` in `[0, 2 * ndim())`.
  ///
  /// Returns:
  ///     std::optional<std::int64_t>: The neighboring cell id, or no value on a boundary facet.
  ///
  /// Raises:
  ///     index_error: `cid` or `lfid` is out of range.
  [[nodiscard]] std::optional<std::int64_t> neighbor_across_facet(std::int64_t cid,
                                                                  std::int64_t lfid) const {
    const std::vector<std::int64_t> all = facet_neighbors(cid, lfid, /*first_only=*/true);
    if (all.empty()) {
      return std::nullopt;
    }
    return all.front();
  }

  /// Get all active neighbors across facet `lfid` of `cid`.
  ///
  /// One neighbor for a conforming or coarser interface; for a finer one, every active leaf
  /// touching the face, depth first along the face (row-major over the non-normal axes at each
  /// level).
  ///
  /// Args:
  ///     cid (std::int64_t): Cell identifier.
  ///     lfid (std::int64_t): Local facet identifier in `[0, 2 * ndim())`.
  ///
  /// Returns:
  ///     std::vector<std::int64_t>: Neighboring cell ids; empty on a boundary facet.
  ///
  /// Raises:
  ///     index_error: `cid` or `lfid` is out of range.
  [[nodiscard]] std::vector<std::int64_t> hanging_neighbors(std::int64_t cid,
                                                            std::int64_t lfid) const {
    return facet_neighbors(cid, lfid, /*first_only=*/false);
  }

  /// Get every facet on the grid's outer boundary as `(cid, lfid)` rows.
  ///
  /// The outer boundary only, so level interfaces are excluded. Computed from the active blocks:
  /// whether a block face lies on the boundary is decided once per face, and the cells of such a
  /// face are emitted together.
  ///
  /// Returns:
  ///     NdArray<std::int64_t>: Shape `(n, 2)`, rows `(cid, lfid)` with `lfid = 2 * axis + side`,
  ///     sorted lexicographically.
  [[nodiscard]] NdArray<std::int64_t> boundary_facets() const {
    const std::size_t nd = ndim();
    std::vector<std::pair<std::int64_t, std::int64_t>> rows;
    for (std::size_t level = 0; level < rep_->blocks.size(); ++level) {
      std::vector<std::int64_t> cells(nd);
      for (std::size_t k = 0; k < nd; ++k) {
        cells[k] = cells_at(static_cast<std::int64_t>(level), k);
      }
      for (auto b = static_cast<std::size_t>(rep_->level_start[level]);
           b < static_cast<std::size_t>(rep_->level_start[level + 1]); ++b) {
        const std::int64_t* lo = rep_->block_lo.data() + b * nd;
        const std::int64_t* hi = rep_->block_hi.data() + b * nd;
        detail::Index shape(nd);
        detail::Index stride(nd, 1);
        for (std::size_t k = 0; k < nd; ++k) {
          shape[k] = hi[k] - lo[k];
        }
        for (std::size_t k = nd - 1; k > 0; --k) {
          stride[k - 1] = stride[k] * shape[k];
        }
        for (std::size_t axis = 0; axis < nd; ++axis) {
          for (std::int64_t side = 0; side < 2; ++side) {
            const bool touches = side == 0 ? lo[axis] == 0 : hi[axis] == cells[axis];
            if (!touches) {
              continue;
            }
            // Fix the normal axis at the block's boundary layer and walk the others row-major.
            detail::Index idx(nd, 0);
            idx[axis] = side == 0 ? 0 : shape[axis] - 1;
            while (true) {
              std::int64_t offset = 0;
              for (std::size_t k = 0; k < nd; ++k) {
                offset += idx[k] * stride[k];
              }
              rows.emplace_back(rep_->block_base[b] + offset,
                                2 * static_cast<std::int64_t>(axis) + side);
              bool advanced = false;
              for (std::size_t k = nd; k-- > 0;) {
                if (k == axis) {
                  continue;
                }
                if (++idx[k] < shape[k]) {
                  advanced = true;
                  break;
                }
                idx[k] = 0;
              }
              if (!advanced) {
                break;
              }
            }
          }
        }
      }
    }
    std::sort(rows.begin(), rows.end());
    NdArray<std::int64_t> out({rows.size(), 2}, for_overwrite);
    for (std::size_t i = 0; i < rows.size(); ++i) {
      out.data()[2 * i] = rows[i].first;
      out.data()[2 * i + 1] = rows[i].second;
    }
    return out;
  }

  /// Get the root-cell-aligned bounding-box sub-grid spanning `ids`.
  ///
  /// The window is the multi-index bounding box, in root-cell coordinates, of the root cells
  /// containing the requested leaves. The sub-grid's root is the matching slice of this grid's root
  /// breakpoints, it keeps the same factor, and its leaves are the per-level intersections of this
  /// grid's blocks with the window. Restricting one deep leaf therefore returns the whole leaf
  /// tiling of its root cell, with only the requested leaf flagged in `in_subset`.
  ///
  /// Args:
  ///     ids (std::span<const std::int64_t>): Flat cell identifiers to span; duplicates are
  ///         ignored. Each must be in `[0, num_cells())`.
  ///
  /// Returns:
  ///     HierarchicalGridRestriction: The windowed grid (version 1), the local-to-global cell map
  ///     and the requested-versus-fill mask.
  ///
  /// Raises:
  ///     value_error: `ids` is empty.
  ///     index_error: A cell id is out of range.
  ///     state_error: A windowed leaf has no counterpart in this grid (an invariant; unreachable).
  [[nodiscard]] HierarchicalGridRestriction restrict(std::span<const std::int64_t> ids) const;

  /// Get the refinement level of cell `cid`.
  ///
  /// Args:
  ///     cid (std::int64_t): Cell identifier.
  ///
  /// Returns:
  ///     std::int64_t: The level, 0 for an unrefined root cell.
  ///
  /// Raises:
  ///     index_error: `cid` is out of range.
  [[nodiscard]] std::int64_t cell_level(std::int64_t cid) const {
    check_cid(cid);
    detail::Index midx(ndim());
    return detail::decode_flat_id(*rep_, cid, midx.data());
  }

  /// Get the per-axis index of cell `cid` in its level's coordinates.
  ///
  /// Args:
  ///     cid (std::int64_t): Cell identifier.
  ///
  /// Returns:
  ///     std::vector<std::int64_t>: `ndim()` indices.
  ///
  /// Raises:
  ///     index_error: `cid` is out of range.
  [[nodiscard]] std::vector<std::int64_t> cell_multi_index(std::int64_t cid) const {
    check_cid(cid);
    detail::Index midx(ndim());
    static_cast<void>(detail::decode_flat_id(*rep_, cid, midx.data()));
    return midx;
  }

  /// Get the flat id of the active leaf `(level, midx)`, if it is one.
  ///
  /// Refining and coarsening reassign every flat id, so a caller tracking cells across a change
  /// keeps `(level, midx)` pairs, which are stable, and resolves them here once per change.
  ///
  /// Args:
  ///     level (std::int64_t): Hierarchy level; any value is accepted.
  ///     midx (std::span<const std::int64_t>): Per-axis index in level coordinates; any values.
  ///
  /// Returns:
  ///     std::optional<std::int64_t>: The flat id, or no value when the position is out of range,
  ///     not yet created, refined away, or `midx` has the wrong length.
  [[nodiscard]] std::optional<std::int64_t> cell_id(std::int64_t level,
                                                    std::span<const std::int64_t> midx) const {
    // A negative index lies below every block (block lower bounds are at least 0), so
    // `encode_midx` finds no leaf for it.
    return detail::encode_midx(*rep_, level, detail::Index(midx.begin(), midx.end()));
  }

  /// Report whether `(level, midx)` is an active leaf.
  ///
  /// Args:
  ///     level (std::int64_t): Hierarchy level; any value is accepted.
  ///     midx (std::span<const std::int64_t>): Per-axis index in level coordinates.
  ///
  /// Returns:
  ///     bool: Whether `cell_id(level, midx)` has a value.
  [[nodiscard]] bool is_active_leaf(std::int64_t level, std::span<const std::int64_t> midx) const {
    return cell_id(level, midx).has_value();
  }

  /// Get the per-axis cell count of the level-`level` grid.
  ///
  /// A pure formula: `level` need not be an existing level, and a value above `max_level()` gives
  /// the count of the hypothetical finer grid.
  ///
  /// Args:
  ///     level (std::int64_t): Hierarchy level, at least 0.
  ///
  /// Returns:
  ///     std::vector<std::int64_t>: `root.cells_per_axis()[k] * factor[k] ** level` per axis.
  ///
  /// Raises:
  ///     value_error: `level < 0`.
  ///     std::overflow_error: A count does not fit in `int64`.
  [[nodiscard]] std::vector<std::int64_t> level_cells_per_axis(std::int64_t level) const {
    if (level < 0) {
      throw value_error("level must be >= 0; got " + std::to_string(level) + ".");
    }
    std::vector<std::int64_t> counts(ndim());
    for (std::size_t k = 0; k < counts.size(); ++k) {
      const std::optional<std::int64_t> n =
          detail::try_cells_at_level(rep_->root, rep_->factor, k, level);
      if (!n.has_value()) {
        throw std::overflow_error(level_overflow_message(level, k));
      }
      counts[k] = *n;
    }
    return counts;
  }

  /// Get the active-leaf blocks at `level`.
  ///
  /// Args:
  ///     level (std::int64_t): Hierarchy level in `[0, max_level()]`.
  ///
  /// Returns:
  ///     const std::vector<detail::Block>&: The sorted, non-overlapping `(lo, hi)` rectangles, in
  ///     level coordinates, valid while this grid lives.
  ///
  /// Raises:
  ///     value_error: `level` is outside `[0, max_level()]`.
  [[nodiscard]] const std::vector<detail::Block>& active_blocks(std::int64_t level) const {
    check_level(level);
    return rep_->blocks[static_cast<std::size_t>(level)];
  }

  /// Get every level's blocks.
  ///
  /// Returns:
  ///     const std::vector<std::vector<detail::Block>>&: `blocks()[l]` is `active_blocks(l)`.
  [[nodiscard]] const std::vector<std::vector<detail::Block>>& blocks() const noexcept {
    return rep_->blocks;
  }

  /// Get a mask of the active-leaf cells at `level`.
  ///
  /// Args:
  ///     level (std::int64_t): Hierarchy level in `[0, max_level()]`.
  ///
  /// Returns:
  ///     NdArray<std::uint8_t>: Shape `level_cells_per_axis(level)`; 1 where `(level, midx)` is an
  ///     active leaf.
  ///
  /// Raises:
  ///     value_error: `level` is outside `[0, max_level()]`.
  [[nodiscard]] NdArray<std::uint8_t> active_leaf_mask(std::int64_t level) const {
    check_level(level);
    const detail::Index counts = level_cells_per_axis(level);
    NdArray<std::uint8_t> mask(std::vector<std::size_t>(counts.begin(), counts.end()));
    for (const detail::Block& block : rep_->blocks[static_cast<std::size_t>(level)]) {
      detail::fill_box(mask.flat(), counts, block.first, block.second, 1);
    }
    return mask;
  }

  /// Get a mask of the level-`level` refined subdomain.
  ///
  /// A level-`level` cell is in the subdomain iff no active leaf of a coarser level covers it. The
  /// mask is computed at level-`level` resolution by projecting each coarser block up to `level`.
  ///
  /// Args:
  ///     level (std::int64_t): Hierarchy level in `[0, max_level()]`.
  ///
  /// Returns:
  ///     NdArray<std::uint8_t>: Shape `level_cells_per_axis(level)`; 1 where the cell is in the
  ///     subdomain.
  ///
  /// Raises:
  ///     value_error: `level` is outside `[0, max_level()]`.
  [[nodiscard]] NdArray<std::uint8_t> subdomain_mask(std::int64_t level) const {
    check_level(level);
    const std::size_t nd = ndim();
    const detail::Index counts = level_cells_per_axis(level);
    NdArray<std::uint8_t> mask(std::vector<std::size_t>(counts.begin(), counts.end()));
    const std::span<std::uint8_t> cells = mask.flat();
    std::fill(cells.begin(), cells.end(), std::uint8_t{1});
    for (std::int64_t m = 0; m < level; ++m) {
      detail::Index scale(nd);
      for (std::size_t k = 0; k < nd; ++k) {
        scale[k] = detail::checked_pow(rep_->factor[k], level - m);
      }
      for (const detail::Block& block : rep_->blocks[static_cast<std::size_t>(m)]) {
        detail::Index lo(nd);
        detail::Index hi(nd);
        for (std::size_t k = 0; k < nd; ++k) {
          lo[k] = block.first[k] * scale[k];
          hi[k] = block.second[k] * scale[k];
        }
        detail::fill_box(mask.flat(), counts, lo, hi, 0);
      }
    }
    return mask;
  }

  /// Refine the active part of the rectangle `[lo, hi)` at `level` to `level + 1`.
  ///
  /// Union semantics: only the currently active part of the rectangle is refined, cells already
  /// deeper stay as they are, and a rectangle with no active cell changes nothing. That costs
  /// invertibility: promoting part of a box sends several grids to the same result, so no single
  /// `coarsened` undoes this in general. `coarsened(level, lo, hi)` reverses it only when every
  /// cell of the rectangle was an active leaf at `level` beforehand; to coarsen without risking
  /// refinement the caller did not mean to lose, name cells with `coarsened_cells`.
  ///
  /// All flat ids of the result differ from this grid's.
  ///
  /// Args:
  ///     level (std::int64_t): Level at which the region lives, in `[0, max_level()]`.
  ///     lo (std::span<const std::int64_t>): Per-axis start (inclusive) in level coordinates.
  ///     hi (std::span<const std::int64_t>): Per-axis end (exclusive).
  ///
  /// Returns:
  ///     HierarchicalGrid: The refined grid with version `version() + 1`; this grid itself (same
  ///     representation and version) when no active cell lies in the rectangle.
  ///
  /// Raises:
  ///     value_error: `level` is out of range, `lo` or `hi` has the wrong length, some `lo[k] >=
  ///         hi[k]`, or the rectangle leaves the level's index range.
  ///     std::overflow_error: The new level would have more cells along an axis than `int64`
  ///         holds, or the grid more cells in total.
  [[nodiscard]] HierarchicalGrid refined(std::int64_t level, std::span<const std::int64_t> lo,
                                         std::span<const std::int64_t> hi) const {
    if (!(0 <= level && level <= max_level())) {
      throw value_error("level must be in [0, " + std::to_string(max_level()) + "]; got " +
                        std::to_string(level) + ".");
    }
    const auto [lo_t, hi_t] = check_region(level, lo, hi);
    std::vector<detail::Block> kept;
    std::vector<detail::Block> overlaps;
    for (const detail::Block& block : rep_->blocks[static_cast<std::size_t>(level)]) {
      std::optional<detail::Block> inter =
          detail::rect_intersect(block.first, block.second, lo_t, hi_t);
      if (!inter.has_value()) {
        kept.push_back(block);
        continue;
      }
      for (detail::Block& slab :
           detail::peel(block.first, block.second, inter->first, inter->second)) {
        kept.push_back(std::move(slab));
      }
      overlaps.push_back(std::move(*inter));
    }
    if (overlaps.empty()) {
      return *this;
    }
    const std::size_t nd = ndim();
    if (level == max_level()) {
      for (std::size_t k = 0; k < nd; ++k) {
        if (!detail::try_cells_at_level(rep_->root, rep_->factor, k, level + 1).has_value()) {
          throw std::overflow_error("refining level " + std::to_string(level) +
                                    " would give level " + std::to_string(level + 1) +
                                    " more than 9223372036854775807 cells along axis " +
                                    std::to_string(k) + ".");
        }
      }
    }
    std::vector<std::vector<detail::Block>> new_blocks = rep_->blocks;
    const auto lv = static_cast<std::size_t>(level);
    new_blocks[lv] = detail::normalize_blocks(std::move(kept));
    if (new_blocks.size() <= lv + 1) {
      new_blocks.emplace_back();
    }
    std::vector<detail::Block> finer = std::move(new_blocks[lv + 1]);
    for (const detail::Block& inter : overlaps) {
      detail::Block child{detail::Index(nd), detail::Index(nd)};
      for (std::size_t k = 0; k < nd; ++k) {
        child.first[k] = inter.first[k] * rep_->factor[k];
        child.second[k] = inter.second[k] * rep_->factor[k];
      }
      finer.push_back(std::move(child));
    }
    new_blocks[lv + 1] = detail::normalize_blocks(std::move(finer));
    return HierarchicalGrid(
        detail::make_hier_rep(rep_->root, rep_->factor, std::move(new_blocks), version() + 1));
  }

  /// Refine a set of active cells by per-level bounding-box aggregation.
  ///
  /// Groups `ids` by level, takes the bounding box of the cells at each level, and applies
  /// `refined` once per level in increasing level order, so the result can promote a cell no id
  /// named (refining removes no cell, so that costs nothing).
  ///
  /// Args:
  ///     ids (std::span<const std::int64_t>): Flat cell ids to refine; repeats are ignored.
  ///
  /// Returns:
  ///     HierarchicalGrid: The refined grid, whose version is this one's plus the number of levels
  ///     that changed; this grid itself when `ids` is empty or nothing changed.
  ///
  /// Raises:
  ///     index_error: An id is out of range.
  ///     std::overflow_error: As `refined`.
  [[nodiscard]] HierarchicalGrid refined_cells(std::span<const std::int64_t> ids) const {
    std::map<std::int64_t, detail::Block> boxes;
    detail::Index midx(ndim());
    for (const std::int64_t cid : ids) {
      check_cid(cid);
      const std::int64_t level = detail::decode_flat_id(*rep_, cid, midx.data());
      const auto [it, inserted] = boxes.try_emplace(level);
      detail::Block& box = it->second;
      for (std::size_t k = 0; k < midx.size(); ++k) {
        if (inserted) {
          box.first.push_back(midx[k]);
          box.second.push_back(midx[k] + 1);
        } else {
          box.first[k] = (std::min)(box.first[k], midx[k]);
          box.second[k] = (std::max)(box.second[k], midx[k] + 1);
        }
      }
    }
    HierarchicalGrid current = *this;
    for (const auto& [level, box] : boxes) {
      current = current.refined(level, box.first, box.second);
    }
    return current;
  }

  /// Demote the rectangle `[lo, hi)` from level `level + 1` back to `level`.
  ///
  /// The region must be fully refined to exactly level `level + 1`: every child cell in
  /// `[lo * factor, hi * factor)` must be an active leaf at `level + 1`, none refined further and
  /// none still a leaf at `level`. The whole box is demoted, whereas `refined` promotes only the
  /// box's active part, so this inverts a refinement only when every cell of the box was an active
  /// leaf at `level` before it; otherwise it can remove children an earlier refinement created.
  /// Refining `(level, lo, hi)` always undoes a coarsening of it.
  ///
  /// All flat ids of the result differ from this grid's.
  ///
  /// Args:
  ///     level (std::int64_t): Level whose cells are reactivated, in `[0, max_level())`.
  ///     lo (std::span<const std::int64_t>): Per-axis start (inclusive) in level coordinates.
  ///     hi (std::span<const std::int64_t>): Per-axis end (exclusive).
  ///
  /// Returns:
  ///     HierarchicalGrid: The coarsened grid with version `version() + 1`.
  ///
  /// Raises:
  ///     value_error: `level` is out of range, `lo` or `hi` has the wrong length, some `lo[k] >=
  ///         hi[k]`, the rectangle leaves the level's index range, or the region is not fully
  ///         refined to exactly level `level + 1` (the message names the offending cells, up to
  ///         `detail::max_named_cells` per reason).
  [[nodiscard]] HierarchicalGrid coarsened(std::int64_t level, std::span<const std::int64_t> lo,
                                           std::span<const std::int64_t> hi) const {
    if (!(0 <= level && level < max_level())) {
      throw value_error("level must be in [0, " + std::to_string(max_level()) + "); got " +
                        std::to_string(level) + ".");
    }
    const auto [lo_t, hi_t] = check_region(level, lo, hi);
    const std::size_t nd = ndim();
    detail::Index child_lo(nd);
    detail::Index child_hi(nd);
    for (std::size_t k = 0; k < nd; ++k) {
      child_lo[k] = lo_t[k] * rep_->factor[k];
      child_hi[k] = hi_t[k] * rep_->factor[k];
    }
    // The children region must be fully tiled by active leaves at level + 1. Its size can exceed
    // int64 (a region of a deep level need not exist), in which case it cannot be covered.
    const std::optional<std::int64_t> child_size = detail::try_block_size(child_lo, child_hi);
    std::int64_t covered = 0;
    std::vector<detail::Block> new_finer;
    for (const detail::Block& block : rep_->blocks[static_cast<std::size_t>(level) + 1]) {
      const std::optional<detail::Block> inter =
          detail::rect_intersect(block.first, block.second, child_lo, child_hi);
      if (!inter.has_value()) {
        new_finer.push_back(block);
        continue;
      }
      covered += detail::block_size(inter->first, inter->second);
      for (detail::Block& slab :
           detail::peel(block.first, block.second, inter->first, inter->second)) {
        new_finer.push_back(std::move(slab));
      }
    }
    if (!child_size.has_value() || covered != *child_size) {
      throw value_error(refusal_message(level, lo_t, hi_t));
    }
    std::vector<std::vector<detail::Block>> new_blocks = rep_->blocks;
    const auto lv = static_cast<std::size_t>(level);
    new_blocks[lv + 1] = detail::normalize_blocks(std::move(new_finer));
    new_blocks[lv].emplace_back(lo_t, hi_t);
    new_blocks[lv] = detail::normalize_blocks(std::move(new_blocks[lv]));
    while (new_blocks.size() > 1 && new_blocks.back().empty()) {
      new_blocks.pop_back();
    }
    return HierarchicalGrid(
        detail::make_hier_rep(rep_->root, rep_->factor, std::move(new_blocks), version() + 1));
  }

  /// Demote every parent whose children are all named in `ids`.
  ///
  /// The route that destroys only what the caller named. A parent is reactivated, its children
  /// removed, only when every one of its `prod(factor)` children is both an active leaf and in
  /// `ids`. A parent only some of whose children are named, a parent with a child refined further
  /// and an id at level 0 are skipped without error, and no cell outside `ids` is ever removed.
  ///
  /// Parents are processed deepest level first, then lexicographically by index, which fixes the
  /// order of the block lists and so the flat ids. Coarsening does not cascade: a cell reborn by
  /// this call was not a leaf when the ids were chosen, so its own parent is never complete.
  ///
  /// Args:
  ///     ids (std::span<const std::int64_t>): Flat ids of active leaves to coarsen away; repeats
  ///         and level-0 ids are ignored.
  ///
  /// Returns:
  ///     HierarchicalGrid: The coarsened grid, whose version is this one's plus the number of
  ///     parents demoted; this grid itself when none was.
  ///
  /// Raises:
  ///     index_error: An id is out of range.
  [[nodiscard]] HierarchicalGrid coarsened_cells(std::span<const std::int64_t> ids) const {
    const std::size_t nd = ndim();
    std::set<std::pair<std::int64_t, detail::Index>> marked;
    detail::Index midx(nd);
    for (const std::int64_t cid : ids) {
      check_cid(cid);
      const std::int64_t level = detail::decode_flat_id(*rep_, cid, midx.data());
      marked.emplace(level, midx);
    }
    // Keyed (-level, index) so that iteration is deepest first, then lexicographic.
    std::set<std::pair<std::int64_t, detail::Index>> parents;
    for (const auto& [level, index] : marked) {
      if (level < 1) {
        continue;
      }
      detail::Index parent(nd);
      for (std::size_t k = 0; k < nd; ++k) {
        parent[k] = index[k] / rep_->factor[k];
      }
      parents.emplace(-(level - 1), std::move(parent));
    }
    HierarchicalGrid current = *this;
    for (const auto& [neg_level, parent] : parents) {
      const std::int64_t parent_level = -neg_level;
      // Every child must be named and, in the grid as coarsened so far, still a leaf. The second
      // test cannot fail once a child is named, since only this parent's own coarsening could
      // remove it; it is kept because it is the documented rule, not a property of this loop.
      detail::Index child(nd);
      for (std::size_t k = 0; k < nd; ++k) {
        child[k] = parent[k] * rep_->factor[k];
      }
      bool complete = true;
      while (true) {
        if (!marked.contains({parent_level + 1, child}) ||
            !current.is_active_leaf(parent_level + 1, child)) {
          complete = false;
          break;
        }
        bool advanced = false;
        for (std::size_t k = nd; k-- > 0;) {
          if (++child[k] < (parent[k] + 1) * rep_->factor[k]) {
            advanced = true;
            break;
          }
          child[k] = parent[k] * rep_->factor[k];
        }
        if (!advanced) {
          break;
        }
      }
      if (complete) {
        detail::Index parent_hi = parent;
        for (std::int64_t& p : parent_hi) {
          p += 1;
        }
        current = current.coarsened(parent_level, parent, parent_hi);
      }
    }
    return current;
  }

  /// Get the work of `collect_cell_bounds` per cell, which sizes its chunks.
  ///
  /// One item is one cell. Per axis the bounds cost an integer division and remainder, two
  /// breakpoint loads, a subtraction, a division, a multiplication and two additions, about six
  /// operations per axis, as the cost of a block is `cells * 6 * ndim`; a chunk starts with two
  /// binary searches, which a chunk of this size amortizes. The loop runs over cells rather than
  /// blocks so that one large block still splits. An estimate of the algorithm's work, not a
  /// timing; the Python binding also uses it to decide whether to release the GIL.
  ///
  /// Returns:
  ///     pantr::parallel::WorkEstimate: Operations to evaluate the bounds of one cell.
  [[nodiscard]] pantr::parallel::WorkEstimate collect_work() const {
    return pantr::parallel::WorkEstimate{6.0 * static_cast<double>(ndim())};
  }

  /// Materialize per-cell `(lo, hi)` in flat-id order, in parallel over the cells.
  ///
  /// Each cell's bounds are the same expression `cell_bounds` evaluates, so the arrays equal the
  /// per-cell results exactly, whatever the number of threads.
  ///
  /// Returns:
  ///     std::pair<NdArray<double>, NdArray<double>>: `(cell_lo, cell_hi)`, each of shape
  ///     `(num_cells(), ndim())`.
  [[nodiscard]] std::pair<NdArray<double>, NdArray<double>> collect_cell_bounds() const {
    const std::size_t nd = ndim();
    const auto n = static_cast<std::size_t>(rep_->num_cells);
    NdArray<double> cell_lo({n, nd}, for_overwrite);
    NdArray<double> cell_hi({n, nd}, for_overwrite);
    // Each cell writes `2 * ndim` doubles.
    const detail::HierarchicalGridRep& rep = *rep_;
    double* const dst_lo = cell_lo.data();
    double* const dst_hi = cell_hi.data();
    pantr::parallel::parallel_for(n, collect_work(), 2 * nd * sizeof(double),
                                  [&rep, dst_lo, dst_hi](std::size_t begin, std::size_t end) {
                                    detail::collect_bounds_range(
                                        rep, static_cast<std::int64_t>(begin),
                                        static_cast<std::int64_t>(end), dst_lo, dst_hi);
                                  });
    return {std::move(cell_lo), std::move(cell_hi)};
  }

  /// Get deduplicated leaf-cell vertices and their connectivity.
  ///
  /// One vertex per distinct corner of the active leaves, shared exactly between the cells that
  /// meet there, including across levels (a hanging vertex appears once). Corners are identified on
  /// the finest-level integer lattice, where a level-`l` cell at `midx` contributes
  /// `(midx[k] + bit[k]) * factor[k] ** (max_level - l)`: equal corners are equal integers, and
  /// each distinct node is mapped to coordinates by one formula, so coincident corners are bitwise
  /// identical and no tolerance is involved.
  ///
  /// Corner `c` of a row takes the upper bound on axis `k` iff bit `k` of `c` is set, axis 0 the
  /// least significant bit (the basix and dolfinx convention, unlike the row-major cell ids).
  ///
  /// Note:
  ///     `points[conn[cid]]` reproduces the corners of `cell_bounds` to within
  ///     `2 gamma_2 |x| + 2 gamma_4 |x - b|`, where `b` is the lower breakpoint of the root cell
  ///     holding the corner and `gamma_m = m u / (1 - m u)` with `u = eps / 2`. Both sides build
  ///     `b + s * w / factor ** level` from different expressions; forming `w`, the division, the
  ///     integer-scaled multiply and the addition are four correctly rounded operations each,
  ///     with a fifth on the `cell_bounds` side of a `hi` corner, which it writes as
  ///     `lo + size`.
  ///
  ///     The second term is what the bound is about. `|x - b|` is bounded by the root cell's
  ///     width and not by `|x|`, so only where every coordinate dominates its own offset (a
  ///     domain with `b >= 0` is the usual reason) does this collapse to a multiple of
  ///     `eps |x|`. A corner small against the width of the root cell holding it is a
  ///     counterexample rather than a corner case.
  ///
  ///     It is bitwise whenever every intermediate is exactly representable: a dyadic `factor`
  ///     on dyadic root breakpoints, which covers the usual unit-domain power-of-two case.
  ///     Bitwise agreement in general is not merely unimplemented, it is unattainable:
  ///     `cell_bounds` scales a root cell's width by `factor ** level`, so for a corner shared
  ///     between two levels it evaluates a different expression on each side and can return two
  ///     floats one unit in the last place apart, and no single deduplicated vertex can equal
  ///     both.
  ///
  ///     Stated hypothesis: coordinates are normal, not subnormal. The bound is a product of
  ///     relative factors with the magnitudes, so below roughly `2e-308` both terms underflow to
  ///     exactly zero while the difference they bound is still a nonzero subnormal: the
  ///     expression stops implementing the inequality, though the inequality itself still holds
  ///     in the reals. Nothing excludes that domain, so this is a hypothesis and not a
  ///     guarantee.
  ///
  /// Warning:
  ///     Materializes `O(num_cells * 2 ** ndim * ndim)` integers before deduplication, so a
  ///     large 3D hierarchy costs several times `conn` in peak memory.
  ///
  /// Returns:
  ///     std::pair<NdArray<double>, NdArray<std::int64_t>>: `(points, conn)`. `points` has shape
  ///     `(num_vertices, ndim())`, sorted lexicographically by lattice coordinate; `conn` has shape
  ///     `(num_cells(), 2 ** ndim())` and indexes `points`, one row per leaf in flat-id order.
  ///
  /// Raises:
  ///     capacity_error: `num_cells() * 2 ** ndim() * ndim()` overflows `std::size_t`.
  [[nodiscard]] std::pair<NdArray<double>, NdArray<std::int64_t>> export_cells() const {
    const std::size_t nd = ndim();
    const std::size_t nc = std::size_t{1} << nd;
    const auto n = static_cast<std::size_t>(rep_->num_cells);
    const std::int64_t top = max_level();
    // The two sizes below are products of runtime counts: refuse a wrapped one before allocating.
    const std::array<std::size_t, 2> corner_dims{n, nc};
    const std::array<std::size_t, 3> lattice_dims{n, nc, nd};
    const std::size_t total = pantr::checked_product(corner_dims);
    const std::size_t lattice_size = pantr::checked_product(lattice_dims);
    // lattice[(cell * nc + corner) * nd + k]: corner `corner` of cell `cell` on axis `k`.
    std::vector<std::int64_t> lattice(lattice_size);
    detail::Index scale(nd);
    detail::Index midx(nd);
    for (std::size_t b = 0; b < rep_->block_base.size(); ++b) {
      const std::int64_t level = detail::level_of_block(*rep_, b);
      const std::int64_t* lo = rep_->block_lo.data() + b * nd;
      const std::int64_t* hi = rep_->block_hi.data() + b * nd;
      for (std::size_t k = 0; k < nd; ++k) {
        scale[k] = detail::checked_pow(rep_->factor[k], top - level);
        midx[k] = lo[k];
      }
      const std::int64_t count =
          detail::block_size(detail::Index(lo, lo + nd), detail::Index(hi, hi + nd));
      for (std::int64_t c = 0; c < count; ++c) {
        const auto cell = static_cast<std::size_t>(rep_->block_base[b] + c);
        for (std::size_t corner = 0; corner < nc; ++corner) {
          for (std::size_t k = 0; k < nd; ++k) {
            const auto bit = static_cast<std::int64_t>((corner >> k) & 1U);
            lattice[(cell * nc + corner) * nd + k] = (midx[k] + bit) * scale[k];
          }
        }
        for (std::size_t k = nd; k-- > 0;) {
          if (++midx[k] < hi[k]) {
            break;
          }
          midx[k] = lo[k];
        }
      }
    }
    // The distinct lattice rows in lexicographic order, with each row's rank among them.
    std::vector<std::size_t> order(total);
    std::iota(order.begin(), order.end(), std::size_t{0});
    const auto row_less = [&lattice, nd](std::size_t a, std::size_t b) {
      return std::lexicographical_compare(
          lattice.begin() + static_cast<std::ptrdiff_t>(a * nd),
          lattice.begin() + static_cast<std::ptrdiff_t>((a + 1) * nd),
          lattice.begin() + static_cast<std::ptrdiff_t>(b * nd),
          lattice.begin() + static_cast<std::ptrdiff_t>((b + 1) * nd));
    };
    std::sort(order.begin(), order.end(), row_less);
    NdArray<std::int64_t> conn({n, nc}, for_overwrite);
    std::vector<std::size_t> node_rows;
    for (std::size_t i = 0; i < total; ++i) {
      if (i == 0 || row_less(order[i - 1], order[i])) {
        node_rows.push_back(order[i]);
      }
      conn.data()[order[i]] = static_cast<std::int64_t>(node_rows.size()) - 1;
    }
    NdArray<double> points({node_rows.size(), nd}, for_overwrite);
    for (std::size_t r = 0; r < node_rows.size(); ++r) {
      for (std::size_t k = 0; k < nd; ++k) {
        points.data()[r * nd + k] = lattice_coordinate(k, lattice[node_rows[r] * nd + k], top);
      }
    }
    return {std::move(points), std::move(conn)};
  }

  /// Get the lazy block holding the spatial index (for `GridBase::cell_bvh`).
  ///
  /// Returns:
  ///     const pantr::detail::Lazy<BVH>&: The block, valid while the grid lives.
  [[nodiscard]] const pantr::detail::Lazy<BVH>& bvh_lazy() const noexcept { return rep_->bvh; }

  /// Get the shared representation, for an owner that must keep the data alive.
  ///
  /// Returns:
  ///     const std::shared_ptr<const detail::HierarchicalGridRep>&: The representation.
  [[nodiscard]] const std::shared_ptr<const detail::HierarchicalGridRep>& rep() const noexcept {
    return rep_;
  }

  /// Report whether two values share one representation.
  ///
  /// Args:
  ///     other (const HierarchicalGrid&): The other grid.
  ///
  /// Returns:
  ///     bool: Whether the representations are the same object.
  [[nodiscard]] bool same_rep(const HierarchicalGrid& other) const noexcept {
    return rep_ == other.rep_;
  }

 private:
  explicit HierarchicalGrid(std::shared_ptr<const detail::HierarchicalGridRep> rep)
      : rep_(std::move(rep)) {}

  /// Check a subdivision factor against the grid dimension.
  ///
  /// Args:
  ///     nd (std::size_t): The dimension.
  ///     factor (const std::vector<std::int64_t>&): The per-axis factors.
  ///
  /// Raises:
  ///     value_error: `factor` does not have `nd` entries, or an entry is below 1.
  static void check_factor(std::size_t nd, const std::vector<std::int64_t>& factor) {
    if (factor.size() != nd) {
      throw value_error("factor must be a scalar or length-" + std::to_string(nd) +
                        " sequence; got length " + std::to_string(factor.size()) + ".");
    }
    if (std::any_of(factor.begin(), factor.end(), [](std::int64_t f) { return f < 1; })) {
      throw value_error("every factor entry must be >= 1 (1 = no subdivision on that axis); got " +
                        detail::py_repr_ints(factor) + ".");
    }
  }

  /// Build the representation of an unrefined grid: one level holding the whole root as a block.
  ///
  /// Args:
  ///     root (TensorProductGrid): The level-0 grid.
  ///     factor (std::vector<std::int64_t>): The per-axis subdivision factors.
  ///
  /// Returns:
  ///     std::shared_ptr<const detail::HierarchicalGridRep>: The representation, version 1.
  ///
  /// Raises:
  ///     value_error: `factor` is invalid (see `check_factor`).
  static std::shared_ptr<const detail::HierarchicalGridRep> make_initial(
      TensorProductGrid root, std::vector<std::int64_t> factor) {
    check_factor(root.ndim(), factor);
    const std::span<const std::int64_t> cells = root.cells_per_axis();
    std::vector<std::vector<detail::Block>> blocks(1);
    blocks[0].emplace_back(detail::Index(root.ndim(), 0),
                           detail::Index(cells.begin(), cells.end()));
    return detail::make_hier_rep(std::move(root), std::move(factor), std::move(blocks), 1);
  }

  /// Build the message for a level whose per-axis cell count does not fit `int64`.
  ///
  /// Args:
  ///     level (std::int64_t): The level that would be created.
  ///     axis (std::size_t): The axis whose count overflows.
  ///
  /// Returns:
  ///     std::string: The message.
  static std::string level_overflow_message(std::int64_t level, std::size_t axis) {
    return "level " + std::to_string(level) +
           " would have more than 9223372036854775807 cells along axis " + std::to_string(axis) +
           ".";
  }

  /// Get the cells along `axis` at an existing `level`.
  ///
  /// Args:
  ///     level (std::int64_t): An existing level.
  ///     axis (std::size_t): The axis, below `ndim()`.
  ///
  /// Returns:
  ///     std::int64_t: `root.cells_per_axis()[axis] * factor[axis] ** level`; the count of an
  ///     existing level fits `int64`, so this does not overflow.
  [[nodiscard]] std::int64_t cells_at(std::int64_t level, std::size_t axis) const {
    return detail::checked_mul(rep_->root.cells_per_axis()[axis],
                               detail::checked_pow(rep_->factor[axis], level));
  }

  /// Check that `level` is an existing level.
  ///
  /// Args:
  ///     level (std::int64_t): The level to check.
  ///
  /// Raises:
  ///     value_error: `level` is outside `[0, max_level()]`.
  void check_level(std::int64_t level) const {
    if (!(0 <= level && level <= max_level())) {
      throw value_error("level must be in [0, " + std::to_string(max_level()) + "]; got " +
                        std::to_string(level) + ".");
    }
  }

  /// Validate a region at an existing level and return it as two index vectors.
  ///
  /// Args:
  ///     level (std::int64_t): An existing level.
  ///     lo (std::span<const std::int64_t>): Inclusive lower corner.
  ///     hi (std::span<const std::int64_t>): Exclusive upper corner.
  ///
  /// Returns:
  ///     std::pair<detail::Index, detail::Index>: `(lo, hi)` as vectors.
  ///
  /// Raises:
  ///     value_error: A length differs from `ndim()`, `lo >= hi` on some axis, or the region
  ///         leaves `[0, cells)` of the level.
  [[nodiscard]] std::pair<detail::Index, detail::Index> check_region(
      std::int64_t level, std::span<const std::int64_t> lo,
      std::span<const std::int64_t> hi) const {
    const std::size_t nd = ndim();
    if (lo.size() != nd || hi.size() != nd) {
      throw value_error("lo and hi must have length " + std::to_string(nd) + "; got " +
                        std::to_string(lo.size()) + " and " + std::to_string(hi.size()) + ".");
    }
    for (std::size_t k = 0; k < nd; ++k) {
      if (lo[k] >= hi[k]) {
        throw value_error("lo must be strictly less than hi in every dimension; got lo=" +
                          detail::py_repr_ints(lo) + ", hi=" + detail::py_repr_ints(hi) + ".");
      }
    }
    for (std::size_t k = 0; k < nd; ++k) {
      const std::int64_t n_k = cells_at(level, k);
      if (lo[k] < 0 || hi[k] > n_k) {
        throw value_error("[lo, hi) out of bounds at level " + std::to_string(level) + ": axis " +
                          std::to_string(k) + " needs [0, " + std::to_string(n_k) + "), got [" +
                          std::to_string(lo[k]) + ", " + std::to_string(hi[k]) + ").");
      }
    }
    return {detail::Index(lo.begin(), lo.end()), detail::Index(hi.begin(), hi.end())};
  }

  /// Build the message of a refused coarsening, naming the offending cells when the region is
  /// small.
  ///
  /// Args:
  ///     level (std::int64_t): The level being coarsened.
  ///     lo (const detail::Index&): Inclusive lower corner of the region.
  ///     hi (const detail::Index&): Exclusive upper corner of the region.
  ///
  /// Returns:
  ///     std::string: The message.
  [[nodiscard]] std::string refusal_message(std::int64_t level, const detail::Index& lo,
                                            const detail::Index& hi) const {
    const std::optional<std::int64_t> region = detail::try_block_size(lo, hi);
    std::string extra;
    if (!region.has_value()) {
      // Python's unbounded integers print the exact count; one beyond int64 prints as a bound.
      extra = " The region spans more than 9223372036854775807 cells, too many to name.";
    } else if (*region > detail::max_diagnosed_cells) {
      extra = " The region spans " + std::to_string(*region) + " cells, too many to name.";
    } else {
      const std::string obstacles = detail::coarsen_obstacles(*rep_, level, lo, hi);
      if (!obstacles.empty()) {
        extra =
            " Offending cells in level-" + std::to_string(level) + " indices: " + obstacles + ".";
      }
    }
    return "cannot coarsen [" + detail::py_repr_ints(lo) + ", " + detail::py_repr_ints(hi) +
           ") at level " + std::to_string(level) +
           ": the region is not fully refined to exactly level " + std::to_string(level + 1) + "." +
           extra;
  }

  /// Get the neighbors across a facet: all of them, or just the first.
  ///
  /// Args:
  ///     cid (std::int64_t): Flat cell id.
  ///     lfid (std::int64_t): Local facet id.
  ///     first_only (bool): Keep only the first neighbor of a finer side.
  ///
  /// Returns:
  ///     std::vector<std::int64_t>: The neighbor ids; empty on the outer boundary.
  ///
  /// Raises:
  ///     index_error: `cid` or `lfid` is out of range.
  [[nodiscard]] std::vector<std::int64_t> facet_neighbors(std::int64_t cid, std::int64_t lfid,
                                                          bool first_only) const {
    check_lfid(cid, lfid);
    const auto axis = static_cast<std::size_t>(lfid / 2);
    const std::int64_t side = lfid % 2;
    detail::Index nbr(ndim());
    const std::int64_t level = detail::decode_flat_id(*rep_, cid, nbr.data());
    const std::int64_t new_ik = nbr[axis] + (side == 0 ? -1 : 1);
    if (new_ik < 0 || new_ik >= cells_at(level, axis)) {
      return {};  // The grid's outer boundary.
    }
    nbr[axis] = new_ik;
    // Same level: a conforming neighbor.
    if (const std::optional<std::int64_t> same = detail::encode_midx(*rep_, level, nbr)) {
      return {*same};
    }
    // Coarser: the leaf covering the position, however many levels up.
    if (const std::optional<std::int64_t> up = detail::nearest_active_ancestor(*rep_, level, nbr)) {
      return {*up};
    }
    // Finer: the leaves touching the shared face, however many levels down.
    const std::int64_t face_j = side == 0 ? rep_->factor[axis] - 1 : 0;
    std::vector<std::int64_t> below;
    detail::active_face_descendants(*rep_, level, nbr, detail::FaceTarget{axis, face_j}, below);
    if (first_only && below.size() > 1) {
      below.resize(1);
    }
    return below;
  }

  /// Map a finest-level lattice coordinate along `axis` to a parametric coordinate.
  ///
  /// The lattice has `factor ** top` steps per root cell, so every leaf corner of every level is a
  /// node. The far edge takes its breakpoint exactly. Each product is its own statement, which
  /// stops contraction under the project's `-ffp-contract=on` (a statement boundary); it would not
  /// under `-ffp-contract=fast`.
  ///
  /// Args:
  ///     axis (std::size_t): The axis.
  ///     coord (std::int64_t): The lattice coordinate, in `[0, steps * cells]`.
  ///     top (std::int64_t): The finest level, `max_level()`.
  ///
  /// Returns:
  ///     double: The coordinate, `bp + offset * (width / steps)` from the root cell's lower
  ///     breakpoint, or the last breakpoint exactly at the far edge.
  [[nodiscard]] double lattice_coordinate(std::size_t axis, std::int64_t coord,
                                          std::int64_t top) const {
    const std::vector<double>& bp = rep_->root.breakpoints()[axis];
    const std::int64_t num_root = rep_->root.cells_per_axis()[axis];
    const std::int64_t steps = detail::checked_pow(rep_->factor[axis], top);
    if (coord == detail::checked_mul(steps, num_root)) {
      return bp[static_cast<std::size_t>(num_root)];
    }
    // coord < steps * num_root here (the far edge returned above), so root_cell < num_root.
    const std::int64_t root_cell = coord / steps;
    const std::int64_t offset = coord - root_cell * steps;
    const double width =
        bp[static_cast<std::size_t>(root_cell) + 1] - bp[static_cast<std::size_t>(root_cell)];
    const double unit = width / static_cast<double>(steps);
    const double step = static_cast<double>(offset) * unit;
    return bp[static_cast<std::size_t>(root_cell)] + step;
  }

  std::shared_ptr<const detail::HierarchicalGridRep> rep_;
};

/// The result of `HierarchicalGrid::restrict`: a windowed sub-grid plus index maps.
///
/// Attributes:
///     grid (HierarchicalGrid): The window's leaves as a grid of its own (version 1).
///     local_to_global (std::vector<std::int64_t>): For each sub-grid cell id, its id in the
///         original grid.
///     in_subset (std::vector<std::uint8_t>): For each sub-grid cell, 1 if it was among the
///         requested ids, 0 if it is bounding-box fill.
struct HierarchicalGridRestriction {
  HierarchicalGrid grid;
  std::vector<std::int64_t> local_to_global;
  std::vector<std::uint8_t> in_subset;
};

inline HierarchicalGridRestriction HierarchicalGrid::restrict(
    std::span<const std::int64_t> ids) const {
  if (ids.empty()) {
    throw value_error("restrict: cell_ids must be non-empty.");
  }
  const auto [lo_it, hi_it] = std::minmax_element(ids.begin(), ids.end());
  if (*lo_it < 0 || *hi_it >= rep_->num_cells) {
    throw index_error("restrict: cell id out of range [0, " + std::to_string(rep_->num_cells) +
                      "); got [" + std::to_string(*lo_it) + ", " + std::to_string(*hi_it) + "].");
  }
  const std::size_t nd = ndim();
  std::vector<std::int64_t> sorted_ids(ids.begin(), ids.end());
  std::sort(sorted_ids.begin(), sorted_ids.end());

  // The root-cell bounding box over the requested leaves.
  detail::Index r_lo(rep_->root.cells_per_axis().begin(), rep_->root.cells_per_axis().end());
  detail::Index r_hi(nd, 0);
  detail::Index midx(nd);
  for (const std::int64_t cid : sorted_ids) {
    const std::int64_t level = detail::decode_flat_id(*rep_, cid, midx.data());
    for (std::size_t k = 0; k < nd; ++k) {
      const std::int64_t root_ik = midx[k] / detail::checked_pow(rep_->factor[k], level);
      r_lo[k] = (std::min)(r_lo[k], root_ik);
      r_hi[k] = (std::max)(r_hi[k], root_ik + 1);
    }
  }

  // The sub-root is a pure slice of the root breakpoints, never re-clamped.
  std::vector<std::vector<double>> sub_breakpoints(nd);
  for (std::size_t k = 0; k < nd; ++k) {
    const std::vector<double>& bp = rep_->root.breakpoints()[k];
    sub_breakpoints[k].assign(bp.begin() + static_cast<std::ptrdiff_t>(r_lo[k]),
                              bp.begin() + static_cast<std::ptrdiff_t>(r_hi[k] + 1));
  }
  TensorProductGrid sub_root(std::move(sub_breakpoints));

  // Per level: the blocks intersected with the window, translated to sub coordinates.
  std::vector<std::vector<detail::Block>> sub_blocks(rep_->blocks.size());
  for (std::size_t level = 0; level < rep_->blocks.size(); ++level) {
    detail::Index w_lo(nd);
    detail::Index w_hi(nd);
    for (std::size_t k = 0; k < nd; ++k) {
      const std::int64_t scale =
          detail::checked_pow(rep_->factor[k], static_cast<std::int64_t>(level));
      w_lo[k] = r_lo[k] * scale;
      w_hi[k] = r_hi[k] * scale;
    }
    for (const detail::Block& block : rep_->blocks[level]) {
      const std::optional<detail::Block> inter =
          detail::rect_intersect(block.first, block.second, w_lo, w_hi);
      if (!inter.has_value()) {
        continue;
      }
      detail::Block local{detail::Index(nd), detail::Index(nd)};
      for (std::size_t k = 0; k < nd; ++k) {
        local.first[k] = inter->first[k] - w_lo[k];
        local.second[k] = inter->second[k] - w_lo[k];
      }
      sub_blocks[level].push_back(std::move(local));
    }
  }
  HierarchicalGrid sub = from_blocks(std::move(sub_root), rep_->factor, std::move(sub_blocks), 1);

  // The local-to-global map: each sub leaf, translated back to global coordinates. The sub grid
  // numbers its cells block by block, row-major, so the blocks are walked in that order.
  const detail::HierarchicalGridRep& srep = *sub.rep_;
  std::vector<std::int64_t> local_to_global(static_cast<std::size_t>(srep.num_cells));
  detail::Index g_midx(nd);
  detail::Index s_midx(nd);
  for (std::size_t b = 0; b < srep.block_base.size(); ++b) {
    const std::int64_t level = detail::level_of_block(srep, b);
    const std::int64_t* lo = srep.block_lo.data() + b * nd;
    const std::int64_t* hi = srep.block_hi.data() + b * nd;
    detail::Index offset(nd);
    for (std::size_t k = 0; k < nd; ++k) {
      offset[k] = r_lo[k] * detail::checked_pow(rep_->factor[k], level);
      s_midx[k] = lo[k];
    }
    const std::int64_t count =
        detail::block_size(detail::Index(lo, lo + nd), detail::Index(hi, hi + nd));
    for (std::int64_t c = 0; c < count; ++c) {
      for (std::size_t k = 0; k < nd; ++k) {
        g_midx[k] = s_midx[k] + offset[k];
      }
      const std::int64_t g_cid = detail::block_of_midx(*rep_, level, g_midx.data());
      if (g_cid < 0) {
        throw state_error(
            "restrict: a windowed leaf has no counterpart in the grid (internal invariant).");
      }
      local_to_global[static_cast<std::size_t>(srep.block_base[b] + c)] = g_cid;
      for (std::size_t k = nd; k-- > 0;) {
        if (++s_midx[k] < hi[k]) {
          break;
        }
        s_midx[k] = lo[k];
      }
    }
  }
  std::vector<std::uint8_t> in_subset(local_to_global.size(), 0);
  for (std::size_t i = 0; i < local_to_global.size(); ++i) {
    in_subset[i] =
        std::binary_search(sorted_ids.begin(), sorted_ids.end(), local_to_global[i]) ? 1 : 0;
  }
  return HierarchicalGridRestriction{std::move(sub), std::move(local_to_global),
                                     std::move(in_subset)};
}

}  // namespace pantr::grid
