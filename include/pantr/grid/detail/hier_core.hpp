/// @file hier_core.hpp
/// @brief The representation of a `HierarchicalGrid` and the kernels over it: flat-id encoding and
/// decoding, cell bounds, point location, facet adjacency and the coarsening diagnostics.
///
/// The active cells of a hierarchical grid are rectangular blocks, one sorted list per level. For
/// the kernels the lists are packed into flat arrays in flat-cell-id order (level by level, block
/// by block, row-major within a block), so a flat id is decoded by two binary searches.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/lazy.hpp"
#include "pantr/grid/bvh.hpp"
#include "pantr/grid/detail/hier_blocks.hpp"
#include "pantr/grid/detail/py_text.hpp"
#include "pantr/grid/tensor_product_grid.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pantr::grid::detail {

/// The shared representation of a `HierarchicalGrid`.
///
/// Attributes:
///     root (TensorProductGrid): The level-0 grid.
///     factor (std::vector<std::int64_t>): Per-axis subdivision factor, each at least 1.
///     blocks (std::vector<std::vector<Block>>): `blocks[l]` is the sorted list of non-overlapping
///         active blocks at level `l`, in level-`l` coordinates.
///     block_lo (std::vector<std::int64_t>): Packed block lower bounds, row-major
///         `(n_blocks, ndim)`, level by level in flat-id order.
///     block_hi (std::vector<std::int64_t>): Packed block upper bounds, same layout.
///     block_base (std::vector<std::int64_t>): Flat id of each block's first cell.
///     level_start (std::vector<std::int64_t>): Block index range of each level, one entry more
///     than
///         there are levels.
///     level_base (std::vector<std::int64_t>): Flat-id base of each level, one entry more than
///     there
///         are levels (the last is `num_cells`).
///     num_cells (std::int64_t): Total active cells.
///     version (std::int64_t): Structural-change counter, at least 1.
///     bvh (pantr::detail::Lazy<BVH>): The spatial index over the cells, built on first use.
struct HierarchicalGridRep {
  explicit HierarchicalGridRep(TensorProductGrid r) : root(std::move(r)) {}

  TensorProductGrid root;
  std::vector<std::int64_t> factor;
  std::vector<std::vector<Block>> blocks;
  std::vector<std::int64_t> block_lo;
  std::vector<std::int64_t> block_hi;
  std::vector<std::int64_t> block_base;
  std::vector<std::int64_t> level_start;
  std::vector<std::int64_t> level_base;
  std::int64_t num_cells = 0;
  std::int64_t version = 1;
  pantr::detail::Lazy<BVH> bvh;
};

/// Get the number of cells along one axis at a level, or no value if it exceeds `int64`.
///
/// Args:
///     root (const TensorProductGrid&): The level-0 grid.
///     factor (std::span<const std::int64_t>): Per-axis subdivision factor.
///     axis (std::size_t): The axis.
///     level (std::int64_t): Hierarchy level, at least 0.
///
/// Returns:
///     std::optional<std::int64_t>: `root.cells_per_axis()[axis] * factor[axis] ** level`, or no
///     value when it does not fit in `int64`.
[[nodiscard]] inline std::optional<std::int64_t> try_cells_at_level(
    const TensorProductGrid& root, std::span<const std::int64_t> factor, std::size_t axis,
    std::int64_t level) noexcept {
  const std::optional<std::int64_t> scale = try_pow(factor[axis], level);
  if (!scale.has_value()) {
    return std::nullopt;
  }
  return try_mul(root.cells_per_axis()[axis], *scale);
}

/// Assemble a representation from per-level block lists: flat-id bases, the cell count, the packed
/// arrays.
///
/// Note:
///     The blocks are taken as they are: sorted, non-empty, non-overlapping, inside their level's
///     index range, and together tiling the root consistently with `factor`. The callers
///     establish this.
///
/// Args:
///     root (TensorProductGrid): The level-0 grid.
///     factor (std::vector<std::int64_t>): Per-axis subdivision factor.
///     blocks (std::vector<std::vector<Block>>): The active blocks of every level (at least one).
///     version (std::int64_t): The version to record.
///
/// Returns:
///     std::shared_ptr<const HierarchicalGridRep>: The representation.
///
/// Raises:
///     std::overflow_error: The number of cells does not fit in `int64`.
[[nodiscard]] inline std::shared_ptr<const HierarchicalGridRep> make_hier_rep(
    TensorProductGrid root, std::vector<std::int64_t> factor,
    std::vector<std::vector<Block>> blocks, std::int64_t version) {
  const std::size_t nd = root.ndim();
  auto rep = std::make_shared<HierarchicalGridRep>(std::move(root));
  rep->factor = std::move(factor);
  rep->version = version;
  const std::size_t n_levels = blocks.size();
  std::size_t n_blocks = 0;
  for (const std::vector<Block>& level_blocks : blocks) {
    n_blocks += level_blocks.size();
  }
  rep->block_lo.reserve(n_blocks * nd);
  rep->block_hi.reserve(n_blocks * nd);
  rep->block_base.reserve(n_blocks);
  rep->level_start.reserve(n_levels + 1);
  rep->level_base.reserve(n_levels + 1);
  std::int64_t base = 0;
  for (const std::vector<Block>& level_blocks : blocks) {
    rep->level_start.push_back(static_cast<std::int64_t>(rep->block_base.size()));
    rep->level_base.push_back(base);
    for (const Block& block : level_blocks) {
      rep->block_lo.insert(rep->block_lo.end(), block.first.begin(), block.first.end());
      rep->block_hi.insert(rep->block_hi.end(), block.second.begin(), block.second.end());
      rep->block_base.push_back(base);
      const std::optional<std::int64_t> size = try_block_size(block.first, block.second);
      if (!size.has_value() || *size > (std::numeric_limits<std::int64_t>::max)() - base) {
        throw std::overflow_error(
            "the hierarchical grid would have more than 9223372036854775807 cells.");
      }
      base += *size;
    }
  }
  rep->level_start.push_back(static_cast<std::int64_t>(rep->block_base.size()));
  rep->level_base.push_back(base);
  rep->num_cells = base;
  rep->blocks = std::move(blocks);
  return rep;
}

/// Get the flat id of the active leaf `(level, midx)`.
///
/// Scans the level's blocks and, on a hit, adds the row-major offset of `midx` inside the block to
/// the block's base.
///
/// Note:
///     No validation is performed: `level` is below the number of levels and `midx` holds `ndim`
///     indices.
///
/// Args:
///     rep (const HierarchicalGridRep&): The grid.
///     level (std::int64_t): Hierarchy level.
///     midx (const std::int64_t*): Per-axis index in level-`level` coordinates.
///
/// Returns:
///     std::int64_t: The flat id, or -1 when `(level, midx)` is not an active leaf.
[[nodiscard]] inline std::int64_t block_of_midx(const HierarchicalGridRep& rep, std::int64_t level,
                                                const std::int64_t* midx) noexcept {
  const std::size_t nd = rep.factor.size();
  const auto first = static_cast<std::size_t>(rep.level_start[static_cast<std::size_t>(level)]);
  const auto last = static_cast<std::size_t>(rep.level_start[static_cast<std::size_t>(level) + 1]);
  for (std::size_t b = first; b < last; ++b) {
    const std::int64_t* lo = rep.block_lo.data() + b * nd;
    const std::int64_t* hi = rep.block_hi.data() + b * nd;
    bool inside = true;
    for (std::size_t k = 0; k < nd; ++k) {
      if (midx[k] < lo[k] || midx[k] >= hi[k]) {
        inside = false;
        break;
      }
    }
    if (!inside) {
      continue;
    }
    std::int64_t offset = 0;
    for (std::size_t k = 0; k < nd; ++k) {
      offset = offset * (hi[k] - lo[k]) + (midx[k] - lo[k]);
    }
    return rep.block_base[b] + offset;
  }
  return -1;
}

/// Get the level of block `b`: the level whose block range holds it.
///
/// Note:
///     No validation is performed: `b` is below the number of blocks.
///
/// Args:
///     rep (const HierarchicalGridRep&): The grid.
///     b (std::size_t): Block index.
///
/// Returns:
///     std::int64_t: The level. Empty levels have equal start offsets and are skipped.
[[nodiscard]] inline std::int64_t level_of_block(const HierarchicalGridRep& rep,
                                                 std::size_t b) noexcept {
  const auto it = std::upper_bound(rep.level_start.begin(), rep.level_start.end(),
                                   static_cast<std::int64_t>(b));
  return static_cast<std::int64_t>(it - rep.level_start.begin()) - 1;
}

/// Get the block holding flat id `cid`: the last block whose base is at most `cid`.
///
/// Note:
///     No validation is performed: `cid` is in `[0, num_cells)`.
///
/// Args:
///     rep (const HierarchicalGridRep&): The grid.
///     cid (std::int64_t): Flat cell id.
///
/// Returns:
///     std::size_t: The block index.
[[nodiscard]] inline std::size_t block_of_cid(const HierarchicalGridRep& rep,
                                              std::int64_t cid) noexcept {
  const auto it = std::upper_bound(rep.block_base.begin(), rep.block_base.end(), cid);
  return static_cast<std::size_t>(it - rep.block_base.begin()) - 1;
}

/// Decode a flat cell id into its level and per-axis multi-index.
///
/// Note:
///     No validation is performed: `cid` is in `[0, num_cells)` and `out_midx` holds `ndim`
///     entries.
///
/// Args:
///     rep (const HierarchicalGridRep&): The grid.
///     cid (std::int64_t): Flat cell id.
///     out_midx (std::int64_t*): Receives the index in level coordinates.
///
/// Returns:
///     std::int64_t: The cell's level.
inline std::int64_t decode_flat_id(const HierarchicalGridRep& rep, std::int64_t cid,
                                   std::int64_t* out_midx) noexcept {
  const std::size_t nd = rep.factor.size();
  const std::size_t b = block_of_cid(rep, cid);
  const std::int64_t* lo = rep.block_lo.data() + b * nd;
  const std::int64_t* hi = rep.block_hi.data() + b * nd;
  std::int64_t offset = cid - rep.block_base[b];
  for (std::size_t k = nd; k-- > 0;) {
    const std::int64_t extent = hi[k] - lo[k];
    out_midx[k] = lo[k] + offset % extent;
    offset /= extent;
  }
  return level_of_block(rep, b);
}

/// Get the per-axis `(lo, hi)` bounds of a cell along one axis.
///
/// Evaluates `lo = root_lo + sub * size`, `hi = lo + size` with `size = (root_hi - root_lo) /
/// m_pow`, each product in a statement of its own so the build's contraction setting cannot fuse
/// it into the sum: the Python kernels do not contract, and this keeps the bounds equal to theirs.
///
/// Note:
///     No validation is performed: `bp` holds the axis's root breakpoints and `ik / m_pow` indexes
///     a root cell.
///
/// Args:
///     bp (const std::vector<double>&): The axis's root breakpoints.
///     ik (std::int64_t): Cell index along the axis at its level.
///     m_pow (std::int64_t): `factor ** level` along the axis.
///     lo (double&): Receives the lower bound.
///     hi (double&): Receives the upper bound.
inline void axis_bounds(const std::vector<double>& bp, std::int64_t ik, std::int64_t m_pow,
                        double& lo, double& hi) noexcept {
  const std::int64_t root_ik = ik / m_pow;
  const std::int64_t sub_ik = ik % m_pow;
  const double root_lo = bp[static_cast<std::size_t>(root_ik)];
  const double root_hi = bp[static_cast<std::size_t>(root_ik) + 1];
  const double size = (root_hi - root_lo) / static_cast<double>(m_pow);
  const double step = static_cast<double>(sub_ik) * size;
  lo = root_lo + step;
  hi = lo + size;
}

/// Get the `(lo, hi)` corners of the cell `(level, midx)`.
///
/// Args:
///     rep (const HierarchicalGridRep&): The grid.
///     level (std::int64_t): Hierarchy level of an existing cell.
///     midx (const std::int64_t*): Per-axis index in level coordinates.
///
/// Returns:
///     std::pair<std::vector<double>, std::vector<double>>: `ndim` entries each.
///
/// Raises:
///     std::overflow_error: `factor ** level` does not fit in `int64`, which no existing level
///         allows.
[[nodiscard]] inline std::pair<std::vector<double>, std::vector<double>> level_cell_bounds(
    const HierarchicalGridRep& rep, std::int64_t level, const std::int64_t* midx) {
  const std::size_t nd = rep.factor.size();
  std::vector<double> lo(nd);
  std::vector<double> hi(nd);
  for (std::size_t k = 0; k < nd; ++k) {
    axis_bounds(rep.root.breakpoints()[k], midx[k], checked_pow(rep.factor[k], level), lo[k],
                hi[k]);
  }
  return {std::move(lo), std::move(hi)};
}

/// The per-thread scratch of `locate_point`: the cell's index and corners, updated while
/// descending.
struct LocateScratch {
  explicit LocateScratch(std::size_t nd) : midx(nd), lo(nd), hi(nd) {}
  std::vector<std::int64_t> midx;
  std::vector<double> lo;
  std::vector<double> hi;
};

/// Pick the child of a cell that contains a point from the ratio `(x - lo) / size`.
///
/// Truncates the ratio toward zero and clamps it to `[0, fk - 1]`. The clamp is applied before the
/// conversion because the ratio can be as large as `fk`, which reaches `2**63` for a factor near
/// the `int64` limit, where converting a double to `int64` is undefined; a non-positive ratio (the
/// point on or, by round-off, just below the cell's lower face) selects child 0.
///
/// Args:
///     ratio (double): `(x - lo) / size`.
///     fk (std::int64_t): The axis's subdivision factor.
///
/// Returns:
///     std::int64_t: The child offset in `[0, fk - 1]`.
[[nodiscard]] inline std::int64_t child_offset(double ratio, std::int64_t fk) noexcept {
  if (!(ratio > 0.0)) {
    return 0;
  }
  if (ratio >= 9223372036854775808.0) {
    return fk - 1;
  }
  return (std::min)(static_cast<std::int64_t>(ratio), fk - 1);
}

/// Locate one point on a hierarchical grid by a top-down descent.
///
/// The point is located in the root grid by a binary search per axis (outside the root domain, NaN
/// included, means outside); then, level by level, the position is looked up in the active blocks,
/// and on a miss the containing child is computed from the floating-point bounds of the current
/// cell, the point on a child boundary going to the upper child.
///
/// Note:
///     No validation is performed: `pt` holds `ndim` coordinates and `scratch` was built for
///     `ndim`.
///
/// Args:
///     rep (const HierarchicalGridRep&): The grid.
///     pt (const double*): The point.
///     scratch (LocateScratch&): Working storage, overwritten.
///
/// Returns:
///     std::int64_t: The flat cell id, or -1 for a point outside the grid or with a NaN coordinate.
[[nodiscard]] inline std::int64_t locate_point(const HierarchicalGridRep& rep, const double* pt,
                                               LocateScratch& scratch) noexcept {
  const std::size_t nd = rep.factor.size();
  for (std::size_t d = 0; d < nd; ++d) {
    const std::vector<double>& bp = rep.root.breakpoints()[d];
    const double x = pt[d];
    // The negated chained comparison is NaN-safe: NaN fails both bounds.
    if (!(bp.front() <= x && x <= bp.back())) {
      return -1;
    }
    // lower_bound - 1 clamped below: it is at most the last cell since x <= bp.back(), and -1 only
    // for x == bp.front(), which belongs to cell 0.
    const auto idx =
        static_cast<std::int64_t>(std::lower_bound(bp.begin(), bp.end(), x) - bp.begin()) - 1;
    const std::int64_t cell_i = (std::max)(idx, std::int64_t{0});
    scratch.midx[d] = cell_i;
    scratch.lo[d] = bp[static_cast<std::size_t>(cell_i)];
    scratch.hi[d] = bp[static_cast<std::size_t>(cell_i) + 1];
  }
  const auto n_levels = static_cast<std::int64_t>(rep.blocks.size());
  for (std::int64_t level = 0; level < n_levels; ++level) {
    const std::int64_t cid = block_of_midx(rep, level, scratch.midx.data());
    if (cid >= 0) {
      return cid;
    }
    if (level >= n_levels - 1) {
      return -1;  // Unreachable in a consistent grid: every position is covered by a leaf.
    }
    for (std::size_t k = 0; k < nd; ++k) {
      const std::int64_t fk = rep.factor[k];
      const double size = (scratch.hi[k] - scratch.lo[k]) / static_cast<double>(fk);
      const std::int64_t j = child_offset((pt[k] - scratch.lo[k]) / size, fk);
      const double step = static_cast<double>(j) * size;
      scratch.lo[k] = scratch.lo[k] + step;
      scratch.hi[k] = scratch.lo[k] + size;
      scratch.midx[k] = scratch.midx[k] * fk + j;
    }
  }
  return -1;
}

/// Write the bounds of the cells with flat ids in `[first, last)` into the output arrays.
///
/// Locates the block of `first` once, then walks the cells in row-major order with an odometer,
/// moving on to the next block when one ends. Each cell's bounds depend on that cell alone, so any
/// partition of the id range into calls gives the same bits.
///
/// Note:
///     No validation is performed: `0 <= first <= last <= num_cells`, and the outputs hold
///     `(num_cells, ndim)` doubles each.
///
/// Args:
///     rep (const HierarchicalGridRep&): The grid.
///     first (std::int64_t): First flat id.
///     last (std::int64_t): One past the last flat id.
///     out_lo (double*): Lower corners, row-major `(num_cells, ndim)`.
///     out_hi (double*): Upper corners, same layout.
inline void collect_bounds_range(const HierarchicalGridRep& rep, std::int64_t first,
                                 std::int64_t last, double* out_lo, double* out_hi) {
  if (first >= last) {
    return;
  }
  const std::size_t nd = rep.factor.size();
  std::size_t b = block_of_cid(rep, first);
  std::vector<std::int64_t> midx(nd);
  std::vector<std::int64_t> m_pow(nd);
  // Position at block `b`, `skip` cells in.
  const auto enter_block = [&](std::size_t blk, std::int64_t skip) {
    const std::int64_t level = level_of_block(rep, blk);
    const std::int64_t* lo = rep.block_lo.data() + blk * nd;
    const std::int64_t* hi = rep.block_hi.data() + blk * nd;
    std::int64_t offset = skip;
    for (std::size_t k = nd; k-- > 0;) {
      const std::int64_t extent = hi[k] - lo[k];
      midx[k] = lo[k] + offset % extent;
      offset /= extent;
      m_pow[k] = checked_pow(rep.factor[k], level);
    }
  };
  enter_block(b, first - rep.block_base[b]);
  std::int64_t block_end = (b + 1 < rep.block_base.size()) ? rep.block_base[b + 1] : rep.num_cells;
  for (std::int64_t cid = first; cid < last; ++cid) {
    const auto row = static_cast<std::size_t>(cid) * nd;
    for (std::size_t k = 0; k < nd; ++k) {
      axis_bounds(rep.root.breakpoints()[k], midx[k], m_pow[k], out_lo[row + k], out_hi[row + k]);
    }
    if (cid + 1 == block_end) {
      if (cid + 1 < last) {
        ++b;
        enter_block(b, 0);
        block_end = (b + 1 < rep.block_base.size()) ? rep.block_base[b + 1] : rep.num_cells;
      }
      continue;
    }
    const std::int64_t* lo = rep.block_lo.data() + b * nd;
    const std::int64_t* hi = rep.block_hi.data() + b * nd;
    for (std::size_t k = nd; k-- > 0;) {
      if (++midx[k] < hi[k]) {
        break;
      }
      midx[k] = lo[k];
    }
  }
}

/// Get the flat id of `(level, midx)` if it is an active leaf of some level of the grid.
///
/// Args:
///     rep (const HierarchicalGridRep&): The grid.
///     level (std::int64_t): Hierarchy level; any value is accepted.
///     midx (const Index&): Per-axis index; any values are accepted.
///
/// Returns:
///     std::optional<std::int64_t>: The flat id, or no value when `level` is not a level of the
///     grid, `midx` has the wrong length, or the position is not an active leaf.
[[nodiscard]] inline std::optional<std::int64_t> encode_midx(const HierarchicalGridRep& rep,
                                                             std::int64_t level,
                                                             const Index& midx) noexcept {
  if (level < 0 || level >= static_cast<std::int64_t>(rep.blocks.size()) ||
      midx.size() != rep.factor.size()) {
    return std::nullopt;
  }
  const std::int64_t cid = block_of_midx(rep, level, midx.data());
  if (cid < 0) {
    return std::nullopt;
  }
  return cid;
}

/// Get the active leaf covering `(level, midx)` at the nearest strictly coarser level.
///
/// Args:
///     rep (const HierarchicalGridRep&): The grid.
///     level (std::int64_t): Level of the queried position.
///     midx (Index): Per-axis index in level-`level` coordinates (taken by value, divided up).
///
/// Returns:
///     std::optional<std::int64_t>: The covering leaf's id, or no value when no ancestor is active
///     (the position is either a leaf itself or subdivided into finer leaves).
[[nodiscard]] inline std::optional<std::int64_t> nearest_active_ancestor(
    const HierarchicalGridRep& rep, std::int64_t level, Index midx) {
  for (std::int64_t lvl = level - 1; lvl >= 0; --lvl) {
    for (std::size_t k = 0; k < midx.size(); ++k) {
      midx[k] /= rep.factor[k];
    }
    const std::optional<std::int64_t> cid = encode_midx(rep, lvl, midx);
    if (cid.has_value()) {
      return cid;
    }
  }
  return std::nullopt;
}

/// Collect the active leaves inside `(level, midx)` that touch one of its faces.
///
/// Descends through the subdivision tree keeping, at each step, only the children on the facet
/// plane (offset `face_j` along `axis`) and enumerating the other axes in row-major order, depth
/// first, so the ids come out ordered by their position along the face.
///
/// Args:
///     rep (const HierarchicalGridRep&): The grid.
///     level (std::int64_t): Level of the starting position.
///     midx (const Index&): Per-axis index in level-`level` coordinates.
///     axis (std::size_t): Axis normal to the face.
///     face_j (std::int64_t): Child offset along `axis` next to the facet plane.
///     out (std::vector<std::int64_t>&): Receives the ids, appended; the id of `(level, midx)`
///         itself when it is an active leaf.
inline void active_face_descendants(const HierarchicalGridRep& rep, std::int64_t level,
                                    const Index& midx, std::size_t axis, std::int64_t face_j,
                                    std::vector<std::int64_t>& out) {
  const std::optional<std::int64_t> cid = encode_midx(rep, level, midx);
  if (cid.has_value()) {
    out.push_back(*cid);
    return;
  }
  if (level + 1 >= static_cast<std::int64_t>(rep.blocks.size())) {
    return;
  }
  const std::size_t nd = midx.size();
  Index offsets(nd, 0);
  for (std::size_t k = 0; k < nd; ++k) {
    offsets[k] = (k == axis) ? face_j : 0;
  }
  Index child(nd);
  while (true) {
    for (std::size_t k = 0; k < nd; ++k) {
      child[k] = midx[k] * rep.factor[k] + offsets[k];
    }
    active_face_descendants(rep, level + 1, child, axis, face_j, out);
    // Odometer over the axes other than `axis`, last axis fastest.
    bool advanced = false;
    for (std::size_t k = nd; k-- > 0;) {
      if (k == axis) {
        continue;
      }
      if (++offsets[k] < rep.factor[k]) {
        advanced = true;
        break;
      }
      offsets[k] = 0;
    }
    if (!advanced) {
      return;
    }
  }
}

/// Name the cells of `[lo, hi)` that stop it being demoted from level `level + 1`.
///
/// A cell of the region blocks a coarsening in exactly one of three ways: it is still an active
/// leaf at `level`, it is refined past `level + 1`, or it is absent at `level` because a coarser
/// active leaf covers it. Each class is read off the block lists: leaves from the level's own
/// blocks, hidden cells from the coarser blocks scaled up to `level`, over-refined cells from the
/// deeper blocks scaled down to `level`.
///
/// Note:
///     Reached only from the failing path of a coarsening, so it favours a precise message over
///     speed and allocates one region-shaped mask per reason. No validation is performed: the
///     caller keeps the region within `max_diagnosed_cells` and inside the level's index range.
///
/// Args:
///     rep (const HierarchicalGridRep&): The grid.
///     level (std::int64_t): Level whose cells were to be reactivated, below the deepest level.
///     lo (const Index&): Region lower bound (inclusive), level-`level` coordinates.
///     hi (const Index&): Region upper bound (exclusive).
///
/// Returns:
///     std::string: Semicolon-separated clauses, each a reason followed by the offending cells;
///     empty when every cell of the region is refined to exactly `level + 1`.
[[nodiscard]] inline std::string coarsen_obstacles(const HierarchicalGridRep& rep,
                                                   std::int64_t level, const Index& lo,
                                                   const Index& hi) {
  const std::size_t nd = lo.size();
  Index shape(nd);
  for (std::size_t k = 0; k < nd; ++k) {
    shape[k] = hi[k] - lo[k];
  }
  const auto n_region = static_cast<std::size_t>(block_size(lo, hi));
  std::vector<std::uint8_t> still_leaf(n_region, 0);
  std::vector<std::uint8_t> over_refined(n_region, 0);
  std::vector<std::uint8_t> hidden(n_region, 0);
  const auto n_levels = static_cast<std::int64_t>(rep.blocks.size());

  for (const Block& block : rep.blocks[static_cast<std::size_t>(level)]) {
    mark_region_cells(still_leaf, lo, hi, block);
  }
  for (std::int64_t coarser = 0; coarser < level; ++coarser) {
    Index up(nd);
    for (std::size_t k = 0; k < nd; ++k) {
      up[k] = checked_pow(rep.factor[k], level - coarser);
    }
    for (const Block& block : rep.blocks[static_cast<std::size_t>(coarser)]) {
      Block scaled{Index(nd), Index(nd)};
      for (std::size_t k = 0; k < nd; ++k) {
        scaled.first[k] = block.first[k] * up[k];
        scaled.second[k] = block.second[k] * up[k];
      }
      mark_region_cells(hidden, lo, hi, scaled);
    }
  }
  for (std::int64_t deeper = level + 2; deeper < n_levels; ++deeper) {
    Index down(nd);
    for (std::size_t k = 0; k < nd; ++k) {
      down[k] = checked_pow(rep.factor[k], deeper - level);
    }
    for (const Block& block : rep.blocks[static_cast<std::size_t>(deeper)]) {
      // Floor the start and ceil the end: a level-`level` cell holding any part of a deeper block
      // has a descendant below level + 1.
      Block scaled{Index(nd), Index(nd)};
      for (std::size_t k = 0; k < nd; ++k) {
        scaled.first[k] = block.first[k] / down[k];
        scaled.second[k] = block.second[k] / down[k] + (block.second[k] % down[k] != 0 ? 1 : 0);
      }
      mark_region_cells(over_refined, lo, hi, scaled);
    }
  }

  std::string out;
  const auto add_reason = [&](const std::vector<std::uint8_t>& mask, const std::string& reason) {
    if (std::none_of(mask.begin(), mask.end(), [](std::uint8_t c) { return c != 0; })) {
      return;
    }
    if (!out.empty()) {
      out += "; ";
    }
    out += reason + ": " + name_marked_cells(mask, shape, lo);
  };
  add_reason(still_leaf, "still active leaves at level " + std::to_string(level) +
                             ", with no children to remove");
  add_reason(over_refined, "refined beyond level " + std::to_string(level + 1));
  add_reason(hidden,
             "covered by a coarser active leaf and absent at level " + std::to_string(level));
  return out;
}

}  // namespace pantr::grid::detail
