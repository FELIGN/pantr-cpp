/// @file bvh.hpp
/// @brief `BVH`: a bounding-volume hierarchy over a fixed collection of axis-aligned boxes.
///
/// Built once by median-of-longest-axis splits, queried by iterative descent. The tree is five
/// parallel arrays: per-node box corners (`node_lo`, `node_hi`, row-major `n_nodes x ndim`), child
/// indices (`node_left`, `node_right`, `-1` on leaves) and the cell id of a leaf (`node_cell`,
/// `-1` on internal nodes). The root is node 0 and covers every cell; for `N` cells there are
/// `2N - 1` nodes. Construction stops at one cell per leaf, so leaves and cells correspond one to
/// one, and the construction order (preorder, left to right) is deterministic.
///
/// **The overlap test is a separating-axis test and nothing else**, and it is not
/// `geometry::AABB::overlaps`. It compares inclusively, so a query box sharing a face with a cell
/// is reported; but it has no emptiness branch, so a *reversed* query interval is reported
/// against any cell whose own interval contains it, where the box reports nothing. One cell
/// `[0, 10]` in one dimension, queried with `lo = 5, hi = 3`, returns cell 0. This is the
/// behavior the Python implementation has; reconciling the two predicates has to move both at
/// once.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/geometry/aabb.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::grid {

namespace detail {

/// Depth of the fixed traversal stack, in nodes. An admitted bound, not a measurement: a median
/// split over `n <= 2^63` cells has depth at most `ceil(log2 n) + 1 <= 64`, so 128 is twice what
/// `from_cell_bounds` can reach; `BVH`'s raw constructor refuses an arbitrary tree deeper than it.
inline constexpr std::int64_t bvh_stack_depth = 128;

/// The shared representation of a `BVH`.
///
/// Attributes:
///     ndim (std::size_t): Dimension of the boxes, at least 1.
///     n_cells (std::int64_t): Number of indexed cells (leaves).
///     n_nodes (std::int64_t): `2 * n_cells - 1`, or 0 when there is no cell.
///     node_lo (std::vector<double>): Node lower corners, row-major `n_nodes x ndim`.
///     node_hi (std::vector<double>): Node upper corners, the same layout.
///     node_left (std::vector<std::int64_t>): Left child per node, `-1` on leaves.
///     node_right (std::vector<std::int64_t>): Right child per node, `-1` on leaves.
///     node_cell (std::vector<std::int64_t>): Cell id per leaf, `-1` on internal nodes.
struct BVHRep {
  std::size_t ndim = 0;
  std::int64_t n_cells = 0;
  std::int64_t n_nodes = 0;
  std::vector<double> node_lo;
  std::vector<double> node_hi;
  std::vector<std::int64_t> node_left;
  std::vector<std::int64_t> node_right;
  std::vector<std::int64_t> node_cell;
};

/// Check that the node arrays encode a tree the traversal can walk.
///
/// The traversal decides "leaf or internal" from `node_cell` alone and then pushes both children
/// without testing either against `-1`; so the two encodings of leafness have to agree, and an
/// internal node's children have to be real indices.
///
/// Args:
///     rep (const BVHRep&): The arrays to check.
///
/// Raises:
///     value_error: `node_cell` and the child pointers disagree about which nodes are leaves, an
///         internal node's child index lies outside `[0, n_nodes)`, or a leaf's cell id is
///         `>= n_cells`.
inline void check_tree_structure(const BVHRep& rep) {
  const auto n = static_cast<std::size_t>(rep.n_nodes);
  for (std::size_t i = 0; i < n; ++i) {
    const bool is_leaf = rep.node_cell[i] >= 0;
    if (is_leaf != (rep.node_left[i] == -1 && rep.node_right[i] == -1)) {
      throw value_error(
          "BVH: node_cell and the child pointers disagree about which nodes are "
          "leaves. A node is a leaf iff node_cell >= 0, and exactly then must "
          "node_left and node_right both be -1.");
    }
  }
  const std::pair<const std::vector<std::int64_t>*, const char*> children[2] = {
      {&rep.node_left, "node_left"}, {&rep.node_right, "node_right"}};
  for (const auto& [arr, name] : children) {
    bool any = false;
    std::int64_t lo = 0;
    std::int64_t hi = 0;
    for (std::size_t i = 0; i < n; ++i) {
      if (rep.node_cell[i] >= 0) {
        continue;
      }
      const std::int64_t c = (*arr)[i];
      lo = any ? (std::min)(lo, c) : c;
      hi = any ? (std::max)(hi, c) : c;
      any = true;
    }
    if (any && (lo < 0 || hi >= rep.n_nodes)) {
      throw value_error("BVH: " + std::string(name) + " contains values outside [0, " +
                        std::to_string(rep.n_nodes) + ") on internal nodes: range is [" +
                        std::to_string(lo) + ", " + std::to_string(hi) + "].");
    }
  }
  bool any_leaf = false;
  std::int64_t max_cell = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (rep.node_cell[i] >= 0) {
      max_cell = any_leaf ? (std::max)(max_cell, rep.node_cell[i]) : rep.node_cell[i];
      any_leaf = true;
    }
  }
  if (any_leaf && max_cell >= rep.n_cells) {
    throw value_error("BVH: node_cell contains values outside [0, " + std::to_string(rep.n_cells) +
                      ") on leaves: maximum is " + std::to_string(max_cell) + ".");
  }
}

/// Measure a node array's maximum root-to-leaf depth, capped at `limit`.
///
/// The walk stops as soon as it finds a path deeper than `limit`, which is all the caller needs.
/// That alone does not bound it: node arrays in which internal nodes share children (a DAG) have
/// exponentially many root-to-leaf paths, none of them deep. A tree reaches each node exactly
/// once, so the walk counts its pops and refuses as soon as they exceed `n_nodes`; that count is
/// also what ends it on a cycle.
///
/// Args:
///     rep (const BVHRep&): A structure-checked tree.
///     limit (std::int64_t): Depth beyond which the exact value no longer matters.
///
/// Returns:
///     std::int64_t: The number of nodes on the longest root-to-leaf path (the root counts as
///     depth 1); 0 for no nodes. A value greater than `limit` means only "deeper than `limit`".
///
/// Raises:
///     value_error: A node is reachable from the root more than once, so the arrays are not a
///         tree.
[[nodiscard]] inline std::int64_t max_tree_depth(const BVHRep& rep, std::int64_t limit) {
  if (rep.n_nodes == 0) {
    return 0;
  }
  std::vector<std::pair<std::int64_t, std::int64_t>> stack;
  stack.emplace_back(0, 1);
  std::int64_t max_depth = 0;
  std::int64_t pops = 0;
  while (!stack.empty()) {
    const auto [node, depth] = stack.back();
    stack.pop_back();
    if (++pops > rep.n_nodes) {
      throw value_error(
          "BVH: the node arrays do not encode a tree: a node is reachable from the root more "
          "than once.");
    }
    max_depth = (std::max)(max_depth, depth);
    if (max_depth > limit) {
      return max_depth;
    }
    const std::int64_t left = rep.node_left[static_cast<std::size_t>(node)];
    const std::int64_t right = rep.node_right[static_cast<std::size_t>(node)];
    if (left != -1) {
      stack.emplace_back(left, depth + 1);
    }
    if (right != -1) {
      stack.emplace_back(right, depth + 1);
    }
  }
  return max_depth;
}

/// Build the BVH arrays from per-cell boxes (median-of-longest-axis splits).
///
/// Iterative top-down construction with an explicit work stack; the root claims node 0 and
/// subsequent splits claim indices in preorder, left to right. At each node the cells are sorted
/// by centroid along the longest axis of the node's box (the first on a tie) with a stable sort
/// and split at the median, so the tree is a deterministic function of the input. The work stack
/// holds at most one pending right-sibling range per level of the current descent path, so its
/// peak size is bounded by the tree height: the median split bounds the height by
/// `ceil(log2 n) + 1 <= 64 < bvh_stack_depth`, which is why the fixed stack is safe.
///
/// Note:
///     No validation is performed (finite corners, `hi >= lo`, `n_cells >= 1`). For general use
///     call `BVH::from_cell_bounds`.
///
/// Args:
///     cell_lo (std::span<const double>): Cell lower corners, row-major `n_cells x ndim`.
///     cell_hi (std::span<const double>): Cell upper corners, the same layout.
///     ndim (std::size_t): The dimension.
///     n_cells (std::int64_t): Number of cells, at least 1.
///
/// Returns:
///     BVHRep: The tree.
[[nodiscard]] inline BVHRep bvh_build_core(std::span<const double> cell_lo,
                                           std::span<const double> cell_hi, std::size_t ndim,
                                           std::int64_t n_cells) {
  const auto n = static_cast<std::size_t>(n_cells);
  const auto max_nodes = 2 * n - 1;
  BVHRep rep;
  rep.ndim = ndim;
  rep.n_cells = n_cells;
  rep.n_nodes = static_cast<std::int64_t>(max_nodes);
  rep.node_lo.assign(max_nodes * ndim, 0.0);
  rep.node_hi.assign(max_nodes * ndim, 0.0);
  rep.node_left.assign(max_nodes, -1);
  rep.node_right.assign(max_nodes, -1);
  rep.node_cell.assign(max_nodes, -1);

  std::vector<double> centroid(n * ndim);
  for (std::size_t i = 0; i < n * ndim; ++i) {
    centroid[i] = 0.5 * (cell_lo[i] + cell_hi[i]);
  }
  std::vector<std::int64_t> perm(n);
  std::iota(perm.begin(), perm.end(), std::int64_t{0});

  // Work stack of (node index, first cell, one past the last cell) triples.
  std::array<std::size_t, bvh_stack_depth> stack_idx{};
  std::array<std::size_t, bvh_stack_depth> stack_start{};
  std::array<std::size_t, bvh_stack_depth> stack_end{};
  stack_idx[0] = 0;
  stack_start[0] = 0;
  stack_end[0] = n;
  std::size_t sp = 1;
  std::size_t next_idx = 1;

  while (sp > 0) {
    --sp;
    const std::size_t idx = stack_idx[sp];
    const std::size_t start = stack_start[sp];
    const std::size_t end = stack_end[sp];

    // Tight box of the node: min/max over its cells.
    for (std::size_t k = 0; k < ndim; ++k) {
      const auto first = static_cast<std::size_t>(perm[start]);
      double lo_k = cell_lo[first * ndim + k];
      double hi_k = cell_hi[first * ndim + k];
      for (std::size_t c = start + 1; c < end; ++c) {
        const auto cell = static_cast<std::size_t>(perm[c]);
        lo_k = (std::min)(lo_k, cell_lo[cell * ndim + k]);
        hi_k = (std::max)(hi_k, cell_hi[cell * ndim + k]);
      }
      rep.node_lo[idx * ndim + k] = lo_k;
      rep.node_hi[idx * ndim + k] = hi_k;
    }

    const std::size_t count = end - start;
    if (count == 1) {
      rep.node_cell[idx] = perm[start];
      continue;
    }

    // Longest axis of the node box (first maximum).
    std::size_t axis = 0;
    double best = rep.node_hi[idx * ndim] - rep.node_lo[idx * ndim];
    for (std::size_t k = 1; k < ndim; ++k) {
      const double extent_k = rep.node_hi[idx * ndim + k] - rep.node_lo[idx * ndim + k];
      if (extent_k > best) {
        best = extent_k;
        axis = k;
      }
    }
    // Stable sort of the node's cells by centroid along the split axis.
    std::stable_sort(perm.begin() + static_cast<std::ptrdiff_t>(start),
                     perm.begin() + static_cast<std::ptrdiff_t>(end),
                     [&](std::int64_t a, std::int64_t b) {
                       return centroid[static_cast<std::size_t>(a) * ndim + axis] <
                              centroid[static_cast<std::size_t>(b) * ndim + axis];
                     });

    const std::size_t mid = start + count / 2;
    const std::size_t left_idx = next_idx;
    const std::size_t right_idx = next_idx + 1;
    next_idx += 2;
    rep.node_left[idx] = static_cast<std::int64_t>(left_idx);
    rep.node_right[idx] = static_cast<std::int64_t>(right_idx);
    // Push right first so left pops first (preorder, left to right).
    stack_idx[sp] = right_idx;
    stack_start[sp] = mid;
    stack_end[sp] = end;
    stack_idx[sp + 1] = left_idx;
    stack_start[sp + 1] = start;
    stack_end[sp + 1] = mid;
    sp += 2;
  }
  return rep;
}

/// Collect the cell ids of the leaves whose box is not separated from the query box.
///
/// The traversal pushes the left child, then the right one, so it pops the right one first: the
/// tree is visited in preorder, right to left. Each node costs `ndim` comparisons.
///
/// Note:
///     No validation is performed: the tree must have passed `check_tree_structure` and have depth
///     at most `bvh_stack_depth`, and the query corners must have `ndim` entries.
///
/// Args:
///     rep (const BVHRep&): A validated tree with at least one node.
///     qlo (const double*): Query lower corner.
///     qhi (const double*): Query upper corner.
///
/// Returns:
///     std::vector<std::int64_t>: The cell ids, in traversal order.
[[nodiscard]] inline std::vector<std::int64_t> bvh_query_emit_core(const BVHRep& rep,
                                                                   const double* qlo,
                                                                   const double* qhi) {
  const std::size_t ndim = rep.ndim;
  std::array<std::int64_t, bvh_stack_depth> stack{};
  stack[0] = 0;
  std::size_t top = 1;
  std::vector<std::int64_t> out;
  while (top > 0) {
    --top;
    const auto idx = static_cast<std::size_t>(stack[top]);
    bool overlaps = true;
    for (std::size_t d = 0; d < ndim; ++d) {
      if (qhi[d] < rep.node_lo[idx * ndim + d] || qlo[d] > rep.node_hi[idx * ndim + d]) {
        overlaps = false;
        break;
      }
    }
    if (!overlaps) {
      continue;
    }
    const std::int64_t cell = rep.node_cell[idx];
    if (cell >= 0) {
      out.push_back(cell);
    } else {
      stack[top] = rep.node_left[idx];
      ++top;
      stack[top] = rep.node_right[idx];
      ++top;
    }
  }
  return out;
}

}  // namespace detail

/// A bounding-volume hierarchy indexing a fixed collection of axis-aligned boxes.
///
/// An immutable value: the arrays are copied or built at construction and shared by every copy.
/// Queries allocate their result.
///
/// Attributes:
///     rep_ (std::shared_ptr<const detail::BVHRep>): The five node arrays and the counts.
class BVH {
 public:
  /// Build a tree from its raw arrays after validating them (mostly for tests and round-trip
  /// serialization; prefer `from_cell_bounds`). The dtype checks of the Python constructor live in
  /// the Python wrapper.
  ///
  /// Args:
  ///     node_lo (std::span<const double>): Node lower corners, row-major over `lo_shape`.
  ///     node_hi (std::span<const double>): Node upper corners, row-major over `hi_shape`.
  ///     lo_shape (std::span<const std::size_t>): Extents of `node_lo`; must be `(n_nodes, ndim)`.
  ///     hi_shape (std::span<const std::size_t>): Extents of `node_hi`; must equal `lo_shape`.
  ///     node_left (std::span<const std::int64_t>): Left child per node, `-1` on leaves.
  ///     node_right (std::span<const std::int64_t>): Right child per node, `-1` on leaves.
  ///     node_cell (std::span<const std::int64_t>): Cell id per leaf, `-1` on internal nodes.
  ///     n_cells (std::int64_t): Number of indexed cells (leaves).
  ///
  /// Raises:
  ///     value_error: Shapes are inconsistent, `ndim < 1`, `n_nodes != 2 * n_cells - 1` (0 when
  ///         `n_cells <= 0`), `node_cell` and the child pointers disagree about which nodes are
  ///         leaves, an internal node's children or a leaf's cell id are out of range, the
  ///         arrays are not a tree (a node reachable twice), or the tree is deeper than the
  ///         traversal stack allows.
  ///     capacity_error: A shape's product overflows `std::size_t`.
  BVH(std::span<const double> node_lo, std::span<const double> node_hi,
      std::span<const std::size_t> lo_shape, std::span<const std::size_t> hi_shape,
      std::span<const std::int64_t> node_left, std::span<const std::int64_t> node_right,
      std::span<const std::int64_t> node_cell, std::int64_t n_cells)
      : rep_(make_rep(node_lo, node_hi, lo_shape, hi_shape, node_left, node_right, node_cell,
                      n_cells)) {}

  /// Build a tree over `n_cells` cell boxes by top-down median-of-longest-axis splits.
  ///
  /// Cells are sorted by centroid on the longest axis; the median splits the list into two halves
  /// of equal size (+/- 1). Each leaf indexes exactly one cell. The tree depth is at most
  /// `ceil(log2 n_cells) + 1 <= 64`, below the traversal stack depth for every `int64` cell
  /// count, so there is no depth limit to report here.
  ///
  /// Args:
  ///     lo (std::span<const double>): Cell lower corners, row-major over `lo_shape`.
  ///     hi (std::span<const double>): Cell upper corners, row-major over `hi_shape`.
  ///     lo_shape (std::span<const std::size_t>): Extents of `lo`; must be `(n_cells, ndim)`.
  ///     hi_shape (std::span<const std::size_t>): Extents of `hi`; must equal `lo_shape`.
  ///
  /// Returns:
  ///     BVH: The hierarchy; the empty tree (no node) when `n_cells == 0`.
  ///
  /// Raises:
  ///     value_error: Shapes are inconsistent, `ndim < 1`, a corner is not finite, or a cell has
  ///         `hi < lo` on some axis.
  ///     capacity_error: A shape's product overflows `std::size_t`.
  [[nodiscard]] static BVH from_cell_bounds(std::span<const double> lo, std::span<const double> hi,
                                            std::span<const std::size_t> lo_shape,
                                            std::span<const std::size_t> hi_shape) {
    if (lo_shape.size() != 2) {
      throw value_error("cell_lo must be 2-D (n_cells, ndim); got shape " + py_repr(lo_shape) +
                        ".");
    }
    if (!std::equal(hi_shape.begin(), hi_shape.end(), lo_shape.begin(), lo_shape.end())) {
      throw value_error("cell_hi shape " + py_repr(hi_shape) + " must match cell_lo shape " +
                        py_repr(lo_shape) + ".");
    }
    const std::size_t n_cells = lo_shape[0];
    const std::size_t ndim = lo_shape[1];
    if (ndim < 1) {
      throw value_error("BVH ndim must be >= 1; got " + std::to_string(ndim) + ".");
    }
    require_holds(lo.size(), lo_shape, "cell_lo");
    require_holds(hi.size(), hi_shape, "cell_hi");
    const auto finite = [](double x) { return std::isfinite(x); };
    if (!std::all_of(lo.begin(), lo.end(), finite) || !std::all_of(hi.begin(), hi.end(), finite)) {
      throw value_error(
          "BVH.from_cell_bounds: cell_lo and cell_hi must contain only finite "
          "values; got NaN or Inf.");
    }
    for (std::size_t i = 0; i < lo.size(); ++i) {
      if (hi[i] < lo[i]) {
        throw value_error(
            "Every cell must satisfy cell_hi >= cell_lo on every axis; "
            "at least one cell violates this.");
      }
    }
    if (n_cells == 0) {
      auto rep = std::make_shared<detail::BVHRep>();
      rep->ndim = ndim;
      return BVH(std::move(rep));
    }
    return BVH(std::make_shared<const detail::BVHRep>(
        detail::bvh_build_core(lo, hi, ndim, static_cast<std::int64_t>(n_cells))));
  }

  /// Get the ids of every leaf cell whose box is not separated from the query box.
  ///
  /// A separating-axis test, not `AABB::overlaps` (see the file comment for how they differ on an
  /// empty query box). A NaN corner is refused, as `geometry::AABB` refuses it: every comparison
  /// against NaN is false, so the test would report every cell.
  ///
  /// Args:
  ///     qlo (std::span<const double>): Query lower corner, `ndim()` entries.
  ///     qhi (std::span<const double>): Query upper corner, `ndim()` entries.
  ///
  /// Returns:
  ///     std::vector<std::int64_t>: Overlapping cell ids in the internal preorder traversal order;
  ///     callers that need a particular order should sort the result.
  ///
  /// Raises:
  ///     value_error: A corner does not have `ndim()` entries or contains NaN.
  [[nodiscard]] std::vector<std::int64_t> query_aabb(std::span<const double> qlo,
                                                     std::span<const double> qhi) const {
    if (qlo.size() != ndim() || qhi.size() != ndim()) {
      throw value_error("BVH.query_aabb: query corners must have length " + std::to_string(ndim()) +
                        "; got " + std::to_string(qlo.size()) + " and " +
                        std::to_string(qhi.size()) + ".");
    }
    const auto is_nan = [](double x) { return std::isnan(x); };
    if (std::any_of(qlo.begin(), qlo.end(), is_nan) ||
        std::any_of(qhi.begin(), qhi.end(), is_nan)) {
      throw value_error("BVH.query_aabb: query corners must not contain NaN.");
    }
    // A raw tree may carry `n_cells < 0` with no node (as the Python constructor accepts), so the
    // node count, not the cell count, says whether there is anything to walk.
    if (rep_->n_nodes == 0) {
      return {};
    }
    return detail::bvh_query_emit_core(*rep_, qlo.data(), qhi.data());
  }

  /// Get the ids of every leaf cell whose box is not separated from `aabb`.
  ///
  /// Args:
  ///     aabb (const geometry::AABB&): Query box; its dimension must match `ndim()`.
  ///
  /// Returns:
  ///     std::vector<std::int64_t>: As the corner overload.
  ///
  /// Raises:
  ///     value_error: `aabb.ndim() != ndim()`.
  [[nodiscard]] std::vector<std::int64_t> query_aabb(const geometry::AABB& aabb) const {
    if (aabb.ndim() != ndim()) {
      throw value_error("BVH.query_aabb: aabb.ndim (" + std::to_string(aabb.ndim()) +
                        ") must match self.ndim (" + std::to_string(ndim()) + ").");
    }
    return query_aabb(aabb.lo(), aabb.hi());
  }

  /// Get the dimension of the indexed boxes.
  ///
  /// Returns:
  ///     std::size_t: At least 1.
  [[nodiscard]] std::size_t ndim() const noexcept { return rep_->ndim; }

  /// Get the number of indexed cells (the number of leaves).
  ///
  /// Returns:
  ///     std::int64_t: The cell count.
  [[nodiscard]] std::int64_t n_cells() const noexcept { return rep_->n_cells; }

  /// Get the total number of nodes.
  ///
  /// Returns:
  ///     std::int64_t: `2 * n_cells() - 1` for `n_cells() > 0`, else 0.
  [[nodiscard]] std::int64_t n_nodes() const noexcept { return rep_->n_nodes; }

  /// Get the per-node lower corners.
  ///
  /// Returns:
  ///     std::span<const double>: Row-major `n_nodes() x ndim()`.
  [[nodiscard]] std::span<const double> node_lo() const noexcept { return rep_->node_lo; }

  /// Get the per-node upper corners.
  ///
  /// Returns:
  ///     std::span<const double>: Row-major `n_nodes() x ndim()`.
  [[nodiscard]] std::span<const double> node_hi() const noexcept { return rep_->node_hi; }

  /// Get the per-node left-child indices.
  ///
  /// Returns:
  ///     std::span<const std::int64_t>: `n_nodes()` entries; `-1` on leaves.
  [[nodiscard]] std::span<const std::int64_t> node_left() const noexcept { return rep_->node_left; }

  /// Get the per-node right-child indices.
  ///
  /// Returns:
  ///     std::span<const std::int64_t>: `n_nodes()` entries; `-1` on leaves.
  [[nodiscard]] std::span<const std::int64_t> node_right() const noexcept {
    return rep_->node_right;
  }

  /// Get the per-leaf cell identifiers.
  ///
  /// Returns:
  ///     std::span<const std::int64_t>: `n_nodes()` entries; `-1` on internal nodes, a cell id in
  ///     `[0, n_cells())` on leaves.
  [[nodiscard]] std::span<const std::int64_t> node_cell() const noexcept { return rep_->node_cell; }

  /// Get the shared representation, for an owner that must keep the arrays alive.
  ///
  /// Returns:
  ///     const std::shared_ptr<const detail::BVHRep>&: The representation.
  [[nodiscard]] const std::shared_ptr<const detail::BVHRep>& rep() const noexcept { return rep_; }

 private:
  explicit BVH(std::shared_ptr<const detail::BVHRep> rep) : rep_(std::move(rep)) {}

  static void require_holds(std::size_t held, std::span<const std::size_t> shape,
                            const char* name) {
    const std::size_t need = pantr::checked_product(shape);
    if (held != need) {
      throw value_error(std::string(name) + " holds " + std::to_string(held) +
                        " values, but its shape " + py_repr(shape) + " needs " +
                        std::to_string(need) + ".");
    }
  }

  static std::shared_ptr<const detail::BVHRep> make_rep(
      std::span<const double> node_lo, std::span<const double> node_hi,
      std::span<const std::size_t> lo_shape, std::span<const std::size_t> hi_shape,
      std::span<const std::int64_t> node_left, std::span<const std::int64_t> node_right,
      std::span<const std::int64_t> node_cell, std::int64_t n_cells) {
    if (lo_shape.size() != 2) {
      throw value_error("node_lo must be 2-D (n_nodes, ndim); got shape " + py_repr(lo_shape) +
                        ".");
    }
    if (!std::equal(hi_shape.begin(), hi_shape.end(), lo_shape.begin(), lo_shape.end())) {
      throw value_error("node_hi shape " + py_repr(hi_shape) + " must match node_lo shape " +
                        py_repr(lo_shape) + ".");
    }
    const std::size_t n_nodes = lo_shape[0];
    const std::size_t ndim = lo_shape[1];
    if (ndim < 1) {
      throw value_error("BVH ndim must be >= 1; got " + std::to_string(ndim) + ".");
    }
    require_holds(node_lo.size(), lo_shape, "node_lo");
    require_holds(node_hi.size(), hi_shape, "node_hi");
    const std::pair<std::span<const std::int64_t>, const char*> int_arrays[3] = {
        {node_left, "node_left"}, {node_right, "node_right"}, {node_cell, "node_cell"}};
    for (const auto& [arr, name] : int_arrays) {
      if (arr.size() != n_nodes) {
        const std::size_t got[1] = {arr.size()};
        throw value_error(std::string(name) + " must have shape (" + std::to_string(n_nodes) +
                          ",); got " + py_repr(got) + ".");
      }
    }
    // 2 * n_cells - 1 fits an unsigned 64-bit integer for every positive int64 n_cells.
    const std::uint64_t expected_nodes =
        n_cells > 0 ? 2 * static_cast<std::uint64_t>(n_cells) - 1 : 0;
    if (static_cast<std::uint64_t>(n_nodes) != expected_nodes) {
      throw value_error("BVH: n_cells=" + std::to_string(n_cells) +
                        " implies n_nodes=" + std::to_string(expected_nodes) +
                        "; got node arrays with " + std::to_string(n_nodes) + " rows.");
    }
    auto rep = std::make_shared<detail::BVHRep>();
    rep->ndim = ndim;
    rep->n_cells = n_cells;
    rep->n_nodes = static_cast<std::int64_t>(n_nodes);
    rep->node_lo.assign(node_lo.begin(), node_lo.end());
    rep->node_hi.assign(node_hi.begin(), node_hi.end());
    rep->node_left.assign(node_left.begin(), node_left.end());
    rep->node_right.assign(node_right.begin(), node_right.end());
    rep->node_cell.assign(node_cell.begin(), node_cell.end());
    // Runs before the depth walk, which indexes the children itself.
    detail::check_tree_structure(*rep);
    // Guard the fixed-depth traversal stack. `from_cell_bounds` needs no such check, its median
    // split keeping the tree balanced; this constructor takes arbitrary node arrays, and an
    // unbalanced one overflows the stack on the first query. A depth-d traversal occupies at most
    // d slots, so depth == bvh_stack_depth still fits.
    const std::int64_t tree_depth = detail::max_tree_depth(*rep, detail::bvh_stack_depth);
    if (tree_depth > detail::bvh_stack_depth) {
      throw value_error("BVH: the given node arrays encode a tree of depth " +
                        std::to_string(tree_depth) +
                        " or more, "
                        "exceeding the stack depth " +
                        std::to_string(detail::bvh_stack_depth) +
                        " that the traversal kernels "
                        "allow. Build the tree more evenly, or use BVH.from_cell_bounds, whose "
                        "median split keeps the depth logarithmic in the cell count.");
    }
    return rep;
  }

  std::shared_ptr<const detail::BVHRep> rep_;
};

}  // namespace pantr::grid
