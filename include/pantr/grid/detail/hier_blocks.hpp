/// @file hier_blocks.hpp
/// @brief The integer-rectangle ("block") helpers of the hierarchical grid, and the checked integer
/// arithmetic its level formulas need.
///
/// A block is a pair `(lo, hi)` of per-axis inclusive-start and exclusive-end integer indices in
/// the coordinates of one level. These are pure functions of integers, kept apart from the grid so
/// the Python test suite can bind and exercise them directly.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/grid/detail/checked.hpp"
#include "pantr/grid/detail/py_text.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pantr::grid::detail {

/// A per-axis integer index or bound.
using Index = std::vector<std::int64_t>;

/// An integer rectangle `(lo, hi)`: inclusive start and exclusive end per axis.
using Block = std::pair<Index, Index>;

/// Cells spelled out per reason when a coarsening is refused. Ported from the Python
/// implementation: enough to show the pattern of the offending set while keeping the message
/// readable; the remainder is reported as a count.
inline constexpr std::int64_t max_named_cells = 6;

/// Region size above which a refused coarsening skips naming cells. Ported from the Python
/// implementation, where it is a memory budget: the diagnostic materializes three masks shaped
/// like the region (one byte per cell here), so this bounds them at 3 MiB together. Past it the
/// message reports the region's extent instead.
inline constexpr std::int64_t max_diagnosed_cells = std::int64_t{1} << 20;

/// Multiply two non-negative `int64` values, reporting a product that does not fit.
///
/// Args:
///     a (std::int64_t): First factor, at least 0.
///     b (std::int64_t): Second factor, at least 0.
///
/// Returns:
///     std::optional<std::int64_t>: `a * b`, or no value when it exceeds the `int64` range.
[[nodiscard]] inline std::optional<std::int64_t> try_mul(std::int64_t a, std::int64_t b) noexcept {
  if (a == 0 || b == 0) {
    return 0;
  }
  if (a > (std::numeric_limits<std::int64_t>::max)() / b) {
    return std::nullopt;
  }
  return a * b;
}

/// Raise `base` to a non-negative integer power, reporting a result that does not fit.
///
/// A base of 1 gives 1 for every exponent without looping, so a huge exponent costs nothing; a base
/// of at least 2 overflows within 63 multiplications, which bounds the loop.
///
/// Args:
///     base (std::int64_t): At least 1.
///     exponent (std::int64_t): At least 0.
///
/// Returns:
///     std::optional<std::int64_t>: `base ** exponent`, or no value when it exceeds `int64`.
[[nodiscard]] inline std::optional<std::int64_t> try_pow(std::int64_t base,
                                                         std::int64_t exponent) noexcept {
  if (base == 1 || exponent == 0) {
    return 1;
  }
  std::int64_t p = 1;
  for (std::int64_t i = 0; i < exponent; ++i) {
    const std::optional<std::int64_t> next = try_mul(p, base);
    if (!next.has_value()) {
      return std::nullopt;
    }
    p = *next;
  }
  return p;
}

/// Raise `base` to a non-negative integer power, refusing a result that does not fit.
///
/// Args:
///     base (std::int64_t): At least 1.
///     exponent (std::int64_t): At least 0.
///
/// Returns:
///     std::int64_t: `base ** exponent`.
///
/// Raises:
///     std::overflow_error: The power is outside the `int64` range.
[[nodiscard]] inline std::int64_t checked_pow(std::int64_t base, std::int64_t exponent) {
  const std::optional<std::int64_t> p = try_pow(base, exponent);
  if (!p.has_value()) {
    throw std::overflow_error("integer overflow in a power of a subdivision factor");
  }
  return *p;
}

/// Get the number of cells in the integer rectangle `[lo, hi)`, or no value if it exceeds `int64`.
///
/// Note:
///     No validation is performed: `lo` and `hi` have equal length and `lo[k] <= hi[k]`.
///
/// Args:
///     lo (const Index&): Per-axis start indices (inclusive).
///     hi (const Index&): Per-axis end indices (exclusive).
///
/// Returns:
///     std::optional<std::int64_t>: The product of `hi[k] - lo[k]` (0 if any factor is 0, however
///     large the others), or no value when it does not fit in `int64`.
[[nodiscard]] inline std::optional<std::int64_t> try_block_size(const Index& lo,
                                                                const Index& hi) noexcept {
  // A zero extent makes the product 0 whatever the other factors are, so test it before any
  // partial product can overflow.
  for (std::size_t k = 0; k < lo.size(); ++k) {
    if (hi[k] == lo[k]) {
      return 0;
    }
  }
  std::int64_t size = 1;
  for (std::size_t k = 0; k < lo.size(); ++k) {
    const std::optional<std::int64_t> next = try_mul(size, hi[k] - lo[k]);
    if (!next.has_value()) {
      return std::nullopt;
    }
    size = *next;
  }
  return size;
}

/// Get the number of cells in the integer rectangle `[lo, hi)`.
///
/// Note:
///     No validation is performed beyond the overflow check: `lo` and `hi` have equal length and
///     `lo[k] <= hi[k]`.
///
/// Args:
///     lo (const Index&): Per-axis start indices (inclusive).
///     hi (const Index&): Per-axis end indices (exclusive).
///
/// Returns:
///     std::int64_t: The product of `hi[k] - lo[k]`.
///
/// Raises:
///     std::overflow_error: The product does not fit in `int64`.
[[nodiscard]] inline std::int64_t block_size(const Index& lo, const Index& hi) {
  const std::optional<std::int64_t> size = try_block_size(lo, hi);
  if (!size.has_value()) {
    throw std::overflow_error("integer overflow in a product of grid sizes");
  }
  return *size;
}

/// Report whether `midx` lies inside `[lo, hi)`.
///
/// Note:
///     No validation is performed: the three arguments have equal length.
///
/// Args:
///     midx (const Index&): Per-axis index to test.
///     lo (const Index&): Block lower bound (inclusive).
///     hi (const Index&): Block upper bound (exclusive).
///
/// Returns:
///     bool: Whether `lo[k] <= midx[k] < hi[k]` for every `k`.
[[nodiscard]] inline bool in_block(const Index& midx, const Index& lo, const Index& hi) noexcept {
  for (std::size_t k = 0; k < midx.size(); ++k) {
    if (!(lo[k] <= midx[k] && midx[k] < hi[k])) {
      return false;
    }
  }
  return true;
}

/// Intersect two rectangles.
///
/// Note:
///     No validation is performed: the four arguments have equal length.
///
/// Args:
///     lo1 (const Index&): Lower bound of the first rectangle.
///     hi1 (const Index&): Upper bound of the first rectangle.
///     lo2 (const Index&): Lower bound of the second rectangle.
///     hi2 (const Index&): Upper bound of the second rectangle.
///
/// Returns:
///     std::optional<Block>: `(lo, hi)` of the intersection, or no value when the rectangles do not
///     overlap in a non-empty set of cells.
[[nodiscard]] inline std::optional<Block> rect_intersect(const Index& lo1, const Index& hi1,
                                                         const Index& lo2, const Index& hi2) {
  Block out{Index(lo1.size()), Index(lo1.size())};
  bool overlaps = true;
  for (std::size_t k = 0; k < lo1.size(); ++k) {
    out.first[k] = (std::max)(lo1[k], lo2[k]);
    out.second[k] = (std::min)(hi1[k], hi2[k]);
    overlaps = overlaps && out.first[k] < out.second[k];
  }
  if (!overlaps) {
    return std::nullopt;
  }
  return out;
}

/// Subtract `[inner_lo, inner_hi)` from `[outer_lo, outer_hi)`.
///
/// Returns at most `2 * ndim` non-overlapping slabs covering the outer rectangle minus the inner
/// one, in the order the axes are peeled: for each axis in turn, the slab below the inner
/// rectangle, then the slab above it, the remaining core shrinking as it goes. Empty slabs are
/// omitted.
///
/// Note:
///     No validation is performed: the four arguments have equal length and the inner rectangle
///     lies inside the outer one.
///
/// Args:
///     outer_lo (const Index&): Outer rectangle lower bound.
///     outer_hi (const Index&): Outer rectangle upper bound.
///     inner_lo (const Index&): Inner rectangle lower bound.
///     inner_hi (const Index&): Inner rectangle upper bound.
///
/// Returns:
///     std::vector<Block>: The slabs.
[[nodiscard]] inline std::vector<Block> peel(const Index& outer_lo, const Index& outer_hi,
                                             const Index& inner_lo, const Index& inner_hi) {
  // With the inner rectangle inside the outer one every extent below is non-negative, so a slab is
  // non-empty exactly when all its extents are positive; testing that avoids forming a product.
  const auto non_empty = [](const Index& lo, const Index& hi) {
    for (std::size_t k = 0; k < lo.size(); ++k) {
      if (!(lo[k] < hi[k])) {
        return false;
      }
    }
    return true;
  };
  std::vector<Block> slabs;
  Index a = outer_lo;
  Index b = outer_hi;
  for (std::size_t k = 0; k < a.size(); ++k) {
    if (a[k] < inner_lo[k]) {
      Index hi_k = b;
      hi_k[k] = inner_lo[k];
      if (non_empty(a, hi_k)) {
        slabs.emplace_back(a, hi_k);
      }
      a[k] = inner_lo[k];
    }
    if (inner_hi[k] < b[k]) {
      Index lo_k = a;
      lo_k[k] = inner_hi[k];
      if (non_empty(lo_k, b)) {
        slabs.emplace_back(lo_k, b);
      }
      b[k] = inner_hi[k];
    }
  }
  return slabs;
}

/// Merge two blocks that differ in exactly one axis and are adjacent there.
///
/// Note:
///     No validation is performed: the four arguments have equal length.
///
/// Args:
///     lo1 (const Index&): First block lower bound.
///     hi1 (const Index&): First block upper bound.
///     lo2 (const Index&): Second block lower bound.
///     hi2 (const Index&): Second block upper bound.
///
/// Returns:
///     std::optional<Block>: The merged block if the two are adjacent and aligned in every other
///     axis, else no value (also when they are equal).
[[nodiscard]] inline std::optional<Block> try_merge(const Index& lo1, const Index& hi1,
                                                    const Index& lo2, const Index& hi2) {
  std::ptrdiff_t merge_axis = -1;
  for (std::size_t k = 0; k < lo1.size(); ++k) {
    if (lo1[k] == lo2[k] && hi1[k] == hi2[k]) {
      continue;
    }
    if (merge_axis >= 0) {
      return std::nullopt;
    }
    if (hi1[k] == lo2[k] || hi2[k] == lo1[k]) {
      merge_axis = static_cast<std::ptrdiff_t>(k);
    } else {
      return std::nullopt;
    }
  }
  if (merge_axis < 0) {
    return std::nullopt;
  }
  Block out{lo1, hi1};
  const auto m = static_cast<std::size_t>(merge_axis);
  out.first[m] = (std::min)(lo1[m], lo2[m]);
  out.second[m] = (std::max)(hi1[m], hi2[m]);
  return out;
}

/// Greedily merge adjacent aligned blocks, then sort.
///
/// Repeats passes until no pair merges: in a pass, each unused block absorbs every later unused
/// block it can merge with, in list order. The result depends on the input order, which is why
/// callers fix it. The final order is lexicographic on `(lo, hi)`.
///
/// Note:
///     No validation is performed: the blocks have equal length and do not overlap.
///
/// Args:
///     blocks (std::vector<Block>): Non-overlapping blocks.
///
/// Returns:
///     std::vector<Block>: Sorted, compacted non-overlapping blocks.
[[nodiscard]] inline std::vector<Block> normalize_blocks(std::vector<Block> blocks) {
  bool changed = blocks.size() > 1;
  while (changed) {
    changed = false;
    std::vector<char> used(blocks.size(), 0);
    std::vector<Block> merged_blocks;
    for (std::size_t i = 0; i < blocks.size(); ++i) {
      if (used[i] != 0) {
        continue;
      }
      Block merged = blocks[i];
      for (std::size_t j = i + 1; j < blocks.size(); ++j) {
        if (used[j] != 0) {
          continue;
        }
        std::optional<Block> result =
            try_merge(merged.first, merged.second, blocks[j].first, blocks[j].second);
        if (result.has_value()) {
          merged = std::move(*result);
          used[j] = 1;
          changed = true;
        }
      }
      merged_blocks.push_back(std::move(merged));
    }
    blocks = std::move(merged_blocks);
  }
  std::sort(blocks.begin(), blocks.end());
  return blocks;
}

/// Set the cells of a row-major mask that the box `[lo, hi)` covers.
///
/// Note:
///     No validation is performed: `lo`, `hi` and `shape` have equal length, `0 <= lo <= hi <=
///     shape` per axis, and the mask holds `prod(shape)` entries.
///
/// Args:
///     cells (std::span<std::uint8_t>): Row-major mask of shape `shape`, written in place.
///     shape (const Index&): Per-axis extents of the mask.
///     lo (const Index&): Box lower bound (inclusive), in mask coordinates.
///     hi (const Index&): Box upper bound (exclusive), in mask coordinates.
///     value (std::uint8_t): The value written.
inline void fill_box(std::span<std::uint8_t> cells, const Index& shape, const Index& lo,
                     const Index& hi, std::uint8_t value) noexcept {
  const std::size_t nd = shape.size();
  for (std::size_t k = 0; k < nd; ++k) {
    if (!(lo[k] < hi[k])) {
      return;
    }
  }
  Index stride(nd, 1);
  for (std::size_t k = nd - 1; k > 0; --k) {
    stride[k - 1] = stride[k] * shape[k];
  }
  Index idx = lo;
  const std::int64_t run = hi[nd - 1] - lo[nd - 1];
  while (true) {
    std::int64_t offset = lo[nd - 1];
    for (std::size_t k = 0; k + 1 < nd; ++k) {
      offset += idx[k] * stride[k];
    }
    std::fill_n(cells.begin() + offset, run, value);
    // Advance the odometer over the axes before the last, last of those fastest.
    bool advanced = false;
    for (std::size_t k = nd - 1; k-- > 0;) {
      if (++idx[k] < hi[k]) {
        advanced = true;
        break;
      }
      idx[k] = lo[k];
    }
    if (!advanced) {
      return;
    }
  }
}

/// Set the cells of a region mask that a block covers.
///
/// The mask is indexed relative to `lo`, so its shape is `hi[k] - lo[k]` per axis. A block that
/// misses the region leaves it untouched.
///
/// Note:
///     No validation is performed: the arguments have equal length and the mask holds
///     `prod(hi - lo)` entries.
///
/// Args:
///     mask (std::span<std::uint8_t>): Row-major region-shaped mask, written in place.
///     lo (const Index&): Region lower bound (inclusive).
///     hi (const Index&): Region upper bound (exclusive).
///     block (const Block&): Rectangle to mark, in the coordinates of `lo` and `hi`.
inline void mark_region_cells(std::span<std::uint8_t> mask, const Index& lo, const Index& hi,
                              const Block& block) {
  const std::optional<Block> inter = rect_intersect(block.first, block.second, lo, hi);
  if (!inter.has_value()) {
    return;
  }
  Index shape(lo.size());
  Index from(lo.size());
  Index to(lo.size());
  for (std::size_t k = 0; k < lo.size(); ++k) {
    shape[k] = hi[k] - lo[k];
    from[k] = inter->first[k] - lo[k];
    to[k] = inter->second[k] - lo[k];
  }
  fill_box(mask, shape, from, to, 1);
}

/// Render the marked cells of a region mask as absolute index tuples.
///
/// At most `max_named_cells` cells are spelled out, in row-major order; any remainder is summarized
/// as a count, so a large rejected region still yields a readable message.
///
/// Note:
///     No validation is performed: the mask holds `prod(shape)` entries and `origin` has the length
///     of `shape`.
///
/// Args:
///     mask (std::span<const std::uint8_t>): Row-major region-shaped mask; nonzero marks a cell.
///     shape (const Index&): Per-axis extents of the mask.
///     origin (const Index&): Index of the region's first cell, added to every mask index.
///
/// Returns:
///     std::string: Comma-separated index tuples, followed by `and N more` when truncated; empty
///     when nothing is marked.
[[nodiscard]] inline std::string name_marked_cells(std::span<const std::uint8_t> mask,
                                                   const Index& shape, const Index& origin) {
  const std::size_t nd = shape.size();
  Index idx(nd, 0);
  std::string named;
  std::int64_t marked = 0;
  for (const std::uint8_t cell : mask) {
    if (cell != 0) {
      if (marked < max_named_cells) {
        Index absolute(nd);
        for (std::size_t k = 0; k < nd; ++k) {
          absolute[k] = idx[k] + origin[k];
        }
        if (marked > 0) {
          named += ", ";
        }
        named += py_repr_ints(absolute);
      }
      ++marked;
    }
    for (std::size_t k = nd; k-- > 0;) {
      if (++idx[k] < shape[k]) {
        break;
      }
      idx[k] = 0;
    }
  }
  const std::int64_t hidden = marked - max_named_cells;
  if (hidden > 0) {
    named += " and " + std::to_string(hidden) + " more";
  }
  return named;
}

}  // namespace pantr::grid::detail
