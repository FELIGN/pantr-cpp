/// @file partition.hpp
/// @brief `Partition` (a per-cell owner assignment) and `partition_grid`, a native partitioner.
///
/// A `Partition` records, for every cell of a grid, which rank owns it, or -1 for an inactive cell
/// excluded from the partition. `partition_grid` produces one without any external dependency or
/// MPI, with two backends and an `automatic` dispatch:
///
/// - `block`: a Cartesian split of a `TensorProductGrid` into contiguous, aspect-ratio-aware box
///   subdomains. It throws rather than produce empty ranks when the part count does not factor
///   onto the axes, and supports neither cell weights nor activity.
/// - `rcb`: recursive coordinate bisection on cell centroids. Geometric and grid-agnostic,
///   weight-aware (it balances total cell cost) and activity-aware (inactive cells get owner -1).
/// - `automatic`: `block` when the grid is a `TensorProductGrid`, no weights or activity are given
///   and the part count factors onto the axes; otherwise `rcb`.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/grid/detail/checked.hpp"
#include "pantr/grid/detail/fsum.hpp"
#include "pantr/grid/detail/py_text.hpp"
#include "pantr/grid/grid_base.hpp"
#include "pantr/grid/tensor_product_grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pantr::grid {

namespace detail {

/// The shared representation of a `Partition`.
///
/// Attributes:
///     cell_owner (std::vector<std::int32_t>): Owner rank per cell, -1 for an inactive cell.
///     n_parts (std::int64_t): Number of parts (ranks), at least 1.
struct PartitionRep {
  std::vector<std::int32_t> cell_owner;
  std::int64_t n_parts = 1;
};

}  // namespace detail

/// A per-cell owner assignment over a grid's cells.
///
/// Records, for every cell, the rank that owns it, or -1 for an inactive cell excluded from the
/// partition (for example an exterior or trimmed cell). Immutable: the owner array is copied in at
/// construction and shared by every copy. The Python wrapper checks that the owner array is 1-D
/// integer and converts it to `int32` before calling.
///
/// Attributes:
///     rep_ (std::shared_ptr<const detail::PartitionRep>): The owners and the part count.
class Partition {
 public:
  /// Build a partition from a per-cell owner array.
  ///
  /// Args:
  ///     cell_owner (std::vector<std::int32_t>): Per-cell owner ranks, -1 for inactive cells.
  ///     n_parts (std::int64_t): Number of parts (ranks); must be at least 1.
  ///
  /// Raises:
  ///     value_error: `n_parts < 1`, or an owner is outside `[-1, n_parts)`.
  Partition(std::vector<std::int32_t> cell_owner, std::int64_t n_parts) {
    if (n_parts < 1) {
      throw value_error("n_parts must be >= 1; got " + std::to_string(n_parts) + ".");
    }
    if (!cell_owner.empty()) {
      const auto [mn, mx] = std::minmax_element(cell_owner.begin(), cell_owner.end());
      if (*mn < -1 || static_cast<std::int64_t>(*mx) >= n_parts) {
        throw value_error("cell_owner values must lie in [-1, " + std::to_string(n_parts) +
                          "); got range [" + std::to_string(*mn) + ", " + std::to_string(*mx) +
                          "].");
      }
    }
    auto rep = std::make_shared<detail::PartitionRep>();
    rep->cell_owner = std::move(cell_owner);
    rep->n_parts = n_parts;
    rep_ = std::move(rep);
  }

  /// Get the per-cell owner array.
  ///
  /// Returns:
  ///     std::span<const std::int32_t>: `n_cells()` owners; -1 for inactive cells.
  [[nodiscard]] std::span<const std::int32_t> cell_owner() const noexcept {
    return rep_->cell_owner;
  }

  /// Get the number of parts (ranks).
  ///
  /// Returns:
  ///     std::int64_t: The part count, at least 1.
  [[nodiscard]] std::int64_t n_parts() const noexcept { return rep_->n_parts; }

  /// Get the total number of cells, active and inactive.
  ///
  /// Returns:
  ///     std::int64_t: The length of `cell_owner()`.
  [[nodiscard]] std::int64_t n_cells() const noexcept {
    return static_cast<std::int64_t>(rep_->cell_owner.size());
  }

  /// Get a mask of the active cells (owned by some rank).
  ///
  /// Returns:
  ///     std::vector<std::uint8_t>: `n_cells()` entries, 1 where the cell owner is not -1.
  [[nodiscard]] std::vector<std::uint8_t> active_mask() const {
    std::vector<std::uint8_t> mask(rep_->cell_owner.size());
    for (std::size_t i = 0; i < mask.size(); ++i) {
      mask[i] = rep_->cell_owner[i] >= 0 ? 1 : 0;
    }
    return mask;
  }

  /// Get the flat ids of the cells owned by `rank`, ascending.
  ///
  /// Args:
  ///     rank (std::int64_t): Owner rank in `[0, n_parts())`.
  ///
  /// Returns:
  ///     std::vector<std::int64_t>: Sorted cell ids with `cell_owner == rank`.
  ///
  /// Raises:
  ///     value_error: `rank` is outside `[0, n_parts())`.
  [[nodiscard]] std::vector<std::int64_t> owned_cells(std::int64_t rank) const {
    if (!(0 <= rank && rank < rep_->n_parts)) {
      throw value_error("rank must be in [0, " + std::to_string(rep_->n_parts) + "); got " +
                        std::to_string(rank) + ".");
    }
    std::vector<std::int64_t> out;
    for (std::size_t i = 0; i < rep_->cell_owner.size(); ++i) {
      if (static_cast<std::int64_t>(rep_->cell_owner[i]) == rank) {
        out.push_back(static_cast<std::int64_t>(i));
      }
    }
    return out;
  }

  /// Get the shared representation.
  ///
  /// Returns:
  ///     const std::shared_ptr<const detail::PartitionRep>&: The representation.
  [[nodiscard]] const std::shared_ptr<const detail::PartitionRep>& rep() const noexcept {
    return rep_;
  }

 private:
  std::shared_ptr<const detail::PartitionRep> rep_;
};

/// The partitioning backends of `partition_grid`.
enum class PartitionBackend { automatic, block, rcb };

namespace detail {

/// Narrow an owner rank to `int32`, refusing one that does not fit.
///
/// Args:
///     rank (std::int64_t): A rank in `[0, n_parts)`.
///
/// Returns:
///     std::int32_t: The rank.
///
/// Raises:
///     std::overflow_error: `rank` exceeds the `int32` range (a part count above `2^31 - 1`).
[[nodiscard]] inline std::int32_t to_owner(std::int64_t rank) {
  if (rank > (std::numeric_limits<std::int32_t>::max)()) {
    throw std::overflow_error("partition_grid: the rank " + std::to_string(rank) +
                              " does not fit in a 32-bit owner.");
  }
  return static_cast<std::int32_t>(rank);
}

/// Get all positive divisors of `n`, ascending.
///
/// Args:
///     n (std::int64_t): A positive integer.
///
/// Returns:
///     std::vector<std::int64_t>: The sorted divisors of `n` (including 1 and `n`).
[[nodiscard]] inline std::vector<std::int64_t> divisors(std::int64_t n) {
  std::vector<std::int64_t> low_half;
  std::vector<std::int64_t> high_half;
  // `i <= n / i` is `i * i <= n` without the product, which overflows for n near the int64 limit.
  for (std::int64_t i = 1; i <= n / i; ++i) {
    if (n % i == 0) {
      low_half.push_back(i);
      if (i != n / i) {
        high_half.push_back(n / i);
      }
    }
  }
  low_half.insert(low_half.end(), high_half.rbegin(), high_half.rend());
  return low_half;
}

/// Get the variance of the log block extents (lower is more cube-like).
///
/// The block extent along an axis is `cells_per_axis[d] / blocks[d]`, the mean number of cells per
/// block along that axis. Log extents make the metric scale-invariant, so the minimizer favors
/// subdomains that are as cube-like as possible. The sums are exactly rounded (`fsum`), so a tie
/// between two factorizations is broken as it is by the Python implementation.
///
/// Args:
///     cells_per_axis (std::span<const std::int64_t>): Cell count along each axis.
///     blocks (std::span<const std::int64_t>): Candidate block count along each axis.
///
/// Returns:
///     double: The variance of `log(cells_per_axis[d] / blocks[d])` over the axes.
///
/// Raises:
///     value_error: The two spans differ in length (Python's strict `zip`).
[[nodiscard]] inline double extent_imbalance(std::span<const std::int64_t> cells_per_axis,
                                             std::span<const std::int64_t> blocks) {
  if (blocks.size() != cells_per_axis.size()) {
    throw value_error(blocks.size() < cells_per_axis.size()
                          ? "zip() argument 2 is shorter than argument 1"
                          : "zip() argument 2 is longer than argument 1");
  }
  std::vector<double> logs(cells_per_axis.size());
  for (std::size_t d = 0; d < logs.size(); ++d) {
    logs[d] = std::log(static_cast<double>(cells_per_axis[d]) / static_cast<double>(blocks[d]));
  }
  const double n = static_cast<double>(logs.size());
  const double mean = fsum(logs) / n;
  std::vector<double> sq(logs.size());
  for (std::size_t d = 0; d < logs.size(); ++d) {
    sq[d] = std::pow(logs[d] - mean, 2.0);
  }
  return fsum(sq) / n;
}

/// Factor `n_parts` into a per-axis block count proportional to cell counts.
///
/// Searches all factorizations of `n_parts` into `cells_per_axis.size()` ordered factors with
/// `blocks[d] <= cells_per_axis[d]` and returns the one whose block extents are the most uniform
/// (minimizing the variance of their logarithms), the first of equals in enumeration order. The
/// search is exact, so it finds a valid factorization whenever one exists; the divisor count of
/// `n_parts` is small, so the enumeration is cheap.
///
/// Args:
///     cells_per_axis (std::span<const std::int64_t>): Cell count along each axis, each at least 1.
///     n_parts (std::int64_t): Number of parts, at least 1.
///
/// Returns:
///     std::vector<std::int64_t>: Block count per axis; the product equals `n_parts` and every
///     entry satisfies `blocks[d] <= cells_per_axis[d]`.
///
/// Raises:
///     value_error: No such factorization exists: `n_parts` is too large or factors too coarsely
///         for this grid, so some rank would get no cells.
///     index_error: `cells_per_axis` is empty (Python's `IndexError`, from reading axis 0).
[[nodiscard]] inline std::vector<std::int64_t> factor_blocks(
    std::span<const std::int64_t> cells_per_axis, std::int64_t n_parts) {
  if (cells_per_axis.empty()) {
    throw index_error("tuple index out of range");
  }
  const std::size_t ndim = cells_per_axis.size();
  std::optional<std::vector<std::int64_t>> best;
  double best_score = std::numeric_limits<double>::infinity();
  std::vector<std::int64_t> acc;

  // Enumerate in the order of the divisor lists; the strict `<` keeps the first of equal scores.
  auto recurse = [&](auto&& self, std::size_t axis, std::int64_t remaining) -> void {
    if (axis == ndim - 1) {
      if (remaining <= cells_per_axis[axis]) {
        acc.push_back(remaining);
        const double score = extent_imbalance(cells_per_axis, acc);
        if (score < best_score) {
          best_score = score;
          best = acc;
        }
        acc.pop_back();
      }
      return;
    }
    for (const std::int64_t divisor : divisors(remaining)) {
      if (divisor <= cells_per_axis[axis]) {
        acc.push_back(divisor);
        self(self, axis + 1, remaining / divisor);
        acc.pop_back();
      }
    }
  };
  recurse(recurse, 0, n_parts);
  if (!best.has_value()) {
    throw value_error("cannot factor n_parts=" + std::to_string(n_parts) +
                      " across axes with cells_per_axis=" + py_repr_ints(cells_per_axis) +
                      " without leaving a rank empty. "
                      "Use a part count that divides the grid better, or the 'rcb' backend.");
  }
  return *best;
}

/// Compute the per-cell owner array for the Cartesian `block` backend.
///
/// Factors `n_parts` into a per-axis block count (`factor_blocks`), then maps each cell's
/// multi-index to its block by the balanced contiguous split `block_d = cell_d * blocks_d /
/// cells_d` (integer division) and flattens the block multi-index (C-order, as cell ids are) to
/// the owner rank.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): `n_parts >= 1` is not checked.
///     For general use, call `partition_grid`.
///
/// Args:
///     grid (const TensorProductGrid&): The grid to partition.
///     n_parts (std::int64_t): Number of parts, at least 1.
///
/// Returns:
///     std::vector<std::int32_t>: `num_cells()` owner ranks in `[0, n_parts)`, in cell-id order.
///
/// Raises:
///     value_error: `n_parts` cannot be factored onto the grid's axes (see `factor_blocks`).
///     std::overflow_error: A block coordinate product or a rank overflows.
[[nodiscard]] inline std::vector<std::int32_t> block_partition(const TensorProductGrid& grid,
                                                               std::int64_t n_parts) {
  const std::span<const std::int64_t> cpa = grid.cells_per_axis();
  const std::vector<std::int64_t> bpa = factor_blocks(cpa, n_parts);
  const std::size_t nd = cpa.size();
  const auto n = static_cast<std::size_t>(grid.num_cells());
  // Per axis, each cell index's contribution to the owner rank: its block `i * b_d / c_d` times
  // the C-order stride of the block grid. Tabulated once, so the integer divisions and overflow
  // checks stay out of the per-cell loop; the ranks are the same integers.
  std::vector<std::vector<std::int64_t>> contribution(nd);
  std::int64_t block_stride = 1;
  for (std::size_t d = nd; d-- > 0;) {
    contribution[d].resize(static_cast<std::size_t>(cpa[d]));
    for (std::int64_t i = 0; i < cpa[d]; ++i) {
      const std::int64_t block = checked_mul(i, bpa[d]) / cpa[d];
      contribution[d][static_cast<std::size_t>(i)] = checked_mul(block, block_stride);
    }
    block_stride = checked_mul(block_stride, bpa[d]);
  }
  std::vector<std::int32_t> owner(n);
  std::vector<std::int64_t> multi(nd, 0);
  for (std::size_t c = 0; c < n; ++c) {
    std::int64_t rank = 0;  // a sum of per-axis terms, < prod(bpa) = n_parts: no overflow
    for (std::size_t d = 0; d < nd; ++d) {
      rank += contribution[d][static_cast<std::size_t>(multi[d])];
    }
    owner[c] = to_owner(rank);
    for (std::size_t d = nd; d-- > 0;) {
      if (++multi[d] < cpa[d]) {
        break;
      }
      multi[d] = 0;
    }
  }
  return owner;
}

/// Partition a grid by recursive coordinate bisection of its active cells.
///
/// Operates on the centroids of the active cells (the midpoints of `collect_cell_bounds`). It is
/// geometric, so it works on any grid, and balances total weight (uniform without weights).
/// Inactive cells get owner -1. A set of cells is split into the parts `[part_lo, part_hi)` by
/// weight, cutting the axis of largest centroid spread (the first on a tie) at the weighted split
/// point, clamped so that each side keeps at least as many cells as parts (no rank is left
/// empty); the cells are ordered along the axis by a stable sort.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): the lengths of `weights` and
///     `active` (`num_cells()` each), the weights' finiteness and sign, and `n_parts >= 1` are
///     checked by the public `partition_grid`, not here. For general use, call `partition_grid`.
///
/// Args:
///     grid (const G&): The grid to partition.
///     n_parts (std::int64_t): Number of parts, at least 1.
///     weights (std::optional<std::span<const double>>): Per-cell weights (finite, non-negative),
///         or no value for uniform.
///     active (std::optional<std::span<const std::uint8_t>>): Per-cell activity (nonzero is
///         active), or no value for all active.
///
/// Returns:
///     std::vector<std::int32_t>: `num_cells()` owners, -1 for inactive cells, otherwise in
///     `[0, n_parts)`.
///
/// Raises:
///     value_error: `n_parts` exceeds the number of active cells.
template <GridLike G>
[[nodiscard]] std::vector<std::int32_t> rcb_partition(
    const G& grid, std::int64_t n_parts, std::optional<std::span<const double>> weights,
    std::optional<std::span<const std::uint8_t>> active) {
  const auto n_cells = static_cast<std::size_t>(grid.num_cells());
  std::vector<std::int64_t> active_idx;
  if (active.has_value()) {
    for (std::size_t c = 0; c < n_cells; ++c) {
      if ((*active)[c] != 0) {
        active_idx.push_back(static_cast<std::int64_t>(c));
      }
    }
  } else {
    active_idx.resize(n_cells);
    std::iota(active_idx.begin(), active_idx.end(), std::int64_t{0});
  }
  const std::size_t n_active = active_idx.size();
  if (n_parts > static_cast<std::int64_t>(n_active)) {
    throw value_error("n_parts=" + std::to_string(n_parts) +
                      " exceeds the number of active cells (" + std::to_string(n_active) +
                      "); cannot assign every rank a cell.");
  }

  const std::size_t nd = grid.ndim();
  const auto [cell_lo, cell_hi] = grid.collect_cell_bounds();
  std::vector<double> centroids(n_active * nd);
  std::vector<double> w_active(n_active, 1.0);
  for (std::size_t a = 0; a < n_active; ++a) {
    const auto c = static_cast<std::size_t>(active_idx[a]);
    for (std::size_t d = 0; d < nd; ++d) {
      centroids[a * nd + d] = 0.5 * (cell_lo.data()[c * nd + d] + cell_hi.data()[c * nd + d]);
    }
    if (weights.has_value()) {
      w_active[a] = (*weights)[c];
    }
  }

  std::vector<std::int32_t> owner_active(n_active, 0);
  // `order` holds positions into the active list; a bisection step reorders its own subrange in
  // place and hands the two halves to the recursion, as the Python version hands on `order[:split]`
  // and `order[split:]`.
  std::vector<std::int64_t> order(n_active);
  std::iota(order.begin(), order.end(), std::int64_t{0});
  std::vector<double> cumw;

  auto bisect = [&](auto&& self, std::size_t first, std::size_t last, std::int64_t part_lo,
                    std::int64_t part_hi) -> void {
    const std::int64_t k = part_hi - part_lo;
    if (k == 1) {
      const std::int32_t rank = to_owner(part_lo);
      for (std::size_t i = first; i < last; ++i) {
        owner_active[static_cast<std::size_t>(order[i])] = rank;
      }
      return;
    }
    // Axis of largest centroid spread, the first on a tie.
    std::size_t axis = 0;
    double best_extent = 0.0;
    for (std::size_t d = 0; d < nd; ++d) {
      double mn = centroids[static_cast<std::size_t>(order[first]) * nd + d];
      double mx = mn;
      for (std::size_t i = first + 1; i < last; ++i) {
        const double v = centroids[static_cast<std::size_t>(order[i]) * nd + d];
        mn = (std::min)(mn, v);
        mx = (std::max)(mx, v);
      }
      const double extent = mx - mn;
      if (d == 0 || extent > best_extent) {
        best_extent = extent;
        axis = d;
      }
    }
    std::stable_sort(order.begin() + static_cast<std::ptrdiff_t>(first),
                     order.begin() + static_cast<std::ptrdiff_t>(last),
                     [&](std::int64_t a, std::int64_t b) {
                       return centroids[static_cast<std::size_t>(a) * nd + axis] <
                              centroids[static_cast<std::size_t>(b) * nd + axis];
                     });
    // Cumulative weights, summed sequentially as np.cumsum does.
    cumw.resize(last - first);
    double running = 0.0;
    for (std::size_t i = first; i < last; ++i) {
      const double w = w_active[static_cast<std::size_t>(order[i])];
      running = i == first ? w : running + w;
      cumw[i - first] = running;
    }
    const std::int64_t k_left = k / 2;
    // (total * k_left) / k, evaluated in this order as Python evaluates `total * k_left / k`.
    const double target = cumw.back() * static_cast<double>(k_left) / static_cast<double>(k);
    const auto at = std::lower_bound(cumw.begin(), cumw.end(), target) - cumw.begin();
    std::int64_t split = static_cast<std::int64_t>(at) + 1;
    const auto count = static_cast<std::int64_t>(last - first);
    split = (std::max)(k_left, (std::min)(split, count - (k - k_left)));
    const auto mid = first + static_cast<std::size_t>(split);
    self(self, first, mid, part_lo, part_lo + k_left);
    self(self, mid, last, part_lo + k_left, part_hi);
  };
  bisect(bisect, std::size_t{0}, n_active, std::int64_t{0}, n_parts);

  std::vector<std::int32_t> owner(n_cells, -1);
  for (std::size_t a = 0; a < n_active; ++a) {
    owner[static_cast<std::size_t>(active_idx[a])] = owner_active[a];
  }
  return owner;
}

}  // namespace detail

/// Partition a grid's cells into `n_parts` rank subdomains.
///
/// Returns a `Partition` for the serial, communication-free distribution of a grid. The weights
/// and the activity mask let a consumer (for example an immersed or unfitted code) drive the
/// partition without pantr storing any classification: per-cell assembly cost via `weights` (the
/// `rcb` backend balances total weight), and an active subset via `active` (inactive cells get
/// owner -1). The `block` backend supports neither.
///
/// The Python wrapper validates the backend name, converts `weights` to `float64` and `active` to
/// a boolean array, and checks that both are 1-D (the shape messages of other ranks spell numpy
/// shapes); here a 1-D array of the wrong length is refused with the same message.
///
/// Args:
///     grid (const G&): The grid to partition. `block` requires a `TensorProductGrid`; `rcb`
///         accepts any grid.
///     n_parts (std::int64_t): Number of parts (ranks); must be at least 1.
///     backend (PartitionBackend): The backend.
///     weights (std::optional<std::span<const double>>): Per-cell cost, `num_cells()` entries,
///         finite and non-negative; no value means uniform. Used by `rcb`, rejected by `block`.
///     active (std::optional<std::span<const std::uint8_t>>): Per-cell activity, `num_cells()`
///         entries, nonzero meaning active; no value means all active. Used by `rcb`, rejected by
///         `block`.
///
/// Returns:
///     Partition: A per-cell owner assignment with `n_parts` parts. Cells excluded by `active`
///     have owner -1; every active cell is assigned a rank in `[0, n_parts)` and no rank is left
///     empty.
///
/// Raises:
///     value_error: `n_parts < 1`; `block` is used on a grid that is not a `TensorProductGrid`
///         or with weights or activity; `n_parts` cannot be factored onto the axes (`block`);
///         `weights` or `active` has the wrong length or invalid values; or `n_parts` exceeds the
///         number of active cells (`rcb`).
///     std::overflow_error: A rank does not fit in 32 bits.
template <GridLike G>
[[nodiscard]] Partition partition_grid(
    const G& grid, std::int64_t n_parts, PartitionBackend backend = PartitionBackend::automatic,
    std::optional<std::span<const double>> weights = std::nullopt,
    std::optional<std::span<const std::uint8_t>> active = std::nullopt) {
  if (n_parts < 1) {
    throw value_error("n_parts must be >= 1; got " + std::to_string(n_parts) + ".");
  }
  const auto n_cells = static_cast<std::size_t>(grid.num_cells());
  if (weights.has_value()) {
    if (weights->size() != n_cells) {
      const std::size_t got[1] = {weights->size()};
      throw value_error("cell_weights must have shape (" + std::to_string(n_cells) + ",); got " +
                        py_repr(got) + ".");
    }
    const bool bad = std::any_of(weights->begin(), weights->end(),
                                 [](double w) { return !std::isfinite(w) || w < 0.0; });
    if (bad) {
      throw value_error("cell_weights must be finite and non-negative.");
    }
  }
  if (active.has_value()) {
    if (active->size() != n_cells) {
      const std::size_t got[1] = {active->size()};
      throw value_error("cell_active must have shape (" + std::to_string(n_cells) + ",); got " +
                        py_repr(got) + ".");
    }
    if (!std::any_of(active->begin(), active->end(), [](std::uint8_t a) { return a != 0; })) {
      throw value_error("cell_active must mark at least one cell active.");
    }
  }

  std::vector<std::int32_t> owner;
  if (backend == PartitionBackend::block) {
    if constexpr (std::is_same_v<G, TensorProductGrid>) {
      if (weights.has_value() || active.has_value()) {
        throw value_error(
            "the 'block' backend does not support cell_weights or cell_active; "
            "use backend='rcb' (or 'auto').");
      }
      owner = detail::block_partition(grid, n_parts);
    } else {
      throw value_error("the 'block' backend requires a TensorProductGrid; got " +
                        std::string(G::type_name) + ".");
    }
  } else if (backend == PartitionBackend::rcb) {
    owner = detail::rcb_partition(grid, n_parts, weights, active);
  } else {
    bool done = false;
    if constexpr (std::is_same_v<G, TensorProductGrid>) {
      if (!weights.has_value() && !active.has_value()) {
        // Prefer the cheap Cartesian split; fall back to rcb when n_parts does not factor onto
        // the axes (awkward or prime part counts), as the Python version does.
        try {
          owner = detail::block_partition(grid, n_parts);
          done = true;
        } catch (const value_error&) {
          owner = detail::rcb_partition(grid, n_parts, std::nullopt, std::nullopt);
          done = true;
        }
      }
    }
    if (!done) {
      owner = detail::rcb_partition(grid, n_parts, weights, active);
    }
  }
  return Partition(std::move(owner), n_parts);
}

}  // namespace pantr::grid
