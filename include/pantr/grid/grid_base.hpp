/// @file grid_base.hpp
/// @brief `GridBase`: the defaults every structured cell grid shares, and the `GridLike` concept.
///
/// A grid is a partition of a parametric domain into axis-aligned box cells with *implicit*
/// (computed, not stored) connectivity. A concrete grid defines only the primitives
///
/// - `ndim()`, `num_cells()`: size metadata;
/// - `cell_bounds(cid)`: the `(lo, hi)` corners of a cell;
/// - `locate(pt)`: the cell containing a point, if any;
/// - `neighbor_across_facet(cid, lfid)`: the cell across a local facet, if any;
/// - `collect_cell_bounds()`: every cell's corners as two `(num_cells, ndim)` arrays;
/// - `locate_many(points, npts)`: a batch of `locate`;
/// - `bvh_lazy()`: the lazy block holding the grid's `BVH`;
///
/// and `GridBase<Derived>` supplies everything else with the defaults of an axis-aligned box grid
/// with `2 * ndim` facets: cell boxes, reference maps, facet bounds, neighbor lists,
/// boundary-facet detection and a box-overlap query backed by a lazily built `BVH`. The
/// derivation is static (CRTP, no virtual call): `GridBase` calls the derived class's members
/// through `Derived`, so a derived class that defines a method of the same name shadows the
/// default, and every default that uses it sees the shadow.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/lazy.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/geometry/aabb.hpp"
#include "pantr/grid/bvh.hpp"
#include "pantr/transform/affine_transform.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pantr::grid {

/// The members a generic grid consumer (`partition_grid`, `cell_quadrature`) needs.
///
/// `type_name` is the name the grid's Python class goes by, for messages that name a grid type.
template <class G>
concept GridLike = requires(const G& g, std::int64_t cid) {
  { g.ndim() } -> std::convertible_to<std::size_t>;
  { g.num_cells() } -> std::convertible_to<std::int64_t>;
  { g.cell_bounds(cid) } -> std::same_as<std::pair<std::vector<double>, std::vector<double>>>;
  { g.collect_cell_bounds() } -> std::same_as<std::pair<NdArray<double>, NdArray<double>>>;
  { G::type_name } -> std::convertible_to<std::string_view>;
};

/// The defaults of an axis-aligned box grid, in terms of the primitives of `Derived`.
///
/// Attributes:
///     (none): `GridBase` holds no state; the lazy `BVH` lives in the derived class's
///         representation and is reached through `Derived::bvh_lazy()`.
template <class Derived>
class GridBase {
 public:
  /// Raise `index_error` if `cid` is out of range.
  ///
  /// Args:
  ///     cid (std::int64_t): Candidate cell identifier.
  ///
  /// Raises:
  ///     index_error: `cid` is negative or `>= num_cells()`.
  void check_cid(std::int64_t cid) const {
    const std::int64_t n = derived().num_cells();
    if (!(0 <= cid && cid < n)) {
      throw index_error("cell id " + std::to_string(cid) + " is out of range [0, " +
                        std::to_string(n) + ").");
    }
  }

  /// Raise `index_error` if `lfid` is not a valid facet of `cid`.
  ///
  /// Args:
  ///     cid (std::int64_t): Cell identifier (validated first).
  ///     lfid (std::int64_t): Candidate local facet identifier.
  ///
  /// Raises:
  ///     index_error: `cid` is out of range, or `lfid` is not in `[0, num_local_facets(cid))`.
  void check_lfid(std::int64_t cid, std::int64_t lfid) const {
    check_cid(cid);
    const std::int64_t n_facets = derived().num_local_facets(cid);
    if (!(0 <= lfid && lfid < n_facets)) {
      throw index_error("local facet id " + std::to_string(lfid) + " is out of range [0, " +
                        std::to_string(n_facets) + ").");
    }
  }

  /// Get the refinement level of cell `cid`: 0 for a flat grid (hierarchical grids shadow this).
  ///
  /// Args:
  ///     cid (std::int64_t): Cell identifier.
  ///
  /// Returns:
  ///     std::int64_t: The level.
  ///
  /// Raises:
  ///     index_error: `cid` is out of range.
  [[nodiscard]] std::int64_t cell_level(std::int64_t cid) const {
    check_cid(cid);
    return 0;
  }

  /// Get the immediate refinement children of cell `cid`: none for a flat grid.
  ///
  /// Args:
  ///     cid (std::int64_t): Cell identifier.
  ///
  /// Returns:
  ///     std::vector<std::int64_t>: Child cell ids; empty for a flat grid.
  ///
  /// Raises:
  ///     index_error: `cid` is out of range.
  [[nodiscard]] std::vector<std::int64_t> child_cells(std::int64_t cid) const {
    check_cid(cid);
    return {};
  }

  /// Get the number of local facets of cell `cid`: `2 * ndim` (an axis-aligned box).
  ///
  /// Args:
  ///     cid (std::int64_t): Cell identifier.
  ///
  /// Returns:
  ///     std::int64_t: `2 * ndim`.
  ///
  /// Raises:
  ///     index_error: `cid` is out of range.
  [[nodiscard]] std::int64_t num_local_facets(std::int64_t cid) const {
    check_cid(cid);
    return 2 * static_cast<std::int64_t>(derived().ndim());
  }

  /// Get `(axis, side)` for local facet `lfid` of cell `cid`.
  ///
  /// The conventional encoding is `lfid = 2 * axis + side`, `side` 0 for the low face and 1 for
  /// the high face.
  ///
  /// Args:
  ///     cid (std::int64_t): Cell identifier.
  ///     lfid (std::int64_t): Local facet identifier in `[0, 2 * ndim)`.
  ///
  /// Returns:
  ///     std::pair<std::int64_t, std::int64_t>: `(axis, side)`.
  ///
  /// Raises:
  ///     index_error: `cid` or `lfid` is out of range.
  [[nodiscard]] std::pair<std::int64_t, std::int64_t> local_facet_axis_side(
      std::int64_t cid, std::int64_t lfid) const {
    check_lfid(cid, lfid);
    return {lfid / 2, lfid % 2};
  }

  /// Get the degenerate `(lo, hi)` box of facet `lfid` of cell `cid`.
  ///
  /// On the facet's normal axis both corners coincide (the cell's lower corner for `side == 0`,
  /// its upper corner for `side == 1`); the other axes span the cell's extent.
  ///
  /// Args:
  ///     cid (std::int64_t): Cell identifier.
  ///     lfid (std::int64_t): Local facet identifier in `[0, 2 * ndim)`.
  ///
  /// Returns:
  ///     std::pair<std::vector<double>, std::vector<double>>: The corners, `ndim` entries each.
  ///
  /// Raises:
  ///     index_error: `cid` or `lfid` is out of range.
  [[nodiscard]] std::pair<std::vector<double>, std::vector<double>> local_facet_bounds(
      std::int64_t cid, std::int64_t lfid) const {
    const auto [axis, side] = derived().local_facet_axis_side(cid, lfid);
    auto [lo, hi] = derived().cell_bounds(cid);
    const auto a = static_cast<std::size_t>(axis);
    if (side == 0) {
      hi[a] = lo[a];
    } else {
      lo[a] = hi[a];
    }
    return {std::move(lo), std::move(hi)};
  }

  /// Report whether facet `lfid` of cell `cid` is on the grid's outer boundary: no neighbor
  /// across it.
  ///
  /// Args:
  ///     cid (std::int64_t): Cell identifier.
  ///     lfid (std::int64_t): Local facet identifier in `[0, num_local_facets(cid))`.
  ///
  /// Returns:
  ///     bool: Whether no neighboring cell shares the facet.
  ///
  /// Raises:
  ///     index_error: `cid` or `lfid` is out of range.
  [[nodiscard]] bool is_mesh_boundary_facet(std::int64_t cid, std::int64_t lfid) const {
    return !derived().neighbor_across_facet(cid, lfid).has_value();
  }

  /// Get the facet-neighbor cell ids of `cid`: `neighbor_across_facet` over every local facet,
  /// boundary facets dropped.
  ///
  /// Args:
  ///     cid (std::int64_t): Cell identifier.
  ///
  /// Returns:
  ///     std::vector<std::int64_t>: Neighboring cell ids, in local-facet order.
  ///
  /// Raises:
  ///     index_error: `cid` is out of range.
  [[nodiscard]] std::vector<std::int64_t> neighbors(std::int64_t cid) const {
    std::vector<std::int64_t> result;
    const std::int64_t n_facets = derived().num_local_facets(cid);
    for (std::int64_t lfid = 0; lfid < n_facets; ++lfid) {
      const std::optional<std::int64_t> neighbor = derived().neighbor_across_facet(cid, lfid);
      if (neighbor.has_value()) {
        result.push_back(*neighbor);
      }
    }
    return result;
  }

  /// Get every facet on the grid's outer boundary as `(cid, lfid)` rows.
  ///
  /// Only the outer (domain) boundary: a facet shared with any neighboring cell is excluded,
  /// whether that neighbor is conforming, coarser or finer. The default enumerates
  /// `is_mesh_boundary_facet` over every cell and local facet, costing `num_cells * 2 * ndim`
  /// neighbor queries; a derived class with structure shadows it.
  ///
  /// Returns:
  ///     NdArray<std::int64_t>: Shape `(n, 2)`, rows `(cid, lfid)` with `lfid = 2 * axis + side`,
  ///     sorted lexicographically by `(cid, lfid)`.
  [[nodiscard]] NdArray<std::int64_t> boundary_facets() const {
    std::vector<std::int64_t> rows;
    const std::int64_t n = derived().num_cells();
    for (std::int64_t cid = 0; cid < n; ++cid) {
      const std::int64_t n_facets = derived().num_local_facets(cid);
      for (std::int64_t lfid = 0; lfid < n_facets; ++lfid) {
        if (derived().is_mesh_boundary_facet(cid, lfid)) {
          rows.push_back(cid);
          rows.push_back(lfid);
        }
      }
    }
    NdArray<std::int64_t> out({rows.size() / 2, 2}, for_overwrite);
    std::copy(rows.begin(), rows.end(), out.data());
    return out;
  }

  /// Get all active cells sharing facet `lfid` of `cid`.
  ///
  /// For a conforming grid this is `neighbor_across_facet` as a list; a hierarchical grid shadows
  /// it, since a coarse face may abut several fine cells.
  ///
  /// Args:
  ///     cid (std::int64_t): Cell identifier.
  ///     lfid (std::int64_t): Local facet identifier in `[0, num_local_facets(cid))`.
  ///
  /// Returns:
  ///     std::vector<std::int64_t>: All neighboring cell ids across the facet; empty on the outer
  ///     boundary.
  ///
  /// Raises:
  ///     index_error: `cid` or `lfid` is out of range.
  [[nodiscard]] std::vector<std::int64_t> hanging_neighbors(std::int64_t cid,
                                                            std::int64_t lfid) const {
    const std::optional<std::int64_t> nbr = derived().neighbor_across_facet(cid, lfid);
    if (!nbr.has_value()) {
      return {};
    }
    return {*nbr};
  }

  /// Get cell `cid`'s axis-aligned bounding box.
  ///
  /// Args:
  ///     cid (std::int64_t): Cell identifier.
  ///
  /// Returns:
  ///     geometry::AABB: The box with the corners of `cell_bounds(cid)`.
  ///
  /// Raises:
  ///     index_error: `cid` is out of range.
  [[nodiscard]] geometry::AABB cell_aabb(std::int64_t cid) const {
    const auto [lo, hi] = derived().cell_bounds(cid);
    return {lo, hi};
  }

  /// Get the affine map from the unit cube to cell `cid`, `T(u) = diag(hi - lo) u + lo`.
  ///
  /// Args:
  ///     cid (std::int64_t): Cell identifier.
  ///
  /// Returns:
  ///     transform::AffineTransform: The push-forward from `[0, 1]^ndim` to the cell.
  ///
  /// Raises:
  ///     index_error: `cid` is out of range.
  [[nodiscard]] transform::AffineTransform reference_map(std::int64_t cid) const {
    const auto [lo, hi] = derived().cell_bounds(cid);
    const std::size_t n = lo.size();
    std::vector<double> diag(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      diag[i * n + i] = hi[i] - lo[i];
    }
    const std::size_t matrix_shape[2] = {n, n};
    const std::size_t translation_shape[1] = {n};
    return {diag, matrix_shape, std::span<const double>(lo), translation_shape};
  }

  /// Get the `BVH` over the grid's cell boxes, built on first call from `collect_cell_bounds` and
  /// kept for the grid's lifetime.
  ///
  /// Building materializes `O(num_cells)` node arrays, so it is deferred until a query needs it:
  /// an un-queried grid never pays this cost. Concurrent first calls build it once.
  ///
  /// Returns:
  ///     const BVH&: The grid's spatial index, valid while the grid (or a copy sharing its
  ///     representation) lives.
  [[nodiscard]] const BVH& cell_bvh() const {
    return derived().bvh_lazy().get([this] {
      const auto [cell_lo, cell_hi] = derived().collect_cell_bounds();
      return BVH::from_cell_bounds(cell_lo.flat(), cell_hi.flat(), cell_lo.shape(),
                                   cell_hi.shape());
    });
  }

  /// Report whether the `BVH` has been built yet.
  ///
  /// Returns:
  ///     bool: Whether `cell_bvh()` has been called.
  [[nodiscard]] bool bvh_built() const noexcept { return derived().bvh_lazy().ready(); }

  /// Get the ids of every cell whose box overlaps `aabb`, backed by `cell_bvh()`.
  ///
  /// The overlap test is inclusive on every axis, so cells touching `aabb` on any face, edge or
  /// corner are included.
  ///
  /// Args:
  ///     aabb (const geometry::AABB&): Query box; its dimension must match `ndim()`.
  ///
  /// Returns:
  ///     std::vector<std::int64_t>: Overlapping cell ids (unordered).
  ///
  /// Raises:
  ///     value_error: `aabb.ndim() != ndim()`.
  [[nodiscard]] std::vector<std::int64_t> query_aabb(const geometry::AABB& aabb) const {
    return cell_bvh().query_aabb(aabb);
  }

 protected:
  GridBase() = default;
  GridBase(const GridBase&) = default;
  GridBase& operator=(const GridBase&) = default;
  ~GridBase() = default;

 private:
  [[nodiscard]] const Derived& derived() const noexcept {
    return static_cast<const Derived&>(*this);
  }
};

}  // namespace pantr::grid
