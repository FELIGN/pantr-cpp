/// @file cell_quadrature.hpp
/// @brief `cell_quadrature`: push a reference rule on the unit cube forward onto grid cells.
///
/// A quadrature rule lives on the reference unit cube `[0, 1]^ndim`. `cell_quadrature` pushes it
/// forward onto a grid's cells with the per-cell affine map `T(u) = lo + (hi - lo) * u`: points
/// are mapped by `T` and weights are scaled by the cell volume `prod(hi - lo)`, the Jacobian
/// determinant of `T`. The reference weights sum to 1, so the mapped weights of a cell sum to that
/// cell's volume. This is the uncut-cell quadrature bridge: a downstream consumer takes this rule
/// for interior cells and substitutes its own on cells flagged as cut.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/grid/grid_base.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::grid {

/// Map a reference quadrature rule onto grid cells.
///
/// For each selected cell, the rule's unit-cube points are affinely mapped into the cell box and
/// its weights are scaled by the cell volume. Only the requested cells are computed. The Python
/// wrapper checks that `cells` is a 1-D integer array and that the rule is a `QuadratureRule`.
///
/// The point is `lo + span * p` formed as a product and then a sum in separate statements, so a
/// compiler that contracts within an expression does not fuse them: the Python implementation
/// rounds the product first. The volume is the product of the spans taken in axis order.
///
/// Args:
///     grid (const G&): The grid whose cells are integrated; `grid.ndim()` must equal
///         `rule_ndim`.
///     rule_points (std::span<const double>): Reference points, row-major `(npts, rule_ndim)`.
///     npts (std::size_t): Number of reference points.
///     rule_ndim (std::size_t): Dimension of the rule.
///     rule_weights (std::span<const double>): `npts` reference weights.
///     cells (std::optional<std::span<const std::int64_t>>): Cell ids to map the rule onto, each
///         in `[0, num_cells)`; order and duplicates are kept. No value selects every cell in id
///         order.
///
/// Returns:
///     std::pair<NdArray<double>, NdArray<double>>: `(points, weights)`, `points` of shape
///     `(num_selected, npts, ndim)` and `weights` of shape `(num_selected, npts)`, ordered to
///     match `cells`.
///
/// Raises:
///     value_error: `rule_ndim != grid.ndim()`, an id is outside `[0, num_cells)`, or the rule's
///         arrays do not have the lengths `npts * rule_ndim` and `npts`.
template <GridLike G>
[[nodiscard]] std::pair<NdArray<double>, NdArray<double>> cell_quadrature(
    const G& grid, std::span<const double> rule_points, std::size_t npts, std::size_t rule_ndim,
    std::span<const double> rule_weights,
    std::optional<std::span<const std::int64_t>> cells = std::nullopt) {
  const std::size_t nd = grid.ndim();
  if (rule_ndim != nd) {
    throw value_error("rule.ndim (" + std::to_string(rule_ndim) + ") must match grid.ndim (" +
                      std::to_string(nd) + ").");
  }
  if (rule_points.size() != npts * rule_ndim || rule_weights.size() != npts) {
    throw value_error("cell_quadrature: the rule holds " + std::to_string(rule_points.size()) +
                      " point values and " + std::to_string(rule_weights.size()) +
                      " weights, but " + std::to_string(npts) + " points of dimension " +
                      std::to_string(rule_ndim) + " need " + std::to_string(npts * rule_ndim) +
                      " and " + std::to_string(npts) + ".");
  }
  const std::int64_t num_cells = grid.num_cells();
  if (cells.has_value() && !cells->empty()) {
    const auto [mn, mx] = std::minmax_element(cells->begin(), cells->end());
    if (*mn < 0 || *mx >= num_cells) {
      throw value_error("cells must lie in [0, " + std::to_string(num_cells) + "); got range [" +
                        std::to_string(*mn) + ", " + std::to_string(*mx) + "].");
    }
  }
  const std::size_t n_sel = cells.has_value() ? cells->size() : static_cast<std::size_t>(num_cells);

  NdArray<double> points({n_sel, npts, nd}, for_overwrite);
  NdArray<double> weights({n_sel, npts}, for_overwrite);

  // Without a selection the corners of every cell come from one pass over the grid.
  std::optional<std::pair<NdArray<double>, NdArray<double>>> all;
  if (!cells.has_value()) {
    all.emplace(grid.collect_cell_bounds());
  }
  std::vector<double> ext(nd);
  for (std::size_t s = 0; s < n_sel; ++s) {
    const double* lo = nullptr;
    const double* hi = nullptr;
    std::pair<std::vector<double>, std::vector<double>> one;
    if (all.has_value()) {
      lo = all->first.data() + s * nd;
      hi = all->second.data() + s * nd;
    } else {
      one = grid.cell_bounds((*cells)[s]);
      lo = one.first.data();
      hi = one.second.data();
    }
    double volume = 1.0;
    for (std::size_t d = 0; d < nd; ++d) {
      ext[d] = hi[d] - lo[d];
      volume = d == 0 ? ext[d] : volume * ext[d];
    }
    for (std::size_t q = 0; q < npts; ++q) {
      for (std::size_t d = 0; d < nd; ++d) {
        const double scaled = ext[d] * rule_points[q * nd + d];
        points.data()[(s * npts + q) * nd + d] = lo[d] + scaled;
      }
      weights.data()[s * npts + q] = rule_weights[q] * volume;
    }
  }
  return {std::move(points), std::move(weights)};
}

}  // namespace pantr::grid
