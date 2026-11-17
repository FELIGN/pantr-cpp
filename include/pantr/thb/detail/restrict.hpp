/// @file restrict.hpp
/// @brief The dof map of a windowed THB space: each local function to the global function of the
/// same level and multi-index, if the global space has it active.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed). For general use, call
///     `pantr::thb::THBSplineSpace::restrict`.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/grid/hierarchical_grid.hpp"
#include "pantr/thb/detail/rep.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace pantr::thb::detail {

/// Map every active function of a windowed space to the global dof of the same function.
///
/// A local function `(level, sub_multi)` is the global function `(level, sub_multi + origin)`,
/// where the origin of a level is the first global function supported on the window's first cell of
/// that level. The global dof is that function's position in the global active list, or -1 when the
/// global space does not have it active (near the window boundary the Kraft selection of the
/// window can differ from the global one).
///
/// Args:
///     global (const THBSpaceRep<T>&): The space that was windowed.
///     sub (const THBSpaceRep<T>&): The space rebuilt on the window.
///     r_lo (std::span<const std::int64_t>): Per-direction index, among the global root cells, of
///         the window's first root cell.
///
/// Returns:
///     std::vector<std::int64_t>: One entry per dof of `sub`.
///
/// Raises:
///     state_error: A local function lands outside the global level's function range, which would
///         mean the windowed level space does not hold the same functions as the global one (the
///         reference implementation fails at this point too, in `numpy.ravel_multi_index`).
template <StorageFloat T>
[[nodiscard]] std::vector<std::int64_t> restriction_dof_map(const THBSpaceRep<T>& global,
                                                            const THBSpaceRep<T>& sub,
                                                            std::span<const std::int64_t> r_lo) {
  const std::size_t dim = global.root_space.dim();
  const std::span<const std::int64_t> factor = global.hier.factor();
  std::vector<std::int64_t> map(static_cast<std::size_t>(sub.func_offset.back()), -1);
  std::vector<std::int64_t> origin(dim);
  std::vector<std::int64_t> multi(dim);
  for (std::size_t level = 0; level < sub.active.size(); ++level) {
    for (std::size_t k = 0; k < dim; ++k) {
      const std::int64_t first_cell =
          r_lo[k] * grid::detail::checked_pow(factor[k], static_cast<std::int64_t>(level));
      origin[k] = global.support[level][k].first_basis[static_cast<std::size_t>(first_cell)];
    }
    const std::vector<std::int64_t>& sub_nb = sub.num_basis[level];
    const std::vector<std::int64_t>& glob_nb = global.num_basis[level];
    const std::vector<std::int64_t>& glob_active = global.active[level];
    for (std::size_t pos = 0; pos < sub.active[level].size(); ++pos) {
      std::int64_t rest = sub.active[level][pos];
      for (std::size_t k = dim; k-- > 0;) {
        multi[k] = rest % sub_nb[k];
        rest /= sub_nb[k];
      }
      std::int64_t glob_flat = 0;
      for (std::size_t k = 0; k < dim; ++k) {
        const std::int64_t g = multi[k] + origin[k];
        if (g >= glob_nb[k]) {
          throw state_error("restrict: local function " + std::to_string(multi[k]) + " of level " +
                            std::to_string(level) + " along axis " + std::to_string(k) +
                            " maps to global function " + std::to_string(g) + ", beyond the " +
                            std::to_string(glob_nb[k]) + " the global level space has.");
        }
        glob_flat = glob_flat * glob_nb[k] + g;
      }
      const auto at = std::lower_bound(glob_active.begin(), glob_active.end(), glob_flat);
      if (at != glob_active.end() && *at == glob_flat) {
        map[static_cast<std::size_t>(sub.func_offset[level]) + pos] =
            global.func_offset[level] + static_cast<std::int64_t>(at - glob_active.begin());
      }
    }
  }
  return map;
}

}  // namespace pantr::thb::detail
