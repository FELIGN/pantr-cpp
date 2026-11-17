/// @file coupling_graph.hpp
/// @brief The cell-coupling graph of a THB space.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/partition/coupling_graph.hpp"
#include "pantr/thb/space.hpp"
#include "pantr/thb/types.hpp"

#include <optional>
#include <span>

namespace pantr::thb {

/// Build the cell-coupling graph of a THB space.
///
/// The edge weight of two cells is the number of active functions whose tensor-product support
/// meets both, as `active_basis` lists them (a truncated function that vanishes on a cell it
/// supports still counts); the graph is `partition::coupling_graph_from_incidence` of
/// `cell_dof_incidence()`.
///
/// Args:
///     space (const THBSplineSpace<T>&): The space.
///     cell_weights (std::optional<std::span<const double>>): Per-cell weights, finite and
///         non-negative; uniform 1 when absent.
///
/// Returns:
///     partition::CouplingGraph: The graph over `space.grid().num_cells()` vertices.
///
/// Raises:
///     value_error: The weights are invalid.
template <StorageFloat T>
[[nodiscard]] partition::CouplingGraph coupling_graph(
    const THBSplineSpace<T>& space,
    std::optional<std::span<const double>> cell_weights = std::nullopt) {
  const CellIncidence inc = space.cell_dof_incidence();
  return partition::coupling_graph_from_incidence(space.grid().num_cells(), space.num_total_basis(),
                                                  inc.cell_ptr, inc.dofs, cell_weights);
}

}  // namespace pantr::thb
