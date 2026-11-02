/// @file coupling_graph.hpp
/// @brief `CouplingGraph`, the cell-coupling (dual) graph of a space, and its assembly from a
/// cell -> DOF incidence.
///
/// The vertices are the cells of a space's grid; two cells are joined when they share at least one
/// basis function (DOF), weighted by the number of shared functions. The graph is stored in the CSR
/// adjacency format METIS and Scotch take (`xadj`, `adjncy`, `adjwgt`, `vwgt`), so a partitioner
/// hands it to those libraries with no reshaping. The tensor-product builder is
/// `pantr/bspline/coupling_graph.hpp`; a space that only knows its active functions per cell (a
/// THB space) builds the graph from that incidence with `coupling_graph_from_incidence`.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"

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

namespace pantr::partition {

namespace detail {

/// The shared representation of a `CouplingGraph`.
///
/// Attributes:
///     num_vertices (std::int64_t): Number of vertices (cells).
///     xadj (std::vector<std::int64_t>): CSR row pointers, `num_vertices + 1` entries.
///     adjncy (std::vector<std::int64_t>): Neighbor vertex ids, `xadj.back()` entries.
///     edge_weights (std::vector<std::int64_t>): Weight per adjacency entry, aligned with `adjncy`.
///     vertex_weights (std::vector<double>): Weight per vertex.
struct CouplingGraphRep {
  std::int64_t num_vertices = 0;
  std::vector<std::int64_t> xadj;
  std::vector<std::int64_t> adjncy;
  std::vector<std::int64_t> edge_weights;
  std::vector<double> vertex_weights;
};

/// Build the message of a length check, in the spelling of a NumPy shape.
///
/// Args:
///     name (const char*): The argument's name.
///     expected (std::size_t): The required length.
///     got (std::size_t): The length received.
///
/// Returns:
///     std::string: `"<name> must have shape (<expected>,); got (<got>,)."`.
[[nodiscard]] inline std::string shape_message(const char* name, std::size_t expected,
                                               std::size_t got) {
  const std::size_t shape[1] = {got};
  return std::string(name) + " must have shape (" + std::to_string(expected) + ",); got " +
         py_repr(shape) + ".";
}

/// Check per-vertex weights: the length, and every entry finite and non-negative.
///
/// Args:
///     weights (std::span<const double>): The weights.
///     n (std::size_t): The required length.
///     name (const char*): The argument's name, for the messages.
///
/// Raises:
///     value_error: The length is not `n`, or an entry is negative or not finite.
inline void check_vertex_weights(std::span<const double> weights, std::size_t n, const char* name) {
  if (weights.size() != n) {
    throw value_error(shape_message(name, n, weights.size()));
  }
  if (std::any_of(weights.begin(), weights.end(),
                  [](double w) { return !std::isfinite(w) || w < 0.0; })) {
    throw value_error(std::string(name) + " must be finite and non-negative.");
  }
}

}  // namespace detail

/// The cell-coupling graph of a space, in METIS / Scotch CSR adjacency format.
///
/// The neighbors of vertex `c` are `adjncy()[xadj()[c] .. xadj()[c + 1])`, each with the weight at
/// the same position of `edge_weights()`. A graph built by `coupling_graph` or
/// `coupling_graph_from_incidence` is symmetric, has no self-loops and no zero weights, and lists
/// each row's neighbors in ascending order; a graph built from arrays is checked only for what an
/// out-of-range read would need (lengths, row pointers, neighbor ids) and for its vertex weights.
/// Immutable: the arrays are moved in at construction and shared by every copy.
///
/// Attributes:
///     rep_ (std::shared_ptr<const detail::CouplingGraphRep>): The arrays.
class CouplingGraph {
 public:
  /// Build a graph from its CSR arrays.
  ///
  /// Args:
  ///     num_vertices (std::int64_t): Number of vertices, at least 0.
  ///     xadj (std::vector<std::int64_t>): Row pointers, `num_vertices + 1` entries, starting at
  ///         0, non-decreasing, ending at `adjncy.size()`.
  ///     adjncy (std::vector<std::int64_t>): Neighbor ids, each in `[0, num_vertices)`.
  ///     edge_weights (std::vector<std::int64_t>): One weight per `adjncy` entry.
  ///     vertex_weights (std::vector<double>): One finite, non-negative weight per vertex.
  ///
  /// Raises:
  ///     value_error: Any of the conditions above fails.
  CouplingGraph(std::int64_t num_vertices, std::vector<std::int64_t> xadj,
                std::vector<std::int64_t> adjncy, std::vector<std::int64_t> edge_weights,
                std::vector<double> vertex_weights) {
    if (num_vertices < 0) {
      throw value_error("num_vertices must be >= 0; got " + std::to_string(num_vertices) + ".");
    }
    const auto n = static_cast<std::size_t>(num_vertices);
    if (xadj.size() != n + 1) {
      throw value_error(detail::shape_message("xadj", n + 1, xadj.size()));
    }
    if (xadj.front() != 0) {
      throw value_error("xadj must start at 0; got " + std::to_string(xadj.front()) + ".");
    }
    if (!std::is_sorted(xadj.begin(), xadj.end())) {
      throw value_error("xadj must be non-decreasing.");
    }
    if (static_cast<std::uint64_t>(xadj.back()) != adjncy.size()) {
      throw value_error("xadj[-1] must equal the length of adjncy (" +
                        std::to_string(adjncy.size()) + "); got " + std::to_string(xadj.back()) +
                        ".");
    }
    if (edge_weights.size() != adjncy.size()) {
      throw value_error(detail::shape_message("edge_weights", adjncy.size(), edge_weights.size()));
    }
    if (std::any_of(adjncy.begin(), adjncy.end(),
                    [&](std::int64_t v) { return v < 0 || v >= num_vertices; })) {
      throw value_error("adjncy entries must lie in [0, " + std::to_string(num_vertices) + ").");
    }
    detail::check_vertex_weights(vertex_weights, n, "vertex_weights");
    auto rep = std::make_shared<detail::CouplingGraphRep>();
    rep->num_vertices = num_vertices;
    rep->xadj = std::move(xadj);
    rep->adjncy = std::move(adjncy);
    rep->edge_weights = std::move(edge_weights);
    rep->vertex_weights = std::move(vertex_weights);
    rep_ = std::move(rep);
  }

  /// Get the number of vertices (cells).
  ///
  /// Returns:
  ///     std::int64_t: The vertex count.
  [[nodiscard]] std::int64_t num_vertices() const noexcept { return rep_->num_vertices; }

  /// Get the CSR row pointers.
  ///
  /// Returns:
  ///     std::span<const std::int64_t>: `num_vertices() + 1` entries.
  [[nodiscard]] std::span<const std::int64_t> xadj() const noexcept { return rep_->xadj; }

  /// Get the concatenated neighbor ids.
  ///
  /// Returns:
  ///     std::span<const std::int64_t>: `xadj().back()` entries.
  [[nodiscard]] std::span<const std::int64_t> adjncy() const noexcept { return rep_->adjncy; }

  /// Get the weight of each adjacency entry (the number of shared functions).
  ///
  /// Returns:
  ///     std::span<const std::int64_t>: One weight per `adjncy()` entry.
  [[nodiscard]] std::span<const std::int64_t> edge_weights() const noexcept {
    return rep_->edge_weights;
  }

  /// Get the weight of each vertex (its assembly cost).
  ///
  /// Returns:
  ///     std::span<const double>: `num_vertices()` weights.
  [[nodiscard]] std::span<const double> vertex_weights() const noexcept {
    return rep_->vertex_weights;
  }

  /// Get the shared representation.
  ///
  /// Returns:
  ///     const std::shared_ptr<const detail::CouplingGraphRep>&: The representation.
  [[nodiscard]] const std::shared_ptr<const detail::CouplingGraphRep>& rep() const noexcept {
    return rep_;
  }

 private:
  std::shared_ptr<const detail::CouplingGraphRep> rep_;
};

namespace detail {

/// Resolve optional cell weights to `n` checked weights, uniform 1 when absent.
///
/// Args:
///     cell_weights (std::optional<std::span<const double>>): The weights, or none.
///     n (std::size_t): The number of cells.
///
/// Returns:
///     std::vector<double>: `n` weights.
///
/// Raises:
///     value_error: The weights have the wrong length or an entry is negative or not finite.
[[nodiscard]] inline std::vector<double> resolve_cell_weights(
    std::optional<std::span<const double>> cell_weights, std::size_t n) {
  if (!cell_weights.has_value()) {
    return std::vector<double>(n, 1.0);
  }
  check_vertex_weights(*cell_weights, n, "cell_weights");
  return {cell_weights->begin(), cell_weights->end()};
}

}  // namespace detail

/// Build the coupling graph from a cell -> DOF incidence.
///
/// The edge weight of two distinct cells is the number of DOFs both list (the off-diagonal entry
/// of `B B^T`, `B` the cell-by-DOF incidence); the diagonal is dropped and each row's neighbors are
/// sorted. A DOF listed twice for one cell counts twice, as a summed incidence entry does, so an
/// incidence is expected to be duplicate-free.
///
/// Args:
///     n_cells (std::int64_t): Number of cells (vertices), at least 0.
///     n_dofs (std::int64_t): Number of DOFs, at least 0.
///     cell_ptr (std::span<const std::int64_t>): Incidence row pointers, `n_cells + 1` entries,
///         starting at 0, non-decreasing, ending at `dofs.size()`.
///     dofs (std::span<const std::int64_t>): DOF ids per cell, each in `[0, n_dofs)`.
///     cell_weights (std::optional<std::span<const double>>): Per-cell weights, finite and
///         non-negative; uniform 1 when absent.
///
/// Returns:
///     CouplingGraph: The graph.
///
/// Raises:
///     value_error: A count is negative, the incidence is malformed, or the weights are invalid.
[[nodiscard]] inline CouplingGraph coupling_graph_from_incidence(
    std::int64_t n_cells, std::int64_t n_dofs, std::span<const std::int64_t> cell_ptr,
    std::span<const std::int64_t> dofs,
    std::optional<std::span<const double>> cell_weights = std::nullopt) {
  if (n_cells < 0 || n_dofs < 0) {
    throw value_error("n_cells and n_dofs must be >= 0; got " + std::to_string(n_cells) + " and " +
                      std::to_string(n_dofs) + ".");
  }
  const auto nc = static_cast<std::size_t>(n_cells);
  const auto nd = static_cast<std::size_t>(n_dofs);
  if (cell_ptr.size() != nc + 1 || cell_ptr.front() != 0 ||
      !std::is_sorted(cell_ptr.begin(), cell_ptr.end()) ||
      static_cast<std::uint64_t>(cell_ptr.back()) != dofs.size()) {
    throw value_error(
        "cell_ptr must have n_cells + 1 non-decreasing entries from 0 to the number of dofs.");
  }
  if (std::any_of(dofs.begin(), dofs.end(), [&](std::int64_t d) { return d < 0 || d >= n_dofs; })) {
    throw value_error("dof ids must lie in [0, " + std::to_string(n_dofs) + ").");
  }
  std::vector<double> weights = detail::resolve_cell_weights(cell_weights, nc);

  // The transpose (DOF -> cells), by counting sort: cells ascending within each DOF.
  std::vector<std::int64_t> dof_ptr(nd + 1, 0);
  for (const std::int64_t d : dofs) {
    ++dof_ptr[static_cast<std::size_t>(d) + 1];
  }
  for (std::size_t d = 0; d < nd; ++d) {
    dof_ptr[d + 1] += dof_ptr[d];
  }
  std::vector<std::int64_t> dof_cells(dofs.size());
  {
    std::vector<std::int64_t> fill(dof_ptr.begin(), dof_ptr.end() - 1);
    for (std::size_t c = 0; c < nc; ++c) {
      for (auto k = static_cast<std::size_t>(cell_ptr[c]);
           k < static_cast<std::size_t>(cell_ptr[c + 1]); ++k) {
        const auto d = static_cast<std::size_t>(dofs[k]);
        dof_cells[static_cast<std::size_t>(fill[d]++)] = static_cast<std::int64_t>(c);
      }
    }
  }

  // Row c of B B^T by a dense accumulator over the cells: `count[c']` collects the shared DOFs,
  // `touched` lists the cells reached, reset after each row.
  std::vector<std::int64_t> xadj(nc + 1, 0);
  std::vector<std::int64_t> adjncy;
  std::vector<std::int64_t> edge_weights;
  std::vector<std::int64_t> count(nc, 0);
  std::vector<std::int64_t> touched;
  for (std::size_t c = 0; c < nc; ++c) {
    for (auto k = static_cast<std::size_t>(cell_ptr[c]);
         k < static_cast<std::size_t>(cell_ptr[c + 1]); ++k) {
      const auto d = static_cast<std::size_t>(dofs[k]);
      for (auto j = static_cast<std::size_t>(dof_ptr[d]);
           j < static_cast<std::size_t>(dof_ptr[d + 1]); ++j) {
        const auto other = static_cast<std::size_t>(dof_cells[j]);
        if (other == c) {
          continue;
        }
        if (count[other] == 0) {
          touched.push_back(static_cast<std::int64_t>(other));
        }
        ++count[other];
      }
    }
    std::sort(touched.begin(), touched.end());
    for (const std::int64_t other : touched) {
      adjncy.push_back(other);
      edge_weights.push_back(count[static_cast<std::size_t>(other)]);
      count[static_cast<std::size_t>(other)] = 0;
    }
    touched.clear();
    xadj[c + 1] = static_cast<std::int64_t>(adjncy.size());
  }
  return CouplingGraph(n_cells, std::move(xadj), std::move(adjncy), std::move(edge_weights),
                       std::move(weights));
}

}  // namespace pantr::partition
