// The arrays of a coupling graph as read-only views sharing its representation, as every binding
// that returns a graph hands it to Python.
#pragma once

#include "arrays.hpp"

#include "pantr/partition/coupling_graph.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <tuple>

namespace pantr::bindings {

/// A read-only view of one array of `g`, kept alive by `g`'s shared representation.
template <class T>
NumpyView<T> graph_view(std::span<const T> s, const partition::CouplingGraph& g) {
  const std::size_t shape[1] = {s.size()};
  return view_of<T>(s.data(), shape, g.rep());
}

/// The graph's arrays as read-only views sharing its representation:
/// (num_vertices, xadj, adjncy, edge_weights, vertex_weights).
inline auto graph_views(const partition::CouplingGraph& g) {
  return std::make_tuple(g.num_vertices(), graph_view(g.xadj(), g), graph_view(g.adjncy(), g),
                         graph_view(g.edge_weights(), g), graph_view(g.vertex_weights(), g));
}

}  // namespace pantr::bindings
