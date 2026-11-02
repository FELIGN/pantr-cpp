// The coupling graph and the graph partitioners, for pantr.bspline's coupling_graph and
// partition_graph wrappers. Arguments arrive normalized (C-contiguous arrays of the exact dtype);
// the C++ API validates their values.
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/bspline/coupling_graph.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/grid/partition.hpp"
#include "pantr/partition/coupling_graph.hpp"
#include "pantr/partition/partition_graph.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

using partition::CouplingGraph;

using F64Vec = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using I64Vec = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using BoolVec = nb::ndarray<const bool, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
std::span<const T> span_of(
    const nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>& a) {
  return {a.data(), a.size()};
}

template <class T>
std::vector<T> vector_of(
    const nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>& a) {
  return {a.data(), a.data() + a.size()};
}

std::optional<std::span<const double>> optional_span(const std::optional<F64Vec>& a) {
  if (!a) {
    return std::nullopt;
  }
  return span_of(*a);
}

std::optional<std::span<const std::uint8_t>> optional_mask(const std::optional<BoolVec>& a) {
  if (!a) {
    return std::nullopt;
  }
  // Booleans are bytes holding 0 or 1.
  return std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(a->data()), a->size());
}

template <class T>
NumpyArray<T> copy_to_numpy(const std::vector<T>& v) {
  NdArray<T> a({v.size()}, for_overwrite);
  std::copy(v.begin(), v.end(), a.data());
  return to_numpy(std::move(a));
}

template <class T, class R>
NumpyView<T> view_1d(std::span<const T> s, std::shared_ptr<const R> keep) {
  const std::size_t shape[1] = {s.size()};
  return view_of<T>(s.data(), shape, std::move(keep));
}

/// The graph's arrays as read-only views sharing its representation:
/// (num_vertices, xadj, adjncy, edge_weights, vertex_weights).
auto graph_views(const CouplingGraph& g) {
  return std::make_tuple(g.num_vertices(), view_1d(g.xadj(), g.rep()), view_1d(g.adjncy(), g.rep()),
                         view_1d(g.edge_weights(), g.rep()), view_1d(g.vertex_weights(), g.rep()));
}

CouplingGraph graph_of(std::int64_t num_vertices, const I64Vec& xadj, const I64Vec& adjncy,
                       const I64Vec& edge_weights, const F64Vec& vertex_weights) {
  return {num_vertices, vector_of(xadj), vector_of(adjncy), vector_of(edge_weights),
          vector_of(vertex_weights)};
}

template <class T>
void def_space_graph(nb::module_& m) {
  m.def(
      "coupling_graph",
      [](const bspline::BsplineSpace<T>& space, const std::optional<F64Vec>& cell_weights) {
        return graph_views(bspline::coupling_graph(space, optional_span(cell_weights)));
      },
      nb::arg("space"), nb::arg("cell_weights").none(),
      "The coupling graph of a tensor-product space: (n, xadj, adjncy, edge_weights, "
      "vertex_weights), read-only.");
}

}  // namespace

void register_partition_graph(nb::module_& m) {
  def_space_graph<double>(m);
  def_space_graph<float>(m);
  m.def(
      "coupling_graph_from_incidence",
      [](std::int64_t n_cells, std::int64_t n_dofs, const I64Vec& cell_ptr, const I64Vec& dofs,
         const std::optional<F64Vec>& cell_weights) {
        return graph_views(partition::coupling_graph_from_incidence(
            n_cells, n_dofs, span_of(cell_ptr), span_of(dofs), optional_span(cell_weights)));
      },
      nb::arg("n_cells"), nb::arg("n_dofs"), nb::arg("cell_ptr"), nb::arg("dofs"),
      nb::arg("cell_weights").none(),
      "The coupling graph of a cell -> DOF incidence in CSR form, as coupling_graph returns it.");
  m.def(
      "partition_graph_spectral",
      [](std::int64_t num_vertices, const I64Vec& xadj, const I64Vec& adjncy,
         const I64Vec& edge_weights, const F64Vec& vertex_weights, std::int64_t n_parts,
         const std::optional<BoolVec>& active) {
        const CouplingGraph g = graph_of(num_vertices, xadj, adjncy, edge_weights, vertex_weights);
        return partition::partition_graph(g, n_parts, optional_mask(active)).partition;
      },
      nb::arg("num_vertices"), nb::arg("xadj"), nb::arg("adjncy"), nb::arg("edge_weights"),
      nb::arg("vertex_weights"), nb::arg("n_parts"), nb::arg("active").none(),
      "Partition a coupling graph by recursive spectral bisection (a grid Partition).");
  m.def(
      "graph_active_subgraph",
      [](std::int64_t num_vertices, const I64Vec& xadj, const I64Vec& adjncy,
         const I64Vec& edge_weights, const F64Vec& vertex_weights, std::int64_t n_parts,
         const std::optional<BoolVec>& active) {
        const CouplingGraph g = graph_of(num_vertices, xadj, adjncy, edge_weights, vertex_weights);
        const partition::detail::ActiveSubgraph sub =
            partition::detail::select_active(g, n_parts, optional_mask(active));
        return std::make_tuple(copy_to_numpy(sub.active_idx), copy_to_numpy(sub.xadj),
                               copy_to_numpy(sub.adjncy), copy_to_numpy(sub.edge_weights),
                               copy_to_numpy(sub.vertex_weights));
      },
      nb::arg("num_vertices"), nb::arg("xadj"), nb::arg("adjncy"), nb::arg("edge_weights"),
      nb::arg("vertex_weights"), nb::arg("n_parts"), nb::arg("active").none(),
      "Check a partition request and return the active subgraph: (active_idx, xadj, adjncy, "
      "edge_weights, vertex_weights), each row's neighbors ascending.");
}

void register_partition_graph_testing(nb::module_& m) {
  m.attr("dense_fiedler_max") = partition::dense_fiedler_max;
  m.def(
      "spectral_partition",
      [](std::int64_t num_vertices, const I64Vec& xadj, const I64Vec& adjncy,
         const I64Vec& edge_weights, const F64Vec& vertex_weights, std::int64_t n_parts,
         const std::optional<BoolVec>& active, std::int64_t dense_max,
         std::optional<std::int64_t> lanczos_max_iter) {
        const CouplingGraph g = graph_of(num_vertices, xadj, adjncy, edge_weights, vertex_weights);
        partition::Spectral spectral{dense_max, lanczos_max_iter};
        partition::GraphPartition r = spectral.partition(g, n_parts, optional_mask(active));
        return std::make_pair(std::move(r.partition),
                              std::make_tuple(r.info.dense_solves, r.info.lanczos_solves,
                                              r.info.dense_fallbacks, r.info.component_orders));
      },
      nb::arg("num_vertices"), nb::arg("xadj"), nb::arg("adjncy"), nb::arg("edge_weights"),
      nb::arg("vertex_weights"), nb::arg("n_parts"), nb::arg("active").none(), nb::arg("dense_max"),
      nb::arg("lanczos_max_iter").none(),
      "Spectral partition with its solver options: (Partition, (dense_solves, lanczos_solves, "
      "dense_fallbacks, component_orders)).");
}

}  // namespace pantr::bindings
