// _pantr.grid: the grids, the BVH, the tag registries, the partition and the generic consumers
// (overlay, cell_quadrature, partition_grid), for pantr.grid's wrappers. Arguments arrive
// normalised (C-contiguous float64 / int64 arrays, Python ints); the wrappers keep the checks whose
// messages spell a NumPy dtype or shape. The unit reaches Eigen only through the grids'
// reference_map (an AffineTransform); the MSVC v4 rung defines EIGEN_DONT_VECTORIZE on its whole
// target, which covers it.
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/geometry/aabb.hpp"
#include "pantr/grid/bvh.hpp"
#include "pantr/grid/cell_quadrature.hpp"
#include "pantr/grid/hierarchical_grid.hpp"
#include "pantr/grid/overlay.hpp"
#include "pantr/grid/partition.hpp"
#include "pantr/grid/tags.hpp"
#include "pantr/grid/tensor_product_grid.hpp"
#include "pantr/transform/affine_transform.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

using grid::BVH;
using grid::CellTags;
using grid::FacetTags;
using grid::HierarchicalGrid;
using grid::Partition;
using grid::PartitionBackend;
using grid::TensorProductGrid;

using F64Vec = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using F64Mat = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using F64Any = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
using I64Vec = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using I64Any = nb::ndarray<const std::int64_t, nb::c_contig, nb::device::cpu>;
using I32Vec = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using BoolVec = nb::ndarray<const bool, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class A>
auto span_of(const A& a) {
  using T = std::remove_const_t<std::remove_pointer_t<decltype(a.data())>>;
  return std::span<const T>(a.data(), a.size());
}

template <class A>
std::vector<std::size_t> shape_of(const A& a) {
  std::vector<std::size_t> s(a.ndim());
  for (std::size_t i = 0; i < a.ndim(); ++i) {
    s[i] = a.shape(i);
  }
  return s;
}

/// A NumPy array owning a copy of `v`.
template <class T>
NumpyArray<T> copy_to_numpy(const std::vector<T>& v) {
  NdArray<T> a({v.size()}, for_overwrite);
  std::copy(v.begin(), v.end(), a.data());
  return to_numpy(std::move(a));
}

/// A boolean NumPy array over the buffer of a 0/1 byte array (no copy).
NumpyArray<bool> to_numpy_bool(NdArray<std::uint8_t>&& a) {
  std::vector<std::size_t> shape(a.shape().begin(), a.shape().end());
  auto* buffer = new std::shared_ptr<std::uint8_t[]>(std::move(a).into_buffer());
  nb::capsule owner(
      buffer, [](void* p) noexcept { delete static_cast<std::shared_ptr<std::uint8_t[]>*>(p); });
  // The bytes hold 0 or 1 and are only ever read through NumPy's bool dtype.
  return NumpyArray<bool>(reinterpret_cast<bool*>(buffer->get()), shape.size(), shape.data(),
                          owner);
}

NumpyArray<bool> bytes_to_numpy_bool(const std::vector<std::uint8_t>& v) {
  NdArray<std::uint8_t> a({v.size()}, for_overwrite);
  std::copy(v.begin(), v.end(), a.data());
  return to_numpy_bool(std::move(a));
}

template <class T, class R>
NumpyView<T> view_1d(std::span<const T> s, std::shared_ptr<const R> keep) {
  const std::size_t shape[1] = {s.size()};
  return view_of<T>(s.data(), shape, std::move(keep));
}

template <class T, class R>
NumpyView<T> view_2d(std::span<const T> s, std::size_t cols, std::shared_ptr<const R> keep) {
  const std::size_t shape[2] = {cols == 0 ? 0 : s.size() / cols, cols};
  return view_of<T>(s.data(), shape, std::move(keep));
}

/// The corners of a cell as two fresh NumPy arrays.
std::pair<NumpyArray<double>, NumpyArray<double>> corners_to_numpy(
    const std::pair<std::vector<double>, std::vector<double>>& c) {
  return {copy_to_numpy(c.first), copy_to_numpy(c.second)};
}

/// Build `g`'s lazy BVH, with the GIL released, unless a call has built it already.
///
/// The build reads the grid and writes only C++ state, so it needs no GIL. A thread that finds the
/// build under way waits on the block's mutex; it cannot deadlock against the builder, which takes
/// the GIL back only after `cell_bvh` has returned and released that mutex. The release costs one
/// lock round trip, paid only by a call that finds the tree unbuilt (once per grid, or a few
/// times when threads race to the first query), and spares the other Python threads a stall for
/// the length of the build.
template <class G>
void ensure_bvh(const G& g) {
  if (!g.bvh_built()) {
    const nb::gil_scoped_release release;
    static_cast<void>(g.cell_bvh());
  }
}

/// The members both grids bind alike: the primitives and the defaults of `GridBase`.
template <class G>
void def_common(nb::class_<G>& cls) {
  cls.def_prop_ro("ndim", [](const G& g) { return static_cast<std::int64_t>(g.ndim()); })
      .def_prop_ro("num_cells", &G::num_cells)
      .def(
          "cell_bounds",
          [](const G& g, std::int64_t cid) { return corners_to_numpy(g.cell_bounds(cid)); },
          nb::arg("cid"), "The (lo, hi) corners of a cell, as fresh arrays.")
      .def(
          "locate", [](const G& g, const F64Vec& pt) { return g.locate(span_of(pt)); },
          nb::arg("pt"), "The cell containing a point, or None.")
      .def(
          "locate_many",
          [](const G& g, const F64Mat& points) {
            const std::size_t n = points.shape(0);
            NdArray<std::int64_t> out = [&] {
              const ReleaseIfParallel release(n, g.locate_work().ops_per_item);
              return g.locate_many(span_of(points), n);
            }();
            return to_numpy(std::move(out));  // with the GIL: it builds a Python object
          },
          nb::arg("points"), "The cell of each row of (npts, ndim) points; -1 outside.")
      .def("neighbor_across_facet", &G::neighbor_across_facet, nb::arg("cid"), nb::arg("lfid"),
           "The cell across a local facet, or None.")
      .def(
          "collect_cell_bounds",
          [](const G& g) {
            auto bounds = [&] {
              // Only the hierarchical grid collects in parallel; the tensor-product one is serial.
              std::optional<ReleaseIfParallel> release;
              if constexpr (std::is_same_v<G, HierarchicalGrid>) {
                release.emplace(static_cast<std::size_t>(g.num_cells()),
                                g.collect_work().ops_per_item);
              }
              return g.collect_cell_bounds();
            }();
            return std::pair(to_numpy(std::move(bounds.first)), to_numpy(std::move(bounds.second)));
          },
          "Every cell's (lo, hi) corners as two (num_cells, ndim) arrays.")
      .def(
          "cell_level", [](const G& g, std::int64_t cid) { return g.cell_level(cid); },
          nb::arg("cid"), "The refinement level of a cell.")
      .def(
          "child_cells", [](const G& g, std::int64_t cid) { return g.child_cells(cid); },
          nb::arg("cid"), "The refinement children of a cell.")
      .def(
          "num_local_facets", [](const G& g, std::int64_t cid) { return g.num_local_facets(cid); },
          nb::arg("cid"), "The number of local facets of a cell.")
      .def(
          "local_facet_axis_side",
          [](const G& g, std::int64_t cid, std::int64_t lfid) {
            return g.local_facet_axis_side(cid, lfid);
          },
          nb::arg("cid"), nb::arg("lfid"), "(axis, side) of a local facet.")
      .def(
          "local_facet_bounds",
          [](const G& g, std::int64_t cid, std::int64_t lfid) {
            return corners_to_numpy(g.local_facet_bounds(cid, lfid));
          },
          nb::arg("cid"), nb::arg("lfid"), "The degenerate (lo, hi) box of a facet.")
      .def(
          "is_mesh_boundary_facet",
          [](const G& g, std::int64_t cid, std::int64_t lfid) {
            return g.is_mesh_boundary_facet(cid, lfid);
          },
          nb::arg("cid"), nb::arg("lfid"), "Whether a facet is on the outer boundary.")
      .def(
          "neighbors", [](const G& g, std::int64_t cid) { return g.neighbors(cid); },
          nb::arg("cid"), "The facet-neighbor cell ids of a cell.")
      .def(
          "boundary_facets", [](const G& g) { return to_numpy(g.boundary_facets()); },
          "Every outer-boundary facet as (cid, lfid) rows.")
      .def(
          "hanging_neighbors",
          [](const G& g, std::int64_t cid, std::int64_t lfid) {
            return g.hanging_neighbors(cid, lfid);
          },
          nb::arg("cid"), nb::arg("lfid"), "All active cells sharing a facet.")
      .def(
          "cell_aabb", [](const G& g, std::int64_t cid) { return g.cell_aabb(cid); },
          nb::arg("cid"), "A cell's bounding box.")
      .def(
          "reference_map", [](const G& g, std::int64_t cid) { return g.reference_map(cid); },
          nb::arg("cid"), "The affine map from the unit cube onto a cell.")
      .def(
          "cell_bvh",
          [](const G& g) {
            ensure_bvh(g);
            return BVH(g.cell_bvh());
          },
          "The BVH over the cell boxes, built on first use.")
      .def(
          "bvh_built", [](const G& g) { return g.bvh_built(); },
          "Whether the BVH has been built yet.")
      .def(
          "query_aabb",
          [](const G& g, const geometry::AABB& aabb) {
            ensure_bvh(g);
            return copy_to_numpy(g.query_aabb(aabb));
          },
          nb::arg("aabb"), "The ids of the cells whose boxes overlap a box.");
}

void register_tensor_product_grid(nb::module_& m) {
  nb::class_<TensorProductGrid> cls(m, "TensorProductGrid",
                                    "An immutable tensor-product grid (private).");
  cls.def(
      "__init__",
      [](TensorProductGrid* self, const std::vector<F64Vec>& axes) {
        std::vector<std::vector<double>> breakpoints;
        breakpoints.reserve(axes.size());
        for (const F64Vec& a : axes) {
          breakpoints.emplace_back(a.data(), a.data() + a.shape(0));
        }
        new (self) TensorProductGrid(std::move(breakpoints));
      },
      nb::arg("breakpoints"), "Validate and copy the per-axis breakpoints.");
  def_common(cls);
  cls.def_prop_ro("cells_per_axis",
                  [](const TensorProductGrid& g) {
                    const auto c = g.cells_per_axis();
                    return std::vector<std::int64_t>(c.begin(), c.end());
                  })
      .def_prop_ro("is_uniform", &TensorProductGrid::is_uniform)
      .def_prop_ro(
          "bounds",
          [](const TensorProductGrid& g) { return view_2d<double>(g.bounds(), 2, g.rep()); },
          nb::rv_policy::automatic)
      .def(
          "breakpoints",
          [](const TensorProductGrid& g, std::int64_t axis) {
            return view_1d<double>(g.breakpoints(static_cast<std::size_t>(axis)), g.rep());
          },
          nb::arg("axis"), "Read-only view of the breakpoints of one axis.")
      .def(
          "cell_multi_index",
          [](const TensorProductGrid& g, std::int64_t cid) { return g.cell_multi_index(cid); },
          nb::arg("cid"), "The per-axis indices of a cell.")
      .def(
          "flat_cell_index",
          [](const TensorProductGrid& g, const std::vector<std::int64_t>& multi) {
            return g.flat_cell_index(multi);
          },
          nb::arg("multi"), "The flat id of a per-axis multi-index.")
      .def(
          "restrict",
          [](const TensorProductGrid& g, const I64Vec& ids) {
            auto r = g.restrict(span_of(ids));
            return std::tuple(std::move(r.grid), copy_to_numpy(r.local_to_global),
                              bytes_to_numpy_bool(r.in_subset));
          },
          nb::arg("ids"), "(sub-grid, local_to_global, in_subset) of the bounding-box window.")
      .def("same_rep", &TensorProductGrid::same_rep, nb::arg("other"),
           "Whether both share one representation.");
}

void register_hierarchical_grid(nb::module_& m) {
  using grid::detail::Block;
  nb::class_<HierarchicalGrid> cls(m, "HierarchicalGrid",
                                   "An immutable hierarchical grid (private).");
  cls.def(
      "__init__",
      [](HierarchicalGrid* self, const TensorProductGrid& root,
         const std::vector<std::int64_t>& factor) { new (self) HierarchicalGrid(root, factor); },
      nb::arg("root"), nb::arg("factor"), "Every root cell active at level 0 (version 1).");
  cls.def_static(
      "from_blocks",
      [](const TensorProductGrid& root, const std::vector<std::int64_t>& factor,
         std::vector<std::vector<Block>> blocks, std::int64_t grid_version) {
        return HierarchicalGrid::from_blocks(root, factor, std::move(blocks), grid_version);
      },
      nb::arg("root"), nb::arg("factor"), nb::arg("blocks"), nb::arg("version"),
      "A grid from per-level block lists (pickling and windowing).");
  def_common(cls);
  cls.def_prop_ro("root", [](const HierarchicalGrid& g) { return TensorProductGrid(g.root()); })
      .def_prop_ro("factor",
                   [](const HierarchicalGrid& g) {
                     const auto f = g.factor();
                     return std::vector<std::int64_t>(f.begin(), f.end());
                   })
      .def_prop_ro("max_level", &HierarchicalGrid::max_level)
      .def_prop_ro("version", &HierarchicalGrid::version)
      .def(
          "cell_multi_index",
          [](const HierarchicalGrid& g, std::int64_t cid) { return g.cell_multi_index(cid); },
          nb::arg("cid"), "The per-axis index of a cell in its level's coordinates.")
      .def(
          "cell_id",
          [](const HierarchicalGrid& g, std::int64_t level, const std::vector<std::int64_t>& midx) {
            return g.cell_id(level, midx);
          },
          nb::arg("level"), nb::arg("midx"), "The flat id of an active leaf, or None.")
      .def(
          "is_active_leaf",
          [](const HierarchicalGrid& g, std::int64_t level, const std::vector<std::int64_t>& midx) {
            return g.is_active_leaf(level, midx);
          },
          nb::arg("level"), nb::arg("midx"), "Whether (level, midx) is an active leaf.")
      .def("level_cells_per_axis", &HierarchicalGrid::level_cells_per_axis, nb::arg("level"),
           "The cell count per axis at a level.")
      .def(
          "active_blocks",
          [](const HierarchicalGrid& g, std::int64_t level) { return g.active_blocks(level); },
          nb::arg("level"), "The active-leaf blocks at a level.")
      .def(
          "blocks", [](const HierarchicalGrid& g) { return g.blocks(); },
          "Every level's active-leaf blocks.")
      .def(
          "active_leaf_mask",
          [](const HierarchicalGrid& g, std::int64_t level) {
            return to_numpy_bool(g.active_leaf_mask(level));
          },
          nb::arg("level"), "The mask of the active leaves at a level.")
      .def(
          "subdomain_mask",
          [](const HierarchicalGrid& g, std::int64_t level) {
            return to_numpy_bool(g.subdomain_mask(level));
          },
          nb::arg("level"), "The mask of a level's refined subdomain.")
      .def(
          "refined",
          [](const HierarchicalGrid& g, std::int64_t level, const std::vector<std::int64_t>& lo,
             const std::vector<std::int64_t>& hi) { return g.refined(level, lo, hi); },
          nb::arg("level"), nb::arg("lo"), nb::arg("hi"),
          "The grid with a region refined; this grid itself when nothing changes.")
      .def(
          "refined_cells",
          [](const HierarchicalGrid& g, const std::vector<std::int64_t>& ids) {
            return g.refined_cells(ids);
          },
          nb::arg("ids"), "The grid with the given cells refined.")
      .def(
          "coarsened",
          [](const HierarchicalGrid& g, std::int64_t level, const std::vector<std::int64_t>& lo,
             const std::vector<std::int64_t>& hi) { return g.coarsened(level, lo, hi); },
          nb::arg("level"), nb::arg("lo"), nb::arg("hi"),
          "The grid with a region coarsened; this grid itself when nothing changes.")
      .def(
          "coarsened_cells",
          [](const HierarchicalGrid& g, const std::vector<std::int64_t>& ids) {
            return g.coarsened_cells(ids);
          },
          nb::arg("ids"), "The grid with the given cells coarsened.")
      .def(
          "restrict",
          [](const HierarchicalGrid& g, const I64Vec& ids) {
            auto r = g.restrict(span_of(ids));
            return std::tuple(std::move(r.grid), copy_to_numpy(r.local_to_global),
                              bytes_to_numpy_bool(r.in_subset));
          },
          nb::arg("ids"), "(sub-grid, local_to_global, in_subset) of the bounding-box window.")
      .def(
          "export_cells",
          [](const HierarchicalGrid& g) {
            auto cells = g.export_cells();
            return std::pair(to_numpy(std::move(cells.first)), to_numpy(std::move(cells.second)));
          },
          "The unique nodes (n_nodes, ndim) and the connectivity (num_cells, 2^ndim).")
      .def("same_rep", &HierarchicalGrid::same_rep, nb::arg("other"),
           "Whether both share one representation.");
}

void register_bvh(nb::module_& m) {
  nb::class_<BVH>(m, "BVH", "An immutable bounding-volume hierarchy (private).")
      .def(
          "__init__",
          [](BVH* self, const F64Any& node_lo, const F64Any& node_hi, const I64Any& node_left,
             const I64Any& node_right, const I64Any& node_cell, std::int64_t n_cells) {
            new (self) BVH(span_of(node_lo), span_of(node_hi), shape_of(node_lo), shape_of(node_hi),
                           span_of(node_left), span_of(node_right), span_of(node_cell), n_cells);
          },
          nb::arg("node_lo"), nb::arg("node_hi"), nb::arg("node_left"), nb::arg("node_right"),
          nb::arg("node_cell"), nb::arg("n_cells"),
          "Validate and copy a raw tree (node arrays, child pointers, leaf cells).")
      .def_static(
          "from_cell_bounds",
          [](const F64Any& lo, const F64Any& hi) {
            return BVH::from_cell_bounds(span_of(lo), span_of(hi), shape_of(lo), shape_of(hi));
          },
          nb::arg("cell_lo"), nb::arg("cell_hi"),
          "Build the tree over (n_cells, ndim) cell corners.")
      .def_prop_ro("ndim", [](const BVH& b) { return static_cast<std::int64_t>(b.ndim()); })
      .def_prop_ro("n_cells", &BVH::n_cells)
      .def_prop_ro("n_nodes", &BVH::n_nodes)
      .def_prop_ro(
          "node_lo", [](const BVH& b) { return view_2d<double>(b.node_lo(), b.ndim(), b.rep()); },
          nb::rv_policy::automatic)
      .def_prop_ro(
          "node_hi", [](const BVH& b) { return view_2d<double>(b.node_hi(), b.ndim(), b.rep()); },
          nb::rv_policy::automatic)
      .def_prop_ro(
          "node_left", [](const BVH& b) { return view_1d(b.node_left(), b.rep()); },
          nb::rv_policy::automatic)
      .def_prop_ro(
          "node_right", [](const BVH& b) { return view_1d(b.node_right(), b.rep()); },
          nb::rv_policy::automatic)
      .def_prop_ro(
          "node_cell", [](const BVH& b) { return view_1d(b.node_cell(), b.rep()); },
          nb::rv_policy::automatic)
      .def(
          "query_aabb",
          [](const BVH& b, const geometry::AABB& aabb) {
            return copy_to_numpy(b.query_aabb(aabb));
          },
          nb::arg("aabb"), "The ids of the cells whose boxes overlap a box.");
}

/// The members `CellTags` and `FacetTags` bind alike.
template <class T>
void def_tags_common(nb::class_<T>& cls) {
  cls.def_prop_ro("num_cells", &T::num_cells)
      .def(
          "remove", [](const T& t, const std::string& name) { return t.remove(name); },
          nb::arg("name"), "The registry without the tag.")
      .def(
          "contains", [](const T& t, const std::string& name) { return t.contains(name); },
          nb::arg("name"), "Whether a tag of that name exists.")
      .def("names", &T::names, "The tag names in insertion order.")
      .def(
          "size", [](const T& t) { return static_cast<std::int64_t>(t.size()); },
          "The number of tags.");
}

void register_tags(nb::module_& m) {
  nb::class_<CellTags> cell(m, "CellTags", "An immutable registry of named cell tags (private).");
  cell.def(nb::init<std::int64_t>(), nb::arg("num_cells"), "An empty registry.")
      .def(
          "set",
          [](const CellTags& t, const std::string& name, const I64Vec& ids, const I64Vec& values) {
            return t.set(name, span_of(ids), span_of(values));
          },
          nb::arg("name"), nb::arg("ids"), nb::arg("values"),
          "The registry with the tag created or replaced.")
      .def(
          "get",
          [](const CellTags& t, const std::string& name) {
            const auto [ids, values] = t.get(name);
            return std::pair(view_1d(ids, t.rep()), view_1d(values, t.rep()));
          },
          nb::arg("name"), "Read-only (ids, values) views of a tag.");
  def_tags_common(cell);

  nb::class_<FacetTags> facet(m, "FacetTags",
                              "An immutable registry of named facet tags (private).");
  facet
      .def(nb::init<std::int64_t, std::int64_t>(), nb::arg("num_cells"), nb::arg("facets_per_cell"),
           "An empty registry.")
      .def_prop_ro("facets_per_cell", &FacetTags::facets_per_cell)
      .def(
          "set",
          [](const FacetTags& t, const std::string& name, const I64Vec& keys,
             const I64Vec& values) { return t.set(name, span_of(keys), span_of(values)); },
          nb::arg("name"), nb::arg("keys"), nb::arg("values"),
          "The registry with the tag created or replaced; keys are the flattened (M, 2) rows.")
      .def(
          "get",
          [](const FacetTags& t, const std::string& name) {
            const auto [keys, values] = t.get(name);
            return std::pair(view_2d(keys, 2, t.rep()), view_1d(values, t.rep()));
          },
          nb::arg("name"), "Read-only (keys (M, 2), values) views of a tag.");
  def_tags_common(facet);
}

void register_partition(nb::module_& m) {
  nb::enum_<PartitionBackend>(m, "PartitionBackend", "The backends of partition_grid.")
      .value("automatic", PartitionBackend::automatic)
      .value("block", PartitionBackend::block)
      .value("rcb", PartitionBackend::rcb);

  nb::class_<Partition>(m, "Partition", "An immutable per-cell owner assignment (private).")
      .def(
          "__init__",
          [](Partition* self, const I32Vec& cell_owner, std::int64_t n_parts) {
            const auto owner = span_of(cell_owner);
            new (self) Partition(std::vector<std::int32_t>(owner.begin(), owner.end()), n_parts);
          },
          nb::arg("cell_owner"), nb::arg("n_parts"), "Validate and copy the owners.")
      .def_prop_ro(
          "cell_owner", [](const Partition& p) { return view_1d(p.cell_owner(), p.rep()); },
          nb::rv_policy::automatic)
      .def_prop_ro("n_parts", &Partition::n_parts)
      .def_prop_ro("n_cells", &Partition::n_cells)
      .def(
          "active_mask", [](const Partition& p) { return bytes_to_numpy_bool(p.active_mask()); },
          "A fresh mask of the cells with an owner.")
      .def(
          "owned_cells",
          [](const Partition& p, std::int64_t rank) { return copy_to_numpy(p.owned_cells(rank)); },
          nb::arg("rank"), "The ascending ids of the cells a rank owns.");
}

/// The generic consumers of a grid, one overload per grid type.
template <class G>
void def_generic(nb::module_& m) {
  m.def(
      "cell_quadrature",
      [](const G& g, const F64Mat& rule_points, const F64Vec& rule_weights,
         const std::optional<I64Vec>& cells) {
        std::optional<std::span<const std::int64_t>> selected;
        if (cells) {
          selected = span_of(*cells);
        }
        auto [points, weights] = [&] {
          // Without a selection a hierarchical grid collects every cell's corners in parallel;
          // a selection is read cell by cell, serially, and a tensor-product grid collects
          // serially.
          std::optional<ReleaseIfParallel> release;
          if constexpr (std::is_same_v<G, HierarchicalGrid>) {
            if (!selected.has_value()) {
              release.emplace(static_cast<std::size_t>(g.num_cells()),
                              g.collect_work().ops_per_item);
            }
          }
          return grid::cell_quadrature(g, span_of(rule_points), rule_points.shape(0),
                                       rule_points.shape(1), span_of(rule_weights), selected);
        }();
        return std::pair(to_numpy(std::move(points)), to_numpy(std::move(weights)));
      },
      nb::arg("grid"), nb::arg("rule_points"), nb::arg("rule_weights"), nb::arg("cells").none(),
      "Map a reference rule onto the cells of a grid: (points, weights).");
  m.def(
      "partition_grid",
      [](const G& g, std::int64_t n_parts, PartitionBackend backend,
         const std::optional<F64Vec>& weights, const std::optional<BoolVec>& active) {
        std::optional<std::span<const double>> w;
        if (weights) {
          w = span_of(*weights);
        }
        std::optional<std::span<const std::uint8_t>> a;
        if (active) {
          // Booleans are bytes holding 0 or 1.
          a = std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(active->data()),
                                            active->size());
        }
        // A hierarchical grid has no block backend, so its partition always collects the cell
        // corners in parallel; a tensor-product grid collects serially.
        std::optional<ReleaseIfParallel> release;
        if constexpr (std::is_same_v<G, HierarchicalGrid>) {
          release.emplace(static_cast<std::size_t>(g.num_cells()), g.collect_work().ops_per_item);
        }
        return grid::partition_grid(g, n_parts, backend, w, a);
      },
      nb::arg("grid"), nb::arg("n_parts"), nb::arg("backend"), nb::arg("weights").none(),
      nb::arg("active").none(), "Partition a grid's cells into n_parts rank subdomains.");
}

}  // namespace

void register_grid(nb::module_& m) {
  register_tensor_product_grid(m);
  register_hierarchical_grid(m);
  register_bvh(m);
  register_tags(m);
  register_partition(m);
  m.def("overlay", &grid::overlay, nb::arg("grid_a"), nb::arg("grid_b"),
        "The coarsest tensor-product grid refining both.");
  def_generic<TensorProductGrid>(m);
  def_generic<HierarchicalGrid>(m);
}

}  // namespace pantr::bindings
