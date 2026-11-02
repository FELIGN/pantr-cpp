// _pantr.bspline: the truncated hierarchical B-spline classes (THBSplineSpace, THBSpline,
// MultiLevelExtraction, THBQuasiInterpolation, one per storage dtype) and the windowing functions
// of a distributed space (the halo and the dof owner of a tensor-product and of a THB space, the
// coupling graph of a THB space), for pantr.bspline's wrappers; and, in _testing, the tolerances
// the space's checks use and the support of a 1D space.
//
// This translation unit reaches Eigen (the prolongation's least-squares solves and the Lagrange and
// cardinal operators of the per-level extractions). The MSVC v4 rung defines EIGEN_DONT_VECTORIZE
// on its whole target, so Eigen's products run unvectorized in it, this unit included.
//
// Arguments arrive as the wrapper normalised them (C-contiguous arrays of the exact dtype, Python
// ints, a writable `out`); validation of values is the C++ API's, with the messages the Python
// suite matches. The GIL is released only around a call that can reach the pool: the lazy per-cell
// table of a space (computed on the first call that needs it), the evaluation and prolongation
// loops, the quasi-interpolant's contraction, the halo and the owner kernels. Where the pool is
// reached only above a size, the decision is the C++ one (`parallel::worth_parallel`) on an
// estimate that errs high, so the GIL is released at least whenever the pool is used.
#include "arrays.hpp"
#include "graph_views.hpp"
#include "interface.hpp"

#include "pantr/bspline/detail/support_1d.hpp"
#include "pantr/bspline/local_space.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/bspline/spanwise_extraction.hpp"
#include "pantr/grid/hierarchical_grid.hpp"
#include "pantr/parallel/work_model.hpp"
#include "pantr/thb/coupling_graph.hpp"
#include "pantr/thb/local_space.hpp"
#include "pantr/thb/multilevel_extraction.hpp"
#include "pantr/thb/quasi_interpolate.hpp"
#include "pantr/thb/space.hpp"
#include "pantr/thb/spline.hpp"
#include "pantr/thb/tolerances.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

using F64Vec = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using F64Mat = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using I64Vec = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using I32Vec = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using OutF64Mat = nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <class A>
auto span_of(const A& a) {
  using E = std::remove_const_t<std::remove_pointer_t<decltype(a.data())>>;
  return std::span<const E>(a.data(), a.size());
}

mdview<const double, 2> view_of_points(const F64Mat& a) {
  return mdview<const double, 2>(a.data(), a.shape(0), a.shape(1));
}

/// A NumPy array owning a copy of `s`.
template <class E>
NumpyArray<E> copy_span(std::span<const E> s) {
  NdArray<E> a({s.size()}, for_overwrite);
  std::copy(s.begin(), s.end(), a.data());
  return to_numpy(std::move(a));
}

/// A read-only one-dimensional view of `s`, owned by a copy of `keep`.
template <class E, class R>
NumpyView<E> view_1d(std::span<const E> s, std::shared_ptr<const R> keep) {
  const std::size_t shape[1] = {s.size()};
  return view_of<E>(s.data(), shape, std::move(keep));
}

/// Release the GIL for the lifetime of the guard when `release` holds.
class ReleaseGil {
 public:
  explicit ReleaseGil(bool release) {
    if (release) {
      guard_.emplace();
    }
  }

 private:
  std::optional<nb::gil_scoped_release> guard_;
};

/// Narrow a regularity from the binding's 64-bit integer to the C++ API's `int`, saturating: a
/// value outside `int` is outside `[-1, degree)` whatever it is, so the C++ range check refuses it
/// (naming the saturated value).
std::vector<std::optional<int>> narrow_regularity(
    const std::vector<std::optional<long long>>& regularity) {
  constexpr long long lo = (std::numeric_limits<int>::min)();
  constexpr long long hi = (std::numeric_limits<int>::max)();
  std::vector<std::optional<int>> out;
  out.reserve(regularity.size());
  for (const std::optional<long long>& r : regularity) {
    if (r.has_value()) {
      out.emplace_back(static_cast<int>(std::clamp(*r, lo, hi)));
    } else {
      out.emplace_back(std::nullopt);
    }
  }
  return out;
}

/// Operations the per-cell table costs per cell, as `thb::detail::build_cell_block` estimates it
/// (levels, times the candidate functions per level, times a flat index and a binary search each).
template <class T>
double cell_table_work(const thb::THBSplineSpace<T>& space) {
  double local_functions = 1.0;
  for (const int p : space.degrees()) {
    local_functions *= static_cast<double>(p) + 1.0;
  }
  std::int64_t most_active = 0;
  for (const std::int64_t count : space.num_basis_per_level()) {
    most_active = (std::max)(most_active, count);
  }
  const double per_candidate =
      static_cast<double>(space.dim()) + std::log2(static_cast<double>(most_active) + 1.0);
  return (std::max)(1.0, static_cast<double>(space.num_levels()) * local_functions * per_candidate);
}

/// Whether the call about to run builds the space's per-cell table on the pool: the table is lazy
/// and not built yet, and building it would be parallel.
template <class T>
bool table_on_pool(const thb::THBSplineSpace<T>& space) {
  if (space.rep()->cells.ready()) {
    return false;
  }
  return parallel::worth_parallel(static_cast<std::size_t>(space.grid().num_cells()),
                                  parallel::WorkEstimate{cell_table_work(space)});
}

/// Whether a loop of `n` items of at most `ops` operations each runs on the pool.
bool loop_on_pool(std::size_t n, double ops) {
  return parallel::worth_parallel(n, parallel::WorkEstimate{(std::max)(1.0, ops)});
}

/// Functions a cell can hold, at most: the level count times `prod(degree + 1)`.
template <class T>
double functions_per_cell_bound(const thb::THBSplineSpace<T>& space) {
  double local_functions = 1.0;
  for (const int p : space.degrees()) {
    local_functions *= static_cast<double>(p) + 1.0;
  }
  return static_cast<double>(space.num_levels()) * local_functions;
}

/// Operations one point of `THBSpline::evaluate` costs, as `THBSpline::point_work` estimates it
/// with the number of functions on a cell bounded by `functions_per_cell_bound`.
template <class T>
double evaluate_point_work(const thb::THBSplineSpace<T>& space, std::size_t rank) {
  const auto d = static_cast<double>(space.dim());
  int p_max = 0;
  for (const int p : space.degrees()) {
    p_max = (std::max)(p_max, p);
  }
  const auto width = static_cast<double>(p_max) + 1.0;
  const double tabulation = static_cast<double>(space.num_levels()) * d * 2.0 * width * width;
  return tabulation + functions_per_cell_bound(space) * (d + 2.0 * static_cast<double>(rank));
}

/// Operations one function's chain of coefficient boxes costs in `MultiLevelExtraction`, as
/// `build_boxes` estimates it for the longest chain (a function met at level 0 and on cells of the
/// deepest level).
template <class T>
double box_chain_work(const thb::THBSplineSpace<T>& space) {
  const std::size_t d = space.dim();
  const grid::HierarchicalGrid grid = space.grid();
  std::int64_t most_active = 0;
  for (const std::int64_t count : space.num_basis_per_level()) {
    most_active = (std::max)(most_active, count);
  }
  double per_entry = static_cast<double>(d) + std::log2(static_cast<double>(most_active) + 1.0);
  std::vector<double> width(d, 1.0);
  for (std::size_t k = 0; k < d; ++k) {
    per_entry += 2.0 * (static_cast<double>(space.degrees()[k]) + 1.0);
  }
  double total = 0.0;
  for (std::int64_t s = 1; s < space.num_levels(); ++s) {
    double entries = 1.0;
    for (std::size_t k = 0; k < d; ++k) {
      const auto p = static_cast<double>(space.degrees()[k]);
      width[k] = static_cast<double>(grid.factor()[k]) * (width[k] + p) - p;
      entries *= width[k];
    }
    total += entries * per_entry;
  }
  return (std::max)(1.0, total);
}

/// Whether a call of `MultiLevelExtraction` may build the boxes (once per object) or the cell table
/// on the pool.
template <class T>
bool extraction_on_pool(const thb::THBSplineSpace<T>& space) {
  return table_on_pool(space) ||
         loop_on_pool(static_cast<std::size_t>(space.num_total_basis()), box_chain_work(space));
}

template <class T>
void def_thb_space(nb::module_& m, const char* name) {
  using S = thb::THBSplineSpace<T>;
  using Restriction = thb::THBSplineSpaceRestriction<T>;
  nb::class_<S>(m, name, "An immutable truncated hierarchical B-spline space.")
      .def(
          "__init__",
          [](S* self, const bspline::BsplineSpace<T>& root, const grid::HierarchicalGrid& grid,
             bool truncate, const std::vector<std::optional<long long>>& regularity) {
            new (self) S(root, grid, truncate, narrow_regularity(regularity));
          },
          nb::arg("root"), nb::arg("grid"), nb::arg("truncate"), nb::arg("regularity"),
          "Build the space on a root space and a grid; regularity has one entry per direction.")
      .def_prop_ro(
          "grid", [](const S& s) { return s.grid(); }, "The grid the space owns.")
      .def_prop_ro("num_levels", &S::num_levels)
      .def_prop_ro("num_total_basis", &S::num_total_basis)
      .def_prop_ro("num_basis_per_level", &S::num_basis_per_level)
      .def_prop_ro(
          "func_offset", [](const S& s) { return view_1d(s.func_offset(), s.rep()); },
          nb::rv_policy::automatic, "Read-only view of the first dof of each level.")
      .def(
          "level_space_1d",
          [](const S& s, std::int64_t level) { return s.level_space(level).spaces(); },
          nb::arg("level"), "The 1D spaces of the tensor-product space at a level.")
      .def(
          "active_function_indices",
          [](const S& s, std::int64_t level) {
            return copy_span(s.active_function_indices(level));
          },
          nb::arg("level"), "A fresh array of the flat indices of the active functions of a level.")
      .def(
          "active_basis",
          [](const S& s, std::int64_t cid) {
            std::vector<std::int64_t> dofs = [&] {
              const ReleaseGil release(table_on_pool(s));
              const std::span<const std::int64_t> view = s.active_basis(cid);
              return std::vector<std::int64_t>(view.begin(), view.end());
            }();
            return vector_to_numpy(std::move(dofs));
          },
          nb::arg("cid"), "A fresh array of the global dofs whose support meets a cell.")
      .def(
          "num_active_on",
          [](const S& s, std::int64_t cid) {
            const ReleaseGil release(table_on_pool(s));
            return static_cast<std::int64_t>(s.num_active_on(cid));
          },
          nb::arg("cid"), "The number of functions whose support meets a cell.")
      .def(
          "max_active_per_cell",
          [](const S& s) {
            const ReleaseGil release(table_on_pool(s));
            return s.max_active_per_cell();
          },
          "The largest number of active functions on a cell.")
      .def(
          "tabulate_basis",
          [](const S& s, std::int64_t cid, const F64Mat& pts,
             const std::vector<std::int64_t>& orders) {
            thb::TabulatedBasis tab = [&] {
              const ReleaseGil release(table_on_pool(s));
              return s.tabulate_basis(cid, view_of_points(pts), orders);
            }();
            return std::make_pair(to_numpy(std::move(tab.values)), to_numpy(std::move(tab.dofs)));
          },
          nb::arg("cid"), nb::arg("pts"), nb::arg("orders"),
          "(values (n, K), dofs (K,)) of the mixed partial of the functions on a cell.")
      .def(
          "restrict",
          [](const S& s, const I64Vec& cell_ids) {
            Restriction r = s.restrict(span_of(cell_ids));
            return std::make_tuple(std::move(r.space),
                                   vector_to_numpy(std::move(r.local_to_global_dof)),
                                   vector_to_numpy(std::move(r.local_to_global_cell)));
          },
          nb::arg("cell_ids"),
          "(windowed space, local_to_global_dof, local_to_global_cell) of the cells' window.")
      .def(
          "refine",
          [](const S& s, const I64Vec& cell_ids, std::optional<std::int64_t> admissible_class) {
            return s.refine(span_of(cell_ids), admissible_class);
          },
          nb::arg("cell_ids"), nb::arg("admissible_class").none(),
          "The space with the named cells refined.")
      .def(
          "refine_region",
          [](const S& s, std::int64_t level, const std::vector<std::int64_t>& lo,
             const std::vector<std::int64_t>& hi, std::optional<std::int64_t> admissible_class) {
            return s.refine_region(level, lo, hi, admissible_class);
          },
          nb::arg("level"), nb::arg("lo"), nb::arg("hi"), nb::arg("admissible_class").none(),
          "The space with a region refined.")
      .def(
          "coarsen",
          [](const S& s, const I64Vec& cell_ids, std::optional<std::int64_t> admissible_class) {
            return s.coarsen(span_of(cell_ids), admissible_class);
          },
          nb::arg("cell_ids"), nb::arg("admissible_class").none(),
          "The space with the named cells coarsened away.")
      .def(
          "prolongation_csr",
          [](const S& s, const S& fine) {
            thb::ProlongationCsr p = [&] {
              // The columns are solved on the pool.
              const nb::gil_scoped_release release;
              return s.prolongation_csr(fine);
            }();
            return std::make_tuple(p.rows, p.cols, vector_to_numpy(std::move(p.indptr)),
                                   vector_to_numpy(std::move(p.indices)),
                                   vector_to_numpy(std::move(p.data)));
          },
          nb::arg("fine"),
          "(rows, cols, indptr, indices, data) of the prolongation to a refinement, in CSR form.");
}

template <class T>
void def_thb_spline(nb::module_& m, const char* name) {
  using S = thb::THBSplineSpace<T>;
  using F = thb::THBSpline<T>;
  nb::class_<F>(m, name, "An immutable THB spline function.")
      .def(
          "__init__",
          [](F* self, const S& space, const F64Mat& control_points) {
            NdArray<double> cp({control_points.shape(0), control_points.shape(1)}, for_overwrite);
            std::copy(control_points.data(), control_points.data() + control_points.size(),
                      cp.data());
            new (self) F(space, std::move(cp));
          },
          nb::arg("space"), nb::arg("control_points"),
          "Copy `(num_total_basis, rank)` control points onto a space.")
      .def_prop_ro(
          "control_points",
          [](const F& f) {
            const std::size_t shape[2] = {f.shape()[0], f.shape()[1]};
            return view_of<double>(f.control_points().data(), shape, f.rep());
          },
          nb::rv_policy::automatic, "Read-only view of the control points.")
      .def(
          "evaluate",
          [](const F& f, const F64Mat& pts, const std::vector<std::int64_t>& orders) {
            NdArray<double> out = [&] {
              // Two loops run on the pool: the location of the points and the cells' blocks.
              const double ops = (std::max)(evaluate_point_work(f.space(), f.rank()),
                                            f.space().grid().locate_work().ops_per_item);
              const ReleaseGil release(table_on_pool(f.space()) || loop_on_pool(pts.shape(0), ops));
              return f.evaluate(view_of_points(pts), orders);
            }();
            return to_numpy(std::move(out));
          },
          nb::arg("pts"), nb::arg("orders"),
          "The mixed partial of the spline at the rows of `(m, dim)` points: `(m, rank)`.")
      .def(
          "prolong_to",
          [](const F& f, const S& fine) {
            const nb::gil_scoped_release release;  // the control points are prolonged on the pool
            return f.prolong_to(fine);
          },
          nb::arg("fine"), "The same function on a refinement of the space.");
}

template <class T>
void def_multilevel_extraction(nb::module_& m, const char* name) {
  using S = thb::THBSplineSpace<T>;
  using E = thb::MultiLevelExtraction<T>;
  nb::class_<E>(m, name, "Per-element multi-level extraction operators of a THB space.")
      .def(
          "__init__",
          [](E* self, const S& space, int target) {
            new (self) E(space, bspline::to_extraction_target(target));
          },
          nb::arg("space"), nb::arg("target"),
          "The extraction onto a target basis (0 Bezier, 1 Lagrange, 2 cardinal).")
      .def(
          "num_active_on",
          [](const E& e, std::int64_t cid) {
            const ReleaseGil release(table_on_pool(e.space()));
            return static_cast<std::int64_t>(e.space().num_active_on(cid));
          },
          nb::arg("cid"), "The rows of the operators on a cell.")
      .def(
          "multilevel_operator",
          [](const E& e, std::int64_t cid, OutF64Mat& out) {
            const ReleaseGil release(extraction_on_pool(e.space()));
            e.multilevel_operator(cid, mdview<double, 2>(out.data(), out.shape(0), out.shape(1)));
          },
          nb::arg("cid"), nb::arg("out").noconvert(),
          "Write the multi-level extraction operator of a cell into `out`.")
      .def(
          "extraction_operator",
          [](const E& e, std::int64_t cid, OutF64Mat& out) {
            const ReleaseGil release(extraction_on_pool(e.space()));
            e.extraction_operator(cid, mdview<double, 2>(out.data(), out.shape(0), out.shape(1)));
          },
          nb::arg("cid"), nb::arg("out").noconvert(),
          "Write the multi-level Bezier (target) extraction operator of a cell into `out`.");
}

template <class T>
void def_quasi_interpolation(nb::module_& m, const char* name) {
  using S = thb::THBSplineSpace<T>;
  using Q = thb::THBQuasiInterpolation<T>;
  nb::class_<Q>(m, name, "The staged quasi-interpolant onto a THB space.")
      .def(
          "__init__",
          [](Q* self, const S& space) {
            const ReleaseGil release(table_on_pool(space));
            new (self) Q(space);
          },
          nb::arg("space"), "Choose each function's leaf cell and build its points and weights.")
      .def(
          "points", [](const Q& q) { return to_numpy(q.points()); },
          "The sample points `(num_functions * block, dim)`.")
      .def(
          "coefficients",
          [](const Q& q, const F64Mat& values) {
            // The contraction of one function costs two operations per entry of each successive
            // intermediate array, as `THBQuasiInterpolation::coefficients` estimates it.
            double ops = 0.0;
            double size = static_cast<double>(q.block()) * static_cast<double>(values.shape(1));
            for (const int p : q.space().degrees()) {
              ops += 2.0 * size;
              size /= static_cast<double>(p) + 1.0;
            }
            NdArray<double> out = [&] {
              const ReleaseGil release(loop_on_pool(q.num_functions(), ops));
              return q.coefficients(view_of_points(values));
            }();
            return to_numpy(std::move(out));
          },
          nb::arg("values"), "The coefficients `(num_functions, rank)` from the samples.");
}

template <class T>
void def_windowing(nb::module_& m) {
  using S = thb::THBSplineSpace<T>;
  m.def(
      "compute_halo",
      [](const bspline::BsplineSpace<T>& space, const I64Vec& owned_cells) {
        // The closure of a cell spans at most `2 degree + 1` cells per direction.
        double ops = 1.0;
        for (const auto& s : space.spaces()) {
          ops *= 2.0 * static_cast<double>(s.degree()) + 1.0;
        }
        std::vector<std::int64_t> halo = [&] {
          const ReleaseGil release(loop_on_pool(owned_cells.size(), ops));
          return bspline::compute_halo(space, span_of(owned_cells));
        }();
        return vector_to_numpy(std::move(halo));
      },
      nb::arg("space"), nb::arg("owned_cells"),
      "The support-closure halo of owned cells of a tensor-product space, ascending.");
  m.def(
      "dof_owner_windowed",
      [](const bspline::BsplineSpace<T>& space, const I32Vec& cell_owner, const I64Vec& dofs) {
        // The support of a function spans at most `degree + 1` cells per direction.
        double ops = static_cast<double>(space.dim());
        double walk = 1.0;
        for (const auto& s : space.spaces()) {
          walk *= static_cast<double>(s.degree()) + 1.0;
        }
        std::vector<std::int32_t> owners = [&] {
          const ReleaseGil release(loop_on_pool(dofs.size(), ops + walk));
          return bspline::dof_owner_windowed(space, span_of(cell_owner), span_of(dofs));
        }();
        return vector_to_numpy(std::move(owners));
      },
      nb::arg("space"), nb::arg("cell_owner"), nb::arg("dofs"),
      "The owner of the requested functions of a tensor-product space, in request order.");
  m.def(
      "dof_owner",
      [](const bspline::BsplineSpace<T>& space, const I32Vec& cell_owner) {
        std::int64_t n_dofs = 1;
        double walk = 1.0;
        for (const auto& s : space.spaces()) {
          n_dofs *= s.num_basis();
          walk *= static_cast<double>(s.degree()) + 1.0;
        }
        std::vector<std::int32_t> owners = [&] {
          const ReleaseGil release(loop_on_pool(static_cast<std::size_t>(n_dofs),
                                                static_cast<double>(space.dim()) + walk));
          return bspline::dof_owner(space, span_of(cell_owner));
        }();
        return vector_to_numpy(std::move(owners));
      },
      nb::arg("space"), nb::arg("cell_owner"),
      "The owner of every function of a tensor-product space.");
  m.def(
      "thb_compute_halo",
      [](const S& space, const I64Vec& owned_cells) {
        // One cell costs one read per function it holds, at most the per-cell bound.
        std::vector<std::int64_t> halo = [&] {
          const ReleaseGil release(table_on_pool(space) ||
                                   loop_on_pool(static_cast<std::size_t>(space.grid().num_cells()),
                                                functions_per_cell_bound(space)));
          return thb::compute_halo(space, span_of(owned_cells));
        }();
        return vector_to_numpy(std::move(halo));
      },
      nb::arg("space"), nb::arg("owned_cells"),
      "The cross-level support-closure halo of owned cells of a THB space, ascending.");
  m.def(
      "thb_dof_owner",
      [](const S& space, const I32Vec& cell_owner) {
        std::vector<std::int32_t> owners = [&] {
          const ReleaseGil release(table_on_pool(space));  // the pass over the table is serial
          return thb::dof_owner(space, span_of(cell_owner));
        }();
        return vector_to_numpy(std::move(owners));
      },
      nb::arg("space"), nb::arg("cell_owner"), "The owner of every function of a THB space.");
  m.def(
      "coupling_graph_thb",
      [](const S& space, const std::optional<F64Vec>& cell_weights) {
        std::optional<std::span<const double>> weights;
        if (cell_weights) {
          weights = span_of(*cell_weights);
        }
        const partition::CouplingGraph graph = [&] {
          const ReleaseGil release(table_on_pool(space));
          return thb::coupling_graph(space, weights);
        }();
        return graph_views(graph);
      },
      nb::arg("space"), nb::arg("cell_weights").none(),
      "The coupling graph of a THB space: (n, xadj, adjncy, edge_weights, vertex_weights), "
      "read-only.");
}

template <class T>
void def_support(nb::module_& m) {
  m.def(
      "func_support_1d",
      [](const bspline::BsplineSpace1D<T>& space) {
        const bspline::detail::Support1D s = bspline::detail::func_support_1d(space);
        return std::make_tuple(copy_span(std::span<const std::int64_t>(s.first_basis)),
                               copy_span(std::span<const std::int64_t>(s.first_cell)),
                               copy_span(std::span<const std::int64_t>(s.last_cell)));
      },
      nb::arg("space"), "(first_basis, first_cell, last_cell) of a 1D space's functions.");
}

}  // namespace

void register_thb(nb::module_& m) {
  def_thb_space<double>(m, "THBSplineSpaceF64");
  def_thb_space<float>(m, "THBSplineSpaceF32");
  def_thb_spline<double>(m, "THBSplineF64");
  def_thb_spline<float>(m, "THBSplineF32");
  def_multilevel_extraction<double>(m, "MultiLevelExtractionF64");
  def_multilevel_extraction<float>(m, "MultiLevelExtractionF32");
  def_quasi_interpolation<double>(m, "THBQuasiInterpolationF64");
  def_quasi_interpolation<float>(m, "THBQuasiInterpolationF32");
  def_windowing<double>(m);
  def_windowing<float>(m);
}

void register_thb_testing(nb::module_& m) {
  m.def(
      "cell_membership_tolerance",
      [](double lo, double hi) { return thb::cell_membership_tolerance(lo, hi); }, nb::arg("lo"),
      nb::arg("hi"), "The slack allowed at the boundary of the cell `[lo, hi]` along one axis.");
  m.def(
      "prolongation_residual_tolerance",
      [](double max_coarse_val) { return thb::prolongation_residual_tolerance(max_coarse_val); },
      nb::arg("max_coarse_val"), "The residual above which a coarse column is not reproduced.");
  def_support<double>(m);
  def_support<float>(m);
}

}  // namespace pantr::bindings
