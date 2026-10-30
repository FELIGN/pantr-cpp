// _pantr.bspline: knot insertion, conversion to open and periodic form, subdivision, restriction
// and splitting of a Bspline; the two knot-vector functions BsplineSpace1D shares; the 1D kernels
// the Python callers not yet ported reach (_kernels); and the blossom (_testing).
//
// This translation unit reaches Eigen (the periodic conversion's QR), which the module's
// vectorization settings already cover for every unit (EIGEN_DONT_VECTORIZE on MSVC's v4 rung).
// Nothing here uses the pool, so no call releases the GIL. Arguments arrive as the wrapper
// normalised them (C-contiguous arrays of the exact dtype); validation of values is the C++
// API's, with the messages the Python suite matches.
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/bspline/blossom.hpp"
#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/knot_insertion.hpp"
#include "pantr/bspline/periodic.hpp"
#include "pantr/bspline/restrict.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

template <class T>
using In1D = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
template <class T>
using In2D = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <class T>
std::span<const T> as_span(const In1D<T>& a) {
  return {a.data(), a.shape(0)};
}

template <class T>
mdview<const T, 2> as_view(const In2D<T>& a) {
  return mdview<const T, 2>(a.data(), a.shape(0), a.shape(1));
}

template <class T>
NumpyArray<T> knots_to_numpy(const std::vector<T>& v) {
  NdArray<T> a({v.size()}, for_overwrite);
  std::copy(v.begin(), v.end(), a.data());
  return to_numpy(std::move(a));
}

/// A refined knot vector and its control points as a pair of NumPy arrays.
template <class T>
std::pair<NumpyArray<T>, NumpyArray<T>> to_pair(bspline::KnotsAndCtrl<T>&& r) {
  NumpyArray<T> knots = knots_to_numpy(r.knots);
  return {std::move(knots), to_numpy(std::move(r.ctrl))};
}

/// One direction's new knots, or none to skip the direction.
template <class T>
using MaybeKnots = std::optional<In1D<T>>;

template <class T>
void def_refine(nb::class_<bspline::Bspline<T>>& cls) {
  using B = bspline::Bspline<T>;
  cls.def(
         "insert_knots",
         [](const B& b, const std::vector<MaybeKnots<T>>& per_dim) {
           std::vector<std::span<const T>> views;
           views.reserve(per_dim.size());
           for (const auto& k : per_dim) {
             views.push_back(k ? as_span(*k) : std::span<const T>());
           }
           return bspline::insert_knots<T>(b, views);
         },
         nb::arg("new_knots"),
         "Insert knots per direction (none or empty skips a direction); a periodic direction is "
         "refined through its open form.")
      .def(
          "to_open", [](const B& b) { return bspline::to_open_bspline<T>(b); },
          "Convert every direction to open (clamped) non-periodic form.")
      .def(
          "to_periodic",
          [](const B& b, const std::vector<std::optional<long long>>& continuity) {
            return bspline::to_periodic<T>(b, continuity);
          },
          nb::arg("continuity"),
          "Convert the directions with a continuity (-1 for the maximum; none skips) to periodic "
          "form.")
      .def(
          "subdivide",
          [](const B& b, const std::vector<long long>& counts,
             std::optional<long long> regularity) {
            return bspline::subdivide<T>(b, counts, regularity);
          },
          nb::arg("counts"), nb::arg("regularity").none(),
          "Subdivide each direction whose count is at least 2 uniformly.")
      .def(
          "restrict",
          [](const B& b, const std::vector<std::optional<std::pair<double, double>>>& bounds) {
            std::vector<std::optional<bspline::Bounds>> bs;
            bs.reserve(bounds.size());
            for (const auto& x : bounds) {
              bs.push_back(x ? std::optional<bspline::Bounds>(bspline::Bounds{x->first, x->second})
                             : std::nullopt);
            }
            return bspline::restrict<T>(b, bs);
          },
          nb::arg("bounds"), "Restrict each direction with bounds to that interval.")
      .def(
          "split",
          [](const B& b, long long direction, double value) {
            return bspline::split<T>(b, direction, value);
          },
          nb::arg("direction"), nb::arg("value"), "Split at a parameter value in one direction.");
}

template <class T>
void def_functions(nb::module_& m, const std::string& suffix) {
  m.def(("compute_inserted_knot_vector" + suffix).c_str(),
        [](const In1D<T>& knots, long long degree, const In1D<T>& new_knots, double tol) {
          return knots_to_numpy(bspline::compute_inserted_knot_vector<T>(
              as_span(knots), saturate_degree(degree), as_span(new_knots), tol));
        },
        nb::arg("knots").noconvert(), nb::arg("degree"), nb::arg("new_knots").noconvert(),
        nb::arg("tol"), "Validate new knots and merge them into a knot vector.");
  m.def(("compute_uniform_subdivision_knots" + suffix).c_str(),
        [](const In1D<T>& knots, long long degree, double tol, long long n_subdivisions,
           std::optional<long long> regularity) {
          return knots_to_numpy(bspline::compute_uniform_subdivision_knots<T>(
              as_span(knots), saturate_degree(degree), tol, n_subdivisions, regularity));
        },
        nb::arg("knots").noconvert(), nb::arg("degree"), nb::arg("tol"), nb::arg("n_subdivisions"),
        nb::arg("regularity").none(), "The knots that subdivide every span uniformly.");
}

template <class T>
void def_kernels(nb::module_& m, const std::string& suffix) {
  m.def(("to_open_1d" + suffix).c_str(),
        [](const In1D<T>& knots, long long degree, const In2D<T>& ctrl, bool periodic, double tol) {
          return to_pair(bspline::to_open_1d<T>(as_span(knots), saturate_degree(degree),
                                                as_view(ctrl), periodic, tol));
        },
        nb::arg("knots").noconvert(), nb::arg("degree"), nb::arg("ctrl").noconvert(),
        nb::arg("periodic"), nb::arg("tol"),
        "One direction to open form: (knots, control points (n_stored, cols)) -> (knots, points).");
  m.def(("to_periodic_1d" + suffix).c_str(),
        [](const In1D<T>& knots, long long degree, const In2D<T>& ctrl, long long m_bdy, double tol,
           std::optional<double> tol_deviation) {
          return to_pair(bspline::to_periodic_1d<T>(as_span(knots), saturate_degree(degree),
                                                    as_view(ctrl), m_bdy, tol, tol_deviation));
        },
        nb::arg("open_knots").noconvert(), nb::arg("degree"), nb::arg("ctrl").noconvert(),
        nb::arg("m_bdy"), nb::arg("tol"), nb::arg("tol_deviation").none() = nb::none(),
        "One direction to periodic form: (knots, points) -> (periodic knots, periodic points).");
  m.def(("oslo_matrix" + suffix).c_str(),
        [](long long degree, const In1D<T>& old_knots, const In1D<T>& new_knots) {
          return to_numpy(bspline::oslo_matrix<T>(saturate_degree(degree), as_span(old_knots),
                                                  as_span(new_knots)));
        },
        nb::arg("degree"), nb::arg("old_knots").noconvert(), nb::arg("new_knots").noconvert(),
        "The dense Oslo refinement matrix (m + 1, n + 1).");
}

template <class T>
void def_blossom(nb::module_& m, const std::string& suffix) {
  m.def(("oslo_rows" + suffix).c_str(),
        [](long long degree, const In1D<T>& old_knots, const In1D<T>& new_knots) {
          const int p = saturate_degree(degree);
          // The kernel reads at least degree + 2 knots of each vector (the dense matrix checks).
          static_cast<void>(bspline::oslo_matrix<T>(p, as_span(old_knots), as_span(new_knots)));
          bspline::detail::OsloRows<T> rows =
              bspline::detail::oslo_rows<T>(p, as_span(old_knots), as_span(new_knots));
          NdArray<std::int64_t> first({rows.first_col.size()}, for_overwrite);
          std::copy(rows.first_col.begin(), rows.first_col.end(), first.data());
          return std::make_pair(to_numpy(std::move(rows.alphas)), to_numpy(std::move(first)));
        },
        nb::arg("degree"), nb::arg("old_knots").noconvert(), nb::arg("new_knots").noconvert(),
        "The banded rows of the Oslo matrix: (alphas (m + 1, degree + 1), first_col (m + 1,)).");
  m.def(("oslo_matrix" + suffix).c_str(),
        [](long long degree, const In1D<T>& old_knots, const In1D<T>& new_knots) {
          return to_numpy(bspline::oslo_matrix<T>(saturate_degree(degree), as_span(old_knots),
                                                  as_span(new_knots)));
        },
        nb::arg("degree"), nb::arg("old_knots").noconvert(), nb::arg("new_knots").noconvert(),
        "The dense Oslo refinement matrix (m + 1, n + 1).");
  m.def(("evaluate_blossom_1d" + suffix).c_str(),
        [](const In1D<T>& knots, long long degree, const In2D<T>& ctrl, const In1D<T>& u_values,
           double tol) {
          return knots_to_numpy(bspline::blossom<T>(as_span(knots), saturate_degree(degree),
                                                    as_view(ctrl), as_span(u_values), tol));
        },
        nb::arg("knots").noconvert(), nb::arg("degree"), nb::arg("control_points").noconvert(),
        nb::arg("u_values").noconvert(), nb::arg("tol"),
        "The blossom of a 1D B-spline at `degree` values: (rank,).");
  m.def(("blossom_span" + suffix).c_str(),
        [](const In1D<T>& knots, T u_last, long long n) {
          return static_cast<long long>(bspline::detail::blossom_span<T>(
              as_span(knots), u_last, static_cast<std::ptrdiff_t>(n)));
        },
        nb::arg("knots").noconvert(), nb::arg("u_last"), nb::arg("n"),
        "The span the blossom's recurrence starts from, in [0, n].");
}

}  // namespace

void register_bspline_refine(nb::module_& m) {
  // The classes exist already (register_bspline_ops runs first); their refinement methods are
  // bound here, in the translation unit that reaches Eigen.
  auto f64 = nb::borrow<nb::class_<bspline::Bspline<double>>>(m.attr("BsplineF64"));
  auto f32 = nb::borrow<nb::class_<bspline::Bspline<float>>>(m.attr("BsplineF32"));
  def_refine<double>(f64);
  def_refine<float>(f32);
  def_functions<double>(m, "");
  def_functions<float>(m, "_f32");
}

void register_bspline_kernels(nb::module_& m) {
  def_kernels<double>(m, "");
  def_kernels<float>(m, "_f32");
}

void register_bspline_blossom_testing(nb::module_& m) {
  def_blossom<double>(m, "");
  def_blossom<float>(m, "_f32");
}

}  // namespace pantr::bindings
