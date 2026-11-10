// _pantr.bspline: the Bézier, Lagrange and cardinal extraction operators of a 1D space, the
// conversion of a Bspline to Bézier form; and, in _testing, the raw extraction operators and
// cardinal intervals on a knot vector with an explicit tolerance.
//
// This translation unit reaches Eigen: the Lagrange and cardinal operators fetch their
// change-of-basis matrices from the process tables, whose builders solve with Eigen's LU. The
// periodic directions of `to_beziers` go through the conversion to open form, which does not.
// The decomposition runs on the pool, so `to_beziers` releases the GIL when its work warrants it;
// the Python objects of the result are built after the release. Arguments arrive as the wrapper
// normalised them (C-contiguous arrays of the exact dtype and shape); validation of values is the
// C++ API's, with the messages the Python suite matches.
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/basis/lagrange_variant.hpp"
#include "pantr/bezier/bezier.hpp"
#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/extraction.hpp"
#include "pantr/bspline/knots.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/bspline/to_beziers.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/mdspan.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstddef>
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
using Out3D = nb::ndarray<nb::numpy, T, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
using OutBool1D = nb::ndarray<nb::numpy, bool, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
std::span<const T> as_span(const In1D<T>& a) {
  return {a.data(), a.shape(0)};
}

template <class T>
mdview<T, 3> as_view(Out3D<T>& a) {
  return mdview<T, 3>(a.data(), a.shape(0), a.shape(1), a.shape(2));
}

/// Work one `to_beziers` call does, for the GIL decision only: per direction, every control point
/// is touched `degree + 1` times (a multiply-add each).
template <class T>
double decomposition_ops(const bspline::Bspline<T>& b) {
  double order = 1.0;
  const bspline::BsplineSpace<T> whole = b.space();
  for (const auto& s : whole.spaces()) {
    order = (std::max)(order, static_cast<double>(s.degree()) + 1.0);
  }
  return 2.0 * order * static_cast<double>(b.size()) * static_cast<double>(b.dim());
}

template <class T>
void def_space_operators(nb::class_<bspline::BsplineSpace1D<T>>& cls) {
  using S = bspline::BsplineSpace1D<T>;
  cls.def(
         "bezier_extraction",
         [](const S& s, Out3D<T>& out) {
           bspline::bezier_extraction_operators<T>(s, as_view(out));
         },
         nb::arg("out").noconvert(),
         "Write the Bezier extraction operators (intervals, degree + 1, degree + 1) into `out`.")
      .def(
          "lagrange_extraction",
          [](const S& s, int variant, Out3D<T>& out) {
            bspline::lagrange_extraction_operators<T>(s, basis::to_lagrange_variant(variant),
                                                      as_view(out));
          },
          nb::arg("variant"), nb::arg("out").noconvert(),
          "Write the Lagrange extraction operators of a node variant into `out`.")
      .def(
          "cardinal_extraction",
          [](const S& s, Out3D<T>& out) {
            bspline::cardinal_extraction_operators<T>(s, as_view(out));
          },
          nb::arg("out").noconvert(),
          "Write the cardinal B-spline extraction operators into `out`.")
      .def(
          "bezier_identity_mask",
          [](const S& s, OutBool1D& out) {
            bspline::bezier_structural_identity_mask<T>(s,
                                                        std::span<bool>(out.data(), out.shape(0)));
          },
          nb::arg("out").noconvert(),
          "Write into `out` which intervals are already Bezier patches (identity operator).");
}

template <class T>
void def_conversions(nb::class_<bspline::Bspline<T>>& cls) {
  using B = bspline::Bspline<T>;
  cls.def(
         "to_bezier", [](const B& b, bool copy) { return bspline::to_bezier<T>(b, copy); },
         nb::arg("copy"),
         "The Bezier of a single-element B-spline (periodic and unclamped ones through open "
         "form); with `copy` false a Bezier-like B-spline shares its buffer.")
      .def(
          "to_beziers",
          [](const B& b) {
            bspline::BezierDecomposition<T> r = [&] {
              const ReleaseIfParallel gil(1, decomposition_ops<T>(b));
              return bspline::to_beziers<T>(b);
            }();
            return std::make_pair(std::move(r.extents), std::move(r.beziers));
          },
          "The Bezier patches of every element, in C order, and the elements per direction.");
}

template <class T>
void def_testing(nb::module_& m, const std::string& suffix) {
  m.def(("bezier_extraction_1d_into" + suffix).c_str(),
        [](const In1D<T>& knots, long long degree, double tol, Out3D<T>& out) {
          bspline::bezier_extraction_operators<T>(as_span(knots), saturate_degree(degree), tol,
                                                  as_view(out));
        },
        nb::arg("knots").noconvert(), nb::arg("degree"), nb::arg("tol"), nb::arg("out").noconvert(),
        "Write the Bezier extraction operators into `out`.");
  m.def(("lagrange_extraction_1d_into" + suffix).c_str(),
        [](const In1D<T>& knots, long long degree, double tol, int variant, Out3D<T>& out) {
          bspline::lagrange_extraction_operators<T>(as_span(knots), saturate_degree(degree), tol,
                                                    basis::to_lagrange_variant(variant),
                                                    as_view(out));
        },
        nb::arg("knots").noconvert(), nb::arg("degree"), nb::arg("tol"), nb::arg("variant"),
        nb::arg("out").noconvert(), "Write the Lagrange extraction operators into `out`.");
  m.def(("cardinal_extraction_1d_into" + suffix).c_str(),
        [](const In1D<T>& knots, long long degree, double tol, Out3D<T>& out) {
          bspline::cardinal_extraction_operators<T>(as_span(knots), saturate_degree(degree), tol,
                                                    as_view(out));
        },
        nb::arg("knots").noconvert(), nb::arg("degree"), nb::arg("tol"), nb::arg("out").noconvert(),
        "Write the cardinal extraction operators into `out`.");
  m.def(("num_intervals_1d" + suffix).c_str(),
        [](const In1D<T>& knots, long long degree, double tol) {
          const int p = saturate_degree(degree);
          if (tol < 0) {
            throw value_error("tol must be non-negative");
          }
          bspline::detail::check_knots_and_degree<T>(as_span(knots), p);
          return bspline::unique_knots_and_multiplicity<T>(as_span(knots), p, tol, true)
                     .multiplicities.size() -
                 1;
        },
        nb::arg("knots").noconvert(), nb::arg("degree"), nb::arg("tol"),
        "The number of in-domain intervals of a knot vector with an explicit tolerance.");
  m.def(("cardinal_intervals_1d_into" + suffix).c_str(),
        [](const In1D<T>& knots, long long degree, double tol, OutBool1D& out) {
          const int p = saturate_degree(degree);
          if (tol < 0) {
            throw value_error("tol must be non-negative");
          }
          bspline::detail::check_knots_and_degree<T>(as_span(knots), p);
          const std::size_t n =
              bspline::unique_knots_and_multiplicity<T>(as_span(knots), p, tol, true)
                  .multiplicities.size() -
              1;
          if (out.shape(0) != n) {
            throw value_error("Output array has shape (" + std::to_string(out.shape(0)) +
                              ",), but expected shape (" + std::to_string(n) + ",)");
          }
          bspline::cardinal_intervals<T>(as_span(knots), p, tol, std::span<bool>(out.data(), n));
        },
        nb::arg("knots").noconvert(), nb::arg("degree"), nb::arg("tol"), nb::arg("out").noconvert(),
        "Write the cardinal-interval flags of a knot vector with an explicit tolerance.");
}

}  // namespace

void register_bspline_extraction(nb::module_& m) {
  // The classes exist already (register_bspline and register_bspline_ops run first).
  auto s64 = nb::borrow<nb::class_<bspline::BsplineSpace1D<double>>>(m.attr("BsplineSpace1DF64"));
  auto s32 = nb::borrow<nb::class_<bspline::BsplineSpace1D<float>>>(m.attr("BsplineSpace1DF32"));
  def_space_operators<double>(s64);
  def_space_operators<float>(s32);
  auto b64 = nb::borrow<nb::class_<bspline::Bspline<double>>>(m.attr("BsplineF64"));
  auto b32 = nb::borrow<nb::class_<bspline::Bspline<float>>>(m.attr("BsplineF32"));
  def_conversions<double>(b64);
  def_conversions<float>(b32);
}

void register_bspline_extraction_testing(nb::module_& m) {
  def_testing<double>(m, "");
  def_testing<float>(m, "_f32");
}

}  // namespace pantr::bindings
