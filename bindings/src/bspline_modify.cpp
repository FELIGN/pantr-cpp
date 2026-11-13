// _pantr.bspline: the operations that rebuild a Bspline over other knots or another degree: degree
// elevation and reduction, knot removal and the derivative, with the common refinement of two
// B-splines the rational derivative calls; and, in _testing, the tolerance helpers of knot
// removal, the binomial kernel and the 1D degree kernels the tests reach.
//
// This translation unit reaches Eigen through the periodic directions of elevation and reduction
// (the QR of the periodic conversion), which the module's vectorization settings already cover for
// every unit (EIGEN_DONT_VECTORIZE on MSVC's v4 rung). Nothing here uses the pool, so no call
// releases the GIL. Arguments arrive as the wrapper normalised them (C-contiguous arrays of the
// exact dtype); validation of values is the C++ API's, with the messages the Python suite matches.
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/degree.hpp"
#include "pantr/bspline/derivative.hpp"
#include "pantr/bspline/knot_removal.hpp"
#include "pantr/core/binomial.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

/// The columns of a control net handed to a tolerance helper, refused at zero: the helpers divide
/// the flat length by it, which is a division by zero (a hardware trap, not an exception) for a
/// net of no columns.
template <class T>
std::size_t columns_of(const In2D<T>& ctrl) {
  if (ctrl.shape(1) == 0) {
    throw value_error("ctrl needs at least one column per control point, got shape (" +
                      std::to_string(ctrl.shape(0)) + ", 0).");
  }
  return ctrl.shape(1);
}

/// One direction's knot values, or none to skip the direction.
template <class T>
using MaybeKnots = std::optional<In1D<T>>;

/// A knot vector as a NumPy array.
template <class T>
NumpyArray<T> knots_to_numpy(const std::vector<T>& v) {
  NdArray<T> a({v.size()}, for_overwrite);
  std::copy(v.begin(), v.end(), a.data());
  return to_numpy(std::move(a));
}

/// A knot vector and its control points as a pair of NumPy arrays.
template <class T>
std::pair<NumpyArray<T>, NumpyArray<T>> to_pair(bspline::KnotsAndCtrl<T>&& r) {
  NumpyArray<T> knots = knots_to_numpy(r.knots);
  return {std::move(knots), to_numpy(std::move(r.ctrl))};
}

template <class T>
void def_modify(nb::class_<bspline::Bspline<T>>& cls) {
  using B = bspline::Bspline<T>;
  cls.def(
         "elevate_degree",
         [](const B& b, const std::vector<long long>& increments) {
           return bspline::elevate_degree<T>(b, increments);
         },
         nb::arg("increments"),
         "Elevate the degree in each direction by its increment (a periodic direction is "
         "elevated through its open form).")
      .def(
          "reduce_degree",
          [](const B& b, const std::vector<long long>& decrements) {
            return bspline::reduce_degree<T>(b, decrements);
          },
          nb::arg("decrements"),
          "Reduce the degree in each direction by its decrement (an approximation).")
      .def(
          "remove_knots",
          [](const B& b, const std::vector<MaybeKnots<T>>& per_dim, std::optional<long long> num,
             std::optional<double> tol) {
            std::vector<std::span<const T>> views;
            views.reserve(per_dim.size());
            for (const auto& k : per_dim) {
              views.push_back(k ? as_span(*k) : std::span<const T>());
            }
            return bspline::remove_knots<T>(b, views, num, tol);
          },
          nb::arg("knot_values"), nb::arg("num").none(), nb::arg("tol").none(),
          "Remove the knots per direction (none or empty skips a direction), up to `num` times "
          "each, while the deviation stays within `tol` (the round-off floor when none).")
      .def(
          "derivative",
          [](const B& b, long long direction) {
            return bspline::derivative_nonrational<T>(b, direction);
          },
          nb::arg("direction"), "The partial derivative of a non-rational B-spline.")
      .def(
          "derivative_keep_degree",
          [](const B& b, long long direction) {
            return bspline::derivative_keep_degree_nonrational<T>(b, direction);
          },
          nb::arg("direction"),
          "The partial derivative of a non-rational B-spline, elevated back to its degree.");
}

template <class T>
void def_functions(nb::module_& m, const std::string& suffix) {
  using B = bspline::Bspline<T>;
  m.def(("refine_to_common_space_1d" + suffix).c_str(),
        [](const B& a, const B& b, std::size_t direction) {
          return bspline::refine_to_common_space_1d<T>(a, b, direction);
        },
        nb::arg("a"), nb::arg("b"), nb::arg("direction"),
        "Insert into each of two B-splines the knots the other has in one direction and it lacks.");
}

template <class T>
void def_testing(nb::module_& m, const std::string& suffix) {
  m.def(("control_point_scale" + suffix).c_str(),
        [](const In2D<T>& ctrl) {
          return bspline::control_point_scale<T>(std::span<const T>(ctrl.data(), ctrl.size()),
                                                 columns_of(ctrl));
        },
        nb::arg("ctrl").noconvert(),
        "max(bounding-box diagonal, largest norm) of the rows of (points, coordinates).");
  m.def(("roundoff_deviation_floor" + suffix).c_str(),
        [](const In2D<T>& ctrl, long long degree, std::size_t num_stacked) {
          return bspline::roundoff_deviation_floor<T>(std::span<const T>(ctrl.data(), ctrl.size()),
                                                      columns_of(ctrl), saturate_degree(degree),
                                                      num_stacked);
        },
        nb::arg("ctrl").noconvert(), nb::arg("degree"), nb::arg("num_stacked"),
        "The deviation budget that admits exactly the removals round-off allows.");
  m.def(("homogeneous_deviation_tolerance" + suffix).c_str(),
        [](const In2D<T>& ctrl, double tol) {
          return bspline::homogeneous_deviation_tolerance<T>(
              std::span<const T>(ctrl.data(), ctrl.size()), columns_of(ctrl), tol);
        },
        nb::arg("ctrl").noconvert(), nb::arg("tol_euclidean"),
        "A projected-space deviation budget pulled back to homogeneous control points.");
  m.def(("find_knot_index_and_multiplicity" + suffix).c_str(),
        [](const In1D<T>& knots, long long degree, double knot_value, double tol) {
          const auto [r, s] = bspline::detail::find_knot_index_and_multiplicity<T>(
              as_span(knots), saturate_degree(degree), knot_value, tol);
          return std::make_pair(static_cast<long long>(r), static_cast<long long>(s));
        },
        nb::arg("knots").noconvert(), nb::arg("degree"), nb::arg("knot_value"), nb::arg("tol"),
        "The last index and the multiplicity of the knot class a value falls in.");
  m.def(("degree_elevate_1d" + suffix).c_str(),
        [](long long degree, const In2D<T>& ctrl, const In1D<T>& knots, long long increment) {
          return to_pair(bspline::elevate_degree_1d<T>(saturate_degree(degree), as_view(ctrl),
                                                       as_span(knots), saturate_degree(increment)));
        },
        nb::arg("degree"), nb::arg("ctrl").noconvert(), nb::arg("knots").noconvert(),
        nb::arg("increment"),
        "Degree elevate a 1D B-spline: (control points (n, rank), knots) -> (points, knots).");
  m.def(
      ("degree_reduce_1d" + suffix).c_str(),
      [](long long degree, const In2D<T>& ctrl, const In1D<T>& knots, long long decrement) {
        return to_pair(bspline::reduce_degree_1d<T>(saturate_degree(degree), as_view(ctrl),
                                                    as_span(knots), saturate_degree(decrement)));
      },
      nb::arg("degree"), nb::arg("ctrl").noconvert(), nb::arg("knots").noconvert(),
      nb::arg("decrement"),
      "Degree reduce a 1D B-spline to Bezier form: (points (n, rank), knots) -> (points, knots).");
}

}  // namespace

void register_bspline_modify(nb::module_& m) {
  // The classes exist already (register_bspline_ops runs first); their methods are bound here, in
  // the translation unit that reaches Eigen through the periodic directions.
  auto f64 = nb::borrow<nb::class_<bspline::Bspline<double>>>(m.attr("BsplineF64"));
  auto f32 = nb::borrow<nb::class_<bspline::Bspline<float>>>(m.attr("BsplineF32"));
  def_modify<double>(f64);
  def_modify<float>(f32);
  def_functions<double>(m, "");
  def_functions<float>(m, "_f32");
}

void register_bspline_modify_testing(nb::module_& m) {
  m.attr("bincoeff_max_n") = pantr::bincoeff_max_n;
  m.def(
      "bincoeff", [](long long n, long long k) { return pantr::bincoeff(n, k); }, nb::arg("n"),
      nb::arg("k"), "The exact-integer binomial coefficient C(n, k) as a double (n <= 61).");
  m.def(
      "check_bincoeff_envelope",
      [](long long n, const std::string& what) { pantr::check_bincoeff_envelope(n, what); },
      nb::arg("n"), nb::arg("what"),
      "Refuse an upper index beyond the exactness envelope of the binomial kernel.");
  m.def(
      "num_stacked_control_points",
      [](const std::vector<std::size_t>& shape, std::size_t axis) {
        return bspline::num_stacked_control_points(shape, axis);
      },
      nb::arg("shape"), nb::arg("axis"),
      "The control points the removal kernel's one distance spans: the product of the extents "
      "other than `axis` (the last extent, the coordinates, excluded).");
  def_testing<double>(m, "");
  def_testing<float>(m, "_f32");
}

}  // namespace pantr::bindings
