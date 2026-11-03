// _pantr.bspline: B-spline fitting (collocation on tensor-product and scattered nodes, L2
// projection, Lee-Lyche-Mørken quasi-interpolation). Each callable-based fit is bound in its staged
// form (design §3.6): a class built before any value is known says what to sample, and the Python
// wrapper calls the user's function and hands the values back. Reaches Eigen's LU and SVD (the
// solves' dense paths).
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/bspline/fit.hpp"
#include "pantr/bspline/l2_project.hpp"
#include "pantr/bspline/quasi_interpolate.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

/// A C-contiguous array of any rank.
template <class T>
using Any = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;

template <class T>
using Vec = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
using Mat = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

/// A zero divisor in the basis derivative recurrence: the error Python's float division raised.
[[noreturn]] void raise_zero_division() {
  PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
  throw nb::python_error();
}

template <class T>
std::vector<std::size_t> shape_of(const Any<T>& a) {
  std::vector<std::size_t> s(a.ndim());
  for (std::size_t i = 0; i < s.size(); ++i) {
    s[i] = a.shape(i);
  }
  return s;
}

template <class T>
std::span<const T> flat_of(const Any<T>& a) {
  return {a.data(), a.size()};
}

/// A copy of an `NdArray` as a fresh NumPy array.
template <class T>
NumpyArray<T> copy_out(const NdArray<T>& a) {
  NdArray<T> c(std::vector<std::size_t>(a.shape().begin(), a.shape().end()), for_overwrite);
  std::copy(a.data(), a.data() + a.size(), c.data());
  return to_numpy(std::move(c));
}

template <class T>
std::vector<NumpyArray<T>> list_out(const std::vector<NdArray<T>>& arrays) {
  std::vector<NumpyArray<T>> out;
  out.reserve(arrays.size());
  for (const NdArray<T>& a : arrays) {
    out.push_back(copy_out(a));
  }
  return out;
}

/// The suffix naming a dtype in the binding's function names (`f32`, `f64`).
template <class T>
std::string tag() {
  return sizeof(T) == 4 ? "f32" : "f64";
}

using Derivs = std::vector<std::optional<std::pair<long long, long long>>>;

std::vector<std::optional<bspline::BoundaryDerivatives>> to_derivs(const Derivs& d) {
  std::vector<std::optional<bspline::BoundaryDerivatives>> out(d.size());
  for (std::size_t i = 0; i < d.size(); ++i) {
    if (d[i]) {
      out[i] = bspline::BoundaryDerivatives{d[i]->first, d[i]->second};
    }
  }
  return out;
}

template <class P>
void def_tensor_collocation(nb::module_& m, const char* name) {
  using TC = bspline::TensorCollocation<P>;
  nb::class_<TC> cls(
      m, name, "The staged tensor-product collocation system (matrices built from the nodes).");
  cls.def_prop_ro(
         "matrices", [](const TC& t) { return list_out(t.matrices()); }, nb::rv_policy::move,
         "The per-direction collocation matrices, with their derivative rows (copies).")
      .def_prop_ro(
          "grid_shape", [](const TC& t) { return t.grid_shape(); }, "The node grid's extents.");
  const auto solve = [](const TC& t, const Any<double>& values, std::size_t rank, double tol) {
    return to_numpy(t.template solve<double>(flat_of(values), rank, tol));
  };
  const auto solve32 = [](const TC& t, const Any<float>& values, std::size_t rank, double tol) {
    return to_numpy(t.template solve<float>(flat_of(values), rank, tol));
  };
  cls.def("solve_f64", solve, nb::arg("values").noconvert(), nb::arg("rank"), nb::arg("tol"),
          "The coefficients `(n_0, ..., n_{d-1}, rank)` of float64 values on the node grid.")
      .def("solve_f32", solve32, nb::arg("values").noconvert(), nb::arg("rank"), nb::arg("tol"),
           "The coefficients of float32 values on the node grid.");
}

template <class T, class P>
void def_tensor_collocation_factory(nb::module_& m) {
  m.def(("tensor_collocation_" + tag<P>()).c_str(),
        [](const bspline::BsplineSpace<T>& space, const std::vector<Vec<P>>& nodes,
           const Derivs& derivatives) {
          std::vector<std::span<const P>> spans;
          spans.reserve(nodes.size());
          for (const Vec<P>& v : nodes) {
            spans.emplace_back(v.data(), v.shape(0));
          }
          try {
            return bspline::TensorCollocation<P>(space, spans, to_derivs(derivatives));
          } catch (const std::domain_error&) {
            raise_zero_division();
          }
        },
        nb::arg("space"), nb::arg("nodes").noconvert(), nb::arg("derivatives"),
        "Build the tensor-product collocation system of `space` at `nodes`.");
}

template <class T, class P, class C>
void def_fit_scattered(nb::module_& m) {
  m.def(("fit_scattered_" + tag<P>() + "_" + tag<C>()).c_str(),
        [](const bspline::BsplineSpace<T>& space, const Mat<P>& pts, const Mat<C>& values,
           double tol) {
          return to_numpy(bspline::fit_scattered_coefficients<T, P, C>(
              space, mdview<const P, 2>(pts.data(), pts.shape(0), pts.shape(1)),
              mdview<const C, 2>(values.data(), values.shape(0), values.shape(1)), tol));
        },
        nb::arg("space"), nb::arg("pts").noconvert(), nb::arg("values").noconvert(), nb::arg("tol"),
        "The coefficients `(n_0, ..., n_{d-1}, rank)` fitted to values at scattered points.");
}

template <class T>
void def_l2(nb::module_& m, const char* name) {
  using L2 = bspline::L2Projection<T>;
  nb::class_<L2> cls(m, name, "The staged L2 projection (mass matrices, lattice and traces).");
  cls.def(
         "__init__",
         [](L2* self, const bspline::BsplineSpace<T>& space, const std::vector<long long>& n_quad,
            bool lobatto, const std::vector<std::pair<bool, bool>>& boundary_interpolation) {
           new (self) L2(space, n_quad,
                         lobatto ? bspline::L2Quadrature::gauss_lobatto
                                 : bspline::L2Quadrature::gauss_legendre,
                         boundary_interpolation);
         },
         nb::arg("space"), nb::arg("n_quad"), nb::arg("lobatto"), nb::arg("boundary_interpolation"))
      .def_prop_ro(
          "nodes", [](const L2& p) { return list_out(p.nodes()); }, nb::rv_policy::move,
          "Per direction, the global quadrature nodes (copies).")
      .def_prop_ro(
          "weights", [](const L2& p) { return list_out(p.weights()); }, nb::rv_policy::move,
          "Per direction, the quadrature weights (copies).")
      .def_prop_ro(
          "mass", [](const L2& p) { return list_out(p.mass()); }, nb::rv_policy::move,
          "Per direction, the mass matrix with its boundary rows (copies).")
      .def_prop_ro(
          "traces",
          [](const L2& p) {
            std::vector<std::vector<int>> out;
            for (const bspline::BoundaryTrace& t : p.traces()) {
              out.push_back(t.side);
            }
            return out;
          },
          "The boundary traces in application order: per direction -1 (free), 0 (start), 1 (end).")
      .def(
          "trace_nodes",
          [](const L2& p, std::size_t trace) {
            std::vector<NumpyArray<T>> out;
            for (const std::vector<T>& x : p.trace_nodes(trace)) {
              NdArray<T> a({x.size()}, for_overwrite);
              std::copy(x.begin(), x.end(), a.data());
              out.push_back(to_numpy(std::move(a)));
            }
            return out;
          },
          nb::arg("trace"), "Per direction, the nodes a trace samples.");
  const auto project = [](const L2& p, const Any<double>& values,
                          const std::vector<Any<double>>& traces, double tol) {
    std::vector<std::span<const double>> spans;
    for (const Any<double>& t : traces) {
      spans.push_back(flat_of(t));
    }
    return to_numpy(p.template project<double>(flat_of(values), spans, tol));
  };
  const auto project32 = [](const L2& p, const Any<float>& values,
                            const std::vector<Any<float>>& traces, double tol) {
    std::vector<std::span<const float>> spans;
    for (const Any<float>& t : traces) {
      spans.push_back(flat_of(t));
    }
    return to_numpy(p.template project<float>(flat_of(values), spans, tol));
  };
  cls.def("project_f64", project, nb::arg("values").noconvert(), nb::arg("traces").noconvert(),
          nb::arg("tol"), "The coefficients of one component from its lattice and trace values.")
      .def("project_f32", project32, nb::arg("values").noconvert(), nb::arg("traces").noconvert(),
           nb::arg("tol"));
}

template <class T>
void def_qi(nb::module_& m, const char* name) {
  using QI = bspline::QuasiInterpolation<T>;
  nb::class_<QI>(m, name, "The staged Lee-Lyche-Mørken quasi-interpolant (points, then weights).")
      .def(nb::init<const bspline::BsplineSpace<T>&>(), nb::arg("space"))
      .def(
          "points", [](const QI& q) { return to_numpy(q.points()); },
          "Every sample point, `(M, dim)` float64.")
      .def(
          "coefficients",
          [](const QI& q, const Mat<double>& values) {
            return to_numpy(q.coefficients(
                mdview<const double, 2>(values.data(), values.shape(0), values.shape(1))));
          },
          nb::arg("values").noconvert(),
          "The coefficients `(n_0, ..., n_{d-1}, rank)` float64 from `(M, rank)` samples.");
}

}  // namespace

void register_bspline_fit(nb::module_& m) {
  def_tensor_collocation<double>(m, "TensorCollocationF64");
  def_tensor_collocation<float>(m, "TensorCollocationF32");
  def_tensor_collocation_factory<double, double>(m);
  def_tensor_collocation_factory<double, float>(m);
  def_tensor_collocation_factory<float, float>(m);
  def_tensor_collocation_factory<float, double>(m);
  def_fit_scattered<double, double, double>(m);
  def_fit_scattered<double, float, float>(m);
  def_fit_scattered<double, float, double>(m);
  def_fit_scattered<double, double, float>(m);
  def_fit_scattered<float, float, float>(m);
  def_fit_scattered<float, double, double>(m);
  def_fit_scattered<float, float, double>(m);
  def_fit_scattered<float, double, float>(m);
  def_l2<double>(m, "L2ProjectionF64");
  def_l2<float>(m, "L2ProjectionF32");
  def_qi<double>(m, "QuasiInterpolationF64");
  def_qi<float>(m, "QuasiInterpolationF32");
}

}  // namespace pantr::bindings
