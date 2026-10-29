// _pantr.bezier (continued): degree reduction and minimisation, and fitting, for pantr.bezier's
// wrapper; the reduction operator for the B-spline reduction not yet ported (_kernels, DR-2); and
// the internals the tests reach through _pantr._testing. A translation unit of its own because it
// reaches Eigen's SVD (bezier.cpp reaches no Eigen).
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/bezier/bezier.hpp"
#include "pantr/bezier/fit.hpp"
#include "pantr/bezier/reduce.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

using bezier::Bezier;

template <class T>
using Any = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;

template <class T>
using Vec = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
using Mat = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <class T>
std::vector<std::size_t> shape_of(const Any<T>& a) {
  std::vector<std::size_t> s(a.ndim());
  for (std::size_t i = 0; i < a.ndim(); ++i) {
    s[i] = a.shape(i);
  }
  return s;
}

/// A read-only view of a process-table operator, kept alive by the table's shared pointer.
NumpyView<double> operator_view(std::shared_ptr<const NdArray<double>> op) {
  const NdArray<double>* raw = op.get();
  return view_of<double>(raw->data(), raw->shape(), std::move(op));
}

template <class T>
void def_reduction(nb::module_& m) {
  m.def(
      "reduce_degree",
      [](const Bezier<T>& b, const std::vector<long long>& decrements) {
        return bezier::reduce_degree(b, decrements);
      },
      nb::arg("bezier"), nb::arg("decrements"), "Endpoint-interpolating L2 degree reduction.");
  m.def(
      "degree_reduction_error",
      [](const Bezier<T>& b, const std::vector<long long>& decrements) {
        return bezier::degree_reduction_error(b, decrements);
      },
      nb::arg("bezier"), nb::arg("decrements"), "The exact L2 error of reduce_degree.");
  m.def(
      "minimize_degree",
      [](const Bezier<T>& b, std::optional<double> tol) { return bezier::minimize_degree(b, tol); },
      nb::arg("bezier"), nb::arg("tol").none(), "The lowest degree within a relative tolerance.");
}

template <class T>
void def_fit(nb::module_& m, const std::string& sfx) {
  m.def(("fit_tensor" + sfx).c_str(),
        [](const Any<T>& values, const std::vector<Vec<T>>& nodes,
           std::optional<std::vector<long long>> degrees, std::optional<double> tol) {
          std::vector<std::span<const T>> spans;
          spans.reserve(nodes.size());
          for (const Vec<T>& n : nodes) {
            spans.emplace_back(n.data(), n.shape(0));
          }
          const std::vector<std::size_t> shape = shape_of(values);
          return bezier::fit_tensor<T>(std::span<const T>(values.data(), values.size()), shape,
                                       spans, degrees, tol);
        },
        nb::arg("values"), nb::arg("nodes"), nb::arg("degrees").none(), nb::arg("tol").none(),
        "Fit on a tensor grid: values (*n_pts, rank), one node array per direction.");
  m.def(("fit_scattered" + sfx).c_str(),
        [](const Mat<T>& values, const Mat<T>& pts, const std::vector<long long>& degrees,
           std::optional<double> tol) {
          return bezier::fit_scattered<T>(
              mdview<const T, 2>(values.data(), values.shape(0), values.shape(1)),
              mdview<const T, 2>(pts.data(), pts.shape(0), pts.shape(1)), degrees, tol);
        },
        nb::arg("values"), nb::arg("pts"), nb::arg("degrees"), nb::arg("tol").none(),
        "Fit at scattered points: values (n_pts, rank), points (n_pts, dim).");
}

}  // namespace

void register_bezier_approx(nb::module_& m) {
  def_reduction<double>(m);
  def_reduction<float>(m);
  def_fit<double>(m, "");
  def_fit<float>(m, "_f32");
}

void register_bezier_approx_kernels(nb::module_& m) {
  m.def(
      "interpolating_reduction_operator",
      [](long long degree, long long decrement) {
        return operator_view(bezier::cached_reduction_operator(bezier::ReductionKind::interpolating,
                                                               degree, decrement));
      },
      nb::arg("degree"), nb::arg("decrement"),
      "The endpoint-interpolating reduction operator (read-only, from the process table).");
}

void register_bezier_approx_testing(nb::module_& m) {
  m.def(
      "l2_reduction_operator",
      [](long long degree, long long decrement) {
        return operator_view(
            bezier::cached_reduction_operator(bezier::ReductionKind::l2, degree, decrement));
      },
      nb::arg("degree"), nb::arg("decrement"));
  m.def(
      "elevation_matrix",
      [](long long degree, long long increment) {
        return to_numpy(bezier::elevation_matrix(degree, increment));
      },
      nb::arg("degree"), nb::arg("increment"));
  m.attr("reduction_cache_max_degree") = bezier::reduction_cache_max_degree;
  m.def("reduction_cache_size", &bezier::reduction_cache_size);
  m.def("reduction_cache_clear", &bezier::reduction_cache_clear);
  m.def(
      "squared_l2_norm",
      [](const Any<double>& c) {
        std::vector<std::size_t> shape = shape_of(c);
        if (shape.empty()) {
          shape.push_back(1);
        }
        return bezier::detail::squared_l2_norm(std::span<const double>(c.data(), c.size()), shape);
      },
      nb::arg("coeffs"));
  m.def("projected_quadrature_size", &bezier::detail::projected_quadrature_size, nb::arg("degree"));
  m.def(
      "tensor_gauss_weights",
      [](const std::vector<std::size_t>& num_nodes) {
        return to_numpy(bezier::detail::tensor_gauss_weights(num_nodes));
      },
      nb::arg("num_nodes"));
  m.def(
      "sample_projected",
      [](const Any<double>& ctrl,
         const std::vector<std::size_t>& num_nodes) -> std::optional<NumpyArray<double>> {
        const std::vector<std::size_t> shape = shape_of(ctrl);
        if (shape.size() < 2 || num_nodes.size() != shape.size() - 1) {
          throw value_error("sample_projected needs (*orders, rank + 1) and one count per order");
        }
        auto r = bezier::detail::sample_projected<double>(
            std::span<const double>(ctrl.data(), ctrl.size()), shape, num_nodes);
        if (!r) {
          return std::nullopt;
        }
        return to_numpy(std::move(*r));
      },
      nb::arg("ctrl"), nb::arg("num_nodes"));
  m.def(
      "projected_relative_deviation",
      [](std::optional<Any<double>> values, std::optional<Any<double>> trial,
         const Any<double>& w) {
        const auto to_opt = [](const std::optional<Any<double>>& a) {
          std::optional<NdArray<double>> o;
          if (a) {
            o.emplace(shape_of(*a), for_overwrite);
            std::copy(a->data(), a->data() + a->size(), o->data());
          }
          return o;
        };
        const std::size_t need = w.size() * (values ? values->shape(values->ndim() - 1) : 0);
        if ((values && values->size() != need) || (trial && trial->size() != need)) {
          throw value_error("projected_relative_deviation: (*num_nodes, rank) values expected");
        }
        NdArray<double> wa(shape_of(w), for_overwrite);
        std::copy(w.data(), w.data() + w.size(), wa.data());
        return bezier::detail::projected_relative_deviation(to_opt(values), to_opt(trial), wa);
      },
      nb::arg("values").none(), nb::arg("trial_values").none(), nb::arg("quad_weights"));
}

}  // namespace pantr::bindings
