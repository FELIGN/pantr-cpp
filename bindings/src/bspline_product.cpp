// _pantr.bspline: the product of two B-splines and the rational derivative built on it; in
// _testing, the knot classes of a raw knot vector. The product and the rational derivative reach
// Eigen's SVD (the product's change of representation).
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/derivative_rational.hpp"
#include "pantr/bspline/extraction.hpp"
#include "pantr/bspline/knots.hpp"
#include "pantr/bspline/product.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <tuple>
#include <utility>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

template <class T>
using Vec = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
void def_product(nb::class_<bspline::Bspline<T>>& cls) {
  using B = bspline::Bspline<T>;
  cls.def(
         "multiply", [](const B& f, const B& g) { return bspline::multiply<T>(f, g); },
         nb::arg("other"), "The exact pointwise product, on the minimal product knot vectors.")
      .def(
          "derivative_rational",
          [](const B& b, long long direction) {
            return bspline::derivative_rational<T>(b, direction);
          },
          nb::arg("direction"), "The partial derivative of a rational B-spline (quotient rule).");
}

template <class T>
void def_kernels(nb::module_& m, const std::string& suffix) {
  m.def(("unique_knots_and_multiplicity_1d" + suffix).c_str(),
        [](const Vec<T>& knots, long long degree, double tol, bool in_domain) {
          const int p = saturate_degree(degree);
          const std::span<const T> k(knots.data(), knots.shape(0));
          bspline::detail::check_knots_and_degree<T>(k, p);
          bspline::KnotClasses<T> c =
              bspline::unique_knots_and_multiplicity<T>(k, p, tol, in_domain);
          NdArray<T> values({c.knots.size()}, for_overwrite);
          std::copy(c.knots.begin(), c.knots.end(), values.data());
          NdArray<std::int64_t> mults({c.multiplicities.size()}, for_overwrite);
          std::copy(c.multiplicities.begin(), c.multiplicities.end(), mults.data());
          return std::make_tuple(to_numpy(std::move(values)), to_numpy(std::move(mults)));
        },
        nb::arg("knots").noconvert(), nb::arg("degree"), nb::arg("tol"), nb::arg("in_domain"),
        "The knot classes of a raw knot vector at an explicit absolute tolerance, as `(knots, "
        "multiplicities)`.");
}

}  // namespace

void register_bspline_product(nb::module_& m) {
  // The classes exist already (register_bspline_ops runs first); their methods are bound here.
  auto f64 = nb::borrow<nb::class_<bspline::Bspline<double>>>(m.attr("BsplineF64"));
  auto f32 = nb::borrow<nb::class_<bspline::Bspline<float>>>(m.attr("BsplineF32"));
  def_product<double>(f64);
  def_product<float>(f32);
}

void register_bspline_product_testing(nb::module_& m) {
  def_kernels<double>(m, "");
  def_kernels<float>(m, "_f32");
}

}  // namespace pantr::bindings
