// _pantr.bspline: the product of two B-splines, the rational derivative built on it, and the
// roots of a scalar B-spline; in _testing, the root finder's kernel and merge. One translation unit
// for the three because each reaches Eigen's SVD (the product's change of representation).
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/derivative_rational.hpp"
#include "pantr/bspline/extraction.hpp"
#include "pantr/bspline/knots.hpp"
#include "pantr/bspline/product.hpp"
#include "pantr/bspline/roots.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

template <class T>
using Vec = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// A fresh 1-D float64 array holding `values`.
NumpyArray<double> doubles(const std::vector<double>& values) {
  NdArray<double> out({values.size()}, for_overwrite);
  std::copy(values.begin(), values.end(), out.data());
  return to_numpy(std::move(out));
}

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
          nb::arg("direction"), "The partial derivative of a rational B-spline (quotient rule).")
      .def(
          "find_roots",
          [](const B& b, std::optional<double> tol, std::optional<long long> max_insertions) {
            std::optional<int> budget;
            if (max_insertions) {
              budget = static_cast<int>(*max_insertions);
            }
            const bspline::BsplineRootSet r = bspline::find_roots<T>(b, tol, budget);
            return std::make_tuple(doubles(r.roots), static_cast<std::int32_t>(r.status),
                                   r.candidates, r.truncated, r.abandoned, r.budget, r.root_tol);
          },
          nb::arg("tol").none(), nb::arg("max_insertions").none(),
          "The zeros of a scalar univariate B-spline (Morken-Reimers), as `(roots, status, "
          "candidates, truncated, abandoned, budget, root_tol)`.");
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

void register_bspline_product_kernels(nb::module_& m) {
  def_kernels<double>(m, "");
  def_kernels<float>(m, "_f32");
}

void register_bspline_product(nb::module_& m) {
  // The classes exist already (register_bspline_ops runs first); their methods are bound here.
  auto f64 = nb::borrow<nb::class_<bspline::Bspline<double>>>(m.attr("BsplineF64"));
  auto f32 = nb::borrow<nb::class_<bspline::Bspline<float>>>(m.attr("BsplineF32"));
  def_product<double>(f64);
  def_product<float>(f32);
  m.attr("ROOT_STATUS_COMPLETE") = static_cast<std::int32_t>(bspline::RootStatus::complete);
  m.attr("ROOT_STATUS_BUDGET_EXHAUSTED") =
      static_cast<std::int32_t>(bspline::RootStatus::budget_exhausted);
  m.attr("ROOT_STATUS_STALLED_AT_LAST_ITERATE") =
      static_cast<std::int32_t>(bspline::RootStatus::stalled_at_last_iterate);
}

void register_bspline_product_testing(nb::module_& m) {
  m.def(
      "bspline_max_insertions", [](int degree) { return bspline::max_insertions(degree); },
      nb::arg("degree"), "The Morken-Reimers insertion budget per zero at a degree.");
  m.def(
      "bspline_morken_reimers_roots",
      [](const Vec<double>& knots, int degree, const Vec<double>& coeffs, double tol,
         double zero_tol, double root_tol, int max_insertions) {
        const auto r = bspline::detail::morken_reimers_roots(
            std::span<const double>(knots.data(), knots.shape(0)), degree,
            std::span<const double>(coeffs.data(), coeffs.shape(0)), tol, zero_tol, root_tol,
            max_insertions);
        return std::make_tuple(doubles(r.roots), r.count, r.truncated, r.abandoned);
      },
      nb::arg("knots"), nb::arg("degree"), nb::arg("coeffs"), nb::arg("tol"), nb::arg("zero_tol"),
      nb::arg("root_tol"), nb::arg("max_insertions"),
      "The Morken-Reimers kernel on an open knot vector, as `(roots, count, truncated, "
      "abandoned)` (no validation).");
  m.def(
      "bspline_merge_roots",
      [](const Vec<double>& roots, const Vec<double>& radii) {
        const auto r =
            bspline::detail::merge_roots(std::span<const double>(roots.data(), roots.shape(0)),
                                         std::span<const double>(radii.data(), radii.shape(0)));
        NdArray<std::int64_t> labels({r.labels.size()}, for_overwrite);
        std::copy(r.labels.begin(), r.labels.end(), labels.data());
        return std::make_tuple(doubles(r.merged), to_numpy(std::move(labels)));
      },
      nb::arg("roots"), nb::arg("radii"),
      "Merge runs of ascending roots within their radii, as `(merged, labels)` (no validation).");
}

}  // namespace pantr::bindings
