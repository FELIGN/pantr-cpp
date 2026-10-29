// _pantr.change_basis: the change-of-basis builders and their process tables, for
// pantr.change_basis's wrappers.
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/basis/lagrange_variant.hpp"
#include "pantr/change_basis.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

/// The builders, by the wire value the wrapper passes.
enum class Builder : int {
  lagrange_to_bernstein = 0,
  bernstein_to_lagrange = 1,
  bernstein_to_cardinal = 2,
  cardinal_to_bernstein = 3,
  legendre_to_cardinal = 4,
  cardinal_to_legendre = 5,
  cardinal_dual_legendre_coeffs = 6,
  monomial_to_bernstein = 7,
};

template <class T>
NdArray<T> build(int builder, int degree, basis::LagrangeVariant v) {
  namespace cb = change_basis;
  switch (static_cast<Builder>(builder)) {
    case Builder::lagrange_to_bernstein:
      return cb::lagrange_to_bernstein<T>(degree, v);
    case Builder::bernstein_to_lagrange:
      return cb::bernstein_to_lagrange<T>(degree, v);
    case Builder::bernstein_to_cardinal:
      return cb::bernstein_to_cardinal<T>(degree);
    case Builder::cardinal_to_bernstein:
      return cb::cardinal_to_bernstein<T>(degree);
    case Builder::legendre_to_cardinal:
      return cb::legendre_to_cardinal<T>(degree);
    case Builder::cardinal_to_legendre:
      return cb::cardinal_to_legendre<T>(degree);
    case Builder::cardinal_dual_legendre_coeffs:
      return cb::cardinal_dual_legendre_coeffs<T>(degree);
    case Builder::monomial_to_bernstein:
      return cb::monomial_to_bernstein<T>(degree);
  }
  throw value_error("no change-of-basis builder has the value " + std::to_string(builder));
}

template <class T>
void def_dtype(nb::module_& m, const std::string& suffix) {
  m.def(("build" + suffix).c_str(),
        [](int builder, int degree, int variant) {
          return to_numpy(build<T>(builder, degree, basis::to_lagrange_variant(variant)));
        },
        nb::arg("builder"), nb::arg("degree"), nb::arg("variant"),
        "Build a change-of-basis matrix (a fresh array).");
  m.def(("cached" + suffix).c_str(),
        [](int kind, int degree, int variant) {
          if (kind < 0 || kind > 5) {
            throw value_error("no change-of-basis table has the value " + std::to_string(kind));
          }
          std::shared_ptr<const NdArray<T>> mat = change_basis::cached<T>(
              static_cast<change_basis::Kind>(kind), degree, basis::to_lagrange_variant(variant));
          const T* data = mat->data();
          const std::size_t shape[2] = {mat->shape()[0], mat->shape()[1]};
          return view_of<T>(data, shape, std::move(mat));
        },
        nb::arg("kind"), nb::arg("degree"), nb::arg("variant"),
        "The process table's matrix for a key, as a read-only view of the shared matrix.");
}

}  // namespace

void register_change_basis(nb::module_& m) {
  def_dtype<double>(m, "");
  def_dtype<float>(m, "_f32");
  // The degree domains, as (float32, float64) pairs.
  namespace cb = pantr::change_basis;
  const auto pair = [](cb::DegreeLimit l) { return std::pair<int, int>(l.float32, l.float64); };
  m.def(
      "cardinal_to_bernstein_max_degree",
      [pair] { return pair(cb::cardinal_to_bernstein_max_degree); },
      "The (float32, float64) degree limit of cardinal_to_bernstein.");
  m.def(
      "bernstein_to_cardinal_max_degree",
      [pair] { return pair(cb::bernstein_to_cardinal_max_degree); },
      "The (float32, float64) degree limit of bernstein_to_cardinal.");
  m.def(
      "cardinal_to_legendre_max_degree",
      [pair] { return pair(cb::cardinal_to_legendre_max_degree); },
      "The (float32, float64) degree limit of cardinal_to_legendre and the dual coefficients.");
  m.def(
      "bernstein_to_lagrange_max_degree",
      [pair](int variant) {
        return pair(cb::bernstein_to_lagrange_max_degree(basis::to_lagrange_variant(variant)));
      },
      nb::arg("variant"), "The (float32, float64) degree limit of bernstein_to_lagrange.");
}

}  // namespace pantr::bindings
