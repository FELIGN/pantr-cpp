// _pantr.basis: the 1D and tensor-product basis tabulations, for pantr.basis's wrappers.
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/basis/bernstein.hpp"
#include "pantr/basis/lagrange_variant.hpp"
#include "pantr/basis/tabulate.hpp"
#include "pantr/quad/points_lattice.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

template <class T>
using In1D = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
template <class T>
using In2D = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
template <class T>
using Out2D = nb::ndarray<nb::numpy, T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <class T>
nb::object tabulate_1d(int family, long long degree_in, int variant, const In1D<T>& pts,
                       std::optional<Out2D<T>> out) {
  const int degree = saturate_degree(degree_in);
  basis::detail::check_degree(degree);  // before the work estimate, which assumes it
  const basis::Family fam = basis::to_family(family);
  const basis::LagrangeVariant var = basis::to_lagrange_variant(variant);
  const std::span<const T> p(pts.data(), pts.shape(0));
  if (out) {
    const mdview<T, 2> o(out->data(), out->shape(0), out->shape(1));
    {
      const ReleaseIfParallel gil(p.size(), basis::ops_per_point(fam, degree));
      basis::tabulate_1d<T>(fam, degree, var, p, o);
    }
    return nb::none();
  }
  basis::detail::check_degree(degree);
  NdArray<T> result({p.size(), static_cast<std::size_t>(degree) + 1}, for_overwrite);
  {
    const ReleaseIfParallel gil(p.size(), basis::ops_per_point(fam, degree));
    basis::tabulate_1d<T>(fam, degree, var, p, result.template view<2>());
  }
  return nb::cast(to_numpy(std::move(result)));
}

std::size_t total_functions(const std::vector<int>& degrees) {
  std::vector<std::size_t> counts;
  counts.reserve(degrees.size());
  for (const int d : degrees) {
    basis::detail::check_degree(d);
    counts.push_back(static_cast<std::size_t>(d) + 1);
  }
  return checked_product(counts);
}

template <class T>
nb::object tabulate_lattice(int family, const std::vector<long long>& degrees_in, int variant,
                            const quad::PointsLattice<T>& lattice, int order,
                            std::optional<Out2D<T>> out) {
  const std::vector<int> degrees = saturate_degrees(degrees_in);
  const basis::Family fam = basis::to_family(family);
  const basis::LagrangeVariant var = basis::to_lagrange_variant(variant);
  const Order ord = order == 0 ? Order::C : Order::F;
  if (degrees.size() != lattice.dim()) {
    throw value_error("The number of evaluators must be equal to the dimension of the points.");
  }
  const std::size_t n_fun = total_functions(degrees);
  const std::size_t n_pts = lattice.num_points();
  std::optional<NdArray<T>> result;
  mdview<T, 2> o;
  if (out) {
    o = mdview<T, 2>(out->data(), out->shape(0), out->shape(1));
  } else {
    result.emplace(std::vector<std::size_t>{n_pts, n_fun}, for_overwrite);
    o = result->template view<2>();
  }
  {
    const ReleaseIfParallel gil(n_pts, 2.0 * static_cast<double>(n_fun));
    basis::tabulate_tensor<T>(fam, degrees, var, lattice, ord, o);
  }
  return out ? nb::none() : nb::cast(to_numpy(std::move(*result)));
}

template <class T>
nb::object tabulate_points(int family, const std::vector<long long>& degrees_in, int variant,
                           const In2D<T>& pts, int order, std::optional<Out2D<T>> out) {
  const std::vector<int> degrees = saturate_degrees(degrees_in);
  const basis::Family fam = basis::to_family(family);
  const basis::LagrangeVariant var = basis::to_lagrange_variant(variant);
  const Order ord = order == 0 ? Order::C : Order::F;
  const mdview<const T, 2> p(pts.data(), pts.shape(0), pts.shape(1));
  if (p.extent(1) < 1) {
    throw value_error("The dimension of the points must be at least 1.");
  }
  if (degrees.size() != p.extent(1)) {
    throw value_error("The number of evaluators must be equal to the dimension of the points.");
  }
  const std::size_t n_fun = total_functions(degrees);
  std::optional<NdArray<T>> result;
  mdview<T, 2> o;
  if (out) {
    o = mdview<T, 2>(out->data(), out->shape(0), out->shape(1));
  } else {
    result.emplace(std::vector<std::size_t>{p.extent(0), n_fun}, for_overwrite);
    o = result->template view<2>();
  }
  {
    const ReleaseIfParallel gil(p.extent(0), 2.0 * static_cast<double>(n_fun));
    basis::tabulate_tensor<T>(fam, degrees, var, p, ord, o);
  }
  return out ? nb::none() : nb::cast(to_numpy(std::move(*result)));
}

}  // namespace

void register_basis(nb::module_& m) {
  const char* tab_doc =
      "Tabulate the 1D basis `family` (0 Bernstein, 1 cardinal B-spline, 2 Legendre, 3 Lagrange "
      "on node family `variant`) of `degree` at `pts`, into `out` when given (then returning "
      "None).";
  m.attr("max_degree") = (std::numeric_limits<int>::max)();
  m.def("tabulate_1d", &tabulate_1d<double>, nb::arg("family"), nb::arg("degree"),
        nb::arg("variant"), nb::arg("pts"), nb::arg("out").none().noconvert(), tab_doc);
  m.def("tabulate_1d", &tabulate_1d<float>, nb::arg("family"), nb::arg("degree"),
        nb::arg("variant"), nb::arg("pts"), nb::arg("out").none().noconvert(), tab_doc);
  const char* lat_doc =
      "Tabulate the tensor-product basis of `family` with `degrees` on a lattice, points and "
      "functions in `order` (0 C, 1 F), into `out` when given (then returning None).";
  m.def("tabulate_lattice", &tabulate_lattice<double>, nb::arg("family"), nb::arg("degrees"),
        nb::arg("variant"), nb::arg("lattice"), nb::arg("order"), nb::arg("out").none().noconvert(),
        lat_doc);
  m.def("tabulate_lattice", &tabulate_lattice<float>, nb::arg("family"), nb::arg("degrees"),
        nb::arg("variant"), nb::arg("lattice"), nb::arg("order"), nb::arg("out").none().noconvert(),
        lat_doc);
  const char* pts_doc =
      "Tabulate the tensor-product basis of `family` with `degrees` at scattered points "
      "`(n, dim)`, functions in `order` (0 C, 1 F), into `out` when given (then returning None).";
  m.def("tabulate_points", &tabulate_points<double>, nb::arg("family"), nb::arg("degrees"),
        nb::arg("variant"), nb::arg("pts"), nb::arg("order"), nb::arg("out").none().noconvert(),
        pts_doc);
  m.def("tabulate_points", &tabulate_points<float>, nb::arg("family"), nb::arg("degrees"),
        nb::arg("variant"), nb::arg("pts"), nb::arg("order"), nb::arg("out").none().noconvert(),
        pts_doc);
}

}  // namespace pantr::bindings
