// _pantr.basis: the 1D and tensor-product basis tabulations, for pantr.basis's wrappers; and
// _pantr._kernels: the interim surface unported Python callers reach the Bernstein kernels through
// (deleted with the last of them, in the Numba retirement stage).
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
using AnyStride1D = nb::ndarray<const T, nb::ndim<1>, nb::device::cpu>;
template <class T>
using AnyStride2D = nb::ndarray<T, nb::ndim<2>, nb::device::cpu>;
template <class T>
using AnyStride3D = nb::ndarray<T, nb::ndim<3>, nb::device::cpu>;

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

// --- _kernels: any-stride arrays, as the Numba kernels they replace accepted
// ----------------------

template <class T>
std::vector<T> gather(const AnyStride1D<T>& a) {
  std::vector<T> v(a.shape(0));
  const auto view = a.view();
  for (std::size_t i = 0; i < v.size(); ++i) {
    v[i] = view(i);
  }
  return v;
}

template <class T>
void kernel_bernstein_1d(long long degree_in, const AnyStride1D<T>& pts, AnyStride2D<T>& out) {
  const int degree = saturate_degree(degree_in);
  basis::detail::check_degree(degree);  // before the work estimate, which assumes it
  const std::vector<T> p = gather(pts);
  const std::size_t rows = out.shape(0);
  const std::size_t cols = out.shape(1);
  const bool contiguous = out.stride(1) == 1 && out.stride(0) == static_cast<std::int64_t>(cols);
  std::vector<T> tmp;
  T* dst = out.data();
  if (!contiguous) {
    tmp.resize(rows * cols);
    dst = tmp.data();
  }
  {
    const ReleaseIfParallel gil(p.size(), basis::detail::bernstein_ops_per_point(degree));
    basis::tabulate_bernstein_1d<T>(degree, p, mdview<T, 2>(dst, rows, cols));
  }
  if (!contiguous) {
    auto v = out.view();
    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
        v(i, j) = tmp[i * cols + j];
      }
    }
  }
}

template <class T>
void kernel_bernstein_derivatives_1d(long long degree_in, const AnyStride1D<T>& pts, int n_deriv,
                                     AnyStride3D<T>& out) {
  const int degree = saturate_degree(degree_in);
  basis::detail::check_degree(degree);  // before the work estimate, which assumes them
  if (n_deriv < 0) {
    throw value_error("n_deriv must be non-negative");
  }
  const std::vector<T> p = gather(pts);
  const std::size_t s0 = out.shape(0);
  const std::size_t s1 = out.shape(1);
  const std::size_t s2 = out.shape(2);
  const bool contiguous = out.stride(2) == 1 && out.stride(1) == static_cast<std::int64_t>(s2) &&
                          out.stride(0) == static_cast<std::int64_t>(s1 * s2);
  std::vector<T> tmp;
  T* dst = out.data();
  if (!contiguous) {
    tmp.resize(s0 * s1 * s2);
    dst = tmp.data();
  }
  {
    const ReleaseIfParallel gil(
        p.size(), basis::detail::bernstein_derivatives_ops_per_point(degree, n_deriv));
    basis::tabulate_bernstein_derivatives_1d<T>(degree, p, n_deriv, mdview<T, 3>(dst, s0, s1, s2));
  }
  if (!contiguous) {
    auto v = out.view();
    for (std::size_t i = 0; i < s0; ++i) {
      for (std::size_t k = 0; k < s1; ++k) {
        for (std::size_t j = 0; j < s2; ++j) {
          v(i, k, j) = tmp[(i * s1 + k) * s2 + j];
        }
      }
    }
  }
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

void register_kernels(nb::module_& m) {
  const char* b_doc =
      "Write the Bernstein basis of `degree` at `pts` into `out` (any strides, same dtype).";
  m.def("tabulate_bernstein_1d", &kernel_bernstein_1d<double>, nb::arg("degree"), nb::arg("pts"),
        nb::arg("out").noconvert(), b_doc);
  m.def("tabulate_bernstein_1d", &kernel_bernstein_1d<float>, nb::arg("degree"), nb::arg("pts"),
        nb::arg("out").noconvert(), b_doc);
  const char* d_doc =
      "Write the Bernstein derivatives of orders 0..n_deriv at `pts` into `out` "
      "(n, n_deriv + 1, degree + 1), any strides, same dtype.";
  m.def("tabulate_bernstein_derivatives_1d", &kernel_bernstein_derivatives_1d<double>,
        nb::arg("degree"), nb::arg("pts"), nb::arg("n_deriv"), nb::arg("out").noconvert(), d_doc);
  m.def("tabulate_bernstein_derivatives_1d", &kernel_bernstein_derivatives_1d<float>,
        nb::arg("degree"), nb::arg("pts"), nb::arg("n_deriv"), nb::arg("out").noconvert(), d_doc);
}

}  // namespace pantr::bindings
