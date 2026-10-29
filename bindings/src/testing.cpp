// _pantr._testing: probes for the tests and the benchmark harness. Not a public API.
#include "arrays.hpp"
#include "interface.hpp"
#include "probes.hpp"

#include "pantr/basis/bernstein.hpp"
#include "pantr/basis/lagrange_variant.hpp"
#include "pantr/basis/tabulate.hpp"
#include "pantr/change_basis.hpp"
#include "pantr/parallel/shared.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

// The Bernstein row kernels at one point, for the tests that pin which recurrence runs where.
template <class T, void (*Row)(int, T, T*)>
NumpyArray<T> bernstein_row(int n, T u) {
  pantr::basis::detail::check_degree(n);
  NdArray<T> row({static_cast<std::size_t>(n) + 1});
  if (n == 0) {
    row.data()[0] = static_cast<T>(1);
  } else {
    Row(n, u, row.data());
  }
  return to_numpy(std::move(row));
}

template <class T>
NumpyArray<T> bernstein_derivatives_point(int n, T s, int n_deriv) {
  pantr::basis::detail::check_degree(n);
  const auto ord = static_cast<std::size_t>(n) + 1;
  const auto nd1 = static_cast<std::size_t>(n_deriv) + 1;
  NdArray<T> out({nd1, ord});
  std::vector<T> ndu(ord * ord);
  std::vector<T> a(2 * nd1);
  pantr::basis::detail::bernstein_derivatives_point<T>(n, s, n_deriv, out.data(), ndu.data(),
                                                       a.data());
  return to_numpy(std::move(out));
}

template <class T>
NumpyArray<T> bernstein_derivatives(
    int n, const nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>& pts,
    int n_deriv) {
  pantr::basis::detail::check_degree(n);
  if (n_deriv < 0) {
    throw pantr::value_error("n_deriv must be non-negative");
  }
  const std::span<const T> p(pts.data(), pts.shape(0));
  NdArray<T> out(
      {p.size(), static_cast<std::size_t>(n_deriv) + 1, static_cast<std::size_t>(n) + 1});
  pantr::basis::tabulate_bernstein_derivatives_1d<T>(n, p, n_deriv, out.template view<3>());
  return to_numpy(std::move(out));
}

template <class T>
void def_bernstein(nb::module_& m, const char* suffix) {
  using pantr::basis::detail::bernstein_row_mirrored;
  using pantr::basis::detail::bernstein_row_unmirrored;
  m.def((std::string("bernstein_row_mirrored") + suffix).c_str(),
        &bernstein_row<T, &bernstein_row_mirrored<T>>, nb::arg("n"), nb::arg("u"),
        "Bernstein values at one point by the mirrored ratio recurrence.");
  m.def((std::string("bernstein_row_unmirrored") + suffix).c_str(),
        &bernstein_row<T, &bernstein_row_unmirrored<T>>, nb::arg("n"), nb::arg("u"),
        "Bernstein values at one point by the plain forward recurrence (low degrees only).");
  m.def((std::string("bernstein_derivatives_point") + suffix).c_str(),
        &bernstein_derivatives_point<T>, nb::arg("n"), nb::arg("s"), nb::arg("n_deriv"),
        "Bernstein derivatives of orders 0..n_deriv at one point, (n_deriv + 1, n + 1).");
  m.def(
      (std::string("tabulate_bernstein_derivatives_1d") + suffix).c_str(),
      &bernstein_derivatives<T>, nb::arg("n"), nb::arg("pts"), nb::arg("n_deriv"),
      "Bernstein derivatives of orders 0..n_deriv at each point, (len(pts), n_deriv + 1, n + 1).");
}

// The generic Gram projection behind the change-of-basis builders, over any two families.
template <class T>
NumpyArray<T> gram_projection(int new_family, int old_family, int degree, long long n_quad) {
  const auto fnew = pantr::basis::to_family(new_family);
  const auto fold = pantr::basis::to_family(old_family);
  const auto eval = [](pantr::basis::Family f) {
    return [f](int d, std::span<const T> p) {
      pantr::basis::detail::check_degree(d);
      NdArray<T> out({p.size(), static_cast<std::size_t>(d) + 1});
      pantr::basis::tabulate_1d<T>(f, d, pantr::basis::LagrangeVariant::equispaces, p,
                                   out.template view<2>());
      return out;
    };
  };
  return to_numpy(
      pantr::change_basis::detail::gram_projection<T>(degree, n_quad, eval(fnew), eval(fold)));
}

}  // namespace

void register_testing(nb::module_& m) {
  m.def("count_threads_used", &count_threads_used, nb::arg("chunks"), nb::arg("hold_us"),
        nb::call_guard<nb::gil_scoped_release>(),
        "Count the distinct threads that run a parallel loop of `chunks` chunks holding "
        "`hold_us` microseconds each.");
  m.def("routes_shared_parallelism", &pantr::parallel::routes_shared_parallelism,
        "Report whether this module routes its parallel state through another module.");
  m.attr("bernstein_max_unmirrored_degree_double") =
      pantr::basis::bernstein_max_unmirrored_degree_double;
  m.attr("bernstein_max_unmirrored_degree_float") =
      pantr::basis::bernstein_max_unmirrored_degree_float;
  def_bernstein<double>(m, "");
  def_bernstein<float>(m, "_f32");
  m.def("gram_projection", &gram_projection<double>, nb::arg("new_family"), nb::arg("old_family"),
        nb::arg("degree"), nb::arg("n_quad_pts"),
        "The matrix M with old = M new, by an n_quad_pts-point Gauss-Legendre Gram projection.");
  m.def("gram_projection_f32", &gram_projection<float>, nb::arg("new_family"),
        nb::arg("old_family"), nb::arg("degree"), nb::arg("n_quad_pts"),
        "The float instantiation of gram_projection.");
  namespace cb = pantr::change_basis;
  m.attr("change_basis_cache_max_degree") = cb::change_basis_cache_max_degree;
  m.def(
      "change_basis_cache_size", [] { return cb::cache_size<double>() + cb::cache_size<float>(); },
      "Number of matrices the change-of-basis process tables hold, both dtypes.");
  m.def(
      "change_basis_cache_clear",
      [] {
        cb::cache_clear<double>();
        cb::cache_clear<float>();
      },
      "Empty the change-of-basis process tables (matrices handed out stay valid).");
}

}  // namespace pantr::bindings
