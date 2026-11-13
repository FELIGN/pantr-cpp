// _pantr._testing: probes for the tests and the benchmark harness. Not a public API.
#include "arrays.hpp"
#include "interface.hpp"
#include "probes.hpp"

#include "pantr/basis/bernstein.hpp"
#include "pantr/basis/lagrange_variant.hpp"
#include "pantr/basis/tabulate.hpp"
#include "pantr/change_basis.hpp"
#include "pantr/grid/detail/hier_blocks.hpp"
#include "pantr/grid/partition.hpp"
#include "pantr/parallel/shared.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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
  m.def(
      "empty_views",
      [] {
        // Null data, as an empty std::vector may have: the views must still be read-only.
        const auto keep = std::make_shared<const int>(0);
        const std::size_t shape[1] = {0};
        return std::make_pair(view_of<double>(nullptr, shape, keep),
                              view_of<std::int64_t>(nullptr, shape, keep));
      },
      "Return two empty read-only views over null data (float64 and int64), as `view_of` "
      "builds them.");
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

// The pure block and factorization helpers of the grids, for the tests that pin them directly.
void register_grid_testing(nb::module_& m) {
  namespace gd = pantr::grid::detail;
  using Mask = nb::ndarray<bool, nb::c_contig, nb::device::cpu>;
  using ConstMask = nb::ndarray<const bool, nb::c_contig, nb::device::cpu>;
  m.attr("max_named_cells") = gd::max_named_cells;
  m.attr("max_diagnosed_cells") = gd::max_diagnosed_cells;
  m.def("block_size", &gd::block_size, nb::arg("lo"), nb::arg("hi"),
        "The number of cells in the block [lo, hi).");
  m.def("in_block", &gd::in_block, nb::arg("midx"), nb::arg("lo"), nb::arg("hi"),
        "Whether midx lies in [lo, hi).");
  m.def("rect_intersect", &gd::rect_intersect, nb::arg("lo1"), nb::arg("hi1"), nb::arg("lo2"),
        nb::arg("hi2"), "The intersection of two blocks, or None.");
  m.def("peel", &gd::peel, nb::arg("outer_lo"), nb::arg("outer_hi"), nb::arg("inner_lo"),
        nb::arg("inner_hi"), "The slabs of the outer block minus the inner one.");
  m.def("try_merge", &gd::try_merge, nb::arg("lo1"), nb::arg("hi1"), nb::arg("lo2"), nb::arg("hi2"),
        "The merge of two adjacent aligned blocks, or None.");
  m.def("normalize_blocks", &gd::normalize_blocks, nb::arg("blocks"),
        "Greedily merge, then sort, a list of blocks.");
  m.def(
      "mark_region_cells",
      [](Mask mask, const gd::Index& lo, const gd::Index& hi, const gd::Block& block) {
        gd::mark_region_cells(
            std::span<std::uint8_t>(reinterpret_cast<std::uint8_t*>(mask.data()), mask.size()), lo,
            hi, block);
      },
      nb::arg("mask"), nb::arg("lo"), nb::arg("hi"), nb::arg("block"),
      "Set, in place, the cells of a region mask that a block covers.");
  m.def(
      "name_marked_cells",
      [](const ConstMask& mask, const gd::Index& origin) {
        gd::Index shape(mask.ndim());
        for (std::size_t k = 0; k < mask.ndim(); ++k) {
          shape[k] = static_cast<std::int64_t>(mask.shape(k));
        }
        return gd::name_marked_cells(
            std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(mask.data()),
                                          mask.size()),
            shape, origin);
      },
      nb::arg("mask"), nb::arg("origin"), "Spell the marked cells of a region mask.");
  m.def(
      "factor_blocks",
      [](const std::vector<std::int64_t>& cells_per_axis, std::int64_t n_parts) {
        return gd::factor_blocks(cells_per_axis, n_parts);
      },
      nb::arg("cells_per_axis"), nb::arg("n_parts"),
      "The per-axis block counts of the Cartesian partition.");
  m.def("divisors", &gd::divisors, nb::arg("n"), "The positive divisors of n, ascending.");
}

}  // namespace pantr::bindings
