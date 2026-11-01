// _pantr.bspline: point inversion of a B-spline (`locate`) and its context; the internals the
// tests reach go to _pantr._testing.
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/locate.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
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

namespace bs = pantr::bspline;
namespace lk = pantr::bspline::detail::locate_kernels;

using Points = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Ids = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Vec = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// Raise the `ZeroDivisionError` Python's float division raised where a Jacobian's basis
/// derivative recurrence met a zero divisor.
[[noreturn]] void raise_zero_division() {
  PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
  throw nb::python_error();
}

/// A `(rows, cols)` NumPy array copied from a row-major vector.
NumpyArray<double> matrix_to_numpy(const std::vector<double>& v, std::size_t rows,
                                   std::size_t cols) {
  NdArray<double> a({rows, cols}, for_overwrite);
  std::copy(v.begin(), v.end(), a.data());
  return to_numpy(std::move(a));
}

/// Check that `n` rows of `cols` columns arrived (the testing probes' only shape check).
void check_rows(const Points& p, std::size_t n, std::size_t cols, const char* what) {
  if (p.shape(0) != n || p.shape(1) != cols) {
    throw value_error(std::string(what) + " has the wrong shape");
  }
}

}  // namespace

void register_bspline_locate(nb::module_& m) {
  nb::class_<bs::LocateContext>(
      m, "LocateContext",
      "The per-spline state of `locate`: the promoted spline, the cell hierarchy and the lengths "
      "the default thresholds are expressed in (immutable).")
      .def_prop_ro("tier", &bs::LocateContext::tier, "`get_default` of the caller's dtype.")
      .def_prop_ro("physical", &bs::LocateContext::physical, "The image-only length.")
      .def_prop_ro("parametric", &bs::LocateContext::parametric, "The parametric length.")
      .def_prop_ro("diagonal", &bs::LocateContext::diagonal, "The control-point box diagonal.")
      .def_prop_ro("scale", &bs::LocateContext::scale, "`max(physical, parametric)`.")
      .def(
          "default_thresholds",
          [](const bs::LocateContext& c) {
            const bs::LocateThresholds t = c.default_thresholds();
            return std::make_pair(t.stop, t.accept);
          },
          "The default `(stop, accept)` thresholds, or a refusal.");
  m.def(
      "locate_context", [](const bs::Bspline<double>& b) { return bs::LocateContext(b); },
      nb::arg("spline"), "Validate a float64 B-spline for inversion and build its locate context.");
  m.def(
      "locate_context", [](const bs::Bspline<float>& b) { return bs::LocateContext(b); },
      nb::arg("spline"),
      "Validate a float32 B-spline for inversion and build its locate context (promoted).");
  m.def(
      "locate",
      [](const bs::LocateContext& ctx, const Points& points, std::optional<double> tol,
         std::int64_t max_iter) {
        const std::size_t n = points.shape(0);
        if (points.shape(1) != ctx.spline().rank()) {
          throw value_error("locate: points must have " + std::to_string(ctx.spline().rank()) +
                            " columns");
        }
        bs::LocateResult r = [&] {
          const ReleaseIfParallel release(n, bs::locate_work(ctx).ops_per_item);
          return bs::locate(ctx, std::span<const double>(points.data(), points.size()), n, tol,
                            max_iter);
        }();
        if (!r.complete) {
          raise_zero_division();
        }
        return std::make_pair(to_numpy(std::move(r.cell_ids)), to_numpy(std::move(r.ref_coords)));
      },
      nb::arg("ctx"), nb::arg("points"), nb::arg("tol").none(), nb::arg("max_iter"),
      "Invert the mapping at `(n, rank)` points: `(cell_ids, ref_coords)`.");
}

void register_bspline_locate_testing(nb::module_& m) {
  m.def(
      "locate_cell_physical_bounds",
      [](const bs::LocateContext& ctx) {
        const auto [lo, hi] = lk::cell_physical_bounds(ctx.spline());
        const std::size_t rank = ctx.spline().rank();
        return std::make_pair(matrix_to_numpy(lo, lo.size() / rank, rank),
                              matrix_to_numpy(hi, hi.size() / rank, rank));
      },
      nb::arg("ctx"), "The per-cell physical control-point boxes `(lo, hi)`, `(n_cells, rank)`.");
  m.def(
      "locate_parametric_reach",
      [](const Vec& lo, const Vec& hi) {
        if (lo.shape(0) != hi.shape(0)) {
          throw value_error("lo and hi differ in length");
        }
        const std::vector<double> r =
            lk::parametric_reach(std::span<const double>(lo.data(), lo.size()),
                                 std::span<const double>(hi.data(), hi.size()));
        NdArray<double> out({r.size()}, for_overwrite);
        std::copy(r.begin(), r.end(), out.data());
        return to_numpy(std::move(out));
      },
      nb::arg("lo"), nb::arg("hi"), "Per direction, `max(|lo|, |hi|) / (hi - lo)` (0 if empty).");
  m.def("locate_geometric_scale", &lk::geometric_scale, nb::arg("physical"), nb::arg("parametric"),
        "`max(physical, parametric)`.");
  m.def(
      "locate_newton_refine",
      [](const bs::LocateContext& ctx, const Points& targets, const Points& starts, double stop,
         double accept, std::int64_t max_iter) {
        const std::size_t n = targets.shape(0);
        const std::size_t dim = ctx.spline().dim();
        check_rows(targets, n, ctx.spline().rank(), "targets");
        check_rows(starts, n, dim, "starts");
        NdArray<bool> converged({n}, for_overwrite);
        NdArray<double> xi({n, dim}, for_overwrite);
        lk::Workspace ws(ctx.spline(), ctx.domain_lo(), ctx.domain_hi());
        for (std::size_t i = 0; i < n; ++i) {
          double* row = xi.data() + i * dim;
          std::copy(starts.data() + i * dim, starts.data() + (i + 1) * dim, row);
          converged.data()[i] = lk::newton_refine(ws, targets.data() + i * ctx.spline().rank(), row,
                                                  {stop, accept}, max_iter);
          if (ws.zero_divisor) {
            raise_zero_division();
          }
        }
        return std::make_pair(to_numpy(std::move(converged)), to_numpy(std::move(xi)));
      },
      nb::arg("ctx"), nb::arg("targets"), nb::arg("starts"), nb::arg("stop"), nb::arg("accept"),
      nb::arg("max_iter"),
      "Run one Newton solve per target from its start: `(converged, last iterate)`.");
  m.def(
      "locate_nearest_corner_starts",
      [](const bs::LocateContext& ctx, const Ids& cells, const Points& targets) {
        const std::size_t n = cells.shape(0);
        const std::size_t dim = ctx.spline().dim();
        check_rows(targets, n, ctx.spline().rank(), "targets");
        NdArray<double> out({n, dim}, for_overwrite);
        lk::Workspace ws(ctx.spline(), ctx.domain_lo(), ctx.domain_hi());
        std::vector<double> lo(dim);
        std::vector<double> hi(dim);
        std::int64_t n_cells = 1;
        for (const std::size_t k : ctx.rep()->num_intervals) {
          n_cells *= static_cast<std::int64_t>(k);
        }
        for (std::size_t i = 0; i < n; ++i) {
          if (cells.data()[i] < 0 || cells.data()[i] >= n_cells) {
            throw value_error("cell id out of range");
          }
          ctx.cell_box(cells.data()[i], lo.data(), hi.data());
          lk::nearest_corner_start(ws, targets.data() + i * ctx.spline().rank(), lo.data(),
                                   hi.data(), out.data() + i * dim);
        }
        return to_numpy(std::move(out));
      },
      nb::arg("ctx"), nb::arg("cells"), nb::arg("targets"),
      "The corner of each cell whose image is nearest its target, `(n, dim)`.");
}

}  // namespace pantr::bindings
