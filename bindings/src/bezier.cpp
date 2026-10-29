// _pantr.bezier: the Bezier value type and its operations, for pantr.bezier's wrapper; and the
// Bezier kernels the not-yet-ported Python callers reach through _pantr._kernels (DR-2). Arguments
// arrive normalised (C-contiguous, the exact dtype); the wrapper keeps the checks whose messages
// spell a Python value or a NumPy dtype.
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/bezier/bezier.hpp"
#include "pantr/bezier/compose.hpp"
#include "pantr/bezier/derivative.hpp"
#include "pantr/bezier/elevate.hpp"
#include "pantr/bezier/evaluate.hpp"
#include "pantr/bezier/product.hpp"
#include "pantr/bezier/reparametrize.hpp"
#include "pantr/bezier/subdivide.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/quad/points_lattice.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

using bezier::Bezier;

template <class T>
using Any = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;

template <class T>
using Points = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <class T>
using Vec = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
std::span<const T> span_of(const Any<T>& a) {
  return {a.data(), a.size()};
}

template <class T>
std::vector<std::size_t> shape_of(const Any<T>& a) {
  std::vector<std::size_t> s(a.ndim());
  for (std::size_t i = 0; i < a.ndim(); ++i) {
    s[i] = a.shape(i);
  }
  return s;
}

template <class T>
mdview<const T, 2> view_of_points(const Points<T>& p) {
  return mdview<const T, 2>(p.data(), p.shape(0), p.shape(1));
}

/// Work per evaluated point, for the GIL decision only: one multiply-add per control-point
/// coordinate. It is the 1D kernel's order, and below the tensor contraction's (which also
/// tabulates the bases), so the GIL may stay held over a short parallel call; that is safe
/// because no pool task touches Python or owns a Bezier (they capture by reference), and only
/// costs other Python threads a wait.
template <class T>
double ops_per_point(const Bezier<T>& b) {
  return static_cast<double>(b.size());
}

template <class T>
void def_bezier(nb::module_& m, const char* name) {
  using B = Bezier<T>;
  using Lattice = quad::PointsLattice<T>;
  nb::class_<B>(m, name, "An immutable Bezier map (control points in a shared, read-only buffer).")
      .def(
          "__init__",
          [](B* self, const Any<T>& cp, bool rational) {
            new (self) B(span_of(cp), shape_of(cp), rational);
          },
          nb::arg("control_points"), nb::arg("rational"), "Validate and copy the control points.")
      .def_static(
          "borrow",
          [](const Any<T>& cp, bool rational) {
            // Interim (until the B-spline is ported): a `copy=False` conversion from a B-spline
            // shares the B-spline's array, as the previous release did. The buffer aliases that
            // array and keeps it alive. The deleter runs where the last copy of the value dies:
            // a Python thread today, since no pool task owns a Bezier; the explicit acquire is
            // for any other thread (nanobind's own release of the array also takes the thread
            // state, so it is redundant on a Python thread, and harmless). A pool task that did
            // own a Bezier while its submitter held the GIL would deadlock here.
            auto* keep = new Any<T>(cp);
            std::shared_ptr<const T[]> data(cp.data(), [keep](const T* /*unused*/) {
              const nb::gil_scoped_acquire gil;
              delete keep;
            });
            return B(std::move(data), shape_of(cp), rational);
          },
          nb::arg("control_points").noconvert(), nb::arg("rational"),
          "A Bezier over the caller's array, without copying it (interim).")
      .def_prop_ro(
          "shape",
          [](const B& b) {
            const std::span<const std::size_t> s = b.shape();
            return std::vector<std::size_t>(s.begin(), s.end());
          },
          "The control-point extents.")
      .def_prop_ro("rational", &B::rational, "Whether the last coordinate is a weight.")
      .def_prop_ro(
          "control_points",
          [](const B& b) { return view_of<T>(b.rep()->data.get(), b.shape(), b.rep()); },
          nb::rv_policy::automatic, "Read-only view of the control points.")
      .def("same_rep", &B::same_rep, nb::arg("other"), "Whether both share one representation.")
      .def(
          "evaluate",
          [](const B& b, const Points<T>& pts) {
            NdArray<T> out = [&] {
              const ReleaseIfParallel release(pts.shape(0), ops_per_point(b));
              return bezier::evaluate(b, view_of_points(pts));
            }();
            return to_numpy(std::move(out));  // with the GIL: it builds a Python object
          },
          nb::arg("pts"), "Values at (n, dim) points: (n, rank).")
      .def(
          "evaluate_lattice",
          [](const B& b, const Lattice& lattice) {
            NdArray<T> out = [&] {
              const ReleaseIfParallel release(lattice.num_points(), ops_per_point(b));
              return bezier::evaluate(b, lattice);
            }();
            return to_numpy(std::move(out));  // with the GIL: it builds a Python object
          },
          nb::arg("lattice"), "Values on a lattice: (n_0, ..., n_{dim-1}, rank).")
      .def(
          "evaluate_derivatives",
          [](const B& b, const Points<T>& pts, const std::vector<long long>& orders) {
            NdArray<T> out = [&] {
              const ReleaseIfParallel release(pts.shape(0), ops_per_point(b));
              return bezier::evaluate_derivatives(b, view_of_points(pts), orders);
            }();
            return to_numpy(std::move(out));  // with the GIL: it builds a Python object
          },
          nb::arg("pts"), nb::arg("orders"), "A mixed partial derivative at (n, dim) points.")
      .def(
          "evaluate_derivatives_lattice",
          [](const B& b, const Lattice& lattice, const std::vector<long long>& orders) {
            NdArray<T> out = [&] {
              const ReleaseIfParallel release(lattice.num_points(), ops_per_point(b));
              return bezier::evaluate_derivatives(b, lattice, orders);
            }();
            return to_numpy(std::move(out));  // with the GIL: it builds a Python object
          },
          nb::arg("lattice"), nb::arg("orders"), "A mixed partial derivative on a lattice.")
      .def(
          "derivative",
          [](const B& b, long long direction, bool keep_degree) {
            return bezier::derivative(b, direction, keep_degree);
          },
          nb::arg("direction"), nb::arg("keep_degree"), "The hodograph in one direction.")
      .def(
          "elevate_degree",
          [](const B& b, const std::vector<long long>& increments) {
            return bezier::elevate_degree(b, increments);
          },
          nb::arg("increments"), "Degree elevation per direction.")
      .def(
          "multiply", [](const B& a, const B& b) { return bezier::multiply(a, b); },
          nb::arg("other"), "The pointwise product.")
      .def(
          "compose", [](const B& outer, const B& inner) { return bezier::compose(outer, inner); },
          nb::arg("inner"), "self(inner(t)).")
      .def(
          "split",
          [](const B& b, long long direction, double value) {
            return bezier::split(b, direction, value);
          },
          nb::arg("direction"), nb::arg("value"), "Split at a parameter value: (left, right).")
      .def(
          "slice",
          [](const B& b, long long axis, double value) { return bezier::slice(b, axis, value); },
          nb::arg("axis"), nb::arg("value"), "Fix one direction (dim >= 2).")
      .def(
          "slice_point",
          [](const B& b, long long axis, double value) {
            return to_numpy(bezier::slice_point(b, axis, value));
          },
          nb::arg("axis"), nb::arg("value"), "The point of a curve at a parameter (projected).")
      .def(
          "restrict",
          [](const B& b, const std::vector<std::optional<std::pair<double, double>>>& bounds) {
            return bezier::restrict(b, bounds);
          },
          nb::arg("bounds"), "The restriction to a sub-box, reparametrised to [0, 1]^dim.")
      .def(
          "collapse_along_axis",
          [](const B& b, long long axis, const Vec<T>& values) {
            return bezier::collapse_along_axis(b, axis,
                                               std::span<const T>(values.data(), values.shape(0)));
          },
          nb::arg("axis"), nb::arg("values"), "Fix every direction but one.")
      .def(
          "reversed",
          [](const B& b, long long direction) { return bezier::reversed(b, direction); },
          nb::arg("direction"), "Reverse one direction.")
      .def(
          "permuted",
          [](const B& b, const std::vector<long long>& permutation) {
            return bezier::permuted(b, permutation);
          },
          nb::arg("permutation"), "Reorder the directions.")
      .def(
          "transformed",
          [](const B& b, const Any<double>& matrix, const Any<double>& offset) {
            const std::vector<std::size_t> ms = shape_of(matrix);
            return bezier::transformed(b, span_of(matrix), ms, span_of(offset));
          },
          nb::arg("matrix"), nb::arg("offset"), "Apply x -> A x + b to the geometry.");
}

template <class T>
void def_bezier_kernels(nb::module_& m, const std::string& suffix) {
  m.def(("bezier_elevate_degree_1d" + suffix).c_str(),
        [](std::size_t degree, const Points<T>& ctrl, std::size_t increment) {
          return to_numpy(bezier::elevate_degree_1d(degree, view_of_points(ctrl), increment));
        },
        nb::arg("degree"), nb::arg("ctrl").noconvert(), nb::arg("increment"),
        "Elevate the degree of a (degree + 1, n_cols) Bernstein array.");
  m.def(("bernstein_product" + suffix).c_str(),
        [](const Any<T>& a, const Any<T>& b) {
          const std::vector<std::size_t> as = shape_of(a);
          const std::vector<std::size_t> bs = shape_of(b);
          return to_numpy(bezier::bernstein_product(span_of(a), as, span_of(b), bs));
        },
        nb::arg("a").noconvert(), nb::arg("b").noconvert(),
        "Bernstein coefficients of the pointwise product of two (*degrees_plus_1, rank) arrays "
        "(the 1D formula for one direction, the tensor formula otherwise).");
  m.def(("bernstein_product_nd" + suffix).c_str(),
        [](const Any<T>& a, const Any<T>& b) {
          const std::vector<std::size_t> as = shape_of(a);
          const std::vector<std::size_t> bs = shape_of(b);
          return to_numpy(bezier::bernstein_product_nd(span_of(a), as, span_of(b), bs));
        },
        nb::arg("a").noconvert(), nb::arg("b").noconvert(),
        "As bernstein_product, always by the tensor formula.");
}

}  // namespace

void register_bezier(nb::module_& m) {
  def_bezier<double>(m, "BezierF64");
  def_bezier<float>(m, "BezierF32");
}

void register_bezier_kernels(nb::module_& m) {
  // Distinct names per dtype: overloads differing only in an unshaped array's dtype overlap for
  // mypy, and the caller knows the dtype.
  def_bezier_kernels<double>(m, "");
  def_bezier_kernels<float>(m, "_f32");
}

}  // namespace pantr::bindings
