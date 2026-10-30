// _pantr.bspline: the Bspline value type and its evaluation, for pantr.bspline's wrapper. Arguments
// arrive normalised (C-contiguous, the exact dtype, an `out` of the exact extents); the wrapper
// keeps the checks whose messages spell a Python value or a NumPy dtype, and the shaping of what it
// returns.
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/evaluate.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/quad/points_lattice.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

template <class T>
using Any = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;

template <class T>
using Points = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <class T>
using Out2D = nb::ndarray<nb::numpy, T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <class T>
mdview<const T, 2> points_view(const Points<T>& p) {
  return mdview<const T, 2>(p.data(), p.shape(0), p.shape(1));
}

template <class T>
mdview<T, 2> out_view(Out2D<T>& o) {
  return mdview<T, 2>(o.data(), o.shape(0), o.shape(1));
}

/// Narrow derivative orders to the C++ API's `int`: one past `int` is a `capacity_error`, a
/// negative one becomes `INT_MIN`, which the API refuses as negative with its own message.
std::vector<int> narrow_orders(const std::vector<long long>& orders) {
  constexpr long long lo = (std::numeric_limits<int>::min)();
  constexpr long long hi = (std::numeric_limits<int>::max)();
  std::vector<int> out(orders.size());
  for (std::size_t i = 0; i < orders.size(); ++i) {
    if (orders[i] > hi) {
      throw capacity_error("derivative order " + std::to_string(orders[i]) + " is too large");
    }
    out[i] = static_cast<int>(orders[i] < lo ? lo : orders[i]);
  }
  return out;
}

/// A zero divisor in the basis derivative recurrence: the error Python's float division raises, as
/// `BsplineSpace1D.tabulate_basis_derivatives` raises it.
[[noreturn]] void raise_zero_division() {
  PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
  throw nb::python_error();
}

template <class T>
void def_bspline(nb::module_& m, const char* name) {
  using B = bspline::Bspline<T>;
  using Space = bspline::BsplineSpace<T>;

  // One lattice overload per coordinate type: the wrapper requires the spline's dtype for a
  // lattice of two or more directions, and not for a 1D one (as before).
  const auto def_lattice_overloads = [](nb::class_<B>& cls, auto lattice_tag) {
    using L = typename decltype(lattice_tag)::type;
    using Lattice = quad::PointsLattice<L>;
    cls.def(
        "evaluate_lattice",
        [](const B& b, const Lattice& lattice, Out2D<T>& out) {
          const ReleaseIfParallel gil(
              lattice.num_points(),
              bspline::detail::tabulation_ops(b) +
                  bspline::detail::contraction_ops(bspline::detail::make_layout(b)));
          bspline::evaluate<T, L>(b, lattice, out_view(out));
        },
        nb::arg("lattice"), nb::arg("out").noconvert(),
        "Write the values on a lattice (C order) into `out` (num_points, cp_size); the weight "
        "column of a rational spline stays.");
    cls.def(
        "evaluate_derivatives_lattice",
        [](const B& b, const Lattice& lattice, const std::vector<long long>& orders,
           Out2D<T>& out) {
          const std::vector<int> o = narrow_orders(orders);
          bool complete = false;
          {
            const ReleaseIfParallel gil(lattice.num_points(),
                                        bspline::detail::derivative_ops_per_point(b, o) +
                                            bspline::detail::tabulation_ops(b));
            complete = bspline::evaluate_derivatives<T, L>(b, lattice, o, out_view(out));
          }
          if (!complete) {
            raise_zero_division();
          }
        },
        nb::arg("lattice"), nb::arg("orders"), nb::arg("out").noconvert(),
        "Write a mixed partial derivative on a lattice into `out` (num_points, rank).");
  };

  nb::class_<B> cls(m, name,
                    "An immutable B-spline map (a space and control points in a shared, read-only "
                    "buffer).");
  cls.def(
         "__init__",
         [](B* self, const Space& space, const Any<T>& cp, bool rational) {
           new (self) B(space, std::span<const T>(cp.data(), cp.size()), rational);
         },
         nb::arg("space"), nb::arg("control_points"), nb::arg("rational"),
         "Validate and copy the control points (any shape: only their number is read).")
      .def_static(
          "cp_size_for",
          [](const Space& space, std::size_t count) { return B::cp_size_for(space, count); },
          nb::arg("space"), nb::arg("count"),
          "The coordinates per control point that `count` values make in `space`, or the "
          "ValueError of a count that is not a multiple of the number of basis functions.")
      .def_prop_ro(
          "shape",
          [](const B& b) {
            const std::span<const std::size_t> s = b.shape();
            return std::vector<std::size_t>(s.begin(), s.end());
          },
          "The control-point extents (*num_basis, cp_size).")
      .def_prop_ro("cp_size", &B::cp_size, "The coordinates per control point.")
      .def_prop_ro("rank", &B::rank, "The output rank, the weight excluded.")
      .def_prop_ro("is_rational", &B::rational, "Whether the last coordinate is a weight.")
      .def_prop_ro(
          "space", [](const B& b) { return b.space(); }, "The space.")
      .def_prop_ro(
          "control_points",
          [](const B& b) { return view_of<T>(b.rep()->data.get(), b.shape(), b.rep()); },
          nb::rv_policy::automatic, "Read-only view of the control points.")
      .def("same_rep", &B::same_rep, nb::arg("other"), "Whether both share one representation.")
      .def(
          "evaluate",
          [](const B& b, const Points<T>& pts, Out2D<T>& out) {
            const ReleaseIfParallel gil(pts.shape(0), bspline::detail::tabulation_ops(b) +
                                                          bspline::detail::contraction_ops(
                                                              bspline::detail::make_layout(b)));
            bspline::evaluate<T, T>(b, points_view(pts), out_view(out));
          },
          nb::arg("pts"), nb::arg("out").noconvert(),
          "Write the values at (n, dim) points into `out` (n, cp_size); the weight column of a "
          "rational spline stays.")
      .def(
          "evaluate_derivatives",
          [](const B& b, const Points<T>& pts, const std::vector<long long>& orders,
             Out2D<T>& out) {
            const std::vector<int> o = narrow_orders(orders);
            bool complete = false;
            {
              const ReleaseIfParallel gil(pts.shape(0),
                                          bspline::detail::derivative_ops_per_point(b, o) +
                                              bspline::detail::tabulation_ops(b));
              complete = bspline::evaluate_derivatives<T, T>(b, points_view(pts), o, out_view(out));
            }
            if (!complete) {
              raise_zero_division();
            }
          },
          nb::arg("pts"), nb::arg("orders"), nb::arg("out").noconvert(),
          "Write a mixed partial derivative at (n, dim) points into `out` (n, rank).");
  def_lattice_overloads(cls, std::type_identity<double>{});
  def_lattice_overloads(cls, std::type_identity<float>{});
}

}  // namespace

void register_bspline_ops(nb::module_& m) {
  def_bspline<double>(m, "BsplineF64");
  def_bspline<float>(m, "BsplineF32");
}

}  // namespace pantr::bindings
