// _pantr.bspline: BsplineSpace1D, BsplineSpace and the knot factories, for pantr.bspline's
// wrappers; and the knot scale, tolerance and snapping check in _pantr._testing.
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/bspline/knot_factories.hpp"
#include "pantr/bspline/knots.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/quad/points_lattice.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

template <class T>
using In1D = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
template <class T>
using In2D = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
template <class T>
using AnyStride1D = nb::ndarray<const T, nb::ndim<1>, nb::device::cpu>;
template <class T, std::size_t Rank>
using OutND = nb::ndarray<nb::numpy, T, nb::ndim<Rank>, nb::c_contig, nb::device::cpu>;
using OutFirst1D = OutND<std::int64_t, 1>;
using OutFirst2D = OutND<std::int64_t, 2>;
using OutBool1D = OutND<bool, 1>;

template <class T>
std::span<const T> as_span(const In1D<T>& a) {
  return {a.data(), a.shape(0)};
}

template <class T>
std::vector<T> gather(const AnyStride1D<T>& a) {
  std::vector<T> v(a.shape(0));
  const auto view = a.view();
  for (std::size_t i = 0; i < v.size(); ++i) {
    v[i] = view(i);
  }
  return v;
}

template <class E, class T>
NumpyView<E> view_1d(std::span<const E> s,
                     std::shared_ptr<const typename bspline::BsplineSpace1D<T>::Rep> keep) {
  const std::size_t shape[1] = {s.size()};
  return view_of<E>(s.data(), shape, std::move(keep));
}

/// A NumPy array owning a copy of `v`.
template <class E>
NumpyArray<E> copy_to_numpy(const std::vector<E>& v) {
  NdArray<E> a({v.size()}, for_overwrite);
  std::copy(v.begin(), v.end(), a.data());
  return pantr::bindings::to_numpy(std::move(a));
}

/// Tabulate the values (or, with `n_deriv >= 0`, the derivatives) into `out` and `first`.
template <class T, class P>
void tabulate_1d(const bspline::BsplineSpace1D<T>& s, const In1D<P>& pts, OutND<P, 2>& out,
                 OutFirst1D& first, bool validate) {
  const std::span<const P> p = as_span(pts);
  const mdview<P, 2> o(out.data(), out.shape(0), out.shape(1));
  const ReleaseIfParallel gil(p.size(), bspline::detail::basis_ops_per_point(s.degree()));
  s.template tabulate_basis<P>(p, o, std::span<std::int64_t>(first.data(), first.shape(0)),
                               validate);
}

template <class T, class P>
void tabulate_derivs_1d(const bspline::BsplineSpace1D<T>& s, const In1D<P>& pts, int n_deriv,
                        std::optional<OutND<P, 3>> out, std::optional<OutFirst1D> first,
                        bool validate) {
  if (n_deriv < 0 || !out || !first) {
    // The checks the wrapper relies on to size its outputs: n_deriv first, as before.
    if (n_deriv < 0) {
      throw value_error("n_deriv must be non-negative, got " + std::to_string(n_deriv));
    }
    throw value_error("tabulate_basis_derivatives needs both output arrays");
  }
  const std::span<const P> p = as_span(pts);
  const mdview<P, 3> o(out->data(), out->shape(0), out->shape(1), out->shape(2));
  bool complete = false;
  {
    const ReleaseIfParallel gil(p.size(),
                                bspline::detail::derivs_ops_per_point(s.degree(), n_deriv));
    complete = s.template tabulate_basis_derivatives<P>(
        p, n_deriv, o, std::span<std::int64_t>(first->data(), first->shape(0)), validate);
  }
  if (!complete) {
    // A zero divisor in the derivative recurrence: the error Python's float division raises, as
    // the previous implementation raised it (for fewer than 4096 points).
    PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
    throw nb::python_error();
  }
}

template <class T>
void def_space_1d(nb::module_& m, const char* name) {
  using S = bspline::BsplineSpace1D<T>;
  nb::class_<S> cls(m, name, "An immutable 1D B-spline space (knots and degree).");
  cls.def(
         "__init__",
         [](S* self, const In1D<T>& knots, long long degree, bool periodic, bool snap_knots) {
           if (degree > (std::numeric_limits<int>::max)() &&
               static_cast<double>(knots.shape(0)) < 2.0 * static_cast<double>(degree) + 2.0) {
             // The size check the constructor makes first, for a degree past `int`.
             throw value_error("knots must have at least 2*degree+2 elements");
           }
           new (self) S(as_span(knots), saturate_degree(degree), periodic, snap_knots);
         },
         nb::arg("knots"), nb::arg("degree"), nb::arg("periodic"), nb::arg("snap_knots"),
         "Validate and copy the knots into a new space.")
      .def_static(
          "from_state",
          [](const In1D<T>& knots, long long degree, bool periodic, double tol) {
            return S::from_state(as_span(knots), saturate_degree(degree), periodic, tol);
          },
          nb::arg("knots"), nb::arg("degree"), nb::arg("periodic"), nb::arg("tol"),
          "Rebuild a space from its stored knots and tolerance (no snapping).")
      .def_prop_ro("degree", &S::degree, "The degree.")
      .def_prop_ro("periodic", &S::periodic, "Whether the space is periodic.")
      .def_prop_ro("tolerance", &S::tolerance, "The absolute parametric tolerance.")
      .def_prop_ro("num_basis", &S::num_basis, "The number of basis functions.")
      .def_prop_ro("num_intervals", &S::num_intervals, "The number of in-domain intervals.")
      .def_prop_ro("left_open", &S::has_left_end_open, "Whether the left end is open.")
      .def_prop_ro("right_open", &S::has_right_end_open, "Whether the right end is open.")
      .def_prop_ro("bezier_like", &S::has_bezier_like_knots, "Whether the space is one piece.")
      .def_prop_ro(
          "knots", [](const S& s) { return view_1d<T, T>(s.knots(), s.rep()); },
          nb::rv_policy::automatic, "Read-only view of the knot vector.")
      .def(
          "unique_knots_and_multiplicity",
          [](const S& s, bool in_domain) {
            const auto [ks, ms] = s.unique_knots_and_multiplicity(in_domain);
            return std::make_pair(view_1d<T, T>(ks, s.rep()),
                                  view_1d<std::int64_t, T>(ms, s.rep()));
          },
          nb::arg("in_domain"), "Read-only views of the distinct knots and their multiplicities.")
      .def(
          "first_basis_per_interval",
          [](const S& s) {
            return view_1d<std::int64_t, T>(s.first_basis_per_interval(), s.rep());
          },
          "Read-only view of the first supported function of each interval.")
      .def(
          "cardinal_intervals",
          [](const S& s, OutBool1D& out) {
            s.cardinal_intervals(std::span<bool>(out.data(), out.shape(0)));
          },
          nb::arg("out").noconvert(), "Write the cardinal-interval flags into `out`.")
      .def(
          "restrict",
          [](const S& s, long long lo, long long hi) {
            auto [w, map] = s.restrict(lo, hi);
            return std::make_pair(std::move(w), copy_to_numpy(map));
          },
          nb::arg("lo"), nb::arg("hi"), "The windowed space over intervals [lo, hi) and its map.")
      .def(
          "greville", [](const S& s) { return copy_to_numpy(s.greville()); },
          "The Greville abscissae.")
      .def(
          "all_in_domain",
          [](const S& s, const In1D<double>& pts) {
            return s.template all_in_domain<double>(as_span(pts));
          },
          nb::arg("pts"), "Whether every point lies in the domain.")
      .def(
          "all_in_domain",
          [](const S& s, const In1D<float>& pts) {
            return s.template all_in_domain<float>(as_span(pts));
          },
          nb::arg("pts"), "Whether every point lies in the domain.")
      .def("tabulate_basis", &tabulate_1d<T, double>, nb::arg("pts"), nb::arg("out").noconvert(),
           nb::arg("first").noconvert(), nb::arg("validate"),
           "Write the basis values and first indices at `pts` into `out` and `first`.")
      .def("tabulate_basis", &tabulate_1d<T, float>, nb::arg("pts"), nb::arg("out").noconvert(),
           nb::arg("first").noconvert(), nb::arg("validate"),
           "Write the basis values and first indices at `pts` into `out` and `first`.")
      .def("tabulate_basis_derivatives", &tabulate_derivs_1d<T, double>, nb::arg("pts"),
           nb::arg("n_deriv"), nb::arg("out").none().noconvert(),
           nb::arg("first").none().noconvert(), nb::arg("validate"),
           "Write the basis derivatives of orders 0..n_deriv at `pts` into `out` and `first`.")
      .def("tabulate_basis_derivatives", &tabulate_derivs_1d<T, float>, nb::arg("pts"),
           nb::arg("n_deriv"), nb::arg("out").none().noconvert(),
           nb::arg("first").none().noconvert(), nb::arg("validate"),
           "Write the basis derivatives of orders 0..n_deriv at `pts` into `out` and `first`.");
}

/// Operations one tensor-product point costs: the product (two per function) and, at most, each
/// direction's 1D recurrence (a lattice shares those across points, so this overestimates it).
template <class T>
double tensor_ops_per_point(const bspline::BsplineSpace<T>& s) {
  double ops = 2.0 * static_cast<double>(s.num_local_functions());
  for (const auto& d : s.spaces()) {
    ops += bspline::detail::basis_ops_per_point(d.degree());
  }
  return ops;
}

template <class T, class P>
void tabulate_points(const bspline::BsplineSpace<T>& s, const In2D<P>& pts, OutND<T, 2>& out,
                     OutFirst2D& first) {
  const mdview<const P, 2> p(pts.data(), pts.shape(0), pts.shape(1));
  const ReleaseIfParallel gil(pts.shape(0), tensor_ops_per_point(s));
  s.template tabulate_basis<P>(
      p, mdview<T, 2>(out.data(), out.shape(0), out.shape(1)),
      mdview<std::int64_t, 2>(first.data(), first.shape(0), first.shape(1)));
}

template <class T, class L>
void tabulate_lattice(const bspline::BsplineSpace<T>& s, const quad::PointsLattice<L>& lattice,
                      OutND<L, 2>& out, OutFirst2D& first) {
  const ReleaseIfParallel gil(lattice.num_points(), tensor_ops_per_point(s));
  s.template tabulate_basis<L>(
      lattice, mdview<L, 2>(out.data(), out.shape(0), out.shape(1)),
      mdview<std::int64_t, 2>(first.data(), first.shape(0), first.shape(1)));
}

template <class T, class P>
std::ptrdiff_t first_outside_points(const bspline::BsplineSpace<T>& s, const In2D<P>& pts) {
  return s.template first_direction_outside<P>(
      mdview<const P, 2>(pts.data(), pts.shape(0), pts.shape(1)));
}

template <class T, class L>
std::ptrdiff_t first_outside_lattice(const bspline::BsplineSpace<T>& s,
                                     const quad::PointsLattice<L>& lattice) {
  return s.template first_direction_outside<L>(lattice);
}

template <class T>
void def_space(nb::module_& m, const char* name) {
  using S = bspline::BsplineSpace<T>;
  nb::class_<S>(m, name, "An immutable tensor-product B-spline space.")
      .def(
          "__init__",
          [](S* self, const std::vector<bspline::BsplineSpace1D<T>>& spaces) {
            new (self) S(spaces);
          },
          nb::arg("spaces"), "The tensor product of `spaces`.")
      .def_prop_ro("dim", &S::dim, "The number of directions.")
      .def_prop_ro("num_local_functions", &S::num_local_functions,
                   "The non-zero functions per point.")
      .def("first_direction_outside", &first_outside_points<T, double>, nb::arg("pts"),
           "The first direction with a point outside the domain, or -1.")
      .def("first_direction_outside", &first_outside_points<T, float>, nb::arg("pts"),
           "The first direction with a point outside the domain, or -1.")
      .def("first_direction_outside", &first_outside_lattice<T, double>, nb::arg("lattice"),
           "The first direction with a coordinate outside the domain, or -1.")
      .def("first_direction_outside", &first_outside_lattice<T, float>, nb::arg("lattice"),
           "The first direction with a coordinate outside the domain, or -1.")
      .def("tabulate_points", &tabulate_points<T, double>, nb::arg("pts"),
           nb::arg("out").noconvert(), nb::arg("first").noconvert(),
           "Write the basis at points `(n, dim)` into `out` `(n, nfun)` and `first` `(n, dim)`.")
      .def("tabulate_points", &tabulate_points<T, float>, nb::arg("pts"),
           nb::arg("out").noconvert(), nb::arg("first").noconvert(),
           "Write the basis at points `(n, dim)` into `out` `(n, nfun)` and `first` `(n, dim)`.")
      .def("tabulate_lattice", &tabulate_lattice<T, double>, nb::arg("lattice"),
           nb::arg("out").noconvert(), nb::arg("first").noconvert(),
           "Write the basis on a lattice (C order) into `out` and `first`.")
      .def("tabulate_lattice", &tabulate_lattice<T, float>, nb::arg("lattice"),
           nb::arg("out").noconvert(), nb::arg("first").noconvert(),
           "Write the basis on a lattice (C order) into `out` and `first`.");
}

template <class T>
void def_factories(nb::module_& m, const std::string& suffix) {
  m.def(("uniform_open_knots" + suffix).c_str(),
        [](long long n, long long degree, long long continuity, T start, T end) {
          const std::vector<T> k =
              bspline::uniform_open_knots<T>(n, degree, continuity, start, end);
          return copy_to_numpy(k);
        },
        nb::arg("num_intervals"), nb::arg("degree"), nb::arg("continuity"), nb::arg("start"),
        nb::arg("end"), "A uniform open knot vector.");
  m.def(("uniform_periodic_knots" + suffix).c_str(),
        [](long long n, long long degree, long long continuity, T start, T end) {
          const std::vector<T> k =
              bspline::uniform_periodic_knots<T>(n, degree, continuity, start, end);
          return copy_to_numpy(k);
        },
        nb::arg("num_intervals"), nb::arg("degree"), nb::arg("continuity"), nb::arg("start"),
        nb::arg("end"), "A uniform periodic knot vector.");
  m.def(("cardinal_knots" + suffix).c_str(),
        [](long long n, long long degree) {
          const std::vector<T> k = bspline::cardinal_knots<T>(n, degree);
          return copy_to_numpy(k);
        },
        nb::arg("num_intervals"), nb::arg("degree"), "The cardinal B-spline knot vector.");
}

/// Refuse an empty knot vector, which the knot helpers read the ends of.
template <class T>
std::vector<T> non_empty(std::vector<T> v) {
  if (v.empty()) {
    throw value_error("the knot vector is empty");
  }
  return v;
}

template <class T>
void def_knot_kernels(nb::module_& m) {
  m.def(
      "knot_scale",
      [](const AnyStride1D<T>& k) { return bspline::knot_scale<T>(non_empty(gather(k))); },
      nb::arg("knots"), "max(span, |knots[0]|, |knots[-1]|) of a non-empty knot vector.");
  m.def(
      "knot_tolerance",
      [](const AnyStride1D<T>& k) { return bspline::knot_tolerance<T>(non_empty(gather(k))); },
      nb::arg("knots"), "The absolute parametric tolerance of a non-empty knot vector.");
  m.def(
      "check_snapping_kept_an_interval",
      [](const AnyStride1D<T>& raw, const AnyStride1D<T>& snapped, double tol) {
        bspline::check_snapping_kept_an_interval<T>(non_empty(gather(raw)),
                                                    non_empty(gather(snapped)), tol);
      },
      nb::arg("raw"), nb::arg("snapped"), nb::arg("tol"),
      "Refuse a knot vector that snapping collapsed onto one point.");
}

}  // namespace

void register_bspline(nb::module_& m) {
  def_space_1d<double>(m, "BsplineSpace1DF64");
  def_space_1d<float>(m, "BsplineSpace1DF32");
  def_space<double>(m, "BsplineSpaceF64");
  def_space<float>(m, "BsplineSpaceF32");
  def_factories<double>(m, "");
  def_factories<float>(m, "_f32");
}

void register_bspline_testing(nb::module_& m) {
  def_knot_kernels<double>(m);
  def_knot_kernels<float>(m);
}

}  // namespace pantr::bindings
