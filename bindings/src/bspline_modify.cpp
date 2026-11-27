// _pantr.bspline: the operations that rebuild a Bspline over other knots or another degree: degree
// elevation and reduction, knot removal and the derivative, with the common refinement of two
// B-splines the rational derivative calls; and, in _testing, the tolerance helpers of knot
// removal, the binomial kernel and the 1D degree kernels the tests reach.
//
// This translation unit reaches Eigen through the periodic directions of elevation and reduction
// (the QR of the periodic conversion), which the module's vectorization settings already cover for
// every unit (EIGEN_DONT_VECTORIZE on MSVC's v4 rung). Arguments arrive as the wrapper normalised
// them (C-contiguous arrays of the exact dtype); validation of values is the C++ API's, with the
// messages the Python suite matches.
//
// Nothing here reaches the pool, so a call releases the GIL through `ReleaseIfLong`, on a lower
// bound of its work, as in `bspline_refine.cpp`: a refusal the estimate can read off the arguments
// alone counts no work; every other one (a knot that is not one, an unclamped knot vector) is left
// unmirrored, and such a call may release for nothing, which costs a wait. The testing hooks at
// the end keep the GIL: they are reached from the test-only `_testing` module alone.
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/degree.hpp"
#include "pantr/bspline/derivative.hpp"
#include "pantr/bspline/knot_removal.hpp"
#include "pantr/core/binomial.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
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
std::span<const T> as_span(const In1D<T>& a) {
  return {a.data(), a.shape(0)};
}

template <class T>
mdview<const T, 2> as_view(const In2D<T>& a) {
  return mdview<const T, 2>(a.data(), a.shape(0), a.shape(1));
}

/// The columns of a control net handed to a tolerance helper, refused at zero: the helpers divide
/// the flat length by it, which is a division by zero (a hardware trap, not an exception) for a
/// net of no columns.
template <class T>
std::size_t columns_of(const In2D<T>& ctrl) {
  if (ctrl.shape(1) == 0) {
    throw value_error("ctrl needs at least one column per control point, got shape (" +
                      std::to_string(ctrl.shape(0)) + ", 0).");
  }
  return ctrl.shape(1);
}

/// One direction's knot values, or none to skip the direction.
template <class T>
using MaybeKnots = std::optional<In1D<T>>;

/// Work of `elevate_degree`, for the GIL decision only: each elevated direction's
/// `elevation_ops`. Zero for the increments the call refuses before any work: a count other than
/// the dimension, a negative one, none positive, or an elevated degree above `bincoeff_max_n`.
template <class T>
double elevate_ops(const bspline::Bspline<T>& b, const std::vector<long long>& increments) {
  if (increments.size() != b.dim() ||
      std::any_of(increments.begin(), increments.end(), [](long long t) { return t < 0; })) {
    return 0.0;
  }
  const bspline::BsplineSpace<T> whole = b.space();
  double ops = 0.0;
  for (std::size_t d = 0; d < b.dim(); ++d) {
    const long long t = increments[d];
    if (t == 0) {
      continue;
    }
    if (t > bincoeff_max_n - whole.spaces()[d].degree()) {
      return 0.0;
    }
    ops += elevation_ops(b, d, static_cast<double>(t));
  }
  return ops;
}

/// Work of `reduce_degree`, for the GIL decision only, in the first reduced direction, of degree
/// `p`, decrement `t` and `N` intervals, for each of the `rest = b.size() / shape[d]` values the
/// other extents and the coordinates hold:
///
///  - the `(q + 1, p + 1)` reduction operator, `q = p - t`, applied to each of at least `N` Bezier
///    segments (`detail::degree_reduce_kernel` splits at every distinct knot value),
///    `N (q + 1) (p + 1)` multiply-adds;
///  - the forced removal that restores the continuity (`coarsen_knots_after_reduction`): each
///    interior class whose Bezier multiplicity `q` exceeds its target `max(1, m - t)` is removed
///    by one call of `remove_knot_1d`, which copies the whole net into its working rows, at least
///    the `q + N` rows of the coarsened result.
///
/// A lower bound: it leaves out the later directions, whose nets the reduction can shrink, the
/// Bezier extraction, and the build of an operator the process table does not hold. Zero for the
/// decrements the call refuses before any work: a count other than the dimension, a negative one,
/// none positive, or one above its direction's degree.
template <class T>
double reduce_ops(const bspline::Bspline<T>& b, const std::vector<long long>& decrements) {
  if (decrements.size() != b.dim()) {
    return 0.0;
  }
  const bspline::BsplineSpace<T> whole = b.space();
  double ops = 0.0;
  for (std::size_t d = 0; d < b.dim(); ++d) {
    const bspline::BsplineSpace1D<T>& space = whole.spaces()[d];
    const long long p = space.degree();
    const long long t = decrements[d];
    if (t < 0 || t > p) {
      return 0.0;
    }
    if (t == 0 || ops > 0.0) {
      continue;
    }
    const long long q = p - t;
    const auto n_int = static_cast<double>(space.num_intervals());
    const std::span<const std::int64_t> mults = space.unique_knots_and_multiplicity(true).second;
    double removals = 0.0;
    for (std::size_t i = 1; i + 1 < mults.size(); ++i) {
      removals += q > (std::max)(1LL, static_cast<long long>(mults[i]) - t) ? 1.0 : 0.0;
    }
    const auto rest = static_cast<double>(b.size() / b.shape()[d]);
    ops = rest * (n_int * static_cast<double>((q + 1) * (p + 1)) +
                  removals * (static_cast<double>(q) + n_int));
  }
  return ops;
}

/// Work of `remove_knots`, for the GIL decision only: the first direction with knots, of `n` rows
/// and degree `p`, copies the net, and the removal of its `k`-th value (from 0) copies the net as
/// it stands into the kernel's working rows (`detail::remove_knot_kernel`), at least `n - k p`
/// rows, since each removal takes at most `p`; each row holds `b.size() / n` values. A lower
/// bound: it leaves out the kernel's result and the later directions, whose nets the removal
/// shrinks. Zero for the arguments the call refuses before any
/// work: a periodic direction, a count of directions other than the dimension, every direction
/// skipped, or `num` below 1.
template <class T>
double removal_ops(const bspline::Bspline<T>& b, std::span<const std::span<const T>> knot_values,
                   std::optional<long long> num) {
  const bspline::BsplineSpace<T> whole = b.space();
  const bool periodic = std::any_of(whole.spaces().begin(), whole.spaces().end(),
                                    [](const auto& s) { return s.periodic(); });
  if (periodic || knot_values.size() != b.dim() || (num.has_value() && *num < 1)) {
    return 0.0;
  }
  for (std::size_t d = 0; d < b.dim(); ++d) {
    if (knot_values[d].empty()) {
      continue;
    }
    const auto n = static_cast<double>(b.shape()[d]);
    const auto p = static_cast<double>(whole.spaces()[d].degree());
    double rows = n;
    for (std::size_t k = 0; k < knot_values[d].size() && n > static_cast<double>(k) * p; ++k) {
      rows += n - static_cast<double>(k) * p;
    }
    return static_cast<double>(b.size() / b.shape()[d]) * rows;
  }
  return 0.0;
}

/// Work of `derivative`, for the GIL decision only: each of the `shape[d] - 1` rows of the
/// hodograph forms a difference of two rows and stores it, scaled, for each of the
/// `b.size() / shape[d]` values the other extents and the coordinates hold
/// (`derivative_ctrl_1d`). Zero for a rational B-spline, a direction out of range or one of
/// degree 0, which the call refuses before any work.
template <class T>
double derivative_ops(const bspline::Bspline<T>& b, long long direction) {
  if (b.rational() || direction < 0 || direction >= static_cast<long long>(b.dim())) {
    return 0.0;
  }
  const auto d = static_cast<std::size_t>(direction);
  if (b.space().spaces()[d].degree() < 1) {
    return 0.0;
  }
  const std::size_t n = b.shape()[d];
  return 2.0 * static_cast<double>(b.size() / n) * static_cast<double>(n - 1);
}

/// Work of `derivative_keep_degree`, for the GIL decision only: `derivative_ops`, plus the
/// hodograph's elevation by one, whose rows store `shape[d] - 1 + S` rows, `S >= num_intervals`
/// (as `elevation_ops`). Zero where `derivative_ops` is, and for a degree above `bincoeff_max_n`,
/// which the call refuses before any work.
template <class T>
double derivative_keep_degree_ops(const bspline::Bspline<T>& b, long long direction) {
  const double hodograph = derivative_ops(b, direction);
  if (hodograph == 0.0) {
    return 0.0;
  }
  const auto d = static_cast<std::size_t>(direction);
  const bspline::BsplineSpace1D<T> space = b.space().spaces()[d];
  if (space.degree() > bincoeff_max_n) {
    return 0.0;
  }
  const std::size_t n = b.shape()[d];
  const auto rows = static_cast<double>(n - 1) + static_cast<double>(space.num_intervals());
  return hodograph + static_cast<double>(b.size() / n) * rows;
}

/// The knots of the in-domain classes of direction `d` of `b`.
template <class T>
double in_domain_knots(const bspline::Bspline<T>& b, std::size_t d) {
  const bspline::BsplineSpace1D<T> space = b.space().spaces()[d];
  const auto mults = space.unique_knots_and_multiplicity(true).second;
  return static_cast<double>(std::accumulate(mults.begin(), mults.end(), std::int64_t{0}));
}

/// Work of `refine_to_common_space_1d`, for the GIL decision only: each of the two receives every
/// in-domain knot of the other that it lacks, so the one with fewer in-domain knots receives at
/// least the difference, `refinement_ops` for them. A lower bound when the classes of either are
/// further apart than the larger of the two tolerances, the comparison the call matches them by.
/// Zero for a direction outside either B-spline, which the call does not check.
template <class T>
double common_space_ops(const bspline::Bspline<T>& a, const bspline::Bspline<T>& b,
                        std::size_t direction) {
  if (direction >= a.dim() || direction >= b.dim()) {
    return 0.0;
  }
  const double ka = in_domain_knots(a, direction);
  const double kb = in_domain_knots(b, direction);
  if (ka == kb) {
    return 0.0;
  }
  return ka < kb ? refinement_ops(a, direction, kb - ka) : refinement_ops(b, direction, ka - kb);
}

/// A knot vector as a NumPy array.
template <class T>
NumpyArray<T> knots_to_numpy(const std::vector<T>& v) {
  NdArray<T> a({v.size()}, for_overwrite);
  std::copy(v.begin(), v.end(), a.data());
  return to_numpy(std::move(a));
}

/// A knot vector and its control points as a pair of NumPy arrays.
template <class T>
std::pair<NumpyArray<T>, NumpyArray<T>> to_pair(bspline::KnotsAndCtrl<T>&& r) {
  NumpyArray<T> knots = knots_to_numpy(r.knots);
  return {std::move(knots), to_numpy(std::move(r.ctrl))};
}

template <class T>
void def_modify(nb::class_<bspline::Bspline<T>>& cls) {
  using B = bspline::Bspline<T>;
  cls.def(
         "elevate_degree",
         [](const B& b, const std::vector<long long>& increments) {
           const ReleaseIfLong release(elevate_ops(b, increments));
           return bspline::elevate_degree<T>(b, increments);
         },
         nb::arg("increments"),
         "Elevate the degree in each direction by its increment (a periodic direction is "
         "elevated through its open form).")
      .def(
          "reduce_degree",
          [](const B& b, const std::vector<long long>& decrements) {
            const ReleaseIfLong release(reduce_ops(b, decrements));
            return bspline::reduce_degree<T>(b, decrements);
          },
          nb::arg("decrements"),
          "Reduce the degree in each direction by its decrement (an approximation).")
      .def(
          "remove_knots",
          [](const B& b, const std::vector<MaybeKnots<T>>& per_dim, std::optional<long long> num,
             std::optional<double> tol) {
            std::vector<std::span<const T>> views;
            views.reserve(per_dim.size());
            for (const auto& k : per_dim) {
              views.push_back(k ? as_span(*k) : std::span<const T>());
            }
            const ReleaseIfLong release(removal_ops<T>(b, views, num));
            return bspline::remove_knots<T>(b, views, num, tol);
          },
          nb::arg("knot_values"), nb::arg("num").none(), nb::arg("tol").none(),
          "Remove the knots per direction (none or empty skips a direction), up to `num` times "
          "each, while the deviation stays within `tol` (the round-off floor when none).")
      .def(
          "derivative",
          [](const B& b, long long direction) {
            const ReleaseIfLong release(derivative_ops(b, direction));
            return bspline::derivative_nonrational<T>(b, direction);
          },
          nb::arg("direction"), "The partial derivative of a non-rational B-spline.")
      .def(
          "derivative_keep_degree",
          [](const B& b, long long direction) {
            const ReleaseIfLong release(derivative_keep_degree_ops(b, direction));
            return bspline::derivative_keep_degree_nonrational<T>(b, direction);
          },
          nb::arg("direction"),
          "The partial derivative of a non-rational B-spline, elevated back to its degree.");
}

template <class T>
void def_functions(nb::module_& m, const std::string& suffix) {
  using B = bspline::Bspline<T>;
  m.def(("refine_to_common_space_1d" + suffix).c_str(),
        [](const B& a, const B& b, std::size_t direction) {
          const ReleaseIfLong release(common_space_ops(a, b, direction));
          return bspline::refine_to_common_space_1d<T>(a, b, direction);
        },
        nb::arg("a"), nb::arg("b"), nb::arg("direction"),
        "Insert into each of two B-splines the knots the other has in one direction and it lacks.");
}

template <class T>
void def_testing(nb::module_& m, const std::string& suffix) {
  m.def(("control_point_scale" + suffix).c_str(),
        [](const In2D<T>& ctrl) {
          return bspline::control_point_scale<T>(std::span<const T>(ctrl.data(), ctrl.size()),
                                                 columns_of(ctrl));
        },
        nb::arg("ctrl").noconvert(),
        "max(bounding-box diagonal, largest norm) of the rows of (points, coordinates).");
  m.def(("roundoff_deviation_floor" + suffix).c_str(),
        [](const In2D<T>& ctrl, long long degree, std::size_t num_stacked) {
          return bspline::roundoff_deviation_floor<T>(std::span<const T>(ctrl.data(), ctrl.size()),
                                                      columns_of(ctrl), saturate_degree(degree),
                                                      num_stacked);
        },
        nb::arg("ctrl").noconvert(), nb::arg("degree"), nb::arg("num_stacked"),
        "The round-off deviation budget for the first copy of a knot.");
  m.def(("homogeneous_deviation_tolerance" + suffix).c_str(),
        [](const In2D<T>& ctrl, double tol) {
          return bspline::homogeneous_deviation_tolerance<T>(
              std::span<const T>(ctrl.data(), ctrl.size()), columns_of(ctrl), tol);
        },
        nb::arg("ctrl").noconvert(), nb::arg("tol_euclidean"),
        "A projected-space deviation budget pulled back to homogeneous control points.");
  m.def(("find_knot_index_and_multiplicity" + suffix).c_str(),
        [](const In1D<T>& knots, long long degree, double knot_value, double tol) {
          const auto [r, s] = bspline::detail::find_knot_index_and_multiplicity<T>(
              as_span(knots), saturate_degree(degree), knot_value, tol);
          return std::make_pair(static_cast<long long>(r), static_cast<long long>(s));
        },
        nb::arg("knots").noconvert(), nb::arg("degree"), nb::arg("knot_value"), nb::arg("tol"),
        "The last index and the multiplicity of the knot class a value falls in.");
  m.def(("degree_elevate_1d" + suffix).c_str(),
        [](long long degree, const In2D<T>& ctrl, const In1D<T>& knots, long long increment) {
          return to_pair(bspline::elevate_degree_1d<T>(saturate_degree(degree), as_view(ctrl),
                                                       as_span(knots), saturate_degree(increment)));
        },
        nb::arg("degree"), nb::arg("ctrl").noconvert(), nb::arg("knots").noconvert(),
        nb::arg("increment"),
        "Degree elevate a 1D B-spline: (control points (n, rank), knots) -> (points, knots).");
  m.def(
      ("degree_reduce_1d" + suffix).c_str(),
      [](long long degree, const In2D<T>& ctrl, const In1D<T>& knots, long long decrement) {
        return to_pair(bspline::reduce_degree_1d<T>(saturate_degree(degree), as_view(ctrl),
                                                    as_span(knots), saturate_degree(decrement)));
      },
      nb::arg("degree"), nb::arg("ctrl").noconvert(), nb::arg("knots").noconvert(),
      nb::arg("decrement"),
      "Degree reduce a 1D B-spline to Bezier form: (points (n, rank), knots) -> (points, knots).");
}

}  // namespace

void register_bspline_modify(nb::module_& m) {
  // The classes exist already (register_bspline_ops runs first); their methods are bound here, in
  // the translation unit that reaches Eigen through the periodic directions.
  auto f64 = nb::borrow<nb::class_<bspline::Bspline<double>>>(m.attr("BsplineF64"));
  auto f32 = nb::borrow<nb::class_<bspline::Bspline<float>>>(m.attr("BsplineF32"));
  def_modify<double>(f64);
  def_modify<float>(f32);
  def_functions<double>(m, "");
  def_functions<float>(m, "_f32");
}

void register_bspline_modify_testing(nb::module_& m) {
  m.attr("bincoeff_max_n") = pantr::bincoeff_max_n;
  m.def(
      "bincoeff", [](long long n, long long k) { return pantr::bincoeff(n, k); }, nb::arg("n"),
      nb::arg("k"), "The exact-integer binomial coefficient C(n, k) as a double (n <= 61).");
  m.def(
      "check_bincoeff_envelope",
      [](long long n, const std::string& what) { pantr::check_bincoeff_envelope(n, what); },
      nb::arg("n"), nb::arg("what"),
      "Refuse an upper index beyond the exactness envelope of the binomial kernel.");
  m.def(
      "num_stacked_control_points",
      [](const std::vector<std::size_t>& shape, std::size_t axis) {
        return bspline::num_stacked_control_points(shape, axis);
      },
      nb::arg("shape"), nb::arg("axis"),
      "The control points the removal kernel's one distance spans: the product of the extents "
      "other than `axis` (the last extent, the coordinates, excluded).");
  def_testing<double>(m, "");
  def_testing<float>(m, "_f32");
}

}  // namespace pantr::bindings
