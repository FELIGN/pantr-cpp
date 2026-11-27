// _pantr.cad: the constructive-geometry functions of pantr.cad that run in the compiled library
// (the primitives, make_compat, create_ruled, create_revolution, the two Coons constructions and
// join), for pantr.cad's thin wrappers, and the Coons edge readings in `_testing`.
//
// B-splines arrive as the compiled handles of either dtype (a `std::variant`), since the Python API
// accepts B-splines of different dtypes in these functions and each one has its own rule for the
// mixture, reproduced here from the Python implementation this replaces:
//
//   make_compat   homogeneous lists run in their own type; a mixed list is widened to double, made
//                 compatible there, and each result cast back to its own input's dtype.
//   create_ruled  the second B-spline is converted to the first's dtype.
//   create_coons_surface / create_coons_volume
//                 every curve or face is converted to the first one's dtype, and the corner and
//                 edge comparisons are graded at the coarsest dtype among the inputs as supplied.
//   join          a float32 pair or a float64 pair runs in its own type; a float64 first with a
//                 float32 second, or a curve pair of either order, runs in double graded at the
//                 coarsest dtype; a float32 first with a float64 second joined along a dimension
//                 of two or more is refused (as the Python implementation refused it, at the very
//                 end, so the earlier checks win: the join is run first and its refusals stand).
//   create_revolution
//                 float64 only; a float32 input is refused after the same checks, for the same
//                 reason.
//
// None of these calls reaches the pool. `make_compat`, `create_ruled`, `create_revolution`, the
// two Coons constructions and `join` release the GIL through `ReleaseIfLong`, on a lower bound of
// their work (design §2.8): the B-splines arrive as C++ values and the result leaves as one, so
// nothing inside the guard touches a Python object. A refusal the estimate can read off the
// arguments alone (dimensions that differ, an axis out of range, nothing to do) counts no work, so
// such a call keeps the GIL. The other refusals are left unmirrored, since mirroring them would
// repeat the library's own comparisons: a corner, edge or seam mismatch and a ruled pair whose
// widths or nets differ are refused after the compatibility but before the work the estimate
// counts, so such a call may release for nothing, which costs a wait and changes no result; a
// mixed dtype refused at the end has done the work. The primitives build a handful of control
// points and keep the GIL, as does the testing hook.
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/convert.hpp"
#include "pantr/cad/compat.hpp"
#include "pantr/cad/coons.hpp"
#include "pantr/cad/detail/validation.hpp"
#include "pantr/cad/join.hpp"
#include "pantr/cad/operations.hpp"
#include "pantr/cad/primitives.hpp"
#include "pantr/core/errors.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

using F32 = bspline::Bspline<float>;
using F64 = bspline::Bspline<double>;
/// A B-spline of either dtype.
using AnyB = cad::AnyBspline;
using Sweep = std::optional<std::array<double, 2>>;

/// The message of every refusal of a mixture the Python implementation refused.
constexpr const char* mixed_dtype_message = "All B-spline spaces must have the same data type.";

bool is_f32(const AnyB& b) { return std::holds_alternative<F32>(b); }

/// Convert a B-spline of either dtype to type `T`.
template <class T>
bspline::Bspline<T> as(const AnyB& b) {
  return std::visit([](const auto& v) { return bspline::astype<T>(v); }, b);
}

/// The relative tolerance tier of the comparisons among B-splines of the given dtypes: that of the
/// coarsest dtype present (`cad::detail::conservative_tier`).
double tier_of(std::span<const AnyB> all) {
  const bool any_f32 = std::any_of(all.begin(), all.end(), is_f32);
  return any_f32 ? cad::detail::conservative_tier<float>()
                 : cad::detail::conservative_tier<double>();
}

/// Run `fn` on B-splines of type `T`: each argument converted to the first one's dtype.
template <class Fn>
AnyB in_first_dtype(const AnyB& first, Fn&& fn) {
  if (is_f32(first)) {
    return AnyB(fn(std::type_identity<float>{}));
  }
  return AnyB(fn(std::type_identity<double>{}));
}

// Work estimates, for the GIL decisions only. Each reads the arguments as supplied and bounds the
// call's work from below.

std::size_t dim_of(const AnyB& b) {
  return std::visit([](const auto& v) { return v.dim(); }, b);
}

std::size_t size_of(const AnyB& b) {
  return std::visit([](const auto& v) { return v.size(); }, b);
}

std::size_t extent_of(const AnyB& b, std::size_t d) {
  return std::visit([d](const auto& v) { return v.shape()[d]; }, b);
}

std::size_t cp_size_of(const AnyB& b) {
  return std::visit([](const auto& v) { return v.cp_size(); }, b);
}

/// The knots of the in-domain classes of a direction other than its two end classes.
template <class T>
double interior_knots(const bspline::BsplineSpace1D<T>& s) {
  const std::span<const std::int64_t> m = s.unique_knots_and_multiplicity(true).second;
  if (m.size() <= 2) {
    return 0.0;
  }
  return static_cast<double>(std::accumulate(m.begin() + 1, m.end() - 1, std::int64_t{0}));
}

/// Work of `make_compat` on `bsplines` along `axes`, for the GIL decision only. Counted:
///
///  - in a list of mixed dtypes, each float32 input's conversion to double, a copy of its net;
///  - each input periodic along a selected axis, converted to open form (`to_open_ops`'s
///    `b.size()`);
///  - along each selected axis, each input below the largest degree there, elevated to it
///    (`elevation_ops`);
///  - along each selected axis, each input already at the largest degree, given the in-domain
///    interior knots it lacks (`refinement_ops`). The outputs share their interior classes and
///    their multiplicities there, and insertion and elevation only add knots, so every output
///    holds at least the most interior knots any input holds after its elevation, `I + t (N - 1)`
///    for one with `I` interior knots, `N` intervals and increment `t` (A5.9 adds `t` copies of
///    every distinct interior value). An input at the largest degree is not elevated, so it
///    receives at least the difference. A lower bound when the breakpoints are further apart than
///    the merge tolerance; a chain of closer ones merges into fewer classes.
///
/// Zero for fewer than two inputs, which the call returns unchanged, and for the arguments it
/// refuses or returns unchanged before any work: dimensions that differ, an axis out of range, or
/// an empty list of axes.
double make_compat_ops(std::span<const AnyB> bsplines,
                       const std::optional<std::vector<long long>>& axes) {
  if (bsplines.size() < 2) {
    return 0.0;
  }
  const std::size_t dim = dim_of(bsplines[0]);
  if (std::any_of(bsplines.begin(), bsplines.end(),
                  [dim](const AnyB& b) { return dim_of(b) != dim; })) {
    return 0.0;
  }
  std::vector<bool> selected(dim, !axes.has_value());
  if (axes) {
    if (axes->empty()) {
      return 0.0;
    }
    for (const long long a : *axes) {
      if (a < 0 || a >= static_cast<long long>(dim)) {
        return 0.0;
      }
      selected[static_cast<std::size_t>(a)] = true;
    }
  }
  const bool any_f32 = std::any_of(bsplines.begin(), bsplines.end(), is_f32);
  const bool mixed = any_f32 && !std::all_of(bsplines.begin(), bsplines.end(), is_f32);
  double ops = 0.0;
  for (const AnyB& b : bsplines) {
    std::visit(
        [&](const auto& v) {
          const auto whole = v.space();
          bool periodic = false;
          for (std::size_t a = 0; a < dim; ++a) {
            periodic = periodic || (selected[a] && whole.spaces()[a].periodic());
          }
          const auto net = static_cast<double>(v.size());
          ops += (mixed && is_f32(b) ? net : 0.0) + (periodic ? net : 0.0);
        },
        b);
  }
  for (std::size_t a = 0; a < dim; ++a) {
    if (!selected[a]) {
      continue;
    }
    int top = 0;
    for (const AnyB& b : bsplines) {
      top = (std::max)(
          top, std::visit([a](const auto& v) { return v.space().spaces()[a].degree(); }, b));
    }
    double most = 0.0;
    for (const AnyB& b : bsplines) {
      most = (std::max)(most, std::visit(
                                  [a, top](const auto& v) {
                                    const auto space = v.space().spaces()[a];
                                    const auto t = static_cast<double>(top - space.degree());
                                    const auto n = static_cast<double>(space.num_intervals());
                                    return interior_knots(space) + t * (n - 1.0);
                                  },
                                  b));
    }
    for (const AnyB& b : bsplines) {
      ops += std::visit(
          [a, top, most](const auto& v) {
            const auto space = v.space().spaces()[a];
            if (space.degree() < top) {
              return elevation_ops(v, a, static_cast<double>(top - space.degree()));
            }
            const double lacking = most - interior_knots(space);
            return lacking > 0.0 ? refinement_ops(v, a, lacking) : 0.0;
          },
          b);
    }
  }
  return ops;
}

/// Work of `create_ruled`, for the GIL decision only: `make_compat_ops` for the pair, plus the
/// stacked net, `2 * points * width` values written and then copied into the result, where
/// `points * width` holds at least either input's net (compatibility only grows a net, and the
/// second's width is the first's or 1). Zero for dimensions that differ or above 2, which the call
/// refuses before any work.
double ruled_ops(const AnyB& b1, const AnyB& b2) {
  const std::size_t dim = dim_of(b1);
  if (dim != dim_of(b2) || dim > cad::detail::max_dim_for_operations) {
    return 0.0;
  }
  const std::array<AnyB, 2> pair = {b1, b2};
  const auto nets = static_cast<double>(size_of(b1) + size_of(b2));
  return make_compat_ops(pair, std::nullopt) + 2.0 * nets;
}

/// Work of `create_revolution`, for the GIL decision only: every control point of the input,
/// promoted to rational, is multiplied into each of the arc's control points, a `(4, 4)` matrix
/// product of 16 multiply-adds (`detail::revolve_control_points`); the full circle has 9 arc points
/// and any sweep at least 3 (one quadratic span). A lower bound: it leaves out the two affine maps
/// and the promotion. Zero for the arguments the call refuses before any work: a dimension above
/// 2, a rank other than 3, a point or an axis of more than three coordinates, or an axis whose
/// squared norm is zero.
double revolution_ops(const AnyB& b, const std::vector<double>& point,
                      const std::vector<double>& axis, const Sweep& sweep) {
  const std::size_t rank = std::visit([](const auto& v) { return v.rank(); }, b);
  if (dim_of(b) > cad::detail::max_dim_for_operations || rank != cad::detail::physical_dim ||
      point.size() > cad::detail::physical_dim || axis.size() > cad::detail::physical_dim) {
    return 0.0;
  }
  double squared = 0.0;
  for (const double c : axis) {
    squared += c * c;
  }
  if (squared == 0.0) {
    return 0.0;
  }
  const auto points = static_cast<double>(size_of(b) / cp_size_of(b));
  const double arc_points = sweep.has_value() ? 3.0 : 9.0;
  return 16.0 * points * arc_points;
}

/// Work of `create_coons_surface`, for the GIL decision only: the result's net is at least as
/// large as the u-curves' points by the v-curves' (compatibility only grows a net), of at least
/// the narrowest input's width, and `combine_terms` adds three terms into each of its values, a
/// multiply-add each. A lower bound: it leaves out the compatibility and the ruled surfaces. Zero
/// when a curve is not 1D, which the call refuses before any work.
double coons_surface_ops(const AnyB& v0, const AnyB& v1, const AnyB& u0, const AnyB& u1) {
  const std::array<const AnyB*, 4> all = {&v0, &v1, &u0, &u1};
  if (std::any_of(all.begin(), all.end(), [](const AnyB* c) { return dim_of(*c) != 1; })) {
    return 0.0;
  }
  std::size_t width = cp_size_of(v0);
  for (const AnyB* c : all) {
    width = (std::min)(width, cp_size_of(*c));
  }
  const auto n_u = static_cast<double>((std::max)(extent_of(u0, 0), extent_of(u1, 0)));
  const auto n_v = static_cast<double>((std::max)(extent_of(v0, 0), extent_of(v1, 0)));
  return 3.0 * n_u * n_v * static_cast<double>(width);
}

/// Work of `create_coons_volume`, for the GIL decision only: as `coons_surface_ops`, with seven
/// terms added into a net at least as large, along each direction, as either face pair that spans
/// it (the u- and v-faces, `(v, w)` and `(u, w)`; the w-faces, `(u, v)`). Zero when a face is not
/// 2D, which the call refuses before any work.
double coons_volume_ops(const std::array<AnyB, 6>& f) {
  if (std::any_of(f.begin(), f.end(), [](const AnyB& b) { return dim_of(b) != 2; })) {
    return 0.0;
  }
  std::size_t width = cp_size_of(f[0]);
  for (const AnyB& b : f) {
    width = (std::min)(width, cp_size_of(b));
  }
  const auto most = [&](std::size_t i, std::size_t ai, std::size_t j, std::size_t aj) {
    const std::array<std::size_t, 4> e = {extent_of(f[i], ai), extent_of(f[i + 1], ai),
                                          extent_of(f[j], aj), extent_of(f[j + 1], aj)};
    return static_cast<double>(*std::max_element(e.begin(), e.end()));
  };
  const double n_u = most(2, 0, 4, 0);  // the v-faces' axis 0, the w-faces' axis 0
  const double n_v = most(0, 0, 4, 1);  // the u-faces' axis 0, the w-faces' axis 1
  const double n_w = most(0, 1, 2, 1);  // the u-faces' axis 1, the v-faces' axis 1
  return 7.0 * n_u * n_v * n_w * static_cast<double>(width);
}

/// Work of `join`, for the GIL decision only: `make_compat_ops` along the other axes, plus the
/// joined net, written and then copied into the result, which holds at least either input's net.
/// A lower bound: it leaves out the opening and the junction's knot removal. Zero for dimensions
/// that differ or an axis out of range, which the call refuses before any work.
double join_ops(const AnyB& b1, const AnyB& b2, long long axis) {
  const std::size_t dim = dim_of(b1);
  if (dim != dim_of(b2) || axis < 0 || axis >= static_cast<long long>(dim)) {
    return 0.0;
  }
  std::vector<long long> others;
  for (std::size_t a = 0; a < dim; ++a) {
    if (a != static_cast<std::size_t>(axis)) {
      others.push_back(static_cast<long long>(a));
    }
  }
  const std::array<AnyB, 2> pair = {b1, b2};
  const double compat = others.empty() ? 0.0 : make_compat_ops(pair, others);
  return compat + 2.0 * static_cast<double>((std::max)(size_of(b1), size_of(b2)));
}

AnyB ruled_any(const AnyB& b1, const AnyB& b2) {
  const ReleaseIfLong release(ruled_ops(b1, b2));
  return in_first_dtype(b1, [&](auto tag) {
    using T = typename decltype(tag)::type;
    return cad::create_ruled<T>(as<T>(b1), as<T>(b2));
  });
}

std::vector<AnyB> make_compat_any(const std::vector<AnyB>& bsplines,
                                  const std::optional<std::vector<long long>>& axes) {
  const ReleaseIfLong release(make_compat_ops(bsplines, axes));
  std::vector<F32> f32;
  std::vector<F64> f64;
  for (const AnyB& b : bsplines) {
    if (const auto* single = std::get_if<F32>(&b)) {
      f32.push_back(*single);
    } else {
      f64.push_back(std::get<F64>(b));
    }
  }
  std::vector<AnyB> out;
  if (f64.empty()) {
    for (F32& r : cad::make_compat<float>(f32, axes)) {
      out.emplace_back(std::move(r));
    }
  } else if (f32.empty()) {
    for (F64& r : cad::make_compat<double>(f64, axes)) {
      out.emplace_back(std::move(r));
    }
  } else {
    out = cad::detail::make_compat(std::span<const AnyB>(bsplines), axes);
  }
  return out;
}

AnyB coons_surface_any(const AnyB& v0, const AnyB& v1, const AnyB& u0, const AnyB& u1) {
  const ReleaseIfLong release(coons_surface_ops(v0, v1, u0, u1));
  const std::array<AnyB, 4> all = {v0, v1, u0, u1};
  const double tier = tier_of(all);
  return in_first_dtype(v0, [&](auto tag) {
    using T = typename decltype(tag)::type;
    const std::array<std::array<bspline::Bspline<T>, 2>, 2> curves = {
        {{as<T>(v0), as<T>(v1)}, {as<T>(u0), as<T>(u1)}}};
    return cad::detail::create_coons_surface<T>(curves, tier);
  });
}

/// The six faces `((u0, u1), (v0, v1), (w0, w1))`, each converted to type `T`.
template <class T>
std::array<std::array<bspline::Bspline<T>, 2>, 3> faces_as(const std::array<AnyB, 6>& f) {
  return {{{as<T>(f[0]), as<T>(f[1])}, {as<T>(f[2]), as<T>(f[3])}, {as<T>(f[4]), as<T>(f[5])}}};
}

AnyB coons_volume_any(const std::array<AnyB, 6>& f) {
  const ReleaseIfLong release(coons_volume_ops(f));
  const double tier = tier_of(f);
  return in_first_dtype(f[0], [&](auto tag) {
    using T = typename decltype(tag)::type;
    return cad::detail::create_coons_volume<T>(faces_as<T>(f), tier);
  });
}

AnyB join_any(const AnyB& b1, const AnyB& b2, long long axis) {
  const ReleaseIfLong release(join_ops(b1, b2, axis));
  if (is_f32(b1) == is_f32(b2)) {
    if (is_f32(b1)) {
      return AnyB(cad::join<float>(std::get<F32>(b1), std::get<F32>(b2), axis));
    }
    return AnyB(cad::join<double>(std::get<F64>(b1), std::get<F64>(b2), axis));
  }
  // Mixed: in double, graded at the coarsest of the two (the float32 one).
  const double tier = cad::detail::conservative_tier<float>();
  const F64 joined = cad::detail::join<double>(as<double>(b1), as<double>(b2), axis, tier);
  if (is_f32(b1) && std::get<F32>(b1).dim() >= 2) {
    // The Python implementation joined in each input's own dtype along the join direction and
    // refused the result's mixed spaces, after every check above; so does this.
    throw value_error(mixed_dtype_message);
  }
  return AnyB(joined);
}

AnyB revolution_any(const AnyB& b, const std::vector<double>& point,
                    const std::vector<double>& axis, const Sweep& sweep) {
  const ReleaseIfLong release(revolution_ops(b, point, axis, sweep));
  const F64 revolved = cad::create_revolution(as<double>(b), std::span<const double>(point),
                                              std::span<const double>(axis), sweep);
  if (is_f32(b)) {
    // The arc is built in double, and the Python implementation refused the mixed spaces of the
    // result, after every check above; so does this.
    throw value_error(mixed_dtype_message);
  }
  return AnyB(revolved);
}

/// An edge reading as it crosses the binding: label, face_a, curve_a, face_b, curve_b, the two
/// corner labels.
using EdgeTuple = std::tuple<std::string, std::string, nb::object, std::string, nb::object,
                             std::string, std::string>;

template <class T>
std::vector<EdgeTuple> edge_pairs_of(const std::array<AnyB, 6>& f) {
  const std::array<std::array<bspline::Bspline<T>, 2>, 3> faces = faces_as<T>(f);
  std::vector<EdgeTuple> out;
  for (const cad::detail::EdgeReadings<T>& e : cad::detail::coons_edge_pairs<T>(faces)) {
    out.emplace_back(e.label, e.face_a, nb::cast(e.curve_a), e.face_b, nb::cast(e.curve_b),
                     e.corners[0], e.corners[1]);
  }
  return out;
}

}  // namespace

void register_cad(nb::module_& m) {
  m.def(
      "create_line",
      [](const std::vector<double>& p0, const std::vector<double>& p1) {
        return cad::create_line(std::span<const double>(p0), std::span<const double>(p1));
      },
      nb::arg("p0"), nb::arg("p1"),
      "A degree-1 curve between two points (up to three coordinates).");
  m.def(
      "create_circle",
      [](double radius, const std::optional<std::vector<double>>& center, const Sweep& sweep) {
        if (center) {
          return cad::create_circle(radius, std::span<const double>(*center), sweep);
        }
        return cad::create_circle(radius, std::nullopt, sweep);
      },
      nb::arg("radius"), nb::arg("center").none(), nb::arg("sweep").none(),
      "A rational quadratic circle or arc; `sweep` is (start, end) in radians, none for the full "
      "circle.");
  m.def(
      "create_bilinear", []() { return cad::create_bilinear(); },
      "The bilinear unit square [-0.5, 0.5]^2 in the xy-plane.");
  m.def(
      "create_bilinear",
      [](const std::vector<double>& values, const std::vector<std::size_t>& shape) {
        return cad::create_bilinear(std::span<const double>(values),
                                    std::span<const std::size_t>(shape));
      },
      nb::arg("values"), nb::arg("shape"),
      "A bilinear surface from corner coordinates, row-major over `shape` = (2, 2, rank).");
  m.def(
      "create_trilinear", []() { return cad::create_trilinear(); },
      "The trilinear unit cube [-0.5, 0.5]^3.");
  m.def(
      "create_trilinear",
      [](const std::vector<double>& values, const std::vector<std::size_t>& shape) {
        return cad::create_trilinear(std::span<const double>(values),
                                     std::span<const std::size_t>(shape));
      },
      nb::arg("values"), nb::arg("shape"),
      "A trilinear volume from corner coordinates, row-major over `shape` = (2, 2, 2, rank).");
  m.def("make_compat", &make_compat_any, nb::arg("bsplines"), nb::arg("axes").none(),
        "B-splines made compatible (degree and knots) along the axes; none means every axis.");
  m.def("create_ruled", &ruled_any, nb::arg("bspline1"), nb::arg("bspline2"),
        "The ruled surface or volume between two B-splines.");
  m.def("create_revolution", &revolution_any, nb::arg("bspline"), nb::arg("point"), nb::arg("axis"),
        nb::arg("sweep").none(),
        "A curve or surface revolved about the axis through `point` with direction `axis`.");
  m.def("create_coons_surface", &coons_surface_any, nb::arg("c_v0"), nb::arg("c_v1"),
        nb::arg("c_u0"), nb::arg("c_u1"), "The Coons patch of four boundary curves.");
  m.def(
      "create_coons_volume",
      [](const AnyB& u0, const AnyB& u1, const AnyB& v0, const AnyB& v1, const AnyB& w0,
         const AnyB& w1) { return coons_volume_any({u0, u1, v0, v1, w0, w1}); },
      nb::arg("face_u0"), nb::arg("face_u1"), nb::arg("face_v0"), nb::arg("face_v1"),
      nb::arg("face_w0"), nb::arg("face_w1"), "The Coons volume of six boundary faces.");
  m.def("join", &join_any, nb::arg("bspline1"), nb::arg("bspline2"), nb::arg("axis"),
        "Two B-splines joined along a parametric axis with C0 continuity.");
}

void register_cad_testing(nb::module_& m) {
  m.def(
      "coons_edge_pairs",
      [](const AnyB& u0, const AnyB& u1, const AnyB& v0, const AnyB& v1, const AnyB& w0,
         const AnyB& w1) {
        const std::array<AnyB, 6> f = {u0, u1, v0, v1, w0, w1};
        return is_f32(u0) ? edge_pairs_of<float>(f) : edge_pairs_of<double>(f);
      },
      nb::arg("face_u0"), nb::arg("face_u1"), nb::arg("face_v0"), nb::arg("face_v1"),
      nb::arg("face_w0"), nb::arg("face_w1"),
      "The 12 edges of a Coons volume, each read from both faces that carry it, as (label, "
      "face_a, curve_a, face_b, curve_b, corner_low, corner_high) tuples.");
}

}  // namespace pantr::bindings
