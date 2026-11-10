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
// None of these functions releases the GIL: each is short beside the cost of releasing it (design
// §2.8, `gil_release_min_ops`).
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

AnyB ruled_any(const AnyB& b1, const AnyB& b2) {
  return in_first_dtype(b1, [&](auto tag) {
    using T = typename decltype(tag)::type;
    return cad::create_ruled<T>(as<T>(b1), as<T>(b2));
  });
}

std::vector<AnyB> make_compat_any(const std::vector<AnyB>& bsplines,
                                  const std::optional<std::vector<long long>>& axes) {
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
  const double tier = tier_of(f);
  return in_first_dtype(f[0], [&](auto tag) {
    using T = typename decltype(tag)::type;
    return cad::detail::create_coons_volume<T>(faces_as<T>(f), tier);
  });
}

AnyB join_any(const AnyB& b1, const AnyB& b2, long long axis) {
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
