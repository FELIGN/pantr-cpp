// _pantr.multipatch: the interface check, the control-point matching of two faces, the detection of
// the interfaces among patches and the validation of a MultiPatch, for pantr.multipatch's wrappers.
//
// Patches arrive as the compiled B-spline handles of either dtype (a `std::variant`), since the
// Python API accepts patches of different dtypes in `detect_interfaces`. Where the dtypes differ
// the patches are converted to double (`bspline::astype`, exact for float32) and compared at the
// relative tolerance each pair's own dtypes give: the larger of the two patches' strict tolerances,
// as the Python implementation computed it. An interface crosses the binding as the tuple
// `(patch_a, face_a, patch_b, face_b, axis_map, flips)`; the Python `Interface` is built from it.
//
// The GIL is released around a detection that runs on the pool (the pairs are tested there,
// `ReleaseGil` on the detection's own work estimate), once the arguments are plain C++ values.
// Every check is the C++ API's, with the messages the Python suite matches.
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/convert.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/multipatch/detect.hpp"
#include "pantr/multipatch/interface.hpp"
#include "pantr/multipatch/match.hpp"
#include "pantr/multipatch/multi_patch.hpp"
#include "pantr/tolerance.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

using F32 = bspline::Bspline<float>;
using F64 = bspline::Bspline<double>;
/// A patch of either dtype.
using AnyPatch = std::variant<F32, F64>;
/// An interface as it crosses the binding.
using InterfaceTuple = std::tuple<long long, long long, long long, long long,
                                  std::vector<long long>, std::vector<bool>>;

std::vector<InterfaceTuple> to_tuples(const std::vector<multipatch::Interface>& found) {
  std::vector<InterfaceTuple> out;
  out.reserve(found.size());
  for (const multipatch::Interface& i : found) {
    out.emplace_back(i.patch_a(), i.face_a(), i.patch_b(), i.face_b(), i.axis_map(), i.flips());
  }
  return out;
}

/// Whether detecting the interfaces among `n_patches` patches of parametric dimension `dim` (the
/// first patch's) and largest rank `max_rank` runs on the pool: its loop over every face, priced as
/// the detection prices it (`multipatch::detail::detection_work`).
bool detection_on_pool(std::size_t n_patches, std::size_t dim, std::size_t max_rank) {
  const std::size_t n_sides = n_patches * 2 * dim;
  return n_sides > 0 && parallel::worth_parallel(
                            n_sides, multipatch::detail::detection_work(n_sides, dim, max_rank));
}

multipatch::Interface from_tuple(const InterfaceTuple& t) {
  const auto& [patch_a, face_a, patch_b, face_b, axis_map, flips] = t;
  return multipatch::Interface(patch_a, face_a, patch_b, face_b, axis_map, flips);
}

/// Detect the interfaces among patches of either dtype.
///
/// Homogeneous lists run in their own type. A list of both dtypes is converted to double and each
/// patch carries the relative tolerance of its own dtype (a pair uses the larger of its two).
std::vector<multipatch::Interface> detect_any(const std::vector<AnyPatch>& patches,
                                              std::optional<double> tol) {
  std::vector<F32> f32;
  std::vector<F64> f64;
  std::size_t max_rank = 1;
  for (const AnyPatch& p : patches) {
    if (const auto* single = std::get_if<F32>(&p)) {
      f32.push_back(*single);
      max_rank = (std::max)(max_rank, single->rank());
    } else {
      f64.push_back(std::get<F64>(p));
      max_rank = (std::max)(max_rank, f64.back().rank());
    }
  }
  const std::size_t dim =
      patches.empty() ? 0 : std::visit([](const auto& b) { return b.dim(); }, patches.front());
  const ReleaseGil release(detection_on_pool(patches.size(), dim, max_rank));
  if (f64.empty()) {
    return multipatch::detect_interfaces<float>(f32, tol);
  }
  if (f32.empty()) {
    return multipatch::detect_interfaces<double>(f64, tol);
  }
  std::vector<F64> all;
  std::vector<double> relative;
  all.reserve(patches.size());
  relative.reserve(patches.size());
  for (const AnyPatch& p : patches) {
    if (const auto* single = std::get_if<F32>(&p)) {
      all.push_back(bspline::astype<double>(*single));
      relative.push_back(static_cast<double>(tolerance::get_strict<float>()));
    } else {
      all.push_back(std::get<F64>(p));
      relative.push_back(static_cast<double>(tolerance::get_strict<double>()));
    }
  }
  return multipatch::detail::detect_interfaces<double>(all, relative, tol);
}

/// Validate a MultiPatch of patches of one dtype, returning its interfaces.
template <class T>
std::vector<multipatch::Interface> multi_patch_of(
    const std::vector<bspline::Bspline<T>>& patches,
    const std::optional<std::vector<InterfaceTuple>>& interfaces, std::optional<double> tol) {
  std::optional<std::vector<multipatch::Interface>> given;
  if (interfaces) {
    given.emplace();
    for (const InterfaceTuple& t : *interfaces) {
      given->push_back(from_tuple(t));
    }
  }
  // Only detection runs on the pool; verifying a supplied list is serial and reads only the faces
  // it names, so it keeps the GIL.
  std::size_t max_rank = 1;
  for (const bspline::Bspline<T>& p : patches) {
    max_rank = (std::max)(max_rank, p.rank());
  }
  const std::size_t dim = patches.empty() ? 0 : patches.front().dim();
  const ReleaseGil release(!given && detection_on_pool(patches.size(), dim, max_rank));
  return multipatch::MultiPatch<T>(patches, std::move(given), tol).interfaces();
}

}  // namespace

void register_multipatch(nb::module_& m) {
  m.def(
      "validate_interface",
      [](long long patch_a, long long face_a, long long patch_b, long long face_b,
         const std::vector<long long>& axis_map, const std::vector<bool>& flips) {
        const multipatch::Interface checked(patch_a, face_a, patch_b, face_b, axis_map, flips);
        static_cast<void>(checked);
      },
      nb::arg("patch_a"), nb::arg("face_a"), nb::arg("patch_b"), nb::arg("face_b"),
      nb::arg("axis_map"), nb::arg("flips"),
      "Refuse an interface record that cannot hold (the checks of the C++ `Interface`).");
  m.def(
      "match_face_cps",
      [](const std::vector<std::size_t>& num_basis_a, long long face_a,
         const std::vector<std::size_t>& num_basis_b, long long face_b,
         const std::vector<long long>& axis_map, const std::vector<bool>& flips) {
        multipatch::MatchedCps matched =
            multipatch::match_face_cps(std::span<const std::size_t>(num_basis_a), face_a,
                                       std::span<const std::size_t>(num_basis_b), face_b,
                                       std::span<const long long>(axis_map), flips);
        return std::make_pair(vector_to_numpy(std::move(matched.a)),
                              vector_to_numpy(std::move(matched.b)));
      },
      nb::arg("num_basis_a"), nb::arg("face_a"), nb::arg("num_basis_b"), nb::arg("face_b"),
      nb::arg("axis_map"), nb::arg("flips"),
      "The flat control-point ids of two faces, paired under an axis correspondence.");
  m.def(
      "detect_interfaces",
      [](const std::vector<AnyPatch>& patches, std::optional<double> tol) {
        return to_tuples(detect_any(patches, tol));
      },
      nb::arg("patches"), nb::arg("tol").none(),
      "The conforming interfaces among patches, as (patch_a, face_a, patch_b, face_b, axis_map, "
      "flips) tuples sorted by their first four entries.");
  m.def(
      "multi_patch_interfaces",
      [](const std::variant<std::vector<F32>, std::vector<F64>>& patches,
         const std::optional<std::vector<InterfaceTuple>>& interfaces, std::optional<double> tol) {
        return to_tuples(std::visit(
            [&](const auto& list) { return multi_patch_of(list, interfaces, tol); }, patches));
      },
      nb::arg("patches"), nb::arg("interfaces").none(), nb::arg("tol").none(),
      "Validate a collection of patches of one dtype (and the supplied interfaces, each verified "
      "against the geometry), returning its interfaces: the supplied ones, or the detected ones.");
}

}  // namespace pantr::bindings
