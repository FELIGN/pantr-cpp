/// @file match.hpp
/// @brief Control-point index arithmetic across a pair of patch faces.
///
/// Pure index arithmetic: no geometry is read, so nothing here says whether the two faces actually
/// coincide in space; `detect_interfaces` is what establishes that.
#pragma once

#include "pantr/bspline/space.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/multipatch/interface.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::multipatch {

/// The flat control-point ids of two faces, paired.
///
/// Attributes:
///     a (std::vector<std::int64_t>): Flat ids on the first face, ascending.
///     b (std::vector<std::int64_t>): `b[i]` is the id matched to `a[i]` on the second face.
struct MatchedCps {
  std::vector<std::int64_t> a;
  std::vector<std::int64_t> b;
};

namespace detail {

/// Get the per-direction basis counts of a space.
///
/// Args:
///     space (const bspline::BsplineSpace<T>&): The space.
///
/// Returns:
///     std::vector<std::size_t>: One count per direction.
template <StorageFloat T>
[[nodiscard]] std::vector<std::size_t> num_basis_of(const bspline::BsplineSpace<T>& space) {
  std::vector<std::size_t> out;
  out.reserve(space.dim());
  for (const auto& s : space.spaces()) {
    out.push_back(static_cast<std::size_t>(s.num_basis()));
  }
  return out;
}

/// Get the C-order strides of a patch's control-point grid, in control points (the last extent of
/// the control-point array, the coordinates per point, excluded).
///
/// Args:
///     num_basis (std::span<const std::size_t>): Per-direction basis counts.
///
/// Returns:
///     std::vector<std::size_t>: `stride[d]` is the product of `num_basis[d + 1 ...]`.
[[nodiscard]] inline std::vector<std::size_t> c_strides(std::span<const std::size_t> num_basis) {
  std::vector<std::size_t> stride(num_basis.size(), 1);
  for (std::size_t d = num_basis.size(); d-- > 1;) {
    stride[d - 1] = stride[d] * num_basis[d];
  }
  return stride;
}

/// Refuse a pair of faces whose matched tangential directions hold different numbers of basis
/// functions.
///
/// Args:
///     num_basis_a (std::span<const std::size_t>): Basis counts of the first space.
///     face_a (long long): Face id on the first space.
///     num_basis_b (std::span<const std::size_t>): Basis counts of the second space.
///     axis_map (std::span<const long long>): Patch-b axis per tangential axis of face a.
///
/// Returns:
///     bool: Whether every matched pair of directions holds the same number of functions.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
[[nodiscard]] inline bool tangential_counts_agree(std::span<const std::size_t> num_basis_a,
                                                  long long face_a,
                                                  std::span<const std::size_t> num_basis_b,
                                                  std::span<const long long> axis_map) {
  const std::vector<long long> tang_a = tangential_axes(face_a, num_basis_a.size());
  for (std::size_t k = 0; k < tang_a.size(); ++k) {
    if (num_basis_a[static_cast<std::size_t>(tang_a[k])] !=
        num_basis_b[static_cast<std::size_t>(axis_map[k])]) {
      return false;
    }
  }
  return true;
}

/// Pair the flat control-point ids of two faces under an axis correspondence.
///
/// The ids are visited in the order of a C-order ravel over the tangential axes of face a taken in
/// increasing order (the first one slowest), which is ascending in the flat id on face a; face b's
/// id is that of the matched node, with each flipped axis counted from the other end.
///
/// Args:
///     num_basis_a (std::span<const std::size_t>): Basis counts of the first space.
///     face_a (long long): Face id on the first space.
///     num_basis_b (std::span<const std::size_t>): Basis counts of the second space.
///     face_b (long long): Face id on the second space.
///     axis_map (std::span<const long long>): Patch-b axis per tangential axis of face a.
///     flips (const std::vector<bool>&): Whether each of those axis pairs runs opposed.
///
/// Returns:
///     MatchedCps: The paired ids.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): equal dimensions, faces in
///     range, `axis_map` a permutation of face b's tangential axes, `flips` of its length, matched
///     directions of equal size, and flat ids that fit `std::int64_t`. For general use, call
///     `match_face_cps` instead.
[[nodiscard]] inline MatchedCps build_matched_cps(std::span<const std::size_t> num_basis_a,
                                                  long long face_a,
                                                  std::span<const std::size_t> num_basis_b,
                                                  long long face_b,
                                                  std::span<const long long> axis_map,
                                                  const std::vector<bool>& flips) {
  const std::size_t dim = num_basis_a.size();
  const std::vector<std::size_t> stride_a = c_strides(num_basis_a);
  const std::vector<std::size_t> stride_b = c_strides(num_basis_b);
  const auto [own_a, side_a] = face_axis_side(face_a);
  const auto [own_b, side_b] = face_axis_side(face_b);
  const auto axis_own_a = static_cast<std::size_t>(own_a);
  const auto axis_own_b = static_cast<std::size_t>(own_b);
  const std::size_t base_a = side_a == 0 ? 0 : (num_basis_a[axis_own_a] - 1) * stride_a[axis_own_a];
  const std::size_t base_b = side_b == 0 ? 0 : (num_basis_b[axis_own_b] - 1) * stride_b[axis_own_b];

  const std::vector<long long> tang_a = tangential_axes(face_a, dim);
  const std::size_t n_tang = tang_a.size();
  std::vector<std::size_t> extent(n_tang);
  std::vector<std::size_t> step_a(n_tang);  // flat-id step of one tangential index on face a
  std::vector<std::size_t> step_b(n_tang);  // the same on face b, before any flip
  std::vector<std::size_t> top_b(n_tang);   // flat id of the last node of the matched b axis
  std::size_t count = 1;
  for (std::size_t k = 0; k < n_tang; ++k) {
    const auto axis_a = static_cast<std::size_t>(tang_a[k]);
    const auto axis_b = static_cast<std::size_t>(axis_map[k]);
    extent[k] = num_basis_a[axis_a];
    step_a[k] = stride_a[axis_a];
    step_b[k] = stride_b[axis_b];
    top_b[k] = (num_basis_b[axis_b] - 1) * stride_b[axis_b];
    count *= extent[k];
  }

  MatchedCps out;
  out.a.reserve(count);
  out.b.reserve(count);
  std::vector<std::size_t> index(n_tang, 0);  // the odometer over the tangential axes
  for (std::size_t n = 0; n < count; ++n) {
    std::size_t flat_a = base_a;
    std::size_t flat_b = base_b;
    for (std::size_t k = 0; k < n_tang; ++k) {
      flat_a += index[k] * step_a[k];
      flat_b += flips[k] ? top_b[k] - index[k] * step_b[k] : index[k] * step_b[k];
    }
    out.a.push_back(static_cast<std::int64_t>(flat_a));
    out.b.push_back(static_cast<std::int64_t>(flat_b));
    for (std::size_t k = n_tang; k-- > 0;) {
      if (++index[k] < extent[k]) {
        break;
      }
      index[k] = 0;
    }
  }
  return out;
}

}  // namespace detail

/// Pair the flat control-point ids of two faces under an axis correspondence.
///
/// Flat ids are C-order over the per-direction basis counts. The `k`-th tangential axis of face
/// `face_a`, in increasing-axis order, is taken to correspond to patch axis `axis_map[k]` of the
/// second space, running backwards when `flips[k]`. The result is in the order of a C-order ravel
/// over face a's tangential axes (the first slowest): `a` is ascending.
///
/// This is index arithmetic only. It does not check that the two faces coincide geometrically;
/// `detect_interfaces` is what establishes that.
///
/// Args:
///     num_basis_a (std::span<const std::size_t>): Basis counts of the first space, one per
///         direction.
///     face_a (long long): Face id on the first space.
///     num_basis_b (std::span<const std::size_t>): Basis counts of the second space.
///     face_b (long long): Face id on the second space.
///     axis_map (std::span<const long long>): Patch-b axis per tangential axis of face a; a
///         permutation of face b's tangential axes.
///     flips (const std::vector<bool>&): Whether each of those axis pairs runs opposed.
///
/// Returns:
///     MatchedCps: Equal-length id lists; `a[i]` is matched to `b[i]`.
///
/// Raises:
///     value_error: The two spaces differ in dimension, a face id is out of range, `flips` and
///         `axis_map` differ in length, `axis_map` has not the length `dim - 1`, is not a
///         permutation of face b's tangential axes, or the matched tangential directions hold
///         different numbers of basis functions (checked in that order).
///     capacity_error: The flat ids do not fit `std::int64_t`.
[[nodiscard]] inline MatchedCps match_face_cps(std::span<const std::size_t> num_basis_a,
                                               long long face_a,
                                               std::span<const std::size_t> num_basis_b,
                                               long long face_b,
                                               std::span<const long long> axis_map,
                                               const std::vector<bool>& flips) {
  if (num_basis_a.size() != num_basis_b.size()) {
    throw value_error("Both spaces must have the same parametric dimension; got " +
                      std::to_string(num_basis_a.size()) + " and " +
                      std::to_string(num_basis_b.size()));
  }
  const std::size_t dim = num_basis_a.size();
  const auto n_faces = 2 * static_cast<long long>(dim);
  const std::pair<const char*, long long> faces[] = {{"face_a", face_a}, {"face_b", face_b}};
  for (const auto& [name, face] : faces) {
    if (!(0 <= face && face < n_faces)) {
      throw value_error(std::string(name) + "=" + std::to_string(face) + " out of range for a " +
                        std::to_string(dim) + "D space; expected 0 <= " + name + " < " +
                        std::to_string(n_faces));
    }
  }
  if (flips.size() != axis_map.size()) {
    throw value_error("flips and axis_map must have equal length; got " +
                      std::to_string(flips.size()) + " and " + std::to_string(axis_map.size()));
  }
  if (axis_map.size() != dim - 1) {
    throw value_error("axis_map must have length " + std::to_string(dim - 1) + " for a " +
                      std::to_string(dim) + "D space");
  }
  detail::check_axis_map_permutation(axis_map, face_b, dim);

  const std::vector<long long> tang_a = tangential_axes(face_a, dim);
  for (std::size_t k = 0; k < tang_a.size(); ++k) {
    const auto axis_a = static_cast<std::size_t>(tang_a[k]);
    const auto axis_b = static_cast<std::size_t>(axis_map[k]);
    if (num_basis_a[axis_a] != num_basis_b[axis_b]) {
      throw value_error(
          "Matched tangential directions hold different numbers of basis functions: axis " +
          std::to_string(axis_a) + " of space_a has " + std::to_string(num_basis_a[axis_a]) +
          ", axis " + std::to_string(axis_b) + " of space_b has " +
          std::to_string(num_basis_b[axis_b]));
    }
  }
  // The flat ids are below the product of each space's counts; the build narrows them to int64.
  const auto fits = [](std::span<const std::size_t> counts) {
    return checked_product(counts) <=
           static_cast<std::size_t>((std::numeric_limits<std::int64_t>::max)());
  };
  if (!fits(num_basis_a) || !fits(num_basis_b)) {
    throw capacity_error("the flat control-point ids of the spaces do not fit a 64-bit integer");
  }
  return detail::build_matched_cps(num_basis_a, face_a, num_basis_b, face_b, axis_map, flips);
}

/// Pair the flat control-point ids of two faces of two spaces, of any storage types.
///
/// Args:
///     space_a (const bspline::BsplineSpace<Ta>&): The first space.
///     face_a (long long): Face id on the first space.
///     space_b (const bspline::BsplineSpace<Tb>&): The second space.
///     face_b (long long): Face id on the second space.
///     axis_map (std::span<const long long>): Patch-b axis per tangential axis of face a.
///     flips (const std::vector<bool>&): Whether each of those axis pairs runs opposed.
///
/// Returns:
///     MatchedCps: As the overload over basis counts.
///
/// Raises:
///     value_error: As the overload over basis counts.
///     capacity_error: As the overload over basis counts.
template <StorageFloat Ta, StorageFloat Tb>
[[nodiscard]] MatchedCps match_face_cps(const bspline::BsplineSpace<Ta>& space_a, long long face_a,
                                        const bspline::BsplineSpace<Tb>& space_b, long long face_b,
                                        std::span<const long long> axis_map,
                                        const std::vector<bool>& flips) {
  const std::vector<std::size_t> n_a = detail::num_basis_of(space_a);
  const std::vector<std::size_t> n_b = detail::num_basis_of(space_b);
  return match_face_cps(n_a, face_a, n_b, face_b, axis_map, flips);
}

}  // namespace pantr::multipatch
