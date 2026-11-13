/// @file interface.hpp
/// @brief `Interface`: a validated record of one conforming contact between two patch faces.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::multipatch {

/// Split a face id into its parametric axis and side.
///
/// Faces are numbered `2 * axis + side` with `side == 0` the low-parameter face and `side == 1`
/// the high one.
///
/// Args:
///     face (long long): Face id, at least 0.
///
/// Returns:
///     std::pair<long long, long long>: `(axis, side)`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): a negative id is divided
///     towards zero, not floored as Python's `divmod` does.
[[nodiscard]] inline std::pair<long long, long long> face_axis_side(long long face) noexcept {
  return {face / 2, face % 2};
}

/// Get the patch axes tangential to a face, in increasing order.
///
/// Args:
///     face (long long): Face id, in `[0, 2 * dim)`.
///     dim (std::size_t): Parametric dimension of the patch.
///
/// Returns:
///     std::vector<long long>: The `dim - 1` axes other than the face's own axis.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
[[nodiscard]] inline std::vector<long long> tangential_axes(long long face, std::size_t dim) {
  const long long own = face_axis_side(face).first;
  std::vector<long long> axes;
  axes.reserve(dim);
  for (long long k = 0; k < static_cast<long long>(dim); ++k) {
    if (k != own) {
      axes.push_back(k);
    }
  }
  return axes;
}

namespace detail {

/// Spell a list of integers as Python spells the tuple: `()`, `(1,)`, `(1, 2)`.
[[nodiscard]] inline std::string tuple_repr(std::span<const long long> values) {
  std::string out = "(";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out += ", ";
    }
    out += std::to_string(values[i]);
  }
  if (values.size() == 1) {
    out += ',';
  }
  out += ')';
  return out;
}

/// Spell a list of booleans as Python spells the tuple: `()`, `(True,)`, `(True, False)`.
[[nodiscard]] inline std::string tuple_repr(const std::vector<bool>& values) {
  std::string out = "(";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out += ", ";
    }
    out += values[i] ? "True" : "False";
  }
  if (values.size() == 1) {
    out += ',';
  }
  out += ')';
  return out;
}

/// Refuse an axis map that is not a permutation of the tangential axes of a face.
///
/// Args:
///     axis_map (std::span<const long long>): The map to check.
///     face_b (long long): Face id on the second patch, in `[0, 2 * dim)`.
///     dim (std::size_t): Parametric dimension.
///
/// Raises:
///     value_error: `axis_map` is not a permutation of `face_b`'s tangential axes (a repeated
///         entry included).
///
/// Note:
///     `face_b` is assumed to be in range (no validation performed).
inline void check_axis_map_permutation(std::span<const long long> axis_map, long long face_b,
                                       std::size_t dim) {
  const std::vector<long long> expected = tangential_axes(face_b, dim);
  std::vector<long long> sorted(axis_map.begin(), axis_map.end());
  std::sort(sorted.begin(), sorted.end());
  // `expected` is increasing and has no repeat, so equality of the sorted lists is the equality of
  // the two sets together with the absence of a repeated entry in `axis_map`.
  if (sorted != expected) {
    throw value_error("axis_map=" + tuple_repr(axis_map) +
                      " must be a permutation of the tangential axes " + tuple_repr(expected) +
                      " of face_b=" + std::to_string(face_b));
  }
}

}  // namespace detail

/// A conforming contact between two patch faces: which faces meet, how their tangential axes
/// correspond, and which of those axes run in opposite directions.
///
/// An immutable value, validated at construction. The two sides are in canonical order,
/// `(patch_a, face_a)` lexicographically smaller than `(patch_b, face_b)`; a patch may meet itself
/// on two different faces, which is how a closed ring is expressed.
///
/// `axis_map` and `flips` describe the reparametrization between the two faces. Entry `k` refers
/// to the `k`-th tangential axis of `face_a` taken in increasing-axis order; `axis_map[k]` is the
/// patch axis of `patch_b` it corresponds to, and `flips[k]` says whether the two run in opposite
/// directions.
///
/// Attributes:
///     patch_a_ (long long): Index of the first patch.
///     face_a_ (long long): Face id on `patch_a`, as `2 * axis + side`.
///     patch_b_ (long long): Index of the second patch.
///     face_b_ (long long): Face id on `patch_b`.
///     axis_map_ (std::vector<long long>): Length `dim - 1`; the patch-`b` axis corresponding to
///         each tangential axis of `face_a`. A permutation of the tangential axes of `face_b`.
///     flips_ (std::vector<bool>): Length `dim - 1`; true where that tangential axis of `face_a`
///         runs against its partner's direction.
class Interface {
 public:
  /// Build an interface, checking everything the record can check without the patches.
  ///
  /// Args:
  ///     patch_a (long long): Index of the first patch.
  ///     face_a (long long): Face id on `patch_a`.
  ///     patch_b (long long): Index of the second patch.
  ///     face_b (long long): Face id on `patch_b`.
  ///     axis_map (std::vector<long long>): Patch-`b` axis per tangential axis of `face_a`.
  ///     flips (std::vector<bool>): Whether each of those axis pairs runs opposed.
  ///
  /// Raises:
  ///     value_error: `flips` and `axis_map` differ in length; a face id is out of range for the
  ///         dimension `axis_map.size() + 1`; the interface joins a face to itself; `axis_map` is
  ///         not a permutation of `face_b`'s tangential axes; or the two sides are not in
  ///         canonical order (checked in that order).
  Interface(long long patch_a, long long face_a, long long patch_b, long long face_b,
            std::vector<long long> axis_map, std::vector<bool> flips)
      : patch_a_(patch_a),
        face_a_(face_a),
        patch_b_(patch_b),
        face_b_(face_b),
        axis_map_(std::move(axis_map)),
        flips_(std::move(flips)) {
    const std::size_t dim = axis_map_.size() + 1;
    if (flips_.size() != axis_map_.size()) {
      throw value_error("flips and axis_map must have equal length; got " +
                        std::to_string(flips_.size()) + " and " + std::to_string(axis_map_.size()));
    }
    const long long n_faces = 2 * static_cast<long long>(dim);
    const std::pair<const char*, long long> faces[] = {{"face_a", face_a_}, {"face_b", face_b_}};
    for (const auto& [name, face] : faces) {
      if (!(0 <= face && face < n_faces)) {
        throw value_error(std::string(name) + "=" + std::to_string(face) +
                          " out of range for dim=" + std::to_string(dim) +
                          "; expected 0 <= face < " + std::to_string(n_faces));
      }
    }
    if (patch_a_ == patch_b_ && face_a_ == face_b_) {
      throw value_error("An interface cannot join a face to itself (patch " +
                        std::to_string(patch_a_) + ", face " + std::to_string(face_a_) + ")");
    }
    detail::check_axis_map_permutation(axis_map_, face_b_, dim);
    if (!(patch_a_ < patch_b_ || (patch_a_ == patch_b_ && face_a_ < face_b_))) {
      throw value_error("Interface sides must be in canonical order: (" + std::to_string(patch_a_) +
                        ", " + std::to_string(face_a_) +
                        ") must be lexicographically smaller than (" + std::to_string(patch_b_) +
                        ", " + std::to_string(face_b_) + ")");
    }
  }

  /// Get the index of the first patch.
  [[nodiscard]] long long patch_a() const noexcept { return patch_a_; }
  /// Get the face id on the first patch.
  [[nodiscard]] long long face_a() const noexcept { return face_a_; }
  /// Get the index of the second patch.
  [[nodiscard]] long long patch_b() const noexcept { return patch_b_; }
  /// Get the face id on the second patch.
  [[nodiscard]] long long face_b() const noexcept { return face_b_; }
  /// Get the patch-`b` axis per tangential axis of `face_a`.
  [[nodiscard]] const std::vector<long long>& axis_map() const noexcept { return axis_map_; }
  /// Get whether each of those axis pairs runs opposed.
  [[nodiscard]] const std::vector<bool>& flips() const noexcept { return flips_; }

  /// Get the parametric dimension of the patches this interface joins.
  ///
  /// Returns:
  ///     std::size_t: `axis_map().size() + 1`.
  [[nodiscard]] std::size_t dim() const noexcept { return axis_map_.size() + 1; }

  /// Spell the interface as the Python dataclass does.
  ///
  /// Returns:
  ///     std::string: E.g. `Interface(patch_a=0, face_a=1, patch_b=1, face_b=0, axis_map=(1,),
  ///     flips=(False,))`.
  [[nodiscard]] std::string repr() const {
    return "Interface(patch_a=" + std::to_string(patch_a_) + ", face_a=" + std::to_string(face_a_) +
           ", patch_b=" + std::to_string(patch_b_) + ", face_b=" + std::to_string(face_b_) +
           ", axis_map=" + detail::tuple_repr(axis_map_) + ", flips=" + detail::tuple_repr(flips_) +
           ")";
  }

  /// Compare two interfaces field by field.
  [[nodiscard]] bool operator==(const Interface&) const = default;

 private:
  long long patch_a_;
  long long face_a_;
  long long patch_b_;
  long long face_b_;
  std::vector<long long> axis_map_;
  std::vector<bool> flips_;
};

}  // namespace pantr::multipatch
