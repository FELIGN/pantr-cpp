/// @file multi_patch.hpp
/// @brief `MultiPatch`: a validated collection of B-spline patches and the interfaces joining them.
#pragma once

#include "pantr/bspline/bspline.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/multipatch/detect.hpp"
#include "pantr/multipatch/interface.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::multipatch {

/// A set of B-spline patches together with their conforming interfaces.
///
/// The patches must agree on everything that would make joining them meaningless otherwise:
/// parametric dimension, physical rank, and whether they are rational (the storage type is the
/// class's). Interfaces are detected on construction unless supplied, and supplied ones are
/// verified rather than trusted. An immutable value.
///
/// Attributes:
///     patches_ (std::vector<bspline::Bspline<T>>): The patches, in the order given.
///     interfaces_ (std::vector<Interface>): The interfaces joining them.
template <StorageFloat T>
class MultiPatch {
 public:
  /// Validate the patches and establish the interfaces.
  ///
  /// Args:
  ///     patches (std::vector<bspline::Bspline<T>>): At least one patch. All must share parametric
  ///         dimension, physical rank and rationality.
  ///     interfaces (std::optional<std::vector<Interface>>): Interfaces to use. Each is verified
  ///         against the geometry. Detected automatically when absent.
  ///     tol (std::optional<double>): Relative tolerance override for detection and verification.
  ///         Defaults to the strict preset of `T`.
  ///
  /// Raises:
  ///     value_error: No patches are given, the patches disagree on dimension, rank or
  ///         rationality, an interface refers to a patch that does not exist or implies another
  ///         dimension, a supplied interface does not hold for the geometry, or the detection or
  ///         verification refuses the patches (see `detect_interfaces`).
  MultiPatch(std::vector<bspline::Bspline<T>> patches,
             std::optional<std::vector<Interface>> interfaces, std::optional<double> tol)
      : patches_(std::move(patches)) {
    if (patches_.empty()) {
      throw value_error("A MultiPatch needs at least one patch");
    }
    const bspline::Bspline<T>& first = patches_.front();
    for (std::size_t index = 1; index < patches_.size(); ++index) {
      const bspline::Bspline<T>& patch = patches_[index];
      check_shared("dim", first.dim(), patch.dim(), index);
      check_shared("rank", first.rank(), patch.rank(), index);
      check_shared("is_rational", first.rational(), patch.rational(), index);
    }

    const std::span<const bspline::Bspline<T>> view(patches_);
    if (!interfaces) {
      interfaces_ = detect_interfaces<T>(view, tol);
      return;
    }
    const std::size_t n_patches = patches_.size();
    for (const Interface& iface : *interfaces) {
      const std::pair<const char*, long long> sides[] = {{"patch_a", iface.patch_a()},
                                                         {"patch_b", iface.patch_b()}};
      for (const auto& [side, index] : sides) {
        if (!(0 <= index && static_cast<unsigned long long>(index) < n_patches)) {
          throw value_error("Interface " + std::string(side) + "=" + std::to_string(index) +
                            " is out of range for " + std::to_string(n_patches) + " patches");
        }
      }
      if (iface.dim() != first.dim()) {
        throw value_error("Interface implies dim=" + std::to_string(iface.dim()) +
                          " but the patches are " + std::to_string(first.dim()) + "D");
      }
      if (!verify_interface<T>(view, iface, tol)) {
        throw value_error("Interface does not hold for the given geometry: " + iface.repr() +
                          ". The matched control points or the tangential knot vectors disagree.");
      }
    }
    interfaces_ = std::move(*interfaces);
  }

  /// Get the patches.
  ///
  /// Returns:
  ///     const std::vector<bspline::Bspline<T>>&: The patches, in the order given.
  [[nodiscard]] const std::vector<bspline::Bspline<T>>& patches() const noexcept {
    return patches_;
  }

  /// Get the interfaces joining the patches.
  ///
  /// Returns:
  ///     const std::vector<Interface>&: The supplied or detected interfaces.
  [[nodiscard]] const std::vector<Interface>& interfaces() const noexcept { return interfaces_; }

  /// Get the shared parametric dimension of the patches.
  ///
  /// Returns:
  ///     std::size_t: Parametric dimension.
  [[nodiscard]] std::size_t dim() const noexcept { return patches_.front().dim(); }

  /// Get the number of patches.
  ///
  /// Returns:
  ///     std::size_t: Patch count, at least 1.
  [[nodiscard]] std::size_t size() const noexcept { return patches_.size(); }

 private:
  /// Refuse a patch whose attribute differs from the first patch's.
  template <class V>
  static void check_shared(const char* attribute, V expected, V actual, std::size_t index) {
    if (expected != actual) {
      throw value_error(std::string("All patches must share ") + attribute + "; patch 0 has " +
                        spell(expected) + " but patch " + std::to_string(index) + " has " +
                        spell(actual));
    }
  }

  static std::string spell(std::size_t v) { return std::to_string(v); }
  static std::string spell(bool v) { return v ? "True" : "False"; }

  std::vector<bspline::Bspline<T>> patches_;
  std::vector<Interface> interfaces_;
};

}  // namespace pantr::multipatch
