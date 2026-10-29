/// @file points_lattice.hpp
/// @brief `PointsLattice`: a tensor-product grid of points, one coordinate array per direction.
#pragma once

#include "pantr/basis/lagrange_variant.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <concepts>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace pantr {

/// Index order of a flattened tensor product: `C` makes the last index vary fastest, `F` the
/// first. The values are the wire values the Python binding passes.
enum class Order : int {
  C = 0,  ///< Row-major: the last index varies fastest.
  F = 1,  ///< Column-major: the first index varies fastest.
};

}  // namespace pantr

namespace pantr::quad {

/// A tensor-product grid of points: one non-empty coordinate array per direction.
///
/// An immutable value: the coordinates are copied in at construction and shared, read-only, by
/// every copy of the value (a copy is a reference-count increment).
///
/// Attributes:
///     rep_ (std::shared_ptr<const std::vector<std::vector<T>>>): The coordinates per direction.
template <StorageFloat T>
class PointsLattice {
 public:
  /// The shared representation: one coordinate vector per direction.
  using Rep = std::vector<std::vector<T>>;

  /// Build a lattice from its coordinate arrays.
  ///
  /// Args:
  ///     pts_per_dir (std::vector<std::vector<T>>): One coordinate array per direction.
  ///
  /// Raises:
  ///     value_error: No direction, or a direction with no point.
  ///     capacity_error: The number of points overflows `std::size_t`.
  explicit PointsLattice(std::vector<std::vector<T>> pts_per_dir)
      : rep_(std::make_shared<const Rep>(std::move(pts_per_dir))) {
    if (rep_->empty()) {
      throw value_error("Points lattice must have at least 1 dimension");
    }
    std::vector<std::size_t> counts;
    counts.reserve(rep_->size());
    for (const auto& pts : *rep_) {
      if (pts.empty()) {
        throw value_error("All points must have at least 1 point");
      }
      counts.push_back(pts.size());
    }
    static_cast<void>(checked_product(counts));  // num_points() then cannot overflow
  }

  /// Get the number of directions.
  ///
  /// Returns:
  ///     std::size_t: The dimension, at least 1.
  [[nodiscard]] std::size_t dim() const noexcept { return rep_->size(); }

  /// Get the coordinates of one direction (a numeric view into the shared representation).
  ///
  /// Args:
  ///     d (std::size_t): The direction, `< dim()`.
  ///
  /// Returns:
  ///     std::span<const T>: The coordinates, valid while any copy of the lattice lives.
  ///
  /// Raises:
  ///     index_error: `d >= dim()`.
  [[nodiscard]] std::span<const T> pts(std::size_t d) const {
    if (d >= dim()) {
      throw index_error("direction " + std::to_string(d) + " is out of range for a lattice of " +
                        "dimension " + std::to_string(dim()));
    }
    return (*rep_)[d];
  }

  /// Get the total number of points, the product of the per-direction counts.
  ///
  /// Returns:
  ///     std::size_t: The number of points.
  [[nodiscard]] std::size_t num_points() const noexcept {
    std::size_t n = 1;
    for (const auto& pts : *rep_) {
      n *= pts.size();
    }
    return n;
  }

  /// Get every point of the lattice.
  ///
  /// Args:
  ///     order (Order): `C` (the last direction's index varies fastest) or `F` (the first's).
  ///
  /// Returns:
  ///     NdArray<T>: Extents `(num_points(), dim())`.
  [[nodiscard]] NdArray<T> all_points(Order order) const {
    const std::size_t dimension = dim();
    const std::size_t n = num_points();
    NdArray<T> out({n, dimension}, for_overwrite);
    T* data = out.data();
    std::vector<std::size_t> idx(dimension, 0);
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t d = 0; d < dimension; ++d) {
        data[p * dimension + d] = (*rep_)[d][idx[d]];
      }
      // Advance the multi-index, fastest direction first.
      for (std::size_t s = 0; s < dimension; ++s) {
        const std::size_t d = order == Order::C ? dimension - 1 - s : s;
        if (++idx[d] < (*rep_)[d].size()) {
          break;
        }
        idx[d] = 0;
      }
    }
    return out;
  }

  /// Get the shared representation, for an owner that must keep the coordinates alive (a view
  /// handed to another language).
  ///
  /// Returns:
  ///     const std::shared_ptr<const Rep>&: The representation.
  [[nodiscard]] const std::shared_ptr<const Rep>& rep() const noexcept { return rep_; }

 private:
  std::shared_ptr<const Rep> rep_;
};

/// Build the lattice of Lagrange nodes of a family, one direction per count.
///
/// Args:
///     variant (basis::LagrangeVariant): The node family.
///     n_pts_per_dir (std::span<const long long>): Nodes per direction, each at least 1 (2 for
///         the Lobatto families).
///
/// Returns:
///     PointsLattice<T>: The lattice.
///
/// Raises:
///     value_error: A count below 1, a count below the family's minimum, or no direction.
template <StorageFloat T>
[[nodiscard]] PointsLattice<T> lagrange_points_lattice(basis::LagrangeVariant variant,
                                                       std::span<const long long> n_pts_per_dir) {
  for (const long long n : n_pts_per_dir) {
    if (n < 1) {
      throw value_error("All number of points must be at least 1");
    }
  }
  std::vector<std::vector<T>> pts;
  pts.reserve(n_pts_per_dir.size());
  for (const long long n : n_pts_per_dir) {
    const NdArray<T> nodes = basis::lagrange_points<T>(variant, n);
    pts.emplace_back(nodes.data(), nodes.data() + nodes.size());
  }
  return PointsLattice<T>(std::move(pts));
}

}  // namespace pantr::quad
