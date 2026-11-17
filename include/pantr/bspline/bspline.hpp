/// @file bspline.hpp
/// @brief `Bspline`: a tensor-product B-spline (or NURBS) map, a space with its control points.
#pragma once

#include "pantr/bspline/space.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bspline {

namespace detail {

/// The shared representation of a `Bspline`: its space and its control points.
///
/// Attributes:
///     space (BsplineSpace<T>): The tensor-product space (held by value: a vector of handles,
///         each sharing its direction's representation).
///     data (std::shared_ptr<const T[]>): The control points, row-major over `shape`. Possibly
///         shared with another immutable value.
///     shape (std::vector<std::size_t>): `(num_basis[0], ..., num_basis[dim - 1], cp_size)`.
///     rational (bool): Whether the last coordinate is a homogeneous weight.
template <StorageFloat T>
struct BsplineRep {
  BsplineSpace<T> space;
  std::shared_ptr<const T[]> data;
  std::vector<std::size_t> shape;
  bool rational;
};

}  // namespace detail

/// A B-spline curve, surface or volume: a `BsplineSpace` with one control point per basis
/// function.
///
/// An immutable value (design §2.2): the space and the control points are fixed at construction
/// and shared, read-only, by every copy (a copy is a reference-count increment). Every operation
/// constructs a new value. The shared-buffer constructor is the one way to break that: a caller
/// that hands it memory someone else may write makes a value that follows those writes, and a
/// write during an operation is a data race.
///
/// The control points have extents `(num_basis[0], ..., num_basis[dim - 1], cp_size)`, row-major.
/// For a rational B-spline the last of the `cp_size` coordinates is the weight and the others are
/// weighted (homogeneous) coordinates, so `rank() == cp_size() - 1`; otherwise `rank() ==
/// cp_size()`. For a periodic direction the control points are the stored ones: the function
/// reads them modulo `num_basis` (the ghost functions repeat the first ones).
///
/// Attributes:
///     rep_ (std::shared_ptr<const detail::BsplineRep<T>>): The shared representation.
template <StorageFloat T>
class Bspline {
 public:
  /// The shared representation type.
  using Rep = detail::BsplineRep<T>;

  /// Build a B-spline by copying its control points.
  ///
  /// Args:
  ///     space (BsplineSpace<T>): The space.
  ///     control_points (std::span<const T>): `num_total_basis * cp_size` values, row-major over
  ///         `(num_basis[0], ..., num_basis[dim - 1], cp_size)`; `cp_size` is their number over
  ///         the number of basis functions.
  ///     rational (bool): Whether the last coordinate is a homogeneous weight.
  ///
  /// Raises:
  ///     value_error: The number of values is not a multiple of the number of basis functions, or
  ///         the rank is below one.
  Bspline(BsplineSpace<T> space, std::span<const T> control_points, bool rational)
      : rep_(make_rep(std::move(space), control_points, rational)) {}

  /// Build a B-spline over an existing buffer, without copying (a result, or a conversion that
  /// shares another value's control points). The buffer must not be written while the value
  /// lives, or the value is not immutable (see the class note).
  ///
  /// Args:
  ///     space (BsplineSpace<T>): The space.
  ///     data (std::shared_ptr<const T[]>): `num_total_basis * cp_size` values, row-major. A
  ///         pointer cannot say how many it points to, so `count` does.
  ///     count (std::size_t): The number of values `data` holds.
  ///     cp_size (std::size_t): The coordinates per control point, the weight included.
  ///     rational (bool): Whether the last coordinate is a homogeneous weight.
  ///
  /// Raises:
  ///     value_error: `count` is not `num_total_basis * cp_size` (checked first); the rank is
  ///         below one.
  Bspline(BsplineSpace<T> space, std::shared_ptr<const T[]> data, std::size_t count,
          std::size_t cp_size, bool rational)
      : rep_(make_rep(std::move(space), std::move(data), count, cp_size, rational)) {}

  /// Build a B-spline that adopts an array's buffer (no copy). Only the array's number of values
  /// is read, as if it were flat.
  ///
  /// Args:
  ///     space (BsplineSpace<T>): The space.
  ///     control_points (NdArray<T>&&): Consumed.
  ///     rational (bool): Whether the last coordinate is a homogeneous weight.
  ///
  /// Raises:
  ///     value_error: As the copying constructor.
  Bspline(BsplineSpace<T> space, NdArray<T>&& control_points, bool rational)
      : rep_(make_rep(std::move(space), std::move(control_points), rational)) {}

  /// Get the number of coordinates per control point that `count` values make in `space`.
  ///
  /// Args:
  ///     space (const BsplineSpace<T>&): The space.
  ///     count (std::size_t): The number of control-point values.
  ///
  /// Returns:
  ///     std::size_t: `count / space.num_total_basis()`.
  ///
  /// Raises:
  ///     value_error: `count` is not a multiple of the number of basis functions.
  [[nodiscard]] static std::size_t cp_size_for(const BsplineSpace<T>& space, std::size_t count) {
    const std::size_t total = num_total_basis_of(space);
    if (count % total != 0) {
      throw value_error(
          "The number of control points must be a multiple of the number of basis functions."
          "Got " +
          std::to_string(count) + " control points and " + std::to_string(total) +
          " basis functions.");
    }
    return count / total;
  }

  /// Get the space (a copy of its handles: each direction shares its representation).
  ///
  /// Returns:
  ///     BsplineSpace<T>: The space.
  [[nodiscard]] BsplineSpace<T> space() const { return rep_->space; }

  /// Get the parametric dimension.
  ///
  /// Returns:
  ///     std::size_t: The number of parametric directions, at least 1.
  [[nodiscard]] std::size_t dim() const noexcept { return rep_->shape.size() - 1; }

  /// Get the degrees per direction.
  ///
  /// Returns:
  ///     std::vector<std::size_t>: One degree per parametric direction.
  [[nodiscard]] std::vector<std::size_t> degrees() const {
    std::vector<std::size_t> out;
    out.reserve(dim());
    for (const auto& s : rep_->space.spaces()) {
      out.push_back(static_cast<std::size_t>(s.degree()));
    }
    return out;
  }

  /// Get the number of basis functions in one direction.
  ///
  /// Args:
  ///     d (std::size_t): The direction, `< dim()` (unchecked).
  ///
  /// Returns:
  ///     std::size_t: `shape()[d]`.
  [[nodiscard]] std::size_t num_basis(std::size_t d) const noexcept { return rep_->shape[d]; }

  /// Get the number of coordinates per control point, the weight included.
  ///
  /// Returns:
  ///     std::size_t: The last extent.
  [[nodiscard]] std::size_t cp_size() const noexcept { return rep_->shape.back(); }

  /// Get the output rank (the weight excluded).
  ///
  /// Returns:
  ///     std::size_t: `cp_size() - 1` when rational, else `cp_size()`; at least 1.
  [[nodiscard]] std::size_t rank() const noexcept { return cp_size() - (rational() ? 1 : 0); }

  /// Report whether the last coordinate is a homogeneous weight.
  ///
  /// Returns:
  ///     bool: Whether the B-spline is rational.
  [[nodiscard]] bool rational() const noexcept { return rep_->rational; }

  /// Get the extents of the control points.
  ///
  /// Returns:
  ///     std::span<const std::size_t>: `(num_basis[0], ..., cp_size)`.
  [[nodiscard]] std::span<const std::size_t> shape() const noexcept { return rep_->shape; }

  /// Get the number of control-point coordinates, the product of the extents.
  ///
  /// Returns:
  ///     std::size_t: The size.
  [[nodiscard]] std::size_t size() const noexcept {
    std::size_t n = 1;
    for (const std::size_t e : rep_->shape) {
      n *= e;
    }
    return n;
  }

  /// Get the control points (a numeric view into the shared representation).
  ///
  /// Returns:
  ///     std::span<const T>: Row-major over `shape()`, valid while any copy of the value lives.
  [[nodiscard]] std::span<const T> control_points() const noexcept {
    return {rep_->data.get(), size()};
  }

  /// Get the shared representation, for an owner that must keep the control points alive (a view
  /// handed to another language).
  ///
  /// Returns:
  ///     const std::shared_ptr<const Rep>&: The representation.
  [[nodiscard]] const std::shared_ptr<const Rep>& rep() const noexcept { return rep_; }

  /// Report whether two values share one representation.
  ///
  /// Args:
  ///     other (const Bspline&): The other value.
  ///
  /// Returns:
  ///     bool: Whether the representations are the same object.
  [[nodiscard]] bool same_rep(const Bspline& other) const noexcept { return rep_ == other.rep_; }

 private:
  static std::size_t num_total_basis_of(const BsplineSpace<T>& space) {
    std::size_t total = 1;
    for (const auto& s : space.spaces()) {
      total *= static_cast<std::size_t>(s.num_basis());
    }
    return total;
  }

  static std::shared_ptr<const Rep> make_rep(BsplineSpace<T> space, NdArray<T>&& a, bool rational) {
    const std::size_t cp = cp_size_for(space, a.size());
    const std::size_t count = a.size();
    return make_rep(std::move(space), std::shared_ptr<const T[]>(std::move(a).into_buffer()), count,
                    cp, rational);
  }

  static std::shared_ptr<const Rep> make_rep(BsplineSpace<T> space, std::span<const T> src,
                                             bool rational) {
    const std::size_t cp = cp_size_for(space, src.size());
    std::shared_ptr<T[]> buf(new T[src.size()]);
    std::copy(src.begin(), src.end(), buf.get());
    return make_rep(std::move(space), std::shared_ptr<const T[]>(std::move(buf)), src.size(), cp,
                    rational);
  }

  static std::shared_ptr<const Rep> make_rep(BsplineSpace<T> space, std::shared_ptr<const T[]> data,
                                             std::size_t count, std::size_t cp_size,
                                             bool rational) {
    const std::size_t total = num_total_basis_of(space);
    // `total * cp_size` is formed only when it fits: a product past size_t matches no count.
    const bool fits = cp_size == 0 || total <= (std::numeric_limits<std::size_t>::max)() / cp_size;
    if (!fits || total * cp_size != count) {
      throw value_error("The buffer holds " + std::to_string(count) +
                        " control-point values, but the space has " + std::to_string(total) +
                        " basis functions of " + std::to_string(cp_size) + " coordinates each.");
    }
    const long long rank = static_cast<long long>(cp_size) - (rational ? 1 : 0);
    if (rank <= 0) {
      throw value_error("The B-spline must have at least rank one. Got rank " +
                        std::to_string(rank));
    }
    std::vector<std::size_t> shape;
    shape.reserve(space.dim() + 1);
    for (const auto& s : space.spaces()) {
      shape.push_back(static_cast<std::size_t>(s.num_basis()));
    }
    shape.push_back(cp_size);
    return std::make_shared<const Rep>(
        Rep{std::move(space), std::move(data), std::move(shape), rational});
  }

  std::shared_ptr<const Rep> rep_;
};

}  // namespace pantr::bspline
