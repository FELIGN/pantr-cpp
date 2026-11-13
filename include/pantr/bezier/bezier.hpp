/// @file bezier.hpp
/// @brief `Bezier`: a tensor-product Bernstein polynomial (or rational) map on `[0, 1]^dim`.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bezier {

namespace detail {

/// The shared representation of a `Bezier`: its control points and how to read them.
///
/// Attributes:
///     data (std::shared_ptr<const T[]>): The control points, row-major over `shape`. Possibly
///         shared with another immutable value (an aliasing pointer into its buffer).
///     shape (std::vector<std::size_t>): `(degree[0] + 1, ..., degree[dim - 1] + 1, cp_size)`.
///     rational (bool): Whether the last coordinate is a homogeneous weight.
template <StorageFloat T>
struct BezierRep {
  std::shared_ptr<const T[]> data;
  std::vector<std::size_t> shape;
  bool rational;
};

}  // namespace detail

/// A Bézier curve, surface or volume: control points over `[0, 1]^dim` in the Bernstein basis.
///
/// An immutable value (design §2.2): the control points are fixed at construction and shared,
/// read-only, by every copy (a copy is a reference-count increment). Every operation constructs a
/// new value. The buffer constructor is the one way to break that: a caller that hands it memory
/// someone else may write makes a value that follows those writes, and a write during an
/// operation is a data race. A conversion from a B-spline that shares its control points hands it
/// an aliasing pointer into the B-spline's immutable buffer, which the Bézier then keeps alive.
///
/// The control points have extents `(degree[0] + 1, ..., degree[dim - 1] + 1, cp_size)`, row-major.
/// For a rational Bézier the last of the `cp_size` coordinates is the weight and the others are
/// weighted (homogeneous) coordinates, so `rank() == cp_size() - 1`; otherwise `rank() ==
/// cp_size()`.
///
/// Attributes:
///     rep_ (std::shared_ptr<const detail::BezierRep<T>>): The shared representation.
template <StorageFloat T>
class Bezier {
 public:
  /// The shared representation type.
  using Rep = detail::BezierRep<T>;

  /// Build a Bézier by copying its control points.
  ///
  /// Args:
  ///     control_points (std::span<const T>): Row-major over `shape`.
  ///     shape (std::vector<std::size_t>): `(n_0, ..., n_{dim-1}, cp_size)`, at least two extents.
  ///     rational (bool): Whether the last coordinate is a homogeneous weight.
  ///
  /// Raises:
  ///     value_error: Fewer than two extents, a parametric extent below 1, a rank below 1, or a
  ///         `control_points` size that does not match `shape`.
  ///     capacity_error: The product of the extents overflows `std::size_t`.
  Bezier(std::span<const T> control_points, std::vector<std::size_t> shape, bool rational)
      // `shape` is copied, not moved: the order in which the two arguments are evaluated is
      // unspecified, and a moved-from shape would skip `copy_of`'s size check.
      : Bezier(copy_of(control_points, shape), shape, rational) {}

  /// Build a Bézier over an existing buffer, without copying (a result, or a conversion that
  /// shares another value's control points). The buffer must not be written while the value
  /// lives, or the value is not immutable (see the class note).
  ///
  /// Args:
  ///     data (std::shared_ptr<const T[]>): Row-major over `shape`; must hold
  ///     `checked_product(shape)`
  ///         elements, which cannot be checked here.
  ///     shape (std::vector<std::size_t>): As in the copying constructor.
  ///     rational (bool): Whether the last coordinate is a homogeneous weight.
  ///
  /// Raises:
  ///     value_error: As in the copying constructor, except the size check.
  Bezier(std::shared_ptr<const T[]> data, std::vector<std::size_t> shape, bool rational)
      : rep_(make_rep(std::move(data), std::move(shape), rational)) {}

  /// Build a Bézier that adopts an array's buffer (no copy).
  ///
  /// Args:
  ///     control_points (NdArray<T>&&): Consumed; its extents are the shape.
  ///     rational (bool): Whether the last coordinate is a homogeneous weight.
  ///
  /// Raises:
  ///     value_error: As in the copying constructor.
  Bezier(NdArray<T>&& control_points, bool rational)
      : Bezier(adopt(std::move(control_points)), rational) {}

  /// Get the parametric dimension.
  ///
  /// Returns:
  ///     std::size_t: The number of parametric directions, at least 1.
  [[nodiscard]] std::size_t dim() const noexcept { return rep_->shape.size() - 1; }

  /// Get the degree in one direction.
  ///
  /// Args:
  ///     d (std::size_t): The direction, `< dim()` (unchecked).
  ///
  /// Returns:
  ///     std::size_t: `shape()[d] - 1`.
  [[nodiscard]] std::size_t degree(std::size_t d) const noexcept { return rep_->shape[d] - 1; }

  /// Get the degrees per direction.
  ///
  /// Returns:
  ///     std::vector<std::size_t>: One degree per parametric direction.
  [[nodiscard]] std::vector<std::size_t> degrees() const {
    std::vector<std::size_t> out(dim());
    for (std::size_t d = 0; d < out.size(); ++d) {
      out[d] = degree(d);
    }
    return out;
  }

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
  ///     bool: Whether the Bézier is rational.
  [[nodiscard]] bool rational() const noexcept { return rep_->rational; }

  /// Get the extents of the control points.
  ///
  /// Returns:
  ///     std::span<const std::size_t>: `(degree[0] + 1, ..., cp_size)`.
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

  /// Get the shared representation, for an owner that must keep the control points alive (a
  /// view handed to another language).
  ///
  /// Returns:
  ///     const std::shared_ptr<const Rep>&: The representation.
  [[nodiscard]] const std::shared_ptr<const Rep>& rep() const noexcept { return rep_; }

  /// Report whether two values share one representation (the same control-point buffer, shape
  /// and rationality), which is what lets an operation that changes nothing return its input.
  ///
  /// Args:
  ///     other (const Bezier&): The other value.
  ///
  /// Returns:
  ///     bool: Whether the representations are the same object.
  [[nodiscard]] bool same_rep(const Bezier& other) const noexcept { return rep_ == other.rep_; }

 private:
  struct Adopted {
    std::shared_ptr<const T[]> data;
    std::vector<std::size_t> shape;
  };

  Bezier(Adopted a, bool rational) : Bezier(std::move(a.data), std::move(a.shape), rational) {}

  static Adopted adopt(NdArray<T>&& a) {
    std::vector<std::size_t> shape(a.shape().begin(), a.shape().end());
    return {std::shared_ptr<const T[]>(std::move(a).into_buffer()), std::move(shape)};
  }

  static std::shared_ptr<const T[]> copy_of(std::span<const T> src,
                                            const std::vector<std::size_t>& shape) {
    if (shape.size() >= 2 && checked_product(shape) != src.size()) {
      throw value_error("control points hold " + std::to_string(src.size()) +
                        " values, but the shape " + py_repr(std::span<const std::size_t>(shape)) +
                        " needs " + std::to_string(checked_product(shape)));
    }
    std::shared_ptr<T[]> buf(new T[src.size()]);
    std::copy(src.begin(), src.end(), buf.get());
    return buf;
  }

  static std::shared_ptr<const Rep> make_rep(std::shared_ptr<const T[]> data,
                                             std::vector<std::size_t> shape, bool rational) {
    if (shape.size() < 2) {
      throw value_error("Control points must be at least 2D: (*degrees_plus_1, cp_size).");
    }
    for (std::size_t d = 0; d + 1 < shape.size(); ++d) {
      if (shape[d] < 1) {
        throw value_error("Control points must have at least 1 entry in parametric direction " +
                          std::to_string(d) + ", got " + std::to_string(shape[d]) + ".");
      }
    }
    static_cast<void>(checked_product(shape));
    const std::size_t cp = shape.back();
    const long long rank = static_cast<long long>(cp) - (rational ? 1 : 0);
    if (rank <= 0) {
      throw value_error("The Bézier must have at least rank one. Got rank " + std::to_string(rank) +
                        ".");
    }
    return std::make_shared<const Rep>(Rep{std::move(data), std::move(shape), rational});
  }

  std::shared_ptr<const Rep> rep_;
};

}  // namespace pantr::bezier
