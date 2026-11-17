/// @file aabb.hpp
/// @brief `AABB`: an axis-aligned bounding box in any number of dimensions.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/ndarray.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::geometry {

namespace detail {

/// Spell a list of doubles as Python spells `repr(list_of_floats)`: `[0.0, inf]`.
inline std::string py_repr_list(std::span<const double> xs) {
  std::string out = "[";
  for (std::size_t i = 0; i < xs.size(); ++i) {
    if (i > 0) {
      out += ", ";
    }
    out += py_repr(xs[i]);
  }
  out += ']';
  return out;
}

/// The shared representation of an `AABB`.
///
/// Attributes:
///     lo (std::vector<double>): Lower corner.
///     hi (std::vector<double>): Upper corner, the same length.
struct AABBRep {
  std::vector<double> lo;
  std::vector<double> hi;
};

}  // namespace detail

/// An axis-aligned bounding box `[lo_0, hi_0] x ... x [lo_{n-1}, hi_{n-1}]`, closed.
///
/// An immutable value: the corners are copied in at construction and shared by every copy. A
/// box with `lo[i] > hi[i]` on some axis is *empty*; the canonical empty box has `lo = +inf`,
/// `hi = -inf` on every axis, so it is the identity of `union_with`. Infinite bounds are allowed;
/// NaN is refused, since every comparison against it is false and the box would contain nothing
/// and overlap nothing without being empty.
///
/// Attributes:
///     rep_ (std::shared_ptr<const detail::AABBRep>): The corners.
class AABB {
 public:
  /// Build a box from its corners.
  ///
  /// Args:
  ///     lo (std::span<const double>): Lower corner.
  ///     hi (std::span<const double>): Upper corner.
  ///
  /// Raises:
  ///     value_error: The corners differ in length, have no entry, or contain NaN.
  AABB(std::span<const double> lo, std::span<const double> hi)
      : rep_(std::make_shared<const detail::AABBRep>(
            detail::AABBRep{{lo.begin(), lo.end()}, {hi.begin(), hi.end()}})) {
    if (lo.size() != hi.size()) {
      const std::size_t ls[1] = {lo.size()};
      const std::size_t hs[1] = {hi.size()};
      throw value_error("AABB.lo and AABB.hi must share shape; got " + py_repr(ls) + " vs " +
                        py_repr(hs) + ".");
    }
    if (lo.empty()) {
      throw value_error("AABB ndim must be >= 1; got 0.");
    }
    const auto is_nan = [](double x) { return std::isnan(x); };
    if (std::any_of(lo.begin(), lo.end(), is_nan) || std::any_of(hi.begin(), hi.end(), is_nan)) {
      throw value_error("AABB bounds must not contain NaN; got lo=" + detail::py_repr_list(lo) +
                        ", hi=" + detail::py_repr_list(hi) + ".");
    }
  }

  /// Build the box that contains every point, `[-inf, inf]^ndim`.
  ///
  /// Args:
  ///     ndim (long long): The dimension, at least 1.
  ///
  /// Returns:
  ///     AABB: The unbounded box.
  ///
  /// Raises:
  ///     value_error: `ndim < 1`.
  [[nodiscard]] static AABB unbounded(long long ndim) {
    check_ndim(ndim, "AABB.unbounded");
    const auto n = static_cast<std::size_t>(ndim);
    const std::vector<double> lo(n, -std::numeric_limits<double>::infinity());
    const std::vector<double> hi(n, std::numeric_limits<double>::infinity());
    return {lo, hi};
  }

  /// Build the canonical empty box, `lo = +inf`, `hi = -inf` on every axis.
  ///
  /// Args:
  ///     ndim (long long): The dimension, at least 1.
  ///
  /// Returns:
  ///     AABB: The empty box, the identity of `union_with`.
  ///
  /// Raises:
  ///     value_error: `ndim < 1`.
  [[nodiscard]] static AABB empty(long long ndim) {
    check_ndim(ndim, "AABB.empty");
    const auto n = static_cast<std::size_t>(ndim);
    const std::vector<double> lo(n, std::numeric_limits<double>::infinity());
    const std::vector<double> hi(n, -std::numeric_limits<double>::infinity());
    return {lo, hi};
  }

  /// Build a box from an `(ndim, 2)` array of `[lo, hi]` rows.
  ///
  /// Args:
  ///     bounds (std::span<const double>): Row-major over `shape`.
  ///     shape (std::span<const std::size_t>): The array's extents; must be `(ndim, 2)`.
  ///
  /// Returns:
  ///     AABB: The box.
  ///
  /// Raises:
  ///     value_error: A shape other than `(ndim, 2)`, or the constructor's errors.
  [[nodiscard]] static AABB from_bounds(std::span<const double> bounds,
                                        std::span<const std::size_t> shape) {
    if (shape.size() != 2 || shape[1] != 2) {
      throw value_error("from_bounds: bounds must have shape (ndim, 2); got " + py_repr(shape) +
                        ".");
    }
    if (bounds.size() != 2 * shape[0]) {
      throw value_error("from_bounds: bounds hold " + std::to_string(bounds.size()) +
                        " values, but the shape " + py_repr(shape) + " needs " +
                        std::to_string(2 * shape[0]) + ".");
    }
    std::vector<double> lo(shape[0]);
    std::vector<double> hi(shape[0]);
    for (std::size_t i = 0; i < shape[0]; ++i) {
      lo[i] = bounds[2 * i];
      hi[i] = bounds[2 * i + 1];
    }
    return {lo, hi};
  }

  /// Get the dimension.
  ///
  /// Returns:
  ///     std::size_t: The number of axes, at least 1.
  [[nodiscard]] std::size_t ndim() const noexcept { return rep_->lo.size(); }

  /// Get the lower corner (a numeric view into the shared representation).
  ///
  /// Returns:
  ///     std::span<const double>: `ndim()` values.
  [[nodiscard]] std::span<const double> lo() const noexcept { return rep_->lo; }

  /// Get the upper corner (a numeric view into the shared representation).
  ///
  /// Returns:
  ///     std::span<const double>: `ndim()` values.
  [[nodiscard]] std::span<const double> hi() const noexcept { return rep_->hi; }

  /// Report whether the box is empty: `lo[i] > hi[i]` on some axis.
  ///
  /// Returns:
  ///     bool: Whether the box contains no point.
  [[nodiscard]] bool is_empty() const noexcept {
    for (std::size_t i = 0; i < ndim(); ++i) {
      if (rep_->lo[i] > rep_->hi[i]) {
        return true;
      }
    }
    return false;
  }

  /// Report whether a point lies in the closed box.
  ///
  /// Args:
  ///     x (std::span<const double>): The point, `ndim()` coordinates.
  ///
  /// Returns:
  ///     bool: Whether `lo <= x <= hi` on every axis.
  ///
  /// Raises:
  ///     value_error: `x` has the wrong length or contains NaN.
  [[nodiscard]] bool contains_point(std::span<const double> x) const {
    if (x.size() != ndim()) {
      const std::size_t xs[1] = {x.size()};
      throw value_error("contains_point: x must have length " + std::to_string(ndim()) + "; got " +
                        py_repr(xs) + ".");
    }
    if (std::any_of(x.begin(), x.end(), [](double v) { return std::isnan(v); })) {
      throw value_error("contains_point: x must not contain NaN.");
    }
    for (std::size_t i = 0; i < ndim(); ++i) {
      if (!(x[i] >= rep_->lo[i] && x[i] <= rep_->hi[i])) {
        return false;
      }
    }
    return true;
  }

  /// Report whether two boxes share a point (touching counts; an empty box overlaps nothing).
  ///
  /// Args:
  ///     other (const AABB&): A box of the same dimension.
  ///
  /// Returns:
  ///     bool: Whether the closed boxes intersect.
  ///
  /// Raises:
  ///     value_error: The dimensions differ.
  [[nodiscard]] bool overlaps(const AABB& other) const {
    require_same_ndim(other, "overlaps");
    if (is_empty() || other.is_empty()) {
      return false;
    }
    for (std::size_t i = 0; i < ndim(); ++i) {
      if ((std::max)(rep_->lo[i], other.rep_->lo[i]) > (std::min)(rep_->hi[i], other.rep_->hi[i])) {
        return false;
      }
    }
    return true;
  }

  /// Get the smallest box containing both (an empty operand is the identity and is returned as
  /// it is: the result shares its representation).
  ///
  /// Args:
  ///     other (const AABB&): A box of the same dimension.
  ///
  /// Returns:
  ///     AABB: The union's bounding box.
  ///
  /// Raises:
  ///     value_error: The dimensions differ.
  [[nodiscard]] AABB union_with(const AABB& other) const {
    require_same_ndim(other, "union");
    if (is_empty()) {
      return other;
    }
    if (other.is_empty()) {
      return *this;
    }
    std::vector<double> lo(ndim());
    std::vector<double> hi(ndim());
    for (std::size_t i = 0; i < ndim(); ++i) {
      lo[i] = (std::min)(rep_->lo[i], other.rep_->lo[i]);
      hi[i] = (std::max)(rep_->hi[i], other.rep_->hi[i]);
    }
    return {lo, hi};
  }

  /// Get the intersection.
  ///
  /// Args:
  ///     other (const AABB&): A box of the same dimension.
  ///
  /// Returns:
  ///     std::optional<AABB>: The intersection, or no value when it is empty (an empty operand
  ///     included).
  ///
  /// Raises:
  ///     value_error: The dimensions differ.
  [[nodiscard]] std::optional<AABB> intersect(const AABB& other) const {
    require_same_ndim(other, "intersect");
    if (is_empty() || other.is_empty()) {
      return std::nullopt;
    }
    std::vector<double> lo(ndim());
    std::vector<double> hi(ndim());
    for (std::size_t i = 0; i < ndim(); ++i) {
      lo[i] = (std::max)(rep_->lo[i], other.rep_->lo[i]);
      hi[i] = (std::min)(rep_->hi[i], other.rep_->hi[i]);
      if (lo[i] > hi[i]) {
        return std::nullopt;
      }
    }
    return AABB(lo, hi);
  }

  /// Grow the box by `r[i]` on both sides of each axis (shrink where `r[i] < 0`).
  ///
  /// Args:
  ///     r (std::span<const double>): One finite padding per axis.
  ///
  /// Returns:
  ///     AABB: `[lo - r, hi + r]`; an infinite bound stays infinite.
  ///
  /// Raises:
  ///     value_error: `r` has the wrong length or a non-finite entry.
  [[nodiscard]] AABB pad(std::span<const double> r) const {
    if (r.size() != ndim()) {
      const std::size_t rs[1] = {r.size()};
      throw value_error("pad(r) requires r scalar or shape (" + std::to_string(ndim()) +
                        ",); got " + py_repr(rs) + ".");
    }
    if (!std::all_of(r.begin(), r.end(), [](double v) { return std::isfinite(v); })) {
      throw value_error("pad(r) entries must be finite; got " + detail::py_repr_list(r) + ".");
    }
    std::vector<double> lo(ndim());
    std::vector<double> hi(ndim());
    for (std::size_t i = 0; i < ndim(); ++i) {
      lo[i] = rep_->lo[i] - r[i];
      hi[i] = rep_->hi[i] + r[i];
    }
    return {lo, hi};
  }

  /// Grow the box by the same padding on every axis.
  ///
  /// Args:
  ///     r (double): A finite padding.
  ///
  /// Returns:
  ///     AABB: See the per-axis overload.
  ///
  /// Raises:
  ///     value_error: `r` is not finite.
  [[nodiscard]] AABB pad(double r) const {
    const std::vector<double> rs(ndim(), r);
    return pad(std::span<const double>(rs));
  }

  /// Get the bounding box of the image of this box under `x -> A x + b`.
  ///
  /// Per output axis `i`, the image's bounds are `b[i] + sum_j min/max(A[i,j] lo[j],
  /// A[i,j] hi[j])`, exact for an axis-aligned box. A zero `A[i,j]` contributes nothing even
  /// against an infinite bound (rather than `0 * inf = NaN`). An empty box maps to the canonical
  /// empty box, before any check of the map.
  ///
  /// Args:
  ///     affine_dim (long long): The map's dimension.
  ///     matrix (std::span<const double>): `A`, row-major over `matrix_shape`.
  ///     matrix_shape (std::span<const std::size_t>): `A`'s extents; must be `(ndim, ndim)`.
  ///     offset (std::span<const double>): `b`, `ndim()` values.
  ///
  /// Returns:
  ///     AABB: The image's bounding box.
  ///
  /// Raises:
  ///     value_error: `affine_dim != ndim()`, a matrix of another shape, an offset of another
  ///         length, or a NaN bound in the result (infinite bounds that cancel).
  [[nodiscard]] AABB transform(long long affine_dim, std::span<const double> matrix,
                               std::span<const std::size_t> matrix_shape,
                               std::span<const double> offset) const {
    const std::size_t n = ndim();
    if (is_empty()) {
      return empty(static_cast<long long>(n));
    }
    if (affine_dim != static_cast<long long>(n)) {
      throw value_error("transform(): affine dim (" + std::to_string(affine_dim) +
                        ") must match AABB ndim (" + std::to_string(n) + ").");
    }
    if (matrix_shape.size() != 2 || matrix_shape[0] != n || matrix_shape[1] != n) {
      throw value_error("transform(): affine.matrix must be (" + std::to_string(n) + ", " +
                        std::to_string(n) + "); got " + py_repr(matrix_shape) + ".");
    }
    if (matrix.size() != n * n) {
      throw value_error("transform(): affine.matrix holds " + std::to_string(matrix.size()) +
                        " values, but its shape needs " + std::to_string(n * n) + ".");
    }
    if (offset.size() != n) {
      const std::size_t os[1] = {offset.size()};
      throw value_error("transform(): affine.offset must have shape (" + std::to_string(n) +
                        ",); got " + py_repr(os) + ".");
    }
    std::vector<double> lo(n);
    std::vector<double> hi(n);
    for (std::size_t i = 0; i < n; ++i) {
      // Summed left to right from the first term, as numpy's reduction does for a short row, so a
      // row of zero contributions keeps the sign its first term carries.
      double sum_lo = 0.0;
      double sum_hi = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        const double a = matrix[i * n + j];
        const double t_lo = a == 0.0 ? 0.0 : a * rep_->lo[j];
        const double t_hi = a == 0.0 ? 0.0 : a * rep_->hi[j];
        // NaN propagates, as through np.minimum / np.maximum (std::min and std::max would drop
        // it in one argument position): an infinite entry of A against a zero bound gives one.
        const bool nan = std::isnan(t_lo) || std::isnan(t_hi);
        const double c_lo = nan ? t_lo + t_hi : (std::min)(t_lo, t_hi);
        const double c_hi = nan ? t_lo + t_hi : (std::max)(t_lo, t_hi);
        sum_lo = j == 0 ? c_lo : sum_lo + c_lo;
        sum_hi = j == 0 ? c_hi : sum_hi + c_hi;
      }
      lo[i] = sum_lo + offset[i];
      hi[i] = sum_hi + offset[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
      if (std::isnan(lo[i]) || std::isnan(hi[i])) {
        throw value_error(
            "AABB.transform produced NaN bounds; the transform is incompatible with this AABB "
            "(for example, a singular matrix combined with infinite bounds).");
      }
    }
    return {lo, hi};
  }

  /// Get the box as an `(ndim, 2)` array of `[lo, hi]` rows.
  ///
  /// Returns:
  ///     NdArray<double>: A fresh array.
  [[nodiscard]] NdArray<double> as_bounds() const {
    NdArray<double> out({ndim(), 2}, for_overwrite);
    for (std::size_t i = 0; i < ndim(); ++i) {
      out.data()[2 * i] = rep_->lo[i];
      out.data()[2 * i + 1] = rep_->hi[i];
    }
    return out;
  }

  /// Compare two boxes by value (IEEE equality per bound, so `-0.0 == 0.0`; boxes of different
  /// dimensions are unequal).
  ///
  /// Args:
  ///     other (const AABB&): The other box.
  ///
  /// Returns:
  ///     bool: Whether the corners are equal.
  [[nodiscard]] bool operator==(const AABB& other) const noexcept {
    return rep_->lo.size() == other.rep_->lo.size() &&
           std::equal(rep_->lo.begin(), rep_->lo.end(), other.rep_->lo.begin()) &&
           std::equal(rep_->hi.begin(), rep_->hi.end(), other.rep_->hi.begin());
  }

  /// Get the shared representation, for an owner that must keep the corners alive.
  ///
  /// Returns:
  ///     const std::shared_ptr<const detail::AABBRep>&: The representation.
  [[nodiscard]] const std::shared_ptr<const detail::AABBRep>& rep() const noexcept { return rep_; }

  /// Report whether two values share one representation.
  ///
  /// Args:
  ///     other (const AABB&): The other box.
  ///
  /// Returns:
  ///     bool: Whether the representations are the same object.
  [[nodiscard]] bool same_rep(const AABB& other) const noexcept { return rep_ == other.rep_; }

 private:
  static void check_ndim(long long ndim, const char* who) {
    if (ndim < 1) {
      throw value_error(std::string(who) + ": ndim must be >= 1; got " + std::to_string(ndim) +
                        ".");
    }
  }

  void require_same_ndim(const AABB& other, const char* op) const {
    if (ndim() != other.ndim()) {
      throw value_error("AABB." + std::string(op) +
                        ": dimension mismatch (a.ndim=" + std::to_string(ndim()) +
                        " vs b.ndim=" + std::to_string(other.ndim()) + ").");
    }
  }

  std::shared_ptr<const detail::AABBRep> rep_;
};

}  // namespace pantr::geometry
