/// @file affine_transform.hpp
/// @brief `AffineTransform`: the map `x -> A x + b` on R^n.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/linalg/dense.hpp"
#include "pantr/parallel/parallel_for.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::transform {

namespace detail {

/// The shared representation of an `AffineTransform`.
///
/// Attributes:
///     matrix (NdArray<double>): `A`, extents `(n, n)`.
///     translation (NdArray<double>): `b`, extents `(n,)`.
struct AffineRep {
  NdArray<double> matrix;
  NdArray<double> translation;
};

}  // namespace detail

/// An affine map `x -> A x + b` on R^n, with `A` any square matrix (singular included).
///
/// An immutable value: `A` and `b` are copied in at construction and shared by every copy.
/// Composition and application compute what the Python implementation this replaces computed
/// (`A @ B`, then `A @ b' + b`), summing each product over the index in ascending order; NumPy's
/// BLAS products may sum in another order or contract into FMAs, so results agree to rounding,
/// not bit for bit.
///
/// Attributes:
///     rep_ (std::shared_ptr<const detail::AffineRep>): `A` and `b`.
class AffineTransform {
 public:
  /// Build a map from its matrix and translation.
  ///
  /// Args:
  ///     matrix (std::span<const double>): `A`, row-major over `matrix_shape`.
  ///     matrix_shape (std::span<const std::size_t>): Must be `(n, n)`.
  ///     translation (std::optional<std::span<const double>>): `b`; no value means zero.
  ///     translation_shape (std::span<const std::size_t>): `b`'s extents; must be `(n,)` when
  ///         `translation` holds a value.
  ///
  /// Raises:
  ///     value_error: `A` is not square and rank 2, or `b` does not have shape `(n,)`.
  AffineTransform(std::span<const double> matrix, std::span<const std::size_t> matrix_shape,
                  std::optional<std::span<const double>> translation = std::nullopt,
                  std::span<const std::size_t> translation_shape = {})
      : rep_(make_rep(matrix, matrix_shape, translation, translation_shape)) {}

  /// Build the identity on R^n.
  ///
  /// Args:
  ///     n (std::size_t): The dimension (0 is allowed, as numpy's `eye(0)`).
  ///
  /// Returns:
  ///     AffineTransform: `x -> x`.
  [[nodiscard]] static AffineTransform identity(std::size_t n) {
    const std::vector<double> eye = identity_matrix(n);
    const std::size_t shape[2] = {n, n};
    return {eye, shape};
  }

  /// Build the translation `x -> x + offset`.
  ///
  /// Args:
  ///     offset (std::span<const double>): `b`; its length is the dimension.
  ///
  /// Returns:
  ///     AffineTransform: The translation.
  [[nodiscard]] static AffineTransform translation(std::span<const double> offset) {
    const std::size_t n = offset.size();
    const std::vector<double> eye = identity_matrix(n);
    const std::size_t shape[2] = {n, n};
    const std::size_t tshape[1] = {n};
    return {eye, shape, offset, tshape};
  }

  /// Build the scaling `x -> diag(factors) x`, optionally about a center.
  ///
  /// Args:
  ///     factors (std::span<const double>): One finite, non-zero factor per axis.
  ///     center (std::optional<std::span<const double>>): The fixed point; the origin if none.
  ///
  /// Returns:
  ///     AffineTransform: The scaling.
  ///
  /// Raises:
  ///     value_error: A non-finite or zero factor, or a center of another length.
  [[nodiscard]] static AffineTransform scaling(
      std::span<const double> factors,
      std::optional<std::span<const double>> center = std::nullopt) {
    // Every factor finite first, then none zero, in the previous release's order.
    if (!std::all_of(factors.begin(), factors.end(), [](double f) { return std::isfinite(f); })) {
      throw value_error("scaling factors must be finite.");
    }
    if (std::any_of(factors.begin(), factors.end(), [](double f) { return f == 0.0; })) {
      throw value_error("scaling factors must be non-zero (singular transform).");
    }
    const std::size_t n = factors.size();
    std::vector<double> mat(square_size(n), 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      mat[i * n + i] = factors[i];
    }
    const std::size_t shape[2] = {n, n};
    return with_optional_center(AffineTransform(mat, shape), center);
  }

  /// Build the rotation of R^2 by `angle` radians (counter-clockwise), optionally about a center.
  ///
  /// Args:
  ///     angle (double): A finite angle.
  ///     center (std::optional<std::span<const double>>): The fixed point; the origin if none.
  ///
  /// Returns:
  ///     AffineTransform: The rotation.
  ///
  /// Raises:
  ///     value_error: A non-finite angle, or a center of another length.
  [[nodiscard]] static AffineTransform rotation_2d(
      double angle, std::optional<std::span<const double>> center = std::nullopt) {
    check_angle(angle);
    using std::cos;
    using std::sin;
    const double c = cos(angle);
    const double s = sin(angle);
    const std::vector<double> mat = {c, -s, s, c};
    const std::size_t shape[2] = {2, 2};
    return with_optional_center(AffineTransform(mat, shape), center);
  }

  /// Build the rotation of R^3 by `angle` radians about the axis `u` (Rodrigues' formula
  /// `R = cos(t) I + (1 - cos(t)) u u^T + sin(t) [u]_x`), optionally about a center.
  ///
  /// Args:
  ///     angle (double): A finite angle.
  ///     axis (std::span<const double>): Three coordinates of a finite non-zero vector; it is
  ///         normalised here.
  ///     center (std::optional<std::span<const double>>): A point on the axis; the origin if none.
  ///
  /// Returns:
  ///     AffineTransform: The rotation.
  ///
  /// Raises:
  ///     value_error: An axis that is not three finite coordinates of a non-zero vector, a
  ///         non-finite angle, or a center of another length.
  [[nodiscard]] static AffineTransform rotation_3d(
      double angle, std::span<const double> axis,
      std::optional<std::span<const double>> center = std::nullopt) {
    if (axis.size() != 3) {
      const std::size_t as[1] = {axis.size()};
      throw value_error("Rotation axis must have shape (3,), got " + py_repr(as) + ".");
    }
    const double norm = euclidean_norm(axis);
    if (norm == 0.0 || !std::isfinite(norm)) {
      throw value_error("Rotation axis must be a finite non-zero vector.");
    }
    const double u[3] = {axis[0] / norm, axis[1] / norm, axis[2] / norm};
    check_angle(angle);
    using std::cos;
    using std::sin;
    const double c = cos(angle);
    const double s = sin(angle);
    const double k[9] = {0.0, -u[2], u[1], u[2], 0.0, -u[0], -u[1], u[0], 0.0};
    std::vector<double> mat(9);
    for (std::size_t i = 0; i < 3; ++i) {
      for (std::size_t j = 0; j < 3; ++j) {
        // numpy's `c * eye + (1 - c) * outer(u, u) + s * K`, term by term in that order.
        const double eye = i == j ? 1.0 : 0.0;
        mat[3 * i + j] = c * eye + (1.0 - c) * (u[i] * u[j]) + s * k[3 * i + j];
      }
    }
    const std::size_t shape[2] = {3, 3};
    return with_optional_center(AffineTransform(mat, shape), center);
  }

  /// Build the reflection through the hyperplane with normal `normal`, `I - 2 n n^T` for the
  /// normalised `n`, optionally through a center.
  ///
  /// Args:
  ///     normal (std::span<const double>): A finite non-zero vector; its length is the dimension.
  ///     center (std::optional<std::span<const double>>): A point on the hyperplane.
  ///
  /// Returns:
  ///     AffineTransform: The reflection.
  ///
  /// Raises:
  ///     value_error: A zero or non-finite normal, or a center of another length.
  [[nodiscard]] static AffineTransform mirror(
      std::span<const double> normal,
      std::optional<std::span<const double>> center = std::nullopt) {
    const double norm = euclidean_norm(normal);
    if (norm == 0.0 || !std::isfinite(norm)) {
      throw value_error("Mirror normal must be a finite non-zero vector.");
    }
    const std::size_t n = normal.size();
    std::vector<double> u(n);
    for (std::size_t i = 0; i < n; ++i) {
      u[i] = normal[i] / norm;
    }
    std::vector<double> mat(square_size(n));
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        mat[i * n + j] = (i == j ? 1.0 : 0.0) - 2.0 * (u[i] * u[j]);
      }
    }
    const std::size_t shape[2] = {n, n};
    return with_optional_center(AffineTransform(mat, shape), center);
  }

  /// Build the shear adding `factor * x[direction]` to `x[component]`.
  ///
  /// Args:
  ///     dim (long long): The dimension.
  ///     component (long long): The coordinate that changes, in `[0, dim)`.
  ///     direction (long long): The coordinate it moves along, in `[0, dim)`, not `component`.
  ///     factor (double): A finite factor.
  ///
  /// Returns:
  ///     AffineTransform: The shear.
  ///
  /// Raises:
  ///     value_error: Equal indices, an index out of range, or a non-finite factor.
  [[nodiscard]] static AffineTransform shear(long long dim, long long component,
                                             long long direction, double factor) {
    if (component == direction) {
      throw value_error("component and direction must differ.");
    }
    if (component < 0 || component >= dim) {
      throw value_error("component must be in [0, " + std::to_string(dim) + "), got " +
                        std::to_string(component) + ".");
    }
    if (direction < 0 || direction >= dim) {
      throw value_error("direction must be in [0, " + std::to_string(dim) + "), got " +
                        std::to_string(direction) + ".");
    }
    if (!std::isfinite(factor)) {
      throw value_error("factor must be finite, got " + py_repr(factor) + ".");
    }
    const auto n = static_cast<std::size_t>(dim);
    std::vector<double> mat = identity_matrix(n);
    mat[static_cast<std::size_t>(component) * n + static_cast<std::size_t>(direction)] = factor;
    const std::size_t shape[2] = {n, n};
    return {mat, shape};
  }

  /// Get the dimension.
  ///
  /// Returns:
  ///     std::size_t: `n`.
  [[nodiscard]] std::size_t dim() const noexcept { return rep_->translation.size(); }

  /// Get `A` (a numeric view into the shared representation).
  ///
  /// Returns:
  ///     std::span<const double>: `n * n` values, row-major.
  [[nodiscard]] std::span<const double> matrix() const noexcept { return rep_->matrix.flat(); }

  /// Get `b` (a numeric view into the shared representation).
  ///
  /// Returns:
  ///     std::span<const double>: `n` values.
  [[nodiscard]] std::span<const double> offset() const noexcept { return rep_->translation.flat(); }

  /// Get the inverse map `x -> A^-1 x - A^-1 b`.
  ///
  /// Returns:
  ///     AffineTransform: The inverse.
  ///
  /// Raises:
  ///     value_error: `A` is singular: an LU pivot is exactly zero or NaN (numpy.linalg.inv's
  ///         rule, `linalg::try_inverse`).
  [[nodiscard]] AffineTransform inverse() const {
    std::optional<NdArray<double>> inv = linalg::try_inverse(rep_->matrix);
    if (!inv) {
      throw value_error("Cannot invert a singular affine transformation.");
    }
    const std::size_t n = dim();
    std::vector<double> t(n);
    const double* a = inv->data();
    const std::span<const double> b = offset();
    for (std::size_t i = 0; i < n; ++i) {
      double acc = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        acc += a[i * n + j] * b[j];
      }
      t[i] = -acc;
    }
    const std::size_t shape[2] = {n, n};
    const std::size_t tshape[1] = {n};
    return {inv->flat(), shape, std::span<const double>(t), tshape};
  }

  /// Compose: `(this o other)(x) = this(other(x))`.
  ///
  /// Args:
  ///     other (const AffineTransform&): The map applied first, of the same dimension.
  ///
  /// Returns:
  ///     AffineTransform: `x -> A_this (A_other x + b_other) + b_this`.
  ///
  /// Raises:
  ///     value_error: The dimensions differ.
  [[nodiscard]] AffineTransform compose(const AffineTransform& other) const {
    const std::size_t n = dim();
    if (other.dim() != n) {
      throw value_error("Cannot compose transforms of different dimensions (" + std::to_string(n) +
                        " and " + std::to_string(other.dim()) + ").");
    }
    const std::span<const double> a = matrix();
    const std::span<const double> b = other.matrix();
    const std::span<const double> tb = other.offset();
    const std::span<const double> ta = offset();
    std::vector<double> mat(n * n);
    std::vector<double> t(n);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        double acc = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
          acc += a[i * n + k] * b[k * n + j];
        }
        mat[i * n + j] = acc;
      }
      double acc = 0.0;
      for (std::size_t k = 0; k < n; ++k) {
        acc += a[i * n + k] * tb[k];
      }
      t[i] = acc + ta[i];
    }
    const std::size_t shape[2] = {n, n};
    const std::size_t tshape[1] = {n};
    return {mat, shape, std::span<const double>(t), tshape};
  }

  /// Apply the map to points.
  ///
  /// Args:
  ///     points (std::span<const double>): `m * n` coordinates, row-major `(m, n)` (any leading
  ///         shape flattened to `m`).
  ///
  /// Returns:
  ///     NdArray<double>: Extents `(m, n)`, `A p + b` per point.
  ///
  /// Raises:
  ///     value_error: `points.size()` is not a multiple of `n` (for `n == 0`, not empty).
  [[nodiscard]] NdArray<double> apply(std::span<const double> points) const {
    const std::size_t n = dim();
    if (n == 0 ? !points.empty() : points.size() % n != 0) {
      throw value_error("points do not hold a whole number of " + std::to_string(n) +
                        "-coordinate points");
    }
    const std::size_t m = n == 0 ? 0 : points.size() / n;
    NdArray<double> out({m, n}, for_overwrite);
    apply(points, out.flat());
    return out;
  }

  /// Apply the map to points, writing into `out`.
  ///
  /// Args:
  ///     points (std::span<const double>): `m * n` coordinates, row-major.
  ///     out (std::span<double>): `m * n` values; may not alias `points`.
  ///
  /// Raises:
  ///     value_error: The sizes differ, or are not a multiple of `n`.
  void apply(std::span<const double> points, std::span<double> out) const {
    const std::size_t n = dim();
    if (out.size() != points.size() || (n != 0 && points.size() % n != 0)) {
      throw value_error("points and out must hold the same whole number of " + std::to_string(n) +
                        "-coordinate points");
    }
    const std::span<const double> a = matrix();
    const std::span<const double> b = offset();
    const std::size_t m = n == 0 ? 0 : points.size() / n;
    parallel::parallel_for(m, ops_per_point(), [&](std::size_t begin, std::size_t end) {
      for (std::size_t p = begin; p < end; ++p) {
        const double* x = points.data() + p * n;
        double* y = out.data() + p * n;
        for (std::size_t i = 0; i < n; ++i) {
          double acc = 0.0;
          for (std::size_t j = 0; j < n; ++j) {
            acc += x[j] * a[i * n + j];
          }
          y[i] = acc + b[i];
        }
      }
    });
  }

  /// Work per point of `apply`: a multiply-add per entry of `A`, and an addition per coordinate.
  ///
  /// Returns:
  ///     parallel::WorkEstimate: The estimate, at least one operation.
  [[nodiscard]] parallel::WorkEstimate ops_per_point() const noexcept {
    const auto n = static_cast<double>(dim());
    return parallel::WorkEstimate{(std::max)(1.0, 2.0 * n * n + n)};
  }

  /// Get the shared representation, for an owner that must keep `A` and `b` alive.
  ///
  /// Returns:
  ///     const std::shared_ptr<const detail::AffineRep>&: The representation.
  [[nodiscard]] const std::shared_ptr<const detail::AffineRep>& rep() const noexcept {
    return rep_;
  }

 private:
  /// `n * n`, refused when it overflows (a wrapped product would size a buffer the diagonal then
  /// writes past).
  static std::size_t square_size(std::size_t n) {
    const std::size_t extents[2] = {n, n};
    return checked_product(extents);
  }

  static std::vector<double> identity_matrix(std::size_t n) {
    std::vector<double> eye(square_size(n), 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      eye[i * n + i] = 1.0;
    }
    return eye;
  }

  static double euclidean_norm(std::span<const double> v) {
    using std::sqrt;
    // As numpy.linalg.norm computes it for a vector, sqrt(x . x), unscaled: a vector near the
    // overflow threshold has an infinite norm, which every caller refuses, as before.
    double sum = 0.0;
    for (const double x : v) {
      sum += x * x;
    }
    return sqrt(sum);
  }

  static void check_angle(double angle) {
    if (!std::isfinite(angle)) {
      throw value_error("angle must be finite, got " + py_repr(angle) + ".");
    }
  }

  static AffineTransform with_optional_center(AffineTransform t,
                                              std::optional<std::span<const double>> center) {
    if (!center) {
      return t;
    }
    if (center->size() != t.dim()) {
      const std::size_t cs[1] = {center->size()};
      throw value_error("center must have shape (" + std::to_string(t.dim()) + ",), got " +
                        py_repr(cs) + ".");
    }
    std::vector<double> neg(center->size());
    for (std::size_t i = 0; i < neg.size(); ++i) {
      neg[i] = -(*center)[i];
    }
    // numpy's `t_pos @ t @ t_neg`, left to right.
    return translation(*center).compose(t).compose(translation(neg));
  }

  static std::shared_ptr<const detail::AffineRep> make_rep(
      std::span<const double> matrix, std::span<const std::size_t> matrix_shape,
      std::optional<std::span<const double>> translation,
      std::span<const std::size_t> translation_shape) {
    if (matrix_shape.size() != 2 || matrix_shape[0] != matrix_shape[1]) {
      throw value_error("matrix must be a square 2-D array, got shape " + py_repr(matrix_shape) +
                        ".");
    }
    const std::size_t n = matrix_shape[0];
    if (matrix.size() != square_size(n)) {
      throw value_error("matrix holds " + std::to_string(matrix.size()) +
                        " values, but its shape needs " + std::to_string(n * n));
    }
    NdArray<double> mat({n, n}, for_overwrite);
    std::copy(matrix.begin(), matrix.end(), mat.data());
    NdArray<double> t({n});
    if (translation) {
      if (translation_shape.size() != 1 || translation_shape[0] != n || translation->size() != n) {
        throw value_error("translation must have shape (" + std::to_string(n) + ",), got " +
                          py_repr(translation_shape) + ".");
      }
      std::copy(translation->begin(), translation->end(), t.data());
    }
    return std::make_shared<const detail::AffineRep>(
        detail::AffineRep{std::move(mat), std::move(t)});
  }

  std::shared_ptr<const detail::AffineRep> rep_;
};

}  // namespace pantr::transform
