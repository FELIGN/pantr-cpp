/// @file primitives.hpp
/// @brief Constructive geometry primitives: `create_line`, `create_circle`, `create_bilinear` and
///        `create_trilinear`.
///
/// Basic B-spline objects from geometric descriptions (points, corners, a radius, angles). Every
/// primitive produces rank-3 (3D) `double` output so they compose freely with the operations of
/// `cad/operations.hpp`.
#pragma once

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/cad/detail/validation.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/transform/affine_transform.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::cad {

namespace detail {

/// The degree of the rational B-spline that represents a circular arc exactly.
inline constexpr int circle_degree = 2;

/// Create the degree-1 space on `[0, 1]`, knots `[0, 0, 1, 1]`.
///
/// Returns:
///     bspline::BsplineSpace1D<T>: The clamped linear space.
template <StorageFloat T>
[[nodiscard]] bspline::BsplineSpace1D<T> linear_space_1d() {
  const std::array<T, 4> knots = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(1),
                                  static_cast<T>(1)};
  return bspline::BsplineSpace1D<T>(std::span<const T>(knots), 1);
}

/// Rotate weighted homogeneous control points about the z axis.
///
/// For a point `(w x, w y, w z, w)` the rotation acts on the spatial part `(w x, w y, w z)` and
/// the weight is copied. The rotation matrix is that of `AffineTransform::rotation_3d(angle,
/// e_z)`, and each rotated coordinate is the sum over the columns of `cw[..., :3] @ R^T`, added
/// in the order of the column index.
///
/// Args:
///     in (const double*): `rows` control points of four coordinates, row-major.
///     out (double*): Room for `rows` control points; must not overlap `in`.
///     rows (std::size_t): The number of control points.
///     angle (double): The rotation angle in radians (`rotation_3d` refuses a non-finite one).
///
/// Raises:
///     value_error: `angle` is not finite.
inline void rotate_weighted_z(const double* in, double* out, std::size_t rows, double angle) {
  const std::array<double, 3> e_z = {0.0, 0.0, 1.0};
  const transform::AffineTransform rotation = transform::AffineTransform::rotation_3d(angle, e_z);
  const std::span<const double> r = rotation.matrix();
  for (std::size_t p = 0; p < rows; ++p) {
    const double* x = in + p * (physical_dim + 1);
    double* y = out + p * (physical_dim + 1);
    for (std::size_t i = 0; i < physical_dim; ++i) {
      double acc = 0.0;
      for (std::size_t j = 0; j < physical_dim; ++j) {
        acc += x[j] * r[i * physical_dim + j];
      }
      y[i] = acc;
    }
    y[physical_dim] = x[physical_dim];
  }
}

/// Build the weighted homogeneous control points of a full circle: nine points of four
/// coordinates, row-major.
///
/// Args:
///     radius (double): The circle's radius.
///
/// Returns:
///     std::vector<double>: The `9 * 4` values. Corner points carry weight 1, the points between
///     them the weight `sqrt(2) / 2` and the corresponding factor on their coordinates.
[[nodiscard]] inline std::vector<double> build_full_circle(double radius) {
  constexpr std::size_t count = 9;
  constexpr std::array<std::array<double, 2>, count> unit = {
      {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}, {1, 0}}};
  const double wm = std::sqrt(2.0) / 2.0;
  std::vector<double> cw(count * (physical_dim + 1), 0.0);
  for (std::size_t p = 0; p < count; ++p) {
    double* row = cw.data() + p * (physical_dim + 1);
    row[0] = unit[p][0] * radius;
    row[1] = unit[p][1] * radius;
    row[physical_dim] = 1.0;
    if (p % 2 == 1) {
      for (std::size_t c = 0; c < physical_dim + 1; ++c) {
        row[c] *= wm;
      }
    }
  }
  return cw;
}

/// Build the weighted homogeneous control points of a circular arc.
///
/// A template arc bisected by the +x axis is rotated to the start angle; each further span is the
/// previous span's last two control points rotated by the span's angle `2 alpha`.
///
/// Args:
///     radius (double): The circle's radius.
///     start (double): The start angle in radians.
///     sweep (double): The sweep angle in radians (negative for a clockwise arc).
///     spans (int): The number of quadratic arc spans, at least 1.
///
/// Returns:
///     std::vector<double>: The `(2 * spans + 1) * 4` values, row-major.
///
/// Raises:
///     value_error: A rotation angle is not finite (the sweep or the start is NaN or infinite).
[[nodiscard]] inline std::vector<double> build_arc(double radius, double start, double sweep,
                                                   int spans) {
  const double alpha = sweep / static_cast<double>(2 * spans);
  const double sin_a = std::sin(alpha);
  const double cos_a = std::cos(alpha);
  const double tan_a = std::tan(alpha);
  const double x = radius * cos_a;
  const double y = radius * sin_a;
  const double wm = cos_a;
  const double xm = x + y * tan_a;

  // The template arc: three control points bisected by the +x axis.
  const std::array<double, 12> tmpl = {x,       -y,  0.0, 1.0,   // start
                                       wm * xm, 0.0, 0.0, wm,    // middle
                                       x,       y,   0.0, 1.0};  // end
  const auto rows = static_cast<std::size_t>(2 * spans + 1);
  std::vector<double> cw(rows * (physical_dim + 1));
  rotate_weighted_z(tmpl.data(), cw.data(), 3, alpha + start);

  // Each following span is a rotation of the previous one.
  const double two_alpha = 2.0 * alpha;
  for (std::size_t i = 1; i < static_cast<std::size_t>(spans); ++i) {
    const std::size_t n = 2 * i + 1;
    rotate_weighted_z(cw.data() + (n - 2) * (physical_dim + 1), cw.data() + n * (physical_dim + 1),
                      2, two_alpha);
  }
  return cw;
}

/// The number of arc spans of a sweep: how many of the quadrant bounds `0, pi/2, pi, 3 pi/2` lie
/// strictly below `|sweep|` (`numpy.searchsorted` with side left), at least 1. A NaN sorts last,
/// so it gives 4; the rotation then refuses it.
///
/// Args:
///     sweep (double): The sweep angle in radians.
///
/// Returns:
///     int: The number of spans, in `[1, 4]`.
[[nodiscard]] inline int arc_spans(double sweep) noexcept {
  constexpr std::array<double, 4> bounds = {0.0, std::numbers::pi / 2.0, std::numbers::pi,
                                            3.0 * std::numbers::pi / 2.0};
  if (std::isnan(sweep)) {
    return 4;
  }
  const double magnitude = std::abs(sweep);
  int below = 0;
  for (const double b : bounds) {
    below += b < magnitude ? 1 : 0;
  }
  return (std::max)(below, 1);
}

/// Validate the shape of a corner array and zero-pad its coordinates to three.
///
/// Args:
///     values (std::span<const double>): The corner coordinates, row-major over `shape`.
///     shape (std::span<const std::size_t>): The extents, required to be `(2, ..., 2, rank)` with
///         `dim` leading twos and `rank <= 3`.
///     dim (std::size_t): The number of leading twos, 2 or 3.
///
/// Returns:
///     std::vector<double>: The `2^dim * 3` control points, row-major, the coordinates beyond
///     `rank` zero.
///
/// Raises:
///     value_error: `shape` is not `(2, ..., 2, rank)` ("corners must have shape ..."); `rank` is
///         above 3; `values` does not hold the number of values `shape` implies.
[[nodiscard]] inline std::vector<double> pad_corners(std::span<const double> values,
                                                     std::span<const std::size_t> shape,
                                                     std::size_t dim) {
  bool shaped = shape.size() == dim + 1;
  for (std::size_t d = 0; shaped && d < dim; ++d) {
    shaped = shape[d] == 2;
  }
  if (!shaped) {
    std::string expected = "(";
    for (std::size_t d = 0; d < dim; ++d) {
      expected += "2, ";
    }
    throw value_error("corners must have shape " + expected + "rank), got " + py_repr(shape) + ".");
  }
  const std::size_t rank = shape[dim];
  if (rank > physical_dim) {
    throw value_error("corners rank must be at most " + std::to_string(physical_dim) + ", got " +
                      std::to_string(rank) + ".");
  }
  const std::size_t corners = std::size_t{1} << dim;
  if (values.size() != corners * rank) {
    throw value_error("corners holds " + std::to_string(values.size()) + " values, but its shape " +
                      py_repr(shape) + " needs " + std::to_string(corners * rank) + ".");
  }
  std::vector<double> cp(corners * physical_dim, 0.0);
  for (std::size_t c = 0; c < corners; ++c) {
    for (std::size_t i = 0; i < rank; ++i) {
      cp[c * physical_dim + i] = values[c * rank + i];
    }
  }
  return cp;
}

/// Build the space of `dim` linear directions on `[0, 1]`.
[[nodiscard]] inline bspline::BsplineSpace<double> linear_space(std::size_t dim) {
  std::vector<bspline::BsplineSpace1D<double>> spaces;
  spaces.reserve(dim);
  for (std::size_t d = 0; d < dim; ++d) {
    spaces.push_back(linear_space_1d<double>());
  }
  return bspline::BsplineSpace<double>(std::move(spaces));
}

}  // namespace detail

/// Construct a straight-line B-spline curve between two points.
///
/// A degree-1, non-rational curve with two control points and the knots `[0, 0, 1, 1]`. A point of
/// fewer than three coordinates is zero-padded.
///
/// Args:
///     p0 (std::span<const double>): The start point, up to three coordinates.
///     p1 (std::span<const double>): The end point, up to three coordinates.
///
/// Returns:
///     bspline::Bspline<double>: A 1D, degree-1, rank-3, non-rational curve.
///
/// Raises:
///     value_error: A point has more than three coordinates (`p0` is checked first).
[[nodiscard]] inline bspline::Bspline<double> create_line(std::span<const double> p0,
                                                          std::span<const double> p1) {
  const std::array<double, detail::physical_dim> a = detail::pad_to_3d(p0);
  const std::array<double, detail::physical_dim> b = detail::pad_to_3d(p1);
  const std::array<double, 2 * detail::physical_dim> cp = {a[0], a[1], a[2], b[0], b[1], b[2]};
  return bspline::Bspline<double>(detail::linear_space(1), std::span<const double>(cp), false);
}

/// Construct a NURBS circular arc or full circle.
///
/// A degree-2 rational curve in the xy-plane. The arc is split into spans of at most 90 degrees;
/// the interior knots have multiplicity 2 (the degree), so the curve is `C^0` at the junctions,
/// the standard exact representation of conics by rational quadratics. The number of spans is the
/// number of the bounds `0, 90, 180, 270` degrees strictly below the sweep's magnitude, at least
/// 1: 1 span and 3 control points up to 90 degrees, 2 and 5 up to 180, 3 and 7 up to 270, 4 and 9
/// beyond. The sweep `0` gives one span of three coincident control points; `10` radians gives
/// four spans (the arc is not wrapped).
///
/// Args:
///     radius (double): The circle's radius. Defaults to 1.
///     center (std::optional<std::span<const double>>): The center, up to three coordinates,
///         zero-padded; no value means the origin.
///     sweep (std::optional<std::array<double, 2>>): `(start, end)` in radians; no value means the
///         full circle. A single angle `a` of the Python interface is `(0, a)`.
///
/// Returns:
///     bspline::Bspline<double>: A 1D, degree-2, rank-3, rational curve.
///
/// Raises:
///     value_error: The center has more than three coordinates; an angle is not finite (the
///         message is the rotation's: `angle must be finite, got ...`).
[[nodiscard]] inline bspline::Bspline<double> create_circle(
    double radius = 1.0, std::optional<std::span<const double>> center = std::nullopt,
    std::optional<std::array<double, 2>> sweep = std::nullopt) {
  std::vector<double> cw;
  int spans = 4;
  if (!sweep) {
    cw = detail::build_full_circle(radius);
  } else {
    const double start = (*sweep)[0];
    const double delta = (*sweep)[1] - start;
    spans = detail::arc_spans(delta);
    cw = detail::build_arc(radius, start, delta, spans);
  }

  if (center) {
    // For weighted homogeneous points `(w x, w y, w z, w)` the translation by `c` is
    // `(w x + w c_x, ...)`.
    const std::array<double, detail::physical_dim> c = detail::pad_to_3d(*center);
    for (std::size_t p = 0; p < cw.size() / (detail::physical_dim + 1); ++p) {
      double* row = cw.data() + p * (detail::physical_dim + 1);
      for (std::size_t i = 0; i < detail::physical_dim; ++i) {
        row[i] += row[detail::physical_dim] * c[i];
      }
    }
  }

  // The knot vector `[0, 0, 0, u1, u1, u2, u2, ..., 1, 1, 1]`, the interior values `i / spans` as
  // `numpy.linspace` forms them (`i * (1 / spans)`).
  const auto n_spans = static_cast<std::size_t>(spans);
  std::vector<double> knots(2 * (n_spans + 1) + 2);
  knots.front() = 0.0;
  knots.back() = 1.0;
  const double step = 1.0 / static_cast<double>(spans);
  for (std::size_t i = 0; i <= n_spans; ++i) {
    const double u = i == n_spans ? 1.0 : static_cast<double>(i) * step;
    knots[1 + 2 * i] = u;
    knots[2 + 2 * i] = u;
  }

  std::vector<bspline::BsplineSpace1D<double>> dirs;
  dirs.emplace_back(std::span<const double>(knots), detail::circle_degree);
  return bspline::Bspline<double>(bspline::BsplineSpace<double>(std::move(dirs)),
                                  std::span<const double>(cw), true);
}

/// Construct the bilinear B-spline surface of the unit square `[-0.5, 0.5]^2 x {0}` in the
/// xy-plane.
///
/// Returns:
///     bspline::Bspline<double>: A 2D, degree-(1, 1), rank-3, non-rational surface, 2 x 2 control
///     points, the knots `[0, 0, 1, 1]` in each direction.
[[nodiscard]] inline bspline::Bspline<double> create_bilinear() {
  const std::array<double, 12> cp = {-0.5, -0.5, 0.0,   // corners[0, 0]
                                     -0.5, 0.5,  0.0,   // corners[0, 1]
                                     0.5,  -0.5, 0.0,   // corners[1, 0]
                                     0.5,  0.5,  0.0};  // corners[1, 1]
  return bspline::Bspline<double>(detail::linear_space(2), std::span<const double>(cp), false);
}

/// Construct a bilinear B-spline surface from four corner points.
///
/// A degree-(1, 1), non-rational surface with 2 x 2 control points, the corners in the
/// tensor-product order `corners[i, j]` with `i` along `u` and `j` along `v`: `corners[0, 0]` at
/// `(0, 0)`, `corners[1, 0]` at `(1, 0)`, `corners[0, 1]` at `(0, 1)`. A corner of fewer than
/// three coordinates is zero-padded; a rank of 0 gives all-zero corners.
///
/// Args:
///     values (std::span<const double>): The corner coordinates, row-major over `shape`.
///     shape (std::span<const std::size_t>): The extents, `(2, 2, rank)` with `rank <= 3`.
///
/// Returns:
///     bspline::Bspline<double>: A 2D, degree-(1, 1), rank-3, non-rational surface.
///
/// Raises:
///     value_error: `shape` is not `(2, 2, rank)`; `rank` is above 3; `values` does not hold the
///         number of values `shape` implies (checked in that order).
[[nodiscard]] inline bspline::Bspline<double> create_bilinear(std::span<const double> values,
                                                              std::span<const std::size_t> shape) {
  const std::vector<double> cp = detail::pad_corners(values, shape, 2);
  return bspline::Bspline<double>(detail::linear_space(2), std::span<const double>(cp), false);
}

/// Construct the trilinear B-spline volume of the unit cube `[-0.5, 0.5]^3`.
///
/// Returns:
///     bspline::Bspline<double>: A 3D, degree-(1, 1, 1), rank-3, non-rational volume, 2 x 2 x 2
///     control points, the knots `[0, 0, 1, 1]` in each direction.
[[nodiscard]] inline bspline::Bspline<double> create_trilinear() {
  std::vector<double> cp;
  cp.reserve(8 * detail::physical_dim);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        cp.push_back(static_cast<double>(i) - 0.5);
        cp.push_back(static_cast<double>(j) - 0.5);
        cp.push_back(static_cast<double>(k) - 0.5);
      }
    }
  }
  return bspline::Bspline<double>(detail::linear_space(3), std::span<const double>(cp), false);
}

/// Construct a trilinear B-spline volume from eight corner points.
///
/// A degree-(1, 1, 1), non-rational volume with 2 x 2 x 2 control points, the corners in the
/// tensor-product order `corners[i, j, k]` with `i` along `u`, `j` along `v` and `k` along `w`. A
/// corner of fewer than three coordinates is zero-padded; a rank of 0 gives all-zero corners.
///
/// Args:
///     values (std::span<const double>): The corner coordinates, row-major over `shape`.
///     shape (std::span<const std::size_t>): The extents, `(2, 2, 2, rank)` with `rank <= 3`.
///
/// Returns:
///     bspline::Bspline<double>: A 3D, degree-(1, 1, 1), rank-3, non-rational volume.
///
/// Raises:
///     value_error: `shape` is not `(2, 2, 2, rank)`; `rank` is above 3; `values` does not hold
///         the number of values `shape` implies (checked in that order).
[[nodiscard]] inline bspline::Bspline<double> create_trilinear(std::span<const double> values,
                                                               std::span<const std::size_t> shape) {
  const std::vector<double> cp = detail::pad_corners(values, shape, 3);
  return bspline::Bspline<double>(detail::linear_space(3), std::span<const double>(cp), false);
}

}  // namespace pantr::cad
