/// @file operations.hpp
/// @brief Constructive operations: `create_ruled` and `create_revolution`.
///
/// Higher-dimensional B-splines from existing ones: the ruled surface or volume between two
/// B-splines, and the revolution of a curve or surface about an axis. (Extrusion and the
/// translational sweep stay in the Python layer.)
#pragma once

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/reparametrize.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/cad/compat.hpp"
#include "pantr/cad/detail/validation.hpp"
#include "pantr/cad/primitives.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
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

/// The largest parametric dimension of the operations (they add one direction: results are at most
/// volumes).
inline constexpr std::size_t max_dim_for_operations = 2;

/// The threshold on the norm of `cross(axis, z)` below which the rotation axis is taken as
/// (anti)parallel to z.
///
/// Inherited from the Python implementation this replaces, and not derived: nothing here says why
/// `1e-14` rather than a multiple of the unit roundoff. The quantity compared (the sine of the
/// angle between the unit axis and z) is dimensionless, so the constant is at least scale-free. It
/// is kept as it was; a derivation would be the way to change it.
inline constexpr double norm_tol = 1e-14;

/// Normalize a rotation axis to a unit 3D vector.
///
/// Args:
///     axis (std::span<const double>): One to three coordinates, zero-padded.
///
/// Returns:
///     std::array<double, 3>: The coordinates divided by their Euclidean norm. A NaN coordinate
///     gives NaN throughout (the norm is NaN). An infinite one makes the norm infinite: that
///     coordinate becomes NaN (`inf / inf`) and the others 0, so `[inf, 0, 1]` gives `[nan, 0, 0]`.
///
/// Raises:
///     value_error: More than three coordinates; a zero vector.
[[nodiscard]] inline std::array<double, physical_dim> normalize_axis_vector(
    std::span<const double> axis) {
  if (axis.size() > physical_dim) {
    throw value_error("Rotation axis must have at most " + std::to_string(physical_dim) +
                      " coordinates, got " + std::to_string(axis.size()) + ".");
  }
  std::array<double, physical_dim> v = {0.0, 0.0, 0.0};
  std::copy(axis.begin(), axis.end(), v.begin());
  double sum = 0.0;
  for (const double c : v) {
    sum += c * c;
  }
  const double norm = std::sqrt(sum);
  if (norm == 0.0) {
    throw value_error("Rotation axis must be non-zero.");
  }
  for (double& c : v) {
    c /= norm;
  }
  return v;
}

/// Build the transform that translates `pt` to the origin and aligns the axis `v` with z.
///
/// The rotation is by `acos(v_z)` about `cross(v, z)` when that is longer than `norm_tol` (the
/// angle is then `acos` of a number near 1, whose error grows as `1 / sin(angle)`: inherited); a
/// negative `v_z` with no such cross product (the axis is -z) is a rotation by pi about x; an axis
/// along +z is the translation alone. A NaN axis fails both comparisons, so only the translation
/// is applied (the revolution is then about the z axis through `pt`).
///
/// Args:
///     pt (const std::array<double, 3>&): A point on the rotation axis.
///     v (const std::array<double, 3>&): The unit rotation axis.
///
/// Returns:
///     transform::AffineTransform: The translate-then-rotate transform.
[[nodiscard]] inline transform::AffineTransform build_axis_alignment_transform(
    const std::array<double, physical_dim>& pt, const std::array<double, physical_dim>& v) {
  // `cross(v, z)` with z = (0, 0, 1), term by term as NumPy forms it.
  const std::array<double, physical_dim> n = {v[1] - v[2] * 0.0, v[2] * 0.0 - v[0],
                                              v[0] * 0.0 - v[1] * 0.0};
  const std::array<double, physical_dim> minus_pt = {-pt[0], -pt[1], -pt[2]};
  const transform::AffineTransform translate = transform::AffineTransform::translation(minus_pt);
  const double n_norm = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (n_norm > norm_tol) {
    const double gamma = std::acos(std::clamp(v[2], -1.0, 1.0));
    return transform::AffineTransform::rotation_3d(gamma, n).compose(translate);
  }
  if (v[2] < 0.0) {
    // v = -z: flip by a rotation of pi about x.
    const std::array<double, physical_dim> e_x = {1.0, 0.0, 0.0};
    return transform::AffineTransform::rotation_3d(std::numbers::pi, e_x).compose(translate);
  }
  return translate;
}

/// Revolve weighted homogeneous control points about the z axis.
///
/// For each control point `(w x, w y, w z, w)` of the input, with `rho = hypot(w x, w y)` and
/// `theta = atan2(w y, w x)` taken into `[0, 2 pi)`, the arc's control points `a` are mapped by
/// `M = Rz(theta) Tz(w z) Sxy(rho)`, written as the matrix with rows `(rho cos theta, -rho sin
/// theta, 0, 0)`, `(rho sin theta, rho cos theta, 0, 0)`, `(0, 0, 0, w z)`, `(0, 0, 0, 1)`, giving
/// `a M^T` with each weight then multiplied by `w`. Each product is summed over the four columns
/// in order.
///
/// Args:
///     cw (std::span<const double>): The input control points in the z-aligned frame, `count`
///         points of four coordinates, row-major.
///     arc (const bspline::Bspline<double>&): The circular arc (rational, `cp_size() == 4`).
///
/// Returns:
///     std::vector<double>: The revolved control points, extents `(count, arc points, 4)`,
///     row-major.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
[[nodiscard]] inline std::vector<double> revolve_control_points(
    std::span<const double> cw, const bspline::Bspline<double>& arc) {
  constexpr std::size_t width = physical_dim + 1;
  const std::span<const double> aw = arc.control_points();
  const std::size_t n_arc = aw.size() / width;
  const std::size_t count = cw.size() / width;
  std::vector<double> qw(count * n_arc * width);
  for (std::size_t p = 0; p < count; ++p) {
    const double wx = cw[p * width];
    const double wy = cw[p * width + 1];
    const double wz = cw[p * width + 2];
    const double w = cw[p * width + 3];
    const double rho = std::hypot(wx, wy);
    double theta = std::atan2(wy, wx);
    if (theta < 0.0) {
      theta += 2.0 * std::numbers::pi;
    }
    const double r_cos = rho * std::cos(theta);
    const double r_sin = rho * std::sin(theta);
    const std::array<std::array<double, width>, width> m = {{{r_cos, -r_sin, 0.0, 0.0},
                                                             {r_sin, r_cos, 0.0, 0.0},
                                                             {0.0, 0.0, 0.0, wz},
                                                             {0.0, 0.0, 0.0, 1.0}}};
    for (std::size_t a = 0; a < n_arc; ++a) {
      double* out = qw.data() + (p * n_arc + a) * width;
      for (std::size_t i = 0; i < width; ++i) {
        double acc = 0.0;
        for (std::size_t j = 0; j < width; ++j) {
          acc += aw[a * width + j] * m[i][j];
        }
        out[i] = acc;
      }
      out[physical_dim] *= w;
    }
  }
  return qw;
}

}  // namespace detail

/// Construct a ruled surface or volume between two B-splines.
///
/// Interpolates linearly between `bspline1` (at parameter 0) and `bspline2` (at parameter 1) along
/// a new last parametric direction of degree 1 and knots `[0, 0, 1, 1]`. The two inputs are first
/// made compatible with `make_compat`; if one is rational and the other not, the polynomial one
/// gets unit weights.
///
/// The control points of the two inputs are stacked along the new axis, which needs them to have
/// one width after the promotion. One exception is kept from the NumPy implementation this
/// replaces, which broadcast: a second B-spline of control-point width 1 (a rank-1, polynomial
/// one) is repeated over the columns of a wider first. The reverse, and every other mismatch, is
/// refused.
///
/// Args:
///     bspline1 (const bspline::Bspline<T>&): The first boundary (a curve or a surface).
///     bspline2 (const bspline::Bspline<T>&): The second boundary, of the same dimension.
///
/// Returns:
///     bspline::Bspline<T>: A B-spline with one more parametric direction, rational when either
///     input is.
///
/// Raises:
///     value_error: The dimensions differ ("Both B-splines must have the same dim, got ..."); the
///         dimension is above 2 ("ruled requires dim <= 2, got ..."); `make_compat` refuses; the
///         control-point widths differ after the promotion and the second is not of width 1
///         ("ruled requires control points of the same width ..."); the compatible B-splines
///         have different numbers of control points (the guard below).
template <StorageFloat T>
[[nodiscard]] bspline::Bspline<T> create_ruled(const bspline::Bspline<T>& bspline1,
                                               const bspline::Bspline<T>& bspline2) {
  if (bspline1.dim() != bspline2.dim()) {
    throw value_error("Both B-splines must have the same dim, got " +
                      std::to_string(bspline1.dim()) + " and " + std::to_string(bspline2.dim()) +
                      ".");
  }
  if (bspline1.dim() > detail::max_dim_for_operations) {
    throw value_error("ruled requires dim <= " + std::to_string(detail::max_dim_for_operations) +
                      ", got " + std::to_string(bspline1.dim()) + ".");
  }

  const std::array<bspline::Bspline<T>, 2> inputs = {bspline1, bspline2};
  std::vector<bspline::Bspline<T>> made =
      make_compat<T>(std::span<const bspline::Bspline<T>>(inputs));
  const bool rational = made[0].rational() || made[1].rational();
  if (rational) {
    made[0] = detail::promote_to_rational<T>(made[0]);
    made[1] = detail::promote_to_rational<T>(made[1]);
  }
  const bspline::Bspline<T>& b1 = made[0];
  const bspline::Bspline<T>& b2 = made[1];

  const std::size_t width = b1.cp_size();
  const std::size_t width2 = b2.cp_size();
  if (width2 != width && width2 != 1) {
    throw value_error("ruled requires control points of the same width, got " +
                      std::to_string(width) + " and " + std::to_string(width2) +
                      " after promotion (only a second B-spline of width 1 is broadcast).");
  }
  // After `make_compat` the two have one number of basis functions per direction, which the
  // matching of knot classes at a tolerance is meant to guarantee; the stacking below reads one
  // point of the second per point of the first, so a difference would read out of bounds, and the
  // NumPy implementation raised (a broadcast error) there.
  const std::size_t points = b1.size() / width;
  if (b2.size() / width2 != points) {
    throw value_error(
        "ruled requires B-splines with equal numbers of control points after "
        "making them compatible, got " +
        std::to_string(points) + " and " + std::to_string(b2.size() / width2) + ".");
  }

  const std::span<const T> cp1 = b1.control_points();
  const std::span<const T> cp2 = b2.control_points();
  std::vector<T> stacked(points * 2 * width);
  for (std::size_t p = 0; p < points; ++p) {
    T* row = stacked.data() + p * 2 * width;
    for (std::size_t c = 0; c < width; ++c) {
      row[c] = cp1[p * width + c];
      row[width + c] = width2 == 1 ? cp2[p] : cp2[p * width + c];
    }
  }

  std::vector<bspline::BsplineSpace1D<T>> spaces = b1.space().spaces();
  spaces.push_back(detail::linear_space_1d<T>());
  return bspline::Bspline<T>(bspline::BsplineSpace<T>(std::move(spaces)),
                             std::span<const T>(stacked), rational);
}

/// Revolve a B-spline curve or surface about an axis.
///
/// Creates a B-spline with one more parametric direction, the angular one, appended last and with
/// the span structure of `create_circle` (one span per 90 degrees, `C^0` at the junctions). The
/// input is promoted to rational, moved by an affine map to a frame whose z axis is the rotation
/// axis, revolved with the control points of the circular arc, and moved back.
///
/// Args:
///     bspline (const bspline::Bspline<double>&): A curve or a surface of rank 3 (`double` only:
///         the arc is built in `double`).
///     point (std::span<const double>): A point on the axis, up to three coordinates, zero-padded.
///     axis (std::span<const double>): The axis direction, up to three coordinates, zero-padded
///         and normalized here. A NaN coordinate is accepted and revolves about the z axis through
///         `point` (see `build_axis_alignment_transform`).
///     sweep (std::optional<std::array<double, 2>>): `(start, end)` in radians, as for
///         `create_circle`; no value means the full turn.
///
/// Returns:
///     bspline::Bspline<double>: A rational B-spline with `dim + 1` directions.
///
/// Raises:
///     value_error: The dimension is above 2 ("create_revolution requires dim <= 2, got ...");
///         the rank is not 3 ("create_revolution requires rank == 3, got ..."); the point has more
///         than three coordinates; the axis has more than three or is the zero vector ("Rotation
///         axis must be non-zero."); an angle is not finite; (checked in that order).
[[nodiscard]] inline bspline::Bspline<double> create_revolution(
    const bspline::Bspline<double>& bspline, std::span<const double> point,
    std::span<const double> axis, std::optional<std::array<double, 2>> sweep = std::nullopt) {
  if (bspline.dim() > detail::max_dim_for_operations) {
    throw value_error(
        "create_revolution requires dim <= " + std::to_string(detail::max_dim_for_operations) +
        ", got " + std::to_string(bspline.dim()) + ".");
  }
  if (bspline.rank() != detail::physical_dim) {
    throw value_error("create_revolution requires rank == " + std::to_string(detail::physical_dim) +
                      ", got " + std::to_string(bspline.rank()) + ".");
  }

  const std::array<double, detail::physical_dim> pt = detail::pad_to_3d(point);
  const std::array<double, detail::physical_dim> v = detail::normalize_axis_vector(axis);
  const transform::AffineTransform t = detail::build_axis_alignment_transform(pt, v);
  const std::array<std::size_t, 2> matrix_shape = {detail::physical_dim, detail::physical_dim};

  const bspline::Bspline<double> nrb = detail::promote_to_rational<double>(bspline);
  const bspline::Bspline<double> aligned =
      bspline::transformed<double>(nrb, t.matrix(), matrix_shape, t.offset());

  const bspline::Bspline<double> arc = create_circle(1.0, std::nullopt, sweep);
  const std::vector<double> qw = detail::revolve_control_points(aligned.control_points(), arc);

  std::vector<bspline::BsplineSpace1D<double>> spaces = aligned.space().spaces();
  const std::vector<bspline::BsplineSpace1D<double>> arc_spaces = arc.space().spaces();
  spaces.insert(spaces.end(), arc_spaces.begin(), arc_spaces.end());
  const bspline::Bspline<double> revolved(bspline::BsplineSpace<double>(std::move(spaces)),
                                          std::span<const double>(qw), true);

  const transform::AffineTransform back = t.inverse();
  return bspline::transformed<double>(revolved, back.matrix(), matrix_shape, back.offset());
}

}  // namespace pantr::cad
