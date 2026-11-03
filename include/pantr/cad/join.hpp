/// @file join.hpp
/// @brief `join`: the C0 concatenation of two B-splines along a parametric axis.
///
/// The two inputs are made compatible on the other axes, brought to one degree on the join axis,
/// and concatenated; the shared boundary rows are averaged, after the two have been checked to
/// agree. `detail::join` takes the tolerance tier of that check as a `double`, for a binding that
/// accepts B-splines of both storage types.
#pragma once

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/degree.hpp"
#include "pantr/bspline/knot_insertion.hpp"
#include "pantr/bspline/knot_removal.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/cad/compat.hpp"
#include "pantr/cad/detail/grading.hpp"
#include "pantr/cad/detail/validation.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/tolerance.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::cad {

namespace detail {

/// Two B-splines prepared for concatenation.
///
/// Attributes:
///     b1 (bspline::Bspline<T>): The first, open, of the common degree on the join axis.
///     b2 (bspline::Bspline<T>): The second, the same, with its join axis remapped to start where
///         `b1`'s ends.
///     rational (bool): Whether either input was rational (both then are).
template <StorageFloat T>
struct PreparedJoin {
  bspline::Bspline<T> b1;
  bspline::Bspline<T> b2;
  bool rational;
};

/// Prepare two B-splines for joining along an axis.
///
/// Makes them compatible on the non-join axes, promotes to rational if mixed, opens (clamps) a
/// periodic or unclamped B-spline, elevates to the larger degree on the join axis, and remaps `b2`
/// so it starts where `b1` ends.
///
/// Args:
///     bspline1 (const bspline::Bspline<T>&): The first B-spline.
///     bspline2 (const bspline::Bspline<T>&): The second B-spline, of the same dimension.
///     axis (std::size_t): The join axis, `< dim`.
///
/// Returns:
///     PreparedJoin<T>: The two, ready for `concatenate_along_axis`.
///
/// Raises:
///     value_error: `make_compat`, the conversion to open form, the degree elevation or the
///         remapping refuses (their messages).
template <StorageFloat T>
[[nodiscard]] PreparedJoin<T> prepare_for_join(const bspline::Bspline<T>& bspline1,
                                               const bspline::Bspline<T>& bspline2,
                                               std::size_t axis) {
  const std::size_t dim = bspline1.dim();

  // Make compatible on the non-join axes.
  bspline::Bspline<T> b1 = bspline1;
  bspline::Bspline<T> b2 = bspline2;
  std::vector<long long> non_join;
  for (std::size_t i = 0; i < dim; ++i) {
    if (i != axis) {
      non_join.push_back(static_cast<long long>(i));
    }
  }
  if (!non_join.empty()) {
    const std::array<bspline::Bspline<T>, 2> in = {bspline1, bspline2};
    const std::vector<bspline::Bspline<T>> made = cad::make_compat<T>(
        std::span<const bspline::Bspline<T>>(in), std::optional<std::vector<long long>>(non_join));
    b1 = made[0];
    b2 = made[1];
  }

  // Promote to rational if mixed.
  const bool rational = b1.rational() || b2.rational();
  if (rational) {
    b1 = promote_to_rational<T>(b1);
    b2 = promote_to_rational<T>(b2);
  }

  // Clamp if needed.
  const auto needs_open = [](const bspline::Bspline<T>& b) {
    const bspline::BsplineSpace<T> whole = b.space();
    return std::any_of(
        whole.spaces().begin(), whole.spaces().end(),
        [](const bspline::BsplineSpace1D<T>& s) { return s.periodic() || !s.has_open_knots(); });
  };
  if (needs_open(b1)) {
    b1 = bspline::to_open_bspline<T>(b1);
  }
  if (needs_open(b2)) {
    b2 = bspline::to_open_bspline<T>(b2);
  }

  // Elevate to the larger degree on the join axis.
  const std::size_t p1 = b1.degrees()[axis];
  const std::size_t p2 = b2.degrees()[axis];
  const std::size_t p = (std::max)(p1, p2);
  if (p > p1) {
    std::vector<long long> inc(dim, 0);
    inc[axis] = static_cast<long long>(p - p1);
    b1 = bspline::elevate_degree<T>(b1, std::span<const long long>(inc));
  }
  if (p > p2) {
    std::vector<long long> inc(dim, 0);
    inc[axis] = static_cast<long long>(p - p2);
    b2 = bspline::elevate_degree<T>(b2, std::span<const long long>(inc));
  }

  // Remap b2 so it starts where b1 ends.
  const bspline::BsplineSpace<T> whole1 = b1.space();
  const bspline::BsplineSpace<T> whole2 = b2.space();
  const double u1_end = static_cast<double>(whole1.spaces()[axis].domain().second);
  const double u2_start = static_cast<double>(whole2.spaces()[axis].domain().first);
  const double u2_end = static_cast<double>(whole2.spaces()[axis].domain().second);
  const double new_end = u1_end + (u2_end - u2_start);
  b2 = remap_domain<T>(b2, axis, u1_end, new_end);

  return PreparedJoin<T>{std::move(b1), std::move(b2), rational};
}

/// Verify that the two control rows meeting at the junction agree, before averaging them.
///
/// The two rows are not independent data: they are two readings of the same shared face, and
/// `concatenate_along_axis` fuses them by averaging. When they agree the average is that same row
/// and the join is exact; when they do not, it is a new row belonging to neither input, and the
/// result misses both by *half the gap* -- an error that grows with the mistake rather than
/// staying near roundoff. Refusing is the same verdict `create_coons_surface` and
/// `create_coons_volume` reach on the same class of input.
///
/// Both rows are compared in full, not only at the corners of the shared face. Two patches meeting
/// exactly at both ends of the seam and disagreeing in between produce a result that misses them
/// along the whole interior of it, so the corners are necessary but not sufficient. By this point
/// `prepare_for_join` has made the non-join axes compatible, so the two rows span the same space,
/// where a spline is determined by its control points; comparing those is therefore an exact
/// comparison of the two boundary patches and needs no sampling.
///
/// The tolerance is `verify_corners_2d`'s tier (coons.hpp; its justification lives there and is
/// not restated) times the largest absolute value over the two rows compared, applied to what is
/// graded here, which is a control-point coefficient rather than a point on a curve.
///
/// **Why that magnitude.** Not because a genuinely shared seam arrives bitwise equal; that is true
/// only when one side is refined and the other left alone. When each input carries its own
/// interior knots on a non-join axis -- two surfaces meeting at a seam, each with its own
/// transverse knot vector, which is the ordinary case -- `make_compat` refines *both*, by
/// different operator sequences, and the readings differ. Every step between the caller's data
/// and this comparison (rational promotion, join-axis elevation, which leaves the clamped row
/// fixed, and non-join-axis elevation and insertion) is a convex combination, so its absolute
/// rounding error is bounded by `c * eps * max|coefficient|` with `max|coefficient|`
/// non-increasing under convex combination. The floor is therefore `O(eps) * scale`, and measured
/// it is **0.5 to 3.3 ulps** of `scale` -- flat in the number of insertions, since knot insertion
/// is convex and the two sides' errors are correlated rather than independent -- across degrees 2
/// to 5 and magnitudes `1e-9` to `1e15`. Allowing a few ulps of build slack for FMA, vector width
/// and libm, `16 * eps * scale` covers it and the conservative tier clears that by a further
/// factor of 250. That margin is the tier's own robustness reading, not an accident: the slack is
/// for seam data the caller *computed* rather than copied -- a transform, an intersection, a
/// reparametrization -- and is sized to catch pieces given in the wrong place or the wrong
/// orientation, which miss by a fraction of the model rather than by epsilons.
///
/// This is also what justifies the `* scale` multiplier twice over: `max|row|` is both the
/// magnitude that transports `eps` into the compared space (verified proportional over 24
/// decades) and, by the convex hull below, the geometric magnitude it maps onto.
///
/// **The scale is the seam's, not the model's**, since the two rows are the whole of what is
/// compared. That is never looser than the Coons rule, whose population spans the whole skeleton,
/// because the seam is part of the model. It is stricter, and the contract it puts on the caller
/// is worth stating: *the seam must be supplied to within the tier times its own magnitude, not
/// the model's*. A seam near the origin on a large model, whose coordinates came out of a
/// model-scale chain and so carry model-scale absolute error, is refused. Erring that way is
/// deliberate; it is the direction that reports rather than the direction that accepts.
///
/// **Each column group is graded against its own magnitude.** A rational row carries homogeneous
/// coordinates, in units of length times weight, beside a dimensionless weight. One max over the
/// row would grade the weight against a threshold proportional to the model's *length*, which is
/// dimensionally wrong and breaks scale invariance: sending `x -> lambda*x` scales the coordinates
/// and leaves the weights alone. Measured on a seam containing a control point at the origin, the
/// accepted geometric error then grows linearly with coordinate magnitude, from 0.12 times the
/// tier at magnitude 1 to **1.2e7 times the tier at 1e8**, while the non-rational control stays
/// flat at 0.50 at every magnitude. Grading the coordinate columns against `max|X|` and the weight
/// column against `max|w|` restores it: under `x -> lambda*x` both group scales and both group
/// gaps carry `lambda`, and under a global weight rescale `w -> mu*w` the homogeneous coordinates
/// `X = w*x` carry `mu` alongside the weights, so both groups carry it too. It does not disarm the
/// check: a *one-sided* weight rescale still leaves `gap_w = (mu - 1)*w` against `tier * max(w,
/// mu*w)` and is still refused. The weight column's own roundoff floor against `max|w|` is
/// measured at `<= 3.2 * eps`, independent of magnitude, degree and refinement, so the tier clears
/// it by about 1290.
///
/// The tier is evaluated at the **coarser of the two inputs' precisions** (the caller passes it),
/// not at `double`, because the library's B-splines are `float` as well. If the true shared face
/// is `F`, the two stored readings satisfy `|row_d - F| <= 0.5 * eps_d * |F|`, so for a genuinely
/// shared face `|row1 - row2| <= 0.5 * (eps_1 + eps_2) * scale`, which is about `0.5 * eps_coarse
/// * scale`. That is a hard floor on any achievable agreement: any tolerance below it refuses a
/// perfect seam. The tier clears it by 8192. Taking the *promoted* type instead would be exactly
/// backwards, missing the floor by a factor of `5.4e8` for a mixed pair. The consequence is worth
/// stating plainly, since "the same tier as the Coons path" reads as "the same strictness" and it
/// is not: at `float` the tier is `4.88e-4` relative, so a `float` join accepts a seam gap of
/// `1e-5` of the seam scale and refuses `1e-3`. It still catches what the check exists for; it
/// does not catch a small displacement error that `double` would refuse. The rule also grades by
/// *container* rather than by provenance, so `float`-resolution data already copied into a
/// `double` container is graded at `double`.
///
/// For non-rational inputs the basis is non-negative and sums to one -- and a tensor product of
/// non-negative partitions of unity is one -- so the largest control-point gap bounds the largest
/// gap between the two boundary patches themselves: the number reported is then a distance in
/// space. Measured over 400 random pairs the ratio saturates at exactly 1. For rational inputs it
/// is not: the comparison is of **homogeneous** control points, which is what the average
/// combines, and turning that into a distance would need a lower bound on the weights that this
/// function does not compute; measured, the ratio reaches `9e4` at weights of 0.01. Comparing them
/// homogeneously is still the right test -- two rows that project to the same points under
/// different weights average to a row whose weights are neither input's, so the curve on both
/// sides of the seam is not the curve either input described -- but the reported gap is then a
/// coefficient gap.
///
/// One consequence of comparing coefficients: what is required is that the two boundary patches
/// agree **as parametrized**, which is strictly stronger than agreeing as point sets. That is the
/// right condition for a join, whose result has to be one spline, but it means two seams tracing
/// the same curve at different speeds are refused.
///
/// The comparison is `!(gap <= tol)` to refuse, so a NaN gap is refused.
///
/// Args:
///     row1 (std::span<const T>): The first B-spline's control row at the end of the join axis,
///         with extents `row_shape` (the join axis removed from the control net, the coefficient
///         last), row-major.
///     row2 (std::span<const T>): The second B-spline's control row at the start of the join axis.
///     row1_shape (std::span<const std::size_t>): The extents of `row1`.
///     row2_shape (std::span<const std::size_t>): The extents of `row2`.
///     axis (long long): The join axis, named in the message so a caller can tell which seam
///         failed.
///     rational (bool): Whether the rows are homogeneous, i.e. whether the trailing column is a
///         weight and has to be graded against its own magnitude.
///     tier (double): The relative tolerance tier.
///
/// Raises:
///     value_error: The two rows disagree anywhere on the shared face, in either column group, by
///         more than that group's tolerance; or the two rows differ in shape (a different number
///         of control points on a non-join axis after `make_compat`, or control points of a
///         different number of coordinates; the reference implementation raised a NumPy broadcast
///         error at the subtraction).
template <StorageFloat T>
void verify_shared_boundary(std::span<const T> row1, std::span<const T> row2,
                            std::span<const std::size_t> row1_shape,
                            std::span<const std::size_t> row2_shape, long long axis, bool rational,
                            double tier) {
  if (!std::equal(row1_shape.begin(), row1_shape.end(), row2_shape.begin(), row2_shape.end())) {
    throw value_error("Shared boundary mismatch on axis " + std::to_string(axis) +
                      ": the two B-splines have junction faces of different shapes after being "
                      "made compatible, " +
                      py_repr(row1_shape) + " and " + py_repr(row2_shape) + ".");
  }
  const std::size_t width = row1_shape.back();
  const std::span<const std::size_t> face_shape = row1_shape.first(row1_shape.size() - 1);

  // Coordinates first: a seam given in the wrong place fails there, and it is the more
  // informative half to report when both groups are out.
  for (std::size_t g = 0; g < group_count(rational); ++g) {
    const ColumnGroup group = group_at(rational, g, width);
    // No floor: a group whose two halves are all zero has no scale, and is also bitwise equal, so
    // a zero tolerance is the right answer there rather than a hazard.
    const double scale =
        py_max(max_abs_in<T>(row1, width, group), max_abs_in<T>(row2, width, group));
    const double tol = tier * scale;
    const GroupGap graded = group_gap<T>(row1, row2, width, group);
    if (graded.gap <= tol) {
      continue;
    }

    // Report the control point that disagrees most, so the gap named is the gap compared. The
    // whole point is printed even though one group decided the verdict.
    const std::vector<std::size_t> index = unravel(graded.index, face_shape);
    // A curve's shared face is a single control point, whose index is the empty tuple: printing it
    // would read like a defect rather than like a location.
    const std::string where =
        index.empty() ? std::string() : " at index " + py_repr(std::span<const std::size_t>(index));
    throw value_error(
        "Shared boundary mismatch on axis " + std::to_string(axis) + where +
        ": the first B-spline gives " + row_repr<T>(row1.subspan(graded.index * width, width)) +
        ", the second gives " + row_repr<T>(row2.subspan(graded.index * width, width)) + " (gap " +
        py_format_exp(graded.gap, 3) + ", above " + py_format_exp(tol, 3) + " at boundary " +
        std::string(group.name) + " scale " + py_format_exp(scale, 3) + ").");
  }
}

/// Try removing junction knots to increase smoothness.
///
/// Attempts to remove up to `p - 1` copies of the junction knot (`remove_knots` at its default
/// deviation budget, the round-off floor, which removes what is redundant to within the
/// arithmetic and nothing more).
///
/// Args:
///     b (const bspline::Bspline<T>&): The joined B-spline.
///     axis (std::size_t): The join axis.
///     u_junction (T): The knot value at the junction.
///     p (int): The degree on the join axis, at least 2.
///
/// Returns:
///     bspline::Bspline<T>: The B-spline with some junction knots removed, or equal to the
///     original when removal was not possible.
///
/// Raises:
///     value_error: `remove_knots` refuses (its messages).
template <StorageFloat T>
[[nodiscard]] bspline::Bspline<T> try_remove_junction_knots(const bspline::Bspline<T>& b,
                                                            std::size_t axis, T u_junction, int p) {
  std::vector<std::span<const T>> per_dim(b.dim());
  per_dim[axis] = std::span<const T>(&u_junction, 1);
  return bspline::remove_knots<T>(b, std::span<const std::span<const T>>(per_dim),
                                  static_cast<long long>(p - 1), std::nullopt);
}

/// Concatenate two prepared B-splines along an axis.
///
/// Verifies that the two shared boundary control rows agree, averages them, merges the knot
/// vectors, and builds the result. The junction knots take the incoming knots' storage type, and
/// the result is `C0` there: the junction knot has multiplicity `p`, and the removal of up to `p -
/// 1` of them is then tried when `p > 1`.
///
/// Args:
///     b1 (const bspline::Bspline<T>&): The first B-spline (left side).
///     b2 (const bspline::Bspline<T>&): The second B-spline (right side, already remapped).
///     axis (std::size_t): The join axis.
///     rational (bool): Whether the result is rational.
///     tier (double): The relative tolerance tier of the shared-boundary check.
///
/// Returns:
///     bspline::Bspline<T>: The concatenated B-spline.
///
/// Raises:
///     value_error: The two shared boundary control rows disagree, as `verify_shared_boundary`
///         grades it; or the removal of knots refuses.
template <StorageFloat T>
[[nodiscard]] bspline::Bspline<T> concatenate_along_axis(const bspline::Bspline<T>& b1,
                                                         const bspline::Bspline<T>& b2,
                                                         std::size_t axis, bool rational,
                                                         double tier) {
  const bspline::BsplineSpace<T> whole1 = b1.space();
  const bspline::BsplineSpace<T> whole2 = b2.space();
  const int p = whole1.spaces()[axis].degree();
  const std::span<const T> cp1 = b1.control_points();
  const std::span<const T> cp2 = b2.control_points();
  const std::span<const std::size_t> shape1 = b1.shape();
  const std::span<const std::size_t> shape2 = b2.shape();

  // The control net is row-major over (n_0, ..., n_{dim-1}, width): `outer` slabs before the join
  // axis, each holding `n` rows of `inner` values (the later axes and the coefficient).
  const std::size_t n1 = shape1[axis];
  const std::size_t n2 = shape2[axis];
  std::size_t outer1 = 1;
  std::size_t inner1 = 1;
  for (std::size_t d = 0; d < shape1.size(); ++d) {
    if (d < axis) {
      outer1 *= shape1[d];
    } else if (d > axis) {
      inner1 *= shape1[d];
    }
  }
  std::size_t outer2 = 1;
  std::size_t inner2 = 1;
  for (std::size_t d = 0; d < shape2.size(); ++d) {
    if (d < axis) {
      outer2 *= shape2[d];
    } else if (d > axis) {
      inner2 *= shape2[d];
    }
  }

  // The rows at the junction: the last row of b1 and the first of b2, with the join axis removed.
  std::vector<std::size_t> row1_shape;
  std::vector<std::size_t> row2_shape;
  for (std::size_t d = 0; d < shape1.size(); ++d) {
    if (d != axis) {
      row1_shape.push_back(shape1[d]);
    }
  }
  for (std::size_t d = 0; d < shape2.size(); ++d) {
    if (d != axis) {
      row2_shape.push_back(shape2[d]);
    }
  }
  std::vector<T> row1(outer1 * inner1);
  std::vector<T> row2(outer2 * inner2);
  for (std::size_t o = 0; o < outer1; ++o) {
    std::copy_n(cp1.begin() + static_cast<std::ptrdiff_t>((o * n1 + (n1 - 1)) * inner1), inner1,
                row1.begin() + static_cast<std::ptrdiff_t>(o * inner1));
  }
  for (std::size_t o = 0; o < outer2; ++o) {
    std::copy_n(cp2.begin() + static_cast<std::ptrdiff_t>((o * n2) * inner2), inner2,
                row2.begin() + static_cast<std::ptrdiff_t>(o * inner2));
  }
  verify_shared_boundary<T>(
      std::span<const T>(row1), std::span<const T>(row2), std::span<const std::size_t>(row1_shape),
      std::span<const std::size_t>(row2_shape), static_cast<long long>(axis), rational, tier);

  // Exact when the two rows agree, and they are known to by now: (x + x) / 2 returns x for every
  // finite x whose sum does not overflow. Averaging rather than picking a side keeps the result
  // symmetric in its two inputs.
  std::vector<T> mid(row1.size());
  for (std::size_t i = 0; i < mid.size(); ++i) {
    mid[i] = (row1[i] + row2[i]) / static_cast<T>(2);
  }

  // The rows agree in shape (verified), so the two nets differ only along the join axis.
  std::vector<T> new_cp;
  new_cp.reserve(outer1 * (n1 + n2 - 1) * inner1);
  for (std::size_t o = 0; o < outer1; ++o) {
    const auto left_begin = cp1.begin() + static_cast<std::ptrdiff_t>(o * n1 * inner1);
    new_cp.insert(new_cp.end(), left_begin,
                  left_begin + static_cast<std::ptrdiff_t>((n1 - 1) * inner1));
    new_cp.insert(new_cp.end(), mid.begin() + static_cast<std::ptrdiff_t>(o * inner1),
                  mid.begin() + static_cast<std::ptrdiff_t>((o + 1) * inner1));
    const auto right_begin = cp2.begin() + static_cast<std::ptrdiff_t>((o * n2 + 1) * inner1);
    new_cp.insert(new_cp.end(), right_begin,
                  right_begin + static_cast<std::ptrdiff_t>((n2 - 1) * inner1));
  }

  // Merge the knot vectors. The junction knots take the incoming knots' storage type, so the
  // knot vector stays in `T` along with the control points.
  const T u_junction = whole1.spaces()[axis].domain().second;
  const std::span<const T> k1 = whole1.spaces()[axis].knots();
  const std::span<const T> k2 = whole2.spaces()[axis].knots();
  const std::size_t skip = static_cast<std::size_t>(p) + 1;
  std::vector<T> new_knots(k1.begin(), k1.end() - static_cast<std::ptrdiff_t>(skip));
  new_knots.insert(new_knots.end(), static_cast<std::size_t>(p), u_junction);
  new_knots.insert(new_knots.end(), k2.begin() + static_cast<std::ptrdiff_t>(skip), k2.end());

  std::vector<bspline::BsplineSpace1D<T>> spaces = whole1.spaces();
  spaces[axis] = bspline::BsplineSpace1D<T>(std::span<const T>(new_knots), p);
  const bspline::Bspline<T> result(bspline::BsplineSpace<T>(std::move(spaces)),
                                   std::span<const T>(new_cp), rational);

  // Try to remove junction knots for a smoother join.
  if (p > 1) {
    return try_remove_junction_knots<T>(result, axis, u_junction, p);
  }
  return result;
}

/// Join two B-splines along a parametric axis, grading the seam at a given tier.
///
/// The body of `join`; see there for the construction and its refusals. The tier is the relative
/// tolerance of the shared-boundary comparison (`conservative_tier` of the coarser of the inputs'
/// storage types as supplied).
///
/// Args:
///     bspline1 (const bspline::Bspline<T>&): The first B-spline (left side of the join).
///     bspline2 (const bspline::Bspline<T>&): The second B-spline (right side of the join).
///     axis (long long): The parametric axis along which to join (0-indexed).
///     tier (double): The relative tolerance tier of the shared-boundary check.
///
/// Returns:
///     bspline::Bspline<T>: A single B-spline spanning both inputs, in `T`.
///
/// Raises:
///     value_error: As `join`.
template <StorageFloat T>
[[nodiscard]] bspline::Bspline<T> join(const bspline::Bspline<T>& bspline1,
                                       const bspline::Bspline<T>& bspline2, long long axis,
                                       double tier) {
  const std::size_t dim = bspline1.dim();
  if (dim != bspline2.dim()) {
    throw value_error("Both B-splines must have the same dim, got " +
                      std::to_string(bspline1.dim()) + " and " + std::to_string(bspline2.dim()) +
                      ".");
  }
  if (axis < 0 || axis >= static_cast<long long>(dim)) {
    throw value_error("axis must be in [0, " + std::to_string(dim) + "), got " +
                      std::to_string(axis) + ".");
  }
  const auto axis_index = static_cast<std::size_t>(axis);

  PreparedJoin<T> prepared = prepare_for_join<T>(bspline1, bspline2, axis_index);
  return concatenate_along_axis<T>(prepared.b1, prepared.b2, axis_index, prepared.rational, tier);
}

}  // namespace detail

/// Join two B-splines along a parametric axis with C0 continuity.
///
/// The two inputs are made compatible on all non-join axes (degree and knots), elevated to the
/// same degree on the join axis, and then concatenated. The shared boundary control points are
/// averaged.
///
/// Optionally, knots are removed at the junction to recover higher smoothness when the geometry
/// permits it.
///
/// **The two inputs must already meet.** The face where they join is not independent data:
/// `bspline1`'s boundary there and `bspline2`'s are two readings of the same thing, and averaging
/// them is exact only because they agree. Pieces that do not meet are refused rather than
/// reconciled, since the average of two disagreeing rows lies on neither input and the result
/// would then pass through neither, missing each by half the gap. For rational inputs -- including
/// a rational joined to a polynomial, which is promoted first -- agreement is required of the
/// **homogeneous** control points, weights included, so two boundaries that describe the same
/// points of space under different weights are refused too.
///
/// The two must have control points of one rank (a rank-2 curve cannot be joined to a rank-3
/// one).
///
/// Args:
///     bspline1 (const bspline::Bspline<T>&): The first B-spline (left side of the join).
///     bspline2 (const bspline::Bspline<T>&): The second B-spline (right side of the join).
///     axis (long long): The parametric axis along which to join (0-indexed).
///
/// Returns:
///     bspline::Bspline<T>: A single B-spline spanning both inputs.
///
/// Raises:
///     value_error: The inputs have different parametric dimensions ("Both B-splines must have the
///         same dim, got ..."); `axis` is out of range ("axis must be in [0, dim), got ...");
///         the two inputs disagree anywhere on the face they are joined along, by more than `4096
///         * eps` times the largest absolute control-point value over the two boundary rows
///         compared. This is `create_coons_surface`'s tier, justified in
///         `detail::verify_corners_2d`, with two differences that follow from what is graded here
///         and are derived in `detail::verify_shared_boundary`: the scale is taken over the
///         **seam** rather than over the whole boundary skeleton, so a seam must agree to within
///         the tier times its own magnitude, not the model's; and for a rational join the
///         homogeneous coordinates and the weights are graded against their own magnitudes
///         separately, since they do not share a physical dimension. Also: the inputs' control
///         points differ in rank or, after making them compatible, in number; or `make_compat`,
///         the conversion to open form, the degree elevation, the remapping or the knot removal
///         refuses (their messages).
template <StorageFloat T>
[[nodiscard]] bspline::Bspline<T> join(const bspline::Bspline<T>& bspline1,
                                       const bspline::Bspline<T>& bspline2, long long axis) {
  return detail::join<T>(bspline1, bspline2, axis, detail::conservative_tier<T>());
}

}  // namespace pantr::cad
