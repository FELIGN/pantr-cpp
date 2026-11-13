/// @file coons.hpp
/// @brief Coons patch and volume constructions: bilinear blending of four boundary curves and
///        trilinear blending of six boundary faces.
///
/// `create_coons_surface` builds a surface from its four boundary curves, `create_coons_volume` a
/// volume from its six boundary faces. Both refuse boundaries that disagree where they meet, and
/// both blend rational boundaries in homogeneous space (see each function).
///
/// Each has a `detail::` overload that takes the tolerance tier of the comparisons as a `double`:
/// the public function passes the tier of its own storage type, and a binding that accepts
/// B-splines of both storage types converts them to one and passes the tier of the coarsest.
#pragma once

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/reparametrize.hpp"
#include "pantr/bspline/slice.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/cad/compat.hpp"
#include "pantr/cad/detail/grading.hpp"
#include "pantr/cad/detail/validation.hpp"
#include "pantr/cad/operations.hpp"
#include "pantr/cad/primitives.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/tolerance.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pantr::cad {

namespace detail {

/// The names of a Coons volume's parametric directions, in axis order.
inline constexpr std::array<char, 3> direction_names = {'u', 'v', 'w'};

/// One edge of a Coons volume, as read off each of the two faces that carry it.
///
/// Two faces share a whole edge, so each of the twelve edges can be read twice, and the two
/// readings must agree for the volume to interpolate both faces.
///
/// Attributes:
///     label (std::string): Edge name, `{free}_{fixed}{side}_{fixed}{side}` (e.g. `w_u0_v1`).
///     face_a (std::string): Name of the first face carrying the edge (e.g. `face_u0`).
///     curve_a (bspline::Bspline<T>): The edge as `face_a` sees it.
///     face_b (std::string): Name of the second face carrying the edge.
///     curve_b (bspline::Bspline<T>): The edge as `face_b` sees it.
///     corners (std::array<std::string, 2>): Labels of the volume corners at the low and high
///         ends of the edge, in the `(i,j,k)` form of `corner_label`.
template <StorageFloat T>
struct EdgeReadings {
  std::string label;
  std::string face_a;
  bspline::Bspline<T> curve_a;
  std::string face_b;
  bspline::Bspline<T> curve_b;
  std::array<std::string, 2> corners;
};

/// Name a corner of a Coons volume by the side it occupies in each direction.
///
/// Args:
///     sides (const std::array<int, 3>&): Side (0 or 1) along u, v and w.
///
/// Returns:
///     std::string: Corner label `(i,j,k)`, matching the `(i,j)` form of the 2D messages.
[[nodiscard]] inline std::string corner_label(const std::array<int, 3>& sides) {
  return "(" + std::to_string(sides[0]) + "," + std::to_string(sides[1]) + "," +
         std::to_string(sides[2]) + ")";
}

/// Make a pair of B-splines compatible (`make_compat` of the two).
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): the callers check the
///     parametric dimension, and the refusals are `make_compat`'s alone.
template <StorageFloat T>
[[nodiscard]] std::array<bspline::Bspline<T>, 2> compat_pair(const bspline::Bspline<T>& a,
                                                             const bspline::Bspline<T>& b) {
  const std::array<bspline::Bspline<T>, 2> in = {a, b};
  std::vector<bspline::Bspline<T>> made =
      cad::make_compat<T>(std::span<const bspline::Bspline<T>>(in), std::nullopt);
  return {made[0], made[1]};
}

/// Linearly combine control points of compatible B-splines.
///
/// The sum is accumulated in the B-spline's storage type, starting from zero, and the result gets
/// the **first** term's space. Every term is built in the precision of the data it was read from,
/// including the corner interpolant, so nothing in the sum is promoted or narrowed: a `float`
/// model stays `float` (accumulating in `double` and handing back `double` coefficients for a
/// `float` space would not build, and would be a precision escalation no other operation of the
/// library performs; `create_ruled` and `make_compat` stay in the type they were given). If any
/// term is rational, every term is promoted to rational first (unit weights).
///
/// Args:
///     terms (const std::vector<bspline::Bspline<T>>&): B-splines that share one space (after
///         `make_compat`); the first decides the space of the result.
///     signs (std::span<const int>): The coefficient (+1 or -1) of each term.
///
/// Returns:
///     bspline::Bspline<T>: The combination, rational when any term is.
///
/// Raises:
///     value_error: The terms' control nets have different shapes. After `make_compat` they have
///         one number of basis functions per direction, which the matching of knot classes at a
///         tolerance is meant to guarantee, but a knot a rounding away from another can leave
///         different numbers (the case `create_ruled` refuses too); the reference implementation
///         raised a NumPy broadcast error there.
template <StorageFloat T>
[[nodiscard]] bspline::Bspline<T> combine_terms(const std::vector<bspline::Bspline<T>>& terms,
                                                std::span<const int> signs) {
  const bool rational = std::any_of(terms.begin(), terms.end(),
                                    [](const bspline::Bspline<T>& t) { return t.rational(); });
  std::vector<bspline::Bspline<T>> all;
  all.reserve(terms.size());
  for (const auto& t : terms) {
    all.push_back(rational ? promote_to_rational<T>(t) : t);
  }
  for (std::size_t i = 1; i < all.size(); ++i) {
    // Guards an out-of-bounds read of the later terms in the sum below. No input reaching it has
    // been constructed: the edge check fires first on the inputs tried.
    const std::span<const std::size_t> s0 = all[0].shape();
    const std::span<const std::size_t> si = all[i].shape();
    if (!std::equal(s0.begin(), s0.end(), si.begin(), si.end())) {
      throw value_error(
          "Coons blend terms have control nets of different shapes after being made "
          "compatible, got " +
          py_repr(s0) + " and " + py_repr(si) + ".");
    }
  }
  std::vector<T> cp(all[0].size(), static_cast<T>(0));
  for (std::size_t t = 0; t < all.size(); ++t) {
    const std::span<const T> x = all[t].control_points();
    const T sign = static_cast<T>(signs[t]);
    for (std::size_t i = 0; i < cp.size(); ++i) {
      cp[i] += sign * x[i];
    }
  }
  return bspline::Bspline<T>(all[0].space(), std::span<const T>(cp), rational);
}

/// Build the corner term of a Coons blend from its `2^dim` corner coefficients.
///
/// `create_bilinear` and `create_trilinear` cannot serve here, for two reasons that both come from
/// their being public *geometric* primitives: they cap the rank at three, so they reject the
/// homogeneous `(w x, w y, w z, w)` corner of a rational boundary, and they build their space in
/// `double` whatever they were handed, which a `float` model cannot then be combined with.
///
/// One construction serves the patch's bilinear term and the volume's trilinear one; both are the
/// construction `build_bilinear_volume` uses for its two blended directions. Keeping them one
/// function is what stops the two paths drifting apart: a corner term built from projected points
/// beside two homogeneous terms does not cancel in the inclusion-exclusion, and the blend misses
/// its boundary curves (pantr issue 327).
///
/// Args:
///     corners (std::span<const T>): `2^dim` corners of `width` coefficients, row-major, indexed
///         by the side along u, v (and w), the coefficient last, as `extract_corners_2d` and
///         `extract_corners_3d` return them.
///     dim (std::size_t): The parametric dimension, 2 or 3.
///     rational (bool): Whether the corners carry a trailing weight.
///
/// Returns:
///     bspline::Bspline<T>: A `dim`-dimensional B-spline of degree 1 in every direction, in the
///     axis order the corners are indexed by and in their precision.
template <StorageFloat T>
[[nodiscard]] bspline::Bspline<T> build_multilinear_corner_term(std::span<const T> corners,
                                                                std::size_t dim, bool rational) {
  std::vector<bspline::BsplineSpace1D<T>> spaces;
  spaces.reserve(dim);
  for (std::size_t d = 0; d < dim; ++d) {
    spaces.push_back(linear_space_1d<T>());
  }
  return bspline::Bspline<T>(bspline::BsplineSpace<T>(std::move(spaces)), corners, rational);
}

/// Extract a Coons patch's 4 corner coefficients from the two curves free in u.
///
/// The array is read and sized in **one** space, which is what the corner term of the blend needs.
/// `bspline::boundary_point` **projects**, so reading a rational corner through it and promoting
/// the result afterwards gives `(x, y, z, 1)` where the other two terms carry `(w x, w y, w z,
/// w)`; the three-term inclusion-exclusion then fails to cancel and the patch misses the curves it
/// was built from by a fraction of the model. Measured with four curves agreeing homogeneously at
/// every corner and all four corner weights 2, that was 2.2% of the model, with nothing refused
/// (pantr issue 327). `homogeneous_endpoint` is the read that stays in the space the width was
/// taken from, as it is for the volume's `extract_corners_3d`.
///
/// `rational` is the **patch's**, not these two curves': the corner term is one of three that are
/// added together, so it has to live in the same space as the other two even when the u-curves are
/// polynomial and only a v-curve is rational. Promoting them gives those corners a unit weight,
/// which is the right value exactly when the rational curves carry a unit weight there too, and
/// `verify_corners_2d` requires precisely that, since it compares each corner's two readings
/// homogeneously and a polynomial reading promotes to weight one.
///
/// Args:
///     u_curves (const std::array<bspline::Bspline<T>, 2>&): `(C_u0, C_u1)`, the boundaries at
///         `v = 0` and `v = 1`, each free in u and already made compatible with each other.
///     rational (bool): Whether the patch being built is rational, i.e. whether the corners are
///         wanted as homogeneous coefficients.
///
/// Returns:
///     std::vector<T>: Corner coefficients with extents `(2, 2, width)`, row-major, indexed
///     `[i, j]` by side along u and v, where `width` counts the weight when `rational`.
///
/// Raises:
///     value_error: The two curves' control points have different widths (the reference
///         implementation raised a NumPy broadcast error filling the array).
template <StorageFloat T>
[[nodiscard]] std::vector<T> extract_corners_2d(const std::array<bspline::Bspline<T>, 2>& u_curves,
                                                bool rational) {
  std::array<bspline::Bspline<T>, 2> curves = u_curves;
  if (rational) {
    curves = {promote_to_rational<T>(curves[0]), promote_to_rational<T>(curves[1])};
  }
  const std::size_t width = curves[0].cp_size();
  if (curves[1].cp_size() != width) {
    throw value_error("create_coons_surface requires curves of one rank, got control points of " +
                      std::to_string(width) + " and " + std::to_string(curves[1].cp_size()) +
                      " coordinates (weights included) on C_u0 and C_u1.");
  }
  std::vector<T> corners(2 * 2 * width);
  for (std::size_t j = 0; j < 2; ++j) {
    for (long long i = 0; i < 2; ++i) {
      const NdArray<T> end = homogeneous_endpoint<T>(curves[j], i);
      std::copy(end.flat().begin(), end.flat().end(),
                corners.begin() +
                    static_cast<std::ptrdiff_t>((static_cast<std::size_t>(i) * 2 + j) * width));
    }
  }
  return corners;
}

/// One corner of a Coons patch, as read off each of the two curves that carry it.
///
/// Two curves meet at each corner of a patch, so each of the four corners can be read twice, and
/// the two readings must agree, in homogeneous data, weights included, for the patch to
/// interpolate both curves.
///
/// Attributes:
///     label (std::string): Corner name `(i,j)`, by side along u and v.
///     u_curve (std::string_view): Name of the u-curve carrying it, as the mismatch message
///         prints it.
///     from_u (std::vector<T>): The corner as `u_curve` sees it, in that curve's own precision.
///     v_curve (std::string_view): Name of the v-curve carrying it.
///     from_v (std::vector<T>): The corner as `v_curve` sees it, in that curve's own precision.
template <StorageFloat T>
struct CornerReadings {
  std::string label;
  std::string_view u_curve;
  std::vector<T> from_u;
  std::string_view v_curve;
  std::vector<T> from_v;
};

/// Verify that the corners read off the u-curves match what the v-curves say there.
///
/// The tolerance is `tier * scale` with `scale` that column group's own largest absolute value
/// over **all eight** corner readings, not just the pair under test, and `tier` the conservative
/// tier at the coarsest of the eight readings' own precisions (the caller passes it).
///
/// **Why the conservative tier.** Both curves meeting at a corner read it straight off a clamped
/// control point, and neither knot insertion nor degree elevation moves a clamped endpoint, so
/// `make_compat` leaves it alone: two curves that genuinely share a corner produce **bitwise
/// equal** values and need no tolerance at all. The slack exists only for a corner the caller
/// *computed* rather than copied -- an intersection, a reparametrization, a transform -- and the
/// length of that chain is not something this function can see. So the threshold is not sized
/// against a known accumulation; it is sized to catch the mistake the check exists for, which is
/// curves given in the wrong order or the wrong orientation, missing by a fraction of the patch
/// rather than by epsilons. Failing the other way is the expensive one: a `value_error` on a
/// geometrically perfect patch. The conservative tier is the widest the table sanctions without a
/// derivation of its own, and at unit scale it lands on `9.1e-13` for `double`, within a factor of
/// 1.1 of an absolute `1e-12`. Taking the scale over all eight is what makes the verdict
/// independent of which corner happens to sit at the origin: a corner at `(0, 0, 0)` supplies no
/// scale of its own, and grading it against zero would demand bitwise agreement of a coordinate
/// the caller may well have computed.
///
/// A fixed absolute tolerance cannot do this. Measured across twelve decades of `1e-12 / L`, a
/// one-ulp corner mismatch was rejected at `L >= 1e6` -- a micron-unit model of a metre-scale part
/// failing on a geometrically perfect patch -- while a `1e-9` *relative* gap was accepted at `L =
/// 1e-6`.
///
/// **The tier is read at the readings' own precision**, not at `double`, because the library's
/// B-splines are `float` as well: `conservative_tier` carries that derivation. At `double` the
/// tier is `9.09e-13` relative while one `float` ulp at magnitude 1 is `1.19e-07`, so a `float`
/// patch whose corner is as close as its format allows was refused for a disagreement it cannot
/// even express, by a factor of `eps32 / (4096 eps64) = 2^17 = 131072`. (Pantr issue 319 states
/// `5.4e5` here; that is the ratio of the two *epsilons* divided by 4096 done the wrong way round
/// -- `eps32 / eps64` is `2^29 = 5.4e8` -- and the figure against the *tier* is `1.3e5`. The test
/// "the float tier is the only one a float corner can clear" pins it.)
///
/// **A rational patch is compared homogeneously, in two column groups.** The three terms of the
/// blend are formed from homogeneous coefficients and projected only on evaluation, which is what
/// makes the boundary property survive projection -- but only if the four curves agree in
/// homogeneous data. A homogeneous representation is unique just up to one constant scaling all of
/// a patch's weights, so two curves can name the same point of space at a shared corner while
/// supplying different `(w x, w y, w z, w)` there, and the three-term inclusion-exclusion then
/// fails to cancel. Reading the corners through `boundary_point`, which **projects**, hid exactly
/// that: both readings were physical points and agreed whenever the geometry did, so the weight
/// never entered the comparison. Measured with a single rational boundary whose weights run 1 to
/// 2, every corner matched to `1.1e-16` and the patch missed one of the four curves it was handed
/// by 17% of the model (pantr issue 327). `homogeneous_endpoint` is the read that stops before the
/// division, and `extract_corners_2d` is where it happens.
///
/// So a rational reading carries a dimensionless weight beside coordinates in units of length
/// times weight, and **each column group is graded against its own magnitude**. One magnitude over
/// the whole row would grade a weight against a threshold proportional to the model's *length*,
/// which is dimensionally wrong and destroys scale invariance: sending `x -> lambda x` scales the
/// homogeneous coordinates and leaves the weights alone. That rule, its scale-invariance argument
/// and its measured roundoff floor are `verify_shared_boundary`'s (join.hpp), and
/// `verify_edges_3d` applies them to the same kind of comparison; this is the third site and does
/// not restate them. A polynomial patch has one group covering every column.
///
/// Unlike `verify_edges_3d`, rationality here is the **patch's** and not the corner's, so every
/// one of the eight readings carries both groups and both scale populations are the whole set.
/// That is what `extract_corners_2d` needs anyway: the corner term is one of three summed in a
/// single space.
///
/// **Each of the eight readings is taken from its own curve**, rather than four of them from
/// `extract_corners_2d`'s output. That array is sized from `C_u0` alone, so routing `C_u1`'s
/// corners through it would widen them to `C_u0`'s precision and the tier would no longer see the
/// precision they were actually supplied in. In this library a call is homogeneous in the storage
/// type, so the two coincide; a binding that converts curves of different precisions to one type
/// passes the coarsest tier instead (the direction that errs towards accepting them).
///
/// The comparison is `gap > tol` to refuse, so a NaN gap is not refused.
///
/// Args:
///     u_curves (const std::array<bspline::Bspline<T>, 2>&): `(C_u0, C_u1)`, the boundaries at
///         `v = 0` and `v = 1`, each free in u and already made compatible with each other.
///     v_curves (const std::array<bspline::Bspline<T>, 2>&): `(C_v0, C_v1)`, the boundaries at
///         `u = 0` and `u = 1`, each free in v.
///     rational (bool): Whether the patch being built is rational, i.e. whether the readings
///         carry a trailing weight column that has to be graded against the weights.
///     tier (double): The relative tolerance tier.
///
/// Raises:
///     value_error: A pair of corners does not match, in either column group; or the two curves
///         meeting at a corner carry control points of different widths (the reference
///         implementation raised a NumPy broadcast error at the subtraction).
template <StorageFloat T>
void verify_corners_2d(const std::array<bspline::Bspline<T>, 2>& u_curves,
                       const std::array<bspline::Bspline<T>, 2>& v_curves, bool rational,
                       double tier) {
  std::array<bspline::Bspline<T>, 2> u = u_curves;
  std::array<bspline::Bspline<T>, 2> v = v_curves;
  if (rational) {
    u = {promote_to_rational<T>(u[0]), promote_to_rational<T>(u[1])};
    v = {promote_to_rational<T>(v[0]), promote_to_rational<T>(v[1])};
  }
  const auto read_end = [](const bspline::Bspline<T>& curve, long long side) {
    const NdArray<T> end = homogeneous_endpoint<T>(curve, side);
    return std::vector<T>(end.flat().begin(), end.flat().end());
  };
  const std::vector<CornerReadings<T>> corners = {
      {"(0,0)", "C_u0", read_end(u[0], 0), "C_v0", read_end(v[0], 0)},
      {"(1,0)", "C_u0", read_end(u[0], 1), "C_v1", read_end(v[1], 0)},
      {"(0,1)", "C_u1", read_end(u[1], 0), "C_v0", read_end(v[0], 1)},
      {"(1,1)", "C_u1", read_end(u[1], 1), "C_v1", read_end(v[1], 1)},
  };
  std::vector<const std::vector<T>*> readings;
  for (const auto& c : corners) {
    readings.push_back(&c.from_u);
    readings.push_back(&c.from_v);
  }
  // No floor: a group whose eight readings are all zero has no scale, and is also bitwise equal,
  // so a zero tolerance is the right answer there rather than a hazard.
  std::vector<double> scales;
  for (std::size_t g = 0; g < group_count(rational); ++g) {
    double scale = 0.0;
    for (std::size_t r = 0; r < readings.size(); ++r) {
      const std::vector<T>& row = *readings[r];
      const double magnitude =
          max_abs_in<T>(std::span<const T>(row), row.size(), group_at(rational, g, row.size()));
      scale = r == 0 ? magnitude : py_max(scale, magnitude);
    }
    scales.push_back(scale);
  }

  for (const auto& c : corners) {
    if (c.from_u.size() != c.from_v.size()) {
      throw value_error("Corner " + c.label + ": " + std::string(c.u_curve) + " carries " +
                        std::to_string(c.from_u.size()) + " coordinates per control point and " +
                        std::string(c.v_curve) + " " + std::to_string(c.from_v.size()) +
                        "; the curves must have one rank (weights included).");
    }
    // Coordinates first: a curve given in the wrong place fails there, and it is the more
    // informative half to report when both groups are out.
    for (std::size_t g = 0; g < group_count(rational); ++g) {
      const ColumnGroup group = group_at(rational, g, c.from_u.size());
      const double gap = group_gap<T>(std::span<const T>(c.from_u), std::span<const T>(c.from_v),
                                      c.from_u.size(), group)
                             .gap;
      const double scale = scales[g];
      const double tol = tier * scale;
      if (gap > tol) {
        throw value_error(
            "Corner " + c.label + " mismatch: " + std::string(c.u_curve) + " gives " +
            row_repr<T>(std::span<const T>(c.from_u)) + ", " + std::string(c.v_curve) + " gives " +
            row_repr<T>(std::span<const T>(c.from_v)) + " (gap " + py_format_exp(gap, 3) +
            ", above " + py_format_exp(tol, 3) + " at corner " + std::string(group.name) +
            " scale " + py_format_exp(scale, 3) + ").");
      }
    }
  }
}

/// Construct a Coons patch from four boundary curves, grading their corners at a given tier.
///
/// The body of `create_coons_surface`; see there for the construction and its refusals. The tier
/// is the relative tolerance of the corner comparison (`conservative_tier` of the coarsest storage
/// type among the inputs as supplied).
///
/// Args:
///     curves (const std::array<std::array<bspline::Bspline<T>, 2>, 2>&): `((C_v0, C_v1), (C_u0,
///         C_u1))`.
///     tier (double): The relative tolerance tier of the corner comparison.
///
/// Returns:
///     bspline::Bspline<T>: A surface in `T`.
///
/// Raises:
///     value_error: As `create_coons_surface`.
template <StorageFloat T>
[[nodiscard]] bspline::Bspline<T> create_coons_surface(
    const std::array<std::array<bspline::Bspline<T>, 2>, 2>& curves, double tier) {
  const bspline::Bspline<T>& v0 = curves[0][0];
  const bspline::Bspline<T>& v1 = curves[0][1];
  const bspline::Bspline<T>& u0 = curves[1][0];
  const bspline::Bspline<T>& u1 = curves[1][1];

  for (const bspline::Bspline<T>* c : {&v0, &v1, &u0, &u1}) {
    if (c->dim() != 1) {
      throw value_error("All curves must be 1D, got dim=" + std::to_string(c->dim()) + ".");
    }
  }

  // Make opposite pairs compatible.
  const std::array<bspline::Bspline<T>, 2> u_pair = compat_pair<T>(u0, u1);
  const std::array<bspline::Bspline<T>, 2> v_pair = compat_pair<T>(v0, v1);

  // Rationality is the patch's, not each curve's: the three terms are added together and so have
  // to share one space, in which a polynomial curve carries unit weights.
  const bool rational =
      u_pair[0].rational() || u_pair[1].rational() || v_pair[0].rational() || v_pair[1].rational();

  // Read the corners off the u-curves in the space the three terms are combined in, and refuse
  // v-curves that contradict them there. These are read in the curves' own precision: it is the
  // precision the disagreement is graded at.
  const std::vector<T> corners = extract_corners_2d<T>(u_pair, rational);
  verify_corners_2d<T>(u_pair, v_pair, rational, tier);

  // R0: ruled in v from C_v0 to C_v1, then transposed to (u, v).
  const std::array<long long, 2> swap_axes = {1, 0};
  const bspline::Bspline<T> r0 = bspline::permuted<T>(create_ruled<T>(v_pair[0], v_pair[1]),
                                                      std::span<const long long>(swap_axes));
  // R1: ruled in u from C_u0 to C_u1 (already in (u, v) order).
  const bspline::Bspline<T> r1 = create_ruled<T>(u_pair[0], u_pair[1]);
  // B: bilinear from the corners.
  const bspline::Bspline<T> b =
      build_multilinear_corner_term<T>(std::span<const T>(corners), 2, rational);

  const std::array<bspline::Bspline<T>, 3> terms = {r0, r1, b};
  const std::vector<bspline::Bspline<T>> made =
      cad::make_compat<T>(std::span<const bspline::Bspline<T>>(terms), std::nullopt);
  const std::array<int, 3> signs = {+1, +1, -1};
  return combine_terms<T>(made, std::span<const int>(signs));
}

/// Extract the 12 edges of a volume, each from both faces that carry it.
///
/// An edge free in direction `f` is fixed in the other two, and the face pair of either fixed
/// direction carries it: the edge free in w at `u = i, v = j` is a boundary of `face_u{i}` and a
/// boundary of `face_v{j}`. `curve_a` is the reading from the face of the lower fixed direction,
/// which is the one the construction goes on to use.
///
/// Within a face the two surviving directions keep their volume order, so a fixed direction sits
/// at face axis 0 when it precedes `f` and at face axis 1 when it follows.
///
/// Args:
///     faces (const std::array<std::array<bspline::Bspline<T>, 2>, 3>&): `((face_u0, face_u1),
///         (face_v0, face_v1), (face_w0, face_w1))`: the faces at `u = 0, 1`, parameterized by
///         `(v, w)`; at `v = 0, 1`, by `(u, w)`; at `w = 0, 1`, by `(u, v)`.
///
/// Returns:
///     std::vector<EdgeReadings<T>>: The 12 edges, ordered by free direction w, then v, then u (and
///     within one, by the side of the first fixed direction, then of the second), and named as
///     `create_coons_volume` expects.
///
/// Raises:
///     value_error: A face is not a surface (`bspline::boundary`'s refusal).
template <StorageFloat T>
[[nodiscard]] std::vector<EdgeReadings<T>> coons_edge_pairs(
    const std::array<std::array<bspline::Bspline<T>, 2>, 3>& faces) {
  std::vector<EdgeReadings<T>> readings;
  readings.reserve(12);
  for (const int free_dir : {2, 1, 0}) {
    std::array<int, 2> fixed_dirs = {0, 0};
    int slot = 0;
    for (int axis = 0; axis < 3; ++axis) {
      if (axis != free_dir) {
        fixed_dirs[static_cast<std::size_t>(slot++)] = axis;
      }
    }
    const int axis_a = fixed_dirs[0];
    const int axis_b = fixed_dirs[1];
    const char name_free = direction_names[static_cast<std::size_t>(free_dir)];
    const char name_a = direction_names[static_cast<std::size_t>(axis_a)];
    const char name_b = direction_names[static_cast<std::size_t>(axis_b)];
    const long long face_axis_a = axis_b < free_dir ? 0 : 1;
    const long long face_axis_b = axis_a < free_dir ? 0 : 1;
    for (int side_a = 0; side_a < 2; ++side_a) {
      for (int side_b = 0; side_b < 2; ++side_b) {
        std::array<int, 3> sides = {0, 0, 0};
        sides[static_cast<std::size_t>(axis_a)] = side_a;
        sides[static_cast<std::size_t>(axis_b)] = side_b;
        std::array<std::string, 2> ends;
        for (int end = 0; end < 2; ++end) {
          sides[static_cast<std::size_t>(free_dir)] = end;
          ends[static_cast<std::size_t>(end)] = corner_label(sides);
        }
        readings.push_back(EdgeReadings<T>{
            std::string(1, name_free) + "_" + name_a + std::to_string(side_a) + "_" + name_b +
                std::to_string(side_b),
            std::string("face_") + name_a + std::to_string(side_a),
            bspline::boundary<T>(
                faces[static_cast<std::size_t>(axis_a)][static_cast<std::size_t>(side_a)],
                face_axis_a, side_b),
            std::string("face_") + name_b + std::to_string(side_b),
            bspline::boundary<T>(
                faces[static_cast<std::size_t>(axis_b)][static_cast<std::size_t>(side_b)],
                face_axis_b, side_a),
            ends});
      }
    }
  }
  return readings;
}

/// Verify that the three faces meeting at each corner of the volume agree there.
///
/// The tolerance is `verify_corners_2d`'s, which is where its derivation lives: the conservative
/// tier, at the coarsest of the readings' own precisions (`tier`), times the largest absolute
/// coordinate over **all** corner readings, so the verdict does not depend on which corner sits at
/// the origin. Three faces meet at a corner of a volume rather than two curves at a corner of a
/// patch, so the scale is taken over twenty-four readings instead of eight and each corner is
/// graded by three comparisons instead of one; nothing else changes, including that these are
/// projected points and so carry one physical dimension between them.
///
/// The w-faces are the reason this reads every edge from both of its carriers. Reading each edge,
/// and so each corner, off the u- and v-faces alone would compare those faces with themselves,
/// which agree by construction, and a w-face contradicting them would be compared with nothing.
///
/// The comparison is `gap > tol` to refuse, so a NaN gap is not refused.
///
/// Args:
///     edges (const std::vector<EdgeReadings<T>>&): The 12 edges of the volume, read from both
///         carriers, as `coons_edge_pairs` returns them.
///     tier (double): The relative tolerance tier.
///
/// Raises:
///     value_error: Two faces meeting at a corner disagree there by more than the tolerance; or
///         carry points of different numbers of coordinates (the reference implementation raised
///         a NumPy broadcast error at the subtraction).
template <StorageFloat T>
void verify_corners_3d(const std::vector<EdgeReadings<T>>& edges, double tier) {
  // Each (corner, face) pair is an endpoint of two of the twelve edges; read it once.
  struct Reading {
    std::string corner;
    std::string face;
    std::vector<T> point;
  };
  std::vector<Reading> readings;
  for (const auto& edge : edges) {
    for (long long end = 0; end < 2; ++end) {
      const std::string& corner = edge.corners[static_cast<std::size_t>(end)];
      const std::array<std::pair<const std::string*, const bspline::Bspline<T>*>, 2> carriers = {
          {{&edge.face_a, &edge.curve_a}, {&edge.face_b, &edge.curve_b}}};
      for (const auto& carrier : carriers) {
        const std::string& face = *carrier.first;
        const bspline::Bspline<T>& curve = *carrier.second;
        const bool seen = std::any_of(readings.begin(), readings.end(), [&](const Reading& r) {
          return r.corner == corner && r.face == face;
        });
        if (!seen) {
          const NdArray<T> point = bspline::boundary_point<T>(curve, 0, end);
          readings.push_back(
              Reading{corner, face, std::vector<T>(point.flat().begin(), point.flat().end())});
        }
      }
    }
  }
  const auto point_of = [&](const std::string& corner, const std::string& face) {
    // Every (corner, face) pair of the eight corners and their three faces is among the readings:
    // `coons_edge_pairs` visits each as an endpoint of an edge it carries.
    return std::find_if(readings.begin(), readings.end(),
                        [&](const Reading& r) { return r.corner == corner && r.face == face; })
        ->point;
  };

  // No floor: corners all at the origin have no scale, and are also bitwise equal, so a zero
  // tolerance is the right answer there rather than a hazard.
  double scale = 0.0;
  for (std::size_t r = 0; r < readings.size(); ++r) {
    const std::vector<T>& p = readings[r].point;
    const double magnitude =
        max_abs_in<T>(std::span<const T>(p), p.size(), ColumnGroup{"coordinate", 0, p.size()});
    scale = r == 0 ? magnitude : py_max(scale, magnitude);
  }
  const double tol = tier * scale;

  // The three pairs among the three faces meeting at a corner, in `itertools.combinations` order.
  const std::array<std::pair<int, int>, 3> face_pairs = {{{0, 1}, {0, 2}, {1, 2}}};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        const std::string corner = corner_label({i, j, k});
        const std::array<std::string, 3> meeting = {"face_u" + std::to_string(i),
                                                    "face_v" + std::to_string(j),
                                                    "face_w" + std::to_string(k)};
        for (const auto& [ia, ib] : face_pairs) {
          const std::string& face_a = meeting[static_cast<std::size_t>(ia)];
          const std::string& face_b = meeting[static_cast<std::size_t>(ib)];
          const std::vector<T> point_a = point_of(corner, face_a);
          const std::vector<T> point_b = point_of(corner, face_b);
          if (point_a.size() != point_b.size()) {
            throw value_error("Corner " + corner + ": " + face_a + " gives a point of " +
                              std::to_string(point_a.size()) + " coordinates and " + face_b +
                              " one of " + std::to_string(point_b.size()) +
                              "; the faces must have one rank.");
          }
          const double gap =
              group_gap<T>(std::span<const T>(point_a), std::span<const T>(point_b), point_a.size(),
                           ColumnGroup{"coordinate", 0, point_a.size()})
                  .gap;
          if (gap > tol) {
            throw value_error("Corner " + corner + " mismatch: " + face_a + " gives " +
                              row_repr<T>(std::span<const T>(point_a)) + ", " + face_b + " gives " +
                              row_repr<T>(std::span<const T>(point_b)) + " (gap " +
                              py_format_exp(gap, 3) + ", above " + py_format_exp(tol, 3) +
                              " at corner scale " + py_format_exp(scale, 3) + ").");
          }
        }
      }
    }
  }
}

/// One edge's two readings, made comparable, with the column groups that grade them.
///
/// Rationality is decided per edge, not per volume: an edge shared by two polynomial faces is
/// compared in three columns of coordinates, while one carried by a rational face is promoted and
/// compared in four, the last of which is a weight. The groups travel with the arrays because a
/// mixed model has both kinds at once, and a weight column must never be graded against a
/// coordinate magnitude.
///
/// Attributes:
///     edge (const EdgeReadings<T>*): The edge, for the labels the mismatch message names.
///     a (bspline::Bspline<T>): `face_a`'s reading, made compatible.
///     b (bspline::Bspline<T>): `face_b`'s reading, same space.
///     rational (bool): Whether the two readings carry a trailing weight column.
template <StorageFloat T>
struct ComparedEdge {
  const EdgeReadings<T>* edge;
  bspline::Bspline<T> a;
  bspline::Bspline<T> b;
  bool rational;
};

/// Verify that the two faces sharing each edge of the volume agree along all of it.
///
/// Agreeing at the eight corners is necessary but not sufficient: two faces share a whole edge,
/// and a volume built from faces that meet only at the endpoints of a shared edge misses one of
/// them by the size of the disagreement in between. Agreeing on all twelve edges *is* sufficient,
/// and a face's interior is free: displacing an interior control point of one face by half the
/// model size leaves all six faces interpolated to `2e-15`. That is the classical compatibility
/// condition for transfinite interpolation, observed here rather than proved.
///
/// Made compatible, the two readings span the same space, where a spline is determined by its
/// control points; comparing those is therefore an exact comparison of the two curves and needs no
/// sampling. The tolerance is `verify_corners_2d`'s tier and rule applied to what is graded here,
/// which is a control-point coefficient rather than a point on the curve: the conservative tier
/// times the largest absolute coefficient over all twelve compared edges.
///
/// **Each column group is graded against its own magnitude.** A rational edge is compared as
/// homogeneous control points, whose coordinate columns carry units of length times weight beside
/// a dimensionless weight. One magnitude over the whole row would grade a weight against a
/// threshold proportional to the model's *length*, which is dimensionally wrong and destroys scale
/// invariance: sending `x -> lambda x` scales the coordinates and leaves the weights alone, so the
/// verdict on a weight drifts with the size of the model. Measured on the cube of side `s` with
/// two rational faces, a weight displaced by four times the tier was refused at `s = 1` and
/// **accepted at every `s >= 1e2`** -- the failure this check exists to stop, reappearing on the
/// rational path once the model is large. Grading the coordinate columns against `max|X|` and the
/// weight column against `max|w|` restores it: under `x -> lambda x` both group scales and both
/// group gaps carry `lambda`, and under a global weight rescale `w -> mu w` the homogeneous
/// coordinates `X = w x` carry `mu` alongside the weights, so both groups carry it too. It does
/// not disarm the check a one-sided rescale exists to catch: that still leaves `gap_w = (mu - 1)
/// w` against `tier * max(w, mu w)`, and is refused. The weight column's own roundoff floor
/// against `max|w|` is `<= 3.2 eps`, independent of magnitude, degree and refinement, so the tier
/// clears it by about 1290. `verify_shared_boundary` (join.hpp) carries those measurements; this
/// is the same rule on the same kind of comparison.
///
/// **Each group's scale spans the whole model, and only the edges that carry the group.** The
/// scale population is all twelve edges, as it is for the corners, so the verdict does not depend
/// on which edge happens to sit at the origin. But an edge shared by two polynomial faces has no
/// weight column, so it contributes to the coordinate scale and says nothing about the weights --
/// which is what keeps a mixed model's weight tolerance from being set by unit weights the caller
/// never supplied.
///
/// **The tier is read at the readings' own precision**, not at `double`, for the reason
/// `conservative_tier` gives: at `double` it is `9.09e-13` relative, while one `float` ulp at
/// magnitude 1 is `1.19e-07`, so a `float` model was graded `2^17 = 131072` times tighter than
/// anything it can express. The coarsest of the compared readings decides, matching the scale's
/// population; that is never below the per-edge floor, so no genuinely shared edge is refused by
/// it.
///
/// **Why any slack at all.** Two readings of a genuinely shared edge are usually **bitwise
/// equal**, and not only when the two faces store it identically: reaching the common space runs
/// the same degree elevation and knot insertion on the same coefficients, so the roundoff is the
/// same on both sides and cancels. Measured on a shared edge arriving at degree 1 from one face
/// and degree 2 from the other, and on one arriving with different knot vectors, the gap was
/// exactly zero in both cases. So the threshold is not sized against that roundoff. As in two
/// dimensions it exists for face data the caller *computed* rather than copied -- a transform, an
/// intersection, a reparametrization -- and is sized to catch faces given in the wrong order or
/// the wrong orientation, which miss by a fraction of the model rather than by epsilons.
///
/// For non-rational faces the convex-hull property makes the control-point gap an upper bound on
/// the geometric gap, so the number reported is a distance in space. For rational faces it is not:
/// the comparison is of **homogeneous** control points, which is what the seven-term formula
/// combines, and turning that into a distance would need a lower bound on the weights that this
/// function does not compute. Comparing them is still the right test -- scaling one face's
/// homogeneous control points by 2 leaves its geometry bitwise untouched yet, with this check
/// disabled, makes the volume miss each of the four faces adjacent to it by 17% of the model --
/// but the reported gap is then a coefficient gap, not a distance, and for the weight group it is
/// not even a length.
///
/// The comparison is `!(gap <= tol)` to refuse, so a NaN gap is refused.
///
/// Args:
///     edges (const std::vector<EdgeReadings<T>>&): The 12 edges of the volume, read from both
///         carriers, as `coons_edge_pairs` returns them.
///     tier (double): The relative tolerance tier.
///
/// Raises:
///     value_error: Two faces sharing an edge disagree along it, in either column group, by more
///         than that group's tolerance; or their readings differ in the number of control points
///         or of coordinates per control point after being made compatible (the reference
///         implementation raised a NumPy broadcast error at the subtraction).
template <StorageFloat T>
void verify_edges_3d(const std::vector<EdgeReadings<T>>& edges, double tier) {
  std::vector<ComparedEdge<T>> compared;
  compared.reserve(edges.size());
  for (const auto& edge : edges) {
    const bool rational = edge.curve_a.rational() || edge.curve_b.rational();
    const auto pair =
        compat_pair<T>(rational ? promote_to_rational<T>(edge.curve_a) : edge.curve_a,
                       rational ? promote_to_rational<T>(edge.curve_b) : edge.curve_b);
    compared.push_back(ComparedEdge<T>{&edge, pair[0], pair[1], rational});
  }

  // No floor: a group whose readings are all zero has no scale, and is also bitwise equal, so a
  // zero tolerance is the right answer there rather than a hazard. Only the edges that carry a
  // group contribute to its scale: an edge shared by two polynomial faces has no weight column
  // and says nothing about the weights.
  std::array<double, 2> scales = {0.0, 0.0};
  for (const auto& c : compared) {
    for (std::size_t g = 0; g < group_count(c.rational); ++g) {
      const double from_a = max_abs_in<T>(c.a.control_points(), c.a.cp_size(),
                                          group_at(c.rational, g, c.a.cp_size()));
      const double from_b = max_abs_in<T>(c.b.control_points(), c.b.cp_size(),
                                          group_at(c.rational, g, c.b.cp_size()));
      scales[g] = py_max(scales[g], py_max(from_a, from_b));
    }
  }

  for (const auto& c : compared) {
    const std::span<const std::size_t> shape_a = c.a.shape();
    const std::span<const std::size_t> shape_b = c.b.shape();
    if (!std::equal(shape_a.begin(), shape_a.end(), shape_b.begin(), shape_b.end())) {
      throw value_error("Edge " + c.edge->label + ": " + c.edge->face_a + " and " + c.edge->face_b +
                        " give control nets of different shapes after being made compatible, " +
                        py_repr(shape_a) + " and " + py_repr(shape_b) + ".");
    }
    const std::size_t width = c.a.cp_size();
    for (std::size_t g = 0; g < group_count(c.rational); ++g) {
      const ColumnGroup group = group_at(c.rational, g, width);
      const GroupGap graded =
          group_gap<T>(c.a.control_points(), c.b.control_points(), width, group);
      const double scale = scales[g];
      const double tol = tier * scale;
      if (graded.gap <= tol) {
        continue;
      }

      // Name the control point that disagrees most, so the gap reported is the gap compared, as
      // `verify_shared_boundary` does for the same class of mismatch. The whole coefficient is
      // printed even though one column group decided the verdict.
      const std::size_t index = graded.index;
      throw value_error(
          "Edge " + c.edge->label + " mismatch at control point " + std::to_string(index) + ": " +
          c.edge->face_a + " gives " +
          row_repr<T>(c.a.control_points().subspan(index * width, width)) + ", " + c.edge->face_b +
          " gives " + row_repr<T>(c.b.control_points().subspan(index * width, width)) +
          ", and they disagree along the edge they share (gap " + py_format_exp(graded.gap, 3) +
          ", above " + py_format_exp(tol, 3) + " at edge " + std::string(group.name) + " scale " +
          py_format_exp(scale, 3) + ").");
    }
  }
}

/// Refuse a rational Coons volume whose weight field is not certified positive.
///
/// A NURBS volume is `X(u) / w(u)` and is only defined where its weight field stays away from
/// zero. Positive weights on the six faces do **not** carry over to the blend: the seven-term
/// formula is an inclusion-exclusion and subtracts three of its terms, so a weight field assembled
/// from strictly positive faces can dip to zero or below inside, and the volume then has a pole
/// the caller never put there.
///
/// **What a control weight settles.** The result's weight field is `w(u) = sum_i N_i(u) w_i` over
/// its own tensor-product B-spline basis, which is non-negative and sums to one on the domain -- a
/// tensor product of non-negative partitions of unity is one -- so
///
///     `w(u) - min_j w_j = sum_i N_i(u) (w_i - min_j w_j) >= 0`
///
/// for every `u`. The weight field therefore never goes below the smallest control weight, and
/// reading the weight column of the **result's** control points bounds it from below without
/// evaluating anything.
///
/// **Sufficient, not necessary, and deliberately so.** The converse of that inequality is false:
/// `w(u)` is a convex combination of the `w_i` only at the ends, and in between the basis is
/// strictly positive on more than one coefficient, so a volume whose weight field never comes near
/// zero can still carry a negative control weight and be refused here. This check is therefore
/// conservative: everything it accepts has a positive weight field, but it does not accept
/// everything that has one. It reports the bound it can prove rather than pretending to detect the
/// pole itself, which would need a global minimum of `w` over the domain.
///
/// The smallest weight is taken as NumPy takes it: a NaN weight is the smallest, and is refused.
///
/// Args:
///     volume (const bspline::Bspline<T>&): The rational volume just assembled, whose control
///         points carry a trailing weight column.
///
/// Raises:
///     value_error: A control weight of `volume` is zero, negative or NaN, so that the lower
///         bound no longer keeps the weight field away from zero.
template <StorageFloat T>
void verify_positive_weights(const bspline::Bspline<T>& volume) {
  const std::span<const T> cp = volume.control_points();
  const std::size_t width = volume.cp_size();
  double smallest = static_cast<double>(cp[width - 1]);
  std::size_t at = 0;
  for (std::size_t p = 1; p * width < cp.size(); ++p) {
    const double w = static_cast<double>(cp[p * width + width - 1]);
    if (!std::isnan(smallest) && (std::isnan(w) || w < smallest)) {
      smallest = w;
      at = p;
    }
  }
  if (smallest > 0.0) {
    return;
  }

  const std::span<const std::size_t> shape = volume.shape();
  const std::vector<std::size_t> index = unravel(at, shape.first(shape.size() - 1));
  throw value_error(
      "Coons volume weight is not certified positive: control weight " +
      py_format_exp(smallest, 3) + " at index " + py_repr(std::span<const std::size_t>(index)) +
      " is not above zero, so the bound w(u) >= min_i w_i no longer keeps the blended weight "
      "field away from zero and the volume may have a pole. The seven-term blend subtracts "
      "three of its terms, so positive weights on the six faces do not imply positive weights "
      "on the volume; supply faces whose weights vary less across the model. The test is "
      "sufficient, not necessary: a volume whose weight field never vanishes can still fail "
      "it.");
}

/// Extract the volume's 8 corner coefficients from the four edges free in u.
///
/// The array is sized and filled in **one** space, which is what the corner term of the blend
/// needs. A width taken from the edges' coefficients, which are homogeneous for a rational face,
/// and filled from `boundary_point`, which projects, mixes two spaces: a rational volume then
/// fails on shapes `(3,)` and `(4,)` rather than on anything the caller supplied (pantr issue
/// 309). `homogeneous_endpoint` is the read that stays in the space the width was taken from.
///
/// `rational` is the **volume's**, not these four edges': the corner term is one of seven that are
/// added together, so it has to live in the same space as the other six even when the faces that
/// happen to carry the u-edges are polynomial. Promoting them gives those corners a unit weight,
/// which is the right value exactly when the faces that *are* rational carry a unit weight there
/// too -- and `verify_edges_3d` has already required that, since it compares each edge's two
/// readings homogeneously and a polynomial reading promotes to weight one.
///
/// Args:
///     edges (const std::map<std::string, bspline::Bspline<T>>&): The 12 named edge curves.
///     rational (bool): Whether the volume being built is rational, i.e. whether the corners are
///         wanted as homogeneous coefficients.
///
/// Returns:
///     std::vector<T>: Corner coefficients with extents `(2, 2, 2, width)`, row-major, indexed
///     `[i, j, k]` by side along u, v and w, where `width` counts the weight when `rational`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): the four edges carry control
///     points of one width, which `verify_corners_3d` has established (it refuses faces whose
///     points differ in their number of coordinates).
template <StorageFloat T>
[[nodiscard]] std::vector<T> extract_corners_3d(
    const std::map<std::string, bspline::Bspline<T>>& edges, bool rational) {
  std::vector<bspline::Bspline<T>> u_edges;
  for (int j = 0; j < 2; ++j) {
    for (int k = 0; k < 2; ++k) {
      const bspline::Bspline<T>& e = edges.at("u_v" + std::to_string(j) + "_w" + std::to_string(k));
      u_edges.push_back(rational ? promote_to_rational<T>(e) : e);
    }
  }
  const std::size_t width = u_edges[0].cp_size();
  std::vector<T> corners(2 * 2 * 2 * width);
  for (std::size_t jk = 0; jk < 4; ++jk) {
    for (std::size_t i = 0; i < 2; ++i) {
      const NdArray<T> end = homogeneous_endpoint<T>(u_edges[jk], static_cast<long long>(i));
      std::copy(end.flat().begin(), end.flat().end(),
                corners.begin() + static_cast<std::ptrdiff_t>((i * 4 + jk) * width));
    }
  }
  return corners;
}

/// Build a trivariate B-spline bilinear in two directions, free in one.
///
/// The four input edges are curves that share the same free parameter. They are placed at the four
/// combinations of the two bilinear parameters. The raw axes are `(bilinear_0, bilinear_1, free)`,
/// then permuted to `(u, v, w)` by `permutation`.
///
/// Args:
///     e_00 (const bspline::Bspline<T>&): Edge at `(bilinear_0 = 0, bilinear_1 = 0)`.
///     e_10 (const bspline::Bspline<T>&): Edge at `(bilinear_0 = 1, bilinear_1 = 0)`.
///     e_01 (const bspline::Bspline<T>&): Edge at `(bilinear_0 = 0, bilinear_1 = 1)`.
///     e_11 (const bspline::Bspline<T>&): Edge at `(bilinear_0 = 1, bilinear_1 = 1)`.
///     permutation (const std::array<long long, 3>&): The permutation with `new[i] =
///         old[permutation[i]]`, mapping the raw `(bilinear_0, bilinear_1, free)` to `(u, v, w)`.
///
/// Returns:
///     bspline::Bspline<T>: A volume in `(u, v, w)` order, rational when any edge is.
///
/// Raises:
///     value_error: `make_compat` refuses, or the four edges differ in the number of control
///         points or of coordinates per control point after being made compatible (the reference
///         implementation raised a NumPy broadcast error filling the array).
template <StorageFloat T>
[[nodiscard]] bspline::Bspline<T> build_bilinear_volume(
    const bspline::Bspline<T>& e_00, const bspline::Bspline<T>& e_10,
    const bspline::Bspline<T>& e_01, const bspline::Bspline<T>& e_11,
    const std::array<long long, 3>& permutation) {
  const std::array<bspline::Bspline<T>, 4> in = {e_00, e_01, e_10, e_11};
  std::vector<bspline::Bspline<T>> made =
      cad::make_compat<T>(std::span<const bspline::Bspline<T>>(in), std::nullopt);

  const bool rational = std::any_of(made.begin(), made.end(),
                                    [](const bspline::Bspline<T>& e) { return e.rational(); });
  if (rational) {
    for (auto& e : made) {
      e = promote_to_rational<T>(e);
    }
  }

  const std::size_t width = made[0].cp_size();
  const std::size_t n_free = made[0].num_basis(0);
  for (const auto& e : made) {
    // Guards an out-of-bounds read of the edges' control points below. No input reaching it has
    // been constructed: the edge check fires first on the inputs tried.
    if (e.cp_size() != width || e.num_basis(0) != n_free) {
      throw value_error(
          "A Coons blend volume needs four edges with equal control nets after being made "
          "compatible, got " +
          py_repr(made[0].shape()) + " and " + py_repr(e.shape()) + ".");
    }
  }

  // Extents (2, 2, n_free, width) = (bilinear_0, bilinear_1, free, coefficient). The type is the
  // edges' own: the free direction reuses their space.
  std::vector<T> new_cp;
  new_cp.reserve(4 * n_free * width);
  // Row-major over (bilinear_0, bilinear_1): the blocks (0, 0), (0, 1), (1, 0), (1, 1), which is
  // the order of `made`.
  for (const auto& e : made) {
    const std::span<const T> cp = e.control_points();
    new_cp.insert(new_cp.end(), cp.begin(), cp.end());
  }

  std::vector<bspline::BsplineSpace1D<T>> spaces;
  spaces.push_back(linear_space_1d<T>());
  spaces.push_back(linear_space_1d<T>());
  spaces.push_back(made[0].space().spaces()[0]);
  const bspline::Bspline<T> vol(bspline::BsplineSpace<T>(std::move(spaces)),
                                std::span<const T>(new_cp), rational);
  return bspline::permuted<T>(vol, std::span<const long long>(permutation));
}

/// Construct a trivariate Coons blending volume from 6 boundary faces, grading them at a given
/// tier.
///
/// The body of `create_coons_volume`; see there for the construction and its refusals. The tier is
/// the relative tolerance of the corner and edge comparisons (`conservative_tier` of the coarsest
/// storage type among the inputs as supplied).
///
/// Args:
///     faces (const std::array<std::array<bspline::Bspline<T>, 2>, 3>&): Three pairs of opposite
///         boundary faces.
///     tier (double): The relative tolerance tier of the comparisons.
///
/// Returns:
///     bspline::Bspline<T>: A volume in `T`.
///
/// Raises:
///     value_error: As `create_coons_volume`.
template <StorageFloat T>
[[nodiscard]] bspline::Bspline<T> create_coons_volume(
    const std::array<std::array<bspline::Bspline<T>, 2>, 3>& faces, double tier) {
  for (const auto& pair : faces) {
    for (const auto& f : pair) {
      if (f.dim() != 2) {
        throw value_error("All faces must be 2D B-splines, got dim=" + std::to_string(f.dim()) +
                          ".");
      }
    }
  }

  // Make opposite face pairs compatible.
  const std::array<std::array<bspline::Bspline<T>, 2>, 3> compat = {
      {compat_pair<T>(faces[0][0], faces[0][1]), compat_pair<T>(faces[1][0], faces[1][1]),
       compat_pair<T>(faces[2][0], faces[2][1])}};

  // Read every edge from both faces that carry it, and refuse faces that disagree.
  const std::vector<EdgeReadings<T>> edge_readings = coons_edge_pairs<T>(compat);
  verify_corners_3d<T>(edge_readings, tier);
  verify_edges_3d<T>(edge_readings, tier);

  // The construction reads each edge from the first of its two carriers; the other reading is
  // only there to be checked against, and by now is known to agree.
  std::map<std::string, bspline::Bspline<T>> edges;
  for (const auto& r : edge_readings) {
    edges.emplace(r.label, r.curve_a);
  }

  // Extract corners from edges, in the space the other six terms are combined in.
  bool rational_faces = false;
  for (const auto& pair : compat) {
    for (const auto& f : pair) {
      rational_faces = rational_faces || f.rational();
    }
  }
  const std::vector<T> corners = extract_corners_3d<T>(edges, rational_faces);

  // Build 3 ruled volumes.
  const std::array<long long, 3> perm_u = {2, 0, 1};
  const std::array<long long, 3> perm_v = {0, 2, 1};
  const bspline::Bspline<T> r_u = bspline::permuted<T>(create_ruled<T>(compat[0][0], compat[0][1]),
                                                       std::span<const long long>(perm_u));
  const bspline::Bspline<T> r_v = bspline::permuted<T>(create_ruled<T>(compat[1][0], compat[1][1]),
                                                       std::span<const long long>(perm_v));
  const bspline::Bspline<T> r_w = create_ruled<T>(compat[2][0], compat[2][1]);  // (u, v, w)

  // Build 3 bilinear blend volumes.
  // B_uv: raw axes (u, v, w) -> permute to (u, v, w) = identity.
  const bspline::Bspline<T> b_uv =
      build_bilinear_volume<T>(edges.at("w_u0_v0"), edges.at("w_u1_v0"), edges.at("w_u0_v1"),
                               edges.at("w_u1_v1"), {0, 1, 2});
  // B_uw: raw axes (u, w, v) -> permute to (u, v, w).
  const bspline::Bspline<T> b_uw =
      build_bilinear_volume<T>(edges.at("v_u0_w0"), edges.at("v_u1_w0"), edges.at("v_u0_w1"),
                               edges.at("v_u1_w1"), {0, 2, 1});
  // B_vw: raw axes (v, w, u) -> permute to (u, v, w).
  const bspline::Bspline<T> b_vw =
      build_bilinear_volume<T>(edges.at("u_v0_w0"), edges.at("u_v1_w0"), edges.at("u_v0_w1"),
                               edges.at("u_v1_w1"), {2, 0, 1});

  // Build the trilinear volume.
  const bspline::Bspline<T> t =
      build_multilinear_corner_term<T>(std::span<const T>(corners), 3, rational_faces);

  // Make all 7 terms compatible and combine.
  const std::array<bspline::Bspline<T>, 7> terms = {r_u, r_v, r_w, b_uv, b_uw, b_vw, t};
  const std::vector<bspline::Bspline<T>> made =
      cad::make_compat<T>(std::span<const bspline::Bspline<T>>(terms), std::nullopt);
  const std::array<int, 7> signs = {+1, +1, +1, -1, -1, -1, +1};
  const bspline::Bspline<T> volume = combine_terms<T>(made, std::span<const int>(signs));
  if (volume.rational()) {
    verify_positive_weights<T>(volume);
  }
  return volume;
}

}  // namespace detail

/// Construct a Coons patch from four boundary curves.
///
/// Builds a bilinearly blended surface from four boundary curves using the formula `S = R0 + R1 -
/// B`, where `R0` and `R1` are ruled surfaces and `B` is the bilinear interpolant of the four
/// corners.
///
/// The curve layout is:
///
///     |           C_u1
///     |    o--------------o
///     |    |  v           |
///     |    |  ^           |
///     |    |  |     C_v1  |
///     |    |  +---> u     |
///     |    o--------------o
///     |    C_v0   C_u0
///
/// **Rational curves are supported, and the blend is applied in homogeneous space.** All three
/// terms are formed from homogeneous coefficients `(w x, w y, w z, w)` and the result is projected
/// only when it is evaluated, so the formula stays the linear one above and nothing has to be said
/// about how weights combine under division. That is what makes the boundary property survive:
/// projection is pointwise, so it commutes with restriction to a boundary, and a blend whose
/// *homogeneous* restriction to `u = 0` is `C_v0`'s homogeneous data projects to `C_v0` itself. A
/// polynomial curve among rational ones is promoted with unit weights, so mixed input takes the
/// same path rather than a third one.
///
/// That argument has a hypothesis, and it is the one the corner check enforces: **the four curves
/// must agree in homogeneous data where they meet, weights included, not merely projectively.** A
/// homogeneous representation is unique only up to one constant scaling all of a patch's weights,
/// so two curves can name the same point of space at a shared corner while their coefficients
/// differ by a factor, and the three-term inclusion-exclusion then does not cancel. Curves that
/// disagree are **refused** rather than blended, which is the verdict `create_coons_volume`
/// reaches on faces that disagree along a shared edge. Rescaling *every* curve's homogeneous
/// coefficients by one constant is not a disagreement and is accepted, since it describes the same
/// patch.
///
/// This is measured rather than feared: with a single rational boundary whose weights run 1 to 2,
/// every corner matched to `1.1e-16` and the patch missed one of the four curves it was handed by
/// 17% of the model (pantr issue 327).
///
/// The curves may have control points of any one rank (a planar patch from rank-2 curves is
/// accepted); curves of different ranks are refused.
///
/// Args:
///     curves (const std::array<std::array<bspline::Bspline<T>, 2>, 2>&): `((C_v0, C_v1), (C_u0,
///         C_u1))` -- two pairs of opposite boundary curves. `C_v0` and `C_v1` run in the
///         v-direction (left/right). `C_u0` and `C_u1` run in the u-direction (bottom/top). All
///         must be curves.
///
/// Returns:
///     bspline::Bspline<T>: A surface, in the precision the curves were given in. Because it
///     returns rather than raising, the four curves agree at every shared corner, and the patch
///     therefore restricts to each of them on the corresponding boundary. That interpolation is
///     exact in exact arithmetic; the floating-point residual is **observed** to stay near
///     `1e-16` relative for `double`, three orders below the tolerance the check below uses, and
///     is not a proved bound.
///
/// Raises:
///     value_error: A curve is not 1D ("All curves must be 1D, got dim=..."); two curves
///         disagree at a shared corner by more than `4096 * eps` times that column group's own
///         magnitude over all eight corner readings: the largest absolute homogeneous coordinate
///         for the coordinate columns, and the largest absolute weight for the weight column of a
///         rational patch. `eps` is that of `T`, so that a `float` model is graded against
///         something `float` can express; `detail::verify_corners_2d` derives all of it. Also: the
///         curves carry control points of different ranks, or `make_compat` or `create_ruled`
///         refuses (their messages).
template <StorageFloat T>
[[nodiscard]] bspline::Bspline<T> create_coons_surface(
    const std::array<std::array<bspline::Bspline<T>, 2>, 2>& curves) {
  return detail::create_coons_surface<T>(curves, detail::conservative_tier<T>());
}

/// Construct a trivariate Coons blending volume from 6 boundary faces.
///
/// Uses the trilinear Coons formula:
///
/// `V = (R_u + R_v + R_w) - (B_uv + B_uw + B_vw) + T`
///
/// where `R` are ruled volumes from opposite face pairs, `B` are bilinear blend volumes from edge
/// quadruples, and `T` is the trilinear corner interpolant.
///
/// Edges and corners are derived automatically from face boundaries.
///
/// The six faces are not independent data: each of the twelve edges is shared by two of them and
/// each of the eight corners by three, so they have to agree where they meet. Faces that do not
/// are refused rather than blended, because the formula spreads the inconsistency over the volume
/// and the result then misses faces the caller supplied correctly.
///
/// **Rational faces are supported, and the blend is applied in homogeneous space.** All seven
/// terms are formed from homogeneous coefficients `(w x, w y, w z, w)` and the result is projected
/// only when it is evaluated, so the formula stays the linear one above and nothing has to be said
/// about how weights combine under division. That is what makes the boundary property survive:
/// projection is pointwise, so it commutes with restriction to a face, and a blend whose
/// *homogeneous* restriction to `u = 0` is `face_u0`'s homogeneous data projects to `face_u0`
/// itself. A polynomial face among rational ones is promoted with unit weights, so mixed input
/// takes the same path rather than a third one.
///
/// That argument has a hypothesis, and it is the one the checks below enforce: the faces must
/// agree in **homogeneous** data, weights included, not merely projectively. A homogeneous
/// representation is unique only up to one constant scaling all of a patch's weights, so two faces
/// can trace the same edge in space while their coefficients differ by a factor, and then the
/// inclusion-exclusion does not cancel. This is measured rather than feared: scaling one
/// `w`-face's homogeneous control points by 2 leaves that face's geometry bitwise unchanged yet,
/// with the checks disabled, makes the volume miss each of the four faces adjacent to it by 17% of
/// the model.
///
/// Face labelling convention:
///
/// - `faces[0] = (face_u0, face_u1)`: faces at u=0 and u=1, each a surface parameterized by
///   (v, w).
/// - `faces[1] = (face_v0, face_v1)`: faces at v=0 and v=1, each a surface parameterized by
///   (u, w).
/// - `faces[2] = (face_w0, face_w1)`: faces at w=0 and w=1, each a surface parameterized by
///   (u, v).
///
/// Args:
///     faces (const std::array<std::array<bspline::Bspline<T>, 2>, 3>&): Three pairs of opposite
///         boundary faces.
///
/// Returns:
///     bspline::Bspline<T>: A volume, in the precision the faces were given in. Because it returns
///     rather than raising, the six faces agree where they meet, and the volume therefore
///     restricts to each of them on the corresponding boundary. That interpolation is exact in
///     exact arithmetic; the floating-point residual is **observed** to stay near `1e-15`
///     relative for `double`, three orders below the tolerance the checks below use, and is not a
///     proved bound.
///
/// Raises:
///     value_error: A face is not a surface ("All faces must be 2D B-splines, got dim=...");
///         two faces meeting at a corner of the volume disagree there by more than `4096 * eps`
///         times the largest absolute coordinate over all twenty-four corner readings, derived in
///         `detail::verify_corners_2d` and applied by `detail::verify_corners_3d`. Same tier as
///         `create_coons_surface`, but one undivided scale rather than that function's two column
///         groups: these are **projected** points and so carry one physical dimension between
///         them, while a patch compares homogeneous coefficients. A weight disagreement on the
///         volume path is caught by `detail::verify_edges_3d` instead, which does split the
///         columns. Two faces sharing an edge disagree along it by more than `4096 * eps` times
///         that column group's own magnitude over the twelve compared edges: the largest absolute
///         homogeneous coordinate for the coordinate columns, and the largest absolute weight for
///         the weight column of a rational edge, derived in `detail::verify_edges_3d`. Both
///         tolerances read `eps` at that of `T` rather than at `double`. The blend of rational
///         faces carries a control weight that is not above zero, so that the weight field of the
///         result cannot be certified positive. The bound is `w(u) >= min_i w_i` and is
///         sufficient rather than necessary, so this refuses some volumes whose weight field
///         never vanishes; `detail::verify_positive_weights` derives it. Also: the faces carry
///         points of different ranks, or `make_compat` or `create_ruled` refuses (their
///         messages).
template <StorageFloat T>
[[nodiscard]] bspline::Bspline<T> create_coons_volume(
    const std::array<std::array<bspline::Bspline<T>, 2>, 3>& faces) {
  return detail::create_coons_volume<T>(faces, detail::conservative_tier<T>());
}

}  // namespace pantr::cad
