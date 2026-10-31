/// @file roots.hpp
/// @brief Root finding for scalar B-splines by the Morken-Reimers method.
///
/// Computes the zeros of a scalar B-spline directly on its own knot vector, by repeatedly inserting
/// the zero of the control polygon as a new knot (Morken and Reimers, "An unconditionally
/// convergent method for computing zeros of splines and polynomials", Math. Comp. 76 (2007)). Every
/// arithmetic operation is a convex combination (Boehm knot insertion), the iteration needs no
/// starting value, and it converges for any spline: the control polygon of a spline with a zero
/// always has a sign change, and the variation-diminishing property keeps the sign-change count
/// from growing under refinement.
///
/// The kernels (`detail::`) refine one working copy of the whole knot vector in place. Reporting a
/// zero splits the spline there, raising the zero to knot multiplicity `degree + 1`, and everything
/// to its left is then dropped: what is left is again an open (clamped) knot vector, this time on
/// `[zero, end]`. That bounds the memory, since the insertions spent on one zero are discarded when
/// the next one is reported, and it keeps every span search inside the buffer.
///
/// Nothing is reported on the strength of the branch that produced it. The paper's certificates (a
/// repeated iterate, an iterate at the end of its Greville interval) are theorems of exact
/// arithmetic, and a floating-point iteration can satisfy their conclusions' *tests* without being
/// anywhere near a zero. So `detail::track_zero` hands back `|f(x)|` on every exit, its status says
/// only how the iteration stopped, and `detail::morken_reimers_roots` reports an iterate if and
/// only if that residual is within `root_tol`.
///
/// **Precision contract.** Knots and coefficients are promoted to `double` before the iteration,
/// which is exact for a `float` spline, so the arithmetic is `double` throughout whatever the
/// storage type. The tolerances, however, are resolved from the *storage* type: a `float` spline
/// carries float-level information, and a residual below that level cannot be told from zero no
/// matter how the iteration is carried out.
///
/// **What is found.** The control polygon says where to look: a spline whose coefficients have no
/// sign change has no zeros at all, so every zero where the spline changes sign is bracketed by a
/// sign change and is searched for. What it does not do is certify. The tracking of a sign change
/// stops for three different reasons, and none of them implies that the point it stopped at is a
/// zero, so a value reaches the result only when its residual `|f(x)|` is within what a zero may
/// legitimately leave behind: the error of evaluating the spline, plus the value the spline reaches
/// at the parametric resolution the iteration stops at, `|f'| * 2 * tol * scale`. That residual
/// test is the only way a zero of even multiplicity can be reported at all, since the sign changes
/// bracketing it disappear under refinement. The two domain endpoints are decided on the evaluation
/// error alone, being read straight off a coefficient with no location error to allow for.
///
/// **Floating point.** The kernels port the Numba kernels operation by operation, and the Numba
/// kernels never contracted a product into a fused multiply-add. Every product here is therefore
/// formed in a statement of its own before it is added, so that `-ffp-contract=on` (the flag
/// `pantr::pantr` sets) cannot fuse it (see `pantr/roots/detail/scalar.hpp`). The branches are sign
/// tests and comparisons on such sums, a discrete decision a last-bit change can flip. A consumer
/// building with `-ffp-contract=fast` loses that property.
#pragma once

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/evaluate.hpp"
#include "pantr/bspline/knot_insertion.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/roots/detail/scalar.hpp"
#include "pantr/tolerance.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bspline {

/// How a root search ended. The values are wire values (the Python binding and the `int` status
/// codes of a C API carry them), so they are pinned.
enum class RootStatus : std::int32_t {
  complete = 0,                 ///< Every sign change was resolved or certified away.
  budget_exhausted = 1,         ///< A sign change was left without a zero: a root may be missing.
  stalled_at_last_iterate = 2,  ///< A reported zero sits at the last iterate, not a stagnated one.
};

static_assert(static_cast<std::int32_t>(RootStatus::complete) == 0);
static_assert(static_cast<std::int32_t>(RootStatus::budget_exhausted) == 1);
static_assert(static_cast<std::int32_t>(RootStatus::stalled_at_last_iterate) == 2);

/// The zeros of a scalar B-spline, and how complete the search was.
///
/// The Python `RuntimeWarning`s are rendered from these fields: the first, when `truncated > 0`,
/// reads "`truncated` of `candidates` roots exhausted their insertion budget of `budget` ...", and
/// the second, when `abandoned > 0`, "`abandoned` sign changes exhausted their insertion budget of
/// `budget` without the residual ever falling to `root_tol` ...".
///
/// Attributes:
///     roots (std::vector<double>): The roots, ascending, with repeated reports of one zero
///         merged (the Python return value).
///     status (RootStatus): `budget_exhausted` if `abandoned > 0`, else `stalled_at_last_iterate`
///         if `truncated > 0`, else `complete`.
///     candidates (std::int64_t): The roots the kernel reported, before merging.
///     truncated (std::int64_t): The reported zeros whose insertion budget ran out first, so that
///         they sit at the last iterate reached rather than at a stagnated one.
///     abandoned (std::int64_t): The sign changes left without a zero to show for them, whether
///         the budget ran out before the residual certified one or the working arrays had no room
///         to track it; each may be a zero this call does not report.
///     budget (int): The insertion budget per zero that was used.
///     root_tol (double): The residual threshold a tracked iterate had to meet.
struct [[nodiscard]] BsplineRootSet {
  std::vector<double> roots;
  RootStatus status;
  std::int64_t candidates;
  std::int64_t truncated;
  std::int64_t abandoned;
  int budget;
  double root_tol;
};

namespace detail {

/// Insertion budget for the linear phase of a single zero.
///
/// A zero of multiplicity greater than one is approached linearly, and the observed ratio is one
/// half (seen on an exact double zero), so exhausting the 53-bit significand takes
/// `log2(1 / eps) = 53` insertions. Rounded up to 64, matching `max_newton_iter` in
/// `pantr/roots/detail/yuksel.hpp`, which bounds the same bisection-grade phase for the Bernstein
/// solver.
inline constexpr int insertions_base = 64;

/// Extra insertions granted per unit of degree.
///
/// The error is quadratic once per `degree - 1` insertions and not once per insertion
/// (Morken-Reimers, Theorem 22), so a zero the theorem covers needs roughly six quadratic steps of
/// `degree - 1` insertions each to reach machine precision. Eight per degree leaves a margin over
/// the worst case seen on random splines.
///
/// The theorem's hypotheses are `f'(z) != 0` **and** `f''(z) != 0`, not simplicity alone: a simple
/// zero with a vanishing second derivative is outside it, and so is every multiple zero. Nothing
/// rests on the hypotheses holding, since the budget is soft (exhausting it stops the tracking with
/// `TrackStatus::budget` and leaves the residual to decide), but the quadratic-step count that
/// sizes the budget is only justified where they do.
inline constexpr int insertions_per_degree = 8;

/// Lowest degree whose splines have a second derivative, hence a curvature merge radius.
inline constexpr int curvature_degree = 2;

/// Safety factor on the evaluation-error bound used as the residual threshold.
///
/// Matches the factor applied to the de Casteljau bound in `pantr/roots/detail/clipping.hpp`; de
/// Boor's algorithm combines the same `degree + 1` coefficients through the same number of
/// convex-combination levels, so the bound has the same shape.
inline constexpr double zero_tol_safety = 4.0;

/// How far a reported zero may sit from the true one, in units of the stopping spread.
///
/// The tracking stops when the last `degree` iterates span less than `tol * scale`. Morken and
/// Reimers bound the *convergence rate* but not the distance from the last iterate to the limit at
/// an arbitrary stopping point, so this is **observed, not proved**: the spread bounds how far the
/// iterates still move, and one further step of the same size covers the tail. Two is the factor
/// the Python test suite measures the residual against over random splines at several parametric
/// scales, and clears with margin on the families that break this method. Being an estimate, it is
/// used with the safety factors those tests carry rather than as a bound in its own right.
inline constexpr double stop_slack = 2.0;

/// Lowest degree at which Corollary 14 of Morken-Reimers asks for a collapsed knot run.
///
/// The corollary needs `degree - 1` active knots collapsed onto the iterate, so at degree 1 it asks
/// for none and a repeated iterate satisfies it on its own.
inline constexpr int corollary_14_degree = 2;

}  // namespace detail

/// Compute the insertion budget for one zero at a given degree.
///
/// Args:
///     degree (int): Polynomial degree of the spline.
///
/// Returns:
///     int: Maximum number of knot insertions spent on a single zero,
///     `64 + 8 * (degree - 1)`.
[[nodiscard]] constexpr int max_insertions(int degree) noexcept {
  return detail::insertions_base + detail::insertions_per_degree * (degree - 1);
}

namespace detail {

using ::pantr::roots::detail::lerp_uncontracted;

/// How a tracked sign change stopped.
enum class TrackStatus : int {
  /// The iteration reached a point it will not leave.
  ///
  /// Covers the three stopping rules: an iterate repeated exactly with Corollary 14's collapsed
  /// knot run behind it, the last `degree` iterates spanned less than `tol`, and an iterate reached
  /// the right end of its Greville interval, where every value of `lam` gives the same point.
  stagnated = 1,
  /// The tracked sign change disappeared under refinement.
  ///
  /// Either a tangential zero, where the two sign changes bracketing it collapse, or a false
  /// warning of the variation-diminishing bound.
  vanished = 2,
  /// The insertion budget ran out before the iteration stopped.
  budget = 3,
};

/// Compute the Greville abscissa `(t[index+1] + ... + t[index+degree]) / degree`.
///
/// Args:
///     knots (std::span<const double>): Knot vector.
///     degree (int): Polynomial degree, at least 1.
///     index (std::int64_t): Coefficient index whose knot average is requested.
///
/// Returns:
///     double: The knot average (Greville abscissa) of coefficient `index`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `find_roots` instead.
[[nodiscard]] inline double knot_average(std::span<const double> knots, int degree,
                                         std::int64_t index) noexcept {
  const double* k = knots.data();
  double total = 0.0;
  for (std::int64_t j = index + 1; j <= index + degree; ++j) {
    total += k[j];
  }
  return total / static_cast<double>(degree);
}

/// Find the smallest zero index of the control polygon at or after `start`.
///
/// Following Morken and Reimers, `a` is a zero index when `coeffs[a - 1]` is non-zero and
/// `coeffs[a - 1] * coeffs[a] <= 0`; the non-zero requirement is what keeps the secant denominator
/// `coeffs[a] - coeffs[a - 1]` away from zero across a run of vanishing coefficients.
///
/// Args:
///     coeffs (std::span<const double>): B-spline coefficients.
///     num_coeffs (std::int64_t): Number of valid entries in `coeffs`.
///     start (std::int64_t): First index to test; clamped to 1 from below.
///
/// Returns:
///     std::int64_t: The zero index, or `-1` if the polygon has no sign change left.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `find_roots` instead.
[[nodiscard]] inline std::int64_t zero_index(std::span<const double> coeffs,
                                             std::int64_t num_coeffs, std::int64_t start) noexcept {
  const double* c = coeffs.data();
  std::int64_t index = start > 1 ? start : 1;
  while (index < num_coeffs) {
    const double previous = c[index - 1];
    if (previous != 0.0 && previous * c[index] <= 0.0) {
      return index;
    }
    ++index;
  }
  return -1;
}

/// Check whether `index` is a zero index of the control polygon.
///
/// Args:
///     coeffs (std::span<const double>): B-spline coefficients.
///     num_coeffs (std::int64_t): Number of valid entries in `coeffs`.
///     index (std::int64_t): Index to test.
///
/// Returns:
///     bool: True if `coeffs[index - 1]` is non-zero and the product
///     `coeffs[index - 1] * coeffs[index]` is non-positive.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `find_roots` instead.
[[nodiscard]] inline bool is_zero_index(std::span<const double> coeffs, std::int64_t num_coeffs,
                                        std::int64_t index) noexcept {
  if (index < 1 || index >= num_coeffs) {
    return false;
  }
  const double* c = coeffs.data();
  const double previous = c[index - 1];
  return previous != 0.0 && previous * c[index] <= 0.0;
}

/// Evaluate a scalar spline at one point with de Boor's algorithm.
///
/// Args:
///     knots (std::span<const double>): Knot vector.
///     coeffs (std::span<const double>): B-spline coefficients.
///     degree (int): Polynomial degree.
///     point (double): Evaluation point, inside `[knots[span], knots[span+1]]`.
///     span (std::int64_t): Knot span index of `point`.
///     work (std::span<double>): Scratch buffer of at least `degree + 1` entries.
///
/// Returns:
///     double: The value of the spline at `point`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `find_roots` instead.
[[nodiscard]] inline double deboor_point(std::span<const double> knots,
                                         std::span<const double> coeffs, int degree, double point,
                                         std::int64_t span, std::span<double> work) noexcept {
  const double* k = knots.data();
  const double* c = coeffs.data();
  double* w = work.data();
  for (int i = 0; i <= degree; ++i) {
    w[i] = c[span - degree + i];
  }
  for (int level = 1; level <= degree; ++level) {
    for (int i = degree; i >= level; --i) {
      const std::int64_t left = span - degree + i;
      const double denominator = k[left + degree - level + 1] - k[left];
      const double alpha = denominator > 0.0 ? (point - k[left]) / denominator : 0.0;
      w[i] = lerp_uncontracted(1.0 - alpha, alpha, w[i - 1], w[i]);
    }
  }
  return w[degree];
}

/// Locate the knot span of `point` in an open (clamped) knot window.
///
/// The search is bounded above by `num_coeffs - 1`, the last span of an open knot vector, so that a
/// point at the end of the domain is handled from the left instead of walking off the end of the
/// buffer.
///
/// Args:
///     knots (std::span<const double>): Open (clamped) knot window.
///     num_coeffs (std::int64_t): Number of valid coefficients in the window.
///     degree (int): Polynomial degree.
///     point (double): Point inside the window's domain.
///
/// Returns:
///     std::int64_t: Span index in `[degree, num_coeffs - 1]`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `find_roots` instead.
[[nodiscard]] inline std::int64_t span_at(std::span<const double> knots, std::int64_t num_coeffs,
                                          int degree, double point) noexcept {
  const double* k = knots.data();
  std::int64_t span = degree;
  while (span + 1 < num_coeffs && point >= k[span + 1]) {
    ++span;
  }
  return span;
}

/// Evaluate `|f(point)|` on a refined window.
///
/// This is the certificate every reported zero has to produce, so it is computed from the window as
/// it stands rather than assumed from the branch that reached it.
///
/// Args:
///     knots (std::span<const double>): Open (clamped) knot window.
///     coeffs (std::span<const double>): Coefficient window.
///     num_coeffs (std::int64_t): Number of valid entries in `coeffs`.
///     degree (int): Polynomial degree.
///     point (double): Evaluation point, inside the window's domain.
///     work (std::span<double>): Scratch buffer of at least `degree + 1` entries.
///
/// Returns:
///     double: `|f(point)|`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `find_roots` instead.
[[nodiscard]] inline double residual_at(std::span<const double> knots,
                                        std::span<const double> coeffs, std::int64_t num_coeffs,
                                        int degree, double point, std::span<double> work) noexcept {
  const std::int64_t span = span_at(knots, num_coeffs, degree, point);
  return std::abs(deboor_point(knots, coeffs, degree, point, span, work));
}

/// Insert one knot in place by Boehm's algorithm, growing both buffers by one.
///
/// Args:
///     knots (std::span<double>): Knot buffer holding `num_coeffs + degree + 1` valid entries, with
///         room for one more.
///     coeffs (std::span<double>): Coefficient buffer holding `num_coeffs` valid entries, with room
///         for one more.
///     num_coeffs (std::int64_t): Number of valid coefficients before the insertion.
///     degree (int): Polynomial degree.
///     point (double): Knot value to insert.
///     span (std::int64_t): Knot span index of `point`, with `knots[span] <= point <
///         knots[span + 1]` (the divisions below are then by non-zero knot differences).
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `find_roots` instead.
inline void insert_knot(std::span<double> knots, std::span<double> coeffs, std::int64_t num_coeffs,
                        int degree, double point, std::int64_t span) noexcept {
  double* k = knots.data();
  double* c = coeffs.data();
  for (std::int64_t i = num_coeffs; i > span; --i) {
    c[i] = c[i - 1];
  }
  for (std::int64_t i = span; i > span - degree; --i) {
    const double alpha = (point - k[i]) / (k[i + degree] - k[i]);
    c[i] = lerp_uncontracted(1.0 - alpha, alpha, c[i - 1], c[i]);
  }

  for (std::int64_t i = num_coeffs + degree + 1; i > span + 1; --i) {
    k[i] = k[i - 1];
  }
  k[span + 1] = point;
}

/// The window left by `split_at_root`.
///
/// Attributes:
///     num_coeffs (std::int64_t): The updated coefficient count.
///     offset (std::int64_t): The index at which the remainder begins, that is the first index of
///         the knot run carrying the split point. Both arrays are to be shifted down by it.
struct SplitResult {
  std::int64_t num_coeffs;
  std::int64_t offset;
};

/// Raise `point` to multiplicity `degree + 1` and pin its coefficient to zero.
///
/// Splitting the spline at a zero that has already been reported is what keeps the next sign change
/// from drifting back onto it: a tracked sequence converges to *a* zero of the spline inside
/// `[t[a], t[a+degree]]`, not necessarily to the one its own Greville interval brackets, so an
/// already-reported zero has to be removed from that bracket first. Morken and Reimers prescribe
/// exactly this, splitting the spline at each zero before looking for the next one.
///
/// Multiplicity `degree + 1` is what makes the split a genuine one: the two sides then share no
/// basis function, and dropping the left side leaves an open (clamped) knot vector on `[point,
/// end]`, so every later span search and every later Boehm insertion sees the same structure as the
/// original input. Multiplicity `degree` would leave one knot of the old spline hanging to the left
/// of the new domain, and a span search starting from a coefficient index would then be free to run
/// off the front of the buffer.
///
/// The first coefficient of that clamped remainder is `f(point)`, which is zero because `point` is
/// a zero of the spline, so pinning it to exactly zero removes the rounding left by the insertions
/// and makes it a barrier: `zero_index` never reports an index whose left neighbour is exactly
/// zero.
///
/// Args:
///     knots (std::span<double>): Knot vector, refined in place.
///     coeffs (std::span<double>): Coefficients, refined in place; its size is the buffer capacity.
///     num_coeffs (std::int64_t): Number of valid entries in `coeffs`.
///     degree (int): Polynomial degree.
///     point (double): Zero of the spline, below the end of the domain.
///     zero_tol (double): Residual below which a coefficient counts as zero.
///
/// Returns:
///     SplitResult: The updated coefficient count and the offset of the remainder.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `find_roots` instead.
[[nodiscard]] inline SplitResult split_at_root(std::span<double> knots, std::span<double> coeffs,
                                               std::int64_t num_coeffs, int degree, double point,
                                               double zero_tol) noexcept {
  double* k = knots.data();
  double* c = coeffs.data();
  const auto capacity = static_cast<std::int64_t>(coeffs.size());

  // Count only the run that ends at this span: the same value may well appear elsewhere in the
  // knot vector, and it is this run that has to be raised.
  std::int64_t span = 0;
  while (point >= k[span + 1]) {
    ++span;
  }
  std::int64_t multiplicity = 0;
  std::int64_t index = span;
  while (index >= 0 && k[index] == point) {
    ++multiplicity;
    --index;
  }

  while (multiplicity <= degree && num_coeffs < capacity) {
    insert_knot(knots, coeffs, num_coeffs, degree, point, span);
    ++num_coeffs;
    ++multiplicity;
    ++span;
  }

  std::int64_t offset = span;
  while (offset > 0 && k[offset - 1] == point) {
    --offset;
  }

  c[offset] = 0.0;
  // The coefficients just right of the barrier average knots that are all the split point, so they
  // carry f(point) = 0 up to rounding. Left as they are, their arbitrary sign seeds a sign change
  // that tracks straight back to the split point; pinning them extends the barrier.
  std::int64_t i = offset + 1;
  while (i < num_coeffs && std::abs(c[i]) <= zero_tol) {
    c[i] = 0.0;
    ++i;
  }
  return {num_coeffs, offset};
}

/// Discard the first `offset` coefficients of the window, shifting the rest down.
///
/// Args:
///     knots (std::span<double>): Knot window, shifted in place.
///     coeffs (std::span<double>): Coefficient window, shifted in place.
///     num_coeffs (std::int64_t): Number of valid entries in `coeffs` before the shift.
///     degree (int): Polynomial degree.
///     offset (std::int64_t): Number of leading coefficients to drop.
///
/// Returns:
///     std::int64_t: The coefficient count after the shift, `num_coeffs - offset`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `find_roots` instead.
inline std::int64_t drop_window_head(std::span<double> knots, std::span<double> coeffs,
                                     std::int64_t num_coeffs, int degree,
                                     std::int64_t offset) noexcept {
  double* k = knots.data();
  double* c = coeffs.data();
  const std::int64_t remaining = num_coeffs - offset;
  for (std::int64_t i = 0; i < remaining; ++i) {
    c[i] = c[i + offset];
  }
  for (std::int64_t i = 0; i < remaining + degree + 1; ++i) {
    k[i] = k[i + offset];
  }
  return remaining;
}

/// Where `track_zero` stopped.
///
/// Attributes:
///     x (double): The last polygon zero computed.
///     status (TrackStatus): How the iteration stopped, never whether `x` may be reported, which is
///         the caller's residual test alone.
///     residual (double): `|f(x)|`, evaluated on the refined window on *every* exit.
///     num_coeffs (std::int64_t): The coefficient count left behind in the refined window.
///     index (std::int64_t): The tracked index left behind in the refined window.
struct TrackResult {
  double x;
  TrackStatus status;
  double residual;
  std::int64_t num_coeffs;
  std::int64_t index;
};

/// Track one sign change of the control polygon to a zero of the spline.
///
/// Runs Algorithm 2 of Morken and Reimers on the window buffers: the zero of the control polygon is
/// inserted as a new knot, and the tracked index follows the same sign change, which after the
/// insertion sits at `index` or at `index + 1` and nowhere else. Both buffers are refined in place.
///
/// The stopping rule is the paper's: the spread of the last `degree` iterates, measured relative to
/// `max(|knots[index]|, |knots[index+degree]|, domain_length)`. That window spans exactly one
/// quadratic step, since the error is quadratic per `degree - 1` insertions and not per insertion.
///
/// Args:
///     knots (std::span<double>): Knot window, refined in place.
///     coeffs (std::span<double>): Coefficient window, refined in place; its size is the buffer
///         capacity.
///     num_coeffs (std::int64_t): Number of valid entries in `coeffs`.
///     degree (int): Polynomial degree, at least 1.
///     index (std::int64_t): Zero index to track, inside the window. The caller guarantees that
///         `is_zero_index` holds for it.
///     tol (double): Relative stagnation tolerance.
///     domain_length (double): Length of the spline's parametric domain, used as a floor for the
///         tolerance scale.
///     max_insertions_per_zero (int): Insertion budget for this zero.
///     iterates (std::span<double>): Ring buffer of at least `degree` entries.
///     work (std::span<double>): Scratch buffer of at least `degree + 1` entries.
///
/// Returns:
///     TrackResult: The last polygon zero, the stop reason, `|f(x)|` and the window left behind.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `find_roots` instead.
[[nodiscard]] inline TrackResult track_zero(std::span<double> knots, std::span<double> coeffs,
                                            std::int64_t num_coeffs, int degree, std::int64_t index,
                                            double tol, double domain_length,
                                            int max_insertions_per_zero, std::span<double> iterates,
                                            std::span<double> work) noexcept {
  double* k = knots.data();
  double* c = coeffs.data();
  double* ring = iterates.data();

  std::int64_t num_iterates = 0;
  double previous_x = 0.0;
  double x = 0.0;

  // Never write past the working buffers: the budget is whichever runs out first, the insertions
  // allowed for this zero or the room left in the arrays, keeping `degree + 1` slots for the split
  // that follows a reported zero.
  const std::int64_t room = static_cast<std::int64_t>(coeffs.size()) - degree - 1 - num_coeffs;
  const std::int64_t budget = (std::min)(static_cast<std::int64_t>(max_insertions_per_zero), room);
  std::int64_t inserted = 0;

  for (;;) {
    const double left_value = c[index - 1];
    const double right_value = c[index];
    const double left_abscissa = knot_average(knots, degree, index - 1);
    const double right_abscissa = knot_average(knots, degree, index);

    // Zero of the segment joining (left_abscissa, left_value) and (right_abscissa, right_value),
    // as a clamped convex combination: the two values straddle zero, so the parameter is in [0,
    // 1] up to rounding.
    double lam = -left_value / (right_value - left_value);
    lam = (std::min)((std::max)(lam, 0.0), 1.0);
    const double extent = right_abscissa - left_abscissa;
    const double step = lam * extent;
    x = left_abscissa + step;
    x = (std::min)((std::max)(x, left_abscissa), right_abscissa);

    // A knot average is a sum of `degree` terms divided by `degree`, so it carries a relative error
    // of a few ulp and can land below its own smallest term once the knots have piled up to within
    // that much. The theory places every iterate in `[t[index], t[index+degree]]`, and the span
    // search below only ever walks right from `index`, so restore the left end rather than let the
    // search return a span that does not bracket x.
    x = (std::max)(x, k[index]);

    // A repeated iterate is a fixed point of the iteration, hence a zero of the spline
    // (Morken-Reimers, Lemma 13 and Corollary 14). The corollary is a statement about exact
    // arithmetic, so what it licenses is stopping here, not the value being a zero: the residual
    // is what says that.
    //
    // Its hypothesis is that the tracked sign change has `degree - 1` of its active knots collapsed
    // onto the iterate, `t[index + 1] = ... = t[index + degree - 1] = x`, and the knots being
    // ascending the two ends of that run decide it. Repetition alone is a much weaker signal: it is
    // also what a secant that is nearly horizontal in the control polygon produces, once the
    // correction it asks for falls below an ulp of the Greville abscissa, and that happens at a
    // shallow non-zero minimum just as readily as at a zero. Falling through instead inserts x
    // again, which collapses one more knot onto it, so the hypothesis is reached in at most `degree
    // - 1` further insertions and the budget bounds the rest. For degree 1 the run is empty and the
    // corollary asks for nothing.
    if (num_iterates > 0 && x == previous_x) {
      const bool collapsed =
          degree < corollary_14_degree || (k[index + 1] == x && k[index + degree - 1] == x);
      if (collapsed) {
        const double residual = residual_at(knots, coeffs, num_coeffs, degree, x, work);
        return {x, TrackStatus::stagnated, residual, num_coeffs, index};
      }
    }

    // Their Lemma 3: an iterate that reaches the right end of its Greville interval carries
    // `coeffs[index] = 0`, hence f(x) = 0.
    //
    // The lemma places x in `(knot_average(index - 1), knot_average(index)]`, and that interval is
    // empty exactly when the two abscissae coincide, which happens exactly when `knots[index] ==
    // knots[index + degree]`, that is when the knot run at x carries multiplicity `degree + 1` and
    // the spline is C^-1 there. In that one excluded case the secant through the two coefficients
    // is vertical, its zero is x for every `lam`, and nothing forces `coeffs[index]` to vanish; the
    // sign change is then the spline jumping across the axis rather than meeting it.
    //
    // Either way the conclusion is tested rather than assumed, so the two cases need no separating.
    // Reaching this branch means `x` is exactly the knot `knots[index + degree]` and the run ending
    // there has multiplicity at least `degree`, so exactly one basis function is non-zero at x on
    // the side the tracking came from and `coeffs[index]` is the value of the spline there: the
    // right limit when the run reaches back to `index`, the left limit when it starts past it, and
    // both at once where the spline is continuous. Refining instead would insert a knot into a run
    // the method keeps at multiplicity at most `degree`, and divide by a zero-length span.
    //
    // It is a coefficient rather than an evaluation, but that does not make it exact and it gets
    // the same threshold as every other exit. The window it belongs to was refined by Boehm
    // insertions whose alphas are differences of knots, so it carries the coordinate's own
    // resolution. Only the two domain endpoints, read off the *unrefined* input, are free of it.
    if (x >= k[index + degree]) {
      return {x, TrackStatus::stagnated, std::abs(c[index]), num_coeffs, index};
    }

    ring[num_iterates % degree] = x;
    previous_x = x;
    ++num_iterates;

    if (num_iterates >= degree) {
      double lowest = ring[0];
      double highest = ring[0];
      for (int i = 1; i < degree; ++i) {
        lowest = (std::min)(lowest, ring[i]);
        highest = (std::max)(highest, ring[i]);
      }
      double scale = (std::max)(std::abs(k[index]), std::abs(k[index + degree]));
      scale = (std::max)(scale, domain_length);
      const double spread = highest - lowest;
      const double allowed = tol * scale;
      if (spread <= allowed) {
        const double residual = residual_at(knots, coeffs, num_coeffs, degree, x, work);
        return {x, TrackStatus::stagnated, residual, num_coeffs, index};
      }
    }

    if (inserted >= budget) {
      const double residual = residual_at(knots, coeffs, num_coeffs, degree, x, work);
      return {x, TrackStatus::budget, residual, num_coeffs, index};
    }

    std::int64_t span = index;
    while (x >= k[span + 1]) {
      ++span;
    }
    insert_knot(knots, coeffs, num_coeffs, degree, x, span);
    ++num_coeffs;
    ++inserted;

    // Their Algorithm 2, step 3: the tracked sign change is now at `index` or at `index + 1`; if it
    // is at neither, it has vanished.
    if (!is_zero_index(coeffs, num_coeffs, index)) {
      ++index;
      if (!is_zero_index(coeffs, num_coeffs, index)) {
        const double residual = residual_at(knots, coeffs, num_coeffs, degree, x, work);
        return {x, TrackStatus::vanished, residual, num_coeffs, index};
      }
    }
  }
}

/// The zeros the kernel reported, and what it gave up on.
///
/// Attributes:
///     roots (std::vector<double>): The reported zeros, ascending; exactly `count` entries.
///     count (std::int64_t): The number of reported zeros.
///     truncated (std::int64_t): The reported zeros whose insertion budget ran out first, so that
///         they sit at the last iterate reached rather than at a stagnated one.
///     abandoned (std::int64_t): The sign changes left without a zero to show for them, whether
///         because the budget ran out before the residual certified one or because the working
///         arrays had no room to track them, each of which may be a zero this call does not report.
struct MorkenReimersResult {
  std::vector<double> roots;
  std::int64_t count;
  std::int64_t truncated;
  std::int64_t abandoned;
};

/// Compute every zero of a scalar spline by the Morken-Reimers method.
///
/// Scans the control polygon left to right and tracks each sign change. What the polygon supplies
/// is a *place to look*, never a certificate: the tracking stops for three different reasons, none
/// of which implies that the point it stopped at is a zero, so every iterate is reported only when
/// `|f(x)| <= root_tol`. That test is what separates a tangential zero, where the two sign changes
/// bracketing it collapse, from a false warning of the variation-diminishing bound; what rejects a
/// sign change across a knot of multiplicity `degree + 1`, where the spline is C^-1 and jumps
/// across the axis without reaching it; and what rejects an iteration that stopped moving somewhere
/// other than a zero.
///
/// The two domain endpoints are decided by the tighter `zero_tol` instead, on the coefficient an
/// open knot vector interpolates there. That coefficient comes from the input untouched, so unlike
/// everything the tracking produces it carries no positional error at all and only the evaluation
/// term applies.
///
/// Args:
///     knots (std::span<const double>): Open (clamped) knot vector of size `coeffs.size() + degree
///         + 1`.
///     degree (int): Polynomial degree, at least 1.
///     coeffs (std::span<const double>): B-spline coefficients of the scalar spline.
///     tol (double): Relative stagnation tolerance for the iterates.
///     zero_tol (double): Absolute residual below which a value taken from the unrefined input
///         counts as zero, that is the error of evaluating the spline. Used for the two domain
///         endpoints and for the split barrier.
///     root_tol (double): Absolute residual below which an iterate the tracking produced counts as
///         a zero. Larger than `zero_tol` by what the parametric tolerance the iteration stops at
///         leaves behind, and derived in `find_roots`.
///     max_insertions (int): Insertion budget per zero, at least 0.
///
/// Returns:
///     MorkenReimersResult: The zeros, ascending, and the counts of the zeros it truncated and the
///     sign changes it abandoned.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `find_roots` instead.
[[nodiscard]] inline MorkenReimersResult morken_reimers_roots(
    std::span<const double> knots, int degree, std::span<const double> coeffs, double tol,
    double zero_tol, double root_tol, int max_insertions_per_zero) {
  const auto num_coeffs = static_cast<std::int64_t>(coeffs.size());
  // A connected spline has at most `num_coeffs - 1` sign changes in its control polygon and
  // therefore at most that many interior zeros, plus the two endpoints. The extra slot is what lets
  // the loop below stop on a full array instead of writing past it.
  const std::int64_t roots_size = num_coeffs + 2;
  std::vector<double> roots(static_cast<std::size_t>(roots_size));
  double* r = roots.data();
  std::int64_t count = 0;
  std::int64_t truncated = 0;
  std::int64_t abandoned = 0;

  const double start = knots[static_cast<std::size_t>(degree)];
  const double end = knots[static_cast<std::size_t>(num_coeffs)];
  const double domain_length = end - start;

  // The working arrays hold the spline from the zero reported last to the end of the domain,
  // refined in place. They only ever grow by insertions between two reported zeros, because
  // reporting one compacts everything behind it.
  const std::int64_t capacity = num_coeffs + degree + 2 + 4 * std::int64_t{max_insertions_per_zero};
  std::vector<double> live_coeffs(static_cast<std::size_t>(capacity));
  std::vector<double> live_knots(static_cast<std::size_t>(capacity + degree + 1));
  std::vector<double> iterates(static_cast<std::size_t>((std::max)(degree, 1)));
  std::vector<double> work(static_cast<std::size_t>(degree) + 1);

  for (std::int64_t i = 0; i < num_coeffs; ++i) {
    live_coeffs[static_cast<std::size_t>(i)] = coeffs[static_cast<std::size_t>(i)];
  }
  for (std::int64_t i = 0; i < num_coeffs + degree + 1; ++i) {
    live_knots[static_cast<std::size_t>(i)] = knots[static_cast<std::size_t>(i)];
  }
  std::int64_t num_live = num_coeffs;

  // An open knot vector interpolates its first and last coefficient, so an endpoint zero is read
  // straight off the coefficients (their pseudo code).
  double previous_root = start - domain_length - 1.0;
  if (std::abs(coeffs[0]) <= zero_tol) {
    r[count] = start;
    ++count;
    previous_root = start;
  }

  std::int64_t scan_from = 1;
  while (count + 1 < roots_size) {
    std::int64_t index = zero_index(live_coeffs, num_live, scan_from);
    if (index < 0) {
      break;
    }

    // Stop while a full tracking and the split that may follow it still fit. Reaching this means
    // the insertions between two reported zeros outran the room the compaction reclaims, which the
    // capacity above is sized to prevent; the count of unfinished zeros is what tells the caller.
    // This sign change is never tracked at all, so it is abandoned rather than truncated.
    if (num_live + max_insertions_per_zero + 2 * (std::int64_t{degree} + 1) > capacity) {
      ++abandoned;
      break;
    }

    const TrackResult tracked = track_zero(live_knots, live_coeffs, num_live, degree, index, tol,
                                           domain_length, max_insertions_per_zero, iterates, work);
    const double x = tracked.x;
    num_live = tracked.num_coeffs;
    index = tracked.index;

    // The residual, and nothing else, is what makes an iterate a zero. The three ways the tracking
    // can stop are three provenances, not three entitlements: the iteration stopped moving, the
    // polygon lost its sign change, or the budget ran out, and each of them can stop where the
    // spline does not vanish. The theory that says otherwise holds in exact arithmetic, where a
    // coefficient reaches zero only near a zero of the spline; in floating point it can reach zero
    // for reasons of its own, so the conclusion is tested rather than inherited.
    //
    // `root_tol` and not `zero_tol`, for every one of them alike. The iteration stops at a
    // *parametric* resolution, so a correctly located zero leaves `|f'|` times that behind however
    // exactly f is then evaluated, and the same is true of a residual read off a coefficient of the
    // refined window, since the insertions that produced it take differences of knots and so carry
    // the coordinate's resolution too. Only the two domain endpoints below, taken from the
    // unrefined input, are free of it.
    const bool certified = tracked.residual <= root_tol;

    if (tracked.status == TrackStatus::budget) {
      if (certified) {
        ++truncated;
      } else {
        ++abandoned;
      }
    }

    // The sweep is strictly left to right, so an iterate that did not pass the zero reported last
    // is that same zero seen again through another sign change of the polygon, not a new one.
    if (certified && previous_root < x && x <= end) {
      r[count] = x;
      ++count;
      previous_root = x;
      scan_from = index + 1;
      if (x < live_knots[static_cast<std::size_t>(num_live)]) {
        // Split at the zero just reported so that the next sign change cannot drift back onto it,
        // then drop everything left of the split: it is behind the sweep for good, and dropping it
        // is what keeps the working arrays from growing with each zero and what leaves an open knot
        // vector on the rest of the domain.
        const SplitResult split =
            split_at_root(live_knots, live_coeffs, num_live, degree, x, zero_tol);
        num_live =
            drop_window_head(live_knots, live_coeffs, split.num_coeffs, degree, split.offset);
        scan_from = 1;
      }
    } else {
      // One index past the sign change just rejected, and no further: the refinement left by the
      // tracking can leave a second sign change of its own immediately to the right, and that one
      // may well bracket the next zero. Stepping over the knot run instead would skip it. This is
      // what makes the sweep terminate, the index returned by `track_zero` being at least the one
      // the scan handed it.
      scan_from = index + 1;
    }
  }

  if (std::abs(coeffs[static_cast<std::size_t>(num_coeffs - 1)]) <= zero_tol) {
    r[count] = end;
    ++count;
  }

  roots.resize(static_cast<std::size_t>(count));
  return {std::move(roots), count, truncated, abandoned};
}

/// The runs of root candidates after merging.
///
/// Attributes:
///     merged (std::vector<double>): The midpoint of each run, ascending; empty for no input.
///     labels (std::vector<std::int64_t>): For each input root, the index in `merged` of the run it
///         joined.
struct MergedRoots {
  std::vector<double> merged;
  std::vector<std::int64_t> labels;
};

/// Merge runs of ascending roots that lie within their own merge radii.
///
/// The scan reports one root per sign change of the control polygon, and a zero of even
/// multiplicity is bracketed by two of them, so consecutive reports have to be collapsed. Two
/// neighbours join the same run when their separation does not exceed the larger of the two radii;
/// each run is replaced by its midpoint, which for a bracketed tangential zero is the better
/// estimate of the two.
///
/// A midpoint is a value nobody tracked, so the run each input joined is reported alongside it: the
/// caller cannot certify the midpoint from here, and without the labels it could not put a run back
/// the way it found it when the certificate fails.
///
/// Args:
///     roots (std::span<const double>): Ascending root candidates.
///     radii (std::span<const double>): Per-root merge radius, same length.
///
/// Returns:
///     MergedRoots: The run midpoints and, per input root, the run it joined.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `find_roots` instead.
[[nodiscard]] inline MergedRoots merge_roots(std::span<const double> roots,
                                             std::span<const double> radii) {
  const std::size_t count = roots.size();
  MergedRoots out{{}, std::vector<std::int64_t>(count)};
  if (count == 0) {
    return out;
  }

  out.merged.reserve(count);
  double run_start = roots[0];
  double run_end = roots[0];
  out.labels[0] = 0;
  for (std::size_t i = 1; i < count; ++i) {
    if (roots[i] - run_end <= (std::max)(radii[i], radii[i - 1])) {
      run_end = roots[i];
    } else {
      out.merged.push_back(0.5 * (run_start + run_end));
      run_start = roots[i];
      run_end = roots[i];
    }
    out.labels[i] = static_cast<std::int64_t>(out.merged.size());
  }
  out.merged.push_back(0.5 * (run_start + run_end));
  return out;
}

/// Take the maximum of the absolute values, a NaN in the input giving NaN (as `numpy.max` does).
///
/// Args:
///     values (std::span<const double>): The values; every `|v|` is `>= 0`, so `0` is the identity
///         of the maximum.
///
/// Returns:
///     double: `max |v|`, or NaN if any value is NaN.
[[nodiscard]] inline double max_abs_propagating_nan(std::span<const double> values) noexcept {
  double best = 0.0;
  for (const double v : values) {
    const double a = std::abs(v);
    if (std::isnan(a)) {
      return a;
    }
    if (a > best) {
      best = a;
    }
  }
  return best;
}

/// Take the elementwise minimum of two values, NaN if either is (as `numpy.minimum` does).
[[nodiscard]] inline double minimum_propagating_nan(double a, double b) noexcept {
  return (a < b || std::isnan(a)) ? a : b;
}

/// Take the elementwise maximum of two values, NaN if either is (as `numpy.maximum` does).
[[nodiscard]] inline double maximum_propagating_nan(double a, double b) noexcept {
  return (a > b || std::isnan(a)) ? a : b;
}

/// Widen a storage-typed range to `double`, exactly.
template <StorageFloat T>
[[nodiscard]] std::vector<double> widen(std::span<const T> values) {
  std::vector<double> out;
  out.reserve(values.size());
  for (const T v : values) {
    out.push_back(static_cast<double>(v));
  }
  return out;
}

/// Validate that a B-spline is a scalar univariate one with positive weights.
///
/// Args:
///     b (const Bspline<T>&): Input to validate.
///
/// Returns:
///     Bspline<T>: The B-spline, converted to an open (clamped), non-periodic representation when
///     it is not already one.
///
/// Raises:
///     value_error: It is not univariate (`dim != 1`), not scalar-valued (`rank != 1`), of degree
///         zero, or rational with a non-positive weight (a NaN weight is not positive).
template <StorageFloat T>
[[nodiscard]] Bspline<T> validate_bspline_for_roots(const Bspline<T>& b) {
  if (b.dim() != 1) {
    throw value_error("Bspline must be univariate (dim == 1), got dim=" + std::to_string(b.dim()));
  }
  if (b.rank() != 1) {
    throw value_error("Bspline must be scalar valued (rank == 1), got rank=" +
                      std::to_string(b.rank()));
  }
  if (b.space().spaces()[0].degree() < 1) {
    throw value_error(
        "Root finding needs degree >= 1: a degree-zero spline is piecewise "
        "constant, so its zero set is a union of knot intervals, not a set of points");
  }
  if (b.rational()) {
    const std::span<const T> cp = b.control_points();
    const std::size_t cols = b.cp_size();
    double min_weight = static_cast<double>(cp[cols - 1]);
    bool all_positive = min_weight > 0.0;
    for (std::size_t i = 1; i < cp.size() / cols; ++i) {
      const auto w = static_cast<double>(cp[i * cols + cols - 1]);
      all_positive = all_positive && w > 0.0;
      // The minimum propagates NaN, as `numpy.min` does, so the message names it.
      if (!std::isnan(min_weight) && (std::isnan(w) || w < min_weight)) {
        min_weight = w;
      }
    }
    if (!all_positive) {
      throw value_error(
          "Rational B-splines must have strictly positive weights, so that the "
          "zeros of the numerator are the zeros of the mapping. Got min weight " +
          py_repr(min_weight));
    }
  }

  if (!b.space().spaces()[0].has_open_knots()) {
    return to_open_bspline<T>(b);
  }
  return b;
}

/// Reject a spline that vanishes identically on a knot interval.
///
/// Such a spline is *disconnected* in the sense of Morken and Reimers: on that interval every zero
/// is a zero of the spline, so there is no set of isolated roots to report and the
/// variation-diminishing bound that the method rests on no longer holds. The paper assumes
/// connectedness throughout and notes that the degeneracy is easy to detect, which is what this
/// does.
///
/// The window maximum of the Python implementation is replaced by the equivalent "every
/// coefficient is within `zero_tol`": a maximum is `<= zero_tol` exactly when each of its terms
/// is, a NaN term or a NaN `zero_tol` making both false.
///
/// Args:
///     coeffs (std::span<const double>): Scalar B-spline coefficients.
///     knots (std::span<const double>): Open knot vector.
///     degree (int): Polynomial degree.
///     zero_tol (double): Residual below which a coefficient counts as zero.
///
/// Raises:
///     value_error: Every coefficient active on some non-empty knot interval is zero to within
///         `zero_tol`; the first such interval is named.
inline void check_connected(std::span<const double> coeffs, std::span<const double> knots,
                            int degree, double zero_tol) {
  const auto num_coeffs = static_cast<std::int64_t>(coeffs.size());
  for (std::int64_t span = degree; span < num_coeffs; ++span) {
    const auto lo = static_cast<std::size_t>(span);
    if (!(knots[lo] < knots[lo + 1])) {
      continue;
    }
    bool vanishing = true;
    for (std::int64_t j = span - degree; j <= span; ++j) {
      vanishing = vanishing && std::abs(coeffs[static_cast<std::size_t>(j)]) <= zero_tol;
    }
    if (vanishing) {
      throw value_error("The spline vanishes identically on the knot interval [" +
                        py_repr(knots[lo]) + ", " + py_repr(knots[lo + 1]) +
                        "], so its zero set is not a set of isolated points. Split the spline to "
                        "exclude that interval.");
    }
  }
}

/// Bound `|f'|` over the whole parametric domain, from the hodograph.
///
/// The derivative of a spline is the spline of degree `degree - 1` whose coefficients are `degree *
/// (c[i] - c[i-1]) / (t[i+degree] - t[i])`, and a spline never leaves the range of its own
/// coefficients, so the largest of those difference quotients bounds `|f'|` everywhere.
///
/// A zero-length denominator is skipped rather than read as an infinite slope. It occurs only at a
/// knot of multiplicity `degree + 1`, where the spline jumps and no finite slope describes it; an
/// infinite bound there would turn the residual test into an unconditional accept, which is the
/// defect this whole threshold exists to prevent, and the jump itself is rejected on its own
/// residual. At least one denominator is always positive: an open knot vector carries exactly
/// `degree + 1` copies of the start, so `knots[degree + 1] > knots[1]`.
///
/// The bound is over the whole domain, where the threshold it feeds is applied at one zero at a
/// time. A spline with one steep region therefore gets a threshold as loose everywhere as that
/// region needs, which errs towards accepting rather than towards rejecting a genuine zero.
/// Sharpening it to the knot span holding the zero is a tolerance-policy question and is *not* a
/// matter of computing the same quantity on the working window: after a hundred insertions the
/// local knot gaps there are degenerate and the same formula returns a bound larger than the
/// spline's own range. A correct local version would have to index the *original* knot vector by
/// parametric position, and `root_tol` reaches the kernel as one scalar decided before any zero is
/// located, so it would also have to become a per-span quantity the kernel indexes.
///
/// That work is deliberately not done, and the reason is that the looseness has not been shown to
/// cost anything: a randomized search for a live false positive (a value this threshold accepts
/// that the local bound would reject) found none. It is therefore a tightening rather than a bug
/// fix, and the direction that would be fatal here is the opposite one: for a root finder, a
/// `root_tol` that is too *large* returns values that are not roots, and no local sharpening can
/// make the threshold larger than this bound.
///
/// Args:
///     coeffs (std::span<const double>): Scalar B-spline coefficients.
///     knots (std::span<const double>): Open knot vector.
///     degree (int): Polynomial degree, at least 1.
///
/// Returns:
///     double: An upper bound on `|f'|` over the domain, NaN if a quotient is NaN (as `numpy.max`
///     would give). Zero for a spline whose coefficients are all equal, which is the correct bound
///     for a constant.
[[nodiscard]] inline double slope_bound(std::span<const double> coeffs,
                                        std::span<const double> knots, int degree) noexcept {
  const std::size_t num_coeffs = coeffs.size();
  const auto p = static_cast<std::size_t>(degree);
  // Every quotient is `>= 0` or NaN, so 0 is the identity of the maximum.
  double largest = 0.0;
  for (std::size_t i = 0; i + 1 < num_coeffs; ++i) {
    const double gap = knots[p + 1 + i] - knots[1 + i];
    if (!(gap > 0.0)) {
      continue;
    }
    const double quotient = std::abs(coeffs[i + 1] - coeffs[i]) / gap;
    if (std::isnan(quotient)) {
      largest = quotient;
      break;
    }
    if (quotient > largest) {
      largest = quotient;
    }
  }
  return static_cast<double>(degree) * largest;
}

/// Build the float64, non-rational spline whose zeros are being reported.
///
/// For a rational spline the zeros of the mapping are the zeros of its numerator, the weights being
/// strictly positive, so the numerator is what the derivatives used by `merge_radii` must be taken
/// of.
///
/// Args:
///     knots (std::span<const double>): Its knot vector, as `double`.
///     coeffs (std::span<const double>): Its scalar coefficients, as `double`.
///     degree (int): Polynomial degree.
///
/// Returns:
///     Bspline<double>: A non-rational `double` spline with the same zeros (knots not snapped).
inline Bspline<double> scalar_numerator(std::span<const double> knots,
                                        std::span<const double> coeffs, int degree) {
  std::vector<BsplineSpace1D<double>> directions;
  directions.emplace_back(knots, degree, false, false);
  return Bspline<double>(BsplineSpace<double>(std::move(directions)), coeffs, false);
}

/// Evaluate the `order`-th derivative of a scalar spline at the roots, in absolute value.
///
/// Raises:
///     value_error: The basis derivative recurrence met a zero divisor (a knot repeated more than
///         `degree + 1` times within a root's support); Python raises `ZeroDivisionError` there.
inline std::vector<double> abs_derivative_at(const Bspline<double>& numerator,
                                             std::span<const double> roots, int order) {
  const mdview<const double, 2> points(roots.data(), roots.size(), 1);
  const int orders[1] = {order};
  const DerivativeValues<double> derived = bspline::evaluate_derivatives<double, double>(
      numerator, points, std::span<const int>(orders));
  if (!derived.complete) {
    throw value_error(
        "division by zero in the basis derivative recurrence: a knot is repeated more than "
        "degree + 1 times in the support of a root");
  }
  std::vector<double> out(roots.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = std::abs(derived.values.data()[i]);
  }
  return out;
}

/// Compute the per-root radius within which two reports are the same zero.
///
/// Around a zero of multiplicity `m` the set where `|f| <= zero_tol` has half-width `(m! * zero_tol
/// / |f^(m)|) ** (1 / m)`, from the first non-vanishing term of the Taylor expansion. The two cases
/// that are computed are `m = 1` and `m = 2`; every one of them is a length, so all of them
/// transform correctly under a change of parametric scale.
///
/// The result is floored at the parametric resolution `tol * max(|x|, domain_length)` and capped at
/// `domain_length * (degree! * zero_tol / coeff_scale) ** (1 / degree)`, which is that same
/// half-width for the highest multiplicity the spline space admits, evaluated where `f^(degree)`
/// has its natural magnitude `coeff_scale / domain_length ** degree`. The cap is what covers a zero
/// of multiplicity three or more, where the two computed derivatives both vanish and their radii
/// would otherwise be dominated by rounding; without a cap at all, such a zero would merge every
/// later root into itself.
///
/// `degree!` is accumulated in `double`, which is exact up to `22!` (every factorial to there has
/// an odd part below `2^53`). Above that the running product may differ from Python's correctly
/// rounded `int` to `float` conversion of the exact factorial by an ulp or more: an admitted
/// difference in a merge-radius cap, which is a heuristic length and not a bound.
///
/// Args:
///     numerator (const Bspline<double>&): Scalar, non-rational spline whose zeros are sought.
///     roots (std::span<const double>): Ascending root candidates.
///     zero_tol (double): Residual below which a value counts as zero.
///     coeff_scale (double): Largest absolute coefficient of the spline.
///     domain_length (double): Length of the parametric domain.
///     tol (double): Relative parametric tolerance.
///
/// Returns:
///     std::vector<double>: Merge radius for each root.
///
/// Raises:
///     value_error: As `abs_derivative_at`.
[[nodiscard]] inline std::vector<double> merge_radii(const Bspline<double>& numerator,
                                                     std::span<const double> roots, double zero_tol,
                                                     double coeff_scale, double domain_length,
                                                     double tol) {
  const int degree = numerator.space().spaces()[0].degree();
  const double infinity = std::numeric_limits<double>::infinity();
  const std::size_t n = roots.size();

  const std::vector<double> first = abs_derivative_at(numerator, roots, 1);
  std::vector<double> radius(n);
  for (std::size_t i = 0; i < n; ++i) {
    radius[i] = first[i] > 0.0 ? zero_tol / first[i] : infinity;
  }

  if (degree >= curvature_degree) {
    const std::vector<double> second = abs_derivative_at(numerator, roots, 2);
    const double twice_zero_tol = 2.0 * zero_tol;
    for (std::size_t i = 0; i < n; ++i) {
      const double curved = std::sqrt(second[i] > 0.0 ? twice_zero_tol / second[i] : infinity);
      radius[i] = minimum_propagating_nan(radius[i], curved);
    }
  }

  double factorial = 1.0;
  for (int k = 2; k <= degree; ++k) {
    factorial *= static_cast<double>(k);
  }
  const double scaled = factorial * zero_tol;
  const double ratio = scaled / coeff_scale;
  const double half_width = std::pow(ratio, 1.0 / static_cast<double>(degree));
  const double cap = domain_length * half_width;
  for (std::size_t i = 0; i < n; ++i) {
    const double floor_scale = maximum_propagating_nan(std::abs(roots[i]), domain_length);
    const double lower_bound = tol * floor_scale;
    radius[i] = maximum_propagating_nan(minimum_propagating_nan(radius[i], cap), lower_bound);
  }
  return radius;
}

/// Collapse repeated reports of one zero, keeping only the merges that are zeros.
///
/// The midpoint of a run is a value nobody tracked, so it inherits no certificate from the reports
/// it replaces and takes the same residual test they did. Where it fails, the run was not one zero
/// seen several times: the reports are kept as they were, each already certified on its own.
///
/// Without that test a single over-wide radius is enough to return a value that is not a root,
/// which is what happens at high degree: the cap `domain_length * (degree! * zero_tol /
/// coeff_scale) ** (1 / degree)` grows with degree and passes the whole domain around degree 17,
/// and runs are joined on the *larger* of two radii, so one such radius joins every later root into
/// its own run.
///
/// The candidates are finite and ascending (each satisfies `previous_root < x <= end`), so the
/// final sort meets no NaN.
///
/// Args:
///     numerator (const Bspline<double>&): Scalar, non-rational spline whose zeros are sought.
///     roots (std::span<const double>): Ascending root candidates.
///     radii (std::span<const double>): Per-root merge radius, same length.
///     root_tol (double): Residual below which a value counts as a zero.
///
/// Returns:
///     std::vector<double>: Ascending roots, with every certified run collapsed to its midpoint and
///     every other run left as it was.
inline std::vector<double> merge_certified(const Bspline<double>& numerator,
                                           std::span<const double> roots,
                                           std::span<const double> radii, double root_tol) {
  const MergedRoots runs = merge_roots(roots, radii);
  const mdview<const double, 2> points(runs.merged.data(), runs.merged.size(), 1);
  const NdArray<double> values = bspline::evaluate<double, double>(numerator, points);
  std::vector<bool> certified(runs.merged.size());
  bool all_certified = true;
  for (std::size_t i = 0; i < certified.size(); ++i) {
    certified[i] = std::abs(values.data()[i]) <= root_tol;
    all_certified = all_certified && certified[i];
  }
  if (all_certified) {
    return runs.merged;
  }
  std::vector<double> out;
  for (std::size_t i = 0; i < certified.size(); ++i) {
    if (certified[i]) {
      out.push_back(runs.merged[i]);
    }
  }
  for (std::size_t i = 0; i < roots.size(); ++i) {
    if (!certified[static_cast<std::size_t>(runs.labels[i])]) {
      out.push_back(roots[i]);
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace detail

/// Find every zero of a scalar univariate B-spline.
///
/// Uses the Morken-Reimers method: the zero of the control polygon is inserted as a new knot, over
/// and over, until the iterates stagnate. The iteration needs no starting value, converges for any
/// spline, and reaches a simple zero quadratically (once per `degree - 1` insertions, not once per
/// insertion).
///
/// A value is reported only when the residual there is within what a zero may legitimately leave
/// behind: the de Boor evaluation-error bound of the spline, plus `|f'| * 2 * tol * scale`, the
/// value the spline reaches at the parametric resolution the iteration stops at. The control
/// polygon says where to look and never certifies on its own. The residual test is also the only
/// way a zero of even multiplicity is reported, since it does not change the sign of the spline and
/// the sign changes bracketing it vanish under refinement.
///
/// At a knot of multiplicity `degree + 1` the spline is discontinuous, and a zero of the piece on
/// either side is reported at the knot. Evaluating there returns the *right* limit, so such a root
/// can show a residual as large as the jump when the zero belongs to the left piece.
///
/// Args:
///     b (const Bspline<T>&): A univariate (`dim() == 1`), scalar-valued (`rank() == 1`) B-spline.
///         Periodic or unclamped splines are converted to an open representation first. For
///         rational splines the zeros are those of the numerator, which are the zeros of the
///         mapping because the weights are positive.
///     tol (std::optional<double>): Relative parametric tolerance. The iteration stops when the
///         spread of the last `degree` iterates falls to `tol * max(|t_a|, |t_{a+degree}|,
///         domain_length)`, so that it transforms correctly when the parametric domain is scaled
///         or moved away from the origin. Must satisfy `!(tol <= 0)`, so a NaN passes (and
///         certifies nothing). Defaults to `tolerance::get_strict<T>()`.
///     max_insertions_override (std::optional<int>): Insertion budget per zero, at least 0.
///         Defaults to `max_insertions(degree)`. The Python implementation has no such argument; it
///         lets a caller exercise the exhaustion paths.
///
/// Returns:
///     BsplineRootSet: The roots, ascending and always `double`, empty if the spline has no zero;
///     and, for the two `RuntimeWarning`s of the Python implementation, the status and counts.
///
/// Raises:
///     value_error: `b` is not univariate and scalar valued, has degree zero, is rational with a
///         non-positive (or NaN) weight, `tol` is not positive, the budget is negative, the spline
///         is identically zero, it vanishes identically on a knot interval, or the basis
///         derivative recurrence met a zero divisor while merging.
template <StorageFloat T>
[[nodiscard]] BsplineRootSet find_roots(const Bspline<T>& b,
                                        std::optional<double> tol = std::nullopt,
                                        std::optional<int> max_insertions_override = std::nullopt) {
  const Bspline<T> spline = detail::validate_bspline_for_roots<T>(b);
  if (tol.has_value() && *tol <= 0.0) {
    throw value_error("tol must be positive, got " + py_repr(*tol));
  }
  if (max_insertions_override.has_value() && *max_insertions_override < 0) {
    throw value_error("max_insertions must be non-negative, got " +
                      std::to_string(*max_insertions_override));
  }

  const double resolved_tol =
      tol.has_value() ? *tol : static_cast<double>(tolerance::get_strict<T>());
  const auto epsilon = static_cast<double>(tolerance::machine_epsilon<T>());

  const BsplineSpace<T> whole = spline.space();
  const BsplineSpace1D<T>& space = whole.spaces()[0];
  const int degree = space.degree();
  const std::vector<double> knots = detail::widen<T>(space.knots());
  const std::span<const T> stored = spline.control_points();
  const std::size_t cols = spline.cp_size();
  std::vector<double> coeffs(stored.size() / cols);
  for (std::size_t i = 0; i < coeffs.size(); ++i) {
    coeffs[i] = static_cast<double>(stored[i * cols]);
  }
  const std::size_t num_coeffs = coeffs.size();

  const double coeff_scale = detail::max_abs_propagating_nan(coeffs);
  if (coeff_scale == 0.0) {
    throw value_error("The spline is identically zero, so every point of its domain is a root");
  }
  const double evaluation_floor =
      static_cast<double>(degree + 1) * detail::zero_tol_safety * epsilon;
  const double zero_tol = coeff_scale * (std::max)(evaluation_floor, resolved_tol);
  detail::check_connected(coeffs, knots, degree, zero_tol);

  // What a *tracked* iterate may leave behind, as opposed to a value read straight off a
  // coefficient. `zero_tol` is the error of evaluating the spline, and on its own it rejects the
  // genuine zeros of a steep one: the iteration stops at a parametric resolution of `stop_slack *
  // tol * scale`, and a zero located to that much leaves `|f'|` times it behind however exactly the
  // spline is then evaluated. Both terms are values of the spline, so the sum transforms correctly
  // under a change of either scale: the second is a slope times a length.
  const double domain_start = knots[static_cast<std::size_t>(degree)];
  const double domain_end = knots[num_coeffs];
  const double domain_length = domain_end - domain_start;
  double parametric_scale = (std::max)(std::abs(domain_start), std::abs(domain_end));
  parametric_scale = (std::max)(parametric_scale, domain_length);
  const double slope = detail::slope_bound(coeffs, knots, degree);
  const double stop_slope = detail::stop_slack * slope;
  const double stop_length = resolved_tol * parametric_scale;
  const double stop_residual = stop_slope * stop_length;
  const double root_tol = zero_tol + stop_residual;

  const int budget =
      max_insertions_override.has_value() ? *max_insertions_override : max_insertions(degree);
  detail::MorkenReimersResult raw =
      detail::morken_reimers_roots(knots, degree, coeffs, resolved_tol, zero_tol, root_tol, budget);

  std::vector<double> roots = std::move(raw.roots);
  if (raw.count > 1) {
    const Bspline<double> numerator = detail::scalar_numerator(knots, coeffs, degree);
    const std::vector<double> radii =
        detail::merge_radii(numerator, roots, zero_tol, coeff_scale, domain_length, resolved_tol);
    roots = detail::merge_certified(numerator, roots, radii, root_tol);
  }

  RootStatus status = RootStatus::complete;
  if (raw.abandoned > 0) {
    status = RootStatus::budget_exhausted;
  } else if (raw.truncated > 0) {
    status = RootStatus::stalled_at_last_iterate;
  }
  return {std::move(roots), status, raw.count, raw.truncated, raw.abandoned, budget, root_tol};
}

}  // namespace pantr::bspline
