/// @file knot_removal.hpp
/// @brief Knot removal (Piegl and Tiller A5.8) with a tolerance on the deviation of the control
/// points, and the tolerances that decide what the default removes.
///
/// **What the kernel's tolerance grades.** `detail::remove_knot_kernel` accepts a removal when the
/// Euclidean distance between two reconstructions of the same control point stays within `tol`.
/// That distance is taken over *the columns of the array it is given*, which for a rational spline
/// are the homogeneous coordinates `[w x, w y, w z, w]` and not the projected point. A caller's
/// budget, on the other hand, is a distance in projected space. The two are different quantities,
/// and `homogeneous_deviation_tolerance` is where they are reconciled.
#pragma once

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/detail/refine_common.hpp"
#include "pantr/bspline/knots.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/tolerance.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bspline {

/// Extra factor on the strict tier for the round-off floor of an exact removal.
///
/// With `get_strict` this is `8 eps`, and it is applied per step of the reconstruction chain: the
/// floor is `8 (degree + 1) eps * control-point scale`.
///
/// **Shape.** Removing a knot once runs Algorithm A5.8's two one-sided reconstructions towards each
/// other, at most `degree + 1` control points from each side, and reads off how far apart they
/// land. Each step is one divided difference of the previous result, so an `eps`-sized error enters
/// at every one of the `degree + 1` of them and the largest quantity any of them handles is the
/// control-point scale. The steps are *not* convex combinations (the `1 / alpha` in the recurrence
/// can amplify), so this is an operation count and not a proved bound; the eight epsilons per step
/// are the same doubling of the strict tier that `knot_merge_safety` applies, covering an FMA
/// contraction and a differing summation order on top of the count.
///
/// **Checked** by a test (`cpp/tests/test_bspline_knot_removal.cpp`): an exactly removable knot,
/// one just inserted, is removed under the default floor over degrees, ranks and geometry scales,
/// with a margin against `8 (degree + 1)`, and the margin shows no trend with the degree, so the
/// `degree + 1` is headroom the derivation asks for and not a term fitted to data.
///
/// All of that describes **one** control point, which is what a curve has per index. See
/// `roundoff_deviation_floor` for the `sqrt` that a surface or a volume adds.
inline constexpr double removal_floor_safety = 2.0;

/// Get the magnitude a control-point distance is relative to.
///
/// `max(bounding-box diagonal, largest ||P_i||)` over the rows of `ctrl`, the same pairing of
/// extent and coordinate magnitude that `knot_scale` makes in parametric space. The extent alone is
/// not enough: a small part sitting at `x = 1e6` has control points whose differences carry an
/// absolute error of `eps * 1e6` however small the part is.
///
/// A third length, what the *parametrization* can resolve, has no counterpart here: knot removal
/// grades a distance between control points, which no parametric coordinate enters.
///
/// Computed on the array as given, so for a rational spline it is the scale of the *homogeneous*
/// control points, which is the space the kernel measures in. Computed in `double`.
///
/// Args:
///     ctrl (std::span<const T>): Control points, flat; every `cols` consecutive values are one
///         control point (any order of the points).
///     cols (std::size_t): The coordinates per control point, at least 1.
///
/// Returns:
///     double: The scale, zero only for a control net that is identically zero; NaN when a
///     coordinate is NaN (as NumPy's `max` propagates it).
template <StorageFloat T>
[[nodiscard]] double control_point_scale(std::span<const T> ctrl, std::size_t cols) {
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  const std::size_t rows = ctrl.size() / cols;
  std::vector<double> lo(cols, std::numeric_limits<double>::infinity());
  std::vector<double> hi(cols, -std::numeric_limits<double>::infinity());
  double magnitude = 0.0;
  for (std::size_t r = 0; r < rows; ++r) {
    double sq = 0.0;
    for (std::size_t c = 0; c < cols; ++c) {
      const auto x = static_cast<double>(ctrl[r * cols + c]);
      if (std::isnan(x) || std::isnan(lo[c])) {
        lo[c] = nan;
        hi[c] = nan;
      } else {
        lo[c] = (std::min)(lo[c], x);
        hi[c] = (std::max)(hi[c], x);
      }
      sq += x * x;
    }
    const double norm = std::sqrt(sq);
    if (std::isnan(magnitude)) {
      continue;
    }
    magnitude = std::isnan(norm) ? norm : (std::max)(magnitude, norm);
  }
  double diag_sq = 0.0;
  for (std::size_t c = 0; c < cols; ++c) {
    const double extent = hi[c] - lo[c];
    diag_sq += extent * extent;
  }
  const double diagonal = std::sqrt(diag_sq);
  // Python's `max(diagonal, magnitude)`: the second wins only when strictly larger, so a NaN
  // magnitude loses to the diagonal and a NaN diagonal wins.
  return magnitude > diagonal ? magnitude : diagonal;
}

/// Count the control points the kernel's one distance will span, reducing `axis`.
///
/// The control net is flattened along `axis` to `(num_basis[axis], rest)`, so the kernel sees every
/// control point of every *other* direction at once and its Euclidean distance runs over all of
/// them. That count is the product of the basis counts excluding `axis`: one for a curve.
///
/// Kept as its own function because getting it the wrong way round (taking the reduced axis
/// instead of the others) is a mistake no behavioural test can catch: the floor carries a large
/// margin, so any positive multiplier still admits the removal. Only a direct assertion on this
/// number can.
///
/// Args:
///     shape (std::span<const std::size_t>): The control-net extents `(*num_basis, cp_size)`.
///     axis (std::size_t): The parametric direction being reduced, `< shape.size() - 1`.
///
/// Returns:
///     std::size_t: The number of stacked control points, at least one for a non-empty net.
[[nodiscard]] inline std::size_t num_stacked_control_points(std::span<const std::size_t> shape,
                                                            std::size_t axis) {
  std::size_t n = 1;
  for (std::size_t d = 0; d + 1 < shape.size(); ++d) {
    if (d != axis) {
      n *= shape[d];
    }
  }
  return n;
}

/// Get the deviation budget that admits exactly the removals round-off allows.
///
/// `removal_floor_safety * get_strict<T>() * (degree + 1) * sqrt(num_stacked) * scale`:
/// `8 (degree + 1) eps` times `control_point_scale` for one control point (derivation in
/// `removal_floor_safety`), times a `sqrt` for how many control points the kernel grades at once.
///
/// This is the default budget, and it is deliberately *not* a geometric one. A knot that a spline
/// does not actually need is removable exactly, and the only thing standing between it and a
/// bit-for-bit reconstruction is the round-off of the reconstruction itself. A budget at that level
/// therefore means "remove what is genuinely redundant and nothing else", which is the one default
/// that needs no arbitrary constant. Anything looser is a statement about how much geometry the
/// caller is willing to lose, which only the caller can make.
///
/// **Why `sqrt(num_stacked)`.** For a curve the kernel compares one control point against one
/// control point. For a surface or a volume it does not: the flattening collapses every *other*
/// direction into the trailing axis first, so the single Euclidean distance A5.8 computes runs over
/// `num_stacked` control points at once. Stacking is exactly what a 2-norm aggregates,
/// `|| (d_1, ..., d_n) || = sqrt(sum ||d_k||^2) <= sqrt(n) max_k ||d_k||`, so a slice whose every
/// point is at its own per-point floor lands at `sqrt(num_stacked)` times that floor. This is a
/// bound, not a statistical estimate: it holds however the per-point errors are correlated. Without
/// it the default is too tight by that factor on anything but a curve, and it *silently* refuses an
/// exactly redundant knot, with no error and no warning, once the stack is large enough.
///
/// What the aggregate cannot see is *which* point moved. A slice could in principle spend the
/// whole budget on one control point, which would then sit `sqrt(num_stacked)` above its own floor.
/// Grading per point instead would need the kernel to return a per-point distance rather than one
/// scalar. In practice the reconstructions share their weights and run on data of one magnitude, so
/// the error is spread across the slice rather than concentrated.
///
/// It is expressed in the units the kernel measures in, so for a rational spline it is a
/// homogeneous distance and does **not** go through `homogeneous_deviation_tolerance`: a floor is a
/// statement about the arithmetic, and the arithmetic happens in homogeneous coordinates.
///
/// Args:
///     ctrl (std::span<const T>): Control points as the kernel will see them, homogeneous for a
///         rational spline; every `cols` consecutive values are one control point.
///     cols (std::size_t): The coordinates per control point.
///     degree (int): Polynomial degree in the direction being reduced.
///     num_stacked (std::size_t): How many control points the kernel's distance spans, i.e. the
///         product of the basis counts of every direction *except* the one being reduced. One for a
///         curve.
///
/// Returns:
///     double: Absolute deviation budget, in the units of `ctrl`.
template <StorageFloat T>
[[nodiscard]] double roundoff_deviation_floor(std::span<const T> ctrl, std::size_t cols, int degree,
                                              std::size_t num_stacked) {
  const double scale = control_point_scale<T>(ctrl, cols);
  const double per_point = removal_floor_safety * static_cast<double>(tolerance::get_strict<T>()) *
                           static_cast<double>(degree + 1) * scale;
  return per_point * std::sqrt(static_cast<double>(num_stacked));
}

/// Pull a projected-space deviation budget back to homogeneous control points.
///
/// `TOL = d w_min / (1 + |P|_max)`, with `|P|_max = max_i ||P_i||_2` in **Euclidean** coordinates
/// and `w_min = min_i w_i`. This is Eq. (5.30) of Piegl and Tiller, *The NURBS Book*, 2nd ed.
/// (Springer, 1997), p. 185, given there for Algorithm A5.8, which `remove_knot_kernel` otherwise
/// implements faithfully.
///
/// Why it is needed: the kernel measures one Euclidean distance `D` over every column of the array
/// it is handed, which for a rational spline are the homogeneous coordinates `[w x, w y, w z, w]`.
/// Moving the homogeneous point by `dP^w` moves the projected point by `(dP^w_xyz - P dP^w_w) / w`,
/// so
///
///     |dP| <= (|dP^w_xyz| + |P| |dP^w_w|) / w <= D (1 + |P|) / w,
///
/// and requiring `D <= d w_min / (1 + |P|_max)` gives `|dP| <= d` for every point, since
/// `w >= w_min` and `|P| <= |P|_max`. Without it, a deviation carried by the **weight** column is
/// graded as though it were a coordinate: with the perturbation in the weight and the coordinates
/// large, the actual projected deviation exceeds the requested tolerance by orders of magnitude. A
/// circle cannot show this, because its weights are structurally tied to its coordinates.
///
/// Two honest caveats.
///
/// `w_min = min_i w_i`, and weights are required strictly positive (see Raises), so it is a minimum
/// and not a minimum of absolute values.
///
/// The bound is **tight for a weight-carried perturbation and conservative for a purely geometric
/// one**, by up to a factor `1 + |P|_max`. Splitting `D` optimally between the two parts gives
/// `sqrt(1 + |P|_max^2)`, which differs from `1 + |P|_max` by at most `sqrt(2)`, so almost none of
/// that conservatism comes from the triangle inequality: it comes from the kernel compressing a
/// length and a weight into a single Euclidean distance, which loses the direction. Removing it
/// would mean the kernel grading the two parts separately.
///
/// That is also why `1 + |P|_max` adds a pure number to a length, and the consequence has to be
/// stated rather than waved past. The inhomogeneity is not an error in the formula: it is inherited
/// from `D` itself, whose squared value adds `(length * weight)^2` to `weight^2`, so `D` is not a
/// quantity of one unit and no pullback out of it can be. The **bound is sound at every scale**
/// (the derivation above uses nothing about the size of `|P|`) but its *tightness* is not scale
/// invariant, and where the crossover sits depends on the unit the model is written in:
///
///   * `|P|_max >> 1`: tight for a weight-carried deviation, loose by `|P|_max` for a purely
///     coordinate-carried one.
///   * `|P|_max << 1`: the reverse. Tight for a coordinate-carried deviation, loose by
///     `1 / |P|_max` for a weight-carried one.
///
/// The verdict is therefore covariant in the safe direction only: a small model refuses removals
/// its budget would allow, and never accepts one it would not.
///
/// Closing that gap is a kernel change, not a better pullback. It needs the kernel to grade
/// `||dP^w_xyz|| / w` and `|P| |dP^w_w| / w` against `d` **separately**, which is dimensionally
/// clean, tight in both regimes and fully covariant. Compressing them into one Euclidean distance
/// is what loses the direction, and once lost it cannot be recovered here.
///
/// Args:
///     ctrl (std::span<const T>): Homogeneous control points, flat; every `cols` consecutive values
///         are one control point, the last the weight.
///     cols (std::size_t): The coordinates per control point, the weight included, at least 2.
///     tol_euclidean (double): The caller's deviation budget, a distance in projected space.
///
/// Returns:
///     double: The equivalent budget on homogeneous control points.
///
/// Raises:
///     value_error: Any weight is non-positive. The conversion divides by the weights to recover
///         `|P|_max`, and a zero weight has no projected point to convert to; the same
///         precondition is checked by `locate` and `find_roots`, and stating it here is what keeps
///         a degenerate weight from silently producing `TOL = 0` and refusing every removal in the
///         direction.
template <StorageFloat T>
[[nodiscard]] double homogeneous_deviation_tolerance(std::span<const T> ctrl, std::size_t cols,
                                                     double tol_euclidean) {
  const std::size_t rows = ctrl.size() / cols;
  double w_min = std::numeric_limits<double>::infinity();
  for (std::size_t r = 0; r < rows; ++r) {
    const auto w = static_cast<double>(ctrl[r * cols + cols - 1]);
    if (std::isnan(w)) {
      w_min = w;
      break;
    }
    w_min = (std::min)(w_min, w);
  }
  if (w_min <= 0.0) {
    throw value_error(
        "Knot removal with an explicit tolerance needs strictly positive weights to convert the "
        "tolerance into homogeneous coordinates; the smallest is " +
        py_repr(w_min) + ".");
  }
  double p_max = 0.0;
  for (std::size_t r = 0; r < rows; ++r) {
    const auto w = static_cast<double>(ctrl[r * cols + cols - 1]);
    double sq = 0.0;
    for (std::size_t c = 0; c + 1 < cols; ++c) {
      const double x = static_cast<double>(ctrl[r * cols + c]) / w;
      sq += x * x;
    }
    const double norm = std::sqrt(sq);
    if (std::isnan(p_max)) {
      continue;
    }
    p_max = std::isnan(norm) ? norm : (std::max)(p_max, norm);
  }
  return tol_euclidean * w_min / (1.0 + p_max);
}

/// The result of removing knots from one direction.
///
/// Attributes:
///     knots (std::vector<T>): The knot vector after the removals.
///     ctrl (NdArray<T>): Control points, extents `(rows, cols)`.
///     removals (int): How many knots were removed, `0 <= removals <= num`.
template <StorageFloat T>
struct RemovalResult {
  std::vector<T> knots;
  NdArray<T> ctrl;
  int removals;
};

namespace detail {

/// Convert a tolerance to the storage type for a comparison made in it, as NumPy compares a
/// `float32` array with a Python float: the float is rounded to `float32` first. A tolerance past
/// the range of `float` becomes infinite (the conversion itself would be undefined).
template <StorageFloat T>
[[nodiscard]] T tolerance_in(double tol) noexcept {
  if constexpr (std::same_as<T, float>) {
    constexpr double largest = static_cast<double>((std::numeric_limits<float>::max)());
    if (tol > largest) {
      return std::numeric_limits<float>::infinity();
    }
    if (tol < -largest) {
      return -std::numeric_limits<float>::infinity();
    }
  }
  return static_cast<T>(tol);
}

/// Find the last index and the multiplicity of the knot class a value falls in.
///
/// The class is the first one (of all of them, ghost knots included) whose representative lies
/// within `tol` of `knot_value`. The comparison is absolute: a relative one (NumPy's `isclose` with
/// its default `rtol = 1e-5`) would match a knot up to `1e-5 |knot_value|` away, so a removal
/// request would silently target a different knot (10.0 away at a knot value of `1e6`). It is made
/// in the storage type, as NumPy makes `unique_knots - knot_value` against a Python float.
///
/// Args:
///     knots (std::span<const T>): The knot vector.
///     degree (int): Degree.
///     knot_value (double): The knot value to locate.
///     tol (double): Absolute parametric tolerance.
///
/// Returns:
///     std::pair<std::size_t, std::int64_t>: `(r, s)`, the index of the last knot of the class and
///     its multiplicity.
///
/// Raises:
///     value_error: No class within `tol` of `knot_value`.
template <StorageFloat T>
[[nodiscard]] std::pair<std::size_t, std::int64_t> find_knot_index_and_multiplicity(
    std::span<const T> knots, int degree, double knot_value, double tol) {
  const KnotClasses<T> classes = unique_knots_and_multiplicity<T>(knots, degree, tol, false);
  const T value = static_cast<T>(knot_value);
  const T limit = tolerance_in<T>(tol);
  std::size_t end = 0;
  for (std::size_t i = 0; i < classes.knots.size(); ++i) {
    end += static_cast<std::size_t>(classes.multiplicities[i]);
    if (std::abs(static_cast<T>(classes.knots[i] - value)) <= limit) {
      return {end - 1, classes.multiplicities[i]};
    }
  }
  throw value_error("Knot value " + py_repr(knot_value) +
                    " not found in knot vector (tolerance=" + py_repr(tol) + ").");
}

/// Remove a single knot value up to `num` times from a 1D B-spline.
///
/// Implements Algorithm A5.8 from *The NURBS Book* (Piegl and Tiller, 1997). Each removal is
/// accepted only if the resulting geometric deviation does not exceed `tol`.
///
/// Arithmetic. The knots are of the storage type and so are the control points; the blending
/// weights `alpha` are formed in `double` from the (exact) value and the knots, and every control
/// point written into `temp` is rounded to the storage type, as the kernel this replaces did.
///
/// Args:
///     degree (int): Polynomial degree.
///     knots (std::span<const T>): Knot vector, `n + degree + 2` entries.
///     ctrl (mdview<const T, 2>): Control points, extents `(n + 1, rank)`.
///     knot_value (double): The knot value to remove.
///     knot_index (std::size_t): The index `r` of the last knot of the class of `knot_value`, with
///         `degree + multiplicity <= r <= n` (a knot strictly inside the domain).
///     multiplicity (int): The multiplicity `s` of that class.
///     num (int): The most removals to attempt. **Load-bearing for memory safety**: see the note.
///     tol (double): The largest Euclidean distance between the two intermediate control-point
///         approximations that still accepts a removal.
///
/// Returns:
///     RemovalResult<T>: The knots, the control points and the number of removals.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed). For general use, call
///     `remove_knot_1d`.
///
///     **Precondition on `num`, memory safety and not merely correctness:**
///     `2 * num <= degree + multiplicity`. The scratch buffer `temp` is sized `(2 degree + 1,
///     rank)` once, before the removal loop. Pass `t` writes up to row `last + 1 - off` with `off =
///     first - 1`; since `first` decreases and `last` increases by one per pass, that highest row
///     is `degree - s + 2 t + 2`, growing by two each time. Keeping it within `2 degree` gives `t
///     <= (degree + s - 2) / 2` for the final pass `t = num - 1`, which is the bound above. The
///     same passes read control rows `off` and `last + 1`, which the precondition on `knot_index`
///     keeps inside `[0, n]`: `off >= s - num >= 0` and `last + 1 <= r <= n`. `remove_knot_1d`
///     establishes both by capping `num` at `min(multiplicity, degree)`, which implies the bound
///     and is one removal conservative for some pairs (`degree = 4`, `multiplicity = 2` would admit
///     `num = 3`), and by refusing a knot that is not strictly inside the domain.
template <StorageFloat T>
[[nodiscard]] RemovalResult<T> remove_knot_kernel(int degree, std::span<const T> knots,
                                                  mdview<const T, 2> ctrl, double knot_value,
                                                  std::size_t knot_index, int multiplicity, int num,
                                                  double tol) {
  using Index = std::ptrdiff_t;
  const std::size_t rank = ctrl.extent(1);
  const auto n = static_cast<Index>(ctrl.extent(0)) - 1;  // last control point index
  const Index m = n + degree + 1;                         // last knot index
  const Index order = degree + 1;
  const auto r = static_cast<Index>(knot_index);
  const Index s = multiplicity;

  // Work on copies so the originals are not modified.
  std::vector<T> wk(knots.begin(), knots.end());
  std::vector<T> wc(ctrl.data_handle(), ctrl.data_handle() + ctrl.extent(0) * rank);
  // Temporary control point buffer (the maximum size needed: 2 degree + 1 rows).
  std::vector<T> temp(static_cast<std::size_t>(2 * degree + 1) * rank);

  const auto at = [rank](std::vector<T>& v, Index row, std::size_t col) -> T& {
    return v[static_cast<std::size_t>(row) * rank + col];
  };

  const Index fout = (2 * r - s - degree) / 2;  // first output row that changes (non-negative)
  Index first = r - degree;
  Index last = r - s;

  int removals = 0;
  for (Index t = 0; t < num; ++t) {
    const Index off = first - 1;
    // Seed the temp buffer.
    for (std::size_t d = 0; d < rank; ++d) {
      at(temp, 0, d) = at(wc, off, d);
      at(temp, last + 1 - off, d) = at(wc, last + 1, d);
    }

    // Compute new control points from both sides.
    Index i = first;
    Index ii = 1;
    Index j = last;
    Index jj = last - off;

    while (j - i > t) {
      // The knot differences are of the storage type, the blending weights of `double`. A zero
      // difference gives a zero weight, as in the source; a valid space cannot produce one here (it
      // needs `degree + 2` equal knots).
      const T denom_i = static_cast<T>(wk[static_cast<std::size_t>(i + order + t)] -
                                       wk[static_cast<std::size_t>(i)]);
      const double alpha_i =
          denom_i != T{0} ? (knot_value - static_cast<double>(wk[static_cast<std::size_t>(i)])) /
                                static_cast<double>(denom_i)
                          : 0.0;

      const T denom_j = static_cast<T>(wk[static_cast<std::size_t>(j + order)] -
                                       wk[static_cast<std::size_t>(j - t)]);
      const double alpha_j =
          denom_j != T{0}
              ? (knot_value - static_cast<double>(wk[static_cast<std::size_t>(j - t)])) /
                    static_cast<double>(denom_j)
              : 0.0;

      for (std::size_t d = 0; d < rank; ++d) {
        // A zero `alpha` here is a division by zero (inf or NaN, not a trap). It needs the knot
        // value on a knot of an earlier class, or on the next one: the value is within `tol` of its
        // class and the classes are more than `tol` apart, so it is unreachable for a class found
        // by `find_knot_index_and_multiplicity` in a space.
        //
        // Each product is a statement of its own, so no multiply-add is contracted: whether a
        // removal is accepted compares a distance of a few units of round-off with a budget of
        // that size, and a fused rung would round the recurrence differently from the others (and
        // from the kernel this replaces) and could flip the verdict of an adversarial input.
        const double from_left = (1.0 - alpha_i) * static_cast<double>(at(temp, ii - 1, d));
        const double from_right = alpha_j * static_cast<double>(at(temp, jj + 1, d));
        at(temp, ii, d) = static_cast<T>((static_cast<double>(at(wc, i, d)) - from_left) / alpha_i);
        at(temp, jj, d) =
            static_cast<T>((static_cast<double>(at(wc, j, d)) - from_right) / (1.0 - alpha_j));
      }

      ++i;
      ++ii;
      --j;
      --jj;
    }

    // Check deviation.
    bool accepted = false;
    if (j - i < t) {
      // Odd case: the distance between the two meeting points. The differences are of the storage
      // type, their squares too, and the sum of `double`.
      double dist = 0.0;
      for (std::size_t d = 0; d < rank; ++d) {
        const T diff = static_cast<T>(at(temp, ii - 1, d) - at(temp, jj + 1, d));
        const double square = static_cast<double>(static_cast<T>(diff * diff));
        dist += square;
      }
      dist = std::sqrt(dist);
      accepted = dist <= tol;
    } else {
      // Even case: the blended point against the both-side approximation.
      const T denom_i = static_cast<T>(wk[static_cast<std::size_t>(i + order + t)] -
                                       wk[static_cast<std::size_t>(i)]);
      const double alpha_i =
          denom_i != T{0} ? (knot_value - static_cast<double>(wk[static_cast<std::size_t>(i)])) /
                                static_cast<double>(denom_i)
                          : 0.0;
      double dist = 0.0;
      for (std::size_t d = 0; d < rank; ++d) {
        const double from_right = alpha_i * static_cast<double>(at(temp, ii + t + 1, d));
        const double from_left = (1.0 - alpha_i) * static_cast<double>(at(temp, ii - 1, d));
        const double blended = from_right + from_left;
        const double diff = static_cast<double>(at(wc, i, d)) - blended;
        const double square = diff * diff;
        dist += square;
      }
      dist = std::sqrt(dist);
      accepted = dist <= tol;
    }

    if (!accepted) {
      break;
    }

    // Accept the removal: write temp back into the working control points.
    i = first;
    j = last;
    while (j - i > t) {
      for (std::size_t d = 0; d < rank; ++d) {
        at(wc, i, d) = at(temp, i - off, d);
        at(wc, j, d) = at(temp, j - off, d);
      }
      ++i;
      --j;
    }

    --first;
    ++last;
    ++removals;
  }

  if (removals == 0) {
    NdArray<T> same({ctrl.extent(0), ctrl.extent(1)}, for_overwrite);
    std::copy(ctrl.data_handle(), ctrl.data_handle() + ctrl.extent(0) * ctrl.extent(1),
              same.data());
    return RemovalResult<T>{std::vector<T>(knots.begin(), knots.end()), std::move(same), 0};
  }

  // Compact the knot vector: shift the entries after the removed knots.
  for (Index k = r + 1; k <= m; ++k) {
    wk[static_cast<std::size_t>(k - removals)] = wk[static_cast<std::size_t>(k)];
  }
  wk.resize(knots.size() - static_cast<std::size_t>(removals));

  // Compact the control points: shift the entries to fill the removed rows. The gap boundaries
  // after the alternating shift:
  Index j_out = fout;
  Index i_out = j_out;
  for (Index k = 1; k < removals; ++k) {
    if (k % 2 == 1) {
      ++i_out;
    } else {
      --j_out;
    }
  }
  Index dest = j_out;
  for (Index k = i_out + 1; k <= n; ++k) {
    for (std::size_t d = 0; d < rank; ++d) {
      at(wc, dest, d) = at(wc, k, d);
    }
    ++dest;
  }

  const std::size_t new_rows = ctrl.extent(0) - static_cast<std::size_t>(removals);
  NdArray<T> new_ctrl({new_rows, ctrl.extent(1)}, for_overwrite);
  std::copy(wc.begin(), wc.begin() + static_cast<std::ptrdiff_t>(new_rows * rank), new_ctrl.data());
  return RemovalResult<T>{std::move(wk), std::move(new_ctrl), removals};
}

}  // namespace detail

/// Remove a single knot value from a 1D B-spline.
///
/// Args:
///     knots (std::span<const T>): Knot vector, `n + degree + 2` entries.
///     degree (int): Polynomial degree.
///     ctrl (mdview<const T, 2>): Control points, extents `(n + 1, rank)`.
///     knot_value (double): The knot value to remove.
///     num (std::optional<long long>): The most removals; no value removes as many as possible, up
///         to the multiplicity capped at the degree.
///     tol_space (double): Absolute parametric tolerance for the knot comparisons.
///     tol_deviation (double): The largest control-point deviation an accepted removal may have,
///         already in the units the kernel measures in (homogeneous for a rational spline; see the
///         file note). Infinity accepts every removal.
///
/// Returns:
///     RemovalResult<T>: The new knots, control points and the number of removals.
///
/// Raises:
///     value_error: `num < 1`; `knot_value` is not found in the knot vector; it is a boundary knot
///         of the domain, or not strictly inside it (a ghost knot of an unclamped vector); or the
///         control points do not match the knots.
template <StorageFloat T>
[[nodiscard]] RemovalResult<T> remove_knot_1d(std::span<const T> knots, int degree,
                                              mdview<const T, 2> ctrl, double knot_value,
                                              std::optional<long long> num, double tol_space,
                                              double tol_deviation) {
  if (num.has_value() && *num < 1) {
    throw value_error("num must be a positive integer or None, got " + std::to_string(*num) + ".");
  }
  detail::check_net_matches("remove_knots", ctrl.extent(0), knots.size(), degree);

  const auto [r, s] =
      detail::find_knot_index_and_multiplicity<T>(knots, degree, knot_value, tol_space);

  // Boundary knots of the domain cannot be removed.
  const auto p = static_cast<std::size_t>(degree);
  const auto domain_lo = static_cast<double>(knots[p]);
  const auto domain_hi = static_cast<double>(knots[knots.size() - p - 1]);
  if (std::abs(knot_value - domain_lo) <= tol_space) {
    throw value_error("Cannot remove boundary knot " + py_repr(knot_value) + " (domain start).");
  }
  if (std::abs(knot_value - domain_hi) <= tol_space) {
    throw value_error("Cannot remove boundary knot " + py_repr(knot_value) + " (domain end).");
  }
  // The kernel reads control rows `r - degree - num .. r`: only a class strictly between the class
  // of the first and the one of the last domain knot keeps them inside the net. A value outside the
  // domain (the ghost knots of an unclamped vector) is such a class too, and the source read (and
  // wrote) outside its arrays there.
  if (r < p + static_cast<std::size_t>(s) || r + p + 1 >= knots.size()) {
    throw value_error("Cannot remove knot " + py_repr(knot_value) +
                      ": it is not an interior knot of the domain [" + py_repr(domain_lo) + ", " +
                      py_repr(domain_hi) + "].");
  }

  // Cap at the actual multiplicity (and at degree per the algorithm). This clamp is load-bearing
  // for memory safety, not just a sanity bound: the kernel sizes its scratch buffer for
  // `2 num <= degree + s` and writes outside it past that. `min(s, degree)` implies that bound, so
  // the clamp has to stay, including when the caller supplied `num` explicitly.
  const long long max_removals =
      (std::min)(static_cast<long long>(s), static_cast<long long>(degree));
  const long long capped = num.has_value() ? (std::min)(*num, max_removals) : max_removals;
  return detail::remove_knot_kernel<T>(degree, knots, ctrl, knot_value, r, static_cast<int>(s),
                                       static_cast<int>(capped), tol_deviation);
}

/// Remove knots, direction by direction, from a B-spline.
///
/// For each direction, iterates over the distinct knot values to remove, applying single-knot
/// removal sequentially (each removal changes the knot indices for the next value). The deviation
/// budget is resolved per direction, from the control points as they stand: the degree, the
/// control-point scale and the number of stacked points all differ between directions, and the net
/// is carried forward from the previous one.
///
/// Args:
///     b (const Bspline<T>&): The B-spline: non-periodic in every direction.
///     knot_values (std::span<const std::span<const T>>): For each direction the distinct knot
///         values to remove; an empty span skips the direction.
///     num (std::optional<long long>): The most removals per knot value; no value removes as many
///         as possible.
///     tol (std::optional<double>): The deviation budget, a distance in **projected** space. No
///         value uses the round-off floor of `roundoff_deviation_floor`, which removes what is
///         redundant to within the arithmetic and nothing more.
///
/// Returns:
///     Bspline<T>: The B-spline with the reduced knot vectors.
///
/// Raises:
///     value_error: A direction is periodic; `knot_values.size() != dim`; every direction is
///         skipped; or as `remove_knot_1d` and `homogeneous_deviation_tolerance` for a direction.
template <StorageFloat T>
[[nodiscard]] Bspline<T> remove_knots(const Bspline<T>& b,
                                      std::span<const std::span<const T>> knot_values,
                                      std::optional<long long> num, std::optional<double> tol) {
  const BsplineSpace<T> whole = b.space();
  const std::vector<BsplineSpace1D<T>>& spaces = whole.spaces();
  for (std::size_t d = 0; d < spaces.size(); ++d) {
    if (spaces[d].periodic()) {
      throw value_error("Knot removal is not supported for periodic B-splines (direction " +
                        std::to_string(d) + " is periodic).");
    }
  }
  if (knot_values.size() != b.dim()) {
    throw value_error("knot_values sequence length (" + std::to_string(knot_values.size()) +
                      ") must match dim (" + std::to_string(b.dim()) + ").");
  }
  const bool any = std::any_of(knot_values.begin(), knot_values.end(),
                               [](std::span<const T> k) { return !k.empty(); });
  if (!any) {
    throw value_error(
        "At least one direction must have a non-empty array of knot values to remove.");
  }
  const std::size_t cols = b.cp_size();
  // The control-net extents as the directions are processed in order: the number of stacked
  // points of a direction is read from the net as it stands.
  std::vector<std::size_t> extents(b.shape().begin(), b.shape().end());
  return detail::map_directions<T>(
      b, [&](std::size_t d) { return !knot_values[d].empty(); },
      [&](std::size_t d, const BsplineSpace1D<T>& space,
          mdview<const T, 2> flat) -> std::pair<BsplineSpace1D<T>, NdArray<T>> {
        // The flattened net holds the control points of the net as it stands, `cols` consecutive
        // values to a point; the scale and the weights do not depend on their order.
        const std::span<const T> all(flat.data_handle(), flat.extent(0) * flat.extent(1));
        double tol_deviation = 0.0;
        if (!tol.has_value()) {
          tol_deviation = roundoff_deviation_floor<T>(all, cols, space.degree(),
                                                      num_stacked_control_points(extents, d));
        } else if (b.rational()) {
          // Not scaled by `num_stacked`: an explicit budget is a promise about each control
          // point, and widening the aggregate would let one point absorb the whole of it. Grading
          // the stacked distance against the per-point budget is conservative, which is the sound
          // direction for a caller's own bound.
          tol_deviation = homogeneous_deviation_tolerance<T>(all, cols, *tol);
        } else {
          tol_deviation = *tol;
        }

        std::vector<T> knots(space.knots().begin(), space.knots().end());
        NdArray<T> ctrl({flat.extent(0), flat.extent(1)}, for_overwrite);
        std::copy(flat.data_handle(), flat.data_handle() + all.size(), ctrl.data());
        for (const T val : knot_values[d]) {
          RemovalResult<T> r =
              remove_knot_1d<T>(std::span<const T>(knots), space.degree(), ctrl.template view<2>(),
                                static_cast<double>(val), num, space.tolerance(), tol_deviation);
          knots = std::move(r.knots);
          ctrl = std::move(r.ctrl);
        }
        extents[d] = ctrl.shape()[0];
        return {BsplineSpace1D<T>(std::span<const T>(knots), space.degree()), std::move(ctrl)};
      });
}

}  // namespace pantr::bspline
