/// @file tolerances.hpp
/// @brief The two tolerances of `THBSplineSpace`: cell membership of a query point, and the
/// residual of a prolongation column.
#pragma once

#include "pantr/config.hpp"
#include "pantr/tolerance.hpp"

#include <algorithm>
#include <cmath>

namespace pantr::thb {

/// Extra factor on the strict tier for deciding that a point lies in a cell.
///
/// Together with `tolerance::get_strict<double>()` this is `8 eps`, the same rule and the same
/// number as `pantr::bspline::knot_tolerance`, deliberately, because it is the same question. A
/// cell boundary is a parametric coordinate, and asking whether a point has fallen off it is asking
/// whether two parametric coordinates are the same one.
///
/// What has to be covered is the caller's route to the point. A point in cell `[lo, hi]` is almost
/// always formed as `lo + xi * (hi - lo)` for some `xi` in `[0, 1]`: a subtraction, a
/// multiplication and an addition, three roundings, each on a quantity no larger than
/// `max(|lo|, |hi|)`, so the computed point sits within `3 eps max(|lo|, |hi|)` of the exact one.
/// The bounds themselves arrive from `HierarchicalGrid::cell_bounds`, which subdivides the axis by
/// the same kind of arithmetic and at the same magnitude. Four epsilons cover either route and
/// eight covers both plus an FMA contraction, which is the doubling
/// `pantr::bspline::knot_merge_safety` applies for the same reason.
///
/// The magnitude it multiplies is the cell's own, not the axis's: see
/// `cell_membership_tolerance`.
inline constexpr double cell_membership_safety = 2.0;

/// Get the absolute slack allowed at one boundary of a cell along one axis.
///
/// `cell_membership_safety * get_strict<double>() * max(|lo|, |hi|, hi - lo)`, i.e. `8 eps` times
/// the cell's own magnitude. The magnitude is what `pantr::bspline::knot_scale` computes, applied
/// to the two-element vector `[lo, hi]` rather than to a whole knot vector.
///
/// The scale is the cell's, not the axis's, and the difference is not cosmetic. A point in the cell
/// is formed from `lo` and `hi`, so its round-off is relative to those two coordinates; a cell of
/// width `1e-3` on a domain reaching `1e6` would, on the axis scale, accept a point half a billion
/// cell-widths outside. The coordinate terms `|lo|` and `|hi|` are what carry the case of a narrow
/// cell far from the origin, where the width alone would demand agreement the arithmetic that
/// produced the bounds cannot deliver.
///
/// No floor is applied, for the reason `pantr::bspline::knot_scale` gives: a floor of one would be
/// a physical choice this layer is not entitled to make, and it destroys covariance on a domain
/// smaller than one unit.
///
/// The precision is `double`, not the root space's, which is where this departs from
/// `pantr::bspline::knot_tolerance`'s `get_strict<T>()`. It grades different objects:
/// `cell_bounds` returns `double` whatever the root space is made of, and the query points are
/// `double`, so `double` is the precision of both sides here. A caller whose points carry only
/// `float` information should widen them before asking, rather than have the containment check
/// widened by eight million for everyone.
///
/// Args:
///     lo (double): The cell's lower bound along the axis.
///     hi (double): The cell's upper bound along the axis.
///
/// Returns:
///     double: The slack, in the units of the parametric coordinate.
[[nodiscard]] inline double cell_membership_tolerance(double lo, double hi) noexcept {
  const double scale = (std::max)((std::max)(std::abs(lo), std::abs(hi)), hi - lo);
  return cell_membership_safety * tolerance::get_strict<double>() * scale;
}

/// Get the residual above which a coarse column is not reproduced by the fine basis.
///
/// What is graded is `max_i |A x - b|` over every column of the prolongation, where `b` is a coarse
/// function's coefficient vector in a common tensor-product level and `A`'s columns are the
/// candidate fine functions' vectors in the same basis. All the entries are refinement (Oslo)
/// coefficients, so the quantity is dimensionless and its natural size is
/// `max_coarse_val = ||b||_inf`, which for a genuine refinement is the partition-of-unity value 1.
/// The `1 +` is the floor that keeps the threshold positive for a coarse function whose column is
/// empty.
///
/// A least-squares solve by the SVD (LAPACK's `gelsd`) is normwise backward stable (Golub and Van
/// Loan, *Matrix Computations*, 4th ed., section 5.5; Higham, *Accuracy and Stability of Numerical
/// Algorithms*, 2nd ed., chapter 20): the computed `x_hat` is the exact least-squares solution of
/// `(A + dA) y ~= b + db` with `||dA|| <= c eps ||A||` and `||db|| <= c eps ||b||`.
///
/// The step from that to a residual bound is worth spelling out, because `x_hat` does not satisfy
/// the perturbed system exactly: a least-squares solution leaves its own residual `r_pert`. Writing
/// `x` for the exact solution of the unperturbed system, which for a genuine refinement is
/// consistent (`A x = b`), and using `x` as a trial vector for the perturbed minimisation:
///
///     ||r_pert|| <= ||(A + dA) x - (b + db)|| = ||dA x - db|| <= ||dA|| ||x|| + ||db||,
///     ||A x_hat - b|| <= ||r_pert|| + ||dA|| ||x_hat|| + ||db||
///                     <= 2 (||dA|| max(||x||, ||x_hat||) + ||db||),
///
/// which is `O(c eps (||A|| ||x|| + ||b||))`. The factor of two is that argument's, not a safety
/// pad. Every factor on the right is of order one here: the entries of `A`, `x` and `b` are all
/// refinement coefficients in `[0, 1]`. The residual is therefore a small multiple of `eps`, and
/// `c` is the part no closed form is available for. The code grades `max_i |A x - b|`, an infinity
/// norm, which is at most the two-norm the bound is stated in, so the bound covers it.
///
/// The tier is `tolerance::get_conservative<double>()`, `4096 eps`, whose stated meaning is a long
/// accumulation, which an SVD-based least-squares solve is. It stands in for the unknown `c`: an
/// admitted heuristic factor, not a derived one.
///
/// A `sqrt(M)` term with `M` the number of rows was considered, since `||b||_2` carries one, and
/// rejected on measurement of the Python implementation's residuals, which did not grow with `M`;
/// carrying it would loosen the gate for the largest systems for a growth that is not there. If
/// one ever appears, that term is where it belongs.
///
/// Args:
///     max_coarse_val (double): Largest absolute coefficient over every coarse column,
///         `||b||_inf`.
///
/// Returns:
///     double: The residual threshold, in the coefficients' own dimensionless units.
[[nodiscard]] inline double prolongation_residual_tolerance(double max_coarse_val) noexcept {
  return tolerance::get_conservative<double>() * (1.0 + max_coarse_val);
}

}  // namespace pantr::thb
