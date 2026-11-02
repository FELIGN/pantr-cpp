/// @file locate.hpp
/// @brief Physical-to-parametric point inversion of a `Bspline` (`locate`).
///
/// Given physical points, recover the parametric coordinates whose image they are, together with
/// the knot-span cell that contains them. Two stages:
///
/// 1. **Candidate cells.** The image of a knot-span cell lies inside the axis-aligned box of the
///    control points supporting that cell: the convex-hull property of the B-spline basis, which
///    for a NURBS holds for the *projected* control points provided every weight is positive. Those
///    per-cell boxes are indexed by a `grid::BVH`, so the cells that can contain a query point are
///    found by a tree descent instead of by trying Newton on every cell.
/// 2. **Newton.** Per candidate cell, in ascending cell id, a damped Newton's method on
///    `F(xi) - x = 0` starting from the cell's parametric midpoint and, if that does not converge,
///    from the cell corner whose image is nearest the query. Iterates are clamped to the parametric
///    domain box, so an iterate may migrate out of its candidate cell: the candidate box is a
///    superset test, never a constraint on the solution.
///
///    The damping is a backtracking line search over a step first capped to one parametric domain
///    extent (`detail::locate_kernels::max_step_extents`), and it is what makes the iteration
///    usable away from the basin of attraction: an undamped Newton step is only guaranteed to
///    reduce the residual near a root, and out there it can be longer than the domain the answer
///    has to lie in. Accepting such a step and clamping the result can be *periodic* (the clamp
///    maps two off-basin iterates onto each other), and then no budget and no tolerance recovers
///    the point. Requiring each accepted step to reduce the residual removes both failure modes,
///    and makes the residuals of the accepted iterates a non-increasing sequence.
///
///    Stopping and acceptance are two thresholds (`LocateThresholds`): the iteration runs until the
///    residual is below what the *image* alone entitles the caller to, and a result counts as found
///    once it is below what the *parametrization* can attain.
///
///    What damping cannot do is change which root the iteration finds. On a mapping that folds, the
///    residual has minima on the domain boundary that are not roots; a start that descends into one
///    is stuck there legitimately. That is what the second start is for, and why it is a *different
///    point of the same cell* rather than a longer iteration.
///
/// What is guaranteed is `F(ref_coords[i]) == points[i]` within the acceptance threshold, and *not*
/// that `ref_coords[i]` is any particular preimage: a folded mapping sends several parametric
/// points to the same physical point, and which one comes back depends on the candidate order and
/// the Newton path. A point on an interior knot line belongs to either adjacent cell; the reported
/// cell is the one `grid::TensorProductGrid::locate` assigns to the returned coordinates (the lower
/// cell on an interior breakpoint).
///
/// **Precision contract.** The inversion always runs in `double`, promoting a `float` spline to an
/// exact `double` copy first (casting `float` knots and coefficients to `double` is exact, so the
/// promoted spline is the same mapping). The reason is margin: the residual of a `float` evaluation
/// cannot be resolved much below `get_default<float>()`, while the `double` arithmetic floor is
/// eight orders below it. The default thresholds are nevertheless taken from the caller's storage
/// type (`get_default<T>()`), so a `float` caller's answer inverts the promoted map to its own
/// declared precision, not to `double` accuracy; evaluating it on the original `float` spline
/// reproduces the query point to `float` accuracy only.
///
/// Every point is solved independently of every other, so the batch runs over the thread pool and
/// no result depends on the number of threads.
///
/// Only square maps are inverted (`rank == dim`): planar and volumetric geometry maps. An embedded
/// curve or surface (`rank > dim`) needs a Gauss-Newton closest-point solve, not implemented.
#pragma once

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/evaluate.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/grid/bvh.hpp"
#include "pantr/grid/tensor_product_grid.hpp"
#include "pantr/parallel/parallel_for.hpp"
#include "pantr/parallel/work_model.hpp"
#include "pantr/tolerance.hpp"

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pantr::bspline {

/// The two residual thresholds a solve is governed by, with `stop <= accept`.
///
/// One number cannot do both jobs. `stop` is the **accuracy contract**, what a caller is entitled
/// to expect of a coordinate that comes back; `accept` is an **attainability** statement, the
/// residual below which no representable parameter can go, so that a query whose exact preimage is
/// unrepresentable is still reported found. Serving both from one number forces the accuracy
/// contract up to the attainability bound: the achieved residual tracks whatever threshold the
/// iteration stops at. So the Newton loop iterates while the residual is above `stop` and takes the
/// verdict at `accept`.
///
/// **Why only `accept` grows with the parametrization.** The default `stop` is
/// `get_default<T>() * physical`, read off the image alone (`LocateContext::physical`); `accept` is
/// the same tier times that length maximised against the parametric length
/// (`LocateContext::scale`). How finely a `double` parametric coordinate resolves the geometry
/// decides whether a solution *exists* among representable parameters; it says nothing about the
/// accuracy a caller is owed, which is a property of the image alone. For a knot vector spanning
/// `[0, L]` the two are equal bit for bit (the reach is exactly `1.0` and no net span exceeds the
/// diagonal).
///
/// **Coverage cannot shrink, by construction.** Every accepted step strictly reduces the residual
/// (`detail::locate_kernels::armijo_decrease`), so lowering the stopping test from `accept` to
/// `stop` only extends trajectories the single-threshold iteration would have cut short, and a row
/// whose residual once passed `accept` still passes it at exit. The verdict is a test of the
/// residual actually achieved, taken once at the end.
///
/// **The ordering is constructed, not enforced.** Both members are the same positive tier times a
/// length, the acceptance length is a maximum over the stopping one, and round-to-nearest
/// multiplication is monotone, so `stop <= accept` cannot round the wrong way. Were `accept < stop`
/// ever built, the loop would end against the larger number and the verdict be taken against the
/// smaller: coverage would collapse (points reported not found), but no coordinate above `accept`
/// could be returned, so the failure is one-sided and no guard is added. An explicit tolerance sets
/// both members to itself.
///
/// Attributes:
///     stop (double): The residual at or below which the iteration stops, in physical units.
///     accept (double): The residual at or below which a result counts as found; never below
///     `stop`.
struct LocateThresholds {
  double stop;
  double accept;
};

namespace detail::locate_kernels {

/// Relative threshold at which a Newton Jacobian counts as rank-deficient.
///
/// A candidate is abandoned when `sigma_min <= dim * jacobian_rank_rel_tol * sigma_max`, the rule
/// and the magnitude `numpy.linalg.matrix_rank` uses by default (`max(M, N) * eps * sigma_max`):
/// below it the null direction is indistinguishable from rounding noise and the solve returns an
/// arbitrary step. It is deliberately permissive (it only rejects a condition number reaching
/// `1 / (dim * eps)`), because a merely ill-conditioned Jacobian still gives a usable step and
/// `max_step_extents` bounds an overlong one.
///
/// The test is a ratio, so it is invariant under scaling the geometry and under scaling the
/// parametrization **isotropically**. It is *not* invariant under scaling the parametric directions
/// independently (a different knot-vector span per direction right-multiplies the Jacobian by a
/// diagonal matrix): `J = diag(1, 100 eps)` passes against `2 eps`, while under
/// `T = diag(1, eps / 100)` the same map has ratio `eps^2` and is rejected. This bites only where
/// the Jacobian already sits near the guard.
inline constexpr double jacobian_rank_rel_tol = std::numeric_limits<double>::epsilon();

/// Fraction of its own size the residual norm must lose per unit step length, to be accepted.
///
/// A damped step `lam * delta` passes when
/// `||F(xi - lam delta) - x|| <= (1 - armijo_decrease * lam) ||F(xi) - x||`: the Armijo
/// sufficient-decrease rule on the merit function `||F - x||`, whose directional derivative along
/// the exact Newton step at `lam = 0` is `-||r||`, so `lam ||r||` is what the linear model promises
/// and this constant is the fraction of it that must materialize.
///
/// **When a step is accepted at all.** With `dtilde` the step with every coordinate the clip holds
/// fixed zeroed, and `Jv` the Jacobian of the polynomial piece the segment `xi - t dtilde` enters,
/// some small enough `lam` passes iff `<r, Jv dtilde> > armijo_decrease ||r||^2`. An iterate in the
/// open interior of a cell with no coordinate held always passes (`Jv == J`, the left side is
/// `||r||^2`): this needs only differentiability at that point and a nonsingular Jacobian. An
/// iterate on a knot line where the map is only C0 (a simple interior knot at degree 1, or a knot
/// of multiplicity at least the degree) can fail at every `lam`: the span search is right-closed,
/// so `J` is the right-hand derivative, and a kink across which the directional derivative shrinks
/// by more than `1 / armijo_decrease` has every step rejected while the residual still decreases
/// (observed at degree 1 with slopes `2e-5` and `1.99998`, and at degree 3 across a triple knot).
/// It is reachable, since corner starts are knot values, and it is a limitation of a monotone line
/// search on a C0 join that no budget fixes.
///
/// **Why this magnitude.** It must be small enough never to reject the asymptotic step (near a root
/// `||r_next|| ~ K ||r||^2`, which beats `(1 - armijo_decrease) ||r||` by orders of magnitude, so
/// `lam == 1` is accepted throughout the endgame) and far above the rounding of the comparison
/// (twelve decades above `eps` at `lam == 1`, nine at the shortest step `max_step_halvings`
/// admits). `1e-4` is the established value for exactly this rule, sufficient decrease tested on
/// `||F||` itself for Newton on `F(x) = 0` (C. T. Kelley, *Iterative Methods for Linear and
/// Nonlinear Equations*, SIAM, 1995, Eq. (8.1) and Algorithm 8.2.1). Both bounds hold with decades
/// to spare, so the choice is insensitive.
inline constexpr double armijo_decrease = 1.0e-4;

/// Longest first trial step of the line search, in parametric domain extents per direction.
///
/// The search starts at `max_step_extents / max(reach, max_step_extents)` of the Newton step, with
/// `reach = max_k |delta_k| / (hi_k - lo_k)`: a trust region. The step `J^-1 r` grows like
/// `1 / sigma_min(J)`, so a merely ill-conditioned Jacobian gives a step orders of magnitude longer
/// than the domain, and the halvings needed to walk it back grow like `log2(||delta|| / L)`, which
/// is unbounded over legal inputs. A step longer than the domain cannot be a step toward a solution
/// inside it (the clip truncates it anyway), so capping at one extent removes exactly that term and
/// leaves the halvings the curvature demands. The cap is inactive near the root, so the asymptotic
/// rate is untouched. Normalizing per direction by its own extent keeps the cap invariant under
/// scaling the parametric directions independently.
inline constexpr double max_step_extents = 1.0;

/// Number of times a Newton step may be halved before its candidate cell is abandoned.
///
/// The shortest step tried is `2^-max_step_halvings` of the first, which `max_step_extents` bounds;
/// with that cap what remains is set by the map's curvature. Observed requirement on the warped-map
/// sweep of the Python suite (dimensions 1 to 3, degrees 1 to 5, folding maps) and on near-critical
/// configurations: every query converges at a budget of 2; this budget is four doublings beyond.
/// What it does *not* cover, stated plainly: a single solve can need more (up to 15 halvings were
/// observed on the folding family), and is then abandoned; such queries were decided by another
/// candidate or the second start, and a query all of whose candidates and both starts need more
/// than 8 would be lost. Cost is what keeps it from being larger: a candidate that will never
/// converge spends its whole budget, and those dominate queries off the image. An admitted
/// heuristic, not a derived bound. Exhausting it abandons that candidate, not the query.
inline constexpr int max_step_halvings = 8;

/// Clip `v` into `[lo, hi]` as `numpy.clip` does (a NaN stays NaN).
[[nodiscard]] inline double clip(double v, double lo, double hi) noexcept {
  const double up = v < lo ? lo : v;
  return up > hi ? hi : up;
}

/// The Euclidean norm of `n` values, as a plain sum of squares in order.
[[nodiscard]] inline double norm2(const double* v, std::size_t n) noexcept {
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double sq = v[i] * v[i];
    s = s + sq;
  }
  return std::sqrt(s);
}

/// Evaluates a `double` B-spline and its Jacobian at one parametric point (one per thread).
///
/// The basis rows come from each direction's own `tabulate_basis` / `tabulate_basis_derivatives`
/// and are combined and contracted as `evaluate` does for scattered points (`tensor_row`, then
/// `contract_local`); a rational value is divided by its weight and a rational derivative takes the
/// first-order quotient rule `(H_k - H_k[w] F) / w`, as `evaluate_derivatives` does.
///
/// Attributes:
///     b_ (const Bspline<double>*): The spline (outlives the evaluator).
///     L_ (Layout): Its control-point addressing.
///     rank_ (std::size_t): Output rank.
///     values_, derivs_ (std::vector<std::vector<double>>): Per-direction scratch rows.
///     first_ (std::vector<std::int64_t>): First function per direction.
///     offs_, product_, weights_ (std::vector<...>): Contraction scratch, `nloc` each.
///     rows_ (std::vector<const double*>): Rows handed to `tensor_row`.
///     h0_, hk_ (std::vector<double>): Homogeneous values, `cp_size` each.
class MapEvaluator {
 public:
  explicit MapEvaluator(const Bspline<double>& b)
      : b_(&b),
        L_(make_layout(b)),
        rank_(b.rank()),
        first_(L_.dim),
        offs_(L_.nloc),
        product_(L_.nloc),
        weights_(L_.nloc),
        rows_(L_.dim),
        h0_(L_.cp_size),
        hk_(L_.cp_size) {
    values_.resize(L_.dim);
    derivs_.resize(L_.dim);
    for (std::size_t d = 0; d < L_.dim; ++d) {
      values_[d].resize(L_.order[d]);
      derivs_[d].resize(2 * L_.order[d]);
    }
  }

  /// Evaluate `F(xi)` into `out` (`rank` values). `xi` must lie in the domain (unchecked).
  void value(const double* xi, double* out) {
    const auto& spaces = b_->rep()->space.spaces();
    for (std::size_t d = 0; d < L_.dim; ++d) {
      spaces[d].template tabulate_basis<double>(
          std::span<const double>(xi + d, 1), mdview<double, 2>(values_[d].data(), 1, L_.order[d]),
          std::span<std::int64_t>(&first_[d], 1), false);
      rows_[d] = values_[d].data();
    }
    contract(h0_.data());
    if (b_->rational()) {
      project_in_place(h0_.data(), rank_);
    }
    std::copy(h0_.begin(), h0_.begin() + static_cast<std::ptrdiff_t>(rank_), out);
  }

  /// Evaluate the Jacobian `dF / dxi` into `jac` (`rank x dim`, row-major).
  ///
  /// The first-derivative rows of the basis recurrence divide by knot differences, and one is
  /// zero at the domain's upper end when the span search clamps to an empty last span: knots
  /// `[0, 0, 0, 1, 2, 2, 2, 3]` at degree 2 put a knot of full multiplicity at the end `2` of the
  /// domain, before a trailing knot. The iteration reaches that end by the clip, so the case is
  /// reachable from `locate`; there the derivative is undefined (the previous implementation's
  /// was NaN).
  ///
  /// Returns:
  ///     bool: False when a direction's recurrence met a zero divisor; `jac` is then incomplete
  ///     and the caller abandons the candidate, as for a Jacobian that is not finite.
  [[nodiscard]] bool jacobian(const double* xi, double* jac) {
    const auto& spaces = b_->rep()->space.spaces();
    for (std::size_t d = 0; d < L_.dim; ++d) {
      const bool complete = spaces[d].template tabulate_basis_derivatives<double>(
          std::span<const double>(xi + d, 1), 1,
          mdview<double, 3>(derivs_[d].data(), 1, 2, L_.order[d]),
          std::span<std::int64_t>(&first_[d], 1), false);
      if (!complete) {
        return false;
      }
    }
    const std::size_t dim = L_.dim;
    const bool rational = b_->rational();
    if (rational) {
      for (std::size_t d = 0; d < dim; ++d) {
        rows_[d] = derivs_[d].data();
      }
      contract(h0_.data());
      project_in_place(h0_.data(), rank_);
    }
    for (std::size_t k = 0; k < dim; ++k) {
      for (std::size_t d = 0; d < dim; ++d) {
        rows_[d] = derivs_[d].data() + (d == k ? L_.order[d] : 0);
      }
      contract(hk_.data());
      for (std::size_t c = 0; c < rank_; ++c) {
        double v = hk_[c];
        if (rational) {
          const double cw = hk_[rank_];
          const double t = cw * h0_[c];
          v = (v - t) / h0_[rank_];
        }
        jac[c * dim + k] = v;
      }
    }
    return true;
  }

 private:
  void contract(double* out) {
    tensor_row<double, double>(rows_, L_.order, product_.data(), weights_.data());
    local_offsets(L_, first_.data(), offs_.data());
    contract_local<double>(b_->control_points().data(), offs_.data(), weights_.data(), L_.nloc,
                           L_.cp_size, out);
  }

  const Bspline<double>* b_;
  Layout L_;
  std::size_t rank_;
  std::vector<std::vector<double>> values_;
  std::vector<std::vector<double>> derivs_;
  std::vector<std::int64_t> first_;
  std::vector<std::size_t> offs_;
  std::vector<double> product_;
  std::vector<double> weights_;
  std::vector<const double*> rows_;
  std::vector<double> h0_;
  std::vector<double> hk_;
};

/// The per-thread state of a solve: the evaluator and the scratch of one Newton iteration.
///
/// Attributes:
///     ev (MapEvaluator): The map and Jacobian evaluator.
///     lo, hi (std::vector<double>): The parametric domain box, `dim` each.
///     residual, trial_xi, trial_residual, corner, best_corner (std::vector<double>): Scratch.
///     jac (Eigen::MatrixXd): The Jacobian, `dim x dim`.
///     svd (Eigen::JacobiSVD<Eigen::MatrixXd>): Singular values of `jac` (no vectors).
///     lu (Eigen::PartialPivLU<Eigen::MatrixXd>): The Newton solve, partial pivoting as LAPACK's
///         `gesv` (what `numpy.linalg.solve` calls).
///     rhs, delta (Eigen::VectorXd): The residual and the Newton step.
struct Workspace {
  MapEvaluator ev;
  std::vector<double> lo;
  std::vector<double> hi;
  std::vector<double> residual;
  std::vector<double> trial_xi;
  std::vector<double> trial_residual;
  std::vector<double> corner;
  std::vector<double> best_corner;
  std::vector<double> jac_rows;
  Eigen::MatrixXd jac;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd;
  Eigen::PartialPivLU<Eigen::MatrixXd> lu;
  Eigen::VectorXd rhs;
  Eigen::VectorXd delta;

  Workspace(const Bspline<double>& b, std::span<const double> dom_lo,
            std::span<const double> dom_hi)
      : ev(b),
        lo(dom_lo.begin(), dom_lo.end()),
        hi(dom_hi.begin(), dom_hi.end()),
        residual(b.rank()),
        trial_xi(b.dim()),
        trial_residual(b.rank()),
        corner(b.dim()),
        best_corner(b.dim()),
        jac_rows(b.rank() * b.dim()),
        jac(static_cast<Eigen::Index>(b.rank()), static_cast<Eigen::Index>(b.dim())),
        svd(static_cast<Eigen::Index>(b.rank()), static_cast<Eigen::Index>(b.dim())),
        lu(static_cast<Eigen::Index>(b.dim())),
        rhs(static_cast<Eigen::Index>(b.rank())),
        delta(static_cast<Eigen::Index>(b.dim())) {}
};

/// Take the longest damped Newton step that reduces the residual, at one point.
///
/// A backtracking line search on `||F(xi) - x||`: the first trial is shortened to at most
/// `max_step_extents` domain extents per direction, then halved until the Armijo test passes, up to
/// `max_step_halvings` halvings. The trial iterate is clipped to the domain box, so the search runs
/// along the projected path `lam -> clip(xi - lam delta)`, which still tends to `xi` as `lam` does
/// (`clip` is 1-Lipschitz and fixes `xi`). What is rejected at every `lam` is a constant projected
/// path, not merely a direction that leaves the box.
///
/// A false return means this candidate made no progress, not that the point is a minimum of the
/// residual: in two or more dimensions `sign(delta[k])` need not agree with the sign of the merit
/// gradient, so a corner can be abandoned while the residual still decreases into the box.
///
/// Args:
///     ws (Workspace&): Scratch; `ws.delta` holds the undamped step `J^-1 r`.
///     x (const double*): The query point, `rank` values.
///     xi (double*): The iterate, `dim` values; updated when a step is accepted.
///     r (double*): `F(xi) - x`, `rank` values; updated with `xi`.
///     rn (double&): `||r||`; updated with `xi`.
///
/// Returns:
///     bool: Whether a step was accepted.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
[[nodiscard]] inline bool accept_damped_step(Workspace& ws, const double* x, double* xi, double* r,
                                             double& rn) {
  const std::size_t dim = ws.lo.size();
  const std::size_t rank = ws.residual.size();
  // `numpy.max` propagates a NaN: a NaN ratio makes the first step NaN, and every trial fails.
  double reach = -std::numeric_limits<double>::infinity();
  for (std::size_t k = 0; k < dim; ++k) {
    const double ratio = std::abs(ws.delta(static_cast<Eigen::Index>(k))) / (ws.hi[k] - ws.lo[k]);
    reach = (std::isnan(ratio) || std::isnan(reach)) ? std::numeric_limits<double>::quiet_NaN()
                                                     : (std::max)(reach, ratio);
  }
  const double capped = std::isnan(reach) ? reach : (std::max)(reach, max_step_extents);
  const double first = max_step_extents / capped;
  double shrink = 1.0;
  for (int h = 0; h <= max_step_halvings; ++h) {
    const double step = shrink * first;
    for (std::size_t k = 0; k < dim; ++k) {
      const double move = step * ws.delta(static_cast<Eigen::Index>(k));
      ws.trial_xi[k] = clip(xi[k] - move, ws.lo[k], ws.hi[k]);
    }
    ws.ev.value(ws.trial_xi.data(), ws.trial_residual.data());
    for (std::size_t c = 0; c < rank; ++c) {
      ws.trial_residual[c] = ws.trial_residual[c] - x[c];
    }
    const double trial_norm = norm2(ws.trial_residual.data(), rank);
    const double bound = (1.0 - armijo_decrease * step) * rn;
    if (trial_norm <= bound) {
      std::copy(ws.trial_xi.begin(), ws.trial_xi.end(), xi);
      std::copy(ws.trial_residual.begin(), ws.trial_residual.end(), r);
      rn = trial_norm;
      return true;
    }
    shrink *= 0.5;
  }
  return false;
}

/// Run the damped Newton inversion of one point from one starting guess.
///
/// Only `th.stop` ends a trajectory (a starting guess already inside it is returned untouched); a
/// point is also dropped when its Jacobian is rank-deficient (`jacobian_rank_rel_tol`) or its line
/// search is exhausted. The verdict is one test of the residual actually achieved against
/// `th.accept`, taken at the end, so it covers every way the loop ends, `max_iter` included.
///
/// Every accepted step reduces the residual, so the residuals are non-increasing; they strictly
/// decrease for every residual in the normal range (the bound `(1 - armijo_decrease lam) ||r||`
/// is below `||r||` by nine decades more than its rounding at the shortest step), so a point that
/// is not converging exhausts its line search instead of cycling. The one gap is a subnormal
/// residual, 300 decades below any threshold but a subnormal explicit tolerance.
///
/// Args:
///     ws (Workspace&): Scratch for this thread.
///     x (const double*): The query point, `rank` values.
///     xi (double*): In: the starting guess (`dim` values); out: the last accepted iterate.
///     th (LocateThresholds): The thresholds.
///     max_iter (std::int64_t): Maximum number of accepted Newton steps; line-search evaluations
///         are not steps.
///
/// Returns:
///     bool: `||F(xi) - x|| <= th.accept` at exit.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
[[nodiscard]] inline bool newton_refine(Workspace& ws, const double* x, double* xi,
                                        LocateThresholds th, std::int64_t max_iter) {
  const std::size_t dim = ws.lo.size();
  const std::size_t rank = ws.residual.size();
  for (std::size_t k = 0; k < dim; ++k) {
    xi[k] = clip(xi[k], ws.lo[k], ws.hi[k]);
  }
  double* r = ws.residual.data();
  ws.ev.value(xi, r);
  for (std::size_t c = 0; c < rank; ++c) {
    r[c] = r[c] - x[c];
  }
  double rn = norm2(r, rank);
  // Tested before the first step, so a guess already inside the stopping threshold is returned
  // untouched; a NaN residual is not above it and ends the solve here.
  bool active = rn > th.stop;
  const double guard = static_cast<double>(dim) * jacobian_rank_rel_tol;
  for (std::int64_t it = 0; it < max_iter && active; ++it) {
    // An undefined (zero-divisor) or non-finite Jacobian abandons the candidate, as a
    // rank-deficient one does; the previous implementation raised numpy's "SVD did not
    // converge" on the NaN Jacobian it formed there.
    if (!ws.ev.jacobian(xi, ws.jac_rows.data())) {
      break;
    }
    for (std::size_t c = 0; c < rank; ++c) {
      for (std::size_t k = 0; k < dim; ++k) {
        ws.jac(static_cast<Eigen::Index>(c), static_cast<Eigen::Index>(k)) =
            ws.jac_rows[c * dim + k];
      }
      ws.rhs(static_cast<Eigen::Index>(c)) = r[c];
    }
    ws.svd.compute(ws.jac);
    // A non-finite entry (an overflowed Jacobian, reachable with coordinates near 1e154 and an
    // explicit tolerance) leaves the singular values unset: abandon the candidate, as a
    // rank-deficient one is.
    if (ws.svd.info() != Eigen::Success) {
      break;
    }
    const auto& sv = ws.svd.singularValues();
    const double smax = sv(0);
    const double smin = sv(sv.size() - 1);
    if (!(smin > guard * smax)) {
      break;
    }
    ws.lu.compute(ws.jac);
    ws.delta = ws.lu.solve(ws.rhs);
    if (!accept_damped_step(ws, x, xi, r, rn)) {
      break;
    }
    active = rn > th.stop;
  }
  return rn <= th.accept;
}

/// Find the corner of a cell whose image is nearest the query point (the second start).
///
/// A monotone iteration reaches only the root whose basin it starts in, and a cell midpoint is one
/// guess out of many: on a mapping that folds, a midpoint start can descend into a boundary minimum
/// of the residual and never leave it, while another point of the same cell descends into the root.
/// Among the `2^dim` corners, the one with the nearest image starts at the smallest residual, the
/// only ordering the merit function gives. Corner 0 (all lower bounds) is the fallback, so a map
/// whose evaluation is not finite (every distance NaN, every comparison false) still gets a corner.
///
/// Args:
///     ws (Workspace&): Scratch.
///     x (const double*): The query point.
///     cell_lo, cell_hi (const double*): The cell's parametric box, `dim` values each.
///     out (double*): The chosen corner, `dim` values.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
inline void nearest_corner_start(Workspace& ws, const double* x, const double* cell_lo,
                                 const double* cell_hi, double* out) {
  const std::size_t dim = ws.lo.size();
  const std::size_t rank = ws.residual.size();
  std::copy(cell_lo, cell_lo + dim, out);
  double best = std::numeric_limits<double>::infinity();
  const std::size_t n_corners = std::size_t{1} << dim;
  for (std::size_t corner = 0; corner < n_corners; ++corner) {
    for (std::size_t k = 0; k < dim; ++k) {
      ws.corner[k] = ((corner >> k) & 1U) != 0 ? cell_hi[k] : cell_lo[k];
    }
    ws.ev.value(ws.corner.data(), ws.trial_residual.data());
    for (std::size_t c = 0; c < rank; ++c) {
      ws.trial_residual[c] = ws.trial_residual[c] - x[c];
    }
    const double distance = norm2(ws.trial_residual.data(), rank);
    if (distance < best) {
      std::copy(ws.corner.begin(), ws.corner.end(), out);
      best = distance;
    }
  }
}

/// The physical (projected) control points of a `double` spline, row-major over
/// `(num_basis..., rank)`.
[[nodiscard]] inline std::vector<double> physical_control_points(const Bspline<double>& b) {
  const std::span<const double> cp = b.control_points();
  const std::size_t rank = b.rank();
  const std::size_t cps = b.cp_size();
  const std::size_t n = cp.size() / cps;
  std::vector<double> out(n * rank);
  for (std::size_t i = 0; i < n; ++i) {
    const double* row = cp.data() + i * cps;
    for (std::size_t c = 0; c < rank; ++c) {
      out[i * rank + c] = b.rational() ? row[c] / row[rank] : row[c];
    }
  }
  return out;
}

/// The physical box of every knot-span cell's supporting control points.
///
/// Reduces one parametric axis at a time, a running min/max over the `degree + 1` functions of each
/// interval, so the work is `O(num_cells * sum(degree + 1))` and no per-cell support table is
/// formed (min and max being exact, the result is that of the full support).
///
/// Args:
///     b (const Bspline<double>&): A non-periodic spline.
///
/// Returns:
///     std::pair<std::vector<double>, std::vector<double>>: `(lo, hi)`, each `(num_cells, rank)`
///     row-major, cells flat in C order over the intervals per direction.
[[nodiscard]] inline std::pair<std::vector<double>, std::vector<double>> cell_physical_bounds(
    const Bspline<double>& b) {
  const std::size_t dim = b.dim();
  const std::size_t rank = b.rank();
  std::vector<double> lo = physical_control_points(b);
  std::vector<double> hi = lo;
  std::vector<std::size_t> extents(b.shape().begin(), b.shape().end() - 1);
  for (std::size_t axis = 0; axis < dim; ++axis) {
    const auto& sub = b.rep()->space.spaces()[axis];
    const std::span<const std::int64_t> first = sub.first_basis_per_interval();
    const auto order = static_cast<std::size_t>(sub.degree()) + 1;
    std::size_t outer = 1;
    for (std::size_t d = 0; d < axis; ++d) {
      outer *= extents[d];
    }
    std::size_t inner = rank;
    for (std::size_t d = axis + 1; d < dim; ++d) {
      inner *= extents[d];
    }
    const std::size_t n_in = extents[axis];
    const std::size_t n_out = first.size();
    std::vector<double> new_lo(outer * n_out * inner);
    std::vector<double> new_hi(outer * n_out * inner);
    for (std::size_t o = 0; o < outer; ++o) {
      for (std::size_t i = 0; i < n_out; ++i) {
        const auto f = static_cast<std::size_t>(first[i]);
        double* dl = new_lo.data() + (o * n_out + i) * inner;
        double* dh = new_hi.data() + (o * n_out + i) * inner;
        const double* sl = lo.data() + (o * n_in + f) * inner;
        const double* sh = hi.data() + (o * n_in + f) * inner;
        std::copy(sl, sl + inner, dl);
        std::copy(sh, sh + inner, dh);
        for (std::size_t a = 1; a < order; ++a) {
          const double* tl = sl + a * inner;
          const double* th = sh + a * inner;
          for (std::size_t j = 0; j < inner; ++j) {
            dl[j] = (std::min)(dl[j], tl[j]);
            dh[j] = (std::max)(dh[j], th[j]);
          }
        }
      }
    }
    lo = std::move(new_lo);
    hi = std::move(new_hi);
    extents[axis] = n_out;
  }
  return {std::move(lo), std::move(hi)};
}

/// Per direction, how far the parametric box sits from the origin in its own widths.
///
/// `max(|lo_k|, |hi_k|) / (hi_k - lo_k)`: the factor by which the representable parameters around
/// the box are coarser, relative to the extent they resolve, than on a box touching the origin. One
/// for `[0, L]`, one half for `[-a, a]`, growing linearly once the box is further from the origin
/// than it is wide. Kept per direction because it is only meaningful paired with that direction's
/// own physical span (`parametric_scale`). A direction with no extent contributes zero rather than
/// dividing by zero: it carries no cell, and an infinite scale would report every query found.
///
/// Args:
///     lo, hi (std::span<const double>): The parametric box, `dim` values each.
///
/// Returns:
///     std::vector<double>: `dim` values, non-negative and finite.
[[nodiscard]] inline std::vector<double> parametric_reach(std::span<const double> lo,
                                                          std::span<const double> hi) {
  std::vector<double> out(lo.size(), 0.0);
  for (std::size_t k = 0; k < lo.size(); ++k) {
    const double extent = hi[k] - lo[k];
    if (extent > 0.0) {
      out[k] = (std::max)(std::abs(lo[k]), std::abs(hi[k])) / extent;
    }
  }
  return out;
}

/// Per direction, the physical length one line of the control net along it sweeps.
///
/// For direction `k`, the diagonal of the bounding box of one net line along `k`, maximised over
/// the transverse indices. It stands for `L_k ||J e_k||`, exactly for an affine map (an open knot
/// vector's Greville abscissae span the extent, and linear precision reproduces the map on them).
/// The box rather than the chord, because the chord vanishes on a direction that doubles back,
/// which would delete the parametric term exactly where the map is hardest.
///
/// Args:
///     b (const Bspline<double>&): The spline (rational ones read through projected points).
///
/// Returns:
///     std::vector<double>: `dim` values, non-negative.
[[nodiscard]] inline std::vector<double> net_span_per_direction(const Bspline<double>& b) {
  const std::size_t dim = b.dim();
  const std::size_t rank = b.rank();
  const std::vector<double> cp = physical_control_points(b);
  const std::vector<std::size_t> extents(b.shape().begin(), b.shape().end() - 1);
  std::vector<double> spans(dim, 0.0);
  std::vector<double> ext_lo(rank);
  std::vector<double> ext_hi(rank);
  std::vector<double> diff(rank);
  for (std::size_t axis = 0; axis < dim; ++axis) {
    std::size_t outer = 1;
    for (std::size_t d = 0; d < axis; ++d) {
      outer *= extents[d];
    }
    std::size_t inner = 1;
    for (std::size_t d = axis + 1; d < dim; ++d) {
      inner *= extents[d];
    }
    const std::size_t n = extents[axis];
    double best = -std::numeric_limits<double>::infinity();
    for (std::size_t o = 0; o < outer; ++o) {
      for (std::size_t in = 0; in < inner; ++in) {
        for (std::size_t c = 0; c < rank; ++c) {
          ext_lo[c] = cp[((o * n) * inner + in) * rank + c];
          ext_hi[c] = ext_lo[c];
        }
        for (std::size_t i = 1; i < n; ++i) {
          const double* row = cp.data() + ((o * n + i) * inner + in) * rank;
          for (std::size_t c = 0; c < rank; ++c) {
            ext_lo[c] = (std::min)(ext_lo[c], row[c]);
            ext_hi[c] = (std::max)(ext_hi[c], row[c]);
          }
        }
        for (std::size_t c = 0; c < rank; ++c) {
          diff[c] = ext_hi[c] - ext_lo[c];
        }
        best = (std::max)(best, norm2(diff.data(), rank));
      }
    }
    spans[axis] = best;
  }
  return spans;
}

/// The physical length the parametrization's own resolution puts a floor at.
///
/// `max_k [reach_k * net_span_k]` (`parametric_reach`, `net_span_per_direction`), formed per
/// direction and maximised afterwards, times `eps(double) / eps(T)`. One ulp of a parametric
/// coordinate is `eps |xi_k|`, which the map turns into `eps |xi_k| ||J e_k||`; splitting it at the
/// extent, `eps (|xi_k| / L_k) (L_k ||J e_k||)`, separates the reach from the span. Pairing the
/// factors per direction matters: maximising them separately pairs the worst-offset direction with
/// another direction's stretch and over-loosens the threshold by their ratio. The precision ratio
/// cancels the caller's format back out of the tier the threshold is multiplied by, since the
/// iterate is a `double` whatever `T` is (exactly `1.0` for `double`).
///
/// A map whose local stretch exceeds the average over its own direction has a floor above this
/// length, by `A = max_k L_k sup||J e_k|| / net_span_k`; the default tier absorbs `A` of order 100,
/// and `A` is unbounded over legal inputs, so the term is a mitigation, not a guarantee (the
/// physical diagonal has always stood in for stretch the same way).
///
/// Args:
///     b (const Bspline<double>&): The promoted spline.
///     lo, hi (std::span<const double>): Its parametric domain box.
///     precision_ratio (double): `eps(double) / eps(T)` of the caller's storage type `T`.
///
/// Returns:
///     double: A physical length, non-negative.
[[nodiscard]] inline double parametric_scale(const Bspline<double>& b, std::span<const double> lo,
                                             std::span<const double> hi, double precision_ratio) {
  const std::vector<double> reach = parametric_reach(lo, hi);
  const std::vector<double> spans = net_span_per_direction(b);
  double best = -std::numeric_limits<double>::infinity();
  for (std::size_t k = 0; k < reach.size(); ++k) {
    const double product = reach[k] * spans[k];
    best = (std::isnan(product) || std::isnan(best)) ? std::numeric_limits<double>::quiet_NaN()
                                                     : (std::max)(best, product);
  }
  return precision_ratio * best;
}

/// The diagonal and the largest coordinate magnitude of the boxes' union.
///
/// Attributes:
///     diagonal (double): `||box_hi - box_lo||`.
///     magnitude (double): `max(|box_lo|, |box_hi|)` over the coordinates.
struct BoxLengths {
  double diagonal;
  double magnitude;
};

/// Measure the union of the cell boxes (`n_cells x rank`, row-major; at least one cell).
[[nodiscard]] inline BoxLengths box_lengths(std::span<const double> lo, std::span<const double> hi,
                                            std::size_t rank) {
  const std::size_t n = lo.size() / rank;
  std::vector<double> box_lo(lo.begin(), lo.begin() + static_cast<std::ptrdiff_t>(rank));
  std::vector<double> box_hi(hi.begin(), hi.begin() + static_cast<std::ptrdiff_t>(rank));
  for (std::size_t i = 1; i < n; ++i) {
    for (std::size_t c = 0; c < rank; ++c) {
      box_lo[c] = (std::min)(box_lo[c], lo[i * rank + c]);
      box_hi[c] = (std::max)(box_hi[c], hi[i * rank + c]);
    }
  }
  std::vector<double> diff(rank);
  double magnitude = 0.0;
  for (std::size_t c = 0; c < rank; ++c) {
    diff[c] = box_hi[c] - box_lo[c];
    magnitude = (std::max)({magnitude, std::abs(box_lo[c]), std::abs(box_hi[c])});
  }
  return {norm2(diff.data(), rank), magnitude};
}

/// The length the *accuracy* a returned coordinate is entitled to is a multiple of.
///
/// The larger of the diagonal of the geometry's bounding box (what makes a residual threshold a
/// relative statement) and its largest coordinate magnitude (the computed residual `F(xi) - x`
/// cannot be resolved below about `eps max|coordinate|`, whatever the extent: a unit patch at
/// `x = 1e6` has a residual floor near `1e-10`). Both are properties of the **image** alone, so
/// reparametrizing a geometry does not change what a caller may expect. A geometry with no length
/// at all (every control point identical, at the origin) falls back to `1.0`: a fallback, not a
/// floor, so the threshold keeps shrinking with the model's units.
///
/// Args:
///     lengths (BoxLengths): From `box_lengths`.
///
/// Returns:
///     double: Strictly positive.
[[nodiscard]] inline double physical_scale(BoxLengths lengths) {
  const double scale = (std::max)(lengths.diagonal, lengths.magnitude);
  return scale > 0.0 ? scale : 1.0;
}

/// The length the default *acceptance* threshold is a multiple of: `max(physical, parametric)`.
///
/// The parametric term (`parametric_scale`) is invisible to the image-only lengths (translating
/// the knot vector changes it and neither of them), and without it a threshold built from the image
/// alone is unreachable once the knot vector sits far from the parametric origin. For a knot vector
/// spanning `[0, L]` the reach is exactly `1.0` and no net span exceeds the diagonal, so the
/// maximum is the physical length bit for bit. The default tier `get_default` (`64 eps`) is an
/// accuracy contract, not a margin: the achieved residual tracks whatever threshold is set.
[[nodiscard]] inline double geometric_scale(double physical, double parametric) {
  return (std::max)(physical, parametric);
}

/// Format `x` as Python's `f"{x:.3e}"` (for the refusal message).
[[nodiscard]] inline std::string sci3(double x) {
  if (std::isnan(x)) {
    return "nan";
  }
  if (std::isinf(x)) {
    return x > 0 ? "inf" : "-inf";
  }
  char buf[64];
  const int n = std::snprintf(buf, sizeof buf, "%.3e", x);
  return std::string(buf, static_cast<std::size_t>(n));
}

}  // namespace detail::locate_kernels

/// The result of `locate`.
///
/// Attributes:
///     cell_ids (NdArray<std::int64_t>): `(n,)`: the flat knot-span cell id (C order over the
///         intervals per direction) of each point, `-1` where it was not found.
///     ref_coords (NdArray<double>): `(n, dim)`: parametric coordinates, NaN where not found.
struct [[nodiscard]] LocateResult {
  NdArray<std::int64_t> cell_ids;
  NdArray<double> ref_coords;
};

namespace detail {

/// The shared representation of a `LocateContext`.
///
/// Attributes:
///     spline (Bspline<double>): The spline to invert, promoted to `double`.
///     bvh (grid::BVH): Over the per-cell physical control-point boxes.
///     grid (grid::TensorProductGrid): Over the in-domain breakpoints (cell ids and the tie rule).
///     breakpoints (std::vector<std::vector<double>>): Per direction.
///     num_intervals (std::vector<std::size_t>): Per direction.
///     lo, hi (std::vector<double>): The parametric domain box.
///     tier (double): `get_default<T>()` of the caller's storage type.
///     physical, parametric, diagonal, scale (double): See `LocateContext`.
struct LocateContextRep {
  Bspline<double> spline;
  grid::BVH bvh;
  grid::TensorProductGrid grid;
  std::vector<std::vector<double>> breakpoints;
  std::vector<std::size_t> num_intervals;
  std::vector<double> lo;
  std::vector<double> hi;
  double tier;
  double physical;
  double parametric;
  double diagonal;
  double scale;
};

}  // namespace detail

/// The per-spline state `locate` reuses across calls: an immutable value (copies share it).
///
/// Holds the spline promoted to `double`, the hierarchy over its per-cell physical boxes, the grid
/// that assigns cell ids, and the lengths the two default thresholds are expressed in. None of it
/// depends on the query points; the hierarchy costs `O(num_cells)` to build.
///
/// Attributes:
///     rep_ (std::shared_ptr<const detail::LocateContextRep>): The shared representation.
class LocateContext {
 public:
  /// Validate `spline` for inversion and build its context.
  ///
  /// Args:
  ///     spline (const Bspline<T>&): The spline to invert.
  ///
  /// Raises:
  ///     not_implemented: `rank > dim` (an embedded curve or surface).
  ///     value_error: `rank < dim`; a periodic direction; a rational spline with a non-positive
  ///         (or NaN) weight; a non-finite projected control point (from `grid::BVH`).
  template <StorageFloat T>
  explicit LocateContext(const Bspline<T>& spline) : rep_(build(spline)) {}

  /// Get the promoted spline. Returns: `Bspline<double>`.
  [[nodiscard]] const Bspline<double>& spline() const noexcept { return rep_->spline; }
  /// Get the hierarchy over the cell boxes. Returns: `const grid::BVH&`.
  [[nodiscard]] const grid::BVH& bvh() const noexcept { return rep_->bvh; }
  /// Get the grid of the in-domain breakpoints. Returns: `const grid::TensorProductGrid&`.
  [[nodiscard]] const grid::TensorProductGrid& grid() const noexcept { return rep_->grid; }
  /// Get the default tier, `get_default<T>()` of the caller's type. Returns: `double`.
  [[nodiscard]] double tier() const noexcept { return rep_->tier; }
  /// Get the image-only length the default stop is a multiple of. Returns: `double`, positive.
  [[nodiscard]] double physical() const noexcept { return rep_->physical; }
  /// Get the parametric length (`detail::locate_kernels::parametric_scale`). Returns: `double`.
  [[nodiscard]] double parametric() const noexcept { return rep_->parametric; }
  /// Get the bounding-box diagonal of the control points. Returns: `double`.
  [[nodiscard]] double diagonal() const noexcept { return rep_->diagonal; }
  /// Get the length the default accept is a multiple of, `max(physical, parametric)`.
  /// Returns: `double`.
  [[nodiscard]] double scale() const noexcept { return rep_->scale; }
  /// Get the parametric domain box's lower corner. Returns: `std::span<const double>`.
  [[nodiscard]] std::span<const double> domain_lo() const noexcept { return rep_->lo; }
  /// Get the parametric domain box's upper corner. Returns: `std::span<const double>`.
  [[nodiscard]] std::span<const double> domain_hi() const noexcept { return rep_->hi; }

  /// Get the default thresholds, or refuse a parametrization that can carry none.
  ///
  /// `tier * physical` to stop at and `tier * scale` to accept at. Refused when
  /// `tier * parametric > diagonal`: past it a "found" verdict admits more than a whole diameter of
  /// error, and clamping the scale back would restore an accuracy that cannot be attained. The line
  /// compares two lengths the geometry supplies, so no constant is invented; it fires above a reach
  /// of `1 / (64 eps)`, about `7e13`. Below it the threshold can still be a sizeable fraction of
  /// the geometry (a reach of `1e13` buys `0.14` of a diameter); that is stated in the Python
  /// `Bspline.locate` docstring rather than legislated here.
  ///
  /// Returns:
  ///     LocateThresholds: Both strictly positive, `stop <= accept`.
  ///
  /// Raises:
  ///     value_error: The parametric term alone would put the acceptance threshold above the
  ///         geometry's own bounding-box diagonal.
  [[nodiscard]] LocateThresholds default_thresholds() const {
    namespace lk = detail::locate_kernels;
    const double t = rep_->tier;
    if (t * rep_->parametric > rep_->diagonal) {
      throw value_error(
          "locate: this knot vector sits so far from the parametric origin that one ulp of a "
          "parametric coordinate moves the image by more than the geometry's own size (" +
          lk::sci3(t * rep_->parametric) + " against a bounding-box diagonal of " +
          lk::sci3(rep_->diagonal) +
          "), so no residual threshold can mean anything. Move the knot vector nearer the "
          "parametric origin, or pass an explicit tol to say what distance you want.");
    }
    return {t * rep_->physical, t * rep_->scale};
  }

  /// Get each cell's parametric box (consecutive breakpoints per direction).
  ///
  /// Args:
  ///     cell (std::int64_t): A flat cell id in `[0, num_cells)` (unchecked).
  ///     lo, hi (double*): Outputs, `dim` values each.
  void cell_box(std::int64_t cell, double* lo, double* hi) const noexcept {
    const std::size_t dim = rep_->num_intervals.size();
    auto rest = static_cast<std::size_t>(cell);
    for (std::size_t d = dim; d-- > 0;) {
      const std::size_t n = rep_->num_intervals[d];
      const std::size_t i = rest % n;
      rest /= n;
      lo[d] = rep_->breakpoints[d][i];
      hi[d] = rep_->breakpoints[d][i + 1];
    }
  }

  /// Get the representation, for an owner keeping it alive. Returns: the shared pointer.
  [[nodiscard]] const std::shared_ptr<const detail::LocateContextRep>& rep() const noexcept {
    return rep_;
  }

 private:
  template <StorageFloat T>
  static Bspline<double> promote(const Bspline<T>& s) {
    if constexpr (std::is_same_v<T, double>) {
      return s;
    } else {
      std::vector<BsplineSpace1D<double>> dirs;
      for (const auto& sub : s.rep()->space.spaces()) {
        const std::span<const T> k = sub.knots();
        const std::vector<double> k64(k.begin(), k.end());
        dirs.emplace_back(std::span<const double>(k64), sub.degree(), sub.periodic());
      }
      const std::span<const T> cp = s.control_points();
      const std::vector<double> cp64(cp.begin(), cp.end());
      return Bspline<double>(BsplineSpace<double>(std::move(dirs)), std::span<const double>(cp64),
                             s.rational());
    }
  }

  template <StorageFloat T>
  static void check_invertible(const Bspline<T>& s) {
    const std::size_t dim = s.dim();
    const std::size_t rank = s.rank();
    if (rank > dim) {
      throw not_implemented("locate: embedded geometries (rank " + std::to_string(rank) +
                            " > dim " + std::to_string(dim) +
                            ") need a Gauss-Newton closest-point solve, which is not implemented; "
                            "only square maps (rank == dim) can be inverted.");
    }
    if (rank < dim) {
      throw value_error(
          "locate: a B-spline mapping cannot be inverted with fewer physical coordinates than "
          "parametric directions; got rank " +
          std::to_string(rank) + " < dim " + std::to_string(dim) + ".");
    }
    const auto& spaces = s.rep()->space.spaces();
    for (std::size_t axis = 0; axis < spaces.size(); ++axis) {
      if (spaces[axis].periodic()) {
        throw value_error("locate: periodic B-spline spaces are not supported (axis " +
                          std::to_string(axis) + ").");
      }
    }
    if (s.rational()) {
      const std::span<const T> cp = s.control_points();
      const std::size_t cps = s.cp_size();
      for (std::size_t i = cps - 1; i < cp.size(); i += cps) {
        // Compared in double, as the previous implementation cast the weights first; NaN fails.
        if (!(static_cast<double>(cp[i]) > 0.0)) {
          throw value_error(
              "locate: every NURBS weight must be strictly positive; the convex-hull property the "
              "candidate search relies on does not hold otherwise.");
        }
      }
    }
  }

  template <StorageFloat T>
  static std::shared_ptr<const detail::LocateContextRep> build(const Bspline<T>& s) {
    namespace lk = detail::locate_kernels;
    check_invertible(s);
    Bspline<double> b = promote(s);
    const std::size_t dim = b.dim();
    const std::size_t rank = b.rank();
    std::vector<std::vector<double>> breakpoints;
    std::vector<std::size_t> num_intervals;
    std::vector<double> lo(dim);
    std::vector<double> hi(dim);
    std::size_t n_cells = 1;
    for (std::size_t d = 0; d < dim; ++d) {
      const auto& sub = b.rep()->space.spaces()[d];
      const std::span<const double> u = sub.unique_knots_and_multiplicity(true).first;
      breakpoints.emplace_back(u.begin(), u.end());
      num_intervals.push_back(u.size() - 1);
      n_cells *= u.size() - 1;
      const auto [a, z] = sub.domain();
      lo[d] = a;
      hi[d] = z;
    }
    auto [cell_lo, cell_hi] = lk::cell_physical_bounds(b);
    const double eps64 = std::numeric_limits<double>::epsilon();
    const double ratio = eps64 / static_cast<double>(std::numeric_limits<T>::epsilon());
    const double parametric = lk::parametric_scale(b, lo, hi, ratio);
    const lk::BoxLengths lengths = lk::box_lengths(cell_lo, cell_hi, rank);
    const double physical = lk::physical_scale(lengths);
    const std::size_t shape[2] = {n_cells, rank};
    grid::BVH bvh = grid::BVH::from_cell_bounds(cell_lo, cell_hi, shape, shape);
    grid::TensorProductGrid tpg(breakpoints);
    return std::make_shared<const detail::LocateContextRep>(detail::LocateContextRep{
        std::move(b), std::move(bvh), std::move(tpg), std::move(breakpoints),
        std::move(num_intervals), std::move(lo), std::move(hi),
        static_cast<double>(tolerance::get_default<T>()), physical, parametric, lengths.diagonal,
        lk::geometric_scale(physical, parametric)});
  }

  std::shared_ptr<const detail::LocateContextRep> rep_;
};

/// Get the work of locating one point, which sizes `locate`'s chunks.
///
/// One item is one point. A lower bound, the cheapest point that is found: a tree descent, one map
/// evaluation at the first candidate's midpoint and one Newton step, which is a Jacobian (about
/// `dim + 1` evaluations' worth of tabulation and contraction), its singular values and solve
/// (a few Jacobi sweeps and an LU of a `dim x dim` matrix, counted as `10 dim^3`, an admitted
/// order of magnitude) and one trial evaluation: about `dim + 3` value evaluations, each
/// `value_ops_per_point`. Points off the image or on folded maps cost many times more (every
/// candidate, both starts, the halvings), so this underestimates: chunks come out larger than
/// `chunk_ops` intends and a batch runs serially up to a larger size than it would at its true
/// cost. Both err toward less pool overhead on small batches, and the chunks are claimed
/// dynamically, so the imbalance of large chunks is bounded by one chunk per thread. An estimate
/// of the algorithm's work, not a timing.
///
/// Args:
///     ctx (const LocateContext&): The context.
///
/// Returns:
///     pantr::parallel::WorkEstimate: Operations per point.
[[nodiscard]] inline pantr::parallel::WorkEstimate locate_work(const LocateContext& ctx) {
  const Bspline<double>& b = ctx.spline();
  const double evals = static_cast<double>(b.dim()) + 3.0;
  const double descent = 2.0 * static_cast<double>(b.rank()) *
                         static_cast<double>(std::bit_width(static_cast<std::uint64_t>(
                             (std::max)(ctx.bvh().n_nodes(), std::int64_t{1}))));
  const double d = static_cast<double>(b.dim());
  const double linalg = 10.0 * d * d * d;
  return pantr::parallel::WorkEstimate{evals * detail::value_ops_per_point(b) + linalg + descent};
}

/// Invert the mapping at `n` physical points.
///
/// Per point: the candidate cells are the leaves whose box meets the query point padded by the
/// acceptance threshold (the threshold a returned coordinate is graded against, so no cell that
/// could hold a solution is missed), tried in ascending cell id; per candidate a Newton solve from
/// the cell midpoint and, if that does not converge, from the nearest corner
/// (`detail::locate_kernels::nearest_corner_start`). The first converged solve wins, and its cell
/// id is the grid's for the returned coordinates.
///
/// Args:
///     ctx (const LocateContext&): The context of the spline to invert.
///     points (std::span<const double>): Row-major `(n, rank)` physical points.
///     n (std::size_t): The number of points.
///     tol (std::optional<double>): An explicit threshold on `||F(xi) - x||_2`, governing stopping
///         and acceptance alike, or none for `ctx.default_thresholds()`.
///     max_iter (std::int64_t): Maximum number of Newton steps per candidate and start.
///
/// Returns:
///     LocateResult: Cell ids and parametric coordinates.
///
/// Raises:
///     value_error: `max_iter < 1`; `tol` not positive and finite; `points` not `n * rank` values
///         or not finite; the default thresholds refused (`default_thresholds`) or overflowed to
///         infinity (with at least one point).
[[nodiscard]] inline LocateResult locate(const LocateContext& ctx, std::span<const double> points,
                                         std::size_t n, std::optional<double> tol,
                                         std::int64_t max_iter) {
  namespace lk = detail::locate_kernels;
  const Bspline<double>& b = ctx.spline();
  const std::size_t dim = b.dim();
  const std::size_t rank = b.rank();
  if (max_iter < 1) {
    throw value_error("locate: max_iter must be >= 1; got " + std::to_string(max_iter) + ".");
  }
  if (tol && !(*tol > 0.0 && std::isfinite(*tol))) {
    throw value_error("locate: tol must be positive and finite; got " + py_repr(*tol) + ".");
  }
  if (points.size() != n * rank) {
    throw value_error("locate: points must have shape (n, " + std::to_string(rank) +
                      ") for a rank-" + std::to_string(rank) + " B-spline; got " +
                      std::to_string(points.size()) + " values for n = " + std::to_string(n) + ".");
  }
  if (!std::all_of(points.begin(), points.end(), [](double v) { return std::isfinite(v); })) {
    throw value_error("locate: points must be finite; got a NaN or infinite coordinate.");
  }
  const LocateThresholds th = tol ? LocateThresholds{*tol, *tol} : ctx.default_thresholds();
  // A default threshold overflows when the control points' box diagonal does (coordinates near
  // 1e154 and beyond); an infinite pad would make every point "found" at its first start. Refused
  // with the message the previous implementation raised when it padded the query box by it.
  if (n > 0 && !std::isfinite(th.accept)) {
    std::string list = "[";
    for (std::size_t c = 0; c < rank; ++c) {
      list += (c == 0 ? "" : ", ") + py_repr(th.accept);
    }
    throw value_error("pad(r) entries must be finite; got " + list + "].");
  }
  LocateResult res{NdArray<std::int64_t>({n}, for_overwrite), NdArray<double>({n, dim})};
  std::int64_t* const ids = res.cell_ids.data();
  double* const ref = res.ref_coords.data();
  std::fill(ids, ids + n, std::int64_t{-1});
  std::fill(ref, ref + n * dim, std::numeric_limits<double>::quiet_NaN());
  const double* const src = points.data();
  pantr::parallel::parallel_for(n, locate_work(ctx), [&](std::size_t begin, std::size_t end) {
    lk::Workspace ws(b, ctx.domain_lo(), ctx.domain_hi());
    std::vector<double> qlo(rank);
    std::vector<double> qhi(rank);
    std::vector<double> cell_lo(dim);
    std::vector<double> cell_hi(dim);
    std::vector<double> xi(dim);
    for (std::size_t p = begin; p < end; ++p) {
      const double* x = src + p * rank;
      for (std::size_t c = 0; c < rank; ++c) {
        qlo[c] = x[c] - th.accept;
        qhi[c] = x[c] + th.accept;
      }
      std::vector<std::int64_t> candidates = ctx.bvh().query_aabb(qlo, qhi);
      std::sort(candidates.begin(), candidates.end());
      for (const std::int64_t cell : candidates) {
        ctx.cell_box(cell, cell_lo.data(), cell_hi.data());
        for (std::size_t k = 0; k < dim; ++k) {
          xi[k] = 0.5 * (cell_lo[k] + cell_hi[k]);
        }
        bool found = lk::newton_refine(ws, x, xi.data(), th, max_iter);
        if (!found) {
          lk::nearest_corner_start(ws, x, cell_lo.data(), cell_hi.data(), xi.data());
          found = lk::newton_refine(ws, x, xi.data(), th, max_iter);
        }
        if (found) {
          std::copy(xi.begin(), xi.end(), ref + p * dim);
          ids[p] = ctx.grid().locate(std::span<const double>(xi)).value_or(-1);
          break;
        }
      }
    }
  });
  return res;
}

}  // namespace pantr::bspline
