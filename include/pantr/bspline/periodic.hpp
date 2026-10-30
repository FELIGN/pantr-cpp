/// @file periodic.hpp
/// @brief Conversion of an open (or non-periodic) B-spline to periodic form: an exact change of
/// basis, solved by a QR factorization.
///
/// The Oslo matrix maps the periodic (ghost-extended, wrapped) basis to the open one; the periodic
/// control points are the least-squares solution of `M x = c_open`, `M = oslo W`, with `W` the
/// wrapping matrix, `W[i, i mod n_per] = 1`. The factorization is Householder QR in `double`
/// (`Eigen::HouseholderQR`) followed by a triangular solve, whatever the storage type, and the
/// result is cast back. A residual check refuses an input that is not periodic to the requested
/// regularity.
#pragma once

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/detail/oslo.hpp"
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

#include <Eigen/Core>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bspline {

/// The continuity value that means "maximum regularity, `degree - 1`" in `to_periodic`.
inline constexpr long long convert_default_continuity = -1;

/// Refuse a target boundary multiplicity outside `[1, degree]`.
///
/// The precondition of `to_periodic_1d`, enforced rather than assumed because
/// `build_periodic_knot_vector` cannot survive its violation: with `m_bdy = 0` the per-period tile
/// it replicates has total multiplicity 0 at the seam, so its right-hand loop appends nothing and
/// increments its shift forever.
///
/// Every caller but one derives `m_bdy` from a range that already satisfies this: `to_periodic`
/// computes `degree - continuity` with a validated continuity, and knot insertion and degree
/// elevation carry a multiplicity forward from an existing periodic space. Degree reduction
/// computes `m_bdy - decrement`, which reaches 0.
///
/// Args:
///     m_bdy (long long): Target boundary multiplicity of the periodic knot vector.
///     degree (int): Degree of the periodic space.
///
/// Raises:
///     value_error: `m_bdy` is outside `[1, degree]`.
inline void check_boundary_multiplicity(long long m_bdy, int degree) {
  if (!(1 <= m_bdy && m_bdy <= degree)) {
    throw value_error(
        "A periodic knot vector needs a boundary multiplicity in [1, degree] = [1, " +
        std::to_string(degree) + "], got " + std::to_string(m_bdy) +
        ". At degree 0 that range is empty: with no ghost knots there is nothing to wrap, and the "
        "periodic form of a piecewise constant is its open form. Work in the open representation "
        "instead.");
  }
}

/// Build a periodic knot vector from an open knot vector's breakpoints.
///
/// Constructs the in-domain part `[a^{m_bdy}, interior, b^{m_bdy}]` and extends it periodically
/// with ghost knots on each side. Ghost knots are computed by shifting individual breakpoint
/// values (`bp + k * period`) rather than whole arrays, to minimise floating-point drift, in the
/// storage type.
///
/// Both ghost loops terminate for the same reason, and only for that reason: the multiplicities of
/// the tile sum to at least `m_bdy`, which `check_boundary_multiplicity` has put at 1 or more. Each
/// pass then appends at least one entry (the right loop from `shift = 2` on, since its `shift == 1`
/// pass skips the seam entry), so `n_ghost` is reached in at most `n_ghost + 1` passes. At
/// `m_bdy = 0` with no interior breakpoint the tile is empty and both would spin forever: that is
/// why the caller checks rather than assumes.
///
/// When the tile holds fewer than `n_ghost` entries (a periodic space of fewer than `degree + 1`
/// intervals, which no space can have) the left loop takes several passes and keeps the last
/// `n_ghost` entries, which are then not in increasing order: the vector is returned as main
/// builds it, and the space built from it refuses it.
///
/// Args:
///     open_knots (std::span<const T>): Open knot vector.
///     degree (int): Degree.
///     interior_bp (std::span<const T>): Unique interior breakpoints (strictly between `a` and
///         `b`).
///     interior_mults (std::span<const std::int64_t>): Multiplicity of each interior breakpoint,
///         the same length as `interior_bp`.
///     m_bdy (long long): Target boundary multiplicity, in `[1, degree]`.
///
/// Returns:
///     std::vector<T>: The periodic knot vector with ghost knots.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] std::vector<T> build_periodic_knot_vector(
    std::span<const T> open_knots, int degree, std::span<const T> interior_bp,
    std::span<const std::int64_t> interior_mults, long long m_bdy) {
  const auto p = static_cast<std::size_t>(degree);
  const T a = open_knots[p];
  const T b = open_knots[open_knots.size() - p - 1];
  const T period = static_cast<T>(b - a);
  const auto n_ghost = static_cast<std::size_t>(static_cast<long long>(degree) + 1 - m_bdy);

  // The tile: [a, xi_1, ..., xi_{N-1}], each with its multiplicity.
  std::vector<T> bp_vals;
  std::vector<std::int64_t> bp_mults;
  bp_vals.push_back(a);
  bp_mults.push_back(m_bdy);
  bp_vals.insert(bp_vals.end(), interior_bp.begin(), interior_bp.end());
  bp_mults.insert(bp_mults.end(), interior_mults.begin(), interior_mults.end());

  // In-domain part: the tile, then b^{m_bdy}.
  std::vector<T> in_domain;
  for (std::size_t i = 0; i < bp_vals.size(); ++i) {
    in_domain.insert(in_domain.end(), static_cast<std::size_t>(bp_mults[i]), bp_vals[i]);
  }
  in_domain.insert(in_domain.end(), static_cast<std::size_t>(m_bdy), b);
  if (n_ghost == 0) {
    return in_domain;
  }

  // Right ghosts: the interior breakpoints shifted by one period, then the whole tile shifted by
  // two, and so on; the leading `a` of the first pass is the seam, already in the domain.
  std::vector<T> right;
  T shift = static_cast<T>(1);
  while (right.size() < n_ghost) {
    for (std::size_t i = 0; i < bp_vals.size() && right.size() < n_ghost; ++i) {
      if (shift == static_cast<T>(1) && bp_vals[i] == a) {
        continue;
      }
      for (std::int64_t k = 0; k < bp_mults[i]; ++k) {
        const T shifted = static_cast<T>(shift * period);
        right.push_back(static_cast<T>(bp_vals[i] + shifted));
        if (right.size() == n_ghost) {
          break;
        }
      }
    }
    shift += static_cast<T>(1);
  }

  // Left ghosts: the whole tile shifted by -1, -2, ..., keeping the last n_ghost entries.
  std::vector<T> left;
  shift = static_cast<T>(-1);
  while (left.size() < n_ghost) {
    for (std::size_t i = 0; i < bp_vals.size(); ++i) {
      for (std::int64_t k = 0; k < bp_mults[i]; ++k) {
        const T shifted = static_cast<T>(shift * period);
        left.push_back(static_cast<T>(bp_vals[i] + shifted));
      }
    }
    shift -= static_cast<T>(1);
  }

  std::vector<T> out;
  out.reserve(n_ghost + in_domain.size() + n_ghost);
  out.insert(out.end(), left.end() - static_cast<std::ptrdiff_t>(n_ghost), left.end());
  out.insert(out.end(), in_domain.begin(), in_domain.end());
  out.insert(out.end(), right.begin(), right.end());
  return out;
}

/// Get the magnitude the periodic admissibility tests are relative to: `max |c|` over every entry.
///
/// An inf-norm on purpose: both quantities graded below are inf-norms, so the scale has to be one
/// too. A row 2-norm would be right for the knot-removal kernel, which aggregates a Euclidean
/// distance over its stacked directions; here the array has been flattened to `(n, rank *
/// n_other)`, so a row 2-norm would sum over every other parametric direction and inflate the
/// budget by `sqrt(rank * n_other)`, which grows with the size of the net while the graded quantity
/// does not.
///
/// What it is relative to is the **whole net**, not the neighbourhood of the seam. A small feature
/// beside a large one is therefore graded against the large one, which is loose for the small
/// feature; the tighter statement would use the `2 (p + 1)` control points in the boundary
/// supports. That is a real limitation, stated rather than fixed, because narrowing it changes
/// which splines convert.
///
/// Computed on the array as given, so for a rational spline it is the scale of the *homogeneous*
/// control points, the space these tests measure in.
///
/// Args:
///     ctrl (std::span<const T>): Control points, any shape, flat.
///
/// Returns:
///     double: The magnitude: zero for an identically zero or empty net, NaN when any entry is NaN
///     (as `numpy.max` propagates it).
template <class T>
[[nodiscard]] double control_point_magnitude(std::span<const T> ctrl) {
  return detail::max_abs_propagating<T>(ctrl);
}

/// Get how far the two end control points may sit apart and still close the seam:
/// `get_conservative<T>() * control_point_magnitude(ctrl)`.
///
/// The magnitude is a control-point one because the quantity graded is a distance between two
/// control points. It is emphatically *not* the parametric knot tolerance the caller also holds:
/// that is an absolute parametric length and is not the factor to scale a physical coordinate or a
/// spline coefficient by. Grading a value-space deviation against it makes the verdict track the
/// knot vector's magnitude rather than the function's: the same endpoint mismatch, refused at unit
/// knot scale, would be accepted once the knot vector is scaled up far enough.
///
/// The conservative tier, the same as `closure_deviation_budget`, and the argument is consistency
/// rather than a count of operations. The two grade the *same* defect (a least-squares closure
/// splits a seam mismatch across the two boundary rows), so the seam deviation is of the same order
/// as the residual that follows it. A budget tighter than the one grading the same quantity
/// downstream can only produce refusals the downstream test would not have made, such as refusing
/// the derivative of a genuinely periodic NURBS of high degree over many elements as "not
/// periodic". The control points reach this test through a short chain on the plain conversion,
/// but not on the rational derivative's path: the quotient rule runs at degree `2p` through the
/// product and a refinement to a common space first. A budget for this site has to cover its
/// longest caller.
///
/// Args:
///     ctrl (std::span<const T>): Control points, flat.
///
/// Returns:
///     double: The budget, in the units of a control-point distance.
template <StorageFloat T>
[[nodiscard]] double seam_deviation_budget(std::span<const T> ctrl) {
  return static_cast<double>(tolerance::get_conservative<T>()) * control_point_magnitude<T>(ctrl);
}

/// Get how far the open form may sit from the periodic subspace and still convert:
/// `get_conservative<T>() * max(magnitude(ctrl), magnitude(ctrl_per))`, raised to `tol_deviation`
/// when the caller supplies one.
///
/// **Why both vectors.** The graded quantity is the residual of a least-squares solve,
/// `max|M x - b|`, and Householder QR is backward stable with a constant polynomial in the shape
/// and free of `cond(M)`: the bound is `c(m, n) eps ||M|| ||x||`. Here `M = oslo W` is entrywise
/// non-negative with unit row sums, so `||M||_inf` is exactly 1 and drops out, but the bound is
/// against `x`, the periodic control points, not against `b`. Normalising by `b` alone would
/// re-import the conditioning through `||x|| / ||b||`. Taking the larger of the two removes
/// conditioning from the argument instead of budgeting for it. The conservative tier's 4096
/// epsilons then cover the shape constant and the Oslo chain's own accumulation, the rational
/// derivative's chain included.
///
/// Both budgets read `eps` from the control points' own storage type (the expression they replace
/// read it from `double` unconditionally, which for a `float` spline demanded of the data a
/// precision it does not carry).
///
/// A caller whose input is only approximately periodic *by design* passes its own budget, in the
/// same control-point units, which is why the `max` here is legitimate where the one it replaces
/// was not: both operands are control-point distances rather than one of each kind. For a rational
/// spline those units are **homogeneous**.
///
/// Args:
///     ctrl (std::span<const T>): Open-form control points, the right-hand side of the solve.
///     ctrl_per (std::span<const double>): Periodic control points, its solution.
///     tol_deviation (std::optional<double>): The caller's budget in control-point units, or no
///         value for the round-off floor alone.
///
/// Returns:
///     double: The budget, in the units of a control-point distance.
template <StorageFloat T>
[[nodiscard]] double closure_deviation_budget(std::span<const T> ctrl,
                                              std::span<const double> ctrl_per,
                                              std::optional<double> tol_deviation) {
  // Python's max(a, b): a unless b > a (so a NaN first operand stays).
  const double mag_a = control_point_magnitude<T>(ctrl);
  const double mag_b = control_point_magnitude<double>(ctrl_per);
  const double scale = (mag_b > mag_a) ? mag_b : mag_a;
  const double floor = static_cast<double>(tolerance::get_conservative<T>()) * scale;
  if (!tol_deviation.has_value()) {
    return floor;
  }
  return (floor > *tol_deviation) ? floor : *tol_deviation;
}

namespace detail {

/// The `double` row-major rows of the periodic change of basis, one at a time.
///
/// Row `r` of `M = oslo W` is the sum of the columns of `oslo` that wrap onto the same periodic
/// function. `oslo` is the Oslo matrix from the ghost-extended periodic knot vector to the open
/// one, trimmed of the ghost region (one factor, or the product of two when the open knot vector
/// is not the trimmed intermediate one); its rows are banded, so a row is formed from the bands,
/// in the storage type as the dense product was, and folded in `double`.
template <StorageFloat T>
class PeriodicChange {
 public:
  PeriodicChange(int degree, std::span<const T> t_per, std::span<const T> t_inter,
                 std::size_t n_ghost, std::span<const T> open_knots, std::size_t n_open,
                 std::size_t n_per)
      : width_(static_cast<std::size_t>(degree) + 1),
        n_ghost_(n_ghost),
        n_open_(n_open),
        n_per_(n_per),
        n_full_(t_per.size() - static_cast<std::size_t>(degree) - 1),
        rows1_(oslo_rows<T>(degree, t_per, t_inter)),
        row_(n_full_, static_cast<T>(0)) {
    // Step 3: the trimmed intermediate has as many control points as the open form unless the
    // open knot vector is not the trimmed one; then a second Oslo step maps one to the other.
    two_step_ = (n_full_ != n_open_);
    if (two_step_) {
      rows2_.emplace(
          oslo_rows<T>(degree, t_inter.subspan(n_ghost, t_inter.size() - 2 * n_ghost), open_knots));
    }
  }

  /// Add row `r` of `M` into `dst[0 .. n_per)` (which the caller has zeroed).
  void add_row(std::size_t r, double* dst) {
    std::size_t lo = n_full_;
    std::size_t hi = 0;
    const auto touch = [&](std::size_t col) {
      lo = (std::min)(lo, col);
      hi = (std::max)(hi, col + 1);
    };
    if (!two_step_) {
      accumulate_row1(n_ghost_ + r, static_cast<T>(1), touch);
    } else {
      const T* alpha2 = rows2_->alphas.data() + r * width_;
      const std::int64_t base2 = rows2_->first_col[r];
      for (std::size_t l = 0; l < width_; ++l) {
        const std::int64_t c2 = base2 + static_cast<std::int64_t>(l);
        if (0 <= c2 && c2 < static_cast<std::int64_t>(n_full_)) {
          accumulate_row1(n_ghost_ + static_cast<std::size_t>(c2), alpha2[l], touch);
        }
      }
    }
    // Fold in double, columns ascending: the sum `oslo @ W` forms.
    for (std::size_t col = lo; col < hi; ++col) {
      dst[col % n_per_] += static_cast<double>(row_[col]);
      row_[col] = static_cast<T>(0);
    }
  }

 private:
  template <class Touch>
  void accumulate_row1(std::size_t t, T weight, Touch& touch) {
    const T* alpha1 = rows1_.alphas.data() + t * width_;
    const std::int64_t base1 = rows1_.first_col[t];
    for (std::size_t l = 0; l < width_; ++l) {
      const std::int64_t col = base1 + static_cast<std::int64_t>(l);
      if (0 <= col && col < static_cast<std::int64_t>(n_full_)) {
        const auto cu = static_cast<std::size_t>(col);
        if (two_step_) {
          row_[cu] += weight * alpha1[l];
        } else {
          row_[cu] = alpha1[l];
        }
        touch(cu);
      }
    }
  }

  std::size_t width_;
  std::size_t n_ghost_;
  std::size_t n_open_;
  std::size_t n_per_;
  std::size_t n_full_;
  OsloRows<T> rows1_;
  std::optional<OsloRows<T>> rows2_;
  std::vector<T> row_;
  bool two_step_ = false;
};

}  // namespace detail

/// Convert an open (clamped) 1D B-spline to periodic form.
///
/// Performs an exact change of basis from the open representation to a periodic one with boundary
/// multiplicity `m_bdy`. The Oslo matrix defines the linear map from the periodic (ghost-extended,
/// wrapped) basis to the open basis; inverting this map by QR recovers the periodic control points.
///
/// Args:
///     open_knots (std::span<const T>): Open (clamped) knot vector with multiplicity `degree + 1`
///         at both boundaries (or a non-clamped one, which skips the seam test).
///     degree (int): Degree.
///     ctrl (mdview<const T, 2>): Control points, extents `(n_open, cols)`.
///     m_bdy (long long): Target boundary multiplicity of the periodic knot vector, in
///         `[1, degree]`.
///     tol (double): Absolute parametric tolerance, used only for knot comparisons; the two
///         admissibility tests measure control-point distances and take `tol_deviation` instead.
///     tol_deviation (std::optional<double>): How far the open form may sit from the periodic
///         subspace, in control-point units (homogeneous ones for a rational spline), or no value
///         for the round-off floor alone. Pass infinity from a caller whose input is only
///         approximately periodic by design. It raises the **closure** budget only: the C0 seam
///         test keeps its derived floor, because an approximate caller does not thereby move the
///         two end control points (the reduction operator's first and last rows are exact unit
///         vectors, so the boundary points arrive bit for bit unchanged). Defaults to no value.
///
/// Returns:
///     KnotsAndCtrl<T>: The periodic knot vector (ghost knots included) and the periodic control
///     points, extents `(n_per, cols)`.
///
/// Raises:
///     value_error: `m_bdy` is outside `[1, degree]`; the control points do not match the knots;
///         the end control points differ by more than the seam budget (clamped knots); the
///         interior breakpoints are inconsistent at the tolerance; the factor `R` is singular
///         (`numpy.linalg.LinAlgError` is a `ValueError`); or the residual exceeds the closure
///         budget, in main's words.
template <StorageFloat T>
[[nodiscard]] KnotsAndCtrl<T> to_periodic_1d(std::span<const T> open_knots, int degree,
                                             mdview<const T, 2> ctrl, long long m_bdy, double tol,
                                             std::optional<double> tol_deviation = std::nullopt) {
  using ColMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  const int p = degree;
  const auto pu = static_cast<std::size_t>(p);

  check_boundary_multiplicity(m_bdy, p);
  detail::check_net_matches("to_periodic", ctrl.extent(0), open_knots.size(), p);

  const std::size_t n_open = ctrl.extent(0);
  const std::size_t cols = ctrl.extent(1);
  const std::span<const T> ctrl_flat(ctrl.data_handle(), n_open * cols);
  const auto a = static_cast<double>(open_knots[pu]);
  const auto b = static_cast<double>(open_knots[open_knots.size() - pu - 1]);

  // Quick C^0 check when the knots are clamped (P_0 = f(a), P_{n-1} = f(b)).
  int m_left = 0;
  int m_right = 0;
  for (std::size_t i = 0; i <= pu; ++i) {
    if (static_cast<double>(std::abs(static_cast<T>(open_knots[i] - static_cast<T>(a)))) <= tol) {
      ++m_left;
    }
    if (static_cast<double>(std::abs(static_cast<T>(open_knots[open_knots.size() - pu - 1 + i] -
                                                    static_cast<T>(b)))) <= tol) {
      ++m_right;
    }
  }
  if (m_left == p + 1 && m_right == p + 1) {
    const double c0_tol = seam_deviation_budget<T>(ctrl_flat);
    std::vector<T> diff(cols);
    for (std::size_t r = 0; r < cols; ++r) {
      diff[r] = static_cast<T>(ctrl(0, r) - ctrl(n_open - 1, r));
    }
    const double c0_dev = detail::max_abs_propagating<T>(diff);
    if (c0_dev > c0_tol) {
      throw value_error("Function values do not match at the domain endpoints (max deviation " +
                        py_format_exp(c0_dev, 2) + ", tolerance " + py_format_exp(c0_tol, 2) +
                        "). The B-spline is not periodic.");
    }
  }

  const KnotClasses<T> in_domain = unique_knots_and_multiplicity<T>(open_knots, p, tol, true);
  // Exact breakpoint values taken from the open knots (the first occurrence of each group), to
  // avoid the drift of the representative the classification chooses.
  std::vector<T> bp_exact;
  bp_exact.push_back(open_knots[pu]);
  for (std::size_t i = pu + 1; i + pu < open_knots.size(); ++i) {
    const T v = open_knots[i];
    if (std::abs(static_cast<double>(v) - static_cast<double>(bp_exact.back())) > tol) {
      bp_exact.push_back(v);
    }
  }
  const std::size_t n_bp = bp_exact.size();
  const std::size_t n_cls = in_domain.multiplicities.size();
  const std::size_t n_interior_bp = n_bp >= 2 ? n_bp - 2 : 0;
  const std::size_t n_interior_mult = n_cls >= 2 ? n_cls - 2 : 0;
  if (n_interior_bp != n_interior_mult) {
    throw value_error(
        "The interior breakpoints of the knot vector are not consistent at the "
        "tolerance " +
        py_repr(tol) + ": " + std::to_string(n_interior_bp) + " breakpoints, " +
        std::to_string(n_interior_mult) + " multiplicities.");
  }
  const std::span<const T> interior_bp = std::span<const T>(bp_exact).subspan(1, n_interior_bp);
  const std::span<const std::int64_t> interior_mults =
      std::span<const std::int64_t>(in_domain.multiplicities).subspan(1, n_interior_mult);

  std::vector<T> t_per =
      build_periodic_knot_vector<T>(open_knots, p, interior_bp, interior_mults, m_bdy);

  // The Oslo chain: T_per -> T_inter (clamping knots inserted) -> open knots.
  const auto n_ghost = static_cast<std::size_t>(static_cast<long long>(p) + 1 - m_bdy);
  std::vector<T> t_inter(t_per);
  t_inter.insert(t_inter.end(), n_ghost, static_cast<T>(a));
  t_inter.insert(t_inter.end(), n_ghost, static_cast<T>(b));
  std::sort(t_inter.begin(), t_inter.end());

  const long long n_per_signed = num_basis<T>(t_per, p, true, tol);
  const auto n_per = static_cast<std::size_t>(n_per_signed);

  // M = oslo W, in double, row by row.
  ColMatrix m_mat =
      ColMatrix::Zero(static_cast<Eigen::Index>(n_open), static_cast<Eigen::Index>(n_per));
  {
    detail::PeriodicChange<T> change(p, t_per, t_inter, n_ghost, open_knots, n_open, n_per);
    std::vector<double> row(n_per);
    for (std::size_t r = 0; r < n_open; ++r) {
      std::fill(row.begin(), row.end(), 0.0);
      change.add_row(r, row.data());
      for (std::size_t c = 0; c < n_per; ++c) {
        m_mat(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = row[c];
      }
    }
  }

  // Exact solve by QR: M C_per = C_open.
  ColMatrix c_open(static_cast<Eigen::Index>(n_open), static_cast<Eigen::Index>(cols));
  for (std::size_t r = 0; r < n_open; ++r) {
    for (std::size_t c = 0; c < cols; ++c) {
      c_open(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) =
          static_cast<double>(ctrl(r, c));
    }
  }
  const Eigen::HouseholderQR<ColMatrix> qr(m_mat);
  const auto np = static_cast<Eigen::Index>(n_per);
  const ColMatrix qt_c = qr.householderQ().transpose() * c_open;
  const auto r_factor = qr.matrixQR().topLeftCorner(np, np).template triangularView<Eigen::Upper>();
  for (Eigen::Index i = 0; i < np; ++i) {
    if (qr.matrixQR()(i, i) == 0.0) {
      throw value_error("Singular matrix");
    }
  }
  const ColMatrix ctrl_per = r_factor.solve(qt_c.topRows(np));

  // Residual check.
  const ColMatrix residual_mat = m_mat * ctrl_per - c_open;
  double residual = 0.0;
  for (Eigen::Index j = 0; j < residual_mat.cols() && !std::isnan(residual); ++j) {
    for (Eigen::Index i = 0; i < residual_mat.rows(); ++i) {
      const double v = std::abs(residual_mat(i, j));
      if (std::isnan(v)) {
        residual = v;
        break;
      }
      residual = (std::max)(residual, v);
    }
  }
  std::vector<double> per_flat(n_per * cols);
  for (std::size_t i = 0; i < n_per; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      per_flat[i * cols + j] = ctrl_per(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
    }
  }
  const double residual_tol = closure_deviation_budget<T>(ctrl_flat, per_flat, tol_deviation);
  if (residual > residual_tol) {
    throw value_error("Residual " + py_format_exp(residual, 2) + " exceeds tolerance " +
                      py_format_exp(residual_tol, 2) +
                      ". The B-spline does not have sufficient smoothness at the seam for the "
                      "requested periodic regularity.");
  }

  NdArray<T> out({n_per, cols}, for_overwrite);
  for (std::size_t i = 0; i < per_flat.size(); ++i) {
    out.data()[i] = static_cast<T>(per_flat[i]);
  }
  return KnotsAndCtrl<T>{std::move(t_per), std::move(out)};
}

/// Convert the non-periodic directions of a B-spline to periodic form.
///
/// For each direction that is not already periodic and not skipped, the exact change of basis from
/// the open (or non-open) representation to a periodic one with the requested continuity at the
/// seam (`to_periodic_1d`). Directions are converted in order.
///
/// This assumes the B-spline represents an inherently periodic function: values and derivatives
/// match at the domain boundaries up to the requested continuity. Gross violations raise (through
/// the residual check); small deviations may pass silently.
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///     continuity (std::span<const std::optional<long long>>): The target continuity at the seam
///         for each direction, `dim` entries: no value skips the direction, `-1`
///         (`convert_default_continuity`) means the maximum, `degree - 1`; otherwise it must be in
///         `[0, degree - 1]`.
///
/// Returns:
///     Bspline<T>: The B-spline with the requested directions made periodic.
///
/// Raises:
///     value_error: `continuity.size() != dim` ("continuity tuple length ... != dimension ...");
///         no direction would be converted; a continuity outside `[0, degree - 1]`; or the
///         conversion of a direction refuses (see `to_periodic_1d`).
template <StorageFloat T>
[[nodiscard]] Bspline<T> to_periodic(const Bspline<T>& b,
                                     std::span<const std::optional<long long>> continuity) {
  const std::size_t dim = b.dim();
  if (continuity.size() != dim) {
    throw value_error("continuity tuple length " + std::to_string(continuity.size()) +
                      " != dimension " + std::to_string(dim) + ".");
  }
  const BsplineSpace<T> whole = b.space();
  const std::vector<BsplineSpace1D<T>>& spaces = whole.spaces();
  const auto will_convert = [&](std::size_t d) {
    return !spaces[d].periodic() && continuity[d].has_value();
  };
  bool any = false;
  for (std::size_t d = 0; d < dim; ++d) {
    any = any || will_convert(d);
  }
  if (!any) {
    throw value_error(
        "No direction to convert: all directions are either already periodic or explicitly "
        "skipped.");
  }
  return detail::map_directions<T>(
      b, will_convert,
      [&](std::size_t d, const BsplineSpace1D<T>& space,
          mdview<const T, 2> flat) -> std::pair<BsplineSpace1D<T>, NdArray<T>> {
        const int p = space.degree();
        long long c = *continuity[d];
        if (c == convert_default_continuity) {
          c = static_cast<long long>(p) - 1;
        }
        if (!(0 <= c && c <= static_cast<long long>(p) - 1)) {
          throw value_error("continuity must be in [0, degree-1]=[0, " + std::to_string(p - 1) +
                            "] for direction " + std::to_string(d) + ", got " + std::to_string(c) +
                            ".");
        }
        const long long m_bdy = static_cast<long long>(p) - c;
        KnotsAndCtrl<T> r = to_periodic_1d<T>(space.knots(), p, flat, m_bdy, space.tolerance());
        BsplineSpace1D<T> per(std::span<const T>(r.knots), p, true);
        return {std::move(per), std::move(r.ctrl)};
      });
}

}  // namespace pantr::bspline
