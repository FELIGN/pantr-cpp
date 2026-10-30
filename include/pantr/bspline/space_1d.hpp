/// @file space_1d.hpp
/// @brief `BsplineSpace1D`: a knot vector and a degree, with the basis evaluated at points.
#pragma once

#include "pantr/basis/bernstein.hpp"
#include "pantr/basis/detail/tabulate_1d.hpp"
#include "pantr/bspline/detail/basis_kernels.hpp"
#include "pantr/bspline/knot_factories.hpp"
#include "pantr/bspline/knots.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pantr::bspline {

namespace detail {

/// The shared representation of a `BsplineSpace1D`: the knots, and the state derived from them at
/// construction (eager, since none of it costs more than the validation that precedes it).
///
/// Attributes:
///     knots (std::vector<T>): The knot vector, snapped if snapping was requested.
///     degree (int): Degree.
///     periodic (bool): Periodicity.
///     tol (double): Absolute parametric tolerance (`knot_tolerance` of the knots as given).
///     num_basis (long long): Number of basis functions.
///     classes (KnotClasses<T>): Every knot class.
///     lo_class (std::size_t): The class holding the first domain knot.
///     hi_class (std::size_t): The class holding the last domain knot.
///     first_basis (std::vector<std::int64_t>): First supported function of each interval.
///     left_open (bool): The first `degree + 1` knots are one knot (never for a periodic space).
///     right_open (bool): The last `degree + 1` knots are one knot (never for a periodic space).
///     bezier_like (bool): Non-periodic, open at both ends and `num_basis == degree + 1`.
template <StorageFloat T>
struct BsplineSpace1DRep {
  std::vector<T> knots;
  int degree = 0;
  bool periodic = false;
  double tol = 0.0;
  long long num_basis = 0;
  KnotClasses<T> classes;
  std::size_t lo_class = 0;
  std::size_t hi_class = 0;
  std::vector<std::int64_t> first_basis;
  bool left_open = false;
  bool right_open = false;
  bool bezier_like = false;
};

/// Refuse a knot vector and degree that define no space (the checks before any is derived).
///
/// Raises:
///     value_error: `degree < 0`, fewer than `2 * degree + 2` knots, decreasing (or NaN) knots,
///         in that order.
template <StorageFloat T>
void check_knots_and_degree(std::span<const T> knots, int degree) {
  if (degree < 0) {
    throw value_error("degree must be non-negative");
  }
  if (static_cast<long long>(knots.size()) < 2LL * degree + 2) {
    throw value_error("knots must have at least 2*degree+2 elements");
  }
  for (std::size_t i = 1; i < knots.size(); ++i) {
    if (!(static_cast<T>(knots[i] - knots[i - 1]) >= static_cast<T>(0))) {
      throw value_error("knots must be non-decreasing");
    }
  }
}

}  // namespace detail

/// The basis derivatives at points, and whether they are complete.
///
/// Attributes:
///     values (NdArray<P>): `(n, n_deriv + 1, degree + 1)` derivatives.
///     first (NdArray<std::int64_t>): `(n,)` first supported function per point.
///     complete (bool): False when the recurrence met a zero divisor (see
///         `BsplineSpace1D::tabulate_basis_derivatives`); `values` is then incomplete.
template <StorageFloat P>
struct [[nodiscard]] BasisDerivatives {
  NdArray<P> values;
  NdArray<std::int64_t> first;
  bool complete;
};

/// A 1D B-spline space: a non-decreasing knot vector and a degree, possibly periodic.
///
/// An immutable value (design §2.2): the knots are copied in at construction, everything derived
/// from them is computed then, and every copy shares one read-only representation. An instance
/// always has at least one interval.
///
/// The tolerance (`tolerance()`) is an absolute parametric length, `8 eps * max(span, |knots[0]|,
/// |knots[-1]|)` of the knots as given: two knots closer than it are one knot, a point within it
/// of an end is in the domain, two intervals differing by less are the same length.
///
/// Attributes:
///     rep_ (std::shared_ptr<const detail::BsplineSpace1DRep<T>>): The shared representation.
template <StorageFloat T>
class BsplineSpace1D {
 public:
  /// The shared representation type.
  using Rep = detail::BsplineSpace1DRep<T>;

  /// Build a space from a knot vector (copied) and a degree.
  ///
  /// With `snap_knots`, each class of knots meant to be one knot (consecutive steps within the
  /// tolerance) is stored as its first knot, which keeps the values supplied and is idempotent.
  ///
  /// Args:
  ///     knots (std::span<const T>): Non-decreasing knot vector, at least `2 * degree + 2` knots.
  ///     degree (int): Degree, `>= 0`.
  ///     periodic (bool): Whether the space is periodic. Defaults to false.
  ///     snap_knots (bool): Whether to merge each knot class onto one value. Defaults to true.
  ///
  /// Raises:
  ///     value_error: `degree < 0`; fewer than `2 * degree + 2` knots; decreasing or NaN knots;
  ///         fewer than `degree + 1` basis functions; snapping collapsed distinct knots onto one
  ///         point; the domain spans no interval; or, at positive degree, the last knot repeats
  ///         more than `degree + 1` times (its basis would not sum to one at the right end).
  BsplineSpace1D(std::span<const T> knots, int degree, bool periodic = false,
                 bool snap_knots = true)
      : rep_(build(knots, degree, periodic, snap_knots, nullptr)) {}

  /// Rebuild a space from its stored state (a pickle): the knots exactly as stored, no snapping,
  /// and the tolerance it was built with, so the rebuilt space equals the original.
  ///
  /// Args:
  ///     knots (std::span<const T>): The stored knot vector.
  ///     degree (int): Degree.
  ///     periodic (bool): Periodicity.
  ///     tol (double): The stored tolerance.
  ///
  /// Returns:
  ///     BsplineSpace1D: The space.
  ///
  /// Raises:
  ///     value_error: As the constructor, but for the snapping check.
  [[nodiscard]] static BsplineSpace1D from_state(std::span<const T> knots, int degree,
                                                 bool periodic, double tol) {
    return BsplineSpace1D(build(knots, degree, periodic, false, &tol));
  }

  /// The knot vector.
  [[nodiscard]] std::span<const T> knots() const noexcept { return rep_->knots; }
  /// The degree.
  [[nodiscard]] int degree() const noexcept { return rep_->degree; }
  /// Whether the space is periodic.
  [[nodiscard]] bool periodic() const noexcept { return rep_->periodic; }
  /// The absolute parametric tolerance.
  [[nodiscard]] double tolerance() const noexcept { return rep_->tol; }
  /// The number of basis functions.
  [[nodiscard]] long long num_basis() const noexcept { return rep_->num_basis; }
  /// The number of in-domain intervals, at least 1.
  [[nodiscard]] long long num_intervals() const noexcept {
    return static_cast<long long>(rep_->hi_class - rep_->lo_class);
  }
  /// The distinct knots (each class's first) and their multiplicities, all of them or only the
  /// classes from the first domain knot's to the last's (whole classes).
  ///
  /// Returns:
  ///     std::pair<std::span<const T>, std::span<const std::int64_t>>: Views into the space.
  [[nodiscard]] std::pair<std::span<const T>, std::span<const std::int64_t>>
  unique_knots_and_multiplicity(bool in_domain) const noexcept {
    const std::span<const T> k(rep_->classes.knots);
    const std::span<const std::int64_t> m(rep_->classes.multiplicities);
    if (!in_domain) {
      return {k, m};
    }
    const std::size_t count = rep_->hi_class - rep_->lo_class + 1;
    return {k.subspan(rep_->lo_class, count), m.subspan(rep_->lo_class, count)};
  }
  /// The domain `(knots[degree], knots[size - degree - 1])`.
  [[nodiscard]] std::pair<T, T> domain() const noexcept {
    const auto d = static_cast<std::size_t>(rep_->degree);
    return {rep_->knots[d], rep_->knots[rep_->knots.size() - d - 1]};
  }
  /// Whether the first `degree + 1` knots are one knot (false for a periodic space).
  [[nodiscard]] bool has_left_end_open() const noexcept { return rep_->left_open; }
  /// Whether the last `degree + 1` knots are one knot (false for a periodic space).
  [[nodiscard]] bool has_right_end_open() const noexcept { return rep_->right_open; }
  /// Whether both ends are open.
  [[nodiscard]] bool has_open_knots() const noexcept { return rep_->left_open && rep_->right_open; }
  /// Whether the space is one Bézier piece (open ends and `degree + 1` functions).
  [[nodiscard]] bool has_bezier_like_knots() const noexcept { return rep_->bezier_like; }

  /// The first supported basis function of each interval: `k - degree` where `k` is the last
  /// position of the interval's starting knot, exact integer arithmetic.
  ///
  /// Returns:
  ///     std::span<const std::int64_t>: `num_intervals()` entries, non-decreasing.
  ///
  /// Raises:
  ///     value_error: The space is periodic.
  [[nodiscard]] std::span<const std::int64_t> first_basis_per_interval() const {
    if (rep_->periodic) {
      throw value_error("first_basis_per_interval: periodic B-spline spaces are not supported.");
    }
    return rep_->first_basis;
  }

  /// Flag each interval that is cardinal (see `bspline::cardinal_intervals`).
  ///
  /// Args:
  ///     out (std::span<bool>): One flag per interval.
  ///
  /// Raises:
  ///     value_error: `out` does not hold `num_intervals()` flags.
  void cardinal_intervals(std::span<bool> out) const {
    const auto n = static_cast<std::size_t>(num_intervals());
    if (out.size() != n) {
      const std::size_t got[1] = {out.size()};
      const std::size_t want[1] = {n};
      throw value_error("Output array has shape " + py_repr(std::span<const std::size_t>(got)) +
                        ", but expected shape " + py_repr(std::span<const std::size_t>(want)));
    }
    bspline::cardinal_intervals<T>(knots(), degree(), tolerance(), out);
  }

  /// Whether `pt` lies in the domain up to the tolerance.
  template <StorageFloat P>
  [[nodiscard]] bool contains(P pt) const noexcept {
    return is_in_domain<T, P>(knots(), degree(), pt, tolerance());
  }

  /// Whether every point lies in the domain up to the tolerance (a NaN point does not).
  template <StorageFloat P>
  [[nodiscard]] bool all_in_domain(std::span<const P> pts) const noexcept {
    for (const P p : pts) {
      if (!contains<P>(p)) {
        return false;
      }
    }
    return true;
  }

  /// Refuse points outside the domain.
  ///
  /// Args:
  ///     pts (std::span<const P>): The points.
  ///     what (const std::string&): How the message names the points (`pts`).
  ///
  /// Raises:
  ///     value_error: A point (NaN included) is outside the domain.
  template <StorageFloat P>
  void check_in_domain(std::span<const P> pts, const std::string& what = "pts") const {
    if (!all_in_domain<P>(pts)) {
      const auto [a, b] = domain();
      throw value_error("One or more values in " + what + " are outside the knot vector domain (" +
                        py_repr(static_cast<double>(a)) + ", " + py_repr(static_cast<double>(b)) +
                        ")");
    }
  }

  /// Evaluate the `degree + 1` non-zero basis functions at each point into `out`, and the index of
  /// the first of them into `first`.
  ///
  /// A Bézier-like space is evaluated by the Bernstein recurrence on `(pt - a) / (b - a)`, any
  /// other by Cox-de Boor (Piegl and Tiller A2.2) on the clamped span. Points outside the domain
  /// with `validate == false` are evaluated on the nearest in-domain span (no extrapolation
  /// guarantee is made).
  ///
  /// Args:
  ///     pts (std::span<const P>): The points.
  ///     out (mdview<P, 2>): Extents `(pts.size(), degree + 1)`.
  ///     first (std::span<std::int64_t>): `pts.size()` entries.
  ///     validate (bool): Refuse points outside the domain. Defaults to true.
  ///
  /// Raises:
  ///     value_error: `validate` and a point outside the domain; `out` or `first` of another size.
  template <StorageFloat P>
  void tabulate_basis(std::span<const P> pts, mdview<P, 2> out, std::span<std::int64_t> first,
                      bool validate = true) const {
    basis::detail::check_out<P, 2>(out, {pts.size(), static_cast<std::size_t>(degree()) + 1});
    check_first(first, pts.size());
    if (validate) {
      check_in_domain<P>(pts);
    }
    tabulate_basis_unchecked<P>(pts, out.data_handle(), first.data());
  }

  /// Evaluate the basis at each point (see the `out` overload).
  ///
  /// Returns:
  ///     std::pair<NdArray<P>, NdArray<std::int64_t>>: Values `(n, degree + 1)` and first
  ///     indices `(n,)`.
  template <StorageFloat P>
  [[nodiscard]] std::pair<NdArray<P>, NdArray<std::int64_t>> tabulate_basis(
      std::span<const P> pts, bool validate = true) const {
    if (validate) {
      check_in_domain<P>(pts);
    }
    NdArray<P> values({pts.size(), static_cast<std::size_t>(degree()) + 1}, for_overwrite);
    NdArray<std::int64_t> first({pts.size()}, for_overwrite);
    tabulate_basis_unchecked<P>(pts, values.data(), first.data());
    return {std::move(values), std::move(first)};
  }

  /// Evaluate the derivatives of orders `0..n_deriv` of the non-zero basis functions at each point
  /// (Piegl and Tiller A2.3; the Bernstein derivatives scaled by the chain rule for a Bézier-like
  /// space). Slice 0 equals `tabulate_basis`; orders above the degree are zero.
  ///
  /// Args:
  ///     pts (std::span<const P>): The points.
  ///     n_deriv (int): Highest order, `>= 0`.
  ///     out (mdview<P, 3>): Extents `(pts.size(), n_deriv + 1, degree + 1)`.
  ///     first (std::span<std::int64_t>): `pts.size()` entries.
  ///     validate (bool): Refuse points outside the domain. Defaults to true.
  ///
  /// Returns:
  ///     bool: True when every point was evaluated; false when the derivative recurrence met a zero
  ///     divisor, which a basis function of empty support produces (a knot repeated more than
  ///     `degree + 1` times in a point's support). `out` is then incomplete. A2.3 divides there
  ///     unguarded; the previous implementation raised `ZeroDivisionError`.
  ///
  /// Raises:
  ///     value_error: `n_deriv < 0` (first), `validate` and a point outside the domain, `out` or
  ///         `first` of another size.
  template <StorageFloat P>
  [[nodiscard]] bool tabulate_basis_derivatives(std::span<const P> pts, int n_deriv,
                                                mdview<P, 3> out, std::span<std::int64_t> first,
                                                bool validate = true) const {
    check_n_deriv(n_deriv);
    basis::detail::check_out<P, 3>(out, {pts.size(), static_cast<std::size_t>(n_deriv) + 1,
                                         static_cast<std::size_t>(degree()) + 1});
    check_first(first, pts.size());
    if (validate) {
      check_in_domain<P>(pts);
    }
    return tabulate_derivatives_unchecked<P>(pts, n_deriv, out.data_handle(), first.data());
  }

  /// Evaluate the derivatives at each point (see the `out` overload).
  ///
  /// Returns:
  ///     BasisDerivatives<P>: Values `(n, n_deriv + 1, degree + 1)`, first indices `(n,)` and
  ///     whether every point was evaluated.
  template <StorageFloat P>
  [[nodiscard]] BasisDerivatives<P> tabulate_basis_derivatives(std::span<const P> pts, int n_deriv,
                                                               bool validate = true) const {
    check_n_deriv(n_deriv);
    if (validate) {
      check_in_domain<P>(pts);
    }
    NdArray<P> values(
        {pts.size(), static_cast<std::size_t>(n_deriv) + 1, static_cast<std::size_t>(degree()) + 1},
        for_overwrite);
    NdArray<std::int64_t> first({pts.size()}, for_overwrite);
    const bool complete =
        tabulate_derivatives_unchecked<P>(pts, n_deriv, values.data(), first.data());
    return {std::move(values), std::move(first), complete};
  }

  /// The windowed sub-space over the intervals `[lo, hi)`: a slice of this space's knots (never
  /// re-clamped, not snapped again) whose basis functions are exactly this space's functions
  /// supported on those intervals, and the global index of each.
  ///
  /// Args:
  ///     lo (long long): First interval, inclusive.
  ///     hi (long long): One past the last interval, `0 <= lo < hi <= num_intervals()`.
  ///
  /// Returns:
  ///     std::pair<BsplineSpace1D, std::vector<std::int64_t>>: The windowed space and its
  ///     local-to-global function map.
  ///
  /// Raises:
  ///     value_error: The space is periodic, or the range is invalid.
  [[nodiscard]] std::pair<BsplineSpace1D, std::vector<std::int64_t>> restrict(long long lo,
                                                                              long long hi) const {
    if (periodic()) {
      throw value_error("restrict: periodic B-spline spaces are not supported.");
    }
    const long long n_int = num_intervals();
    if (!(0 <= lo && lo < hi && hi <= n_int)) {
      throw value_error(
          "restrict: require 0 <= interval_lo < interval_hi <= " + std::to_string(n_int) +
          "; got [" + std::to_string(lo) + ", " + std::to_string(hi) + ").");
    }
    const std::span<const T> u = unique_knots_and_multiplicity(true).first;
    const auto ulo = static_cast<std::size_t>(lo);
    const auto uhi = static_cast<std::size_t>(hi);
    const T half = static_cast<T>(0.5);
    const T s0 = u[ulo] + u[ulo + 1];
    const T s1 = u[uhi - 1] + u[uhi];
    const T mids[2] = {half * s0, half * s1};
    std::int64_t first[2];
    std::vector<T> values(2 * (static_cast<std::size_t>(degree()) + 1));
    tabulate_basis_unchecked<T>(std::span<const T>(mids, 2), values.data(), first);
    const std::int64_t j_lo = first[0];
    const std::int64_t j_hi = first[1] + degree();
    const auto begin = static_cast<std::size_t>(j_lo);
    const auto end = static_cast<std::size_t>(j_hi + degree() + 2);
    BsplineSpace1D windowed(std::span<const T>(rep_->knots).subspan(begin, end - begin), degree(),
                            false, false);
    std::vector<std::int64_t> map(static_cast<std::size_t>(j_hi - j_lo + 1));
    for (std::size_t i = 0; i < map.size(); ++i) {
      map[i] = j_lo + static_cast<std::int64_t>(i);
    }
    return {std::move(windowed), std::move(map)};
  }

  /// The Greville abscissae, one per basis function (see `bspline::greville_abscissae`).
  [[nodiscard]] std::vector<T> greville() const {
    const auto [a, b] = domain();
    return greville_abscissae<T>(knots(), degree(), static_cast<std::size_t>(num_basis()),
                                 periodic(), a, b);
  }

  /// The shared representation (for a binding's read-only views).
  [[nodiscard]] const std::shared_ptr<const Rep>& rep() const noexcept { return rep_; }

 private:
  explicit BsplineSpace1D(std::shared_ptr<const Rep> rep) : rep_(std::move(rep)) {}

  static void check_n_deriv(int n_deriv) {
    if (n_deriv < 0) {
      throw value_error("n_deriv must be non-negative, got " + std::to_string(n_deriv));
    }
  }

  static void check_first(std::span<std::int64_t> first, std::size_t n) {
    if (first.size() != n) {
      const std::size_t got[1] = {first.size()};
      const std::size_t want[1] = {n};
      throw value_error("Output array has shape " + py_repr(std::span<const std::size_t>(got)) +
                        ", but expected shape " + py_repr(std::span<const std::size_t>(want)));
    }
  }

  /// `(pt - a) / (b - a)` per point, the difference `b - a` formed in `T` and the rest in the
  /// common type, stored in `P` (as the previous implementation's NumPy expression computed it).
  template <StorageFloat P>
  std::vector<P> to_unit(std::span<const P> pts) const {
    using W = std::common_type_t<T, P>;
    const auto [a, b] = domain();
    const T len = b - a;
    std::vector<P> u(pts.size());
    for (std::size_t i = 0; i < pts.size(); ++i) {
      const W shifted = static_cast<W>(pts[i]) - static_cast<W>(a);
      u[i] = static_cast<P>(shifted / static_cast<W>(len));
    }
    return u;
  }

  template <StorageFloat P>
  void tabulate_basis_unchecked(std::span<const P> pts, P* out, std::int64_t* first) const {
    const int p = degree();
    if (rep_->bezier_like) {
      const std::vector<P> u = to_unit<P>(pts);
      const auto ord = static_cast<std::size_t>(p) + 1;
      basis::detail::tabulate_bernstein_1d_kernel<P>(p, std::span<const P>(u),
                                                     mdview<P, 2>(out, pts.size(), ord));
      for (std::size_t i = 0; i < pts.size(); ++i) {
        first[i] = 0;
      }
      return;
    }
    detail::tabulate_basis_kernel<T, P>(knots(), p, pts, out, first);
  }

  template <StorageFloat P>
  [[nodiscard]] bool tabulate_derivatives_unchecked(std::span<const P> pts, int n_deriv, P* out,
                                                    std::int64_t* first) const {
    const int p = degree();
    if (rep_->bezier_like) {
      const std::vector<P> u = to_unit<P>(pts);
      const auto ord = static_cast<std::size_t>(p) + 1;
      const auto nd1 = static_cast<std::size_t>(n_deriv) + 1;
      basis::detail::tabulate_bernstein_derivatives_1d_kernel<P>(
          p, std::span<const P>(u), n_deriv, mdview<P, 3>(out, pts.size(), nd1, ord));
      // Chain rule: the k-th derivative in x is the k-th in s times (1 / (b - a))^k, the power
      // accumulated in double and each factor applied in P.
      const auto [a, b] = domain();
      const T len = b - a;
      const double inv_span = 1.0 / static_cast<double>(len);
      double scale = inv_span;
      for (std::size_t k = 1; k < nd1; ++k) {
        const auto f = static_cast<P>(scale);
        for (std::size_t i = 0; i < pts.size(); ++i) {
          P* row = out + (i * nd1 + k) * ord;
          for (std::size_t j = 0; j < ord; ++j) {
            row[j] = row[j] * f;
          }
        }
        scale *= inv_span;
      }
      for (std::size_t i = 0; i < pts.size(); ++i) {
        first[i] = 0;
      }
      return true;
    }
    return detail::tabulate_derivs_kernel<T, P>(knots(), p, n_deriv, pts, out, first);
  }

  static std::shared_ptr<const Rep> build(std::span<const T> raw, int degree, bool periodic,
                                          bool snap_knots, const double* stored_tol) {
    detail::check_knots_and_degree<T>(raw, degree);
    const double raw_tol = stored_tol != nullptr ? *stored_tol : knot_tolerance<T>(raw);
    if (bspline::num_basis<T>(raw, degree, periodic, raw_tol) < degree + 1LL) {
      throw value_error("Not enough knots for the specified degree");
    }
    auto rep = std::make_shared<Rep>();
    rep->degree = degree;
    rep->periodic = periodic;
    rep->tol = raw_tol;
    rep->knots.assign(raw.begin(), raw.end());
    // One classification serves the snapping and the space: snapping writes each class's first
    // knot over its members, and the snapped vector's classes are the raw vector's (the step
    // between two representatives is at least the step that separated their classes, rounding
    // being monotone, and a class's members become equal), as `unique_knots_and_multiplicity`'s
    // idempotence states.
    KnotClassification<T> cls = classify_knots<T>(raw, degree, rep->tol);
    if (snap_knots) {
      auto it = rep->knots.begin();
      for (std::size_t c = 0; c < cls.classes.knots.size(); ++c) {
        it = std::fill_n(it, cls.classes.multiplicities[c], cls.classes.knots[c]);
      }
      check_snapping_kept_an_interval<T>(raw, rep->knots, rep->tol);
    }
    const std::span<const T> k(rep->knots);
    rep->classes = std::move(cls.classes);
    rep->lo_class = cls.lo_class;
    rep->hi_class = cls.hi_class;
    check_space_has_an_interval<T>(k, degree, static_cast<long long>(rep->hi_class - rep->lo_class),
                                   rep->tol);
    if (degree > 0 && rep->classes.multiplicities.back() > degree + 1LL) {
      throw value_error("the last knot may repeat at most " + std::to_string(degree + 1) +
                        " times");
    }
    rep->num_basis = bspline::num_basis<T>(k, degree, periodic, rep->tol);
    // The functions non-zero on the span starting at knot index p are p - degree .. p, p the last
    // position of the span's starting knot; the classes whose last position lies in
    // [degree, num_basis) start exactly the in-domain intervals.
    std::int64_t last = -1;
    for (const std::int64_t m : rep->classes.multiplicities) {
      last += m;
      if (last >= degree && last < rep->num_basis) {
        rep->first_basis.push_back(last - degree);
      }
    }
    const auto d = static_cast<std::size_t>(degree);
    if (!periodic) {
      rep->left_open = std::abs(static_cast<double>(k[0]) - static_cast<double>(k[d])) <= rep->tol;
      rep->right_open = std::abs(static_cast<double>(k[k.size() - d - 1]) -
                                 static_cast<double>(k.back())) <= rep->tol;
    }
    rep->bezier_like = !periodic && rep->left_open && rep->right_open &&
                       rep->num_basis == static_cast<long long>(degree) + 1;
    return rep;
  }

  std::shared_ptr<const Rep> rep_;
};

}  // namespace pantr::bspline
