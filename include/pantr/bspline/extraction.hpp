/// @file extraction.hpp
/// @brief Bézier, Lagrange and cardinal extraction operators of a 1D B-spline space, and the
///        mask of the elements whose Bézier operator is structurally the identity.
///
/// For each interval `i` the Bézier extraction operator `C_i` satisfies `N_i(x) = C_i B(xi)`: `N_i`
/// are the `degree + 1` B-spline basis functions that are nonzero on the interval, `B` the
/// Bernstein basis of the reference interval `[0, 1]`. The Lagrange and cardinal operators map the
/// Lagrange or cardinal B-spline basis of the reference element the same way, so they are the
/// Bézier operator times a change-of-basis matrix.
///
/// Each operator comes as a raw entry point on a knot vector, a degree and an absolute tolerance
/// (the knots are validated, the output shape is checked), and as one on a `BsplineSpace1D`, whose
/// knots, degree and tolerance are already validated and whose knot classes are already known.
#pragma once

#include "pantr/basis/detail/tabulate_1d.hpp"
#include "pantr/basis/lagrange_variant.hpp"
#include "pantr/bspline/detail/mode_product.hpp"
#include "pantr/bspline/knots.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/change_basis.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace pantr::bspline {

namespace detail {

/// The Bézier extraction operators, by the knot insertion that makes each interval Bézier.
///
/// For the first interval of an unclamped knot vector the missing end multiplicity is inserted at
/// the domain start; then, interval by interval, the knots of each interval's boundary are
/// inserted up to multiplicity `degree` and the columns that the next interval shares are carried
/// over. Every operation is in `T`.
///
/// Args:
///     knots (std::span<const T>): The knot vector.
///     degree (int): The degree.
///     tol (double): Absolute parametric tolerance.
///     mults (std::span<const std::int64_t>): Multiplicities of the in-domain knot classes, one
///         more than the intervals.
///     out (T*): `(mults.size() - 1) * (degree + 1)^2` values, written.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed). The knot vector must hold the
///     domain: at least `2 * degree + 2` knots.
template <StorageFloat T>
void bezier_extraction_core(std::span<const T> knots, int degree, double tol,
                            std::span<const std::int64_t> mults, T* out) {
  const auto p = static_cast<std::ptrdiff_t>(degree);
  const auto order = static_cast<std::size_t>(degree) + 1;
  const std::size_t n_elems = mults.size() - 1;
  const T one = static_cast<T>(1);
  std::fill(out, out + n_elems * order * order, static_cast<T>(0));
  // An identity per interval; element `e` is the `order x order` block at `out + e * order^2`.
  for (std::size_t e = 0; e < n_elems; ++e) {
    for (std::size_t i = 0; i < order; ++i) {
      out[e * order * order + i * order + i] = one;
    }
  }
  const auto col = [&](std::size_t e, std::ptrdiff_t r, std::ptrdiff_t c) -> T& {
    return out[e * order * order + static_cast<std::size_t>(r) * order +
               static_cast<std::size_t>(c)];
  };

  // A knot vector whose first domain knot is not repeated degree + 1 times needs knot insertion at
  // the start before the first interval is Bézier.
  const int first_mult = multiplicity_of_first_knot_in_domain<T>(knots, degree, tol);
  if (first_mult < degree + 1) {
    const std::ptrdiff_t reg = p - first_mult;
    const T t = knots[static_cast<std::size_t>(p)];
    for (std::ptrdiff_t r = 0; r < reg; ++r) {
      const T* lcl = knots.data() + r;
      for (std::ptrdiff_t k = 1; k < p - r; ++k) {
        const T alpha = (t - lcl[k]) / (lcl[k + p - r] - lcl[k]);
        for (std::ptrdiff_t row = 0; row < p + 1; ++row) {
          col(0, row, k - 1) = alpha * col(0, row, k) + (one - alpha) * col(0, row, k - 1);
        }
      }
    }
  }

  std::vector<T> alphas(static_cast<std::size_t>((std::max)(p - 1, std::ptrdiff_t{0})));
  std::ptrdiff_t knt_id = p;
  std::ptrdiff_t mult = 0;
  for (std::size_t e = 0; e < n_elems; ++e) {
    knt_id += mult;
    mult = static_cast<std::ptrdiff_t>(mults[e + 1]);
    if (mult >= p) {
      continue;
    }
    const T* lcl = knots.data() + knt_id;
    for (std::ptrdiff_t j = 0; j < p - mult; ++j) {
      alphas[static_cast<std::size_t>(j)] = (lcl[1] - lcl[0]) / (lcl[mult + 1 + j] - lcl[0]);
    }
    const std::ptrdiff_t reg = p - mult;
    for (std::ptrdiff_t r = 1; r <= reg; ++r) {
      const std::ptrdiff_t s = mult + r;
      for (std::ptrdiff_t k = p; k >= s; --k) {
        const T alpha = alphas[static_cast<std::size_t>(k - s)];
        for (std::ptrdiff_t row = 0; row < p + 1; ++row) {
          col(e, row, k) = alpha * col(e, row, k) + (one - alpha) * col(e, row, k - 1);
        }
      }
      if (e + 1 < n_elems) {
        for (std::ptrdiff_t q = 0; q <= r; ++q) {
          col(e + 1, reg - r + q, reg - r) = col(e, p - r + q, p);
        }
      }
    }
  }
}

/// Refuse a tolerance that is not a finite non-negative number, and a knot vector and degree that
/// define no space.
///
/// Raises:
///     value_error: `tol < 0` ("tol must be non-negative"); `tol` NaN or infinite ("tol must be
///         finite"; a NaN passes every comparison, so it is named rather than tested by `<`); or
///         as `check_knots_and_degree`.
template <StorageFloat T>
void check_extraction_input(std::span<const T> knots, int degree, double tol) {
  if (tol < 0) {
    throw value_error("tol must be non-negative");
  }
  if (!std::isfinite(tol)) {
    throw value_error("tol must be finite, got " + py_repr(tol));
  }
  check_knots_and_degree<T>(knots, degree);
}

/// Validate a raw operator request and group its knots: the classes of the in-domain knots.
///
/// The core writes the start-insertion block and the per-interval identity into an output of
/// `classes - 1` intervals, so a tolerance that merges the whole domain into one class (zero
/// intervals) is refused here, before any output is sized or written.
///
/// Args:
///     knots (std::span<const T>): The knot vector.
///     degree (int): The degree.
///     tol (double): Absolute parametric tolerance.
///
/// Returns:
///     KnotClasses<T>: The in-domain classes, at least two.
///
/// Raises:
///     value_error: As `check_extraction_input`; or the classes are fewer than two ("knot vector
///         spans no interval ...", as `BsplineSpace1D` refuses the same vector).
template <StorageFloat T>
[[nodiscard]] KnotClasses<T> classify_for_extraction(std::span<const T> knots, int degree,
                                                     double tol) {
  check_extraction_input<T>(knots, degree, tol);
  KnotClasses<T> cls = unique_knots_and_multiplicity<T>(knots, degree, tol, true);
  check_space_has_an_interval<T>(knots, degree,
                                 static_cast<long long>(cls.multiplicities.size()) - 1, tol);
  return cls;
}

/// The extents `(intervals, degree + 1, degree + 1)` an operator array has.
[[nodiscard]] inline std::array<std::size_t, 3> extraction_extents(std::size_t num_intervals,
                                                                   int degree) {
  const auto order = static_cast<std::size_t>(degree) + 1;
  return {num_intervals, order, order};
}

/// Write the product `out[i] <- out[i] M` for every interval, `M` the change-of-basis matrix
/// `(order x order)` of the target basis (a batched matrix product on the right; the Bézier
/// operator times `M` expresses the operator on the target basis).
///
/// Args:
///     out (mdview<T, 3>): The Bézier operators `(n, order, order)`; overwritten.
///     matrix (const NdArray<T>&): `M`.
template <StorageFloat T>
void right_multiply_in_place(mdview<T, 3> out, const NdArray<T>& matrix) {
  const std::size_t n = out.extent(0);
  const std::size_t order = out.extent(1);
  const std::size_t total = n * order * order;
  const std::vector<T> source(out.data_handle(), out.data_handle() + total);
  const std::size_t extents[3] = {n, order, order};
  // out[i][r][c] = sum_k source[i][r][k] * M[k][c]: the transpose of M applied along the last mode.
  const ModeOperator<T> op{matrix.flat(), order, order, true};
  mode_product_whole<T>(
      op, {std::span<const T>(source), std::span<const std::size_t>(extents)}, 2,
      {std::span<T>(out.data_handle(), total), std::span<const std::size_t>(extents)});
}

/// Refuse a mask output that does not hold one flag per interval.
///
/// Raises:
///     value_error: `out.size() != n` ("Output array has shape ..., but expected shape ...").
inline void check_mask_size(std::size_t n, std::span<const bool> out) {
  if (out.size() != n) {
    const std::size_t got[1] = {out.size()};
    const std::size_t want[1] = {n};
    throw value_error("Output array has shape " + py_repr(std::span<const std::size_t>(got)) +
                      ", but expected shape " + py_repr(std::span<const std::size_t>(want)));
  }
}

/// Overwrite with the identity the operators of the flagged intervals.
///
/// Args:
///     operators (T*): `flags.size() * order * order` values; the blocks of the flagged intervals
///         are overwritten.
///     order (std::size_t): `degree + 1`.
///     flags (std::span<const bool>): One flag per interval.
template <StorageFloat T>
void set_identity_where(T* operators, std::size_t order, std::span<const bool> flags) {
  for (std::size_t e = 0; e < flags.size(); ++e) {
    if (!flags[e]) {
      continue;
    }
    T* block = operators + e * order * order;
    std::fill(block, block + order * order, static_cast<T>(0));
    for (std::size_t i = 0; i < order; ++i) {
      block[i * order + i] = static_cast<T>(1);
    }
  }
}

}  // namespace detail

/// Compute the Bézier extraction operators of a knot vector.
///
/// Args:
///     knots (std::span<const T>): The knot vector, non-decreasing, at least `2 * degree + 2`.
///     degree (int): The degree, `>= 0`.
///     tol (double): Absolute parametric tolerance (two knots closer are one knot), `>= 0`.
///     out (mdview<T, 3>): Extents `(intervals, degree + 1, degree + 1)`; written.
///
/// Raises:
///     value_error: `tol` is negative, NaN or infinite; the knots or the degree define no space;
///         `tol` merges the whole domain into one knot (no interval); `out` has other extents.
template <StorageFloat T>
void bezier_extraction_operators(std::span<const T> knots, int degree, double tol,
                                 mdview<T, 3> out) {
  const KnotClasses<T> cls = detail::classify_for_extraction<T>(knots, degree, tol);
  basis::detail::check_out<T, 3>(out,
                                 detail::extraction_extents(cls.multiplicities.size() - 1, degree));
  detail::bezier_extraction_core<T>(knots, degree, tol, cls.multiplicities, out.data_handle());
}

/// Compute the Bézier extraction operators of a knot vector (see the overload writing into `out`).
///
/// Returns:
///     NdArray<T>: Extents `(intervals, degree + 1, degree + 1)`.
template <StorageFloat T>
[[nodiscard]] NdArray<T> bezier_extraction_operators(std::span<const T> knots, int degree,
                                                     double tol) {
  const KnotClasses<T> cls = detail::classify_for_extraction<T>(knots, degree, tol);
  const auto ext = detail::extraction_extents(cls.multiplicities.size() - 1, degree);
  NdArray<T> out(std::vector<std::size_t>(ext.begin(), ext.end()), for_overwrite);
  detail::bezier_extraction_core<T>(knots, degree, tol, cls.multiplicities, out.data());
  return out;
}

/// Compute the Bézier extraction operators of a space.
///
/// Args:
///     space (const BsplineSpace1D<T>&): The space.
///     out (mdview<T, 3>): Extents `(intervals, degree + 1, degree + 1)`; written.
///
/// Raises:
///     value_error: `out` has other extents.
template <StorageFloat T>
void bezier_extraction_operators(const BsplineSpace1D<T>& space, mdview<T, 3> out) {
  basis::detail::check_out<T, 3>(
      out,
      detail::extraction_extents(static_cast<std::size_t>(space.num_intervals()), space.degree()));
  detail::bezier_extraction_core<T>(space.knots(), space.degree(), space.tolerance(),
                                    space.unique_knots_and_multiplicity(true).second,
                                    out.data_handle());
}

/// Compute the Bézier extraction operators of a space (see the overload writing into `out`).
///
/// Returns:
///     NdArray<T>: Extents `(intervals, degree + 1, degree + 1)`.
template <StorageFloat T>
[[nodiscard]] NdArray<T> bezier_extraction_operators(const BsplineSpace1D<T>& space) {
  const auto ext =
      detail::extraction_extents(static_cast<std::size_t>(space.num_intervals()), space.degree());
  NdArray<T> out(std::vector<std::size_t>(ext.begin(), ext.end()), for_overwrite);
  bezier_extraction_operators<T>(space, out.template view<3>());
  return out;
}

/// Compute the Lagrange extraction operators of a knot vector: `C_i` with `C_i [Lagrange values] =
/// [B-spline values on interval i]`, the Bézier operator times the Lagrange-to-Bernstein matrix.
///
/// Args:
///     knots (std::span<const T>): The knot vector.
///     degree (int): The degree.
///     tol (double): Absolute parametric tolerance, `>= 0`.
///     variant (basis::LagrangeVariant): The Lagrange node distribution.
///     out (mdview<T, 3>): Extents `(intervals, degree + 1, degree + 1)`; written.
///
/// Raises:
///     value_error: As `bezier_extraction_operators`; or `degree` is beyond the degrees the
///         Lagrange-to-Bernstein matrix of `variant` can be built for (checked after the output
///         extents, before anything is written).
template <StorageFloat T>
void lagrange_extraction_operators(std::span<const T> knots, int degree, double tol,
                                   basis::LagrangeVariant variant, mdview<T, 3> out) {
  const KnotClasses<T> cls = detail::classify_for_extraction<T>(knots, degree, tol);
  basis::detail::check_out<T, 3>(out,
                                 detail::extraction_extents(cls.multiplicities.size() - 1, degree));
  const auto matrix =
      change_basis::cached<T>(change_basis::Kind::lagrange_to_bernstein, degree, variant);
  detail::bezier_extraction_core<T>(knots, degree, tol, cls.multiplicities, out.data_handle());
  detail::right_multiply_in_place<T>(out, *matrix);
}

/// Compute the Lagrange extraction operators of a space (see the raw overload).
///
/// Args:
///     space (const BsplineSpace1D<T>&): The space.
///     variant (basis::LagrangeVariant): The Lagrange node distribution.
///     out (mdview<T, 3>): Extents `(intervals, degree + 1, degree + 1)`; written.
template <StorageFloat T>
void lagrange_extraction_operators(const BsplineSpace1D<T>& space, basis::LagrangeVariant variant,
                                   mdview<T, 3> out) {
  basis::detail::check_out<T, 3>(
      out,
      detail::extraction_extents(static_cast<std::size_t>(space.num_intervals()), space.degree()));
  const auto matrix =
      change_basis::cached<T>(change_basis::Kind::lagrange_to_bernstein, space.degree(), variant);
  bezier_extraction_operators<T>(space, out);
  detail::right_multiply_in_place<T>(out, *matrix);
}

/// Compute the cardinal B-spline extraction operators of a knot vector: the Bézier operator times
/// the cardinal-to-Bernstein matrix, except on the cardinal intervals (equal knot spacing over the
/// whole support window), where the operator is exactly the identity.
///
/// Args:
///     knots (std::span<const T>): The knot vector.
///     degree (int): The degree.
///     tol (double): Absolute parametric tolerance, `>= 0`.
///     out (mdview<T, 3>): Extents `(intervals, degree + 1, degree + 1)`; written.
///
/// Raises:
///     value_error: As `lagrange_extraction_operators`, for the cardinal-to-Bernstein matrix.
template <StorageFloat T>
void cardinal_extraction_operators(std::span<const T> knots, int degree, double tol,
                                   mdview<T, 3> out) {
  const KnotClasses<T> cls = detail::classify_for_extraction<T>(knots, degree, tol);
  const std::size_t n_elems = cls.multiplicities.size() - 1;
  basis::detail::check_out<T, 3>(out, detail::extraction_extents(n_elems, degree));
  const auto matrix = change_basis::cached<T>(change_basis::Kind::cardinal_to_bernstein, degree,
                                              basis::LagrangeVariant::equispaces);
  detail::bezier_extraction_core<T>(knots, degree, tol, cls.multiplicities, out.data_handle());
  detail::right_multiply_in_place<T>(out, *matrix);
  // `std::vector<bool>` is not a span of `bool`, so the flags live in a plain array.
  const std::unique_ptr<bool[]> flags(new bool[n_elems]);
  cardinal_intervals<T>(knots, degree, tol, std::span<bool>(flags.get(), n_elems));
  detail::set_identity_where<T>(out.data_handle(), static_cast<std::size_t>(degree) + 1,
                                std::span<const bool>(flags.get(), n_elems));
}

/// Compute the cardinal extraction operators of a space (see the raw overload).
///
/// Args:
///     space (const BsplineSpace1D<T>&): The space.
///     out (mdview<T, 3>): Extents `(intervals, degree + 1, degree + 1)`; written.
///
/// Raises:
///     value_error: `out` has other extents, or the degree is beyond the cardinal-to-Bernstein
///         matrix (checked before anything is written).
template <StorageFloat T>
void cardinal_extraction_operators(const BsplineSpace1D<T>& space, mdview<T, 3> out) {
  const auto n_elems = static_cast<std::size_t>(space.num_intervals());
  basis::detail::check_out<T, 3>(out, detail::extraction_extents(n_elems, space.degree()));
  const auto matrix = change_basis::cached<T>(change_basis::Kind::cardinal_to_bernstein,
                                              space.degree(), basis::LagrangeVariant::equispaces);
  // The space's own classes, as the Bézier overload reads them, and its own cardinal flags.
  detail::bezier_extraction_core<T>(space.knots(), space.degree(), space.tolerance(),
                                    space.unique_knots_and_multiplicity(true).second,
                                    out.data_handle());
  detail::right_multiply_in_place<T>(out, *matrix);
  const std::unique_ptr<bool[]> flags(new bool[n_elems]);
  space.cardinal_intervals(std::span<bool>(flags.get(), n_elems));
  detail::set_identity_where<T>(out.data_handle(), static_cast<std::size_t>(space.degree()) + 1,
                                std::span<const bool>(flags.get(), n_elems));
}

/// Flag the intervals whose Bézier extraction operator is structurally the identity: both bounding
/// knots have multiplicity `degree + 1` or more, so the interval is already a Bézier patch.
///
/// Args:
///     space (const BsplineSpace1D<T>&): The space.
///     out (std::span<bool>): One flag per interval.
///
/// Raises:
///     value_error: `out` does not hold `num_intervals()` flags.
template <StorageFloat T>
void bezier_structural_identity_mask(const BsplineSpace1D<T>& space, std::span<bool> out) {
  const std::span<const std::int64_t> mult = space.unique_knots_and_multiplicity(true).second;
  const std::size_t n = mult.size() - 1;
  detail::check_mask_size(n, out);
  const std::int64_t threshold = static_cast<std::int64_t>(space.degree()) + 1;
  for (std::size_t e = 0; e < n; ++e) {
    out[e] = mult[e] >= threshold && mult[e + 1] >= threshold;
  }
}

/// Flag the intervals whose Lagrange extraction operator is structurally the identity.
///
/// A degree-0 space has the 1x1 operator `[[1]]` on every interval. Otherwise the operator is the
/// Bézier one times the Lagrange-to-Bernstein matrix, so it is the identity exactly where the
/// Bézier operator is and that matrix is the identity (the Lagrange nodes are the Bernstein
/// abscissae `i / degree`: degree 1 with equispaced, Gauss-Lobatto or Chebyshev-Lobatto nodes);
/// for any other matrix no interval is flagged.
///
/// Args:
///     space (const BsplineSpace1D<T>&): The space.
///     variant (basis::LagrangeVariant): The Lagrange node distribution.
///     out (std::span<bool>): One flag per interval.
///
/// Raises:
///     value_error: `out` does not hold `num_intervals()` flags; or, for a degree above 0, the
///         degree is beyond the degrees the Lagrange-to-Bernstein matrix of `variant` can be built
///         for.
template <StorageFloat T>
void lagrange_structural_identity_mask(const BsplineSpace1D<T>& space,
                                       basis::LagrangeVariant variant, std::span<bool> out) {
  detail::check_mask_size(static_cast<std::size_t>(space.num_intervals()), out);
  if (space.degree() == 0) {
    std::fill(out.begin(), out.end(), true);
    return;
  }
  const auto matrix =
      change_basis::cached<T>(change_basis::Kind::lagrange_to_bernstein, space.degree(), variant);
  const auto order = static_cast<std::size_t>(space.degree()) + 1;
  const T* entries = matrix->data();
  bool is_identity = true;
  for (std::size_t i = 0; i < order; ++i) {
    for (std::size_t j = 0; j < order; ++j) {
      is_identity = is_identity && entries[i * order + j] == (i == j ? T{1} : T{0});
    }
  }
  if (is_identity) {
    bezier_structural_identity_mask<T>(space, out);
  } else {
    std::fill(out.begin(), out.end(), false);
  }
}

}  // namespace pantr::bspline
