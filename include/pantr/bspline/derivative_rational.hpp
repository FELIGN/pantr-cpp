/// @file derivative_rational.hpp
/// @brief The partial derivative of a rational B-spline, by the quotient rule over `multiply`.
///
/// For `f = A / w`, `f' = (A' w - A w') / w^2`. `A'` and `w'` are taken with the degree kept
/// (`derivative_keep_degree_nonrational`), so `A' w`, `A w'` and `w^2` all have degree `2p` in the
/// direction and the numerator needs no elevation of its own. The numerator and `w^2` are then
/// refined to a common knot vector in the direction (`refine_to_common_space_1d`).
///
/// A periodic direction is differentiated in open form and closed again: the seam multiplicity the
/// products need does not fit a periodic knot vector. Continuity is what is carried across the
/// seam, not multiplicity, because the degree changes: a seam of multiplicity `m` at degree `p` is
/// `C^(p - m)`; the numerator carries one differentiation (`C^(p - m - 1)`) and `w^2` none, so at
/// degree `2p` the common seam has multiplicity `2p - (p - m - 1) = p + m + 1`, at most `2p`
/// because a `C^0` seam (`m == p`) is refused first: the derivative is discontinuous there.
#pragma once

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/derivative.hpp"
#include "pantr/bspline/detail/refine_common.hpp"
#include "pantr/bspline/knot_insertion.hpp"
#include "pantr/bspline/knots.hpp"
#include "pantr/bspline/periodic.hpp"
#include "pantr/bspline/product.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bspline {

namespace detail {

/// Repeat the single column of a scalar B-spline `cols` times.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] Bspline<T> tile_columns(const Bspline<T>& w, std::size_t cols) {
  const std::span<const T> src = w.control_points();
  std::vector<std::size_t> shape(w.shape().begin(), w.shape().end() - 1);
  shape.push_back(cols);
  NdArray<T> out(shape, for_overwrite);
  for (std::size_t i = 0; i < src.size(); ++i) {
    std::fill_n(out.data() + i * cols, cols, src[i]);
  }
  return Bspline<T>(w.space(), std::move(out), false);
}

/// The quotient rule on a rational B-spline whose `direction` is not periodic.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] Bspline<T> derivative_rational_open(const Bspline<T>& b, long long direction) {
  const std::size_t rank = b.cp_size() - 1;
  const Bspline<T> a = select_columns<T>(b, 0, rank);
  const Bspline<T> w = select_columns<T>(b, rank, 1);
  const Bspline<T> a_prime = derivative_keep_degree_nonrational<T>(a, direction);
  const Bspline<T> w_prime = derivative_keep_degree_nonrational<T>(w, direction);

  const Bspline<T> num1 = multiply<T>(a_prime, tile_columns<T>(w, rank));
  const Bspline<T> num2 = multiply<T>(a, tile_columns<T>(w_prime, rank));
  // The two products have the same space: the operands' spaces agree pairwise.
  const std::span<const T> c1 = num1.control_points();
  const std::span<const T> c2 = num2.control_points();
  NdArray<T> diff(std::vector<std::size_t>(num1.shape().begin(), num1.shape().end()),
                  for_overwrite);
  for (std::size_t i = 0; i < c1.size(); ++i) {
    diff.data()[i] = c1[i] - c2[i];
  }
  const Bspline<T> numerator(num1.space(), std::move(diff), false);
  const Bspline<T> denominator = multiply<T>(w, w);
  auto [n, d] =
      refine_to_common_space_1d<T>(numerator, denominator, static_cast<std::size_t>(direction));
  return join_columns<T>(n, d, true);
}

}  // namespace detail

/// Compute the partial derivative of a rational B-spline.
///
/// Args:
///     b (const Bspline<T>&): A rational B-spline.
///     direction (long long): The parametric direction, in `[0, dim)`, of degree at least 1.
///
/// Returns:
///     Bspline<T>: The rational derivative, of degree `2p` in `direction` (the other degrees
///     doubled as well, `multiply` adding the operands' degrees in every direction), periodic in
///     `direction` exactly when `b` is.
///
/// Raises:
///     value_error: `b` is not rational; the direction is out of range or has degree 0; the
///         direction is periodic with a `C^0` seam (seam multiplicity equal to the degree); or as
///         `derivative_keep_degree_nonrational` (the degree envelope of the binomial kernel, an
///         unclamped direction, a `C^-1` interior knot).
template <StorageFloat T>
[[nodiscard]] Bspline<T> derivative_rational(const Bspline<T>& b, long long direction) {
  if (!b.rational()) {
    throw value_error("derivative_rational is for rational B-splines.");
  }
  if (direction < 0 || direction >= static_cast<long long>(b.dim())) {
    throw value_error("direction must be in [0, " + std::to_string(b.dim()) + "), got " +
                      std::to_string(direction) + ".");
  }
  const auto dir = static_cast<std::size_t>(direction);
  const BsplineSpace<T> whole = b.space();
  const BsplineSpace1D<T>& s = whole.spaces()[dir];
  const int p = s.degree();
  if (p < 1) {
    throw value_error("Derivative of a degree-0 B-spline is not defined.");
  }
  if (!s.periodic()) {
    return detail::derivative_rational_open<T>(b, direction);
  }

  const double tol = s.tolerance();
  const std::int64_t m_bdy =
      unique_knots_and_multiplicity<T>(s.knots(), p, tol, true).multiplicities[0];
  if (m_bdy >= p) {
    throw value_error(
        "Cannot differentiate a periodic rational B-spline whose seam is C^0 (seam multiplicity " +
        std::to_string(m_bdy) + " equals the degree " + std::to_string(p) +
        "): its derivative is discontinuous at the seam and has no periodic representation.");
  }
  const Bspline<T> open = detail::map_directions<T>(
      b, [dir](std::size_t d) { return d == dir; },
      [&](std::size_t /*d*/, const BsplineSpace1D<T>& space,
          mdview<const T, 2> flat) -> std::pair<BsplineSpace1D<T>, NdArray<T>> {
        KnotsAndCtrl<T> o = to_open_1d<T>(space.knots(), p, flat, true, tol);
        return {BsplineSpace1D<T>(std::span<const T>(o.knots), p), std::move(o.ctrl)};
      });
  const Bspline<T> result = detail::derivative_rational_open<T>(open, direction);
  const long long m_bdy_new = static_cast<long long>(p) + m_bdy + 1;
  return detail::map_directions<T>(
      result, [dir](std::size_t d) { return d == dir; },
      [&](std::size_t /*d*/, const BsplineSpace1D<T>& space,
          mdview<const T, 2> flat) -> std::pair<BsplineSpace1D<T>, NdArray<T>> {
        KnotsAndCtrl<T> per =
            to_periodic_1d<T>(space.knots(), space.degree(), flat, m_bdy_new, space.tolerance());
        return {BsplineSpace1D<T>(std::span<const T>(per.knots), space.degree(), true),
                std::move(per.ctrl)};
      });
}

}  // namespace pantr::bspline
