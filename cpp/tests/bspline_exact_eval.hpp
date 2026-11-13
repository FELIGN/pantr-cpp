// The exact value of a `Bspline` at a point, from the recursive definition of the basis in
// bspline_oracle.hpp (not the triangle the library runs): the sum over the supported basis
// functions of the control points, in exact rational arithmetic. A tensor product of any
// dimension; a periodic direction reads its stored control points modulo their number (the ghost
// functions repeat the first ones), on the full knot vector with its ghost knots.
//
// The result carries every stored coordinate, the homogeneous weight of a rational spline
// included, so two splines compare in homogeneous coordinates (which is where elevation, reduction
// and knot removal work).
#pragma once

#include "bspline_oracle.hpp"

#include <pantr/bspline/bspline.hpp>
#include <pantr/core/scalar.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace bspline_oracle {

/// The exact value (all `cp_size` coordinates) of `b`, or of one of its partial derivatives, at the
/// parameters `x`, one per direction.
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///     x (std::span<const Rational>): The point, `dim` exact coordinates inside the domain.
///     orders (std::span<const int>): The derivative order per direction, `dim` values, or none for
///         the value. The derivative is that of the polynomial piece of the span `x` is evaluated
///         on, the one the oracle's span rule picks.
///
/// Returns:
///     std::vector<Rational>: `sum_i P_i D^orders N_i(x)` per coordinate (of the homogeneous
///     coordinates, for a rational spline).
template <pantr::StorageFloat T>
std::vector<Rational> exact_value(const pantr::bspline::Bspline<T>& b, std::span<const Rational> x,
                                  std::span<const int> orders = {}) {
  const auto spaces = b.space().spaces();
  const std::size_t dim = spaces.size();
  const std::size_t cols = b.cp_size();
  const std::span<const std::size_t> shape = b.shape();

  // The supported functions of each direction: first index, and the exact weights.
  std::vector<std::size_t> first(dim);
  std::vector<std::vector<Rational>> weights(dim);
  for (std::size_t d = 0; d < dim; ++d) {
    const auto p = static_cast<std::size_t>(spaces[d].degree());
    BasisAt at(exact_knots<T>(spaces[d].knots()), spaces[d].degree(), x[d]);
    first[d] = at.span() - p;
    for (std::size_t j = 0; j <= p; ++j) {
      weights[d].push_back(at.deriv(first[d] + j, orders.empty() ? 0 : orders[d]));
    }
  }

  // Row-major strides of the control net over the stored numbers of functions.
  std::vector<std::size_t> stride(dim);
  std::size_t s = cols;
  for (std::size_t d = dim; d-- > 0;) {
    stride[d] = s;
    s *= shape[d];
  }

  std::vector<Rational> out(cols, Rational(0));
  std::vector<std::size_t> j(dim, 0);
  const auto ctrl = b.control_points();
  for (;;) {
    Rational w = 1;
    std::size_t offset = 0;
    for (std::size_t d = 0; d < dim; ++d) {
      w *= weights[d][j[d]];
      offset += ((first[d] + j[d]) % shape[d]) * stride[d];
    }
    if (w != 0) {
      for (std::size_t c = 0; c < cols; ++c) {
        out[c] += w * exact(static_cast<double>(ctrl[offset + c]));
      }
    }
    std::size_t d = dim;
    while (d-- > 0) {
      if (++j[d] <= static_cast<std::size_t>(spaces[d].degree())) {
        break;
      }
      j[d] = 0;
      if (d == 0) {
        return out;
      }
    }
  }
}

/// The largest absolute value among the stored control points, as a double.
template <pantr::StorageFloat T>
double max_abs_control_point(const pantr::bspline::Bspline<T>& b) {
  double m = 0.0;
  for (const T v : b.control_points()) {
    m = (std::max)(m, std::abs(static_cast<double>(v)));
  }
  return m;
}

}  // namespace bspline_oracle
