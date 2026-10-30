/// @file refine_common.hpp
/// @brief What the knot-insertion, periodic-conversion, restriction and splitting operations share:
/// the result of a one-direction operation, the per-direction loop over an N-D control net, and
/// small text and reduction helpers.
#pragma once

#include "pantr/bezier/detail/along_axis.hpp"
#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bspline {

/// A knot vector and the control points over it, as a one-direction operation returns them.
///
/// Attributes:
///     knots (std::vector<T>): The knot vector.
///     ctrl (NdArray<T>): Control points, extents `(num_basis, cols)`.
template <StorageFloat T>
struct KnotsAndCtrl {
  std::vector<T> knots;
  NdArray<T> ctrl;
};

namespace detail {

/// Spell a storage-type value as NumPy spells an element of an array of that dtype in a message:
/// the shortest decimal string that reads back to the same value in that type.
///
/// Args:
///     v (T): The value.
///
/// Returns:
///     std::string: The Python-style spelling of that shortest string.
template <StorageFloat T>
[[nodiscard]] std::string repr_of(T v) {
  if constexpr (std::same_as<T, float>) {
    return py_repr_f32(v);
  } else {
    return py_repr(v);
  }
}

/// Get the largest multiplicity of a knot vector, without building the classes.
///
/// Groups knots exactly as `classify_knots` does (a class ends where the step to the next knot,
/// formed in `T`, exceeds `tol`) and returns the size of the largest group: the same number as
/// the maximum of `unique_knots_and_multiplicity(knots, degree, tol, false).multiplicities`, in one
/// pass and without allocating.
///
/// Args:
///     knots (std::span<const T>): Non-decreasing knot vector.
///     tol (double): Absolute parametric tolerance.
///
/// Returns:
///     std::int64_t: The largest group size, 0 for an empty vector.
template <StorageFloat T>
[[nodiscard]] std::int64_t max_multiplicity(std::span<const T> knots, double tol) {
  std::int64_t best = 0;
  std::int64_t run = 0;
  for (std::size_t i = 0; i < knots.size(); ++i) {
    if (i > 0 && static_cast<double>(static_cast<T>(knots[i] - knots[i - 1])) > tol) {
      run = 0;
    }
    ++run;
    best = (std::max)(best, run);
  }
  return best;
}

/// The largest `|x|` of a range, propagating a NaN as `numpy.max` does.
///
/// Args:
///     values (std::span<const T>): The values, any length.
///
/// Returns:
///     double: Zero for an empty range; NaN when any value is NaN; else the largest magnitude.
template <class T>
[[nodiscard]] double max_abs_propagating(std::span<const T> values) {
  double result = 0.0;
  for (const T v : values) {
    const double a = std::abs(static_cast<double>(v));
    if (std::isnan(a)) {
      return a;
    }
    if (a > result) {
      result = a;
    }
  }
  return result;
}

/// Check that a control net matches a knot vector and degree in one direction.
///
/// Args:
///     what (const char*): The operation, for the message.
///     rows (std::size_t): The control-point rows.
///     num_knots (std::size_t): The knots.
///     degree (int): The degree.
///
/// Raises:
///     value_error: `degree < 0`; or `rows != num_knots - degree - 1` (or fewer than `degree + 2`
///         knots).
inline void check_net_matches(const char* what, std::size_t rows, std::size_t num_knots,
                              int degree) {
  if (degree < 0) {
    throw value_error("degree must be non-negative");
  }
  const auto d = static_cast<std::size_t>(degree);
  if (num_knots < d + 2 || rows != num_knots - d - 1) {
    throw value_error(std::string(what) + ": " + std::to_string(rows) +
                      " control-point rows do not match " + std::to_string(num_knots) +
                      " knots at degree " + std::to_string(degree) + ".");
  }
}

/// Check that a knot vector can hold a domain at a degree, before any index derived from them.
///
/// The domain endpoints sit at `knots[degree]` and `knots[size - degree - 1]`, which are in range
/// (and ordered) exactly when `size >= 2 * degree + 2`.
///
/// Args:
///     num_knots (std::size_t): The knots.
///     degree (int): The degree.
///
/// Raises:
///     value_error: `degree < 0`, or fewer than `2 * degree + 2` knots (the messages
///         `BsplineSpace1D` gives).
inline void check_domain_indexable(std::size_t num_knots, int degree) {
  if (degree < 0) {
    throw value_error("degree must be non-negative");
  }
  if (num_knots / 2 < static_cast<std::size_t>(degree) + 1) {
    throw value_error("knots must have at least 2*degree+2 elements");
  }
}

/// Check the arguments of an Oslo kernel: a degree and two knot vectors that can hold it.
///
/// Args:
///     degree (int): The degree.
///     num_old (std::size_t): The knots of the original vector.
///     num_new (std::size_t): The knots of the refined vector.
///
/// Raises:
///     value_error: `degree < 0`, or a knot vector with fewer than `degree + 2` knots.
inline void check_oslo_sizes(int degree, std::size_t num_old, std::size_t num_new) {
  if (degree < 0) {
    throw value_error("degree must be non-negative");
  }
  const auto min_size = static_cast<std::size_t>(degree) + 2;
  if (num_old < min_size || num_new < min_size) {
    throw value_error("Both knot vectors need at least degree + 2 = " + std::to_string(min_size) +
                      " knots, got " + std::to_string(num_old) + " and " + std::to_string(num_new) +
                      ".");
  }
}

/// An array with no elements, to be assigned over (`NdArray` has no default constructor).
template <StorageFloat T>
[[nodiscard]] NdArray<T> empty_net() {
  return NdArray<T>(std::vector<std::size_t>{0});
}

/// Apply a one-direction operation along each direction of a B-spline's control net, in turn.
///
/// The net is flattened to `(num_basis[d], rest)` along `d` (without a copy when nothing precedes
/// the axis), `op` maps the direction's current space and flattened points to a new space and
/// points, and the points are folded back.
/// `op` is called only for the directions `selected` names; the others keep their space.
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///     selected (Pred): `bool(std::size_t direction)`.
///     op (Op): `(std::size_t direction, const BsplineSpace1D<T>&, mdview<const T, 2>) ->
///         std::pair<BsplineSpace1D<T>, NdArray<T>>`: the new space and the new flattened points.
///
/// Returns:
///     Bspline<T>: The B-spline over the new spaces and points, rational as `b` is.
template <StorageFloat T, class Pred, class Op>
[[nodiscard]] Bspline<T> map_directions(const Bspline<T>& b, Pred&& selected, Op&& op) {
  std::vector<BsplineSpace1D<T>> spaces = b.space().spaces();
  NdArray<T> owned = empty_net<T>();
  std::span<const T> current = b.control_points();
  std::vector<std::size_t> shape(b.shape().begin(), b.shape().end());
  for (std::size_t d = 0; d < spaces.size(); ++d) {
    if (!selected(d)) {
      continue;
    }
    const bezier::detail::AxisSplit around = bezier::detail::split_around(shape, d);
    std::pair<BsplineSpace1D<T>, NdArray<T>> result{spaces[d], empty_net<T>()};
    NdArray<T> next = empty_net<T>();
    if (around.outer == 1) {
      // Nothing precedes the axis: the net is already the matrix `(num_basis[d], rest)`, row-major,
      // and the result matrix is already the new net, so neither is copied.
      result = op(d, spaces[d], mdview<const T, 2>(current.data(), shape[d], around.inner));
      shape[d] = result.second.shape()[0];
      next = std::move(result.second);
    } else {
      const NdArray<T> flat = bezier::detail::flatten_along_axis<T>(current, shape, d);
      result = op(d, spaces[d], flat.template view<2>());
      next = bezier::detail::unflatten_along_axis<T>(result.second.template view<2>(), shape, d);
      shape[d] = result.second.shape()[0];
    }
    spaces[d] = std::move(result.first);
    owned = std::move(next);
    current = owned.flat();
  }
  return Bspline<T>(BsplineSpace<T>(std::move(spaces)), std::move(owned), b.rational());
}

}  // namespace detail
}  // namespace pantr::bspline
