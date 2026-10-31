/// @file slice.hpp
/// @brief Fixing one parametric direction of a `Bspline` at a value: `slice`, `boundary`.
///
/// The control points of the slice are the de Boor corner-cutting (Piegl and Tiller A5.1) of the
/// `degree + 1` control points that carry the function at the parameter, in the sliced direction,
/// for every row of the other directions at once. A parameter at a knot of multiplicity `s` needs
/// `degree - s` rounds, and at multiplicity `degree` or more the slice is a control point.
#pragma once

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/detail/refine_common.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace pantr::bspline {

namespace detail {

/// The knot span a parameter value falls in, and the value used for it.
///
/// Attributes:
///     k (std::size_t): The span, `knots[k] <= u < knots[k + 1]` (the last span at the right end).
///     u (double): The parameter: the value, or the domain end it was within tolerance of.
struct SliceSpan {
  std::size_t k;
  double u;
};

/// Find the span of a parameter value (Piegl and Tiller convention): a value within `tol` of an end
/// of the domain is taken as that end, any other by binary search over the knot vector.
///
/// Args:
///     knots (std::span<const T>): The knot vector.
///     degree (int): The degree.
///     n_basis (std::size_t): The basis functions of the (unwrapped) direction:
///         `knots.size() - degree - 1`.
///     value (double): The parameter, inside the domain.
///     tol (double): Absolute parametric tolerance.
///
/// Returns:
///     SliceSpan: The span and the parameter.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] SliceSpan find_slice_span(std::span<const T> knots, int degree, std::size_t n_basis,
                                        double value, double tol) {
  const auto p = static_cast<std::size_t>(degree);
  const double u_left = static_cast<double>(knots[p]);
  const double u_right = static_cast<double>(knots[n_basis]);
  if (value <= u_left + tol) {
    return {p, u_left};
  }
  if (value >= u_right - tol) {
    return {n_basis - 1, u_right};
  }
  const auto above = std::upper_bound(knots.begin(), knots.end(), value, [](double v, T knot) {
    return v < static_cast<double>(knot);
  });
  return {static_cast<std::size_t>(above - knots.begin()) - 1, value};
}

/// Count the knots, from the one at `k` leftwards and at most `degree + 1` of them, within `tol`
/// of `u`, stopping at the first that is not.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] std::size_t count_slice_multiplicity(std::span<const T> knots, std::size_t k,
                                                   int degree, double u, double tol) {
  std::size_t s = 0;
  const std::size_t stop =
      k >= static_cast<std::size_t>(degree) ? k - static_cast<std::size_t>(degree) : 0;
  for (std::size_t i = k + 1; i-- > stop;) {
    if (!(std::abs(static_cast<double>(knots[i]) - u) <= tol)) {
      break;
    }
    ++s;
  }
  return s;
}

/// One corner-cutting step over `w` contiguous columns: `a[c] = alpha b[c] + beta a[c]`, in `T`.
///
/// A `noexcept` function of its own so that GCC vectorises it (a contracted multiply-add in a body
/// that may throw leaves the loop scalar).
///
/// Note:
///     Inputs are assumed to be correct (no validation performed). `a` and `b` must not overlap.
template <StorageFloat T>
void corner_cut_rows(T* a, const T* b, std::size_t w, T alpha, T beta) noexcept {
  for (std::size_t c = 0; c < w; ++c) {
    a[c] = alpha * b[c] + beta * a[c];
  }
}

/// Compute the control net of the slice of a B-spline at a parameter, in one direction.
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///     axis (std::size_t): The direction, `< b.dim()`.
///     value (double): The parameter, inside the domain.
///
/// Returns:
///     NdArray<T>: The control points, extents `b.shape()` without `axis`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] NdArray<T> slice_net(const Bspline<T>& b, std::size_t axis, double value) {
  const BsplineSpace<T> whole = b.space();
  const BsplineSpace1D<T>& space = whole.spaces()[axis];
  const std::span<const T> knots = space.knots();
  const int degree = space.degree();
  const auto p = static_cast<std::size_t>(degree);
  const double tol = space.tolerance();
  const std::span<const std::size_t> shape = b.shape();
  const std::size_t n_stored = shape[axis];
  // A periodic direction stores its first control points once; the function reads them modulo
  // `n_stored` over `knots.size() - degree - 1` positions.
  const bool periodic = space.periodic();
  const std::size_t n_basis = knots.size() - p - 1;

  const SliceSpan span = find_slice_span<T>(knots, degree, n_basis, value, tol);
  const std::size_t k = span.k;
  const std::size_t s = count_slice_multiplicity<T>(knots, k, degree, span.u, tol);

  const bezier::detail::AxisSplit around = bezier::detail::split_around(shape, axis);
  const std::size_t rest = around.outer * around.inner;
  std::vector<std::size_t> out_shape;
  out_shape.reserve(shape.size() - 1);
  for (std::size_t d = 0; d < shape.size(); ++d) {
    if (d != axis) {
      out_shape.push_back(shape[d]);
    }
  }
  NdArray<T> out(std::move(out_shape), for_overwrite);
  const T* src = b.control_points().data();
  // Row `a` of the window of control points that carry the function on the span, as a run of
  // `rest` values (the other directions' rows, in order).
  const auto load_row = [&](std::size_t a, T* dst) {
    const std::size_t index = periodic ? (k - p + a) % n_stored : k - p + a;
    for (std::size_t o = 0; o < around.outer; ++o) {
      std::copy_n(src + (o * n_stored + index) * around.inner, around.inner,
                  dst + o * around.inner);
    }
  };

  // At a knot of multiplicity `degree` or more only one function is nonzero: the slice is its
  // control point.
  if (s >= p) {
    load_row(0, out.data());
    return out;
  }
  const std::size_t r = p - s;
  std::vector<T> work((r + 1) * rest);
  for (std::size_t a = 0; a <= r; ++a) {
    load_row(a, work.data() + a * rest);
  }
  for (std::size_t j = 1; j <= r; ++j) {
    for (std::size_t i = 0; i + j <= r; ++i) {
      const double left_knot = static_cast<double>(knots[k - p + j + i]);
      const double right_knot = static_cast<double>(knots[i + k + 1]);
      const double denom = right_knot - left_knot;
      // Repeated knots in the window make the denominator exactly zero, and the weight is then 0.
      // An exact test: the difference of two knots of a space is zero or at least its tolerance.
      const double alpha = denom == 0.0 ? 0.0 : (span.u - left_knot) / denom;
      // The weights are formed in double and rounded to `T`, as the operands they multiply.
      corner_cut_rows<T>(work.data() + i * rest, work.data() + (i + 1) * rest, rest,
                         static_cast<T>(alpha), static_cast<T>(1.0 - alpha));
    }
  }
  std::copy_n(work.data(), rest, out.data());
  return out;
}

/// Check a direction against the dimension, with the message `Bspline.slice` gives.
///
/// Raises:
///     value_error: `axis` is not in `[0, dim)`.
inline std::size_t checked_axis(long long axis, std::size_t dim) {
  if (axis < 0 || axis >= static_cast<long long>(dim)) {
    throw value_error("axis must be in [0, " + std::to_string(dim) + "), got " +
                      std::to_string(axis) + ".");
  }
  return static_cast<std::size_t>(axis);
}

/// Check that a parameter lies in the domain of a direction, up to the space's tolerance.
///
/// A NaN is outside.
///
/// Raises:
///     value_error: `value` is outside the domain widened by the tolerance.
template <StorageFloat T>
void check_slice_value(const BsplineSpace1D<T>& space, std::size_t axis, double value) {
  const auto [a, b] = space.domain();
  const double tol = space.tolerance();
  const bool inside =
      value >= static_cast<double>(a) - tol && value <= static_cast<double>(b) + tol;
  if (!inside) {
    throw value_error("value " + py_repr(value) + " is outside the domain [" + repr_of<T>(a) +
                      ", " + repr_of<T>(b) + "] of direction " + std::to_string(axis) + ".");
  }
}

/// The parameter at one end of a direction's domain.
///
/// Args:
///     space (const BsplineSpace1D<T>&): The direction's space.
///     side (long long): 0 for the start of the domain, anything else for its end (checked by
///         `check_side` before).
///
/// Returns:
///     double: The end, as a double.
template <StorageFloat T>
[[nodiscard]] double domain_end(const BsplineSpace1D<T>& space, long long side) {
  return static_cast<double>(side == 0 ? space.domain().first : space.domain().second);
}

/// Check the side of a boundary.
///
/// Raises:
///     value_error: `side` is neither 0 nor 1.
inline void check_side(long long side) {
  if (side != 0 && side != 1) {
    throw value_error("side must be 0 or 1, got " + std::to_string(side) + ".");
  }
}

/// Project the slice of a curve to its point: the control point, or for a rational curve the
/// weighted coordinates divided by the weight.
template <StorageFloat T>
[[nodiscard]] NdArray<T> project_point(const NdArray<T>& row, bool rational) {
  const std::size_t rank = row.size() - (rational ? 1 : 0);
  NdArray<T> point({rank}, for_overwrite);
  for (std::size_t i = 0; i < rank; ++i) {
    point.data()[i] = rational ? row.data()[i] / row.data()[rank] : row.data()[i];
  }
  return point;
}

}  // namespace detail

/// Fix one parametric direction of a surface or a volume at a parameter value.
///
/// Reduces the parametric dimension by one by de Boor corner cutting on the control points along
/// the direction. At a knot of multiplicity `s` only `degree - s` rounds are needed, and at
/// `s >= degree` the slice is a control point. A rational B-spline stays rational. A periodic
/// direction is read through its wrapped control points. A parameter within the space's
/// tolerance of an end of the domain is taken as that end.
///
/// Args:
///     b (const Bspline<T>&): The B-spline, of dimension two or more (`slice_point` is the
///     curve's). axis (long long): The direction to fix, in `[0, dim)`. value (double): The
///     parameter, in the direction's domain up to the space's tolerance.
///
/// Returns:
///     Bspline<T>: The B-spline of the other directions, rational as `b` is.
///
/// Raises:
///     value_error: `b` is a curve; `axis` is out of range; `value` is outside the domain (NaN
///         included).
template <StorageFloat T>
[[nodiscard]] Bspline<T> slice(const Bspline<T>& b, long long axis, double value) {
  if (b.dim() < 2) {
    throw value_error("slice: a curve has no lower-dimensional B-spline; use slice_point.");
  }
  const std::size_t d = detail::checked_axis(axis, b.dim());
  const BsplineSpace<T> whole = b.space();
  detail::check_slice_value<T>(whole.spaces()[d], d, value);
  NdArray<T> net = detail::slice_net<T>(b, d, value);
  std::vector<BsplineSpace1D<T>> spaces;
  spaces.reserve(b.dim() - 1);
  for (std::size_t i = 0; i < b.dim(); ++i) {
    if (i != d) {
      spaces.push_back(whole.spaces()[i]);
    }
  }
  return Bspline<T>(BsplineSpace<T>(std::move(spaces)), std::move(net), b.rational());
}

/// Evaluate a curve at a parameter by de Boor corner cutting.
///
/// Args:
///     b (const Bspline<T>&): A curve (dimension one).
///     axis (long long): The direction, which must be 0.
///     value (double): The parameter, in the domain up to the space's tolerance.
///
/// Returns:
///     NdArray<T>: The point, extents `(rank,)`; for a rational curve the weighted coordinates
///     divided by the weight (projected).
///
/// Raises:
///     value_error: `b` is not a curve; `axis` is out of range; `value` is outside the domain (NaN
///         included).
template <StorageFloat T>
[[nodiscard]] NdArray<T> slice_point(const Bspline<T>& b, long long axis, double value) {
  if (b.dim() != 1) {
    throw value_error("slice_point: only a curve slices to a point; use slice.");
  }
  const std::size_t d = detail::checked_axis(axis, b.dim());
  detail::check_slice_value<T>(b.space().spaces()[d], d, value);
  const NdArray<T> row = detail::slice_net<T>(b, d, value);
  return detail::project_point<T>(row, b.rational());
}

/// Extract the boundary of a surface or a volume along one direction: its restriction to one end
/// of the domain, `slice` at the domain's start (`side == 0`) or end (`side == 1`).
///
/// Args:
///     b (const Bspline<T>&): The B-spline, of dimension two or more.
///     axis (long long): The direction, in `[0, dim)`.
///     side (long long): 0 for the start of the domain, 1 for the end.
///
/// Returns:
///     Bspline<T>: The boundary B-spline.
///
/// Raises:
///     value_error: `side` is not 0 or 1; `axis` is out of range; `b` is a curve.
template <StorageFloat T>
[[nodiscard]] Bspline<T> boundary(const Bspline<T>& b, long long axis, long long side) {
  detail::check_side(side);
  const std::size_t d = detail::checked_axis(axis, b.dim());
  return slice<T>(b, axis, detail::domain_end<T>(b.space().spaces()[d], side));
}

/// Extract the end point of a curve: `slice_point` at the domain's start or end.
///
/// Args:
///     b (const Bspline<T>&): A curve.
///     axis (long long): The direction, which must be 0.
///     side (long long): 0 for the start of the domain, 1 for the end.
///
/// Returns:
///     NdArray<T>: The point, extents `(rank,)`, projected for a rational curve.
///
/// Raises:
///     value_error: `side` is not 0 or 1; `axis` is out of range; `b` is not a curve.
template <StorageFloat T>
[[nodiscard]] NdArray<T> boundary_point(const Bspline<T>& b, long long axis, long long side) {
  detail::check_side(side);
  const std::size_t d = detail::checked_axis(axis, b.dim());
  return slice_point<T>(b, axis, detail::domain_end<T>(b.space().spaces()[d], side));
}

}  // namespace pantr::bspline
