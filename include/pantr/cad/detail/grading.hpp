/// @file grading.hpp
/// @brief What the comparisons of control points in `pantr::cad` share: the magnitude and the gap
///        of a column group, the order in which a maximum is kept, and a row spelled for a message.
///
/// The Coons constructions and `join` each refuse input whose pieces disagree where they meet, by
/// grading a gap against `tier * scale`. The tier, the scale populations and the message differ
/// per site and stay with the site; the arithmetic of one group's magnitude and gap, and what it
/// does with a NaN, is the same everywhere and has one spelling here.
///
/// A NaN is kept as the reference implementation (NumPy) kept it, so that the verdict on it is the
/// same: `max` and `argmax` of an array propagate it (the first NaN is the maximum), while Python's
/// built-in `max` over scalars never lets a NaN displace the value held. Which comparison
/// then refuses it differs by site and is stated there (`gap > tol` accepts a NaN gap, `!(gap <=
/// tol)` refuses it).
#pragma once

#include "pantr/cad/detail/validation.hpp"
#include "pantr/config.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/scalar.hpp"

#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace pantr::cad::detail {

/// Python's `max(a, b)`: `a`, unless `b` is greater (so a NaN `b` never wins, and a NaN `a` is
/// never replaced).
[[nodiscard]] constexpr double py_max(double a, double b) noexcept { return b > a ? b : a; }

/// Fold a value into a running NumPy maximum or argmax: a NaN, once held, is kept (it is the
/// first NaN), and a NaN value replaces any other.
///
/// Returns:
///     bool: Whether `value` is the new running maximum.
[[nodiscard]] inline bool numpy_max_takes(double best, double value) noexcept {
  if (std::isnan(best)) {
    return false;
  }
  return std::isnan(value) || value > best;
}

/// Get the `index`th column group of a row of `cp_size` columns.
///
/// Args:
///     rational (bool): Whether the trailing column is a weight.
///     index (std::size_t): 0 for the coordinates, 1 for the weight (a rational row only).
///     cp_size (std::size_t): The number of columns.
///
/// Returns:
///     ColumnGroup: `column_groups(rational, cp_size)[index]`.
[[nodiscard]] inline ColumnGroup group_at(bool rational, std::size_t index, std::size_t cp_size) {
  return column_groups(rational, cp_size)[index];
}

/// Get the number of column groups of a control-point array: one, or two when rational.
[[nodiscard]] constexpr std::size_t group_count(bool rational) noexcept { return rational ? 2 : 1; }

/// Get the largest absolute value in one column group over every point of an array
/// (`np.abs(cp[..., columns]).max()`: a NaN anywhere makes it NaN).
///
/// Args:
///     cp (std::span<const T>): `cp.size() / width` points of `width` columns, row-major.
///     width (std::size_t): The columns per point, at least 1.
///     group (const ColumnGroup&): The columns graded together.
///
/// Returns:
///     double: The magnitude (0 when no value is non-zero).
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): `cp` holds at least one
///     point, as every B-spline does.
template <StorageFloat T>
[[nodiscard]] double max_abs_in(std::span<const T> cp, std::size_t width,
                                const ColumnGroup& group) {
  double best = 0.0;
  for (std::size_t p = 0; p + width <= cp.size(); p += width) {
    for (std::size_t c = group.begin; c < group.end; ++c) {
      const double v = static_cast<double>(std::abs(cp[p + c]));
      if (numpy_max_takes(best, v)) {
        best = v;
      }
    }
  }
  return best;
}

/// The largest gap between two arrays over one column group, and the point carrying it.
///
/// Attributes:
///     gap (double): `max |a - b|` over the group's columns and every point (a NaN anywhere makes
///         it NaN); each difference is taken in the storage type, as NumPy subtracts the arrays.
///     index (std::size_t): The point with the largest gap of its own, the first on a tie (and
///         the first NaN, a NaN counting as the largest), `np.argmax` of the per-point maxima.
struct GroupGap {
  double gap;
  std::size_t index;
};

/// Grade two arrays of one shape against each other over one column group.
///
/// Args:
///     a (std::span<const T>): `a.size() / width` points of `width` columns, row-major.
///     b (std::span<const T>): The same shape.
///     width (std::size_t): The columns per point, at least 1.
///     group (const ColumnGroup&): The columns graded together.
///
/// Returns:
///     GroupGap: The largest gap and the point it sits at.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): the two arrays hold the same
///     number of points, at least one.
template <StorageFloat T>
[[nodiscard]] GroupGap group_gap(std::span<const T> a, std::span<const T> b, std::size_t width,
                                 const ColumnGroup& group) {
  GroupGap out{0.0, 0};
  bool first = true;
  for (std::size_t p = 0; p + width <= a.size(); p += width) {
    double point_gap = 0.0;
    for (std::size_t c = group.begin; c < group.end; ++c) {
      const T diff = a[p + c] - b[p + c];
      const double v = static_cast<double>(std::abs(diff));
      if (numpy_max_takes(point_gap, v)) {
        point_gap = v;
      }
    }
    if (first || numpy_max_takes(out.gap, point_gap)) {
      out.gap = point_gap;
      out.index = p / width;
      first = false;
    }
  }
  return out;
}

/// Spell a row of coefficients as a NumPy array is spelled in a message, `[a, b, c]`, each element
/// as `repr` of the double it is (of the double nearest its shortest decimal string for a `float`).
///
/// Args:
///     row (std::span<const T>): The coefficients.
///
/// Returns:
///     std::string: The spelling. This is the one place where a message differs from the
///     reference implementation's, which printed NumPy's own array layout (`[0. 1. 0.]`).
template <StorageFloat T>
[[nodiscard]] std::string row_repr(std::span<const T> row) {
  std::string out = "[";
  for (std::size_t i = 0; i < row.size(); ++i) {
    if (i > 0) {
      out += ", ";
    }
    if constexpr (std::same_as<T, float>) {
      out += py_repr_f32(row[i]);
    } else {
      out += py_repr(row[i]);
    }
  }
  out += ']';
  return out;
}

/// Get the multi-index of a row-major flat position within `shape`.
///
/// Args:
///     flat (std::size_t): A position in `[0, prod(shape))`.
///     shape (std::span<const std::size_t>): The extents, every one at least 1 (empty: a single
///         position).
///
/// Returns:
///     std::vector<std::size_t>: The multi-index (`np.unravel_index`).
[[nodiscard]] inline std::vector<std::size_t> unravel(std::size_t flat,
                                                      std::span<const std::size_t> shape) {
  std::vector<std::size_t> index(shape.size(), 0);
  for (std::size_t d = shape.size(); d-- > 0;) {
    index[d] = flat % shape[d];
    flat /= shape[d];
  }
  return index;
}

}  // namespace pantr::cad::detail
