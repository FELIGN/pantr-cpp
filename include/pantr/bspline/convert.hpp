/// @file convert.hpp
/// @brief `astype`: a B-spline converted to another storage type.
#pragma once

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/scalar.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace pantr::bspline {

namespace detail {

/// Convert one value to another storage type, refusing a finite value beyond its range.
///
/// A finite `double` above `FLT_MAX` has no `float` to convert to, and C++ leaves the conversion
/// undefined (IEEE 754 would give infinity); widening, and a value of `T` that `U` can hold, are
/// exact or round to nearest.
///
/// Args:
///     v (T): The value.
///
/// Returns:
///     U: `v`, converted (non-finite values keep their class).
///
/// Raises:
///     value_error: `v` is finite and its magnitude exceeds the largest finite `U`.
template <StorageFloat U, StorageFloat T>
[[nodiscard]] U convert_value(T v) {
  if constexpr (sizeof(U) < sizeof(T)) {
    if (std::isfinite(v) && std::abs(v) > static_cast<T>((std::numeric_limits<U>::max)())) {
      throw value_error("The value " + py_repr(static_cast<double>(v)) +
                        " is outside the range of the target floating-point type.");
    }
  }
  return static_cast<U>(v);
}

/// Convert one direction to another storage type: the knots cast, the degree and the periodicity
/// kept, the space (its tolerance included) built from the cast knots as the constructor does.
template <StorageFloat U, StorageFloat T>
[[nodiscard]] BsplineSpace1D<U> astype_space_1d(const BsplineSpace1D<T>& s) {
  const std::span<const T> knots = s.knots();
  std::vector<U> cast;
  cast.reserve(knots.size());
  for (const T k : knots) {
    cast.push_back(convert_value<U>(k));
  }
  return BsplineSpace1D<U>(std::span<const U>(cast), s.degree(), s.periodic());
}

}  // namespace detail

/// Convert a B-spline to another storage type.
///
/// The knots and the control points are cast to `U`; the degrees, the periodicity and the
/// rationality are kept. Each direction's space is rebuilt from its cast knots as the space
/// constructor does (so its tolerance is that of `U`, and snapping may merge knots a narrowing
/// cast brought together). Widening is exact. The same type returns `b` itself, sharing its
/// representation.
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///
/// Returns:
///     Bspline<U>: The converted B-spline.
///
/// Raises:
///     value_error: A knot or control-point value is finite and beyond the range of `U`, or the
///         cast knots no longer define a space (the constructor's refusals).
template <StorageFloat U, StorageFloat T>
[[nodiscard]] Bspline<U> astype(const Bspline<T>& b) {
  if constexpr (std::is_same_v<U, T>) {
    return b;
  } else {
    std::vector<BsplineSpace1D<U>> spaces;
    spaces.reserve(b.dim());
    const BsplineSpace<T> source = b.space();
    for (const auto& s : source.spaces()) {
      spaces.push_back(detail::astype_space_1d<U>(s));
    }
    const std::span<const T> src = b.control_points();
    std::shared_ptr<U[]> data(new U[src.size()]);
    U* out = data.get();
    for (std::size_t i = 0; i < src.size(); ++i) {
      out[i] = detail::convert_value<U>(src[i]);
    }
    return Bspline<U>(BsplineSpace<U>(std::move(spaces)),
                      std::shared_ptr<const U[]>(std::move(data)), src.size(), b.cp_size(),
                      b.rational());
  }
}

}  // namespace pantr::bspline
