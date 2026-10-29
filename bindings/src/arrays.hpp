// Arrays across the binding: handing a C++ result to NumPy without a copy, exposing a read-only
// view of a shared immutable representation, and releasing the GIL only around work that reaches
// the pool.
#pragma once

#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/parallel/work_model.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bindings {

namespace nb = nanobind;

/// A NumPy array of `T` with any rank (a fresh, writable, owned result).
template <class T>
using NumpyArray = nb::ndarray<nb::numpy, T>;

/// A read-only NumPy array of `T` with any rank (a view of an immutable C++ value).
template <class T>
using NumpyView = nb::ndarray<nb::numpy, const T>;

/// Hand an `NdArray` to NumPy without copying: the array's buffer moves into a capsule that the
/// NumPy array owns.
///
/// Args:
///     a (NdArray<T>&&): The result; consumed.
///
/// Returns:
///     NumpyArray<T>: A writable C-contiguous array of the same shape.
template <class T>
NumpyArray<T> to_numpy(NdArray<T>&& a) {
  std::vector<std::size_t> shape(a.shape().begin(), a.shape().end());
  auto* buffer = new std::shared_ptr<T[]>(std::move(a).into_buffer());
  nb::capsule owner(buffer, [](void* p) noexcept { delete static_cast<std::shared_ptr<T[]>*>(p); });
  return NumpyArray<T>(buffer->get(), shape.size(), shape.data(), owner);
}

/// Expose `n` elements of shape `shape` at `data`, read-only, owned by a copy of `keep` (the
/// shared representation the data lives in), so the view outlives every C++ handle on it.
///
/// Args:
///     data (const T*): The first element (row-major).
///     shape (std::span<const std::size_t>): The extents.
///     keep (std::shared_ptr<const R>): The owner of the data.
///
/// Returns:
///     NumpyView<T>: A read-only array.
template <class T, class R>
NumpyView<T> view_of(const T* data, std::span<const std::size_t> shape,
                     std::shared_ptr<const R> keep) {
  auto* holder = new std::shared_ptr<const R>(std::move(keep));
  nb::capsule owner(holder,
                    [](void* p) noexcept { delete static_cast<std::shared_ptr<const R>*>(p); });
  return NumpyView<T>(data, shape.size(), shape.data(), owner);
}

/// Narrow a degree from the binding's 64-bit integer to the C++ API's `int`.
///
/// A negative degree outside `int` becomes `INT_MIN`, which every C++ entry point refuses as
/// negative with its own message; a positive one is refused here. Saturating it instead would let
/// a tabulation succeed at the wrong degree.
///
/// Args:
///     degree (long long): The degree as given.
///
/// Returns:
///     int: `degree`, or `INT_MIN` for a negative degree outside `int`.
///
/// Raises:
///     capacity_error: `degree` exceeds `INT_MAX`.
inline int saturate_degree(long long degree) {
  constexpr long long lo = (std::numeric_limits<int>::min)();
  constexpr long long hi = (std::numeric_limits<int>::max)();
  if (degree > hi) {
    throw capacity_error("degree " + std::to_string(degree) +
                         " exceeds the largest supported degree " + std::to_string(hi));
  }
  return static_cast<int>(degree < lo ? lo : degree);
}

/// Narrow a list of degrees as `saturate_degree` does.
inline std::vector<int> saturate_degrees(const std::vector<long long>& degrees) {
  std::vector<int> out(degrees.size());
  for (std::size_t i = 0; i < degrees.size(); ++i) {
    out[i] = saturate_degree(degrees[i]);
  }
  return out;
}

/// Release the GIL for the lifetime of the guard when `n` items of `ops` work each will run in
/// parallel (and only then: releasing costs a lock round trip a serial call need not pay).
class ReleaseIfParallel {
 public:
  ReleaseIfParallel(std::size_t n, double ops) {
    if (parallel::worth_parallel(n, parallel::WorkEstimate{ops})) {
      release_.emplace();
    }
  }

 private:
  std::optional<nb::gil_scoped_release> release_;
};

}  // namespace pantr::bindings
