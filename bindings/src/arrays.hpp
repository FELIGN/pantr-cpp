// Arrays across the binding: handing a C++ result to NumPy without a copy, exposing a read-only
// view of a shared immutable representation, and releasing the GIL only around work that reaches
// the pool.
#pragma once

#include "pantr/core/ndarray.hpp"
#include "pantr/parallel/work_model.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <span>
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

/// Narrow a degree from Python's unbounded `int` to the C++ API's `int`, saturating.
///
/// A degree outside `int` is far outside every domain: saturating keeps the C++ refusal (a
/// negative degree, a degree past a builder's domain, an output too large to allocate), all
/// `ValueError`s, where a narrowing argument would fail the binding's type check with a
/// `TypeError` first. Only the degree a refusal message quotes is the saturated one.
///
/// Args:
///     degree (long long): The degree as given.
///
/// Returns:
///     int: `degree`, clamped to `[INT_MIN, INT_MAX]`.
inline int saturate_degree(long long degree) {
  constexpr long long lo = (std::numeric_limits<int>::min)();
  constexpr long long hi = (std::numeric_limits<int>::max)();
  return static_cast<int>(degree < lo ? lo : (degree > hi ? hi : degree));
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
