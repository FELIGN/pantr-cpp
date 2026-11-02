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
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pantr::bindings {

namespace nb = nanobind;

/// A NumPy array of 64-bit integers, retyped as `numpy.int64` on its way to Python.
///
/// nanobind hands an array to NumPy through the buffer protocol, whose format for a 64-bit integer
/// is `'q'` (`long long`); NumPy reads that as `numpy.longlong`, which on Linux is a type distinct
/// from `numpy.int64` (`numpy.long`): the dtypes compare equal, but `dtype.type is numpy.int64`,
/// `isinstance(a[0], numpy.int64)` and `numpy.issubdtype(a.dtype, numpy.int64)` are false. This
/// type's caster (below) exports the array as nanobind does and returns `a.view(numpy.int64)`, a
/// view of the same memory with the same writability. Every int64 `NumpyArray` and `NumpyView` is
/// this type, so every binding gets the retyping without a call site of its own.
template <class T>
struct Int64Numpy : nb::ndarray<nb::numpy, T> {
  using nb::ndarray<nb::numpy, T>::ndarray;
};

/// Whether `T` (const or not) is a 64-bit signed integer.
template <class T>
inline constexpr bool is_int64_v = std::is_same_v<std::remove_const_t<T>, std::int64_t>;

/// A NumPy array of `T` with any rank (a fresh, writable, owned result).
template <class T>
using NumpyArray = std::conditional_t<is_int64_v<T>, Int64Numpy<T>, nb::ndarray<nb::numpy, T>>;

/// A read-only NumPy array of `T` with any rank (a view of an immutable C++ value).
template <class T>
using NumpyView =
    std::conditional_t<is_int64_v<T>, Int64Numpy<const T>, nb::ndarray<nb::numpy, const T>>;

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

/// Hand a vector to NumPy without copying: the vector moves into a capsule that the NumPy array
/// owns.
///
/// Args:
///     v (std::vector<T>&&): The result; consumed.
///
/// Returns:
///     NumpyArray<T>: A writable one-dimensional array over the vector's elements.
template <class T>
NumpyArray<T> vector_to_numpy(std::vector<T>&& v) {
  if (v.empty()) {
    // An empty vector's data may be null, which NumPy would not take as an owned buffer.
    return to_numpy(NdArray<T>(std::vector<std::size_t>{std::size_t{0}}, for_overwrite));
  }
  auto* holder = new std::vector<T>(std::move(v));
  nb::capsule owner(holder, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  const std::size_t shape[1] = {holder->size()};
  return NumpyArray<T>(holder->data(), 1, shape, owner);
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
  // An empty vector's data may be null, and NumPy answers a null buffer with a fresh writable
  // array where the view must be read-only; any valid address will do for zero elements.
  static const T unused{};
  if (data == nullptr) {
    data = &unused;
  }
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

namespace nanobind::detail {

/// Export an `Int64Numpy` as nanobind exports its base, then retype it as `numpy.int64` (see
/// `Int64Numpy`). Return-only: no binding takes one as an argument.
template <class T>
struct type_caster<pantr::bindings::Int64Numpy<T>> {
  using Base = type_caster<ndarray<numpy, T>>;
  NB_TYPE_CASTER(pantr::bindings::Int64Numpy<T>, Base::Name)

  bool from_python(handle /*src*/, uint32_t /*flags*/, cleanup_list* /*cleanup*/) noexcept {
    return false;
  }

  static handle from_cpp(const pantr::bindings::Int64Numpy<T>& value, rv_policy policy,
                         cleanup_list* cleanup) noexcept {
    handle exported = Base::from_cpp(value, policy, cleanup);
    if (!exported.is_valid()) {
      return exported;
    }
    PyObject* dtype = int64_dtype();
    if (dtype == nullptr) {
      exported.dec_ref();
      return {};
    }
    // Interned once and kept for the process, as the dtype is.
    static PyObject* const view_name = PyUnicode_InternFromString("view");
    if (view_name == nullptr) {
      exported.dec_ref();
      return {};
    }
    // `args[0]` is the receiver. Without PY_VECTORCALL_ARGUMENTS_OFFSET, since no slot precedes
    // `args` for the callee to borrow.
    PyObject* args[2] = {exported.ptr(), dtype};
    PyObject* retyped = PyObject_VectorcallMethod(view_name, args, 2, nullptr);
    exported.dec_ref();
    return retyped;
  }

 private:
  /// `numpy.dtype(numpy.int64)`, created on first use and kept for the process (never released:
  /// a release at exit could run after the interpreter it belongs to). Null, with the Python error
  /// set, when it cannot be created.
  static PyObject* int64_dtype() noexcept {
    // Built under the GIL, which every caller holds; the import may release it, so two threads
    // can both build it and one reference is then leaked, which is harmless. A failed attempt
    // leaves it null with the error set, and the next call tries again.
    static PyObject* dtype = nullptr;
    if (dtype == nullptr) {
      PyObject* np = PyImport_ImportModule("numpy");
      if (np == nullptr) {
        return nullptr;
      }
      PyObject* type = PyObject_GetAttrString(np, "int64");
      dtype = type == nullptr ? nullptr : PyObject_CallMethod(np, "dtype", "O", type);
      Py_XDECREF(type);
      Py_DECREF(np);
    }
    return dtype;
  }
};

}  // namespace nanobind::detail
