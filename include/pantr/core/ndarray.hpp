/// @file ndarray.hpp
/// @brief `pantr::NdArray`: an owning, contiguous, row-major array with runtime extents.
///
/// The return type of the convenience overloads (the `out` overloads write into a caller's
/// `mdspan` instead). Its buffer is a `std::shared_ptr<T[]>`, which the Python binding hands to
/// NumPy without a copy by consuming the array (`std::move(a).into_buffer()`).
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/mdspan.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pantr {

/// Owning, contiguous, row-major (C order) array whose rank and extents are runtime values.
///
/// A value type: copying copies the elements, so two arrays never alias. A moved-from array has
/// `size() == 0` and `data() == nullptr`, and may only be assigned to or destroyed.
///
/// Attributes:
///     shape_ (std::vector<std::size_t>): Extents, outermost first.
///     size_ (std::size_t): Number of elements, the product of the extents.
///     data_ (std::shared_ptr<T[]>): The elements; never shared with another `NdArray`.
template <class T>
  requires std::is_arithmetic_v<T>
class NdArray {
 public:
  /// Allocate a zero-initialized array of the given shape.
  ///
  /// Args:
  ///     shape (std::vector<std::size_t>): Extents, outermost first. An empty vector gives a
  ///         rank-0 array holding one element, as in NumPy.
  ///
  /// Raises:
  ///     capacity_error: The element count or its size in bytes does not fit in `std::size_t`.
  explicit NdArray(std::vector<std::size_t> shape)
      : shape_(std::move(shape)),
        size_(checked_size(shape_)),
        data_(std::make_shared<T[]>(size_)) {}

  NdArray(const NdArray& other)
      : shape_(other.shape_), size_(other.size_), data_(std::make_shared<T[]>(other.size_)) {
    std::copy_n(other.data_.get(), size_, data_.get());
  }

  NdArray(NdArray&& other) noexcept
      : shape_(std::move(other.shape_)),
        size_(std::exchange(other.size_, 0)),
        data_(std::move(other.data_)) {
    other.reset_to_empty();
  }

  NdArray& operator=(const NdArray& other) {
    if (this != &other) {
      NdArray copy(other);
      swap(copy);
    }
    return *this;
  }

  NdArray& operator=(NdArray&& other) noexcept {
    if (this != &other) {
      NdArray moved(std::move(other));
      swap(moved);
    }
    return *this;
  }

  ~NdArray() = default;

  /// Get the number of dimensions.
  ///
  /// Returns:
  ///     std::size_t: The rank.
  [[nodiscard]] std::size_t rank() const noexcept { return shape_.size(); }

  /// Get the extents.
  ///
  /// Returns:
  ///     std::span<const std::size_t>: The extents, outermost first.
  [[nodiscard]] std::span<const std::size_t> shape() const noexcept { return shape_; }

  /// Get the number of elements.
  ///
  /// Returns:
  ///     std::size_t: The product of the extents.
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

  /// Get the elements in row-major order.
  ///
  /// Returns:
  ///     T*: Pointer to the first element (null for a moved-from array).
  [[nodiscard]] T* data() noexcept { return data_.get(); }

  /// Get the elements in row-major order.
  ///
  /// Returns:
  ///     const T*: Pointer to the first element (null for a moved-from array).
  [[nodiscard]] const T* data() const noexcept { return data_.get(); }

  /// Get the elements as a flat span.
  ///
  /// Returns:
  ///     std::span<T>: All elements in row-major order.
  [[nodiscard]] std::span<T> flat() noexcept { return {data_.get(), size_}; }

  /// Get the elements as a flat span.
  ///
  /// Returns:
  ///     std::span<const T>: All elements in row-major order.
  [[nodiscard]] std::span<const T> flat() const noexcept { return {data_.get(), size_}; }

  /// View the array as a row-major `mdspan` of static rank `Rank`.
  ///
  /// Returns:
  ///     mdview<T, Rank>: A view of the elements, valid while the array lives and is not moved.
  ///
  /// Raises:
  ///     value_error: `Rank` differs from `rank()`.
  template <std::size_t Rank>
  [[nodiscard]] mdview<T, Rank> view() {
    return make_view<T, Rank>(data_.get());
  }

  /// View the array as a read-only row-major `mdspan` of static rank `Rank`.
  ///
  /// Returns:
  ///     mdview<const T, Rank>: A view of the elements, valid while the array lives.
  ///
  /// Raises:
  ///     value_error: `Rank` differs from `rank()`.
  template <std::size_t Rank>
  [[nodiscard]] mdview<const T, Rank> view() const {
    return make_view<const T, Rank>(data_.get());
  }

  /// Give up the buffer, consuming the array (for a zero-copy hand-off to another owner).
  ///
  /// Returns:
  ///     std::shared_ptr<T[]>: The elements; this array is left in the moved-from state.
  [[nodiscard]] std::shared_ptr<T[]> into_buffer() && {
    std::shared_ptr<T[]> buffer = std::move(data_);
    reset_to_empty();
    return buffer;
  }

  /// Exchange contents with another array.
  ///
  /// Args:
  ///     other (NdArray&): The array to swap with.
  void swap(NdArray& other) noexcept {
    shape_.swap(other.shape_);
    std::swap(size_, other.size_);
    data_.swap(other.data_);
  }

 private:
  // The moved-from state: no elements and no buffer. The shape is left empty rather than set to
  // (0,), which would allocate inside a noexcept move.
  void reset_to_empty() noexcept {
    shape_.clear();
    size_ = 0;
    data_.reset();
  }

  // NumPy's rule (PyArray_NewFromDescr): the item size times the product of the nonzero extents
  // must fit in a signed pointer difference, and a zero extent then makes the array empty. Taking
  // the same rule means every shape an NdArray accepts can be handed to NumPy (into_buffer), and
  // every shape NumPy refuses is refused here first, with a C++ error rather than at the boundary.
  static std::size_t checked_size(const std::vector<std::size_t>& shape) {
    constexpr auto max_bytes =
        static_cast<std::size_t>((std::numeric_limits<std::ptrdiff_t>::max)());
    constexpr std::size_t max_elements = max_bytes / sizeof(T);
    std::size_t n = 1;
    bool empty = false;
    for (const std::size_t extent : shape) {
      if (extent == 0) {
        empty = true;
        continue;
      }
      if (n > max_elements / extent) {
        throw capacity_error("NdArray shape " + py_repr(std::span<const std::size_t>(shape)) +
                             " has more elements than fit in memory");
      }
      n *= extent;
    }
    return empty ? 0 : n;
  }

  template <class U, std::size_t Rank>
  mdview<U, Rank> make_view(U* ptr) const {
    if (Rank != shape_.size()) {
      throw value_error("cannot view an array of rank " + std::to_string(shape_.size()) +
                        " as rank " + std::to_string(Rank));
    }
    std::array<std::size_t, Rank> ext{};
    for (std::size_t i = 0; i < Rank; ++i) {
      ext[i] = shape_[i];
    }
    return mdview<U, Rank>(ptr, ext);
  }

  std::vector<std::size_t> shape_;
  std::size_t size_;
  std::shared_ptr<T[]> data_;
};

}  // namespace pantr
