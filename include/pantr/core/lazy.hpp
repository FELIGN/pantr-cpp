/// @file lazy.hpp
/// @brief `pantr::detail::Lazy`: a block of derived state computed once, on first use,
/// thread-safely.
///
/// pantr's domain values are immutable and shared between threads. Derived quantities too costly
/// to compute eagerly live in one grouped block per type behind `Lazy<Block>`: double-checked
/// locking, with an acquire load of an atomic flag on the fast path and a mutex on the slow path.
/// Not a bare `mutable std::optional` (a data race that value tests do not see) and not
/// `std::call_once` (which needs a `std::once_flag` beside each block and offers no `ready()`
/// query).
#pragma once

#include "pantr/config.hpp"

#include <atomic>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>

namespace pantr::detail {

/// A value of type `Block` computed at most once, by the first caller of `get`.
///
/// Thread-safe: concurrent `get` calls compute the block once and all return the same object. If
/// the computation throws, nothing is stored, the exception reaches that caller, and a later call
/// computes again.
///
/// Copying gives an empty `Lazy`: a copy is typically made to build a new value from an old one,
/// and the old derived state need not describe the new value, so it is recomputed on demand
/// rather than carried over. Assignment is deleted, because replacing a block that another thread
/// may be reading through a returned reference would invalidate that reference.
///
/// Attributes:
///     ready_ (std::atomic<bool>): Set, with release ordering, once `value_` holds the block.
///     mutex_ (std::mutex): Serializes the computation.
///     value_ (std::optional<Block>): The block, written once under `mutex_`.
template <class Block>
  requires std::is_object_v<Block> && std::is_destructible_v<Block>
class Lazy {
 public:
  Lazy() = default;
  Lazy(const Lazy& /*other*/) noexcept {}
  Lazy& operator=(const Lazy&) = delete;
  ~Lazy() = default;

  /// Return the block, computing it with `compute()` if no call has stored it yet.
  ///
  /// Args:
  ///     compute (F&&): Callable returning a `Block`; called at most once per stored block. It
  ///         must not call `get` on this same `Lazy`, which would deadlock on its mutex.
  ///
  /// Returns:
  ///     const Block&: The stored block, valid for the lifetime of this `Lazy`.
  ///
  /// Raises:
  ///     Whatever `compute` throws; nothing is stored in that case.
  template <class F>
    requires std::is_convertible_v<std::invoke_result_t<F&>, Block>
  const Block& get(F&& compute) const {
    if (!ready_.load(std::memory_order_acquire)) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!ready_.load(std::memory_order_relaxed)) {
        value_.emplace(compute());
        ready_.store(true, std::memory_order_release);
      }
    }
    return *value_;
  }

  /// Tell whether the block has been computed.
  ///
  /// Returns:
  ///     bool: True once a `get` call has stored the block.
  [[nodiscard]] bool ready() const noexcept { return ready_.load(std::memory_order_acquire); }

 private:
  mutable std::atomic<bool> ready_{false};
  mutable std::mutex mutex_;
  mutable std::optional<Block> value_;
};

}  // namespace pantr::detail
