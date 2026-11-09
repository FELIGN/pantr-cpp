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
///
/// Fork safety: a child forked while another thread builds a block inherits that block's mutex
/// locked by a thread the child lacks. Every build in progress is registered, and a
/// `pthread_atfork` child handler gives each registered block a fresh mutex, so the child builds
/// the block itself instead of blocking forever (design/threading.md).
#pragma once

#include "pantr/config.hpp"

#include <atomic>
#include <cstddef>
#include <map>
#include <mutex>
#include <new>
#include <optional>
#include <system_error>
#include <type_traits>
#include <utility>

#if !defined(_WIN32)
#include <pthread.h>
#endif

namespace pantr::detail {

#if !defined(_WIN32)
/// Guards `lazy_builds`; held across `fork` as the pool's and the tables' mutexes are.
inline constinit std::mutex lazy_builds_mutex;

/// The mutex of every `Lazy` that a thread is building or waiting to build, with the number of
/// such threads. A mutex can be locked only by a thread counted here, so after a `fork` these are
/// the only mutexes the child may find held, by threads that do not exist in it.
inline std::map<std::mutex*, std::size_t> lazy_builds;

/// Registration of the fork handlers; nonzero means it failed, which the first build reports.
/// The child handler runs before the child has any other thread, so no thread can hold or wait
/// on a registered mutex there: each is replaced by a fresh one (the old one is never unlocked or
/// destroyed, since its owner is gone) and the registry is emptied.
inline const int lazy_atfork_status =
    ::pthread_atfork([] { lazy_builds_mutex.lock(); }, [] { lazy_builds_mutex.unlock(); },
                     [] {
                       for (const auto& entry : lazy_builds) {
                         ::new (static_cast<void*>(entry.first)) std::mutex;
                       }
                       lazy_builds.clear();
                       lazy_builds_mutex.unlock();
                     });
#endif

/// Registers a `Lazy` mutex in `lazy_builds` for the guard's lifetime (no-op on Windows, which
/// has no `fork`). Taken before the mutex is locked and released after it is unlocked, so a fork
/// at any point of a build finds the mutex registered whenever it may be held.
class LazyBuildRegistration {
 public:
  /// Register `mutex`.
  ///
  /// Args:
  ///     mutex (std::mutex&): The `Lazy` mutex about to be locked.
  ///
  /// Raises:
  ///     std::system_error: The fork handlers could not be registered.
  explicit LazyBuildRegistration(std::mutex& mutex) : mutex_(&mutex) {
#if !defined(_WIN32)
    if (lazy_atfork_status != 0) {
      throw std::system_error(lazy_atfork_status, std::generic_category(),
                              "pthread_atfork: the lazy blocks' fork handler could not be "
                              "registered");
    }
    const std::lock_guard<std::mutex> lock(lazy_builds_mutex);
    ++lazy_builds[mutex_];
#endif
  }
  LazyBuildRegistration(const LazyBuildRegistration&) = delete;
  LazyBuildRegistration& operator=(const LazyBuildRegistration&) = delete;
  ~LazyBuildRegistration() {
#if !defined(_WIN32)
    const std::lock_guard<std::mutex> lock(lazy_builds_mutex);
    const auto it = lazy_builds.find(mutex_);  // present: only a fork empties the registry
    if (--it->second == 0) {
      lazy_builds.erase(it);
    }
#endif
  }

 private:
  std::mutex* mutex_;
};

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
  ///         must not call `get` on this same `Lazy`, which would deadlock on its mutex, and must
  ///         not `fork`, since the child's fork handler replaces the mutex its caller holds.
  ///
  /// Returns:
  ///     const Block&: The stored block, valid for the lifetime of this `Lazy`.
  ///
  /// Raises:
  ///     std::system_error: The fork handlers could not be registered (first build only).
  ///     Whatever `compute` throws; nothing is stored in that case.
  template <class F>
    requires std::is_convertible_v<std::invoke_result_t<F&>, Block>
  const Block& get(F&& compute) const {
    if (!ready_.load(std::memory_order_acquire)) {
      const LazyBuildRegistration registration(mutex_);
      const std::lock_guard<std::mutex> lock(mutex_);
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
