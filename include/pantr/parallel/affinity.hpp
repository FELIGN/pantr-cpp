/// @file affinity.hpp
/// @brief How many CPUs this process may run on, and the pool capacity derived from it.
///
/// The pool is sized from the process's CPU affinity, never from
/// `std::thread::hardware_concurrency()`, which reports the machine's CPUs and ignores a
/// restricted affinity mask (a shared server, `taskset`, a batch scheduler's CPU set).
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/parallel/shared.hpp"

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <string_view>
#include <system_error>

#if defined(__linux__)
#include <sched.h>

#include <new>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#define PANTR_DETAIL_UNDEF_NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#define PANTR_DETAIL_UNDEF_WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#ifdef PANTR_DETAIL_UNDEF_NOMINMAX
#undef NOMINMAX
#undef PANTR_DETAIL_UNDEF_NOMINMAX
#endif
#ifdef PANTR_DETAIL_UNDEF_WIN32_LEAN_AND_MEAN
#undef WIN32_LEAN_AND_MEAN
#undef PANTR_DETAIL_UNDEF_WIN32_LEAN_AND_MEAN
#endif
#else
#error "pantr: no CPU affinity query is implemented for this platform"
#endif

namespace pantr::parallel {

/// Count the CPUs the calling process may run on.
///
/// Linux: the popcount of `sched_getaffinity` (the mask is grown until the kernel accepts its
/// size, so machines with more than 1024 CPUs are counted). macOS, which has no affinity masks:
/// `hw.activecpu`. Windows: the popcount of `GetProcessAffinityMask`, which covers the process's
/// processor group only (at most 64 CPUs).
///
/// Returns:
///     std::size_t: The count; at least 1, since a running process is allowed at least one CPU.
///
/// Raises:
///     std::system_error: The operating system query failed.
[[nodiscard]] inline std::size_t affinity_cpu_count() {
#if defined(__linux__)
  // EINVAL means the mask is smaller than the kernel's; any other errno is a real failure.
  for (std::size_t ncpu = 1024; ncpu <= (std::size_t{1} << 22); ncpu *= 2) {
    cpu_set_t* set = CPU_ALLOC(ncpu);
    if (set == nullptr) {
      throw std::bad_alloc();
    }
    const std::size_t bytes = CPU_ALLOC_SIZE(ncpu);
    CPU_ZERO_S(bytes, set);
    const int rc = sched_getaffinity(0, bytes, set);
    const int err = errno;
    const int count = rc == 0 ? CPU_COUNT_S(bytes, set) : 0;
    CPU_FREE(set);
    if (rc == 0) {
      return static_cast<std::size_t>(count);
    }
    if (err != EINVAL) {
      throw std::system_error(err, std::generic_category(), "sched_getaffinity");
    }
  }
  throw std::system_error(EINVAL, std::generic_category(),
                          "sched_getaffinity: no mask size up to 2^22 CPUs was accepted");
#elif defined(__APPLE__)
  int count = 0;
  std::size_t len = sizeof(count);
  if (sysctlbyname("hw.activecpu", &count, &len, nullptr, 0) != 0) {
    throw std::system_error(errno, std::generic_category(), "sysctlbyname(hw.activecpu)");
  }
  return static_cast<std::size_t>(count);
#elif defined(_WIN32)
  DWORD_PTR process_mask = 0;
  DWORD_PTR system_mask = 0;
  if (!GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)) {
    throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
                            "GetProcessAffinityMask");
  }
  std::size_t count = 0;
  for (; process_mask != 0; process_mask &= process_mask - 1) {
    ++count;
  }
  return count;
#endif
}

namespace detail {

/// Parse the value of `PANTR_NUM_THREADS`.
///
/// Args:
///     value (const char*): The variable's value, or null when it is unset.
///
/// Returns:
///     std::size_t: The requested thread count, or 0 when the variable is unset or empty.
///
/// Raises:
///     value_error: The value is not a positive decimal integer.
[[nodiscard]] inline std::size_t parse_num_threads_env(const char* value) {
  if (value == nullptr || *value == '\0') {
    return 0;
  }
  const std::string_view text(value);
  std::size_t n = 0;
  for (const char c : text) {
    if (c < '0' || c > '9' || n > (static_cast<std::size_t>(-1) - 9) / 10) {
      throw value_error("PANTR_NUM_THREADS must be a positive integer, got '" + std::string(text) +
                        "'");
    }
    n = n * 10 + static_cast<std::size_t>(c - '0');
  }
  if (n == 0) {
    throw value_error("PANTR_NUM_THREADS must be a positive integer, got '" + std::string(text) +
                      "'");
  }
  return n;
}

/// Capacity from the affinity count and the parsed `PANTR_NUM_THREADS` (0 meaning unset).
///
/// Args:
///     affinity (std::size_t): CPUs the process may run on.
///     requested (std::size_t): Parsed `PANTR_NUM_THREADS`, 0 when unset.
///
/// Returns:
///     std::size_t: `affinity` when unset, else `min(affinity, requested)`: the variable can lower
///     the capacity, never raise it above the CPUs the process may use.
[[nodiscard]] constexpr std::size_t capacity_from(std::size_t affinity,
                                                  std::size_t requested) noexcept {
  return requested == 0 || requested > affinity ? affinity : requested;
}

/// Cached pool capacity; 0 until the first `max_threads()` call computes it.
inline constinit std::atomic<std::size_t> capacity_cache{0};

/// This module's pool capacity, ignoring any attached table (see `max_threads`).
///
/// Returns:
///     std::size_t: At least 1.
///
/// Raises:
///     value_error: `PANTR_NUM_THREADS` is set to something other than a positive integer.
///     std::system_error: The affinity query failed.
[[nodiscard]] inline std::size_t local_max_threads() {
  // A constant-initialized atomic rather than a function-local static: a static's initialization
  // guard could be inherited mid-initialization by a forked child, which would then block on it
  // forever. Concurrent first calls compute the same value; a call that throws stores nothing.
  std::size_t cap = capacity_cache.load(std::memory_order_acquire);
  if (cap == 0) {
    cap = capacity_from(affinity_cpu_count(),
                        parse_num_threads_env(std::getenv("PANTR_NUM_THREADS")));
    capacity_cache.store(cap, std::memory_order_release);
  }
  return cap;
}

}  // namespace detail

/// Get the pool capacity: the most threads a pantr parallel call may use.
///
/// The affinity count, lowered by the environment variable `PANTR_NUM_THREADS` when it is set.
/// Read on the first call in the process (and possibly by concurrent first calls); later changes to
/// the affinity or the environment are not seen. In a module attached to another's shared state
/// (shared.hpp), the owner's capacity.
///
/// Returns:
///     std::size_t: At least 1.
///
/// Raises:
///     value_error: `PANTR_NUM_THREADS` is set to something other than a positive integer.
///     std::system_error: The affinity query failed (the owner's failure, in an attached module, is
///         raised as `std::runtime_error` with the owner's message).
[[nodiscard]] inline std::size_t max_threads() {
  if (const SharedParallelism* shared = detail::shared_table.load(std::memory_order_acquire)) {
    std::size_t out = 0;
    char message[detail::kSharedMessageSize];
    const int status = shared->max_threads(&out, message, sizeof(message));
    if (status != 0) {
      detail::raise_shared_failure(status, message);
    }
    return out;
  }
  return detail::local_max_threads();
}

}  // namespace pantr::parallel
