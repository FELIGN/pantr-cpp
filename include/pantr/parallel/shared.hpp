/// @file shared.hpp
/// @brief One pool and one per-thread setting per process across pantr's extension modules.
///
/// The pool, the capacity cache and the per-thread settings are inline variables. A program built
/// as one binary has one copy of each. A process that loads several shared objects built from these
/// headers with their symbols kept local (every Python extension module: hidden visibility, and
/// separate DLLs on Windows) gets one copy per module: one pool each, and a thread count set
/// through one module that the others never see.
///
/// The fix is ownership. One module (pantr's `_pantr`) owns the state and hands out a table of
/// functions over it (`owner_shared_parallelism`), passed to the other modules as a Python capsule
/// named `kSharedParallelismCapsule`. Another module installs that table once, at import
/// (`attach_shared_parallelism`), and from then on its `parallel_for`, `max_threads`,
/// `get_num_threads` and `set_num_threads` act on the owner's pool and the owner's per-thread
/// settings. A program that never attaches a table, every plain C++ consumer included, uses its own
/// state as before.
///
/// Every function in the table is `noexcept` and reports failure by a status code and a message:
/// an exception thrown in one module and caught in another may fail to match its handler where the
/// two modules each carry the exception type's identity. A chunk's exception crosses only as a
/// `std::exception_ptr` stored in the job and rethrown by the module that started the loop.
///
/// Assumption, stated rather than checkable: the modules sharing a table are built with the same
/// compiler and standard library (pantr's extension modules are built together, from one source
/// tree, against one nanobind pin), since a `Job` holds `std::atomic` and `std::exception_ptr`
/// members that both sides read and write. What attach does check is the table's layout version and
/// size and the size and alignment of `Job`. Design note: design/threading.md.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>

namespace pantr::parallel {

namespace detail {

/// One parallel loop in flight: `n_chunks` chunks claimed through an atomic counter.
///
/// Lives on the calling thread's stack for the duration of `Pool::run`. Part of the table's ABI:
/// in a module attached to another, the owner's workers run the chunks of a job this module built.
/// Fields marked "pool mutex" are read and written only under the pool's mutex.
///
/// Attributes:
///     n_chunks (std::size_t): Number of chunks.
///     ctx (void*): The loop body, type-erased.
///     run_chunk (void (*)(void*, std::size_t)): Runs chunk `k` of the body at `ctx`.
///     max_workers (std::size_t): Pool workers allowed to join (the caller's setting minus one).
///     next (std::atomic<std::size_t>): Next chunk to claim; values `>= n_chunks` mean none left.
///     failed (std::atomic<bool>): Set by the first chunk that throws.
///     error (std::exception_ptr): That chunk's exception, written once by the thread that set
///         `failed`, read by the caller after the join.
///     joined (std::size_t): Workers that have joined so far (pool mutex).
///     active (std::size_t): Workers currently running chunks (pool mutex).
struct Job {
  std::size_t n_chunks;
  void* ctx;
  void (*run_chunk)(void*, std::size_t);
  std::size_t max_workers;
  std::atomic<std::size_t> next{0};
  std::atomic<bool> failed{false};
  std::exception_ptr error{};
  std::size_t joined = 0;
  std::size_t active = 0;

  /// Claim and run chunks until none are left; after a failure, claim the rest without running.
  void run_chunks() noexcept {
    for (;;) {
      const std::size_t k = next.fetch_add(1, std::memory_order_relaxed);
      if (k >= n_chunks) {
        return;
      }
      if (failed.load(std::memory_order_relaxed)) {
        continue;  // a chunk failed: drain the counter, run nothing more
      }
      try {
        run_chunk(ctx, k);
      } catch (...) {
        bool expected = false;
        if (failed.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
          error = std::current_exception();
        }
      }
    }
  }
};

}  // namespace detail

/// Layout version of `SharedParallelism` and of `detail::Job`, which crosses the table. Bump it
/// with any change to either, so that modules built from different headers refuse to share.
inline constexpr std::uint32_t kSharedParallelismAbi = 1;

/// Name of the Python capsule that carries a `SharedParallelism*` between extension modules.
inline constexpr const char* kSharedParallelismCapsule = "pantr.shared_parallelism.v1";

/// Status codes of the table's functions.
enum class SharedStatus : int {
  ok = 0,             ///< Success.
  value_error = 1,    ///< The owner raised `value_error` (e.g. a malformed `PANTR_NUM_THREADS`).
  runtime_error = 2,  ///< The owner raised anything else (e.g. a thread could not be started).
};

/// The owner's parallel state, as functions another module calls instead of its own copies.
///
/// Attributes:
///     abi (std::uint32_t): `kSharedParallelismAbi` of the owner's headers.
///     size (std::uint32_t): `sizeof(SharedParallelism)` in the owner.
///     job_size (std::uint32_t): `sizeof(detail::Job)` in the owner.
///     job_align (std::uint32_t): `alignof(detail::Job)` in the owner.
///     max_threads (int (*)(std::size_t*, char*, std::size_t) noexcept): Writes the owner's pool
///         capacity; on failure writes a message (truncated, null-terminated) and returns a
///         `SharedStatus`.
///     run (int (*)(detail::Job*, char*, std::size_t) noexcept): Runs every chunk of the job on the
///         owner's pool. A chunk's exception stays in the job for the caller to rethrow; a nonzero
///         status means the pool itself failed.
///     thread_setting (std::size_t* (*)() noexcept): The calling thread's parallelism setting in
///         the owner (0 meaning unset).
///     thread_in_region (bool* (*)() noexcept): The calling thread's in-parallel-region flag in the
///         owner.
struct SharedParallelism {
  std::uint32_t abi;
  std::uint32_t size;
  std::uint32_t job_size;
  std::uint32_t job_align;
  int (*max_threads)(std::size_t* out, char* message, std::size_t message_size) noexcept;
  int (*run)(detail::Job* job, char* message, std::size_t message_size) noexcept;
  std::size_t* (*thread_setting)() noexcept;
  bool* (*thread_in_region)() noexcept;
};

namespace detail {

/// The table this module routes through, or null when it uses its own state.
inline constinit std::atomic<const SharedParallelism*> shared_table{nullptr};

/// Size of the message buffer passed to the table's functions.
inline constexpr std::size_t kSharedMessageSize = 512;

/// Raise, in this module, the failure the owner reported.
///
/// Args:
///     status (int): A nonzero `SharedStatus`.
///     message (const char*): The owner's message.
///
/// Raises:
///     value_error: `status` is `SharedStatus::value_error`.
///     std::runtime_error: Any other status.
[[noreturn]] inline void raise_shared_failure(int status, const char* message) {
  if (status == static_cast<int>(SharedStatus::value_error)) {
    throw value_error(message);
  }
  throw std::runtime_error(message);
}

/// Copy `text` into `buffer` (truncated, null-terminated).
inline void write_shared_message(const char* text, char* buffer, std::size_t size) noexcept {
  if (size == 0) {
    return;
  }
  std::size_t i = 0;
  for (; i + 1 < size && text[i] != '\0'; ++i) {
    buffer[i] = text[i];
  }
  buffer[i] = '\0';
}

}  // namespace detail

/// Route this module's parallel state through another module's table.
///
/// Called once, when an extension module that is not the owner is imported. Attaching the same
/// table again does nothing.
///
/// Args:
///     table (const SharedParallelism*): The owner's table (from its capsule).
///
/// Raises:
///     value_error: `table` is null.
///     state_error: The table's layout version or size, or the size or alignment of a job, differs
///         from this module's headers, or a
///         different table is already attached.
inline void attach_shared_parallelism(const SharedParallelism* table) {
  if (table == nullptr) {
    throw value_error("the shared-parallelism table is null");
  }
  if (table->abi != kSharedParallelismAbi || table->size != sizeof(SharedParallelism) ||
      table->job_size != sizeof(detail::Job) || table->job_align != alignof(detail::Job)) {
    throw state_error(
        "pantr extension modules were built from different headers: the owner's "
        "shared-parallelism layout is version " +
        std::to_string(table->abi) + " (size " + std::to_string(table->size) +
        "), this module's is version " + std::to_string(kSharedParallelismAbi) + " (size " +
        std::to_string(sizeof(SharedParallelism)) + ")");
  }
  const SharedParallelism* expected = nullptr;
  if (!detail::shared_table.compare_exchange_strong(expected, table, std::memory_order_acq_rel) &&
      expected != table) {
    throw state_error("a different shared-parallelism table is already attached");
  }
}

/// Report whether this module routes its parallel state through another module.
///
/// Returns:
///     bool: True after `attach_shared_parallelism`.
[[nodiscard]] inline bool routes_shared_parallelism() noexcept {
  return detail::shared_table.load(std::memory_order_acquire) != nullptr;
}

}  // namespace pantr::parallel
