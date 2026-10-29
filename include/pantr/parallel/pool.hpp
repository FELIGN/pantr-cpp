/// @file pool.hpp
/// @brief The process thread pool and the calling thread's parallelism setting.
///
/// One pool per process (strictly, per copy of these headers' inline variables: see the note
/// below), created on first use with `max_threads() - 1` workers: the thread that
/// calls a parallel loop always works on it too, so a call never waits for a free worker, and
/// `max_threads()` threads run at full parallelism. Concurrent calls from several threads share the
/// workers; a parallel loop started from inside a running one runs inline on its thread.
///
/// The parallelism of a call is the calling thread's own setting (`set_num_threads`), as with
/// Numba: setting it in one thread does not change another's.
///
/// Fork safety: a forked child inherits the pool's memory but none of its threads. The pool
/// records the process id that created it; a child that calls a parallel loop finds a different
/// id, leaves the inherited pool untouched (its mutexes may have been held at the fork) and builds
/// its own. The holder is constant-initialized and its fork handler registered by dynamic
/// initialization, so no static-initialization guard or creation mutex is inherited mid-update.
/// The pool is never destroyed (see `detail::PoolHolder`).
///
/// Note: the pool, the holder and the thread-local settings are inline variables. Shared objects
/// that each include these headers and keep their symbols local (hidden visibility, or `dlopen`
/// with `RTLD_LOCAL` on toolchains without GNU unique symbols) each get their own copy. pantr's
/// extension modules share one pool by routing through the owner's table (shared.hpp): in a module
/// that attached one, `parallel_for` and the thread settings act on the owner's state, and
/// `Pool::instance()` refuses. Design note: design/threading.md.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/parallel/affinity.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <pthread.h>
#include <unistd.h>
#endif

namespace pantr::parallel {

namespace detail {

/// The calling thread's parallelism setting; 0 means "not set", i.e. `max_threads()`.
inline thread_local std::size_t tl_num_threads = 0;

/// True while the thread executes chunks of a parallel loop (always, on a pool worker).
inline thread_local bool tl_in_parallel_region = false;

/// The calling thread's parallelism setting: the owner's when attached (shared.hpp), else this
/// module's `tl_num_threads`.
[[nodiscard]] inline std::size_t& thread_setting() noexcept {
  if (const SharedParallelism* shared = shared_table.load(std::memory_order_acquire)) {
    return *shared->thread_setting();
  }
  return tl_num_threads;
}

/// The calling thread's in-parallel-region flag: the owner's when attached, else this module's.
[[nodiscard]] inline bool in_parallel_region() noexcept {
  if (const SharedParallelism* shared = shared_table.load(std::memory_order_acquire)) {
    return *shared->thread_in_region();
  }
  return tl_in_parallel_region;
}

/// Process id, used to detect that the pool was inherited through `fork`.
[[nodiscard]] inline std::int64_t current_pid() noexcept {
#if defined(_WIN32)
  return 0;  // no fork on Windows
#else
  return static_cast<std::int64_t>(::getpid());
#endif
}

/// One parallel loop in flight: `n_chunks` chunks claimed through an atomic counter.
///
/// Lives on the calling thread's stack for the duration of `Pool::run`. Fields marked "pool mutex"
/// are read and written only under the pool's mutex.
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

/// Sets `tl_in_parallel_region` for a scope and restores the previous value.
class RegionGuard {
 public:
  RegionGuard() noexcept : previous_(tl_in_parallel_region) { tl_in_parallel_region = true; }
  RegionGuard(const RegionGuard&) = delete;
  RegionGuard& operator=(const RegionGuard&) = delete;
  ~RegionGuard() { tl_in_parallel_region = previous_; }

 private:
  bool previous_;
};

}  // namespace detail

/// A fixed set of worker threads executing the chunks of parallel loops.
///
/// Use `Pool::instance()`; constructing another pool is allowed (tests do) but is never needed.
///
/// Attributes:
///     mutex_ (std::mutex): Guards `jobs_`, `stop_` and every job's `joined`/`active`.
///     work_cv_ (std::condition_variable): Signals workers that a job was posted or `stop_` set.
///     done_cv_ (std::condition_variable): Signals callers that a job's last worker left.
///     jobs_ (std::vector<detail::Job*>): Jobs open to workers, oldest first.
///     stop_ (bool): Set by the destructor; workers exit.
///     pid_ (std::int64_t): Process that created the pool.
///     threads_ (std::vector<std::thread>): The workers.
///     inherited_ (Pool*): The pool this one replaced after a `fork`, kept reachable, never
///         touched; null otherwise.
class Pool {
 public:
  /// Start `workers` worker threads.
  ///
  /// Args:
  ///     workers (std::size_t): Number of worker threads; 0 gives a pool that runs every loop on
  ///         the calling thread.
  ///
  /// Raises:
  ///     std::system_error: A thread could not be started (the started ones are joined first).
  explicit Pool(std::size_t workers) : pid_(detail::current_pid()) {
    threads_.reserve(workers);
    try {
      for (std::size_t i = 0; i < workers; ++i) {
        threads_.emplace_back([this] { worker_loop(); });
      }
    } catch (...) {
      shutdown();
      throw;
    }
  }

  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;

  /// Stop and join the workers. No loop may be running on this pool.
  ~Pool() { shutdown(); }

  /// Get the number of worker threads.
  ///
  /// Returns:
  ///     std::size_t: Workers, not counting the threads that call `run`.
  [[nodiscard]] std::size_t workers() const noexcept { return threads_.size(); }

  /// Get the id of the process that created the pool.
  ///
  /// Returns:
  ///     std::int64_t: The creator's process id (0 on Windows).
  [[nodiscard]] std::int64_t creator_pid() const noexcept { return pid_; }

  /// Run every chunk of `job`, on the calling thread and up to `job.max_workers` workers.
  ///
  /// Returns after every claimed chunk has run (after a throw, unstarted chunks are skipped) and
  /// every worker has left the job.
  ///
  /// Args:
  ///     job (detail::Job&): The loop; must outlive the call.
  ///
  /// Raises:
  ///     The exception of the first chunk that threw, after the join.
  void run(detail::Job& job) {
    execute(job);
    if (job.error) {
      std::rethrow_exception(job.error);
    }
  }

  /// Run every chunk of `job` as `run` does, leaving a chunk's exception in `job.error`.
  ///
  /// Args:
  ///     job (detail::Job&): The loop; must outlive the call.
  ///
  /// Raises:
  ///     std::system_error: Locking the pool's mutex failed.
  void execute(detail::Job& job) {
    const std::size_t wanted = (std::min)(job.max_workers, threads_.size());
    if (wanted > 0) {
      {
        const std::lock_guard<std::mutex> lock(mutex_);
        jobs_.push_back(&job);
      }
      // When the job admits every worker, one broadcast is cheapest. Otherwise wake only as many
      // as it admits, so the others do not contend for the mutex just to go back to sleep.
      if (wanted == threads_.size()) {
        work_cv_.notify_all();
      } else {
        for (std::size_t i = 0; i < wanted; ++i) {
          work_cv_.notify_one();
        }
      }
    }
    {
      const detail::RegionGuard region;
      job.run_chunks();
    }
    {
      std::unique_lock<std::mutex> lock(mutex_);
      // No chunk is left to claim, so no new worker may join; wait for the ones that did.
      jobs_.erase(std::remove(jobs_.begin(), jobs_.end(), &job), jobs_.end());
      done_cv_.wait(lock, [&job] { return job.active == 0; });
    }
  }

  /// Get the process pool, creating it on first use (and again in a forked child).
  ///
  /// Returns:
  ///     Pool&: The pool of the calling process, with `max_threads() - 1` workers.
  ///
  /// Raises:
  ///     value_error: `PANTR_NUM_THREADS` is malformed (see `max_threads`).
  ///     state_error: This module routes its parallel work through another module's pool
  ///         (shared.hpp); use `parallel_for`.
  ///     std::system_error: The affinity query or a thread start failed, or the fork handler could
  ///         not be registered.
  static Pool& instance();

  /// Get this module's own pool, ignoring any attached table (see `instance`).
  ///
  /// Returns:
  ///     Pool&: This module's pool, created on first use (and again in a forked child).
  ///
  /// Raises:
  ///     As `instance`, except `state_error`.
  static Pool& local_instance();

 private:
  void shutdown() noexcept {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    work_cv_.notify_all();
    for (std::thread& t : threads_) {
      t.join();
    }
    threads_.clear();
  }

  detail::Job* find_joinable() const noexcept {
    for (detail::Job* job : jobs_) {
      if (job->joined < job->max_workers &&
          job->next.load(std::memory_order_relaxed) < job->n_chunks) {
        return job;
      }
    }
    return nullptr;
  }

  void worker_loop() {
    detail::tl_in_parallel_region = true;  // a loop started from a chunk runs inline
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      detail::Job* job = nullptr;
      work_cv_.wait(lock, [&] { return stop_ || (job = find_joinable()) != nullptr; });
      if (stop_) {
        return;
      }
      ++job->joined;
      ++job->active;
      lock.unlock();
      job->run_chunks();
      lock.lock();
      if (--job->active == 0) {
        done_cv_.notify_all();
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable work_cv_;
  std::condition_variable done_cv_;
  std::vector<detail::Job*> jobs_;
  bool stop_ = false;
  std::int64_t pid_;
  std::vector<std::thread> threads_;
  Pool* inherited_ = nullptr;
};

namespace detail {

/// Owner of the process pool. Constant-initialized (no static-initialization guard), so a forked
/// child can never inherit it half-constructed.
///
/// The pool is deliberately never destroyed: its workers stay blocked on their condition variable
/// until the process ends. Destroying it at exit would race every static destructor, in any
/// translation unit, that still runs a parallel loop (their order relative to this object is
/// unspecified), and joining idle workers buys nothing at exit.
///
/// Attributes:
///     mutex (std::mutex): Serializes pool creation.
///     current (std::atomic<Pool*>): The pool of the process that owns it, or null.
struct PoolHolder {
  std::mutex mutex;
  std::atomic<Pool*> current{nullptr};

  constexpr PoolHolder() noexcept = default;
  PoolHolder(const PoolHolder&) = delete;
  PoolHolder& operator=(const PoolHolder&) = delete;
  ~PoolHolder() = default;
};

/// The process's pool holder.
inline constinit PoolHolder pool_holder;

#if !defined(_WIN32)
/// Registered by dynamic initialization (in practice when the library is loaded; at the latest
/// before `Pool::instance` reads it): `pool_holder.mutex` is held by the forking thread across
/// `fork` and released in both processes, so a child never inherits it locked by a thread it
/// lacks. Nonzero means registration failed, which `Pool::instance` reports.
inline const int pool_atfork_status =
    ::pthread_atfork([] { pool_holder.mutex.lock(); }, [] { pool_holder.mutex.unlock(); },
                     [] { pool_holder.mutex.unlock(); });
#endif

}  // namespace detail

inline Pool& Pool::instance() {
  if (routes_shared_parallelism()) {
    throw state_error(
        "this module runs parallel loops on another module's pool; "
        "Pool::instance() would start a second pool");
  }
  return local_instance();
}

inline Pool& Pool::local_instance() {
  detail::PoolHolder& h = detail::pool_holder;
#if !defined(_WIN32)
  // Reading the status here also odr-uses it, so its registration cannot be deferred past this.
  if (detail::pool_atfork_status != 0) {
    throw std::system_error(detail::pool_atfork_status, std::generic_category(),
                            "pthread_atfork: the pool's fork handler could not be registered");
  }
#endif
  Pool* p = h.current.load(std::memory_order_acquire);
  if (p != nullptr && p->pid_ == detail::current_pid()) {
    return *p;
  }
  const std::lock_guard<std::mutex> lock(h.mutex);
  p = h.current.load(std::memory_order_relaxed);
  if (p == nullptr || p->pid_ != detail::current_pid()) {
    // In a forked child the inherited pool's threads do not exist and its mutex may be locked:
    // never touch it. The new pool keeps a pointer to it, so it stays reachable (no leak report)
    // and is never destroyed.
    Pool* inherited = p;
    p = new Pool(detail::local_max_threads() - 1);
    p->inherited_ = inherited;
    h.current.store(p, std::memory_order_release);
  }
  return *p;
}

/// Set the calling thread's parallelism for pantr's parallel loops.
///
/// Args:
///     n (std::int64_t): Threads a loop called from this thread may use, the caller included.
///
/// Raises:
///     value_error: `n < 1` or `n > max_threads()`, or `PANTR_NUM_THREADS` is malformed.
///     std::system_error: The affinity query failed (see `max_threads`).
inline void set_num_threads(std::int64_t n) {
  if (n < 1) {
    throw value_error("n must be >= 1, got " + std::to_string(n));
  }
  const std::size_t cap = max_threads();
  if (static_cast<std::uint64_t>(n) > cap) {
    throw value_error("n must be <= the pool capacity (" + std::to_string(cap) + "), got " +
                      std::to_string(n));
  }
  detail::thread_setting() = static_cast<std::size_t>(n);
}

/// Get the calling thread's parallelism for pantr's parallel loops.
///
/// Returns:
///     std::size_t: The value last set on this thread, else `max_threads()`.
///
/// Raises:
///     value_error: Nothing was set on this thread and `PANTR_NUM_THREADS` is malformed.
///     std::system_error: Nothing was set on this thread and the affinity query failed.
[[nodiscard]] inline std::size_t get_num_threads() {
  const std::size_t setting = detail::thread_setting();
  return setting != 0 ? setting : max_threads();
}

namespace detail {

/// Run `job` on the process pool: the owner's when attached (shared.hpp), else this module's.
///
/// Args:
///     job (Job&): The loop; must outlive the call.
///
/// Raises:
///     The exception of the first chunk that threw, after the join; the errors of
///     `Pool::instance()` (in an attached module, the owner's, see `raise_shared_failure`).
inline void run_job(Job& job) {
  if (const SharedParallelism* shared = shared_table.load(std::memory_order_acquire)) {
    char message[kSharedMessageSize];
    const int status = shared->run(&job, message, sizeof(message));
    if (status != 0) {
      raise_shared_failure(status, message);
    }
    if (job.error) {
      std::rethrow_exception(job.error);
    }
    return;
  }
  Pool::instance().run(job);
}

/// Report a caught exception through a table function's status and message.
inline int shared_failure(char* message, std::size_t size) noexcept {
  try {
    throw;
  } catch (const value_error& e) {
    write_shared_message(e.what(), message, size);
    return static_cast<int>(SharedStatus::value_error);
  } catch (const std::exception& e) {
    write_shared_message(e.what(), message, size);
  } catch (...) {
    write_shared_message("unknown error in the pool owner", message, size);
  }
  return static_cast<int>(SharedStatus::runtime_error);
}

}  // namespace detail

/// Get the table over this module's own parallel state, for other modules to attach.
///
/// Only the owning module calls this (pantr's `_pantr`); its functions always serve the module
/// they were compiled into, whether or not that module itself attached a table.
///
/// Returns:
///     const SharedParallelism&: The table, valid for the life of the process.
[[nodiscard]] inline const SharedParallelism& owner_shared_parallelism() noexcept {
  static constexpr SharedParallelism table{
      kSharedParallelismAbi,
      static_cast<std::uint32_t>(sizeof(SharedParallelism)),
      [](std::size_t* out, char* message, std::size_t size) noexcept -> int {
        try {
          *out = detail::local_max_threads();
          return 0;
        } catch (...) {
          return detail::shared_failure(message, size);
        }
      },
      [](detail::Job* job, char* message, std::size_t size) noexcept -> int {
        try {
          Pool::local_instance().execute(*job);
          return 0;
        } catch (...) {
          return detail::shared_failure(message, size);
        }
      },
      []() noexcept -> std::size_t* { return &detail::tl_num_threads; },
      []() noexcept -> bool* { return &detail::tl_in_parallel_region; },
  };
  return table;
}

}  // namespace pantr::parallel
