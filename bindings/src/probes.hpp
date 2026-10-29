// Probes shared by the _pantr extension and the test-only twin modules.
#pragma once

#include "pantr/parallel/parallel_for.hpp"

#include <chrono>
#include <cstddef>
#include <mutex>
#include <set>
#include <thread>

namespace pantr::bindings {

/// Run a parallel loop of `chunks` one-item chunks, each holding its thread for `hold_us`
/// microseconds, and count the distinct threads that ran a chunk.
///
/// The hold gives every thread the loop admits time to claim a chunk, so the count reaches the
/// calling thread's setting whenever there are enough chunks; it can never exceed it.
///
/// Args:
///     chunks (std::size_t): Number of chunks.
///     hold_us (std::size_t): Time each chunk holds its thread, in microseconds.
///
/// Returns:
///     std::size_t: Distinct threads that ran at least one chunk, the caller included.
///
/// Raises:
///     As `pantr::parallel::parallel_for`.
inline std::size_t count_threads_used(std::size_t chunks, std::size_t hold_us) {
  std::mutex mutex;
  std::set<std::thread::id> ids;
  pantr::parallel::parallel_for(chunks, std::size_t{1}, [&](std::size_t, std::size_t) {
    {
      const std::lock_guard<std::mutex> lock(mutex);
      ids.insert(std::this_thread::get_id());
    }
    std::this_thread::sleep_for(std::chrono::microseconds(hold_us));
  });
  return ids.size();
}

}  // namespace pantr::bindings
