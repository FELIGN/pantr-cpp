/// @file parallel_for.hpp
/// @brief `parallel_for`: a loop over `[0, n)` split into fixed chunks, run on the process pool.
///
/// The chunks are `[k * grain, min(n, (k + 1) * grain))`, a function of `n` and `grain` only. The
/// serial and the parallel paths run exactly the same chunks, so a body that writes disjoint
/// outputs per item gives the same result for every thread count. A body must not reduce across
/// chunks: the order in which chunks run is unspecified.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/parallel/pool.hpp"
#include "pantr/parallel/work_model.hpp"

// `(std::min)` throughout: a consumer may include <windows.h> without NOMINMAX first.

#include <algorithm>
#include <concepts>
#include <cstddef>

namespace pantr::parallel {

namespace detail {

/// Run chunk `k` of a loop over `[0, n)` with chunks of `grain` items.
template <class F>
void run_chunk(F& body, std::size_t n, std::size_t grain, std::size_t k) {
  const std::size_t begin = k * grain;
  const std::size_t end = (std::min)(n, begin + grain);
  body(begin, end);
}

}  // namespace detail

/// Run `body(begin, end)` over the chunks of `[0, n)`, in parallel when it can.
///
/// Runs on the calling thread alone when there is one chunk, when the calling thread's setting is
/// one thread, or when called from inside another parallel loop (no nested oversubscription, no
/// deadlock). Otherwise the calling thread and up to `get_num_threads() - 1` pool workers claim
/// chunks until none are left.
///
/// Args:
///     n (std::size_t): Number of items.
///     grain (std::size_t): Items per chunk; the last chunk may be shorter.
///     body (F&&): Callable as `body(std::size_t begin, std::size_t end)`; called once per chunk,
///         possibly concurrently from several threads.
///
/// Raises:
///     value_error: `grain` is 0.
///     The first exception a chunk threw, after every running chunk has finished; chunks not yet
///     started when it was thrown are skipped.
template <class F>
requires std::invocable<F&, std::size_t, std::size_t>
void parallel_for(std::size_t n, std::size_t grain, F&& body) {
  if (grain == 0) {
    throw value_error("grain must be >= 1, got 0");
  }
  if (n == 0) {
    return;
  }
  const std::size_t n_chunks = (n - 1) / grain + 1;
  const std::size_t threads = get_num_threads();
  if (n_chunks == 1 || threads == 1 || detail::tl_in_parallel_region) {
    for (std::size_t k = 0; k < n_chunks; ++k) {
      detail::run_chunk(body, n, grain, k);
    }
    return;
  }
  struct Context {
    F& body;
    std::size_t n;
    std::size_t grain;
  } ctx{body, n, grain};
  detail::Job job{n_chunks, &ctx,
                  [](void* p, std::size_t k) {
                    auto* c = static_cast<Context*>(p);
                    detail::run_chunk(c->body, c->n, c->grain, k);
                  },
                  (std::min)(threads - 1, n_chunks - 1)};
  Pool::instance().run(job);
}

/// Run `body(begin, end)` over `[0, n)` with the chunk size and the serial-or-parallel decision
/// taken from the work per item (see work_model.hpp).
///
/// Below `parallel_min_ops` total operations the chunks run serially on the calling thread; the
/// chunks are the same either way.
///
/// Args:
///     n (std::size_t): Number of items.
///     work (WorkEstimate): Operations per item, derived from the algorithm.
///     body (F&&): As for the overload taking a grain.
///
/// Raises:
///     value_error: `work.ops_per_item` is not a finite number >= 1.
///     The first exception a chunk threw (see the overload taking a grain).
template <class F>
requires std::invocable<F&, std::size_t, std::size_t>
void parallel_for(std::size_t n, WorkEstimate work, F&& body) {
  const std::size_t grain = grain_for(work);
  if (worth_parallel(n, work)) {
    parallel_for(n, grain, body);
    return;
  }
  const std::size_t n_chunks = n == 0 ? 0 : (n - 1) / grain + 1;
  for (std::size_t k = 0; k < n_chunks; ++k) {
    detail::run_chunk(body, n, grain, k);
  }
}

}  // namespace pantr::parallel
