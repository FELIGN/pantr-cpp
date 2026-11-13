/// @file work_model.hpp
/// @brief From a per-item work estimate to a chunk size and a serial-or-parallel decision.
///
/// A parallel kernel states how much work one item costs, in elementary floating-point operations
/// derived from its algorithm (for example `(p + 1)^2` per point for Cox-de Boor of degree `p`).
/// From that, `grain_for` gives the chunk size and `worth_parallel` decides whether waking the pool
/// pays at all. Both depend on the problem only, never on the thread count, which is what makes a
/// parallel loop's chunk boundaries, and so its results, independent of the number of threads.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"

#include <cmath>
#include <cstddef>

namespace pantr::parallel {

/// Work per item of a parallel loop, in elementary floating-point operations.
///
/// Attributes:
///     ops_per_item (double): Operations one item costs, derived from the algorithm; at least 1,
///         since even an empty body costs one loop iteration.
struct WorkEstimate {
  double ops_per_item;
};

/// Operations one chunk should carry. An admitted heuristic, not a measurement: claiming a chunk
/// is one atomic increment on a contended cache line, of the order of 10^2 cycles, and a chunk
/// writes at least one cache line of output; about 10^4 operations per chunk is two orders of
/// magnitude more work than that overhead at one operation per cycle (less margin for a kernel
/// that retires several per cycle). 2^14 is that order of magnitude. Never tuned on a particular
/// machine.
inline constexpr double chunk_ops = 16384.0;

/// Operations below which a loop runs serially. An admitted heuristic, not a measurement: waking
/// pool workers and joining them costs futex system calls and cross-core wake-ups, of the order of
/// 10^4 cycles; a job of about ten times that is an order of magnitude more work than the
/// overhead at one operation per cycle. 2^17 is that order of magnitude. Never tuned on a
/// particular machine.
inline constexpr double parallel_min_ops = 131072.0;

/// Bytes of output a chunk of a memory-bound loop should write. An admitted heuristic, not a
/// measurement: a loop whose items cost a few operations each is limited by memory traffic, not by
/// `chunk_ops`, and a chunk sized by its operations alone then streams only a few kilobytes, too
/// little to amortize a claim and the hardware prefetchers' ramp-up on a new stream. 2^18 bytes is
/// the order of a core's L2 cache, the scale at which a streaming chunk's start-up is a small
/// fraction of it. Never tuned on a particular machine.
inline constexpr double chunk_bytes = 262144.0;

namespace detail {

inline void check_work(WorkEstimate w) {
  if (!(w.ops_per_item >= 1.0) || !std::isfinite(w.ops_per_item)) {
    throw value_error("ops_per_item must be a finite number >= 1, got " + py_repr(w.ops_per_item));
  }
}

}  // namespace detail

/// Chunk size for a loop whose items each cost `w`: `max(1, ceil(chunk_ops / ops_per_item))`.
///
/// The quotient is a rounded double, so where the exact quotient sits within a rounding error of
/// an integer the chunk can be one item smaller than the exact ceiling. The chunk size is an
/// admitted heuristic (see `chunk_ops`); one item either way changes no result, since the chunks
/// are a function of `n` and the grain only.
///
/// Args:
///     w (WorkEstimate): Work per item.
///
/// Returns:
///     std::size_t: Items per chunk, between 1 and `chunk_ops`.
///
/// Raises:
///     value_error: `w.ops_per_item` is not a finite number >= 1.
[[nodiscard]] inline std::size_t grain_for(WorkEstimate w) {
  detail::check_work(w);
  // ops_per_item >= 1 bounds the quotient by chunk_ops, and a positive quotient's ceiling is >= 1.
  return static_cast<std::size_t>(std::ceil(chunk_ops / w.ops_per_item));
}

/// Chunk size for a loop whose items each cost `w` and write `bytes_per_item` bytes of output:
/// the larger of `grain_for(w)` and `ceil(chunk_bytes / bytes_per_item)`, so a cheap item writing
/// little still gets a chunk that streams `chunk_bytes`.
///
/// Args:
///     w (WorkEstimate): Work per item.
///     bytes_per_item (std::size_t): Bytes of output one item writes; 0 adds no bound.
///
/// Returns:
///     std::size_t: Items per chunk, at least 1.
///
/// Raises:
///     value_error: `w.ops_per_item` is not a finite number >= 1.
[[nodiscard]] inline std::size_t grain_for(WorkEstimate w, std::size_t bytes_per_item) {
  const std::size_t by_ops = grain_for(w);
  if (bytes_per_item == 0) {
    return by_ops;
  }
  const auto by_bytes =
      static_cast<std::size_t>(std::ceil(chunk_bytes / static_cast<double>(bytes_per_item)));
  return by_ops > by_bytes ? by_ops : by_bytes;
}

/// Decide whether a loop of `n` items costing `w` each is worth running in parallel.
///
/// Args:
///     n (std::size_t): Number of items.
///     w (WorkEstimate): Work per item.
///
/// Returns:
///     bool: True when `n * ops_per_item >= parallel_min_ops`.
///
/// Raises:
///     value_error: `w.ops_per_item` is not a finite number >= 1.
[[nodiscard]] inline bool worth_parallel(std::size_t n, WorkEstimate w) {
  detail::check_work(w);
  return static_cast<double>(n) * w.ops_per_item >= parallel_min_ops;
}

}  // namespace pantr::parallel
