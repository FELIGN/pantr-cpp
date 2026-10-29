/// @file tabulate_1d.hpp
/// @brief What every one-dimensional basis tabulation shares: argument checks and the point loop.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/parallel/parallel_for.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <string>

namespace pantr::basis::detail {

/// Refuse a negative degree.
///
/// Raises:
///     value_error: `degree < 0`, with the message the Python suite matches.
inline void check_degree(long long degree) {
  if (degree < 0) {
    throw value_error("degree must be non-negative");
  }
}

/// Refuse an output view whose extents differ from the expected ones.
///
/// Raises:
///     value_error: The extents differ; the message spells both shapes as Python tuples.
template <class T, std::size_t Rank>
void check_out(const mdview<T, Rank>& out, const std::array<std::size_t, Rank>& expected) {
  std::array<std::size_t, Rank> got{};
  for (std::size_t r = 0; r < Rank; ++r) {
    got[r] = out.extent(r);
  }
  if (got != expected) {
    throw value_error("Output array has shape " + py_repr(std::span<const std::size_t>(got)) +
                      ", but expected shape " + py_repr(std::span<const std::size_t>(expected)));
  }
}

/// Run `chunk(begin, end)` over the chunks of `[0, n)`, in parallel when the work warrants it.
///
/// The chunks hold `parallel::grain_for(ops, bytes)` items (a function of the problem only), and
/// run on the pool when `parallel::worth_parallel` says the operations pay for it, else in order
/// on the calling thread.
///
/// Args:
///     n (std::size_t): Number of items.
///     ops_per_item (double): Floating-point operations one item costs, derived from the kernel
///         (at least 1).
///     bytes_per_item (std::size_t): Bytes of output one item writes.
///     chunk (F&&): Callable as `chunk(std::size_t begin, std::size_t end)`; writes only the
///         output of its items.
template <class F>
void for_each_chunk(std::size_t n, double ops_per_item, std::size_t bytes_per_item, F&& chunk) {
  const parallel::WorkEstimate work{ops_per_item};
  const std::size_t grain = parallel::grain_for(work, bytes_per_item);
  if (parallel::worth_parallel(n, work)) {
    parallel::parallel_for(n, grain, chunk);
  } else {
    chunk(std::size_t{0}, n);
  }
}

/// Run `row(i)` for every point `i` in `[0, n_pts)`, chunked as `for_each_chunk`.
///
/// Args:
///     n_pts (std::size_t): Number of points.
///     ops_per_point (double): Floating-point operations one point costs (at least 1).
///     bytes_per_point (std::size_t): Bytes of output one point writes.
///     row (F&&): Callable as `row(std::size_t i)`; writes only point `i`'s output.
template <class F>
void for_each_point(std::size_t n_pts, double ops_per_point, std::size_t bytes_per_point, F&& row) {
  for_each_chunk(n_pts, ops_per_point, bytes_per_point, [&row](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      row(i);
    }
  });
}

}  // namespace pantr::basis::detail
