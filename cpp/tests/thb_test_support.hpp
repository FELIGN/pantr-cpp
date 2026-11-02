// What the THB test files (test_thb_space.cpp, test_thb_refine.cpp) share: the aliases and knot
// vectors both build their spaces from, the rounding model every comparison is bounded by, the
// thread-count helpers of the determinism tests, and the helpers that read a space's values.
#pragma once

#include <doctest/doctest.h>

#include <pantr/core/mdspan.hpp>
#include <pantr/parallel/parallel_for.hpp>
#include <pantr/thb/space.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <thread>
#include <vector>

// The functions under test are [[nodiscard]]; a throw check discards the value on purpose.
#define THB_THROWS_AS(expr, ex) CHECK_THROWS_AS(static_cast<void>(expr), ex)
#define THB_THROWS_WITH_AS(expr, msg, ex) CHECK_THROWS_WITH_AS(static_cast<void>(expr), msg, ex)
#define THB_NOTHROW(expr) CHECK_NOTHROW(static_cast<void>(expr))

namespace thb_test {

namespace par = pantr::parallel;

using I64 = std::int64_t;
using Ivec = std::vector<I64>;
using Reg = std::vector<std::optional<int>>;

inline constexpr double eps = std::numeric_limits<double>::epsilon();

/// The slack of a floating-point build over a rounding model (FMA contraction, vector width): the
/// factor of four `pantr::tolerance` gives its default tier for the same reason.
inline constexpr double build_slack = 4.0;

// ---- Knot vectors ----------------------------------------------------------------------------

/// Degree 2, four equal spans on [0, 1].
inline const std::vector<double> k2 = {0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0};
/// Degree 2, four unequal spans on [0, 1].
inline const std::vector<double> k2n = {0.0, 0.0, 0.0, 0.3, 0.55, 0.8, 1.0, 1.0, 1.0};
/// Degree 2, two equal spans on [0, 1].
inline const std::vector<double> k2_3d = {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};

// ---- Threads ---------------------------------------------------------------------------------

/// Restores the calling thread's parallelism setting on scope exit.
class ThreadsGuard {
 public:
  explicit ThreadsGuard(std::size_t n) : previous_(par::get_num_threads()) {
    par::set_num_threads(static_cast<std::int64_t>(n));
  }
  ThreadsGuard(const ThreadsGuard&) = delete;
  ThreadsGuard& operator=(const ThreadsGuard&) = delete;
  ~ThreadsGuard() { par::set_num_threads(static_cast<std::int64_t>(previous_)); }

 private:
  std::size_t previous_;
};

/// The many-thread side of a determinism comparison: eight threads, capped by the CPU mask.
inline std::size_t eight_threads() { return (std::min)(std::size_t{8}, par::max_threads()); }

/// Check that the many-thread side of a comparison really runs on more than one thread, where the
/// capacity allows it.
///
/// With one CPU both sides are serial and their agreement says nothing about the pool, which is
/// reported with a message instead. Otherwise a probe loop under `ThreadsGuard(eight_threads())`
/// has every chunk wait (bounded) until two distinct threads have run one, which a pool that
/// never wakes a worker cannot provide.
inline void check_many_threads_are_used() {
  if (par::max_threads() <= 1) {
    MESSAGE(
        "max_threads() == 1: the many-thread side is serial, so the comparison does not "
        "exercise the pool");
    return;
  }
  const std::size_t threads = eight_threads();
  CHECK(threads > 1);
  const ThreadsGuard guard(threads);
  std::mutex m;
  std::set<std::thread::id> ids;
  std::atomic<bool> two_seen{false};
  // One deadline for the loop, so a pool that never wakes a worker fails in seconds.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  par::parallel_for(4 * threads, std::size_t{1}, [&](std::size_t, std::size_t) {
    {
      const std::lock_guard<std::mutex> lock(m);
      ids.insert(std::this_thread::get_id());
      if (ids.size() >= 2) {
        two_seen = true;
      }
    }
    while (!two_seen.load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
  });
  CHECK(ids.size() > 1);
}

// ---- Reading a space -------------------------------------------------------------------------

/// A bound on the relative rounding error of one hierarchical function's value, in units of `eps`,
/// all terms being non-negative: `2 p` for a 1D Cox-de Boor recurrence (`p` combinations, two
/// operations each) in each of the `D` directions, `D` for their product, and for a truncated
/// function the size of its coefficient box for the sum (taken at the largest box of the space).
template <class T>
double value_error_units(const pantr::thb::THBSplineSpace<T>& space) {
  const int p = *std::max_element(space.degrees().begin(), space.degrees().end());
  const auto d = static_cast<double>(space.dim());
  std::size_t box = 1;
  for (I64 dof = 0; dof < space.num_total_basis(); ++dof) {
    if (const auto* f = space.truncated(dof)) {
      box = (std::max)(box, f->coeffs.size());
    }
  }
  return 2.0 * p * d + d + static_cast<double>(box);
}

/// Points of cell `cid`: `lo + t * (hi - lo)` for `t` in `ts` along every axis (the product grid
/// of the parameters, C order).
template <class T>
std::vector<double> cell_points(const pantr::thb::THBSplineSpace<T>& space, I64 cid,
                                const std::vector<double>& ts) {
  const auto [lo, hi] = space.grid().cell_bounds(cid);
  const std::size_t d = space.dim();
  std::size_t count = 1;
  for (std::size_t k = 0; k < d; ++k) {
    count *= ts.size();
  }
  std::vector<double> pts(count * d);
  for (std::size_t i = 0; i < count; ++i) {
    std::size_t rest = i;
    for (std::size_t k = d; k-- > 0;) {
      pts[i * d + k] = lo[k] + ts[rest % ts.size()] * (hi[k] - lo[k]);
      rest /= ts.size();
    }
  }
  return pts;
}

}  // namespace thb_test
