// The pool and parallel_for: concurrent callers, nesting, exceptions, thread-count invariance,
// the thread-local setting, and fork safety. Run under TSan in the sanitizer build.
#include <doctest/doctest.h>

#include <pantr/parallel/parallel_for.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

#if defined(__SANITIZE_THREAD__)
#define PANTR_TEST_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define PANTR_TEST_TSAN 1
#endif
#endif

namespace par = pantr::parallel;

namespace {

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

/// Highest thread count the tests exercise: enough to cross every code path, bounded so a large
/// machine does not turn the suite into a stress test.
std::size_t test_threads() { return std::min<std::size_t>(par::max_threads(), 8); }

/// A per-item kernel with real floating-point work: 20 terms of a sine series.
constexpr int kTerms = 20;
double kernel_value(std::size_t i) {
  double s = 0.0;
  for (int j = 1; j <= kTerms; ++j) {
    s += std::sin(static_cast<double>(i) * j * 1e-3) / j;
  }
  return s;
}
/// Its work estimate: kTerms terms of a sine (order 20 operations) plus a multiply and a divide.
constexpr par::WorkEstimate kKernelWork{kTerms * 22.0};

struct RunRecord {
  std::vector<double> values;
  std::set<std::pair<std::size_t, std::size_t>> chunks;
  std::set<std::thread::id> threads;
};

RunRecord run_kernel(std::size_t n) {
  RunRecord rec;
  rec.values.assign(n, std::numeric_limits<double>::quiet_NaN());
  std::mutex m;
  par::parallel_for(n, kKernelWork, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      rec.values[i] = kernel_value(i);
    }
    const std::lock_guard<std::mutex> lock(m);
    rec.chunks.emplace(b, e);
    rec.threads.insert(std::this_thread::get_id());
  });
  return rec;
}

}  // namespace

TEST_CASE("parallel_for: every item is visited exactly once, in chunks of the grain") {
  const ThreadsGuard guard(test_threads());
  for (const std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{64},
                              std::size_t{1000}, std::size_t{100003}}) {
    for (const std::size_t grain :
         {std::size_t{1}, std::size_t{3}, std::size_t{64}, std::size_t{5000}}) {
      CAPTURE(n);
      CAPTURE(grain);
      std::vector<std::atomic<int>> visits(n);
      std::atomic<bool> bad_chunk{false};
      par::parallel_for(n, grain, [&](std::size_t b, std::size_t e) {
        if (b % grain != 0 || e - b > grain || e > n || (e - b < grain && e != n)) {
          bad_chunk = true;
        }
        for (std::size_t i = b; i < e; ++i) {
          visits[i].fetch_add(1, std::memory_order_relaxed);
        }
      });
      CHECK_FALSE(bad_chunk.load());
      CHECK(std::all_of(visits.begin(), visits.end(), [](const auto& v) { return v.load() == 1; }));
    }
  }
}

TEST_CASE("parallel_for: grain 0 is a value_error") {
  CHECK_THROWS_WITH_AS(par::parallel_for(10, std::size_t{0}, [](std::size_t, std::size_t) {}),
                       "grain must be >= 1, got 0", pantr::value_error);
}

TEST_CASE("parallel_for: results and chunks do not depend on the thread count") {
  // Sizes on both sides of the grain (one chunk vs two) and of the serial threshold.
  const std::size_t grain = par::grain_for(kKernelWork);
  const auto serial_limit =
      static_cast<std::size_t>(std::ceil(par::parallel_min_ops / kKernelWork.ops_per_item));
  for (const std::size_t n : {std::size_t{1}, grain - 1, grain, grain + 1, serial_limit - 1,
                              serial_limit, serial_limit + 1, 20 * serial_limit}) {
    CAPTURE(n);
    RunRecord reference;
    {
      const ThreadsGuard guard(1);
      reference = run_kernel(n);
    }
    for (std::size_t t = 2; t <= test_threads(); ++t) {
      CAPTURE(t);
      const ThreadsGuard guard(t);
      const RunRecord rec = run_kernel(n);
      // Chunk boundaries are integers: exact equality is the criterion.
      CHECK(rec.chunks == reference.chunks);
      CHECK(rec.threads.size() <= t);
      // Values: each item is the same expression evaluated on some thread. Bound the difference
      // by the recursive-summation bound over the terms, kTerms * eps * sum|term|, which holds for
      // any evaluation of the series, so the check does not require bit-identity.
      bool within = true;
      for (std::size_t i = 0; i < n; ++i) {
        double abs_terms = 0.0;
        for (int j = 1; j <= kTerms; ++j) {
          abs_terms += std::abs(std::sin(static_cast<double>(i) * j * 1e-3) / j);
        }
        const double bound = kTerms * std::numeric_limits<double>::epsilon() * abs_terms;
        within = within && std::abs(rec.values[i] - reference.values[i]) <= bound;
      }
      CHECK(within);
    }
  }
}

TEST_CASE("parallel_for: concurrent callers from many threads") {
  const ThreadsGuard guard(test_threads());
  constexpr int kCallers = 12;
  constexpr int kRepeats = 20;
  constexpr std::size_t kN = 20000;
  std::vector<int> ok(kCallers, 0);
  std::vector<std::thread> callers;
  callers.reserve(kCallers);
  for (int c = 0; c < kCallers; ++c) {
    callers.emplace_back([&, c] {
      // Each caller thread has its own setting, at the pool capacity by default.
      std::vector<double> out(kN);
      bool good = true;
      for (int r = 0; r < kRepeats; ++r) {
        par::parallel_for(kN, std::size_t{256}, [&](std::size_t b, std::size_t e) {
          for (std::size_t i = b; i < e; ++i) {
            out[i] = static_cast<double>(i) * c + r;
          }
        });
        for (std::size_t i = 0; i < kN; ++i) {
          good = good && out[i] == static_cast<double>(i) * c + r;
        }
      }
      ok[static_cast<std::size_t>(c)] = good ? 1 : 0;
    });
  }
  for (auto& t : callers) {
    t.join();
  }
  CHECK(std::all_of(ok.begin(), ok.end(), [](int v) { return v == 1; }));
}

TEST_CASE("parallel_for: a nested loop runs inline on the thread of the outer chunk") {
  const ThreadsGuard guard(test_threads());
  constexpr std::size_t kOuter = 64;
  constexpr std::size_t kInner = 500;
  std::vector<double> out(kOuter * kInner, 0.0);
  std::atomic<bool> left_thread{false};
  par::parallel_for(kOuter, std::size_t{1}, [&](std::size_t ob, std::size_t oe) {
    const auto outer_thread = std::this_thread::get_id();
    for (std::size_t o = ob; o < oe; ++o) {
      par::parallel_for(kInner, std::size_t{10}, [&](std::size_t ib, std::size_t ie) {
        if (std::this_thread::get_id() != outer_thread) {
          left_thread = true;
        }
        for (std::size_t i = ib; i < ie; ++i) {
          out[o * kInner + i] = static_cast<double>(o + i);
        }
      });
    }
  });
  CHECK_FALSE(left_thread.load());
  bool all = true;
  for (std::size_t o = 0; o < kOuter; ++o) {
    for (std::size_t i = 0; i < kInner; ++i) {
      all = all && out[o * kInner + i] == static_cast<double>(o + i);
    }
  }
  CHECK(all);
}

TEST_CASE("parallel_for: the first exception reaches the caller, and the pool stays usable") {
  const ThreadsGuard guard(test_threads());
  for (int attempt = 0; attempt < 10; ++attempt) {
    CHECK_THROWS_WITH_AS(par::parallel_for(1000, std::size_t{1},
                                           [&](std::size_t b, std::size_t) {
                                             if (b % 97 == 13) {
                                               throw std::runtime_error("chunk failed");
                                             }
                                           }),
                         "chunk failed", std::runtime_error);
  }
  // A pantr exception keeps its type through the pool.
  CHECK_THROWS_AS(
      par::parallel_for(100, std::size_t{1},
                        [](std::size_t, std::size_t) { throw pantr::value_error("from a chunk"); }),
      pantr::value_error);
  // The pool still runs loops to completion afterwards.
  std::vector<int> out(5000, 0);
  par::parallel_for(out.size(), std::size_t{7}, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      out[i] = 1;
    }
  });
  CHECK(std::all_of(out.begin(), out.end(), [](int v) { return v == 1; }));
}

TEST_CASE("parallel_for: one thread runs every chunk on the caller, in order") {
  const ThreadsGuard guard(1);
  const auto caller = std::this_thread::get_id();
  std::vector<std::size_t> order;
  bool other_thread = false;
  par::parallel_for(100, std::size_t{9}, [&](std::size_t b, std::size_t) {
    other_thread = other_thread || std::this_thread::get_id() != caller;
    order.push_back(b);
  });
  CHECK_FALSE(other_thread);
  CHECK(std::is_sorted(order.begin(), order.end()));
  CHECK(order.size() == 12);
}

TEST_CASE("Pool: after a chunk throws, the chunks not yet started are skipped") {
  // A pool with no worker runs the chunks on the caller in claim order, so the skip is
  // deterministic: chunk 0 throws, and none of the other nine may run.
  par::Pool pool(0);
  std::atomic<int> ran{0};
  par::detail::Job job{10, &ran,
                       [](void* p, std::size_t k) {
                         static_cast<std::atomic<int>*>(p)->fetch_add(1);
                         if (k == 0) {
                           throw std::runtime_error("first chunk failed");
                         }
                       },
                       0};
  CHECK_THROWS_WITH_AS(pool.run(job), "first chunk failed", std::runtime_error);
  CHECK(ran.load() == 1);
}

TEST_CASE("parallel_for: pool workers take part") {
  if (par::max_threads() < 2) {
    MESSAGE("one CPU available: no worker exists to take part");
    return;
  }
  const ThreadsGuard guard(2);
  // Each chunk waits (bounded) until a chunk has run on a second thread, which only a worker can
  // provide: a pool that never woke its workers would leave one thread id.
  std::mutex m;
  std::set<std::thread::id> ids;
  std::atomic<bool> two_seen{false};
  par::parallel_for(8, std::size_t{1}, [&](std::size_t, std::size_t) {
    {
      const std::lock_guard<std::mutex> lock(m);
      ids.insert(std::this_thread::get_id());
      if (ids.size() >= 2) {
        two_seen = true;
      }
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!two_seen.load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
  });
  CHECK(ids.size() == 2);
}

TEST_CASE("Pool: the caller completes a loop with no worker at all") {
  par::Pool pool(0);
  std::vector<int> out(100, 0);
  struct Ctx {
    std::vector<int>* out;
  } ctx{&out};
  par::detail::Job job{10, &ctx,
                       [](void* p, std::size_t k) {
                         auto* c = static_cast<Ctx*>(p);
                         for (std::size_t i = 10 * k; i < 10 * k + 10; ++i) {
                           (*c->out)[i] = 1;
                         }
                       },
                       4};
  pool.run(job);
  CHECK(std::all_of(out.begin(), out.end(), [](int v) { return v == 1; }));
}

TEST_CASE("set_num_threads: validated, and local to the calling thread") {
  CHECK_THROWS_WITH_AS(par::set_num_threads(0), "n must be >= 1, got 0", pantr::value_error);
  CHECK_THROWS_WITH_AS(par::set_num_threads(-3), "n must be >= 1, got -3", pantr::value_error);
  const auto cap = static_cast<std::int64_t>(par::max_threads());
  CHECK_THROWS_WITH_AS(par::set_num_threads(cap + 1),
                       ("n must be <= the pool capacity (" + std::to_string(cap) + "), got " +
                        std::to_string(cap + 1))
                           .c_str(),
                       pantr::value_error);
  const ThreadsGuard guard(1);
  CHECK(par::get_num_threads() == 1);
  std::size_t seen_elsewhere = 0;
  std::thread other([&] { seen_elsewhere = par::get_num_threads(); });
  other.join();
  CHECK(seen_elsewhere == par::max_threads());
}

#if !defined(_WIN32) && !defined(PANTR_TEST_TSAN)
// Not under TSan: its runtime does not support a child that starts threads after a
// multi-threaded fork (die_after_fork), which is exactly what this test does.
TEST_CASE("Pool: a forked child builds its own pool and runs parallel loops") {
  if (par::max_threads() < 2) {
    MESSAGE("one CPU available: the child would run serially, so the pool path is not reached");
    return;
  }
  const ThreadsGuard guard(test_threads());
  // Make sure the parent's pool exists and its workers are running.
  std::vector<int> warm(10000, 0);
  par::parallel_for(warm.size(), std::size_t{100}, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      warm[i] = 1;
    }
  });
  const std::int64_t parent_pid = par::Pool::instance().creator_pid();
  REQUIRE(parent_pid == static_cast<std::int64_t>(::getpid()));

  const pid_t child = ::fork();
  REQUIRE(child >= 0);
  if (child == 0) {
    // Child: report through the exit status only (no doctest machinery after a fork). A hang
    // ends with SIGALRM instead of stalling the suite.
    ::alarm(60);
    int status = 0;
    try {
      std::vector<double> out(50000, 0.0);
      par::parallel_for(out.size(), std::size_t{100}, [&](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) {
          out[i] = 3.0 * static_cast<double>(i);
        }
      });
      for (std::size_t i = 0; i < out.size(); ++i) {
        if (out[i] != 3.0 * static_cast<double>(i)) {
          status = 1;
        }
      }
      if (par::Pool::instance().creator_pid() != static_cast<std::int64_t>(::getpid())) {
        status = 2;
      }
      if (par::Pool::instance().workers() + 1 != par::max_threads()) {
        status = 3;
      }
    } catch (...) {
      status = 4;
    }
    ::_exit(status);
  }
  int wstatus = 0;
  REQUIRE(::waitpid(child, &wstatus, 0) == child);
  REQUIRE(WIFEXITED(wstatus));
  CHECK(WEXITSTATUS(wstatus) == 0);
  // The parent's pool is untouched.
  CHECK(par::Pool::instance().creator_pid() == parent_pid);
}
#endif
