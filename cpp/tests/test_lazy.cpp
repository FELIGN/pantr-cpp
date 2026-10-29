// Lazy<Block>: computed once under concurrent first access (a TSan target), retried after a
// throw, empty after a copy.
#include <doctest/doctest.h>

#include <pantr/core/lazy.hpp>

#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

struct Block {
  std::vector<double> values;
  int id;
};

}  // namespace

TEST_CASE("Lazy: many threads racing on first access compute once and see one block") {
  constexpr int kThreads = 16;
  constexpr int kRounds = 20;
  for (int round = 0; round < kRounds; ++round) {
    const pantr::detail::Lazy<Block> lazy;
    std::atomic<int> computations{0};
    std::atomic<int> ready_to_go{0};
    std::vector<const Block*> seen(kThreads, nullptr);
    std::vector<double> sums(kThreads, 0.0);
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&, t] {
        ready_to_go.fetch_add(1);
        while (ready_to_go.load() < kThreads) {
        }
        const Block& b = lazy.get([&] {
          computations.fetch_add(1);
          return Block{std::vector<double>(64, 1.0), round};
        });
        seen[static_cast<std::size_t>(t)] = &b;
        double s = 0.0;
        for (const double v : b.values) {
          s += v;  // reads the block's contents: a race here is what TSan reports
        }
        sums[static_cast<std::size_t>(t)] = s;
      });
    }
    for (auto& th : threads) {
      th.join();
    }
    CHECK(computations.load() == 1);
    for (int t = 0; t < kThreads; ++t) {
      CHECK(seen[static_cast<std::size_t>(t)] == seen[0]);
      CHECK(sums[static_cast<std::size_t>(t)] == 64.0);
    }
    CHECK(lazy.ready());
  }
}

// The fast path alone: one thread computes, then signals the others through a relaxed flag, which
// orders nothing. Each reader then finds the block ready and reads it with no mutex, so the only
// thing ordering its reads after the computation's writes is the fast path's acquire load; were
// that load relaxed, TSan would report the reads (the racing test above mostly sends readers
// through the mutex, which orders them anyway).
TEST_CASE("Lazy: readers on the fast path see the computed block (a TSan target)") {
  constexpr int kReaders = 8;
  constexpr int kRounds = 50;
  for (int round = 0; round < kRounds; ++round) {
    const pantr::detail::Lazy<Block> lazy;
    std::atomic<bool> published{false};
    std::vector<double> sums(kReaders, 0.0);
    std::vector<std::thread> readers;
    readers.reserve(kReaders);
    for (int t = 0; t < kReaders; ++t) {
      readers.emplace_back([&, t] {
        while (!published.load(std::memory_order_relaxed)) {
        }
        const Block& b = lazy.get([] { return Block{std::vector<double>(64, 2.0), -1}; });
        double s = 0.0;
        for (const double v : b.values) {
          s += v;
        }
        sums[static_cast<std::size_t>(t)] = s;
      });
    }
    static_cast<void>(lazy.get([round] { return Block{std::vector<double>(64, 1.0), round}; }));
    published.store(true, std::memory_order_relaxed);
    for (auto& th : readers) {
      th.join();
    }
    for (const double s : sums) {
      CHECK(s == 64.0);
    }
  }
}

TEST_CASE("Lazy: a computation that throws stores nothing, and the next call computes again") {
  const pantr::detail::Lazy<int> lazy;
  CHECK_THROWS_AS(static_cast<void>(lazy.get([]() -> int { throw std::runtime_error("boom"); })),
                  std::runtime_error);
  CHECK_FALSE(lazy.ready());
  CHECK(lazy.get([] { return 7; }) == 7);
  CHECK(lazy.get([] { return 8; }) == 7);  // stored: the second callable is not called
}

TEST_CASE("Lazy: a copy starts empty") {
  const pantr::detail::Lazy<int> a;
  CHECK(a.get([] { return 1; }) == 1);
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization): the copy is what is under test
  const pantr::detail::Lazy<int> b(a);
  CHECK_FALSE(b.ready());
  CHECK(b.get([] { return 2; }) == 2);
}
