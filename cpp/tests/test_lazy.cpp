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
