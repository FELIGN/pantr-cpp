// Routing through another module's parallel state (shared.hpp), against a fake owner table that
// records every call. A binary of its own, since an attached table cannot be detached; ctest runs
// each case in its own process, so every case after the first attaches the fake itself. The
// two-module case itself, with a real second extension module, is
// tests/test_shared_parallelism.py.
#include <doctest/doctest.h>

#include <pantr/parallel/parallel_for.hpp>
#include <pantr/parallel/shared.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace par = pantr::parallel;

namespace {

// The fake owner's state.
std::size_t fake_setting = 0;
bool fake_region = false;
std::atomic<int> max_threads_calls{0};
std::atomic<int> run_calls{0};
int run_status = 0;  // what the next run reports

int fake_max_threads(std::size_t* out, char* /*message*/, std::size_t /*size*/) noexcept {
  ++max_threads_calls;
  *out = 7;
  return 0;
}

int fake_run(par::detail::Job* job, char* message, std::size_t size) noexcept {
  ++run_calls;
  if (run_status != 0) {
    par::detail::write_shared_message("the owner's pool failed", message, size);
    return run_status;
  }
  job->run_chunks();  // the owner would spread these over its workers
  return 0;
}

std::size_t* fake_thread_setting() noexcept { return &fake_setting; }
bool* fake_thread_in_region() noexcept { return &fake_region; }

constexpr par::SharedParallelism kFake{
    par::kSharedParallelismAbi, static_cast<std::uint32_t>(sizeof(par::SharedParallelism)),
    fake_max_threads,           fake_run,
    fake_thread_setting,        fake_thread_in_region};

}  // namespace

TEST_CASE("shared: a null or mismatched table is refused and nothing is attached") {
  CHECK_THROWS_AS(par::attach_shared_parallelism(nullptr), pantr::value_error);
  par::SharedParallelism other_abi = kFake;
  other_abi.abi = par::kSharedParallelismAbi + 1;
  CHECK_THROWS_WITH_AS(par::attach_shared_parallelism(&other_abi),
                       doctest::Contains("built from different headers"), pantr::state_error);
  par::SharedParallelism other_size = kFake;
  other_size.size = 8;
  CHECK_THROWS_AS(par::attach_shared_parallelism(&other_size), pantr::state_error);
  CHECK_FALSE(par::routes_shared_parallelism());
}

TEST_CASE("shared: attaching routes the capacity and the thread setting to the owner") {
  par::attach_shared_parallelism(&kFake);
  par::attach_shared_parallelism(&kFake);  // the same table again is harmless
  CHECK(par::routes_shared_parallelism());

  const int calls = max_threads_calls.load();
  CHECK(par::max_threads() == 7);
  CHECK(max_threads_calls.load() == calls + 1);

  fake_setting = 0;
  CHECK(par::get_num_threads() == 7);  // unset: the owner's capacity
  par::set_num_threads(5);
  CHECK(fake_setting == 5);  // written into the owner's slot
  CHECK(par::get_num_threads() == 5);
  CHECK_THROWS_AS(par::set_num_threads(8), pantr::value_error);  // above the owner's capacity
  CHECK(fake_setting == 5);
}

TEST_CASE("shared: a different table is refused once one is attached") {
  par::attach_shared_parallelism(&kFake);
  par::SharedParallelism other = kFake;
  CHECK_THROWS_WITH_AS(par::attach_shared_parallelism(&other),
                       doctest::Contains("already attached"), pantr::state_error);
}

TEST_CASE("shared: parallel loops run through the owner and keep their chunks") {
  par::attach_shared_parallelism(&kFake);
  fake_setting = 4;
  fake_region = false;
  run_status = 0;
  const int calls = run_calls.load();
  std::vector<int> hits(100, 0);
  par::parallel_for(hits.size(), std::size_t{7}, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      hits[i] += 1;
    }
  });
  CHECK(run_calls.load() == calls + 1);
  for (const int h : hits) {
    CHECK(h == 1);
  }
}

TEST_CASE("shared: a loop inside the owner's parallel region runs inline") {
  par::attach_shared_parallelism(&kFake);
  fake_setting = 4;
  fake_region = true;
  const int calls = run_calls.load();
  int sum = 0;
  par::parallel_for(std::size_t{10}, std::size_t{1},
                    [&](std::size_t b, std::size_t) { sum += static_cast<int>(b); });
  CHECK(sum == 45);
  CHECK(run_calls.load() == calls);
  fake_region = false;
}

TEST_CASE("shared: a chunk's exception reaches the caller") {
  par::attach_shared_parallelism(&kFake);
  fake_setting = 4;
  run_status = 0;
  CHECK_THROWS_WITH_AS(par::parallel_for(std::size_t{10}, std::size_t{1},
                                         [](std::size_t b, std::size_t) {
                                           if (b == 3) {
                                             throw std::out_of_range("chunk 3");
                                           }
                                         }),
                       "chunk 3", std::out_of_range);
}

TEST_CASE("shared: the owner's failures are raised here with its message") {
  par::attach_shared_parallelism(&kFake);
  fake_setting = 4;
  run_status = static_cast<int>(par::SharedStatus::value_error);
  CHECK_THROWS_WITH_AS(
      par::parallel_for(std::size_t{10}, std::size_t{1}, [](std::size_t, std::size_t) {}),
      "the owner's pool failed", pantr::value_error);
  run_status = static_cast<int>(par::SharedStatus::runtime_error);
  CHECK_THROWS_WITH_AS(
      par::parallel_for(std::size_t{10}, std::size_t{1}, [](std::size_t, std::size_t) {}),
      "the owner's pool failed", std::runtime_error);
  run_status = 0;
}

TEST_CASE("shared: this module's own pool is never started once attached") {
  par::attach_shared_parallelism(&kFake);
  CHECK_THROWS_WITH_AS(par::Pool::instance(), doctest::Contains("another module's pool"),
                       pantr::state_error);
}

TEST_CASE("shared: the owner's table serves the module it was compiled into") {
  par::attach_shared_parallelism(&kFake);
  // Through the table, the owner's capacity and slots are this binary's own state: the local
  // capacity, and the local thread-local variables, whatever is attached.
  const par::SharedParallelism& own = par::owner_shared_parallelism();
  std::size_t cap = 0;
  char message[par::detail::kSharedMessageSize];
  REQUIRE(own.max_threads(&cap, message, sizeof(message)) == 0);
  CHECK(cap == par::detail::local_max_threads());
  CHECK(own.thread_setting() == &par::detail::tl_num_threads);
  CHECK(own.thread_in_region() == &par::detail::tl_in_parallel_region);

  // Its run executes on this binary's own pool, even though this binary is attached elsewhere.
  par::detail::tl_num_threads = 2;
  std::atomic<int> chunks{0};
  auto body = [&](std::size_t, std::size_t) { ++chunks; };
  struct Ctx {
    decltype(body)& f;
  } ctx{body};
  par::detail::Job job{8, &ctx, [](void* p, std::size_t k) { static_cast<Ctx*>(p)->f(k, k + 1); },
                       1};
  CHECK(own.run(&job, message, sizeof(message)) == 0);
  CHECK(chunks.load() == 8);
}
