// PANTR_NUM_THREADS as max_threads() reads it. Not part of the main suite: ctest runs this binary
// with the variable set (see CMakeLists.txt), and PANTR_TEST_EXPECT says which outcome to expect.
#include <doctest/doctest.h>

#include <pantr/parallel/affinity.hpp>
#include <pantr/parallel/pool.hpp>

#include <algorithm>
#include <cstdlib>
#include <string>

TEST_CASE("PANTR_NUM_THREADS: max_threads honours the environment") {
  const char* expect = std::getenv("PANTR_TEST_EXPECT");
  REQUIRE(expect != nullptr);
  const std::string mode(expect);
  if (mode == "malformed") {
    CHECK_THROWS_WITH_AS(static_cast<void>(pantr::parallel::max_threads()),
                         "PANTR_NUM_THREADS must be a positive integer, got 'two'",
                         pantr::value_error);
    // A failed read stores nothing: the next call fails the same way.
    CHECK_THROWS_AS(static_cast<void>(pantr::parallel::max_threads()), pantr::value_error);
  } else {
    // mode is the requested count; the capacity is never above the affinity count.
    const std::size_t requested = std::stoul(mode);
    const std::size_t expected = std::min(requested, pantr::parallel::affinity_cpu_count());
    CHECK(pantr::parallel::max_threads() == expected);
    CHECK(pantr::parallel::get_num_threads() == expected);
    CHECK(pantr::parallel::Pool::instance().workers() == expected - 1);
  }
}
