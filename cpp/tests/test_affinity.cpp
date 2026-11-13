// Affinity count and pool capacity.
#include <doctest/doctest.h>

#include <pantr/parallel/affinity.hpp>

#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>

#if defined(__linux__)
namespace {

// Independent route to the same fact: the kernel's text form of the mask in /proc, not the
// sched_getaffinity system call. Parses a list such as "0-3,8,10-11".
std::size_t count_from_proc_status() {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    const std::string key = "Cpus_allowed_list:";
    if (line.rfind(key, 0) != 0) {
      continue;
    }
    std::stringstream ranges(line.substr(key.size()));
    std::string range;
    std::size_t count = 0;
    while (std::getline(ranges, range, ',')) {
      const auto dash = range.find('-');
      const std::size_t lo = std::stoul(range.substr(0, dash));
      const std::size_t hi = dash == std::string::npos ? lo : std::stoul(range.substr(dash + 1));
      count += hi - lo + 1;
    }
    return count;
  }
  return 0;
}

}  // namespace

TEST_CASE("affinity: the count equals the kernel's Cpus_allowed_list") {
  const std::size_t expected = count_from_proc_status();
  REQUIRE(expected > 0);
  CHECK(pantr::parallel::affinity_cpu_count() == expected);
}
#endif

TEST_CASE("affinity: capacity is at least one and at most the affinity count") {
  const std::size_t cap = pantr::parallel::max_threads();
  CHECK(cap >= 1);
  CHECK(cap <= pantr::parallel::affinity_cpu_count());
}

TEST_CASE("PANTR_NUM_THREADS: parsing") {
  using pantr::parallel::detail::parse_num_threads_env;
  CHECK(parse_num_threads_env(nullptr) == 0);
  CHECK(parse_num_threads_env("") == 0);
  CHECK(parse_num_threads_env("1") == 1);
  CHECK(parse_num_threads_env("64") == 64);
  for (const char* bad : {"0", "-1", "2.5", "four", " 4", "4 ", "99999999999999999999999"}) {
    CAPTURE(bad);
    CHECK_THROWS_AS(static_cast<void>(parse_num_threads_env(bad)), pantr::value_error);
  }
  CHECK_THROWS_WITH(static_cast<void>(parse_num_threads_env("x")),
                    "PANTR_NUM_THREADS must be a positive integer, got 'x'");
}

TEST_CASE("PANTR_NUM_THREADS: lowers the capacity, never raises it") {
  using pantr::parallel::detail::capacity_from;
  CHECK(capacity_from(8, 0) == 8);
  CHECK(capacity_from(8, 3) == 3);
  CHECK(capacity_from(8, 8) == 8);
  CHECK(capacity_from(8, 100) == 8);
}
