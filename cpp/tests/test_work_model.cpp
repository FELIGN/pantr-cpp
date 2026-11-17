// The work model: grain and serial threshold from the work per item, validated.
#include <doctest/doctest.h>

#include <pantr/parallel/work_model.hpp>

#include <cmath>
#include <cstddef>
#include <limits>

using pantr::parallel::chunk_ops;
using pantr::parallel::grain_for;
using pantr::parallel::parallel_min_ops;
using pantr::parallel::WorkEstimate;
using pantr::parallel::worth_parallel;

TEST_CASE("work model: grain is ceil(chunk_ops / ops_per_item), at least one") {
  CHECK(grain_for(WorkEstimate{1.0}) == static_cast<std::size_t>(chunk_ops));
  CHECK(grain_for(WorkEstimate{3.0}) == 5462);  // ceil(16384 / 3)
  CHECK(grain_for(WorkEstimate{chunk_ops}) == 1);
  CHECK(grain_for(WorkEstimate{10.0 * chunk_ops}) == 1);
}

TEST_CASE("work model: parallel from parallel_min_ops total operations on") {
  const WorkEstimate w{16.0};
  const auto threshold = static_cast<std::size_t>(parallel_min_ops / 16.0);
  CHECK_FALSE(worth_parallel(threshold - 1, w));
  CHECK(worth_parallel(threshold, w));
  CHECK_FALSE(worth_parallel(0, w));
}

TEST_CASE("work model: an estimate below one operation, or not finite, is a value_error") {
  for (const double bad : {0.0, 0.5, -1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
    CAPTURE(bad);
    CHECK_THROWS_AS(static_cast<void>(grain_for(WorkEstimate{bad})), pantr::value_error);
    CHECK_THROWS_AS(static_cast<void>(worth_parallel(10, WorkEstimate{bad})), pantr::value_error);
  }
  CHECK_THROWS_WITH(static_cast<void>(grain_for(WorkEstimate{0.5})),
                    "ops_per_item must be a finite number >= 1, got 0.5");
}

TEST_CASE("work model: a memory-bound item's chunk streams at least chunk_bytes") {
  using pantr::parallel::chunk_bytes;
  // 16 ops and 32 bytes per item: by operations alone 1024 items (32 KiB), by bytes 8192.
  CHECK(grain_for(WorkEstimate{16.0}, 32) == static_cast<std::size_t>(chunk_bytes / 32.0));
  // One byte per item: the bytes bound dominates even an item worth a whole chunk of ops.
  CHECK(grain_for(WorkEstimate{chunk_ops}, 1) == static_cast<std::size_t>(chunk_bytes));
  // A large item keeps its operation-derived chunk.
  CHECK(grain_for(WorkEstimate{1.0}, 1 << 20) == static_cast<std::size_t>(chunk_ops));
  // The bytes bound rounds up: 2^18 / 24 = 10922.67.
  CHECK(grain_for(WorkEstimate{16.0}, 24) == 10923);
  CHECK(grain_for(WorkEstimate{3.0}, 0) == grain_for(WorkEstimate{3.0}));
}
