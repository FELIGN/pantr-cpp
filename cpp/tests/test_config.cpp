// Version and configuration headers.
#include <doctest/doctest.h>

#include <pantr/pantr.hpp>

TEST_CASE("version: the constexpr triple matches the macros") {
  static_assert(pantr::version.major == PANTR_VERSION_MAJOR);
  static_assert(pantr::version.minor == PANTR_VERSION_MINOR);
  static_assert(pantr::version.patch == PANTR_VERSION_PATCH);
  CHECK(PANTR_VERSION ==
        PANTR_VERSION_MAJOR * 10000 + PANTR_VERSION_MINOR * 100 + PANTR_VERSION_PATCH);
}

TEST_CASE("config: Eigen is built MPL-2.0-only and without its own threads") {
#if !defined(EIGEN_MPL2_ONLY) || !defined(EIGEN_DONT_PARALLELIZE)
  FAIL("pantr::pantr must define EIGEN_MPL2_ONLY and EIGEN_DONT_PARALLELIZE");
#endif
  CHECK(true);
}
