// The SIMD level macros: exactly one level, consistent name and lane counts. Built twice: in the
// main suite (target ISA) and with PANTR_SIMD_FORCE_SCALAR (PANTR_TEST_EXPECT_SCALAR).
#include <doctest/doctest.h>

#include <pantr/core/simd.hpp>

#include <cstddef>
#include <string>

namespace {

constexpr int kLevels = 0
#if defined(PANTR_SIMD_AVX512)
                        + 1
#endif
#if defined(PANTR_SIMD_AVX2)
                        + 1
#endif
#if defined(PANTR_SIMD_NEON)
                        + 1
#endif
#if defined(PANTR_SIMD_SCALAR)
                        + 1
#endif
    ;
static_assert(kLevels == 1, "exactly one PANTR_SIMD_* level must be defined");

}  // namespace

TEST_CASE("simd: one level, its name, and its lane counts") {
  const std::string name = pantr::simd::isa_name;
#if defined(PANTR_TEST_EXPECT_SCALAR)
  CHECK(name == "scalar");
#endif
#if defined(PANTR_SIMD_AVX512)
  CHECK(name == "avx512");
  CHECK(pantr::simd::lanes<double> == 8);
  CHECK(pantr::simd::lanes<float> == 16);
#elif defined(PANTR_SIMD_AVX2)
  CHECK(name == "avx2");
  CHECK(pantr::simd::lanes<double> == 4);
  CHECK(pantr::simd::lanes<float> == 8);
#elif defined(PANTR_SIMD_NEON)
  CHECK(name == "neon");
  CHECK(pantr::simd::lanes<double> == 2);
  CHECK(pantr::simd::lanes<float> == 4);
#else
  CHECK(name == "scalar");
  CHECK(pantr::simd::lanes<double> == 1);
  CHECK(pantr::simd::lanes<float> == 1);
#endif
}
