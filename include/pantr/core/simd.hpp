/// @file simd.hpp
/// @brief SIMD level of the translation unit, chosen from the target macros.
///
/// Exactly one of `PANTR_SIMD_AVX512`, `PANTR_SIMD_AVX2`, `PANTR_SIMD_NEON` and
/// `PANTR_SIMD_SCALAR` is defined to 1, from the compiler's target macros (the ISA the unit is
/// compiled for, not the machine it runs on). Defining `PANTR_SIMD_FORCE_SCALAR` (CMake option
/// `PANTR_FORCE_SCALAR`) selects the scalar level whatever the target. `PANTR_SIMD_ISA_NAME` names
/// the level.
///
/// Policy: vectorize across independent items, never inside a recurrence; rely on
/// auto-vectorization first; block loops before reaching for intrinsics. These macros select block
/// widths. Explicit intrinsics are written only where a benchmark shows auto-vectorization fails,
/// and then beside a scalar reference that the forced-scalar build checks against.
///
/// No public type's layout may depend on these macros: a program may mix translation units
/// compiled for different ISAs, and a layout that differed between them would be an ODR violation.
#pragma once

#include "pantr/config.hpp"

#include <cstddef>

#if defined(PANTR_SIMD_FORCE_SCALAR)
#define PANTR_SIMD_SCALAR 1
#define PANTR_SIMD_ISA_NAME "scalar"
#define PANTR_SIMD_REGISTER_BYTES 0
#elif defined(__AVX512F__)
#define PANTR_SIMD_AVX512 1
#define PANTR_SIMD_ISA_NAME "avx512"
#define PANTR_SIMD_REGISTER_BYTES 64
#elif defined(__AVX2__)
#define PANTR_SIMD_AVX2 1
#define PANTR_SIMD_ISA_NAME "avx2"
#define PANTR_SIMD_REGISTER_BYTES 32
#elif defined(__ARM_NEON)
#define PANTR_SIMD_NEON 1
#define PANTR_SIMD_ISA_NAME "neon"
#define PANTR_SIMD_REGISTER_BYTES 16
#else
#define PANTR_SIMD_SCALAR 1
#define PANTR_SIMD_ISA_NAME "scalar"
#define PANTR_SIMD_REGISTER_BYTES 0
#endif

namespace pantr::simd {

/// Name of the SIMD level this translation unit was compiled for.
inline constexpr const char* isa_name = PANTR_SIMD_ISA_NAME;

/// Number of `T` lanes in one vector register at this level; 1 at the scalar level.
template <class T>
inline constexpr std::size_t lanes = PANTR_SIMD_REGISTER_BYTES == 0
                                         ? std::size_t{1}
                                         : std::size_t{PANTR_SIMD_REGISTER_BYTES} / sizeof(T);

}  // namespace pantr::simd
