/// @file config.hpp
/// @brief Compile-time configuration every pantr header includes first.
///
/// pantr refuses to compile under fast-math. Those modes license reassociation, assume no NaN or
/// infinity and drop error checks, which voids the floating-point reasoning the library's results
/// rest on (special-value policies, derived tolerances, compensated sums). The check lives in a
/// header, not in the build system, because a header-only library is compiled with the
/// consumer's flags: only a check in the header runs on every consumer compile.
#pragma once

#include "pantr/version.hpp"

// GCC and Clang always define __FINITE_MATH_ONLY__, as 0 or 1, so its value is what matters.
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__) || \
    defined(_M_FP_FAST)
#error "pantr must not be compiled with fast-math (-ffast-math, -ffinite-math-only, /fp:fast)"
#endif

#if !defined(__cplusplus) || \
    (__cplusplus < 202002L && !(defined(_MSVC_LANG) && _MSVC_LANG >= 202002L))
#error "pantr requires C++20"
#endif
