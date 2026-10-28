# Feature probes. They test what the toolchain compiles, never which compiler version it is: the
# floor is the standard library, not the compiler. The answers are cached, so a build directory
# serves one compiler; use one build directory per compiler.
include_guard(GLOBAL)
include(CheckCXXSourceCompiles)

# Probe at the standard the project compiles with: C++20 unless the builder asked for a newer one.
set(_pantr_saved_std "${CMAKE_CXX_STANDARD}")
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

# Hard gate: concepts are used throughout the public headers.
check_cxx_source_compiles(
  [=[
  #include <concepts>
  template <class T> requires std::floating_point<T> T f(T x) { return x; }
  int main() { return static_cast<int>(f(0.0)); }
  ]=]
  PANTR_HAVE_CONCEPTS)
if(NOT PANTR_HAVE_CONCEPTS)
  message(FATAL_ERROR "pantr: the C++ compiler does not support C++20 concepts")
endif()

# Hard gate: floating-point std::to_chars, which core/format.hpp needs to print a double as the
# shortest string that reads back to the same value (Python's repr).
check_cxx_source_compiles(
  [=[
  #include <charconv>
  int main() {
    char buf[32];
    auto r = std::to_chars(buf, buf + sizeof(buf), 0.1);
    return r.ec == std::errc{} ? 0 : 1;
  }
  ]=]
  PANTR_HAVE_FP_TO_CHARS)
if(NOT PANTR_HAVE_FP_TO_CHARS)
  message(FATAL_ERROR "pantr: the C++ standard library has no floating-point std::to_chars")
endif()

# Toggle: a standard <mdspan>. Decided here, once, and set as a compile definition on the exported
# target, so every translation unit of a consumer program sees the same pantr::mdspan type. A
# header-level __has_include test would give two types in a program that mixes C++20 and C++23
# translation units: an ODR violation in every signature that mentions it.
# Hard gate: std::make_shared for arrays (C++20), which core/ndarray.hpp allocates with.
check_cxx_source_compiles(
  [=[
  #include <memory>
  int main() { auto p = std::make_shared<double[]>(3); return static_cast<int>(p[0]); }
  ]=]
  PANTR_HAVE_MAKE_SHARED_ARRAY)
if(NOT PANTR_HAVE_MAKE_SHARED_ARRAY)
  message(FATAL_ERROR "pantr: the C++ standard library has no std::make_shared<T[]>")
endif()

check_cxx_source_compiles(
  [=[
  #include <mdspan>
  int main() {
    double d[6] = {};
    std::mdspan<double, std::dextents<std::size_t, 2>> m(d, 2, 3);
    return static_cast<int>(m.extent(0)) - 2;
  }
  ]=]
  PANTR_HAVE_STD_MDSPAN)
# Until pantr's code can index a std::mdspan (it has no operator(), which pantr code uses; see
# design/mdspan_identity.md), the probe is reported but the bundled implementation is always used.
set(PANTR_USE_STD_MDSPAN OFF CACHE BOOL "Use std::mdspan instead of the bundled Kokkos mdspan")
if(PANTR_USE_STD_MDSPAN)
  message(FATAL_ERROR "pantr: PANTR_USE_STD_MDSPAN=ON is not supported yet: pantr indexes with "
                      "operator(), which std::mdspan does not provide")
endif()
if(PANTR_HAVE_STD_MDSPAN)
  message(STATUS "pantr: this toolchain has <mdspan>; the bundled mdspan is used regardless")
endif()
if(PANTR_USE_STD_MDSPAN)
  message(STATUS "pantr: pantr::mdspan is std::mdspan")
else()
  message(STATUS "pantr: pantr::mdspan is the bundled Kokkos reference mdspan")
endif()

if(_pantr_saved_std)
  set(CMAKE_CXX_STANDARD "${_pantr_saved_std}")
else()
  unset(CMAKE_CXX_STANDARD)
endif()
