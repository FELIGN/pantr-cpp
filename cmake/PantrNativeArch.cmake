# Flag decisions made where the consumer's compiler is known. Installed beside pantrConfig.cmake and
# included both by pantr's own build and by pantrConfig.cmake, so the build tree and an installed
# pantr apply the same rule.
include_guard(GLOBAL)

# Add the native-ISA flag to <target> when <enabled> is true.
#
# Not frozen into the exported target: the machine that builds a consumer is not necessarily the
# one that configured pantr, so the decision is the consumer's (set PANTR_NATIVE_ARCH before
# find_package(pantr)). x86 compilers take -march=native; AArch64 GCC and Clang take -mcpu=native.
# MSVC has no native-ISA switch, so nothing is added there and a message says so. Never applied
# when cross-compiling, where "native" names the wrong machine.
function(pantr_apply_native_arch target enabled)
  if(NOT enabled)
    return()
  endif()
  if(CMAKE_CROSSCOMPILING)
    message(STATUS "pantr: PANTR_NATIVE_ARCH ignored while cross-compiling")
    return()
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message(STATUS "pantr: PANTR_NATIVE_ARCH has no MSVC equivalent; no ISA flag added")
    return()
  endif()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(_flag -mcpu=native)
  else()
    set(_flag -march=native)
  endif()
  target_compile_options(${target} INTERFACE ${_flag})
  message(STATUS "pantr: ${_flag} added to ${target} (PANTR_NATIVE_ARCH=ON)")
endfunction()

# Say what the contraction flag does with the compiler at hand.
#
# GCC implements -ffp-contract=on only from GCC 14; before that the manual documents it as
# behaving as -ffp-contract=off, so no contraction happens. This is the one compiler-version test
# in pantr's build: contraction semantics cannot be probed by compiling a snippet.
function(pantr_report_fp_contract enabled)
  if(NOT enabled)
    message(STATUS "pantr: floating-point contraction flag not set (PANTR_FP_CONTRACT=OFF)")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
    message(STATUS "pantr: -ffp-contract=on behaves as -ffp-contract=off before GCC 14; "
                   "no contraction with GCC ${CMAKE_CXX_COMPILER_VERSION}")
  else()
    message(STATUS "pantr: floating-point contraction within expressions is enabled")
  endif()
endfunction()
