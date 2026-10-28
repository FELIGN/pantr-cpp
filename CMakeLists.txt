# pantr: header-only C++20 library (pantr::pantr) under the Python package.
#
# Layout and the reasons behind each option: design/port_design.md, section 1.
cmake_minimum_required(VERSION 3.28)  # FetchContent_Declare(... EXCLUDE_FROM_ALL) needs 3.28

# The version lives in include/pantr/version.hpp, the one place a C++ consumer can read it.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/pantr/version.hpp" _pantr_version_lines
     REGEX "^#define PANTR_VERSION_(MAJOR|MINOR|PATCH) [0-9]+$")
foreach(_part MAJOR MINOR PATCH)
  if(NOT _pantr_version_lines MATCHES "PANTR_VERSION_${_part} ([0-9]+)")
    message(FATAL_ERROR "pantr: cannot read PANTR_VERSION_${_part} from include/pantr/version.hpp")
  endif()
  set(_pantr_version_${_part} "${CMAKE_MATCH_1}")
endforeach()
set(_pantr_version "${_pantr_version_MAJOR}.${_pantr_version_MINOR}.${_pantr_version_PATCH}")

# The Python package carries the same version; refuse to configure when the two disagree.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/src/pantr/__init__.py" _pantr_py_version_line
     REGEX "^__version__: Final\\[str\\] = \"[^\"]+\"$")
if(NOT _pantr_py_version_line MATCHES "\"([^\"]+)\"" OR NOT CMAKE_MATCH_1 STREQUAL _pantr_version)
  message(FATAL_ERROR "pantr: include/pantr/version.hpp says ${_pantr_version} but "
                      "src/pantr/__init__.py says '${CMAKE_MATCH_1}'; update both together")
endif()

project(pantr VERSION ${_pantr_version} LANGUAGES CXX)

# pantr's own targets compile as ISO C++ (no GNU extensions) and use no C++ modules, so the
# module dependency scan is off.
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(PantrOptions)
include(PantrCompilerProbes)
include(PantrDependencies)
include(PantrFlags)

# --- pantr::pantr -------------------------------------------------------------------------------
add_library(pantr INTERFACE)
add_library(pantr::pantr ALIAS pantr)
target_include_directories(pantr INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(pantr INTERFACE cxx_std_20)
find_package(Threads REQUIRED)
target_link_libraries(pantr INTERFACE Threads::Threads pantr_deps)
target_compile_definitions(
  pantr
  INTERFACE
    # Refuse direct inclusion of any Eigen file that is not MPL-2.0 (a tripwire, not a guarantee).
    EIGEN_MPL2_ONLY
    # Eigen never starts its own threads: pantr's pool is the only parallelism in a process.
    EIGEN_DONT_PARALLELIZE
    # GCC's -Wmaybe-uninitialized fires inside Eigen's AVX-512 TRSM kernels once the ISA is raised;
    # disabling those kernels is the fix that keeps -Werror without silencing the warning globally.
    $<$<CXX_COMPILER_ID:GNU>:EIGEN_USE_AVX512_TRSM_KERNELS=0>
    $<$<BOOL:${PANTR_USE_STD_MDSPAN}>:PANTR_USE_STD_MDSPAN>
    $<$<BOOL:${PANTR_FORCE_SCALAR}>:PANTR_SIMD_FORCE_SCALAR>)
if(PANTR_FP_CONTRACT)
  target_compile_options(pantr INTERFACE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang,IntelLLVM>:-ffp-contract=on>
                                         $<$<CXX_COMPILER_ID:MSVC>:/fp:contract>)
endif()
# -march=native is not part of the exported target: pantrConfig.cmake decides it in the consumer
# from PANTR_NATIVE_ARCH. The build tree applies the same rule to pantr's own tests.
include(PantrNativeArch)
pantr_apply_native_arch(pantr "${PANTR_NATIVE_ARCH}")
pantr_report_fp_contract("${PANTR_FP_CONTRACT}")

# --- pantr::mpi_capsule (component mpi-capsule) --------------------------------------------------
# The C ABI shared by the extension modules: pantr's include directory and nothing else, so that a
# build against an installed pantr that carries no third-party headers can still use it.
add_library(pantr_mpi_capsule INTERFACE)
add_library(pantr::mpi_capsule ALIAS pantr_mpi_capsule)
set_target_properties(pantr_mpi_capsule PROPERTIES EXPORT_NAME mpi_capsule)
target_include_directories(
  pantr_mpi_capsule INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# --- pantr::mpi (component mpi) ------------------------------------------------------------------
# Built only where an MPI implementation is found; the distributed headers themselves land later.
find_package(MPI COMPONENTS CXX QUIET)
if(MPI_CXX_FOUND)
  add_library(pantr_mpi INTERFACE)
  add_library(pantr::mpi ALIAS pantr_mpi)
  set_target_properties(pantr_mpi PROPERTIES EXPORT_NAME mpi)
  target_link_libraries(pantr_mpi INTERFACE pantr MPI::MPI_CXX)
  message(STATUS "pantr: MPI found, the pantr::mpi target is built and exported")
else()
  message(STATUS "pantr: no MPI found, the pantr::mpi target (component mpi) is not built")
endif()

if(PANTR_INSTALL)
  include(PantrInstall)
endif()

if(PANTR_BUILD_TESTS)
  enable_testing()
  add_subdirectory(cpp/tests)
endif()
