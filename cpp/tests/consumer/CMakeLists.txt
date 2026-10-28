# A downstream project using an installed pantr through find_package and nothing else.
cmake_minimum_required(VERSION 3.28)
project(pantr_consumer LANGUAGES CXX)

# No C++ modules here; without this, CMake looks for clang-scan-deps, which the isolated search
# below cannot find.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

option(CONSUMER_WITH_MPI "Also request the mpi component" OFF)
option(CONSUMER_CAPSULE_ONLY "Request only the mpi-capsule component and check what it loads" OFF)

if(CONSUMER_CAPSULE_ONLY)
  # What the pantr-mpi extension build does against a pantr that carries no third-party headers:
  # the capsule target alone, without the core target or its dependencies.
  find_package(pantr 0.7 REQUIRED COMPONENTS mpi-capsule)
  if(NOT TARGET pantr::mpi_capsule OR TARGET pantr::pantr)
    message(FATAL_ERROR "COMPONENTS mpi-capsule must define pantr::mpi_capsule and not pantr::pantr")
  endif()
  return()
elseif(CONSUMER_WITH_MPI)
  find_package(pantr 0.7 REQUIRED COMPONENTS mpi)
else()
  find_package(pantr 0.7 REQUIRED)
endif()

# The package must come from the prefix under test, not from any other installation.
if(DEFINED PANTR_EXPECTED_PREFIX)
  cmake_path(IS_PREFIX PANTR_EXPECTED_PREFIX "${pantr_DIR}" NORMALIZE _under_prefix)
  if(NOT _under_prefix)
    message(FATAL_ERROR "pantr found at ${pantr_DIR}, outside ${PANTR_EXPECTED_PREFIX}")
  endif()
endif()

# The native-ISA flag follows PANTR_NATIVE_ARCH as set here, in the consumer (default ON).
if(NOT CMAKE_CROSSCOMPILING AND NOT MSVC)
  get_target_property(_opts pantr::pantr INTERFACE_COMPILE_OPTIONS)
  string(REGEX MATCH "-m(arch|cpu)=native" _native "${_opts}")
  if(NOT DEFINED PANTR_NATIVE_ARCH OR PANTR_NATIVE_ARCH)
    set(_want_native TRUE)
  else()
    set(_want_native FALSE)
  endif()
  if(_want_native AND NOT _native)
    message(FATAL_ERROR "PANTR_NATIVE_ARCH is ON but pantr::pantr carries no native flag: ${_opts}")
  elseif(NOT _want_native AND _native)
    message(FATAL_ERROR "PANTR_NATIVE_ARCH is OFF but pantr::pantr carries ${_native}")
  endif()
endif()

add_executable(pantr_consumer main.cpp)
set_target_properties(pantr_consumer PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON
                                                CXX_EXTENSIONS OFF)
target_link_libraries(pantr_consumer PRIVATE pantr::pantr pantr::mpi_capsule)
if(CONSUMER_WITH_MPI)
  target_link_libraries(pantr_consumer PRIVATE pantr::mpi)
endif()
