# The _pantr_mpi extension module (package pantr_mpi, distribution pantr-mpi), design §1.7 and
# design/mpi.md.
#
# Built two ways:
#   - from pantr's own build (bindings/CMakeLists.txt, PANTR_BUILD_MPI_MODULE), against the tree's
#     headers, installed in the `mpi` component (make ext puts it in src/pantr_mpi/);
#   - on its own, by bindings/mpi/pyproject.toml (scikit-build-core), against the headers of this
#     source tree (PANTR_INCLUDE_DIR, by default ../../include) or, from the pantr-mpi sdist, of
#     an installed pantr found by find_package(pantr COMPONENTS mpi-capsule): scikit-build-core
#     searches the build environment's site-packages, and elsewhere pass
#     -Dpantr_DIR=$(python -c "import pantr; print(pantr.cmake_dir())").
# It needs only comm_capsule.h and mpi_table.hpp from pantr, an MPI with its C++ wrapper, mpi4py's C
# headers (mpi4py >= 4.0 for the stable ABI) and the nanobind pinned in pantr's pyproject.toml.
#
# The MPI family is recorded in pantr_mpi/_build_info.py, which pantr.mpi reads before loading the
# extension and compares with mpi4py.MPI.Get_library_version(): `openmpi` or `mpich` (the MPICH ABI:
# MPICH, Intel MPI, MVAPICH), read from mpi.h unless PANTR_MPI_FAMILY is set; `custom` disables
# the check.
cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED PROJECT_NAME)
  project(pantr_mpi LANGUAGES CXX)
  set(_pantr_mpi_standalone ON)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  option(PANTR_STABLE_ABI "Build the stable-ABI (abi3) module" ON)
  set(_pantr_python_components Interpreter Development.Module)
  if(PANTR_STABLE_ABI)
    list(APPEND _pantr_python_components Development.SABIModule)
  endif()
  find_package(Python 3.12 REQUIRED COMPONENTS ${_pantr_python_components})
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  find_package(nanobind CONFIG REQUIRED)
  file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _pin REGEX "\"nanobind==[^\"]+\"")
  if(NOT _pin MATCHES "\"nanobind==([^\"]+)\"" OR NOT nanobind_VERSION VERSION_EQUAL CMAKE_MATCH_1)
    message(FATAL_ERROR "pantr-mpi: nanobind ${nanobind_VERSION} does not match its exact pin")
  endif()
  set(PANTR_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
      CACHE PATH "pantr's include directory (holding pantr/mpi/comm_capsule.h)")
  add_library(_pantr_mpi_capsule INTERFACE)
  if(PANTR_INCLUDE_DIR AND EXISTS "${PANTR_INCLUDE_DIR}/pantr/mpi/comm_capsule.h")
    target_include_directories(_pantr_mpi_capsule INTERFACE "${PANTR_INCLUDE_DIR}")
    message(STATUS "pantr-mpi: pantr's headers from ${PANTR_INCLUDE_DIR}")
  else()
    find_package(pantr REQUIRED COMPONENTS mpi-capsule)
    target_link_libraries(_pantr_mpi_capsule INTERFACE pantr::mpi_capsule)
    message(STATUS "pantr-mpi: pantr's headers from the installed pantr at ${pantr_DIR}")
  endif()
  set(_pantr_mpi_capsule_target _pantr_mpi_capsule)
  set(_pantr_mpi_destination pantr_mpi)
  set(_pantr_mpi_warnings "")
  if(PANTR_STABLE_ABI)
    set(_pantr_nb_abi STABLE_ABI)
  else()
    set(_pantr_nb_abi "")
  endif()
else()
  set(_pantr_mpi_standalone OFF)
  set(_pantr_mpi_capsule_target pantr::mpi_capsule)
  set(_pantr_mpi_destination pantr_mpi)
  set(_pantr_mpi_warnings pantr_cxx_warnings)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import mpi4py; print(mpi4py.get_include())"
  OUTPUT_VARIABLE _pantr_mpi4py_include
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _pantr_mpi4py_result)
if(NOT _pantr_mpi4py_result EQUAL 0)
  message(FATAL_ERROR "pantr-mpi: mpi4py is not importable by ${Python_EXECUTABLE}")
endif()

# The MPI family the wheel is built for, written into _build_info.py: read from the macros of the
# mpi.h the module compiles against (Open MPI defines OMPI_MAJOR_VERSION; MPICH and the
# implementations of its ABI, Intel MPI and MVAPICH, define MPICH_VERSION).
set(PANTR_MPI_FAMILY "auto" CACHE STRING "MPI family recorded for the import check: auto, openmpi, mpich or custom")
find_file(_pantr_mpi_h mpi.h PATHS ${MPI_CXX_INCLUDE_DIRS} NO_DEFAULT_PATH NO_CACHE)
set(_pantr_mpi_version_line "unknown MPI")
set(_pantr_mpi_detected custom)
if(_pantr_mpi_h)
  file(STRINGS "${_pantr_mpi_h}" _pantr_mpi_macros
       REGEX "^#[ \t]*define[ \t]+(OMPI_MAJOR_VERSION|OMPI_MINOR_VERSION|OMPI_RELEASE_VERSION|I_MPI_VERSION|MPICH_VERSION|MVAPICH2_VERSION)[ \t]")
  foreach(_line IN LISTS _pantr_mpi_macros)
    if(_line MATCHES "define[ \t]+([A-Z0-9_]+)[ \t]+\"?([^\" \t]+)")
      set(_pantr_mpi_macro_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
    endif()
  endforeach()
  if(DEFINED _pantr_mpi_macro_OMPI_MAJOR_VERSION)
    set(_pantr_mpi_detected openmpi)
    set(_pantr_mpi_version_line "Open MPI v${_pantr_mpi_macro_OMPI_MAJOR_VERSION}.${_pantr_mpi_macro_OMPI_MINOR_VERSION}.${_pantr_mpi_macro_OMPI_RELEASE_VERSION}")
  elseif(DEFINED _pantr_mpi_macro_I_MPI_VERSION)
    set(_pantr_mpi_detected mpich)
    set(_pantr_mpi_version_line "Intel(R) MPI Library ${_pantr_mpi_macro_I_MPI_VERSION}")
  elseif(DEFINED _pantr_mpi_macro_MVAPICH2_VERSION)
    set(_pantr_mpi_detected mpich)
    set(_pantr_mpi_version_line "MVAPICH2 ${_pantr_mpi_macro_MVAPICH2_VERSION}")
  elseif(DEFINED _pantr_mpi_macro_MPICH_VERSION)
    set(_pantr_mpi_detected mpich)
    set(_pantr_mpi_version_line "MPICH ${_pantr_mpi_macro_MPICH_VERSION}")
  endif()
endif()
if(PANTR_MPI_FAMILY STREQUAL "auto")
  set(_pantr_mpi_family ${_pantr_mpi_detected})
  if(_pantr_mpi_family STREQUAL "custom")
    message(WARNING "pantr-mpi: the MPI family of ${_pantr_mpi_h} is not Open MPI or the MPICH ABI; "
                    "recording 'custom', which turns the import-time family check off. Set "
                    "PANTR_MPI_FAMILY=custom to say so explicitly.")
  endif()
else()
  set(_pantr_mpi_family "${PANTR_MPI_FAMILY}")
endif()
if(NOT _pantr_mpi_family MATCHES "^(openmpi|mpich|custom)$")
  message(FATAL_ERROR "PANTR_MPI_FAMILY must be auto, openmpi, mpich or custom; got ${PANTR_MPI_FAMILY}")
endif()
set(_pantr_mpi_version_py "${_pantr_mpi_version_line}")
message(STATUS "pantr-mpi: built for MPI family '${_pantr_mpi_family}' (${_pantr_mpi_version_line})")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/pantr_mpi/_build_info.py.in"
               "${CMAKE_CURRENT_BINARY_DIR}/_build_info.py" @ONLY)

nanobind_add_module(_pantr_mpi ${_pantr_nb_abi} NB_STATIC NOMINSIZE NOSTRIP NB_SUPPRESS_WARNINGS
                    src/module.cpp)
target_compile_features(_pantr_mpi PRIVATE cxx_std_20)
target_link_libraries(_pantr_mpi PRIVATE ${_pantr_mpi_capsule_target} MPI::MPI_CXX
                                         ${_pantr_mpi_warnings})
target_include_directories(_pantr_mpi SYSTEM PRIVATE "${_pantr_mpi4py_include}")
install(TARGETS _pantr_mpi LIBRARY DESTINATION ${_pantr_mpi_destination} COMPONENT mpi
        RUNTIME DESTINATION ${_pantr_mpi_destination} COMPONENT mpi)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/python/pantr_mpi/__init__.py"
              "${CMAKE_CURRENT_BINARY_DIR}/_build_info.py"
        DESTINATION ${_pantr_mpi_destination} COMPONENT mpi)
