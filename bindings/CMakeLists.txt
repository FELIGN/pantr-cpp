# The Python extension modules: _pantr (one copy per ISA rung) and the _pantr_cpu probe.
#
# Layout and reasons: design/port_design.md, sections 1.8 and 1.9, and design/packaging.md.
# Every module is built NB_STATIC, NOMINSIZE (nanobind otherwise appends -Os after -O3 and the last
# flag wins, so kernels silently compile for size; design/lessons_first_port.md), NOSTRIP (nanobind otherwise links release modules
# with -s; pyproject.toml's install.strip = false covers scikit-build-core's own strip) and
# NB_SUPPRESS_WARNINGS (nanobind's and Python's include directories become SYSTEM, so -Werror
# applies to pantr's code only).

# --- Python and nanobind -------------------------------------------------------------------------
# The stable ABI needs the SABIModule component: without it nanobind silently builds a
# version-specific module. scikit-build-core asks for it through SKBUILD_SABI_COMPONENT when the
# wheel is tagged abi3; a plain configure (make ext) asks for it from PANTR_STABLE_ABI.
set(_pantr_python_components Interpreter Development.Module)
if(PANTR_STABLE_ABI)
  list(APPEND _pantr_python_components Development.SABIModule)
endif()
find_package(Python 3.12 REQUIRED COMPONENTS ${_pantr_python_components})

if(NOT nanobind_ROOT AND NOT DEFINED ENV{nanobind_ROOT})
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pantr_nb_result)
  if(NOT _pantr_nb_result EQUAL 0)
    message(FATAL_ERROR "pantr: nanobind is not importable by ${Python_EXECUTABLE}; "
                        "install the version pinned in pyproject.toml")
  endif()
endif()
find_package(nanobind CONFIG REQUIRED)

# One nanobind pin for every pantr extension module (_pantr, _pantr_mpi, _pantr_berf): modules that
# exchange nanobind objects must agree on its ABI. The pin lives in pyproject.toml's build
# requirements; a build against any other nanobind stops here.
file(STRINGS "${PROJECT_SOURCE_DIR}/pyproject.toml" _pantr_nb_pin_line REGEX "\"nanobind==[^\"]+\"")
if(NOT _pantr_nb_pin_line MATCHES "\"nanobind==([^\"]+)\"")
  message(FATAL_ERROR "pantr: no exact nanobind pin (\"nanobind==X.Y.Z\") in pyproject.toml")
endif()
set(PANTR_NANOBIND_PIN "${CMAKE_MATCH_1}")
if(NOT nanobind_VERSION VERSION_EQUAL PANTR_NANOBIND_PIN)
  message(FATAL_ERROR "pantr: nanobind ${nanobind_VERSION} found, pyproject.toml pins "
                      "${PANTR_NANOBIND_PIN}")
endif()

if(PANTR_STABLE_ABI)
  set(_pantr_nb_abi STABLE_ABI)
else()
  set(_pantr_nb_abi "")
endif()

# Every pantr module installs into the `pantr` package, in the `python` component: the wheel
# installs that component only, and `make ext` installs it into src/.
set(PANTR_PYTHON_DESTINATION pantr)

# --- The ISA rungs ------------------------------------------------------------------------------
# The rungs follow the target architecture, not the build host: on Apple an explicit
# CMAKE_OSX_ARCHITECTURES (a cross-compiled wheel) overrides CMAKE_SYSTEM_PROCESSOR.
set(_pantr_target_arch "${CMAKE_SYSTEM_PROCESSOR}")
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
  list(LENGTH CMAKE_OSX_ARCHITECTURES _pantr_osx_arch_count)
  if(_pantr_osx_arch_count GREATER 1)
    message(FATAL_ERROR "pantr: one architecture per extension build; "
                        "CMAKE_OSX_ARCHITECTURES='${CMAKE_OSX_ARCHITECTURES}'")
  endif()
  set(_pantr_target_arch "${CMAKE_OSX_ARCHITECTURES}")
endif()
if(_pantr_target_arch MATCHES "^(x86_64|amd64|AMD64)$")
  set(PANTR_EXT_ARCH x86_64)
elseif(_pantr_target_arch MATCHES "^(arm64|aarch64|ARM64)$")
  set(PANTR_EXT_ARCH arm64)
else()
  set(PANTR_EXT_ARCH other)
endif()

set(_pantr_ext_sources src/module.cpp src/parallel.cpp src/testing.cpp src/quad.cpp src/basis.cpp
                       src/change_basis.cpp src/geometry.cpp
                       src/bezier.cpp src/bezier_approx.cpp src/roots.cpp src/bspline.cpp
                       src/bspline_ops.cpp src/bspline_refine.cpp src/bspline_modify.cpp)

# Build one copy of the extension: module name <name>, rung <isa>, extra compile options in ARGN.
function(pantr_add_extension name isa)
  nanobind_add_module(${name} ${_pantr_nb_abi} NB_STATIC NOMINSIZE NOSTRIP NB_SUPPRESS_WARNINGS
                      ${_pantr_ext_sources})
  target_link_libraries(${name} PRIVATE pantr::pantr pantr_cxx_warnings)
  target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_compile_definitions(${name} PRIVATE PANTR_EXT_MODULE_NAME=${name} PANTR_EXT_ISA="${isa}")
  if(ARGN)
    target_compile_options(${name} PRIVATE ${ARGN})
  endif()
  install(TARGETS ${name} LIBRARY DESTINATION ${PANTR_PYTHON_DESTINATION} COMPONENT python
          RUNTIME DESTINATION ${PANTR_PYTHON_DESTINATION} COMPONENT python)
endfunction()

# The base module: the architecture baseline on x86-64; on arm64 under the rungs, forced scalar, so
# that the NEON rung is a distinct choice (NEON is the AArch64 baseline, so a non-scalar base would
# be the NEON module twice).
pantr_add_extension(_pantr generic)
if(PANTR_MULTI_ISA)
  if(PANTR_EXT_ARCH STREQUAL "x86_64")
    if(MSVC)
      pantr_add_extension(_pantr_v3 v3 /arch:AVX2)
      pantr_add_extension(_pantr_v4 v4 /arch:AVX512)
      # MSVC's optimizer does not finish Eigen's AVX-512 matrix-product kernels (seen with 14.44);
      # Eigen runs unvectorized in this rung, pantr's own kernels keep AVX-512.
      target_compile_definitions(_pantr_v4 PRIVATE EIGEN_DONT_VECTORIZE)
    else()
      pantr_add_extension(_pantr_v3 v3 -march=x86-64-v3)
      pantr_add_extension(_pantr_v4 v4 -march=x86-64-v4)
    endif()
  elseif(PANTR_EXT_ARCH STREQUAL "arm64")
    target_compile_definitions(_pantr PRIVATE PANTR_SIMD_FORCE_SCALAR)
    pantr_add_extension(_pantr_neon neon)
  else()
    # The base module is the portable one: a source build on another architecture still works.
    message(STATUS "pantr: no ISA rungs for '${_pantr_target_arch}'; building the base module only")
  endif()
endif()

# The CPU probe the ladder consults before importing a tuned rung (pantr/_isa.py). Compiled at the
# toolchain baseline with no pantr header, so it loads on any CPU of the architecture.
nanobind_add_module(_pantr_cpu ${_pantr_nb_abi} NB_STATIC NOMINSIZE NOSTRIP NB_SUPPRESS_WARNINGS
                    src/cpu.cpp)
target_compile_features(_pantr_cpu PRIVATE cxx_std_20)
target_link_libraries(_pantr_cpu PRIVATE pantr_cxx_warnings)
install(TARGETS _pantr_cpu LIBRARY DESTINATION ${PANTR_PYTHON_DESTINATION} COMPONENT python
        RUNTIME DESTINATION ${PANTR_PYTHON_DESTINATION} COMPONENT python)

# --- Test-only modules --------------------------------------------------------------------------
# A second extension built from the same headers, as _pantr_mpi and _pantr_berf will be: the twin
# routes its parallel state through _pantr's table, the unrouted twin is the control that keeps its
# own (tests/test_shared_parallelism.py). Installed in the `testing` component, never in a wheel.
if(PANTR_BUILD_PYTHON_TEST_MODULES)
  foreach(_twin _pantr_twin _pantr_twin_unrouted)
    nanobind_add_module(${_twin} ${_pantr_nb_abi} NB_STATIC NOMINSIZE NOSTRIP NB_SUPPRESS_WARNINGS
                        twin/twin.cpp)
    target_link_libraries(${_twin} PRIVATE pantr::pantr pantr_cxx_warnings)
    target_include_directories(${_twin} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_definitions(${_twin} PRIVATE PANTR_TWIN_MODULE_NAME=${_twin})
    install(TARGETS ${_twin} LIBRARY DESTINATION ${PANTR_PYTHON_DESTINATION} COMPONENT testing
            RUNTIME DESTINATION ${PANTR_PYTHON_DESTINATION} COMPONENT testing)
  endforeach()
  target_compile_definitions(_pantr_twin PRIVATE PANTR_TWIN_ROUTED=1)
  target_compile_definitions(_pantr_twin_unrouted PRIVATE PANTR_TWIN_ROUTED=0)
endif()
