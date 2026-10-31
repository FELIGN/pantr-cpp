# The C++ test suite. It owns what the Python suite cannot observe: the C++ API's validation and
# exception types, the pool's concurrency semantics, lazy blocks under TSan, the installed
# package, and the header rules (no mpi.h in the core, no fast-math).
include("${pantr_doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake")

set(_pantr_test_sources
    test_main.cpp
    test_config.cpp
    test_errors.cpp
    test_format.cpp
    test_ndarray.cpp
    test_mdspan.cpp
    test_lazy.cpp
    test_simd.cpp
    test_exact_rational.cpp
    test_affinity.cpp
    test_work_model.cpp
    test_parallel.cpp
    test_dependencies.cpp
    test_quad.cpp
    test_basis.cpp
    test_change_basis.cpp
    test_binomial.cpp
    test_bezier_eval.cpp
    test_bezier_ops.cpp
    test_bezier_subdivide.cpp
    test_bezier_reduce.cpp
    test_bezier_fit.cpp
    test_roots.cpp
    test_geometry.cpp
    test_transform.cpp
    test_bspline_knots.cpp
    test_bspline_basis_1d.cpp
    test_bspline_space.cpp
    test_bspline_space_nd.cpp
    test_bspline.cpp
    test_bspline_knot_insertion.cpp
    test_bspline_knot_removal.cpp
    test_bspline_degree.cpp
    test_bspline_derivative.cpp
    test_bspline_slice.cpp
    test_bspline_reparametrize.cpp
    test_bspline_extraction.cpp)
add_executable(pantr_tests ${_pantr_test_sources})
target_link_libraries(pantr_tests PRIVATE pantr::pantr pantr_native_arch pantr_doctest
                                          pantr_cxx_warnings)
doctest_discover_tests(pantr_tests)

# The scalar SIMD fallback, as the forced-scalar build sees it.
add_executable(pantr_tests_scalar test_main.cpp test_simd.cpp)
target_link_libraries(pantr_tests_scalar PRIVATE pantr::pantr pantr_native_arch pantr_doctest
                                                 pantr_cxx_warnings)
target_compile_definitions(pantr_tests_scalar PRIVATE PANTR_SIMD_FORCE_SCALAR PANTR_TEST_EXPECT_SCALAR)
doctest_discover_tests(pantr_tests_scalar TEST_PREFIX "scalar: ")

# Routing through another module's parallel state: a binary of its own, since an attached table
# cannot be detached.
add_executable(pantr_tests_shared test_main.cpp test_shared.cpp)
target_link_libraries(pantr_tests_shared PRIVATE pantr::pantr pantr_native_arch pantr_doctest
                                                 pantr_cxx_warnings)
doctest_discover_tests(pantr_tests_shared)

# PANTR_NUM_THREADS wiring: one process per setting, since max_threads() reads it once.
add_executable(pantr_tests_env test_main.cpp test_env.cpp)
target_link_libraries(pantr_tests_env PRIVATE pantr::pantr pantr_native_arch pantr_doctest
                                              pantr_cxx_warnings)
add_test(NAME env.num_threads_lowers_capacity COMMAND pantr_tests_env)
set_tests_properties(env.num_threads_lowers_capacity PROPERTIES
                     ENVIRONMENT "PANTR_NUM_THREADS=2;PANTR_TEST_EXPECT=2")
add_test(NAME env.num_threads_cannot_raise_capacity COMMAND pantr_tests_env)
set_tests_properties(env.num_threads_cannot_raise_capacity PROPERTIES
                     ENVIRONMENT "PANTR_NUM_THREADS=100000;PANTR_TEST_EXPECT=100000")
add_test(NAME env.num_threads_malformed COMMAND pantr_tests_env)
set_tests_properties(env.num_threads_malformed PROPERTIES
                     ENVIRONMENT "PANTR_NUM_THREADS=two;PANTR_TEST_EXPECT=malformed")

# --- Header rules ---------------------------------------------------------------------------------
# Every public header compiles on its own (self-contained) with a poisoned mpi.h first on the
# include path, so a core header that reaches mpi.h, directly or through another header, fails the
# build. The MPI communicator header is the one header allowed to include it.
set(PANTR_MPI_HEADER_ALLOWED "pantr/mpi/mpi_communicator.hpp")
file(GLOB_RECURSE _pantr_headers CONFIGURE_DEPENDS RELATIVE "${PROJECT_SOURCE_DIR}/include"
     "${PROJECT_SOURCE_DIR}/include/pantr/*.hpp" "${PROJECT_SOURCE_DIR}/include/pantr/*.h")
set(_pantr_header_tus "")
foreach(_h IN LISTS _pantr_headers)
  if(_h STREQUAL PANTR_MPI_HEADER_ALLOWED)
    continue()
  endif()
  string(MAKE_C_IDENTIFIER "${_h}" _id)
  set(_tu "${CMAKE_CURRENT_BINARY_DIR}/header_check/${_id}.cpp")
  file(CONFIGURE OUTPUT "${_tu}" CONTENT "#include <${_h}>\n")
  list(APPEND _pantr_header_tus "${_tu}")
endforeach()
add_library(pantr_header_check OBJECT ${_pantr_header_tus})
target_include_directories(pantr_header_check BEFORE PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/header_rules/poison_mpi")
target_link_libraries(pantr_header_check PRIVATE pantr::pantr pantr_native_arch pantr_cxx_warnings)

# The poison works: a unit that includes mpi.h through the same include path fails with its message.
add_library(pantr_poison_control OBJECT EXCLUDE_FROM_ALL header_rules/poison_control.cpp)
target_include_directories(pantr_poison_control BEFORE PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/header_rules/poison_mpi")
add_test(NAME header_rules.poison_mpi_control
         COMMAND "${CMAKE_COMMAND}" --build "${PROJECT_BINARY_DIR}" --target pantr_poison_control)
set_tests_properties(header_rules.poison_mpi_control PROPERTIES
                     PASS_REGULAR_EXPRESSION "poisoned mpi.h reached")

# No header names mpi.h at all, outside the allowed one (catches a guarded include the compile
# check would not reach).
add_test(NAME header_rules.no_mpi_include_text
         COMMAND "${CMAKE_COMMAND}" -DINCLUDE_DIR=${PROJECT_SOURCE_DIR}/include
                 -DALLOWED=${PANTR_MPI_HEADER_ALLOWED} -P "${CMAKE_CURRENT_SOURCE_DIR}/header_rules/check_no_mpi.cmake")

# config.hpp rejects fast-math, under each flag that enables it. MSVC has /fp:fast only: no flag of
# its own assumes finite math.
if(MSVC)
  set(_pantr_fast_math_flags /fp:fast)
else()
  set(_pantr_fast_math_flags -ffast-math -ffinite-math-only)
endif()
set(_pantr_fast_math_tests "")
foreach(_flag ${_pantr_fast_math_flags})
  string(MAKE_C_IDENTIFIER "${_flag}" _id)
  string(REGEX REPLACE "[/:]" "-" _name "${_flag}")
  add_library(pantr_reject${_id} OBJECT EXCLUDE_FROM_ALL header_rules/fast_math_probe.cpp)
  target_link_libraries(pantr_reject${_id} PRIVATE pantr::pantr)
  target_compile_options(pantr_reject${_id} PRIVATE ${_flag})
  add_test(NAME header_rules.rejects${_name}
           COMMAND "${CMAKE_COMMAND}" --build "${PROJECT_BINARY_DIR}" --target pantr_reject${_id})
  set_tests_properties(header_rules.rejects${_name} PROPERTIES
                       PASS_REGULAR_EXPRESSION "must not be compiled with fast-math")
  list(APPEND _pantr_fast_math_tests header_rules.rejects${_name})
endforeach()

# Every pinned dependency commit is listed, with its license, in THIRD_PARTY_NOTICES.md.
add_test(NAME notices.pins_listed
         COMMAND "${CMAKE_COMMAND}" -DDEPS_FILE=${PROJECT_SOURCE_DIR}/cmake/PantrDependencies.cmake
                 -DNOTICES_FILE=${PROJECT_SOURCE_DIR}/THIRD_PARTY_NOTICES.md
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/header_rules/check_notices.cmake")

# --- Installed package ---------------------------------------------------------------------------
# Install this build to a fresh prefix, then configure, build and run cpp/tests/consumer against
# that prefix alone.
add_test(NAME install.consumer
         COMMAND "${CMAKE_COMMAND}" -DBUILD_DIR=${PROJECT_BINARY_DIR}
                 -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/consumer_run
                 -DCONSUMER_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/consumer
                 -DCXX_COMPILER=${CMAKE_CXX_COMPILER} -DGENERATOR=${CMAKE_GENERATOR}
                 -DMAKE_PROGRAM=${CMAKE_MAKE_PROGRAM} -DMT_PROGRAM=${CMAKE_MT}
                 -DBUILD_TYPE=${CMAKE_BUILD_TYPE} -DCOMPONENTS=$<IF:$<TARGET_EXISTS:pantr_mpi>,mpi,>
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/consumer/run_consumer_test.cmake")
set_tests_properties(install.consumer PROPERTIES TIMEOUT 600)

# The exported targets carry no ISA flag even when pantr itself is built with PANTR_NATIVE_ARCH=ON.
set(_pantr_dep_sources "")
foreach(_dep EIGEN SPECTRA MDSPAN BOOST_MULTIPRECISION BOOST_CONFIG)
  string(TOLOWER "${_dep}" _lower)
  list(APPEND _pantr_dep_sources "-DFETCHCONTENT_SOURCE_DIR_PANTR_${_dep}=${pantr_${_lower}_SOURCE_DIR}")
endforeach()
string(REPLACE ";" "|" _pantr_dep_sources "${_pantr_dep_sources}")
add_test(NAME install.native_flag_not_exported
         COMMAND "${CMAKE_COMMAND}" -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
                 -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/native_export_run
                 -DCXX_COMPILER=${CMAKE_CXX_COMPILER} -DGENERATOR=${CMAKE_GENERATOR}
                 -DMAKE_PROGRAM=${CMAKE_MAKE_PROGRAM} "-DDEP_SOURCES=${_pantr_dep_sources}"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/consumer/run_native_export_test.cmake")
set_tests_properties(install.native_flag_not_exported PROPERTIES TIMEOUT 600)

# The tests above drive `cmake --build` or `cmake --install` on this very build tree; two of them at
# once would regenerate it concurrently.
set_tests_properties(header_rules.poison_mpi_control ${_pantr_fast_math_tests} install.consumer
                     PROPERTIES RESOURCE_LOCK pantr_build_tree)
