# ctest driver: install pantr's build tree to a fresh prefix, then build and run the consumer
# project against that prefix alone.
#
# Inputs: BUILD_DIR, WORK_DIR, CONSUMER_SOURCE, CXX_COMPILER, GENERATOR, MAKE_PROGRAM, BUILD_TYPE,
# COMPONENTS (empty, or "mpi" when pantr::mpi was built).
cmake_minimum_required(VERSION 3.28)

set(prefix "${WORK_DIR}/prefix")
# The prefix and the consumer build are this test's own output, recreated on every run so a stale
# install can never satisfy it.
file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

function(run_step name)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "${name} failed (exit ${rc}):\n${out}${err}")
  endif()
  message(STATUS "[${name}] ok")
  set(last_output "${out}" PARENT_SCOPE)
endfunction()

run_step(install "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${prefix}")

# The native-ISA flag is the consumer's decision: the exported targets must not carry one.
file(GLOB targets_file "${prefix}/*/cmake/pantr/pantrTargets.cmake")
if(NOT targets_file)
  message(FATAL_ERROR "no pantrTargets.cmake under ${prefix}")
endif()
file(READ "${targets_file}" exported)
if(exported MATCHES "-m(arch|cpu)=")
  message(FATAL_ERROR "the installed pantrTargets.cmake carries an ISA flag")
endif()

# Core: find_package(pantr) with every search location except the fresh prefix switched off.
set(isolate
    -DCMAKE_FIND_USE_CMAKE_SYSTEM_PATH=OFF -DCMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH=OFF
    -DCMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH=OFF -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
    -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF -DCMAKE_FIND_USE_INSTALL_PREFIX=OFF)
set(common -G "${GENERATOR}" -DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}
           -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
           -DCMAKE_PREFIX_PATH=${prefix} -DPANTR_EXPECTED_PREFIX=${prefix})

# The core consumer leaves PANTR_NATIVE_ARCH at its default (ON: built and run on this machine);
# the mpi consumer turns it OFF. Each asserts the flag follows its own setting.
run_step(configure-core "${CMAKE_COMMAND}" -S "${CONSUMER_SOURCE}" -B "${WORK_DIR}/core" ${common}
         ${isolate})
run_step(build-core "${CMAKE_COMMAND}" --build "${WORK_DIR}/core")
run_step(run-core "${WORK_DIR}/core/pantr_consumer")
message(STATUS "consumer output: ${last_output}")

# The same consumer with PANTR_NATIVE_ARCH=OFF: configuring asserts the flag is absent.
run_step(configure-core-off "${CMAKE_COMMAND}" -S "${CONSUMER_SOURCE}" -B "${WORK_DIR}/core-off"
         ${common} ${isolate} -DPANTR_NATIVE_ARCH=OFF)

# Component mpi-capsule alone loads the capsule target and nothing else.
run_step(configure-capsule "${CMAKE_COMMAND}" -S "${CONSUMER_SOURCE}" -B "${WORK_DIR}/capsule"
         ${common} ${isolate} -DCONSUMER_CAPSULE_ONLY=ON)

# Component mpi: MPI itself is found where the system keeps it, so only pantr is isolated (the
# consumer asserts pantr_DIR is under the prefix).
if(COMPONENTS STREQUAL "mpi")
  run_step(configure-mpi "${CMAKE_COMMAND}" -S "${CONSUMER_SOURCE}" -B "${WORK_DIR}/mpi" ${common}
           -DCONSUMER_WITH_MPI=ON -DPANTR_NATIVE_ARCH=OFF -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
           -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF)
  run_step(build-mpi "${CMAKE_COMMAND}" --build "${WORK_DIR}/mpi")
endif()
