# ctest driver: configure pantr with PANTR_NATIVE_ARCH=ON in a fresh tree, install it, and check
# that the exported targets carry no ISA flag. The suite's own build may have the option OFF, where
# the same check on its install would pass whatever the export did.
#
# Inputs: SOURCE_DIR, WORK_DIR, CXX_COMPILER, GENERATOR, MAKE_PROGRAM, DEP_SOURCES (a |-separated
# list of -DFETCHCONTENT_SOURCE_DIR_<NAME>=<dir> arguments, so nothing is fetched again).
cmake_minimum_required(VERSION 3.28)

# The tree and prefix are this test's own output, recreated on every run.
file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")
string(REPLACE "|" ";" dep_args "${DEP_SOURCES}")

function(run_step name)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "${name} failed (exit ${rc}):\n${out}${err}")
  endif()
  message(STATUS "[${name}] ok")
endfunction()

run_step(configure "${CMAKE_COMMAND}" -S "${SOURCE_DIR}" -B "${WORK_DIR}/build" -G "${GENERATOR}"
         -DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM} -DCMAKE_CXX_COMPILER=${CXX_COMPILER}
         -DPANTR_NATIVE_ARCH=ON -DPANTR_BUILD_TESTS=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON
         ${dep_args})
run_step(install "${CMAKE_COMMAND}" --install "${WORK_DIR}/build" --prefix "${WORK_DIR}/prefix")
file(GLOB targets_file "${WORK_DIR}/prefix/*/cmake/pantr/pantrTargets.cmake")
if(NOT targets_file)
  message(FATAL_ERROR "no pantrTargets.cmake under ${WORK_DIR}/prefix")
endif()
file(READ "${targets_file}" exported)
if(exported MATCHES "-m(arch|cpu)=")
  message(FATAL_ERROR "with PANTR_NATIVE_ARCH=ON, the exported targets carry an ISA flag")
endif()
message(STATUS "exported targets carry no ISA flag")
