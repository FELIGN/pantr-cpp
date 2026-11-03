# Fail unless every CMake script under SCRIPT_DIR calls cmake_minimum_required before any other
# command. A script run with `cmake -P` and no such call runs with every policy unset, so on
# CMake 3.x a construct such as if(... IN_LIST ...) (CMP0057) is an error there, while CMake 4,
# which dropped the OLD behavior of policies older than 3.5, accepts it: the script passes
# locally and fails on an older CMake.
cmake_minimum_required(VERSION 3.28)

file(GLOB_RECURSE scripts "${SCRIPT_DIR}/*.cmake")
set(offenders "")
foreach(script IN LISTS scripts)
  file(STRINGS "${script}" lines)
  set(first "")
  foreach(line IN LISTS lines)
    if(line MATCHES "^[ \t]*(#.*)?$")
      continue()
    endif()
    set(first "${line}")
    break()
  endforeach()
  if(NOT first MATCHES "^cmake_minimum_required\\(")
    list(APPEND offenders "${script}")
  endif()
endforeach()
list(LENGTH scripts n)
if(n EQUAL 0)
  message(FATAL_ERROR "no scripts found under ${SCRIPT_DIR}: the check would pass vacuously")
endif()
if(offenders)
  message(FATAL_ERROR "scripts without a leading cmake_minimum_required: ${offenders}")
endif()
message(STATUS "checked ${n} scripts: each sets its policies")
