# Check pantr_mpiexec_needs_oversubscribe (cmake/PantrMpiexecFlags.cmake, under MODULE_DIR) on the
# `mpiexec --version` text of each MPI family. Open MPI refuses more ranks than
# slots unless asked to oversubscribe; the other launchers have no such flag.
cmake_minimum_required(VERSION 3.28)
include("${MODULE_DIR}/PantrMpiexecFlags.cmake")

set(failures "")
function(expect text want)
  pantr_mpiexec_needs_oversubscribe(got "${text}")
  if(NOT got STREQUAL want)
    set(failures "${failures}\n  '${text}': got ${got}, expected ${want}" PARENT_SCOPE)
  endif()
endfunction()

# Open MPI 4 names its runtime OpenRTE; Open MPI 5 names itself.
expect("mpiexec (OpenRTE) 4.1.6\n\nReport bugs to http://www.open-mpi.org/community/help/" TRUE)
expect("mpiexec (Open MPI) 5.0.3\n\nReport bugs to https://www.open-mpi.org/community/help/" TRUE)
expect("Intel(R) MPI Library for Linux* OS, Version 2021.18.0 Build 20260327 (id: 821f207)" FALSE)
expect("HYDRA build details:\n    Version:                                 4.1.2" FALSE)
expect("" FALSE)

if(failures)
  message(FATAL_ERROR "wrong oversubscribe decision:${failures}")
endif()
message(STATUS "oversubscribe decided right for every launcher text")
