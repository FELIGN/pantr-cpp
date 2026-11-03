# Launcher flags for pantr's own MPI tests.
include_guard(GLOBAL)

# Set <out> to TRUE when <version_text>, the output of `mpiexec --version`, is Open MPI's: its
# launcher refuses more ranks than slots (cores, by default) unless given --oversubscribe, which the
# other launchers do not accept. Open MPI 4 names its runtime OpenRTE, Open MPI 5 names itself, and
# both print the project's bug-report address. The launcher is asked rather than FindMPI's
# MPI_CXX_LIBRARY_VERSION_STRING, which stays empty unless MPI_DETERMINE_LIBRARY_VERSION is set.
function(pantr_mpiexec_needs_oversubscribe out version_text)
  if(version_text MATCHES "\\((OpenRTE|Open MPI)\\)|open-mpi\\.org")
    set(${out} TRUE PARENT_SCOPE)
  else()
    set(${out} FALSE PARENT_SCOPE)
  endif()
endfunction()
