# FindMETIS: the METIS graph partitioner (Apache-2.0), which the user provides; pantr never fetches
# or bundles it.
#
# Result: METIS_FOUND and the imported target METIS::metis. Search hints: METIS_ROOT (CMake's
# <PackageName>_ROOT), or METIS_INCLUDE_DIR and METIS_LIBRARY set directly.
find_library(METIS_LIBRARY NAMES metis)
# The header from the library's own prefix first: a metis.h of another installation can declare
# another index width than the library was built with.
set(_metis_hints "")
if(METIS_LIBRARY)
  get_filename_component(_metis_libdir "${METIS_LIBRARY}" DIRECTORY)
  set(_metis_hints "${_metis_libdir}/../include")
endif()
find_path(METIS_INCLUDE_DIR NAMES metis.h HINTS ${_metis_hints} PATH_SUFFIXES metis)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(METIS REQUIRED_VARS METIS_LIBRARY METIS_INCLUDE_DIR)

if(METIS_FOUND AND NOT TARGET METIS::metis)
  add_library(METIS::metis UNKNOWN IMPORTED)
  set_target_properties(METIS::metis PROPERTIES IMPORTED_LOCATION "${METIS_LIBRARY}"
                                                INTERFACE_INCLUDE_DIRECTORIES "${METIS_INCLUDE_DIR}")
endif()
mark_as_advanced(METIS_INCLUDE_DIR METIS_LIBRARY)
