# Every pantr build option, in one place. Reasons: design/port_design.md, section 1.3.
include_guard(GLOBAL)
include(GNUInstallDirs)

option(PANTR_FP_CONTRACT
       "Put -ffp-contract=on (/fp:contract) on pantr::pantr, inherited by consumers" ON)
# Consumed in two places: here for pantr's own tests, and in pantrConfig.cmake, where a consumer
# sets it before find_package(pantr). It is never frozen into the exported target.
option(PANTR_NATIVE_ARCH "Compile for the build machine's ISA (-march=native)" ON)
option(PANTR_FORCE_SCALAR "Define PANTR_SIMD_FORCE_SCALAR (the scalar SIMD fallback)" OFF)
option(PANTR_WARNINGS_AS_ERRORS "Add -Werror (/WX) to pantr's own warnings target" OFF)
option(PANTR_INSTALL "Install pantr's headers, CMake package and bundled dependency headers" ON)
option(PANTR_BUILD_TESTS "Build the C++ test suite" ${PROJECT_IS_TOP_LEVEL})
option(PANTR_WITH_BERF "Build the BeRF-FR root-finding provider (pantr::berf)" OFF)
option(PANTR_WITH_METIS "Build the METIS partitioning provider (pantr::metis)" OFF)

# The provider targets land in their own stages. An option that is ON but builds nothing would
# report a provider that is not there, so say so and stop.
foreach(_provider BERF METIS)
  if(PANTR_WITH_${_provider})
    message(FATAL_ERROR "pantr: PANTR_WITH_${_provider}=ON is not available yet; "
                        "the provider is not part of this version")
  endif()
endforeach()
