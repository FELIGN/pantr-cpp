# install(): pantr's headers, the bundled third-party headers and licenses, and the CMake package.
#
# Two install components. `development` is pantr's own headers and its CMake package, which the
# Python wheel ships (pyproject.toml); `third_party` is the bundled dependency headers with their
# licenses, which only a CMake install carries. A plain `cmake --install` installs both, and
# pantrConfig.cmake tells the two apart by whether the third-party directory is there.
include_guard(GLOBAL)
include(CMakePackageConfigHelpers)

install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/pantr" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        COMPONENT development)

# Third-party headers, each with its license, under include/pantr/third_party/<name>/.
set(PANTR_INSTALL_THIRD_PARTY_DIR "${CMAKE_INSTALL_INCLUDEDIR}/pantr/third_party")
set(_pantr_tp "${PANTR_INSTALL_THIRD_PARTY_DIR}")
install(DIRECTORY "${pantr_eigen_SOURCE_DIR}/Eigen" DESTINATION "${_pantr_tp}/eigen"
        COMPONENT third_party)
install(FILES "${pantr_eigen_SOURCE_DIR}/COPYING.README" "${pantr_eigen_SOURCE_DIR}/COPYING.MPL2"
              "${pantr_eigen_SOURCE_DIR}/COPYING.APACHE" "${pantr_eigen_SOURCE_DIR}/COPYING.BSD"
              "${pantr_eigen_SOURCE_DIR}/COPYING.MINPACK" DESTINATION "${_pantr_tp}/eigen"
        COMPONENT third_party)
install(DIRECTORY "${pantr_spectra_SOURCE_DIR}/include/Spectra" DESTINATION "${_pantr_tp}/spectra"
        COMPONENT third_party)
install(FILES "${pantr_spectra_SOURCE_DIR}/LICENSE" DESTINATION "${_pantr_tp}/spectra"
        COMPONENT third_party)
# Boost.Multiprecision and Boost.Config share one boost/ tree and the Boost Software License.
install(DIRECTORY "${pantr_boost_multiprecision_SOURCE_DIR}/include/boost"
                  "${pantr_boost_config_SOURCE_DIR}/include/boost" DESTINATION "${_pantr_tp}/boost"
        COMPONENT third_party)
install(FILES "${pantr_boost_multiprecision_SOURCE_DIR}/LICENSE" DESTINATION "${_pantr_tp}/boost"
        COMPONENT third_party)
if(NOT PANTR_USE_STD_MDSPAN)
  install(DIRECTORY "${pantr_mdspan_SOURCE_DIR}/include/" DESTINATION "${_pantr_tp}/mdspan"
          COMPONENT third_party FILES_MATCHING PATTERN "*.hpp" PATTERN "mdspan" PATTERN "mdarray")
  install(FILES "${pantr_mdspan_SOURCE_DIR}/LICENSE" DESTINATION "${_pantr_tp}/mdspan"
          COMPONENT third_party)
endif()
# The notices list the bundled dependencies; the wheel carries pantr's license in its metadata.
install(FILES "${PROJECT_SOURCE_DIR}/THIRD_PARTY_NOTICES.md" "${PROJECT_SOURCE_DIR}/LICENSE"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/pantr" COMPONENT third_party)

# One export set per component, so a consumer asking only for mpi-capsule loads no dependency.
set_target_properties(pantr_deps PROPERTIES EXPORT_NAME third_party)
install(TARGETS pantr pantr_deps EXPORT pantrTargets)
install(EXPORT pantrTargets NAMESPACE pantr:: DESTINATION "${PANTR_INSTALL_CMAKEDIR}"
        COMPONENT development)
install(TARGETS pantr_mpi_capsule EXPORT pantrMpiCapsuleTargets)
install(EXPORT pantrMpiCapsuleTargets NAMESPACE pantr:: DESTINATION "${PANTR_INSTALL_CMAKEDIR}"
        COMPONENT development)
if(TARGET pantr_mpi)
  install(TARGETS pantr_mpi EXPORT pantrMpiTargets)
  install(EXPORT pantrMpiTargets NAMESPACE pantr:: DESTINATION "${PANTR_INSTALL_CMAKEDIR}"
          COMPONENT development)
endif()

if(TARGET pantr_metis)
  install(DIRECTORY "${PROJECT_SOURCE_DIR}/providers/metis/include/pantr"
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT development)
  install(TARGETS pantr_metis EXPORT pantrMetisTargets)
  install(EXPORT pantrMetisTargets NAMESPACE pantr:: DESTINATION "${PANTR_INSTALL_CMAKEDIR}"
          COMPONENT development)
  install(FILES "${PROJECT_SOURCE_DIR}/cmake/FindMETIS.cmake" DESTINATION "${PANTR_INSTALL_CMAKEDIR}"
          COMPONENT development)
endif()

configure_package_config_file("${PROJECT_SOURCE_DIR}/cmake/pantrConfig.cmake.in"
                              "${PROJECT_BINARY_DIR}/pantrConfig.cmake"
                              INSTALL_DESTINATION "${PANTR_INSTALL_CMAKEDIR}"
                              PATH_VARS PANTR_INSTALL_THIRD_PARTY_DIR)
# Header-only, so the package is independent of the consumer's pointer size.
write_basic_package_version_file("${PROJECT_BINARY_DIR}/pantrConfigVersion.cmake"
                                 COMPATIBILITY SameMinorVersion ARCH_INDEPENDENT)
install(FILES "${PROJECT_BINARY_DIR}/pantrConfig.cmake" "${PROJECT_BINARY_DIR}/pantrConfigVersion.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/PantrNativeArch.cmake"
        DESTINATION "${PANTR_INSTALL_CMAKEDIR}" COMPONENT development)
