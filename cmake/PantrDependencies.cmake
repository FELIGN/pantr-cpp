# Third-party dependencies: fetched at a full commit SHA, never a branch or a tag (a tag can move).
#
# Every dependency of pantr::pantr is header-only. None is added with add_subdirectory: their own
# build systems would contribute targets, tests and install rules pantr does not want. The sources
# are populated only, and pantr_deps carries their include directories as SYSTEM includes, so
# pantr's -Werror never applies to third-party code. With nothing added as a subdirectory, the
# SYSTEM and EXCLUDE_FROM_ALL keywords below have no effect today; they keep each declaration
# correct should a dependency ever be built from its own CMakeLists. `cmake --install` copies their headers and
# license files under include/pantr/third_party/, which makes an installed pantr self-contained.
#
# Offline builds: -DFETCHCONTENT_SOURCE_DIR_<NAME>=<checkout> per dependency, plus
# -DFETCHCONTENT_FULLY_DISCONNECTED=ON. Licenses and versions: THIRD_PARTY_NOTICES.md, which must
# be updated together with any pin below.
include_guard(GLOBAL)
include(FetchContent)

# A SOURCE_SUBDIR that does not exist makes FetchContent_MakeAvailable populate the source without
# calling add_subdirectory on it.
set(_pantr_no_subdir "pantr-populate-only")

# Eigen 5.0.1 (MPL-2.0). Chosen over 3.4.1 because Spectra 1.2.0's symmetric eigensolver compiles
# and passes against it (cpp/tests/test_dependencies.cpp).
FetchContent_Declare(
  pantr_eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG bc3b39870ecb690a623a3f49149a358b95c5781d # 5.0.1
  SOURCE_SUBDIR ${_pantr_no_subdir}
  SYSTEM EXCLUDE_FROM_ALL)
# Spectra 1.2.0 (MPL-2.0).
FetchContent_Declare(
  pantr_spectra
  GIT_REPOSITORY https://github.com/yixuan/spectra.git
  GIT_TAG 6841bcbacaa0f0a8446210314e682057a084be4e # v1.2.0
  SOURCE_SUBDIR ${_pantr_no_subdir}
  SYSTEM EXCLUDE_FROM_ALL)
# Kokkos reference mdspan 0.6.0 (Apache-2.0 WITH LLVM-exception).
FetchContent_Declare(
  pantr_mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG 9ceface91483775a6c74d06ebf717bbb2768452f # mdspan-0.6.0
  SOURCE_SUBDIR ${_pantr_no_subdir}
  SYSTEM EXCLUDE_FROM_ALL)
# Boost.Multiprecision from Boost 1.90.0 (BSL-1.0), used in standalone mode (BOOST_MP_STANDALONE).
FetchContent_Declare(
  pantr_boost_multiprecision
  GIT_REPOSITORY https://github.com/boostorg/multiprecision.git
  GIT_TAG 529dfac199191a7eb8a5eb7f47256eff6d0db993 # boost-1.90.0
  SOURCE_SUBDIR ${_pantr_no_subdir}
  SYSTEM EXCLUDE_FROM_ALL)
# Boost.Config from Boost 1.90.0 (BSL-1.0): standalone Boost.Multiprecision refuses to compile
# without it (boost/multiprecision/detail/standalone_config.hpp), as its release package bundles it.
FetchContent_Declare(
  pantr_boost_config
  GIT_REPOSITORY https://github.com/boostorg/config.git
  GIT_TAG a7d5a9b05d70c9cfea980dc3539ca3d3461411b3 # boost-1.90.0
  SOURCE_SUBDIR ${_pantr_no_subdir}
  SYSTEM EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(pantr_eigen pantr_spectra pantr_mdspan pantr_boost_multiprecision
                           pantr_boost_config)

if(PANTR_BUILD_TESTS)
  # doctest 2.5.3 (MIT): the C++ test framework; never installed.
  FetchContent_Declare(
    pantr_doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG 2d0a9359a60c51affe2a9bebb1be1dca47868151 # v2.5.3
    SOURCE_SUBDIR ${_pantr_no_subdir}
    SYSTEM EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(pantr_doctest)
  add_library(pantr_doctest INTERFACE)
  target_include_directories(pantr_doctest SYSTEM INTERFACE "${pantr_doctest_SOURCE_DIR}")
endif()

# Each entry: <name>|<include root in the source tree>|<install subdirectory under third_party>.
set(PANTR_BUNDLED_DEPS
    "eigen|${pantr_eigen_SOURCE_DIR}|eigen"
    "spectra|${pantr_spectra_SOURCE_DIR}/include|spectra"
    "boost_multiprecision|${pantr_boost_multiprecision_SOURCE_DIR}/include|boost"
    "boost_config|${pantr_boost_config_SOURCE_DIR}/include|boost")
if(NOT PANTR_USE_STD_MDSPAN)
  list(APPEND PANTR_BUNDLED_DEPS "mdspan|${pantr_mdspan_SOURCE_DIR}/include|mdspan")
endif()

# The two Boost pieces share one install directory (their file sets are disjoint); the build tree
# uses both source roots.
add_library(pantr_deps INTERFACE)
set(_pantr_install_subdirs "")
foreach(_entry IN LISTS PANTR_BUNDLED_DEPS)
  string(REPLACE "|" ";" _entry "${_entry}")
  list(GET _entry 1 _src)
  list(GET _entry 2 _sub)
  target_include_directories(pantr_deps SYSTEM INTERFACE $<BUILD_INTERFACE:${_src}>)
  list(APPEND _pantr_install_subdirs "${_sub}")
endforeach()
list(REMOVE_DUPLICATES _pantr_install_subdirs)
foreach(_sub IN LISTS _pantr_install_subdirs)
  target_include_directories(
    pantr_deps SYSTEM INTERFACE
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/pantr/third_party/${_sub}>)
endforeach()
target_compile_definitions(pantr_deps INTERFACE BOOST_MP_STANDALONE)
if(NOT PANTR_USE_STD_MDSPAN)
  # The Kokkos mdspan picks operator() or a multi-index operator[] from the language version of
  # each translation unit. Fixing both macros keeps the class identical in C++20 and C++23 units.
  target_compile_definitions(pantr_deps INTERFACE MDSPAN_USE_PAREN_OPERATOR=1
                                                  MDSPAN_USE_BRACKET_OPERATOR=0)
endif()
