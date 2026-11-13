/// @file version.hpp
/// @brief Library version of the pantr C++ headers.
///
/// The root CMakeLists.txt reads the three numbers below to set the CMake project version, and
/// fails to configure when they disagree with `__version__` in `src/pantr/__init__.py`, so the C++
/// headers and the Python package cannot drift apart.
#pragma once

#define PANTR_VERSION_MAJOR 0
#define PANTR_VERSION_MINOR 7
#define PANTR_VERSION_PATCH 0

/// Version as a single integer, `major * 10000 + minor * 100 + patch`, for preprocessor tests.
#define PANTR_VERSION \
  (PANTR_VERSION_MAJOR * 10000 + PANTR_VERSION_MINOR * 100 + PANTR_VERSION_PATCH)

namespace pantr {

/// Library version triple, usable in constant expressions.
struct Version {
  int major;  ///< Major version.
  int minor;  ///< Minor version.
  int patch;  ///< Patch version.
};

/// The version of the headers this translation unit was compiled against.
inline constexpr Version version{PANTR_VERSION_MAJOR, PANTR_VERSION_MINOR, PANTR_VERSION_PATCH};

}  // namespace pantr
