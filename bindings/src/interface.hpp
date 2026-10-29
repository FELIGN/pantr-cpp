// What every translation unit of the _pantr extension shares: the interface version and the
// registration functions of each part of the module.
#pragma once

#include <nanobind/nanobind.h>

namespace pantr::bindings {

/// Version of the module's interface. pantr/_ext.py refuses a module whose version differs from
/// its EXPECTED_INTERFACE, so a stale build fails at import instead of as a phantom regression.
/// Bump both together with any change to what the module exposes.
inline constexpr int kInterfaceVersion = 1;

/// Register the translation of pantr's C++ exceptions to Python exceptions.
void register_errors(nanobind::module_& m);

/// Register the thread settings over the C++ pool and the shared-parallelism capsule.
void register_parallel(nanobind::module_& m);

/// Register the `_testing` submodule (probes for tests and benchmarks; not a public API).
void register_testing(nanobind::module_& m);

}  // namespace pantr::bindings
