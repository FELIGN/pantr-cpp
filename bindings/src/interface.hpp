// What every translation unit of the _pantr extension shares: the interface version and the
// registration functions of each part of the module.
#pragma once

#include <nanobind/nanobind.h>

namespace pantr::bindings {

/// Version of the module's interface. pantr/_ext.py refuses a module whose version differs from
/// its EXPECTED_INTERFACE, so a stale build fails at import instead of as a phantom regression.
/// Bump both together with any change to what the module exposes.
inline constexpr int kInterfaceVersion = 5;

/// Register the translation of pantr's C++ exceptions to Python exceptions.
void register_errors(nanobind::module_& m);

/// Register the thread settings over the C++ pool and the shared-parallelism capsule.
void register_parallel(nanobind::module_& m);

/// Register the `quad` submodule (1D rules, PointsLattice, QuadratureRule).
void register_quad(nanobind::module_& m);

/// Register the `tolerance` submodule (the preset factors).
void register_tolerance(nanobind::module_& m);

/// Register the `basis` submodule (1D and tensor-product tabulation).
void register_basis(nanobind::module_& m);

/// Register the `change_basis` submodule (builders and process tables).
void register_change_basis(nanobind::module_& m);

/// Register the `geometry` submodule (AABB).
void register_geometry(nanobind::module_& m);

/// Register the `transform` submodule (AffineTransform).
void register_transform(nanobind::module_& m);

/// Register the `bezier` submodule (the Bezier value type and its operations).
void register_bezier(nanobind::module_& m);

/// Register the Bezier kernels of the `_kernels` submodule (for Python callers not yet ported).
void register_bezier_kernels(nanobind::module_& m);

/// Register degree reduction and fitting in the `bezier` submodule.
void register_bezier_approx(nanobind::module_& m);

/// Register the reduction operator in the `_kernels` submodule (for the B-spline reduction).
void register_bezier_approx_kernels(nanobind::module_& m);

/// Register the reduction and fitting internals in the `_testing` submodule.
void register_bezier_approx_testing(nanobind::module_& m);

/// Register the `roots` submodule (the Builtin root provider).
void register_roots(nanobind::module_& m);

/// Register the root provider's internals in the `_testing` submodule.
void register_roots_testing(nanobind::module_& m);

/// Register the `_kernels` submodule: the interim surface through which Python code not yet ported
/// reaches ported kernels (removed with the last such caller).
void register_kernels(nanobind::module_& m);

/// Register the `_testing` submodule (probes for tests and benchmarks; not a public API).
void register_testing(nanobind::module_& m);

}  // namespace pantr::bindings
