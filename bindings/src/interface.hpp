// What every translation unit of the _pantr extension shares: the interface version and the
// registration functions of each part of the module.
#pragma once

#include <nanobind/nanobind.h>

namespace pantr::bindings {

/// Version of the module's interface. pantr/_ext.py refuses a module whose version differs from
/// its EXPECTED_INTERFACE, so a stale build fails at import instead of as a phantom regression.
/// Bump both together with any change to what the module exposes.
inline constexpr int kInterfaceVersion = 9;

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

/// Register the `bspline` submodule (BsplineSpace1D, BsplineSpace, the knot factories).
void register_bspline(nanobind::module_& m);
/// Register the knot scale, tolerance and snapping check in the `_testing` submodule.
void register_bspline_testing(nanobind::module_& m);

/// Register the Bspline value type and its evaluation in the `bspline` submodule.
void register_bspline_ops(nanobind::module_& m);

/// Register the refinement methods of the Bspline classes (knot insertion, conversion to open and
/// periodic form, subdivision, restriction, splitting) and the knot-vector functions of the
/// `bspline` submodule. After `register_bspline_ops`.
void register_bspline_refine(nanobind::module_& m);

/// Register degree elevation and reduction, knot removal and the derivative of the Bspline
/// classes, and the common refinement of two B-splines, in the `bspline` submodule. After
/// `register_bspline_ops`.
void register_bspline_modify(nanobind::module_& m);

/// Register the 1D refinement kernels in the `_kernels` submodule (for Python callers not yet
/// ported).
void register_bspline_kernels(nanobind::module_& m);

/// Register the knot-removal tolerances, the binomial kernel and the 1D degree kernels in the
/// `_testing` submodule.
void register_bspline_modify_testing(nanobind::module_& m);

/// Register the blossom in the `_testing` submodule.
void register_bspline_blossom_testing(nanobind::module_& m);

/// Register the `_testing` submodule (probes for tests and benchmarks; not a public API).
void register_testing(nanobind::module_& m);

}  // namespace pantr::bindings
