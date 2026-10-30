// The _pantr extension module, compiled once per ISA rung under the name PANTR_EXT_MODULE_NAME.
#include "interface.hpp"

#include "pantr/core/errors.hpp"
#include "pantr/core/simd.hpp"

#include <nanobind/nanobind.h>

#include <exception>

namespace nb = nanobind;

namespace pantr::bindings {

void register_errors(nb::module_& /*m*/) {
  // The table in pantr/core/errors.hpp. Most derived first: each class also derives from a
  // standard exception, which nanobind's default translation would map less precisely.
  nb::register_exception_translator([](const std::exception_ptr& p, void* /*payload*/) {
    try {
      std::rethrow_exception(p);
    } catch (const pantr::value_error& e) {
      PyErr_SetString(PyExc_ValueError, e.what());
    } catch (const pantr::type_error& e) {
      PyErr_SetString(PyExc_TypeError, e.what());
    } catch (const pantr::index_error& e) {
      PyErr_SetString(PyExc_IndexError, e.what());
    } catch (const pantr::state_error& e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    } catch (const pantr::not_implemented& e) {
      PyErr_SetString(PyExc_NotImplementedError, e.what());
    } catch (const pantr::capacity_error& e) {
      PyErr_SetString(PyExc_ValueError, e.what());
    }
  });
}

}  // namespace pantr::bindings

NB_MODULE(PANTR_EXT_MODULE_NAME, m) {
  m.doc() = "pantr's compiled core (private: use the pantr package).";
  m.attr("interface_version") = pantr::bindings::kInterfaceVersion;
  m.attr("isa") = PANTR_EXT_ISA;
  m.attr("simd_isa") = PANTR_SIMD_ISA_NAME;
  pantr::bindings::register_errors(m);
  pantr::bindings::register_parallel(m);
  nb::module_ quad = m.def_submodule("quad", "Quadrature rules and lattices (private).");
  pantr::bindings::register_quad(quad);
  nb::module_ tolerance = m.def_submodule("tolerance", "Tolerance preset factors (private).");
  pantr::bindings::register_tolerance(tolerance);
  nb::module_ basis = m.def_submodule("basis", "Basis tabulation (private).");
  pantr::bindings::register_basis(basis);
  nb::module_ change_basis = m.def_submodule("change_basis", "Change-of-basis matrices (private).");
  pantr::bindings::register_change_basis(change_basis);
  nb::module_ geometry = m.def_submodule("geometry", "Axis-aligned bounding boxes (private).");
  pantr::bindings::register_geometry(geometry);
  nb::module_ transform = m.def_submodule("transform", "Affine maps (private).");
  pantr::bindings::register_transform(transform);
  nb::module_ bezier = m.def_submodule("bezier", "Bezier values and operations (private).");
  pantr::bindings::register_bezier(bezier);
  pantr::bindings::register_bezier_approx(bezier);
  nb::module_ roots = m.def_submodule("roots", "The Builtin root provider (private).");
  pantr::bindings::register_roots(roots);
  nb::module_ bspline = m.def_submodule("bspline", "B-spline spaces and knot vectors (private).");
  pantr::bindings::register_bspline(bspline);
  nb::module_ kernels = m.def_submodule(
      "_kernels", "Ported kernels for Python callers not yet ported (private, interim).");
  pantr::bindings::register_bezier_kernels(kernels);
  pantr::bindings::register_bezier_approx_kernels(kernels);
  nb::module_ testing = m.def_submodule("_testing", "Probes for tests and benchmarks (private).");
  pantr::bindings::register_testing(testing);
  pantr::bindings::register_roots_testing(testing);
  pantr::bindings::register_bezier_approx_testing(testing);
  pantr::bindings::register_bspline_testing(testing);
}
