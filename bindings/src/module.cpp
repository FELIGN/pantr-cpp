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
  nb::module_ testing = m.def_submodule("_testing", "Probes for tests and benchmarks (private).");
  pantr::bindings::register_testing(testing);
}
