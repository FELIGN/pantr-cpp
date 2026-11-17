// A test-only second extension module built from pantr's headers, as _pantr_mpi and _pantr_berf
// will be. Built twice: _pantr_twin attaches to _pantr's pool (PANTR_TWIN_ROUTED=1) and
// _pantr_twin_unrouted keeps its own, the control showing what the routing prevents.
#include "attach_shared.hpp"
#include "probes.hpp"

#include "pantr/parallel/pool.hpp"
#include "pantr/parallel/shared.hpp"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(PANTR_TWIN_MODULE_NAME, m) {
  m.doc() = "Test-only second pantr extension (never shipped).";
#if PANTR_TWIN_ROUTED
  pantr::bindings::attach_shared_parallelism_from_pantr();
#endif
  m.def("routes_shared_parallelism", &pantr::parallel::routes_shared_parallelism);
  m.def("max_threads", &pantr::parallel::max_threads);
  m.def("get_num_threads", &pantr::parallel::get_num_threads);
  m.def("set_num_threads", &pantr::parallel::set_num_threads, nb::arg("n"));
  m.def("count_threads_used", &pantr::bindings::count_threads_used, nb::arg("chunks"),
        nb::arg("hold_us"), nb::call_guard<nb::gil_scoped_release>());
}
