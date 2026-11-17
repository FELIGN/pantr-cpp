// The thread settings of pantr's C++ pool, and the capsule other pantr modules attach to.
#include "interface.hpp"

#include "pantr/parallel/pool.hpp"
#include "pantr/parallel/shared.hpp"

#include <nanobind/nanobind.h>

#include <cstddef>
#include <cstdint>

namespace nb = nanobind;

namespace pantr::bindings {

void register_parallel(nb::module_& m) {
  m.def("max_threads", &pantr::parallel::max_threads,
        "Return the C++ pool capacity: the affinity CPU count, lowered by PANTR_NUM_THREADS.");
  m.def("get_num_threads", &pantr::parallel::get_num_threads,
        "Return the calling thread's parallelism setting for the C++ pool.");
  m.def("set_num_threads", &pantr::parallel::set_num_threads, nb::arg("n"),
        "Set the calling thread's parallelism for the C++ pool (1 <= n <= max_threads()).");
  // _pantr owns the process's pool and per-thread settings; another pantr module built from the
  // same headers attaches this table at import (pantr/parallel/shared.hpp).
  m.attr("_shared_parallelism") = nb::capsule(&pantr::parallel::owner_shared_parallelism(),
                                              pantr::parallel::kSharedParallelismCapsule);
}

}  // namespace pantr::bindings
