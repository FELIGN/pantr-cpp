// The _pantr_mpi extension module (package pantr_mpi, distribution pantr-mpi): the MPI glue and
// nothing else (design §1.7). It turns an mpi4py communicator into a `comm_capsule.h` table over
// MPI (`pantr::mpi::make_mpi_table`), handed to `_pantr` in a capsule, and reports the MPI it was
// built against. No numerics, hence no ISA rungs and no pantr core header: it needs only
// comm_capsule.h and mpi_table.hpp, the `mpi-capsule` component.
#include "pantr/mpi/comm_capsule.h"
#include "pantr/mpi/mpi_table.hpp"

#include <mpi.h>

// mpi4py's C API under the stable ABI goes through capsules (mpi4py >= 4.0, pycapi.h).
#if defined(Py_LIMITED_API) && !defined(MPI4PY_LIMITED_API)
#define MPI4PY_LIMITED_API 1
#endif
#include <mpi4py/mpi4py.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace {

// The MPI implementation the module was compiled against, from its mpi.h.
#if defined(OMPI_MAJOR_VERSION)
constexpr const char* kCompiledImplementation = "openmpi";
#elif defined(I_MPI_VERSION)
constexpr const char* kCompiledImplementation = "intel";
#elif defined(MVAPICH2_VERSION) || defined(MVAPICH_VERSION)
constexpr const char* kCompiledImplementation = "mvapich";
#elif defined(MPICH_VERSION)
constexpr const char* kCompiledImplementation = "mpich";
#else
constexpr const char* kCompiledImplementation = "unknown";
#endif

void drop_reference(void* obj) noexcept { Py_DecRef(static_cast<PyObject*>(obj)); }

void release_table(void* table) noexcept {
  pantr::mpi::release_mpi_table(static_cast<pantr_mpi_comm*>(table));
}

}  // namespace

NB_MODULE(_pantr_mpi, m) {
  m.doc() = "pantr's MPI glue (private: use pantr.mpi).";
  if (import_mpi4py() < 0) {
    throw nb::python_error();
  }
  m.attr("compiled_implementation") = kCompiledImplementation;
  m.attr("comm_abi_version") = PANTR_MPI_COMM_ABI_VERSION;
  m.attr("comm_capsule_name") = PANTR_MPI_COMM_CAPSULE_NAME;

  m.def(
      "comm_capsule",
      [](nb::handle comm) {
        MPI_Comm* handle = PyMPIComm_Get(comm.ptr());
        if (handle == nullptr) {
          throw nb::python_error();
        }
        std::array<char, 512> msg{};
        // The table keeps the mpi4py object alive (its handle lives inside it) until released.
        Py_IncRef(comm.ptr());
        pantr_mpi_comm* table = pantr::mpi::make_mpi_table(*handle, msg.data(), msg.size(),
                                                           comm.ptr(), &drop_reference);
        if (table == nullptr) {
          Py_DecRef(comm.ptr());
          throw std::runtime_error(msg.data());
        }
        return nb::capsule(table, PANTR_MPI_COMM_CAPSULE_NAME, &release_table);
      },
      nb::arg("comm"),
      "A capsule holding the communicator table over an mpi4py communicator (keeps it alive).");

  m.def(
      "query_thread",
      []() {
        int provided = 0;
        if (MPI_Query_thread(&provided) != MPI_SUCCESS) {
          throw std::runtime_error("MPI_Query_thread failed");
        }
        return provided;
      },
      "The thread level MPI was initialized with.");

  m.def(
      "library_version",
      []() {
        std::array<char, MPI_MAX_LIBRARY_VERSION_STRING> text{};
        int len = 0;
        if (MPI_Get_library_version(text.data(), &len) != MPI_SUCCESS) {
          throw std::runtime_error("MPI_Get_library_version failed");
        }
        return std::string(text.data(), static_cast<std::size_t>(len));
      },
      "The MPI library's version string, as the loaded library reports it.");

  m.attr("THREAD_FUNNELED") = static_cast<int>(MPI_THREAD_FUNNELED);
}
