# The communicator concept, the capsule ABI and the duck-typed path

Decision record for how pantr's distributed algorithms reach a communicator. Expensive to reverse
because every distributed algorithm is written against the concept, the `pantr-mpi` distribution
is built against the ABI, and the Python suite's fake communicators depend on the payloads.
Architecture: `design/port_design.md` §5.

## The decision

- **The algorithms are MPI-free templates** in `include/pantr/mpi/` over a `Communicator` concept
  (`communicator.hpp`) with one member per payload the algorithms send: `allgatherv` of a
  `Block<double>` or `Block<int64_t>` (data with one or two extents), `allgather_owned` of the
  pair of owned DOF indices and coefficients, and `allreduce_sum` of a `float` or `double` tensor.
  A gathered block keeps the shape the sender gave it. Each payload is one Python call of the
  duck-typed communicator, so the Python fakes keep the presets they had.
- **Two headers include `<mpi.h>`**: `mpi_table.hpp`, the MPI collectives written as the
  `comm_capsule.h` C table over an `MPI_Comm` (standard library and `<mpi.h>` only, so `pantr-mpi`
  builds against the `mpi-capsule` component), and `mpi_communicator.hpp`, `MpiCommunicator`, that
  table held as a communicator for C++ programs. No other header includes `mpi.h` (the header
  check compiles every other one against a poisoned `mpi.h`).
- **`AnyCommunicator`** erases any communicator behind a virtual interface, and also holds a C
  table from another module (`from_table`). The Python extension instantiates every algorithm
  once, over it.
- **`comm_capsule.h` is the only thing that crosses between `_pantr` and `_pantr_mpi`**: a
  versioned struct of `noexcept` functions returning status codes, a gathered result delivered
  through a sink the caller provides, so neither side frees the other's memory. `_pantr_mpi`
  wraps an mpi4py communicator (`PyMPIComm_Get`, holding a reference to the mpi4py object until
  the table is released) and passes the table in a capsule named `pantr.mpi.comm_v1`.
- **The duck-typed path stays a supported path** (`PyCommunicator`, in `_pantr`): any object with
  `rank` and `size` attributes, read only when needed, and `allgather` and `allreduce`. Its
  exceptions keep their Python type, since it is a C++ class of the same module rather than a C
  table.
- **The partition is computed on rank 0 and broadcast** by `create_distributed_space` (and so by
  `create_distributed_function`) whenever there is more than one rank: rank 0 sends
  `[1, n_parts, owners...]` through `allgatherv`, the others send nothing and do not compute it.
  If rank 0's computation raises, it sends `[0]`, re-raises, and every other rank raises
  `ValueError`, so a failure on rank 0 alone does not leave the others waiting.
- **Threads and MPI**: collectives run only on the calling thread, outside any `parallel_for`, so
  `MPI_THREAD_FUNNELED` suffices; `make_mpi_table` refuses a lower level, and under `FUNNELED` a
  collective from another thread than MPI's main thread is refused. The Python bindings hold the
  GIL, except inside a real-MPI collective (a capsule's table), which they call with it released,
  as mpi4py releases it in its own blocking collectives.
- **The MPI family check** runs before `_pantr_mpi` is first imported: the family recorded at
  build time in `pantr_mpi/_build_info.py` (from the library version FindMPI reports, or set with
  `PANTR_MPI_FAMILY`; `custom` disables the check) is compared, at the family level dolfinx's
  wheel check uses, with `mpi4py.MPI.Get_library_version()`. It runs on first use of an mpi4py
  communicator rather than at `import pantr.mpi`, which must not import mpi4py.

## Why

- A concept with one member per payload, rather than raw `allgatherv` of flat buffers, keeps
  each Python-visible payload one call with the format the fakes preset; flat buffers would have
  turned the QI tuple into two calls and lost the `(m, r)` shape.
- The partition broadcast: a partition derived from floating-point work (spectral bisection,
  weighted recursive bisection) can differ between machines that round differently (another ISA
  rung, another compiler), and every rank must hold the same one. Computing it on rank 0 only
  matches the ruling and spends no work on the other ranks.
- The failure marker costs one entry and removes a deadlock that a broadcast would otherwise add
  (an argument error raised by rank 0's partitioner alone).

## Alternatives not taken

- Orchestration in Python with only the local compute in C++: leaves a C++ user without a
  distributed API.
- Algorithms in `_pantr_mpi`: one build per MPI family and none per ISA, and C++ objects passed
  across extension modules.
- Every rank computing the partition and checking agreement: detects the disagreement instead of
  removing it.
