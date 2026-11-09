# Threading model

Decision record for pantr's C++ parallelism (`include/pantr/parallel/`). Expensive to reverse
because every parallel kernel is written against it and its determinism property is what lets
the Python suite stay the oracle across thread counts. Architecture: `design/port_design.md` §2.8.

## The decision

- **One process pool**, created on first use, with `max_threads() - 1` worker threads.
  `max_threads()` is the CPU count of the process's affinity mask (`sched_getaffinity` on Linux,
  `hw.activecpu` on macOS, `GetProcessAffinityMask` on Windows), lowered by `PANTR_NUM_THREADS` when
  set, read once per process. Never `std::thread::hardware_concurrency()`, which ignores a restricted
  mask. No OpenMP, no TBB (R0).
- **The caller participates.** `parallel_for` posts a job and then claims chunks itself, so a call
  makes progress without any free worker; concurrent calls from many threads share the workers.
  This is why the pool has one worker fewer than the capacity.
- **Nested loops run inline.** A `parallel_for` started from inside a chunk (on a worker or on a
  participating caller) runs its chunks serially on that thread: no oversubscription, no deadlock.
- **Parallelism is a per-thread setting** (`set_num_threads`, thread-local, default the capacity),
  as Numba's is; setting it in one thread does not change another's. A call uses at most that many
  threads, the caller included.
- **Chunking depends on the problem only.** Chunks are `[k * grain, min(n, (k + 1) * grain))`. The
  grain comes from a per-item work estimate (`WorkEstimate`, elementary operations derived from the
  algorithm), `grain = max(1, ceil(chunk_ops / ops_per_item))`, raised for a loop that states the
  bytes each item writes to `ceil(chunk_bytes / bytes_per_item)` so a cheap, memory-bound item still
  gets a chunk that streams enough to amortize its claim, and a loop below `parallel_min_ops` total
  operations runs serially: `parallel_for` runs the same chunks in order, and a kernel may instead
  run the whole range as one chunk (`basis/detail/tabulate_1d.hpp` does). Either is exact because
  a chunk body gives each item the same result whatever chunk holds it. No parallel loop reduces
  across chunks, so a result does not depend on the thread count. A future reduction must fix its
  chunk order.
- **The constants are admitted heuristics**, reasoned from orders of magnitude (an atomic claim on a
  contended cache line against the work of a chunk; a futex wake-and-join against the work of a
  job; an L2-sized stream against a chunk's start-up) and stated at their definition in `work_model.hpp`. They are never fitted on a particular
  machine, where a threshold measured on many cores would be wrong on a laptop.
- **Exceptions**: the first exception a chunk throws is captured, the chunks not yet started are
  skipped, and it is rethrown on the caller after the join.
- **The pool is never destroyed.** Its workers stay blocked until the process ends. Destroying it
  at exit would race every static destructor in any translation unit that still runs a parallel
  loop, since their order relative to the pool's owner is unspecified.
- **Fork safety**: the pool records the process id that created it. A forked child that calls a
  parallel loop sees a different id, leaves the inherited pool untouched (its threads do not exist
  in the child, and its mutexes may have been held at the fork) and builds its own. The mutex that
  guards pool creation is held across `fork` with `pthread_atfork`, registered by dynamic
  initialization (a failed registration is reported when the pool is first needed), so a child never inherits it locked by a thread it does not have. The pool's holder and
  the cached capacity are constant-initialized rather than function-local statics: a static's
  initialization guard, inherited mid-initialization, would block the child forever.
- **Fork safety of lazy blocks** (`core/lazy.hpp`): each `Lazy` block has its own mutex, held by
  the thread building the block, so a child forked during that build would inherit it locked by a
  thread it lacks and block forever on first use. A build registers its mutex before locking it and
  unregisters after unlocking, in a registry whose own mutex is held across `fork`; a
  `pthread_atfork` child handler, which runs before the child has a second thread, replaces every
  registered mutex with a fresh one and empties the registry, so the child builds the block itself.
  The old mutex is neither unlocked nor destroyed, since its owner does not exist. A compute
  function must therefore not fork. Each extension module has its own registry, and a block built
  from two modules at once is registered in both, so every handler that runs resets it. Windows has
  no `fork`, and the registry is compiled out there.

## Consequences

- **One pool per process, by ownership.** The pool, its holder, the capacity cache and the
  per-thread settings are inline variables of header-only code, so shared objects that each include
  the headers and keep their symbols local (every Python extension module: hidden visibility, and
  separate DLLs on Windows) would each get their own pool and setting. pantr's `_pantr` owns them
  and exports a table of functions over them (`pantr/parallel/shared.hpp`) in a capsule; another
  pantr module that runs parallel loops attaches that table when imported, after which its
  `parallel_for`, `max_threads` and thread settings act on the owner's pool, and its own
  `Pool::instance()` refuses. Today only the test module `_pantr_twin` attaches: `_pantr_mpi` runs
  no parallel loop and includes no pool header, and `_pantr_berf` does not exist yet. The table's functions are `noexcept` and report failure by status and
  message, since an exception thrown in one module may not match a handler in another, and its
  layout carries a version so modules from different headers refuse to share. A program that never
  attaches, every plain C++ consumer, is unchanged. Rejected: GNU unique symbols (ELF and GCC only)
  and per-module pools (oversubscription past the affinity mask, and a thread count set through
  one module that another never sees). Tested with a test-only second extension
  (`tests/test_shared_parallelism.py`) and a C++ test against a recording fake owner.
- The Numba warmup barrier and the serial-twin kernels have no C++ counterpart: concurrent callers
  are safe by construction, and the serial path is the parallel path's chunks run in order.
- Thread-count invariance is tested on integers (chunk boundaries, exactly) and on values within a
  derived bound, never as bit-identity of floating-point results.
- TSan's runtime does not support a child that starts threads after a multi-threaded fork, so the
  forked-child test runs in every build except the TSan one.
- MPI: collectives are called only by the calling thread, outside any `parallel_for`, which is why
  `MPI_THREAD_FUNNELED` suffices (§2.8).

## Alternatives not taken

- A global (process-wide) thread count: one Python thread's setting would change another's, unlike
  Numba and unlike what `pantr.mpi`'s per-rank policy documents.
- Chunking by thread count (`n / threads` chunks): results of any future reduction would move with
  the thread count, and load balance would depend on the slowest thread.
- `std::async` or a thread per call: thread start-up per call, and no bound on the threads alive.
