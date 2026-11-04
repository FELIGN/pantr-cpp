# Parallelism

PaNTr's performance-critical kernels are compiled C++ that distributes work across CPU cores on
PaNTr's own thread pool. This page explains which operations run in parallel, how to control the
thread count, and how to avoid over-subscription when composing PaNTr with other parallel
libraries.

## What runs in parallel

The following operations are parallelized over **evaluation points**:

| Operation | Kernel | Parallelism dimension |
|---|---|---|
| B-spline basis evaluation and derivatives (`BsplineSpace1D`, `BsplineSpace`) | C++ (`include/pantr/bspline/`), on pantr's own thread pool | points |
| `Bspline` evaluation and derivatives (points and lattices) | C++ (`include/pantr/bspline/evaluate.hpp`), on pantr's own thread pool | points; for a lattice, tiles of each direction's contraction |
| Bernstein, cardinal B-spline, Legendre and Lagrange basis evaluation (1D and tensor product) | C++ (`include/pantr/basis/`), on pantr's own thread pool | points |
| Bezier evaluation and derivatives | C++ (`include/pantr/bezier/`), on pantr's own thread pool | points |
| Bezier split / slice / restrict | C++ (`include/pantr/bezier/`), on pantr's own thread pool | control point columns |
| Bezier extraction (`to_beziers`) | C++ (`include/pantr/bspline/to_beziers.hpp`), on pantr's own thread pool | element/slab pairs |
| Batch applies of `SpanwiseElementExtraction` (`apply_many`, `apply_transpose_many`, `apply_MT_K_M_many`, `apply_M_K_MT_many`) | C++ (`include/pantr/bspline/spanwise_extraction.hpp`), on pantr's own thread pool, GIL released when the batch is large enough to run in parallel | cells |

By default, every CPU the process may run on is used: the pool sizes itself from the process's
CPU affinity mask (on Linux), not from the machine's core count, so a process restricted with
`taskset`, a cpuset or a batch scheduler's binding uses only the CPUs it was given. A CPU quota
(Docker's `--cpus`, a Kubernetes limit) does not change the mask: set `PANTR_NUM_THREADS` there.

## Controlling the thread count

PaNTr exposes three functions for thread-pool control:

```python
import pantr

# Query the current thread count
n = pantr.get_num_threads()

# Set it for the calling thread
pantr.set_num_threads(4)

# Temporarily override with a context manager
with pantr.num_threads(2):
    result = space.tabulate_basis(pts)  # uses 2 threads
# reverts to previous value here
```

Setting the thread count to **1** serializes all parallel kernels called from that thread.

The setting is **per thread**: `pantr.set_num_threads` changes the parallelism of kernels called
from the calling thread only, and every other thread keeps its own, the maximum by default. The
maximum, which `set_num_threads` cannot exceed, is the pool's capacity: the CPUs of the affinity
mask, lowered by the `PANTR_NUM_THREADS` environment variable when it is set. Both are read once,
on the first call that needs the thread count.

### BLAS / LAPACK coordination

PaNTr's dense solves are compiled into the extension and start no thread of their own, though a
few calls still reach NumPy's BLAS (for example `THBSplineSpace.restriction_to`, through
`numpy.linalg.pinv`). Those, and NumPy and SciPy operations in your code, use their own
thread pools (OpenBLAS, MKL, or Apple Accelerate). When PaNTr's threads *and* BLAS threads are
both active, the total number of OS threads can exceed the number of cores, causing
**over-subscription** and degraded performance. To limit BLAS threads around a block, use
[threadpoolctl](https://github.com/joblib/threadpoolctl) directly (it is not a PaNTr dependency):

```python
from threadpoolctl import threadpool_limits

with pantr.num_threads(4), threadpool_limits(limits=4):
    # PaNTr uses 4 threads; BLAS also limited to 4
    result = bspline.evaluate(pts)
```

## Usage patterns

### Pattern 1: PaNTr owns all parallelism (default)

The simplest approach.  Every PaNTr call distributes work across all cores
internally.  User code stays serial:

```python
# PaNTr parallelizes over 100k points automatically
result = bspline.evaluate(pts)
```

Best when each individual call has a large workload (many evaluation points or
many elements).

### Pattern 2: User owns outer parallelism, PaNTr runs serially

When you have many independent objects (curves, surfaces, patches) to process,
it is often better to parallelize *across* objects yourself and let each PaNTr
call run serially.  This avoids nested thread-pool contention:

```python
import concurrent.futures
import pantr


def evaluate_serially(curve, pts):
    pantr.set_num_threads(1)  # the setting is per thread: set it in the worker
    return curve.evaluate(pts)


with concurrent.futures.ThreadPoolExecutor(8) as pool:
    futures = [pool.submit(evaluate_serially, curve, pts) for curve in curves]
    results = [f.result() for f in futures]
```

This works for the operations that **release the GIL** while they compute, which most parallel
entry points do once the call is large enough to use the pool (whatever the thread setting): the
basis tabulations, B-spline and Bézier evaluation, the spanwise extraction's batch applies, point
location and the grids' batch queries. Several Python threads then run PaNTr kernels
concurrently. A call too small to use the pool keeps the GIL, since its work is shorter than the
cost of releasing it, and so do a few parallel operations (Bézier split, slice and restrict among
them), which serialize the calling threads.

### Pattern 3: Balanced -- user and PaNTr share cores

Split the available cores between PaNTr's inner parallelism and your own outer
parallelism:

```python
def evaluate_on_two(curve, pts):
    pantr.set_num_threads(2)  # PaNTr uses 2 threads per call from this worker
    return curve.evaluate(pts)


with concurrent.futures.ThreadPoolExecutor(4) as pool:
    futures = [pool.submit(evaluate_on_two, curve, pts) for curve in curves]
    results = [f.result() for f in futures]
# 4 user threads x 2 PaNTr threads = 8 total on an 8-core machine
```

Concurrent calls from several threads share the pool's workers, and each call also computes on
the thread that made it, so a call makes progress even when every worker is busy. A parallel
loop started from inside another one runs serially on its thread, so nesting never
over-subscribes the pool.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `PANTR_NUM_THREADS` | unset (the CPUs of the affinity mask) | Lowers the pool's capacity, the most threads `pantr.set_num_threads` accepts; never raises it above the CPUs the process may use. Set to a non-empty value, it also disables the default one-thread-per-rank policy of `pantr.mpi` (see {func}`pantr.mpi.configure_threads`). A value other than a positive integer makes the first parallel call raise `ValueError`. |

`NUMBA_NUM_THREADS`, `NUMBA_THREADING_LAYER` and `NUMBA_DISABLE_JIT` are not read: PaNTr no
longer uses Numba.
