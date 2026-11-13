/* @file comm_capsule.h
 * @brief The C ABI through which one extension module hands a communicator to another.
 *
 * pantr's distributed algorithms are compiled in the `_pantr` extension, which links no MPI; the
 * MPI glue lives in `_pantr_mpi`, built once per MPI family. `_pantr_mpi` wraps a real communicator
 * in a `pantr_mpi_comm` table and passes it to `_pantr` in a Python capsule named
 * `PANTR_MPI_COMM_CAPSULE_NAME`, whose destructor calls `release`. Only this C struct crosses
 * between the two binaries, never a C++ type, so they need not agree on a standard library.
 *
 * Every function returns a status (`PANTR_MPI_OK` or an error code) and never throws or longjmps.
 * On an error it writes a NUL-terminated message of at most `msg_len` bytes (`msg_len` includes the
 * NUL) into `msg`. A gathered result is delivered through a caller-provided sink, called once
 * during the collective with every rank's block, so that neither side frees memory the other
 * allocated.
 *
 * Calling rules: a collective is called by every rank of the communicator, in the same order, from
 * the thread that initialized MPI (pantr requires `MPI_THREAD_FUNNELED`, design §2.8). A Python
 * extension calls the collectives with the GIL released, as mpi4py does in a blocking collective,
 * so other Python threads run while a rank waits; an implementation must therefore touch no Python
 * object in any function but `release`, which is called with the GIL held.
 *
 * Shapes cross as three `int64_t` per block, `(ndim, extent0, extent1)`, with `ndim` 1 or 2 and the
 * unused extent 0. Data are C-contiguous, row-major.
 *
 * Design note: design/mpi.md.
 */
#ifndef PANTR_MPI_COMM_CAPSULE_H
#define PANTR_MPI_COMM_CAPSULE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The layout version of `pantr_mpi_comm`; a table of another version is refused. */
#define PANTR_MPI_COMM_ABI_VERSION 1u

/* The name of the Python capsule holding a `pantr_mpi_comm*`. */
#define PANTR_MPI_COMM_CAPSULE_NAME "pantr.mpi.comm_v1"

/* Status codes. */
#define PANTR_MPI_OK 0
#define PANTR_MPI_ERR_VALUE 1   /* a payload of unexpected size or shape (Python ValueError) */
#define PANTR_MPI_ERR_RUNTIME 2 /* an MPI call failed, or the thread level is insufficient */

/* Element types of a payload. */
#define PANTR_MPI_INT64 0
#define PANTR_MPI_FLOAT64 1
#define PANTR_MPI_FLOAT32 2

/* Receives every rank's block of one gather, in rank order.
 *
 * `shapes` holds `3 * n_ranks` values (see the file comment); `data` the `n_elems` elements of all
 * blocks concatenated, of the type the gather was called with. The sink copies what it needs before
 * it returns; both pointers are valid only during the call. */
typedef struct pantr_mpi_sink {
  void* ctx;
  int (*receive)(void* ctx, int64_t n_ranks, const int64_t* shapes, const void* data,
                 int64_t n_elems, char* msg, size_t msg_len);
} pantr_mpi_sink;

/* A communicator. `ctx` is the implementation's own state, passed back to every function. */
typedef struct pantr_mpi_comm {
  uint32_t abi_version; /* PANTR_MPI_COMM_ABI_VERSION */
  uint32_t struct_size; /* sizeof(pantr_mpi_comm) */
  void* ctx;
  /* This process's rank, and the number of ranks. */
  int (*rank)(void* ctx, int32_t* out, char* msg, size_t msg_len);
  int (*size)(void* ctx, int32_t* out, char* msg, size_t msg_len);
  /* Gather every rank's block of `dtype` (PANTR_MPI_INT64 or PANTR_MPI_FLOAT64) into `sink`. */
  int (*allgatherv)(void* ctx, int dtype, const void* data, const int64_t shape[3],
                    pantr_mpi_sink* sink, char* msg, size_t msg_len);
  /* Gather the pair (int64 indices, float64 values) of every rank, as one collective of the
   * caller's: the indices into `index_sink`, the values into `value_sink`. */
  int (*allgather_owned)(void* ctx, const int64_t* index, const int64_t index_shape[3],
                         const double* value, const int64_t value_shape[3],
                         pantr_mpi_sink* index_sink, pantr_mpi_sink* value_sink, char* msg,
                         size_t msg_len);
  /* Elementwise sum of `count` elements of `dtype` (PANTR_MPI_FLOAT64 or PANTR_MPI_FLOAT32) over
   * the ranks, into `recv`. Every rank passes the same count, or every rank gets ERR_VALUE. */
  int (*allreduce_sum)(void* ctx, int dtype, const void* send, void* recv, int64_t count, char* msg,
                       size_t msg_len);
  /* Free `ctx`. Called once, when the last holder drops the table. */
  void (*release)(void* ctx);
} pantr_mpi_comm;

#ifdef __cplusplus
}
#endif

#endif /* PANTR_MPI_COMM_CAPSULE_H */
