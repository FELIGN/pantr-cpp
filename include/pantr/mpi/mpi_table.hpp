/// @file mpi_table.hpp
/// @brief The `comm_capsule.h` table over an `MPI_Comm`: pantr's MPI collectives, written against
/// `<mpi.h>`, `comm_capsule.h` and the standard library only, so that the `pantr-mpi` extension can
/// be built against a pantr installation that carries no third-party header (the `mpi-capsule`
/// component). One of the two pantr headers that include `<mpi.h>`; `mpi_communicator.hpp` is the
/// other.
///
/// pantr calls MPI only from the thread that calls a distributed algorithm, outside any
/// `parallel_for`, so it needs `MPI_THREAD_FUNNELED` and no more (design §2.8). `make_mpi_table`
/// refuses an MPI initialized below that level, and each collective refuses a call from another
/// thread than the one that initialized MPI unless MPI was initialized at `MPI_THREAD_SERIALIZED`
/// or above. The Python bindings call a collective with the GIL released, so two Python threads can
/// be inside MPI at once: concurrent distributed calls from several threads need
/// `MPI_THREAD_MULTIPLE`, since `MPI_THREAD_SERIALIZED` only promises calls that do not overlap.
///
/// An MPI error code becomes `PANTR_MPI_ERR_RUNTIME` with `MPI_Error_string`'s text (under the
/// default error handler, `MPI_ERRORS_ARE_FATAL`, MPI aborts first). An `allreduce_sum` over
/// tensors of different sizes on different ranks is detected by a reduction of the sizes first, and
/// every rank returns `PANTR_MPI_ERR_VALUE` instead of entering an erroneous collective.
/// Design note: design/mpi.md.
#pragma once

#include "pantr/mpi/comm_capsule.h"

#include <mpi.h>

#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace pantr::mpi {

namespace detail {

/// The context of an MPI-backed table.
///
/// Attributes:
///     comm (MPI_Comm): The communicator (owned by the caller, not freed here).
///     rank (int): This process's rank, read when the table was made.
///     size (int): The number of ranks, read when the table was made.
///     funneled (bool): Whether only the main thread may call MPI.
///     keep (void*): What `drop` releases with the table (a reference keeping `comm` alive).
///     drop (void (*)(void*)): Called once with `keep` on release; may be null.
struct MpiTableCtx {
  MPI_Comm comm;
  int rank;
  int size;
  bool funneled;
  void* keep;
  void (*drop)(void*);
};

/// Write `text` into a table call's message buffer.
inline void put_message(char* msg, std::size_t len, const std::string& text) noexcept {
  if (msg != nullptr && len > 0) {
    std::snprintf(msg, len, "%s", text.c_str());
  }
}

/// `PANTR_MPI_OK` for `MPI_SUCCESS`, else `PANTR_MPI_ERR_RUNTIME` with the MPI error text.
inline int mpi_status(int code, const char* what, char* msg, std::size_t len) noexcept {
  if (code == MPI_SUCCESS) {
    return PANTR_MPI_OK;
  }
  std::array<char, MPI_MAX_ERROR_STRING> text{};
  int n = 0;
  if (MPI_Error_string(code, text.data(), &n) != MPI_SUCCESS) {
    n = 0;
  }
  put_message(
      msg, len,
      std::string(what) + " failed: " + std::string(text.data(), static_cast<std::size_t>(n)));
  return PANTR_MPI_ERR_RUNTIME;
}

inline MpiTableCtx& ctx_of(void* ctx) noexcept { return *static_cast<MpiTableCtx*>(ctx); }

/// `PANTR_MPI_OK`, or `ERR_RUNTIME` when only the main thread may call MPI and this is another.
inline int check_thread(const MpiTableCtx& c, char* msg, std::size_t len) noexcept {
  if (!c.funneled) {
    return PANTR_MPI_OK;
  }
  int is_main = 0;
  const int st = mpi_status(MPI_Is_thread_main(&is_main), "MPI_Is_thread_main", msg, len);
  if (st != PANTR_MPI_OK) {
    return st;
  }
  if (is_main == 0) {
    put_message(msg, len,
                "pantr::mpi: MPI was initialized at MPI_THREAD_FUNNELED, so only the thread that "
                "initialized it may call a distributed algorithm.");
    return PANTR_MPI_ERR_RUNTIME;
  }
  return PANTR_MPI_OK;
}

inline int table_rank(void* ctx, std::int32_t* out, char* /*msg*/, std::size_t /*len*/) noexcept {
  *out = ctx_of(ctx).rank;
  return PANTR_MPI_OK;
}

inline int table_size(void* ctx, std::int32_t* out, char* /*msg*/, std::size_t /*len*/) noexcept {
  *out = ctx_of(ctx).size;
  return PANTR_MPI_OK;
}

/// The element size and MPI type of a payload code; false for an unknown code.
inline bool payload_type(int dtype, std::size_t& bytes, MPI_Datatype& type) noexcept {
  if (dtype == PANTR_MPI_INT64) {
    bytes = sizeof(std::int64_t);
    type = MPI_INT64_T;
  } else if (dtype == PANTR_MPI_FLOAT64) {
    bytes = sizeof(double);
    type = MPI_DOUBLE;
  } else if (dtype == PANTR_MPI_FLOAT32) {
    bytes = sizeof(float);
    type = MPI_FLOAT;
  } else {
    return false;
  }
  return true;
}

inline int table_allgatherv(void* ctx, int dtype, const void* data, const std::int64_t shape[3],
                            pantr_mpi_sink* sink, char* msg, std::size_t len) noexcept {
  const MpiTableCtx& c = ctx_of(ctx);
  std::size_t bytes = 0;
  MPI_Datatype type = MPI_DATATYPE_NULL;
  if (!payload_type(dtype, bytes, type) || dtype == PANTR_MPI_FLOAT32) {
    put_message(msg, len, "allgatherv: unsupported element type " + std::to_string(dtype));
    return PANTR_MPI_ERR_VALUE;
  }
  if (shape[0] < 1 || shape[0] > 2 || shape[1] < 0 || shape[2] < 0) {
    put_message(msg, len, "allgatherv: a payload must have one or two non-negative extents.");
    return PANTR_MPI_ERR_VALUE;
  }
  int st = check_thread(c, msg, len);
  if (st != PANTR_MPI_OK) {
    return st;
  }
  try {
    std::vector<std::int64_t> shapes(3 * static_cast<std::size_t>(c.size));
    st = mpi_status(MPI_Allgather(shape, 3, MPI_INT64_T, shapes.data(), 3, MPI_INT64_T, c.comm),
                    "MPI_Allgather", msg, len);
    if (st != PANTR_MPI_OK) {
      return st;
    }
    std::vector<int> counts(static_cast<std::size_t>(c.size));
    std::vector<int> displs(static_cast<std::size_t>(c.size));
    long long total = 0;
    bool too_big = false;
    for (std::size_t r = 0; r < counts.size(); ++r) {
      const long long n =
          shapes[3 * r] == 2 ? shapes[3 * r + 1] * shapes[3 * r + 2] : shapes[3 * r + 1];
      too_big = too_big || n > INT_MAX - total;
      displs[r] = too_big ? 0 : static_cast<int>(total);
      counts[r] = too_big ? 0 : static_cast<int>(n);
      total += n;
    }
    if (too_big) {
      // Every rank saw the same shapes, so every rank returns here.
      put_message(msg, len, "allgatherv: the gathered payload exceeds INT_MAX elements.");
      return PANTR_MPI_ERR_VALUE;
    }
    std::vector<unsigned char> out(static_cast<std::size_t>(total) * bytes);
    st = mpi_status(MPI_Allgatherv(data, counts[static_cast<std::size_t>(c.rank)], type, out.data(),
                                   counts.data(), displs.data(), type, c.comm),
                    "MPI_Allgatherv", msg, len);
    if (st != PANTR_MPI_OK) {
      return st;
    }
    return sink->receive(sink->ctx, c.size, shapes.data(), out.data(), total, msg, len);
  } catch (const std::exception& e) {
    put_message(msg, len, e.what());
    return PANTR_MPI_ERR_RUNTIME;
  }
}

inline int table_allgather_owned(void* ctx, const std::int64_t* index,
                                 const std::int64_t index_shape[3], const double* value,
                                 const std::int64_t value_shape[3], pantr_mpi_sink* index_sink,
                                 pantr_mpi_sink* value_sink, char* msg, std::size_t len) noexcept {
  const int st = table_allgatherv(ctx, PANTR_MPI_INT64, index, index_shape, index_sink, msg, len);
  if (st != PANTR_MPI_OK) {
    return st;
  }
  return table_allgatherv(ctx, PANTR_MPI_FLOAT64, value, value_shape, value_sink, msg, len);
}

inline int table_allreduce_sum(void* ctx, int dtype, const void* send, void* recv,
                               std::int64_t count, char* msg, std::size_t len) noexcept {
  const MpiTableCtx& c = ctx_of(ctx);
  std::size_t bytes = 0;
  MPI_Datatype type = MPI_DATATYPE_NULL;
  if (!payload_type(dtype, bytes, type) || dtype == PANTR_MPI_INT64 || count < 0) {
    put_message(msg, len, "allreduce_sum: unsupported element type or count");
    return PANTR_MPI_ERR_VALUE;
  }
  int st = check_thread(c, msg, len);
  if (st != PANTR_MPI_OK) {
    return st;
  }
  const std::array<std::int64_t, 2> mine{count, -count};
  std::array<std::int64_t, 2> extremes{};
  st = mpi_status(MPI_Allreduce(mine.data(), extremes.data(), 2, MPI_INT64_T, MPI_MAX, c.comm),
                  "MPI_Allreduce", msg, len);
  if (st != PANTR_MPI_OK) {
    return st;
  }
  if (extremes[0] != -extremes[1]) {
    put_message(msg, len,
                "allreduce_sum: the ranks passed tensors of different sizes (from " +
                    std::to_string(-extremes[1]) + " to " + std::to_string(extremes[0]) +
                    " elements).");
    return PANTR_MPI_ERR_VALUE;
  }
  if (count > INT_MAX) {
    put_message(msg, len, "allreduce_sum: the tensor exceeds INT_MAX elements.");
    return PANTR_MPI_ERR_VALUE;
  }
  return mpi_status(MPI_Allreduce(send, recv, static_cast<int>(count), type, MPI_SUM, c.comm),
                    "MPI_Allreduce", msg, len);
}

inline void table_release(void* ctx) noexcept {
  auto* c = static_cast<MpiTableCtx*>(ctx);
  if (c->drop != nullptr) {
    c->drop(c->keep);
  }
  delete c;
}

}  // namespace detail

/// The table over a communicator, owning it: free it with `release_mpi_table`.
///
/// Args:
///     comm (MPI_Comm): The communicator, valid for the table's lifetime (not freed here).
///     keep (void*): Passed to `drop` when the table is released (for instance a Python reference
///         keeping the object that owns `comm` alive). Defaults to none.
///     drop (void (*)(void*)): Called once with `keep` on release. Defaults to none.
///     msg (char*), msg_len (std::size_t): The message buffer of a failure.
///
/// Returns:
///     pantr_mpi_comm*: The table, or null with a message: MPI not initialized or already
///     finalized, initialized below `MPI_THREAD_FUNNELED`, a null communicator, or an MPI call
///     failed. On failure `drop` is not called.
[[nodiscard]] inline pantr_mpi_comm* make_mpi_table(MPI_Comm comm, char* msg, std::size_t msg_len,
                                                    void* keep = nullptr,
                                                    void (*drop)(void*) = nullptr) noexcept {
  int flag = 0;
  if (detail::mpi_status(MPI_Initialized(&flag), "MPI_Initialized", msg, msg_len) != PANTR_MPI_OK) {
    return nullptr;
  }
  int done = 0;
  if (detail::mpi_status(MPI_Finalized(&done), "MPI_Finalized", msg, msg_len) != PANTR_MPI_OK) {
    return nullptr;
  }
  if (flag == 0 || done != 0) {
    detail::put_message(msg, msg_len, "pantr::mpi needs an initialized, not yet finalized MPI.");
    return nullptr;
  }
  int provided = 0;
  if (detail::mpi_status(MPI_Query_thread(&provided), "MPI_Query_thread", msg, msg_len) !=
      PANTR_MPI_OK) {
    return nullptr;
  }
  if (provided < MPI_THREAD_FUNNELED) {
    detail::put_message(
        msg, msg_len,
        "pantr::mpi needs MPI initialized with at least MPI_THREAD_FUNNELED (MPI_Init_thread); "
        "this MPI provides level " +
            std::to_string(provided) +
            ". pantr calls MPI only from the thread that initialized it, outside its thread pool.");
    return nullptr;
  }
  if (comm == MPI_COMM_NULL) {
    detail::put_message(msg, msg_len, "pantr::mpi: a null communicator (MPI_COMM_NULL).");
    return nullptr;
  }
  int rank = 0;
  int size = 0;
  if (detail::mpi_status(MPI_Comm_rank(comm, &rank), "MPI_Comm_rank", msg, msg_len) !=
          PANTR_MPI_OK ||
      detail::mpi_status(MPI_Comm_size(comm, &size), "MPI_Comm_size", msg, msg_len) !=
          PANTR_MPI_OK) {
    return nullptr;
  }
  try {
    auto ctx = std::make_unique<detail::MpiTableCtx>(
        detail::MpiTableCtx{comm, rank, size, provided < MPI_THREAD_SERIALIZED, keep, drop});
    auto table = std::make_unique<pantr_mpi_comm>();
    table->abi_version = PANTR_MPI_COMM_ABI_VERSION;
    table->struct_size = static_cast<std::uint32_t>(sizeof(pantr_mpi_comm));
    table->rank = &detail::table_rank;
    table->size = &detail::table_size;
    table->allgatherv = &detail::table_allgatherv;
    table->allgather_owned = &detail::table_allgather_owned;
    table->allreduce_sum = &detail::table_allreduce_sum;
    table->release = &detail::table_release;
    table->ctx = ctx.release();
    return table.release();
  } catch (const std::exception& e) {
    detail::put_message(msg, msg_len, e.what());
    return nullptr;
  }
}

/// Release a table made by `make_mpi_table` (calls its `release`, then frees it).
inline void release_mpi_table(pantr_mpi_comm* table) noexcept {
  if (table != nullptr) {
    if (table->release != nullptr) {
      table->release(table->ctx);
    }
    delete table;
  }
}

}  // namespace pantr::mpi
