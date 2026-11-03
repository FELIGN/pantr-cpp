/// @file mpi_communicator.hpp
/// @brief A `Communicator` over an `MPI_Comm`, on the `pantr::mpi` target. One of the two pantr
/// headers that include `<mpi.h>` (through `mpi_table.hpp`, which holds the MPI calls).
///
/// `MpiCommunicator` is the `comm_capsule.h` table of `make_mpi_table` held as an
/// `AnyCommunicator`, so a C++ program and the `pantr-mpi` extension run the same MPI code. Its
/// constructor checks that MPI is initialized at `MPI_THREAD_FUNNELED` or above and refuses
/// otherwise: the caller must be the thread that initialized MPI (design §2.8).
/// Design note: design/mpi.md.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/mpi/comm_capsule.h"
#include "pantr/mpi/communicator.hpp"
#include "pantr/mpi/mpi_table.hpp"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>

namespace pantr::mpi {

/// A communicator over an `MPI_Comm` the caller owns (not freed here). Copies share it.
///
/// Attributes:
///     comm_ (AnyCommunicator): The table over the handle.
///     handle_ (MPI_Comm): The handle.
class MpiCommunicator {
 public:
  /// Wrap a communicator.
  ///
  /// Raises:
  ///     state_error: MPI is not initialized, or already finalized; it was initialized with a
  ///         thread level below `MPI_THREAD_FUNNELED`; a null communicator; an MPI call failed.
  explicit MpiCommunicator(MPI_Comm comm) : comm_(make(comm)), handle_(comm) {}

  /// This process's rank.
  [[nodiscard]] int rank() const { return comm_.rank(); }
  /// The number of ranks.
  [[nodiscard]] int size() const { return comm_.size(); }
  /// The wrapped handle.
  [[nodiscard]] MPI_Comm handle() const noexcept { return handle_; }

  /// `MPI_Allgather` of the shapes, then `MPI_Allgatherv` of the data.
  template <class T>
  [[nodiscard]] Gathered<T> allgatherv(const Block<T>& b) const {
    return comm_.allgatherv(b);
  }
  /// Two `allgatherv`, the indices first.
  [[nodiscard]] std::pair<Gathered<std::int64_t>, Gathered<double>> allgather_owned(
      const Block<std::int64_t>& index, const Block<double>& value) const {
    return comm_.allgather_owned(index, value);
  }
  /// `MPI_Allreduce` with `MPI_SUM`, after checking that every rank passes the same size.
  template <class T>
  [[nodiscard]] NdArray<T> allreduce_sum(std::span<const T> data,
                                         std::span<const std::size_t> shape) const {
    return comm_.allreduce_sum(data, shape);
  }

 private:
  [[nodiscard]] static AnyCommunicator make(MPI_Comm comm) {
    std::array<char, detail::kMessageLen> msg{};
    pantr_mpi_comm* table = make_mpi_table(comm, msg.data(), msg.size());
    if (table == nullptr) {
      throw state_error(msg.data());
    }
    return AnyCommunicator::from_table(std::shared_ptr<pantr_mpi_comm>(table, &release_mpi_table));
  }

  AnyCommunicator comm_;
  MPI_Comm handle_;
};

static_assert(Communicator<MpiCommunicator>);

}  // namespace pantr::mpi
