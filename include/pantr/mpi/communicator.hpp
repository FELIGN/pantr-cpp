/// @file communicator.hpp
/// @brief The `Communicator` concept the distributed algorithms are written against, and its
/// type-erased holder `AnyCommunicator`. MPI-free.
///
/// A communicator has one member per payload the distributed algorithms send, so that a
/// duck-typed Python communicator sees each payload as one call of its own (design §5.4):
///
/// - `allgatherv(Block<double>)`, `allgatherv(Block<std::int64_t>)`: every rank's block of values,
///   in rank order (the collocation values, the dolfinx owner ids, the broadcast partition);
/// - `allgather_owned(indices, values)`: every rank's pair of owned DOF indices and their
///   coefficients (the quasi-interpolants);
/// - `allreduce_sum(data, shape)`: the elementwise sum of one tensor per rank (the L2 load), in
///   `float` or `double`.
///
/// A gathered block keeps its shape, one or two extents, since the payloads are `(m,)` or `(m, r)`
/// arrays and the consumers read `r` from them. A consumer validates what it receives (the count
/// of blocks, their extents), so a communicator returning an inconsistent gather raises
/// `value_error` where the payload is used.
///
/// Every collective is called by every rank, in the same order, outside any `parallel_for` and
/// from the thread that initialized MPI. An exception thrown on one rank before a collective
/// leaves the others waiting in it: the algorithms validate their arguments identically on every
/// rank first, so that only a rank-local failure (a user callable raising on one rank) can do this.
/// Design note: design/mpi.md.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/mpi/comm_capsule.h"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pantr::mpi {

/// One rank's payload: C-contiguous data and its one or two extents.
///
/// Attributes:
///     data (std::span<const T>): The elements, row-major.
///     shape (std::vector<std::size_t>): One extent `(m,)` or two `(m, r)`; their product is
///         `data.size()`.
template <class T>
struct Block {
  std::span<const T> data;
  std::vector<std::size_t> shape;
};

/// Check a block's shape against its data.
///
/// Raises:
///     value_error: Not one or two extents, or their product is not the element count.
template <class T>
void check_block(const Block<T>& b) {
  if (b.shape.empty() || b.shape.size() > 2) {
    throw value_error("a communicator payload must have one or two extents; got " +
                      std::to_string(b.shape.size()) + ".");
  }
  if (checked_product(b.shape) != b.data.size()) {
    throw value_error("a communicator payload's shape does not match its " +
                      std::to_string(b.data.size()) + " elements.");
  }
}

/// A one-dimensional block over `data`.
template <class T>
[[nodiscard]] Block<T> block_1d(std::span<const T> data) {
  return Block<T>{data, {data.size()}};
}

/// Every rank's block of one gather, in rank order.
///
/// Attributes:
///     data (std::vector<T>): The blocks concatenated.
///     shapes (std::vector<std::vector<std::size_t>>): Each block's extents, as the sender gave
///         them; a duck-typed communicator may return any number of extents, `0` included, which
///         the consumers check.
template <class T>
struct Gathered {
  std::vector<T> data;
  std::vector<std::vector<std::size_t>> shapes;

  /// The number of blocks received.
  [[nodiscard]] std::size_t count() const noexcept { return shapes.size(); }

  /// The element offset of block `r` in `data`.
  [[nodiscard]] std::size_t offset(std::size_t r) const {
    std::size_t off = 0;
    for (std::size_t k = 0; k < r; ++k) {
      off += checked_product(shapes[k]);
    }
    return off;
  }

  /// The elements of block `r`.
  [[nodiscard]] std::span<const T> block(std::size_t r) const {
    return std::span<const T>(data).subspan(offset(r), checked_product(shapes[r]));
  }
};

/// The operations the distributed algorithms need from a communicator (see the file comment).
template <class C>
concept Communicator = requires(const C& c, const Block<double>& d, const Block<std::int64_t>& i,
                                std::span<const double> sd, std::span<const float> sf,
                                std::span<const std::size_t> shape) {
  { c.rank() } -> std::convertible_to<int>;
  { c.size() } -> std::convertible_to<int>;
  { c.allgatherv(d) } -> std::same_as<Gathered<double>>;
  { c.allgatherv(i) } -> std::same_as<Gathered<std::int64_t>>;
  { c.allgather_owned(i, d) } -> std::same_as<std::pair<Gathered<std::int64_t>, Gathered<double>>>;
  { c.allreduce_sum(sd, shape) } -> std::same_as<NdArray<double>>;
  { c.allreduce_sum(sf, shape) } -> std::same_as<NdArray<float>>;
};

/// A communicator of one rank: every gather returns the caller's own block, every reduction its
/// own tensor. What a serial program passes to a distributed algorithm.
class SelfCommunicator {
 public:
  /// Always 0.
  [[nodiscard]] int rank() const noexcept { return 0; }
  /// Always 1.
  [[nodiscard]] int size() const noexcept { return 1; }

  /// Return the caller's block.
  template <class T>
  [[nodiscard]] Gathered<T> allgatherv(const Block<T>& b) const {
    check_block(b);
    return Gathered<T>{std::vector<T>(b.data.begin(), b.data.end()), {b.shape}};
  }

  /// Return the caller's pair.
  [[nodiscard]] std::pair<Gathered<std::int64_t>, Gathered<double>> allgather_owned(
      const Block<std::int64_t>& index, const Block<double>& value) const {
    return {allgatherv(index), allgatherv(value)};
  }

  /// Return the caller's tensor.
  template <class T>
  [[nodiscard]] NdArray<T> allreduce_sum(std::span<const T> data,
                                         std::span<const std::size_t> shape) const {
    NdArray<T> out(std::vector<std::size_t>(shape.begin(), shape.end()), for_overwrite);
    if (out.size() != data.size()) {
      throw value_error("allreduce_sum: the shape does not match the data.");
    }
    std::copy(data.begin(), data.end(), out.data());
    return out;
  }
};

namespace detail {

/// The virtual interface `AnyCommunicator` erases to.
class CommModel {
 public:
  CommModel() = default;
  CommModel(const CommModel&) = delete;
  CommModel& operator=(const CommModel&) = delete;
  CommModel(CommModel&&) = delete;
  CommModel& operator=(CommModel&&) = delete;
  virtual ~CommModel() = default;
  [[nodiscard]] virtual int rank() const = 0;
  [[nodiscard]] virtual int size() const = 0;
  [[nodiscard]] virtual Gathered<double> allgatherv(const Block<double>& b) const = 0;
  [[nodiscard]] virtual Gathered<std::int64_t> allgatherv(const Block<std::int64_t>& b) const = 0;
  [[nodiscard]] virtual std::pair<Gathered<std::int64_t>, Gathered<double>> allgather_owned(
      const Block<std::int64_t>& index, const Block<double>& value) const = 0;
  [[nodiscard]] virtual NdArray<double> allreduce_sum(std::span<const double> data,
                                                      std::span<const std::size_t> shape) const = 0;
  [[nodiscard]] virtual NdArray<float> allreduce_sum(std::span<const float> data,
                                                     std::span<const std::size_t> shape) const = 0;
};

template <Communicator C>
class CommAdapter final : public CommModel {
 public:
  explicit CommAdapter(C comm) : comm_(std::move(comm)) {}
  [[nodiscard]] int rank() const override { return static_cast<int>(comm_.rank()); }
  [[nodiscard]] int size() const override { return static_cast<int>(comm_.size()); }
  [[nodiscard]] Gathered<double> allgatherv(const Block<double>& b) const override {
    return comm_.allgatherv(b);
  }
  [[nodiscard]] Gathered<std::int64_t> allgatherv(const Block<std::int64_t>& b) const override {
    return comm_.allgatherv(b);
  }
  [[nodiscard]] std::pair<Gathered<std::int64_t>, Gathered<double>> allgather_owned(
      const Block<std::int64_t>& index, const Block<double>& value) const override {
    return comm_.allgather_owned(index, value);
  }
  [[nodiscard]] NdArray<double> allreduce_sum(std::span<const double> data,
                                              std::span<const std::size_t> shape) const override {
    return comm_.allreduce_sum(data, shape);
  }
  [[nodiscard]] NdArray<float> allreduce_sum(std::span<const float> data,
                                             std::span<const std::size_t> shape) const override {
    return comm_.allreduce_sum(data, shape);
  }

 private:
  C comm_;
};

/// The size of the message buffer every table call gets.
inline constexpr std::size_t kMessageLen = 512;

/// Throw the exception a table status stands for.
inline void throw_status(int status, const char* msg, const char* what) {
  const std::string text = msg[0] != '\0' ? std::string(msg) : std::string(what) + " failed";
  if (status == PANTR_MPI_ERR_VALUE) {
    throw value_error(text);
  }
  throw state_error(text);
}

/// Encode a block's shape as the table's three `int64_t`.
template <class T>
[[nodiscard]] std::array<std::int64_t, 3> wire_shape(const Block<T>& b) {
  check_block(b);
  return {static_cast<std::int64_t>(b.shape.size()), static_cast<std::int64_t>(b.shape[0]),
          b.shape.size() == 2 ? static_cast<std::int64_t>(b.shape[1]) : 0};
}

/// A sink that copies one gather into a `Gathered<T>`.
template <class T>
struct GatherInto {
  Gathered<T> out;

  static int receive(void* ctx, std::int64_t n_ranks, const std::int64_t* shapes, const void* data,
                     std::int64_t n_elems, char* msg, std::size_t msg_len) noexcept {
    try {
      auto* self = static_cast<GatherInto*>(ctx);
      self->out.shapes.clear();
      self->out.shapes.reserve(static_cast<std::size_t>(n_ranks));
      std::size_t total = 0;
      for (std::int64_t r = 0; r < n_ranks; ++r) {
        const std::int64_t ndim = shapes[3 * r];
        if (ndim < 1 || ndim > 2 || shapes[3 * r + 1] < 0 || shapes[3 * r + 2] < 0) {
          throw value_error("a gathered block has an invalid shape.");
        }
        std::vector<std::size_t> s{static_cast<std::size_t>(shapes[3 * r + 1])};
        if (ndim == 2) {
          s.push_back(static_cast<std::size_t>(shapes[3 * r + 2]));
        }
        total += checked_product(s);
        self->out.shapes.push_back(std::move(s));
      }
      if (n_elems < 0 || total != static_cast<std::size_t>(n_elems)) {
        throw value_error("a gather delivered another element count than its shapes hold.");
      }
      const auto* p = static_cast<const T*>(data);
      self->out.data.assign(p, p + total);
      return PANTR_MPI_OK;
    } catch (const std::exception& e) {
      if (msg_len > 0) {
        std::snprintf(msg, msg_len, "%s", e.what());
      }
      return PANTR_MPI_ERR_VALUE;
    }
  }

  [[nodiscard]] pantr_mpi_sink sink() noexcept {
    return pantr_mpi_sink{this, &GatherInto::receive};
  }
};

template <class T>
[[nodiscard]] constexpr int dtype_code() noexcept {
  if constexpr (std::same_as<T, std::int64_t>) {
    return PANTR_MPI_INT64;
  } else if constexpr (std::same_as<T, double>) {
    return PANTR_MPI_FLOAT64;
  } else {
    static_assert(std::same_as<T, float>);
    return PANTR_MPI_FLOAT32;
  }
}

/// A communicator backed by a `pantr_mpi_comm` table (another module's).
class TableComm final : public CommModel {
 public:
  explicit TableComm(std::shared_ptr<const pantr_mpi_comm> table) : t_(std::move(table)) {}

  [[nodiscard]] int rank() const override { return query(t_->rank, "rank"); }
  [[nodiscard]] int size() const override { return query(t_->size, "size"); }

  [[nodiscard]] Gathered<double> allgatherv(const Block<double>& b) const override {
    return gather(b);
  }
  [[nodiscard]] Gathered<std::int64_t> allgatherv(const Block<std::int64_t>& b) const override {
    return gather(b);
  }
  [[nodiscard]] std::pair<Gathered<std::int64_t>, Gathered<double>> allgather_owned(
      const Block<std::int64_t>& index, const Block<double>& value) const override {
    const std::array<std::int64_t, 3> is = wire_shape(index);
    const std::array<std::int64_t, 3> vs = wire_shape(value);
    GatherInto<std::int64_t> gi;
    GatherInto<double> gv;
    pantr_mpi_sink si = gi.sink();
    pantr_mpi_sink sv = gv.sink();
    std::array<char, kMessageLen> msg{};
    const int status = t_->allgather_owned(t_->ctx, index.data.data(), is.data(), value.data.data(),
                                           vs.data(), &si, &sv, msg.data(), msg.size());
    if (status != PANTR_MPI_OK) {
      throw_status(status, msg.data(), "allgather_owned");
    }
    return {std::move(gi.out), std::move(gv.out)};
  }
  [[nodiscard]] NdArray<double> allreduce_sum(std::span<const double> data,
                                              std::span<const std::size_t> shape) const override {
    return reduce(data, shape);
  }
  [[nodiscard]] NdArray<float> allreduce_sum(std::span<const float> data,
                                             std::span<const std::size_t> shape) const override {
    return reduce(data, shape);
  }

 private:
  using Query = int (*)(void*, std::int32_t*, char*, std::size_t);

  [[nodiscard]] int query(Query fn, const char* what) const {
    std::int32_t v = 0;
    std::array<char, kMessageLen> msg{};
    const int status = fn(t_->ctx, &v, msg.data(), msg.size());
    if (status != PANTR_MPI_OK) {
      throw_status(status, msg.data(), what);
    }
    return v;
  }

  template <class T>
  [[nodiscard]] Gathered<T> gather(const Block<T>& b) const {
    const std::array<std::int64_t, 3> s = wire_shape(b);
    GatherInto<T> g;
    pantr_mpi_sink sink = g.sink();
    std::array<char, kMessageLen> msg{};
    const int status = t_->allgatherv(t_->ctx, dtype_code<T>(), b.data.data(), s.data(), &sink,
                                      msg.data(), msg.size());
    if (status != PANTR_MPI_OK) {
      throw_status(status, msg.data(), "allgatherv");
    }
    return std::move(g.out);
  }

  template <class T>
  [[nodiscard]] NdArray<T> reduce(std::span<const T> data,
                                  std::span<const std::size_t> shape) const {
    NdArray<T> out(std::vector<std::size_t>(shape.begin(), shape.end()), for_overwrite);
    if (out.size() != data.size()) {
      throw value_error("allreduce_sum: the shape does not match the data.");
    }
    std::array<char, kMessageLen> msg{};
    const int status =
        t_->allreduce_sum(t_->ctx, dtype_code<T>(), data.data(), out.data(),
                          static_cast<std::int64_t>(data.size()), msg.data(), msg.size());
    if (status != PANTR_MPI_OK) {
      throw_status(status, msg.data(), "allreduce_sum");
    }
    return out;
  }

  std::shared_ptr<const pantr_mpi_comm> t_;
};

}  // namespace detail

/// A communicator of any type, held by shared reference: what the distributed algorithms are
/// instantiated with in the Python extension, so that they are compiled once for every
/// communicator a caller may pass. Immutable; copies share the communicator.
class AnyCommunicator {
 public:
  /// Hold any `Communicator` (moved in).
  template <Communicator C>
    requires(!std::same_as<std::remove_cvref_t<C>, AnyCommunicator>)
  explicit AnyCommunicator(C comm)
      : impl_(std::make_shared<detail::CommAdapter<C>>(std::move(comm))) {}

  /// Hold a communicator given as a C table (from another module's capsule).
  ///
  /// Args:
  ///     table (std::shared_ptr<const pantr_mpi_comm>): The table; its deleter is what releases
  ///         the communicator.
  ///
  /// Raises:
  ///     value_error: A null table, another ABI version, a smaller struct, or a missing function.
  [[nodiscard]] static AnyCommunicator from_table(std::shared_ptr<const pantr_mpi_comm> table) {
    if (!table) {
      throw value_error("a null communicator table");
    }
    if (table->abi_version != PANTR_MPI_COMM_ABI_VERSION ||
        table->struct_size < sizeof(pantr_mpi_comm)) {
      throw value_error("the communicator table has ABI version " +
                        std::to_string(table->abi_version) + " and size " +
                        std::to_string(table->struct_size) + "; this pantr expects version " +
                        std::to_string(PANTR_MPI_COMM_ABI_VERSION) + " and size " +
                        std::to_string(sizeof(pantr_mpi_comm)) +
                        " (pantr and pantr-mpi must be the same version)");
    }
    if (table->rank == nullptr || table->size == nullptr || table->allgatherv == nullptr ||
        table->allgather_owned == nullptr || table->allreduce_sum == nullptr) {
      throw value_error("the communicator table lacks a function");
    }
    return AnyCommunicator(std::make_shared<detail::TableComm>(std::move(table)));
  }

  /// This process's rank.
  [[nodiscard]] int rank() const { return impl_->rank(); }
  /// The number of ranks.
  [[nodiscard]] int size() const { return impl_->size(); }
  /// See the file comment.
  [[nodiscard]] Gathered<double> allgatherv(const Block<double>& b) const {
    return impl_->allgatherv(b);
  }
  /// See the file comment.
  [[nodiscard]] Gathered<std::int64_t> allgatherv(const Block<std::int64_t>& b) const {
    return impl_->allgatherv(b);
  }
  /// See the file comment.
  [[nodiscard]] std::pair<Gathered<std::int64_t>, Gathered<double>> allgather_owned(
      const Block<std::int64_t>& index, const Block<double>& value) const {
    return impl_->allgather_owned(index, value);
  }
  /// See the file comment.
  [[nodiscard]] NdArray<double> allreduce_sum(std::span<const double> data,
                                              std::span<const std::size_t> shape) const {
    return impl_->allreduce_sum(data, shape);
  }
  /// See the file comment.
  [[nodiscard]] NdArray<float> allreduce_sum(std::span<const float> data,
                                             std::span<const std::size_t> shape) const {
    return impl_->allreduce_sum(data, shape);
  }

 private:
  explicit AnyCommunicator(std::shared_ptr<const detail::CommModel> impl)
      : impl_(std::move(impl)) {}

  std::shared_ptr<const detail::CommModel> impl_;
};

static_assert(Communicator<SelfCommunicator>);
static_assert(Communicator<AnyCommunicator>);

/// Build a `pantr_mpi_comm` table over a C++ communicator, for handing to another module.
///
/// Every function catches what the communicator throws and returns it as a status:
/// `value_error` as `PANTR_MPI_ERR_VALUE`, anything else as `PANTR_MPI_ERR_RUNTIME`.
///
/// Args:
///     comm (C): The communicator (moved into the table's context).
///
/// Returns:
///     std::unique_ptr<pantr_mpi_comm, void (*)(pantr_mpi_comm*)>: The table; its deleter calls
///     `release` and frees the table.
template <Communicator C>
[[nodiscard]] std::unique_ptr<pantr_mpi_comm, void (*)(pantr_mpi_comm*)> make_comm_table(C comm);

namespace detail {

template <class F>
[[nodiscard]] int guarded(char* msg, std::size_t msg_len, F&& f) noexcept {
  try {
    f();
    return PANTR_MPI_OK;
  } catch (const value_error& e) {
    if (msg_len > 0) {
      std::snprintf(msg, msg_len, "%s", e.what());
    }
    return PANTR_MPI_ERR_VALUE;
  } catch (const std::exception& e) {
    if (msg_len > 0) {
      std::snprintf(msg, msg_len, "%s", e.what());
    }
    return PANTR_MPI_ERR_RUNTIME;
  } catch (...) {
    if (msg_len > 0) {
      std::snprintf(msg, msg_len, "%s", "an unknown exception");
    }
    return PANTR_MPI_ERR_RUNTIME;
  }
}

[[nodiscard]] inline std::vector<std::size_t> shape_from_wire(const std::int64_t s[3]) {
  if (s[0] < 1 || s[0] > 2 || s[1] < 0 || s[2] < 0) {
    throw value_error("a payload shape must have one or two non-negative extents.");
  }
  std::vector<std::size_t> out{static_cast<std::size_t>(s[1])};
  if (s[0] == 2) {
    out.push_back(static_cast<std::size_t>(s[2]));
  }
  return out;
}

template <class T>
void deliver(const Gathered<T>& g, pantr_mpi_sink* sink) {
  std::vector<std::int64_t> shapes;
  shapes.reserve(3 * g.count());
  for (const std::vector<std::size_t>& s : g.shapes) {
    if (s.empty() || s.size() > 2) {
      throw value_error("a gathered block must have one or two extents.");
    }
    shapes.push_back(static_cast<std::int64_t>(s.size()));
    shapes.push_back(static_cast<std::int64_t>(s[0]));
    shapes.push_back(s.size() == 2 ? static_cast<std::int64_t>(s[1]) : 0);
  }
  std::array<char, kMessageLen> msg{};
  const int status =
      sink->receive(sink->ctx, static_cast<std::int64_t>(g.count()), shapes.data(), g.data.data(),
                    static_cast<std::int64_t>(g.data.size()), msg.data(), msg.size());
  if (status != PANTR_MPI_OK) {
    throw_status(status, msg.data(), "a gather sink");
  }
}

/// Raise unless a reduction returned as many elements as it was sent.
inline void check_reduced(std::size_t got, std::size_t sent) {
  if (got != sent) {
    throw value_error("allreduce_sum returned " + std::to_string(got) + " elements for " +
                      std::to_string(sent) + ".");
  }
}

template <Communicator C>
struct TableFns {
  static C& self(void* ctx) noexcept { return *static_cast<C*>(ctx); }

  static int rank(void* ctx, std::int32_t* out, char* msg, std::size_t len) noexcept {
    return guarded(msg, len, [&] { *out = static_cast<std::int32_t>(self(ctx).rank()); });
  }
  static int size(void* ctx, std::int32_t* out, char* msg, std::size_t len) noexcept {
    return guarded(msg, len, [&] { *out = static_cast<std::int32_t>(self(ctx).size()); });
  }
  static int allgatherv(void* ctx, int dtype, const void* data, const std::int64_t shape[3],
                        pantr_mpi_sink* sink, char* msg, std::size_t len) noexcept {
    return guarded(msg, len, [&] {
      std::vector<std::size_t> s = shape_from_wire(shape);
      const std::size_t n = checked_product(s);
      if (dtype == PANTR_MPI_INT64) {
        const Block<std::int64_t> b{{static_cast<const std::int64_t*>(data), n}, std::move(s)};
        deliver(self(ctx).allgatherv(b), sink);
      } else if (dtype == PANTR_MPI_FLOAT64) {
        const Block<double> b{{static_cast<const double*>(data), n}, std::move(s)};
        deliver(self(ctx).allgatherv(b), sink);
      } else {
        throw value_error("allgatherv: unsupported element type " + std::to_string(dtype));
      }
    });
  }
  static int allgather_owned(void* ctx, const std::int64_t* index, const std::int64_t is[3],
                             const double* value, const std::int64_t vs[3],
                             pantr_mpi_sink* index_sink, pantr_mpi_sink* value_sink, char* msg,
                             std::size_t len) noexcept {
    return guarded(msg, len, [&] {
      std::vector<std::size_t> si = shape_from_wire(is);
      std::vector<std::size_t> sv = shape_from_wire(vs);
      const Block<std::int64_t> bi{{index, checked_product(si)}, std::move(si)};
      const Block<double> bv{{value, checked_product(sv)}, std::move(sv)};
      auto [gi, gv] = self(ctx).allgather_owned(bi, bv);
      deliver(gi, index_sink);
      deliver(gv, value_sink);
    });
  }
  static int allreduce_sum(void* ctx, int dtype, const void* send, void* recv, std::int64_t count,
                           char* msg, std::size_t len) noexcept {
    return guarded(msg, len, [&] {
      if (count < 0) {
        throw value_error("allreduce_sum: a negative count");
      }
      const auto n = static_cast<std::size_t>(count);
      const std::array<std::size_t, 1> shape{n};
      if (dtype == PANTR_MPI_FLOAT64) {
        const NdArray<double> r =
            self(ctx).allreduce_sum(std::span<const double>(static_cast<const double*>(send), n),
                                    std::span<const std::size_t>(shape));
        check_reduced(r.size(), n);
        std::copy(r.data(), r.data() + n, static_cast<double*>(recv));
      } else if (dtype == PANTR_MPI_FLOAT32) {
        const NdArray<float> r =
            self(ctx).allreduce_sum(std::span<const float>(static_cast<const float*>(send), n),
                                    std::span<const std::size_t>(shape));
        check_reduced(r.size(), n);
        std::copy(r.data(), r.data() + n, static_cast<float*>(recv));
      } else {
        throw value_error("allreduce_sum: unsupported element type " + std::to_string(dtype));
      }
    });
  }
  static void release(void* ctx) noexcept { delete static_cast<C*>(ctx); }
};

}  // namespace detail

template <Communicator C>
std::unique_ptr<pantr_mpi_comm, void (*)(pantr_mpi_comm*)> make_comm_table(C comm) {
  using Fns = detail::TableFns<C>;
  auto ctx = std::make_unique<C>(std::move(comm));
  auto* table = new pantr_mpi_comm{};
  table->abi_version = PANTR_MPI_COMM_ABI_VERSION;
  table->struct_size = static_cast<std::uint32_t>(sizeof(pantr_mpi_comm));
  table->ctx = ctx.release();
  table->rank = &Fns::rank;
  table->size = &Fns::size;
  table->allgatherv = &Fns::allgatherv;
  table->allgather_owned = &Fns::allgather_owned;
  table->allreduce_sum = &Fns::allreduce_sum;
  table->release = &Fns::release;
  return {table, [](pantr_mpi_comm* t) {
            if (t->release != nullptr) {
              t->release(t->ctx);
            }
            delete t;
          }};
}

}  // namespace pantr::mpi
