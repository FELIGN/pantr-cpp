// The distributed algorithms of pantr::mpi (include/pantr/mpi/), run over fake communicators and
// so MPI-free (cpp/tests/mpi_fake.hpp: `run_ranks` runs n ranks as n threads with real collective
// semantics, `PresetComm` answers every gather with fixed blocks). Oracles, each from a route
// other than the code under test:
//   - the distributed result against the SERIAL operation on the global space (`interpolate`,
//     `l2_project`, `quasi_interpolate`, `THBQuasiInterpolation`), on 2, 3 and 4 ranks over a real
//     partition (`partition_grid`, rcb), checked on every rank and again on the DOFs the rank owns
//     (the `LocalSpace` maps of `DistributedSpace::local`); tolerances are derived where they are
//     used (`coefficient_tolerance`, `l2_tolerance`, `qi_tolerance`, `thb_qi_tolerance`);
//   - the defining property of a quasi-interpolant: it returns the coefficients of a spline of its
//     own space;
//   - hand-computed expectations for the assembly of gathered payloads (`block_bounds`,
//     `assemble_blocks`, `gather_owned_coefficients`, `partition_from_owner`), with the shapes and
//     the messages the Python implementation had;
//   - the geometry of the quadrature lattice against the cell breakpoints
//     (`owned_quadrature_mask`);
//   - invariants of the partition broadcast: every rank holds rank 0's partition, whatever its own
//     partitioner would have returned, and only rank 0 computes it;
//   - the C table (`make_comm_table`, `AnyCommunicator::from_table`) against the direct
//     communicator, and against payloads written out by hand.
// `MpiCommunicator` is not here: the real-MPI tests are cpp/tests/mpi/.
#include <doctest/doctest.h>

#include "mpi_fake.hpp"
#include "thb_test_support.hpp"

#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/evaluate.hpp>
#include <pantr/bspline/fit.hpp>
#include <pantr/bspline/knot_factories.hpp>
#include <pantr/bspline/l2_project.hpp>
#include <pantr/bspline/quasi_interpolate.hpp>
#include <pantr/bspline/space.hpp>
#include <pantr/bspline/space_1d.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/core/mdspan.hpp>
#include <pantr/core/ndarray.hpp>
#include <pantr/grid/hierarchical_grid.hpp>
#include <pantr/grid/partition.hpp>
#include <pantr/grid/tensor_product_grid.hpp>
#include <pantr/mpi/collocation.hpp>
#include <pantr/mpi/communicator.hpp>
#include <pantr/mpi/distributed_space.hpp>
#include <pantr/mpi/l2.hpp>
#include <pantr/mpi/partition_from_owner.hpp>
#include <pantr/mpi/quasi_interpolate.hpp>
#include <pantr/quad/points_lattice.hpp>
#include <pantr/thb/quasi_interpolate.hpp>
#include <pantr/thb/space.hpp>
#include <pantr/thb/spline.hpp>

#include <Eigen/Core>
#include <Eigen/LU>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace bs = pantr::bspline;
namespace gr = pantr::grid;
namespace mp = pantr::mpi;
namespace mt = pantr::mpi::testing;
namespace th = pantr::thb;
using pantr::NdArray;
using pantr::value_error;
using thb_test::build_slack;
using thb_test::I64;
using I64Vec = std::vector<I64>;
using Owners = std::vector<std::int32_t>;
using Shapes = std::vector<std::vector<std::size_t>>;

template <class T>
constexpr double unit() {
  return static_cast<double>(std::numeric_limits<T>::epsilon());
}

/// `gamma_n = n u / (1 - n u)`, the factor of a recursive sum of `n` terms (Higham, Accuracy and
/// Stability of Numerical Algorithms, 2nd ed., eq. (3.1) and (4.4)).
double gamma_n(double n, double u) { return n * u / (1.0 - n * u); }

// ---- Payload builders ---------------------------------------------------------------------------

/// Blocks of int64, each one-dimensional.
mp::Gathered<I64> int_blocks(const std::vector<I64Vec>& blocks) {
  mp::Gathered<I64> g;
  for (const I64Vec& b : blocks) {
    g.data.insert(g.data.end(), b.begin(), b.end());
    g.shapes.push_back({b.size()});
  }
  return g;
}

/// Blocks of double with the shapes the sender gave them; `data` is the concatenation.
mp::Gathered<double> real_blocks(std::vector<double> data, Shapes shapes) {
  return mp::Gathered<double>{std::move(data), std::move(shapes)};
}

/// A communicator that answers every gather from the presets, whatever it is sent.
mt::PresetComm preset(int rank, int size, mp::Gathered<I64> i64 = {},
                      mp::Gathered<double> f64 = {}) {
  return mt::PresetComm(rank, size, std::move(i64), std::move(f64));
}

/// The communicator of one rank that counts the gathers it is asked for (answering with the
/// caller's own block, so only for `size == 1` or for payloads nobody reads).
class CountingComm {
 public:
  CountingComm(int rank, int size)
      : rank_(rank), size_(size), gathers_(std::make_shared<std::atomic<int>>(0)) {}
  [[nodiscard]] int rank() const noexcept { return rank_; }
  [[nodiscard]] int size() const noexcept { return size_; }
  [[nodiscard]] int gathers() const noexcept { return gathers_->load(); }
  template <class T>
  [[nodiscard]] mp::Gathered<T> allgatherv(const mp::Block<T>& b) const {
    ++*gathers_;
    return mp::SelfCommunicator{}.allgatherv(b);
  }
  [[nodiscard]] std::pair<mp::Gathered<I64>, mp::Gathered<double>> allgather_owned(
      const mp::Block<I64>& index, const mp::Block<double>& value) const {
    ++*gathers_;
    return mp::SelfCommunicator{}.allgather_owned(index, value);
  }
  template <class T>
  [[nodiscard]] NdArray<T> allreduce_sum(std::span<const T> data,
                                         std::span<const std::size_t> shape) const {
    return mp::SelfCommunicator{}.allreduce_sum(data, shape);
  }

 private:
  int rank_;
  int size_;
  std::shared_ptr<std::atomic<int>> gathers_;
};
static_assert(mp::Communicator<CountingComm>);

/// A communicator whose `rank()` throws: an algorithm that must not read the rank runs over it.
class NoRankComm {
 public:
  NoRankComm(int size, mp::Gathered<I64> i64) : inner_(0, size, std::move(i64), {}) {}
  [[nodiscard]] int rank() const { throw pantr::state_error("rank() was read"); }
  [[nodiscard]] int size() const noexcept { return inner_.size(); }
  [[nodiscard]] mp::Gathered<I64> allgatherv(const mp::Block<I64>& b) const {
    return inner_.allgatherv(b);
  }
  [[nodiscard]] mp::Gathered<double> allgatherv(const mp::Block<double>& b) const {
    return inner_.allgatherv(b);
  }
  [[nodiscard]] std::pair<mp::Gathered<I64>, mp::Gathered<double>> allgather_owned(
      const mp::Block<I64>& i, const mp::Block<double>& v) const {
    return inner_.allgather_owned(i, v);
  }
  template <class T>
  [[nodiscard]] NdArray<T> allreduce_sum(std::span<const T> data,
                                         std::span<const std::size_t> shape) const {
    return inner_.allreduce_sum(data, shape);
  }

 private:
  mt::PresetComm inner_;
};
static_assert(mp::Communicator<NoRankComm>);

/// What a rank's body ended in: the type and message of the exception it threw, if any.
struct Outcome {
  std::string kind;
  std::string what;
  bool operator==(const Outcome&) const = default;
};

/// A rank-0 partitioner failure of a type of its own.
struct PartitionerError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

template <class F>
Outcome outcome_of(F&& body) {
  try {
    body();
    return {"none", ""};
  } catch (const PartitionerError& e) {
    return {"PartitionerError", e.what()};
  } catch (const value_error& e) {
    return {"value_error", e.what()};
  } catch (const pantr::state_error& e) {
    return {"state_error", e.what()};
  } catch (const std::exception& e) {
    return {"std::exception", e.what()};
  }
}

// ---- The communicator: Self, Gathered, check_block ----------------------------------------------

TEST_CASE("mpi self communicator reports rank 0 of 1 and returns its own block with its shape") {
  const mp::SelfCommunicator c;
  CHECK(c.rank() == 0);
  CHECK(c.size() == 1);
  const std::vector<double> v{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  const std::vector<I64> w{7, 8, 9};
  const std::vector<std::vector<std::size_t>> shapes{{6}, {2, 3}, {3, 2}};
  for (const auto& shape : shapes) {
    const mp::Gathered<double> g = c.allgatherv(mp::Block<double>{v, shape});
    CHECK(g.data == v);
    CHECK(g.shapes == Shapes{shape});
  }
  const mp::Gathered<I64> gi = c.allgatherv(mp::block_1d<I64>(w));
  CHECK(gi.data == w);
  CHECK(gi.shapes == Shapes{{3}});
  // An empty block keeps the extents the sender gave it: (0,) and (0, 3) are not the same payload.
  CHECK(c.allgatherv(mp::Block<double>{{}, {0}}).shapes == Shapes{{0}});
  CHECK(c.allgatherv(mp::Block<double>{{}, {0, 3}}).shapes == Shapes{{0, 3}});
}

TEST_CASE("mpi self communicator returns the pair of owned blocks and its own reduction") {
  const mp::SelfCommunicator c;
  const std::vector<I64> idx{4, 2};
  const std::vector<double> val{0.5, 1.5, 2.5, 3.5};
  const auto [gi, gv] = c.allgather_owned(mp::block_1d<I64>(idx), mp::Block<double>{val, {2, 2}});
  CHECK(gi.data == idx);
  CHECK(gi.shapes == Shapes{{2}});
  CHECK(gv.data == val);
  CHECK(gv.shapes == Shapes{{2, 2}});

  const std::vector<double> d{1.0, 2.0, 3.0};
  const std::vector<float> f{4.0F, 5.0F};
  const std::array<std::size_t, 2> shape_d{3, 1};
  const std::array<std::size_t, 1> shape_f{2};
  const NdArray<double> rd =
      c.allreduce_sum(std::span<const double>(d), std::span<const std::size_t>(shape_d));
  CHECK(std::vector<double>(rd.data(), rd.data() + rd.size()) == d);
  CHECK(std::vector<std::size_t>(rd.shape().begin(), rd.shape().end()) ==
        std::vector<std::size_t>{3, 1});
  const NdArray<float> rf =
      c.allreduce_sum(std::span<const float>(f), std::span<const std::size_t>(shape_f));
  CHECK(std::vector<float>(rf.data(), rf.data() + rf.size()) == f);
  THB_THROWS_WITH_AS(
      c.allreduce_sum(std::span<const double>(d), std::span<const std::size_t>(shape_f)),
      "allreduce_sum: the shape does not match the data.", value_error);
}

TEST_CASE("mpi a payload with a bad shape is refused with its extent count or element count") {
  const mp::SelfCommunicator c;
  const std::vector<double> v{1.0, 2.0, 3.0, 4.0, 5.0};
  THB_THROWS_WITH_AS(c.allgatherv(mp::Block<double>{v, {}}),
                     "a communicator payload must have one or two extents; got 0.", value_error);
  THB_THROWS_WITH_AS(c.allgatherv(mp::Block<double>{v, {5, 1, 1}}),
                     "a communicator payload must have one or two extents; got 3.", value_error);
  THB_THROWS_WITH_AS(c.allgatherv(mp::Block<double>{v, {2, 3}}),
                     "a communicator payload's shape does not match its 5 elements.", value_error);
  // The product is the element count, not merely a factor of it: (5, 1) holds 5, (5, 2) and (2, 2)
  // do not (too many and too few elements for the shape).
  THB_NOTHROW(c.allgatherv(mp::Block<double>{v, {5, 1}}));
  THB_THROWS_WITH_AS(c.allgatherv(mp::Block<double>{v, {2, 2}}),
                     "a communicator payload's shape does not match its 5 elements.", value_error);
  THB_THROWS_WITH_AS(c.allgatherv(mp::Block<double>{v, {5, 2}}),
                     "a communicator payload's shape does not match its 5 elements.", value_error);
}

TEST_CASE("mpi gathered blocks of ragged shapes are addressed by their offsets") {
  // Four ranks sending (2,), (0,), (2, 3), (1, 1): 2 + 0 + 6 + 1 elements.
  const mp::Gathered<double> g =
      real_blocks({0, 1, 10, 11, 12, 13, 14, 15, 99}, {{2}, {0}, {2, 3}, {1, 1}});
  CHECK(g.count() == 4);
  CHECK(g.offset(0) == 0);
  CHECK(g.offset(1) == 2);
  CHECK(g.offset(2) == 2);
  CHECK(g.offset(3) == 8);
  CHECK(g.offset(4) == 9);
  CHECK(std::vector<double>(g.block(0).begin(), g.block(0).end()) == std::vector<double>{0, 1});
  CHECK(g.block(1).empty());
  CHECK(std::vector<double>(g.block(2).begin(), g.block(2).end()) ==
        std::vector<double>{10, 11, 12, 13, 14, 15});
  CHECK(std::vector<double>(g.block(3).begin(), g.block(3).end()) == std::vector<double>{99});
}

// ---- The C table --------------------------------------------------------------------------------

using TablePtr = std::unique_ptr<pantr_mpi_comm, void (*)(pantr_mpi_comm*)>;

std::shared_ptr<const pantr_mpi_comm> share(TablePtr table) {
  return std::shared_ptr<const pantr_mpi_comm>(std::move(table));
}

template <class C>
mp::AnyCommunicator via_table(C comm) {
  return mp::AnyCommunicator::from_table(share(mp::make_comm_table(std::move(comm))));
}

/// How a `FaultyComm` fails.
enum class Fault { value, runtime, unknown, long_message, empty_value, empty_runtime };

void raise_fault(Fault f) {
  switch (f) {
    case Fault::value:
      throw value_error("faulty comm: bad payload");
    case Fault::runtime:
      throw std::runtime_error("faulty comm: mpi down");
    case Fault::unknown:
      throw 42;
    case Fault::long_message:
      throw value_error(std::string(1000, 'x'));
    case Fault::empty_value:
      throw value_error("");
    case Fault::empty_runtime:
      throw std::runtime_error("");
  }
}

/// A communicator whose every member fails in the given way.
class FaultyComm {
 public:
  explicit FaultyComm(Fault f) : fault_(f) {}
  [[nodiscard]] int rank() const {
    raise_fault(fault_);
    return 0;
  }
  [[nodiscard]] int size() const {
    raise_fault(fault_);
    return 1;
  }
  template <class T>
  [[nodiscard]] mp::Gathered<T> allgatherv(const mp::Block<T>& b) const {
    raise_fault(fault_);
    return mp::SelfCommunicator{}.allgatherv(b);
  }
  [[nodiscard]] std::pair<mp::Gathered<I64>, mp::Gathered<double>> allgather_owned(
      const mp::Block<I64>& i, const mp::Block<double>& v) const {
    raise_fault(fault_);
    return mp::SelfCommunicator{}.allgather_owned(i, v);
  }
  template <class T>
  [[nodiscard]] NdArray<T> allreduce_sum(std::span<const T> data,
                                         std::span<const std::size_t> shape) const {
    raise_fault(fault_);
    return mp::SelfCommunicator{}.allreduce_sum(data, shape);
  }

 private:
  Fault fault_;
};
static_assert(mp::Communicator<FaultyComm>);

struct NamedCall {
  std::string name;
  std::function<void()> run;
};

/// Every member of a communicator, each called with a valid payload.
std::vector<NamedCall> all_calls(const mp::AnyCommunicator& c) {
  static const std::vector<double> d{1.0, 2.0};
  static const std::vector<float> f{1.0F, 2.0F};
  static const std::vector<I64> i{3, 4};
  static const std::array<std::size_t, 1> shape{2};
  return {
      {"rank", [&c] { static_cast<void>(c.rank()); }},
      {"size", [&c] { static_cast<void>(c.size()); }},
      {"allgatherv/double", [&c] { static_cast<void>(c.allgatherv(mp::block_1d<double>(d))); }},
      {"allgatherv/int64", [&c] { static_cast<void>(c.allgatherv(mp::block_1d<I64>(i))); }},
      {"allgather_owned",
       [&c] {
         static_cast<void>(c.allgather_owned(mp::block_1d<I64>(i), mp::block_1d<double>(d)));
       }},
      {"allreduce_sum/double",
       [&c] {
         static_cast<void>(
             c.allreduce_sum(std::span<const double>(d), std::span<const std::size_t>(shape)));
       }},
      {"allreduce_sum/float",
       [&c] {
         static_cast<void>(
             c.allreduce_sum(std::span<const float>(f), std::span<const std::size_t>(shape)));
       }},
  };
}

TEST_CASE("mpi table a value_error thrown in the communicator crosses as value_error") {
  // Bug caught: the status of a value_error mapped to another exception type (or lost, with the
  // message), which would turn a Python ValueError into a RuntimeError.
  const mp::AnyCommunicator c = via_table(FaultyComm(Fault::value));
  for (const NamedCall& call : all_calls(c)) {
    INFO(call.name);
    CHECK_THROWS_WITH_AS(call.run(), "faulty comm: bad payload", value_error);
  }
}

TEST_CASE("mpi table any other exception crosses as state_error") {
  const mp::AnyCommunicator runtime = via_table(FaultyComm(Fault::runtime));
  for (const NamedCall& call : all_calls(runtime)) {
    INFO(call.name);
    CHECK_THROWS_WITH_AS(call.run(), "faulty comm: mpi down", pantr::state_error);
  }
  const mp::AnyCommunicator unknown = via_table(FaultyComm(Fault::unknown));
  for (const NamedCall& call : all_calls(unknown)) {
    INFO(call.name);
    CHECK_THROWS_WITH_AS(call.run(), "an unknown exception", pantr::state_error);
  }
}

TEST_CASE("mpi table an empty message is replaced by the name of the failing call") {
  const std::vector<std::pair<std::string, std::string>> expected{
      {"rank", "rank failed"},
      {"size", "size failed"},
      {"allgatherv/double", "allgatherv failed"},
      {"allgatherv/int64", "allgatherv failed"},
      {"allgather_owned", "allgather_owned failed"},
      {"allreduce_sum/double", "allreduce_sum failed"},
      {"allreduce_sum/float", "allreduce_sum failed"}};
  const mp::AnyCommunicator value = via_table(FaultyComm(Fault::empty_value));
  const mp::AnyCommunicator runtime = via_table(FaultyComm(Fault::empty_runtime));
  const std::vector<NamedCall> value_calls = all_calls(value);
  const std::vector<NamedCall> runtime_calls = all_calls(runtime);
  REQUIRE(value_calls.size() == expected.size());
  for (std::size_t k = 0; k < expected.size(); ++k) {
    INFO(expected[k].first);
    REQUIRE(value_calls[k].name == expected[k].first);
    CHECK_THROWS_WITH_AS(value_calls[k].run(), expected[k].second.c_str(), value_error);
    CHECK_THROWS_WITH_AS(runtime_calls[k].run(), expected[k].second.c_str(), pantr::state_error);
  }
}

TEST_CASE("mpi table a message longer than the buffer is cut to it") {
  // Bug caught: a message written past the table's `kMessageLen` bytes. The message is cut at
  // `kMessageLen - 1` characters, the last byte being the NUL.
  const mp::AnyCommunicator c = via_table(FaultyComm(Fault::long_message));
  const std::string cut(mp::detail::kMessageLen - 1, 'x');
  for (const NamedCall& call : all_calls(c)) {
    INFO(call.name);
    CHECK_THROWS_WITH_AS(call.run(), cut.c_str(), value_error);
  }
}

/// The three routes to one communicator: itself, type-erased, and over a C table.
struct Routes {
  mt::HubComm direct;
  mp::AnyCommunicator any;
  mp::AnyCommunicator table;
  explicit Routes(const mt::HubComm& c) : direct(c), any(c), table(via_table(c)) {}
};

/// Everything a rank received from each communicator member, for comparing the routes.
struct Received {
  int rank = -1;
  int size = -1;
  mp::Gathered<double> reals;
  mp::Gathered<I64> ints;
  std::pair<mp::Gathered<I64>, mp::Gathered<double>> owned;
  std::vector<double> sum_d;
  std::vector<float> sum_f;
  std::vector<std::size_t> sum_shape;
};

template <class C>
Received exercise(const C& c) {
  // Rank r sends: a (r, 2) block of reals (rank 0: an empty (0, 2) block), r + 1 integers, and
  // the pair (r + 1 DOFs, (r + 1, 1) coefficients).
  const auto r = static_cast<std::size_t>(c.rank());
  std::vector<double> reals(r * 2);
  for (std::size_t k = 0; k < reals.size(); ++k) {
    reals[k] = static_cast<double>(10 * r + k);
  }
  const I64Vec ints(r + 1, static_cast<I64>(r));
  const std::vector<double> coeffs(r + 1, 0.5 * static_cast<double>(r + 1));
  const std::vector<double> sd{static_cast<double>(r + 1), 10.0 * static_cast<double>(r + 1)};
  const std::vector<float> sf{static_cast<float>(r + 1), 10.0F * static_cast<float>(r + 1)};
  const std::array<std::size_t, 2> shape{2, 1};
  Received out;
  out.rank = c.rank();
  out.size = c.size();
  out.reals = c.allgatherv(mp::Block<double>{reals, {r, 2}});
  out.ints = c.allgatherv(mp::block_1d<I64>(ints));
  out.owned = c.allgather_owned(mp::block_1d<I64>(ints), mp::Block<double>{coeffs, {r + 1, 1}});
  const NdArray<double> a =
      c.allreduce_sum(std::span<const double>(sd), std::span<const std::size_t>(shape));
  const NdArray<float> b =
      c.allreduce_sum(std::span<const float>(sf), std::span<const std::size_t>(shape));
  out.sum_d.assign(a.data(), a.data() + a.size());
  out.sum_f.assign(b.data(), b.data() + b.size());
  out.sum_shape.assign(a.shape().begin(), a.shape().end());
  return out;
}

void check_received(const Received& got, int rank) {
  CHECK(got.rank == rank);
  CHECK(got.size == 3);
  // Rank 0 sends no value, rank 1 two (10, 11), rank 2 four (20..23).
  CHECK(got.reals.data == std::vector<double>{10, 11, 20, 21, 22, 23});
  CHECK(got.reals.shapes == Shapes{{0, 2}, {1, 2}, {2, 2}});
  CHECK(got.ints.data == I64Vec{0, 1, 1, 2, 2, 2});
  CHECK(got.ints.shapes == Shapes{{1}, {2}, {3}});
  CHECK(got.owned.first.data == I64Vec{0, 1, 1, 2, 2, 2});
  CHECK(got.owned.first.shapes == Shapes{{1}, {2}, {3}});
  CHECK(got.owned.second.data == std::vector<double>{0.5, 1.0, 1.0, 1.5, 1.5, 1.5});
  CHECK(got.owned.second.shapes == Shapes{{1, 1}, {2, 1}, {3, 1}});
  // 1 + 2 + 3 and 10 + 20 + 30, exact in double and in float.
  CHECK(got.sum_d == std::vector<double>{6.0, 60.0});
  CHECK(got.sum_f == std::vector<float>{6.0F, 60.0F});
  CHECK(got.sum_shape == std::vector<std::size_t>{2, 1});
}

TEST_CASE("mpi table and type-erased routes give the payloads a direct communicator gives") {
  // Oracle: the payloads written out by hand in `check_received`, for the direct communicator,
  // for `AnyCommunicator(comm)` and for `from_table(make_comm_table(comm))`, on three ranks.
  struct Three {
    Received direct;
    Received any;
    Received table;
  };
  const std::vector<Three> ranks = mt::run_ranks<Three>(3, [](mt::HubComm comm) {
    const Routes routes(comm);
    return Three{exercise(routes.direct), exercise(routes.any), exercise(routes.table)};
  });
  for (int r = 0; r < 3; ++r) {
    INFO("rank " << r);
    const Three& t = ranks[static_cast<std::size_t>(r)];
    check_received(t.direct, r);
    check_received(t.any, r);
    check_received(t.table, r);
  }
}

TEST_CASE("mpi table of the self communicator keeps the extents of an empty block") {
  // Bug caught: the wire shape (ndim, extent0, extent1) dropping the second extent of a block with
  // no rows, so that a (0, 3) block came back as (0,).
  const mp::AnyCommunicator c = via_table(mp::SelfCommunicator{});
  CHECK(c.rank() == 0);
  CHECK(c.size() == 1);
  CHECK(c.allgatherv(mp::Block<double>{{}, {0}}).shapes == Shapes{{0}});
  CHECK(c.allgatherv(mp::Block<double>{{}, {0, 3}}).shapes == Shapes{{0, 3}});
  CHECK(c.allgatherv(mp::Block<I64>{{}, {0, 0}}).shapes == Shapes{{0, 0}});
  const std::vector<double> v{1, 2, 3, 4, 5, 6};
  const mp::Gathered<double> g = c.allgatherv(mp::Block<double>{v, {3, 2}});
  CHECK(g.data == v);
  CHECK(g.shapes == Shapes{{3, 2}});
}

TEST_CASE("mpi table a gather with an invalid block is refused as value_error") {
  // The communicator's answer is malformed (what a duck-typed Python communicator may return):
  // a block of three extents, and a data length that its shapes do not add up to.
  const std::vector<double> v{1.0, 2.0};
  const mp::AnyCommunicator three =
      via_table(preset(0, 1, {}, real_blocks({1, 2, 3, 4}, {{1, 2, 2}})));
  THB_THROWS_WITH_AS(three.allgatherv(mp::block_1d<double>(v)),
                     "a gathered block must have one or two extents.", value_error);
  const mp::AnyCommunicator short_data =
      via_table(preset(0, 1, {}, real_blocks({1, 2, 3}, {{2, 2}})));
  THB_THROWS_WITH_AS(short_data.allgatherv(mp::block_1d<double>(v)),
                     "a gather delivered another element count than its shapes hold.", value_error);
  // A gather of zero blocks is valid on the wire (the consumer refuses it by its count).
  const mp::AnyCommunicator none = via_table(preset(0, 1, {}, real_blocks({}, {})));
  CHECK(none.allgatherv(mp::block_1d<double>(v)).count() == 0);
}

/// A communicator that counts how many copies of it are alive.
struct Live {
  static inline std::atomic<int> count{0};
};
class LiveComm : public mp::SelfCommunicator {
 public:
  LiveComm() { ++Live::count; }
  LiveComm(const LiveComm& o) : mp::SelfCommunicator(o) { ++Live::count; }
  LiveComm(LiveComm&& o) noexcept : mp::SelfCommunicator(o) { ++Live::count; }
  LiveComm& operator=(const LiveComm&) = default;
  LiveComm& operator=(LiveComm&&) noexcept = default;
  ~LiveComm() { --Live::count; }
};
static_assert(mp::Communicator<LiveComm>);

TEST_CASE("mpi table releases its communicator once, when the last holder drops it") {
  // Bug caught: a leak (the communicator never released) or a double release (a copy of the
  // holder releasing it while another still uses it).
  REQUIRE(Live::count == 0);
  {
    std::shared_ptr<const pantr_mpi_comm> table = share(mp::make_comm_table(LiveComm{}));
    CHECK(Live::count == 1);
    std::optional<mp::AnyCommunicator> first = mp::AnyCommunicator::from_table(table);
    table.reset();  // the holder keeps the table alive
    CHECK(Live::count == 1);
    const mp::AnyCommunicator second = *first;  // copies share the table
    first.reset();
    CHECK(Live::count == 1);
    CHECK(second.size() == 1);  // still usable
  }
  CHECK(Live::count == 0);
  {
    const mp::AnyCommunicator held{LiveComm{}};
    CHECK(Live::count == 1);
    const mp::AnyCommunicator copy = held;
    CHECK(Live::count == 1);
    CHECK(copy.rank() == 0);
  }
  CHECK(Live::count == 0);
}

TEST_CASE(
    "mpi table refuses a null table, another ABI version, a smaller struct and a missing "
    "function") {
  const std::shared_ptr<const pantr_mpi_comm> good =
      share(mp::make_comm_table(mp::SelfCommunicator{}));
  const auto sz = sizeof(pantr_mpi_comm);
  // A table that shares `good`'s context but does not own it: never released.
  const auto copy_with = [&good](const std::function<void(pantr_mpi_comm&)>& edit) {
    auto* raw = new pantr_mpi_comm(*good);
    raw->release = nullptr;
    edit(*raw);
    return std::shared_ptr<const pantr_mpi_comm>(raw, [](const pantr_mpi_comm* t) { delete t; });
  };
  const std::string expects = "; this pantr expects version 1 and size " + std::to_string(sz) +
                              " (pantr and pantr-mpi must be the same version)";
  const auto refusal = [&expects](unsigned version, std::size_t size) {
    return "the communicator table has ABI version " + std::to_string(version) + " and size " +
           std::to_string(size) + expects;
  };

  THB_THROWS_WITH_AS(mp::AnyCommunicator::from_table(nullptr), "a null communicator table",
                     value_error);
  const std::string v2 = refusal(2, sz);
  THB_THROWS_WITH_AS(
      mp::AnyCommunicator::from_table(copy_with([](pantr_mpi_comm& t) { t.abi_version = 2; })),
      v2.c_str(), value_error);
  const std::string v0 = refusal(0, sz);
  THB_THROWS_WITH_AS(
      mp::AnyCommunicator::from_table(copy_with([](pantr_mpi_comm& t) { t.abi_version = 0; })),
      v0.c_str(), value_error);
  // One byte short is refused; the exact size and a larger one (a newer tail) are accepted.
  const std::string one_short = refusal(1, sz - 1);
  THB_THROWS_WITH_AS(mp::AnyCommunicator::from_table(copy_with([](pantr_mpi_comm& t) {
                       t.struct_size = static_cast<std::uint32_t>(sizeof(pantr_mpi_comm) - 1);
                     })),
                     one_short.c_str(), value_error);
  THB_NOTHROW(mp::AnyCommunicator::from_table(copy_with([](pantr_mpi_comm&) {})));
  THB_NOTHROW(mp::AnyCommunicator::from_table(copy_with([](pantr_mpi_comm& t) {
    t.struct_size = static_cast<std::uint32_t>(sizeof(pantr_mpi_comm) + 8);
  })));
  // Each of the five functions the algorithms call is required; `release` is optional.
  const std::vector<std::pair<std::string, std::function<void(pantr_mpi_comm&)>>> missing{
      {"rank", [](pantr_mpi_comm& t) { t.rank = nullptr; }},
      {"size", [](pantr_mpi_comm& t) { t.size = nullptr; }},
      {"allgatherv", [](pantr_mpi_comm& t) { t.allgatherv = nullptr; }},
      {"allgather_owned", [](pantr_mpi_comm& t) { t.allgather_owned = nullptr; }},
      {"allreduce_sum", [](pantr_mpi_comm& t) { t.allreduce_sum = nullptr; }}};
  for (const auto& entry : missing) {
    INFO(entry.first);
    THB_THROWS_WITH_AS(mp::AnyCommunicator::from_table(copy_with(entry.second)),
                       "the communicator table lacks a function", value_error);
  }
}

/// The status and message a table function ended in.
using Ended = std::pair<int, std::string>;

TEST_CASE(
    "mpi table functions refuse an element type, a shape and a count a foreign caller may pass") {
  // The C functions of a table are called by another module, with whatever it passes: each refusal
  // is a status and a message, never an exception or a read of the data.
  const TablePtr table = mp::make_comm_table(mp::SelfCommunicator{});
  std::array<char, mp::detail::kMessageLen> msg{};
  const auto accept = [](void*, std::int64_t, const std::int64_t*, const void*, std::int64_t, char*,
                         std::size_t) { return PANTR_MPI_OK; };
  pantr_mpi_sink sink{nullptr, accept};
  const std::array<double, 2> data{1.0, 2.0};
  const auto end = [&msg](int status) {
    const Ended ended{status, std::string(msg.data())};
    msg.fill('\0');
    return ended;
  };
  const auto gather = [&](int dtype, const std::array<std::int64_t, 3>& shape) {
    return end(table->allgatherv(table->ctx, dtype, data.data(), shape.data(), &sink, msg.data(),
                                 msg.size()));
  };
  const std::array<std::int64_t, 3> good{1, 2, 0};
  CHECK(gather(PANTR_MPI_FLOAT64, good) == Ended{PANTR_MPI_OK, ""});
  CHECK(gather(7, good) == Ended{PANTR_MPI_ERR_VALUE, "allgatherv: unsupported element type 7"});
  CHECK(gather(PANTR_MPI_FLOAT32, good) ==
        Ended{PANTR_MPI_ERR_VALUE, "allgatherv: unsupported element type 2"});
  const Ended bad_shape{PANTR_MPI_ERR_VALUE,
                        "a payload shape must have one or two non-negative extents."};
  for (const auto& shape :
       std::vector<std::array<std::int64_t, 3>>{{0, 2, 0}, {3, 2, 0}, {1, -1, 0}, {2, 0, -1}}) {
    INFO(shape[0] << ", " << shape[1] << ", " << shape[2]);
    CHECK(gather(PANTR_MPI_FLOAT64, shape) == bad_shape);
  }
  // The pair gather checks both shapes.
  const std::array<std::int64_t, 3> ragged{3, 2, 0};
  const std::array<std::int64_t, 3> index_shape{1, 2, 0};
  const std::array<std::int64_t, 2> index{4, 5};
  CHECK(end(table->allgather_owned(table->ctx, index.data(), ragged.data(), data.data(),
                                   index_shape.data(), &sink, &sink, msg.data(), msg.size())) ==
        bad_shape);
  CHECK(end(table->allgather_owned(table->ctx, index.data(), index_shape.data(), data.data(),
                                   ragged.data(), &sink, &sink, msg.data(), msg.size())) ==
        bad_shape);

  std::array<double, 2> sum{0.0, 0.0};
  const auto reduce = [&](int dtype, std::int64_t count) {
    return end(table->allreduce_sum(table->ctx, dtype, data.data(), sum.data(), count, msg.data(),
                                    msg.size()));
  };
  CHECK(reduce(PANTR_MPI_FLOAT64, 2) == Ended{PANTR_MPI_OK, ""});
  CHECK(sum == data);
  CHECK(reduce(PANTR_MPI_FLOAT64, 0) == Ended{PANTR_MPI_OK, ""});
  CHECK(reduce(PANTR_MPI_FLOAT64, -1) ==
        Ended{PANTR_MPI_ERR_VALUE, "allreduce_sum: a negative count"});
  CHECK(reduce(PANTR_MPI_INT64, 2) ==
        Ended{PANTR_MPI_ERR_VALUE, "allreduce_sum: unsupported element type 0"});
}

/// What a foreign sink answers.
struct SinkAnswer {
  int status;
  const char* text;
};

TEST_CASE("mpi table passes on the status and message of a sink that refuses a gather") {
  const TablePtr table = mp::make_comm_table(mp::SelfCommunicator{});
  std::array<char, mp::detail::kMessageLen> msg{};
  const auto refuse = [](void* ctx, std::int64_t, const std::int64_t*, const void*, std::int64_t,
                         char* out, std::size_t len) {
    const auto* answer = static_cast<const SinkAnswer*>(ctx);
    std::snprintf(out, len, "%s", answer->text);
    return answer->status;
  };
  const std::array<double, 2> data{1.0, 2.0};
  const std::array<std::int64_t, 3> shape{1, 2, 0};
  const auto gather = [&](const SinkAnswer& answer) {
    SinkAnswer mine = answer;
    pantr_mpi_sink sink{&mine, refuse};
    msg.fill('\0');
    const int status = table->allgatherv(table->ctx, PANTR_MPI_FLOAT64, data.data(), shape.data(),
                                         &sink, msg.data(), msg.size());
    return Ended{status, std::string(msg.data())};
  };
  CHECK(gather({PANTR_MPI_ERR_VALUE, "sink says no"}) ==
        Ended{PANTR_MPI_ERR_VALUE, "sink says no"});
  CHECK(gather({PANTR_MPI_ERR_RUNTIME, "sink broke"}) ==
        Ended{PANTR_MPI_ERR_RUNTIME, "sink broke"});
  CHECK(gather({PANTR_MPI_ERR_VALUE, ""}) == Ended{PANTR_MPI_ERR_VALUE, "a gather sink failed"});
}

// A table written by hand, as another module writes one: its gather delivers the block shape the
// context holds, and two elements.
int foreign_rank(void*, std::int32_t* out, char*, std::size_t) {
  *out = 1;
  return PANTR_MPI_OK;
}
int foreign_size(void*, std::int32_t* out, char*, std::size_t) {
  *out = 2;
  return PANTR_MPI_OK;
}
int foreign_gather(void* ctx, int, const void*, const std::int64_t*, pantr_mpi_sink* sink,
                   char* msg, std::size_t len) {
  const auto* shape = static_cast<const std::array<std::int64_t, 3>*>(ctx);
  const std::array<double, 2> data{3.0, 4.0};
  return sink->receive(sink->ctx, 1, shape->data(), data.data(), 2, msg, len);
}
int foreign_owned(void*, const std::int64_t*, const std::int64_t*, const double*,
                  const std::int64_t*, pantr_mpi_sink*, pantr_mpi_sink*, char*, std::size_t) {
  return PANTR_MPI_OK;
}
int foreign_reduce(void*, int, const void*, void*, std::int64_t, char*, std::size_t) {
  return PANTR_MPI_OK;
}

TEST_CASE("mpi table refuses a gathered block with an invalid shape or element count") {
  // Bug caught: a foreign gather delivering a block of three extents, a negative extent, or
  // another number of elements than its shapes hold, read as if it were well formed.
  std::array<std::int64_t, 3> shape{};
  pantr_mpi_comm raw{};
  raw.abi_version = PANTR_MPI_COMM_ABI_VERSION;
  raw.struct_size = static_cast<std::uint32_t>(sizeof(raw));
  raw.ctx = &shape;
  raw.rank = &foreign_rank;
  raw.size = &foreign_size;
  raw.allgatherv = &foreign_gather;
  raw.allgather_owned = &foreign_owned;
  raw.allreduce_sum = &foreign_reduce;
  raw.release = nullptr;
  const mp::AnyCommunicator c = mp::AnyCommunicator::from_table(
      std::shared_ptr<const pantr_mpi_comm>(&raw, [](const pantr_mpi_comm*) {}));
  CHECK(c.rank() == 1);
  CHECK(c.size() == 2);
  const std::vector<double> mine{0.0};
  shape = {2, 1, 2};
  const mp::Gathered<double> ok = c.allgatherv(mp::block_1d<double>(mine));
  CHECK(ok.data == std::vector<double>{3.0, 4.0});
  CHECK(ok.shapes == Shapes{{1, 2}});
  for (const auto& bad :
       std::vector<std::array<std::int64_t, 3>>{{3, 1, 2}, {0, 2, 0}, {1, -1, 0}, {2, 2, -1}}) {
    INFO(bad[0] << ", " << bad[1] << ", " << bad[2]);
    shape = bad;
    THB_THROWS_WITH_AS(c.allgatherv(mp::block_1d<double>(mine)),
                       "a gathered block has an invalid shape.", value_error);
  }
  shape = {1, 1, 0};  // one element where two were delivered
  THB_THROWS_WITH_AS(c.allgatherv(mp::block_1d<double>(mine)),
                     "a gather delivered another element count than its shapes hold.", value_error);
}

// ---- block_bounds and assemble_blocks -----------------------------------------------------------

TEST_CASE("mpi block_bounds splits a range into contiguous blocks, the first ones longer") {
  using Bounds = std::pair<std::size_t, std::size_t>;
  // Hand-computed: 10 = 4 + 3 + 3, and a range shorter than the number of parts leaves empty
  // trailing blocks at the end of the range.
  const std::vector<Bounds> ten_in_three{{0, 4}, {4, 7}, {7, 10}};
  for (std::size_t r = 0; r < 3; ++r) {
    CHECK(mp::block_bounds(10, 3, r) == ten_in_three[r]);
  }
  const std::vector<Bounds> two_in_four{{0, 1}, {1, 2}, {2, 2}, {2, 2}};
  for (std::size_t r = 0; r < 4; ++r) {
    CHECK(mp::block_bounds(2, 4, r) == two_in_four[r]);
  }
  CHECK(mp::block_bounds(0, 3, 2) == Bounds{0, 0});
  CHECK(mp::block_bounds(7, 1, 0) == Bounds{0, 7});
  CHECK(mp::block_bounds(8, 4, 3) == Bounds{6, 8});
  THB_THROWS_WITH_AS(mp::block_bounds(10, 3, 3), "block_bounds: rank 3 is outside [0, 3).",
                     value_error);
  THB_THROWS_WITH_AS(mp::block_bounds(10, 0, 0), "block_bounds: rank 0 is outside [0, 0).",
                     value_error);
}

TEST_CASE("mpi block_bounds tiles the range for every length and part count") {
  // Property: the blocks are consecutive, cover [0, n), differ in length by at most one, and the
  // longer ones come first.
  for (std::size_t parts = 1; parts <= 9; ++parts) {
    for (std::size_t n = 0; n <= 40; ++n) {
      std::size_t expected_start = 0;
      for (std::size_t r = 0; r < parts; ++r) {
        const auto [start, stop] = mp::block_bounds(n, parts, r);
        const std::size_t want = n / parts + (r < n % parts ? std::size_t{1} : std::size_t{0});
        REQUIRE(start == expected_start);
        REQUIRE(stop - start == want);
        expected_start = stop;
      }
      CHECK(expected_start == n);
    }
  }
}

TEST_CASE("mpi assemble_blocks concatenates blocks of one component count in rank order") {
  // Two components: ranks send (2, 2), an empty block, (1, 2).
  const mp::ValueField f =
      mp::assemble_blocks(real_blocks({1, 2, 3, 4, 5, 6}, {{2, 2}, {0, 2}, {1, 2}}), 3);
  CHECK(f.rank == 2);
  CHECK(f.values == std::vector<double>{1, 2, 3, 4, 5, 6});
  // Scalar: (2,), (1,), shape (0,) in the middle.
  const mp::ValueField s = mp::assemble_blocks(real_blocks({7, 8, 9}, {{2}, {0}, {1}}), 3);
  CHECK(s.rank == 1);
  CHECK(s.values == std::vector<double>{7, 8, 9});
}

TEST_CASE("mpi assemble_blocks reads the component count from the first non-empty block") {
  // The empty blocks (0,), (0, 0) and (0, 5) are skipped whatever their extents: the first
  // non-empty block, (2, 3), decides that there are three components.
  const mp::ValueField f = mp::assemble_blocks(
      real_blocks({1, 2, 3, 4, 5, 6, 7, 8, 9}, {{0}, {0, 0}, {2, 3}, {0, 5}, {1, 3}}), 3);
  CHECK(f.rank == 3);
  CHECK(f.values == std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9});
  // A (k, 1) block is one component, as a (k,) block is.
  const mp::ValueField one = mp::assemble_blocks(real_blocks({4, 5}, {{2, 1}}), 2);
  CHECK(one.rank == 1);
  CHECK(one.values == std::vector<double>{4, 5});
}

TEST_CASE("mpi assemble_blocks refuses a block with another component count than the first") {
  THB_THROWS_WITH_AS(
      mp::assemble_blocks(real_blocks({1, 2, 3, 4, 5, 6, 7, 8}, {{2, 3}, {1, 2}}), 3),
      "the value block of rank 1 has 2 components per point; expected 3.", value_error);
  // A scalar block after a vector one has one component per point, not three.
  THB_THROWS_WITH_AS(mp::assemble_blocks(real_blocks({1, 2, 3, 4, 5, 6, 7}, {{2, 3}, {1}}), 3),
                     "the value block of rank 1 has 1 components per point; expected 3.",
                     value_error);
}

TEST_CASE("mpi assemble_blocks refuses blocks that do not tile the grid") {
  const auto message = [](std::size_t covered, std::size_t total) {
    return "Per-rank value blocks do not tile the flattened grid: they cover " +
           std::to_string(covered) + " of " + std::to_string(total) +
           " points. On the interpolate path this means func returned the wrong number of values "
           "for a block; on the fit path (values_distributed=True) it means the supplied blocks do "
           "not match the contiguous C-order split of the grid (see fit_bspline_distributed's "
           "documented block layout).";
  };
  // Too few, too many, and nothing at all.
  const std::string two_of_three = message(2, 3);
  THB_THROWS_WITH_AS(mp::assemble_blocks(real_blocks({1, 2}, {{1}, {1}}), 3), two_of_three.c_str(),
                     value_error);
  const std::string four_of_three = message(4, 3);
  THB_THROWS_WITH_AS(mp::assemble_blocks(real_blocks({1, 2, 3, 4}, {{2}, {2}}), 3),
                     four_of_three.c_str(), value_error);
  const std::string none_of_three = message(0, 3);
  THB_THROWS_WITH_AS(mp::assemble_blocks(real_blocks({}, {{0}, {0, 0}}), 3), none_of_three.c_str(),
                     value_error);
  // The count is of rows, not elements: two rows of three components are 2 points.
  const std::string two_of_six = message(2, 6);
  THB_THROWS_WITH_AS(mp::assemble_blocks(real_blocks({1, 2, 3, 4, 5, 6}, {{2, 3}}), 6),
                     two_of_six.c_str(), value_error);
  // An empty grid is tiled by empty blocks, and is a scalar field.
  const mp::ValueField none = mp::assemble_blocks(real_blocks({}, {{0}, {0, 0}}), 0);
  CHECK(none.values.empty());
  CHECK(none.rank == 1);
}

TEST_CASE("mpi collocate_values_distributed refuses a gather of another number of blocks") {
  // One block for two ranks: reported before the blocks are used.
  const std::vector<double> nodes{0.0, 0.5, 1.0};
  const bs::BsplineSpace1D<double> s1(std::vector<double>{0, 0, 0, 1, 1, 1}, 2);
  const bs::BsplineSpace<double> space({s1});
  const std::vector<std::span<const double>> spans{std::span<const double>(nodes)};
  const bs::TensorCollocation<double> system(space, spans);
  const mt::PresetComm comm = preset(0, 2, {}, real_blocks({1, 2, 3}, {{3}}));
  const std::vector<double> mine{1, 2, 3};
  THB_THROWS_WITH_AS(
      (mp::collocate_values_distributed<double, double>(comm, system, mp::block_1d<double>(mine),
                                                        1e-12)),
      "allgather returned 1 entries but comm.size is 2; the communicator's allgather is "
      "inconsistent.",
      value_error);
}

// ---- Spaces, partitions, fields -----------------------------------------------------------------

struct TensorCase {
  const char* name;
  std::vector<std::vector<double>> knots;
  std::vector<int> degrees;
};

/// A degree (2, 3) space of 5 x 5 cells with a double interior knot (a C0 line), a degree
/// (1, 2, 1) space of 3 x 2 x 4 cells, and a cubic of 5 cells: windows of two, three and one
/// directions.
const std::vector<TensorCase>& tensor_cases() {
  static const std::vector<TensorCase> cases{
      {"2d (2,3)",
       {{0.0, 0.0, 0.0, 0.3, 0.55, 0.55, 0.8, 0.9, 1.0, 1.0, 1.0},
        {0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.0, 1.0, 1.0}},
       {2, 3}},
      {"3d (1,2,1)",
       {{0.0, 0.0, 0.4, 0.7, 1.0, 1.0},
        {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0},
        {0.0, 0.0, 0.2, 0.6, 0.9, 1.0, 1.0}},
       {1, 2, 1}},
      {"1d cubic", {{0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.6, 0.9, 1.0, 1.0, 1.0, 1.0}}, {3}}};
  return cases;
}

/// Uniform meshes of low degree, whose mass matrices are well conditioned: the float L2 tests run
/// on them, as the forward bound of `l2_tolerance` grows with the conditioning of the mass matrix
/// and is no bound at all (the size of the coefficients) on the meshes above in float.
const std::vector<TensorCase>& uniform_cases() {
  static const std::vector<TensorCase> cases{
      {"2d uniform (2,2)",
       {{0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0},
        {0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0}},
       {2, 2}},
      {"3d uniform (1,1,1)",
       {{0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0},
        {0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0},
        {0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0}},
       {1, 1, 1}},
      {"1d uniform quadratic", {{0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0}}, {2}}};
  return cases;
}

template <class T>
bs::BsplineSpace<T> make_space(const TensorCase& c) {
  std::vector<bs::BsplineSpace1D<T>> dirs;
  for (std::size_t k = 0; k < c.knots.size(); ++k) {
    std::vector<T> narrowed;
    for (const double v : c.knots[k]) {
      narrowed.push_back(static_cast<T>(v));
    }
    dirs.emplace_back(std::span<const T>(narrowed), c.degrees[k]);
  }
  return bs::BsplineSpace<T>(std::move(dirs));
}

/// The breakpoints of each direction, in double.
template <class T>
std::vector<std::vector<double>> breakpoints_of(const bs::BsplineSpace<T>& space) {
  std::vector<std::vector<double>> bp;
  for (const bs::BsplineSpace1D<T>& s : space.spaces()) {
    std::vector<double> b;
    for (const T v : s.unique_knots_and_multiplicity(true).first) {
      b.push_back(static_cast<double>(v));
    }
    bp.push_back(std::move(b));
  }
  return bp;
}

template <class T>
gr::Partition rcb_of(const bs::BsplineSpace<T>& space, int n_parts) {
  return gr::partition_grid(gr::TensorProductGrid(breakpoints_of(space)), n_parts,
                            gr::PartitionBackend::rcb);
}

template <class T>
I64 count_cells(const bs::BsplineSpace<T>& space) {
  return mp::global_cell_count(space);
}

template <class T>
I64 count_dofs(const bs::BsplineSpace<T>& space) {
  I64 n = 1;
  for (const bs::BsplineSpace1D<T>& s : space.spaces()) {
    n *= static_cast<I64>(s.num_basis());
  }
  return n;
}

Owners owners_of(const gr::Partition& p) {
  return Owners(p.cell_owner().begin(), p.cell_owner().end());
}

/// Rank-dependent on purpose: a rank that computed its own partition would hold another one.
gr::Partition rank_dependent(int rank, std::int64_t n_parts, std::size_t n_cells) {
  Owners owner(n_cells);
  for (std::size_t c = 0; c < n_cells; ++c) {
    owner[c] = static_cast<std::int32_t>((c + static_cast<std::size_t>(rank)) %
                                         static_cast<std::size_t>(n_parts));
  }
  return gr::Partition(std::move(owner), n_parts);
}

/// Component `c` of the test field at a point of `dim` coordinates: smooth, not a polynomial.
double field(const double* x, std::size_t dim, std::size_t c) {
  double s = 0.0;
  for (std::size_t k = 0; k < dim; ++k) {
    s += (1.0 + 0.5 * static_cast<double>(k)) *
         std::sin(1.7 * x[k] + 0.3 * static_cast<double>(k + 1));
  }
  if (c == 0) {
    return s;
  }
  return 1.0 + x[0] * x[dim - 1] + 0.25 * s * s;
}

/// The field at `n` points (row-major `(n, dim)`, in `T`), as `(n,)` or `(n, rank)` doubles.
template <class T>
NdArray<double> field_values(const T* pts, std::size_t n, std::size_t dim, std::size_t rank) {
  NdArray<double> out(rank == 1 ? std::vector<std::size_t>{n} : std::vector<std::size_t>{n, rank});
  std::vector<double> x(dim);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < dim; ++k) {
      x[k] = static_cast<double>(pts[i * dim + k]);
    }
    for (std::size_t c = 0; c < rank; ++c) {
      out.data()[i * rank + c] = field(x.data(), dim, c);
    }
  }
  return out;
}

/// The field on a lattice, rounded to `T`: what a serial `interpolate` or `l2_project` samples.
template <class T>
NdArray<T> lattice_values(const pantr::quad::PointsLattice<T>& lattice, std::size_t rank) {
  const NdArray<T> pts = lattice.all_points(pantr::Order::C);
  const NdArray<double> v = field_values<T>(pts.data(), pts.shape()[0], pts.shape()[1], rank);
  NdArray<T> out(std::vector<std::size_t>(v.shape().begin(), v.shape().end()),
                 pantr::for_overwrite);
  for (std::size_t i = 0; i < v.size(); ++i) {
    out.data()[i] = static_cast<T>(v.data()[i]);
  }
  return out;
}

/// The field at the points of a block, as the distributed algorithms' callables return it.
NdArray<double> block_values(pantr::mdview<const double, 2> pts, std::size_t rank) {
  return field_values<double>(pts.data_handle(), pts.extent(0), pts.extent(1), rank);
}

template <class Space>
I64Vec owned_dofs_of(const mp::DistributedSpace<Space>& ds) {
  I64Vec out;
  if (ds.local()) {
    const auto& local = *ds.local();
    for (std::size_t i = 0; i < local.local_to_global_dof.size(); ++i) {
      if (local.owned_dof_mask[i] != 0) {
        out.push_back(local.local_to_global_dof[i]);
      }
    }
  }
  return out;
}

const std::array<int, 3> kRankCounts{2, 3, 4};

// ---- Partition broadcast ------------------------------------------------------------------------

/// Forwards to a communicator and records the integer payloads it is asked to gather.
template <class C>
class Recorder {
 public:
  struct Sent {
    std::vector<I64Vec> payloads;
    Shapes shapes;
  };
  explicit Recorder(C inner) : inner_(std::move(inner)), sent_(std::make_shared<Sent>()) {}
  [[nodiscard]] int rank() const { return inner_.rank(); }
  [[nodiscard]] int size() const { return inner_.size(); }
  [[nodiscard]] mp::Gathered<I64> allgatherv(const mp::Block<I64>& b) const {
    sent_->payloads.emplace_back(b.data.begin(), b.data.end());
    sent_->shapes.push_back(b.shape);
    return inner_.allgatherv(b);
  }
  [[nodiscard]] mp::Gathered<double> allgatherv(const mp::Block<double>& b) const {
    return inner_.allgatherv(b);
  }
  [[nodiscard]] std::pair<mp::Gathered<I64>, mp::Gathered<double>> allgather_owned(
      const mp::Block<I64>& i, const mp::Block<double>& v) const {
    return inner_.allgather_owned(i, v);
  }
  template <class T>
  [[nodiscard]] NdArray<T> allreduce_sum(std::span<const T> data,
                                         std::span<const std::size_t> shape) const {
    return inner_.allreduce_sum(data, shape);
  }
  [[nodiscard]] const Sent& sent() const { return *sent_; }

 private:
  C inner_;
  std::shared_ptr<Sent> sent_;
};
static_assert(mp::Communicator<Recorder<mt::HubComm>>);

struct BroadcastRun {
  Outcome outcome;
  gr::Partition partition;
  std::size_t partitioner_calls;
  std::vector<I64Vec> sent;
  Shapes sent_shapes;
};

TEST_CASE(
    "mpi broadcast_partition gives every rank rank 0's partition and only rank 0 computes it") {
  // Bug caught: a rank using the partition its own partitioner computed (here the owner of cell c
  // is (c + rank) % n, so any rank computing it for itself differs from rank 0), or every rank
  // calling the partitioner.
  for (const int n : kRankCounts) {
    INFO("ranks " << n);
    std::array<std::atomic<int>, 4> calls{};
    const std::vector<BroadcastRun> runs = mt::run_ranks<BroadcastRun>(n, [&](mt::HubComm hub) {
      const Recorder<mt::HubComm> comm(hub);
      const auto r = static_cast<std::size_t>(comm.rank());
      std::optional<gr::Partition> got;
      const Outcome out = outcome_of([&] {
        got.emplace(mp::broadcast_partition(comm, [&]() -> gr::Partition {
          ++calls[r];
          return rank_dependent(comm.rank(), n, 11);
        }));
      });
      return BroadcastRun{out, got ? *got : rank_dependent(0, 1, 1),
                          static_cast<std::size_t>(calls[r].load()), comm.sent().payloads,
                          comm.sent().shapes};
    });
    const gr::Partition want = rank_dependent(0, n, 11);
    // The oracle can fail: another rank's partition is not rank 0's.
    REQUIRE(owners_of(rank_dependent(1, n, 11)) != owners_of(want));
    for (std::size_t r = 0; r < runs.size(); ++r) {
      INFO("rank " << r);
      CHECK(runs[r].outcome == Outcome{"none", ""});
      CHECK(owners_of(runs[r].partition) == owners_of(want));
      CHECK(runs[r].partition.n_parts() == n);
      CHECK(runs[r].partitioner_calls == (r == 0 ? 1U : 0U));
      // The wire format: rank 0 sends [1, n_parts, owners...]; the others send nothing.
      REQUIRE(runs[r].sent.size() == 1);
      if (r == 0) {
        I64Vec payload{1, n};
        payload.insert(payload.end(), want.cell_owner().begin(), want.cell_owner().end());
        CHECK(runs[r].sent[0] == payload);
        CHECK(runs[r].sent_shapes == Shapes{{13}});
      } else {
        CHECK(runs[r].sent[0].empty());
        CHECK(runs[r].sent_shapes == Shapes{{0}});
      }
    }
  }
}

TEST_CASE("mpi broadcast_partition at one rank calls the partitioner and gathers nothing") {
  const CountingComm comm(0, 1);
  int calls = 0;
  const gr::Partition p = mp::broadcast_partition(comm, [&]() -> gr::Partition {
    ++calls;
    return rank_dependent(0, 1, 7);
  });
  CHECK(calls == 1);
  CHECK(comm.gathers() == 0);
  CHECK(owners_of(p) == owners_of(rank_dependent(0, 1, 7)));
  // A failure at one rank is the partitioner's own, with nobody to tell.
  THB_THROWS_WITH_AS(
      mp::broadcast_partition(comm, [&]() -> gr::Partition { throw value_error("bad backend"); }),
      "bad backend", value_error);
  CHECK(comm.gathers() == 0);
}

TEST_CASE(
    "mpi broadcast_partition rank 0 rethrows its own exception and the others get value_error") {
  // Bug caught: rank 0 skipping the collective after a failure (the others wait forever, here
  // the thread hub would never return), the others hanging on rank 0's exception, or every rank
  // seeing the generic message.
  const std::string generic =
      "rank 0 could not compute the partition (its exception is raised there).";
  for (const int n : kRankCounts) {
    INFO("ranks " << n);
    std::array<std::atomic<int>, 4> calls{};
    const std::vector<BroadcastRun> runs = mt::run_ranks<BroadcastRun>(n, [&](mt::HubComm hub) {
      const Recorder<mt::HubComm> comm(hub);
      const auto r = static_cast<std::size_t>(comm.rank());
      const Outcome out = outcome_of([&] {
        static_cast<void>(mp::broadcast_partition(comm, [&]() -> gr::Partition {
          ++calls[r];
          throw PartitionerError("boom");
        }));
      });
      return BroadcastRun{out, rank_dependent(0, 1, 1), static_cast<std::size_t>(calls[r].load()),
                          comm.sent().payloads, comm.sent().shapes};
    });
    for (std::size_t r = 0; r < runs.size(); ++r) {
      INFO("rank " << r);
      CHECK(runs[r].outcome ==
            (r == 0 ? Outcome{"PartitionerError", "boom"} : Outcome{"value_error", generic}));
      CHECK(runs[r].partitioner_calls == (r == 0 ? 1U : 0U));
      REQUIRE(runs[r].sent.size() == 1);
      // The failure marker: rank 0 sends [0], the others nothing.
      CHECK(runs[r].sent[0] == (r == 0 ? I64Vec{0} : I64Vec{}));
    }
  }
  // Rank 0's own exception keeps its type and message whatever the type.
  const std::vector<BroadcastRun> runs = mt::run_ranks<BroadcastRun>(2, [&](mt::HubComm hub) {
    const Recorder<mt::HubComm> comm(hub);
    const Outcome out = outcome_of([&] {
      static_cast<void>(mp::broadcast_partition(
          comm, [&]() -> gr::Partition { throw value_error("n_parts must be >= 1; got 0."); }));
    });
    return BroadcastRun{out, rank_dependent(0, 1, 1), 0, {}, {}};
  });
  CHECK(runs[0].outcome == Outcome{"value_error", "n_parts must be >= 1; got 0."});
  CHECK(runs[1].outcome == Outcome{"value_error", generic});
}

TEST_CASE(
    "mpi broadcast_partition reads rank 0's payload as it is written, and refuses a malformed "
    "one") {
  struct Row {
    const char* name;
    I64Vec root;
    Outcome expected;  // kind "none" means accepted
    Owners owners;     // the partition then built
    std::int64_t n_parts;
  };
  const Outcome failed{"value_error",
                       "rank 0 could not compute the partition (its exception is raised there)."};
  const Outcome malformed{"value_error", "rank 0's partition payload is malformed."};
  const Outcome wide{"value_error", "rank 0's partition payload holds an owner outside int32."};
  const auto range = [](const char* got, std::int64_t n) {
    return Outcome{"value_error", "cell_owner values must lie in [-1, " + std::to_string(n) +
                                      "); got range " + got + "."};
  };
  constexpr std::int64_t kMax = std::numeric_limits<std::int32_t>::max();
  constexpr std::int64_t kMin = std::numeric_limits<std::int32_t>::min();
  const Outcome ok{"none", ""};
  const std::vector<Row> rows{
      // accepted
      {"owners", {1, 2, 0, 1, 0}, ok, {0, 1, 0}, 2},
      {"inactive cell", {1, 2, -1, 0, 1}, ok, {-1, 0, 1}, 2},
      {"no cell", {1, 2}, ok, {}, 2},
      {"largest part count and owner",
       {1, kMax + 1, kMax},
       ok,
       {static_cast<std::int32_t>(kMax)},
       kMax + 1},
      {"smallest owner is -1", {1, 2, -1}, ok, {-1}, 2},
      // rank 0 failed or sent nothing
      {"failure marker", {0}, failed, {}, 0},
      {"empty payload", {}, failed, {}, 0},
      {"first entry not 1", {2, 2, 0, 1}, failed, {}, 0},
      {"no part count", {1}, malformed, {}, 0},
      // the int32 range of the owners, at its edges
      {"owner above int32", {1, 2, 3000000000}, wide, {}, 0},
      {"owner above int32 by one", {1, kMax + 2, kMax + 1}, wide, {}, 0},
      {"owner below int32", {1, 2, -3000000000}, wide, {}, 0},
      {"owner below int32 by one", {1, 2, kMin - 1}, wide, {}, 0},
      // inside int32 but not a part: the partition refuses it
      {"owner int32 max", {1, 2, kMax}, range("[2147483647, 2147483647]", 2), {}, 0},
      {"owner int32 min", {1, 2, kMin}, range("[-2147483648, -2147483648]", 2), {}, 0},
      {"owner equal to the part count", {1, 2, 0, 2}, range("[0, 2]", 2), {}, 0},
      {"owner -2", {1, 2, -2, 0}, range("[-2, 0]", 2), {}, 0},
      {"no part", {1, 0}, Outcome{"value_error", "n_parts must be >= 1; got 0."}, {}, 0},
      {"negative part count",
       {1, -1},
       Outcome{"value_error", "n_parts must be >= 1; got -1."},
       {},
       0}};
  for (const Row& row : rows) {
    INFO(row.name);
    // Rank 1 of 2: it never computes, and reads block 0 whatever the later blocks hold.
    const mt::PresetComm comm = preset(1, 2, int_blocks({row.root, {0}}));
    std::optional<gr::Partition> got;
    const Outcome out = outcome_of([&] {
      got.emplace(mp::broadcast_partition(comm, [&]() -> gr::Partition {
        throw PartitionerError("rank 1 called the partitioner");
      }));
    });
    CHECK(out == row.expected);
    if (row.expected.kind == "none") {
      REQUIRE(got.has_value());
      CHECK(owners_of(*got) == row.owners);
      CHECK(got->n_parts() == row.n_parts);
    }
  }
}

TEST_CASE("mpi broadcast_partition refuses a gather of another number of blocks") {
  const mt::PresetComm comm = preset(1, 2, int_blocks({{1, 2, 0, 1}, {}, {}}));
  THB_THROWS_WITH_AS(
      mp::broadcast_partition(comm, [&]() -> gr::Partition { return rank_dependent(0, 2, 4); }),
      "allgather returned 3 entries but comm.size is 2; the communicator's allgather is "
      "inconsistent.",
      value_error);
}

// ---- DistributedSpace ---------------------------------------------------------------------------

TEST_CASE("mpi distributed space windows the global space and the owned DOFs tile it") {
  for (const TensorCase& c : tensor_cases()) {
    INFO(c.name);
    const bs::BsplineSpace<double> space = make_space<double>(c);
    const gr::Partition part = rcb_of(space, 3);
    std::vector<int> dof_owners(static_cast<std::size_t>(count_dofs(space)), 0);
    I64 cells_seen = 0;
    for (int r = 0; r < 3; ++r) {
      INFO("rank " << r);
      const mp::DistributedSpace<bs::BsplineSpace<double>> ds(space, part, r, 3);
      CHECK(ds.rank() == r);
      CHECK(ds.n_parts() == 3);
      CHECK(owners_of(ds.partition()) == owners_of(part));
      CHECK(ds.owned_cells() == part.owned_cells(r));
      CHECK(ds.owns_cells());
      REQUIRE(ds.local().has_value());
      const auto& local = *ds.local();
      CHECK(local.n_global_cells == count_cells(space));
      CHECK(local.n_global_dofs == count_dofs(space));
      // The cells the window marks as owned are the partition's, found by the global ids.
      I64Vec owned_by_mask;
      for (std::size_t i = 0; i < local.owned_cell_mask.size(); ++i) {
        if (local.owned_cell_mask[i] != 0) {
          owned_by_mask.push_back(local.local_to_global_cell[i]);
        }
      }
      std::sort(owned_by_mask.begin(), owned_by_mask.end());
      CHECK(owned_by_mask == ds.owned_cells());
      cells_seen += static_cast<I64>(ds.owned_cells().size());
      for (const I64 g : owned_dofs_of(ds)) {
        ++dof_owners[static_cast<std::size_t>(g)];
      }
    }
    CHECK(cells_seen == count_cells(space));
    CHECK(std::all_of(dof_owners.begin(), dof_owners.end(), [](int k) { return k == 1; }));
  }
}

TEST_CASE("mpi distributed space of a rank that owns no cell has no local space") {
  const bs::BsplineSpace<double> space = make_space<double>(tensor_cases()[0]);
  Owners owner(25, 0);
  std::fill(owner.begin() + 12, owner.end(), 1);
  const gr::Partition part(owner, 3);  // rank 2 owns nothing
  const mp::DistributedSpace<bs::BsplineSpace<double>> idle(space, part, 2, 3);
  CHECK_FALSE(idle.owns_cells());
  CHECK_FALSE(idle.local().has_value());
  CHECK(idle.owned_cells().empty());
  CHECK(idle.n_parts() == 3);
  const mp::DistributedSpace<bs::BsplineSpace<double>> busy(space, part, 1, 3);
  CHECK(busy.owns_cells());
  CHECK(busy.owned_cells().size() == 13);
}

TEST_CASE("mpi distributed space refuses a size, a cell count, a rank and a periodic direction") {
  const bs::BsplineSpace<double> space = make_space<double>(tensor_cases()[0]);
  const gr::Partition part = rcb_of(space, 3);
  using Ds = mp::DistributedSpace<bs::BsplineSpace<double>>;
  THB_THROWS_WITH_AS(Ds(space, part, 0, 4), "comm.size (4) must equal partition.n_parts (3).",
                     value_error);
  THB_THROWS_WITH_AS(Ds(space, part, 0, 2), "comm.size (2) must equal partition.n_parts (3).",
                     value_error);
  THB_THROWS_WITH_AS(Ds(space, gr::Partition(Owners(24, 0), 1), 0, 1),
                     "partition has 24 cells; expected 25 (the global space's cell count).",
                     value_error);
  THB_THROWS_WITH_AS(Ds(space, part, 3, 3), "rank must be in [0, 3); got 3.", value_error);
  THB_THROWS_WITH_AS(Ds(space, part, -1, 3), "rank must be in [0, 3); got -1.", value_error);
  // The edges of the rank range are accepted.
  THB_NOTHROW(Ds(space, part, 0, 3));
  THB_NOTHROW(Ds(space, part, 2, 3));
  // A rank that owns no cell builds no local space, so the constructor's own checks are the only
  // ones it meets: a rank that accepted what the others refuse would go on into a collective alone.
  THB_THROWS_WITH_AS(Ds(space, gr::Partition(Owners(24, 0), 3), 2, 3),
                     "partition has 24 cells; expected 25 (the global space's cell count).",
                     value_error);

  const std::vector<double> knots = bs::uniform_periodic_knots<double>(4, 2, 1, 0.0, 1.0);
  const bs::BsplineSpace<double> periodic({bs::BsplineSpace1D<double>(knots, 2, true)});
  THB_THROWS_WITH_AS(Ds(periodic, gr::Partition(Owners{0, 0, 1, 1}, 2), 0, 2),
                     "periodic B-spline spaces are not supported.", value_error);
  THB_THROWS_WITH_AS(Ds(periodic, gr::Partition(Owners{0, 0, 1, 1}, 3), 2, 3),
                     "periodic B-spline spaces are not supported.", value_error);
}

TEST_CASE("mpi distribute takes the rank and size from the communicator") {
  const bs::BsplineSpace<double> space = make_space<double>(tensor_cases()[2]);
  const gr::Partition part = rcb_of(space, 2);
  const auto ds = mp::distribute(space, part, preset(1, 2));
  CHECK(ds.rank() == 1);
  CHECK(ds.n_parts() == 2);
  CHECK(ds.owned_cells() == part.owned_cells(1));
  THB_THROWS_WITH_AS(mp::distribute(space, part, preset(0, 3)),
                     "comm.size (3) must equal partition.n_parts (2).", value_error);
}

TEST_CASE("mpi create_distributed_space builds every rank's view of rank 0's partition") {
  for (const TensorCase& c : tensor_cases()) {
    INFO(c.name);
    const bs::BsplineSpace<double> space = make_space<double>(c);
    const auto n_cells = static_cast<std::size_t>(count_cells(space));
    for (const int n : kRankCounts) {
      INFO("ranks " << n);
      std::array<std::atomic<int>, 4> calls{};
      std::array<std::atomic<std::int64_t>, 4> asked{};
      std::array<std::atomic<std::int64_t>, 4> cells_seen{};
      struct View {
        Owners owners;
        std::vector<std::int64_t> owned_cells;
        int rank;
      };
      const auto views = mt::run_ranks<View>(n, [&](mt::HubComm comm) {
        const auto r = static_cast<std::size_t>(comm.rank());
        const auto ds = mp::create_distributed_space(
            space, comm, [&](const bs::BsplineSpace<double>& s, std::int64_t n_parts) {
              ++calls[r];
              asked[r] = n_parts;
              cells_seen[r] = count_cells(s);
              return rank_dependent(comm.rank(), n_parts, n_cells);
            });
        return View{owners_of(ds.partition()), ds.owned_cells(), ds.rank()};
      });
      const gr::Partition want = rank_dependent(0, n, n_cells);
      for (std::size_t r = 0; r < views.size(); ++r) {
        INFO("rank " << r);
        CHECK(views[r].owners == owners_of(want));
        CHECK(views[r].owned_cells == want.owned_cells(static_cast<std::int64_t>(r)));
        CHECK(views[r].rank == static_cast<int>(r));
        CHECK(calls[r].load() == (r == 0 ? 1 : 0));
      }
      // The partitioner got the part count and the global space.
      CHECK(asked[0].load() == n);
      CHECK(cells_seen[0].load() == count_cells(space));
    }
  }
}

TEST_CASE("mpi create_distributed_space at one rank, and when the partitioner fails on rank 0") {
  const bs::BsplineSpace<double> space = make_space<double>(tensor_cases()[1]);
  const auto n_cells = static_cast<std::size_t>(count_cells(space));
  const CountingComm one(0, 1);
  const auto ds = mp::create_distributed_space(
      space, one, [&](const bs::BsplineSpace<double>&, std::int64_t n) {
        return rank_dependent(0, n, n_cells);
      });
  CHECK(one.gathers() == 0);
  CHECK(ds.owned_cells().size() == n_cells);

  const std::vector<Outcome> outcomes = mt::run_ranks<Outcome>(3, [&](mt::HubComm comm) {
    return outcome_of([&] {
      static_cast<void>(mp::create_distributed_space(
          space, comm, [](const bs::BsplineSpace<double>&, std::int64_t) -> gr::Partition {
            throw PartitionerError("no backend");
          }));
    });
  });
  CHECK(outcomes[0] == Outcome{"PartitionerError", "no backend"});
  const Outcome generic{"value_error",
                        "rank 0 could not compute the partition (its exception is raised there)."};
  CHECK(outcomes[1] == generic);
  CHECK(outcomes[2] == generic);
}

// ---- partition_from_owner -----------------------------------------------------------------------

TEST_CASE("mpi partition_from_owner assembles the same owner array on every rank") {
  // Hand-computed: cells 8 and 9 are sent by nobody (-1); ids need not be sorted, a rank may send
  // none, and a repeat inside one rank's block is accepted.
  const std::vector<std::pair<std::vector<I64Vec>, Owners>> cases{
      {{{0, 3, 6}, {1, 4, 7}, {2, 5}}, {0, 1, 2, 0, 1, 2, 0, 1, -1, -1}},
      {{{7, 0, 3}, {}, {9}}, {0, -1, -1, 0, -1, -1, -1, 0, -1, 2}},
      {{{2, 2, 5}, {1}, {}}, {-1, 1, 0, -1, -1, 0, -1, -1, -1, -1}}};
  for (const auto& entry : cases) {
    const std::vector<I64Vec>& blocks = entry.first;
    const Owners& want = entry.second;
    const std::vector<gr::Partition> parts = mt::run_ranks<gr::Partition>(3, [&](mt::HubComm comm) {
      const I64Vec& mine = blocks[static_cast<std::size_t>(comm.rank())];
      return mp::partition_from_owner(comm, std::span<const I64>(mine), 10);
    });
    for (const gr::Partition& p : parts) {
      CHECK(owners_of(p) == want);
      CHECK(p.n_parts() == 3);
    }
  }
}

TEST_CASE("mpi partition_from_owner never reads the rank") {
  // Bug caught: reading `comm.rank()`, which a duck-typed communicator supplies only when asked.
  const NoRankComm comm(2, int_blocks({{0, 1}, {2}}));
  const gr::Partition p = mp::partition_from_owner(comm, std::span<const I64>{}, 4);
  CHECK(owners_of(p) == Owners{0, 0, 1, -1});
}

TEST_CASE("mpi partition_from_owner refuses n_cells below 1 before the collective") {
  const CountingComm comm(0, 1);
  const I64Vec ids{0};
  THB_THROWS_WITH_AS(mp::partition_from_owner(comm, std::span<const I64>(ids), 0),
                     "n_cells must be >= 1; got 0.", value_error);
  THB_THROWS_WITH_AS(mp::partition_from_owner(comm, std::span<const I64>(ids), -3),
                     "n_cells must be >= 1; got -3.", value_error);
  CHECK(comm.gathers() == 0);
  CHECK(owners_of(mp::partition_from_owner(comm, std::span<const I64>(ids), 1)) == Owners{0});
}

TEST_CASE("mpi partition_from_owner refuses the payloads the Python implementation refused") {
  const I64Vec none;
  const auto run = [&](int size, const mp::Gathered<I64>& g, std::int64_t n_cells) {
    return mp::partition_from_owner(preset(0, size, g), std::span<const I64>(none), n_cells);
  };
  THB_THROWS_WITH_AS(run(3, int_blocks({{0}, {1}}), 10),
                     "allgather returned 2 entries but comm.size is 3; the communicator's "
                     "allgather is inconsistent.",
                     value_error);
  // A block that is not one-dimensional, in rank order: rank 1 is reported, not rank 2.
  mp::Gathered<I64> two_d = int_blocks({{0}, {}, {}});
  two_d.data = {0, 1, 2, 3, 4, 5};
  two_d.shapes = {{1}, {2, 3}, {0, 2}};
  THB_THROWS_WITH_AS(run(3, two_d, 10),
                     "allgather result for rank 1 is not a 1-D array (got shape (2, 3)).",
                     value_error);
  mp::Gathered<I64> zero_d = int_blocks({{0}, {}});
  zero_d.shapes = {{1}, {}};
  zero_d.data = {0, 7};
  THB_THROWS_WITH_AS(run(2, zero_d, 10),
                     "allgather result for rank 1 is not a 1-D array (got shape ()).", value_error);
  // Ids outside [0, n_cells), at both edges.
  THB_THROWS_WITH_AS(run(2, int_blocks({{0}, {3, 12}}), 10),
                     "gathered cell ids from rank 1 are outside [0, 10): got range [3, 12].",
                     value_error);
  THB_THROWS_WITH_AS(run(2, int_blocks({{-1, 4}, {}}), 10),
                     "gathered cell ids from rank 0 are outside [0, 10): got range [-1, 4].",
                     value_error);
  THB_THROWS_WITH_AS(run(2, int_blocks({{}, {10}}), 10),
                     "gathered cell ids from rank 1 are outside [0, 10): got range [10, 10].",
                     value_error);
  THB_NOTHROW(run(2, int_blocks({{0}, {9}}), 10));
  // A cell sent by two ranks: the first conflict in the order of the later rank's block, naming
  // the earlier rank that owned it.
  THB_THROWS_WITH_AS(run(2, int_blocks({{4, 7}, {7, 4}}), 10),
                     "inconsistent partition: cell 7 is claimed by both rank 0 and rank 1.",
                     value_error);
  THB_THROWS_WITH_AS(run(3, int_blocks({{1}, {2}, {1}}), 10),
                     "inconsistent partition: cell 1 is claimed by both rank 0 and rank 2.",
                     value_error);
}

// ---- owned_coefficients and gather_owned_coefficients -------------------------------------------

TEST_CASE("mpi owned_coefficients keeps the owned DOFs' rows in local order") {
  const I64Vec l2g{5, 2, 9, 0};
  const std::vector<std::uint8_t> mask{1, 0, 1, 1};
  const std::vector<double> coeffs{10, 11, 20, 21, 30, 31, 40, 41};
  const mp::OwnedCoefficients o = mp::owned_coefficients(l2g, mask, coeffs, 2, false);
  CHECK(o.dofs == I64Vec{5, 9, 0});
  CHECK(o.coeffs == std::vector<double>{10, 11, 30, 31, 40, 41});
  CHECK(o.rank == 2);
  CHECK_FALSE(o.scalar);
  const std::vector<double> column{1, 2, 3, 4};
  const mp::OwnedCoefficients s = mp::owned_coefficients(l2g, mask, column, 1, true);
  CHECK(s.dofs == I64Vec{5, 9, 0});
  CHECK(s.coeffs == std::vector<double>{1, 3, 4});
  CHECK(s.scalar);
  // A mask that keeps nothing keeps nothing.
  const std::vector<std::uint8_t> none(4, 0);
  CHECK(mp::owned_coefficients(l2g, none, coeffs, 2, false).dofs.empty());
  // Any non-zero flag keeps the DOF.
  const std::vector<std::uint8_t> loud{0, 7, 0, 0};
  CHECK(mp::owned_coefficients(l2g, loud, coeffs, 2, false).dofs == I64Vec{2});
}

TEST_CASE("mpi owned_coefficients refuses maps and coefficients that disagree") {
  const I64Vec l2g{5, 2, 9, 0};
  const std::vector<std::uint8_t> mask{1, 0, 1, 1};
  const std::vector<double> coeffs(8, 1.0);
  THB_THROWS_WITH_AS(
      mp::owned_coefficients(l2g, std::span<const std::uint8_t>(mask).first(3), coeffs, 2, false),
      "owned_coefficients: 4 local DOFs, 3 mask entries and 8 coefficients disagree.", value_error);
  THB_THROWS_WITH_AS(
      mp::owned_coefficients(l2g, mask, std::span<const double>(coeffs).first(7), 2, false),
      "owned_coefficients: 4 local DOFs, 4 mask entries and 7 coefficients disagree.", value_error);
  THB_THROWS_WITH_AS(
      mp::owned_coefficients(l2g, mask, coeffs, 0, false),
      "owned_coefficients: 4 local DOFs, 4 mask entries and 8 coefficients disagree.", value_error);
}

/// Rank 1 of 2 receives these blocks whatever it sends.
mt::PresetComm owned_preset(const std::vector<I64Vec>& ids, std::vector<double> data,
                            Shapes shapes) {
  return preset(1, 2, int_blocks(ids), real_blocks(std::move(data), std::move(shapes)));
}

TEST_CASE("mpi gather_owned_coefficients places each rank's rows at the global DOFs") {
  const mp::OwnedCoefficients mine{{}, {}, 2, false};
  // Rank 0 owns DOFs 2 and 0, rank 1 owns DOF 1; two components.
  const mp::GlobalCoefficients g = mp::gather_owned_coefficients(
      owned_preset({{2, 0}, {1}}, {1, 2, 3, 4, 5, 6}, {{2, 2}, {1, 2}}), 3, mine);
  CHECK(g.coeffs == std::vector<double>{3, 4, 5, 6, 1, 2});
  CHECK(g.rank == 2);
  CHECK_FALSE(g.scalar);
  // Scalar blocks (k,): the rank is 1 and the field is scalar.
  const mp::GlobalCoefficients s = mp::gather_owned_coefficients(
      owned_preset({{1, 0}, {2}}, {1.5, 2.5, 3.5}, {{2}, {1}}), 3, mine);
  CHECK(s.coeffs == std::vector<double>{2.5, 1.5, 3.5});
  CHECK(s.rank == 1);
  CHECK(s.scalar);
  // A (k, 1) block is one component, but not a scalar field.
  const mp::GlobalCoefficients col = mp::gather_owned_coefficients(
      owned_preset({{1, 0}, {2}}, {1.5, 2.5, 3.5}, {{2, 1}, {1, 1}}), 3, mine);
  CHECK(col.coeffs == std::vector<double>{2.5, 1.5, 3.5});
  CHECK(col.rank == 1);
  CHECK_FALSE(col.scalar);
}

TEST_CASE("mpi gather_owned_coefficients reads the shape from the first non-empty block") {
  const mp::OwnedCoefficients mine{{}, {}, 3, false};
  // Rank 0 sends nothing, as (0,) in the DOFs and as (0,) or (0, 0) in the coefficients; the
  // component count is that of the first block holding a DOF.
  for (const auto& empty : {std::vector<std::size_t>{0}, std::vector<std::size_t>{0, 0}}) {
    const mp::GlobalCoefficients g =
        mp::gather_owned_coefficients(owned_preset({{}, {1}}, {7, 8, 9}, {empty, {1, 3}}), 2, mine);
    CHECK(g.coeffs == std::vector<double>{0, 0, 0, 7, 8, 9});
    CHECK(g.rank == 3);
    CHECK_FALSE(g.scalar);
  }
  // Nobody sends: a scalar field, all zero.
  for (const auto& empty : {std::vector<std::size_t>{0}, std::vector<std::size_t>{0, 0}}) {
    const mp::GlobalCoefficients g =
        mp::gather_owned_coefficients(owned_preset({{}, {}}, {}, {empty, empty}), 4, mine);
    CHECK(g.coeffs == std::vector<double>(4, 0.0));
    CHECK(g.rank == 1);
    CHECK(g.scalar);
  }
}

TEST_CASE("mpi gather_owned_coefficients leaves a DOF nobody owns at zero") {
  // Bug caught: the previous release left such a DOF uninitialized. DOFs 1 and 3 are sent by
  // nobody (they are supported on inactive cells only).
  const mp::OwnedCoefficients mine{{}, {}, 2, false};
  const mp::GlobalCoefficients g = mp::gather_owned_coefficients(
      owned_preset({{0}, {2}}, {5, 6, 7, 8}, {{1, 2}, {1, 2}}), 4, mine);
  CHECK(g.coeffs == std::vector<double>{5, 6, 0, 0, 7, 8, 0, 0});
}

TEST_CASE("mpi gather_owned_coefficients refuses a DOF out of range or sent twice") {
  const mp::OwnedCoefficients mine{{}, {}, 1, true};
  // The edges 0 and n_global - 1 are valid; n_global and -1 are not.
  THB_NOTHROW(mp::gather_owned_coefficients(owned_preset({{0}, {4}}, {1, 2}, {{1}, {1}}), 5, mine));
  THB_THROWS_WITH_AS(
      mp::gather_owned_coefficients(owned_preset({{0}, {5}}, {1, 2}, {{1}, {1}}), 5, mine),
      "rank 1 sent DOF 5, outside [0, 5).", value_error);
  THB_THROWS_WITH_AS(
      mp::gather_owned_coefficients(owned_preset({{-1}, {2}}, {1, 2}, {{1}, {1}}), 5, mine),
      "rank 0 sent DOF -1, outside [0, 5).", value_error);
  THB_THROWS_WITH_AS(
      mp::gather_owned_coefficients(owned_preset({{3}, {3}}, {1, 2}, {{1}, {1}}), 5, mine),
      "DOF 3 is owned by more than one rank.", value_error);
}

TEST_CASE("mpi gather_owned_coefficients refuses blocks that disagree with each other") {
  const mp::OwnedCoefficients mine{{}, {}, 2, false};
  // Two DOFs but 6 coefficients where the first block fixes two per DOF.
  THB_THROWS_WITH_AS(
      mp::gather_owned_coefficients(
          owned_preset({{0, 1}, {2}}, {1, 2, 3, 4, 5, 6, 7, 8}, {{3, 2}, {1, 2}}), 3, mine),
      "rank 0 sent 2 DOFs and 6 coefficients; expected 2 per DOF.", value_error);
  // A later block with another component count than the first.
  THB_THROWS_WITH_AS(mp::gather_owned_coefficients(
                         owned_preset({{0}, {1}}, {1, 2, 3, 4, 5}, {{1, 2}, {1, 3}}), 2, mine),
                     "rank 1 sent 1 DOFs and 3 coefficients; expected 2 per DOF.", value_error);
  // One block for two ranks, in the DOFs and then in the coefficients.
  const std::string count =
      "allgather returned 1 entries but comm.size is 2; the communicator's allgather is "
      "inconsistent.";
  THB_THROWS_WITH_AS(mp::gather_owned_coefficients(owned_preset({{0}}, {1, 2}, {{1, 2}}), 2, mine),
                     count.c_str(), value_error);
  THB_THROWS_WITH_AS(
      mp::gather_owned_coefficients(owned_preset({{0}, {1}}, {1, 2}, {{1, 2}}), 2, mine),
      count.c_str(), value_error);
}

/// A one-rank communicator that records the shapes of the owned pair it is sent.
class OwnedSpy {
 public:
  OwnedSpy() : shapes_(std::make_shared<Shapes>()) {}
  [[nodiscard]] int rank() const noexcept { return 0; }
  [[nodiscard]] int size() const noexcept { return 1; }
  template <class T>
  [[nodiscard]] mp::Gathered<T> allgatherv(const mp::Block<T>& b) const {
    return mp::SelfCommunicator{}.allgatherv(b);
  }
  [[nodiscard]] std::pair<mp::Gathered<I64>, mp::Gathered<double>> allgather_owned(
      const mp::Block<I64>& i, const mp::Block<double>& v) const {
    shapes_->push_back(i.shape);
    shapes_->push_back(v.shape);
    return mp::SelfCommunicator{}.allgather_owned(i, v);
  }
  template <class T>
  [[nodiscard]] NdArray<T> allreduce_sum(std::span<const T> data,
                                         std::span<const std::size_t> shape) const {
    return mp::SelfCommunicator{}.allreduce_sum(data, shape);
  }
  [[nodiscard]] const Shapes& shapes() const { return *shapes_; }

 private:
  std::shared_ptr<Shapes> shapes_;
};
static_assert(mp::Communicator<OwnedSpy>);

TEST_CASE(
    "mpi gather_owned_coefficients sends scalar coefficients as (k,) and others as (k, rank)") {
  const OwnedSpy scalar;
  const mp::GlobalCoefficients s =
      mp::gather_owned_coefficients(scalar, 4, mp::OwnedCoefficients{{3, 1}, {0.5, 1.5}, 1, true});
  CHECK(scalar.shapes() == Shapes{{2}, {2}});
  CHECK(s.coeffs == std::vector<double>{0, 1.5, 0, 0.5});
  CHECK(s.scalar);
  const OwnedSpy vector_valued;
  const mp::GlobalCoefficients v = mp::gather_owned_coefficients(
      vector_valued, 4, mp::OwnedCoefficients{{3, 1}, {1, 2, 3, 4}, 2, false});
  CHECK(vector_valued.shapes() == Shapes{{2}, {2, 2}});
  CHECK(v.coeffs == std::vector<double>{0, 0, 3, 4, 0, 0, 1, 2});
  CHECK(v.rank == 2);
  // A rank that owns nothing sends empty blocks, and a one-rank gather of them is all zero.
  const OwnedSpy idle;
  const mp::GlobalCoefficients z =
      mp::gather_owned_coefficients(idle, 3, mp::OwnedCoefficients{{}, {}, 1, false});
  CHECK(idle.shapes()[0] == std::vector<std::size_t>{0});
  CHECK(idle.shapes()[1][0] == 0);
  CHECK(z.coeffs == std::vector<double>(3, 0.0));
}

// ---- Tolerances ---------------------------------------------------------------------------------

template <class P>
Eigen::MatrixXd dense_of(const NdArray<P>& a) {
  const auto rows = static_cast<Eigen::Index>(a.shape()[0]);
  const auto cols = static_cast<Eigen::Index>(a.shape()[1]);
  Eigen::MatrixXd m(rows, cols);
  for (Eigen::Index i = 0; i < rows; ++i) {
    for (Eigen::Index j = 0; j < cols; ++j) {
      m(i, j) = static_cast<double>(a.data()[static_cast<std::size_t>(i * cols + j)]);
    }
  }
  return m;
}

/// The infinity norm (the largest absolute row sum).
double inf_norm(const Eigen::MatrixXd& m) { return m.cwiseAbs().rowwise().sum().maxCoeff(); }

template <class T>
double max_abs_of(std::span<const T> v) {
  double worst = 0.0;
  for (const T x : v) {
    worst = (std::max)(worst, std::abs(static_cast<double>(x)));
  }
  return worst;
}

/// The bound on the difference of two coefficient fields that solve the same collocation system
/// from the same gathered values.
///
/// The two are the output of one Kronecker solve on the same operands in the same order, so they
/// are expected to agree to the last bit; the bound is what another build may legitimately do to
/// that routine (a contraction, a vector width), at most the rounding of one evaluation per side.
/// One entry is a chain of `N = sum_d n_d` accumulations (one direction after the other), and the
/// forward error of a backward-stable solve of `A x = b` is at most `gamma_N kappa(A) |x|` in the
/// infinity norm, with `kappa(A_1 x ... x A_d) = prod_k kappa(A_k)` for that norm. The two sides
/// add: a factor 2; `build_slack` is the factor of four the project's default tolerance tier
/// carries for a build, and `u` the unit of the arithmetic the solve runs in (the matrices' type by
/// default; `collocate_values_distributed` can solve in a wider type). The solver's growth factor
/// is taken as 1, an admitted heuristic: a bound wrong by it is still far below the effect of a
/// wrong block (the size of the coefficients themselves; `check_against_serial` asserts that
/// margin).
template <class P>
double coefficient_tolerance(const bs::TensorCollocation<P>& system, double c_max,
                             double u = unit<P>()) {
  double chain = 0.0;
  double kappa = 1.0;
  for (const NdArray<P>& a : system.matrices()) {
    const Eigen::MatrixXd m = dense_of(a);
    chain += static_cast<double>(m.cols());
    kappa *= inf_norm(m) * inf_norm(m.inverse());
  }
  return 2.0 * build_slack * gamma_n(chain, u) * kappa * c_max;
}

/// The bound on the difference of the distributed and the serial L2 projection.
///
/// The ranks differ from the serial run only in the order of the sum that makes the load: the
/// serial load is one chain through the whole lattice, the distributed one a chain through the
/// lattice with the other ranks' cells set to zero (which add exactly zero), then the sum of the
/// ranks' loads. The terms of a load entry are non-negative (a B-spline and a quadrature weight
/// are) times the sampled values, so the sum of their magnitudes is the load of `|f|`,
/// `load_abs_max` at the worst entry. The chain is `sum_d (lattice extent) + n_ranks` long, each
/// side is within `gamma_chain` of the exact sum, hence `2 gamma_chain load_abs_max` apart; the
/// Kronecker mass solve maps a load difference by at most `prod_d ||M_d^{-1}||_inf`; and the two
/// solves, on two loads, differ besides by their own rounding, `2 gamma_N kappa |c|` as in
/// `coefficient_tolerance` (`N` the sum of the mass matrices' orders). `build_slack` as there.
template <class T>
double l2_tolerance(const bs::L2Projection<T>& proj, double load_abs_max, double c_max,
                    int n_ranks) {
  double load_chain = static_cast<double>(n_ranks);
  for (const std::size_t g : proj.grid_shape()) {
    load_chain += static_cast<double>(g);
  }
  double solve_chain = 0.0;
  double inverse_norm = 1.0;
  double kappa = 1.0;
  for (const NdArray<T>& mass : proj.mass()) {
    const Eigen::MatrixXd m = dense_of(mass);
    const double inv = inf_norm(m.inverse());
    solve_chain += static_cast<double>(m.cols());
    inverse_norm *= inv;
    kappa *= inv * inf_norm(m);
  }
  const double u = unit<T>();
  const double load_term = inverse_norm * 2.0 * gamma_n(load_chain, u) * load_abs_max;
  const double solve_term = 2.0 * gamma_n(solve_chain, u) * kappa * c_max;
  return build_slack * (load_term + solve_term);
}

/// The weights' 1-norm of each direction's Lee-Lyche-Morken functionals, `Lambda_d = max_j sum_i
/// |w_ji|`, and their product; `orders` and `weight_units` as `qi_tolerance` uses them.
template <class T>
double qi_lambda(const bs::BsplineSpace<T>& space, double& orders, double& weight_units) {
  double lambda = 1.0;
  orders = 0.0;
  weight_units = 0.0;
  for (const bs::BsplineSpace1D<T>& s : space.spaces()) {
    const bs::detail::LlmDirection dir = bs::detail::llm_direction<T>(s);
    const std::size_t order = dir.weights.shape()[1];
    double worst = 0.0;
    for (std::size_t j = 0; j < dir.weights.shape()[0]; ++j) {
      double row = 0.0;
      for (std::size_t k = 0; k < order; ++k) {
        row += std::abs(dir.weights.data()[j * order + k]);
      }
      worst = (std::max)(worst, row);
    }
    lambda *= worst;
    orders += static_cast<double>(order);
    weight_units += static_cast<double>(order * order) * worst;
  }
  return lambda;
}

/// The bound on the difference of the distributed and the serial tensor quasi-interpolant, with
/// `extra_units` the caller's own rounding (of the sampled values), in units of `u`.
///
/// Each coefficient is `lambda(f) = sum_k a_k f(x_k)`, contracted one direction after the other
/// (a chain of `sum_d (p_d + 1)` accumulations) in double, on the same points: the rank evaluates
/// it on its windowed space, whose knots near an owned function are the global ones, so the points
/// and the weights agree. The two evaluations are within `2 gamma_chain Lambda f_max`, with
/// `Lambda = prod_d Lambda_d` the 1-norm of the tensor weights; the weights themselves, rows of a
/// local collocation inverse, carry a relative error of order `order_d^2 kappa u` with `kappa` at
/// least their norm `Lambda_d`: the `weight_units` term, an admitted heuristic (the inverse is not
/// exposed), as in test_thb_qi.cpp. A `float` space then rounds the coefficients to float: one
/// more float unit of the largest coefficient, as two doubles within the bound can straddle a
/// rounding boundary.
template <class T>
double qi_tolerance(const bs::BsplineSpace<T>& space, double f_max, double c_max,
                    double extra_units = 0.0) {
  double orders = 0.0;
  double weight_units = 0.0;
  const double lambda = qi_lambda(space, orders, weight_units);
  double tol =
      build_slack * unit<double>() * lambda * f_max * (2.0 * orders + weight_units + extra_units);
  if constexpr (std::is_same_v<T, float>) {
    tol += unit<float>() * c_max;
  }
  return tol;
}

/// The same for one function of a THB space: `qi.weights(dof)` carries the functional.
template <class T>
double thb_qi_tolerance(const th::THBSplineSpace<T>& space, const th::THBQuasiInterpolation<T>& qi,
                        I64 dof, double f_max, double extra_units = 0.0) {
  const std::span<const double> w = qi.weights(dof);
  double lambda = 1.0;
  double orders = 0.0;
  double weight_units = 0.0;
  std::size_t at = 0;
  for (const int p : space.degrees()) {
    const auto order = static_cast<std::size_t>(p) + 1;
    double lambda_k = 0.0;
    for (std::size_t i = 0; i < order; ++i) {
      lambda_k += std::abs(w[at + i]);
    }
    at += order;
    lambda *= lambda_k;
    orders += static_cast<double>(order);
    weight_units += static_cast<double>(order * order) * lambda_k;
  }
  return build_slack * unit<double>() * lambda * f_max *
         (2.0 * orders + weight_units + extra_units);
}

// ---- Checks against the serial result -----------------------------------------------------------

template <class T>
struct RankResult {
  std::vector<T> coeffs;
  I64Vec owned_dofs;
  std::size_t calls = 0;
};

/// The largest `|a_i - b_i|`; infinity when the lengths differ.
template <class A, class B>
double max_diff(std::span<const A> a, std::span<const B> b) {
  if (a.size() != b.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = (std::max)(worst, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  }
  return worst;
}

/// The same over the rows of the given DOFs only (`rank` columns per DOF).
template <class A, class B>
double max_diff_on(std::span<const A> a, std::span<const B> b, const I64Vec& dofs,
                   std::size_t rank) {
  double worst = 0.0;
  for (const I64 g : dofs) {
    for (std::size_t c = 0; c < rank; ++c) {
      const auto i = static_cast<std::size_t>(g) * rank + c;
      worst = (std::max)(worst, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
  }
  return worst;
}

/// Every rank's global coefficients against the serial ones, on all DOFs and on the DOFs the rank
/// owns; the owned DOFs of the ranks tile the DOFs (a rank may own none when `each_owns` is off).
///
/// The check can fail: `c_max > 0.1` shows the compared field is not trivially zero, and
/// `tol < signal_ratio c_max` that the bound is far below what a wrong block or a lost rank leaves
/// (an error of the size of the coefficients).
template <class T>
void check_against_serial(const std::vector<RankResult<T>>& ranks, std::span<const T> serial,
                          std::size_t value_rank, I64 n_dofs, double tol, bool each_owns = true,
                          double signal_ratio = 1e-2) {
  const double c_max = max_abs_of(serial);
  CHECK(c_max > 0.1);
  CHECK(tol < signal_ratio * c_max);
  REQUIRE(!ranks.empty());
  REQUIRE(serial.size() == static_cast<std::size_t>(n_dofs) * value_rank);
  std::vector<int> owners(static_cast<std::size_t>(n_dofs), 0);
  for (std::size_t r = 0; r < ranks.size(); ++r) {
    INFO("rank " << r);
    REQUIRE(ranks[r].coeffs.size() == serial.size());
    const std::span<const T> mine(ranks[r].coeffs);
    CHECK(max_diff<T, T>(mine, serial) <= tol);
    CHECK(max_diff_on<T, T>(mine, serial, ranks[r].owned_dofs, value_rank) <= tol);
    if (each_owns) {
      CHECK_FALSE(ranks[r].owned_dofs.empty());
    }
    for (const I64 g : ranks[r].owned_dofs) {
      ++owners[static_cast<std::size_t>(g)];
    }
  }
  CHECK(std::all_of(owners.begin(), owners.end(), [](int k) { return k == 1; }));
}

// ---- Collocation --------------------------------------------------------------------------------

using Derivatives = std::vector<std::optional<bs::BoundaryDerivatives>>;

template <class T>
std::vector<RankResult<T>> run_interpolate(const bs::BsplineSpace<T>& space,
                                           const gr::Partition& part, int n, std::size_t value_rank,
                                           const std::vector<std::vector<T>>& nodes,
                                           const Derivatives& derivatives) {
  return mt::run_ranks<RankResult<T>>(n, [&](mt::HubComm comm) {
    const auto ds = mp::distribute(space, part, comm);
    RankResult<T> out;
    const bs::Bspline<T> b = mp::interpolate_distributed(
        ds, comm,
        [&](pantr::mdview<const double, 2> pts) {
          ++out.calls;
          return block_values(pts, value_rank);
        },
        nodes, derivatives);
    out.coeffs.assign(b.control_points().begin(), b.control_points().end());
    out.owned_dofs = owned_dofs_of(ds);
    return out;
  });
}

template <class T>
bs::Bspline<T> serial_interpolate(const bs::BsplineSpace<T>& space, std::size_t value_rank,
                                  const std::vector<std::vector<T>>& nodes,
                                  const Derivatives& derivatives) {
  return bs::interpolate(
      space,
      [&](const pantr::quad::PointsLattice<T>& lattice) {
        return lattice_values<T>(lattice, value_rank);
      },
      nodes, derivatives);
}

/// The Greville nodes of a space, or the given ones.
template <class T>
std::vector<std::vector<T>> nodes_or_greville(const bs::BsplineSpace<T>& space,
                                              const std::vector<std::vector<T>>& given) {
  if (!given.empty()) {
    return given;
  }
  std::vector<std::vector<T>> nodes;
  for (const bs::BsplineSpace1D<T>& s : space.spaces()) {
    nodes.push_back(s.greville());
  }
  return nodes;
}

template <class T>
void check_interpolate_against_serial(const std::vector<std::vector<T>>& given,
                                      const Derivatives& derivatives, const TensorCase& c) {
  const bs::BsplineSpace<T> space = make_space<T>(c);
  const std::vector<std::vector<T>> nodes = nodes_or_greville(space, given);
  const std::vector<std::span<const T>> spans(nodes.begin(), nodes.end());
  const bs::TensorCollocation<T> system(space, spans, derivatives);
  for (const auto& value_rank : {std::size_t{1}, std::size_t{2}}) {
    const bs::Bspline<T> serial = serial_interpolate<T>(space, value_rank, given, derivatives);
    const double tol = coefficient_tolerance(system, max_abs_of(serial.control_points()));
    for (const int n : kRankCounts) {
      INFO(c.name << ", components " << value_rank << ", ranks " << n);
      const std::vector<RankResult<T>> ranks =
          run_interpolate<T>(space, rcb_of(space, n), n, value_rank, given, derivatives);
      for (const RankResult<T>& r : ranks) {
        CHECK(r.calls == 1);
      }
      check_against_serial<T>(ranks, serial.control_points(), value_rank, count_dofs(space), tol);
    }
  }
}

TEST_CASE(
    "mpi interpolate_distributed equals the serial interpolant on 2, 3 and 4 ranks (double)") {
  for (const TensorCase& c : tensor_cases()) {
    check_interpolate_against_serial<double>({}, {}, c);
  }
}

TEST_CASE("mpi interpolate_distributed equals the serial interpolant on 2, 3 and 4 ranks (float)") {
  for (const TensorCase& c : tensor_cases()) {
    check_interpolate_against_serial<float>({}, {}, c);
  }
}

TEST_CASE(
    "mpi interpolate_distributed with given nodes and boundary derivatives equals the serial one") {
  // Chebyshev-Lobatto nodes, which are not the Greville abscissae the default uses, and zero
  // boundary derivatives in the first direction (both ends) and two at the start of the second.
  const TensorCase& c = tensor_cases()[0];
  const bs::BsplineSpace<double> space = make_space<double>(c);
  std::vector<std::vector<double>> nodes;
  for (const bs::BsplineSpace1D<double>& s : space.spaces()) {
    const auto m = static_cast<std::size_t>(s.num_basis());
    const auto [lo, hi] = s.domain();
    std::vector<double> x(m);
    for (std::size_t j = 0; j < m; ++j) {
      const double t =
          0.5 *
          (1.0 - std::cos(std::numbers::pi * static_cast<double>(j) / static_cast<double>(m - 1)));
      x[j] = lo + (hi - lo) * t;
    }
    nodes.push_back(std::move(x));
  }
  const std::vector<Derivatives> derivative_sets{{},
                                                 {bs::BoundaryDerivatives{1, 1}, std::nullopt},
                                                 {std::nullopt, bs::BoundaryDerivatives{2, 0}}};
  for (const Derivatives& derivs : derivative_sets) {
    INFO("derivative constraints " << derivs.size());
    check_interpolate_against_serial<double>(nodes, derivs, c);
  }
}

/// What a rank's callable saw.
struct SeenBlock {
  std::vector<double> points;
  std::size_t rows = 0;
  std::size_t calls = 0;
};

TEST_CASE(
    "mpi interpolate_distributed hands each rank a contiguous block of the C-order node grid") {
  // Oracle: the node grid written out by nested loops over the Greville abscissae (last direction
  // fastest), cut by the documented split of `block_bounds`, not by the lattice the code uses.
  const bs::BsplineSpace<double> space = make_space<double>(tensor_cases()[0]);
  const std::vector<double> g0 = space.spaces()[0].greville();
  const std::vector<double> g1 = space.spaces()[1].greville();
  std::vector<double> grid;
  for (const double a : g0) {
    for (const double b : g1) {
      grid.push_back(a);
      grid.push_back(b);
    }
  }
  const std::size_t total = g0.size() * g1.size();
  for (const int n : kRankCounts) {
    INFO("ranks " << n);
    const gr::Partition part = rcb_of(space, n);
    const std::vector<SeenBlock> seen = mt::run_ranks<SeenBlock>(n, [&](mt::HubComm comm) {
      const auto ds = mp::distribute(space, part, comm);
      SeenBlock out;
      static_cast<void>(
          mp::interpolate_distributed(ds, comm, [&](pantr::mdview<const double, 2> pts) {
            ++out.calls;
            out.rows = pts.extent(0);
            out.points.assign(pts.data_handle(), pts.data_handle() + pts.extent(0) * pts.extent(1));
            return block_values(pts, 1);
          }));
      return out;
    });
    std::vector<double> joined;
    for (std::size_t r = 0; r < seen.size(); ++r) {
      INFO("rank " << r);
      const auto [start, stop] = mp::block_bounds(total, static_cast<std::size_t>(n), r);
      CHECK(seen[r].calls == 1);
      CHECK(seen[r].rows == stop - start);
      CHECK(seen[r].points ==
            std::vector<double>(grid.begin() + static_cast<std::ptrdiff_t>(2 * start),
                                grid.begin() + static_cast<std::ptrdiff_t>(2 * stop)));
      joined.insert(joined.end(), seen[r].points.begin(), seen[r].points.end());
    }
    CHECK(joined == grid);
  }
}

TEST_CASE(
    "mpi interpolate_distributed with more ranks than nodes gives the last ranks an empty block") {
  // A quadratic Bezier interval has three nodes. With four ranks the last has none, with five the
  // last two have none: an empty block, (0,) or (0, 2) as the callable returns it, and the first
  // non-empty block gives the number of components. The ranks past the first own no cell.
  const TensorCase bezier{"bezier", {{0.0, 0.0, 0.0, 1.0, 1.0, 1.0}}, {2}};
  const bs::BsplineSpace<double> space = make_space<double>(bezier);
  const std::vector<std::vector<double>> nodes = nodes_or_greville(space, {});
  const std::vector<std::span<const double>> spans(nodes.begin(), nodes.end());
  const bs::TensorCollocation<double> system(space, spans);
  for (const auto& value_rank : {std::size_t{1}, std::size_t{2}}) {
    const bs::Bspline<double> serial = serial_interpolate<double>(space, value_rank, {}, {});
    const double tol = coefficient_tolerance(system, max_abs_of(serial.control_points()));
    for (const int n : {4, 5}) {
      INFO("components " << value_rank << ", ranks " << n);
      const std::vector<RankResult<double>> ranks =
          run_interpolate<double>(space, gr::Partition(Owners{0}, n), n, value_rank, {}, {});
      REQUIRE(ranks.size() == static_cast<std::size_t>(n));
      for (const RankResult<double>& r : ranks) {
        CHECK(r.calls == 1);
        CHECK(max_diff<double, double>(r.coeffs, serial.control_points()) <= tol);
      }
      CHECK(ranks[0].owned_dofs.size() == 3);
      CHECK(ranks[static_cast<std::size_t>(n) - 1].owned_dofs.empty());
    }
  }
}

TEST_CASE("mpi interpolate_distributed refuses a function that returns another number of values") {
  const bs::BsplineSpace<double> space = make_space<double>(tensor_cases()[0]);
  const mp::SelfCommunicator comm;
  const auto ds = mp::distribute(space, rcb_of(space, 1), comm);
  const auto too_many = [](pantr::mdview<const double, 2> pts) {
    return NdArray<double>(std::vector<std::size_t>{pts.extent(0) + 1});
  };
  const auto cube = [](pantr::mdview<const double, 2> pts) {
    return NdArray<double>(std::vector<std::size_t>{pts.extent(0), 1, 1});
  };
  const std::string msg =
      "interpolate_distributed: the function returned another number of values than the 64 "
      "points of this rank's block.";
  THB_THROWS_WITH_AS(mp::interpolate_distributed(ds, comm, too_many), msg.c_str(), value_error);
  THB_THROWS_WITH_AS(mp::interpolate_distributed(ds, comm, cube), msg.c_str(), value_error);
}

TEST_CASE(
    "mpi collocate_values_distributed rounds the gathered values to the space's type before the "
    "solve") {
  // A float space solved in double (Cm = double): the values are rounded to float first, as the
  // previous release did. Oracle: that rounding done by hand, then the serial solve in double. The
  // field's values are not floats, so skipping the rounding changes the solution by about the
  // float unit times the conditioning, far more than the double bound.
  const bs::BsplineSpace<float> space = make_space<float>(tensor_cases()[0]);
  const std::vector<std::vector<float>> nodes = nodes_or_greville<float>(space, {});
  const std::vector<std::span<const float>> spans(nodes.begin(), nodes.end());
  const bs::TensorCollocation<float> system(space, spans);
  const NdArray<float> pts = pantr::quad::PointsLattice<float>(nodes).all_points(pantr::Order::C);
  const NdArray<double> values = field_values<float>(pts.data(), pts.shape()[0], 2, 1);
  std::vector<double> rounded(values.size());
  for (std::size_t i = 0; i < rounded.size(); ++i) {
    rounded[i] = static_cast<double>(static_cast<float>(values.data()[i]));
  }
  const double tol = bs::default_fit_tol<double>(std::nullopt);
  const NdArray<double> want = system.solve<double>(std::span<const double>(rounded), 1, tol);
  const NdArray<double> unrounded = system.solve<double>(values.flat(), 1, tol);
  const NdArray<double> got = mp::collocate_values_distributed<float, double>(
      mp::SelfCommunicator{}, system, mp::Block<double>{values.flat(), {values.size()}}, tol);
  const double bound = coefficient_tolerance(system, max_abs_of(want.flat()), unit<double>());
  CHECK(max_diff<double, double>(got.flat(), want.flat()) <= bound);
  CHECK(max_diff<double, double>(unrounded.flat(), want.flat()) > 10.0 * bound);
}

// ---- L2 projection ------------------------------------------------------------------------------

std::vector<std::pair<bool, bool>> boundary_flags(std::size_t dim, bool on) {
  std::vector<std::pair<bool, bool>> flags(dim, {false, false});
  if (on) {
    for (std::size_t d = 0; d < dim; ++d) {
      flags[d] = {true, d % 2 == 0};  // both ends in the even directions, the start in the others
    }
  }
  return flags;
}

std::vector<long long> quadrature_of(const TensorCase& c, long long extra) {
  std::vector<long long> n_quad;
  for (const int p : c.degrees) {
    n_quad.push_back(p + 1 + extra);
  }
  return n_quad;
}

/// The field on a projection's quadrature lattice and on each of its traces.
template <class T>
struct L2Inputs {
  NdArray<T> values;
  std::vector<NdArray<T>> traces;
};

template <class T>
L2Inputs<T> l2_inputs(const bs::L2Projection<T>& proj, std::size_t value_rank) {
  std::vector<std::vector<T>> nodes;
  for (const NdArray<T>& x : proj.nodes()) {
    nodes.emplace_back(x.data(), x.data() + x.size());
  }
  L2Inputs<T> in{lattice_values<T>(pantr::quad::PointsLattice<T>(std::move(nodes)), value_rank),
                 {}};
  for (std::size_t t = 0; t < proj.traces().size(); ++t) {
    in.traces.push_back(
        lattice_values<T>(pantr::quad::PointsLattice<T>(proj.trace_nodes(t)), value_rank));
  }
  return in;
}

/// The largest load entry of `|f|`, over the components: the sum of the magnitudes of the terms.
template <class T>
double abs_load_max(const bs::L2Projection<T>& proj, const NdArray<T>& values,
                    std::size_t value_rank) {
  const std::size_t n_pts = values.size() / value_rank;
  double worst = 0.0;
  for (std::size_t r = 0; r < value_rank; ++r) {
    std::vector<double> magnitude(n_pts);
    for (std::size_t i = 0; i < n_pts; ++i) {
      magnitude[i] = std::abs(static_cast<double>(values.data()[i * value_rank + r]));
    }
    const NdArray<double> load = proj.template load<double>(std::span<const double>(magnitude));
    worst = (std::max)(worst, max_abs_of(load.flat()));
  }
  return worst;
}

template <class T>
std::vector<RankResult<T>> run_l2(const bs::BsplineSpace<T>& space, const gr::Partition& part,
                                  int n, std::size_t value_rank,
                                  const std::vector<long long>& n_quad,
                                  const std::vector<std::pair<bool, bool>>& flags) {
  return mt::run_ranks<RankResult<T>>(n, [&](mt::HubComm comm) {
    const auto ds = mp::distribute(space, part, comm);
    const bs::L2Projection<T> proj(space, n_quad, bs::L2Quadrature::gauss_legendre, flags);
    const L2Inputs<T> in = l2_inputs(proj, value_rank);
    std::vector<std::span<const T>> traces;
    for (const NdArray<T>& t : in.traces) {
      traces.emplace_back(t.flat());
    }
    const NdArray<T> c =
        mp::l2_project_values_distributed(comm, proj, ds.partition().cell_owner(),
                                          std::span<const T>(in.values.flat()), value_rank, traces);
    RankResult<T> out;
    out.coeffs.assign(c.data(), c.data() + c.size());
    out.owned_dofs = owned_dofs_of(ds);
    out.calls = 1;
    return out;
  });
}

template <class T>
void check_l2_against_serial(const std::vector<TensorCase>& cases, double signal_ratio) {
  for (const TensorCase& c : cases) {
    const bs::BsplineSpace<T> space = make_space<T>(c);
    const std::vector<long long> n_quad = quadrature_of(c, 0);
    for (const bool boundary : {false, true}) {
      const auto flags = boundary_flags(space.dim(), boundary);
      const bs::L2Projection<T> proj(space, n_quad, bs::L2Quadrature::gauss_legendre, flags);
      for (const auto& value_rank : {std::size_t{1}, std::size_t{2}}) {
        const bs::Bspline<T> serial = bs::l2_project(
            space,
            [&](const pantr::quad::PointsLattice<T>& lattice) {
              return lattice_values<T>(lattice, value_rank);
            },
            std::optional<std::vector<long long>>(n_quad), bs::L2Quadrature::gauss_legendre, flags);
        const L2Inputs<T> in = l2_inputs(proj, value_rank);
        const double load_abs = abs_load_max(proj, in.values, value_rank);
        const double c_max = max_abs_of(serial.control_points());
        for (const int n : kRankCounts) {
          INFO(c.name << ", components " << value_rank << ", boundary interpolation " << boundary
                      << ", ranks " << n);
          const std::vector<RankResult<T>> ranks =
              run_l2<T>(space, rcb_of(space, n), n, value_rank, n_quad, flags);
          check_against_serial<T>(ranks, serial.control_points(), value_rank, count_dofs(space),
                                  l2_tolerance(proj, load_abs, c_max, n), true, signal_ratio);
        }
      }
    }
  }
}

TEST_CASE(
    "mpi l2_project_values_distributed equals the serial projection on 2, 3 and 4 ranks (double)") {
  check_l2_against_serial<double>(tensor_cases(), 1e-2);
  check_l2_against_serial<double>(uniform_cases(), 1e-2);
}

TEST_CASE(
    "mpi l2_project_values_distributed equals the serial projection on 2, 3 and 4 ranks (float)") {
  // The bound is a tenth of the coefficients here, not a hundredth: a lost cell leaves an error of
  // the size of the coefficients of the DOFs it touches, ten times the bound at least.
  check_l2_against_serial<float>(uniform_cases(), 0.1);
}

TEST_CASE(
    "mpi owned_quadrature_mask marks the quadrature points that lie in the cells a rank owns") {
  // Oracle: the cell of a point found from its coordinates against the breakpoints (Gauss nodes
  // are strictly inside their interval), not from the index arithmetic `q / n_d` of the code.
  // Quadrature counts: p + 1, p + 2, and 1, 2, 3 per direction. Cells owned by ranks 0..2, one in
  // six inactive (-1), so that every rank, and nobody, owns cells.
  for (const TensorCase& c : tensor_cases()) {
    const bs::BsplineSpace<double> space = make_space<double>(c);
    const std::vector<std::vector<double>> bp = breakpoints_of(space);
    const std::size_t dim = bp.size();
    std::vector<std::size_t> n_int(dim);
    for (std::size_t d = 0; d < dim; ++d) {
      n_int[d] = bp[d].size() - 1;
    }
    Owners owner(static_cast<std::size_t>(count_cells(space)));
    for (std::size_t k = 0; k < owner.size(); ++k) {
      owner[k] = k % 6 == 5 ? -1 : static_cast<std::int32_t>((k * 5 + k / 3) % 3);
    }
    std::vector<std::vector<long long>> quadratures{quadrature_of(c, 0), quadrature_of(c, 1), {}};
    for (std::size_t d = 0; d < dim; ++d) {
      quadratures[2].push_back(static_cast<long long>(d) + 1);
    }
    for (const std::vector<long long>& n_quad : quadratures) {
      const bs::L2Projection<double> proj(space, n_quad, bs::L2Quadrature::gauss_legendre,
                                          boundary_flags(dim, false));
      const std::vector<std::size_t> grid = proj.grid_shape();
      std::size_t n_pts = 1;
      for (const std::size_t g : grid) {
        n_pts *= g;
      }
      for (int rank = 0; rank < 3; ++rank) {
        INFO(c.name << ", quadrature " << n_quad.front() << ", rank " << rank);
        std::vector<std::uint8_t> expected(n_pts);
        std::vector<std::size_t> q(dim, 0);
        for (std::size_t i = 0; i < n_pts; ++i) {
          std::size_t cell = 0;
          for (std::size_t d = 0; d < dim; ++d) {
            const double x = proj.nodes()[d].data()[q[d]];
            const auto above = static_cast<std::size_t>(
                std::upper_bound(bp[d].begin(), bp[d].end(), x) - bp[d].begin());
            cell = cell * n_int[d] + (above - 1);
          }
          expected[i] = owner[cell] == rank ? 1 : 0;
          for (std::size_t d = dim; d-- > 0;) {
            if (++q[d] < grid[d]) {
              break;
            }
            q[d] = 0;
          }
        }
        const std::vector<std::uint8_t> got = mp::owned_quadrature_mask(proj, owner, rank);
        CHECK(got == expected);
        const auto ones = static_cast<std::size_t>(std::count(expected.begin(), expected.end(), 1));
        CHECK(ones > 0);
        CHECK(ones < n_pts);
      }
    }
    const bs::L2Projection<double> proj(
        space, quadrature_of(c, 0), bs::L2Quadrature::gauss_legendre, boundary_flags(dim, false));
    Owners short_owner = owner;
    short_owner.pop_back();
    const std::string msg = "partition has " + std::to_string(owner.size() - 1) +
                            " cells; expected " + std::to_string(owner.size()) +
                            " (the global space's cell count).";
    THB_THROWS_WITH_AS(mp::owned_quadrature_mask(proj, short_owner, 0), msg.c_str(), value_error);
  }
}

/// A communicator whose reduction returns a one-dimensional tensor, whatever it is given.
class FlatReduceComm : public mp::SelfCommunicator {
 public:
  template <class T>
  [[nodiscard]] NdArray<T> allreduce_sum(std::span<const T> data,
                                         std::span<const std::size_t>) const {
    NdArray<T> out(std::vector<std::size_t>{data.size()});
    std::copy(data.begin(), data.end(), out.data());
    return out;
  }
};
static_assert(mp::Communicator<FlatReduceComm>);

TEST_CASE("mpi l2_project_values_distributed refuses values, traces and a load of the wrong size") {
  const TensorCase& c = tensor_cases()[2];  // the cubic: 5 cells, 8 functions
  const bs::BsplineSpace<double> space = make_space<double>(c);
  const bs::L2Projection<double> proj(space, {4}, bs::L2Quadrature::gauss_legendre, {{true, true}});
  const mp::SelfCommunicator comm;
  const Owners owner(5, 0);
  const std::vector<double> values(20, 1.0);
  const std::vector<double> point{0.5};
  const std::vector<std::span<const double>> traces{std::span<const double>(point),
                                                    std::span<const double>(point)};
  const auto run = [&](const auto& cm, std::span<const double> v, std::size_t rank_dim,
                       const std::vector<std::span<const double>>& tr,
                       std::span<const Owners::value_type> o) {
    return mp::l2_project_values_distributed(cm, proj, o, v, rank_dim, tr);
  };
  const std::span<const Owners::value_type> owners(owner);
  THB_NOTHROW(run(comm, values, 1, traces, owners));
  THB_THROWS_WITH_AS(run(comm, std::span<const double>(values).first(19), 1, traces, owners),
                     "l2_project: expected 20 x 1 values on the quadrature lattice, got 19.",
                     value_error);
  const std::vector<double> longer(21, 1.0);
  THB_THROWS_WITH_AS(run(comm, longer, 1, traces, owners),
                     "l2_project: expected 20 x 1 values on the quadrature lattice, got 21.",
                     value_error);
  THB_THROWS_WITH_AS(run(comm, values, 0, traces, owners),
                     "l2_project: expected 20 x 0 values on the quadrature lattice, got 20.",
                     value_error);
  THB_THROWS_WITH_AS(run(comm, values, 1, {}, owners),
                     "L2Projection: expected 2 trace value arrays, got 0.", value_error);
  THB_THROWS_WITH_AS(run(comm, values, 1, traces, owners.first(4)),
                     "partition has 4 cells; expected 5 (the global space's cell count).",
                     value_error);
  // Two components but a trace that holds one value: not a multiple of the component count.
  const std::vector<double> values2(40, 1.0);
  THB_THROWS_WITH_AS(run(comm, values2, 2, traces, owners),
                     "trace 0 holds 1 values, not a multiple of 2.", value_error);
  // A reduction that returns another shape than it was given.
  THB_THROWS_WITH_AS(run(FlatReduceComm{}, values, 1, traces, owners),
                     "Reduced load has shape (8,), expected (8, 1); this indicates 'func' "
                     "returned an inconsistent shape across ranks.",
                     value_error);
}

// ---- Quasi-interpolation ------------------------------------------------------------------------

template <class T>
std::vector<RankResult<T>> run_qi(const bs::BsplineSpace<T>& space, const gr::Partition& part,
                                  int n, std::size_t value_rank) {
  return mt::run_ranks<RankResult<T>>(n, [&](mt::HubComm comm) {
    const auto ds = mp::distribute(space, part, comm);
    RankResult<T> out;
    const bs::Bspline<T> b =
        mp::quasi_interpolate_distributed(ds, comm, [&](pantr::mdview<const double, 2> pts) {
          ++out.calls;
          return block_values(pts, value_rank);
        });
    out.coeffs.assign(b.control_points().begin(), b.control_points().end());
    out.owned_dofs = owned_dofs_of(ds);
    return out;
  });
}

template <class T>
bs::Bspline<T> serial_qi(const bs::BsplineSpace<T>& space, std::size_t value_rank) {
  return bs::quasi_interpolate(
      space, [&](pantr::mdview<const double, 2> pts) { return block_values(pts, value_rank); });
}

/// The largest sampled value of the field at a quasi-interpolant's sample points.
template <class T>
double sampled_max(const bs::BsplineSpace<T>& space, std::size_t value_rank) {
  const NdArray<double> pts = bs::QuasiInterpolation<T>(space).points();
  const NdArray<double> values = block_values(pts.view<2>(), value_rank);
  return max_abs_of(values.flat());
}

template <class T>
void check_qi_against_serial() {
  for (const TensorCase& c : tensor_cases()) {
    const bs::BsplineSpace<T> space = make_space<T>(c);
    for (const auto& value_rank : {std::size_t{1}, std::size_t{2}}) {
      const bs::Bspline<T> serial = serial_qi<T>(space, value_rank);
      const double tol = qi_tolerance(space, sampled_max<T>(space, value_rank),
                                      max_abs_of(serial.control_points()));
      for (const int n : kRankCounts) {
        INFO(c.name << ", components " << value_rank << ", ranks " << n);
        const std::vector<RankResult<T>> ranks = run_qi<T>(space, rcb_of(space, n), n, value_rank);
        for (const RankResult<T>& r : ranks) {
          CHECK(r.calls == 1);
        }
        check_against_serial<T>(ranks, serial.control_points(), value_rank, count_dofs(space), tol);
      }
    }
  }
}

TEST_CASE(
    "mpi quasi_interpolate_distributed equals the serial quasi-interpolant on 2, 3 and 4 ranks "
    "(double)") {
  check_qi_against_serial<double>();
}

TEST_CASE(
    "mpi quasi_interpolate_distributed equals the serial quasi-interpolant on 2, 3 and 4 ranks "
    "(float)") {
  check_qi_against_serial<float>();
}

TEST_CASE(
    "mpi quasi_interpolate_distributed with a rank that owns no cell does not call the function "
    "there") {
  // Rank 2 of 3 owns nothing: it sends empty blocks, never calls the function, and still holds
  // the global quasi-interpolant.
  const bs::BsplineSpace<double> space = make_space<double>(tensor_cases()[0]);
  Owners owner(25, 0);
  std::fill(owner.begin() + 12, owner.end(), 1);
  for (const auto& value_rank : {std::size_t{1}, std::size_t{2}}) {
    const bs::Bspline<double> serial = serial_qi<double>(space, value_rank);
    const double tol = qi_tolerance(space, sampled_max<double>(space, value_rank),
                                    max_abs_of(serial.control_points()));
    const std::vector<RankResult<double>> ranks =
        run_qi<double>(space, gr::Partition(owner, 3), 3, value_rank);
    REQUIRE(ranks.size() == 3);
    CHECK(ranks[0].calls == 1);
    CHECK(ranks[1].calls == 1);
    CHECK(ranks[2].calls == 0);
    CHECK(ranks[2].owned_dofs.empty());
    check_against_serial<double>(ranks, serial.control_points(), value_rank, count_dofs(space), tol,
                                 false);
  }
}

TEST_CASE("mpi quasi_interpolate_distributed returns the coefficients of a spline of the space") {
  // The defining property of a quasi-interpolant, found without the serial operator: the values
  // of a spline of the space (random coefficients in [-1, 1]) are quasi-interpolated back to the
  // spline's own coefficients. The sampled values carry the rounding of the spline's evaluation,
  // `2 p` per direction for the recurrence and one for the sum (`value_units`).
  for (const TensorCase& c : tensor_cases()) {
    const bs::BsplineSpace<double> space = make_space<double>(c);
    const auto n_dofs = static_cast<std::size_t>(count_dofs(space));
    thb_test::Lcg rng(17);
    NdArray<double> cp({n_dofs, 2}, pantr::for_overwrite);
    for (std::size_t i = 0; i < cp.size(); ++i) {
      cp.data()[i] = 2.0 * rng.next() - 1.0;
    }
    std::vector<std::size_t> shape;
    for (const bs::BsplineSpace1D<double>& s : space.spaces()) {
      shape.push_back(static_cast<std::size_t>(s.num_basis()));
    }
    shape.push_back(2);
    NdArray<double> shaped(shape, pantr::for_overwrite);
    std::copy(cp.data(), cp.data() + cp.size(), shaped.data());
    const bs::Bspline<double> spline(space, std::move(shaped), false);
    double value_units = 1.0;
    for (const int p : c.degrees) {
      value_units += 2.0 * p;
    }
    const double tol = qi_tolerance(space, 1.0, 1.0, value_units);
    for (const int n : kRankCounts) {
      INFO(c.name << ", ranks " << n);
      const std::vector<RankResult<double>> ranks =
          mt::run_ranks<RankResult<double>>(n, [&](mt::HubComm comm) {
            const auto ds = mp::distribute(space, rcb_of(space, n), comm);
            const bs::Bspline<double> q = mp::quasi_interpolate_distributed(
                ds, comm, [&](pantr::mdview<const double, 2> pts) {
                  return bs::evaluate<double, double>(spline, pts);
                });
            RankResult<double> out;
            out.coeffs.assign(q.control_points().begin(), q.control_points().end());
            return out;
          });
      for (const RankResult<double>& r : ranks) {
        CHECK(max_diff<double, double>(r.coeffs, spline.control_points()) <= tol);
      }
      CHECK(tol < 1e-6);
    }
  }
}

// ---- THB quasi-interpolation --------------------------------------------------------------------

using ThbSpace = th::THBSplineSpace<double>;
using pantr::thb::THBQuasiInterpolation;

/// The spaces of test_thb_qi.cpp: two levels deep in 2D, 3D, a cubic in 1D, mixed degrees.
std::vector<ThbSpace> thb_spaces() {
  std::vector<ThbSpace> out;
  out.push_back(thb_test::refined_2d(true));
  const auto b3 =
      thb_test::make_thb_space({thb_test::k2_3d, thb_test::k2_3d, thb_test::k2_3d}, {2, 2, 2},
                               {2, 2, 2}, {std::nullopt, std::nullopt, std::nullopt}, true);
  out.push_back(b3.refine(thb_test::Ivec{0, 3}, std::nullopt));
  const auto b1 = thb_test::make_thb_space({thb_test::k3}, {3}, {2}, {0}, true);
  out.push_back(b1.refine_region(0, thb_test::Ivec{0}, thb_test::Ivec{2}, std::nullopt)
                    .refine_region(1, thb_test::Ivec{0}, thb_test::Ivec{2}, std::nullopt));
  const auto mixed = thb_test::make_thb_space({thb_test::k2, thb_test::k3}, {2, 3}, {2, 2},
                                              {std::nullopt, std::nullopt}, true);
  out.push_back(mixed.refine_region(0, thb_test::Ivec{0, 0}, thb_test::Ivec{2, 2}, std::nullopt));
  return out;
}

template <class T>
std::vector<RankResult<double>> run_thb_qi(const th::THBSplineSpace<T>& space,
                                           const gr::Partition& part, int n,
                                           std::size_t value_rank) {
  return mt::run_ranks<RankResult<double>>(n, [&](mt::HubComm comm) {
    const auto ds = mp::distribute(space, part, comm);
    RankResult<double> out;
    const th::THBSpline<T> s =
        mp::quasi_interpolate_distributed(ds, comm, [&](pantr::mdview<const double, 2> pts) {
          ++out.calls;
          return block_values(pts, value_rank);
        });
    out.coeffs.assign(s.control_points().begin(), s.control_points().end());
    out.owned_dofs = owned_dofs_of(ds);
    return out;
  });
}

/// Every rank's THB coefficients against the serial `THBQuasiInterpolation`, DOF by DOF with the
/// bound of that DOF's functional, on all DOFs and on the rank's owned ones.
template <class T>
void check_thb_qi_against_serial(const th::THBSplineSpace<T>& space, const gr::Partition& part,
                                 int n, std::size_t value_rank, bool each_owns = true) {
  const THBQuasiInterpolation<T> qi(space);
  const NdArray<double> pts = qi.points();
  const NdArray<double> values = block_values(pts.template view<2>(), value_rank);
  const std::size_t m = pts.shape()[0];
  const NdArray<double> serial =
      qi.coefficients(pantr::mdview<const double, 2>(values.data(), m, value_rank));
  const double f_max = max_abs_of(values.flat());
  const double c_max = max_abs_of(serial.flat());
  const auto n_dofs = static_cast<std::size_t>(space.num_total_basis());
  REQUIRE(serial.size() == n_dofs * value_rank);
  CHECK(c_max > 0.1);
  std::vector<double> tol(n_dofs);
  for (std::size_t g = 0; g < n_dofs; ++g) {
    tol[g] = thb_qi_tolerance(space, qi, static_cast<I64>(g), f_max);
    CHECK(tol[g] < 1e-2 * c_max);
  }
  const std::vector<RankResult<double>> ranks = run_thb_qi<T>(space, part, n, value_rank);
  std::vector<int> owners(n_dofs, 0);
  for (std::size_t r = 0; r < ranks.size(); ++r) {
    INFO("rank " << r);
    REQUIRE(ranks[r].coeffs.size() == serial.size());
    double worst_ratio = 0.0;  // the largest |difference| / bound over the DOFs
    for (std::size_t g = 0; g < n_dofs; ++g) {
      for (std::size_t k = 0; k < value_rank; ++k) {
        const double d =
            std::abs(ranks[r].coeffs[g * value_rank + k] - serial.data()[g * value_rank + k]);
        worst_ratio = (std::max)(worst_ratio, d / tol[g]);
      }
    }
    CHECK(worst_ratio <= 1.0);
    double worst_owned = 0.0;
    for (const I64 g : ranks[r].owned_dofs) {
      ++owners[static_cast<std::size_t>(g)];
      for (std::size_t k = 0; k < value_rank; ++k) {
        const auto i = static_cast<std::size_t>(g) * value_rank + k;
        worst_owned = (std::max)(worst_owned, std::abs(ranks[r].coeffs[i] - serial.data()[i]) /
                                                  tol[static_cast<std::size_t>(g)]);
      }
    }
    CHECK(worst_owned <= 1.0);
    if (each_owns) {
      CHECK_FALSE(ranks[r].owned_dofs.empty());
      CHECK(ranks[r].calls == 1);
    }
  }
  CHECK(std::all_of(owners.begin(), owners.end(), [](int k) { return k == 1; }));
}

TEST_CASE(
    "mpi thb quasi_interpolate_distributed equals the serial THB quasi-interpolant on 2, 3 and 4 "
    "ranks") {
  const std::vector<ThbSpace> spaces = thb_spaces();
  for (std::size_t s = 0; s < spaces.size(); ++s) {
    for (const auto& value_rank : {std::size_t{1}, std::size_t{2}}) {
      for (const int n : kRankCounts) {
        INFO("space " << s << ", components " << value_rank << ", ranks " << n);
        const gr::Partition part =
            gr::partition_grid(spaces[s].grid(), n, gr::PartitionBackend::rcb);
        check_thb_qi_against_serial<double>(spaces[s], part, n, value_rank);
      }
    }
  }
}

TEST_CASE("mpi thb quasi_interpolate_distributed on a float root equals the serial one") {
  const th::THBSplineSpace<float> space = thb_test::refined_2d<float>(true);
  for (const auto& value_rank : {std::size_t{1}, std::size_t{2}}) {
    for (const int n : kRankCounts) {
      INFO("components " << value_rank << ", ranks " << n);
      const gr::Partition part = gr::partition_grid(space.grid(), n, gr::PartitionBackend::rcb);
      check_thb_qi_against_serial<float>(space, part, n, value_rank);
    }
  }
}

TEST_CASE("mpi thb quasi_interpolate_distributed with a rank that owns no cell") {
  const ThbSpace space = thb_test::refined_2d(true);
  const auto n_cells = static_cast<std::size_t>(space.grid().num_cells());
  Owners owner(n_cells, 0);
  std::fill(owner.begin() + static_cast<std::ptrdiff_t>(n_cells / 2), owner.end(), 1);
  const gr::Partition part(owner, 3);  // rank 2 owns nothing
  for (const auto& value_rank : {std::size_t{1}, std::size_t{2}}) {
    check_thb_qi_against_serial<double>(space, part, 3, value_rank, false);
    const std::vector<RankResult<double>> ranks = run_thb_qi<double>(space, part, 3, value_rank);
    CHECK(ranks[2].calls == 0);
    CHECK(ranks[2].owned_dofs.empty());
    CHECK(ranks[0].calls == 1);
    CHECK(ranks[1].calls == 1);
  }
}

TEST_CASE("mpi thb quasi_interpolate_distributed returns the coefficients of a THB spline") {
  // The defining property (Speleers and Manni): the quasi-interpolant of a THB spline's values is
  // its coefficient vector. Random coefficients in [-1, 1], the bound of test_thb_qi.cpp: the
  // functional's contraction, its weights, and the rounding of the spline's evaluation
  // (`value_error_units`).
  const std::vector<ThbSpace> spaces = thb_spaces();
  for (std::size_t s = 0; s < spaces.size(); ++s) {
    const ThbSpace& space = spaces[s];
    const auto n_dofs = static_cast<std::size_t>(space.num_total_basis());
    thb_test::Lcg rng(5 + s);
    NdArray<double> c({n_dofs, 2}, pantr::for_overwrite);
    for (std::size_t i = 0; i < c.size(); ++i) {
      c.data()[i] = 2.0 * rng.next() - 1.0;
    }
    const th::THBSpline<double> spline(space, c);
    const THBQuasiInterpolation<double> qi(space);
    for (const int n : kRankCounts) {
      INFO("space " << s << ", ranks " << n);
      const gr::Partition part = gr::partition_grid(space.grid(), n, gr::PartitionBackend::rcb);
      const std::vector<RankResult<double>> ranks =
          mt::run_ranks<RankResult<double>>(n, [&](mt::HubComm comm) {
            const auto ds = mp::distribute(space, part, comm);
            const th::THBSpline<double> q = mp::quasi_interpolate_distributed(
                ds, comm, [&](pantr::mdview<const double, 2> pts) { return spline.evaluate(pts); });
            RankResult<double> out;
            out.coeffs.assign(q.control_points().begin(), q.control_points().end());
            return out;
          });
      for (const RankResult<double>& r : ranks) {
        REQUIRE(r.coeffs.size() == c.size());
        for (std::size_t g = 0; g < n_dofs; ++g) {
          const double bound = thb_qi_tolerance(space, qi, static_cast<I64>(g), 1.0,
                                                thb_test::value_error_units(space));
          CHECK(bound < 1e-6);
          for (std::size_t k = 0; k < 2; ++k) {
            CHECK(std::abs(r.coeffs[g * 2 + k] - c.data()[g * 2 + k]) <= bound);
          }
        }
      }
    }
  }
}

TEST_CASE(
    "mpi quasi_interpolate_distributed sends its owned pairs in the shapes the Python fakes "
    "preset") {
  // The tensor-product one sends the coefficients as (k, rank); the THB one as (k,) for a
  // function of one value per point and (k, rank) otherwise (design/mpi.md: one call per payload).
  const bs::BsplineSpace<double> space = make_space<double>(tensor_cases()[0]);
  const OwnedSpy tensor_spy;
  const auto tensor_ds = mp::distribute(space, gr::Partition(Owners(25, 0), 1), tensor_spy);
  static_cast<void>(mp::quasi_interpolate_distributed(
      tensor_ds, tensor_spy,
      [](pantr::mdview<const double, 2> pts) { return block_values(pts, 1); }));
  CHECK(tensor_spy.shapes() == Shapes{{64}, {64, 1}});

  const ThbSpace thb = thb_test::refined_2d(true);
  const auto n_dofs = static_cast<std::size_t>(thb.num_total_basis());
  const auto n_cells = static_cast<std::size_t>(thb.grid().num_cells());
  for (const auto& value_rank : {std::size_t{1}, std::size_t{2}}) {
    const OwnedSpy spy;
    const auto ds = mp::distribute(thb, gr::Partition(Owners(n_cells, 0), 1), spy);
    static_cast<void>(mp::quasi_interpolate_distributed(
        ds, spy,
        [&](pantr::mdview<const double, 2> pts) { return block_values(pts, value_rank); }));
    const Shapes want =
        value_rank == 1 ? Shapes{{n_dofs}, {n_dofs}} : Shapes{{n_dofs}, {n_dofs, value_rank}};
    CHECK(spy.shapes() == want);
  }
}

TEST_CASE(
    "mpi quasi_interpolate_distributed refuses a function that returns another number of values") {
  const mp::SelfCommunicator comm;
  const auto one_short = [](pantr::mdview<const double, 2> pts) {
    return NdArray<double>(std::vector<std::size_t>{pts.extent(0) - 1});
  };
  const auto cube = [](pantr::mdview<const double, 2> pts) {
    return NdArray<double>(std::vector<std::size_t>{pts.extent(0), 1, 1});
  };
  const auto message = [](std::size_t points) {
    return "quasi_interpolate: the function returned another number of values than the " +
           std::to_string(points) + " points.";
  };
  const bs::BsplineSpace<double> space = make_space<double>(tensor_cases()[2]);
  const auto ds = mp::distribute(space, rcb_of(space, 1), comm);
  const std::string tensor_msg = message(bs::QuasiInterpolation<double>(space).points().shape()[0]);
  THB_THROWS_WITH_AS(mp::quasi_interpolate_distributed(ds, comm, one_short), tensor_msg.c_str(),
                     value_error);
  THB_THROWS_WITH_AS(mp::quasi_interpolate_distributed(ds, comm, cube), tensor_msg.c_str(),
                     value_error);

  const ThbSpace thb = thb_test::refined_2d(true);
  const auto n_cells = static_cast<std::size_t>(thb.grid().num_cells());
  const auto thb_ds = mp::distribute(thb, gr::Partition(Owners(n_cells, 0), 1), comm);
  const std::string thb_msg = message(THBQuasiInterpolation<double>(thb).points().shape()[0]);
  THB_THROWS_WITH_AS(mp::quasi_interpolate_distributed(thb_ds, comm, one_short), thb_msg.c_str(),
                     value_error);
  THB_THROWS_WITH_AS(mp::quasi_interpolate_distributed(thb_ds, comm, cube), thb_msg.c_str(),
                     value_error);
}

// ---- The distributed algorithms over each route to a communicator -------------------------------

enum class Route { direct, any, table };

/// Call `body(comm)` with one rank's communicator reached the given way.
template <class Body>
auto with_route(const mt::HubComm& hub, Route route, Body&& body) {
  if (route == Route::direct) {
    return body(hub);
  }
  if (route == Route::any) {
    return body(mp::AnyCommunicator(hub));
  }
  return body(via_table(hub));
}

TEST_CASE(
    "mpi the algorithms give the same results over a type-erased communicator and a C table") {
  // Oracle: the serial results, as in the tests above, now with the collectives going through
  // `AnyCommunicator(comm)` and `from_table(make_comm_table(comm))`: a gather, a reduction and a
  // pair of gathers each cross the table; and the partition of `partition_from_owner` of the owned
  // cells of a partition is that partition.
  const TensorCase& c = tensor_cases()[0];
  const bs::BsplineSpace<double> space = make_space<double>(c);
  const std::size_t value_rank = 2;
  const int n = 3;
  const gr::Partition part = rcb_of(space, n);

  const bs::Bspline<double> serial_interp = serial_interpolate<double>(space, value_rank, {}, {});
  const bs::Bspline<double> serial_l2 = bs::l2_project(
      space,
      [&](const pantr::quad::PointsLattice<double>& lattice) {
        return lattice_values<double>(lattice, value_rank);
      },
      std::optional<std::vector<long long>>(quadrature_of(c, 0)));
  const bs::Bspline<double> serial_q = serial_qi<double>(space, value_rank);
  const std::vector<std::vector<double>> nodes = nodes_or_greville(space, {});
  const std::vector<std::span<const double>> spans(nodes.begin(), nodes.end());
  const double tol_interp = coefficient_tolerance(bs::TensorCollocation<double>(space, spans),
                                                  max_abs_of(serial_interp.control_points()));
  const std::vector<long long> n_quad = quadrature_of(c, 0);
  const std::vector<std::pair<bool, bool>> no_flags = boundary_flags(space.dim(), false);
  const bs::L2Projection<double> proj(space, n_quad, bs::L2Quadrature::gauss_legendre, no_flags);
  const L2Inputs<double> in = l2_inputs(proj, value_rank);
  const double tol_l2 = l2_tolerance(proj, abs_load_max(proj, in.values, value_rank),
                                     max_abs_of(serial_l2.control_points()), n);
  const double tol_qi = qi_tolerance(space, sampled_max<double>(space, value_rank),
                                     max_abs_of(serial_q.control_points()));

  struct Everything {
    RankResult<double> interp;
    RankResult<double> l2;
    RankResult<double> qi;
    Owners rebuilt;
    Owners created;
  };
  for (const Route route : {Route::direct, Route::any, Route::table}) {
    INFO("route " << static_cast<int>(route));
    const std::vector<Everything> ranks = mt::run_ranks<Everything>(n, [&](mt::HubComm hub) {
      return with_route(hub, route, [&](const auto& comm) {
        const auto ds = mp::distribute(space, part, comm);
        Everything out;
        const auto fill = [&](RankResult<double>& r, const bs::Bspline<double>& b) {
          r.coeffs.assign(b.control_points().begin(), b.control_points().end());
          r.owned_dofs = owned_dofs_of(ds);
        };
        fill(out.interp,
             mp::interpolate_distributed(ds, comm, [&](pantr::mdview<const double, 2> p) {
               return block_values(p, value_rank);
             }));
        fill(out.qi,
             mp::quasi_interpolate_distributed(ds, comm, [&](pantr::mdview<const double, 2> p) {
               return block_values(p, value_rank);
             }));
        std::vector<std::span<const double>> traces;
        const NdArray<double> l = mp::l2_project_values_distributed(
            comm, proj, ds.partition().cell_owner(), std::span<const double>(in.values.flat()),
            value_rank, traces);
        out.l2.coeffs.assign(l.data(), l.data() + l.size());
        out.l2.owned_dofs = owned_dofs_of(ds);
        const std::vector<I64> mine = ds.owned_cells();
        out.rebuilt = owners_of(mp::partition_from_owner(comm, std::span<const I64>(mine),
                                                         static_cast<I64>(part.n_cells())));
        out.created = owners_of(mp::create_distributed_space(
                                    space, comm,
                                    [&](const bs::BsplineSpace<double>&, std::int64_t) {
                                      return rank_dependent(
                                          comm.rank(), n, static_cast<std::size_t>(part.n_cells()));
                                    })
                                    .partition());
        return out;
      });
    });
    std::vector<RankResult<double>> interp;
    std::vector<RankResult<double>> l2;
    std::vector<RankResult<double>> qi;
    for (const Everything& e : ranks) {
      interp.push_back(e.interp);
      l2.push_back(e.l2);
      qi.push_back(e.qi);
      CHECK(e.rebuilt == owners_of(part));
      CHECK(e.created == owners_of(rank_dependent(0, n, static_cast<std::size_t>(part.n_cells()))));
    }
    check_against_serial<double>(interp, serial_interp.control_points(), value_rank,
                                 count_dofs(space), tol_interp);
    check_against_serial<double>(l2, serial_l2.control_points(), value_rank, count_dofs(space),
                                 tol_l2);
    check_against_serial<double>(qi, serial_q.control_points(), value_rank, count_dofs(space),
                                 tol_qi);
  }
}

TEST_CASE(
    "mpi partition_from_owner rebuilds a partition with inactive cells from its owned cells") {
  // Round trip: the owned cells of a partition, gathered, assemble to that partition, the
  // inactive cells (-1) being the ones nobody sends.
  const Owners owner{0, 2, -1, 1, 1, -1, 0, 2, 2, 1, -1, 0};
  const gr::Partition part(owner, 3);
  const std::vector<gr::Partition> rebuilt = mt::run_ranks<gr::Partition>(3, [&](mt::HubComm comm) {
    const I64Vec mine = part.owned_cells(comm.rank());
    return mp::partition_from_owner(comm, std::span<const I64>(mine), 12);
  });
  for (const gr::Partition& p : rebuilt) {
    CHECK(owners_of(p) == owner);
    CHECK(p.n_parts() == 3);
  }
}

}  // namespace
