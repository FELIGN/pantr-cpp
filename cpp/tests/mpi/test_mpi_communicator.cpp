// The distributed algorithms over real MPI: MpiCommunicator's collectives and the partition
// broadcast, run under `mpiexec -n 2` and `-n 3` (ctest `mpi.*`). Every check runs on every rank,
// and the process's exit status is the reduction of every rank's failures, so a failure on any
// rank fails the run.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/grid/partition.hpp"
#include "pantr/grid/tensor_product_grid.hpp"
#include "pantr/mpi/collocation.hpp"
#include "pantr/mpi/communicator.hpp"
#include "pantr/mpi/distributed_space.hpp"
#include "pantr/mpi/mpi_communicator.hpp"
#include "pantr/mpi/partition_from_owner.hpp"
#include "pantr/mpi/quasi_interpolate.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace {

using pantr::mpi::Block;
using pantr::mpi::MpiCommunicator;

MpiCommunicator world() { return MpiCommunicator(MPI_COMM_WORLD); }

pantr::bspline::BsplineSpace<double> square_space(long long degree, long long cells) {
  std::vector<pantr::bspline::BsplineSpace1D<double>> dirs;
  for (int d = 0; d < 2; ++d) {
    std::vector<double> knots;
    for (long long i = 0; i <= degree; ++i) {
      knots.push_back(0.0);
    }
    for (long long i = 1; i < cells; ++i) {
      knots.push_back(static_cast<double>(i) / static_cast<double>(cells));
    }
    for (long long i = 0; i <= degree; ++i) {
      knots.push_back(1.0);
    }
    dirs.emplace_back(knots, static_cast<int>(degree));
  }
  return pantr::bspline::BsplineSpace<double>(std::move(dirs));
}

}  // namespace

TEST_CASE("mpi communicator reports the world's rank and size") {
  const MpiCommunicator c = world();
  int rank = -1;
  int size = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  CHECK(c.rank() == rank);
  CHECK(c.size() == size);
}

TEST_CASE("mpi allgatherv returns every rank's block with its shape, in rank order") {
  const MpiCommunicator c = world();
  // Rank r sends r rows of two columns holding (r, k): ranks 0 sends nothing.
  std::vector<double> mine;
  for (int k = 0; k < c.rank(); ++k) {
    mine.push_back(c.rank());
    mine.push_back(k);
  }
  const Block<double> b{mine, {static_cast<std::size_t>(c.rank()), 2}};
  const auto g = c.allgatherv(b);
  REQUIRE(g.count() == static_cast<std::size_t>(c.size()));
  for (int r = 0; r < c.size(); ++r) {
    CHECK(g.shapes[static_cast<std::size_t>(r)] ==
          std::vector<std::size_t>{static_cast<std::size_t>(r), 2});
    const auto blk = g.block(static_cast<std::size_t>(r));
    for (int k = 0; k < r; ++k) {
      CHECK(blk[static_cast<std::size_t>(2 * k)] == r);
      CHECK(blk[static_cast<std::size_t>(2 * k + 1)] == k);
    }
  }
}

TEST_CASE("mpi allreduce_sum adds each rank's tensor in float and double") {
  const MpiCommunicator c = world();
  const std::vector<std::size_t> shape{2, 3};
  std::vector<double> d(6);
  std::vector<float> f(6);
  for (std::size_t i = 0; i < 6; ++i) {
    d[i] = static_cast<double>(i) + c.rank();
    f[i] = static_cast<float>(i) + static_cast<float>(c.rank());
  }
  const auto rd = c.allreduce_sum(std::span<const double>(d), shape);
  const auto rf = c.allreduce_sum(std::span<const float>(f), shape);
  const int n = c.size();
  for (std::size_t i = 0; i < 6; ++i) {
    // Small integers: every partial sum is exact in either type.
    const double expected = static_cast<double>(n) * static_cast<double>(i) + n * (n - 1) / 2.0;
    CHECK(rd.data()[i] == expected);
    CHECK(static_cast<double>(rf.data()[i]) == expected);
  }
}

TEST_CASE("mpi allreduce_sum over different sizes raises on every rank instead of hanging") {
  const MpiCommunicator c = world();
  std::vector<double> d(static_cast<std::size_t>(c.rank() + 1), 1.0);
  const std::vector<std::size_t> shape{d.size()};
  if (c.size() > 1) {
    CHECK_THROWS_AS(static_cast<void>(c.allreduce_sum(std::span<const double>(d), shape)),
                    pantr::value_error);
  }
}

TEST_CASE("mpi broadcast_partition gives every rank rank 0's partition") {
  const MpiCommunicator c = world();
  // Each rank's own computation would differ (a rank-dependent owner pattern); only rank 0's runs.
  int calls = 0;
  const auto p = pantr::mpi::broadcast_partition(c, [&] {
    ++calls;
    std::vector<std::int32_t> owner(10);
    for (std::size_t i = 0; i < owner.size(); ++i) {
      owner[i] = static_cast<std::int32_t>((i + static_cast<std::size_t>(c.rank())) %
                                           static_cast<std::size_t>(c.size()));
    }
    return pantr::grid::Partition(std::move(owner), c.size());
  });
  CHECK(calls == (c.rank() == 0 ? 1 : 0));
  for (std::size_t i = 0; i < 10; ++i) {
    CHECK(p.cell_owner()[i] == static_cast<std::int32_t>(i % static_cast<std::size_t>(c.size())));
  }
}

TEST_CASE("mpi broadcast_partition: rank 0's failure raises on every rank") {
  const MpiCommunicator c = world();
  if (c.size() > 1) {
    bool threw = false;
    try {
      static_cast<void>(pantr::mpi::broadcast_partition(
          c, [&]() -> pantr::grid::Partition { throw pantr::type_error("rank 0 failed"); }));
    } catch (const pantr::type_error&) {
      threw = c.rank() == 0;
    } catch (const pantr::value_error&) {
      threw = c.rank() != 0;
    }
    CHECK(threw);
  }
}

TEST_CASE("mpi partition_from_owner assembles the ranks' cells") {
  const MpiCommunicator c = world();
  // Rank r owns cells r, r + size, ...; the last cell is owned by nobody.
  const std::int64_t n_cells = 3 * c.size() + 1;
  std::vector<std::int64_t> mine;
  for (std::int64_t i = c.rank(); i < n_cells - 1; i += c.size()) {
    mine.push_back(i);
  }
  const auto p = pantr::mpi::partition_from_owner(c, mine, n_cells);
  for (std::int64_t i = 0; i < n_cells - 1; ++i) {
    CHECK(p.cell_owner()[static_cast<std::size_t>(i)] == i % c.size());
  }
  CHECK(p.cell_owner()[static_cast<std::size_t>(n_cells - 1)] == -1);
}

TEST_CASE("mpi distributed interpolation equals the serial interpolant") {
  const MpiCommunicator c = world();
  const auto space = square_space(2, 4);
  const auto ds = pantr::mpi::create_distributed_space(
      space, c, [](const pantr::bspline::BsplineSpace<double>& /*s*/, std::int64_t n) {
        const std::vector<double> bp{0.0, 0.25, 0.5, 0.75, 1.0};
        return pantr::grid::partition_grid(pantr::grid::TensorProductGrid({bp, bp}), n,
                                           pantr::grid::PartitionBackend::rcb);
      });
  const auto f = [](pantr::mdview<const double, 2> x) {
    pantr::NdArray<double> v({x.extent(0)});
    for (std::size_t i = 0; i < x.extent(0); ++i) {
      v.data()[i] = std::sin(x(i, 0)) * std::cos(2.0 * x(i, 1));
    }
    return v;
  };
  const auto dist = pantr::mpi::interpolate_distributed(ds, c, f);
  const auto serial = pantr::mpi::interpolate_distributed(
      pantr::mpi::distribute(space, pantr::grid::Partition(std::vector<std::int32_t>(16, 0), 1),
                             pantr::mpi::SelfCommunicator{}),
      pantr::mpi::SelfCommunicator{}, f);
  const auto a = dist.control_points();
  const auto b = serial.control_points();
  REQUIRE(a.size() == b.size());
  // The same Kronecker solve on the same values: only the gather moved them.
  for (std::size_t i = 0; i < a.size(); ++i) {
    CHECK(a[i] == b[i]);
  }
}

int main(int argc, char** argv) {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  doctest::Context ctx(argc, argv);
  const int local = ctx.run();
  int failed = 0;
  MPI_Allreduce(&local, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return failed != 0 ? 1 : 0;
}
