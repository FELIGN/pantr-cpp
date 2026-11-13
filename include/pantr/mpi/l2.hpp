/// @file l2.hpp
/// @brief Distributed L2 projection onto tensor-product B-spline spaces. MPI-free.
///
/// The load of an L2 projection is a sum over cells, so it splits over the cell partition: each
/// rank contracts the values at the quadrature points of the cells it owns (the others are set to
/// zero) into a load tensor, one `allreduce_sum` adds the ranks' loads, and every rank applies the
/// boundary rows and runs the same Kronecker mass solve (`bspline::L2Projection::solve_load`).
/// Every quadrature point lies in exactly one cell, so the summed load is the serial load up to the
/// order of summation.
///
/// Each rank is handed the values on the whole quadrature lattice, as the previous release
/// evaluated them, and masks them; evaluating only the owned cells is a later speed-up.
/// Design note: design/mpi.md.
#pragma once

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/fit.hpp"
#include "pantr/bspline/l2_project.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/mpi/communicator.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace pantr::mpi {

/// The flags of the quadrature points of an L2 projection that lie in cells `rank` owns.
///
/// The global quadrature nodes of direction `d` are concatenated per interval, `n_d` per
/// interval, so the point `(q_0, ..., q_{k-1})` lies in the cell whose interval index in direction
/// `d` is `q_d / n_d`.
///
/// Args:
///     proj (const bspline::L2Projection<T>&): The projection (its quadrature lattice).
///     cell_owner (std::span<const std::int32_t>): The owner of every cell, C order over the
///         space's intervals.
///     rank (int): This rank.
///
/// Returns:
///     std::vector<std::uint8_t>: One flag per lattice point, C order.
///
/// Raises:
///     value_error: Another number of owners than cells.
template <StorageFloat T>
[[nodiscard]] std::vector<std::uint8_t> owned_quadrature_mask(
    const bspline::L2Projection<T>& proj, std::span<const std::int32_t> cell_owner, int rank) {
  const bspline::BsplineSpace<T>& space = proj.space();
  const std::size_t dim = space.dim();
  std::vector<std::size_t> n_int(dim);
  std::vector<std::size_t> per(dim);
  const std::vector<std::size_t> grid = proj.grid_shape();
  for (std::size_t d = 0; d < dim; ++d) {
    n_int[d] = static_cast<std::size_t>(space.spaces()[d].num_intervals());
    per[d] = grid[d] / n_int[d];
  }
  if (cell_owner.size() != checked_product(n_int)) {
    throw value_error("partition has " + std::to_string(cell_owner.size()) + " cells; expected " +
                      std::to_string(checked_product(n_int)) + " (the global space's cell count).");
  }
  const std::size_t n_pts = checked_product(grid);
  std::vector<std::uint8_t> mask(n_pts);
  std::vector<std::size_t> q(dim, 0);
  for (std::size_t i = 0; i < n_pts; ++i) {
    std::size_t cell = 0;
    for (std::size_t d = 0; d < dim; ++d) {
      cell = cell * n_int[d] + q[d] / per[d];
    }
    mask[i] = cell_owner[cell] == rank ? 1 : 0;
    for (std::size_t d = dim; d-- > 0;) {
      if (++q[d] < grid[d]) {
        break;
      }
      q[d] = 0;
    }
  }
  return mask;
}

/// Project values given on the quadrature lattice and on the boundary traces, summing the loads
/// of the owned cells over the ranks: the staged form.
///
/// Args:
///     comm (const C&): The communicator.
///     proj (const bspline::L2Projection<T>&): The projection, identical on every rank.
///     cell_owner (std::span<const std::int32_t>): The partition's owners, identical on every rank.
///     values (std::span<const Cm>): The function on the whole quadrature lattice, row-major over
///         `(*proj.grid_shape(), rank)`.
///     rank_dim (std::size_t): Components, `>= 1`.
///     trace_values (const std::vector<std::span<const Cm>>&): The function on each trace's
///         lattice, row-major over `(*trace_shape, rank)`, one per trace.
///     tol (std::optional<double>): Truncation tolerance; `100 * eps(T)` when absent.
///
/// Returns:
///     NdArray<Cm>: The coefficients, `(n_0, ..., n_{d-1}, rank)`, identical on every rank.
///
/// Raises:
///     value_error: Another number of values or traces (before the collective); the reduced load
///         has another shape than this rank's (after it), see `bspline::L2Projection`.
template <Communicator C, StorageFloat T, StorageFloat Cm>
[[nodiscard]] NdArray<Cm> l2_project_values_distributed(
    const C& comm, const bspline::L2Projection<T>& proj, std::span<const std::int32_t> cell_owner,
    std::span<const Cm> values, std::size_t rank_dim,
    const std::vector<std::span<const Cm>>& trace_values,
    std::optional<double> tol = std::nullopt) {
  const std::size_t n_pts = checked_product(proj.grid_shape());
  if (rank_dim == 0 || values.size() != n_pts * rank_dim) {
    throw value_error("l2_project: expected " + std::to_string(n_pts) + " x " +
                      std::to_string(rank_dim) + " values on the quadrature lattice, got " +
                      std::to_string(values.size()) + ".");
  }
  if (trace_values.size() != proj.traces().size()) {
    throw value_error("L2Projection: expected " + std::to_string(proj.traces().size()) +
                      " trace value arrays, got " + std::to_string(trace_values.size()) + ".");
  }
  const std::vector<std::uint8_t> mask = owned_quadrature_mask(proj, cell_owner, comm.rank());
  const bspline::BsplineSpace<T>& space = proj.space();
  std::vector<std::size_t> nb;
  for (const bspline::BsplineSpace1D<T>& s : space.spaces()) {
    nb.push_back(static_cast<std::size_t>(s.num_basis()));
  }
  const std::size_t n_coeffs = checked_product(nb);
  std::vector<std::size_t> shape = nb;
  shape.push_back(rank_dim);
  std::vector<Cm> local(n_coeffs * rank_dim);
  std::vector<Cm> comp(n_pts);
  for (std::size_t r = 0; r < rank_dim; ++r) {
    for (std::size_t i = 0; i < n_pts; ++i) {
      comp[i] = mask[i] != 0 ? values[i * rank_dim + r] : Cm(0);
    }
    const NdArray<Cm> l = proj.template load<Cm>(std::span<const Cm>(comp));
    for (std::size_t i = 0; i < n_coeffs; ++i) {
      local[i * rank_dim + r] = l.data()[i];
    }
  }
  const NdArray<Cm> total =
      comm.allreduce_sum(std::span<const Cm>(local), std::span<const std::size_t>(shape));
  if (!std::equal(total.shape().begin(), total.shape().end(), shape.begin(), shape.end())) {
    throw value_error("Reduced load has shape " + py_repr(total.shape()) + ", expected " +
                      py_repr(std::span<const std::size_t>(shape)) +
                      "; this indicates 'func' returned an inconsistent shape across ranks.");
  }
  const double resolved = bspline::default_fit_tol<T>(tol);
  NdArray<Cm> out(std::vector<std::size_t>(shape), for_overwrite);
  std::vector<std::vector<Cm>> trace_comp(trace_values.size());
  for (std::size_t r = 0; r < rank_dim; ++r) {
    NdArray<Cm> load(std::vector<std::size_t>(nb), for_overwrite);
    for (std::size_t i = 0; i < n_coeffs; ++i) {
      load.data()[i] = total.data()[i * rank_dim + r];
    }
    std::vector<std::span<const Cm>> spans;
    for (std::size_t t = 0; t < trace_values.size(); ++t) {
      const std::span<const Cm> tv = trace_values[t];
      if (tv.size() % rank_dim != 0) {
        throw value_error("trace " + std::to_string(t) + " holds " + std::to_string(tv.size()) +
                          " values, not a multiple of " + std::to_string(rank_dim) + ".");
      }
      trace_comp[t].resize(tv.size() / rank_dim);
      for (std::size_t i = 0; i < trace_comp[t].size(); ++i) {
        trace_comp[t][i] = tv[i * rank_dim + r];
      }
      spans.emplace_back(trace_comp[t]);
    }
    const NdArray<Cm> x = proj.template solve_load<Cm>(std::move(load), spans, resolved);
    for (std::size_t i = 0; i < n_coeffs; ++i) {
      out.data()[i * rank_dim + r] = x.data()[i];
    }
  }
  return out;
}

}  // namespace pantr::mpi
