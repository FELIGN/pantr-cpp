/// @file quasi_interpolate.hpp
/// @brief Distributed quasi-interpolation onto B-spline and THB spaces. MPI-free.
///
/// A quasi-interpolant's coefficient of a basis function depends only on the function's values on
/// its support, so each rank quasi-interpolates on its windowed local space, keeps the
/// coefficients of the DOFs it owns, and one `allgather_owned` assembles the global coefficient
/// field from every rank's `(global DOF, coefficient)` pairs.
/// Design note: design/mpi.md.
#pragma once

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/quasi_interpolate.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/mpi/communicator.hpp"
#include "pantr/mpi/distributed_space.hpp"
#include "pantr/thb/quasi_interpolate.hpp"
#include "pantr/thb/spline.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::mpi {

/// One rank's owned coefficients: the global DOF ids and their coefficients.
///
/// Attributes:
///     dofs (std::vector<std::int64_t>): Global DOF ids, one per owned local DOF, in local order.
///     coeffs (std::vector<double>): Row-major `(dofs.size(), rank)`.
///     rank (std::size_t): Components per DOF.
///     scalar (bool): Whether the field is scalar (sent as `(k,)` rather than `(k, 1)`).
struct OwnedCoefficients {
  std::vector<std::int64_t> dofs;
  std::vector<double> coeffs;
  std::size_t rank;
  bool scalar;
};

/// Keep the coefficients of the DOFs this rank owns.
///
/// Args:
///     local_to_global_dof (std::span<const std::int64_t>): Per local DOF, its global id.
///     owned_dof_mask (std::span<const std::uint8_t>): Per local DOF, whether this rank owns it.
///     coeffs (std::span<const double>): Row-major `(n_local, rank)`.
///     rank (std::size_t): Components, `>= 1`.
///     scalar (bool): See `OwnedCoefficients`.
///
/// Returns:
///     OwnedCoefficients: The owned pairs, in local order.
///
/// Raises:
///     value_error: The maps and the coefficients disagree on the local DOF count.
[[nodiscard]] inline OwnedCoefficients owned_coefficients(
    std::span<const std::int64_t> local_to_global_dof, std::span<const std::uint8_t> owned_dof_mask,
    std::span<const double> coeffs, std::size_t rank, bool scalar) {
  const std::size_t n = local_to_global_dof.size();
  if (owned_dof_mask.size() != n || rank == 0 || coeffs.size() != n * rank) {
    throw value_error("owned_coefficients: " + std::to_string(n) + " local DOFs, " +
                      std::to_string(owned_dof_mask.size()) + " mask entries and " +
                      std::to_string(coeffs.size()) + " coefficients disagree.");
  }
  OwnedCoefficients out{{}, {}, rank, scalar};
  for (std::size_t i = 0; i < n; ++i) {
    if (owned_dof_mask[i] != 0) {
      out.dofs.push_back(local_to_global_dof[i]);
      out.coeffs.insert(out.coeffs.end(), coeffs.begin() + static_cast<std::ptrdiff_t>(i * rank),
                        coeffs.begin() + static_cast<std::ptrdiff_t>((i + 1) * rank));
    }
  }
  return out;
}

/// The global coefficient field assembled from every rank's owned pairs.
///
/// Attributes:
///     coeffs (std::vector<double>): Row-major `(n_global, rank)`; a DOF no rank sent is `0`.
///     rank (std::size_t): Components, read from the first non-empty block.
///     scalar (bool): Whether that block was one-dimensional.
struct GlobalCoefficients {
  std::vector<double> coeffs;
  std::size_t rank;
  bool scalar;
};

/// Gather every rank's owned pairs and assemble the global coefficient field.
///
/// The number of components comes from the first block that holds a DOF; a scalar field when no
/// block does. A DOF no rank owns (one supported only on inactive cells) is left `0`: the previous
/// release left it uninitialized.
///
/// Args:
///     comm (const C&): The communicator.
///     n_global (std::int64_t): The global DOF count.
///     mine (const OwnedCoefficients&): This rank's pairs.
///
/// Returns:
///     GlobalCoefficients: The field, identical on every rank.
///
/// Raises:
///     value_error: The gather returned another number of blocks than ranks, a block's DOFs and
///         coefficients disagree, a DOF id is outside `[0, n_global)`, or two ranks send the same
///         DOF.
template <Communicator C>
[[nodiscard]] GlobalCoefficients gather_owned_coefficients(const C& comm, std::int64_t n_global,
                                                           const OwnedCoefficients& mine) {
  std::vector<std::size_t> cshape{mine.dofs.size()};
  if (!mine.scalar) {
    cshape.push_back(mine.dofs.empty() ? 0 : mine.rank);
  }
  const Block<std::int64_t> dofs = block_1d<std::int64_t>(mine.dofs);
  const Block<double> coeffs{mine.coeffs, std::move(cshape)};
  const auto [gd, gc] = comm.allgather_owned(dofs, coeffs);
  detail::check_gather_count(gd, comm.size());
  detail::check_gather_count(gc, comm.size());
  GlobalCoefficients out{{}, 1, true};
  for (std::size_t r = 0; r < gc.count(); ++r) {
    const std::vector<std::size_t>& s = gc.shapes[r];
    if (!s.empty() && s[0] != 0) {
      out.scalar = s.size() == 1;
      out.rank = s.size() >= 2 ? s[1] : 1;
      break;
    }
  }
  const auto n = static_cast<std::size_t>(n_global);
  out.coeffs.assign(n * out.rank, 0.0);
  std::vector<std::uint8_t> seen(n, 0);
  for (std::size_t r = 0; r < gd.count(); ++r) {
    const std::span<const std::int64_t> ids = gd.block(r);
    if (ids.empty()) {
      continue;
    }
    const std::span<const double> c = gc.block(r);
    if (c.size() != ids.size() * out.rank) {
      throw value_error("rank " + std::to_string(r) + " sent " + std::to_string(ids.size()) +
                        " DOFs and " + std::to_string(c.size()) + " coefficients; expected " +
                        std::to_string(out.rank) + " per DOF.");
    }
    for (std::size_t k = 0; k < ids.size(); ++k) {
      const std::int64_t g = ids[k];
      if (g < 0 || g >= n_global) {
        throw value_error("rank " + std::to_string(r) + " sent DOF " + std::to_string(g) +
                          ", outside [0, " + std::to_string(n_global) + ").");
      }
      const auto gi = static_cast<std::size_t>(g);
      if (seen[gi] != 0) {
        throw value_error("DOF " + std::to_string(g) + " is owned by more than one rank.");
      }
      seen[gi] = 1;
      std::copy(c.begin() + static_cast<std::ptrdiff_t>(k * out.rank),
                c.begin() + static_cast<std::ptrdiff_t>((k + 1) * out.rank),
                out.coeffs.begin() + static_cast<std::ptrdiff_t>(gi * out.rank));
    }
  }
  return out;
}

/// Quasi-interpolate a function onto a distributed tensor-product space: each rank on its local
/// space, then one gather.
///
/// Args:
///     ds (const DistributedSpace<bspline::BsplineSpace<T>>&): The distributed space.
///     comm (const C&): Its communicator.
///     f (F&&): As for `bspline::quasi_interpolate`; called only on a rank that owns cells.
///
/// Returns:
///     bspline::Bspline<T>: The global quasi-interpolant, identical on every rank; coefficients of
///     shape `(*num_basis, rank)`.
template <Communicator C, StorageFloat T, class F>
  requires std::invocable<F&, mdview<const double, 2>>
[[nodiscard]] bspline::Bspline<T> quasi_interpolate_distributed(
    const DistributedSpace<bspline::BsplineSpace<T>>& ds, const C& comm, F&& f) {
  OwnedCoefficients mine{{}, {}, 1, false};
  if (ds.local()) {
    const auto& local = *ds.local();
    const bspline::Bspline<T> b = bspline::quasi_interpolate(local.space, f);
    const auto cp = b.control_points();
    const std::size_t n_local = local.local_to_global_dof.size();
    std::vector<double> c(cp.size());
    std::transform(cp.begin(), cp.end(), c.begin(), [](T v) { return static_cast<double>(v); });
    mine = owned_coefficients(local.local_to_global_dof, local.owned_dof_mask, c,
                              cp.size() / n_local, false);
  }
  const bspline::BsplineSpace<T>& space = ds.global_space();
  std::vector<std::size_t> shape;
  for (const bspline::BsplineSpace1D<T>& s : space.spaces()) {
    shape.push_back(static_cast<std::size_t>(s.num_basis()));
  }
  const GlobalCoefficients g =
      gather_owned_coefficients(comm, static_cast<std::int64_t>(checked_product(shape)), mine);
  shape.push_back(g.rank);
  NdArray<T> cp(std::move(shape), for_overwrite);
  std::transform(g.coeffs.begin(), g.coeffs.end(), cp.data(),
                 [](double v) { return static_cast<T>(v); });
  return bspline::Bspline<T>(space, std::move(cp), false);
}

/// Quasi-interpolate a function onto a distributed THB space (Speleers-Manni functionals): each
/// rank on its local space, then one gather.
///
/// Args:
///     ds (const DistributedSpace<thb::THBSplineSpace<T>>&): The distributed space.
///     comm (const C&): Its communicator.
///     f (F&&): Called once per rank that owns cells as `f(points)` on the `(M, dim)` sample points
///         of its local space (`mdview<const double, 2>`); returns an `NdArray<double>` of `M`
///         values or `(M, rank)`.
///
/// Returns:
///     thb::THBSpline<T>: The global quasi-interpolant, identical on every rank.
///
/// Raises:
///     value_error: `f` returned another number of values; see `gather_owned_coefficients`.
template <Communicator C, StorageFloat T, class F>
  requires std::invocable<F&, mdview<const double, 2>>
[[nodiscard]] thb::THBSpline<T> quasi_interpolate_distributed(
    const DistributedSpace<thb::THBSplineSpace<T>>& ds, const C& comm, F&& f) {
  OwnedCoefficients mine{{}, {}, 1, true};
  if (ds.local()) {
    const auto& local = *ds.local();
    const thb::THBQuasiInterpolation<T> qi(local.space);
    const NdArray<double> pts = qi.points();
    const NdArray<double> values = f(pts.template view<2>());
    const std::size_t m = pts.shape()[0];
    if ((values.rank() != 1 && values.rank() != 2) || values.shape()[0] != m) {
      throw value_error(
          "quasi_interpolate: the function returned another number of values "
          "than the " +
          std::to_string(m) + " points.");
    }
    const std::size_t rank = values.rank() == 1 ? 1 : values.shape()[1];
    const NdArray<double> c = qi.coefficients(mdview<const double, 2>(values.data(), m, rank));
    mine = owned_coefficients(local.local_to_global_dof, local.owned_dof_mask, c.flat(), rank,
                              values.rank() == 1);
  }
  const GlobalCoefficients g =
      gather_owned_coefficients(comm, ds.global_space().num_total_basis(), mine);
  NdArray<double> cp({static_cast<std::size_t>(ds.global_space().num_total_basis()), g.rank},
                     for_overwrite);
  std::copy(g.coeffs.begin(), g.coeffs.end(), cp.data());
  return thb::THBSpline<T>(ds.global_space(), std::move(cp));
}

}  // namespace pantr::mpi
