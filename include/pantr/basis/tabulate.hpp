/// @file tabulate.hpp
/// @brief Tensor-product tabulation of the one-dimensional bases, at scattered points or on a
/// lattice.
#pragma once

#include "pantr/basis/bernstein.hpp"
#include "pantr/basis/cardinal_bspline.hpp"
#include "pantr/basis/detail/tabulate_1d.hpp"
#include "pantr/basis/lagrange.hpp"
#include "pantr/basis/lagrange_variant.hpp"
#include "pantr/basis/legendre.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/quad/points_lattice.hpp"

#include <concepts>
#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::basis {

/// A one-dimensional polynomial basis family. The values are the wire values the Python binding
/// passes.
enum class Family : int {
  bernstein = 0,         ///< Bernstein polynomials (`tabulate_bernstein_1d`).
  cardinal_bspline = 1,  ///< Central cardinal B-splines (`tabulate_cardinal_bspline_1d`).
  legendre = 2,          ///< Orthonormal shifted Legendre polynomials (`tabulate_legendre_1d`).
  lagrange = 3,          ///< Lagrange polynomials on a node family (`tabulate_lagrange_1d`).
};

/// Convert a wire value to a `Family`.
///
/// Raises:
///     value_error: `value` names no family.
[[nodiscard]] inline Family to_family(int value) {
  if (value < 0 || value > 3) {
    throw value_error("no basis family has the value " + std::to_string(value));
  }
  return static_cast<Family>(value);
}

/// Operations one point of a family's 1D tabulation costs, as its kernel estimates it (the
/// parallel decision, which the Python binding also reads to release the GIL only then).
///
/// Args:
///     family (Family): The basis family.
///     degree (int): Degree, `>= 0`.
///
/// Returns:
///     double: The estimate, at least 1.
[[nodiscard]] inline double ops_per_point(Family family, int degree) {
  switch (family) {
    case Family::bernstein:
      return detail::bernstein_ops_per_point(degree);
    case Family::cardinal_bspline:
      return detail::cardinal_bspline_ops_per_point(degree);
    case Family::legendre:
      return detail::legendre_ops_per_point(degree);
    case Family::lagrange:
      return detail::lagrange_ops_per_point(degree);
  }
  throw value_error("unknown basis family");
}

/// Evaluate a one-dimensional basis of a family at `pts` into `out`.
///
/// Args:
///     family (Family): The basis family.
///     degree (int): Degree, `>= 0`.
///     variant (LagrangeVariant): Node family; read for `Family::lagrange` only.
///     pts (std::span<const T>): The points.
///     out (mdview<T, 2>): Output, extents `(pts.size(), degree + 1)`.
///
/// Raises:
///     value_error: As the family's own function.
template <StorageFloat T>
void tabulate_1d(Family family, int degree, LagrangeVariant variant, std::span<const T> pts,
                 mdview<T, 2> out) {
  switch (family) {
    case Family::bernstein:
      tabulate_bernstein_1d<T>(degree, pts, out);
      return;
    case Family::cardinal_bspline:
      tabulate_cardinal_bspline_1d<T>(degree, pts, out);
      return;
    case Family::legendre:
      tabulate_legendre_1d<T>(degree, pts, out);
      return;
    case Family::lagrange:
      tabulate_lagrange_1d<T>(degree, variant, pts, out);
      return;
  }
  throw value_error("unknown basis family");
}

namespace detail {

/// Write, into `row`, the tensor product of the rows `rows[d]` (lengths `len[d]`), the function
/// index ordered as `order`; `scratch` holds as many values as `row`. Products accumulate in
/// direction order: `((r_0 r_1) r_2) ...`.
template <StorageFloat T>
void kron_row(std::span<const T* const> rows, std::span<const std::size_t> len, Order order, T* row,
              T* scratch) {
  std::size_t m = len[0];
  T* cur = rows.size() % 2 == 1 ? row : scratch;  // so the last product lands in row
  T* nxt = cur == row ? scratch : row;
  for (std::size_t i = 0; i < m; ++i) {
    cur[i] = rows[0][i];
  }
  for (std::size_t d = 1; d < rows.size(); ++d) {
    const std::size_t md = len[d];
    const T* v = rows[d];
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = 0; j < md; ++j) {
        const std::size_t f = order == Order::C ? i * md + j : j * m + i;
        nxt[f] = cur[i] * v[j];
      }
    }
    m *= md;
    std::swap(cur, nxt);
  }
}

}  // namespace detail

/// Evaluate a tensor-product basis on a lattice into `out`.
///
/// Row `p` is the lattice point of flat index `p` and column `f` the tensor-product function of
/// flat index `f`, both flattened in `order`; the value is the product of the one-dimensional
/// values, multiplied in direction order.
///
/// Args:
///     family (Family): The basis family of every direction.
///     degrees (std::span<const int>): Degree per direction, `>= 0`, one per lattice direction.
///     variant (LagrangeVariant): Node family; read for `Family::lagrange` only.
///     lattice (const quad::PointsLattice<T>&): The points.
///     order (Order): Ordering of the points and of the functions.
///     out (mdview<T, 2>): Output, extents `(lattice.num_points(), prod(degree + 1))`.
///
/// Raises:
///     value_error: The number of degrees differs from the lattice dimension, a degree is
///         negative, or `out` has other extents.
template <StorageFloat T>
void tabulate_tensor(Family family, std::span<const int> degrees, LagrangeVariant variant,
                     const quad::PointsLattice<T>& lattice, Order order, mdview<T, 2> out) {
  const std::size_t dim = lattice.dim();
  if (degrees.size() != dim) {
    throw value_error("The number of evaluators must be equal to the dimension of the points.");
  }
  std::vector<NdArray<T>> tables;
  std::vector<std::size_t> n_pts(dim), n_fun(dim);
  tables.reserve(dim);
  for (std::size_t d = 0; d < dim; ++d) {
    detail::check_degree(degrees[d]);
    const std::span<const T> pts = lattice.pts(d);
    n_pts[d] = pts.size();
    n_fun[d] = static_cast<std::size_t>(degrees[d]) + 1;
    tables.emplace_back(std::vector<std::size_t>{n_pts[d], n_fun[d]}, for_overwrite);
    tabulate_1d<T>(family, degrees[d], variant, pts, tables.back().template view<2>());
  }
  const std::size_t total_pts = checked_product(n_pts);
  const std::size_t total_fun = checked_product(n_fun);
  detail::check_out<T, 2>(out, {total_pts, total_fun});
  T* data = out.data_handle();
  detail::for_each_chunk(total_pts, 2.0 * static_cast<double>(total_fun), sizeof(T) * total_fun,
                         [&](std::size_t begin, std::size_t end) {
                           std::vector<T> scratch(total_fun);
                           std::vector<const T*> rows(dim);
                           for (std::size_t p = begin; p < end; ++p) {
                             std::size_t rest = p;
                             for (std::size_t s = 0; s < dim; ++s) {
                               const std::size_t d = order == Order::C ? dim - 1 - s : s;
                               const std::size_t q = rest % n_pts[d];
                               rest /= n_pts[d];
                               rows[d] = tables[d].data() + q * n_fun[d];
                             }
                             detail::kron_row<T>(rows, n_fun, order, data + p * total_fun,
                                                 scratch.data());
                           }
                         });
}

/// Evaluate a tensor-product basis at scattered points into `out`.
///
/// Row `p` is point `pts(p, :)`; column `f` is the tensor-product function of flat index `f` in
/// `order`.
///
/// Args:
///     family (Family): The basis family of every direction.
///     degrees (std::span<const int>): Degree per direction, `>= 0`, one per point coordinate.
///     variant (LagrangeVariant): Node family; read for `Family::lagrange` only.
///     pts (mdview<const T, 2>): The points, extents `(n_points, dim)`.
///     order (Order): Ordering of the functions.
///     out (mdview<T, 2>): Output, extents `(n_points, prod(degree + 1))`.
///
/// Raises:
///     value_error: `dim` is 0, the number of degrees differs from `dim`, a degree is negative,
///         or `out` has other extents.
template <StorageFloat T>
void tabulate_tensor(Family family, std::span<const int> degrees, LagrangeVariant variant,
                     mdview<const T, 2> pts, Order order, mdview<T, 2> out) {
  const std::size_t n = pts.extent(0);
  const std::size_t dim = pts.extent(1);
  if (dim < 1) {
    throw value_error("The dimension of the points must be at least 1.");
  }
  if (degrees.size() != dim) {
    throw value_error("The number of evaluators must be equal to the dimension of the points.");
  }
  std::vector<NdArray<T>> tables;
  std::vector<std::size_t> n_fun(dim);
  tables.reserve(dim);
  std::vector<T> column(n);
  for (std::size_t d = 0; d < dim; ++d) {
    detail::check_degree(degrees[d]);
    for (std::size_t p = 0; p < n; ++p) {
      column[p] = pts(p, d);
    }
    n_fun[d] = static_cast<std::size_t>(degrees[d]) + 1;
    tables.emplace_back(std::vector<std::size_t>{n, n_fun[d]}, for_overwrite);
    tabulate_1d<T>(family, degrees[d], variant, std::span<const T>(column),
                   tables.back().template view<2>());
  }
  const std::size_t total_fun = checked_product(n_fun);
  detail::check_out<T, 2>(out, {n, total_fun});
  T* data = out.data_handle();
  detail::for_each_chunk(n, 2.0 * static_cast<double>(total_fun), sizeof(T) * total_fun,
                         [&](std::size_t begin, std::size_t end) {
                           std::vector<T> scratch(total_fun);
                           std::vector<const T*> rows(dim);
                           for (std::size_t p = begin; p < end; ++p) {
                             for (std::size_t d = 0; d < dim; ++d) {
                               rows[d] = tables[d].data() + p * n_fun[d];
                             }
                             detail::kron_row<T>(rows, n_fun, order, data + p * total_fun,
                                                 scratch.data());
                           }
                         });
}

}  // namespace pantr::basis
