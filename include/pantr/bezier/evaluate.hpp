/// @file evaluate.hpp
/// @brief Evaluation of a `Bezier` and of its partial derivatives at points or on a lattice.
#pragma once

#include "pantr/basis/bernstein.hpp"
#include "pantr/basis/detail/tabulate_1d.hpp"
#include "pantr/bezier/bezier.hpp"
#include "pantr/bezier/detail/kernels_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/binomial.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/parallel/parallel_for.hpp"
#include "pantr/quad/points_lattice.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bezier {

namespace detail {

/// Read access to a matrix of basis values: element `(j, i)` is `data[j * row_stride + i]`. A
/// Bernstein matrix `(m, p + 1)` has stride `p + 1`; one derivative order of an `(m, order + 1,
/// p + 1)` table has stride `(order + 1) (p + 1)`.
template <StorageFloat T>
struct BasisRef {
  const T* data;
  std::size_t row_stride;
};

/// The basis values of one parametric direction at every point of it: the Bernstein basis
/// (`order == 0`, one row of orders) or the table of derivative orders `0..order`.
///
/// Attributes:
///     values (NdArray<T>): `(m, p + 1)` or `(m, order + 1, p + 1)`.
///     width (std::size_t): `p + 1`.
///     orders (std::size_t): The number of derivative orders held, 1 for the plain basis.
template <StorageFloat T>
struct DirBasis {
  NdArray<T> values;
  std::size_t width;
  std::size_t orders;

  /// Get the values of derivative order `k` as a strided matrix.
  [[nodiscard]] BasisRef<T> row(std::size_t k) const {
    return {values.data() + k * width, orders * width};
  }
};

/// Tabulate one direction: the Bernstein basis for `order == 0`, else the derivatives of orders
/// `0..order`.
template <StorageFloat T>
[[nodiscard]] DirBasis<T> tabulate_direction(std::size_t degree, std::span<const T> pts,
                                             long long order) {
  const int p = static_cast<int>(degree);
  if (order == 0) {
    return {basis::tabulate_bernstein_1d<T>(p, pts), degree + 1, 1};
  }
  const auto rows = static_cast<std::size_t>(order) + 1;
  NdArray<T> table({pts.size(), rows, degree + 1}, for_overwrite);
  basis::tabulate_bernstein_derivatives_1d<T>(p, pts, static_cast<int>(order),
                                              table.template view<3>());
  return {std::move(table), degree + 1, rows};
}

/// The parametric points of an evaluation: arbitrary points (one row each) or a lattice.
///
/// Attributes:
///     lattice (bool): Whether the points are the tensor product of `dirs`.
///     count (std::size_t): The number of points.
///     store (std::vector<std::vector<T>>): The columns of an arbitrary point set, copied out of
///         the row-major array (empty for a lattice).
///     dirs (std::vector<std::span<const T>>): One coordinate array per direction, viewing `store`
///         or the lattice.
template <StorageFloat T>
struct EvalPoints {
  bool lattice = false;
  std::size_t count = 0;
  std::vector<std::vector<T>> store;
  std::vector<std::span<const T>> dirs;
};

/// Gather the columns of a `(n, dim)` point array.
template <StorageFloat T>
[[nodiscard]] EvalPoints<T> make_points(mdview<const T, 2> pts) {
  EvalPoints<T> e;
  e.lattice = false;
  e.count = pts.extent(0);
  e.store.resize(pts.extent(1));
  for (std::size_t d = 0; d < pts.extent(1); ++d) {
    e.store[d].resize(e.count);
    for (std::size_t j = 0; j < e.count; ++j) {
      e.store[d][j] = pts(j, d);
    }
    e.dirs.emplace_back(e.store[d]);
  }
  return e;
}

/// View a lattice's coordinate arrays.
template <StorageFloat T>
[[nodiscard]] EvalPoints<T> make_points(const quad::PointsLattice<T>& lattice) {
  EvalPoints<T> e;
  e.lattice = true;
  e.count = lattice.num_points();
  for (std::size_t d = 0; d < lattice.dim(); ++d) {
    e.dirs.push_back(lattice.pts(d));
  }
  return e;
}

/// Contract the control points with one basis matrix per direction at arbitrary points: for each
/// point, direction 0 first (`einsum("pi,i...->p...")`), then direction 1 on what is left, and so
/// on, each sum ascending in the contracted index and accumulated in `T`.
///
/// Args:
///     ctrl (std::span<const T>): Control points over `shape`.
///     shape (std::span<const std::size_t>): `(n_0, ..., n_{dim-1}, cp)`.
///     bases (const std::vector<BasisRef<T>>&): One basis matrix per direction, `n_pts` rows.
///     n_pts (std::size_t): Number of points.
///     dst (T*): Output `(n_pts, cp)`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
void contract_points(std::span<const T> ctrl, std::span<const std::size_t> shape,
                     const std::vector<BasisRef<T>>& bases, std::size_t n_pts, T* dst) {
  const std::size_t dim = shape.size() - 1;
  const std::size_t cp = shape.back();
  // sizes[d] is the number of values left to contract at stage d, prod_{e >= d} n_e * cp.
  std::vector<std::size_t> sizes(dim + 1);
  sizes[dim] = cp;
  for (std::size_t d = dim; d-- > 0;) {
    sizes[d] = sizes[d + 1] * shape[d];
  }
  // Stage d does one multiply-add per value left to contract, sizes[d] of them.
  double ops = 0.0;
  for (std::size_t d = 0; d < dim; ++d) {
    ops += 2.0 * static_cast<double>(sizes[d]);
  }
  const std::size_t buf_size = dim > 1 ? sizes[1] : 0;
  parallel::parallel_for(
      n_pts, parallel::WorkEstimate{(std::max)(ops, 1.0)}, [&](std::size_t begin, std::size_t end) {
        std::vector<T> buf0(buf_size);
        std::vector<T> buf1(buf_size);
        for (std::size_t j = begin; j < end; ++j) {
          const T* src = ctrl.data();
          for (std::size_t d = 0; d < dim; ++d) {
            const std::size_t m = sizes[d + 1];
            T* out = d + 1 == dim ? dst + j * cp : (d % 2 == 0 ? buf0.data() : buf1.data());
            const T* row = bases[d].data + j * bases[d].row_stride;
            for (std::size_t i = 0; i < m; ++i) {
              out[i] = static_cast<T>(0);
            }
            for (std::size_t k = 0; k < shape[d]; ++k) {
              const T coeff = row[k];
              const T* s = src + k * m;
              for (std::size_t i = 0; i < m; ++i) {
                out[i] = out[i] + coeff * s[i];
              }
            }
            src = out;
          }
        }
      });
}

/// Contract the control points with one basis matrix per direction on a lattice: direction 0
/// first, then direction 1 on the result (`tensordot(basis_d, result, axes=([1], [d]))` with the
/// new axis moved back to position `d`), each sum ascending in the contracted index and
/// accumulated in `T`.
///
/// Args:
///     ctrl (std::span<const T>): Control points over `shape`.
///     shape (std::span<const std::size_t>): `(n_0, ..., n_{dim-1}, cp)`.
///     bases (const std::vector<BasisRef<T>>&): One basis matrix per direction, `counts[d]` rows.
///     counts (std::span<const std::size_t>): Points per direction.
///     dst (T*): Output `(counts[0], ..., counts[dim-1], cp)`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
void contract_lattice(std::span<const T> ctrl, std::span<const std::size_t> shape,
                      const std::vector<BasisRef<T>>& bases, std::span<const std::size_t> counts,
                      T* dst) {
  const std::size_t dim = shape.size() - 1;
  const std::size_t cp = shape.back();
  // Stage d turns (m_0..m_{d-1}, n_d, n_{d+1}..n_{dim-1}, cp) into (m_0..m_d, n_{d+1}.., cp).
  std::vector<std::size_t> out_size(dim);
  std::size_t scratch = 0;
  for (std::size_t d = 0; d < dim; ++d) {
    std::size_t s = cp;
    for (std::size_t e = 0; e < dim; ++e) {
      s *= e <= d ? counts[e] : shape[e];
    }
    out_size[d] = s;
    if (d + 1 < dim) {
      scratch = (std::max)(scratch, s);
    }
  }
  std::vector<T> buf0(scratch);
  std::vector<T> buf1(scratch);
  const T* src = ctrl.data();
  std::size_t outer = 1;
  for (std::size_t d = 0; d < dim; ++d) {
    std::size_t inner = cp;
    for (std::size_t e = d + 1; e < dim; ++e) {
      inner *= shape[e];
    }
    const std::size_t n = shape[d];
    const std::size_t m = counts[d];
    T* out = d + 1 == dim ? dst : (d % 2 == 0 ? buf0.data() : buf1.data());
    const BasisRef<T> bm = bases[d];
    // One output row (an `inner`-long run) costs a multiply-add per contracted index.
    const double ops = 2.0 * static_cast<double>(n) * static_cast<double>(inner);
    const T* stage_src = src;
    parallel::parallel_for(outer * m, parallel::WorkEstimate{(std::max)(ops, 1.0)},
                           [&](std::size_t begin, std::size_t end) {
                             for (std::size_t r = begin; r < end; ++r) {
                               const std::size_t o = r / m;
                               const std::size_t a = r % m;
                               T* row = out + r * inner;
                               for (std::size_t i = 0; i < inner; ++i) {
                                 row[i] = static_cast<T>(0);
                               }
                               const T* base = stage_src + o * n * inner;
                               for (std::size_t k = 0; k < n; ++k) {
                                 const T coeff = bm.data[a * bm.row_stride + k];
                                 const T* s = base + k * inner;
                                 for (std::size_t i = 0; i < inner; ++i) {
                                   row[i] = row[i] + coeff * s[i];
                                 }
                               }
                             }
                           });
    src = out;
    outer *= m;
  }
}

/// Contract the control points with the basis matrices at the given points.
template <StorageFloat T>
void contract(const Bezier<T>& b, const EvalPoints<T>& pts, const std::vector<BasisRef<T>>& bases,
              T* dst) {
  if (pts.lattice) {
    std::vector<std::size_t> counts(pts.dirs.size());
    for (std::size_t d = 0; d < counts.size(); ++d) {
      counts[d] = pts.dirs[d].size();
    }
    contract_lattice<T>(b.control_points(), b.shape(), bases, counts, dst);
  } else {
    contract_points<T>(b.control_points(), b.shape(), bases, pts.count, dst);
  }
}

/// Divide the coordinates of homogeneous values by their weight: `out(j, r) = raw(j, r) /
/// raw(j, rank)`.
template <StorageFloat T>
void project_rational(const T* raw, std::size_t count, std::size_t rank, T* out) {
  const std::size_t cp = rank + 1;
  for (std::size_t j = 0; j < count; ++j) {
    const T w = raw[j * cp + rank];
    for (std::size_t r = 0; r < rank; ++r) {
      out[j * rank + r] = raw[j * cp + r] / w;
    }
  }
}

/// Evaluate at arbitrary points or on a lattice for `dim >= 2`, into `out` (`(count, rank)`).
template <StorageFloat T>
void evaluate_nd_into(const Bezier<T>& b, const EvalPoints<T>& pts, T* out) {
  const std::size_t dim = b.dim();
  std::vector<DirBasis<T>> tab;
  std::vector<BasisRef<T>> refs;
  tab.reserve(dim);
  for (std::size_t d = 0; d < dim; ++d) {
    tab.push_back(tabulate_direction<T>(b.degree(d), pts.dirs[d], 0));
    refs.push_back(tab.back().row(0));
  }
  if (!b.rational()) {
    contract(b, pts, refs, out);
    return;
  }
  std::vector<T> raw(pts.count * b.cp_size());
  contract(b, pts, refs, raw.data());
  project_rational<T>(raw.data(), pts.count, b.rank(), out);
}

/// Evaluate a 1D Bezier at the points `pts` into `out` (`(pts.size(), rank)`).
template <StorageFloat T>
void evaluate_1d_into(const Bezier<T>& b, std::span<const T> pts, T* out) {
  const std::size_t cp = b.cp_size();
  const mdview<const T, 2> ctrl(b.control_points().data(), b.shape()[0], cp);
  if (!b.rational()) {
    evaluate_1d_kernel<T>(ctrl, pts, mdview<T, 2>(out, pts.size(), cp));
    return;
  }
  std::vector<T> raw(pts.size() * cp);
  evaluate_1d_kernel<T>(ctrl, pts, mdview<T, 2>(raw.data(), pts.size(), cp));
  project_rational<T>(raw.data(), pts.size(), b.rank(), out);
}

/// Advance a multi-index in lexicographic order (last component fastest) within `bounds`
/// (exclusive upper limits); return false after the last index, leaving it all zero.
inline bool next_multi_index(std::vector<std::size_t>& idx, std::span<const std::size_t> bounds) {
  for (std::size_t d = idx.size(); d-- > 0;) {
    if (++idx[d] < bounds[d]) {
      return true;
    }
    idx[d] = 0;
  }
  return false;
}

/// Read access to homogeneous derivative values: element `(row, c)` is `data[row * stride + c]`.
template <StorageFloat T>
struct HomRef {
  const T* data;
  std::size_t stride;
};

/// The generalized quotient rule (Algorithm A4.2, extended to several variables) for the
/// partial derivative of highest order of `A / w`, from the homogeneous partial derivatives of
/// `(A, w)`:
///
///     D^k f = (D^k A - sum_{0 < j <= k} C(k, j) D^j w D^(k - j) f) / w,
///
/// with `j <= k` componentwise and `C(k, j)` the product of the componentwise binomials, over all
/// multi-indices `k <= orders` in lexicographic order. Every operation is in `T`, as the NumPy
/// arrays it ports were.
///
/// Args:
///     hom (const std::vector<HomRef<T>>&): The homogeneous derivatives, indexed by the
///         lexicographic position of the multi-index in `0..extents`.
///     extents (std::span<const std::size_t>): `orders[d] + 1` per direction.
///     count (std::size_t): The number of points.
///     rank (std::size_t): The number of coordinates, the weight excluded.
///     out (T*): The derivative of the highest order, `(count, rank)`.
template <StorageFloat T>
void quotient_rule(const std::vector<HomRef<T>>& hom, std::span<const std::size_t> extents,
                   std::size_t count, std::size_t rank, T* out) {
  const std::size_t dim = extents.size();
  std::size_t total = 1;
  std::vector<std::size_t> stride(dim, 1);
  for (std::size_t d = dim; d-- > 0;) {
    stride[d] = total;
    total *= extents[d];
  }
  std::vector<std::vector<T>> results(total);
  std::vector<std::size_t> k_idx(dim, 0);
  std::vector<std::size_t> j_idx(dim, 0);
  std::vector<std::size_t> bound(dim, 0);
  for (std::size_t kf = 0; kf < total; ++kf) {
    results[kf].resize(count * rank);
    T* v = results[kf].data();
    for (std::size_t row = 0; row < count; ++row) {
      for (std::size_t c = 0; c < rank; ++c) {
        v[row * rank + c] = hom[kf].data[row * hom[kf].stride + c];
      }
    }
    for (std::size_t d = 0; d < dim; ++d) {
      bound[d] = k_idx[d] + 1;
      j_idx[d] = 0;
    }
    while (next_multi_index(j_idx, bound)) {
      double binom = 1.0;
      std::size_t jf = 0;
      std::size_t df = 0;
      for (std::size_t d = 0; d < dim; ++d) {
        binom *=
            binomial_double(static_cast<long long>(k_idx[d]), static_cast<long long>(j_idx[d]));
        jf += j_idx[d] * stride[d];
        df += (k_idx[d] - j_idx[d]) * stride[d];
      }
      const T coeff = static_cast<T>(binom);
      for (std::size_t row = 0; row < count; ++row) {
        const T w = hom[jf].data[row * hom[jf].stride + rank];
        const T cw = coeff * w;
        for (std::size_t c = 0; c < rank; ++c) {
          v[row * rank + c] = v[row * rank + c] - cw * results[df][row * rank + c];
        }
      }
    }
    for (std::size_t row = 0; row < count; ++row) {
      const T w = hom[0].data[row * hom[0].stride + rank];
      for (std::size_t c = 0; c < rank; ++c) {
        v[row * rank + c] = v[row * rank + c] / w;
      }
    }
    next_multi_index(k_idx, extents);
  }
  std::copy(results[total - 1].begin(), results[total - 1].end(), out);
}

/// Evaluate the partial derivative of the given orders at arbitrary points or on a lattice, for
/// any `dim`, into `out` (`(count, rank)`).
template <StorageFloat T>
void derivatives_into(const Bezier<T>& b, const EvalPoints<T>& pts,
                      std::span<const long long> orders, T* out) {
  const std::size_t dim = b.dim();
  const std::size_t cp = b.cp_size();
  const std::size_t rank = b.rank();
  const std::size_t count = pts.count;
  if (dim == 1) {
    const auto nd1 = static_cast<std::size_t>(orders[0]) + 1;
    const mdview<const T, 2> ctrl(b.control_points().data(), b.shape()[0], cp);
    NdArray<T> table({count, nd1, cp}, for_overwrite);
    evaluate_derivatives_1d_kernel<T>(ctrl, pts.dirs[0], static_cast<int>(orders[0]),
                                      table.template view<3>());
    if (!b.rational()) {
      for (std::size_t j = 0; j < count; ++j) {
        for (std::size_t c = 0; c < cp; ++c) {
          out[j * cp + c] = table.data()[(j * nd1 + nd1 - 1) * cp + c];
        }
      }
      return;
    }
    std::vector<HomRef<T>> hom;
    for (std::size_t k = 0; k < nd1; ++k) {
      hom.push_back({table.data() + k * cp, nd1 * cp});
    }
    const std::size_t extents[1] = {nd1};
    quotient_rule<T>(hom, extents, count, rank, out);
    return;
  }
  std::vector<DirBasis<T>> tab;
  tab.reserve(dim);
  std::vector<std::size_t> extents(dim);
  for (std::size_t d = 0; d < dim; ++d) {
    tab.push_back(tabulate_direction<T>(b.degree(d), pts.dirs[d], orders[d]));
    extents[d] = static_cast<std::size_t>(orders[d]) + 1;
  }
  std::vector<BasisRef<T>> refs(dim);
  if (!b.rational()) {
    for (std::size_t d = 0; d < dim; ++d) {
      refs[d] = tab[d].row(static_cast<std::size_t>(orders[d]));
    }
    contract(b, pts, refs, out);
    return;
  }
  std::size_t total = 1;
  for (const std::size_t e : extents) {
    total *= e;
  }
  std::vector<std::vector<T>> store(total);
  std::vector<HomRef<T>> hom;
  std::vector<std::size_t> k_idx(dim, 0);
  for (std::size_t kf = 0; kf < total; ++kf) {
    for (std::size_t d = 0; d < dim; ++d) {
      refs[d] = tab[d].row(k_idx[d]);
    }
    store[kf].resize(count * cp);
    contract(b, pts, refs, store[kf].data());
    hom.push_back({store[kf].data(), cp});
    next_multi_index(k_idx, extents);
  }
  quotient_rule<T>(hom, extents, count, rank, out);
}

/// Refuse an orders list of the wrong length or with a negative entry, and one too large for the
/// derivative tabulation.
inline void check_orders(std::size_t dim, std::span<const long long> orders) {
  if (orders.size() != dim) {
    throw value_error("len(orders) (" + std::to_string(orders.size()) + ") must match dim (" +
                      std::to_string(dim) + ").");
  }
  for (const long long o : orders) {
    if (o < 0) {
      throw value_error("All derivative orders must be non-negative.");
    }
  }
  for (const long long o : orders) {
    if (o >= (std::numeric_limits<int>::max)()) {
      throw capacity_error("derivative order " + std::to_string(o) + " is too large");
    }
  }
}

/// Refuse a point array whose columns do not match the Bezier.
template <StorageFloat T>
void check_points(const Bezier<T>& b, mdview<const T, 2> pts) {
  if (b.dim() == 1) {
    if (pts.extent(1) != 1) {
      throw value_error("Points must be a 1D array for a 1D Bézier.");
    }
    return;
  }
  if (pts.extent(1) != b.dim()) {
    throw value_error("pts must be a 2D array with " + std::to_string(b.dim()) + " columns.");
  }
}

/// Refuse a lattice whose dimension does not match the Bezier.
template <StorageFloat T>
void check_lattice(const Bezier<T>& b, const quad::PointsLattice<T>& lattice) {
  if (b.dim() == 1) {
    if (lattice.dim() != 1) {
      throw value_error("PointsLattice must be 1D for a 1D Bézier.");
    }
    return;
  }
  if (lattice.dim() != b.dim()) {
    throw value_error("PointsLattice dim (" + std::to_string(lattice.dim()) +
                      ") must match Bézier dim (" + std::to_string(b.dim()) + ").");
  }
}

/// The extents of a lattice evaluation: the points per direction, then the rank.
template <StorageFloat T>
[[nodiscard]] std::vector<std::size_t> lattice_result_shape(const Bezier<T>& b,
                                                            const quad::PointsLattice<T>& lattice) {
  std::vector<std::size_t> shape;
  for (std::size_t d = 0; d < lattice.dim(); ++d) {
    shape.push_back(lattice.pts(d).size());
  }
  shape.push_back(b.rank());
  return shape;
}

/// Evaluate at arbitrary points (checked) into a raw buffer.
template <StorageFloat T>
void evaluate_points_into(const Bezier<T>& b, mdview<const T, 2> pts, T* out) {
  if (b.dim() == 1) {
    evaluate_1d_into<T>(b, std::span<const T>(pts.data_handle(), pts.extent(0)), out);
  } else {
    evaluate_nd_into<T>(b, make_points<T>(pts), out);
  }
}

/// Evaluate on a lattice (checked) into a raw buffer.
template <StorageFloat T>
void evaluate_lattice_into(const Bezier<T>& b, const quad::PointsLattice<T>& lattice, T* out) {
  if (b.dim() == 1) {
    evaluate_1d_into<T>(b, lattice.pts(0), out);
  } else {
    evaluate_nd_into<T>(b, make_points<T>(lattice), out);
  }
}

}  // namespace detail

/// Evaluate a Bezier at arbitrary parametric points.
///
/// A 1D Bezier uses the fused kernel (Bernstein recurrence and contraction in one pass); a
/// higher-dimensional one tabulates the Bernstein basis per direction and contracts direction by
/// direction, accumulating in `T`. A rational Bezier divides by the interpolated weight.
///
/// Args:
///     b (const Bezier<T>&): The Bezier.
///     pts (mdview<const T, 2>): Points, extents `(n, dim)`; `(n, 1)` for a 1D Bezier.
///
/// Returns:
///     NdArray<T>: Values, extents `(n, rank)` (never squeezed).
///
/// Raises:
///     value_error: `pts.extent(1)` is not `dim` (the message for a 1D Bezier names a 1D array).
template <StorageFloat T>
[[nodiscard]] NdArray<T> evaluate(const Bezier<T>& b, mdview<const T, 2> pts) {
  detail::check_points(b, pts);
  NdArray<T> out({pts.extent(0), b.rank()}, for_overwrite);
  detail::evaluate_points_into<T>(b, pts, out.data());
  return out;
}

/// Evaluate a Bezier at arbitrary parametric points into `out`; see the overload returning the
/// values.
///
/// Args:
///     b (const Bezier<T>&): The Bezier.
///     pts (mdview<const T, 2>): Points, extents `(n, dim)`.
///     out (mdview<T, 2>): Output, extents `(n, rank)`.
///
/// Raises:
///     value_error: `pts.extent(1)` is not `dim`, or `out` has other extents.
template <StorageFloat T>
void evaluate(const Bezier<T>& b, mdview<const T, 2> pts, mdview<T, 2> out) {
  detail::check_points(b, pts);
  basis::detail::check_out<T, 2>(out, {pts.extent(0), b.rank()});
  detail::evaluate_points_into<T>(b, pts, out.data_handle());
}

/// Evaluate a Bezier on a tensor-product lattice of parametric points.
///
/// Args:
///     b (const Bezier<T>&): The Bezier.
///     lattice (const quad::PointsLattice<T>&): The lattice, of dimension `dim`.
///
/// Returns:
///     NdArray<T>: Values, extents `(n_0, ..., n_{dim-1}, rank)` for `n_d` points in direction
///     `d` (never squeezed).
///
/// Raises:
///     value_error: The lattice dimension is not `dim`.
template <StorageFloat T>
[[nodiscard]] NdArray<T> evaluate(const Bezier<T>& b, const quad::PointsLattice<T>& lattice) {
  detail::check_lattice(b, lattice);
  NdArray<T> out(detail::lattice_result_shape(b, lattice), for_overwrite);
  detail::evaluate_lattice_into<T>(b, lattice, out.data());
  return out;
}

/// Evaluate a Bezier on a lattice into `out`; see the overload returning the values.
///
/// Args:
///     b (const Bezier<T>&): The Bezier.
///     lattice (const quad::PointsLattice<T>&): The lattice, of dimension `dim`.
///     out (mdview<T, 2>): Output, extents `(lattice.num_points(), rank)`, the lattice points in C
///         order (the last direction varies fastest).
///
/// Raises:
///     value_error: The lattice dimension is not `dim`, or `out` has other extents.
template <StorageFloat T>
void evaluate(const Bezier<T>& b, const quad::PointsLattice<T>& lattice, mdview<T, 2> out) {
  detail::check_lattice(b, lattice);
  basis::detail::check_out<T, 2>(out, {lattice.num_points(), b.rank()});
  detail::evaluate_lattice_into<T>(b, lattice, out.data_handle());
}

/// Evaluate a partial derivative of a Bezier at arbitrary parametric points.
///
/// Orders above the degree give zero. For a rational Bezier the homogeneous derivatives of every
/// order up to `orders` are formed and the quotient rule (Piegl and Tiller's Algorithm A4.2, in
/// several variables) gives the derivative of the projected map.
///
/// Args:
///     b (const Bezier<T>&): The Bezier.
///     pts (mdview<const T, 2>): Points, extents `(n, dim)`; `(n, 1)` for a 1D Bezier.
///     orders (std::span<const long long>): The derivative order per direction, `dim` values
///         `>= 0`.
///
/// Returns:
///     NdArray<T>: Values, extents `(n, rank)` (never squeezed).
///
/// Raises:
///     value_error: `orders` has the wrong length or a negative entry (checked first), or
///         `pts.extent(1)` is not `dim`.
///     capacity_error: An order is `INT_MAX` or more.
template <StorageFloat T>
[[nodiscard]] NdArray<T> evaluate_derivatives(const Bezier<T>& b, mdview<const T, 2> pts,
                                              std::span<const long long> orders) {
  detail::check_orders(b.dim(), orders);
  detail::check_points(b, pts);
  NdArray<T> out({pts.extent(0), b.rank()}, for_overwrite);
  detail::derivatives_into<T>(b, detail::make_points<T>(pts), orders, out.data());
  return out;
}

/// Evaluate a partial derivative at arbitrary parametric points into `out`; see the overload
/// returning the values.
///
/// Args:
///     b (const Bezier<T>&): The Bezier.
///     pts (mdview<const T, 2>): Points, extents `(n, dim)`.
///     orders (std::span<const long long>): The derivative order per direction.
///     out (mdview<T, 2>): Output, extents `(n, rank)`.
///
/// Raises:
///     value_error: As the overload returning the values, or `out` has other extents.
///     capacity_error: An order is `INT_MAX` or more.
template <StorageFloat T>
void evaluate_derivatives(const Bezier<T>& b, mdview<const T, 2> pts,
                          std::span<const long long> orders, mdview<T, 2> out) {
  detail::check_orders(b.dim(), orders);
  detail::check_points(b, pts);
  basis::detail::check_out<T, 2>(out, {pts.extent(0), b.rank()});
  detail::derivatives_into<T>(b, detail::make_points<T>(pts), orders, out.data_handle());
}

/// Evaluate a partial derivative of a Bezier on a tensor-product lattice.
///
/// Args:
///     b (const Bezier<T>&): The Bezier.
///     lattice (const quad::PointsLattice<T>&): The lattice, of dimension `dim`.
///     orders (std::span<const long long>): The derivative order per direction.
///
/// Returns:
///     NdArray<T>: Values, extents `(n_0, ..., n_{dim-1}, rank)` (never squeezed).
///
/// Raises:
///     value_error: `orders` has the wrong length or a negative entry (checked first), or the
///         lattice dimension is not `dim`.
///     capacity_error: An order is `INT_MAX` or more.
template <StorageFloat T>
[[nodiscard]] NdArray<T> evaluate_derivatives(const Bezier<T>& b,
                                              const quad::PointsLattice<T>& lattice,
                                              std::span<const long long> orders) {
  detail::check_orders(b.dim(), orders);
  detail::check_lattice(b, lattice);
  NdArray<T> out(detail::lattice_result_shape(b, lattice), for_overwrite);
  detail::derivatives_into<T>(b, detail::make_points<T>(lattice), orders, out.data());
  return out;
}

/// Evaluate a partial derivative on a lattice into `out`; see the overload returning the values.
///
/// Args:
///     b (const Bezier<T>&): The Bezier.
///     lattice (const quad::PointsLattice<T>&): The lattice, of dimension `dim`.
///     orders (std::span<const long long>): The derivative order per direction.
///     out (mdview<T, 2>): Output, extents `(lattice.num_points(), rank)`, the lattice points in C
///         order.
///
/// Raises:
///     value_error: As the overload returning the values, or `out` has other extents.
///     capacity_error: An order is `INT_MAX` or more.
template <StorageFloat T>
void evaluate_derivatives(const Bezier<T>& b, const quad::PointsLattice<T>& lattice,
                          std::span<const long long> orders, mdview<T, 2> out) {
  detail::check_orders(b.dim(), orders);
  detail::check_lattice(b, lattice);
  basis::detail::check_out<T, 2>(out, {lattice.num_points(), b.rank()});
  detail::derivatives_into<T>(b, detail::make_points<T>(lattice), orders, out.data_handle());
}

}  // namespace pantr::bezier
