/// @file evaluate.hpp
/// @brief Evaluation of a `Bspline` and of its partial derivatives at points or on a lattice.
///
/// Three schemes, by the shape of the request. All evaluate a rational B-spline on its homogeneous
/// control points and divide by the interpolated weight, read the control points of a periodic
/// direction modulo `num_basis`, and give every output value to exactly one parallel item that
/// computes it alone, from fixed chunks of the problem, so no result depends on the number of
/// threads.
///
///  - Scattered points of two or more directions: the non-zero 1D basis values (or derivative
///    rows) of each direction, tabulated by `BsplineSpace1D` once per distinct coordinate, are
///    multiplied into the tensor-product weights of the point's `prod(degree + 1)` local
///    functions, in `P` and left to right over the directions, and stored in `T`; the control
///    points of those functions are summed in `T`, in ascending C order of the local multi-index.
///  - A lattice of two or more directions (values and, for a non-rational B-spline, derivatives):
///    sum factorisation. The control-point array is contracted with one direction's rows at a
///    time, `out[.., j, ..] = sum_a B_d[j, a] cur[.., first_d[j] + a, ..]` with the terms added in
///    ascending `a` in `T`, the directions in the order (first to last, or last to first) that
///    does the fewer multiply-adds. An output value is therefore a sum of nested sums of
///    products, not the flat sum of the scattered scheme: the two agree to a rounding bound.
///  - One direction (points or a lattice): each point's basis values are formed in a scratch row
///    (Cox-de Boor) and contracted at once, with no table of the points' basis values; a
///    Bezier-like space keeps the tabulated scheme, as `BsplineSpace1D` evaluates it by the
///    Bernstein recurrence there. Derivatives take the tabulated scheme.
#pragma once

#include "pantr/basis/detail/tabulate_1d.hpp"
#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/detail/basis_kernels.hpp"
#include "pantr/bspline/space.hpp"
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
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bspline {

namespace detail {

/// The points of an evaluation: arbitrary points (one row each) or the tensor product of one
/// coordinate array per direction.
///
/// Attributes:
///     lattice (bool): Whether the points are the tensor product of `dirs`.
///     count (std::size_t): The number of points.
///     store (std::vector<std::vector<P>>): The columns of an arbitrary point set of two or more
///         columns, copied out of the row-major array (empty for a lattice and for one column,
///         which is contiguous already).
///     dirs (std::vector<std::span<const P>>): One coordinate array per direction: `count` values
///         viewing `store` or the single column, or the lattice's own coordinates. A view into
///         the caller's array lives as long as that array does.
template <StorageFloat P>
struct PointSet {
  bool lattice = false;
  std::size_t count = 0;
  std::vector<std::vector<P>> store;
  std::vector<std::span<const P>> dirs;
};

/// Gather the columns of a `(n, dim)` point array (a single column is viewed, not copied).
template <StorageFloat P>
[[nodiscard]] PointSet<P> make_point_set(mdview<const P, 2> pts) {
  PointSet<P> e;
  e.count = pts.extent(0);
  if (pts.extent(1) == 1) {
    e.dirs.emplace_back(pts.data_handle(), e.count);
    return e;
  }
  e.store.resize(pts.extent(1));
  for (std::size_t d = 0; d < pts.extent(1); ++d) {
    e.store[d].resize(e.count);
    for (std::size_t i = 0; i < e.count; ++i) {
      e.store[d][i] = pts(i, d);
    }
    e.dirs.emplace_back(e.store[d]);
  }
  return e;
}

/// View a lattice's coordinate arrays.
template <StorageFloat P>
[[nodiscard]] PointSet<P> make_point_set(const quad::PointsLattice<P>& lattice) {
  PointSet<P> e;
  e.lattice = true;
  e.count = lattice.num_points();
  for (std::size_t d = 0; d < lattice.dim(); ++d) {
    e.dirs.push_back(lattice.pts(d));
  }
  return e;
}

/// The 1D basis data of one direction at each of its coordinates.
///
/// Attributes:
///     values (NdArray<P>): `(m, rows, order)`: for each coordinate, `rows` rows of `order =
///         degree + 1` values: one row (the basis) or the derivative rows of orders `0..rows - 1`.
///     first (NdArray<std::int64_t>): `(m,)` first supported function per coordinate.
///     order (std::size_t): `degree + 1`.
///     rows (std::size_t): The rows per coordinate.
template <StorageFloat P>
struct DirTable {
  NdArray<P> values;
  NdArray<std::int64_t> first;
  std::size_t order;
  std::size_t rows;

  /// Get row `k` of coordinate `i`.
  [[nodiscard]] const P* row(std::size_t i, std::size_t k) const noexcept {
    return values.data() + (i * rows + k) * order;
  }
};

/// How a B-spline's control points are addressed from the first function of each direction.
///
/// Attributes:
///     dim (std::size_t): The number of directions.
///     cp_size (std::size_t): Coordinates per control point.
///     nloc (std::size_t): `prod(order)`, the local functions per point.
///     order (std::vector<std::size_t>): `degree + 1` per direction.
///     nb (std::vector<std::size_t>): Basis functions (stored control points) per direction.
///     stride (std::vector<std::size_t>): Elements between consecutive indices of a direction.
///     periodic (std::vector<char>): Whether a direction's indices wrap modulo `nb`.
struct Layout {
  std::size_t dim = 0;
  std::size_t cp_size = 0;
  std::size_t nloc = 1;
  std::vector<std::size_t> order;
  std::vector<std::size_t> nb;
  std::vector<std::size_t> stride;
  std::vector<char> periodic;
};

template <StorageFloat T>
[[nodiscard]] Layout make_layout(const Bspline<T>& b) {
  Layout L;
  L.dim = b.dim();
  L.cp_size = b.cp_size();
  L.stride.assign(L.dim, 0);
  std::size_t stride = L.cp_size;
  for (std::size_t d = L.dim; d-- > 0;) {
    L.stride[d] = stride;
    stride *= b.num_basis(d);
  }
  for (const auto& s : b.rep()->space.spaces()) {
    L.order.push_back(static_cast<std::size_t>(s.degree()) + 1);
    L.nb.push_back(static_cast<std::size_t>(s.num_basis()));
    L.periodic.push_back(s.periodic() ? 1 : 0);
    L.nloc *= L.order.back();
  }
  return L;
}

/// The element offsets, in C order over the local multi-index, of the control points of the
/// `nloc` functions whose first function per direction is `first[d]`.
///
/// Args:
///     L (const Layout&): The addressing.
///     first (const std::int64_t*): `dim` first functions (non-negative).
///     offs (std::size_t*): Output, `nloc` offsets.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
inline void local_offsets(const Layout& L, const std::int64_t* first, std::size_t* offs) noexcept {
  const auto index = [&](std::size_t d, std::size_t j) {
    const auto i = static_cast<std::size_t>(first[d]) + j;
    return (L.periodic[d] != 0 ? i % L.nb[d] : i) * L.stride[d];
  };
  std::size_t size = L.order[0];
  for (std::size_t j = 0; j < size; ++j) {
    offs[j] = index(0, j);
  }
  for (std::size_t d = 1; d < L.dim; ++d) {
    const std::size_t o = L.order[d];
    // Expanded in place from the back: entry a moves to a * o .. a * o + o - 1, at or past a.
    for (std::size_t a = size; a-- > 0;) {
      const std::size_t base = offs[a];
      for (std::size_t j = o; j-- > 0;) {
        offs[a * o + j] = base + index(d, j);
      }
    }
    size *= o;
  }
}

/// `out = sum_j w[j] cp[offs[j] ..]`, summed in ascending `j` in `T`, column by column.
///
/// A `noexcept` function of its own so that GCC vectorises the loops over the columns: written
/// inside a body that may throw (a `parallel_for` chunk), GCC 14 with `-ffp-contract=on` treats the
/// contracted `fma` as a statement that can throw and leaves the loop scalar.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
void contract_local(const T* cp, const std::size_t* offs, const T* w, std::size_t nloc,
                    std::size_t cp_size, T* out) noexcept {
  {
    const T* c = cp + offs[0];
    const T w0 = w[0];
    for (std::size_t k = 0; k < cp_size; ++k) {
      out[k] = w0 * c[k];
    }
  }
  for (std::size_t j = 1; j < nloc; ++j) {
    const T* c = cp + offs[j];
    const T wj = w[j];
    for (std::size_t k = 0; k < cp_size; ++k) {
      out[k] = out[k] + wj * c[k];
    }
  }
}

/// Divide the first `rank` coordinates of a homogeneous value by the weight at `row[rank]`, which
/// stays.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
void project_in_place(T* row, std::size_t rank) noexcept {
  const T w = row[rank];
  for (std::size_t k = 0; k < rank; ++k) {
    row[k] = row[k] / w;
  }
}

/// The position of each point in its direction's tables: arbitrary points use their own index in
/// every direction, a lattice point the C-order multi-index of its flat index.
class PointCursor {
 public:
  /// Start at point `begin`.
  ///
  /// Args:
  ///     lattice (bool): Whether the points form a lattice.
  ///     counts (std::span<const std::size_t>): Coordinates per direction (lattice only).
  ///     dim (std::size_t): The number of directions.
  ///     begin (std::size_t): The first point.
  PointCursor(bool lattice, std::span<const std::size_t> counts, std::size_t dim, std::size_t begin)
      : lattice_(lattice), counts_(counts), idx_(dim, begin) {
    if (lattice_) {
      std::size_t rest = begin;
      for (std::size_t d = dim; d-- > 0;) {
        idx_[d] = rest % counts_[d];
        rest /= counts_[d];
      }
    }
  }

  /// The index of the current point in direction `d`'s table.
  [[nodiscard]] std::size_t at(std::size_t d) const noexcept { return idx_[d]; }

  /// Move to the next point.
  void advance() noexcept {
    if (!lattice_) {
      for (std::size_t& i : idx_) {
        ++i;
      }
      return;
    }
    for (std::size_t d = idx_.size(); d-- > 0;) {
      if (++idx_[d] < counts_[d]) {
        break;
      }
      idx_[d] = 0;
    }
  }

 private:
  bool lattice_;
  std::span<const std::size_t> counts_;
  std::vector<std::size_t> idx_;
};

/// Per-chunk scratch of the point loop.
template <StorageFloat T, StorageFloat P>
struct PointScratch {
  std::vector<const P*> rows;
  std::vector<std::int64_t> first;
  std::vector<P> product;
  std::vector<T> weights;
  std::vector<std::size_t> offs;
  explicit PointScratch(const Layout& L)
      : rows(L.dim), first(L.dim), product(L.nloc), weights(L.nloc), offs(L.nloc) {}
};

/// Refuse points whose column count is not the dimension.
///
/// Raises:
///     value_error: `columns != dim`.
inline void check_columns(std::size_t dim, std::size_t columns) {
  if (columns != dim) {
    throw value_error("pts must have " + std::to_string(dim) + " columns");
  }
}

/// Refuse a lattice of another dimension.
///
/// Raises:
///     value_error: `lattice_dim != dim`.
inline void check_lattice_dim(std::size_t dim, std::size_t lattice_dim) {
  if (lattice_dim != dim) {
    throw value_error("Points lattice dimension " + std::to_string(lattice_dim) +
                      " does not match B-spline dimension " + std::to_string(dim));
  }
}

/// Operations to test one coordinate against a domain: two bounds, each a comparison and, inside
/// the tolerance band, a subtraction and a comparison. An admitted estimate, used only to size the
/// chunks of the check.
inline constexpr double domain_check_ops = 8.0;

/// Refuse coordinates outside a direction's domain (a NaN included), testing them over the pool.
///
/// The test is a conjunction over the coordinates, so which chunk finds a violation cannot change
/// the outcome; the error itself is raised by the serial check of the space.
///
/// Raises:
///     value_error: A coordinate is outside the domain of `s`.
template <StorageFloat T, StorageFloat P>
void check_domain(const BsplineSpace1D<T>& s, std::span<const P> pts) {
  std::atomic<bool> inside{true};
  parallel::parallel_for(pts.size(), parallel::WorkEstimate{domain_check_ops},
                         [&](std::size_t begin, std::size_t end) {
                           if (!s.template all_in_domain<P>(pts.subspan(begin, end - begin))) {
                             inside.store(false, std::memory_order_relaxed);
                           }
                         });
  if (!inside.load(std::memory_order_relaxed)) {
    s.template check_in_domain<P>(pts);
  }
}

/// Tabulate each direction's basis (`orders` empty) or its derivative rows `0..orders[d]`, after
/// checking its coordinates against the domain.
///
/// Returns:
///     bool: False when a derivative recurrence met a zero divisor (see
///     `BsplineSpace1D::tabulate_basis_derivatives`); `tables` is then incomplete.
///
/// Raises:
///     value_error: A coordinate (NaN included) is outside its direction's domain.
template <StorageFloat T, StorageFloat P>
[[nodiscard]] bool tabulate_directions(const BsplineSpace<T>& space, const PointSet<P>& ps,
                                       std::span<const int> orders,
                                       std::vector<DirTable<P>>& tables) {
  bool complete = true;
  for (std::size_t d = 0; d < space.dim(); ++d) {
    const BsplineSpace1D<T>& s = space.spaces()[d];
    check_domain<T, P>(s, ps.dirs[d]);
    const auto order = static_cast<std::size_t>(s.degree()) + 1;
    if (orders.empty()) {
      auto [values, first] = s.template tabulate_basis<P>(ps.dirs[d], false);
      tables.push_back(DirTable<P>{std::move(values), std::move(first), order, 1});
    } else {
      auto r = s.template tabulate_basis_derivatives<P>(ps.dirs[d], orders[d], false);
      complete = complete && r.complete;
      tables.push_back(DirTable<P>{std::move(r.values), std::move(r.first), order,
                                   static_cast<std::size_t>(orders[d]) + 1});
    }
  }
  return complete;
}

/// The coordinates per direction, for a cursor.
template <StorageFloat P>
[[nodiscard]] std::vector<std::size_t> counts_of(const PointSet<P>& ps) {
  std::vector<std::size_t> counts;
  for (const auto& d : ps.dirs) {
    counts.push_back(d.size());
  }
  return counts;
}

/// Set the point's rows (row `k[d]` of direction `d`) and first functions, and form its local
/// weights and offsets.
template <StorageFloat T, StorageFloat P>
void prepare_point(const Layout& L, const std::vector<DirTable<P>>& tables, const PointCursor& cur,
                   std::span<const std::size_t> k, PointScratch<T, P>& s) {
  for (std::size_t d = 0; d < L.dim; ++d) {
    s.rows[d] = tables[d].row(cur.at(d), k[d]);
    s.first[d] = tables[d].first.data()[cur.at(d)];
  }
  tensor_row<P, T>(s.rows, L.order, s.product.data(), s.weights.data());
}

/// Operations per point of the scattered contraction: the tensor weights (at most one product per
/// local function and direction, the first direction forming none) and a multiply-add, two
/// operations, per local function and coordinate. An admitted estimate that counts the products
/// of every direction.
[[nodiscard]] inline double contraction_ops(const Layout& L) {
  return static_cast<double>(L.nloc) *
         (static_cast<double>(L.dim) + 2.0 * static_cast<double>(L.cp_size));
}

/// Operations per point of the 1D value tabulations of every direction, as the tabulation kernels
/// size their own loops (a lattice shares them across its points, so this overestimates it).
template <StorageFloat T>
[[nodiscard]] double tabulation_ops(const Bspline<T>& b) {
  double ops = 0.0;
  for (const auto& s : b.rep()->space.spaces()) {
    ops += basis_ops_per_point(s.degree());
  }
  return ops;
}

/// Operations per point of a whole value evaluation: the tabulations and the contraction (a lattice
/// shares the tabulations across its points, and sum factorisation does less than this, so it
/// overestimates both). A binding sizes its release of the interpreter lock with it.
template <StorageFloat T>
[[nodiscard]] double value_ops_per_point(const Bspline<T>& b) {
  return tabulation_ops(b) + contraction_ops(make_layout(b));
}

/// Rows of one direction for a mode product (see `contract_lattice`).
///
/// Attributes:
///     values (const P*): Row 0 of the direction's table; row `j` starts `stride` values on.
///     stride (std::size_t): Values between consecutive rows.
///     first (const std::int64_t*): First supported function of each row.
///     order (std::size_t): `degree + 1`, the terms of a row.
///     nb (std::size_t): Basis functions of the direction, the extent contracted.
///     periodic (bool): Whether indices wrap modulo `nb`.
template <StorageFloat P>
struct StageRows {
  const P* values;
  std::size_t stride;
  const std::int64_t* first;
  std::size_t order;
  std::size_t nb;
  bool periodic;
};

/// The index `i` of a direction, wrapped modulo `n` when it is periodic.
[[nodiscard]] inline std::size_t wrapped(std::size_t i, std::size_t n, bool periodic) noexcept {
  return periodic && i >= n ? i % n : i;
}

/// `out[i] = sum_a coeff[a] src[wrapped(first + a) * inner + i]` for `i < len`, the terms added in
/// ascending `a` in `T`, the coefficients rounded to `T` first.
///
/// A `noexcept` function of its own so that GCC vectorises the loops over `i`: written inside a
/// body that may throw (a `parallel_for` chunk), GCC 14 with `-ffp-contract=on` treats the
/// contracted `fma` as a statement that can throw and leaves the loop scalar.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T, StorageFloat P>
void combine_terms(T* out, const T* src, const P* coeff, std::size_t order, std::size_t first,
                   std::size_t n, bool periodic, std::size_t inner, std::size_t len) noexcept {
  {
    const T c0 = static_cast<T>(coeff[0]);
    const T* s0 = src + wrapped(first, n, periodic) * inner;
    for (std::size_t i = 0; i < len; ++i) {
      out[i] = c0 * s0[i];
    }
  }
  for (std::size_t a = 1; a < order; ++a) {
    const T ca = static_cast<T>(coeff[a]);
    const T* sa = src + wrapped(first + a, n, periodic) * inner;
    for (std::size_t i = 0; i < len; ++i) {
      out[i] = out[i] + ca * sa[i];
    }
  }
}

/// One mode product of `contract_lattice`: `out(o, j, :) = sum_a B(j, a) src(o, first_j + a, :)`
/// for `o < outer`, `j < m`, over rows of `inner` contiguous values.
///
/// A row is cut into tiles of at most `chunk_ops / (2 order)` values, so that a row too long for
/// one chunk is shared out and a short one is not split; the tile is a function of the extents
/// only. Each tile is one parallel item that writes its own values, each the sum over `a` in
/// ascending order, whichever thread runs it.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T, StorageFloat P>
void lattice_stage(const T* src, const StageRows<P>& dir, std::size_t outer, std::size_t m,
                   std::size_t inner, T* out) {
  const std::size_t order = dir.order;
  const auto tile_cap =
      static_cast<std::size_t>(std::ceil(parallel::chunk_ops / (2.0 * static_cast<double>(order))));
  const std::size_t tile = (std::min)(inner, tile_cap);
  const std::size_t tiles = (inner - 1) / tile + 1;
  // One tile costs a multiply-add per term and value.
  const double ops = 2.0 * static_cast<double>(order) * static_cast<double>(tile);
  parallel::parallel_for(
      outer * m * tiles, parallel::WorkEstimate{ops}, [&](std::size_t begin, std::size_t end) {
        std::size_t t = begin % tiles;
        std::size_t row = begin / tiles;
        std::size_t o = row / m;
        std::size_t j = row - o * m;
        for (std::size_t item = begin; item < end; ++item) {
          const std::size_t offset = t * tile;
          combine_terms<T, P>(out + row * inner + offset, src + o * dir.nb * inner + offset,
                              dir.values + j * dir.stride, order,
                              static_cast<std::size_t>(dir.first[j]), dir.nb, dir.periodic, inner,
                              (std::min)(tile, inner - offset));
          if (++t == tiles) {
            t = 0;
            ++row;
            if (++j == m) {
              j = 0;
              ++o;
            }
          }
        }
      });
}

/// Contract the control points with one row table per direction on a lattice (sum
/// factorisation): the value, or the derivative of order `k[d]` per direction, at every point of
/// the tensor product of the coordinates.
///
/// One mode product per direction (`lattice_stage`), in the order of directions that does the
/// fewer multiply-adds: first to last, where stage `d` turns `(m_0..m_{d-1}, n_d..n_{dim-1}, cp)`
/// into `(m_0..m_d, n_{d+1}.., cp)`, or last to first, where it turns `(n_0..n_d, m_{d+1}.., cp)`
/// into `(n_0..n_{d-1}, m_d.., cp)` (`n` the basis functions, `m` the points of a direction). Last
/// to first is taken when the counts are equal: its last stage, the largest, then has the longest
/// contiguous rows. The two orders add the same products in another order, so they differ only by
/// rounding; the choice depends on the extents alone, never on the thread count.
///
/// Args:
///     L (const Layout&): The addressing.
///     cp (const T*): Control points, row-major over `(nb_0, ..., nb_{dim-1}, cp_size)`.
///     tables (const std::vector<DirTable<P>>&): One table per direction.
///     k (std::span<const std::size_t>): The row of each direction's table to use.
///     counts (std::span<const std::size_t>): Points per direction (each at least 1).
///     dst (T*): Output `(counts[0], ..., counts[dim-1], cp_size)`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T, StorageFloat P>
void contract_lattice(const Layout& L, const T* cp, const std::vector<DirTable<P>>& tables,
                      std::span<const std::size_t> k, std::span<const std::size_t> counts, T* dst) {
  const std::size_t dim = L.dim;
  std::vector<StageRows<P>> dirs;
  dirs.reserve(dim);
  for (std::size_t d = 0; d < dim; ++d) {
    dirs.push_back({tables[d].values.data() + k[d] * L.order[d], tables[d].rows * L.order[d],
                    tables[d].first.data(), L.order[d], L.nb[d], L.periodic[d] != 0});
  }
  // Values written by the stage of direction d, in either order: the directions already
  // contracted carry their point counts, the others their numbers of functions.
  const auto written = [&](std::size_t d, bool forward) {
    std::size_t v = L.cp_size;
    for (std::size_t e = 0; e < dim; ++e) {
      const bool done = forward ? e <= d : e >= d;
      v *= done ? counts[e] : L.nb[e];
    }
    return v;
  };
  double cost_forward = 0.0;
  double cost_backward = 0.0;
  for (std::size_t d = 0; d < dim; ++d) {
    const double terms = static_cast<double>(L.order[d]);
    cost_forward += terms * static_cast<double>(written(d, true));
    cost_backward += terms * static_cast<double>(written(d, false));
  }
  const bool forward = cost_forward < cost_backward;
  std::size_t scratch = 0;
  for (std::size_t s = 0; s + 1 < dim; ++s) {
    scratch = (std::max)(scratch, written(forward ? s : dim - 1 - s, forward));
  }
  // Intermediates alternate between two buffers; with two directions only one is needed.
  NdArray<T> buf0({scratch}, for_overwrite);
  NdArray<T> buf1({dim > 2 ? scratch : 0}, for_overwrite);
  const T* src = cp;
  for (std::size_t s = 0; s < dim; ++s) {
    const std::size_t d = forward ? s : dim - 1 - s;
    std::size_t outer = 1;
    std::size_t inner = L.cp_size;
    for (std::size_t e = 0; e < d; ++e) {
      outer *= forward ? counts[e] : L.nb[e];
    }
    for (std::size_t e = d + 1; e < dim; ++e) {
      inner *= forward ? L.nb[e] : counts[e];
    }
    T* out = s + 1 == dim ? dst : (s % 2 == 0 ? buf0.data() : buf1.data());
    lattice_stage<T, P>(src, dirs[d], outer, counts[d], inner, out);
    src = out;
  }
}

/// Divide the first `rank` coordinates of each of the rows `[begin, end)` of `cp_size` values by
/// the weight at `rank`, which stays.
///
/// A `noexcept` function of its own, for the reason of `combine_terms`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
void project_rows(T* data, std::size_t begin, std::size_t end, std::size_t cp_size,
                  std::size_t rank) noexcept {
  for (std::size_t r = begin; r < end; ++r) {
    project_in_place<T>(data + r * cp_size, rank);
  }
}

/// Project `count` rows of homogeneous values over the pool (a division per coordinate).
template <StorageFloat T>
void project_all(T* data, std::size_t count, std::size_t cp_size, std::size_t rank) {
  parallel::parallel_for(count, parallel::WorkEstimate{(std::max)(static_cast<double>(rank), 1.0)},
                         [&](std::size_t begin, std::size_t end) {
                           project_rows<T>(data, begin, end, cp_size, rank);
                         });
}

/// Scratch of the 1D chunk loop: one point's basis values and the recurrence's knot differences.
template <StorageFloat T, StorageFloat P>
struct Row1DScratch {
  std::vector<P> basis;
  std::vector<T> left;
  std::vector<T> right;
  std::vector<T> weights;
  std::vector<std::size_t> offs;
  explicit Row1DScratch(std::size_t order)
      : basis(order), left(order), right(order), weights(order), offs(order) {}
};

/// Evaluate the points `[begin, end)` of a 1D B-spline: each point's basis row is formed in the
/// scratch (Cox-de Boor, as `BsplineSpace1D::tabulate_basis` forms it) and contracted at once.
///
/// A `noexcept` function of its own, so the loop over the points is not a body that may throw.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed), the domain included.
template <StorageFloat T, StorageFloat P>
void evaluate_chunk_1d(const Layout& L, std::span<const T> knots, int degree, const T* cp,
                       bool rational, std::size_t rank, std::span<const P> pts, std::size_t begin,
                       std::size_t end, T* out, Row1DScratch<T, P>& sc) noexcept {
  const std::size_t order = L.order[0];
  for (std::size_t i = begin; i < end; ++i) {
    const SpanAndFirst sf = span_and_first<T, P>(knots, degree, pts[i]);
    basis_funcs_point<T, P>(knots.data(), degree, sf.knot_id, pts[i], sc.basis.data(),
                            sc.left.data(), sc.right.data());
    for (std::size_t a = 0; a < order; ++a) {
      sc.weights[a] = static_cast<T>(sc.basis[a]);
    }
    local_offsets(L, &sf.first, sc.offs.data());
    T* row = out + i * L.cp_size;
    contract_local<T>(cp, sc.offs.data(), sc.weights.data(), order, L.cp_size, row);
    if (rational) {
      project_in_place<T>(row, rank);
    }
  }
}

/// Evaluate a 1D B-spline (not Bezier-like) at `pts` into `out` (`pts.size() x cp_size`), the
/// weight column kept.
///
/// Raises:
///     value_error: A point (NaN included) is outside the domain; `out` is then unwritten.
template <StorageFloat T, StorageFloat P>
void evaluate_1d(const Bspline<T>& b, const Layout& L, std::span<const P> pts, T* out) {
  const BsplineSpace1D<T>& s = b.rep()->space.spaces()[0];
  check_domain<T, P>(s, pts);
  const int degree = s.degree();
  // The recurrence, then a multiply-add (two operations) per term and coordinate, and the
  // division of a rational B-spline.
  const double ops = basis_ops_per_point(degree) +
                     2.0 * static_cast<double>(L.order[0]) * static_cast<double>(L.cp_size) +
                     static_cast<double>(b.rank());
  const T* cp = b.rep()->data.get();
  const bool rational = b.rational();
  const std::size_t rank = b.rank();
  parallel::parallel_for(
      pts.size(), parallel::WorkEstimate{ops}, [&](std::size_t begin, std::size_t end) {
        Row1DScratch<T, P> sc(L.order[0]);
        evaluate_chunk_1d<T, P>(L, s.knots(), degree, cp, rational, rank, pts, begin, end, out, sc);
      });
}

/// Evaluate at every point of `ps` into `out` (`count x cp_size`), the weight column kept.
template <StorageFloat T, StorageFloat P>
void evaluate_into(const Bspline<T>& b, const PointSet<P>& ps, T* out) {
  const Layout L = make_layout(b);
  if (L.dim == 1 && !b.rep()->space.spaces()[0].has_bezier_like_knots()) {
    evaluate_1d<T, P>(b, L, ps.dirs[0], out);
    return;
  }
  std::vector<DirTable<P>> tables;
  tables.reserve(L.dim);
  static_cast<void>(tabulate_directions<T, P>(b.rep()->space, ps, {}, tables));
  const std::vector<std::size_t> counts = counts_of(ps);
  const std::vector<std::size_t> k(L.dim, 0);
  const T* cp = b.rep()->data.get();
  const bool rational = b.rational();
  const std::size_t rank = b.rank();
  if (ps.lattice && L.dim > 1) {
    contract_lattice<T, P>(L, cp, tables, k, counts, out);
    if (rational) {
      project_all<T>(out, ps.count, L.cp_size, rank);
    }
    return;
  }
  // The tables are formed above, by kernels that size their own loops: this loop contracts.
  parallel::parallel_for(ps.count, parallel::WorkEstimate{contraction_ops(L)},
                         [&](std::size_t begin, std::size_t end) {
                           PointScratch<T, P> s(L);
                           PointCursor cur(ps.lattice, counts, L.dim, begin);
                           for (std::size_t p = begin; p < end; ++p) {
                             prepare_point<T, P>(L, tables, cur, k, s);
                             local_offsets(L, s.first.data(), s.offs.data());
                             T* row = out + p * L.cp_size;
                             contract_local<T>(cp, s.offs.data(), s.weights.data(), L.nloc,
                                               L.cp_size, row);
                             if (rational) {
                               project_in_place<T>(row, rank);
                             }
                             cur.advance();
                           }
                         });
}

/// The order of the quotient rule over the multi-indices `0 <= i <= orders` and, for each, what it
/// subtracts.
///
/// Attributes:
///     nbox (std::size_t): `prod(orders + 1)`.
///     order (std::vector<std::size_t>): The flat (C order) multi-indices, by ascending total
///         order and, within one, ascending flat index.
///     terms (std::vector<std::vector<Term>>): For `order[pos]`, the terms `binomial(k, i)
///         w_i R_{k - i}` for every `0 < i <= k` componentwise, by the same order of `i`.
template <StorageFloat T>
struct QuotientPlan {
  struct Term {
    std::size_t i;
    std::size_t kmi;
    T coeff;
  };
  std::size_t nbox = 0;
  std::vector<std::size_t> order;
  std::vector<std::vector<Term>> terms;
  std::vector<std::size_t> multi;  // nbox x dim
};

/// Build the plan for derivative orders `orders` (non-negative).
///
/// The binomial coefficient `prod_d C(k_d, i_d)` is formed in `double` from correctly rounded
/// factors (exact while it stays below 2^53) and rounded once to `T`.
template <StorageFloat T>
[[nodiscard]] QuotientPlan<T> make_quotient_plan(std::span<const int> orders) {
  const std::size_t dim = orders.size();
  std::vector<std::size_t> ext(dim);
  for (std::size_t d = 0; d < dim; ++d) {
    ext[d] = static_cast<std::size_t>(orders[d]) + 1;
  }
  QuotientPlan<T> plan;
  plan.nbox = checked_product(ext);
  plan.multi.resize(plan.nbox * dim);
  std::vector<std::size_t> total(plan.nbox, 0);
  for (std::size_t f = 0; f < plan.nbox; ++f) {
    std::size_t rest = f;
    for (std::size_t d = dim; d-- > 0;) {
      plan.multi[f * dim + d] = rest % ext[d];
      rest /= ext[d];
      total[f] += plan.multi[f * dim + d];
    }
  }
  std::vector<std::size_t> box_stride(dim, 1);
  for (std::size_t d = dim; d-- > 1;) {
    box_stride[d - 1] = box_stride[d] * ext[d];
  }
  plan.order.resize(plan.nbox);
  for (std::size_t f = 0; f < plan.nbox; ++f) {
    plan.order[f] = f;
  }
  std::stable_sort(plan.order.begin(), plan.order.end(),
                   [&](std::size_t a, std::size_t b) { return total[a] < total[b]; });
  plan.terms.resize(plan.nbox);
  for (std::size_t pos = 0; pos < plan.nbox; ++pos) {
    const std::size_t k = plan.order[pos];
    for (std::size_t q = 1; q < plan.nbox; ++q) {
      const std::size_t i = plan.order[q];
      if (total[i] > total[k]) {
        break;
      }
      bool below = true;
      for (std::size_t d = 0; d < dim && below; ++d) {
        below = plan.multi[i * dim + d] <= plan.multi[k * dim + d];
      }
      if (!below) {
        continue;
      }
      double coeff = 1.0;
      std::size_t kmi = 0;
      for (std::size_t d = 0; d < dim; ++d) {
        const auto kd = static_cast<long long>(plan.multi[k * dim + d]);
        const auto id = static_cast<long long>(plan.multi[i * dim + d]);
        coeff *= binomial_double(kd, id);
        kmi += static_cast<std::size_t>(kd - id) * box_stride[d];
      }
      plan.terms[pos].push_back({i, kmi, static_cast<T>(coeff)});
    }
  }
  return plan;
}

/// The generalised quotient rule (Piegl and Tiller A4.2 in several variables) at one point.
///
/// Given the homogeneous derivatives `H[m]` (`cp_size` values each, the weight last) for every
/// multi-index `m` of the box, writes the derivatives of the projected map `R[k] = (H_k - sum_i
/// binomial(k, i) w_i R_{k - i}) / w_0` into `R` (`rank` values per multi-index), the sum over
/// `0 < i <= k` in the order of the plan and each term formed as `(binomial w_i) R_{k - i}`.
///
/// Args:
///     plan (const QuotientPlan<T>&): The plan.
///     H (const T*): `nbox x cp_size` homogeneous derivatives.
///     cp_size (std::size_t): Coordinates per control point.
///     rank (std::size_t): `cp_size - 1`.
///     R (T*): Output, `nbox x rank`.
///     v (T*): Scratch, `rank` values.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
void quotient_rule(const QuotientPlan<T>& plan, const T* H, std::size_t cp_size, std::size_t rank,
                   T* R, T* v) noexcept {
  const T w0 = H[rank];
  for (std::size_t pos = 0; pos < plan.nbox; ++pos) {
    const std::size_t k = plan.order[pos];
    for (std::size_t c = 0; c < rank; ++c) {
      v[c] = H[k * cp_size + c];
    }
    for (const auto& t : plan.terms[pos]) {
      const T cw = t.coeff * H[t.i * cp_size + rank];
      const T* r = R + t.kmi * rank;
      for (std::size_t c = 0; c < rank; ++c) {
        v[c] = v[c] - cw * r[c];
      }
    }
    for (std::size_t c = 0; c < rank; ++c) {
      R[k * rank + c] = v[c] / w0;
    }
  }
}

/// Refuse derivative orders of the wrong number or sign.
///
/// Raises:
///     value_error: `orders` does not have `dim` entries, or an entry is negative (the length
///     first).
inline void check_orders(std::size_t dim, std::span<const int> orders) {
  if (orders.size() != dim) {
    throw value_error("len(orders) must equal spline.dim (" + std::to_string(dim) + "), got " +
                      std::to_string(orders.size()));
  }
  for (std::size_t d = 0; d < dim; ++d) {
    if (orders[d] < 0) {
      throw value_error("orders[" + std::to_string(d) + "] must be >= 0, got " +
                        std::to_string(orders[d]));
    }
  }
}

/// Operations per point of the contraction of a derivative evaluation, once the rows are
/// tabulated: for the one multi-index requested (not rational), or for every multi-index of the box
/// and the quotient rule's terms (rational: at most `nbox^2 / 2` pairs, each a multiply-add per
/// coordinate, counted as `nbox^2` operations per coordinate). Tolerant of orders of the wrong
/// number or sign, which it reads as zero: a binding sizes its work before the call validates.
template <StorageFloat T>
[[nodiscard]] double derivative_contraction_ops(const Bspline<T>& b, std::span<const int> orders) {
  const Layout L = make_layout(b);
  if (!b.rational()) {
    return contraction_ops(L);
  }
  double nbox = 1.0;
  for (std::size_t d = 0; d < L.dim; ++d) {
    nbox *= static_cast<double>(d < orders.size() ? (std::max)(orders[d], 0) : 0) + 1.0;
  }
  return nbox * contraction_ops(L) + nbox * nbox * static_cast<double>(b.rank());
}

/// Operations per point of a whole derivative evaluation, from the work it does: each direction's
/// derivative-row tabulation and `derivative_contraction_ops` (a lattice shares the tabulations
/// across its points, so this overestimates it). Tolerant of orders of the wrong number or sign,
/// which it reads as zero.
template <StorageFloat T>
[[nodiscard]] double derivative_ops_per_point(const Bspline<T>& b, std::span<const int> orders) {
  double ops = 0.0;
  std::size_t d = 0;
  for (const auto& s : b.rep()->space.spaces()) {
    ops += derivs_ops_per_point(s.degree(), d < orders.size() ? (std::max)(orders[d], 0) : 0);
    ++d;
  }
  return ops + derivative_contraction_ops(b, orders);
}

/// Evaluate a derivative at every point of `ps` into `out` (`count x rank`).
///
/// Returns:
///     bool: False when a derivative recurrence met a zero divisor; `out` is then unwritten.
template <StorageFloat T, StorageFloat P>
[[nodiscard]] bool derivatives_into(const Bspline<T>& b, const PointSet<P>& ps,
                                    std::span<const int> orders, T* out) {
  const Layout L = make_layout(b);
  std::vector<DirTable<P>> tables;
  tables.reserve(L.dim);
  if (!tabulate_directions<T, P>(b.rep()->space, ps, orders, tables)) {
    return false;
  }
  const std::vector<std::size_t> counts = counts_of(ps);
  const T* cp = b.rep()->data.get();
  const std::size_t rank = b.rank();
  // The tables are formed above, by kernels that size their own loops: the loops below contract.
  const parallel::WorkEstimate work{derivative_contraction_ops(b, orders)};
  if (!b.rational()) {
    std::vector<std::size_t> k(L.dim);
    for (std::size_t d = 0; d < L.dim; ++d) {
      k[d] = static_cast<std::size_t>(orders[d]);
    }
    if (ps.lattice && L.dim > 1) {
      contract_lattice<T, P>(L, cp, tables, k, counts, out);
      return true;
    }
    parallel::parallel_for(ps.count, work, [&](std::size_t begin, std::size_t end) {
      PointScratch<T, P> s(L);
      PointCursor cur(ps.lattice, counts, L.dim, begin);
      for (std::size_t p = begin; p < end; ++p) {
        prepare_point<T, P>(L, tables, cur, k, s);
        local_offsets(L, s.first.data(), s.offs.data());
        contract_local<T>(cp, s.offs.data(), s.weights.data(), L.nloc, L.cp_size, out + p * rank);
        cur.advance();
      }
    });
    return true;
  }
  const QuotientPlan<T> plan = make_quotient_plan<T>(orders);
  const std::size_t nbox = plan.nbox;
  parallel::parallel_for(ps.count, work, [&](std::size_t begin, std::size_t end) {
    PointScratch<T, P> s(L);
    std::vector<T> H(nbox * L.cp_size);
    std::vector<T> R(nbox * rank);
    std::vector<T> v(rank);
    std::vector<std::size_t> k(L.dim);
    PointCursor cur(ps.lattice, counts, L.dim, begin);
    for (std::size_t p = begin; p < end; ++p) {
      for (std::size_t m = 0; m < nbox; ++m) {
        for (std::size_t d = 0; d < L.dim; ++d) {
          k[d] = plan.multi[m * L.dim + d];
        }
        prepare_point<T, P>(L, tables, cur, k, s);
        if (m == 0) {
          local_offsets(L, s.first.data(), s.offs.data());
        }
        contract_local<T>(cp, s.offs.data(), s.weights.data(), L.nloc, L.cp_size,
                          H.data() + m * L.cp_size);
      }
      quotient_rule<T>(plan, H.data(), L.cp_size, rank, R.data(), v.data());
      const T* target = R.data() + (nbox - 1) * rank;
      std::copy(target, target + rank, out + p * rank);
      cur.advance();
    }
  });
  return true;
}

/// Get the extents of a lattice evaluation: the points per direction, then `last`.
template <StorageFloat P>
[[nodiscard]] std::vector<std::size_t> lattice_shape(const quad::PointsLattice<P>& lattice,
                                                     std::size_t last) {
  std::vector<std::size_t> shape;
  for (std::size_t d = 0; d < lattice.dim(); ++d) {
    shape.push_back(lattice.pts(d).size());
  }
  shape.push_back(last);
  return shape;
}

/// Copy the first `rank` of `cp_size` columns of `n` rows into a new array of `shape`.
template <StorageFloat T>
[[nodiscard]] NdArray<T> drop_weight(const NdArray<T>& full, std::size_t n, std::size_t cp_size,
                                     std::vector<std::size_t> shape) {
  const std::size_t rank = cp_size - 1;
  NdArray<T> out(std::move(shape), for_overwrite);
  for (std::size_t i = 0; i < n; ++i) {
    std::copy(full.data() + i * cp_size, full.data() + i * cp_size + rank, out.data() + i * rank);
  }
  return out;
}

}  // namespace detail

/// The values of a derivative evaluation, and whether they are complete.
///
/// Attributes:
///     values (NdArray<T>): `(n, rank)` (points) or `(n_0, ..., n_{dim-1}, rank)` (lattice).
///     complete (bool): False when the basis derivative recurrence met a zero divisor, which a
///         basis function of empty support produces (see `BsplineSpace1D::tabulate_basis_
///         derivatives`); `values` is then unwritten.
template <StorageFloat T>
struct [[nodiscard]] DerivativeValues {
  NdArray<T> values;
  bool complete;
};

/// Evaluate a B-spline at arbitrary parametric points into `out`.
///
/// For a rational B-spline the first `rank()` columns of `out` are divided by the weight column,
/// which is left in place (so `out` has `cp_size()` columns either way).
///
/// Special values: a rational B-spline's division by the interpolated weight is not guarded. A zero
/// weight gives `+-Inf` or NaN, and a NaN weight or control point gives NaN, in the affected values
/// only; none is reported by an error or a warning. A non-rational B-spline divides by nothing. A
/// NaN control point reaches every value whose `degree + 1` supporting functions per direction
/// include it, a zero basis value there included (`0 * NaN` is NaN).
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///     pts (mdview<const P, 2>): Points, extents `(n, dim)`; any float type, the arithmetic of a
///         mixed pair of knots and points being done in their common type.
///     out (mdview<T, 2>): Output, extents `(n, cp_size)`.
///
/// Raises:
///     value_error: `pts` has another column count; `out` has other extents; a point (NaN
///         included) is outside the domain of its direction, up to the space's tolerance.
template <StorageFloat T, StorageFloat P>
void evaluate(const Bspline<T>& b, mdview<const P, 2> pts, mdview<T, 2> out) {
  detail::check_columns(b.dim(), pts.extent(1));
  basis::detail::check_out<T, 2>(out, {pts.extent(0), b.cp_size()});
  detail::evaluate_into<T, P>(b, detail::make_point_set<P>(pts), out.data_handle());
}

/// Evaluate a B-spline at arbitrary parametric points.
///
/// Special values: a rational B-spline's division by the interpolated weight is not guarded. A zero
/// weight gives `+-Inf` or NaN, and a NaN weight or control point gives NaN, in the affected values
/// only; none is reported by an error or a warning. A non-rational B-spline divides by nothing. A
/// NaN control point reaches every value whose `degree + 1` supporting functions per direction
/// include it, a zero basis value there included (`0 * NaN` is NaN).
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///     pts (mdview<const P, 2>): Points, extents `(n, dim)`.
///
/// Returns:
///     NdArray<T>: Values, extents `(n, rank)` (never squeezed); the weight of a rational
///     B-spline is divided out and dropped.
///
/// Raises:
///     value_error: As the overload taking `out`.
template <StorageFloat T, StorageFloat P>
[[nodiscard]] NdArray<T> evaluate(const Bspline<T>& b, mdview<const P, 2> pts) {
  detail::check_columns(b.dim(), pts.extent(1));
  NdArray<T> full({pts.extent(0), b.cp_size()}, for_overwrite);
  detail::evaluate_into<T, P>(b, detail::make_point_set<P>(pts), full.data());
  if (!b.rational()) {
    return full;
  }
  return detail::drop_weight<T>(full, pts.extent(0), b.cp_size(), {pts.extent(0), b.rank()});
}

/// Evaluate a B-spline on a tensor-product lattice into `out`.
///
/// Special values: a rational B-spline's division by the interpolated weight is not guarded. A zero
/// weight gives `+-Inf` or NaN, and a NaN weight or control point gives NaN, in the affected values
/// only; none is reported by an error or a warning. A non-rational B-spline divides by nothing. A
/// NaN control point reaches every value whose `degree + 1` supporting functions per direction
/// include it, a zero basis value there included (`0 * NaN` is NaN).
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///     lattice (const quad::PointsLattice<P>&): The lattice, of dimension `dim()`.
///     out (mdview<T, 2>): Output, extents `(lattice.num_points(), cp_size)`, the lattice points in
///         C order (the last direction varies fastest); the weight column as for points.
///
/// Raises:
///     value_error: The lattice has another dimension; `out` has other extents; a coordinate is
///         outside the domain of its direction.
template <StorageFloat T, StorageFloat P>
void evaluate(const Bspline<T>& b, const quad::PointsLattice<P>& lattice, mdview<T, 2> out) {
  detail::check_lattice_dim(b.dim(), lattice.dim());
  basis::detail::check_out<T, 2>(out, {lattice.num_points(), b.cp_size()});
  detail::evaluate_into<T, P>(b, detail::make_point_set<P>(lattice), out.data_handle());
}

/// Evaluate a B-spline on a tensor-product lattice.
///
/// Special values: a rational B-spline's division by the interpolated weight is not guarded. A zero
/// weight gives `+-Inf` or NaN, and a NaN weight or control point gives NaN, in the affected values
/// only; none is reported by an error or a warning. A non-rational B-spline divides by nothing. A
/// NaN control point reaches every value whose `degree + 1` supporting functions per direction
/// include it, a zero basis value there included (`0 * NaN` is NaN).
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///     lattice (const quad::PointsLattice<P>&): The lattice, of dimension `dim()`.
///
/// Returns:
///     NdArray<T>: Values, extents `(n_0, ..., n_{dim-1}, rank)` for `n_d` points in direction
///     `d` (never squeezed); the weight of a rational B-spline is divided out and dropped.
///
/// Raises:
///     value_error: As the overload taking `out`.
template <StorageFloat T, StorageFloat P>
[[nodiscard]] NdArray<T> evaluate(const Bspline<T>& b, const quad::PointsLattice<P>& lattice) {
  detail::check_lattice_dim(b.dim(), lattice.dim());
  if (!b.rational()) {
    NdArray<T> out(detail::lattice_shape<P>(lattice, b.cp_size()), for_overwrite);
    detail::evaluate_into<T, P>(b, detail::make_point_set<P>(lattice), out.data());
    return out;
  }
  const std::size_t n = lattice.num_points();
  NdArray<T> full({n, b.cp_size()}, for_overwrite);
  detail::evaluate_into<T, P>(b, detail::make_point_set<P>(lattice), full.data());
  return detail::drop_weight<T>(full, n, b.cp_size(), detail::lattice_shape<P>(lattice, b.rank()));
}

/// Evaluate a partial derivative of a B-spline at arbitrary parametric points into `out`.
///
/// `orders[d]` is the derivative order in direction `d`; an order above the degree gives zero.
/// For a rational B-spline the homogeneous derivatives of every multi-index up to `orders` are
/// formed and the generalised quotient rule (Piegl and Tiller A4.2 in several variables) gives the
/// derivative of the projected map; the weight column does not appear in `out`.
///
/// Special values: the quotient rule divides by the interpolated weight, unguarded. A zero weight
/// gives `+-Inf` or NaN, and a NaN weight or control point gives NaN, in the affected values only;
/// none is reported by an error or a warning (the one reported failure is the basis recurrence's
/// zero divisor, below). A non-rational B-spline divides by nothing. A NaN control point reaches
/// every value whose `degree + 1` supporting functions per direction include it, as for values.
///
/// The binomial coefficients of the quotient rule come from `binomial_double`: each factor is
/// correctly rounded, and the product of the factors is exact while it stays below 2^53, which
/// holds for orders up to 56 in one direction (`C(56, 28)` is below 2^53 and `C(57, 28)` is not)
/// and for products of smaller ones; past it the coefficient carries one
/// double rounding.
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///     pts (mdview<const P, 2>): Points, extents `(n, dim)`.
///     orders (std::span<const int>): The derivative order per direction, `dim` values `>= 0`.
///     out (mdview<T, 2>): Output, extents `(n, rank)`.
///
/// Returns:
///     bool: True when every point was evaluated; false when the basis derivative recurrence met a
///     zero divisor (a knot repeated more than `degree + 1` times within a point's support), and
///     `out` is then unwritten.
///
/// Raises:
///     value_error: `orders` has the wrong length or a negative entry (checked first); `pts` has
///         another column count; `out` has other extents; a point (NaN included) is outside the
///         domain of its direction.
template <StorageFloat T, StorageFloat P>
[[nodiscard]] bool evaluate_derivatives(const Bspline<T>& b, mdview<const P, 2> pts,
                                        std::span<const int> orders, mdview<T, 2> out) {
  detail::check_orders(b.dim(), orders);
  detail::check_columns(b.dim(), pts.extent(1));
  basis::detail::check_out<T, 2>(out, {pts.extent(0), b.rank()});
  return detail::derivatives_into<T, P>(b, detail::make_point_set<P>(pts), orders,
                                        out.data_handle());
}

/// Evaluate a partial derivative of a B-spline at arbitrary parametric points.
///
/// Special values: the quotient rule divides by the interpolated weight, unguarded. A zero weight
/// gives `+-Inf` or NaN, and a NaN weight or control point gives NaN, in the affected values only;
/// none is reported by an error or a warning (the one reported failure is the basis recurrence's
/// zero divisor, below). A non-rational B-spline divides by nothing. A NaN control point reaches
/// every value whose `degree + 1` supporting functions per direction include it, as for values.
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///     pts (mdview<const P, 2>): Points, extents `(n, dim)`.
///     orders (std::span<const int>): The derivative order per direction.
///
/// Returns:
///     DerivativeValues<T>: Values, extents `(n, rank)` (never squeezed), and whether they are
///     complete.
///
/// Raises:
///     value_error: As the overload taking `out`.
template <StorageFloat T, StorageFloat P>
[[nodiscard]] DerivativeValues<T> evaluate_derivatives(const Bspline<T>& b, mdview<const P, 2> pts,
                                                       std::span<const int> orders) {
  detail::check_orders(b.dim(), orders);
  detail::check_columns(b.dim(), pts.extent(1));
  NdArray<T> out({pts.extent(0), b.rank()}, for_overwrite);
  const bool complete =
      detail::derivatives_into<T, P>(b, detail::make_point_set<P>(pts), orders, out.data());
  return {std::move(out), complete};
}

/// Evaluate a partial derivative of a B-spline on a tensor-product lattice into `out`.
///
/// Special values: the quotient rule divides by the interpolated weight, unguarded. A zero weight
/// gives `+-Inf` or NaN, and a NaN weight or control point gives NaN, in the affected values only;
/// none is reported by an error or a warning (the one reported failure is the basis recurrence's
/// zero divisor, below). A non-rational B-spline divides by nothing. A NaN control point reaches
/// every value whose `degree + 1` supporting functions per direction include it, as for values.
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///     lattice (const quad::PointsLattice<P>&): The lattice, of dimension `dim()`.
///     orders (std::span<const int>): The derivative order per direction.
///     out (mdview<T, 2>): Output, extents `(lattice.num_points(), rank)`, the lattice points in C
///         order.
///
/// Returns:
///     bool: As the overload for points.
///
/// Raises:
///     value_error: `orders` has the wrong length or a negative entry (checked first); the lattice
///         has another dimension; `out` has other extents; a coordinate is outside the domain of
///         its direction.
template <StorageFloat T, StorageFloat P>
[[nodiscard]] bool evaluate_derivatives(const Bspline<T>& b, const quad::PointsLattice<P>& lattice,
                                        std::span<const int> orders, mdview<T, 2> out) {
  detail::check_orders(b.dim(), orders);
  detail::check_lattice_dim(b.dim(), lattice.dim());
  basis::detail::check_out<T, 2>(out, {lattice.num_points(), b.rank()});
  return detail::derivatives_into<T, P>(b, detail::make_point_set<P>(lattice), orders,
                                        out.data_handle());
}

/// Evaluate a partial derivative of a B-spline on a tensor-product lattice.
///
/// Special values: the quotient rule divides by the interpolated weight, unguarded. A zero weight
/// gives `+-Inf` or NaN, and a NaN weight or control point gives NaN, in the affected values only;
/// none is reported by an error or a warning (the one reported failure is the basis recurrence's
/// zero divisor, below). A non-rational B-spline divides by nothing. A NaN control point reaches
/// every value whose `degree + 1` supporting functions per direction include it, as for values.
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///     lattice (const quad::PointsLattice<P>&): The lattice, of dimension `dim()`.
///     orders (std::span<const int>): The derivative order per direction.
///
/// Returns:
///     DerivativeValues<T>: Values, extents `(n_0, ..., n_{dim-1}, rank)` (never squeezed), and
///     whether they are complete.
///
/// Raises:
///     value_error: As the overload taking `out`.
template <StorageFloat T, StorageFloat P>
[[nodiscard]] DerivativeValues<T> evaluate_derivatives(const Bspline<T>& b,
                                                       const quad::PointsLattice<P>& lattice,
                                                       std::span<const int> orders) {
  detail::check_orders(b.dim(), orders);
  detail::check_lattice_dim(b.dim(), lattice.dim());
  NdArray<T> out(detail::lattice_shape<P>(lattice, b.rank()), for_overwrite);
  const bool complete =
      detail::derivatives_into<T, P>(b, detail::make_point_set<P>(lattice), orders, out.data());
  return {std::move(out), complete};
}

}  // namespace pantr::bspline
