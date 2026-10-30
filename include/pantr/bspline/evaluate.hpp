/// @file evaluate.hpp
/// @brief Evaluation of a `Bspline` and of its partial derivatives at points or on a lattice.
///
/// Every point is evaluated alone: the non-zero 1D basis values (or derivative rows) of each
/// direction, tabulated by `BsplineSpace1D` once per distinct coordinate, are multiplied into the
/// tensor-product weights of the point's `prod(degree + 1)` local functions, in `P` and left to
/// right over the directions, and stored in `T`; the control points of those functions (read modulo
/// `num_basis` in a periodic direction) are then summed in `T`, in ascending C order of the local
/// multi-index. A rational B-spline is evaluated on its homogeneous control points and divided by
/// the interpolated weight. Points are the unit of parallelism, and each writes its own output
/// row, so no result depends on the number of threads.
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
///     store (std::vector<std::vector<P>>): The columns of an arbitrary point set, copied out of
///     the
///         row-major array (empty for a lattice).
///     dirs (std::vector<std::span<const P>>): One coordinate array per direction: `count` values
///         viewing `store`, or the lattice's own coordinates.
template <StorageFloat P>
struct PointSet {
  bool lattice = false;
  std::size_t count = 0;
  std::vector<std::vector<P>> store;
  std::vector<std::span<const P>> dirs;
};

/// Gather the columns of a `(n, dim)` point array.
template <StorageFloat P>
[[nodiscard]] PointSet<P> make_point_set(mdview<const P, 2> pts) {
  PointSet<P> e;
  e.count = pts.extent(0);
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
    s.template check_in_domain<P>(ps.dirs[d]);
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

/// Operations per point of the contraction: the tensor weights (one product per local function and
/// direction) and a multiply-add per local function and coordinate.
[[nodiscard]] inline double contraction_ops(const Layout& L) {
  return static_cast<double>(L.nloc) *
         (static_cast<double>(L.dim) + 2.0 * static_cast<double>(L.cp_size));
}

/// Operations per point of the 1D tabulations of every direction (a lattice shares them across
/// points, so this overestimates it).
template <StorageFloat T>
[[nodiscard]] double tabulation_ops(const Bspline<T>& b) {
  double ops = 0.0;
  for (const auto& s : b.rep()->space.spaces()) {
    ops += basis_ops_per_point(s.degree());
  }
  return ops;
}

/// Evaluate at every point of `ps` into `out` (`count x cp_size`), the weight column kept.
template <StorageFloat T, StorageFloat P>
void evaluate_into(const Bspline<T>& b, const PointSet<P>& ps, T* out) {
  const Layout L = make_layout(b);
  std::vector<DirTable<P>> tables;
  tables.reserve(L.dim);
  static_cast<void>(tabulate_directions<T, P>(b.rep()->space, ps, {}, tables));
  const std::vector<std::size_t> counts = counts_of(ps);
  const std::vector<std::size_t> k(L.dim, 0);
  const T* cp = b.rep()->data.get();
  const bool rational = b.rational();
  const std::size_t rank = b.rank();
  const double ops = tabulation_ops(b) + contraction_ops(L);
  parallel::parallel_for(ps.count, parallel::WorkEstimate{ops}, sizeof(T) * L.cp_size,
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

/// Operations per point of a derivative evaluation, from the work it does: each direction's
/// derivative tabulation, the contraction for every multi-index of the box (rational) or for the
/// one requested (not), and the quotient rule's terms (rational). Tolerant of orders of the wrong
/// number or sign, which it reads as zero: a binding sizes its work before the call validates.
template <StorageFloat T>
[[nodiscard]] double derivative_ops_per_point(const Bspline<T>& b, std::span<const int> orders) {
  const Layout L = make_layout(b);
  double nbox = 1.0;
  double ops = 0.0;
  std::size_t d = 0;
  for (const auto& s : b.rep()->space.spaces()) {
    const int k = d < orders.size() ? (std::max)(orders[d], 0) : 0;
    ops += derivs_ops_per_point(s.degree(), k);
    nbox *= static_cast<double>(k) + 1.0;
    ++d;
  }
  if (!b.rational()) {
    return ops + contraction_ops(L);
  }
  // Terms of the quotient rule: at most nbox^2 / 2 pairs, each a multiply-add per coordinate.
  return ops + nbox * contraction_ops(L) + nbox * nbox * static_cast<double>(b.rank());
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
  const double ops = derivative_ops_per_point(b, orders) + tabulation_ops(b);
  if (!b.rational()) {
    std::vector<std::size_t> k(L.dim);
    for (std::size_t d = 0; d < L.dim; ++d) {
      k[d] = static_cast<std::size_t>(orders[d]);
    }
    parallel::parallel_for(ps.count, parallel::WorkEstimate{ops}, sizeof(T) * rank,
                           [&](std::size_t begin, std::size_t end) {
                             PointScratch<T, P> s(L);
                             PointCursor cur(ps.lattice, counts, L.dim, begin);
                             for (std::size_t p = begin; p < end; ++p) {
                               prepare_point<T, P>(L, tables, cur, k, s);
                               local_offsets(L, s.first.data(), s.offs.data());
                               contract_local<T>(cp, s.offs.data(), s.weights.data(), L.nloc,
                                                 L.cp_size, out + p * rank);
                               cur.advance();
                             }
                           });
    return true;
  }
  const QuotientPlan<T> plan = make_quotient_plan<T>(orders);
  const std::size_t nbox = plan.nbox;
  parallel::parallel_for(ps.count, parallel::WorkEstimate{ops}, sizeof(T) * rank,
                         [&](std::size_t begin, std::size_t end) {
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
                               contract_local<T>(cp, s.offs.data(), s.weights.data(), L.nloc,
                                                 L.cp_size, H.data() + m * L.cp_size);
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
