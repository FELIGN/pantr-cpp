/// @file space.hpp
/// @brief `BsplineSpace`: the tensor product of 1D B-spline spaces, with its basis at points.
#pragma once

#include "pantr/basis/detail/tabulate_1d.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
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

/// Write the tensor product of one point's 1D rows into `row` (C order over the directions), each
/// product formed left to right in `P`, `((B_0 * B_1) * B_2) ...`, then stored in `T`.
///
/// Args:
///     rows (const std::vector<const P*>&): Per direction, the point's `orders[d]` values.
///     orders (std::span<const std::size_t>): Values per direction.
///     scratch (P*): Room for the product of all orders.
///     row (T*): Output, the product of all orders.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat P, StorageFloat T>
void tensor_row(const std::vector<const P*>& rows, std::span<const std::size_t> orders, P* scratch,
                T* row) noexcept {
  std::size_t size = orders[0];
  std::copy(rows[0], rows[0] + size, scratch);
  for (std::size_t d = 1; d < orders.size(); ++d) {
    const std::size_t o = orders[d];
    const P* b = rows[d];
    // Expanded in place from the back: entry a moves to a * o .. a * o + o - 1, at or past a.
    for (std::size_t a = size; a-- > 0;) {
      const P v = scratch[a];
      for (std::size_t j = o; j-- > 0;) {
        scratch[a * o + j] = v * b[j];
      }
    }
    size *= o;
  }
  for (std::size_t i = 0; i < size; ++i) {
    row[i] = static_cast<T>(scratch[i]);
  }
}

}  // namespace detail

/// A tensor-product B-spline space: one `BsplineSpace1D` per direction, all of one storage type.
///
/// An immutable value: it holds its directions (themselves immutable, shared values).
///
/// Attributes:
///     spaces_ (std::vector<BsplineSpace1D<T>>): One space per direction, at least one.
template <StorageFloat T>
class BsplineSpace {
 public:
  /// Build the tensor product of `spaces`.
  ///
  /// Raises:
  ///     value_error: `spaces` is empty.
  ///     capacity_error: The number of basis functions, intervals or local functions overflows.
  explicit BsplineSpace(std::vector<BsplineSpace1D<T>> spaces) : spaces_(std::move(spaces)) {
    if (spaces_.empty()) {
      throw value_error("a B-spline space needs at least one direction");
    }
    std::vector<std::size_t> nb;
    std::vector<std::size_t> ni;
    std::vector<std::size_t> no;
    for (const auto& s : spaces_) {
      nb.push_back(static_cast<std::size_t>(s.num_basis()));
      ni.push_back(static_cast<std::size_t>(s.num_intervals()));
      no.push_back(static_cast<std::size_t>(s.degree()) + 1);
    }
    static_cast<void>(checked_product(nb));
    static_cast<void>(checked_product(ni));
    static_cast<void>(checked_product(no));
  }

  /// The number of directions.
  [[nodiscard]] std::size_t dim() const noexcept { return spaces_.size(); }
  /// The directions.
  [[nodiscard]] const std::vector<BsplineSpace1D<T>>& spaces() const noexcept { return spaces_; }
  /// The largest of the directions' tolerances.
  [[nodiscard]] double tolerance() const noexcept {
    double t = spaces_[0].tolerance();
    for (const auto& s : spaces_) {
      t = (std::max)(t, s.tolerance());
    }
    return t;
  }
  /// Whether every direction is one Bézier piece.
  [[nodiscard]] bool has_bezier_like_knots() const noexcept {
    return std::all_of(spaces_.begin(), spaces_.end(),
                       [](const BsplineSpace1D<T>& s) { return s.has_bezier_like_knots(); });
  }
  /// The non-zero functions per point, `prod(degree + 1)`.
  [[nodiscard]] std::size_t num_local_functions() const noexcept {
    std::size_t n = 1;
    for (const auto& s : spaces_) {
      n *= static_cast<std::size_t>(s.degree()) + 1;
    }
    return n;
  }

  /// The first direction in which a row of `pts` lies outside the domain, or -1.
  ///
  /// Raises:
  ///     value_error: `pts` has another column count.
  template <StorageFloat P>
  [[nodiscard]] std::ptrdiff_t first_direction_outside(mdview<const P, 2> pts) const {
    check_columns(pts.extent(1));
    for (std::size_t d = 0; d < dim(); ++d) {
      for (std::size_t i = 0; i < pts.extent(0); ++i) {
        if (!spaces_[d].template contains<P>(pts(i, d))) {
          return static_cast<std::ptrdiff_t>(d);
        }
      }
    }
    return -1;
  }

  /// The first direction of a lattice with a coordinate outside the domain, or -1.
  ///
  /// Raises:
  ///     value_error: The lattice has another dimension.
  template <StorageFloat L>
  [[nodiscard]] std::ptrdiff_t first_direction_outside(
      const quad::PointsLattice<L>& lattice) const {
    check_columns(lattice.dim());
    for (std::size_t d = 0; d < dim(); ++d) {
      if (!spaces_[d].template all_in_domain<L>(lattice.pts(d))) {
        return static_cast<std::ptrdiff_t>(d);
      }
    }
    return -1;
  }

  /// Evaluate the non-zero tensor-product functions at points given as rows of `pts`.
  ///
  /// Row `i` of `out` is the tensor product, in C order over the directions, of the 1D values at
  /// `pts(i, d)`; row `i` of `first` holds each direction's first function. The 1D values are in
  /// `P` and so are the products, stored in `T`.
  ///
  /// Args:
  ///     pts (mdview<const P, 2>): Extents `(n, dim())`.
  ///     out (mdview<T, 2>): Extents `(n, num_local_functions())`.
  ///     first (mdview<std::int64_t, 2>): Extents `(n, dim())`.
  ///
  /// Raises:
  ///     value_error: `pts` has another column count; a point outside its direction's domain;
  ///         `out` or `first` of other extents.
  template <StorageFloat P>
  void tabulate_basis(mdview<const P, 2> pts, mdview<T, 2> out,
                      mdview<std::int64_t, 2> first) const {
    check_columns(pts.extent(1));
    const std::size_t n = pts.extent(0);
    basis::detail::check_out<T, 2>(out, {n, num_local_functions()});
    basis::detail::check_out<std::int64_t, 2>(first, {n, dim()});
    std::vector<std::vector<P>> coords(dim(), std::vector<P>(n));
    for (std::size_t d = 0; d < dim(); ++d) {
      for (std::size_t i = 0; i < n; ++i) {
        coords[d][i] = pts(i, d);
      }
      for (const P p : coords[d]) {
        if (!spaces_[d].template contains<P>(p)) {
          throw value_error("One or more values in pts[:, " + std::to_string(d) +
                            "] are outside the knot vector domain " + domain_repr());
        }
      }
    }
    std::vector<NdArray<P>> values;
    for (std::size_t d = 0; d < dim(); ++d) {
      auto [v, f] = spaces_[d].template tabulate_basis<P>(coords[d], false);
      for (std::size_t i = 0; i < n; ++i) {
        first(i, d) = f.data()[i];
      }
      values.push_back(std::move(v));
    }
    const std::vector<std::size_t> orders = orders_of();
    const std::size_t nfun = num_local_functions();
    basis::detail::for_each_chunk(n, 2.0 * static_cast<double>(nfun), sizeof(T) * nfun,
                                  [&](std::size_t begin, std::size_t end) {
                                    std::vector<P> scratch(nfun);
                                    std::vector<const P*> rows(dim());
                                    for (std::size_t i = begin; i < end; ++i) {
                                      for (std::size_t d = 0; d < dim(); ++d) {
                                        rows[d] = values[d].data() + i * orders[d];
                                      }
                                      detail::tensor_row<P, T>(rows, orders, scratch.data(),
                                                               &out(i, 0));
                                    }
                                  });
  }

  /// Evaluate at points given as rows (see the `out` overload).
  ///
  /// Returns:
  ///     std::pair<NdArray<T>, NdArray<std::int64_t>>: Values `(n, num_local_functions())` and
  ///     first indices `(n, dim())`.
  template <StorageFloat P>
  [[nodiscard]] std::pair<NdArray<T>, NdArray<std::int64_t>> tabulate_basis(
      mdview<const P, 2> pts) const {
    check_columns(pts.extent(1));
    NdArray<T> values({pts.extent(0), num_local_functions()}, for_overwrite);
    NdArray<std::int64_t> first({pts.extent(0), dim()}, for_overwrite);
    tabulate_basis<P>(pts, values.template view<2>(), first.template view<2>());
    return {std::move(values), std::move(first)};
  }

  /// Evaluate the non-zero tensor-product functions at every point of a lattice.
  ///
  /// The points are taken in C order over the lattice's directions; row `p` of `out` is the
  /// tensor product of the 1D values at that point, computed and stored in the lattice's type `L`.
  ///
  /// Args:
  ///     lattice (const quad::PointsLattice<L>&): `dim()` directions.
  ///     out (mdview<L, 2>): Extents `(lattice.num_points(), num_local_functions())`.
  ///     first (mdview<std::int64_t, 2>): Extents `(lattice.num_points(), dim())`.
  ///
  /// Raises:
  ///     value_error: The lattice has another dimension; a coordinate outside its direction's
  ///         domain; `out` or `first` of other extents.
  template <StorageFloat L>
  void tabulate_basis(const quad::PointsLattice<L>& lattice, mdview<L, 2> out,
                      mdview<std::int64_t, 2> first) const {
    check_columns(lattice.dim());
    for (std::size_t d = 0; d < dim(); ++d) {
      spaces_[d].template check_in_domain<L>(lattice.pts(d),
                                             "pts.pts_per_dir[" + std::to_string(d) + "]");
    }
    const std::size_t n = lattice.num_points();
    basis::detail::check_out<L, 2>(out, {n, num_local_functions()});
    basis::detail::check_out<std::int64_t, 2>(first, {n, dim()});
    std::vector<NdArray<L>> values;
    std::vector<NdArray<std::int64_t>> firsts;
    std::vector<std::size_t> counts;
    for (std::size_t d = 0; d < dim(); ++d) {
      auto [v, f] = spaces_[d].template tabulate_basis<L>(lattice.pts(d), false);
      values.push_back(std::move(v));
      firsts.push_back(std::move(f));
      counts.push_back(lattice.pts(d).size());
    }
    const std::vector<std::size_t> orders = orders_of();
    const std::size_t nfun = num_local_functions();
    basis::detail::for_each_chunk(n, 2.0 * static_cast<double>(nfun), sizeof(L) * nfun,
                                  [&](std::size_t begin, std::size_t end) {
                                    std::vector<L> scratch(nfun);
                                    std::vector<const L*> rows(dim());
                                    std::vector<std::size_t> idx(dim());
                                    // The multi-index of `begin`, C order, then incremented point
                                    // by point.
                                    std::size_t rest = begin;
                                    for (std::size_t d = dim(); d-- > 0;) {
                                      idx[d] = rest % counts[d];
                                      rest /= counts[d];
                                    }
                                    for (std::size_t p = begin; p < end; ++p) {
                                      for (std::size_t d = 0; d < dim(); ++d) {
                                        rows[d] = values[d].data() + idx[d] * orders[d];
                                        first(p, d) = firsts[d].data()[idx[d]];
                                      }
                                      detail::tensor_row<L, L>(rows, orders, scratch.data(),
                                                               &out(p, 0));
                                      for (std::size_t d = dim(); d-- > 0;) {
                                        if (++idx[d] < counts[d]) {
                                          break;
                                        }
                                        idx[d] = 0;
                                      }
                                    }
                                  });
  }

 private:
  void check_columns(std::size_t columns) const {
    if (columns != dim()) {
      throw value_error("pts must have " + std::to_string(dim()) + " columns");
    }
  }

  [[nodiscard]] std::vector<std::size_t> orders_of() const {
    std::vector<std::size_t> orders;
    for (const auto& s : spaces_) {
      orders.push_back(static_cast<std::size_t>(s.degree()) + 1);
    }
    return orders;
  }

  /// The domain spelled as nested lists of Python floats, `[[a0, b0], [a1, b1]]`.
  [[nodiscard]] std::string domain_repr() const {
    std::string s = "[";
    for (std::size_t d = 0; d < dim(); ++d) {
      const auto [a, b] = spaces_[d].domain();
      s += (d > 0 ? ", [" : "[") + py_repr(static_cast<double>(a)) + ", " +
           py_repr(static_cast<double>(b)) + "]";
    }
    return s + "]";
  }

  std::vector<BsplineSpace1D<T>> spaces_;
};

}  // namespace pantr::bspline
