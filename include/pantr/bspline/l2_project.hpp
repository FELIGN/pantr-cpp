/// @file l2_project.hpp
/// @brief L2 projection onto a tensor-product B-spline space, by per-element quadrature and one
/// mass-matrix solve per direction.
///
/// The L2 system of a tensor-product space is a Kronecker product of one mass matrix per
/// direction, `M_k = B_k^T W_k B_k` with `B_k` the collocation matrix at the global quadrature
/// nodes and `W_k` their weights. The load of a function sampled on the quadrature lattice is
/// contracted one direction at a time. Optional boundary interpolation replaces a mass row by the
/// collocation row at that domain end; the matching load entries are then the load integrals, over
/// the other directions, of the function's trace on that boundary (point values at corners). The
/// quadrature lattice and the boundary traces are what a caller must evaluate (`nodes`,
/// `trace_nodes`); `project` consumes the values (design §3.6).
#pragma once

#include "pantr/bezier/detail/along_axis.hpp"
#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/fit.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/quad/points_lattice.hpp"
#include "pantr/quad/rules_1d.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bspline {

/// The per-element quadrature rule of an L2 projection.
enum class L2Quadrature : std::uint8_t {
  gauss_legendre,  ///< Gauss-Legendre, `n >= 1` points per element.
  gauss_lobatto,   ///< Gauss-Lobatto-Legendre, `n >= 2` points per element.
};

/// One boundary trace of an L2 projection with boundary interpolation.
///
/// Attributes:
///     side (std::vector<int>): Per direction, `-1` for a free (integrated) direction, `0` for
///         the domain start and `1` for its end.
struct BoundaryTrace {
  std::vector<int> side;
};

/// Contract a tensor along one axis with the weighted basis of a 1D space:
/// `out[..., i, ...] = sum_q w_q N_i(x_q) values[..., q, ...]`.
///
/// Args:
///     space (const BsplineSpace1D<T>&): The space of that direction.
///     values (std::span<const C>): Row-major over `shape`.
///     shape (std::span<const std::size_t>): Extents, `shape[axis] == len(nodes)`.
///     nodes (std::span<const T>): Quadrature nodes, in the domain.
///     weights (std::span<const T>): Their weights.
///     axis (std::size_t): The contracted axis.
///
/// Returns:
///     NdArray<C>: `shape` with `[axis]` replaced by `num_basis`.
///
/// Raises:
///     value_error: Inconsistent extents, or a node outside the domain.
template <StorageFloat T, StorageFloat C>
[[nodiscard]] NdArray<C> assemble_load_1d(const BsplineSpace1D<T>& space, std::span<const C> values,
                                          std::span<const std::size_t> shape,
                                          std::span<const T> nodes, std::span<const T> weights,
                                          std::size_t axis) {
  if (axis >= shape.size() || shape[axis] != nodes.size() || weights.size() != nodes.size() ||
      checked_product(shape) != values.size()) {
    throw value_error("assemble_load_1d: the values, nodes and weights do not match.");
  }
  const auto [basis, first] = space.template tabulate_basis<T>(nodes);
  const auto n = static_cast<std::size_t>(space.num_basis());
  const auto order = static_cast<std::size_t>(space.degree()) + 1;
  const bezier::detail::AxisSplit split = bezier::detail::split_around(shape, axis);
  const std::size_t nq = nodes.size();
  std::vector<std::size_t> out_shape(shape.begin(), shape.end());
  out_shape[axis] = n;
  NdArray<C> out(std::move(out_shape));
  for (std::size_t q = 0; q < nq; ++q) {
    const auto f = static_cast<std::size_t>(first.data()[q]);
    for (std::size_t a = 0; a < order; ++a) {
      // The weighted basis value in T, as the previous implementation formed it.
      const T wb = weights[q] * basis.data()[q * order + a];
      const auto c = static_cast<C>(wb);
      const std::size_t i = space.periodic() ? (f + a) % n : f + a;
      for (std::size_t o = 0; o < split.outer; ++o) {
        const C* src = values.data() + (o * nq + q) * split.inner;
        C* dst = out.data() + (o * n + i) * split.inner;
        for (std::size_t r = 0; r < split.inner; ++r) {
          dst[r] += c * src[r];
        }
      }
    }
  }
  return out;
}

/// The staged L2 projection onto a tensor-product space: the mass matrices, the quadrature
/// lattice and the boundary traces, built before any value is known. Immutable.
///
/// Attributes:
///     space (BsplineSpace<T>): The target space (holds).
///     nodes (std::vector<NdArray<T>>): Per direction, the global quadrature nodes.
///     weights (std::vector<NdArray<T>>): Their weights.
///     mass (std::vector<NdArray<T>>): Per direction, the mass matrix with its boundary rows.
///     traces (std::vector<BoundaryTrace>): The boundary traces, in the order their loads apply.
template <StorageFloat T>
class L2Projection {
 public:
  /// Build the projection.
  ///
  /// Args:
  ///     space (BsplineSpace<T>): The target space.
  ///     n_quad (std::vector<long long>): Quadrature points per element, per direction.
  ///     rule (L2Quadrature): The rule.
  ///     boundary_interpolation (std::vector<std::pair<bool, bool>>): Per direction, whether
  ///         the start and the end are interpolated; ignored in a periodic direction.
  ///
  /// Raises:
  ///     value_error: Not one count and one flag pair per direction, or a count below the rule's
  ///         minimum ("n_pts must be at least ...").
  L2Projection(BsplineSpace<T> space, std::vector<long long> n_quad, L2Quadrature rule,
               std::vector<std::pair<bool, bool>> boundary_interpolation)
      : space_(std::move(space)) {
    const std::size_t dim = space_.dim();
    if (n_quad.size() != dim) {
      throw value_error("n_quad has length " + std::to_string(n_quad.size()) + ", expected " +
                        std::to_string(dim));
    }
    if (boundary_interpolation.size() != dim) {
      throw value_error("boundary_interpolation has length " +
                        std::to_string(boundary_interpolation.size()) + ", expected " +
                        std::to_string(dim));
    }
    for (std::size_t d = 0; d < dim; ++d) {
      build_direction(space_.spaces()[d], n_quad[d], rule, boundary_interpolation[d]);
    }
    build_traces(boundary_interpolation);
  }

  /// The target space.
  [[nodiscard]] const BsplineSpace<T>& space() const noexcept { return space_; }
  /// Per direction, the global quadrature nodes (numeric view).
  [[nodiscard]] const std::vector<NdArray<T>>& nodes() const noexcept { return nodes_; }
  /// Per direction, the quadrature weights (numeric view).
  [[nodiscard]] const std::vector<NdArray<T>>& weights() const noexcept { return weights_; }
  /// Per direction, the mass matrix with its boundary rows (numeric view).
  [[nodiscard]] const std::vector<NdArray<T>>& mass() const noexcept { return mass_; }
  /// The boundary traces, in application order: by number of fixed directions, then the fixed
  /// directions lexicographically, then their sides (start before end).
  [[nodiscard]] const std::vector<BoundaryTrace>& traces() const noexcept { return traces_; }

  /// The quadrature lattice's extents.
  [[nodiscard]] std::vector<std::size_t> grid_shape() const {
    std::vector<std::size_t> g(nodes_.size());
    for (std::size_t d = 0; d < g.size(); ++d) {
      g[d] = nodes_[d].size();
    }
    return g;
  }

  /// The lattice a trace samples: the boundary coordinate in a fixed direction, the quadrature
  /// nodes elsewhere.
  ///
  /// Raises:
  ///     index_error: `trace` out of range.
  [[nodiscard]] std::vector<std::vector<T>> trace_nodes(std::size_t trace) const {
    const BoundaryTrace& t = trace_at(trace);
    std::vector<std::vector<T>> out(nodes_.size());
    for (std::size_t d = 0; d < out.size(); ++d) {
      if (t.side[d] < 0) {
        out[d].assign(nodes_[d].data(), nodes_[d].data() + nodes_[d].size());
      } else {
        const auto [a, b] = space_.spaces()[d].domain();
        out[d].assign(1, t.side[d] == 0 ? a : b);
      }
    }
    return out;
  }

  /// The load of one component sampled on the quadrature lattice.
  ///
  /// Args:
  ///     values (std::span<const C>): Row-major over `grid_shape()`.
  ///
  /// Returns:
  ///     NdArray<C>: Extents `(n_0, ..., n_{d-1})`.
  template <StorageFloat C>
  [[nodiscard]] NdArray<C> load(std::span<const C> values) const {
    std::vector<std::size_t> shape = grid_shape();
    NdArray<C> cur(std::vector<std::size_t>(shape), for_overwrite);
    if (values.size() != cur.size()) {
      throw value_error("L2Projection: expected " + std::to_string(cur.size()) +
                        " values on the quadrature lattice, got " + std::to_string(values.size()) +
                        ".");
    }
    std::copy(values.begin(), values.end(), cur.data());
    for (std::size_t d = 0; d < shape.size(); ++d) {
      cur = assemble_load_1d<T, C>(space_.spaces()[d], cur.flat(), shape, nodes_[d].flat(),
                                   weights_[d].flat(), d);
      shape[d] = static_cast<std::size_t>(space_.spaces()[d].num_basis());
    }
    return cur;
  }

  /// The load of one component's trace: its values on `trace_nodes(trace)` contracted over the
  /// free directions.
  ///
  /// Returns:
  ///     NdArray<C>: The load over the full lattice of basis indices with extent 1 in every fixed
  ///     direction.
  template <StorageFloat C>
  [[nodiscard]] NdArray<C> trace_load(std::size_t trace, std::span<const C> values) const {
    const BoundaryTrace& t = trace_at(trace);
    std::vector<std::size_t> shape(nodes_.size());
    for (std::size_t d = 0; d < shape.size(); ++d) {
      shape[d] = t.side[d] < 0 ? nodes_[d].size() : 1;
    }
    NdArray<C> cur(std::vector<std::size_t>(shape), for_overwrite);
    if (values.size() != cur.size()) {
      throw value_error("L2Projection: expected " + std::to_string(cur.size()) +
                        " values on trace " + std::to_string(trace) + ", got " +
                        std::to_string(values.size()) + ".");
    }
    std::copy(values.begin(), values.end(), cur.data());
    for (std::size_t d = 0; d < shape.size(); ++d) {
      if (t.side[d] < 0) {
        cur = assemble_load_1d<T, C>(space_.spaces()[d], cur.flat(), shape, nodes_[d].flat(),
                                     weights_[d].flat(), d);
        shape[d] = static_cast<std::size_t>(space_.spaces()[d].num_basis());
      }
    }
    return cur;
  }

  /// Project one component: its load, each trace's load written over the entries of its boundary
  /// (later traces over earlier ones), and the Kronecker mass solve.
  ///
  /// Args:
  ///     values (std::span<const C>): The component on the quadrature lattice.
  ///     trace_values (const std::vector<std::span<const C>>&): The component on each trace's
  ///         lattice, one per trace.
  ///     tol (double): Truncation tolerance of the SVD path (see `solve_collocation`).
  ///
  /// Returns:
  ///     NdArray<C>: The coefficients, `(n_0, ..., n_{d-1})`.
  ///
  /// Raises:
  ///     value_error: Another number of values or traces, or an SVD failure.
  template <StorageFloat C>
  [[nodiscard]] NdArray<C> project(std::span<const C> values,
                                   const std::vector<std::span<const C>>& trace_values,
                                   double tol) const {
    if (trace_values.size() != traces_.size()) {
      throw value_error("L2Projection: expected " + std::to_string(traces_.size()) +
                        " trace value arrays, got " + std::to_string(trace_values.size()) + ".");
    }
    return solve_load<C>(load<C>(values), trace_values, tol);
  }

  /// Finish a projection from its interior load: each trace's load written over the entries of
  /// its boundary (later traces over earlier ones), then the Kronecker mass solve. The second half
  /// of `project`, for a load assembled elsewhere (summed over ranks in `pantr::mpi`).
  ///
  /// Args:
  ///     total (NdArray<C>): The interior load, `(n_0, ..., n_{d-1})` (consumed).
  ///     trace_values (const std::vector<std::span<const C>>&): The component on each trace's
  ///         lattice, one per trace.
  ///     tol (double): Truncation tolerance of the SVD path (see `solve_collocation`).
  ///
  /// Returns:
  ///     NdArray<C>: The coefficients, `(n_0, ..., n_{d-1})`.
  ///
  /// Raises:
  ///     value_error: Another load size, another number of traces or trace values, or an SVD
  ///         failure.
  template <StorageFloat C>
  [[nodiscard]] NdArray<C> solve_load(NdArray<C> total,
                                      const std::vector<std::span<const C>>& trace_values,
                                      double tol) const {
    if (trace_values.size() != traces_.size()) {
      throw value_error("L2Projection: expected " + std::to_string(traces_.size()) +
                        " trace value arrays, got " + std::to_string(trace_values.size()) + ".");
    }
    const std::size_t dim = nodes_.size();
    std::vector<std::size_t> nb(dim);
    for (std::size_t d = 0; d < dim; ++d) {
      nb[d] = static_cast<std::size_t>(space_.spaces()[d].num_basis());
    }
    if (total.size() != checked_product(nb)) {
      throw value_error("L2Projection: expected a load of " + std::to_string(checked_product(nb)) +
                        " entries, got " + std::to_string(total.size()) + ".");
    }
    for (std::size_t t = 0; t < traces_.size(); ++t) {
      const NdArray<C> tl = trace_load<C>(t, trace_values[t]);
      write_trace(traces_[t], tl, nb, total);
    }
    return solve_kronecker<T, C>(mass_, total.flat(), nb, tol);
  }

 private:
  [[nodiscard]] const BoundaryTrace& trace_at(std::size_t trace) const {
    if (trace >= traces_.size()) {
      throw index_error("trace " + std::to_string(trace) + " out of range for " +
                        std::to_string(traces_.size()) + " traces");
    }
    return traces_[trace];
  }

  void build_direction(const BsplineSpace1D<T>& s, long long nq, L2Quadrature rule,
                       std::pair<bool, bool> flags) {
    const quad::Rule1D<T> ref = rule == L2Quadrature::gauss_legendre
                                    ? quad::gauss_legendre_1d<T>(nq)
                                    : quad::gauss_lobatto_legendre_1d<T>(nq);
    const std::span<const T> breaks = s.unique_knots_and_multiplicity(true).first;
    const std::size_t nr = ref.nodes.size();
    std::vector<T> x;
    std::vector<T> w;
    for (std::size_t e = 0; e + 1 < breaks.size(); ++e) {
      const T lo = breaks[e];
      const T h = breaks[e + 1] - lo;
      if (!(h > T(0))) {
        continue;  // a class of equal knots has no length (kept from the previous loop's guard)
      }
      for (std::size_t k = 0; k < nr; ++k) {
        const T hx = h * ref.nodes.data()[k];
        x.push_back(lo + hx);
        w.push_back(h * ref.weights.data()[k]);
      }
    }
    NdArray<T> xn({x.size()}, for_overwrite);
    std::copy(x.begin(), x.end(), xn.data());
    NdArray<T> wn({w.size()}, for_overwrite);
    std::copy(w.begin(), w.end(), wn.data());

    // M = B^T diag(w) B, accumulated over the quadrature points (each row of B has `order`
    // nonzeros, so this costs O(n_q order^2) rather than the dense product's O(n_q n^2)).
    const auto [basis, first] = s.template tabulate_basis<T>(xn.flat());
    const auto n = static_cast<std::size_t>(s.num_basis());
    const auto order = static_cast<std::size_t>(s.degree()) + 1;
    NdArray<T> mass({n, n});
    std::vector<std::size_t> cols(order);
    std::vector<T> wb(order);
    for (std::size_t q = 0; q < x.size(); ++q) {
      const auto f = static_cast<std::size_t>(first.data()[q]);
      const T* bq = basis.data() + q * order;
      for (std::size_t a = 0; a < order; ++a) {
        cols[a] = s.periodic() ? (f + a) % n : f + a;
        wb[a] = w[q] * bq[a];
      }
      for (std::size_t a = 0; a < order; ++a) {
        T* row = mass.data() + cols[a] * n;
        for (std::size_t b = 0; b < order; ++b) {
          row[cols[b]] += bq[a] * wb[b];
        }
      }
    }
    if (!s.periodic()) {
      const auto [a, b] = s.domain();
      if (flags.first) {
        put_collocation_row(s, a, 0, mass);
      }
      if (flags.second) {
        put_collocation_row(s, b, n - 1, mass);
      }
    }
    nodes_.push_back(std::move(xn));
    weights_.push_back(std::move(wn));
    mass_.push_back(std::move(mass));
  }

  static void put_collocation_row(const BsplineSpace1D<T>& s, T x, std::size_t row,
                                  NdArray<T>& mass) {
    const NdArray<T> c = collocation_matrix<T, T>(s, std::span<const T>(&x, 1));
    const std::size_t n = mass.shape()[1];
    std::copy(c.data(), c.data() + n, mass.data() + row * n);
  }

  void build_traces(const std::vector<std::pair<bool, bool>>& flags) {
    const std::size_t dim = space_.dim();
    std::vector<std::size_t> flagged;
    std::vector<std::vector<int>> sides(dim);
    for (std::size_t d = 0; d < dim; ++d) {
      if (space_.spaces()[d].periodic()) {
        continue;
      }
      if (flags[d].first) {
        sides[d].push_back(0);
      }
      if (flags[d].second) {
        sides[d].push_back(1);
      }
      if (!sides[d].empty()) {
        flagged.push_back(d);
      }
    }
    const std::size_t nf = flagged.size();
    // Subsets by increasing size, each size's subsets in lexicographic order (itertools'
    // combinations), each subset's side choices in itertools.product order.
    for (std::size_t size = 1; size <= nf; ++size) {
      std::vector<std::size_t> pick(size);
      for (std::size_t i = 0; i < size; ++i) {
        pick[i] = i;
      }
      while (true) {
        std::vector<std::size_t> choice(size, 0);
        while (true) {
          BoundaryTrace t{std::vector<int>(dim, -1)};
          for (std::size_t i = 0; i < size; ++i) {
            const std::size_t d = flagged[pick[i]];
            t.side[d] = sides[d][choice[i]];
          }
          traces_.push_back(std::move(t));
          std::size_t i = size;
          while (i > 0) {
            --i;
            if (++choice[i] < sides[flagged[pick[i]]].size()) {
              break;
            }
            choice[i] = 0;
            if (i == 0) {
              i = size + 1;  // every choice made
              break;
            }
          }
          if (i == size + 1) {
            break;
          }
        }
        // Next combination of `size` indices out of `nf`.
        std::size_t i = size;
        bool more = false;
        while (i > 0) {
          --i;
          if (pick[i] < nf - size + i) {
            ++pick[i];
            for (std::size_t j = i + 1; j < size; ++j) {
              pick[j] = pick[j - 1] + 1;
            }
            more = true;
            break;
          }
        }
        if (!more) {
          break;
        }
      }
    }
  }

  /// Write a trace's load over the entries of its boundary: index 0 or `n - 1` in each fixed
  /// direction, every index in the others.
  template <StorageFloat C>
  static void write_trace(const BoundaryTrace& t, const NdArray<C>& tl,
                          const std::vector<std::size_t>& nb, NdArray<C>& total) {
    const std::size_t dim = nb.size();
    std::vector<std::size_t> idx(dim, 0);
    const std::size_t count = tl.size();
    for (std::size_t flat = 0; flat < count; ++flat) {
      // Unravel `flat` over the trace load's extents (1 in each fixed direction).
      std::size_t rem = flat;
      std::size_t dst = 0;
      for (std::size_t d = dim; d-- > 0;) {
        const std::size_t ext = t.side[d] < 0 ? nb[d] : 1;
        idx[d] = rem % ext;
        rem /= ext;
      }
      for (std::size_t d = 0; d < dim; ++d) {
        std::size_t i = idx[d];
        if (t.side[d] == 1) {
          i = nb[d] - 1;
        } else if (t.side[d] == 0) {
          i = 0;
        }
        dst = dst * nb[d] + i;
      }
      total.data()[dst] = tl.data()[flat];
    }
  }

  BsplineSpace<T> space_;
  std::vector<NdArray<T>> nodes_;
  std::vector<NdArray<T>> weights_;
  std::vector<NdArray<T>> mass_;
  std::vector<BoundaryTrace> traces_;
};

/// L2-project a function onto a tensor-product space: the convenience form taking a C++ callable.
///
/// Args:
///     space (const BsplineSpace<T>&): The target space.
///     f (F&&): Called as `f(lattice)` on the quadrature lattice, then once per boundary trace
///         (`const quad::PointsLattice<T>&`); returns an `NdArray<T>` of `num_points()` values or
///         `(num_points(), rank)`, rows in the lattice's C order.
///     n_quad (std::optional<std::vector<long long>>): Points per element; `degree + 1` per
///         direction when absent.
///     rule (L2Quadrature): Defaults to Gauss-Legendre.
///     boundary_interpolation (std::vector<std::pair<bool, bool>>): Per direction; none when empty.
///     tol (std::optional<double>): Truncation tolerance; `100 * eps(T)` when absent.
///
/// Returns:
///     Bspline<T>: A non-rational B-spline on `space`.
///
/// Raises:
///     value_error: See `L2Projection`; `f` returned another number of values.
template <StorageFloat T, class F>
  requires std::invocable<F&, const quad::PointsLattice<T>&>
[[nodiscard]] Bspline<T> l2_project(const BsplineSpace<T>& space, F&& f,
                                    std::optional<std::vector<long long>> n_quad = std::nullopt,
                                    L2Quadrature rule = L2Quadrature::gauss_legendre,
                                    std::vector<std::pair<bool, bool>> boundary_interpolation = {},
                                    std::optional<double> tol = std::nullopt) {
  const std::size_t dim = space.dim();
  if (!n_quad) {
    n_quad.emplace();
    for (const BsplineSpace1D<T>& s : space.spaces()) {
      n_quad->push_back(static_cast<long long>(s.degree()) + 1);
    }
  }
  if (boundary_interpolation.empty()) {
    boundary_interpolation.assign(dim, {false, false});
  }
  const L2Projection<T> proj(space, std::move(*n_quad), rule, std::move(boundary_interpolation));
  const auto sample = [&f](std::vector<std::vector<T>> per_dir, std::size_t& rank) {
    const quad::PointsLattice<T> lattice(std::move(per_dir));
    NdArray<T> v = f(lattice);
    const std::size_t points = lattice.num_points();
    if ((v.rank() != 1 && v.rank() != 2) || v.shape()[0] != points ||
        (v.rank() == 2 && v.shape()[1] == 0)) {
      throw value_error("l2_project: the function returned another number of values than the " +
                        std::to_string(points) + " points.");
    }
    rank = v.rank() == 1 ? 1 : v.shape()[1];
    return v;
  };
  std::vector<std::vector<T>> lattice_nodes;
  for (const NdArray<T>& x : proj.nodes()) {
    lattice_nodes.emplace_back(x.data(), x.data() + x.size());
  }
  std::size_t rank = 0;
  const NdArray<T> values = sample(std::move(lattice_nodes), rank);
  std::vector<NdArray<T>> traces;
  std::vector<std::size_t> trace_ranks(proj.traces().size());
  for (std::size_t t = 0; t < proj.traces().size(); ++t) {
    traces.push_back(sample(proj.trace_nodes(t), trace_ranks[t]));
    if (trace_ranks[t] != rank) {
      throw value_error("l2_project: a boundary trace returned another number of components.");
    }
  }
  const double resolved = default_fit_tol<T>(tol);
  std::vector<std::size_t> shape;
  for (const BsplineSpace1D<T>& s : space.spaces()) {
    shape.push_back(static_cast<std::size_t>(s.num_basis()));
  }
  const std::size_t coeffs = checked_product(shape);
  shape.push_back(rank);
  NdArray<T> out(std::move(shape), for_overwrite);
  std::vector<T> comp(values.size() / rank);
  std::vector<std::vector<T>> trace_comp(traces.size());
  for (std::size_t r = 0; r < rank; ++r) {
    for (std::size_t i = 0; i < comp.size(); ++i) {
      comp[i] = values.data()[i * rank + r];
    }
    std::vector<std::span<const T>> spans;
    for (std::size_t t = 0; t < traces.size(); ++t) {
      trace_comp[t].resize(traces[t].size() / rank);
      for (std::size_t i = 0; i < trace_comp[t].size(); ++i) {
        trace_comp[t][i] = traces[t].data()[i * rank + r];
      }
      spans.emplace_back(trace_comp[t]);
    }
    const NdArray<T> x = proj.template project<T>(std::span<const T>(comp), spans, resolved);
    for (std::size_t i = 0; i < coeffs; ++i) {
      out.data()[i * rank + r] = x.data()[i];
    }
  }
  return Bspline<T>(space, std::move(out), false);
}

}  // namespace pantr::bspline
