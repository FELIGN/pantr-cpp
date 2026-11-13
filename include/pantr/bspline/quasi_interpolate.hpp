/// @file quasi_interpolate.hpp
/// @brief The Lee-Lyche-Mørken quasi-interpolant onto a tensor-product B-spline space.
///
/// Each coefficient is a local point-value functional `lambda_j(f) = sum_k a_{j,k} f(x_{j,k})`:
/// per direction, the `degree + 1` points are equispaced inside the knot interval holding the
/// function's Greville abscissa, and the weights are the function's row of the inverse of the
/// local collocation block there. On one interval the `degree + 1` nonzero B-splines span the
/// polynomials of that degree, so where the chosen interval lies in the function's support the
/// local interpolant recovers its coefficient exactly and the operator reproduces the space
/// (`Q B_j = B_j`). Two cases inherited from the previous release break that: a Greville abscissa
/// on a break of full multiplicity picks the interval to its right, outside the support, and in
/// a periodic space the abscissae are read in sorted order rather than per function; both are
/// pinned as known failures in the tests. Points, local blocks, weights and coefficients are
/// `double`, as before (the blocks are tabulated over the space's own knots, so a `float` space's
/// rounded knots carry into them); a caller rounds the coefficients to the space's type.
#pragma once

#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/linalg/dense.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bspline {

namespace detail {

/// The points and weights of every function of one direction.
///
/// Attributes:
///     points (NdArray<double>): `(num_basis, order)`, each row inside one knot interval.
///     weights (NdArray<double>): `(num_basis, order)`.
struct LlmDirection {
  NdArray<double> points;
  NdArray<double> weights;
};

/// The Python list repr of `pts` (`[0.25, 0.5]`), for the messages below.
[[nodiscard]] inline std::string py_list(std::span<const double> pts) {
  std::string s = "[";
  for (std::size_t i = 0; i < pts.size(); ++i) {
    s += (i == 0 ? "" : ", ") + py_repr(pts[i]);
  }
  return s + "]";
}

/// The `order` equispaced interior points of `[lo, hi]`, `lo + (hi - lo) k / (order + 1)` for
/// `k = 1, ..., order`.
///
/// Interior placement keeps the points off the knots, where a local collocation block can become
/// singular. The product and the sum are separate statements, so that no fused multiply-add changes
/// the points.
///
/// Args:
///     lo (double): The interval's lower end.
///     hi (double): The interval's upper end.
///     out (std::span<double>): `order` entries, written in full.
inline void llm_interval_points(double lo, double hi, std::span<double> out) noexcept {
  const std::size_t order = out.size();
  for (std::size_t k = 0; k < order; ++k) {
    const double frac = static_cast<double>(k + 1) / static_cast<double>(order + 1);
    const double step = (hi - lo) * frac;
    out[k] = lo + step;
  }
}

/// The weight row of function `j` for the points `pts` of one knot interval: row `j - first` of
/// the inverse of the local collocation block, `first` the first function non-zero there, so that
/// `weights . f(pts)` is the coefficient of `B_j` of the local interpolant.
///
/// Args:
///     space (const BsplineSpace1D<T>&): The 1D space.
///     j (std::size_t): The target function, in `[0, num_basis)`; it must be one of the block's
///         functions.
///     pts (std::span<const double>): `degree + 1` distinct points inside one knot interval.
///     row (std::span<double>): `degree + 1` entries, written in full.
///
/// Raises:
///     value_error: The local collocation block is singular (an exactly zero or NaN pivot) or its
///         inverse is not finite.
///     index_error: `j` is not among the block's functions (a Python index outside
///         `[-order, order)`, as before).
template <StorageFloat T>
void llm_weight_row(const BsplineSpace1D<T>& space, std::size_t j, std::span<const double> pts,
                    std::span<double> row) {
  const auto order = static_cast<long long>(pts.size());
  const auto [block, first] = space.template tabulate_basis<double>(pts);
  const std::optional<NdArray<double>> inv = linalg::try_inverse<double>(block);
  if (!inv) {
    throw value_error("Local collocation matrix for basis function " + std::to_string(j) +
                      " is singular.");
  }
  if (!std::all_of(inv->data(), inv->data() + inv->size(),
                   [](double v) { return std::isfinite(v); })) {
    throw value_error(
        "Local collocation matrix inversion for basis function " + std::to_string(j) +
        " produced non-finite values (condition number too large). Points: " + py_list(pts));
  }
  long long local = static_cast<long long>(j) - first.data()[0];
  if (local < -order || local >= order) {
    throw index_error("index " + std::to_string(local) + " is out of bounds for axis 0 with size " +
                      std::to_string(order));
  }
  if (local < 0) {
    local += order;  // a negative index counts from the end, as before
  }
  const double* src = inv->data() + static_cast<std::size_t>(local) * pts.size();
  std::copy(src, src + pts.size(), row.begin());
}

/// Build one direction's Lee-Lyche-Mørken data.
///
/// Raises:
///     value_error: A local collocation block is singular (an exactly zero or NaN pivot) or its
///         inverse is not finite.
///     index_error: The target function is not among the block's functions (a Python index
///         outside `[-order, order)`, as before).
template <StorageFloat T>
[[nodiscard]] LlmDirection llm_direction(const BsplineSpace1D<T>& space) {
  const auto order = static_cast<std::size_t>(space.degree()) + 1;
  const auto n = static_cast<std::size_t>(space.num_basis());
  const std::vector<T> greville = space.greville();
  const std::span<const T> breaks = space.unique_knots_and_multiplicity(true).first;
  const std::size_t n_intervals = breaks.size() - 1;
  LlmDirection out{NdArray<double>({n, order}, for_overwrite),
                   NdArray<double>({n, order}, for_overwrite)};
  std::vector<double> pts(order);
  for (std::size_t j = 0; j < n; ++j) {
    // searchsorted(breaks, g, side="right") - 1, clamped to an interval.
    const auto g = static_cast<double>(greville[j]);
    const auto it = std::upper_bound(breaks.begin(), breaks.end(), g,
                                     [](double v, T b) { return v < static_cast<double>(b); });
    const auto above = static_cast<std::size_t>(it - breaks.begin());
    const std::size_t mu = above == 0 ? 0 : (std::min)(above - 1, n_intervals - 1);
    llm_interval_points(static_cast<double>(breaks[mu]), static_cast<double>(breaks[mu + 1]), pts);
    llm_weight_row<T>(space, j, pts, std::span<double>(out.weights.data() + j * order, order));
    std::copy(pts.begin(), pts.end(), out.points.data() + j * order);
  }
  return out;
}

}  // namespace detail

/// The staged Lee-Lyche-Mørken quasi-interpolant onto a tensor-product space: the points to
/// sample, then the coefficients from the samples (design §3.6). Immutable.
///
/// The points are listed function by function (the functions in C order, the last direction
/// fastest), each function's `prod(degree + 1)` points as the tensor grid of its per-direction
/// points, again in C order.
///
/// Attributes:
///     space (BsplineSpace<T>): The target space (holds).
///     directions (std::vector<detail::LlmDirection>): Per direction, points and weights.
template <StorageFloat T>
class QuasiInterpolation {
 public:
  /// Build the per-direction functionals.
  ///
  /// Raises:
  ///     value_error: See `detail::llm_direction`.
  explicit QuasiInterpolation(BsplineSpace<T> space) : space_(std::move(space)) {
    for (const BsplineSpace1D<T>& s : space_.spaces()) {
      dirs_.push_back(detail::llm_direction<T>(s));
    }
  }

  /// The target space.
  [[nodiscard]] const BsplineSpace<T>& space() const noexcept { return space_; }

  /// The number of points per function, `prod(degree + 1)`.
  [[nodiscard]] std::size_t block() const {
    std::size_t b = 1;
    for (const detail::LlmDirection& d : dirs_) {
      b *= d.points.shape()[1];
    }
    return b;
  }

  /// The number of functions.
  [[nodiscard]] std::size_t num_functions() const {
    std::size_t n = 1;
    for (const detail::LlmDirection& d : dirs_) {
      n *= d.points.shape()[0];
    }
    return n;
  }

  /// Every sample point.
  ///
  /// Returns:
  ///     NdArray<double>: `(num_functions() * block(), dim)`.
  [[nodiscard]] NdArray<double> points() const {
    const std::size_t dim = dirs_.size();
    const std::size_t blk = block();
    const std::size_t nf = num_functions();
    NdArray<double> out({nf * blk, dim}, for_overwrite);
    std::vector<std::size_t> f_idx(dim, 0);
    std::vector<std::size_t> k_idx(dim, 0);
    double* dst = out.data();
    for (std::size_t f = 0; f < nf; ++f) {
      unravel(f, [this](std::size_t d) { return dirs_[d].points.shape()[0]; }, f_idx);
      for (std::size_t k = 0; k < blk; ++k) {
        unravel(k, [this](std::size_t d) { return dirs_[d].points.shape()[1]; }, k_idx);
        for (std::size_t d = 0; d < dim; ++d) {
          const std::size_t order = dirs_[d].points.shape()[1];
          *dst++ = dirs_[d].points.data()[f_idx[d] * order + k_idx[d]];
        }
      }
    }
    return out;
  }

  /// The coefficients from the samples at `points()`.
  ///
  /// Each function's block of values is contracted with its weights one direction at a time, the
  /// first direction first.
  ///
  /// Args:
  ///     values (mdview<const double, 2>): `(num_functions() * block(), rank)`.
  ///
  /// Returns:
  ///     NdArray<double>: `(n_0, ..., n_{d-1}, rank)`.
  ///
  /// Raises:
  ///     value_error: Another number of rows.
  [[nodiscard]] NdArray<double> coefficients(mdview<const double, 2> values) const {
    const std::size_t blk = block();
    const std::size_t nf = num_functions();
    const std::size_t rank = values.extent(1);
    if (values.extent(0) != nf * blk) {
      throw value_error("QuasiInterpolation: expected " + std::to_string(nf * blk) +
                        " rows of values, got " + std::to_string(values.extent(0)) + ".");
    }
    std::vector<std::size_t> shape;
    for (const detail::LlmDirection& d : dirs_) {
      shape.push_back(d.points.shape()[0]);
    }
    shape.push_back(rank);
    NdArray<double> out(std::move(shape), for_overwrite);
    std::vector<std::size_t> f_idx(dirs_.size(), 0);
    std::vector<double> cur(blk * rank);
    std::vector<double> next(blk * rank);
    for (std::size_t f = 0; f < nf; ++f) {
      unravel(f, [this](std::size_t d) { return dirs_[d].points.shape()[0]; }, f_idx);
      std::copy(values.data_handle() + f * blk * rank, values.data_handle() + (f + 1) * blk * rank,
                cur.begin());
      std::size_t rest = blk * rank;
      for (std::size_t d = 0; d < dirs_.size(); ++d) {
        const std::size_t order = dirs_[d].points.shape()[1];
        const double* w = dirs_[d].weights.data() + f_idx[d] * order;
        rest /= order;
        for (std::size_t r = 0; r < rest; ++r) {
          double acc = 0.0;
          for (std::size_t k = 0; k < order; ++k) {
            acc += w[k] * cur[k * rest + r];
          }
          next[r] = acc;
        }
        std::swap(cur, next);
      }
      std::copy(cur.begin(), cur.begin() + static_cast<std::ptrdiff_t>(rank),
                out.data() + f * rank);
    }
    return out;
  }

 private:
  /// Unravel `flat` into C-order indices over extents given per direction.
  template <class Extent>
  void unravel(std::size_t flat, Extent extent, std::vector<std::size_t>& idx) const {
    for (std::size_t d = dirs_.size(); d-- > 0;) {
      const std::size_t e = extent(d);
      idx[d] = flat % e;
      flat /= e;
    }
  }

  BsplineSpace<T> space_;
  std::vector<detail::LlmDirection> dirs_;
};

/// Quasi-interpolate a function onto a tensor-product space: the convenience form taking a C++
/// callable.
///
/// Args:
///     space (const BsplineSpace<T>&): The target space.
///     f (F&&): Called once as `f(points)` on the `(M, dim)` sample points
///         (`mdview<const double, 2>`); returns an `NdArray<double>` of `M` values or `(M, rank)`.
///
/// Returns:
///     Bspline<T>: A non-rational B-spline, the coefficients rounded to `T`.
///
/// Raises:
///     value_error: See `QuasiInterpolation`; `f` returned another number of values.
template <StorageFloat T, class F>
  requires std::invocable<F&, mdview<const double, 2>>
[[nodiscard]] Bspline<T> quasi_interpolate(const BsplineSpace<T>& space, F&& f) {
  const QuasiInterpolation<T> qi(space);
  const NdArray<double> pts = qi.points();
  const NdArray<double> values = f(pts.template view<2>());
  const std::size_t m = pts.shape()[0];
  if ((values.rank() != 1 && values.rank() != 2) || values.shape()[0] != m) {
    throw value_error(
        "quasi_interpolate: the function returned another number of values than the " +
        std::to_string(m) + " points.");
  }
  const std::size_t rank = values.rank() == 1 ? 1 : values.shape()[1];
  const NdArray<double> c = qi.coefficients(mdview<const double, 2>(values.data(), m, rank));
  std::vector<std::size_t> shape(c.shape().begin(), c.shape().end());
  NdArray<T> cp(std::move(shape), for_overwrite);
  std::transform(c.data(), c.data() + c.size(), cp.data(),
                 [](double v) { return static_cast<T>(v); });
  return Bspline<T>(space, std::move(cp), false);
}

}  // namespace pantr::bspline
