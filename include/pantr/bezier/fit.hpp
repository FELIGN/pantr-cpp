/// @file fit.hpp
/// @brief Bézier fitting: Bernstein coefficients from samples, by a truncated-SVD pseudo-inverse.
///
/// The Bernstein Vandermonde matrix grows ill-conditioned with the degree, so it is inverted
/// through its SVD with the singular values below `tol * sigma_max` dropped (Golub and Van Loan,
/// *Matrix Computations*: regularised least squares). On a tensor-product grid the system
/// separates and one pseudo-inverse per direction is applied along its axis; at scattered points
/// the full tensor-product Vandermonde is built once and its truncated pseudo-inverse applied to
/// the values through a QR, never formed (`truncated_pinv_apply`). As in the previous release
/// (NumPy's `linalg.svd`), the decompositions run in `double`; on a grid the factors are rounded
/// to the storage type `T` and the truncation, the pseudo-inverse and its application are in `T`,
/// while at scattered points the truncation compares in `T` and the solve stays in `double` until
/// its result is rounded to `T`. Eigen's `BDCSVD` (divide and conquer, Jacobi below 16 columns)
/// stands for LAPACK's `gesdd` (design §6).
#pragma once

#include "pantr/basis/bernstein.hpp"
#include "pantr/basis/detail/tabulate_1d.hpp"
#include "pantr/bezier/bezier.hpp"
#include "pantr/bezier/detail/along_axis.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/linalg/dense.hpp"

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SVD>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pantr::bezier {

/// Default truncation tolerance of the fitting SVD, in units of the dtype's epsilon: a singular
/// value below `100 * eps * sigma_max` is treated as zero. Kept from the previous release (an
/// admitted factor: two digits above the backward error of the SVD itself, `O(eps * sigma_max)`).
///
/// It is the default of the fits' public `tol` (Bézier and B-spline), not the numerical rank of a
/// least-squares solve in general: `linalg::lstsq_rcond`, `eps * max(m, n)` with no factor, is the
/// cutoff of NumPy's `lstsq` that the internal changes of representation reproduce. Unifying the
/// two would move every fit whose matrix has a singular value between them, a change of a public
/// default this release does not make. The comparison at a tie also stays as each previous-release
/// fit wrote it: a Bézier fit keeps `s >= tol * s_max` (`truncated_pinv`), a
/// B-spline fit `s > tol * s_max` (`bspline::detail::truncated_svd_solve`); they differ only for a
/// singular value exactly on the cutoff.
inline constexpr double svd_tol_factor = 100.0;

namespace detail {

/// The unsigned integer of the width of `T`, for a bitwise comparison of nodes.
template <class T>
using UInt = std::conditional_t<sizeof(T) == 8, std::uint64_t, std::uint32_t>;

/// The truncated pseudo-inverse `V^+ = V_r Sigma_r^{-1} U_r^T` of a dense matrix, in `T`.
///
/// Args:
///     v (const linalg::RowMatrix<T>&): The matrix, `(m, n)`.
///     tol (double): Relative truncation tolerance; a singular value `s` is kept when
///         `s >= tol * s_max` (compared in `T`).
///
/// Returns:
///     NdArray<T>: `(n, m)`.
///
/// Raises:
///     value_error: `v` has a NaN or infinite entry, or the SVD fails ("SVD did not converge").
template <StorageFloat T>
[[nodiscard]] NdArray<T> truncated_pinv(const linalg::RowMatrix<T>& v, double tol) {
  using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using MatD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  // LAPACK's gesdd reports a NaN or infinite entry as a failure to converge (NumPy's LinAlgError,
  // a ValueError); Eigen would return NaN singular values, which the truncation below would then
  // zero, a silent wrong answer. The same refusal, as a value_error with NumPy's message.
  if (!v.allFinite()) {
    throw value_error("SVD did not converge");
  }
  // In double, as NumPy decomposes every real matrix, then the factors rounded to T.
  const Eigen::BDCSVD<MatD, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(
      v.template cast<double>());
  if (svd.info() != Eigen::Success) {
    throw value_error("SVD did not converge");
  }
  const Eigen::Matrix<T, Eigen::Dynamic, 1> sigma = svd.singularValues().template cast<T>();
  const Mat u = svd.matrixU().template cast<T>();
  const Mat vv = svd.matrixV().template cast<T>();
  const auto k = sigma.size();
  Eigen::Matrix<T, Eigen::Dynamic, 1> inv(k);
  const T cut = k > 0 ? static_cast<T>(tol) * sigma(0) : T(0);
  for (Eigen::Index i = 0; i < k; ++i) {
    inv(i) = sigma(i) >= cut ? T(1) / sigma(i) : T(0);
  }
  const Mat pinv = vv * inv.asDiagonal() * u.transpose();
  NdArray<T> out({static_cast<std::size_t>(pinv.rows()), static_cast<std::size_t>(pinv.cols())},
                 for_overwrite);
  Eigen::Map<linalg::RowMatrix<T>>(out.data(), pinv.rows(), pinv.cols()) = pinv;
  return out;
}

/// `truncated_pinv(v, tol) * b` without forming the pseudo-inverse, for a tall `v` (`m >= n`).
///
/// A Householder QR of `v` first, then the SVD of its `(n, n)` triangular factor, whose singular
/// values are those of `v` up to the QR's backward error, `O(eps * |v|)` (Householder QR is
/// backward stable, Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., ch. 19):
/// the same order as the error of the SVD of `v` itself. The truncation is `truncated_pinv`'s,
/// `s >= tol * s_max` compared in `T`. Neither the `(m, n)` factor `U` nor the `(n, m)`
/// pseudo-inverse is formed, so the cost is the QR's, `O(m n^2)` with a small constant.
///
/// Args:
///     v (const linalg::RowMatrix<T>&): The matrix, `(m, n)` with `m >= n`.
///     b (mdview<const T, 2>): The right-hand sides, `(m, k)`.
///     tol (double): Relative truncation tolerance, as in `truncated_pinv`.
///
/// Returns:
///     linalg::RowMatrix<T>: `(n, k)`.
///
/// Raises:
///     value_error: `v` has a NaN or infinite entry, or the SVD fails ("SVD did not converge").
template <StorageFloat T>
[[nodiscard]] linalg::RowMatrix<T> truncated_pinv_apply(const linalg::RowMatrix<T>& v,
                                                        mdview<const T, 2> b, double tol) {
  using MatD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  if (!v.allFinite()) {
    throw value_error("SVD did not converge");  // as truncated_pinv, for the same reason
  }
  const Eigen::Index n = v.cols();
  const auto m = static_cast<Eigen::Index>(b.extent(0));
  const auto k = static_cast<Eigen::Index>(b.extent(1));
  const Eigen::HouseholderQR<MatD> qr(v.template cast<double>());
  const MatD r = qr.matrixQR().topRows(n).template triangularView<Eigen::Upper>();
  MatD qtb = Eigen::Map<const linalg::RowMatrix<T>>(b.data_handle(), m, k).template cast<double>();
  qtb.applyOnTheLeft(qr.householderQ().adjoint());
  const Eigen::BDCSVD<MatD, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(r);
  if (svd.info() != Eigen::Success) {
    throw value_error("SVD did not converge");
  }
  const Eigen::Matrix<T, Eigen::Dynamic, 1> sigma = svd.singularValues().template cast<T>();
  const auto ns = sigma.size();
  Eigen::VectorXd inv(ns);
  const T cut = ns > 0 ? static_cast<T>(tol) * sigma(0) : T(0);
  for (Eigen::Index i = 0; i < ns; ++i) {
    inv(i) = sigma(i) >= cut ? 1.0 / static_cast<double>(sigma(i)) : 0.0;
  }
  const MatD x = svd.matrixV() * (inv.asDiagonal() * (svd.matrixU().transpose() * qtb.topRows(n)));
  return x.template cast<T>();
}

/// Refuse a NaN or infinite truncation tolerance: NaN keeps no singular value and infinity none
/// either, so the fit would report all-zero coefficients as its answer.
///
/// Args:
///     tol (double): The tolerance.
///
/// Raises:
///     value_error: `tol` is NaN or infinite.
inline void check_finite_tol(double tol) {
  if (!std::isfinite(tol)) {
    throw value_error("tol must be finite, got " + py_repr(tol));
  }
}

/// The truncation tolerance a fit uses: `tol` when given, else `svd_tol_factor * eps`.
///
/// Args:
///     tol (std::optional<double>): The caller's tolerance, if any.
///     eps (double): The machine epsilon of the fit's type.
///
/// Returns:
///     double: The tolerance.
///
/// Raises:
///     value_error: `tol` is NaN or infinite.
[[nodiscard]] inline double resolve_svd_tol(std::optional<double> tol, double eps) {
  if (tol) {
    check_finite_tol(*tol);
    return *tol;
  }
  return svd_tol_factor * eps;
}

}  // namespace detail

/// The matrix mapping values at `nodes` to Bernstein coefficients of degree `degree` by truncated
/// SVD: an interpolation matrix when `degree == len(nodes) - 1`, a least-squares fit below.
///
/// Args:
///     nodes (std::span<const T>): Nodes on `[0, 1]`, at least one.
///     degree (long long): `0 <= degree < len(nodes)`.
///     tol (std::optional<double>): Truncation tolerance; `100 * eps(T)` when absent.
///
/// Returns:
///     NdArray<T>: `(degree + 1, len(nodes))`.
///
/// Raises:
///     value_error: `tol` is NaN or infinite, no node, or `degree` out of range.
template <StorageFloat T>
[[nodiscard]] NdArray<T> bernstein_pinv(std::span<const T> nodes, long long degree,
                                        std::optional<double> tol) {
  const double resolved =
      detail::resolve_svd_tol(tol, static_cast<double>(std::numeric_limits<T>::epsilon()));
  if (nodes.empty() || degree < 0 || static_cast<std::size_t>(degree) >= nodes.size()) {
    throw value_error("bernstein_pinv needs 0 <= degree < len(nodes), got degree " +
                      std::to_string(degree) + " for " + std::to_string(nodes.size()) + " nodes.");
  }
  if (nodes.size() == 1) {
    NdArray<T> one({1, 1}, for_overwrite);
    one.data()[0] = T(1);
    return one;
  }
  const NdArray<T> v = basis::tabulate_bernstein_1d<T>(static_cast<int>(degree), nodes);
  const linalg::RowMatrix<T> vm = Eigen::Map<const linalg::RowMatrix<T>>(
      v.data(), static_cast<Eigen::Index>(v.shape()[0]), static_cast<Eigen::Index>(v.shape()[1]));
  return detail::truncated_pinv<T>(vm, resolved);
}

/// Fit a Bézier to values on a tensor-product grid of nodes.
///
/// Args:
///     values (std::span<const T>): Row-major over `(n_0, ..., n_{d-1}, rank)`.
///     shape (std::span<const std::size_t>): Those extents, `d + 1 >= 2` of them.
///     nodes (const std::vector<std::span<const T>>&): `d` node arrays, `n_k` nodes each.
///     degrees (std::optional<std::vector<long long>>): Per direction, `< n_k`; `n_k - 1`
///         (interpolation) when absent.
///     tol (std::optional<double>): Truncation tolerance; `100 * eps(T)` when absent.
///
/// Returns:
///     Bezier<T>: A non-rational Bézier of orders `degree + 1`.
///
/// Raises:
///     value_error: `tol` is NaN or infinite, or inconsistent extents, node counts or degrees.
template <StorageFloat T>
[[nodiscard]] Bezier<T> fit_tensor(std::span<const T> values, std::span<const std::size_t> shape,
                                   const std::vector<std::span<const T>>& nodes,
                                   const std::optional<std::vector<long long>>& degrees,
                                   std::optional<double> tol) {
  if (tol) {
    detail::check_finite_tol(*tol);
  }
  if (shape.size() < 2 || nodes.size() != shape.size() - 1 ||
      (degrees && degrees->size() != nodes.size())) {
    throw value_error(
        "fit_tensor: values need (*n_pts, rank) extents, one node array and one "
        "degree per direction.");
  }
  if (checked_product(shape) != values.size()) {
    throw value_error("fit_tensor: values do not match their shape.");
  }
  const std::size_t dim = nodes.size();
  std::vector<long long> degs(dim, -1);
  std::vector<std::shared_ptr<const NdArray<T>>> pinvs(dim);
  std::vector<std::size_t> cur(shape.begin(), shape.end());
  NdArray<T> current(std::vector<std::size_t>(cur), for_overwrite);
  std::copy(values.begin(), values.end(), current.data());
  for (std::size_t d = 0; d < dim; ++d) {
    if (nodes[d].size() != shape[d]) {
      throw value_error("fit_tensor: direction " + std::to_string(d) + " has " +
                        std::to_string(nodes[d].size()) + " nodes for " + std::to_string(shape[d]) +
                        " values.");
    }
    const long long deg = degrees ? (*degrees)[d] : static_cast<long long>(shape[d]) - 1;
    // A direction with the same nodes (bit for bit) and degree as an earlier one has the same
    // pseudo-inverse; reusing it saves an SVD, the dominant cost, on the usual square grid.
    std::size_t same = d;
    for (std::size_t e = 0; e < d; ++e) {
      if (degs[e] == deg && std::equal(nodes[e].begin(), nodes[e].end(), nodes[d].begin(),
                                       nodes[d].end(), [](T a, T b) {
                                         return std::bit_cast<detail::UInt<T>>(a) ==
                                                std::bit_cast<detail::UInt<T>>(b);
                                       })) {
        same = e;
        break;
      }
    }
    degs[d] = deg;
    if (same == d) {
      pinvs[d] = std::make_shared<const NdArray<T>>(bernstein_pinv<T>(nodes[d], deg, tol));
    } else {
      pinvs[d] = pinvs[same];
    }
    const NdArray<T>& pinv = *pinvs[d];
    if (cur[d] == 1) {
      continue;  // a single node and degree 0: the value is the coefficient
    }
    const NdArray<T> flat = detail::flatten_along_axis<T>(current.flat(), cur, d);
    const std::size_t m = pinv.shape()[0];
    const std::size_t k = pinv.shape()[1];
    const std::size_t cols = flat.shape()[1];
    using CMat = Eigen::Map<const linalg::RowMatrix<T>>;
    const auto em = static_cast<Eigen::Index>(m);
    const auto ek = static_cast<Eigen::Index>(k);
    const auto ec = static_cast<Eigen::Index>(cols);
    NdArray<T> pm({m, cols}, for_overwrite);
    Eigen::Map<linalg::RowMatrix<T>>(pm.data(), em, ec).noalias() =
        CMat(pinv.data(), em, ek) * CMat(flat.data(), ek, ec);
    current = detail::unflatten_along_axis<T>(pm.template view<2>(), cur, d);
    cur[d] = m;
  }
  return Bezier<T>(std::move(current), false);
}

/// Fit a Bézier to values at scattered points with the full tensor-product Vandermonde.
///
/// Args:
///     values (mdview<const T, 2>): `(n_pts, rank)`.
///     pts (mdview<const T, 2>): `(n_pts, dim)`, in `[0, 1]^dim`.
///     degrees (std::span<const long long>): `dim` degrees, `>= 0`, with `prod(degree + 1) <=
///         n_pts`.
///     tol (std::optional<double>): Truncation tolerance; `100 * eps(T)` when absent.
///
/// Returns:
///     Bezier<T>: A non-rational Bézier.
///
/// Raises:
///     value_error: `tol` is NaN or infinite, inconsistent extents, a negative degree, or fewer
///         points than coefficients.
template <StorageFloat T>
[[nodiscard]] Bezier<T> fit_scattered(mdview<const T, 2> values, mdview<const T, 2> pts,
                                      std::span<const long long> degrees,
                                      std::optional<double> tol) {
  const double resolved =
      detail::resolve_svd_tol(tol, static_cast<double>(std::numeric_limits<T>::epsilon()));
  const std::size_t n_pts = pts.extent(0);
  const std::size_t dim = pts.extent(1);
  if (values.extent(0) != n_pts || degrees.size() != dim || dim == 0 || n_pts == 0) {
    throw value_error(
        "fit_scattered: values (n_pts, rank), points (n_pts, dim) and dim degrees "
        "must agree.");
  }
  std::vector<std::size_t> orders(dim);
  std::size_t n_coeffs = 1;
  for (std::size_t d = 0; d < dim; ++d) {
    basis::detail::check_degree(degrees[d]);  // the tabulation's message, as before
    if (degrees[d] >= static_cast<long long>((std::numeric_limits<int>::max)())) {
      throw capacity_error("fit_scattered: degree " + std::to_string(degrees[d]) +
                           " exceeds the largest supported degree");
    }
    orders[d] = static_cast<std::size_t>(degrees[d]) + 1;
  }
  n_coeffs = checked_product(orders);
  if (n_coeffs > n_pts) {
    throw value_error("Underdetermined: " + std::to_string(n_coeffs) + " coefficients but only " +
                      std::to_string(n_pts) + " sample points.");
  }
  // Row-wise tensor products of the univariate bases, the first direction slowest.
  linalg::RowMatrix<T> v = linalg::RowMatrix<T>::Ones(static_cast<Eigen::Index>(n_pts), 1);
  for (std::size_t d = 0; d < dim; ++d) {
    std::vector<T> col(n_pts);
    for (std::size_t i = 0; i < n_pts; ++i) {
      col[i] = pts(i, d);
    }
    const NdArray<T> b = basis::tabulate_bernstein_1d<T>(static_cast<int>(degrees[d]), col);
    const auto nd = static_cast<Eigen::Index>(orders[d]);
    linalg::RowMatrix<T> next(v.rows(), v.cols() * nd);
    for (Eigen::Index i = 0; i < v.rows(); ++i) {
      for (Eigen::Index a = 0; a < v.cols(); ++a) {
        for (Eigen::Index c = 0; c < nd; ++c) {
          next(i, a * nd + c) = v(i, a) * b.data()[static_cast<std::size_t>(i * nd + c)];
        }
      }
    }
    v = std::move(next);
  }
  const linalg::RowMatrix<T> coeffs = detail::truncated_pinv_apply<T>(v, values, resolved);
  const std::size_t rank = values.extent(1);
  const auto ec = static_cast<Eigen::Index>(n_coeffs);
  const auto er = static_cast<Eigen::Index>(rank);
  std::vector<std::size_t> shape(orders);
  shape.push_back(rank);
  NdArray<T> out(std::move(shape), for_overwrite);
  Eigen::Map<linalg::RowMatrix<T>>(out.data(), ec, er) = coeffs;
  return Bezier<T>(std::move(out), false);
}

}  // namespace pantr::bezier
