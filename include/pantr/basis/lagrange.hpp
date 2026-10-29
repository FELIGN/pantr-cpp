/// @file lagrange.hpp
/// @brief The Lagrange basis on `[0, 1]`, by the barycentric formula.
#pragma once

#include "pantr/basis/detail/tabulate_1d.hpp"
#include "pantr/basis/lagrange_variant.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/tolerance.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>
#include <vector>

namespace pantr::basis {

/// Machine epsilons within which a point is taken to be a node and gets the exact Kronecker delta
/// row, so a point that is a node up to rounding does not carry the tiny off-diagonal values the
/// barycentric quotient leaves there. The value the Python implementation fixed; an admitted
/// heuristic, an absolute distance on the unit interval.
inline constexpr double lagrange_node_snap_eps_factor = 16.0;

namespace detail {

/// Barycentric weights `w_j = 1 / prod_{k != j} (x_j - x_k)` of distinct nodes, up to a common
/// factor (which cancels in the barycentric quotient). Each product is accumulated as a mantissa
/// and a binary exponent (`frexp`), so it neither overflows nor underflows at any degree, and the
/// weights are scaled by a common power of two so the largest has magnitude in `(1, 2]`. This
/// replaces the random permutation of the products SciPy uses for the same purpose, which made the
/// weights depend on a generator's stream.
///
/// Raises:
///     value_error: Two nodes coincide.
inline std::vector<double> barycentric_weights(const std::vector<double>& x) {
  const std::size_t m = x.size();
  std::vector<double> mant(m, 1.0);
  std::vector<long long> expo(m, 0);
  for (std::size_t j = 0; j < m; ++j) {
    double mj = 1.0;
    long long ej = 0;
    for (std::size_t k = 0; k < m; ++k) {
      if (k == j) {
        continue;
      }
      const double d = x[j] - x[k];
      if (d == 0.0) {
        throw value_error("Interpolation points xi must be distinct.");
      }
      int e = 0;
      mj = std::frexp(mj * d, &e);
      ej += e;
    }
    mant[j] = mj;
    expo[j] = ej;
  }
  const long long emin = *std::min_element(expo.begin(), expo.end());
  std::vector<double> w(m);
  for (std::size_t j = 0; j < m; ++j) {
    // 1 / (mant 2^e) = (1 / mant) 2^-e, rescaled by 2^emin; |1 / mant| lies in (1, 2].
    const long long shift = emin - expo[j];
    w[j] = shift < INT_MIN ? 0.0 : std::ldexp(1.0 / mant[j], static_cast<int>(shift));
  }
  return w;
}

/// Operations one point of the Lagrange tabulation costs: per node a snap test, two differences
/// and three quotients, about six.
inline double lagrange_ops_per_point(int degree) {
  return 6.0 * (static_cast<double>(degree) + 1.0);
}

/// Write the Lagrange basis of degree `degree` on the nodes of `variant` at every point into
/// `out`, column `i` for the `i`-th node as `lagrange_points` returns them.
///
/// Computed in double whatever `T` (the nodes are generated in `T`, then widened): the second
/// (true) barycentric formula `L_i(t) = (w_i / (t - x_i)) / sum_k w_k / (t - x_k)`, with a point
/// within `lagrange_node_snap_eps_factor * eps(T)` of a node (distance measured in `T`) given that
/// node's Kronecker row exactly.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `tabulate_lagrange_1d` instead.
template <StorageFloat T>
void tabulate_lagrange_1d_kernel(int degree, LagrangeVariant variant, std::span<const T> pts,
                                 mdview<T, 2> out) {
  T* data = out.data_handle();
  const std::size_t m = static_cast<std::size_t>(degree) + 1;
  if (degree == 0) {
    detail::for_each_point(pts.size(), 1.0, sizeof(T),
                           [&](std::size_t j) { data[j] = static_cast<T>(1); });
    return;
  }
  // lagrange_points returns every family's nodes ascending (a C++ test holds each family to it),
  // so column i is node i and the first node within the snap distance is the nearest one below.
  const NdArray<T> nodes_t = lagrange_points<T>(variant, static_cast<long long>(m));
  const T* nt = nodes_t.data();
  std::vector<double> x(m);
  for (std::size_t k = 0; k < m; ++k) {
    x[k] = static_cast<double>(nt[k]);
  }
  const std::vector<double> w = barycentric_weights(x);
  const T snap = static_cast<T>(lagrange_node_snap_eps_factor *
                                static_cast<double>(tolerance::machine_epsilon<T>()));
  detail::for_each_point(pts.size(), lagrange_ops_per_point(degree), sizeof(T) * m,
                         [&](std::size_t j) {
                           T* row = data + j * m;
                           const T tj = pts[j];
                           for (std::size_t k = 0; k < m; ++k) {
                             using std::abs;
                             if (abs(tj - nt[k]) <= snap) {
                               for (std::size_t i = 0; i < m; ++i) {
                                 row[i] = static_cast<T>(0);
                               }
                               row[k] = static_cast<T>(1);
                               return;
                             }
                           }
                           // No node within the snap distance in T, so no difference below is zero
                           // in double.
                           const auto t = static_cast<double>(tj);
                           double den = 0.0;
                           for (std::size_t k = 0; k < m; ++k) {
                             den += w[k] / (t - x[k]);
                           }
                           for (std::size_t k = 0; k < m; ++k) {
                             row[k] = static_cast<T>((w[k] / (t - x[k])) / den);
                           }
                         });
}

}  // namespace detail

/// Evaluate the Lagrange basis of degree `degree` on the nodes of `variant` at `pts` into `out`.
///
/// `L_i` is the polynomial of degree `degree` equal to 1 at node `i` and 0 at the others, the
/// nodes being `lagrange_points<T>(variant, degree + 1)`. See the kernel for how it is computed.
///
/// Args:
///     degree (int): Degree, `>= 0`.
///     variant (LagrangeVariant): Node family.
///     pts (std::span<const T>): The points.
///     out (mdview<T, 2>): Output, extents `(pts.size(), degree + 1)`.
///
/// Raises:
///     value_error: `degree < 0`, `out` has other extents, or two nodes coincide in `T`.
template <StorageFloat T>
void tabulate_lagrange_1d(int degree, LagrangeVariant variant, std::span<const T> pts,
                          mdview<T, 2> out) {
  detail::check_degree(degree);
  detail::check_out<T, 2>(out, {pts.size(), static_cast<std::size_t>(degree) + 1});
  detail::tabulate_lagrange_1d_kernel<T>(degree, variant, pts, out);
}

/// Evaluate the Lagrange basis of degree `degree` on the nodes of `variant` at `pts`.
///
/// Returns:
///     NdArray<T>: Extents `(pts.size(), degree + 1)`; see the `out` overload.
///
/// Raises:
///     value_error: As the `out` overload, extents aside.
template <StorageFloat T>
[[nodiscard]] NdArray<T> tabulate_lagrange_1d(int degree, LagrangeVariant variant,
                                              std::span<const T> pts) {
  detail::check_degree(degree);
  NdArray<T> out({pts.size(), static_cast<std::size_t>(degree) + 1}, for_overwrite);
  detail::tabulate_lagrange_1d_kernel<T>(degree, variant, pts, out.template view<2>());
  return out;
}

}  // namespace pantr::basis
