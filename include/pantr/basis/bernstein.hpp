/// @file bernstein.hpp
/// @brief Bernstein basis polynomials on `[0, 1]`: values and derivatives at points.
#pragma once

#include "pantr/basis/detail/tabulate_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace pantr::basis {

/// Largest `double` degree at which the plain forward recurrence's seed `(1 - u)^n` cannot
/// underflow for any representable `u < 1`: the smallest positive `1 - u` is `2^-53`, and
/// `(2^-53)^n` underflows to zero once `53 n > 1074`, at `n = 21`.
inline constexpr int bernstein_max_unmirrored_degree_double = 20;

/// `float` analogue of `bernstein_max_unmirrored_degree_double`: `24 n > 149` first at `n = 7`.
inline constexpr int bernstein_max_unmirrored_degree_float = 6;

namespace detail {

template <StorageFloat T>
constexpr int max_unmirrored_degree() {
  return std::same_as<T, double> ? bernstein_max_unmirrored_degree_double
                                 : bernstein_max_unmirrored_degree_float;
}

/// `x^n` for `n >= 0` by binary powering: about `2 log2(n)` multiplications, each rounded once,
/// where `std::pow` is a libm call costing several times the rest of a low-degree row. The seed
/// of the recurrences is in `[0, 1]`, so no intermediate overflows; the result is within
/// `2 log2(n)` roundings of `x^n`.
inline double int_power(double x, int n) {
  double result = 1.0;
  double base = x;
  for (unsigned e = static_cast<unsigned>(n); e != 0; e >>= 1U) {
    if ((e & 1U) != 0) {
      result *= base;
    }
    base *= base;
  }
  return result;
}

/// Bernstein values of degree `n >= 1` at `u` by the O(n) ratio recurrence, run from the
/// endpoint that keeps the seed `>= 0.5^n`: forward from 0 for
/// `u <= 0.5`, else forward on `1 - u` (seed `u^n`) and the row reversed, by the symmetry
/// `B_i(u) = B_{n-i}(1 - u)`. Intermediates in double, each value stored in `T` and read back
/// from it (as the Numba kernel this replaces did). The seed `0.5^n` itself underflows `T` from
/// degree 150 (`float`) and 1075 (`double`) on, where a row near `u = 0.5` comes out zero: the
/// envelope of this recurrence, far past any degree pantr uses.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `tabulate_bernstein_1d` instead.
template <StorageFloat T>
void bernstein_row_mirrored(int n, T u, T* row) {
  const auto ud = static_cast<double>(u);
  const double one_minus_u = 1.0 - ud;
  const bool mirror = ud > 0.5;
  const double seed_base = mirror ? ud : one_minus_u;
  const double ratio = mirror ? one_minus_u / ud : ud / one_minus_u;
  row[0] = static_cast<T>(int_power(seed_base, n));
  for (int i = 1; i <= n; ++i) {
    const double const_factor = (n - i + 1.0) / i;
    row[i] = static_cast<T>(static_cast<double>(row[i - 1]) * const_factor * ratio);
  }
  if (mirror) {
    for (int lo = 0, hi = n; lo < hi; ++lo, --hi) {
      std::swap(row[lo], row[hi]);
    }
  }
}

/// Bernstein values of degree `n >= 1` at `u` by the plain forward recurrence from 0, with
/// `u == 1` set directly. Only for `n <= max_unmirrored_degree<T>()`, where the seed provably
/// cannot underflow; the fast path for low degrees.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed), the degree bound included.
///     For general use, call `tabulate_bernstein_1d` instead.
template <StorageFloat T>
void bernstein_row_unmirrored(int n, T u, T* row) {
  if (u == static_cast<T>(1)) {
    for (int i = 0; i < n; ++i) {
      row[i] = static_cast<T>(0);
    }
    row[n] = static_cast<T>(1);
    return;
  }
  const auto ud = static_cast<double>(u);
  const double one_minus_u = 1.0 - ud;
  row[0] = static_cast<T>(int_power(one_minus_u, n));
  const double ratio = ud / one_minus_u;
  for (int i = 1; i <= n; ++i) {
    const double const_factor = (n - i + 1.0) / i;
    row[i] = static_cast<T>(static_cast<double>(row[i - 1]) * const_factor * ratio);
  }
}

/// Operations one point of the Bernstein tabulation costs: about four per value of the ratio
/// recurrence (two multiplications, a quotient and a store).
inline double bernstein_ops_per_point(int degree) {
  return 4.0 * (static_cast<double>(degree) + 1.0);
}

/// Operations one point of the Bernstein derivative tabulation costs: the `(n + 1)^2` table and
/// `(n_deriv + 1)` rows of `O(n + 1)` work over `n + 1` functions.
inline double bernstein_derivatives_ops_per_point(int degree, int n_deriv) {
  const double ord = static_cast<double>(degree) + 1.0;
  return ord * ord * (static_cast<double>(n_deriv) + 2.0);
}

/// Write the Bernstein values of degree `degree` at every point into `out` (`(n, degree + 1)`).
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `tabulate_bernstein_1d` instead.
template <StorageFloat T>
void tabulate_bernstein_1d_kernel(int degree, std::span<const T> pts, mdview<T, 2> out) {
  T* data = out.data_handle();
  const std::size_t stride = static_cast<std::size_t>(degree) + 1;
  const double ops = bernstein_ops_per_point(degree);
  if (degree == 0) {
    detail::for_each_point(pts.size(), 1.0, sizeof(T),
                           [&](std::size_t j) { data[j] = static_cast<T>(1); });
  } else if (degree <= max_unmirrored_degree<T>()) {
    detail::for_each_point(pts.size(), ops, sizeof(T) * stride, [&](std::size_t j) {
      bernstein_row_unmirrored<T>(degree, pts[j], data + j * stride);
    });
  } else {
    detail::for_each_point(pts.size(), ops, sizeof(T) * stride, [&](std::size_t j) {
      bernstein_row_mirrored<T>(degree, pts[j], data + j * stride);
    });
  }
}

/// Bernstein derivatives of orders `0..n_deriv` at one point (Piegl and Tiller's Algorithm A2.3
/// with every knot difference equal to one), in `T`. `out_pt` is `(n_deriv + 1) x (n + 1)`, row
/// major; `ndu` and `a` are scratch of `(n + 1)^2` and `2 (n_deriv + 1)` values. Rows past the
/// degree are zero.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `tabulate_bernstein_derivatives_1d` instead.
template <StorageFloat T>
void bernstein_derivatives_point(int n, T s, int n_deriv, T* out_pt, T* ndu, T* a) {
  const int order = n + 1;
  const auto nd1 = static_cast<std::size_t>(n_deriv) + 1;
  const auto ord = static_cast<std::size_t>(order);
  const T zero = static_cast<T>(0);
  const T one = static_cast<T>(1);
  auto NDU = [&](int r, int c) -> T& {
    return ndu[static_cast<std::size_t>(r) * ord + static_cast<std::size_t>(c)];
  };
  auto A = [&](int r, int c) -> T& {
    return a[static_cast<std::size_t>(r) * nd1 + static_cast<std::size_t>(c)];
  };
  auto OUT = [&](int k, int j) -> T& {
    return out_pt[static_cast<std::size_t>(k) * ord + static_cast<std::size_t>(j)];
  };
  for (std::size_t i = 0; i < nd1 * ord; ++i) {
    out_pt[i] = zero;
  }
  for (std::size_t i = 0; i < ord * ord; ++i) {
    ndu[i] = zero;
  }
  for (std::size_t i = 0; i < 2 * nd1; ++i) {
    a[i] = zero;
  }
  NDU(0, 0) = one;
  for (int j = 1; j < order; ++j) {
    T saved = zero;
    for (int r = 0; r < j; ++r) {
      NDU(j, r) = one;  // the knot difference, 1 for the Bernstein knots
      const T temp = NDU(r, j - 1);
      NDU(r, j) = saved + (one - s) * temp;
      saved = s * temp;
    }
    NDU(j, j) = saved;
  }
  for (int j = 0; j < order; ++j) {
    OUT(0, j) = NDU(j, n);
  }
  for (int r = 0; r < order; ++r) {
    int s1 = 0;
    int s2 = 1;
    A(0, 0) = one;
    for (int k = 1; k <= n_deriv; ++k) {
      T d = zero;
      const int rk = r - k;
      const int pk = n - k;
      if (r >= k) {
        A(s2, 0) = A(s1, 0);
        d = A(s2, 0) * NDU(rk, pk);
      }
      const int j1 = rk >= -1 ? 1 : -rk;
      const int j2 = (r - 1) <= pk ? k - 1 : n - r;
      for (int jj = j1; jj <= j2; ++jj) {
        A(s2, jj) = A(s1, jj) - A(s1, jj - 1);
        d += A(s2, jj) * NDU(rk + jj, pk);
      }
      if (r <= pk) {
        A(s2, k) = -A(s1, k - 1);
        d += A(s2, k) * NDU(r, pk);
      }
      OUT(k, r) = d;
      std::swap(s1, s2);
    }
  }
  // Row k times n! / (n - k)!, exact in an integer while it fits (as the recurrence it replaces
  // computed it), in double past that; zero from k = n + 1 on.
  std::uint64_t fac = static_cast<std::uint64_t>(n);
  double fac_d = static_cast<double>(n);
  bool exact = true;
  for (int k = 1; k <= n_deriv; ++k) {
    const double f = exact ? static_cast<double>(fac) : fac_d;
    for (int j = 0; j < order; ++j) {
      OUT(k, j) = static_cast<T>(static_cast<double>(OUT(k, j)) * f);
    }
    const int next = n - k;
    if (next <= 0) {
      fac = 0;
      fac_d = 0.0;
      exact = true;
      continue;
    }
    const auto m = static_cast<std::uint64_t>(next);
    if (exact && fac > UINT64_MAX / m) {
      exact = false;
      fac_d = static_cast<double>(fac);
    }
    if (exact) {
      fac *= m;
    } else {
      fac_d *= static_cast<double>(next);
    }
  }
}

/// Write the Bernstein derivatives at every point into `out` (`(n, n_deriv + 1, degree + 1)`).
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `tabulate_bernstein_derivatives_1d` instead.
template <StorageFloat T>
void tabulate_bernstein_derivatives_1d_kernel(int degree, std::span<const T> pts, int n_deriv,
                                              mdview<T, 3> out) {
  const auto ord = static_cast<std::size_t>(degree) + 1;
  const auto nd1 = static_cast<std::size_t>(n_deriv) + 1;
  const std::size_t block = nd1 * ord;
  T* data = out.data_handle();
  const double ops = bernstein_derivatives_ops_per_point(degree, n_deriv);
  detail::for_each_chunk(pts.size(), ops, sizeof(T) * block,
                         [&](std::size_t begin, std::size_t end) {
                           std::vector<T> ndu(ord * ord);
                           std::vector<T> a(2 * nd1);
                           for (std::size_t j = begin; j < end; ++j) {
                             bernstein_derivatives_point<T>(degree, pts[j], n_deriv,
                                                            data + j * block, ndu.data(), a.data());
                           }
                         });
}

}  // namespace detail

/// Evaluate the Bernstein basis of degree `degree` at the points `pts` into `out`.
///
/// `out(j, i)` is `B_{i,degree}(pts[j])`. The recurrence is the O(degree) ratio form, run from
/// the endpoint nearer to the point above `bernstein_max_unmirrored_degree_*` so its seed never
/// underflows. Points outside `[0, 1]` are evaluated by the same formula.
///
/// Args:
///     degree (int): Degree, `>= 0`.
///     pts (std::span<const T>): The points.
///     out (mdview<T, 2>): Output, extents `(pts.size(), degree + 1)`.
///
/// Raises:
///     value_error: `degree < 0`, or `out` has other extents.
template <StorageFloat T>
void tabulate_bernstein_1d(int degree, std::span<const T> pts, mdview<T, 2> out) {
  detail::check_degree(degree);
  detail::check_out<T, 2>(out, {pts.size(), static_cast<std::size_t>(degree) + 1});
  detail::tabulate_bernstein_1d_kernel<T>(degree, pts, out);
}

/// Evaluate the Bernstein basis of degree `degree` at the points `pts`.
///
/// Args:
///     degree (int): Degree, `>= 0`.
///     pts (std::span<const T>): The points.
///
/// Returns:
///     NdArray<T>: Extents `(pts.size(), degree + 1)`; see the `out` overload.
///
/// Raises:
///     value_error: `degree < 0`.
template <StorageFloat T>
[[nodiscard]] NdArray<T> tabulate_bernstein_1d(int degree, std::span<const T> pts) {
  detail::check_degree(degree);
  NdArray<T> out({pts.size(), static_cast<std::size_t>(degree) + 1}, for_overwrite);
  detail::tabulate_bernstein_1d_kernel<T>(degree, pts, out.template view<2>());
  return out;
}

/// Evaluate the derivatives of orders `0..n_deriv` of the Bernstein basis at `pts` into `out`.
///
/// `out(j, k, i)` is the `k`-th derivative of `B_{i,degree}` at `pts[j]`; orders above the
/// degree are zero.
///
/// Args:
///     degree (int): Degree, `>= 0`.
///     pts (std::span<const T>): The points.
///     n_deriv (int): Highest derivative order, `>= 0`.
///     out (mdview<T, 3>): Output, extents `(pts.size(), n_deriv + 1, degree + 1)`.
///
/// Raises:
///     value_error: `degree < 0`, `n_deriv < 0`, or `out` has other extents.
template <StorageFloat T>
void tabulate_bernstein_derivatives_1d(int degree, std::span<const T> pts, int n_deriv,
                                       mdview<T, 3> out) {
  detail::check_degree(degree);
  if (n_deriv < 0) {
    throw value_error("n_deriv must be non-negative");
  }
  detail::check_out<T, 3>(out, {pts.size(), static_cast<std::size_t>(n_deriv) + 1,
                                static_cast<std::size_t>(degree) + 1});
  detail::tabulate_bernstein_derivatives_1d_kernel<T>(degree, pts, n_deriv, out);
}

}  // namespace pantr::basis
