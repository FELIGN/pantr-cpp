/// @file compose.hpp
/// @brief Composition of two Beziers: `outer(inner(t))`.
#pragma once

#include "pantr/bezier/bezier.hpp"
#include "pantr/bezier/detail/kernels_1d.hpp"
#include "pantr/bezier/product.hpp"
#include "pantr/config.hpp"
#include "pantr/core/binomial.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bezier {

namespace detail {

/// The Bernstein product of two scalar Bezier coefficient arrays of shape `(*degrees + 1, 1)`.
///
/// A 1D inner map takes the scalar 1D kernel (`scalar_bernstein_product_1d_kernel`), several
/// directions the tensor-product formula (`bernstein_product_nd`); the two round differently.
///
/// Args:
///     a (const NdArray<T>&): Coefficients of the first factor.
///     b (const NdArray<T>&): Coefficients of the second factor, of the same number of extents.
///     use_1d_kernel (bool): Whether the arrays have two extents and take the 1D kernel.
///
/// Returns:
///     NdArray<T>: The coefficients of the product, extents `(*(p + q) + 1, 1)`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed), `p + q <= bincoeff_max_n`
///     included for the 1D kernel.
template <StorageFloat T>
[[nodiscard]] NdArray<T> scalar_product(const NdArray<T>& a, const NdArray<T>& b,
                                        bool use_1d_kernel) {
  if (use_1d_kernel) {
    NdArray<T> out({a.size() + b.size() - 1, 1}, for_overwrite);
    scalar_bernstein_product_1d_kernel<T>(a.flat(), b.flat(), out.flat());
    return out;
  }
  return bernstein_product_nd<T>(a.flat(), a.shape(), b.flat(), b.shape());
}

/// The powers `g, g^2, ..., g^max_power` of a scalar Bezier, each the Bernstein product of the
/// previous power with `g`.
///
/// Args:
///     g (const NdArray<T>&): Coefficients of `g`, extents `(*degrees + 1, 1)`.
///     max_power (std::size_t): The highest power, `>= 1`.
///     use_1d_kernel (bool): As `scalar_product`.
///
/// Returns:
///     std::vector<NdArray<T>>: `max_power` arrays; entry `k` is `g^(k + 1)`.
template <StorageFloat T>
[[nodiscard]] std::vector<NdArray<T>> scalar_powers(const NdArray<T>& g, std::size_t max_power,
                                                    bool use_1d_kernel) {
  std::vector<NdArray<T>> powers;
  powers.reserve(max_power);
  powers.push_back(g);
  for (std::size_t k = 1; k < max_power; ++k) {
    powers.push_back(scalar_product<T>(powers.back(), g, use_1d_kernel));
  }
  return powers;
}

/// The Bernstein basis functions `B_i^m(g) = C(m, i) g^i (1 - g)^(m - i)`, `i = 0..m`, of a scalar
/// Bezier `g`, each as a Bezier coefficient array of degree `m` times the degree of `g`.
///
/// The end functions are the powers `(1 - g)^m` and `g^m`; the others the Bernstein product of
/// `g^i` and `(1 - g)^(m - i)`, multiplied by `C(m, i)` rounded to `T`. The coefficients of
/// `1 - g` are `1 - g` coefficient by coefficient, in `T` (the partition of unity).
///
/// Args:
///     g (const NdArray<T>&): Coefficients of `g`, extents `(*degrees + 1, 1)`.
///     degree (std::size_t): The Bernstein degree `m`, `>= 0`.
///     use_1d_kernel (bool): As `scalar_product`.
///
/// Returns:
///     std::vector<NdArray<T>>: `m + 1` arrays; for `m == 0` the single constant `1`, all of whose
///     extents are 1.
///
/// Raises:
///     std::overflow_error: `C(m, i)` exceeds the `double` range.
template <StorageFloat T>
[[nodiscard]] std::vector<NdArray<T>> bernstein_bases(const NdArray<T>& g, std::size_t degree,
                                                      bool use_1d_kernel) {
  std::vector<NdArray<T>> bases;
  if (degree == 0) {
    NdArray<T> one(std::vector<std::size_t>(g.rank(), 1));
    one.data()[0] = static_cast<T>(1);
    bases.push_back(std::move(one));
    return bases;
  }
  NdArray<T> one_minus_g(std::vector<std::size_t>(g.shape().begin(), g.shape().end()),
                         for_overwrite);
  for (std::size_t i = 0; i < g.size(); ++i) {
    one_minus_g.data()[i] = static_cast<T>(1) - g.data()[i];
  }
  std::vector<NdArray<T>> g_powers = scalar_powers<T>(g, degree, use_1d_kernel);
  std::vector<NdArray<T>> omg_powers = scalar_powers<T>(one_minus_g, degree, use_1d_kernel);
  bases.reserve(degree + 1);
  // The two end functions take the top powers; the loop below reads only lower ones.
  bases.push_back(std::move(omg_powers[degree - 1]));
  for (std::size_t i = 1; i < degree; ++i) {
    NdArray<T> prod = scalar_product<T>(g_powers[i - 1], omg_powers[degree - i - 1], use_1d_kernel);
    const T binom =
        static_cast<T>(binomial_double(static_cast<long long>(degree), static_cast<long long>(i)));
    for (std::size_t k = 0; k < prod.size(); ++k) {
      prod.data()[k] = binom * prod.data()[k];
    }
    bases.push_back(std::move(prod));
  }
  bases.push_back(std::move(g_powers[degree - 1]));
  return bases;
}

}  // namespace detail

/// Compose two non-rational Beziers: the Bezier `h(t) = outer(inner(t))`.
///
/// Each scalar component `g_d` of `inner` is expanded in the Bernstein basis of `outer`'s degree
/// `m_d` in direction `d`, `B_i^{m_d}(g_d)`, as Bezier coefficients: the powers of `g_d` and of
/// `1 - g_d` are Bernstein products, and the function is a product of two of them times a binomial
/// coefficient. The result sums, over the control points `P_i` of `outer`, `P_i` times the product
/// over `d` of those functions, again by Bernstein products. All of it is in `T`; the degree of
/// the result, `sum(outer degrees) * inner degree[s]` in direction `s`, is exact and its control
/// points carry the roundoff of every one of those products.
///
/// A 1D `inner` forms its products with the scalar 1D kernel, whose binomial coefficients
/// `C(r, k)` run up to the full composed degree `r = sum(outer degrees) * inner degree`: the
/// powers of the inner map climb to `outer degree[d] * inner degree`, and the cross-direction
/// products multiply the per-direction functions together. The exact-integer binomial kernel
/// wraps silently past `bincoeff_max_n`, so the envelope is checked on the composed degree. It is
/// *multiplicative* in the operands, which makes it bind at far lower degrees than that of degree
/// elevation: an outer of degree 6 composed with an inner of degree 11 already asks for `C(66,
/// k)`. Two configurations are exempt:
///
///   - An inner map of several directions takes `bernstein_product_nd`, whose binomials come from
///     `binomial_double` and cannot wrap.
///   - A 1D outer of degree at most 1 forms no product at all (the power ladder needs degree 2 and
///     there is no second direction to multiply into), so the kernel is never called. A
///     degree-(1, 1) surface as outer is not exempt: its cross-direction product runs.
///
/// Args:
///     outer (const Bezier<T>&): The mapping being composed; non-rational.
///     inner (const Bezier<T>&): The reparametrization; non-rational, `inner.rank() ==
///         outer.dim()`.
///
/// Returns:
///     Bezier<T>: A non-rational Bezier of dimension `inner.dim()` and rank `outer.rank()`.
///
/// Raises:
///     type_error: `outer` is rational, or `inner` is.
///     value_error: `inner.rank() != outer.dim()` (checked after the rationality); or a 1D `inner`
///         asks, with an outer of degree above 1 or of several directions, for a composed degree
///         beyond `bincoeff_max_n`.
///     capacity_error: A degree of the result overflows `std::size_t`.
///     std::overflow_error: A binomial coefficient exceeds the `double` range.
template <StorageFloat T>
[[nodiscard]] Bezier<T> compose(const Bezier<T>& outer, const Bezier<T>& inner) {
  if (outer.rational()) {
    throw type_error("Composition is not supported for rational Béziers (outer is rational).");
  }
  if (inner.rational()) {
    throw type_error("Composition is not supported for rational Béziers (inner is rational).");
  }
  if (inner.rank() != outer.dim()) {
    throw value_error("Inner Bézier rank (" + std::to_string(inner.rank()) +
                      ") must equal outer Bézier parametric dimension (" +
                      std::to_string(outer.dim()) + ").");
  }
  const std::size_t dim_f = outer.dim();
  const std::size_t dim_g = inner.dim();
  const std::size_t rank = outer.rank();
  const std::vector<std::size_t> deg_f = outer.degrees();
  std::size_t sum_deg_f = 0;
  for (const std::size_t m : deg_f) {
    sum_deg_f += m;
  }
  if (dim_g == 1 && (dim_f > 1 || deg_f[0] > 1)) {
    // The product saturates at the largest `long long`: past `bincoeff_max_n` the exact value
    // only reaches the message.
    constexpr auto ll_max = static_cast<std::size_t>((std::numeric_limits<long long>::max)());
    const std::size_t n = inner.degree(0);
    const std::size_t composed = (n != 0 && sum_deg_f > ll_max / n) ? ll_max : sum_deg_f * n;
    check_bincoeff_envelope(
        static_cast<long long>(composed),
        "Composition to degree " + std::to_string(composed) + " with a 1D inner Bézier");
  }
  const bool use_1d_kernel = dim_g == 1;

  // The extents of the result, `sum(outer degrees) * inner degree[s] + 1` per direction.
  std::vector<std::size_t> result_shape(dim_g + 1, rank);
  for (std::size_t s = 0; s < dim_g; ++s) {
    const std::array<std::size_t, 2> factors = {sum_deg_f, inner.degree(s)};
    result_shape[s] = checked_product(factors) + 1;
  }

  // Step 1: the Bernstein functions of outer's degree of each component of inner.
  const std::size_t inner_cols = inner.cp_size();
  std::vector<std::size_t> g_shape(inner.shape().begin(), inner.shape().end());
  g_shape.back() = 1;
  std::vector<std::vector<NdArray<T>>> all_bases;
  all_bases.reserve(dim_f);
  for (std::size_t d = 0; d < dim_f; ++d) {
    NdArray<T> g_d(g_shape, for_overwrite);
    for (std::size_t i = 0; i < g_d.size(); ++i) {
      g_d.data()[i] = inner.control_points()[i * inner_cols + d];
    }
    all_bases.push_back(detail::bernstein_bases<T>(g_d, deg_f[d], use_1d_kernel));
  }

  // Step 2: sum, over the control points of outer in row-major order, the point times the product
  // of the per-direction functions.
  NdArray<T> result(result_shape);
  const std::size_t n_out = result.size() / rank;
  const std::size_t n_points = outer.size() / rank;
  std::vector<std::size_t> idx(dim_f, 0);
  for (std::size_t point = 0; point < n_points; ++point) {
    NdArray<T> basis = all_bases[0][idx[0]];
    for (std::size_t d = 1; d < dim_f; ++d) {
      basis = detail::scalar_product<T>(basis, all_bases[d][idx[d]], use_1d_kernel);
    }
    const T* coef = outer.control_points().data() + point * rank;
    T* dst = result.data();
    for (std::size_t i = 0; i < n_out; ++i) {
      for (std::size_t c = 0; c < rank; ++c) {
        dst[i * rank + c] = dst[i * rank + c] + coef[c] * basis.data()[i];
      }
    }
    // The next multi-index of outer's control points, last direction fastest.
    for (std::size_t d = dim_f; d-- > 0;) {
      if (++idx[d] <= deg_f[d]) {
        break;
      }
      idx[d] = 0;
    }
  }
  return Bezier<T>(std::move(result), false);
}

}  // namespace pantr::bezier
