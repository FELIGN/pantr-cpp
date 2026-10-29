/// @file product.hpp
/// @brief The pointwise product of two Beziers, and the Bernstein product formula behind it.
#pragma once

#include "pantr/bezier/bezier.hpp"
#include "pantr/config.hpp"
#include "pantr/core/binomial.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bezier {

namespace detail {

/// The binomial coefficients `C(n, 0..n)` rounded to `T`, as `math.comb` cast to the array dtype.
template <StorageFloat T>
[[nodiscard]] std::vector<T> binomials_as(std::size_t n) {
  std::vector<T> c(n + 1);
  for (std::size_t i = 0; i <= n; ++i) {
    c[i] = static_cast<T>(binomial_double(static_cast<long long>(n), static_cast<long long>(i)));
  }
  return c;
}

/// The reciprocals `1 / C(n, 0..n)`, each formed in `double` and rounded to `T`.
template <StorageFloat T>
[[nodiscard]] std::vector<T> inverse_binomials_as(std::size_t n) {
  std::vector<T> c(n + 1);
  for (std::size_t i = 0; i <= n; ++i) {
    c[i] =
        static_cast<T>(1.0 / binomial_double(static_cast<long long>(n), static_cast<long long>(i)));
  }
  return c;
}

/// The row-major outer product of per-direction vectors, `((v_0[i_0] v_1[i_1]) v_2[i_2]) ...`,
/// every product rounded to `T` (`np.multiply.outer` applied left to right).
template <StorageFloat T>
[[nodiscard]] std::vector<T> outer_product(const std::vector<std::vector<T>>& vs) {
  std::vector<T> w = vs[0];
  for (std::size_t d = 1; d < vs.size(); ++d) {
    std::vector<T> next(w.size() * vs[d].size());
    for (std::size_t i = 0; i < w.size(); ++i) {
      for (std::size_t j = 0; j < vs[d].size(); ++j) {
        next[i * vs[d].size() + j] = w[i] * vs[d][j];
      }
    }
    w = std::move(next);
  }
  return w;
}

/// The Bernstein product of two curves with `rank` independent columns:
///
///     d_k = 1 / C(p + q, k) sum_i C(p, i) C(q, k - i) f_i g_{k - i},
///
/// with the coefficient `C(p, i) C(q, j) / C(p + q, i + j)` formed in `T`, multiplied by `f_i` and
/// then `g_j`, and the terms added in `T` in the order of `i`, then `j`.
///
/// Args:
///     f (std::span<const T>): Control points `(p + 1, rank)`.
///     g (std::span<const T>): Control points `(q + 1, rank)`.
///     p (std::size_t): Degree of `f`.
///     q (std::size_t): Degree of `g`.
///     rank (std::size_t): Number of columns.
///
/// Returns:
///     NdArray<T>: Control points `(p + q + 1, rank)`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] NdArray<T> product_1d_formula(std::span<const T> f, std::span<const T> g,
                                            std::size_t p, std::size_t q, std::size_t rank) {
  const std::vector<T> cp = binomials_as<T>(p);
  const std::vector<T> cq = binomials_as<T>(q);
  const std::vector<T> inv_cr = inverse_binomials_as<T>(p + q);
  NdArray<T> out({p + q + 1, rank});
  T* d = out.data();
  for (std::size_t i = 0; i <= p; ++i) {
    for (std::size_t j = 0; j <= q; ++j) {
      const T coeff = cp[i] * cq[j] * inv_cr[i + j];
      T* row = d + (i + j) * rank;
      for (std::size_t c = 0; c < rank; ++c) {
        row[c] = row[c] + coeff * f[i * rank + c] * g[j * rank + c];
      }
    }
  }
  return out;
}

/// The tensor-product Bernstein product of two patches with `rank` independent columns: both
/// operands are weighted by the outer product of their per-direction binomials, convolved by
/// accumulating, for each index `alpha` of `f`, the product with the whole of `g` at the offset
/// `alpha`, and the result scaled by the outer product of the inverse binomials of the summed
/// degrees. Every operation is in `T`; `alpha` runs in row-major order.
///
/// Args:
///     f (std::span<const T>): Control points `(p_0 + 1, ..., p_{D-1} + 1, rank)`.
///     p (std::span<const std::size_t>): Degrees of `f`.
///     g (std::span<const T>): Control points `(q_0 + 1, ..., q_{D-1} + 1, rank)`.
///     q (std::span<const std::size_t>): Degrees of `g`.
///     rank (std::size_t): Number of columns.
///
/// Returns:
///     NdArray<T>: Control points `(p_0 + q_0 + 1, ..., p_{D-1} + q_{D-1} + 1, rank)`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] NdArray<T> product_nd_formula(std::span<const T> f, std::span<const std::size_t> p,
                                            std::span<const T> g, std::span<const std::size_t> q,
                                            std::size_t rank) {
  const std::size_t ndim = p.size();
  std::vector<std::vector<T>> binom_p(ndim);
  std::vector<std::vector<T>> binom_q(ndim);
  std::vector<std::vector<T>> inv_binom_r(ndim);
  std::vector<std::size_t> r_ext(ndim);
  for (std::size_t d = 0; d < ndim; ++d) {
    binom_p[d] = binomials_as<T>(p[d]);
    binom_q[d] = binomials_as<T>(q[d]);
    inv_binom_r[d] = inverse_binomials_as<T>(p[d] + q[d]);
    r_ext[d] = p[d] + q[d] + 1;
  }
  const std::vector<T> w_f = outer_product(binom_p);
  const std::vector<T> w_g = outer_product(binom_q);
  const std::vector<T> inv_w_r = outer_product(inv_binom_r);
  std::vector<T> weighted_f(f.size());
  for (std::size_t a = 0; a < w_f.size(); ++a) {
    for (std::size_t c = 0; c < rank; ++c) {
      weighted_f[a * rank + c] = w_f[a] * f[a * rank + c];
    }
  }
  std::vector<T> weighted_g(g.size());
  for (std::size_t a = 0; a < w_g.size(); ++a) {
    for (std::size_t c = 0; c < rank; ++c) {
      weighted_g[a * rank + c] = w_g[a] * g[a * rank + c];
    }
  }
  // Row-major strides of the result grid, and the offset of every index of `g` in it.
  std::vector<std::size_t> stride(ndim, 1);
  for (std::size_t d = ndim - 1; d-- > 0;) {
    stride[d] = stride[d + 1] * r_ext[d + 1];
  }
  std::vector<std::size_t> q_ext(ndim);
  for (std::size_t d = 0; d < ndim; ++d) {
    q_ext[d] = q[d] + 1;
  }
  std::vector<std::size_t> beta_off(w_g.size());
  std::vector<std::size_t> idx(ndim, 0);
  for (std::size_t bf = 0; bf < beta_off.size(); ++bf) {
    std::size_t off = 0;
    for (std::size_t d = 0; d < ndim; ++d) {
      off += idx[d] * stride[d];
    }
    beta_off[bf] = off;
    for (std::size_t d = ndim; d-- > 0;) {
      if (++idx[d] < q_ext[d]) {
        break;
      }
      idx[d] = 0;
    }
  }
  std::vector<std::size_t> shape(r_ext);
  shape.push_back(rank);
  NdArray<T> out(std::move(shape));
  const std::size_t n_out = inv_w_r.size();
  std::vector<T> h(n_out * rank, static_cast<T>(0));
  std::vector<std::size_t> p_ext(ndim);
  for (std::size_t d = 0; d < ndim; ++d) {
    p_ext[d] = p[d] + 1;
  }
  std::fill(idx.begin(), idx.end(), std::size_t{0});
  for (std::size_t af = 0; af < w_f.size(); ++af) {
    std::size_t base = 0;
    for (std::size_t d = 0; d < ndim; ++d) {
      base += idx[d] * stride[d];
    }
    for (std::size_t bf = 0; bf < beta_off.size(); ++bf) {
      T* row = h.data() + (base + beta_off[bf]) * rank;
      for (std::size_t c = 0; c < rank; ++c) {
        row[c] = row[c] + weighted_f[af * rank + c] * weighted_g[bf * rank + c];
      }
    }
    for (std::size_t d = ndim; d-- > 0;) {
      if (++idx[d] < p_ext[d]) {
        break;
      }
      idx[d] = 0;
    }
  }
  T* dst = out.data();
  for (std::size_t g_i = 0; g_i < n_out; ++g_i) {
    for (std::size_t c = 0; c < rank; ++c) {
      dst[g_i * rank + c] = inv_w_r[g_i] * h[g_i * rank + c];
    }
  }
  return out;
}

/// Refuse operands of `bernstein_product` that do not describe two arrays of one rank.
template <StorageFloat T>
void check_product_operands(std::span<const T> a, std::span<const std::size_t> a_shape,
                            std::span<const T> b, std::span<const std::size_t> b_shape) {
  if (a_shape.size() < 2 || a_shape.size() != b_shape.size()) {
    throw value_error("Operands must have the same number of dimensions, at least 2; got shapes " +
                      py_repr(a_shape) + " and " + py_repr(b_shape) + ".");
  }
  if (a_shape.back() != b_shape.back() || a_shape.back() == 0) {
    throw value_error("Operands must have the same rank, at least 1; got shapes " +
                      py_repr(a_shape) + " and " + py_repr(b_shape) + ".");
  }
  for (std::size_t d = 0; d + 1 < a_shape.size(); ++d) {
    if (a_shape[d] == 0 || b_shape[d] == 0) {
      throw value_error(
          "Operands must have at least 1 entry in every parametric direction; got "
          "shapes " +
          py_repr(a_shape) + " and " + py_repr(b_shape) + ".");
    }
  }
  if (checked_product(a_shape) != a.size() || checked_product(b_shape) != b.size()) {
    throw value_error("Operand sizes " + std::to_string(a.size()) + " and " +
                      std::to_string(b.size()) + " do not match their shapes " + py_repr(a_shape) +
                      " and " + py_repr(b_shape) + ".");
  }
}

/// The parametric degrees of an operand shape `(p_0 + 1, ..., p_{D-1} + 1, rank)`.
[[nodiscard]] inline std::vector<std::size_t> degrees_of(std::span<const std::size_t> shape) {
  std::vector<std::size_t> p(shape.size() - 1);
  for (std::size_t d = 0; d < p.size(); ++d) {
    p[d] = shape[d] - 1;
  }
  return p;
}

}  // namespace detail

/// Multiply two arrays of Bernstein coefficients with the tensor-product formula, in every case
/// (even for one parametric direction, where `bernstein_product` uses the 1D formula, which
/// rounds differently).
///
/// Args:
///     a (std::span<const T>): Control points of the first operand, row-major over `a_shape`.
///     a_shape (std::span<const std::size_t>): `(p_0 + 1, ..., p_{D-1} + 1, rank)`.
///     b (std::span<const T>): Control points of the second operand.
///     b_shape (std::span<const std::size_t>): `(q_0 + 1, ..., q_{D-1} + 1, rank)`, the same
///         number of extents and the same rank as `a_shape`.
///
/// Returns:
///     NdArray<T>: `(p_0 + q_0 + 1, ..., p_{D-1} + q_{D-1} + 1, rank)`.
///
/// Raises:
///     value_error: The shapes differ in length or rank, an extent is zero, or a span does not
///         hold the product of its shape.
///     std::overflow_error: A binomial coefficient of the product degree exceeds the `double`
///         range.
template <StorageFloat T>
[[nodiscard]] NdArray<T> bernstein_product_nd(std::span<const T> a,
                                              std::span<const std::size_t> a_shape,
                                              std::span<const T> b,
                                              std::span<const std::size_t> b_shape) {
  detail::check_product_operands<T>(a, a_shape, b, b_shape);
  const std::vector<std::size_t> p = detail::degrees_of(a_shape);
  const std::vector<std::size_t> q = detail::degrees_of(b_shape);
  return detail::product_nd_formula<T>(a, p, b, q, a_shape.back());
}

/// Multiply two arrays of Bernstein coefficients (the Bernstein product formula), the columns of
/// the last axis independently.
///
/// One parametric direction (`a_shape` of two extents) uses the 1D formula, more use the
/// tensor-product formula (`bernstein_product_nd`); the two round differently, and this is the
/// choice `multiply` makes. Coefficients are formed with `binomial_double`, so they are rounded
/// once from the exact integers.
///
/// Args:
///     a (std::span<const T>): Control points of the first operand, row-major over `a_shape`.
///     a_shape (std::span<const std::size_t>): `(p + 1, rank)` or `(p_0 + 1, ..., p_{D-1} + 1,
///         rank)`.
///     b (std::span<const T>): Control points of the second operand.
///     b_shape (std::span<const std::size_t>): The same number of extents and the same rank as
///         `a_shape`.
///
/// Returns:
///     NdArray<T>: `(p + q + 1, rank)` or `(p_0 + q_0 + 1, ..., rank)`.
///
/// Raises:
///     value_error: As `bernstein_product_nd`.
///     std::overflow_error: As `bernstein_product_nd`.
template <StorageFloat T>
[[nodiscard]] NdArray<T> bernstein_product(std::span<const T> a,
                                           std::span<const std::size_t> a_shape,
                                           std::span<const T> b,
                                           std::span<const std::size_t> b_shape) {
  detail::check_product_operands<T>(a, a_shape, b, b_shape);
  if (a_shape.size() == 2) {
    return detail::product_1d_formula<T>(a, b, a_shape[0] - 1, b_shape[0] - 1, a_shape[1]);
  }
  const std::vector<std::size_t> p = detail::degrees_of(a_shape);
  const std::vector<std::size_t> q = detail::degrees_of(b_shape);
  return detail::product_nd_formula<T>(a, p, b, q, a_shape.back());
}

/// Compute the pointwise product of two Beziers.
///
/// The result has degree `p_d + q_d` in direction `d`: the degree is exact and the control points
/// carry floating-point roundoff. If either operand is rational, both are taken in homogeneous
/// form (a non-rational one with unit weights) and the numerators and the weights are multiplied
/// separately.
///
/// Args:
///     a (const Bezier<T>&): The first operand.
///     b (const Bezier<T>&): The second operand, of the same dimension and rank.
///
/// Returns:
///     Bezier<T>: A Bezier `h` with `h(t) = a(t) * b(t)`, componentwise.
///
/// Raises:
///     value_error: The operands differ in dimension, or in rank (checked in that order).
///     std::overflow_error: A binomial coefficient of the product degree exceeds the `double`
///         range.
template <StorageFloat T>
[[nodiscard]] Bezier<T> multiply(const Bezier<T>& a, const Bezier<T>& b) {
  if (a.dim() != b.dim()) {
    throw value_error("Operands must have the same dimension. Got " + std::to_string(a.dim()) +
                      " and " + std::to_string(b.dim()) + ".");
  }
  if (a.rank() != b.rank()) {
    throw value_error("Operands must have the same rank. Got " + std::to_string(a.rank()) +
                      " and " + std::to_string(b.rank()) + ".");
  }
  const std::size_t dim = a.dim();
  const std::size_t rank = a.rank();
  if (!a.rational() && !b.rational()) {
    return Bezier<T>(
        bernstein_product<T>(a.control_points(), a.shape(), b.control_points(), b.shape()), false);
  }
  // Homogeneous form: numerator columns (rows, rank) and weight column (rows, 1) of each operand.
  struct Parts {
    std::vector<T> num;
    std::vector<T> den;
    std::vector<std::size_t> num_shape;
    std::vector<std::size_t> den_shape;
  };
  auto split = [dim, rank](const Bezier<T>& x) {
    Parts parts;
    const std::size_t cp = x.cp_size();
    const std::size_t rows = x.size() / cp;
    parts.num.resize(rows * rank);
    parts.den.resize(rows);
    for (std::size_t r = 0; r < rows; ++r) {
      for (std::size_t c = 0; c < rank; ++c) {
        parts.num[r * rank + c] = x.control_points()[r * cp + c];
      }
      parts.den[r] = x.rational() ? x.control_points()[r * cp + rank] : static_cast<T>(1);
    }
    parts.num_shape.assign(x.shape().begin(), x.shape().begin() + static_cast<std::ptrdiff_t>(dim));
    parts.den_shape = parts.num_shape;
    parts.num_shape.push_back(rank);
    parts.den_shape.push_back(1);
    return parts;
  };
  const Parts pa = split(a);
  const Parts pb = split(b);
  const NdArray<T> h_num = bernstein_product<T>(pa.num, pa.num_shape, pb.num, pb.num_shape);
  const NdArray<T> h_den = bernstein_product<T>(pa.den, pa.den_shape, pb.den, pb.den_shape);
  std::vector<std::size_t> shape(h_num.shape().begin(), h_num.shape().end());
  shape.back() = rank + 1;
  NdArray<T> out(std::move(shape), for_overwrite);
  const std::size_t rows = h_den.size();
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < rank; ++c) {
      out.data()[r * (rank + 1) + c] = h_num.data()[r * rank + c];
    }
    out.data()[r * (rank + 1) + rank] = h_den.data()[r];
  }
  return Bezier<T>(std::move(out), true);
}

}  // namespace pantr::bezier
