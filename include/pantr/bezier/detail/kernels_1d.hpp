/// @file kernels_1d.hpp
/// @brief Layer 3 kernels of the Bezier operations: one parametric direction, control points
///        viewed as `(n_coeff, n_cols)`.
///
/// Precision follows the Numba kernels these replace, site by site (design §2.3): wherever the
/// Numba code met a `float64` scalar or literal (the parameter values, `1.0 - u`, `(p - i + 1.0) /
/// i`, a binomial coefficient) it promoted to `float64`, so these kernels compute in `double` and
/// store each result in `T`; a value that Numba kept in the array's dtype (the derivative table)
/// stays in `T`.
///
/// None of these kernels validates anything.
#pragma once

#include "pantr/basis/bernstein.hpp"
#include "pantr/config.hpp"
#include "pantr/core/binomial.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/parallel/parallel_for.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace pantr::bezier::detail {

/// Midpoint at which `evaluate_1d_kernel` switches to the mirrored recurrence. It bounds the
/// recurrence seed (`(1 - u)^p` or `u^p`) below by `0.5^p`, so the seed does not underflow below
/// degree 1075 (the seed is a double in both storage types), the envelope `basis/bernstein.hpp`
/// states; past it the seed is zero and so is the value, as in the previous release.
/// `basis/bernstein.hpp` mirrors at the same midpoint.
inline constexpr double mirror_threshold = 0.5;

/// Evaluate a 1D Bezier at every point, fusing the Bernstein recurrence with the contraction.
///
/// For each point the `p + 1` Bernstein values come from the ratio recurrence and are multiplied
/// into the output at once, so no `(n_pts, p + 1)` basis matrix is allocated. The recurrence runs
/// from the endpoint that keeps its seed away from underflow, by the symmetry `B_{i,p}(u) =
/// B_{p-i,p}(1 - u)`:
///
///   - `u <= 0.5`: forward from `(1 - u)^p`, control points ascending.
///   - `u > 0.5`: the same forward recurrence on `1 - u` (seed `u^p`, ratio `(1 - u) / u`), control
///     points descending (step `i` pairs with `ctrl(p - i)`), which reuses the ascending loop
///     instead of a descending recurrence.
///   - `p == 0` copies `ctrl(0)`; `u == 1` copies `ctrl(p)`.
///
/// The running basis value is a `double` scalar and the output accumulates in `T` (each partial
/// sum is stored, so it is rounded to `T`), as the Numba kernel did.
///
/// Args:
///     ctrl (mdview<const T, 2>): Control points, extents `(p + 1, rank)`.
///     pts (std::span<const T>): Evaluation points.
///     out (mdview<T, 2>): Output, extents `(pts.size(), rank)`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `evaluate` instead.
template <StorageFloat T>
void evaluate_1d_kernel(mdview<const T, 2> ctrl, std::span<const T> pts, mdview<T, 2> out) {
  const std::size_t p = ctrl.extent(0) - 1;
  const std::size_t rank = ctrl.extent(1);
  // One point costs about four operations per step of the recurrence (the quotient
  // `(p - i + 1) / i`, two products for the next basis value, and its store) plus a multiply-add
  // per output column, over `p + 1` steps.
  const double ops = (static_cast<double>(p) + 1.0) * (4.0 + 2.0 * static_cast<double>(rank));
  parallel::parallel_for(
      pts.size(), parallel::WorkEstimate{ops}, [&](std::size_t begin, std::size_t end) {
        for (std::size_t j = begin; j < end; ++j) {
          const double u = static_cast<double>(pts[j]);
          if (p == 0) {
            for (std::size_t r = 0; r < rank; ++r) {
              out(j, r) = ctrl(0, r);
            }
          } else if (u == 1.0) {
            for (std::size_t r = 0; r < rank; ++r) {
              out(j, r) = ctrl(p, r);
            }
          } else if (u > mirror_threshold) {
            const double one_minus_u = 1.0 - u;
            // b = B_{0,p}(1 - u) = u^p = B_{p,p}(u): pairs with ctrl(p).
            double b = basis::detail::int_power(u, static_cast<int>(p));
            for (std::size_t r = 0; r < rank; ++r) {
              out(j, r) = static_cast<T>(b * static_cast<double>(ctrl(p, r)));
            }
            const double ratio = one_minus_u / u;
            for (std::size_t i = 1; i <= p; ++i) {
              const double const_factor =
                  (static_cast<double>(p) - static_cast<double>(i) + 1.0) / static_cast<double>(i);
              b = b * const_factor * ratio;
              // b = B_{i,p}(1 - u) = B_{p-i,p}(u): pairs with ctrl(p - i).
              for (std::size_t r = 0; r < rank; ++r) {
                out(j, r) = static_cast<T>(static_cast<double>(out(j, r)) +
                                           b * static_cast<double>(ctrl(p - i, r)));
              }
            }
          } else {
            const double one_minus_u = 1.0 - u;
            double b = basis::detail::int_power(one_minus_u, static_cast<int>(p));  // B_{0,p}
            for (std::size_t r = 0; r < rank; ++r) {
              out(j, r) = static_cast<T>(b * static_cast<double>(ctrl(0, r)));
            }
            const double ratio = u / one_minus_u;
            for (std::size_t i = 1; i <= p; ++i) {
              const double const_factor =
                  (static_cast<double>(p) - static_cast<double>(i) + 1.0) / static_cast<double>(i);
              b = b * const_factor * ratio;
              for (std::size_t r = 0; r < rank; ++r) {
                out(j, r) = static_cast<T>(static_cast<double>(out(j, r)) +
                                           b * static_cast<double>(ctrl(i, r)));
              }
            }
          }
        }
      });
}

/// Evaluate the derivatives of orders `0..n_deriv` of a 1D Bezier at every point.
///
/// The Bernstein derivative table of each point is built by `bernstein_derivatives_point`
/// (Piegl and Tiller's Algorithm A2.3 with unit knot spans, in `T`, rows past the degree zero) and
/// contracted with the control points. Order 0 accumulates in `T`; orders `>= 1` accumulate each
/// product in `double` and store the partial sum in `T`, as the Numba kernel did. One deviation:
/// the table row already carries the factor `p! / (p - k)!` rounded to `T`, where the Numba kernel
/// formed that product in `double` and used it unrounded; for `T = float` that is one more `float`
/// rounding per table entry, for `T = double` no difference.
///
/// Args:
///     ctrl (mdview<const T, 2>): Control points, extents `(p + 1, rank)`.
///     pts (std::span<const T>): Evaluation points.
///     n_deriv (int): Highest derivative order, `>= 0`.
///     out (mdview<T, 3>): Output, extents `(pts.size(), n_deriv + 1, rank)`; `out(j, k, :)` is the
///         `k`-th derivative at `pts[j]`, zero for `k > p`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `evaluate_derivatives` instead.
template <StorageFloat T>
void evaluate_derivatives_1d_kernel(mdview<const T, 2> ctrl, std::span<const T> pts, int n_deriv,
                                    mdview<T, 3> out) {
  const std::size_t ord = ctrl.extent(0);
  const std::size_t rank = ctrl.extent(1);
  const int degree = static_cast<int>(ord) - 1;
  const auto nd1 = static_cast<std::size_t>(n_deriv) + 1;
  // The table costs `(p + 1)^2 (n_deriv + 2)` (basis::detail::bernstein_derivatives_ops_per_point),
  // and the contraction one multiply-add per table entry and column.
  const double ops = basis::detail::bernstein_derivatives_ops_per_point(degree, n_deriv) +
                     2.0 * static_cast<double>(nd1 * ord * rank);
  parallel::parallel_for(pts.size(), parallel::WorkEstimate{ops},
                         [&](std::size_t begin, std::size_t end) {
                           std::vector<T> table(nd1 * ord);
                           std::vector<T> ndu(ord * ord);
                           std::vector<T> a(2 * nd1);
                           for (std::size_t j = begin; j < end; ++j) {
                             basis::detail::bernstein_derivatives_point<T>(
                                 degree, pts[j], n_deriv, table.data(), ndu.data(), a.data());
                             for (std::size_t r = 0; r < rank; ++r) {
                               T acc = static_cast<T>(0);
                               for (std::size_t i = 0; i < ord; ++i) {
                                 acc = acc + table[i] * ctrl(i, r);
                               }
                               out(j, 0, r) = acc;
                             }
                             for (std::size_t k = 1; k < nd1; ++k) {
                               for (std::size_t r = 0; r < rank; ++r) {
                                 T acc = static_cast<T>(0);
                                 for (std::size_t i = 0; i < ord; ++i) {
                                   acc = static_cast<T>(static_cast<double>(acc) +
                                                        static_cast<double>(table[k * ord + i]) *
                                                            static_cast<double>(ctrl(i, r)));
                                 }
                                 out(j, k, r) = acc;
                               }
                             }
                           }
                         });
}

/// Degree-elevate control points `(p + 1, n_cols)` by `t` degrees, to `(p + t + 1, n_cols)`.
///
/// Builds the matrix `alfs` of Bezier degree-elevation coefficients in `double`,
///
///     Q_i = sum_{j = max(0, i - t)}^{min(p, i)} C(p, j) C(t, i - j) / C(p + t, i) P_j,
///
/// from the first half of the rows and the symmetry `alfs(j, i) = alfs(p - j, p + t - i)`, and
/// applies it, accumulating each term in `double` and storing the partial sum in `T`. Serial, as
/// the Numba kernel was.
///
/// Args:
///     degree (std::size_t): Original degree `p`.
///     increment (std::size_t): Degrees to add `t`, `>= 1`.
///     ctrl (mdview<const T, 2>): Control points, extents `(p + 1, n_cols)`.
///     out (mdview<T, 2>): Output, extents `(p + t + 1, n_cols)`; overwritten.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed), `p + t <= bincoeff_max_n`
///     included. For general use, call `elevate_degree_1d` instead.
template <StorageFloat T>
void elevate_degree_1d_kernel(std::size_t degree, std::size_t increment, mdview<const T, 2> ctrl,
                              mdview<T, 2> out) {
  const std::size_t p = degree;
  const std::size_t t = increment;
  const std::size_t ph = p + t;
  const std::size_t n_cols = ctrl.extent(1);
  const std::size_t ph2 = ph / 2;
  auto bc = [](std::size_t n, std::size_t k) {
    return bincoeff(static_cast<long long>(n), static_cast<long long>(k));
  };
  std::vector<double> alfs((p + 1) * (ph + 1), 0.0);
  auto alf = [&](std::size_t j, std::size_t i) -> double& { return alfs[j * (ph + 1) + i]; };
  alf(0, 0) = 1.0;
  alf(p, ph) = 1.0;
  for (std::size_t i = 1; i <= ph2; ++i) {
    const double inv = 1.0 / bc(ph, i);
    const std::size_t j_hi = (std::min)(p, i);
    for (std::size_t j = i > t ? i - t : 0; j <= j_hi; ++j) {
      alf(j, i) = inv * bc(p, j) * bc(t, i - j);
    }
  }
  for (std::size_t i = ph2 + 1; i < ph; ++i) {
    const std::size_t j_hi = (std::min)(p, i);
    for (std::size_t j = i > t ? i - t : 0; j <= j_hi; ++j) {
      alf(j, i) = alf(p - j, ph - i);
    }
  }
  for (std::size_t i = 0; i <= ph; ++i) {
    for (std::size_t r = 0; r < n_cols; ++r) {
      out(i, r) = static_cast<T>(0);
    }
    const std::size_t j_hi = (std::min)(p, i);
    for (std::size_t j = i > t ? i - t : 0; j <= j_hi; ++j) {
      const double coeff = alf(j, i);
      for (std::size_t r = 0; r < n_cols; ++r) {
        out(i, r) = static_cast<T>(static_cast<double>(out(i, r)) +
                                   coeff * static_cast<double>(ctrl(j, r)));
      }
    }
  }
}

/// Evaluate a 1D Bezier at one parameter value by de Casteljau, column by column.
///
/// The values `0` and `1` copy the first or last row. Each column runs the triangular reduction
/// in a workspace of type `T` with the arithmetic in `double` (the parameter is a `double`).
/// Parallel over columns.
///
/// Args:
///     ctrl (mdview<const T, 2>): Control points, extents `(p + 1, n_cols)`.
///     value (double): Parameter value in `[0, 1]`.
///     out (std::span<T>): Output of `n_cols` values.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
void slice_1d_kernel(mdview<const T, 2> ctrl, double value, std::span<T> out) {
  const std::size_t p = ctrl.extent(0) - 1;
  const std::size_t n_cols = ctrl.extent(1);
  const double u = value;
  const double one_minus_u = 1.0 - u;
  if (u == 0.0 || u == 1.0) {
    const std::size_t row = u == 0.0 ? 0 : p;
    parallel::parallel_for(n_cols, parallel::WorkEstimate{1.0},
                           [&](std::size_t begin, std::size_t end) {
                             for (std::size_t c = begin; c < end; ++c) {
                               out[c] = ctrl(row, c);
                             }
                           });
    return;
  }
  // Per column: `p (p + 1) / 2` reduction steps of three operations, plus the copy in.
  const double pd = static_cast<double>(p);
  const double ops = 1.5 * pd * (pd + 1.0) + pd + 1.0;
  parallel::parallel_for(n_cols, parallel::WorkEstimate{ops},
                         [&](std::size_t begin, std::size_t end) {
                           std::vector<T> d(p + 1);
                           for (std::size_t c = begin; c < end; ++c) {
                             for (std::size_t i = 0; i <= p; ++i) {
                               d[i] = ctrl(i, c);
                             }
                             for (std::size_t r = 1; r <= p; ++r) {
                               for (std::size_t i = 0; i + r <= p; ++i) {
                                 d[i] = static_cast<T>(one_minus_u * static_cast<double>(d[i]) +
                                                       u * static_cast<double>(d[i + 1]));
                               }
                             }
                             out[c] = d[0];
                           }
                         });
}

/// Split a 1D Bezier at a parameter value by one forward de Casteljau pass per column.
///
/// After level `r` the workspace holds `[d_0^r, d_1^(r-1), ..., d_p^0]`: its first entry is the
/// `r`-th control point of the left half `[0, value]`, and the final workspace is the right half
/// `[value, 1]`, both reparametrized to `[0, 1]`. Parallel over columns.
///
/// Args:
///     ctrl (mdview<const T, 2>): Control points, extents `(p + 1, n_cols)`.
///     value (double): Parameter value in `(0, 1)`.
///     out_left (mdview<T, 2>): Left half, extents `(p + 1, n_cols)`.
///     out_right (mdview<T, 2>): Right half, extents `(p + 1, n_cols)`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
void split_1d_kernel(mdview<const T, 2> ctrl, double value, mdview<T, 2> out_left,
                     mdview<T, 2> out_right) {
  const std::size_t p = ctrl.extent(0) - 1;
  const std::size_t n_cols = ctrl.extent(1);
  const double u = value;
  const double one_minus_u = 1.0 - u;
  const double pd = static_cast<double>(p);
  const double ops = 1.5 * pd * (pd + 1.0) + 3.0 * (pd + 1.0);
  // A chunk of columns runs its passes together: the workspace is `(p + 1, w)` with the columns
  // contiguous, so each step of a pass is a loop over `w` columns that vectorises (one column at
  // a time it would read `ctrl` with a stride of `n_cols`). Every column sees the same operations
  // in the same order as on its own.
  // A column writes its two halves, `2 (p + 1)` values: chunked by those bytes too, so a chunk is
  // wide enough for the column loop to vectorise (by work alone a chunk is about ten columns).
  parallel::parallel_for(n_cols, parallel::WorkEstimate{ops}, 2 * (p + 1) * sizeof(T),
                         [&](std::size_t begin, std::size_t end) {
                           const std::size_t w = end - begin;
                           std::vector<T> d((p + 1) * w);
                           for (std::size_t i = 0; i <= p; ++i) {
                             for (std::size_t c = 0; c < w; ++c) {
                               d[i * w + c] = ctrl(i, begin + c);
                             }
                           }
                           for (std::size_t c = 0; c < w; ++c) {
                             out_left(0, begin + c) = d[c];
                           }
                           for (std::size_t r = 1; r <= p; ++r) {
                             for (std::size_t i = 0; i + r <= p; ++i) {
                               T* di = d.data() + i * w;
                               const T* dn = di + w;
                               for (std::size_t c = 0; c < w; ++c) {
                                 di[c] = static_cast<T>(one_minus_u * static_cast<double>(di[c]) +
                                                        u * static_cast<double>(dn[c]));
                               }
                             }
                             for (std::size_t c = 0; c < w; ++c) {
                               out_left(r, begin + c) = d[c];
                             }
                           }
                           for (std::size_t i = 0; i <= p; ++i) {
                             for (std::size_t c = 0; c < w; ++c) {
                               out_right(i, begin + c) = d[i * w + c];
                             }
                           }
                         });
}

/// Restrict a 1D Bezier to `[lower, upper]`, reparametrized to `[0, 1]`, by two de Casteljau
/// passes per column; the order of the passes avoids dividing by a small number.
///
///   - `|upper| >= |lower - 1|`: left pass at `upper`, then right pass at `lower / upper`.
///   - otherwise: right pass at `lower`, then left pass at `(upper - lower) / (1 - lower)`.
///
/// Parallel over columns.
///
/// Args:
///     ctrl (mdview<const T, 2>): Control points, extents `(p + 1, n_cols)`.
///     lower (double): Left bound in `[0, 1)`.
///     upper (double): Right bound in `(0, 1]`.
///     out (mdview<T, 2>): Output, extents `(p + 1, n_cols)`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
void restrict_1d_kernel(mdview<const T, 2> ctrl, double lower, double upper, mdview<T, 2> out) {
  const std::size_t p = ctrl.extent(0) - 1;
  const std::size_t n_cols = ctrl.extent(1);
  using std::abs;
  const bool upper_first = abs(upper) >= abs(lower - 1.0);
  // Two reduction triangles of `p (p + 1) / 2` steps, three operations each, plus the copies.
  const double pd = static_cast<double>(p);
  const double ops = 3.0 * pd * (pd + 1.0) + 2.0 * (pd + 1.0);
  // The left pass (towards `tau`, rows from the end) and the right pass (rows from the start).
  auto left_pass = [p](std::vector<T>& d, double tau) {
    const double one_minus_tau = 1.0 - tau;
    for (std::size_t step = 1; step <= p; ++step) {
      for (std::size_t j = p; j >= step; --j) {
        d[j] = static_cast<T>(static_cast<double>(d[j]) * tau +
                              static_cast<double>(d[j - 1]) * one_minus_tau);
      }
    }
  };
  auto right_pass = [p](std::vector<T>& d, double tau) {
    const double one_minus_tau = 1.0 - tau;
    for (std::size_t step = 1; step <= p; ++step) {
      for (std::size_t j = 0; j + step <= p; ++j) {
        d[j] = static_cast<T>(static_cast<double>(d[j]) * one_minus_tau +
                              static_cast<double>(d[j + 1]) * tau);
      }
    }
  };
  parallel::parallel_for(n_cols, parallel::WorkEstimate{ops},
                         [&](std::size_t begin, std::size_t end) {
                           std::vector<T> d(p + 1);
                           for (std::size_t c = begin; c < end; ++c) {
                             for (std::size_t i = 0; i <= p; ++i) {
                               d[i] = ctrl(i, c);
                             }
                             if (upper_first) {
                               left_pass(d, upper);
                               right_pass(d, lower / upper);
                             } else {
                               right_pass(d, lower);
                               left_pass(d, (upper - lower) / (1.0 - lower));
                             }
                             for (std::size_t i = 0; i <= p; ++i) {
                               out(i, c) = d[i];
                             }
                           }
                         });
}

/// Multiply two scalar 1D Bezier curves of degrees `p` and `q` in the Bernstein basis:
///
///     c_k = 1 / C(p + q, k) sum_{i = max(0, k - q)}^{min(p, k)} C(p, i) C(q, k - i) a_i b_{k - i}.
///
/// The convolution accumulates each term in `double` and stores the partial sum in `T`; the
/// normalization by `C(p + q, k)` is parallel over `k`.
///
/// Args:
///     a (std::span<const T>): Control points of the first curve, `p + 1` values.
///     b (std::span<const T>): Control points of the second curve, `q + 1` values.
///     out (std::span<T>): Product control points, `p + q + 1` values; overwritten.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed), `p + q <= bincoeff_max_n`
///     included. For general use, call `bernstein_product` instead.
template <StorageFloat T>
void scalar_bernstein_product_1d_kernel(std::span<const T> a, std::span<const T> b,
                                        std::span<T> out) {
  const std::size_t p = a.size() - 1;
  const std::size_t q = b.size() - 1;
  const std::size_t r = p + q;
  auto bc = [](std::size_t n, std::size_t k) {
    return bincoeff(static_cast<long long>(n), static_cast<long long>(k));
  };
  std::vector<double> bq(q + 1);
  for (std::size_t j = 0; j <= q; ++j) {
    bq[j] = bc(q, j);
  }
  for (std::size_t k = 0; k <= r; ++k) {
    out[k] = static_cast<T>(0);
  }
  for (std::size_t i = 0; i <= p; ++i) {
    const double ai_scaled = static_cast<double>(a[i]) * bc(p, i);
    for (std::size_t j = 0; j <= q; ++j) {
      out[i + j] = static_cast<T>(static_cast<double>(out[i + j]) +
                                  ai_scaled * static_cast<double>(b[j]) * bq[j]);
    }
  }
  // One quotient and a binomial recurrence of `min(k, r - k)` steps per entry.
  const double ops = 2.0 + static_cast<double>(r);
  parallel::parallel_for(r + 1, parallel::WorkEstimate{ops},
                         [&](std::size_t begin, std::size_t end) {
                           for (std::size_t k = begin; k < end; ++k) {
                             out[k] = static_cast<T>(static_cast<double>(out[k]) / bc(r, k));
                           }
                         });
}

}  // namespace pantr::bezier::detail
