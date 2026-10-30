/// @file basis_kernels.hpp
/// @brief The B-spline basis at points: span search, Cox-de Boor values (Piegl and Tiller A2.2)
///        and derivatives (A2.3), point by point and over batches.
///
/// The knots are of type `K` and the points and outputs of type `P`, which may differ. Each
/// quantity is held in the type the Numba kernels this replaces held it in: the knot differences
/// in `K`, the values of A2.2 in `P` (A2.3 holds its values in `K` and stores them in `P`), and
/// every mixed operation in `std::common_type_t<K, P>`.
#pragma once

#include "pantr/basis/detail/tabulate_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/scalar.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace pantr::bspline::detail {

/// The clamped knot span of a point and the index of the first basis function non-zero there.
///
/// Attributes:
///     knot_id (std::size_t): Span index `k` with `knots[k] <= pt < knots[k + 1]`, clamped to the
///         in-domain spans `[degree, size - degree - 2]`.
///     first (std::int64_t): First supported basis function; for a non-periodic space clamped to
///         `num_basis - degree - 1` so the last point addresses the last `degree + 1` functions.
struct SpanAndFirst {
  std::size_t knot_id;
  std::int64_t first;
};

/// Locate `pt`'s span (`searchsorted(knots, pt, side="right") - 1`, a NaN point past the end) and
/// clamp it as `SpanAndFirst` states.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat K, StorageFloat P>
[[nodiscard]] SpanAndFirst span_and_first(std::span<const K> knots, int degree, bool periodic,
                                          long long num_basis, P pt) {
  using W = std::common_type_t<K, P>;
  const auto p = static_cast<W>(pt);
  const auto it = std::upper_bound(knots.begin(), knots.end(), p,
                                   [](W v, K k) { return v < static_cast<W>(k); });
  const auto raw = static_cast<std::ptrdiff_t>(it - knots.begin()) - 1;
  const auto max_id = static_cast<std::ptrdiff_t>(knots.size()) - degree - 2;
  std::ptrdiff_t id = (std::min)(raw, max_id);
  id = (std::max)(id, static_cast<std::ptrdiff_t>(degree));
  std::int64_t first = id - degree;
  if (!periodic) {
    first = (std::min<std::int64_t>)(first, num_basis - degree - 1);
  }
  return {static_cast<std::size_t>(id), first};
}

/// The `degree + 1` non-zero basis values at `pt` in span `knot_id` (A2.2).
///
/// The denominator is the sum of two non-negative knot differences and is dropped exactly when it
/// is zero (an empty span): the step multiplies `N[r] / denom` straight back by those two terms,
/// whose weights sum to one, so a small denominator costs accuracy in that ratio and never
/// boundedness, and the exact-zero test needs no tolerance and is invariant under shift and scale.
///
/// Args:
///     knots (const K*): Knot vector.
///     degree (int): Degree.
///     knot_id (std::size_t): Clamped span of `pt`.
///     pt (P): Point.
///     out (P*): `degree + 1` values.
///     left (K*): Scratch, `degree + 1` values.
///     right (K*): Scratch, `degree + 1` values.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `BsplineSpace1D::tabulate_basis` instead.
template <StorageFloat K, StorageFloat P>
void basis_funcs_point(const K* knots, int degree, std::size_t knot_id, P pt, P* out, K* left,
                       K* right) noexcept {
  using W = std::common_type_t<K, P>;
  const auto p = static_cast<W>(pt);
  out[0] = static_cast<P>(1);
  for (int j = 1; j <= degree; ++j) {
    const auto uj = static_cast<std::size_t>(j);
    left[j] = static_cast<K>(p - static_cast<W>(knots[knot_id + 1 - uj]));
    right[j] = static_cast<K>(static_cast<W>(knots[knot_id + uj]) - p);
    W saved = 0;
    for (int r = 0; r < j; ++r) {
      const K denom = right[r + 1] + left[j - r];
      const W temp = denom == static_cast<K>(0) ? static_cast<W>(0)
                                                : static_cast<W>(out[r]) / static_cast<W>(denom);
      out[r] = static_cast<P>(saved + static_cast<W>(right[r + 1]) * temp);
      saved = static_cast<W>(left[j - r]) * temp;
    }
    out[j] = static_cast<P>(saved);
  }
}

/// Scratch the derivative recurrence needs at one point, sized for a degree and an order.
///
/// Attributes:
///     ndu (std::vector<K>): `(degree + 1)^2` knot differences and values.
///     left (std::vector<K>): `degree + 1` values.
///     right (std::vector<K>): `degree + 1` values.
///     a (std::vector<K>): `2 * (n_deriv + 1)` coefficients.
template <StorageFloat K>
struct DerivScratch {
  std::vector<K> ndu;
  std::vector<K> left;
  std::vector<K> right;
  std::vector<K> a;
  DerivScratch(int degree, int n_deriv)
      : ndu(static_cast<std::size_t>(degree + 1) * static_cast<std::size_t>(degree + 1)),
        left(static_cast<std::size_t>(degree) + 1),
        right(static_cast<std::size_t>(degree) + 1),
        a(2 * (static_cast<std::size_t>(n_deriv) + 1)) {}
};

/// The derivatives of orders `0..n_deriv` of the `degree + 1` non-zero basis functions at `pt`
/// (A2.3), into `out` (`(n_deriv + 1) x (degree + 1)`, row-major). Row 0 equals
/// `basis_funcs_point`'s result; rows past the degree are zero.
///
/// The knot differences, values and coefficients are held in `K`; each row is scaled by the
/// integer `degree! / (degree - k)!` in double (as an integer factor times a float32 promotes to
/// double), stored in `P`. The `ndu` denominators of the value triangle are guarded as in
/// `basis_funcs_point`; the three divisions of the derivative rows are not (as in A2.3), and one
/// whose divisor is zero, which a basis function of empty support produces (a knot repeated more
/// than `degree + 1` times within the point's support), is reported rather than performed.
///
/// Returns:
///     bool: False when a derivative row met a zero divisor; `out` is then incomplete.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `BsplineSpace1D::tabulate_basis_derivatives` instead.
template <StorageFloat K, StorageFloat P>
[[nodiscard]] bool basis_derivs_point(const K* knots, int degree, int n_deriv, std::size_t knot_id,
                                      P pt, P* out, DerivScratch<K>& s) noexcept {
  using W = std::common_type_t<K, P>;
  const auto ord = static_cast<std::size_t>(degree) + 1;
  const auto nd1 = static_cast<std::size_t>(n_deriv) + 1;
  K* ndu = s.ndu.data();
  K* left = s.left.data();
  K* right = s.right.data();
  K* a = s.a.data();
  const auto NDU = [&](int r, int c) -> K& {
    return ndu[static_cast<std::size_t>(r) * ord + static_cast<std::size_t>(c)];
  };
  const auto A = [&](int r, int c) -> K& {
    return a[static_cast<std::size_t>(r) * nd1 + static_cast<std::size_t>(c)];
  };
  const auto out_at = [&](int k, int j) -> P& {
    return out[static_cast<std::size_t>(k) * ord + static_cast<std::size_t>(j)];
  };
  std::fill(s.ndu.begin(), s.ndu.end(), K{0});
  std::fill(s.a.begin(), s.a.end(), K{0});
  const auto p = static_cast<W>(pt);
  const K zero = 0;
  NDU(0, 0) = 1;
  for (int j = 1; j <= degree; ++j) {
    const auto uj = static_cast<std::size_t>(j);
    left[j] = static_cast<K>(p - static_cast<W>(knots[knot_id + 1 - uj]));
    right[j] = static_cast<K>(static_cast<W>(knots[knot_id + uj]) - p);
    K saved = 0;
    for (int r = 0; r < j; ++r) {
      NDU(j, r) = right[r + 1] + left[j - r];
      const K denom = NDU(j, r);
      const K temp = denom == zero ? zero : NDU(r, j - 1) / denom;
      NDU(r, j) = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    NDU(j, j) = saved;
  }
  for (int j = 0; j <= degree; ++j) {
    out_at(0, j) = static_cast<P>(NDU(j, degree));
  }
  for (int r = 0; r <= degree; ++r) {
    int s1 = 0;
    int s2 = 1;
    A(0, 0) = 1;
    for (int k = 1; k <= n_deriv; ++k) {
      K d = 0;
      const int rk = r - k;
      const int pk = degree - k;
      if (r >= k) {
        if (NDU(pk + 1, rk) == zero) {
          return false;
        }
        A(s2, 0) = A(s1, 0) / NDU(pk + 1, rk);
        d = A(s2, 0) * NDU(rk, pk);
      }
      const int j1 = rk >= -1 ? 1 : -rk;
      const int j2 = (r - 1) <= pk ? k - 1 : degree - r;
      for (int jj = j1; jj <= j2; ++jj) {
        if (NDU(pk + 1, rk + jj) == zero) {
          return false;
        }
        A(s2, jj) = (A(s1, jj) - A(s1, jj - 1)) / NDU(pk + 1, rk + jj);
        d += A(s2, jj) * NDU(rk + jj, pk);
      }
      if (r <= pk) {
        if (NDU(pk + 1, r) == zero) {
          return false;
        }
        A(s2, k) = -A(s1, k - 1) / NDU(pk + 1, r);
        d += A(s2, k) * NDU(r, pk);
      }
      out_at(k, r) = static_cast<P>(d);
      std::swap(s1, s2);
    }
  }
  // Row k times degree! / (degree - k)!, a 64-bit integer product applied in double; zero from
  // k = degree + 1 on. The product wraps as the Numba kernel's did once it exceeds 2^63, which
  // degree! / (degree - k)! can from degree 21 on (21! > 2^63): rows of order 21 and above are then
  // wrong, as they were before.
  std::uint64_t fac = static_cast<std::uint64_t>(static_cast<std::int64_t>(degree));
  for (int k = 1; k <= n_deriv; ++k) {
    const auto f = static_cast<double>(static_cast<std::int64_t>(fac));
    for (int j = 0; j <= degree; ++j) {
      out_at(k, j) = static_cast<P>(static_cast<double>(out_at(k, j)) * f);
    }
    fac *= static_cast<std::uint64_t>(static_cast<std::int64_t>(degree - k));
  }
  return true;
}

/// Operations one point of the value recurrence costs: about `2 (p + 1)^2` for the triangle plus
/// the span search's `log2` of the knot count, taken as an admitted constant of 16 (a binary
/// search over up to 65536 knots), not derived from the vector at hand.
inline double basis_ops_per_point(int degree) {
  const double ord = static_cast<double>(degree) + 1.0;
  return 2.0 * ord * ord + 16.0;
}

/// Operations one point of the derivative recurrence costs: the `(p + 1)^2` triangle and
/// `(p + 1)(n_deriv + 1)` rows of `O(p + 1)` work, about `(p + 1)^2 (2 n_deriv + 3)` with the
/// constant factors an admitted estimate, plus the span search as in `basis_ops_per_point`.
inline double derivs_ops_per_point(int degree, int n_deriv) {
  const double ord = static_cast<double>(degree) + 1.0;
  return ord * ord * (2.0 * static_cast<double>(n_deriv) + 3.0) + 16.0;
}

/// The values and first indices at every point (general knots), over the pool where it pays.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat K, StorageFloat P>
void tabulate_basis_kernel(std::span<const K> knots, int degree, bool periodic, long long num_basis,
                           std::span<const P> pts, P* out, std::int64_t* first) {
  const std::size_t ord = static_cast<std::size_t>(degree) + 1;
  pantr::basis::detail::for_each_chunk(
      pts.size(), basis_ops_per_point(degree), sizeof(P) * ord + sizeof(std::int64_t),
      [&](std::size_t begin, std::size_t end) {
        std::vector<K> left(ord);
        std::vector<K> right(ord);
        for (std::size_t i = begin; i < end; ++i) {
          const SpanAndFirst sf = span_and_first<K, P>(knots, degree, periodic, num_basis, pts[i]);
          first[i] = sf.first;
          basis_funcs_point<K, P>(knots.data(), degree, sf.knot_id, pts[i], out + i * ord,
                                  left.data(), right.data());
        }
      });
}

/// The derivatives and first indices at every point (general knots), over the pool where it pays.
///
/// Returns:
///     bool: False when a point's derivative rows met a zero divisor (see `basis_derivs_point`);
///     the output is then incomplete.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat K, StorageFloat P>
[[nodiscard]] bool tabulate_derivs_kernel(std::span<const K> knots, int degree, bool periodic,
                                          long long num_basis, int n_deriv, std::span<const P> pts,
                                          P* out, std::int64_t* first) {
  const std::size_t block =
      (static_cast<std::size_t>(degree) + 1) * (static_cast<std::size_t>(n_deriv) + 1);
  std::atomic<bool> complete{true};
  pantr::basis::detail::for_each_chunk(
      pts.size(), derivs_ops_per_point(degree, n_deriv), sizeof(P) * block + sizeof(std::int64_t),
      [&](std::size_t begin, std::size_t end) {
        DerivScratch<K> scratch(degree, n_deriv);
        for (std::size_t i = begin; i < end; ++i) {
          const SpanAndFirst sf = span_and_first<K, P>(knots, degree, periodic, num_basis, pts[i]);
          first[i] = sf.first;
          if (!basis_derivs_point<K, P>(knots.data(), degree, n_deriv, sf.knot_id, pts[i],
                                        out + i * block, scratch)) {
            complete.store(false, std::memory_order_relaxed);
            return;
          }
        }
      });
  return complete.load(std::memory_order_relaxed);
}

}  // namespace pantr::bspline::detail
