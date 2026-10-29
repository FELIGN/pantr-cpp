/// @file reduce.hpp
/// @brief Degree reduction and minimisation of a `Bezier`.
///
/// Reduction applies a dense `(q + 1, p + 1)` operator `R` per direction, depending only on the
/// degree pair: the reduced control points are `R c`. The operator is the `L^2([0, 1])`-optimal
/// one that keeps both end values; it is assembled in exact rational arithmetic and rounded to
/// `double` once, entry by entry, to nearest (as `float(Fraction)` rounded in the previous
/// release, so the operators are the same bits). The exact assembly is what makes the operator
/// usable at high degree: the interior Gram block it inverts has a condition number of order
/// `1e10` at `q = 19`, so a `double` solve would lose some ten digits.
///
/// Errors are measured in the space the caller budgets in: for a polynomial Bézier the exact
/// Bernstein-Gram `L^2` norm of the coefficient difference; for a rational one the relative
/// deviation of the projected map, estimated on a Gauss-Legendre grid.
#pragma once

#include "pantr/basis/bernstein.hpp"
#include "pantr/bezier/bezier.hpp"
#include "pantr/bezier/detail/along_axis.hpp"
#include "pantr/bezier/detail/kernels_1d.hpp"
#include "pantr/bezier/elevate.hpp"
#include "pantr/config.hpp"
#include "pantr/core/binomial.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/exact_rational.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/quad/rules_1d.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <pthread.h>
#endif

namespace pantr::bezier {

/// Default relative tolerance of `minimize_degree`, in units of the dtype's epsilon: three digits
/// above the round-off floor of a reduce-then-elevate round trip (each accumulates `O(p)`
/// roundings), which an exactly reducible curve reaches: a small multiple of `eps`, for the
/// polynomial measure and for the rational (projected) one alike. Kept from the previous release;
/// an admitted factor.
inline constexpr double auto_reduction_tol_factor = 1.0e3;

/// Largest degree whose reduction operators the process table keeps; past it an operator is
/// computed per call. A memory bound: with every interpolating key up to it present the table
/// holds `sum_p 8 (p + 1) p (p + 1) / 2` bytes, under 20 MB, and the change-of-basis tables use
/// the same degree. An admitted bound, not a derived one.
inline constexpr long long reduction_cache_max_degree = 64;

namespace detail {

using exact::Integer;
using exact::Rational;

/// A matrix of exact rationals, row-major.
using ExactMatrix = std::vector<std::vector<Rational>>;

/// `C(n, k)` as an exact integer, for `0 <= k <= n` (unchecked: every caller is in range).
[[nodiscard]] inline Integer comb_exact(long long n, long long k) {
  k = (std::min)(k, n - k);
  Integer r(1);
  for (long long i = 1; i <= k; ++i) {
    r *= n - k + i;
    r /= i;
  }
  return r;
}

/// Round an exact rational to the nearest double, ties to even (Python's `float(Fraction)`).
[[nodiscard]] inline double round_to_double(const Rational& q) {
  const Integer num = boost::multiprecision::numerator(q);
  const Integer den = boost::multiprecision::denominator(q);
  if (num == 0) {
    return 0.0;
  }
  if (num < 0) {
    return -exact::correctly_rounded_quotient(-num, den);
  }
  return exact::correctly_rounded_quotient(num, den);
}

/// The degree-`n` Bernstein Gram matrix in exact rationals: `C(n,i) C(n,j) / (C(2n,i+j) (2n+1))`
/// (Farouki and Rajan, CAGD 5, 1988).
[[nodiscard]] inline ExactMatrix bernstein_gram_exact(long long n) {
  ExactMatrix g(static_cast<std::size_t>(n + 1),
                std::vector<Rational>(static_cast<std::size_t>(n + 1)));
  for (long long i = 0; i <= n; ++i) {
    for (long long j = 0; j <= n; ++j) {
      g[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = Rational(
          comb_exact(n, i) * comb_exact(n, j), comb_exact(2 * n, i + j) * Integer(2 * n + 1));
    }
  }
  return g;
}

/// The elevation matrix from degree `q` to `q + t` in exact rationals: `M_ij = C(q,j) C(t,i-j) /
/// C(q+t,i)` for `max(0, i - t) <= j <= min(q, i)`, zero elsewhere.
[[nodiscard]] inline ExactMatrix elevation_matrix_exact(long long q, long long t) {
  const long long p = q + t;
  ExactMatrix m(static_cast<std::size_t>(p + 1),
                std::vector<Rational>(static_cast<std::size_t>(q + 1)));
  for (long long i = 0; i <= p; ++i) {
    for (long long j = (std::max)(0LL, i - t); j <= (std::min)(q, i); ++j) {
      m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
          Rational(comb_exact(q, j) * comb_exact(t, i - j), comb_exact(p, i));
    }
  }
  return m;
}

/// Solve `a x = rhs` exactly by Gauss-Jordan elimination, stepping over a zero pivot by taking the
/// first row of largest magnitude (exact arithmetic needs no pivoting for stability).
///
/// Args:
///     a (ExactMatrix): Square, non-singular.
///     rhs (ExactMatrix): Right-hand sides, one row per row of `a`.
///
/// Returns:
///     ExactMatrix: The solution, the shape of `rhs`.
[[nodiscard]] inline ExactMatrix solve_exact(ExactMatrix a, ExactMatrix rhs) {
  const std::size_t n = a.size();
  for (std::size_t col = 0; col < n; ++col) {
    std::size_t pivot = col;
    for (std::size_t row = col + 1; row < n; ++row) {
      if (abs(a[row][col]) > abs(a[pivot][col])) {
        pivot = row;
      }
    }
    std::swap(a[col], a[pivot]);
    std::swap(rhs[col], rhs[pivot]);
    if (a[col][col] == 0) {
      throw value_error("solve_exact: singular matrix");
    }
    const Rational inv = Rational(1) / a[col][col];
    for (auto& v : a[col]) {
      v *= inv;
    }
    for (auto& v : rhs[col]) {
      v *= inv;
    }
    for (std::size_t row = 0; row < n; ++row) {
      if (row == col || a[row][col] == 0) {
        continue;
      }
      const Rational factor = a[row][col];
      for (std::size_t k = 0; k < n; ++k) {
        a[row][k] -= factor * a[col][k];
      }
      for (std::size_t k = 0; k < rhs[row].size(); ++k) {
        rhs[row][k] -= factor * rhs[col][k];
      }
    }
  }
  return rhs;
}

/// `(M^T G_p)` restricted to rows `[r0, r1)`: row `i` is `sum_k M_ki G_p[k, :]`.
[[nodiscard]] inline ExactMatrix projected_rows(const ExactMatrix& elevation,
                                                const ExactMatrix& gram_p, std::size_t r0,
                                                std::size_t r1) {
  const std::size_t np1 = gram_p.size();
  ExactMatrix out(r1 - r0, std::vector<Rational>(np1));
  for (std::size_t i = r0; i < r1; ++i) {
    for (std::size_t k = 0; k < np1; ++k) {
      if (elevation[k][i] == 0) {
        continue;
      }
      for (std::size_t j = 0; j < np1; ++j) {
        out[i - r0][j] += elevation[k][i] * gram_p[k][j];
      }
    }
  }
  return out;
}

[[nodiscard]] inline NdArray<double> round_matrix(const ExactMatrix& m) {
  const std::size_t cols = m.empty() ? 0 : m[0].size();
  NdArray<double> out({m.size(), cols}, for_overwrite);
  for (std::size_t i = 0; i < m.size(); ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      out.data()[i * cols + j] = round_to_double(m[i][j]);
    }
  }
  return out;
}

inline void check_degree_pair(long long degree, long long decrement) {
  if (degree < 1 || decrement < 1 || decrement > degree) {
    throw value_error("A reduction operator needs 1 <= decrement <= degree, got degree " +
                      std::to_string(degree) + " and decrement " + std::to_string(decrement) + ".");
  }
}

}  // namespace detail

/// The unconstrained `L^2` projection from degree `p` onto degree `q = p - t`, rounded once.
///
/// Solves the normal equations `G_q R = M^T G_p` (`M` the elevation matrix, so `M^T G_p M = G_q`).
///
/// Args:
///     degree (long long): `p >= 1`.
///     decrement (long long): `1 <= t <= p`.
///
/// Returns:
///     NdArray<double>: The `(q + 1, p + 1)` operator.
///
/// Raises:
///     value_error: The degree pair is out of range.
[[nodiscard]] inline NdArray<double> l2_reduction_operator(long long degree, long long decrement) {
  detail::check_degree_pair(degree, decrement);
  const long long q = degree - decrement;
  const detail::ExactMatrix elevation = detail::elevation_matrix_exact(q, decrement);
  const detail::ExactMatrix gram_p = detail::bernstein_gram_exact(degree);
  detail::ExactMatrix rhs =
      detail::projected_rows(elevation, gram_p, 0, static_cast<std::size_t>(q + 1));
  return detail::round_matrix(detail::solve_exact(detail::bernstein_gram_exact(q), std::move(rhs)));
}

/// The endpoint-interpolating `L^2`-optimal reduction operator from degree `p` to `p - t`.
///
/// Minimises the same distance as `l2_reduction_operator` with `q_0 = c_0` and `q_q = c_p`: the
/// interior coefficients solve the interior block of `G_q` against `(M^T G_p)` minus the end
/// columns. A degree-0 target cannot honour two conditions, so `t == p` returns the unconstrained
/// projection (the mean of the coefficients).
///
/// Args:
///     degree (long long): `p >= 1`.
///     decrement (long long): `1 <= t <= p`.
///
/// Returns:
///     NdArray<double>: The `(q + 1, p + 1)` operator; its first and last rows are unit vectors.
///
/// Raises:
///     value_error: The degree pair is out of range.
[[nodiscard]] inline NdArray<double> interpolating_reduction_operator(long long degree,
                                                                      long long decrement) {
  detail::check_degree_pair(degree, decrement);
  const long long p = degree;
  const long long q = p - decrement;
  if (q == 0) {
    return l2_reduction_operator(degree, decrement);
  }
  const auto pu = static_cast<std::size_t>(p);
  const auto qu = static_cast<std::size_t>(q);
  detail::ExactMatrix rows(qu + 1, std::vector<detail::Rational>(pu + 1));
  rows[0][0] = 1;
  rows[qu][pu] = 1;
  if (q >= 2) {
    const detail::ExactMatrix elevation = detail::elevation_matrix_exact(q, decrement);
    const detail::ExactMatrix gram_p = detail::bernstein_gram_exact(p);
    const detail::ExactMatrix gram_q = detail::bernstein_gram_exact(q);
    detail::ExactMatrix interior(qu - 1, std::vector<detail::Rational>(qu - 1));
    for (std::size_t i = 0; i + 1 < qu; ++i) {
      for (std::size_t j = 0; j + 1 < qu; ++j) {
        interior[i][j] = gram_q[i + 1][j + 1];
      }
    }
    detail::ExactMatrix right = detail::projected_rows(elevation, gram_p, 1, qu);
    for (std::size_t i = 0; i + 1 < qu; ++i) {
      right[i][0] -= gram_q[1 + i][0];
      right[i][pu] -= gram_q[1 + i][qu];
    }
    const detail::ExactMatrix solved = detail::solve_exact(std::move(interior), std::move(right));
    for (std::size_t i = 0; i + 1 < qu; ++i) {
      rows[1 + i] = solved[i];
    }
  }
  return detail::round_matrix(rows);
}

/// The elevation matrix from degree `q` to `q + t`, each entry rounded once to nearest.
///
/// Args:
///     degree (long long): `q >= 0`.
///     increment (long long): `t >= 1`.
///
/// Returns:
///     NdArray<double>: The `(q + t + 1, q + 1)` matrix.
///
/// Raises:
///     value_error: `degree < 0` or `increment < 1`.
[[nodiscard]] inline NdArray<double> elevation_matrix(long long degree, long long increment) {
  if (degree < 0 || increment < 1) {
    throw value_error("An elevation matrix needs degree >= 0 and increment >= 1.");
  }
  return detail::round_matrix(detail::elevation_matrix_exact(degree, increment));
}

/// The two reduction operators a process table holds.
enum class ReductionKind : int {
  interpolating = 0,  ///< `interpolating_reduction_operator`.
  l2 = 1,             ///< `l2_reduction_operator`.
};

namespace detail {

/// Guards the reduction-operator table; held across `fork` as the change-of-basis table's mutex
/// is (design/threading.md, fork safety).
inline constinit std::mutex reduction_table_mutex;

/// The table's entries, keyed on `(kind, degree, decrement)`.
inline std::map<std::tuple<int, long long, long long>, std::shared_ptr<const NdArray<double>>>
    reduction_table_entries;

#if !defined(_WIN32)
/// Registration of the table mutex's fork handlers; nonzero means it failed.
inline const int reduction_table_atfork_status =
    ::pthread_atfork([] { reduction_table_mutex.lock(); }, [] { reduction_table_mutex.unlock(); },
                     [] { reduction_table_mutex.unlock(); });
#endif

inline void check_reduction_table_atfork() {
#if !defined(_WIN32)
  if (reduction_table_atfork_status != 0) {
    throw std::system_error(reduction_table_atfork_status, std::generic_category(),
                            "pthread_atfork: the reduction-operator table's fork handler could "
                            "not be registered");
  }
#endif
}

}  // namespace detail

/// Get a reduction operator from the process table, computing it on first use.
///
/// Every call with the same key returns the same immutable matrix (one allocation per key per
/// program image: each shared object built from these headers with hidden symbols, such as each
/// Python extension module, has its own table) for degrees up to `reduction_cache_max_degree`; past
/// it each call computes a fresh one. The matrix is computed outside the lock; when two threads
/// compute one key at once, the first stored wins.
///
/// Args:
///     kind (ReductionKind): Which operator.
///     degree (long long): `p >= 1`.
///     decrement (long long): `1 <= t <= p`.
///
/// Returns:
///     std::shared_ptr<const NdArray<double>>: The operator.
///
/// Raises:
///     value_error: The degree pair is out of range.
///     std::system_error: The table's fork handler could not be registered (POSIX).
[[nodiscard]] inline std::shared_ptr<const NdArray<double>> cached_reduction_operator(
    ReductionKind kind, long long degree, long long decrement) {
  detail::check_degree_pair(degree, decrement);
  detail::check_reduction_table_atfork();
  const std::tuple<int, long long, long long> key{static_cast<int>(kind), degree, decrement};
  const bool keep = degree <= reduction_cache_max_degree;
  if (keep) {
    const std::lock_guard<std::mutex> lock(detail::reduction_table_mutex);
    const auto it = detail::reduction_table_entries.find(key);
    if (it != detail::reduction_table_entries.end()) {
      return it->second;
    }
  }
  auto fresh = std::make_shared<const NdArray<double>>(
      kind == ReductionKind::interpolating ? interpolating_reduction_operator(degree, decrement)
                                           : l2_reduction_operator(degree, decrement));
  if (!keep) {
    return fresh;
  }
  const std::lock_guard<std::mutex> lock(detail::reduction_table_mutex);
  return detail::reduction_table_entries.try_emplace(key, std::move(fresh)).first->second;
}

/// Get the number of operators the process table holds.
[[nodiscard]] inline std::size_t reduction_cache_size() {
  const std::lock_guard<std::mutex> lock(detail::reduction_table_mutex);
  return detail::reduction_table_entries.size();
}

/// Empty the process table (operators already handed out stay valid).
inline void reduction_cache_clear() {
  const std::lock_guard<std::mutex> lock(detail::reduction_table_mutex);
  detail::reduction_table_entries.clear();
}

namespace detail {

/// Apply `op` (`(q + 1, p + 1)`) to the columns of `ctrl` (`(p + 1, n_cols)`): `out = op ctrl`,
/// accumulated in `double` and rounded to `T` once. Each product is a statement of its own, so no
/// `fma` is contracted: the unit rows of the interpolating operator then reproduce their inputs
/// bit for bit whatever the build, for finite inputs: a `-0.0` comes back as `+0.0` (the sum starts
/// at `+0.0`, as the previous release's did), and an infinite input makes every output it meets NaN
/// (`0 * inf`).
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] NdArray<T> apply_operator(const NdArray<double>& op, mdview<const T, 2> ctrl) {
  const std::size_t n_out = op.shape()[0];
  const std::size_t n_in = op.shape()[1];
  const std::size_t cols = ctrl.extent(1);
  NdArray<T> out({n_out, cols}, for_overwrite);
  const double* r = op.data();
  for (std::size_t i = 0; i < n_out; ++i) {
    for (std::size_t c = 0; c < cols; ++c) {
      double acc = 0.0;
      for (std::size_t j = 0; j < n_in; ++j) {
        const double term = r[i * n_in + j] * static_cast<double>(ctrl(j, c));
        acc += term;
      }
      out.data()[i * cols + c] = static_cast<T>(acc);
    }
  }
  return out;
}

/// Apply an operator along one axis of a row-major control net.
template <StorageFloat T>
[[nodiscard]] NdArray<T> apply_along_axis(std::span<const T> data,
                                          std::span<const std::size_t> shape, std::size_t axis,
                                          const NdArray<double>& op) {
  const NdArray<T> flat = flatten_along_axis<T>(data, shape, axis);
  const NdArray<T> reduced = apply_operator<T>(op, flat.template view<2>());
  return unflatten_along_axis<T>(reduced.template view<2>(), shape, axis);
}

/// Validate decrements against a Bézier (the previous release's messages and order).
template <StorageFloat T>
void check_decrements(const Bezier<T>& b, std::span<const long long> decrements) {
  if (decrements.size() != b.dim()) {
    throw value_error("Number of degree decrements (" + std::to_string(decrements.size()) +
                      ") must match dimension (" + std::to_string(b.dim()) + ").");
  }
  bool any = false;
  for (const long long d : decrements) {
    if (d < 0) {
      throw value_error("Degree decrements must be non-negative.");
    }
    any = any || d > 0;
  }
  if (!any) {
    throw value_error("At least one degree decrement must be positive.");
  }
  for (std::size_t d = 0; d < b.dim(); ++d) {
    if (decrements[d] > static_cast<long long>(b.degree(d))) {
      throw value_error("Degree decrement (" + std::to_string(decrements[d]) + ") in direction " +
                        std::to_string(d) + " exceeds current degree (" +
                        std::to_string(b.degree(d)) + ").");
    }
  }
}

/// Apply the interpolating operators, one direction at a time, without judging the result.
template <StorageFloat T>
[[nodiscard]] NdArray<T> apply_reductions(const Bezier<T>& b,
                                          std::span<const long long> decrements) {
  std::vector<std::size_t> shape(b.shape().begin(), b.shape().end());
  NdArray<T> current(std::vector<std::size_t>(shape), for_overwrite);
  std::copy(b.control_points().begin(), b.control_points().end(), current.data());
  for (std::size_t d = 0; d < b.dim(); ++d) {
    if (decrements[d] == 0) {
      continue;
    }
    const auto op = cached_reduction_operator(ReductionKind::interpolating,
                                              static_cast<long long>(shape[d] - 1), decrements[d]);
    current = apply_along_axis<T>(current.flat(), shape, d, *op);
    shape[d] = current.shape()[d];
  }
  return current;
}

/// Format a value as Python's `f"{x:.3e}"`.
[[nodiscard]] inline std::string format_3e(double x) {
  if (std::isnan(x)) {
    return "nan";
  }
  if (std::isinf(x)) {
    return x > 0 ? "inf" : "-inf";
  }
  // `std::to_chars` rather than `printf`: no locale (a comma as decimal point), as format.hpp.
  char buf[64];
  const auto res = std::to_chars(buf, buf + sizeof(buf), x, std::chars_format::scientific, 3);
  std::string out(buf, res.ptr);
  // C++ writes the exponent with at least two digits, as Python does: `1.000e-05`.
  return out;
}

/// Refuse a rational net whose smallest control weight is not strictly positive (NaN included):
/// the denominator can then vanish on the domain, so the net is not a NURBS. The reduction
/// operator is not a convex combination (at degree 3 it has an entry of -0.25) and acts on the
/// weight column too.
template <StorageFloat T>
void refuse_inverted_weight(const NdArray<T>& ctrl, std::string_view remedy) {
  const std::size_t cp = ctrl.shape().back();
  double smallest = std::numeric_limits<double>::infinity();
  for (std::size_t i = cp - 1; i < ctrl.size(); i += cp) {
    const auto w = static_cast<double>(ctrl.data()[i]);
    if (std::isnan(w)) {
      smallest = w;
      break;
    }
    smallest = (std::min)(smallest, w);
  }
  if (smallest > 0.0) {
    return;
  }
  throw value_error("Degree reduction drove a control weight to " + format_3e(smallest) +
                    ", which is not strictly positive, so the reduced rational spline is not a "
                    "valid NURBS: its denominator can vanish on the domain. The reduction "
                    "operator is not a convex combination and acts on the weight column too. " +
                    std::string(remedy));
}

}  // namespace detail

/// Reduce the degree of a Bézier, keeping its values at the ends of the domain.
///
/// Among the polynomials of the lower degree that agree with this one at both ends of each
/// reduced direction, the result is the closest in `L^2([0, 1]^dim)`; the end values are
/// reproduced bit for bit when the control points are finite (a `-0.0` as `+0.0`). A reduction to
/// degree 0 returns the plain projection (the mean).
///
/// Args:
///     b (const Bezier<T>&): The Bézier.
///     decrements (std::span<const long long>): Degrees to remove per direction, `dim` values,
///         each in `[0, degree]`, at least one positive.
///
/// Returns:
///     Bezier<T>: The reduced Bézier.
///
/// Raises:
///     value_error: Invalid decrements (see `detail::check_decrements`), or a rational result
///         with a control weight that is not strictly positive.
template <StorageFloat T>
[[nodiscard]] Bezier<T> reduce_degree(const Bezier<T>& b, std::span<const long long> decrements) {
  detail::check_decrements(b, decrements);
  NdArray<T> reduced = detail::apply_reductions(b, decrements);
  if (b.rational()) {
    detail::refuse_inverted_weight(reduced,
                                   "Use minimize_degree, which reduces only as far as it can "
                                   "without changing the sign of the weight function.");
  }
  return Bezier<T>(std::move(reduced), b.rational());
}

namespace detail {

/// The degree-`n` Bernstein Gram matrix in `double`: the exact binomials rounded once, then
/// `C(n,i) C(n,j) / C(2n,i+j) / (2n+1)` in floating point (the previous release's formula).
[[nodiscard]] inline NdArray<double> bernstein_gram(std::size_t n) {
  std::vector<double> bn(n + 1);
  std::vector<double> b2n(2 * n + 1);
  const auto to_d = [](const Integer& v) {
    return exact::correctly_rounded_quotient(v, Integer(1));
  };
  for (std::size_t i = 0; i <= n; ++i) {
    bn[i] = to_d(comb_exact(static_cast<long long>(n), static_cast<long long>(i)));
  }
  for (std::size_t k = 0; k <= 2 * n; ++k) {
    b2n[k] = to_d(comb_exact(2 * static_cast<long long>(n), static_cast<long long>(k)));
  }
  NdArray<double> g({n + 1, n + 1}, for_overwrite);
  const double scale = 2.0 * static_cast<double>(n) + 1.0;
  for (std::size_t i = 0; i <= n; ++i) {
    for (std::size_t j = 0; j <= n; ++j) {
      g.data()[i * (n + 1) + j] = bn[i] * bn[j] / b2n[i + j] / scale;
    }
  }
  return g;
}

/// Contract `mat` (`(m, shape[axis])`) against one axis of a row-major `double` array.
[[nodiscard]] inline NdArray<double> contract_axis(const NdArray<double>& mat,
                                                   std::span<const double> data,
                                                   std::span<const std::size_t> shape,
                                                   std::size_t axis) {
  const NdArray<double> flat = flatten_along_axis<double>(data, shape, axis);
  const std::size_t m = mat.shape()[0];
  const std::size_t k = mat.shape()[1];
  const std::size_t cols = flat.shape()[1];
  NdArray<double> prod({m, cols});
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < k; ++j) {
      const double a = mat.data()[i * k + j];
      for (std::size_t c = 0; c < cols; ++c) {
        prod.data()[i * cols + c] += a * flat.data()[j * cols + c];
      }
    }
  }
  return unflatten_along_axis<double>(prod.template view<2>(), shape, axis);
}

/// The squared `L^2([0, 1]^d)` norm of a tensor Bernstein polynomial: `c^T (G_0 x ... x G_{d-1})
/// c`, contracting each univariate Gram matrix along its axis; the absolute value guards a tiny
/// negative from round-off of a non-negative quantity.
///
/// Args:
///     coeffs (std::span<const double>): Coefficients, row-major over `shape`.
///     shape (std::span<const std::size_t>): Orders per direction.
///
/// Returns:
///     double: The squared norm.
[[nodiscard]] inline double squared_l2_norm(std::span<const double> coeffs,
                                            std::span<const std::size_t> shape) {
  NdArray<double> gc(std::vector<std::size_t>(shape.begin(), shape.end()), for_overwrite);
  std::copy(coeffs.begin(), coeffs.end(), gc.data());
  for (std::size_t axis = 0; axis < shape.size(); ++axis) {
    gc = contract_axis(bernstein_gram(shape[axis] - 1), gc.flat(), shape, axis);
  }
  double s = 0.0;
  for (std::size_t i = 0; i < coeffs.size(); ++i) {
    s += coeffs[i] * gc.data()[i];
  }
  return std::abs(s);
}

/// Sum over rank components of the squared norms of a control net `(*orders, n_comp)`.
template <class V>
[[nodiscard]] double total_squared_norm(std::span<const V> data,
                                        std::span<const std::size_t> shape) {
  const std::size_t n_comp = shape.back();
  const std::span<const std::size_t> orders = shape.first(shape.size() - 1);
  const std::size_t n = data.size() / n_comp;
  std::vector<double> comp(n);
  double total = 0.0;
  for (std::size_t r = 0; r < n_comp; ++r) {
    for (std::size_t i = 0; i < n; ++i) {
      comp[i] = static_cast<double>(data[i * n_comp + r]);
    }
    total += squared_l2_norm(comp, orders);
  }
  return total;
}

/// Gauss-Legendre nodes per direction grading a rational reduction: `2p + 2`, the smallest rule
/// exact on the numerator `w C~ - w~ C` of the projected difference (degree `4p`) plus one node.
/// The `1 / (w w~)^2` factor is not polynomial and not bounded here: an estimate, checked against a
/// dense reference in the previous release rather than proved.
[[nodiscard]] inline std::size_t projected_quadrature_size(std::size_t degree) noexcept {
  return 2 * degree + 2;
}

/// Tensor-product Gauss-Legendre weights on the unit cube, row-major over `num_nodes`.
[[nodiscard]] inline NdArray<double> tensor_gauss_weights(std::span<const std::size_t> num_nodes) {
  NdArray<double> w(std::vector<std::size_t>(num_nodes.begin(), num_nodes.end()), for_overwrite);
  std::vector<double> acc{1.0};
  for (const std::size_t n : num_nodes) {
    const quad::Rule1D<double> rule = quad::gauss_legendre_1d<double>(static_cast<long long>(n));
    std::vector<double> next(acc.size() * n);
    for (std::size_t i = 0; i < acc.size(); ++i) {
      for (std::size_t k = 0; k < n; ++k) {
        next[i * n + k] = acc[i] * rule.weights.data()[k];
      }
    }
    acc = std::move(next);
  }
  std::copy(acc.begin(), acc.end(), w.data());
  return w;
}

/// Evaluate a homogeneous control net `(*orders, rank + 1)` on the Gauss grid and divide out the
/// weights.
///
/// Returns nothing when a sampled weight is not strictly positive (a sign change of `w` between
/// samples, a pole; or a non-positive weight field, refused on the strictly positive convention).
/// A sign change confined strictly between two nodes is not detected.
///
/// Args:
///     data (std::span<const V>): The net, row-major over `shape`.
///     shape (std::span<const std::size_t>): `(*orders, rank + 1)`.
///     num_nodes (std::span<const std::size_t>): Nodes per direction.
///
/// Returns:
///     std::optional<NdArray<double>>: Projected points `(*num_nodes, rank)`, or nothing.
template <class V>
[[nodiscard]] std::optional<NdArray<double>> sample_projected(
    std::span<const V> data, std::span<const std::size_t> shape,
    std::span<const std::size_t> num_nodes) {
  std::vector<std::size_t> cur(shape.begin(), shape.end());
  NdArray<double> values(std::vector<std::size_t>(cur), for_overwrite);
  std::copy(data.begin(), data.end(), values.data());
  for (std::size_t axis = 0; axis + 1 < shape.size(); ++axis) {
    const quad::Rule1D<double> rule =
        quad::gauss_legendre_1d<double>(static_cast<long long>(num_nodes[axis]));
    const NdArray<double> basis =
        basis::tabulate_bernstein_1d<double>(static_cast<int>(shape[axis] - 1), rule.nodes.flat());
    values = contract_axis(basis, values.flat(), cur, axis);
    cur[axis] = num_nodes[axis];
  }
  const std::size_t cp = cur.back();
  const std::size_t n_pts = values.size() / cp;
  std::vector<std::size_t> out_shape(cur);
  out_shape.back() = cp - 1;
  NdArray<double> out(std::move(out_shape), for_overwrite);
  for (std::size_t i = 0; i < n_pts; ++i) {
    const double w = values.data()[i * cp + cp - 1];
    if (!(w > 0.0)) {
      return std::nullopt;
    }
  }
  for (std::size_t i = 0; i < n_pts; ++i) {
    const double w = values.data()[i * cp + cp - 1];
    for (std::size_t r = 0; r + 1 < cp; ++r) {
      out.data()[i * (cp - 1) + r] = values.data()[i * cp + r] / w;
    }
  }
  return out;
}

/// The relative `L^2` deviation of a projected trial from the current projected map, by
/// quadrature: `sqrt(sum w |trial - values|^2) / sqrt(sum w |values|^2)` (unscaled when the
/// reference is zero). Infinity when either side could not be sampled or the value is not
/// finite, so the caller rejects the trial.
[[nodiscard]] inline double projected_relative_deviation(
    const std::optional<NdArray<double>>& values, const std::optional<NdArray<double>>& trial,
    const NdArray<double>& quad_weights) {
  if (!values || !trial) {
    return std::numeric_limits<double>::infinity();
  }
  const std::size_t rank = values->shape().back();
  const std::size_t n = quad_weights.size();
  double dev2 = 0.0;
  double ref2 = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double d = 0.0;
    double r = 0.0;
    for (std::size_t c = 0; c < rank; ++c) {
      const double v = values->data()[i * rank + c];
      const double e = trial->data()[i * rank + c] - v;
      d += e * e;
      r += v * v;
    }
    dev2 += quad_weights.data()[i] * d;
    ref2 += quad_weights.data()[i] * r;
  }
  double dev = std::sqrt(dev2);
  if (ref2 > 0.0) {
    dev /= std::sqrt(ref2);
  }
  return std::isfinite(dev) ? dev : std::numeric_limits<double>::infinity();
}

}  // namespace detail

/// The exact `L^2` error that `reduce_degree` would introduce: reduce, elevate back (exact), and
/// take the Bernstein-Gram norm of the coefficient difference, rank components combined in the
/// Euclidean sense; for a rational Bézier over the homogeneous coefficients. A reduction that
/// would invert a weight is graded all the same.
///
/// Args:
///     b (const Bezier<T>&): The Bézier.
///     decrements (std::span<const long long>): As for `reduce_degree`.
///
/// Returns:
///     double: The error, in the units of the control points.
///
/// Raises:
///     value_error: Invalid decrements, or an elevated degree beyond `bincoeff_max_n`.
template <StorageFloat T>
[[nodiscard]] double degree_reduction_error(const Bezier<T>& b,
                                            std::span<const long long> decrements) {
  detail::check_decrements(b, decrements);
  const Bezier<T> reduced(detail::apply_reductions(b, decrements), b.rational());
  const Bezier<T> restored = elevate_degree(reduced, decrements);
  const std::span<const T> a = restored.control_points();
  const std::span<const T> c = b.control_points();
  std::vector<T> diff(a.size());
  for (std::size_t i = 0; i < diff.size(); ++i) {
    diff[i] = a[i] - c[i];  // in T, as the previous release subtracted the arrays
  }
  return std::sqrt(detail::total_squared_norm<T>(diff, b.shape()));
}

/// Lower the degree of a Bézier as far as accuracy allows, direction by direction.
///
/// In each direction the degree drops by one while the round trip (reduce, then elevate back)
/// stays below `tol` in relative `L^2` error; the first rejected trial ends that direction. Rank
/// components are graded together. A rational Bézier is graded on the projected map (a quadrature
/// estimate on a Gauss grid sized from the input degrees), and a trial whose sampled weight is not
/// strictly positive is rejected.
///
/// Args:
///     b (const Bezier<T>&): The Bézier.
///     tol (std::optional<double>): Relative tolerance; `1e3 * eps(T)` when absent. Zero or
///         negative keeps the input.
///
/// Returns:
///     Bezier<T>: The reduced Bézier, or a copy of `b` when nothing was reduced.
///
/// Raises:
///     value_error: A direction's degree exceeds `bincoeff_max_n`.
template <StorageFloat T>
[[nodiscard]] Bezier<T> minimize_degree(const Bezier<T>& b, std::optional<double> tol) {
  for (std::size_t d = 0; d < b.dim(); ++d) {
    const auto p = static_cast<long long>(b.degree(d));
    if (p >= 1) {
      check_bincoeff_envelope(p, "Degree minimization of a degree-" + std::to_string(p) +
                                     " direction " + std::to_string(d));
    }
  }
  const double t =
      tol ? *tol
          : auto_reduction_tol_factor * static_cast<double>(std::numeric_limits<T>::epsilon());
  std::vector<std::size_t> shape(b.shape().begin(), b.shape().end());
  // A copy even when nothing is reduced, as the previous release returned: the input may share a
  // buffer someone else writes (`Bezier`'s buffer constructor), and the result must not follow it.
  NdArray<T> result(std::vector<std::size_t>(shape), for_overwrite);
  std::copy(b.control_points().begin(), b.control_points().end(), result.data());
  if (t <= 0.0) {
    return Bezier<T>(std::move(result), b.rational());
  }

  std::vector<std::size_t> num_nodes(b.dim());
  for (std::size_t d = 0; d < b.dim(); ++d) {
    num_nodes[d] = detail::projected_quadrature_size(b.degree(d));
  }
  std::optional<NdArray<double>> quad_weights;
  std::optional<NdArray<double>> projected;
  if (b.rational()) {
    quad_weights = detail::tensor_gauss_weights(num_nodes);
    projected = detail::sample_projected<T>(result.flat(), shape, num_nodes);
  }

  for (std::size_t dim = 0; dim < b.dim(); ++dim) {
    while (shape[dim] >= 2) {
      const std::size_t degree = shape[dim] - 1;
      const auto op = cached_reduction_operator(ReductionKind::interpolating,
                                                static_cast<long long>(degree), 1);
      NdArray<T> reduced = detail::apply_along_axis<T>(result.flat(), shape, dim, *op);
      std::vector<std::size_t> rshape(shape);
      rshape[dim] = degree;
      const NdArray<T> flat = detail::flatten_along_axis<T>(reduced.flat(), rshape, dim);
      NdArray<T> up({degree + 1, flat.shape()[1]}, for_overwrite);
      detail::elevate_degree_1d_kernel<T>(degree - 1, 1, flat.template view<2>(),
                                          up.template view<2>());
      const NdArray<T> elevated =
          detail::unflatten_along_axis<T>(up.template view<2>(), rshape, dim);

      double rel_error = 0.0;
      if (quad_weights) {
        rel_error = detail::projected_relative_deviation(
            projected, detail::sample_projected<T>(elevated.flat(), shape, num_nodes),
            *quad_weights);
      } else {
        std::vector<T> diff(result.size());
        for (std::size_t i = 0; i < diff.size(); ++i) {
          diff[i] = elevated.data()[i] - result.data()[i];
        }
        const double diff_norm2 = detail::total_squared_norm<T>(diff, shape);
        const double orig_norm2 = detail::total_squared_norm<T>(result.flat(), shape);
        rel_error = std::sqrt(diff_norm2);
        if (orig_norm2 > 0.0) {
          rel_error /= std::sqrt(orig_norm2);
        }
      }
      if (rel_error >= t) {
        break;
      }
      result = std::move(reduced);
      shape = rshape;
      if (quad_weights) {
        projected = detail::sample_projected<T>(result.flat(), shape, num_nodes);
      }
    }
  }
  return Bezier<T>(std::move(result), b.rational());
}

}  // namespace pantr::bezier
