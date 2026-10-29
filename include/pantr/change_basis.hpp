/// @file change_basis.hpp
/// @brief Change-of-basis matrices between the one-dimensional polynomial bases on `[0, 1]`.
///
/// Every builder `X_to_Y` returns the `(degree + 1, degree + 1)` matrix `M` with
/// `M [X values](x) = [Y values](x)`, computed in `T`. Five of them recover their result from a
/// linear solve and refuse a `(degree, T)` pair outside the domain where the matrix they solve with
/// satisfies `kappa_inf(M) eps(T) < 1`, the boundary past which the perturbation bound of the solve
/// asserts nothing (the derivation, and the test that checks each tabulated degree against matrices
/// rebuilt in exact arithmetic, are the Python module `pantr.change_basis`'s and
/// `tests/test_change_basis_domain.py`). Linear algebra on Eigen; a solve against the identity is
/// an LU factorisation with partial pivoting, as numpy.linalg.solve's gesv, and refuses an exactly
/// zero pivot as gesv does, and a NaN pivot wherever it appears (gesv only in some positions).
///
/// The process tables (`cached_*`) hold one immutable matrix per key for a whole process, for
/// callers that need the same small matrix many times; see `change_basis_cache_max_degree`.
#pragma once

#include "pantr/basis/bernstein.hpp"
#include "pantr/basis/cardinal_bspline.hpp"
#include "pantr/basis/lagrange_variant.hpp"
#include "pantr/basis/legendre.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/exact_rational.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/linalg/dense.hpp"
#include "pantr/quad/rules_1d.hpp"

#include <Eigen/Core>
#include <Eigen/LU>

#if !defined(_WIN32)
#include <pthread.h>
#endif

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

namespace pantr::change_basis {

using basis::LagrangeVariant;

/// Largest degree a solving builder supports, per floating-point format.
///
/// Attributes:
///     float32 (int): For `float` results.
///     float64 (int): For `double` results.
struct DegreeLimit {
  int float32;
  int float64;
};

/// Supported degrees of `cardinal_to_bernstein`: `kappa_inf(A) eps` of the Bernstein-to-cardinal
/// matrix it inverts crosses 1 between degrees 8 and 9 (float) and 14 and 15 (double).
inline constexpr DegreeLimit cardinal_to_bernstein_max_degree{8, 14};

/// Supported degrees of `bernstein_to_cardinal`: the Bernstein Gram matrix it solves with crosses
/// between 12 and 13 (float) and 26 and 27 (double).
inline constexpr DegreeLimit bernstein_to_cardinal_max_degree{12, 26};

/// Supported degrees of `cardinal_to_legendre` and `cardinal_dual_legendre_coeffs`: the
/// Legendre-to-cardinal matrix crosses between 6 and 7 (float) and 12 and 13 (double).
inline constexpr DegreeLimit cardinal_to_legendre_max_degree{6, 12};

/// Supported degrees of `bernstein_to_lagrange` for a node family: the Lagrange-to-Bernstein
/// matrix it inverts, whose condition number depends on the nodes.
///
/// Args:
///     v (LagrangeVariant): The node family.
///
/// Returns:
///     DegreeLimit: Equispaced 17/37, Gauss-Legendre 22/51, the other three 23/52.
[[nodiscard]] constexpr DegreeLimit bernstein_to_lagrange_max_degree(LagrangeVariant v) {
  switch (v) {
    case LagrangeVariant::equispaces:
      return {17, 37};
    case LagrangeVariant::gauss_legendre:
      return {22, 51};
    case LagrangeVariant::gauss_lobatto_legendre:
    case LagrangeVariant::chebyshev_1st:
    case LagrangeVariant::chebyshev_2nd:
      return {23, 52};
  }
  throw value_error("unknown Lagrange variant");  // only an invalid enumerator value reaches here
}

namespace detail {

template <StorageFloat T>
constexpr const char* dtype_name() {
  return std::same_as<T, double> ? "float64" : "float32";
}

template <StorageFloat T>
void check_domain(int degree, DegreeLimit limit, const std::string& builder) {
  const int max = std::same_as<T, double> ? limit.float64 : limit.float32;
  if (degree <= max) {
    return;
  }
  throw value_error(builder + ": degree " + std::to_string(degree) +
                    " is outside the supported domain for " + dtype_name<T>() +
                    " (supported: degree <= " + std::to_string(limit.float32) +
                    " in float32, degree <= " + std::to_string(limit.float64) +
                    " in float64). Past that degree the matrix this builder solves with is "
                    "singular to working precision, so the error bound on the result reaches "
                    "100% and guarantees no correct digit; see the pantr.change_basis module "
                    "docstring for the derivation.");
}

/// Refuse a degree whose `(degree + 1, degree + 1)` result cannot be allocated, before any
/// work (or any intermediate table) is sized from it: NdArray's own rule, applied up front.
///
/// Raises:
///     capacity_error: `(degree + 1)^2 sizeof(T)` bytes exceed the largest `std::ptrdiff_t`.
template <StorageFloat T>
void check_matrix_fits(int degree) {
  const auto m = static_cast<std::size_t>(degree) + 1;
  constexpr auto max_elements =
      static_cast<std::size_t>((std::numeric_limits<std::ptrdiff_t>::max)()) / sizeof(T);
  if (m > max_elements / m) {
    throw capacity_error("a (" + std::to_string(m) + ", " + std::to_string(m) +
                         ") matrix has more elements than fit in memory");
  }
}

inline void check_non_negative(int degree) {
  if (degree < 0) {
    throw value_error("Degree must be non-negative");
  }
}

inline void check_at_least_one(int degree) {
  if (degree < 1) {
    throw value_error("Degree must at least 1");
  }
}

using linalg::as_matrix;
using linalg::RowMatrix;

/// Solve `A X = I` by LU with partial pivoting (`linalg::try_inverse`).
///
/// Raises:
///     value_error: A pivot of the factorisation is exactly zero or NaN ("Singular matrix", the
///         condition under which numpy.linalg.solve raises LinAlgError, a ValueError).
template <StorageFloat T>
NdArray<T> inverse(const NdArray<T>& a) {
  std::optional<NdArray<T>> inv = linalg::try_inverse(a);
  if (!inv) {
    throw value_error("Singular matrix");
  }
  return std::move(*inv);
}

/// The Gram projection `old = M new`: row `i` of `M` holds the coefficients of the `i`-th old
/// basis function in the new basis, from `G M^T = C` with `G` the new basis's Gram matrix and `C`
/// the mixed one, both by `n_quad` Gauss-Legendre points in `T` (`degree + 1` points integrate
/// every product of two degree-`degree` polynomials exactly).
///
/// Raises:
///     value_error: `n_quad < 1`, or an exactly singular Gram matrix.
template <StorageFloat T, class NewEval, class OldEval>
NdArray<T> gram_projection(int degree, long long n_quad, NewEval new_eval, OldEval old_eval) {
  if (n_quad < 1) {
    throw value_error("Number of quadrature points must be positive.");
  }
  const quad::Rule1D<T> rule = quad::gauss_legendre_1d<T>(n_quad);
  const std::span<const T> pts = rule.nodes.flat();
  const NdArray<T> new_b = new_eval(degree, pts);
  const NdArray<T> old_b = old_eval(degree, pts);
  const auto nb = as_matrix(new_b);
  const auto ob = as_matrix(old_b);
  const auto w = Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>>(
      rule.weights.data(), static_cast<Eigen::Index>(rule.weights.size()));
  const RowMatrix<T> nbw = nb.transpose() * w.asDiagonal();
  const RowMatrix<T> g = nbw * nb;
  const RowMatrix<T> c = nbw * ob;
  const Eigen::PartialPivLU<RowMatrix<T>> lu(g);
  const auto& factors = lu.matrixLU();
  for (Eigen::Index i = 0; i < factors.rows(); ++i) {
    if (factors(i, i) == static_cast<T>(0) || std::isnan(factors(i, i))) {
      throw value_error("Singular matrix");
    }
  }
  const RowMatrix<T> mt = lu.solve(c);
  NdArray<T> out({static_cast<std::size_t>(mt.cols()), static_cast<std::size_t>(mt.rows())});
  Eigen::Map<RowMatrix<T>>(out.data(), mt.cols(), mt.rows()) = mt.transpose();
  return out;
}

}  // namespace detail

/// Lagrange-to-Bernstein matrix `C`, `C[i, k] = B_i(x_k)` at the variant's nodes `x_k`.
///
/// No solve (the Lagrange basis is cardinal at its own nodes), so no degree limit.
///
/// Raises:
///     value_error: `degree < 1`, or the variant needs more nodes.
template <StorageFloat T>
[[nodiscard]] NdArray<T> lagrange_to_bernstein(int degree, LagrangeVariant variant) {
  detail::check_at_least_one(degree);
  detail::check_matrix_fits<T>(degree);
  const NdArray<T> x = basis::lagrange_points<T>(variant, static_cast<long long>(degree) + 1);
  const NdArray<T> b = basis::tabulate_bernstein_1d<T>(degree, x.flat());  // b[k, i] = B_i(x_k)
  const std::size_t m = x.size();
  NdArray<T> out({m, m});
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t k = 0; k < m; ++k) {
      out.data()[i * m + k] = b.data()[k * m + i];
    }
  }
  return out;
}

/// Bernstein-to-Lagrange matrix: the inverse of `lagrange_to_bernstein`.
///
/// Raises:
///     value_error: `degree < 1`, `degree` past `bernstein_to_lagrange_max_degree(variant)`, or an
///         exactly singular factorisation.
template <StorageFloat T>
[[nodiscard]] NdArray<T> bernstein_to_lagrange(int degree, LagrangeVariant variant) {
  detail::check_at_least_one(degree);
  detail::check_domain<T>(degree, bernstein_to_lagrange_max_degree(variant),
                          std::string("compute_bernstein_to_lagrange_1d with ") +
                              basis::variant_name(variant) + " nodes");
  return detail::inverse(lagrange_to_bernstein<T>(degree, variant));
}

/// Bernstein-to-cardinal matrix, by one Gram projection on the Bernstein Gram matrix
/// (`degree + 1` Gauss-Legendre points integrate every product exactly).
///
/// Raises:
///     value_error: `degree < 0`, or past `bernstein_to_cardinal_max_degree`.
template <StorageFloat T>
[[nodiscard]] NdArray<T> bernstein_to_cardinal(int degree) {
  detail::check_non_negative(degree);
  detail::check_domain<T>(degree, bernstein_to_cardinal_max_degree,
                          "compute_bernstein_to_cardinal_1d");
  return detail::gram_projection<T>(
      degree, static_cast<long long>(degree) + 1,
      [](int d, std::span<const T> p) { return basis::tabulate_bernstein_1d<T>(d, p); },
      [](int d, std::span<const T> p) { return basis::tabulate_cardinal_bspline_1d<T>(d, p); });
}

/// Cardinal-to-Bernstein matrix: the inverse of `bernstein_to_cardinal` (a second projection on
/// the cardinal Gram matrix would lose twice the digits).
///
/// Raises:
///     value_error: `degree < 0`, past `cardinal_to_bernstein_max_degree`, or an exactly singular
///         factorisation.
template <StorageFloat T>
[[nodiscard]] NdArray<T> cardinal_to_bernstein(int degree) {
  detail::check_non_negative(degree);
  detail::check_domain<T>(degree, cardinal_to_bernstein_max_degree,
                          "compute_cardinal_to_bernstein_1d");
  return detail::inverse(bernstein_to_cardinal<T>(degree));
}

/// Legendre-to-cardinal matrix, by one Gram projection on the Legendre Gram matrix (the identity
/// up to rounding), so no degree limit.
///
/// Raises:
///     value_error: `degree < 0`.
template <StorageFloat T>
[[nodiscard]] NdArray<T> legendre_to_cardinal(int degree) {
  detail::check_non_negative(degree);
  detail::check_matrix_fits<T>(degree);
  return detail::gram_projection<T>(
      degree, static_cast<long long>(degree) + 1,
      [](int d, std::span<const T> p) { return basis::tabulate_legendre_1d<T>(d, p); },
      [](int d, std::span<const T> p) { return basis::tabulate_cardinal_bspline_1d<T>(d, p); });
}

/// Cardinal-to-Legendre matrix: the inverse of `legendre_to_cardinal`.
///
/// Raises:
///     value_error: `degree < 0`, past `cardinal_to_legendre_max_degree`, or an exactly singular
///         factorisation.
template <StorageFloat T>
[[nodiscard]] NdArray<T> cardinal_to_legendre(int degree) {
  detail::check_non_negative(degree);
  detail::check_domain<T>(degree, cardinal_to_legendre_max_degree,
                          "compute_cardinal_to_legendre_1d");
  return detail::inverse(legendre_to_cardinal<T>(degree));
}

/// The cardinal-dual functionals in the Legendre basis: `cardinal_to_legendre(degree)`
/// transposed, row `i` the Legendre coefficients of the dual function biorthogonal to the `i`-th
/// cardinal B-spline.
///
/// Raises:
///     value_error: `degree < 0`, past `cardinal_to_legendre_max_degree`, or an exactly singular
///         factorisation.
template <StorageFloat T>
[[nodiscard]] NdArray<T> cardinal_dual_legendre_coeffs(int degree) {
  detail::check_non_negative(degree);
  detail::check_domain<T>(degree, cardinal_to_legendre_max_degree,
                          "compute_cardinal_dual_legendre_coeffs_1d");
  const NdArray<T> w = cardinal_to_legendre<T>(degree);
  const std::size_t m = w.shape()[0];
  NdArray<T> out({m, m});
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t k = 0; k < m; ++k) {
      out.data()[i * m + k] = w.data()[k * m + i];
    }
  }
  return out;
}

/// Monomial-to-Bernstein matrix `M[i, j] = C(i, j) / C(degree, j)` for `j <= i`, else 0.
///
/// Each entry is the double nearest the exact quotient (rounded once; in `float`, that double
/// rounded again, as a double-precision division stored in a float array is), at every degree:
/// the binomials are exact integers, divided in double where both are at most 2^53 and by an exact
/// integer division otherwise. The first column is exactly 1.
///
/// Raises:
///     value_error: `degree < 0`.
template <StorageFloat T>
[[nodiscard]] NdArray<T> monomial_to_bernstein(int degree) {
  detail::check_non_negative(degree);
  detail::check_matrix_fits<T>(degree);
  const auto m = static_cast<std::size_t>(degree) + 1;
  NdArray<T> out({m, m});
  // C(d, floor(d / 2)) <= 2^53 up to d = 56: every binomial of the table is then exact in double.
  constexpr int max_exact_double_degree = 56;
  if (degree <= max_exact_double_degree) {
    std::vector<std::vector<std::uint64_t>> c(m, std::vector<std::uint64_t>(m, 0));  // Pascal
    for (std::size_t i = 0; i < m; ++i) {
      c[i][0] = 1;
      for (std::size_t j = 1; j <= i; ++j) {
        c[i][j] = c[i - 1][j - 1] + (j < i ? c[i - 1][j] : 0);
      }
    }
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        out.data()[i * m + j] =
            static_cast<T>(static_cast<double>(c[i][j]) / static_cast<double>(c[m - 1][j]));
      }
    }
    return out;
  }
  std::vector<std::vector<exact::Integer>> c(m, std::vector<exact::Integer>(m, 0));
  for (std::size_t i = 0; i < m; ++i) {
    c[i][0] = 1;
    for (std::size_t j = 1; j <= i; ++j) {
      c[i][j] = c[i - 1][j - 1] + (j < i ? c[i - 1][j] : exact::Integer(0));
    }
  }
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      out.data()[i * m + j] =
          static_cast<T>(exact::correctly_rounded_quotient(c[i][j], c[m - 1][j]));
    }
  }
  return out;
}

/// The change-of-basis matrices a process table holds.
enum class Kind : int {
  lagrange_to_bernstein = 0,
  bernstein_to_lagrange = 1,
  bernstein_to_cardinal = 2,
  cardinal_to_bernstein = 3,
  legendre_to_cardinal = 4,
  cardinal_to_legendre = 5,
};

/// Largest degree whose matrix a process table keeps. Bounds each table at a few hundred entries
/// (six kinds, five node families for two of them, degrees 0 to 64), each (kind, family, type)
/// under a megabyte of matrices, while covering every degree the solving builders accept (at most
/// 52). An admitted bound, not a derived one; a matrix of higher degree is computed per call and
/// not kept.
inline constexpr int change_basis_cache_max_degree = 64;

namespace detail {

/// Guards both process tables. Constant-initialized (no initialization guard a forked child could
/// inherit mid-initialization) and, on POSIX, held by the forking thread across `fork` and released
/// in both processes (`table_atfork_status`), so a child never inherits it locked by a thread it
/// does not have. The same arrangement as the pool's (design/threading.md, fork safety).
inline constinit std::mutex table_mutex;

/// The entries of the process table for `T`, keyed on `(kind, degree, variant)`. A variable with
/// static storage initialized when the program image loads, so no function-local guard is taken on
/// first use.
template <StorageFloat T>
inline std::map<std::tuple<int, int, int>, std::shared_ptr<const NdArray<T>>> table_entries;

#if !defined(_WIN32)
/// Registration of `table_mutex`'s fork handlers; nonzero means it failed, which `cached` reports.
inline const int table_atfork_status = ::pthread_atfork(
    [] { table_mutex.lock(); }, [] { table_mutex.unlock(); }, [] { table_mutex.unlock(); });
#endif

inline void check_table_atfork() {
#if !defined(_WIN32)
  // Reading the status also odr-uses it, so its registration cannot be deferred past this.
  if (table_atfork_status != 0) {
    throw std::system_error(table_atfork_status, std::generic_category(),
                            "pthread_atfork: the change-of-basis table's fork handler could not "
                            "be registered");
  }
#endif
}

template <StorageFloat T>
NdArray<T> compute(Kind kind, int degree, LagrangeVariant variant) {
  switch (kind) {
    case Kind::lagrange_to_bernstein:
      return lagrange_to_bernstein<T>(degree, variant);
    case Kind::bernstein_to_lagrange:
      return bernstein_to_lagrange<T>(degree, variant);
    case Kind::bernstein_to_cardinal:
      return bernstein_to_cardinal<T>(degree);
    case Kind::cardinal_to_bernstein:
      return cardinal_to_bernstein<T>(degree);
    case Kind::legendre_to_cardinal:
      return legendre_to_cardinal<T>(degree);
    case Kind::cardinal_to_legendre:
      return cardinal_to_legendre<T>(degree);
  }
  throw value_error("unknown change-of-basis kind");
}

}  // namespace detail

/// Get a change-of-basis matrix from the process table, computing it on first use.
///
/// Every call with the same key returns the same immutable matrix (one allocation per key per
/// program image: each shared object compiled from these headers with hidden symbols, such as
/// each Python extension module, has its own table), up to `change_basis_cache_max_degree`; past it
/// each call computes a fresh matrix. Thread-safe: the matrix is computed outside the table's lock,
/// and when two threads compute one key at once the first to store it wins and both return that
/// one.
///
/// Args:
///     kind (Kind): Which matrix.
///     degree (int): The degree.
///     variant (LagrangeVariant): The node family; part of the key for the two Lagrange kinds
///         only.
///
/// Returns:
///     std::shared_ptr<const NdArray<T>>: The matrix, as the matching builder returns it.
///
/// Raises:
///     value_error: As the matching builder.
///     std::system_error: The table's fork handler could not be registered (POSIX).
template <StorageFloat T>
[[nodiscard]] std::shared_ptr<const NdArray<T>> cached(Kind kind, int degree,
                                                       LagrangeVariant variant) {
  const bool lagrange = kind == Kind::lagrange_to_bernstein || kind == Kind::bernstein_to_lagrange;
  const std::tuple<int, int, int> key{static_cast<int>(kind), degree,
                                      lagrange ? static_cast<int>(variant) : 0};
  const bool keep = degree >= 0 && degree <= change_basis_cache_max_degree;
  detail::check_table_atfork();
  auto& entries = detail::table_entries<T>;
  if (keep) {
    const std::lock_guard<std::mutex> lock(detail::table_mutex);
    const auto it = entries.find(key);
    if (it != entries.end()) {
      return it->second;
    }
  }
  auto fresh = std::make_shared<const NdArray<T>>(detail::compute<T>(kind, degree, variant));
  if (!keep) {
    return fresh;
  }
  const std::lock_guard<std::mutex> lock(detail::table_mutex);
  return entries.try_emplace(key, std::move(fresh)).first->second;
}

/// Get the number of matrices a process table holds for `T`.
///
/// Returns:
///     std::size_t: The entry count.
template <StorageFloat T>
[[nodiscard]] std::size_t cache_size() {
  const std::lock_guard<std::mutex> lock(detail::table_mutex);
  return detail::table_entries<T>.size();
}

/// Empty the process table for `T`. Matrices already handed out stay valid (they are shared).
template <StorageFloat T>
void cache_clear() {
  const std::lock_guard<std::mutex> lock(detail::table_mutex);
  detail::table_entries<T>.clear();
}

}  // namespace pantr::change_basis
