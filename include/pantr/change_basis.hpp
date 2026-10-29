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
/// zero or NaN pivot as gesv does.
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
#include "pantr/quad/rules_1d.hpp"

#include <Eigen/Core>
#include <Eigen/LU>

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <tuple>
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
[[nodiscard]] constexpr DegreeLimit bernstein_to_lagrange_max_degree(LagrangeVariant v) noexcept {
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
  return {0, 0};
}

namespace detail {

template <std::floating_point T>
constexpr const char* dtype_name() {
  return sizeof(T) == sizeof(double) ? "float64" : "float32";
}

template <std::floating_point T>
void check_domain(int degree, DegreeLimit limit, const std::string& builder) {
  const int max = sizeof(T) == sizeof(double) ? limit.float64 : limit.float32;
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

template <std::floating_point T>
using RowMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <std::floating_point T>
Eigen::Map<const RowMatrix<T>> as_matrix(const NdArray<T>& a) {
  return {a.data(), static_cast<Eigen::Index>(a.shape()[0]),
          static_cast<Eigen::Index>(a.shape()[1])};
}

/// Solve `A X = I` by LU with partial pivoting: an explicit inverse (Du Croz and Higham's
/// Method A), which bounds the residual `A X - I`, not `X A - I`.
///
/// Raises:
///     value_error: A pivot of the factorisation is exactly zero or NaN ("Singular matrix", the
///         condition under which numpy.linalg.solve raises LinAlgError, a ValueError).
template <std::floating_point T>
NdArray<T> inverse(const NdArray<T>& a) {
  const auto m = static_cast<Eigen::Index>(a.shape()[0]);
  const Eigen::PartialPivLU<RowMatrix<T>> lu(as_matrix(a));
  const auto& factors = lu.matrixLU();
  for (Eigen::Index i = 0; i < m; ++i) {
    const T pivot = factors(i, i);
    if (pivot == static_cast<T>(0) || std::isnan(pivot)) {
      throw value_error("Singular matrix");
    }
  }
  NdArray<T> out({a.shape()[0], a.shape()[1]});
  Eigen::Map<RowMatrix<T>>(out.data(), m, m) = lu.solve(RowMatrix<T>::Identity(m, m));
  return out;
}

/// The Gram projection `old = M new`: row `i` of `M` holds the coefficients of the `i`-th old
/// basis function in the new basis, from `G M^T = C` with `G` the new basis's Gram matrix and `C`
/// the mixed one, both by `n_quad` Gauss-Legendre points in `T`.
template <std::floating_point T, class NewEval, class OldEval>
NdArray<T> gram_projection(int degree, NewEval new_eval, OldEval old_eval) {
  const quad::Rule1D<T> rule = quad::gauss_legendre_1d<T>(degree + 1);
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
template <std::floating_point T>
[[nodiscard]] NdArray<T> lagrange_to_bernstein(int degree, LagrangeVariant variant) {
  detail::check_at_least_one(degree);
  const NdArray<T> x = basis::lagrange_points<T>(variant, degree + 1);
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
template <std::floating_point T>
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
template <std::floating_point T>
[[nodiscard]] NdArray<T> bernstein_to_cardinal(int degree) {
  detail::check_non_negative(degree);
  detail::check_domain<T>(degree, bernstein_to_cardinal_max_degree,
                          "compute_bernstein_to_cardinal_1d");
  return detail::gram_projection<T>(
      degree, [](int d, std::span<const T> p) { return basis::tabulate_bernstein_1d<T>(d, p); },
      [](int d, std::span<const T> p) { return basis::tabulate_cardinal_bspline_1d<T>(d, p); });
}

/// Cardinal-to-Bernstein matrix: the inverse of `bernstein_to_cardinal` (a second projection on
/// the cardinal Gram matrix would lose twice the digits).
///
/// Raises:
///     value_error: `degree < 0`, past `cardinal_to_bernstein_max_degree`, or an exactly singular
///         factorisation.
template <std::floating_point T>
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
template <std::floating_point T>
[[nodiscard]] NdArray<T> legendre_to_cardinal(int degree) {
  detail::check_non_negative(degree);
  return detail::gram_projection<T>(
      degree, [](int d, std::span<const T> p) { return basis::tabulate_legendre_1d<T>(d, p); },
      [](int d, std::span<const T> p) { return basis::tabulate_cardinal_bspline_1d<T>(d, p); });
}

/// Cardinal-to-Legendre matrix: the inverse of `legendre_to_cardinal`.
///
/// Raises:
///     value_error: `degree < 0`, past `cardinal_to_legendre_max_degree`, or an exactly singular
///         factorisation.
template <std::floating_point T>
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
template <std::floating_point T>
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
template <std::floating_point T>
[[nodiscard]] NdArray<T> monomial_to_bernstein(int degree) {
  detail::check_non_negative(degree);
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
/// (six kinds, five node families for two of them, degrees 0 to 64) and under a megabyte per kind
/// and type, while covering every degree the solving builders accept (at most 52). An admitted
/// bound, not a derived one; a matrix of higher degree is computed per call and not kept.
inline constexpr int change_basis_cache_max_degree = 64;

namespace detail {

template <std::floating_point T>
struct MatrixTable {
  std::mutex mutex;
  std::map<std::tuple<int, int, int>, std::shared_ptr<const NdArray<T>>> entries;
};

template <std::floating_point T>
MatrixTable<T>& matrix_table() {
  static MatrixTable<T> table;
  return table;
}

template <std::floating_point T>
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
/// process), up to `change_basis_cache_max_degree`; past it each call computes a fresh matrix.
/// Thread-safe: the matrix is computed outside the table's lock, and when two threads compute one
/// key at once the first to store it wins and both return that one.
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
template <std::floating_point T>
[[nodiscard]] std::shared_ptr<const NdArray<T>> cached(Kind kind, int degree,
                                                       LagrangeVariant variant) {
  const bool lagrange = kind == Kind::lagrange_to_bernstein || kind == Kind::bernstein_to_lagrange;
  const std::tuple<int, int, int> key{static_cast<int>(kind), degree,
                                      lagrange ? static_cast<int>(variant) : 0};
  const bool keep = degree >= 0 && degree <= change_basis_cache_max_degree;
  auto& table = detail::matrix_table<T>();
  if (keep) {
    const std::lock_guard<std::mutex> lock(table.mutex);
    const auto it = table.entries.find(key);
    if (it != table.entries.end()) {
      return it->second;
    }
  }
  auto fresh = std::make_shared<const NdArray<T>>(detail::compute<T>(kind, degree, variant));
  if (!keep) {
    return fresh;
  }
  const std::lock_guard<std::mutex> lock(table.mutex);
  return table.entries.try_emplace(key, std::move(fresh)).first->second;
}

/// Get the number of matrices a process table holds for `T`.
///
/// Returns:
///     std::size_t: The entry count.
template <std::floating_point T>
[[nodiscard]] std::size_t cache_size() {
  auto& table = detail::matrix_table<T>();
  const std::lock_guard<std::mutex> lock(table.mutex);
  return table.entries.size();
}

/// Empty the process table for `T`. Matrices already handed out stay valid (they are shared).
template <std::floating_point T>
void cache_clear() {
  auto& table = detail::matrix_table<T>();
  const std::lock_guard<std::mutex> lock(table.mutex);
  table.entries.clear();
}

}  // namespace pantr::change_basis
