/// @file roots.hpp
/// @brief The root-finding seam (design §4.1): claims, completeness and capabilities, the
///        `RootProvider` concept, the `Builtin` provider, and the entry points over a `Bezier`.
///
/// What a result certifies is named in the input's terms: the roots in `[0, 1]` of the stored
/// scalar Bernstein polynomial (for a rational Bézier, of its first homogeneous component, whose
/// roots are the rational function's where the weights are positive). `design/root_seam.md` holds
/// the contract and the duplicate pair-class table.
#pragma once

#include "pantr/bezier/bezier.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/parallel/parallel_for.hpp"
#include "pantr/parallel/work_model.hpp"
#include "pantr/roots/detail/clipping.hpp"
#include "pantr/roots/detail/scalar.hpp"
#include "pantr/roots/detail/yuksel.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::roots {

/// What a provider claims about one root. The integer values are a wire format (they cross into
/// Python and between extension modules) and never change.
enum class Claim : std::int32_t {
  none = 0,       ///< No claim.
  computed = 1,   ///< Returned by the algorithm; nothing further is claimed.
  certified = 2,  ///< Existence and uniqueness of a root within `param_tol` of `x` are proved.
};

/// Whether a result accounts for the whole interval. The integer values are a wire format.
enum class Completeness : std::int32_t {
  complete = 0,         ///< No part of `[0, 1]` was given up on.
  gap_localized = 1,    ///< Some intervals, listed in `gaps`, carry no claim.
  gap_unlocalized = 2,  ///< Some part of `[0, 1]` carries no claim, and where is not known.
};

/// What a provider can do, as a value (no minimum claim can be requested; design §4.1).
///
/// Attributes:
///     max_claim (Claim): The strongest claim it ever makes.
///     batch (bool): Whether it solves many polynomials of one degree in one call.
///     monotone (bool): Whether it has a solver for a polynomial known to be monotone.
///     multiplicity (bool): Whether it reports multiplicities.
struct Capabilities {
  Claim max_claim;
  bool batch;
  bool monotone;
  bool multiplicity;
};

/// One root.
///
/// Attributes:
///     x (R): The parameter, in `[0, 1]`.
///     claim (Claim): What is claimed about it.
///     multiplicity (std::int32_t): Its multiplicity, or -1 for no claim.
template <class R>
struct Root {
  R x;
  Claim claim;
  std::int32_t multiplicity;
};

/// The roots of one polynomial and what they account for.
///
/// The root type is the parameter's, `double` for both coefficient types: the previous release
/// returned `float64` roots for a `float32` polynomial, so a `float` parameter would lose digits
/// the interface promised (a deliberate departure from design §4.1's `RootSet<T>`).
///
/// Attributes:
///     roots (std::vector<Root<R>>): Ascending.
///     completeness (Completeness): Whether the result covers `[0, 1]`.
///     gaps (std::vector<std::pair<R, R>>): The intervals without a claim (`gap_localized`).
template <class R>
struct [[nodiscard]] RootSet {
  std::vector<Root<R>> roots;
  Completeness completeness;
  std::vector<std::pair<R, R>> gaps;
};

/// Tolerances of a request, each in its own frame.
///
/// Attributes:
///     param_tol (double): Absolute, on the parameter interval `[0, 1]`: the bracket width at
///         which a root counts as found.
///     value_tol (double): Absolute, in the units of the coefficients: the floor of every zero
///         test. The Python default passes the dimensionless `get_strict(dtype)` into both, as the
///         previous release did (whether it should scale with the coefficients is left to review).
struct FindRootsOptions {
  double param_tol;
  double value_tol;
};

/// A root provider: its capabilities, and the roots of one Bernstein polynomial of storage type
/// `T`.
template <class P, class T>
concept RootProvider = requires(const P p, std::span<const T> bernstein, FindRootsOptions o) {
  { p.capabilities() } -> std::same_as<Capabilities>;
  { p.find_roots(bernstein, o) } -> std::same_as<RootSet<double>>;
};

/// The roots of many polynomials of one degree.
///
/// Attributes:
///     roots (NdArray<double>): Extents `(n_polys, max(degree, 1))`; row `i` holds `counts[i]`
///         ascending roots, then NaN.
///     counts (std::vector<std::int64_t>): Roots per polynomial.
///     completeness (std::vector<Completeness>): Per polynomial.
struct [[nodiscard]] BatchRootSet {
  NdArray<double> roots;
  std::vector<std::int64_t> counts;
  std::vector<Completeness> completeness;
};

namespace detail {

/// Smallest degree at which clipping is considered: below it Yuksel's derivative chain is at most
/// four levels deep and cheaper than a hull and subdivision. Kept from the previous release.
inline constexpr std::size_t clip_min_degree = 6;

/// Largest `max |c| / min nonzero |c|` for which clipping is used, chosen in the previous release
/// by a sweep of clipping against Yuksel (an empirical bound, admitted as such).
inline constexpr double clip_coeff_range_limit = 1e8;

/// Sort ascending with NaN last (NumPy's order).
inline void sort_nan_last(std::vector<double>& v) {
  std::sort(v.begin(), v.end(),
            [](double a, double b) { return a < b || (!std::isnan(a) && std::isnan(b)); });
}

/// A `RootSet` of computed roots without multiplicities.
inline RootSet<double> computed_set(const std::vector<double>& xs,
                                    std::vector<std::pair<double, double>> gaps) {
  RootSet<double> out{
      {}, gaps.empty() ? Completeness::complete : Completeness::gap_localized, std::move(gaps)};
  out.roots.reserve(xs.size());
  for (const double x : xs) {
    out.roots.push_back({x, Claim::computed, -1});
  }
  return out;
}

/// The single-polynomial dispatch of the previous release: Yuksel below `clip_min_degree` or past
/// the coefficient range limit (sorted, not merged), clipping otherwise (merged).
template <StorageFloat T>
RootSet<double> dispatch_single(std::span<const T> coeff, double param_tol, double geom_tol) {
  const std::size_t n = coeff.size() - 1;
  if (n < 1) {
    return computed_set({}, {});
  }
  // The previous release compared in NumPy, which rounds the tolerance (a Python float) to the
  // array's dtype first (NEP 50): in `T` here, a tolerance past `T`'s range reading as infinite.
  const T tol_t = std::abs(geom_tol) > static_cast<double>((std::numeric_limits<T>::max)())
                      ? std::numeric_limits<T>::infinity()
                      : static_cast<T>(geom_tol);
  bool all_zero = true;
  for (const T c : coeff) {
    all_zero = all_zero && std::abs(c) <= tol_t;  // NaN: not zero
  }
  if (all_zero) {
    return computed_set({}, {});
  }
  bool use_clipping = false;
  if (n >= clip_min_degree) {
    double c_max = 0.0;
    double c_min = std::numeric_limits<double>::infinity();
    bool nan = false;
    for (const T c : coeff) {
      const double a = std::abs(static_cast<double>(c));
      nan = nan || std::isnan(a);
      c_max = (std::max)(c_max, a);
      if (a > 0.0) {
        c_min = (std::min)(c_min, a);
      }
    }
    // NumPy's max propagates NaN, and NaN <= limit is false: Yuksel.
    const double range = nan ? std::numeric_limits<double>::quiet_NaN()
                             : (c_min < std::numeric_limits<double>::infinity()
                                    ? c_max / c_min
                                    : std::numeric_limits<double>::infinity());
    use_clipping = range <= clip_coeff_range_limit;
  }
  if (use_clipping) {
    pantr::roots::detail::ClipResult clip = clip_roots<T>(coeff, param_tol, geom_tol);
    return computed_set(dedup_roots<T>(std::move(clip.candidates), coeff, param_tol, geom_tol),
                        std::move(clip.gaps));
  }
  std::vector<double> r = yuksel_roots<T>(coeff, param_tol);
  sort_nan_last(r);
  return computed_set(r, {});
}

/// The batch dispatch of the previous release, which differs from the single one in two places it
/// kept: Yuksel's roots are merged too, and the all-zero test passes a NaN coefficient.
template <StorageFloat T>
RootSet<double> dispatch_batch_row(std::span<const T> coeff, double param_tol, double geom_tol) {
  const std::size_t n = coeff.size() - 1;
  if (n < 1) {
    return computed_set({}, {});
  }
  bool all_zero = true;
  for (const T c : coeff) {
    if (std::abs(static_cast<double>(c)) > geom_tol) {
      all_zero = false;
      break;
    }
  }
  if (all_zero) {
    return computed_set({}, {});
  }
  bool use_clipping = false;
  if (n >= clip_min_degree) {
    double c_max = 0.0;
    double c_min = std::numeric_limits<double>::infinity();
    for (const T c : coeff) {
      const double a = std::abs(static_cast<double>(c));
      c_max = (std::max)(c_max, a);
      if (a > 0.0 && a < c_min) {
        c_min = a;
      }
    }
    const double range = c_min < std::numeric_limits<double>::infinity()
                             ? c_max / c_min
                             : std::numeric_limits<double>::infinity();
    use_clipping = range <= clip_coeff_range_limit;
  }
  std::vector<double> raw;
  std::vector<std::pair<double, double>> gaps;
  if (use_clipping) {
    pantr::roots::detail::ClipResult clip = clip_roots<T>(coeff, param_tol, geom_tol);
    raw = std::move(clip.candidates);
    gaps = std::move(clip.gaps);
  } else {
    raw = yuksel_roots<T>(coeff, param_tol);
  }
  return computed_set(dedup_roots<T>(std::move(raw), coeff, param_tol, geom_tol), std::move(gaps));
}

/// Operations per polynomial of degree `n`, for the batch's chunking only: a derivative chain of
/// `n^2 / 2` differences and, per root, Newton iterations of `n^2 / 2` each. An admitted
/// heuristic, of the order of a few Newton iterations per root (design §2.8).
[[nodiscard]] inline double ops_per_polynomial(std::size_t n) {
  const auto d = static_cast<double>(n + 1);
  return (std::max)(1.0, 8.0 * d * d * d);
}

inline void check_coeffs(std::size_t n_coeff) {
  if (n_coeff < 1) {
    throw value_error("coeff must have at least 1 element");
  }
}

inline void check_tol(double tol, const char* name) {
  if (tol <= 0.0) {  // NaN passes, as in the previous release
    throw value_error(std::string(name) + " must be positive, got " + py_repr(tol));
  }
}

}  // namespace detail

/// pantr's own provider: Yuksel's algorithm, Bezier clipping and the derivative-aware merge,
/// ported from the previous release. It claims `computed` for every root and reports a gap where
/// clipping gave up at its depth cap.
class Builtin {
 public:
  /// Get the provider's capabilities.
  ///
  /// Returns:
  ///     Capabilities: `computed`, batch and monotone solvers, no multiplicities.
  [[nodiscard]] Capabilities capabilities() const noexcept {
    return {Claim::computed, true, true, false};
  }

  /// Find the roots of one Bernstein polynomial on `[0, 1]`.
  ///
  /// Args:
  ///     coeff (std::span<const T>): Coefficients, at least one.
  ///     o (FindRootsOptions): Tolerances.
  ///
  /// Returns:
  ///     RootSet<double>: Ascending computed roots; `gap_localized` when clipping gave up.
  ///
  /// Raises:
  ///     value_error: `coeff` is empty, or a tolerance is not positive.
  template <StorageFloat T>
  [[nodiscard]] RootSet<double> find_roots(std::span<const T> coeff, FindRootsOptions o) const {
    detail::check_coeffs(coeff.size());
    detail::check_tol(o.param_tol, "param_tol");
    detail::check_tol(o.value_tol, "value_tol");
    return detail::dispatch_single<T>(coeff, o.param_tol, o.value_tol);
  }

  /// Find the roots of each row of `(n_polys, degree + 1)` coefficients, in parallel over rows.
  ///
  /// Args:
  ///     coeffs (mdview<const T, 2>): One polynomial per row, at least one column.
  ///     o (FindRootsOptions): Tolerances.
  ///
  /// Returns:
  ///     BatchRootSet: Roots, counts and completeness per row.
  ///
  /// Raises:
  ///     value_error: No column, or a tolerance is not positive.
  template <StorageFloat T>
  [[nodiscard]] BatchRootSet find_roots_batch(mdview<const T, 2> coeffs, FindRootsOptions o) const {
    detail::check_coeffs(coeffs.extent(1));
    detail::check_tol(o.param_tol, "param_tol");
    detail::check_tol(o.value_tol, "value_tol");
    const std::size_t n_polys = coeffs.extent(0);
    const std::size_t degree = coeffs.extent(1) - 1;
    const std::size_t width = (std::max)(degree, std::size_t{1});
    BatchRootSet out{NdArray<double>({n_polys, width}, for_overwrite),
                     std::vector<std::int64_t>(n_polys, 0),
                     std::vector<Completeness>(n_polys, Completeness::complete)};
    std::fill(out.roots.flat().begin(), out.roots.flat().end(),
              std::numeric_limits<double>::quiet_NaN());
    if (degree < 1) {
      return out;
    }
    double* roots = out.roots.data();
    parallel::parallel_for(n_polys, parallel::WorkEstimate{detail::ops_per_polynomial(degree)},
                           [&](std::size_t begin, std::size_t end) {
                             for (std::size_t i = begin; i < end; ++i) {
                               const std::span<const T> row(&coeffs(i, 0), degree + 1);
                               const RootSet<double> r =
                                   detail::dispatch_batch_row<T>(row, o.param_tol, o.value_tol);
                               // At most `degree` distinct roots; the clamp keeps a merge artefact
                               // inside the row.
                               const std::size_t count = (std::min)(r.roots.size(), width);
                               for (std::size_t j = 0; j < count; ++j) {
                                 roots[i * width + j] = r.roots[j].x;
                               }
                               out.counts[i] = static_cast<std::int64_t>(count);
                               out.completeness[i] = r.completeness;
                             }
                           });
    return out;
  }

  /// Find the root of a polynomial known to be monotone on `[0, 1]`.
  ///
  /// Args:
  ///     coeff (std::span<const T>): Coefficients, at least one.
  ///     param_tol (double): Bracket width at which to stop.
  ///
  /// Returns:
  ///     double: The root, or NaN when the end values do not change sign.
  ///
  /// Raises:
  ///     value_error: `coeff` is empty, or `param_tol` is not positive.
  template <StorageFloat T>
  [[nodiscard]] double find_monotone_root(std::span<const T> coeff, double param_tol) const {
    detail::check_coeffs(coeff.size());
    detail::check_tol(param_tol, "param_tol");
    return detail::solve_monotone<T>(coeff, param_tol);
  }

  /// Find the monotone root of each row, in parallel over rows.
  ///
  /// Args:
  ///     coeffs (mdview<const T, 2>): One polynomial per row, at least one column.
  ///     param_tol (double): Bracket width at which to stop.
  ///
  /// Returns:
  ///     std::vector<double>: One root per row, NaN where there is none.
  ///
  /// Raises:
  ///     value_error: No column, or `param_tol` is not positive.
  template <StorageFloat T>
  [[nodiscard]] std::vector<double> find_monotone_roots_batch(mdview<const T, 2> coeffs,
                                                              double param_tol) const {
    detail::check_coeffs(coeffs.extent(1));
    detail::check_tol(param_tol, "param_tol");
    const std::size_t n_polys = coeffs.extent(0);
    const std::size_t n1 = coeffs.extent(1);
    std::vector<double> out(n_polys);
    parallel::parallel_for(n_polys, parallel::WorkEstimate{detail::ops_per_polynomial(n1 - 1)},
                           [&](std::size_t begin, std::size_t end) {
                             for (std::size_t i = begin; i < end; ++i) {
                               out[i] = detail::solve_monotone<T>(
                                   std::span<const T>(&coeffs(i, 0), n1), param_tol);
                             }
                           });
    return out;
  }
};

static_assert(RootProvider<Builtin, double> && RootProvider<Builtin, float>);

/// The scalar coefficients whose roots a Bézier's root finding seeks: the control points of a
/// non-rational scalar curve, the first homogeneous component of a rational one.
///
/// Args:
///     b (const bezier::Bezier<T>&): The Bézier.
///
/// Returns:
///     std::vector<T>: `degree + 1` coefficients.
///
/// Raises:
///     value_error: `b.dim() != 1` or `b.rank() != 1` (the Python messages).
template <StorageFloat T>
[[nodiscard]] std::vector<T> scalar_coefficients(const bezier::Bezier<T>& b) {
  if (b.dim() != 1) {
    throw value_error("Bezier must be 1-D (dim == 1), got dim=" + std::to_string(b.dim()));
  }
  if (b.rank() != 1) {
    throw value_error("Bezier must be scalar (rank == 1), got rank=" + std::to_string(b.rank()));
  }
  const std::size_t stride = b.cp_size();
  const std::span<const T> cp = b.control_points();
  std::vector<T> out(b.degree(0) + 1);
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = cp[i * stride];
  }
  return out;
}

/// Find the roots in `[0, 1]` of a scalar 1D Bézier (design §4.1's entry point).
///
/// Args:
///     b (const bezier::Bezier<T>&): A curve with `dim == 1` and `rank == 1`.
///     o (FindRootsOptions): Tolerances.
///     provider (const P&): The provider. Defaults to `Builtin`.
///
/// Returns:
///     RootSet<double>: The provider's result.
///
/// Raises:
///     value_error: As `scalar_coefficients`, or as the provider.
template <class P = Builtin, StorageFloat T>
  requires RootProvider<P, T>
[[nodiscard]] RootSet<double> find_roots(const bezier::Bezier<T>& b, FindRootsOptions o,
                                         const P& provider = {}) {
  const std::vector<T> c = scalar_coefficients(b);
  return provider.find_roots(std::span<const T>(c), o);
}

}  // namespace pantr::roots
