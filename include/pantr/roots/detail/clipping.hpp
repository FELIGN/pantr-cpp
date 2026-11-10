/// @file clipping.hpp
/// @brief Bezier clipping and the derivative-aware merge of its candidates (Layer 3 of the Builtin
///        provider).
///
/// Bezier clipping (Nishita, Sederberg and Kakimoto, *Ray Tracing Trimmed Rational Surface
/// Patches*, 1990) narrows the parameter interval to where the convex hull of the control polygon
/// meets zero, subdividing when a clip removes too little. It is ported from the Numba kernel step
/// by step (see `scalar.hpp` for the precision rules), with two differences that only matter where
/// the Numba kernel lost work silently (design §2.7): the work stack and the candidate list grow
/// instead of being fixed buffers (the stack bound was unreachable, see `clip_roots`), and an
/// interval abandoned at the depth cap is reported as a gap instead of passing unremarked. None of
/// these functions validates anything.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/roots/detail/scalar.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace pantr::roots::detail {

/// Smallest fraction of an interval a clip must remove to be kept without subdividing. A
/// heuristic, kept from the previous release.
inline constexpr double clip_reduction_threshold = 0.2;

/// Depth at which an interval is given up on: treated as converged (polished, its candidate kept
/// if the residual allows) and reported as a gap. An admitted bound, kept from the previous
/// release; 64 halvings would take any interval below the spacing of doubles, but a clip may
/// remove as little as `clip_reduction_threshold` per level.
inline constexpr int clip_max_depth = 64;

/// Factor of the de Casteljau evaluation error bound `scale * (n + 1) * 4 * eps`, the zero test of
/// clipping and of the merge (an admitted factor, kept from the previous release).
inline constexpr double clip_eval_error_factor = 4.0;

/// Bisection steps of the refinement at a candidate whose derivative vanishes (a multiple root).
inline constexpr int double_root_bisections = 10;

/// What clipping found: the unsorted candidates and the intervals given up on at the depth cap.
///
/// Attributes:
///     candidates (std::vector<double>): Root candidates, unsorted, possibly repeated.
///     gaps (std::vector<std::pair<double, double>>): Intervals abandoned at `clip_max_depth`,
///         in the order they were met.
struct ClipResult {
  std::vector<double> candidates;
  std::vector<std::pair<double, double>> gaps;
};

/// The zero tolerance `max(scale * (n + 1) * 4 * eps, value_floor)` of a polynomial of degree `n`
/// whose largest coefficient magnitude is `scale`.
[[nodiscard]] inline double zero_tolerance(double scale, std::size_t n,
                                           double value_floor) noexcept {
  // The dimensionless factor first: `scale * (n + 1)` would overflow for a `scale` near the
  // largest double whose bound is representable.
  const double bound = scale * (static_cast<double>(n + 1) * clip_eval_error_factor * dbl_epsilon);
  return (std::max)(bound, value_floor);
}

/// The zero tolerance of a degree-`n` polynomial relative to its largest coefficient magnitude,
/// `max((n + 1) * 4 * eps, value_rel)`: `zero_tolerance` divided by the scale, formed without the
/// division (finite for an infinite or zero scale), a length on `[0, 1]` for the merge radii.
[[nodiscard]] inline double relative_zero_tolerance(std::size_t n, double value_rel) noexcept {
  const double bound = static_cast<double>(n + 1) * clip_eval_error_factor * dbl_epsilon;
  return (std::max)(bound, value_rel);
}

/// The largest coefficient magnitude, NaN skipped (the scale every value-frame bound of the
/// provider is relative to).
///
/// Args:
///     coeff (std::span<const S>): Coefficients.
///
/// Returns:
///     double: `max |c_i|` over the non-NaN coefficients, 0 when there are none.
template <StorageFloat S>
[[nodiscard]] double coefficient_scale(std::span<const S> coeff) noexcept {
  double scale = 0.0;
  for (const S c : coeff) {
    scale = (std::max)(scale, std::abs(static_cast<double>(c)));  // NaN: kept the left operand
  }
  return scale;
}

/// The residual a point of `[lo, hi]` may leave behind when the bracket holds a root: `|f'|`
/// over the bracket times its width, bounded by `n max_i |c_{i+1} - c_i|` of the coefficients
/// restricted to it (the Bernstein bound `|f'| <= n max |c_{i+1} - c_i| / (hi - lo)`), and at
/// least `|df| (hi - lo)`. The derivative at one point alone vanishes between two close roots.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat S>
[[nodiscard]] double bracket_allowance(std::span<const S> coeff, double lo, double hi, double df) {
  const std::vector<double> local = subdivide<S>(coeff, lo, hi);
  double steepest = 0.0;
  for (std::size_t i = 0; i + 1 < local.size(); ++i) {
    steepest = (std::max)(steepest, std::abs(local[i + 1] - local[i]));
  }
  const auto degree = static_cast<double>(local.size() - 1);
  return (std::max)(std::abs(df) * (hi - lo), degree * steepest);
}

/// Find root candidates of a Bernstein polynomial by Bezier clipping.
///
/// Every sub-interval is subdivided from the original coefficients (no accumulated subdivision
/// error). The stack is a depth-first one: each pop pushes at most two intervals one level deeper,
/// so it never holds more than `clip_max_depth + 2` entries; the previous release's fixed 256
/// could not bind.
///
/// Every bound in the value frame (the zero tolerances, the flatness floor, the derivative test of
/// a multiple root) is a multiple of the largest coefficient magnitude, so the candidates are the
/// same bits for `c` and `2^k * c` wherever neither under- nor overflows.
///
/// Args:
///     root_coeff (std::span<const S>): Coefficients on `[0, 1]`, degree `n >= 1`.
///     param_tol (double): Interval width at which an interval counts as converged.
///     value_rel (double): Floor of the zero tolerances, relative to the largest `|c|`.
///     max_depth (int): Depth cap (see `clip_max_depth`).
///
/// Returns:
///     ClipResult: The candidates and the gaps.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat S>
[[nodiscard]] ClipResult clip_roots(  // NOLINT(readability-function-cognitive-complexity)
    std::span<const S> root_coeff, double param_tol, double value_rel,
    int max_depth = clip_max_depth) {
  const std::size_t n = root_coeff.size() - 1;
  ClipResult res;
  const double coeff_scale = coefficient_scale(root_coeff);
  const double value_floor = value_rel * coeff_scale;
  const double deriv_floor = dbl_epsilon * coeff_scale;
  const double zero_tol = zero_tolerance(coeff_scale, n, value_floor);
  if (std::abs(static_cast<double>(root_coeff[0])) <= zero_tol) {
    res.candidates.push_back(0.0);
  }
  if (std::abs(static_cast<double>(root_coeff[n])) <= zero_tol) {
    res.candidates.push_back(1.0);
  }

  struct Item {
    double lo;
    double hi;
    int depth;
  };
  std::vector<Item> stack{{0.0, 1.0, 0}};
  const auto push_halves = [&](double lo, double hi, int depth) {
    const double mid = 0.5 * (lo + hi);
    stack.push_back({lo, mid, depth + 1});
    stack.push_back({mid, hi, depth + 1});
  };

  while (!stack.empty()) {
    const Item item = stack.back();
    stack.pop_back();
    const double lo = item.lo;
    const double hi = item.hi;
    const double span = hi - lo;

    // Step 1: convergence (or the depth cap).
    if (span <= param_tol || item.depth > max_depth) {
      if (span > param_tol) {
        res.gaps.emplace_back(lo, hi);
      }
      double mid = 0.5 * (lo + hi);
      if (lo <= 0.0 && hi <= param_tol) {
        mid = 0.0;
      } else if (lo >= 1.0 - param_tol && hi >= 1.0) {
        mid = 1.0;
      }
      const Polished p = newton_polish<S>(root_coeff, mid, lo, hi, param_tol, deriv_floor);
      mid = p.t;
      // A point within `span` of a root leaves up to `|f'| * span` behind, however exactly it is
      // evaluated: the interval was narrowed to `param_tol`, not to the zero test. Both terms are
      // values, so the test scales with the coefficients (bspline/roots.hpp's `root_tol`).
      const double accept = zero_tol + bracket_allowance<S>(root_coeff, lo, hi, p.df);
      if (std::abs(p.f) <= accept) {
        if (std::abs(p.df) <= deriv_floor) {
          // f' near zero (a multiple root): bisection on |f|.
          double a = lo;
          double b = hi;
          double fa = eval<S>(root_coeff, a);
          for (int it = 0; it < double_root_bisections; ++it) {
            const double m = 0.5 * (a + b);
            const double fm = eval<S>(root_coeff, m);
            if (std::abs(fm) < std::abs(fa)) {
              if (spans_zero(fa, fm)) {
                b = m;
              } else {
                a = m;
                fa = fm;
              }
            } else if (spans_zero(fa, fm)) {
              b = m;
            } else {
              a = m;
              fa = fm;
            }
            if (b - a <= dbl_epsilon) {
              break;
            }
          }
          mid = 0.5 * (a + b);
        }
        if (std::abs(eval<S>(root_coeff, mid)) <= accept) {
          res.candidates.push_back(mid);
        }
      }
      continue;
    }

    // Step 2: local coefficients and their zero tolerance.
    const std::vector<double> local = subdivide<S>(root_coeff, lo, hi);
    double local_scale = 0.0;
    for (const double v : local) {
      local_scale = (std::max)(local_scale, std::abs(v));
    }
    const double local_zero_tol = zero_tolerance(local_scale, n, value_floor);

    // Step 3: coefficient range.
    double c_min = local[0];
    double c_max = local[0];
    for (std::size_t i = 1; i <= n; ++i) {
      c_min = (std::min)(c_min, local[i]);
      c_max = (std::max)(c_max, local[i]);
    }

    // Step 4: quick rejection (one sign), with a polish when the margin is within the noise.
    if (c_min > local_zero_tol || c_max < -local_zero_tol) {
      const double rejection_margin = c_min > local_zero_tol ? c_min : -c_max;
      if (rejection_margin <= zero_tol) {
        const Polished p = newton_polish<S>(root_coeff, 0.5 * (lo + hi), lo, hi, 0.0, deriv_floor);
        if (std::abs(p.f) <= zero_tol) {
          res.candidates.push_back(p.t);
        }
      }
      continue;
    }

    // Step 5: a polynomial flat within the noise.
    if (c_max - c_min <= value_floor) {
      if (std::abs(c_min) <= local_zero_tol || std::abs(c_max) <= local_zero_tol ||
          have_opposite_signs(c_min, c_max)) {
        const double mid = 0.5 * (lo + hi);
        if (std::abs(eval<S>(root_coeff, mid)) <= zero_tol) {
          res.candidates.push_back(mid);
        }
      }
      continue;
    }

    // Step 6: endpoint roots.
    if (std::abs(local[0]) <= local_zero_tol) {
      res.candidates.push_back(lo);
    }
    if (std::abs(local[n]) <= local_zero_tol) {
      res.candidates.push_back(hi);
    }

    // Step 7: sign changes.
    const int n_sc = count_sign_changes(local);
    if (n_sc == 0) {
      continue;
    }

    // Step 8: the linear case.
    if (n == 1) {
      const double c0 = local[0];
      const double c1 = local[1];
      if (c0 != c1) {
        const double r = c0 / (c0 - c1);
        if (0.0 <= r && r <= 1.0) {
          const double offset = r * span;
          res.candidates.push_back(lo + offset);
        }
      }
      continue;
    }

    // Step 9: convex-hull clip, or halve when the hull misses zero.
    const HullClip clip = clip_hull_to_zero(local);
    if (!clip.found) {
      push_halves(lo, hi, item.depth);
      continue;
    }
    // A safety margin against rounding in the hull crossing (admitted, as the zero tolerance).
    const double margin = static_cast<double>(n + 1) * clip_eval_error_factor * dbl_epsilon;
    const double t_lo_safe = (std::max)(clip.t_lo - margin, 0.0);
    const double t_hi_safe = (std::min)(clip.t_hi + margin, 1.0);
    const double lo_offset = t_lo_safe * span;
    const double hi_offset = t_hi_safe * span;
    const double new_lo = lo + lo_offset;
    const double new_hi = lo + hi_offset;
    const double new_span = new_hi - new_lo;

    if (new_span <= param_tol) {
      const Polished p = newton_polish<S>(root_coeff, 0.5 * (new_lo + new_hi), new_lo, new_hi,
                                          param_tol, deriv_floor);
      // As at convergence above: within `new_span` of a root the residual is at most `|f'|` over
      // the bracket times it.
      if (std::abs(p.f) <= zero_tol + bracket_allowance<S>(root_coeff, new_lo, new_hi, p.df)) {
        res.candidates.push_back(p.t);
      }
      continue;
    }

    // Step 10: keep the clip, or subdivide it, or subdivide the interval.
    const double reduction = span > 0.0 ? 1.0 - (new_span / span) : 0.0;
    if (n_sc == 1) {
      if (reduction >= clip_reduction_threshold) {
        stack.push_back({new_lo, new_hi, item.depth + 1});
      } else {
        push_halves(new_lo, new_hi, item.depth);
      }
    } else if (reduction >= clip_reduction_threshold) {
      push_halves(new_lo, new_hi, item.depth);
    } else {
      push_halves(lo, hi, item.depth);
    }
  }
  return res;
}

/// Sort root candidates and merge the duplicates, with a merge radius that widens where the
/// derivative is small.
///
/// The same root reached from several intervals lands within `O(zero_tol / |f'(root)|)` of
/// itself, so a candidate is merged into the last kept root when the gap is at most
/// `max(base, min(4 * zero_tol / max(|f'|, eps * scale), z^(1/3)))`, `base = max(2 * param_tol,
/// 4 * z)`, with `scale = max|c|` and `z = zero_tol / scale` the relative zero tolerance. Each term
/// is a length on `[0, 1]`: `zero_tol / |f'|` is a value over a slope, and `z` is dimensionless,
/// so the merge is the same for `c` and `2^k * c`. The cube root caps the radius at a multiple
/// root, where it would diverge: within `O(z^(1/m))` of a root of multiplicity `m` the residual is
/// under `zero_tol`, so the cap covers multiplicities up to three (the design note's pair-class
/// table says which duplicate classes this layer removes).
///
/// Args:
///     raw (std::vector<double>): The candidates; consumed.
///     coeff (std::span<const S>): Coefficients, for the derivative.
///     param_tol (double): Parametric tolerance.
///     value_rel (double): Floor of the zero tolerance, relative to the largest `|c|`.
///
/// Returns:
///     std::vector<double>: The kept roots, ascending.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat S>
[[nodiscard]] std::vector<double> dedup_roots(std::vector<double> raw, std::span<const S> coeff,
                                              double param_tol, double value_rel) {
  if (raw.empty()) {
    return raw;
  }
  const std::size_t n = coeff.size() - 1;
  const double coeff_scale = coefficient_scale(coeff);
  const double zero_tol = zero_tolerance(coeff_scale, n, value_rel * coeff_scale);
  const double zero_rel = relative_zero_tolerance(n, value_rel);
  const double base_dedup = (std::max)(param_tol * 2.0, zero_rel * 4.0);
  const double radius_cap = std::pow(zero_rel, 1.0 / 3.0);
  const double slope_floor = dbl_epsilon * coeff_scale;
  // Insertion sort, as the previous release: stable, and the order of equal keys is moot.
  for (std::size_t i = 1; i < raw.size(); ++i) {
    const double key = raw[i];
    std::size_t j = i;
    while (j > 0 && raw[j - 1] > key) {
      raw[j] = raw[j - 1];
      --j;
    }
    raw[j] = key;
  }
  std::size_t count = 1;
  for (std::size_t i = 1; i < raw.size(); ++i) {
    const double gap = raw[i] - raw[count - 1];
    if (gap <= base_dedup) {
      continue;
    }
    const double df = eval_and_deriv<S>(coeff, raw[count - 1]).second;
    const double local_tol = zero_tol / (std::max)(std::abs(df), slope_floor);
    if (gap <= (std::max)(base_dedup, (std::min)(local_tol * 4.0, radius_cap))) {
      continue;
    }
    raw[count++] = raw[i];
  }
  raw.resize(count);
  return raw;
}

}  // namespace pantr::roots::detail
