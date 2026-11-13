/// @file degree.hpp
/// @brief Degree elevation (Piegl and Tiller A5.9) and degree reduction of a B-spline.
///
/// Elevation follows *The NURBS Book* (Piegl and Tiller, Algorithm A5.9) and the MATLAB NURBS
/// Toolbox layer. Reduction decomposes the spline into Bézier segments and applies a precomputed
/// reduction operator (`bezier::cached_reduction_operator`) to each. A periodic direction is
/// converted to open form, processed and converted back, so that periodicity is preserved.
///
/// The kernels compare knots with `==`, where the rest of the library compares them within the
/// space's tolerance. The two agree on every knot vector a `BsplineSpace1D` holds: the space snaps
/// the members of a knot class to its first knot at construction, so two knots of one class are
/// bit-equal and two of different classes differ by more than the tolerance.
#pragma once

#include "pantr/bezier/reduce.hpp"
#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/detail/refine_common.hpp"
#include "pantr/bspline/knot_insertion.hpp"
#include "pantr/bspline/knot_removal.hpp"
#include "pantr/bspline/knots.hpp"
#include "pantr/bspline/periodic.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/binomial.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bspline {

/// Report which ends of a knot vector carry a run of `degree + 1` equal knots.
///
/// Bit-identity rather than a tolerance, because a run is what `detail::degree_elevate_kernel`
/// finds by comparing knots with `==`: a run that ties only to within a tolerance is no run to the
/// algorithm, whatever it is to a reader.
///
/// At `degree == 0` both runs are one knot long and the answer is `(true, true)` for any vector.
/// That is right rather than vacuous: a degree-0 curve's domain is the whole knot vector, so the
/// kernel's opening assumption holds by construction and its segment walk terminates on the
/// amended `b <= m` bound rather than on a closing run.
///
/// Args:
///     knots (std::span<const T>): A knot vector of `num_coefficients + degree + 1` entries. The
///         caller guarantees that length and `degree + 1 <= knots.size()`; nothing here checks it.
///     degree (int): The degree that vector carries.
///
/// Returns:
///     std::pair<bool, bool>: Whether the vector opens, and whether it closes, with a run of
///     `degree + 1` bit-identical knots.
template <StorageFloat T>
[[nodiscard]] std::pair<bool, bool> clamped_ends(std::span<const T> knots, int degree) {
  const auto run = static_cast<std::size_t>(degree) + 1;
  const T lo = knots.front();
  const T hi = knots.back();
  bool opens = true;
  bool closes = true;
  for (std::size_t i = 0; i < run; ++i) {
    opens = opens && knots[i] == lo;
    closes = closes && knots[knots.size() - run + i] == hi;
  }
  return {opens, closes};
}

/// Refuse a knot vector the degree kernels cannot walk: one that is not clamped, or that repeats a
/// knot more than `degree + 1` times.
///
/// A5.9 needs a run of `degree + 1` equal knots at **each** end, and the kernels check for neither:
///
///   * the **closing** run is what ends the segment walk. Without it the walk steps past the
///     coefficient array.
///   * the **opening** run is what makes the first `degree + 1` control points the Bézier form of
///     the first segment, which the kernel assumes when it seeds `bpts`. Without it the walk stays
///     in bounds and nothing is raised: the kernel opens its output with `degree + increment + 1`
///     copies of `knots[0]`, so the result is a different function, over a domain starting at the
///     vector's first knot rather than at the original domain's own start.
///
/// And no knot may repeat more than `degree + 1` times. The walk takes `degree - multiplicity` as
/// the number of Bézier segments' worth of control points to carry over, so a run of
/// `degree + 2` or more (a space accepts one: the basis function over the empty interval is
/// zero) makes that number less than -1, and the source then wrote below its working arrays. A
/// derivative whose original has a `C^-1` knot (multiplicity `degree + 1`) hands the kernel such a
/// run, at degree `degree - 1`.
///
/// Args:
///     knots (std::span<const T>): The knot vector about to be handed to a kernel.
///     degree (int): The degree that vector carries.
///     what (std::string_view): Description of the requested operation, for the message.
///     elevating (bool): Whether the operation is an elevation (the message names A5.9 and the
///         elevation) or a reduction.
///
/// Raises:
///     value_error: Either end run is not bit-identical; or a knot repeats more than
///         `degree + 1` times.
template <StorageFloat T>
void check_clamped_knots(std::span<const T> knots, int degree, std::string_view what,
                         bool elevating = true) {
  const auto [opens, closes] = clamped_ends<T>(knots, degree);
  if (!opens || !closes) {
    std::string ends;
    if (!opens) {
      ends = "open";
    }
    if (!closes) {
      ends += ends.empty() ? "close" : " or close";
    }
    const std::string run = std::to_string(degree + 1);
    if (elevating) {
      throw value_error(
          std::string(what) + " needs a clamped knot vector: A5.9 takes the first " + run +
          " coefficients as the first Bézier segment and walks segments until a run of equal "
          "knots reaches the last knot, so it needs a run of " +
          run + " equal knots at each end, and this vector does not " + ends +
          " with one. Elevating an unclamped direction is not supported; "
          "`Bspline.to_open_bspline` converts one to a clamped form that can be elevated.");
    }
    throw value_error(
        std::string(what) +
        " needs a clamped knot vector: the decomposition into Bézier segments "
        "takes the first " +
        run +
        " coefficients as the first segment and walks segments until a run of equal knots reaches "
        "the last knot, so it needs a run of " +
        run + " equal knots at each end, and this vector does not " + ends +
        " with one. Reducing an unclamped direction is not supported; "
        "`Bspline.to_open_bspline` converts one to a clamped form that can be reduced.");
  }
  std::size_t run_start = 0;
  for (std::size_t i = 1; i <= knots.size(); ++i) {
    if (i < knots.size() && knots[i] == knots[run_start]) {
      continue;
    }
    if (i - run_start > static_cast<std::size_t>(degree) + 1) {
      throw value_error(std::string(what) + " cannot walk a knot vector that repeats the knot " +
                        detail::repr_of<T>(knots[run_start]) + " " + std::to_string(i - run_start) +
                        " times at degree " + std::to_string(degree) +
                        ": a knot may repeat at most degree + 1 = " + std::to_string(degree + 1) +
                        " times.");
    }
    run_start = i;
  }
}

namespace detail {

/// Degree elevate a B-spline curve by `increment` (A5.9, in the robust form of the source).
///
/// Implements a robust version of Piegl and Tiller Algorithm A5.9 (degree elevate a B-spline
/// curve). Elevation preserves smoothness: a breakpoint where the spline is `C^s` stays `C^s`, so
/// its multiplicity goes from `m` to `m + t`, which is what the knot-writing step emits. That
/// includes `m = degree + 1`, a `C^-1` breakpoint: the two adjacent Bézier segments share no
/// control point there, so the segment after it starts contributing at its own coefficient 0
/// rather than at 1. How many of its coefficients that pass writes still depends on the
/// breakpoint that *closes* the segment, through `rbz`, exactly as in the published algorithm.
///
/// Arithmetic, as the kernel this replaces: the Bézier coefficient table `bezalfs` and the blending
/// weights `alfs` are `double`; the knots, control points and the working rows are of the storage
/// type; the knot ratios are formed in it; a blend is formed in `double` where one factor is a
/// `double` and rounded to the storage type on every store.
///
/// Args:
///     degree (int): Original degree.
///     ctrl (mdview<const T, 2>): Control points, extents `(n_pts, rank)`.
///     knots (std::span<const T>): Knot vector, **clamped**: a run of `degree + 1` bit-identical
///         knots at each end. The closing run is what ends the segment walk and the opening one is
///         what makes the first `degree + 1` control points a Bézier segment; `check_clamped_knots`
///         is the check that establishes both.
///     increment (int): How much to increase the degree, at least 1.
///
/// Returns:
///     KnotsAndCtrl<T>: The new knot vector and the new control points `(n_new, rank)`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): a clamped knot vector, a net
///     matching it, and `degree + increment <= bincoeff_max_n`. For general use, call
///     `elevate_degree_1d`.
template <StorageFloat T>
[[nodiscard]] KnotsAndCtrl<T> degree_elevate_kernel(int degree, mdview<const T, 2> ctrl,
                                                    std::span<const T> knots, int increment) {
  using Index = std::ptrdiff_t;
  const std::size_t n_pts = ctrl.extent(0);
  const std::size_t rank = ctrl.extent(1);
  const Index d = degree;
  const Index t = increment;

  const auto n = static_cast<Index>(n_pts) - 1;
  const Index ph = d + t;
  const Index ph2 = ph / 2;

  const auto sz = [](Index v) { return static_cast<std::size_t>(v); };
  std::vector<double> bezalfs(sz((d + 1) * (ph + 1)), 0.0);
  std::vector<T> bpts(sz(d + 1) * rank, T{0});
  std::vector<T> ebpts(sz(ph + 1) * rank, T{0});
  std::vector<T> nextbpts(sz(d + 1) * rank, T{0});
  std::vector<double> alfs(sz(d), 0.0);
  const auto bz = [&](Index j, Index i) -> double& { return bezalfs[sz(j * (ph + 1) + i)]; };
  const auto row = [rank](std::vector<T>& v, Index r, std::size_t c) -> T& {
    return v[static_cast<std::size_t>(r) * rank + c];
  };

  const Index m = n + d + 1;

  // Compute the Bézier degree elevation coefficients.
  bz(0, 0) = 1.0;
  bz(d, ph) = 1.0;
  for (Index i = 1; i <= ph2; ++i) {
    const double inv = 1.0 / bincoeff(ph, i);
    const Index mpi = (std::min)(d, i);
    for (Index j = (std::max)(Index{0}, i - t); j <= mpi; ++j) {
      bz(j, i) = inv * bincoeff(d, j) * bincoeff(t, i - j);
    }
  }
  for (Index i = ph2 + 1; i < ph; ++i) {
    const Index mpi = (std::min)(d, i);
    for (Index j = (std::max)(Index{0}, i - t); j <= mpi; ++j) {
      bz(j, i) = bz(d - j, ph - i);
    }
  }

  Index kind = ph + 1;
  Index r = -1;
  Index a = d;
  Index b = d + 1;
  Index cind = 1;
  T ua = knots[0];

  // More than enough space for the new arrays. Both outputs follow their own input's storage type.
  const std::size_t max_new_knots = knots.size() + static_cast<std::size_t>(t) * knots.size();
  const std::size_t max_new_rows = n_pts + static_cast<std::size_t>(t) * knots.size();
  std::vector<T> ik(max_new_knots, T{0});
  std::vector<T> ic(max_new_rows * rank, T{0});

  for (std::size_t ii = 0; ii < rank; ++ii) {
    ic[ii] = ctrl(0, ii);
  }
  for (Index i = 0; i <= ph; ++i) {
    ik[sz(i)] = ua;
  }
  for (Index i = 0; i <= d; ++i) {
    for (std::size_t ii = 0; ii < rank; ++ii) {
      row(bpts, i, ii) = ctrl(sz(i), ii);
    }
  }

  // `b <= m` rather than A5.9's `b < m`: the loop body runs once per segment, indexed by the last
  // knot of the run that closes it, so the final segment needs `b == m`. For `d >= 1` the closing
  // block holds `d + 1` equal knots and the inner run scan reaches `m` on its own, which is why
  // A5.9 gets away with the strict bound; at `d == 0` that block is a single knot, the scan cannot
  // advance, and the strict bound drops the last segment together with the closing knots, leaving
  // a knot vector whose tail is still zero and a domain collapsed to a point. The `break` below is
  // what the strict bound used to provide.
  while (b <= m) {
    Index i = b;
    while (b < m && knots[sz(b)] == knots[sz(b + 1)]) {
      ++b;
    }

    const Index mul = b - i + 1;
    const T ub = knots[sz(b)];
    const Index oldr = r;
    r = d - mul;

    // The first elevated Bézier coefficient this segment contributes. A5.9 assumes interior knots
    // of multiplicity at most d, where consecutive segments meet C^0 and the junction coefficient
    // has already been written, so it starts at 1. A knot of multiplicity d + 1 leaves oldr = -1:
    // the spline is C^-1 there, the segments share nothing, and skipping index 0 would drop a
    // control point. The first pass through the loop also has oldr = -1 but a == d, and its index 0
    // really was written, before the loop.
    Index lbz = 1;
    if (oldr > 0) {
      lbz = (oldr + 2) / 2;
    } else if (oldr < 0 && a != d) {
      lbz = 0;
    }

    const Index rbz = r > 0 ? ph - (r + 1) / 2 : ph;

    if (r > 0) {
      const T numer = static_cast<T>(ub - ua);
      for (Index q = d; q > mul; --q) {
        alfs[sz(q - mul - 1)] = static_cast<double>(numer / static_cast<T>(knots[sz(a + q)] - ua));
      }

      for (Index j = 1; j <= r; ++j) {
        const Index save = r - j;
        const Index s = mul + j;

        for (Index q = d; q >= s; --q) {
          for (std::size_t ii = 0; ii < rank; ++ii) {
            const double tmp1 = alfs[sz(q - s)] * static_cast<double>(row(bpts, q, ii));
            const double tmp2 = (1.0 - alfs[sz(q - s)]) * static_cast<double>(row(bpts, q - 1, ii));
            row(bpts, q, ii) = static_cast<T>(tmp1 + tmp2);
          }
        }

        for (std::size_t ii = 0; ii < rank; ++ii) {
          row(nextbpts, save, ii) = row(bpts, d, ii);
        }
      }
    }

    for (Index k = lbz; k <= ph; ++k) {
      for (std::size_t ii = 0; ii < rank; ++ii) {
        row(ebpts, k, ii) = T{0};
      }

      const Index mpi = (std::min)(d, k);
      for (Index j = (std::max)(Index{0}, k - t); j <= mpi; ++j) {
        for (std::size_t ii = 0; ii < rank; ++ii) {
          const double tmp2 = bz(j, k) * static_cast<double>(row(bpts, j, ii));
          row(ebpts, k, ii) = static_cast<T>(static_cast<double>(row(ebpts, k, ii)) + tmp2);
        }
      }
    }

    if (oldr > 1) {
      Index first = kind - 2;
      Index last = kind;
      const T den = static_cast<T>(ub - ua);
      const T bet = static_cast<T>(ub - ik[sz(kind - 1)]) / den;

      for (Index tr = 1; tr < oldr; ++tr) {
        Index ii_i = first;
        Index jj = last;
        Index kj = jj - kind + 1;
        while (jj - ii_i > tr) {
          if (ii_i < cind) {
            const T alf = static_cast<T>(ub - ik[sz(ii_i)]) / static_cast<T>(ua - ik[sz(ii_i)]);
            for (std::size_t ii = 0; ii < rank; ++ii) {
              const T tmp1 = static_cast<T>(alf * ic[sz(ii_i) * rank + ii]);
              const double tmp2 = (1.0 - static_cast<double>(alf)) *
                                  static_cast<double>(ic[sz(ii_i - 1) * rank + ii]);
              ic[sz(ii_i) * rank + ii] = static_cast<T>(static_cast<double>(tmp1) + tmp2);
            }
          }

          if (jj >= lbz) {
            if (jj - tr <= kind - ph + oldr) {
              const T gam = static_cast<T>(ub - ik[sz(jj - tr)]) / den;
              for (std::size_t ii = 0; ii < rank; ++ii) {
                const T tmp1 = static_cast<T>(gam * row(ebpts, kj, ii));
                const double tmp2 =
                    (1.0 - static_cast<double>(gam)) * static_cast<double>(row(ebpts, kj + 1, ii));
                row(ebpts, kj, ii) = static_cast<T>(static_cast<double>(tmp1) + tmp2);
              }
            } else {
              for (std::size_t ii = 0; ii < rank; ++ii) {
                const T tmp1 = static_cast<T>(bet * row(ebpts, kj, ii));
                const double tmp2 =
                    (1.0 - static_cast<double>(bet)) * static_cast<double>(row(ebpts, kj + 1, ii));
                row(ebpts, kj, ii) = static_cast<T>(static_cast<double>(tmp1) + tmp2);
              }
            }
          }

          ++ii_i;
          --jj;
          --kj;
        }

        --first;
        ++last;
      }
    }

    if (a != d) {
      for (Index k = 0; k < ph - oldr; ++k) {
        ik[sz(kind)] = ua;
        ++kind;
      }
    }

    for (Index j = lbz; j <= rbz; ++j) {
      for (std::size_t ii = 0; ii < rank; ++ii) {
        ic[sz(cind) * rank + ii] = row(ebpts, j, ii);
      }
      ++cind;
    }

    if (b < m) {
      for (Index j = 0; j < r; ++j) {
        for (std::size_t ii = 0; ii < rank; ++ii) {
          row(bpts, j, ii) = row(nextbpts, j, ii);
        }
      }
      // `r = -1` (a knot of multiplicity `d + 1`) starts the source's loop at index -1, which wraps
      // to the last row of `bpts` and is overwritten by the last pass of the loop: it starts at 0.
      for (Index j = (std::max)(r, Index{0}); j <= d; ++j) {
        for (std::size_t ii = 0; ii < rank; ++ii) {
          row(bpts, j, ii) = ctrl(sz(b - d + j), ii);
        }
      }
      a = b;
      ++b;
      ua = ub;
    } else {
      for (Index k = 0; k <= ph; ++k) {
        ik[sz(kind + k)] = ub;
      }
      break;
    }
  }

  NdArray<T> out_ctrl({sz(cind), rank}, for_overwrite);
  std::copy(ic.begin(), ic.begin() + static_cast<std::ptrdiff_t>(sz(cind) * rank), out_ctrl.data());
  std::vector<T> out_knots(ik.begin(), ik.begin() + static_cast<std::ptrdiff_t>(sz(kind + ph + 1)));
  return KnotsAndCtrl<T>{std::move(out_knots), std::move(out_ctrl)};
}

/// Count the Bézier segments a reduction sweep will produce, and the jumps among them.
///
/// Counted with the sweep's own advance rule, so the output arrays can be sized exactly (a bound
/// in terms of the number of knots alone is not an upper bound at high degree, where a short vector
/// still yields more segments than that bound allows, and the kernel would then write past the
/// end).
///
/// Args:
///     knots (std::span<const T>): The clamped knot vector.
///     degree (int): The degree.
///     m (std::ptrdiff_t): The last knot index.
///
/// Returns:
///     std::pair<std::size_t, std::size_t>: `(n_seg, n_jump)`, the segments and the interior
///     breakpoints of multiplicity `degree + 1`.
template <StorageFloat T>
[[nodiscard]] std::pair<std::size_t, std::size_t> count_reduction_segments(std::span<const T> knots,
                                                                           int degree,
                                                                           std::ptrdiff_t m) {
  std::size_t n_seg = 0;
  std::size_t n_jump = 0;
  std::ptrdiff_t scan = static_cast<std::ptrdiff_t>(degree) + 1;
  while (scan < m) {
    const std::ptrdiff_t run_start = scan;
    while (scan < m &&
           knots[static_cast<std::size_t>(scan)] == knots[static_cast<std::size_t>(scan) + 1]) {
      ++scan;
    }
    if (scan < m && scan - run_start + 1 > degree) {
      ++n_jump;
    }
    ++n_seg;
    ++scan;
  }
  return {n_seg, n_jump};
}

/// Degree reduce a B-spline curve by `decrement`.
///
/// Decomposes the B-spline into Bézier segments by iterating through knot spans (the same
/// alpha-blending as the elevation kernel), applies the reduction operator to each segment, and
/// stitches the results into a B-spline in Bézier form.
///
/// Because the operator interpolates the segment endpoints, the last control point of one reduced
/// segment and the first of the next are both the shared breakpoint value of the *original*
/// spline, bit for bit. The junction is therefore written once and the stitch is exactly C0; no
/// averaging of the two sides is involved, and neither segment is perturbed away from its own
/// optimum.
///
/// That argument needs a breakpoint the original spline is continuous at. Where it is not (an
/// interior knot of multiplicity `degree + 1`) the two segments have different endpoint values,
/// both are kept, and the breakpoint carries multiplicity `new_degree + 1` so that the reduced
/// spline jumps where the original does. Reduction preserves smoothness, exactly as elevation does:
/// `C^{degree - m} = C^{new_degree - (m - t)}`.
///
/// Arithmetic: the operator is `double`, the blend of a segment is formed in `double` and rounded
/// to the storage type on every store, and the operator is applied by
/// `bezier::detail::apply_operator`, which accumulates in `double` and rounds once.
///
/// Args:
///     degree (int): Original degree.
///     ctrl (mdview<const T, 2>): Control points, extents `(n_pts, rank)`.
///     knots (std::span<const T>): The clamped knot vector.
///     decrement (int): Degrees to reduce, `1 <= decrement <= degree`.
///     op (const NdArray<double>&): The operator, extents `(degree - decrement + 1, degree + 1)`.
///
/// Returns:
///     KnotsAndCtrl<T>: The knots and control points in Bézier form: every interior breakpoint has
///     multiplicity `new_degree`, or `new_degree + 1` where the input was discontinuous.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): a clamped knot vector and a net
///     matching it. For general use, call `reduce_degree_1d`.
template <StorageFloat T>
[[nodiscard]] KnotsAndCtrl<T> degree_reduce_kernel(int degree, mdview<const T, 2> ctrl,
                                                   std::span<const T> knots, int decrement,
                                                   const NdArray<double>& op) {
  using Index = std::ptrdiff_t;
  const std::size_t n_pts = ctrl.extent(0);
  const std::size_t rank = ctrl.extent(1);
  const Index d = degree;
  const Index t = decrement;
  const Index new_deg = d - t;

  const auto n = static_cast<Index>(n_pts) - 1;
  const Index m = n + d + 1;  // the last knot index
  const auto sz = [](Index v) { return static_cast<std::size_t>(v); };

  std::vector<T> bpts(sz(d + 1) * rank, T{0});
  std::vector<T> nextbpts(sz(d + 1) * rank, T{0});
  std::vector<double> alfs(sz(d), 0.0);
  const auto row = [rank](std::vector<T>& v, Index r, std::size_t c) -> T& {
    return v[static_cast<std::size_t>(r) * rank + c];
  };

  const auto [n_seg, n_jump] = count_reduction_segments<T>(knots, degree, m);

  // Both outputs follow their own input's storage type, as in the elevation kernel.
  const std::size_t out_rows = n_seg * sz(new_deg) + 1 + n_jump;
  std::vector<T> oc(out_rows * rank, T{0});
  std::vector<T> ok(n_seg * sz(new_deg) + sz(new_deg) + 2 + n_jump, T{0});

  // Initialise: the first boundary knots and the first Bézier segment.
  T ua = knots[0];
  for (Index i = 0; i <= new_deg; ++i) {
    ok[sz(i)] = ua;
  }
  Index kind = new_deg + 1;

  for (Index i = 0; i <= d; ++i) {
    for (std::size_t ii = 0; ii < rank; ++ii) {
      row(bpts, i, ii) = ctrl(sz(i), ii);
    }
  }

  Index cind = 0;  // the output control point index
  Index a = d;
  Index b = d + 1;
  Index r = -1;

  while (b < m) {
    // Find the next distinct knot.
    const Index i = b;
    while (b < m && knots[sz(b)] == knots[sz(b + 1)]) {
      ++b;
    }
    const Index mul = b - i + 1;
    const T ub = knots[sz(b)];
    const Index oldr = r;
    r = d - mul;

    // Extract the Bézier control points for the current span.
    if (r > 0) {
      const T numer = static_cast<T>(ub - ua);
      for (Index q = d; q > mul; --q) {
        alfs[sz(q - mul - 1)] = static_cast<double>(numer / static_cast<T>(knots[sz(a + q)] - ua));
      }

      for (Index j = 1; j <= r; ++j) {
        const Index save = r - j;
        const Index s = mul + j;
        for (Index q = d; q >= s; --q) {
          for (std::size_t ii = 0; ii < rank; ++ii) {
            row(bpts, q, ii) =
                static_cast<T>(alfs[sz(q - s)] * static_cast<double>(row(bpts, q, ii)) +
                               (1.0 - alfs[sz(q - s)]) * static_cast<double>(row(bpts, q - 1, ii)));
          }
        }
        for (std::size_t ii = 0; ii < rank; ++ii) {
          row(nextbpts, save, ii) = row(bpts, d, ii);
        }
      }
    }

    // Reduce the current Bézier segment.
    const NdArray<T> reduced =
        bezier::detail::apply_operator<T>(op, mdview<const T, 2>(bpts.data(), sz(d + 1), rank));

    // The first segment contributes all of its control points; every later one skips its first,
    // which the previous segment already wrote with the identical value. `oldr < 0` says the
    // previous breakpoint had multiplicity `d + 1`, so there is no shared value to skip: the two
    // sides of a jump are different numbers and both belong in the output.
    const Index start = (cind == 0 || oldr < 0) ? 0 : 1;
    for (Index j = start; j <= new_deg; ++j) {
      for (std::size_t ii = 0; ii < rank; ++ii) {
        oc[sz(cind) * rank + ii] = reduced.data()[sz(j) * rank + ii];
      }
      ++cind;
    }

    // Write the previous interior breakpoint: multiplicity new_deg for a C0 junction, new_deg + 1
    // where the spline jumps.
    if (a != d) {
      const Index n_write = oldr < 0 ? new_deg + 1 : new_deg;
      for (Index k = 0; k < n_write; ++k) {
        ok[sz(kind)] = ua;
        ++kind;
      }
    }

    // Prepare for the next span. (`b < m` holds here: the closing knots are written below.)
    if (b < m) {
      for (Index j = 0; j < r; ++j) {
        for (std::size_t ii = 0; ii < rank; ++ii) {
          row(bpts, j, ii) = row(nextbpts, j, ii);
        }
      }
      // `r = -1` (a knot of multiplicity `d + 1`) starts the source's loop at index -1, which wraps
      // to the last row of `bpts` and is overwritten by the last pass of the loop: it starts at 0.
      for (Index j = (std::max)(r, Index{0}); j <= d; ++j) {
        for (std::size_t ii = 0; ii < rank; ++ii) {
          row(bpts, j, ii) = ctrl(sz(b - d + j), ii);
        }
      }
      a = b;
      ++b;
      ua = ub;
    } else {
      // Write the closing knots.
      for (Index k = 0; k <= new_deg; ++k) {
        ok[sz(kind + k)] = ub;
      }
    }
  }

  NdArray<T> out_ctrl({sz(cind), rank}, for_overwrite);
  std::copy(oc.begin(), oc.begin() + static_cast<std::ptrdiff_t>(sz(cind) * rank), out_ctrl.data());
  std::vector<T> out_knots(ok.begin(),
                           ok.begin() + static_cast<std::ptrdiff_t>(sz(kind + new_deg + 1)));
  return KnotsAndCtrl<T>{std::move(out_knots), std::move(out_ctrl)};
}

}  // namespace detail

/// Degree elevate a 1D B-spline by `increment`.
///
/// Args:
///     degree (int): Original degree, `>= 0`.
///     ctrl (mdview<const T, 2>): Control points, extents `(n + 1, rank)`.
///     knots (std::span<const T>): A clamped knot vector, `n + degree + 2` entries.
///     increment (int): How much to increase the degree, `>= 1`.
///
/// Returns:
///     KnotsAndCtrl<T>: The elevated knot vector (every multiplicity grown by `increment`) and the
///     control points.
///
/// Raises:
///     value_error: The degree or the net does not match the knots; `increment < 1`;
///         `degree + increment` exceeds `bincoeff_max_n`; the knot vector is not clamped; or it
///         repeats a knot more than `degree + 1` times.
template <StorageFloat T>
[[nodiscard]] KnotsAndCtrl<T> elevate_degree_1d(int degree, mdview<const T, 2> ctrl,
                                                std::span<const T> knots, int increment) {
  detail::check_net_matches("degree elevation", ctrl.extent(0), knots.size(), degree);
  if (increment < 1) {
    throw value_error("degree increment must be at least 1, got " + std::to_string(increment));
  }
  const long long elevated = static_cast<long long>(degree) + increment;
  check_bincoeff_envelope(elevated, "Degree elevation to degree " + std::to_string(elevated));
  check_clamped_knots<T>(knots, degree, "Degree elevation");
  return detail::degree_elevate_kernel<T>(degree, ctrl, knots, increment);
}

/// Degree reduce a 1D B-spline by `decrement`, to Bézier form.
///
/// Args:
///     degree (int): Original degree, `>= 1`.
///     ctrl (mdview<const T, 2>): Control points, extents `(n + 1, rank)`.
///     knots (std::span<const T>): A clamped knot vector, `n + degree + 2` entries.
///     decrement (int): Degrees to reduce, `1 <= decrement <= degree`.
///
/// Returns:
///     KnotsAndCtrl<T>: The knots and control points in Bézier form (see
///     `detail::degree_reduce_kernel`).
///
/// Raises:
///     value_error: The degree pair is out of range, the net does not match the knots, the knot
///         vector is not clamped, or it repeats a knot more than `degree + 1` times.
template <StorageFloat T>
[[nodiscard]] KnotsAndCtrl<T> reduce_degree_1d(int degree, mdview<const T, 2> ctrl,
                                               std::span<const T> knots, int decrement) {
  detail::check_net_matches("degree reduction", ctrl.extent(0), knots.size(), degree);
  check_clamped_knots<T>(knots, degree, "Degree reduction", false);
  const auto op =
      bezier::cached_reduction_operator(bezier::ReductionKind::interpolating, degree, decrement);
  return detail::degree_reduce_kernel<T>(degree, ctrl, knots, decrement, *op);
}

/// Remove the excess knots of a Bézier-form B-spline after degree reduction.
///
/// After degree reduction the output is in Bézier form: every interior breakpoint has multiplicity
/// `new_degree`, or `new_degree + 1` where the original spline was discontinuous. This removes
/// excess copies of each interior knot so that the final multiplicity is `max(1, m_i - t)`, with
/// `m_i` the original multiplicity and `t` the degree decrement, which restores the original
/// continuity `C^{p - m_i} = C^{(p - t) - (m_i - t)}`.
///
/// A discontinuous breakpoint (`m_i = p + 1`) therefore keeps multiplicity `new_degree + 1` and
/// nothing is removed there. Clamping the target at `new_degree` instead would ask for a C0 space
/// where the function jumps, and the reduction would silently stop being exact on curves that
/// reduce exactly.
///
/// The removal is **forced**: the deviation budget is infinity, which does not loosen the kernel's
/// acceptance test (`dist <= tol`) but removes it. That is deliberate: this removal is structural,
/// restoring the continuity the original spline had, so the requested count is what must hold and
/// the deviation is not a criterion. `Bspline.reduce_degree` documents that this forced removal,
/// not the per-segment reduction, is what sets the final error.
///
/// Args:
///     knots (std::vector<T>): Knot vector of the Bézier-form reduced B-spline.
///     new_degree (int): Degree after reduction.
///     ctrl (NdArray<T>): Control points, extents `(n_pts, rank)`.
///     orig_unique (std::span<const T>): Unique interior knot values of the *original* B-spline
///         (before reduction).
///     orig_mults (std::span<const std::int64_t>): The corresponding multiplicities.
///     degree_decrement (int): How many degrees were reduced.
///     tol_space (double): Tolerance for the knot comparisons.
///
/// Returns:
///     KnotsAndCtrl<T>: The coarsened knots and control points.
template <StorageFloat T>
[[nodiscard]] KnotsAndCtrl<T> coarsen_knots_after_reduction(
    std::vector<T> knots, int new_degree, NdArray<T> ctrl, std::span<const T> orig_unique,
    std::span<const std::int64_t> orig_mults, int degree_decrement, double tol_space) {
  const long long orig_degree = static_cast<long long>(new_degree) + degree_decrement;
  for (std::size_t idx = 0; idx < orig_unique.size(); ++idx) {
    const auto knot_val = static_cast<double>(orig_unique[idx]);
    const long long m_orig = orig_mults[idx];
    const long long target_mult = (std::max)(1LL, m_orig - degree_decrement);
    // The multiplicity in the Bézier form: new_degree, or new_degree + 1 at a knot the original
    // spline was discontinuous at.
    const long long bezier_mult = m_orig > orig_degree ? new_degree + 1LL : new_degree;
    const long long num_to_remove = bezier_mult - target_mult;
    if (num_to_remove <= 0) {
      continue;
    }
    RemovalResult<T> removed =
        remove_knot_1d<T>(std::span<const T>(knots), new_degree, ctrl.template view<2>(), knot_val,
                          num_to_remove, tol_space, std::numeric_limits<double>::infinity());
    knots = std::move(removed.knots);
    ctrl = std::move(removed.ctrl);
  }
  return KnotsAndCtrl<T>{std::move(knots), std::move(ctrl)};
}

namespace detail {

/// Whether the sum of two non-negative integers exceeds the largest `long long`.
[[nodiscard]] inline bool sum_overflows(long long a, long long b) noexcept {
  constexpr long long top = (std::numeric_limits<long long>::max)();
  return b > top - a;
}

}  // namespace detail

/// Elevate the degree of a B-spline.
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///     increments (std::span<const long long>): The degrees to add per direction, `dim` values, at
///         least one positive.
///
/// Returns:
///     Bspline<T>: The B-spline of the same mapping at the elevated degrees.
///
/// Raises:
///     value_error: The number of increments does not match the dimension; one is negative; none
///         is positive; an elevated degree exceeds the exactness envelope of the binomial kernel
///         (`bincoeff_max_n`, also when `degree + increment` overflows `long long`, which the
///         message states as a bound and not as a degree); a direction to be elevated repeats a
///         knot more than `degree + 1` times; or one is not periodic and its knot vector is not
///         clamped.
template <StorageFloat T>
[[nodiscard]] Bspline<T> elevate_degree(const Bspline<T>& b,
                                        std::span<const long long> increments) {
  if (increments.size() != b.dim()) {
    throw value_error("Number of degree increments (" + std::to_string(increments.size()) +
                      ") must match dimension (" + std::to_string(b.dim()) + ").");
  }
  bool any = false;
  for (const long long inc : increments) {
    if (inc < 0) {
      throw value_error("Degree increments must be non-negative.");
    }
    any = any || inc > 0;
  }
  if (!any) {
    throw value_error("At least one degree increment must be positive.");
  }

  // The Bézier coefficient table is built from `C(degree + inc, i)`, so the elevated degree is what
  // has to stay in range.
  const BsplineSpace<T> whole = b.space();
  const std::vector<BsplineSpace1D<T>>& spaces = whole.spaces();
  for (std::size_t d = 0; d < spaces.size(); ++d) {
    if (increments[d] > 0) {
      if (detail::sum_overflows(spaces[d].degree(), increments[d])) {
        // The elevated degree does not fit a `long long`, so no number can be printed for it: the
        // message gives the bound it exceeds, and does not state a degree it is not.
        throw value_error("Degree elevation to a degree above " +
                          std::to_string((std::numeric_limits<long long>::max)()) +
                          " in direction " + std::to_string(d) +
                          " is far beyond the largest degree, " + std::to_string(bincoeff_max_n) +
                          ", that pantr's exact-integer binomial kernel can serve without an int64 "
                          "overflow.");
      }
      const long long elevated = spaces[d].degree() + increments[d];
      check_bincoeff_envelope(elevated, "Degree elevation to degree " + std::to_string(elevated) +
                                            " in direction " + std::to_string(d));
    }
  }

  return detail::map_directions<T>(
      b, [&](std::size_t d) { return increments[d] > 0; },
      [&](std::size_t d, const BsplineSpace1D<T>& space,
          mdview<const T, 2> flat) -> std::pair<BsplineSpace1D<T>, NdArray<T>> {
        const int p = space.degree();
        const int inc = static_cast<int>(increments[d]);
        const std::string what = "Degree elevation in direction " + std::to_string(d);
        if (space.periodic()) {
          // Round-trip through open form to preserve periodicity.
          const double tol = space.tolerance();
          const long long m_bdy = space.unique_knots_and_multiplicity(true).second[0];
          KnotsAndCtrl<T> open = to_open_1d<T>(space.knots(), p, flat, true, tol);
          // `to_open_1d` writes both end runs itself, so this states the kernel's precondition on
          // the vector that actually reaches it rather than trusting the conversion; a periodic
          // direction cannot fail it.
          check_clamped_knots<T>(open.knots, p, what);
          KnotsAndCtrl<T> elevated =
              detail::degree_elevate_kernel<T>(p, open.ctrl.template view<2>(), open.knots, inc);
          // Convert back to periodic. Degree elevation increases every knot multiplicity by
          // `inc`, so `m_bdy_new = m_bdy + inc`.
          KnotsAndCtrl<T> per = to_periodic_1d<T>(
              elevated.knots, p + inc, elevated.ctrl.template view<2>(), m_bdy + inc, tol);
          return {BsplineSpace1D<T>(std::span<const T>(per.knots), p + inc, true),
                  std::move(per.ctrl)};
        }
        check_clamped_knots<T>(space.knots(), p, what);
        KnotsAndCtrl<T> elevated = detail::degree_elevate_kernel<T>(p, flat, space.knots(), inc);
        return {BsplineSpace1D<T>(std::span<const T>(elevated.knots), p + inc),
                std::move(elevated.ctrl)};
      });
}

/// Reduce the degree of a B-spline.
///
/// Decomposes to Bézier segments, reduces each segment, and coarsens the knot vector to restore the
/// original continuity structure. Unlike `elevate_degree`, this operation is **not exact** in
/// general: the result is an approximation of the original mapping.
///
/// Each Bézier segment is reduced on its own, to the closest lower-degree polynomial in `L^2` among
/// those reproducing the segment's endpoint values. Adjacent segments therefore agree at every
/// breakpoint bit for bit and the stitch is exactly C0, with neither side moved off its own
/// optimum. Knots are then removed to restore the original continuity structure, and that step,
/// not the reduction, is what sets the final error on an open direction: it is a forced removal
/// with no deviation bound, so the exact per-segment figure `degree_reduction_error` reports does
/// not carry over to the assembled spline.
///
/// A **periodic** direction has a second such step. The reduction runs on the open form and the
/// periodic seam is then closed by a least-squares projection, also with no deviation bound,
/// because the reduced spline no longer lies in the periodic subspace exactly: for a smooth
/// periodic spline that projection is percent-level, not round-off. So on a periodic direction
/// neither of the two unbounded steps dominates by construction, and the error has to be measured
/// rather than attributed.
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///     decrements (std::span<const long long>): The degrees to remove per direction, `dim` values,
///         each in `[0, degree]`, at least one positive.
///
/// Returns:
///     Bspline<T>: The reduced B-spline.
///
/// Raises:
///     value_error: Invalid decrements (the number does not match the dimension; one is negative;
///         none is positive; one exceeds the degree); a direction to be reduced repeats a knot
///         more than `degree + 1` times; one is not periodic and its knot vector is not clamped; or
///         a rational result has a control weight that is not strictly positive.
template <StorageFloat T>
[[nodiscard]] Bspline<T> reduce_degree(const Bspline<T>& b, std::span<const long long> decrements) {
  if (decrements.size() != b.dim()) {
    throw value_error("Number of degree decrements (" + std::to_string(decrements.size()) +
                      ") must match dimension (" + std::to_string(b.dim()) + ").");
  }
  bool any = false;
  for (const long long dec : decrements) {
    if (dec < 0) {
      throw value_error("Degree decrements must be non-negative.");
    }
    any = any || dec > 0;
  }
  if (!any) {
    throw value_error("At least one degree decrement must be positive.");
  }
  const BsplineSpace<T> whole = b.space();
  const std::vector<BsplineSpace1D<T>>& spaces = whole.spaces();
  for (std::size_t d = 0; d < spaces.size(); ++d) {
    if (decrements[d] > spaces[d].degree()) {
      throw value_error("Degree decrement (" + std::to_string(decrements[d]) + ") in direction " +
                        std::to_string(d) + " exceeds current degree (" +
                        std::to_string(spaces[d].degree()) + ").");
    }
  }

  Bspline<T> result = detail::map_directions<T>(
      b, [&](std::size_t d) { return decrements[d] > 0; },
      [&](std::size_t d, const BsplineSpace1D<T>& space,
          mdview<const T, 2> flat) -> std::pair<BsplineSpace1D<T>, NdArray<T>> {
        const int p = space.degree();
        const int dec = static_cast<int>(decrements[d]);
        const double tol = space.tolerance();

        // Record the original interior knot multiplicities before the reduction. The boundary
        // knots (the first and last unique in-domain knots) are excluded from the coarsening: they
        // are clamped and stay at full multiplicity.
        const auto [orig_unique, orig_mults] = space.unique_knots_and_multiplicity(true);
        std::span<const T> interior_knots;
        std::span<const std::int64_t> interior_mults;
        if (orig_unique.size() > 2) {
          interior_knots = orig_unique.subspan(1, orig_unique.size() - 2);
          interior_mults = orig_mults.subspan(1, orig_mults.size() - 2);
        }
        const auto op =
            bezier::cached_reduction_operator(bezier::ReductionKind::interpolating, p, dec);
        const int new_degree = p - dec;

        if (space.periodic()) {
          const std::int64_t m_bdy = orig_mults[0];
          KnotsAndCtrl<T> open = to_open_1d<T>(space.knots(), p, flat, true, tol);
          check_clamped_knots<T>(open.knots, p,
                                 "Degree reduction in direction " + std::to_string(d), false);
          KnotsAndCtrl<T> red = detail::degree_reduce_kernel<T>(p, open.ctrl.template view<2>(),
                                                                open.knots, dec, *op);
          if (!interior_knots.empty()) {
            red = coarsen_knots_after_reduction<T>(std::move(red.knots), new_degree,
                                                   std::move(red.ctrl), interior_knots,
                                                   interior_mults, dec, tol);
          }
          // The same floor as the interior breakpoints in `coarsen_knots_after_reduction`: a
          // breakpoint that has to be present cannot drop below multiplicity 1, and the smoothness
          // the subtraction asks for there (C^{new_degree} or better) is already implied by
          // multiplicity 1. Without it the seam of a maximally smooth periodic spline
          // (`m_bdy = 1`) asks for multiplicity 0 and the reduction fails on a knot vector it
          // never had to build.
          const long long m_bdy_new = (std::max)(1LL, static_cast<long long>(m_bdy) - dec);
          // Infinity removes the closure's least-squares admissibility test rather than loosening
          // it, for the same reason `coarsen_knots_after_reduction` removes the removal kernel's:
          // restoring the seam the original spline had is structural, so the periodic form is what
          // must come out, and its distance from the reduced open form is not a criterion. That
          // distance *is* the reduction's own error. Grading it against a round-off floor asked an
          // approximation to be exact and refused every legitimate reduction of a smooth periodic
          // spline.
          //
          // It does NOT remove the C0 seam test, which keeps its own floor. An approximate
          // reduction does not move the two end control points: the reduction operator's first and
          // last rows are exact unit vectors and `coarsen_knots_after_reduction` touches interior
          // breakpoints only, so the boundary points arrive bit for bit unchanged, and the seam
          // test goes on catching what it was there for.
          KnotsAndCtrl<T> per =
              to_periodic_1d<T>(red.knots, new_degree, red.ctrl.template view<2>(), m_bdy_new, tol,
                                std::numeric_limits<double>::infinity());
          return {BsplineSpace1D<T>(std::span<const T>(per.knots), new_degree, true),
                  std::move(per.ctrl)};
        }

        check_clamped_knots<T>(space.knots(), p,
                               "Degree reduction in direction " + std::to_string(d), false);
        KnotsAndCtrl<T> red = detail::degree_reduce_kernel<T>(p, flat, space.knots(), dec, *op);
        if (!interior_knots.empty()) {
          red = coarsen_knots_after_reduction<T>(std::move(red.knots), new_degree,
                                                 std::move(red.ctrl), interior_knots,
                                                 interior_mults, dec, tol);
        }
        return {BsplineSpace1D<T>(std::span<const T>(red.knots), new_degree), std::move(red.ctrl)};
      });

  if (b.rational()) {
    bezier::detail::refuse_inverted_weight<T>(
        result.control_points(), result.cp_size(),
        "Bspline has no minimize_degree; convert to Bezier form with to_bezier or to_beziers and "
        "use Bezier.minimize_degree on the segments, which reduces only as far as it can without "
        "changing the sign of the weight function.");
  }
  return result;
}

}  // namespace pantr::bspline
