// _pantr.bspline: knot insertion, conversion to open and periodic form, subdivision, restriction
// and splitting of a Bspline; the two knot-vector functions BsplineSpace1D shares; and, in
// _testing, the blossom and the Oslo matrix in its banded and dense forms.
//
// This translation unit reaches Eigen (the periodic conversion's QR), which the module's
// vectorization settings already cover for every unit (EIGEN_DONT_VECTORIZE on MSVC's v4 rung).
// Arguments arrive as the wrapper normalised them (C-contiguous arrays of the exact dtype);
// validation of values is the C++ API's, with the messages the Python suite matches.
//
// Nothing here reaches the pool, so a call releases the GIL through `ReleaseIfLong`, on a lower
// bound of its work, and only around the C++ call: no Python object is touched inside the guard,
// and the NumPy results are built after it. Each estimate runs before the call validates its
// arguments. A refusal it can read off the arguments and the space's cached state (a count other
// than the dimension, a direction or a parameter out of range, nothing to do, such as bounds on
// both ends of a clamped direction) counts no work, so such a call keeps the GIL. Every other
// refusal is left unmirrored, since mirroring it would repeat the library's own checks (a knot
// outside the domain, a multiplicity above `degree + 1`, a capacity limit, a seam that does not
// close): such a call may release for nothing, which costs the caller a wait and changes no result.
// The testing hooks at the end keep the GIL: they are reached from the test-only `_testing` module
// alone.
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/bspline/blossom.hpp"
#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/knot_insertion.hpp"
#include "pantr/bspline/periodic.hpp"
#include "pantr/bspline/restrict.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

template <class T>
using In1D = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
template <class T>
using In2D = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <class T>
std::span<const T> as_span(const In1D<T>& a) {
  return {a.data(), a.shape(0)};
}

template <class T>
mdview<const T, 2> as_view(const In2D<T>& a) {
  return mdview<const T, 2>(a.data(), a.shape(0), a.shape(1));
}

template <class T>
NumpyArray<T> knots_to_numpy(const std::vector<T>& v) {
  NdArray<T> a({v.size()}, for_overwrite);
  std::copy(v.begin(), v.end(), a.data());
  return to_numpy(std::move(a));
}

/// One direction's new knots, or none to skip the direction.
template <class T>
using MaybeKnots = std::optional<In1D<T>>;

/// Whether `num_knots` knots can hold a domain at `degree` (`detail::check_domain_indexable`).
bool domain_indexable(std::size_t num_knots, long long degree) {
  return degree >= 0 && num_knots / 2 >= static_cast<std::size_t>(degree) + 1;
}

/// Work of `insert_knots`, for the GIL decision only: each direction's `refinement_ops` for the
/// knots it is given (a repeated value is inserted again). Zero when the count of directions is
/// not the dimension, or every direction is skipped, which the call refuses.
template <class T>
double insertion_ops(const bspline::Bspline<T>& b, std::span<const std::span<const T>> new_knots) {
  if (new_knots.size() != b.dim()) {
    return 0.0;
  }
  double ops = 0.0;
  for (std::size_t d = 0; d < b.dim(); ++d) {
    if (!new_knots[d].empty()) {
      ops += refinement_ops(b, d, static_cast<double>(new_knots[d].size()));
    }
  }
  return ops;
}

/// Work of `to_open`, for the GIL decision only, over the directions converted in order: a periodic
/// one wraps its stored rows into the full ones and inserts at least one copy of each domain end
/// (its seam multiplicity is at most `degree`), `refinement_ops` with nothing inserted; an
/// unclamped one writes at least its rows, refined or copied, `b.size()` values, and may trim
/// rows, after which nothing more is counted. Zero when every direction is open already, which the
/// call refuses.
template <class T>
double to_open_ops(const bspline::Bspline<T>& b) {
  const bspline::BsplineSpace<T> whole = b.space();
  double ops = 0.0;
  for (std::size_t d = 0; d < b.dim(); ++d) {
    const bspline::BsplineSpace1D<T>& space = whole.spaces()[d];
    if (space.periodic()) {
      ops += refinement_ops(b, d, 0.0);
    } else if (!space.has_open_knots()) {
      return ops + static_cast<double>(b.size());
    }
  }
  return ops;
}

/// Work of `to_periodic`, for the GIL decision only: each direction converted solves its change of
/// basis by a dense Householder QR of the `(n, k)` matrix that maps the `k` periodic coefficients
/// to the `n` open ones (`to_periodic_1d`), `n` the direction's rows and `k = n - c - 1` for seam
/// continuity `c <= degree - 1`, so `k >= n - degree`. Reflector `j` updates the `k - j - 1`
/// columns right of it over `n - j >= k - j` rows, a product and a rank-one update of a
/// multiply-add each per entry, `2 (k - j) (k - j - 1)` at least; summed, `(2 / 3) (k^3 - k)`.
/// Eigen's blocked form applies each block of reflectors with at least as many. A lower bound: it
/// leaves out the matrix's assembly, the right-hand side and the residual check. Zero when the
/// count is not the dimension or nothing is converted, which the call refuses before any work; a
/// continuity out of range is refused when its direction is reached, so only the directions
/// converted before it count.
template <class T>
double to_periodic_ops(const bspline::Bspline<T>& b,
                       const std::vector<std::optional<long long>>& continuity) {
  if (continuity.size() != b.dim()) {
    return 0.0;
  }
  const bspline::BsplineSpace<T> whole = b.space();
  double ops = 0.0;
  for (std::size_t d = 0; d < b.dim(); ++d) {
    const bspline::BsplineSpace1D<T>& space = whole.spaces()[d];
    if (space.periodic() || !continuity[d].has_value()) {
      continue;
    }
    const long long p = space.degree();
    const long long c =
        *continuity[d] == bspline::convert_default_continuity ? p - 1 : *continuity[d];
    if (c < 0 || c > p - 1) {
      return ops;  // refused once its direction is reached, after the earlier conversions
    }
    // At least 1: a space holds at least `degree + 1` functions (its constructor refuses fewer).
    const auto k = static_cast<double>(static_cast<long long>(b.shape()[d]) - p);
    ops += 2.0 / 3.0 * (k * k * k - k);
  }
  return ops;
}

/// Work of `subdivide`, for the GIL decision only: each subdivided direction's `refinement_ops`
/// for the `(count - 1) (degree - regularity)` knots inserted in each of its `num_intervals`
/// intervals (`compute_uniform_subdivision_knots`, whose classes are the space's own). Zero for
/// the arguments the call refuses before any work: a count of counts other than the dimension, a
/// count below 1, none above 1, or a regularity outside `[-1, degree - 1]` for a subdivided
/// direction.
template <class T>
double subdivision_ops(const bspline::Bspline<T>& b, const std::vector<long long>& counts,
                       std::optional<long long> regularity) {
  if (counts.size() != b.dim() ||
      std::any_of(counts.begin(), counts.end(), [](long long c) { return c < 1; })) {
    return 0.0;
  }
  const bspline::BsplineSpace<T> whole = b.space();
  double ops = 0.0;
  for (std::size_t d = 0; d < b.dim(); ++d) {
    if (counts[d] == 1) {
      continue;
    }
    const bspline::BsplineSpace1D<T>& space = whole.spaces()[d];
    const long long p = space.degree();
    const long long eff = regularity.value_or(p - 1);
    if (eff < -1 || eff > p - 1) {
      return 0.0;
    }
    const double per_interval =
        (static_cast<double>(counts[d]) - 1.0) * static_cast<double>(p - eff);
    ops += refinement_ops(b, d, static_cast<double>(space.num_intervals()) * per_interval);
  }
  return ops;
}

/// Work of `restrict`, for the GIL decision only: the first restricted direction inserts knots at
/// its bounds, or copies, over every row of the net before it cuts the piece (`restrict_1d`), so
/// it writes at least `b.size()` values. Zero for the arguments the call refuses before any work:
/// a count of bounds other than the dimension, no direction bounded, or bounds of the first
/// bounded direction, when it has no extents before it, that `validate_restrict_bounds` refuses (a
/// NaN, a lower bound not below the upper one, or a bound beyond a domain end by more than the
/// space's tolerance) or that `compute_boundary_knots_to_insert` refuses (both within the
/// tolerance of the domain's ends, both of them open, in a non-periodic direction). Bounds
/// refused in a later direction are reached after an earlier direction's restriction or the net's
/// flattened copy, `b.size()` values written either way, and a periodic direction refuses the full
/// domain after its conversion to open form.
template <class T>
double restriction_ops(const bspline::Bspline<T>& b,
                       const std::vector<std::optional<std::pair<double, double>>>& bounds) {
  if (bounds.size() != b.dim() ||
      std::none_of(bounds.begin(), bounds.end(), [](const auto& x) { return x.has_value(); })) {
    return 0.0;
  }
  const bspline::BsplineSpace<T> whole = b.space();
  const auto net = static_cast<double>(b.size());
  std::size_t before = 1;  // the extents before direction `d`
  bool first = true;
  for (std::size_t d = 0; d < b.dim(); ++d) {
    const std::size_t outer = before;
    before *= b.shape()[d];
    if (!bounds[d].has_value()) {
      continue;
    }
    const auto [lo, hi] = *bounds[d];
    const bspline::BsplineSpace1D<T>& space = whole.spaces()[d];
    const auto [start, end] = space.domain();
    const auto a = static_cast<double>(start);
    const auto e = static_cast<double>(end);
    const double tol = space.tolerance();
    const bool invalid = std::isnan(lo) || std::isnan(hi) || lo >= hi ||
                         (lo < a && std::abs(lo - a) > tol) || (hi > e && std::abs(hi - e) > tol);
    const bool full = !space.periodic() && std::abs(lo - a) <= tol && std::abs(hi - e) <= tol &&
                      space.knots().front() == start && space.knots().back() == end;
    if (invalid || full) {
      // Refused when its direction is reached: after an earlier direction's restriction, or after
      // the net's flattened copy for a direction with extents before it.
      return first && outer == 1 ? 0.0 : net;
    }
    first = false;
  }
  return net;
}

/// Work of `split`, for the GIL decision only: the net is flattened along the direction (a copy),
/// refined at the value or copied over every row (`split_1d`), and its rows copied into the two
/// halves, `b.size()` values written by each pass at least. Zero for a direction out of range or a
/// value not strictly inside the domain (further than the space's tolerance from both ends), which
/// the call refuses before any work.
template <class T>
double split_ops(const bspline::Bspline<T>& b, long long direction, double value) {
  if (direction < 0 || direction >= static_cast<long long>(b.dim())) {
    return 0.0;
  }
  const bspline::BsplineSpace<T> whole = b.space();
  const bspline::BsplineSpace1D<T>& space = whole.spaces()[static_cast<std::size_t>(direction)];
  const auto [lo, hi] = space.domain();
  const double tol = space.tolerance();
  if (!(value > static_cast<double>(lo) + tol && value < static_cast<double>(hi) - tol)) {
    return 0.0;
  }
  return 3.0 * static_cast<double>(b.size());
}

template <class T>
void def_refine(nb::class_<bspline::Bspline<T>>& cls) {
  using B = bspline::Bspline<T>;
  cls.def(
         "insert_knots",
         [](const B& b, const std::vector<MaybeKnots<T>>& per_dim) {
           std::vector<std::span<const T>> views;
           views.reserve(per_dim.size());
           for (const auto& k : per_dim) {
             views.push_back(k ? as_span(*k) : std::span<const T>());
           }
           const ReleaseIfLong release(insertion_ops<T>(b, views));
           return bspline::insert_knots<T>(b, views);
         },
         nb::arg("new_knots"),
         "Insert knots per direction (none or empty skips a direction); a periodic direction is "
         "refined through its open form.")
      .def(
          "to_open",
          [](const B& b) {
            const ReleaseIfLong release(to_open_ops(b));
            return bspline::to_open_bspline<T>(b);
          },
          "Convert every direction to open (clamped) non-periodic form.")
      .def(
          "to_periodic",
          [](const B& b, const std::vector<std::optional<long long>>& continuity) {
            const ReleaseIfLong release(to_periodic_ops(b, continuity));
            return bspline::to_periodic<T>(b, continuity);
          },
          nb::arg("continuity"),
          "Convert the directions with a continuity (-1 for the maximum; none skips) to periodic "
          "form.")
      .def(
          "subdivide",
          [](const B& b, const std::vector<long long>& counts,
             std::optional<long long> regularity) {
            const ReleaseIfLong release(subdivision_ops(b, counts, regularity));
            return bspline::subdivide<T>(b, counts, regularity);
          },
          nb::arg("counts"), nb::arg("regularity").none(),
          "Subdivide each direction whose count is at least 2 uniformly.")
      .def(
          "restrict",
          [](const B& b, const std::vector<std::optional<std::pair<double, double>>>& bounds) {
            std::vector<std::optional<bspline::Bounds>> bs;
            bs.reserve(bounds.size());
            for (const auto& x : bounds) {
              bs.push_back(x ? std::optional<bspline::Bounds>(bspline::Bounds{x->first, x->second})
                             : std::nullopt);
            }
            const ReleaseIfLong release(restriction_ops(b, bounds));
            return bspline::restrict<T>(b, bs);
          },
          nb::arg("bounds"), "Restrict each direction with bounds to that interval.")
      .def(
          "split",
          [](const B& b, long long direction, double value) {
            const ReleaseIfLong release(split_ops(b, direction, value));
            return bspline::split<T>(b, direction, value);
          },
          nb::arg("direction"), nb::arg("value"), "Split at a parameter value in one direction.");
}

template <class T>
void def_functions(nb::module_& m, const std::string& suffix) {
  m.def(("compute_inserted_knot_vector" + suffix).c_str(),
        [](const In1D<T>& knots, long long degree, const In1D<T>& new_knots, double tol) {
          const int p = saturate_degree(degree);
          std::vector<T> merged = [&] {
            // The merge writes the `n + m` knots and the multiplicity check reads them again; zero
            // for the refusals read off the sizes (a vector that cannot hold the domain, no new
            // knot).
            const std::size_t n = knots.shape(0);
            const std::size_t added = new_knots.shape(0);
            const bool refused = !domain_indexable(n, p) || added == 0;
            const ReleaseIfLong release(refused ? 0.0 : 2.0 * static_cast<double>(n + added));
            return bspline::compute_inserted_knot_vector<T>(as_span(knots), p, as_span(new_knots),
                                                            tol);
          }();
          return knots_to_numpy(merged);  // with the GIL: it builds a Python object
        },
        nb::arg("knots").noconvert(), nb::arg("degree"), nb::arg("new_knots").noconvert(),
        nb::arg("tol"), "Validate new knots and merge them into a knot vector.");
  m.def(("compute_uniform_subdivision_knots" + suffix).c_str(),
        [](const In1D<T>& knots, long long degree, double tol, long long n_subdivisions,
           std::optional<long long> regularity) {
          const int p = saturate_degree(degree);
          std::vector<T> inserted = [&] {
            // The knot classes read every knot. A lower bound: the values written are left out,
            // since knots within `tol` of each other can leave no interval to subdivide. Zero for
            // the refusals read off the arguments (a vector that cannot hold the domain, fewer than
            // two subdivisions, a regularity outside `[-1, degree - 1]`).
            const std::size_t n = knots.shape(0);
            const long long eff = regularity.value_or(static_cast<long long>(p) - 1);
            const bool refused =
                !domain_indexable(n, p) || n_subdivisions < 2 || eff < -1 || eff > p - 1LL;
            const ReleaseIfLong release(refused ? 0.0 : static_cast<double>(n));
            return bspline::compute_uniform_subdivision_knots<T>(as_span(knots), p, tol,
                                                                 n_subdivisions, regularity);
          }();
          return knots_to_numpy(inserted);  // with the GIL: it builds a Python object
        },
        nb::arg("knots").noconvert(), nb::arg("degree"), nb::arg("tol"), nb::arg("n_subdivisions"),
        nb::arg("regularity").none(), "The knots that subdivide every span uniformly.");
}

template <class T>
void def_kernels(nb::module_& m, const std::string& suffix) {
  m.def(("oslo_matrix" + suffix).c_str(),
        [](long long degree, const In1D<T>& old_knots, const In1D<T>& new_knots) {
          return to_numpy(bspline::oslo_matrix<T>(saturate_degree(degree), as_span(old_knots),
                                                  as_span(new_knots)));
        },
        nb::arg("degree"), nb::arg("old_knots").noconvert(), nb::arg("new_knots").noconvert(),
        "The dense Oslo refinement matrix (m + 1, n + 1).");
}

template <class T>
void def_blossom(nb::module_& m, const std::string& suffix) {
  m.def(("oslo_rows" + suffix).c_str(),
        [](long long degree, const In1D<T>& old_knots, const In1D<T>& new_knots) {
          const int p = saturate_degree(degree);
          // The kernel reads at least degree + 2 knots of each vector.
          bspline::detail::check_oslo_sizes(p, old_knots.shape(0), new_knots.shape(0));
          bspline::detail::OsloRows<T> rows =
              bspline::detail::oslo_rows<T>(p, as_span(old_knots), as_span(new_knots));
          NdArray<std::int64_t> first({rows.first_col.size()}, for_overwrite);
          std::copy(rows.first_col.begin(), rows.first_col.end(), first.data());
          return std::make_pair(to_numpy(std::move(rows.alphas)), to_numpy(std::move(first)));
        },
        nb::arg("degree"), nb::arg("old_knots").noconvert(), nb::arg("new_knots").noconvert(),
        "The banded rows of the Oslo matrix: (alphas (m + 1, degree + 1), first_col (m + 1,)).");
  m.def(("evaluate_blossom_1d" + suffix).c_str(),
        [](const In1D<T>& knots, long long degree, const In2D<T>& ctrl, const In1D<T>& u_values,
           double tol) {
          return knots_to_numpy(bspline::blossom<T>(as_span(knots), saturate_degree(degree),
                                                    as_view(ctrl), as_span(u_values), tol));
        },
        nb::arg("knots").noconvert(), nb::arg("degree"), nb::arg("control_points").noconvert(),
        nb::arg("u_values").noconvert(), nb::arg("tol"),
        "The blossom of a 1D B-spline at `degree` values: (rank,).");
  m.def(("blossom_span" + suffix).c_str(),
        [](const In1D<T>& knots, T u_last, long long n) {
          return static_cast<long long>(bspline::detail::blossom_span<T>(
              as_span(knots), u_last, static_cast<std::ptrdiff_t>(n)));
        },
        nb::arg("knots").noconvert(), nb::arg("u_last"), nb::arg("n"),
        "The span the blossom's recurrence starts from, in [0, n].");
}

}  // namespace

void register_bspline_refine(nb::module_& m) {
  // The classes exist already (register_bspline_ops runs first); their refinement methods are
  // bound here, in the translation unit that reaches Eigen.
  auto f64 = nb::borrow<nb::class_<bspline::Bspline<double>>>(m.attr("BsplineF64"));
  auto f32 = nb::borrow<nb::class_<bspline::Bspline<float>>>(m.attr("BsplineF32"));
  def_refine<double>(f64);
  def_refine<float>(f32);
  def_functions<double>(m, "");
  def_functions<float>(m, "_f32");
}

void register_bspline_blossom_testing(nb::module_& m) {
  def_blossom<double>(m, "");
  def_blossom<float>(m, "_f32");
  def_kernels<double>(m, "");
  def_kernels<float>(m, "_f32");
}

}  // namespace pantr::bindings
